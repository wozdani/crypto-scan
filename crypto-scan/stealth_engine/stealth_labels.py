#!/usr/bin/env python3
"""
Stealth Labels System - Auto-labeling dla Stealth Engine alerts
Automatyczne przypisywanie etykiet na podstawie aktywnych sygnałów
dla późniejszego wykorzystania w ML i feedback loop
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional

def generate_stealth_label(active_signals: List[str]) -> str:
    """
    Generuje inteligentną etykietę na podstawie aktywnych sygnałów Stealth Engine
    
    Args:
        active_signals: Lista aktywnych sygnałów z analyze_signals()
        
    Returns:
        str: Etykieta opisująca typ sygnału
    """
    
    # Konwersja do set dla szybszego sprawdzania
    signals_set = set(active_signals)
    
    # PATTERN 1: Early Whale Accumulation
    if {"whale_activity_tracking", "volume_spike_detection"}.issubset(signals_set):
        return "early_whale_accumulation"
    
    # PATTERN 2: Stealth DEX Surge  
    if {"dex_inflow", "orderbook_spoofing_detection"}.issubset(signals_set):
        return "stealth_dex_surge"
    
    # PATTERN 3: Orderbook Manipulation
    if {"bid_wall_detection", "ask_wall_detection", "orderbook_spoofing_detection"}.issubset(signals_set):
        return "orderbook_manipulation"
    
    # PATTERN 4: Volume Accumulation
    if {"volume_spike_detection", "volume_accumulation_detection"}.issubset(signals_set):
        return "volume_accumulation"
    
    # PATTERN 5: Spread Squeeze
    if {"bid_ask_spread_tightening", "liquidity_absorption"}.issubset(signals_set):
        return "spread_squeeze"
    
    # PATTERN 6: Institutional Flow
    if {"dex_inflow", "whale_activity_tracking", "bid_wall_detection"}.issubset(signals_set):
        return "institutional_flow"
    
    # PATTERN 7: Pump Preparation
    if {"orderbook_imbalance", "volume_spike_detection", "bid_wall_detection"}.issubset(signals_set):
        return "pump_preparation"
    
    # PATTERN 8: Stealth Accumulation (subtle signals)
    if {"dex_inflow", "bid_ask_spread_tightening"}.issubset(signals_set):
        return "stealth_accumulation"
    
    # PATTERN 9: Market Structure Change
    if {"orderbook_imbalance", "liquidity_absorption"}.issubset(signals_set):
        return "market_structure_change"
    
    # PATTERN 10: Single strong signal patterns
    if "whale_activity_tracking" in signals_set and len(active_signals) >= 2:
        return "whale_activity"
    elif "dex_inflow" in signals_set and len(active_signals) >= 2:
        return "dex_activity"
    elif "orderbook_spoofing_detection" in signals_set:
        return "spoofing_pattern"
    elif "volume_spike_detection" in signals_set:
        return "volume_pattern"
    
    # DEFAULT: Multiple signals but no clear pattern
    if len(active_signals) >= 3:
        return "complex_multi_signal"
    elif len(active_signals) == 2:
        return "moderate_activity"
    elif len(active_signals) == 1:
        return "single_signal"
    else:
        return "no_clear_pattern"

def get_label_confidence(active_signals: List[str], stealth_score: float) -> float:
    """
    Oblicza poziom pewności dla wygenerowanej etykiety
    
    Args:
        active_signals: Lista aktywnych sygnałów
        stealth_score: Score z Stealth Engine
        
    Returns:
        float: Confidence level (0.0-1.0)
    """
    
    # Base confidence based on number of signals
    base_confidence = min(len(active_signals) * 0.25, 0.8)
    
    # Score bonus
    score_bonus = min(stealth_score / 5.0, 0.2)
    
    # Pattern specificity bonus
    signals_set = set(active_signals)
    specificity_bonus = 0.0
    
    # High specificity patterns
    high_specificity = [
        {"whale_activity_tracking", "volume_spike_detection"},
        {"dex_inflow", "orderbook_spoofing_detection"},
        {"bid_wall_detection", "ask_wall_detection", "orderbook_spoofing_detection"}
    ]
    
    for pattern in high_specificity:
        if pattern.issubset(signals_set):
            specificity_bonus = 0.15
            break
    
    total_confidence = base_confidence + score_bonus + specificity_bonus
    return min(total_confidence, 1.0)

def save_stealth_label(symbol: str, stealth_score: float, active_signals: List[str], 
                      alert_type: str, timestamp: Optional[datetime] = None) -> str:
    """
    Zapisuje etykietę Stealth Engine do pliku JSON
    
    Args:
        symbol: Symbol tokena
        stealth_score: Score z Stealth Engine
        active_signals: Lista aktywnych sygnałów
        alert_type: Typ alertu (strong_stealth_alert, medium_alert, etc.)
        timestamp: Opcjonalny timestamp (domyślnie teraz)
        
    Returns:
        str: Ścieżka do zapisanego pliku
    """
    
    if timestamp is None:
        timestamp = datetime.now()
    
    # Generuj etykietę i confidence
    label = generate_stealth_label(active_signals)
    confidence = get_label_confidence(active_signals, stealth_score)
    
    # Przygotuj dane do zapisu
    label_data = {
        "symbol": symbol,
        "timestamp": timestamp.isoformat(),
        "stealth_score": stealth_score,
        "active_signals": active_signals,
        "alert_type": alert_type,
        "stealth_label": label,
        "label_confidence": confidence,
        "metadata": {
            "signals_count": len(active_signals),
            "score_category": "high" if stealth_score >= 4.0 else "medium" if stealth_score >= 2.5 else "low",
            "pattern_complexity": "complex" if len(active_signals) >= 4 else "moderate" if len(active_signals) >= 2 else "simple"
        }
    }
    
    # Utworz folder labels jeśli nie istnieje
    labels_dir = "labels"
    os.makedirs(labels_dir, exist_ok=True)
    
    # Nazwa pliku z timestamp
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{timestamp_str}_stealth_label.json"
    filepath = os.path.join(labels_dir, filename)
    
    # Zapisz plik
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, indent=2, ensure_ascii=False)
    
    print(f"[STEALTH LABEL] {symbol} → Label: {label} (confidence: {confidence:.3f}) saved to {filepath}")
    
    return filepath

def load_stealth_labels(symbol: Optional[str] = None, hours: int = 24) -> List[Dict]:
    """
    Ładuje zapisane etykiety Stealth Engine
    
    Args:
        symbol: Opcjonalny symbol do filtrowania
        hours: Liczba godzin wstecz do załadowania
        
    Returns:
        List[Dict]: Lista załadowanych etykiet
    """
    
    labels_dir = "labels"
    if not os.path.exists(labels_dir):
        return []
    
    labels = []
    cutoff_time = datetime.now().timestamp() - (hours * 3600)
    
    for filename in os.listdir(labels_dir):
        if not filename.endswith("_stealth_label.json"):
            continue
            
        if symbol and not filename.startswith(symbol):
            continue
        
        filepath = os.path.join(labels_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                
            # Check timestamp
            label_timestamp = datetime.fromisoformat(label_data["timestamp"]).timestamp()
            if label_timestamp >= cutoff_time:
                labels.append(label_data)
                
        except Exception as e:
            print(f"[LABEL ERROR] Failed to load {filepath}: {e}")
    
    # Sort by timestamp (newest first)
    labels.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return labels

def get_label_statistics(hours: int = 24) -> Dict:
    """
    Generuje statystyki etykiet Stealth Engine
    
    Args:
        hours: Liczba godzin wstecz do analizy
        
    Returns:
        Dict: Statystyki etykiet
    """
    
    labels = load_stealth_labels(hours=hours)
    
    if not labels:
        return {"total_labels": 0, "message": "No labels found"}
    
    # Count by label type
    label_counts = {}
    confidence_sum = 0
    score_sum = 0
    alert_type_counts = {}
    
    for label_data in labels:
        label = label_data["stealth_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        
        confidence_sum += label_data["label_confidence"]
        score_sum += label_data["stealth_score"]
        
        alert_type = label_data["alert_type"]
        alert_type_counts[alert_type] = alert_type_counts.get(alert_type, 0) + 1
    
    # Calculate averages
    avg_confidence = confidence_sum / len(labels)
    avg_score = score_sum / len(labels)
    
    # Find most common patterns
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "total_labels": len(labels),
        "average_confidence": avg_confidence,
        "average_score": avg_score,
        "most_common_patterns": sorted_labels[:5],
        "alert_type_distribution": alert_type_counts,
        "pattern_diversity": len(label_counts),
        "time_period_hours": hours
    }

def test_stealth_labels():
    """Test funkcji stealth labels"""
    
    print("🧪 Testing Stealth Labels System...")
    
    # Test label generation
    test_signals = [
        ["whale_activity_tracking", "volume_spike_detection"],
        ["dex_inflow", "orderbook_spoofing_detection"],
        ["bid_wall_detection", "ask_wall_detection"],
        ["volume_spike_detection"],
        []
    ]
    
    for signals in test_signals:
        label = generate_stealth_label(signals)
        confidence = get_label_confidence(signals, 3.5)
        print(f"   Signals: {signals} → Label: {label} (confidence: {confidence:.3f})")
    
    # Test save/load
    test_symbol = "TESTLABEL"
    filepath = save_stealth_label(test_symbol, 4.2, ["dex_inflow", "whale_activity_tracking"], "strong_stealth_alert")
    
    # Load and verify
    loaded_labels = load_stealth_labels(test_symbol, 1)
    if loaded_labels:
        print(f"   ✅ Label saved and loaded successfully: {loaded_labels[0]['stealth_label']}")
    
    # Get statistics
    stats = get_label_statistics(24)
    print(f"   📊 Label statistics: {stats['total_labels']} total labels")
    
    print("   ✅ Stealth Labels System operational")

if __name__ == "__main__":
    test_stealth_labels()