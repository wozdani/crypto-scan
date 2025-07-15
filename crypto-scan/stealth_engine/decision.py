#!/usr/bin/env python3
"""
Diamond Decision Fusion Engine - Stage 3/7
Łączy wyniki trzech detektorów w adaptacyjną decyzję końcową

Detektory:
1. whale_ping_score (klasyczny stealth)
2. whaleclip_score (wizualny embedding portfela)  
3. diamond_score (temporal GNN + RL)

Zwraca fused_score i decyzję TRIGGER/AVOID z powodami
"""

import time
from typing import Dict, List, Any, Optional
import json
import os

def simulate_diamond_decision(whale_score: float, whaleclip_score: float, diamond_score: float, 
                            token: str = "UNKNOWN", volume_24h: float = 0) -> Dict[str, Any]:
    """
    Główna funkcja łącząca wyniki trzech detektorów w adaptacyjną decyzję końcową
    
    Args:
        whale_score: Score z whale_ping detection (0.0-1.0)
        whaleclip_score: Score z WhaleCLIP embedding analysis (0.0-1.0)
        diamond_score: Score z DiamondWhale AI temporal analysis (0.0-1.0)
        token: Symbol tokena dla logowania
        volume_24h: Wolumen 24h dla adaptacyjnych progów
        
    Returns:
        Dict z decyzją, fused_score i powodami
    """
    
    # 🎯 ADAPTIVE WEIGHTS - można później uczyć via feedback loop
    # Diamond AI ma wyższą wagę jako najbardziej zaawansowany detektor
    weights = {
        "whale_ping": 0.25,     # Klasyczny stealth detection
        "whaleclip": 0.35,      # Wizualny embedding analysis
        "diamond": 0.40         # Temporal graph + quantum RL (najwyższa waga)
    }
    
    # 📊 FUSED SCORE CALCULATION - weighted combination
    fused_score = (
        weights["whale_ping"] * whale_score +
        weights["whaleclip"] * whaleclip_score +
        weights["diamond"] * diamond_score
    )
    
    # 🔍 INDIVIDUAL TRIGGER ANALYSIS - check which detectors are active
    trigger_reasons = []
    confidence_level = "LOW"
    
    # High-confidence individual triggers (>0.7 threshold)
    if whale_score > 0.7:
        trigger_reasons.append("🐋 Whale Ping Detected")
    if whaleclip_score > 0.7:
        trigger_reasons.append("🧠 WhaleCLIP Confidence")
    if diamond_score > 0.7:
        trigger_reasons.append("💎 Diamond Temporal Anomaly")
    
    # Medium-confidence individual signals (>0.5 threshold)
    medium_signals = []
    if whale_score > 0.5:
        medium_signals.append("whale_ping")
    if whaleclip_score > 0.5:
        medium_signals.append("whaleclip")
    if diamond_score > 0.5:
        medium_signals.append("diamond_ai")
    
    # 🎯 ADAPTIVE DECISION LOGIC
    decision = "AVOID"
    decision_reason = "insufficient_signal"
    
    # SCENARIO 1: High fused score (>0.7) - strong combined signal
    if fused_score > 0.7:
        decision = "TRIGGER"
        decision_reason = "strong_fused_signal"
        confidence_level = "HIGH"
        
    # SCENARIO 2: Multiple high-confidence triggers (≥2 detectors >0.7)
    elif len(trigger_reasons) >= 2:
        decision = "TRIGGER"
        decision_reason = "multiple_high_confidence"
        confidence_level = "HIGH"
        
    # SCENARIO 3: Diamond AI dominance (diamond >0.8 even if others low)
    elif diamond_score > 0.8:
        decision = "TRIGGER"
        decision_reason = "diamond_ai_dominance"
        confidence_level = "MEDIUM"
        trigger_reasons.append("💎 Diamond AI Dominance")
        
    # SCENARIO 4: Strong combined medium signals (≥3 medium + fused >0.6)
    elif len(medium_signals) >= 3 and fused_score > 0.6:
        decision = "TRIGGER"
        decision_reason = "combined_medium_signals"
        confidence_level = "MEDIUM"
        trigger_reasons.append("🔗 Combined Signal Strength")
        
    # SCENARIO 5: Volume-adjusted threshold for low-cap tokens
    elif volume_24h > 0 and volume_24h < 1_000_000 and fused_score > 0.55:
        decision = "TRIGGER"
        decision_reason = "low_cap_adjusted_threshold"
        confidence_level = "LOW"
        trigger_reasons.append("📈 Low-Cap Volume Adjustment")
    
    # 📊 SCORE BREAKDOWN for transparency
    score_breakdown = {
        "whale_ping": {
            "raw_score": round(whale_score, 4),
            "weight": weights["whale_ping"],
            "contribution": round(weights["whale_ping"] * whale_score, 4)
        },
        "whaleclip": {
            "raw_score": round(whaleclip_score, 4),
            "weight": weights["whaleclip"],
            "contribution": round(weights["whaleclip"] * whaleclip_score, 4)
        },
        "diamond": {
            "raw_score": round(diamond_score, 4),
            "weight": weights["diamond"],
            "contribution": round(weights["diamond"] * diamond_score, 4)
        }
    }
    
    # 🎯 DOMINANT DETECTOR identification
    contributions = [
        ("whale_ping", weights["whale_ping"] * whale_score),
        ("whaleclip", weights["whaleclip"] * whaleclip_score),
        ("diamond", weights["diamond"] * diamond_score)
    ]
    dominant_detector = max(contributions, key=lambda x: x[1])[0]
    
    # 📝 COMPREHENSIVE LOGGING
    print(f"[DIAMOND DECISION] {token}: Fusing scores...")
    print(f"[DIAMOND DECISION] {token}: Whale={whale_score:.3f}, WhaleCLIP={whaleclip_score:.3f}, Diamond={diamond_score:.3f}")
    print(f"[DIAMOND DECISION] {token}: Fused score: {fused_score:.4f} (threshold: 0.7)")
    print(f"[DIAMOND DECISION] {token}: Decision: {decision} ({decision_reason})")
    print(f"[DIAMOND DECISION] {token}: Confidence: {confidence_level}")
    print(f"[DIAMOND DECISION] {token}: Dominant detector: {dominant_detector}")
    if trigger_reasons:
        print(f"[DIAMOND DECISION] {token}: Trigger reasons: {', '.join(trigger_reasons)}")
    
    # 🔄 RESULT STRUCTURE
    result = {
        "decision": decision,
        "fused_score": round(fused_score, 4),
        "confidence": confidence_level,
        "decision_reason": decision_reason,
        "trigger_reasons": trigger_reasons,
        "dominant_detector": dominant_detector,
        "score_breakdown": score_breakdown,
        "individual_scores": {
            "whale_ping": round(whale_score, 4),
            "whaleclip": round(whaleclip_score, 4),
            "diamond": round(diamond_score, 4)
        },
        "weights": weights,
        "timestamp": time.time(),
        "token": token
    }
    
    return result

def get_decision_weights() -> Dict[str, float]:
    """
    Pobierz aktualne wagi decyzyjne (może być później uczony via feedback loop)
    
    Returns:
        Dict z wagami dla każdego detektora
    """
    return {
        "whale_ping": 0.25,
        "whaleclip": 0.35,
        "diamond": 0.40
    }

def update_decision_weights(new_weights: Dict[str, float], 
                          config_path: str = "crypto-scan/cache") -> bool:
    """
    Aktualizuj wagi decyzyjne na podstawie feedback loop
    
    Args:
        new_weights: Nowe wagi dla detektorów
        config_path: Ścieżka do katalogu konfiguracji
        
    Returns:
        True jeśli sukces, False w przypadku błędu
    """
    try:
        weights_file = f"{config_path}/diamond_decision_weights.json"
        os.makedirs(config_path, exist_ok=True)
        
        # Walidacja wag
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"[DIAMOND DECISION] Warning: Weights sum to {total_weight:.3f}, not 1.0")
        
        # Zapisz nowe wagi
        with open(weights_file, 'w') as f:
            json.dump({
                "weights": new_weights,
                "updated_at": time.time(),
                "total_weight": total_weight
            }, f, indent=2)
        
        print(f"[DIAMOND DECISION] Updated decision weights: {new_weights}")
        return True
        
    except Exception as e:
        print(f"[DIAMOND DECISION ERROR] Failed to update weights: {e}")
        return False

def load_decision_weights(config_path: str = "crypto-scan/cache") -> Dict[str, float]:
    """
    Załaduj wagi decyzyjne z pliku (jeśli istnieje) lub użyj domyślnych
    
    Args:
        config_path: Ścieżka do katalogu konfiguracji
        
    Returns:
        Dict z wagami dla detektorów
    """
    try:
        weights_file = f"{config_path}/diamond_decision_weights.json"
        
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                data = json.load(f)
                weights = data.get("weights", get_decision_weights())
                print(f"[DIAMOND DECISION] Loaded custom weights: {weights}")
                return weights
        else:
            weights = get_decision_weights()
            print(f"[DIAMOND DECISION] Using default weights: {weights}")
            return weights
            
    except Exception as e:
        print(f"[DIAMOND DECISION ERROR] Failed to load weights: {e}")
        return get_decision_weights()

def batch_diamond_decisions(tokens_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Przetworz wiele tokenów przez Diamond Decision Engine
    
    Args:
        tokens_data: Lista słowników z danymi tokenów
        [{"token": "BTCUSDT", "whale_score": 0.8, "whaleclip_score": 0.6, "diamond_score": 0.9, "volume_24h": 1000000}]
        
    Returns:
        Lista wyników decyzyjnych
    """
    results = []
    
    print(f"[DIAMOND DECISION] Processing batch of {len(tokens_data)} tokens...")
    
    for token_data in tokens_data:
        try:
            token = token_data.get("token", "UNKNOWN")
            whale_score = token_data.get("whale_score", 0.0)
            whaleclip_score = token_data.get("whaleclip_score", 0.0)
            diamond_score = token_data.get("diamond_score", 0.0)
            volume_24h = token_data.get("volume_24h", 0)
            
            decision_result = simulate_diamond_decision(
                whale_score, whaleclip_score, diamond_score, token, volume_24h
            )
            
            results.append(decision_result)
            
        except Exception as e:
            print(f"[DIAMOND DECISION ERROR] Failed to process {token_data.get('token', 'UNKNOWN')}: {e}")
            
            # Fallback result
            results.append({
                "decision": "AVOID",
                "fused_score": 0.0,
                "confidence": "ERROR",
                "decision_reason": "processing_error",
                "error": str(e),
                "token": token_data.get("token", "UNKNOWN")
            })
    
    # Statystyki batch
    trigger_count = sum(1 for r in results if r["decision"] == "TRIGGER")
    high_confidence = sum(1 for r in results if r.get("confidence") == "HIGH")
    
    print(f"[DIAMOND DECISION] Batch complete: {trigger_count}/{len(results)} triggers, {high_confidence} high-confidence")
    
    return results

def test_diamond_decision():
    """Test funkcji simulate_diamond_decision z różnymi scenariuszami"""
    print("🔍 Testing Diamond Decision Engine...")
    
    test_cases = [
        # High fused score
        {"name": "High Fused Score", "whale": 0.8, "clip": 0.7, "diamond": 0.9, "volume": 50000000},
        # Multiple high triggers
        {"name": "Multiple High Triggers", "whale": 0.8, "clip": 0.8, "diamond": 0.4, "volume": 10000000},
        # Diamond dominance
        {"name": "Diamond Dominance", "whale": 0.3, "clip": 0.2, "diamond": 0.85, "volume": 5000000},
        # Combined medium signals
        {"name": "Combined Medium", "whale": 0.6, "clip": 0.6, "diamond": 0.6, "volume": 2000000},
        # Low-cap adjustment
        {"name": "Low-Cap Adjustment", "whale": 0.5, "clip": 0.5, "diamond": 0.6, "volume": 500000},
        # Insufficient signal
        {"name": "Insufficient Signal", "whale": 0.3, "clip": 0.2, "diamond": 0.4, "volume": 1000000}
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {case['name']} ---")
        result = simulate_diamond_decision(
            case["whale"], case["clip"], case["diamond"], 
            f"TEST{i}", case["volume"]
        )
        print(f"Result: {result['decision']} (score: {result['fused_score']}, confidence: {result['confidence']})")

if __name__ == "__main__":
    test_diamond_decision()