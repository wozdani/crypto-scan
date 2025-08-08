#!/usr/bin/env python3
"""
Scoring Utilities - Dynamiczne ładowanie wag scoringowych dla TJDE + Legacy Functions

Zarządza wagami scoringowymi dla simulate_trader_decision_advanced()
z automatycznym fallback do domyślnych wartości.
Zawiera także legacy functions dla backward compatibility.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta


DEFAULT_TJDE_WEIGHTS = {
    "trend_strength": 0.220,
    "pullback_quality": 0.180,
    "support_reaction": 0.160,
    "clip_confidence_score": 0.250,
    "liquidity_pattern_score": 0.120,
    "psych_score": 0.080,
    "htf_supportive_score": 0.060,
    "market_phase_modifier": 0.030
}


def load_tjde_weights(filepath: str = "data/weights/tjde_weights.json") -> Dict[str, float]:
    """
    Ładuje wagi scoringowe TJDE z pliku JSON
    
    Args:
        filepath: Ścieżka do pliku z wagami
        
    Returns:
        Dict z wagami scoringowymi
    """
    if not os.path.exists(filepath):
        print(f"[TJDE WEIGHTS] Loaded adaptive weights from file: {filepath}")
        logging.debug(f"[TJDE WEIGHTS] Weights file not found, using defaults")
        return DEFAULT_TJDE_WEIGHTS.copy()
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"[TJDE WEIGHTS] Loaded adaptive weights from file: {filepath}")
        logging.debug(f"[TJDE WEIGHTS] Successfully loaded weights file: {filepath}")
        
        # Extract only the weight values (ignore metadata)
        weights = {}
        for key in DEFAULT_TJDE_WEIGHTS.keys():
            if key in data and isinstance(data[key], (int, float)):
                weights[key] = float(data[key])
            else:
                weights[key] = float(DEFAULT_TJDE_WEIGHTS[key])
        
        # Ensure CLIP weight exists
        if "clip_confidence_score" not in weights:
            weights["clip_confidence_score"] = 0.12
            print(f"[TJDE WEIGHTS] Added missing CLIP confidence weight: 0.12")
        
        return weights
        
    except Exception as e:
        print(f"[TJDE WEIGHTS] Error loading weights, fallback to default. Reason: {e}")
        return DEFAULT_TJDE_WEIGHTS.copy()


def save_tjde_weights(weights: Dict[str, float], filepath: str = "data/weights/tjde_weights.json") -> bool:
    """
    Zapisuje wagi scoringowe TJDE do pliku JSON
    
    Args:
        weights: Dict z wagami do zapisania
        filepath: Ścieżka do pliku
        
    Returns:
        bool: True jeśli zapis się powiódł
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            normalized_weights = {k: round(v / total, 4) for k, v in weights.items()}
        else:
            normalized_weights = DEFAULT_TJDE_WEIGHTS.copy()
        
        # Add metadata
        weights_data = {
            **normalized_weights,
            "last_updated": "2025-06-22T19:25:00Z",
            "weights_version": "tjde_v1.0",
            "source": "scoring_utilities"
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(weights_data, f, indent=2, ensure_ascii=False)
        
        print(f"[TJDE WEIGHTS] Weights saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"[TJDE WEIGHTS] Error saving weights: {e}")
        return False


def validate_tjde_weights(weights: Dict[str, float]) -> bool:
    """
    Waliduje czy wagi są poprawne
    
    Args:
        weights: Dict z wagami do walidacji
        
    Returns:
        bool: True jeśli wagi są poprawne
    """
    try:
        # Check if all required keys are present
        required_keys = set(DEFAULT_TJDE_WEIGHTS.keys())
        provided_keys = set(weights.keys())
        
        if not required_keys.issubset(provided_keys):
            missing = required_keys - provided_keys
            print(f"[TJDE WEIGHTS] Missing required weights: {missing}")
            return False
        
        # Check if all values are positive numbers
        for key, value in weights.items():
            if key in required_keys:
                if not isinstance(value, (int, float)) or value < 0:
                    print(f"[TJDE WEIGHTS] Invalid weight value for {key}: {value}")
                    return False
        
        # Check if weights sum is reasonable (between 0.8 and 1.2)
        total = sum(weights[key] for key in required_keys)
        if not (0.8 <= total <= 1.2):
            print(f"[TJDE WEIGHTS] Weights sum ({total:.3f}) outside acceptable range")
            return False
        
        return True
        
    except Exception as e:
        print(f"[TJDE WEIGHTS] Error validating weights: {e}")
        return False


def get_weight_adjustment_factor(market_phase: str) -> Dict[str, float]:
    """
    Zwraca faktory dostosowania wag dla różnych faz rynku
    
    Args:
        market_phase: Nazwa fazy rynku
        
    Returns:
        Dict z faktorami dostosowania
    """
    phase_adjustments = {
        "breakout-continuation": {
            "trend_strength": 1.2,
            "pullback_quality": 1.1,
            "support_reaction": 1.0,
            "liquidity_pattern_score": 1.3,
            "psych_score": 0.8,
            "htf_supportive_score": 1.1,
            "market_phase_modifier": 1.0
        },
        "range-accumulation": {
            "trend_strength": 0.7,
            "pullback_quality": 1.0,
            "support_reaction": 1.2,
            "liquidity_pattern_score": 1.4,
            "psych_score": 1.5,
            "htf_supportive_score": 1.0,
            "market_phase_modifier": 1.0
        },
        "exhaustion-pullback": {
            "trend_strength": 0.8,
            "pullback_quality": 1.3,
            "support_reaction": 1.4,
            "liquidity_pattern_score": 0.9,
            "psych_score": 1.6,
            "htf_supportive_score": 1.2,
            "market_phase_modifier": 1.0
        }
    }
    
    return phase_adjustments.get(market_phase, {k: 1.0 for k in DEFAULT_TJDE_WEIGHTS.keys()})


def apply_phase_adjustments(base_weights: Dict[str, float], market_phase: str) -> Dict[str, float]:
    """
    Aplikuje dostosowania wag dla konkretnej fazy rynku
    
    Args:
        base_weights: Bazowe wagi
        market_phase: Faza rynku
        
    Returns:
        Dict z dostosowanymi wagami
    """
    adjustments = get_weight_adjustment_factor(market_phase)
    adjusted_weights = {}
    
    for key in base_weights:
        adjustment_factor = adjustments.get(key, 1.0)
        adjusted_weights[key] = base_weights[key] * adjustment_factor
    
    # Normalize after adjustments
    total = sum(adjusted_weights.values())
    if total > 0:
        for key in adjusted_weights:
            adjusted_weights[key] /= total
    
    return adjusted_weights


# === LEGACY FUNCTIONS FOR BACKWARD COMPATIBILITY ===

def get_recent_alerts(hours: int = 24) -> List[Dict]:
    """Legacy function for recent alerts"""
    try:
        alerts = []
        alert_files = ["data/alerts/alerts_history.json", "logs/alerts_history.jsonl"]
        
        for file_path in alert_files:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_path.endswith(".jsonl"):
                        for line in f:
                            if line.strip():
                                try:
                                    alerts.append(json.loads(line))
                                except:
                                    continue
                    else:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                alerts.extend(data)
                            elif isinstance(data, dict) and "alerts" in data:
                                alerts.extend(data["alerts"])
                        except:
                            continue
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = []
        
        for alert in alerts:
            try:
                timestamp_str = alert.get("timestamp", "")
                if timestamp_str:
                    alert_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    if alert_time >= cutoff_time:
                        recent_alerts.append(alert)
            except:
                continue
        
        return recent_alerts[-50:]  # Return last 50 alerts
        
    except Exception as e:
        print(f"⚠️ Error getting recent alerts: {e}")
        return []


# Legacy compute_ppwcs removed - using TJDE v2 only


# Legacy should_alert removed - using TJDE v2 only


# Legacy log_ppwcs_score removed - using TJDE v2 only


# Legacy get_previous_score removed - using TJDE v2 only


# Legacy save_score removed - using TJDE v2 only


# Legacy get_top_performers removed - using TJDE v2 only


# Legacy get_symbol_stats removed - using TJDE v2 only


# Legacy compute_combined_scores removed - using TJDE v2 only


# Legacy compute_checklist_score removed - using TJDE v2 only


# Legacy get_alert_level removed - using TJDE v2 only


if __name__ == "__main__":
    # Test weight loading and saving
    print("🧪 Testing TJDE Weight Management...")
    
    # Test loading (should create default if not exists)
    weights = load_tjde_weights()
    print(f"Loaded weights: {weights}")
    
    # Test validation
    is_valid = validate_tjde_weights(weights)
    print(f"Weights validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    
    # Test phase adjustments
    breakout_weights = apply_phase_adjustments(weights, "breakout-continuation")
    print(f"Breakout-adjusted weights: {breakout_weights}")
    
    # Test saving
    save_success = save_tjde_weights(weights)
    print(f"Save test: {'✅ SUCCESS' if save_success else '❌ FAILED'}")
    
    # Test legacy functions
    print("\n🧪 Testing Legacy Functions...")
    test_signals = {"volume_spike": True, "whale_activity": True}
    test_score = compute_ppwcs(test_signals, "TESTUSDT")
    print(f"Legacy PPWCS: {test_score}")
    
    print("✅ TJDE Weight Management + Legacy test complete")