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


def compute_ppwcs(signals: Dict, symbol: str = None) -> float:
    """Legacy PPWCS computation for backward compatibility"""
    try:
        base_score = 0.0
        
        # Simple scoring based on available signals
        if signals.get("volume_spike", False):
            base_score += 15
        if signals.get("price_movement", False):
            base_score += 10
        if signals.get("whale_activity", False):
            base_score += 20
        if signals.get("dex_inflow", False):
            base_score += 15
        if signals.get("social_momentum", False):
            base_score += 10
        
        return min(base_score, 100.0)
        
    except Exception as e:
        print(f"⚠️ Error in compute_ppwcs: {e}")
        return 0.0


def should_alert(score: float, symbol: str = None) -> bool:
    """Legacy alert logic for backward compatibility"""
    return score >= 50.0


def log_ppwcs_score(symbol: str, score: float, signals: Dict) -> bool:
    """Legacy score logging for backward compatibility"""
    try:
        os.makedirs("data/scores", exist_ok=True)
        
        log_entry = {
            "symbol": symbol,
            "score": score,
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("data/scores/ppwcs_log.json", "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(log_entry)}\n")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error logging score: {e}")
        return False


def get_previous_score(symbol: str) -> Optional[float]:
    """Legacy function to get previous score"""
    try:
        score_file = "data/scores/ppwcs_log.json"
        if not os.path.exists(score_file):
            return None
        
        with open(score_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in reversed(lines):
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get("symbol") == symbol:
                        return data.get("score", 0.0)
                except:
                    continue
        
        return None
        
    except Exception:
        return None


def save_score(symbol: str, score: float, data: Dict) -> bool:
    """Legacy score saving function"""
    return log_ppwcs_score(symbol, score, data)


def get_top_performers(hours: int = 24, limit: int = 10) -> List[Dict]:
    """Legacy top performers function"""
    try:
        performers = []
        score_file = "data/scores/ppwcs_log.json"
        
        if not os.path.exists(score_file):
            return performers
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        symbol_scores = {}
        
        with open(score_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        timestamp = datetime.fromisoformat(data.get("timestamp", ""))
                        
                        if timestamp >= cutoff_time:
                            symbol = data.get("symbol", "")
                            score = data.get("score", 0.0)
                            
                            if symbol not in symbol_scores or score > symbol_scores[symbol]["score"]:
                                symbol_scores[symbol] = {
                                    "symbol": symbol,
                                    "score": score,
                                    "timestamp": data.get("timestamp")
                                }
                    except:
                        continue
        
        # Sort by score and return top performers
        sorted_performers = sorted(symbol_scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_performers[:limit]
        
    except Exception as e:
        print(f"⚠️ Error getting top performers: {e}")
        return []


def get_symbol_stats(symbol: str) -> Dict:
    """Legacy symbol statistics function"""
    try:
        stats = {
            "symbol": symbol,
            "recent_score": get_previous_score(symbol) or 0.0,
            "alert_count": 0,
            "last_alert": None
        }
        
        return stats
        
    except Exception as e:
        print(f"⚠️ Error getting symbol stats: {e}")
        return {"symbol": symbol, "recent_score": 0.0, "alert_count": 0}


def compute_combined_scores(signals: Dict, symbol: str = None) -> Tuple[float, Dict, str]:
    """
    Legacy compute_combined_scores function for backward compatibility with stage_minus2_1
    
    Args:
        signals: Dictionary of detected signals
        symbol: Trading symbol (optional)
        
    Returns:
        Tuple[final_score, structure_dict, quality_assessment]
    """
    try:
        # Basic PPWCS-style scoring
        base_score = compute_ppwcs(signals, symbol)
        
        # Structure analysis
        structure = {
            "whale_detected": signals.get("whale_activity", False),
            "dex_flow": signals.get("dex_inflow", False),
            "volume_spike": signals.get("volume_spike", False),
            "compression": signals.get("stage_minus1_detected", False),
            "shadow_sync": signals.get("shadow_sync_active", False)
        }
        
        # Quality assessment
        active_signals = sum(1 for v in structure.values() if v)
        
        if active_signals >= 4:
            quality = "excellent"
        elif active_signals >= 3:
            quality = "strong"
        elif active_signals >= 2:
            quality = "moderate"
        else:
            quality = "weak"
        
        # Adjust score based on structure
        structure_bonus = active_signals * 5
        final_score = min(base_score + structure_bonus, 100.0)
        
        return final_score, structure, quality
        
    except Exception as e:
        print(f"⚠️ Error in compute_combined_scores: {e}")
        return 0.0, {}, "error"


def compute_checklist_score(signals: Dict) -> Tuple[float, Dict]:
    """
    Legacy checklist scoring function for backward compatibility
    
    Args:
        signals: Dictionary of signals
        
    Returns:
        Tuple[checklist_score, summary_dict]
    """
    try:
        checklist_items = {
            "whale_activity": signals.get("whale_activity", False),
            "dex_inflow": signals.get("dex_inflow", False),
            "volume_spike": signals.get("volume_spike", False),
            "compression": signals.get("stage_minus1_detected", False),
            "shadow_sync": signals.get("shadow_sync_active", False),
            "heatmap_exhaustion": signals.get("heatmap_exhaustion", False),
            "spoofing": signals.get("spoofing_suspected", False)
        }
        
        # Calculate score
        active_count = sum(1 for v in checklist_items.values() if v)
        total_possible = len(checklist_items)
        checklist_score = (active_count / total_possible) * 100
        
        summary = {
            "active_signals": active_count,
            "total_signals": total_possible,
            "percentage": checklist_score,
            "items": checklist_items
        }
        
        return checklist_score, summary
        
    except Exception as e:
        print(f"⚠️ Error in compute_checklist_score: {e}")
        return 0.0, {"error": str(e)}


def get_alert_level(ppwcs_score: float, checklist_score: float) -> int:
    """
    Legacy alert level function for backward compatibility
    
    Args:
        ppwcs_score: PPWCS score (0-100)
        checklist_score: Checklist score (0-100)
        
    Returns:
        int: Alert level (0=no alert, 1=low, 2=medium, 3=high)
    """
    try:
        # Combined scoring logic
        if ppwcs_score >= 65:
            return 3  # High alert
        elif ppwcs_score >= 50 or (ppwcs_score >= 35 and checklist_score >= 35):
            return 2  # Medium alert
        elif ppwcs_score >= 35 or checklist_score >= 50:
            return 1  # Low alert
        else:
            return 0  # No alert
            
    except Exception as e:
        print(f"⚠️ Error in get_alert_level: {e}")
        return 0


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