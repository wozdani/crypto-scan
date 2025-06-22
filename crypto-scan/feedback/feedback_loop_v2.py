#!/usr/bin/env python3
"""
Feedback Loop v2 - Advanced Self-Learning System

Automatyczna korekcja wag scoringowych TJDE na podstawie rzeczywistej skuteczności alertów.
Analizuje wyniki po 2h, 4h, 6h i dostosowuje wagi komponentów systemu.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any


# Configuration
WEIGHTS_FILE = "data/weights/tjde_weights.json"
ALERT_LOG = "logs/alerts_history.jsonl"
FEEDBACK_HISTORY = "logs/feedback_v2_history.json"

LEARNING_RATE = 0.05
SUCCESS_THRESHOLD_2H = 0.02  # 2% gain in 2 hours
SUCCESS_THRESHOLD_6H = 0.03  # 3% gain in 6 hours
MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.5


def load_current_weights() -> Dict[str, float]:
    """
    Ładuje aktualne wagi scoringowe z pliku JSON
    
    Returns:
        Dict z wagami lub domyślne jeśli plik nie istnieje
    """
    try:
        if not os.path.exists(WEIGHTS_FILE):
            print(f"[FEEDBACK V2] Weights file not found: {WEIGHTS_FILE}")
            return get_default_weights()
        
        with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract only numeric weights
        weights = {}
        default_weights = get_default_weights()
        
        for key in default_weights.keys():
            if key in data and isinstance(data[key], (int, float)):
                weights[key] = float(data[key])
            else:
                weights[key] = default_weights[key]
        
        print(f"[FEEDBACK V2] Loaded {len(weights)} weights from file")
        return weights
        
    except Exception as e:
        print(f"[FEEDBACK V2] Error loading weights: {e}")
        return get_default_weights()


def get_default_weights() -> Dict[str, float]:
    """Zwraca domyślne wagi TJDE"""
    return {
        "trend_strength": 0.25,
        "pullback_quality": 0.2,
        "support_reaction": 0.15,
        "liquidity_pattern_score": 0.1,
        "psych_score": 0.1,
        "htf_supportive_score": 0.1,
        "market_phase_modifier": 0.1
    }


def save_new_weights(weights: Dict[str, float], feedback_info: Dict = None) -> bool:
    """
    Zapisuje nowe wagi do pliku z metadanami
    
    Args:
        weights: Dict z nowymi wagami
        feedback_info: Opcjonalne info o feedback
        
    Returns:
        bool: True jeśli zapis się powiódł
    """
    try:
        os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            normalized_weights = {k: round(v / total, 4) for k, v in weights.items()}
        else:
            normalized_weights = get_default_weights()
        
        # Prepare data with metadata
        weights_data = {
            **normalized_weights,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "weights_version": "feedback_v2_trained",
            "source": "feedback_loop_v2",
            "feedback_info": feedback_info or {}
        }
        
        with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(weights_data, f, indent=2, ensure_ascii=False)
        
        print(f"[FEEDBACK V2] Weights updated and saved to {WEIGHTS_FILE}")
        return True
        
    except Exception as e:
        print(f"[FEEDBACK V2] Error saving weights: {e}")
        return False


def load_alert_logs() -> List[Dict]:
    """
    Ładuje logi alertów z pliku JSONL
    
    Returns:
        Lista z logami alertów
    """
    alerts = []
    
    try:
        if not os.path.exists(ALERT_LOG):
            print(f"[FEEDBACK V2] Alert log not found: {ALERT_LOG}")
            return alerts
        
        with open(ALERT_LOG, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    alert_data = json.loads(line)
                    alerts.append(alert_data)
                except json.JSONDecodeError as e:
                    print(f"[FEEDBACK V2] Error parsing line {line_num}: {e}")
                    continue
        
        print(f"[FEEDBACK V2] Loaded {len(alerts)} alert logs")
        return alerts
        
    except Exception as e:
        print(f"[FEEDBACK V2] Error loading alert logs: {e}")
        return []


def simulate_price_outcome(alert_data: Dict) -> Tuple[float, float, float]:
    """
    Symuluje wyniki cenowe po alertach (do testów bez prawdziwych danych)
    W produkcji należy zastąpić prawdziwymi danymi z API
    
    Args:
        alert_data: Dane alertu
        
    Returns:
        Tuple[result_2h, result_4h, result_6h] w % change
    """
    import random
    
    final_score = alert_data.get("final_score", 0.0)
    decision = alert_data.get("decision", "avoid")
    
    # Symulacja oparta na score - wyższy score = większa szansa na sukces
    base_performance = final_score * 0.05  # Max 5% for perfect score
    
    # Add randomness and market conditions
    market_noise = random.uniform(-0.02, 0.02)  # ±2% noise
    
    # Different time horizons
    result_2h = base_performance + market_noise + random.uniform(-0.01, 0.01)
    result_4h = result_2h + random.uniform(-0.015, 0.015)
    result_6h = result_4h + random.uniform(-0.01, 0.02)
    
    # Penalty for bad decisions
    if decision == "avoid":
        result_2h *= 0.3
        result_4h *= 0.3
        result_6h *= 0.3
    elif decision == "consider_entry":
        result_2h *= 0.7
        result_4h *= 0.8
        result_6h *= 0.9
    
    return result_2h, result_4h, result_6h


def analyze_alert_performance(alerts: List[Dict]) -> Tuple[int, int, float]:
    """
    Analizuje performance alertów
    
    Args:
        alerts: Lista alertów do analizy
        
    Returns:
        Tuple[total_analyzed, successful, success_rate]
    """
    total_analyzed = 0
    successful = 0
    
    for alert in alerts:
        if alert.get("alert_type") != "TJDE_adaptive":
            continue
        
        decision = alert.get("decision", "")
        if decision not in ["join_trend", "consider_entry"]:
            continue
        
        # Get or simulate results
        result_2h = alert.get("result_after_2h")
        result_6h = alert.get("result_after_6h")
        
        if result_2h is None or result_6h is None:
            # Simulate for testing
            result_2h, _, result_6h = simulate_price_outcome(alert)
            # In production, this would fetch real price data
        
        total_analyzed += 1
        
        # Check success criteria
        if result_2h >= SUCCESS_THRESHOLD_2H or result_6h >= SUCCESS_THRESHOLD_6H:
            successful += 1
    
    success_rate = successful / total_analyzed if total_analyzed > 0 else 0.0
    return total_analyzed, successful, success_rate


def calculate_weight_adjustments(alerts: List[Dict], current_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Oblicza korekty wag na podstawie analizy skuteczności
    
    Args:
        alerts: Lista alertów
        current_weights: Aktualne wagi
        
    Returns:
        Dict z korektami wag
    """
    weight_adjustments = {key: 0.0 for key in current_weights.keys()}
    adjustment_counts = {key: 0 for key in current_weights.keys()}
    
    for alert in alerts:
        if alert.get("alert_type") != "TJDE_adaptive":
            continue
        
        decision = alert.get("decision", "")
        if decision not in ["join_trend", "consider_entry"]:
            continue
        
        final_score = alert.get("final_score", 0.0)
        used_features = alert.get("used_features", {})
        
        # Get or simulate results
        result_2h = alert.get("result_after_2h")
        result_6h = alert.get("result_after_6h")
        
        if result_2h is None or result_6h is None:
            result_2h, _, result_6h = simulate_price_outcome(alert)
        
        # Determine if alert was successful
        was_successful = result_2h >= SUCCESS_THRESHOLD_2H or result_6h >= SUCCESS_THRESHOLD_6H
        
        # Calculate adjustments based on performance vs expectation
        for feature_name in current_weights.keys():
            feature_value = used_features.get(feature_name, 0.0)
            
            if feature_value == 0.0:
                continue
            
            # Feature impact calculation
            if was_successful and final_score < 0.6:
                # Alert was undervalued - increase weight of contributing features
                adjustment = LEARNING_RATE * feature_value
                weight_adjustments[feature_name] += adjustment
            elif not was_successful and final_score > 0.7:
                # Alert was overvalued - decrease weight of contributing features
                adjustment = -LEARNING_RATE * feature_value
                weight_adjustments[feature_name] += adjustment
            
            adjustment_counts[feature_name] += 1
    
    # Average adjustments
    for key in weight_adjustments:
        if adjustment_counts[key] > 0:
            weight_adjustments[key] /= adjustment_counts[key]
        weight_adjustments[key] = round(weight_adjustments[key], 6)
    
    return weight_adjustments


def apply_weight_adjustments(current_weights: Dict[str, float], adjustments: Dict[str, float]) -> Dict[str, float]:
    """
    Aplikuje korekty do wag z ograniczeniami
    
    Args:
        current_weights: Aktualne wagi
        adjustments: Korekty do zastosowania
        
    Returns:
        Dict z nowymi wagami
    """
    new_weights = current_weights.copy()
    
    for key in new_weights:
        adjustment = adjustments.get(key, 0.0)
        new_weight = new_weights[key] + adjustment
        
        # Apply bounds
        new_weights[key] = max(MIN_WEIGHT, min(new_weight, MAX_WEIGHT))
    
    return new_weights


def save_feedback_history(analysis_data: Dict):
    """Zapisuje historię feedback dla audytu"""
    try:
        os.makedirs(os.path.dirname(FEEDBACK_HISTORY), exist_ok=True)
        
        # Load existing history
        if os.path.exists(FEEDBACK_HISTORY):
            with open(FEEDBACK_HISTORY, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = {"feedback_sessions": []}
        
        # Add new session
        session = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **analysis_data
        }
        
        history["feedback_sessions"].append(session)
        
        # Keep only last 100 sessions
        if len(history["feedback_sessions"]) > 100:
            history["feedback_sessions"] = history["feedback_sessions"][-100:]
        
        with open(FEEDBACK_HISTORY, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"[FEEDBACK V2] Error saving feedback history: {e}")


def generate_weight_update_explanations(alerts: List[Dict], weight_adjustments: Dict[str, float]) -> Dict[str, str]:
    """
    Generate explanations for weight adjustments based on alert analysis
    
    Args:
        alerts: List of analyzed alerts
        weight_adjustments: Dictionary of weight changes
        
    Returns:
        Dict with explanations for each weight change
    """
    explanations = {}
    
    try:
        # Analyze specific examples from alerts
        successful_alerts = [a for a in alerts if a.get("result_after_2h", 0) >= SUCCESS_THRESHOLD_2H or a.get("result_after_6h", 0) >= SUCCESS_THRESHOLD_6H]
        failed_alerts = [a for a in alerts if a.get("result_after_2h", 0) < SUCCESS_THRESHOLD_2H and a.get("result_after_6h", 0) < SUCCESS_THRESHOLD_6H]
        
        for feature, adjustment in weight_adjustments.items():
            if abs(adjustment) < 0.005:  # Skip minimal changes
                continue
                
            if adjustment > 0:
                # Weight increased - find successful alerts with high feature value
                best_example = None
                for alert in successful_alerts:
                    feature_value = alert.get("used_features", {}).get(feature, 0)
                    if feature_value > 0.6:
                        best_example = alert
                        break
                
                if best_example:
                    symbol = best_example.get("symbol", "unknown")
                    explanations[feature] = f"Zwiększono - {symbol} miał wysoką wartość {feature} ({feature_value:.2f}) i był udany"
                else:
                    explanations[feature] = f"Zwiększono - analiza wskazuje na niedoszacowanie składnika"
            else:
                # Weight decreased - find failed alerts with high feature value
                worst_example = None
                for alert in failed_alerts:
                    feature_value = alert.get("used_features", {}).get(feature, 0)
                    final_score = alert.get("final_score", 0)
                    if feature_value > 0.6 and final_score > 0.7:
                        worst_example = alert
                        break
                
                if worst_example:
                    symbol = worst_example.get("symbol", "unknown")
                    explanations[feature] = f"Zmniejszono - {symbol} miał wysoki {feature}, ale był false positive"
                else:
                    explanations[feature] = f"Zmniejszono - składnik generował za dużo false positives"
    
    except Exception as e:
        print(f"[FEEDBACK V2] Error generating explanations: {e}")
        
    return explanations


def analyze_and_adjust() -> Dict[str, Any]:
    """
    Główna funkcja analizy i dostosowania wag
    
    Returns:
        Dict: Results including weights changes and explanations, or empty dict if failed
    """
    try:
        print(f"\n[FEEDBACK V2] Starting advanced feedback analysis...")
        
        # Load data
        alerts = load_alert_logs()
        if not alerts:
            print("[FEEDBACK V2] No alert logs found.")
            return {}
        
        current_weights = load_current_weights()
        
        # Analyze performance
        total_analyzed, successful, success_rate = analyze_alert_performance(alerts)
        
        if total_analyzed == 0:
            print("[FEEDBACK V2] No valid alerts found for analysis.")
            return {}
        
        print(f"[FEEDBACK V2] Performance Analysis:")
        print(f"  Total alerts analyzed: {total_analyzed}")
        print(f"  Successful alerts: {successful}")
        print(f"  Success rate: {success_rate:.1%}")
        
        # Calculate adjustments
        weight_adjustments = calculate_weight_adjustments(alerts, current_weights)
        
        # Generate explanations
        explanations = generate_weight_update_explanations(alerts, weight_adjustments)
        
        # Apply adjustments
        new_weights = apply_weight_adjustments(current_weights, weight_adjustments)
        
        # Prepare feedback info
        feedback_info = {
            "alerts_analyzed": total_analyzed,
            "successful_alerts": successful,
            "success_rate": round(success_rate, 4),
            "learning_rate": LEARNING_RATE,
            "weight_adjustments": weight_adjustments,
            "explanations": explanations,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Save new weights
        save_success = save_new_weights(new_weights, feedback_info)
        
        # Save history
        analysis_data = {
            "old_weights": current_weights,
            "new_weights": new_weights,
            "feedback_info": feedback_info
        }
        save_feedback_history(analysis_data)
        
        # Print summary
        print_adjustment_summary(current_weights, new_weights, weight_adjustments, feedback_info)
        
        # Return results for Telegram integration
        return {
            "success": save_success,
            "weights_before": current_weights,
            "weights_after": new_weights,
            "explanations": explanations,
            "adjustments": weight_adjustments,
            "performance": {
                "total_analyzed": total_analyzed,
                "successful": successful,
                "success_rate": success_rate
            }
        }
        
    except Exception as e:
        print(f"[FEEDBACK V2] Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}


def print_adjustment_summary(old_weights: Dict, new_weights: Dict, adjustments: Dict, feedback_info: Dict):
    """Wyświetla podsumowanie dostosowań"""
    print(f"\n📊 FEEDBACK V2 ADJUSTMENT SUMMARY")
    print("=" * 60)
    print(f"📈 Success Rate: {feedback_info['success_rate']:.1%}")
    print(f"🎯 Learning Rate: {LEARNING_RATE}")
    print(f"📅 Analysis Time: {feedback_info['analysis_timestamp']}")
    
    print(f"\n📋 WEIGHT CHANGES:")
    print(f"{'Feature':<25} {'Old':<10} {'New':<10} {'Change':<12}")
    print("-" * 60)
    
    for feature in old_weights:
        old_val = old_weights[feature]
        new_val = new_weights[feature]
        change = adjustments.get(feature, 0.0)
        
        change_str = f"{change:+.6f}" if change != 0 else "0.000000"
        arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        print(f"{feature:<25} {old_val:<10.4f} {new_val:<10.4f} {change_str:<12} {arrow}")


if __name__ == "__main__":
    print("🧠 Feedback Loop v2 - Advanced Self-Learning System")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "analyze":
            success = analyze_and_adjust()
            if success:
                print("✅ Feedback analysis completed successfully")
            else:
                print("❌ Feedback analysis failed")
        elif command == "status":
            weights = load_current_weights()
            print("📊 Current weights:")
            for feature, weight in weights.items():
                print(f"  {feature}: {weight:.4f}")
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: analyze, status")
    else:
        print("Available commands:")
        print("  analyze - Run feedback analysis and weight adjustment")
        print("  status  - Show current weights")
        print("\nExample: python feedback/feedback_loop_v2.py analyze")