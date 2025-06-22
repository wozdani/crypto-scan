#!/usr/bin/env python3
"""
Feedback Loop Trainer - Automatyczne dostrajanie wag scoringowych

Analizuje skuteczność alertów i automatycznie dostraja wagi w simulate_trader_decision_advanced()
na podstawie oznaczonych logów z etykietami 'good' lub 'bad'.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple


WEIGHTS_FILE = "data/weights/tjde_weights.json"
FEEDBACK_LOG = "logs/advanced_trader_log.txt"
FEEDBACK_HISTORY = "logs/feedback_training_history.json"

# Domyślne wagi (jeśli plik nie istnieje)
DEFAULT_WEIGHTS = {
    "trend_strength": 0.25,
    "pullback_quality": 0.2,
    "support_reaction": 0.15,
    "liquidity_pattern_score": 0.1,
    "psych_score": 0.1,
    "htf_supportive_score": 0.1,
    "market_phase_modifier": 0.1
}


def load_weights() -> Dict[str, float]:
    """
    Ładuje wagi scoringowe z pliku lub zwraca domyślne
    
    Returns:
        Dict z wagami dla każdej cechy
    """
    try:
        os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
        
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
                weights_data = json.load(f)
            
            # Extract only the scoring weights
            weights = {}
            for key in DEFAULT_WEIGHTS.keys():
                if key in weights_data and isinstance(weights_data[key], (int, float)):
                    weights[key] = float(weights_data[key])
                else:
                    weights[key] = DEFAULT_WEIGHTS[key]
            
            print(f"[TJDE TRAINER] Loaded adaptive weights from {WEIGHTS_FILE}")
            return weights
        else:
            print(f"[TJDE TRAINER] No weights file found, using defaults")
            return DEFAULT_WEIGHTS.copy()
            
    except Exception as e:
        print(f"⚠️ [TJDE TRAINER] Error loading weights: {e}")
        return DEFAULT_WEIGHTS.copy()


def save_weights(weights: Dict[str, float], training_info: Dict = None):
    """
    Zapisuje wagi scoringowe do pliku z metadanymi treningowymi
    
    Args:
        weights: Dict z wagami
        training_info: Opcjonalne info o treningu
    """
    try:
        os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: round(v / total_weight, 4) for k, v in weights.items()}
        else:
            normalized_weights = DEFAULT_WEIGHTS.copy()
        
        weights_data = {
            **normalized_weights,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "training_version": "tjde_feedback_v1.0",
            "training_info": training_info or {}
        }
        
        with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(weights_data, f, indent=2, ensure_ascii=False)
        
        print(f"[TJDE TRAINER] Weights saved to {WEIGHTS_FILE}")
        
    except Exception as e:
        print(f"❌ [TJDE TRAINER] Error saving weights: {e}")


def parse_feedback_logs() -> Tuple[List[Dict], int, int]:
    """
    Parsuje logi feedback z etykietami good/bad
    
    Returns:
        Tuple[parsed_entries, good_count, bad_count]
    """
    parsed_entries = []
    good_count = 0
    bad_count = 0
    
    try:
        if not os.path.exists(FEEDBACK_LOG):
            print(f"⚠️ [TJDE TRAINER] Feedback log not found: {FEEDBACK_LOG}")
            return parsed_entries, good_count, bad_count
        
        with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        print(f"[TJDE TRAINER] Parsing {len(lines)} log lines...")
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or '"final_score"' not in line:
                continue
            
            try:
                # Try to parse as JSON
                data = json.loads(line)
                
                # Look for manual label
                label = data.get("label", data.get("feedback_label", "unknown"))
                
                if label not in ("good", "bad"):
                    continue
                
                # Extract score breakdown
                score_breakdown = data.get("score_breakdown", data.get("used_features", {}))
                final_score = data.get("final_score", 0.0)
                symbol = data.get("symbol", "UNKNOWN")
                
                if not score_breakdown:
                    continue
                
                entry = {
                    "label": label,
                    "score_breakdown": score_breakdown,
                    "final_score": final_score,
                    "symbol": symbol,
                    "line_number": line_num
                }
                
                parsed_entries.append(entry)
                
                if label == "good":
                    good_count += 1
                else:
                    bad_count += 1
                    
            except json.JSONDecodeError:
                # Try to extract info from non-JSON log lines
                if "label:" in line:
                    continue  # Skip for now, could implement text parsing
                    
    except Exception as e:
        print(f"❌ [TJDE TRAINER] Error parsing logs: {e}")
    
    print(f"[TJDE TRAINER] Parsed {len(parsed_entries)} feedback entries (Good: {good_count}, Bad: {bad_count})")
    
    return parsed_entries, good_count, bad_count


def calculate_weight_adjustments(feedback_entries: List[Dict], learning_rate: float = 0.1) -> Dict[str, float]:
    """
    Oblicza korekty wag na podstawie feedback
    
    Args:
        feedback_entries: Lista parsed feedback entries
        learning_rate: Współczynnik uczenia (0.1 = 10% adjustment)
        
    Returns:
        Dict z korektami dla każdej wagi
    """
    adjustments = defaultdict(float)
    feature_impact = defaultdict(list)
    
    try:
        # Collect feature values for good vs bad alerts
        for entry in feedback_entries:
            label = entry["label"]
            score_breakdown = entry["score_breakdown"]
            multiplier = 1.0 if label == "good" else -1.0
            
            for feature_name, feature_value in score_breakdown.items():
                if feature_name in DEFAULT_WEIGHTS:
                    # Weight adjustment proportional to feature value and outcome
                    impact = multiplier * float(feature_value)
                    feature_impact[feature_name].append(impact)
        
        # Calculate average impact for each feature
        for feature_name in DEFAULT_WEIGHTS.keys():
            if feature_name in feature_impact and feature_impact[feature_name]:
                avg_impact = sum(feature_impact[feature_name]) / len(feature_impact[feature_name])
                # Apply learning rate
                adjustments[feature_name] = avg_impact * learning_rate
            else:
                adjustments[feature_name] = 0.0
        
        print(f"[TJDE TRAINER] Calculated adjustments for {len(adjustments)} features")
        
    except Exception as e:
        print(f"❌ [TJDE TRAINER] Error calculating adjustments: {e}")
    
    return dict(adjustments)


def update_weights_based_on_feedback(learning_rate: float = 0.1, min_weight: float = 0.05, max_weight: float = 0.4) -> bool:
    """
    Główna funkcja aktualizacji wag na podstawie feedback
    
    Args:
        learning_rate: Współczynnik uczenia
        min_weight: Minimalna waga
        max_weight: Maksymalna waga
        
    Returns:
        bool: True jeśli wagi zostały zaktualizowane
    """
    try:
        print(f"\n[TJDE TRAINER] Starting weight update process...")
        
        # Parse feedback logs
        feedback_entries, good_count, bad_count = parse_feedback_logs()
        
        if not feedback_entries:
            print("⚠️ [TJDE TRAINER] No feedback data found for training")
            return False
        
        if good_count == 0 and bad_count == 0:
            print("⚠️ [TJDE TRAINER] No labeled feedback found")
            return False
        
        # Load current weights
        current_weights = load_weights()
        
        # Calculate adjustments
        adjustments = calculate_weight_adjustments(feedback_entries, learning_rate)
        
        # Apply adjustments
        new_weights = {}
        for feature_name in current_weights.keys():
            adjustment = adjustments.get(feature_name, 0.0)
            new_weight = current_weights[feature_name] + adjustment
            
            # Apply bounds
            new_weight = max(min_weight, min(new_weight, max_weight))
            new_weights[feature_name] = new_weight
        
        # Training info for logging
        training_info = {
            "feedback_entries_used": len(feedback_entries),
            "good_feedback": good_count,
            "bad_feedback": bad_count,
            "learning_rate": learning_rate,
            "adjustments_applied": adjustments,
            "training_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Save updated weights
        save_weights(new_weights, training_info)
        
        # Log training history
        _save_training_history(current_weights, new_weights, training_info)
        
        # Print summary
        _print_training_summary(current_weights, new_weights, adjustments, training_info)
        
        return True
        
    except Exception as e:
        print(f"❌ [TJDE TRAINER] Error in weight update: {e}")
        import traceback
        traceback.print_exc()
        return False


def _save_training_history(old_weights: Dict, new_weights: Dict, training_info: Dict):
    """Zapisuje historię treningów"""
    try:
        os.makedirs(os.path.dirname(FEEDBACK_HISTORY), exist_ok=True)
        
        # Load existing history
        if os.path.exists(FEEDBACK_HISTORY):
            with open(FEEDBACK_HISTORY, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = {"training_sessions": []}
        
        # Add new session
        session = {
            "timestamp": training_info["training_timestamp"],
            "old_weights": old_weights,
            "new_weights": new_weights,
            "training_info": training_info
        }
        
        history["training_sessions"].append(session)
        
        # Keep only last 50 sessions
        if len(history["training_sessions"]) > 50:
            history["training_sessions"] = history["training_sessions"][-50:]
        
        history["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        with open(FEEDBACK_HISTORY, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"⚠️ [TJDE TRAINER] Error saving training history: {e}")


def _print_training_summary(old_weights: Dict, new_weights: Dict, adjustments: Dict, training_info: Dict):
    """Wyświetla podsumowanie treningu"""
    print(f"\n📊 TJDE TRAINING SUMMARY")
    print("=" * 60)
    print(f"📈 Feedback Data: {training_info['good_feedback']} good, {training_info['bad_feedback']} bad")
    print(f"🎯 Learning Rate: {training_info['learning_rate']}")
    print(f"📅 Training Time: {training_info['training_timestamp']}")
    
    print(f"\n📋 WEIGHT CHANGES:")
    print(f"{'Feature':<25} {'Old Weight':<12} {'New Weight':<12} {'Change':<12}")
    print("-" * 65)
    
    for feature in old_weights:
        old_val = old_weights[feature]
        new_val = new_weights[feature]
        change = new_val - old_val
        
        change_str = f"{change:+.4f}" if change != 0 else "0.0000"
        arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        print(f"{feature:<25} {old_val:<12.4f} {new_val:<12.4f} {change_str:<12} {arrow}")


def add_manual_feedback(symbol: str, decision: str, score: float, label: str, reason: str = ""):
    """
    Dodaje ręczny feedback do logów
    
    Args:
        symbol: Symbol który był analizowany
        decision: Decyzja systemu (join_trend/consider_entry/avoid)
        score: Final score
        label: "good" lub "bad"
        reason: Opcjonalny powód oceny
    """
    try:
        os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
        
        feedback_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "decision": decision,
            "final_score": score,
            "label": label,
            "feedback_reason": reason,
            "feedback_type": "manual",
            "score_breakdown": {}  # Would need actual breakdown from original analysis
        }
        
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(feedback_entry, ensure_ascii=False)}\n")
        
        print(f"[TJDE TRAINER] Manual feedback added: {symbol} -> {label}")
        
    except Exception as e:
        print(f"❌ [TJDE TRAINER] Error adding manual feedback: {e}")


if __name__ == "__main__":
    print("🧠 TJDE Feedback Loop Trainer")
    print("=" * 50)
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            print("🎯 Starting training based on feedback logs...")
            success = update_weights_based_on_feedback()
            if success:
                print("✅ Training completed successfully")
            else:
                print("❌ Training failed")
                
        elif command == "show":
            print("📊 Current weights:")
            weights = load_weights()
            for feature, weight in weights.items():
                print(f"  {feature}: {weight:.4f}")
                
        elif command == "reset":
            print("🔄 Resetting to default weights...")
            save_weights(DEFAULT_WEIGHTS)
            print("✅ Weights reset to defaults")
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: train, show, reset")
    else:
        print("Available commands:")
        print("  train - Run training on feedback logs")
        print("  show  - Show current weights")
        print("  reset - Reset to default weights")
        print("\nExample: python -m utils.feedback_loop_trainer train")