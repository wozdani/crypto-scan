"""
Dynamic Label Weight Adjustment System
Automatically adjusts AI pattern weights based on prediction success rates
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
from .feedback_cache import get_evaluation_history

WEIGHTS_PATH = "feedback_loop/label_weights.json"

def adjust_label_weights() -> Dict[str, float]:
    """
    Główna funkcja dostrajania wag na podstawie skuteczności predykcji
    
    Returns:
        Dict z nowymi wagami dla każdego labela
    """
    print("[WEIGHT ADJUSTMENT] Starting dynamic label weight adjustment...")
    
    # Pobierz historię ewaluacji z ostatnich 7 dni
    evaluations = get_evaluation_history(days=7)
    print(f"[WEIGHT ADJUSTMENT] Analyzing {len(evaluations)} evaluations from last 7 days")
    
    if not evaluations:
        print("[WEIGHT ADJUSTMENT] No evaluations found - using default weights")
        return load_current_weights()
    
    # Grupuj ewaluacje po labelach i oblicz statystyki
    stats = {}
    
    for entry in evaluations:
        label = entry.get("label", "unknown")
        evaluation_result = entry.get("evaluation_result", {})
        successful = evaluation_result.get("successful", False)
        
        if label not in stats:
            stats[label] = {"total": 0, "success": 0}
        
        stats[label]["total"] += 1
        if successful:
            stats[label]["success"] += 1
    
    print(f"[WEIGHT ADJUSTMENT] Statistics for {len(stats)} labels:")
    for label, stat in stats.items():
        success_rate = stat["success"] / stat["total"] if stat["total"] > 0 else 0
        print(f"  {label}: {stat['success']}/{stat['total']} ({success_rate:.1%})")
    
    # Załaduj istniejące wagi
    weights = load_current_weights()
    print(f"[WEIGHT ADJUSTMENT] Current weights loaded: {len(weights)} entries")
    
    # Dostrajaj wagi na podstawie skuteczności
    updated_count = 0
    
    for label, stat in stats.items():
        total = stat["total"]
        success = stat["success"]
        
        if total == 0:
            continue
            
        success_rate = success / total
        prev_weight = weights.get(label, 1.0)
        
        # Zasady dostrajania wag
        if success_rate > 0.7:
            # Wzrost wagi dla skutecznych setupów
            new_weight = min(prev_weight + 0.1, 2.0)
            adjustment_type = "BOOST"
        elif success_rate < 0.5:
            # Spadek wagi dla nieskutecznych setupów
            new_weight = max(prev_weight - 0.1, 0.1)
            adjustment_type = "PENALTY"
        else:
            # Stabilizacja przez wygładzanie (EMA-like)
            new_weight = prev_weight * 0.98 + 0.02 * 1.0
            adjustment_type = "STABILIZE"
        
        # Zapisz nową wagę
        old_weight = weights.get(label, 1.0)
        weights[label] = round(new_weight, 3)
        updated_count += 1
        
        print(f"[WEIGHT {adjustment_type}] {label}: {old_weight:.3f} → {new_weight:.3f} (success: {success_rate:.1%})")
    
    # Zapisz zaktualizowane wagi
    save_weights(weights)
    
    print(f"[WEIGHT ADJUSTMENT] ✅ Updated {updated_count} label weights")
    return weights

def load_current_weights() -> Dict[str, float]:
    """
    Załaduj istniejące wagi z pliku lub użyj domyślnych
    
    Returns:
        Dict z wagami dla każdego labela
    """
    if os.path.exists(WEIGHTS_PATH):
        try:
            with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
                weights = json.load(f)
            print(f"[WEIGHT LOADER] Loaded {len(weights)} weights from {WEIGHTS_PATH}")
            return weights
        except Exception as e:
            print(f"[WEIGHT LOADER ERROR] Failed to load weights: {e}")
            return {}
    else:
        print(f"[WEIGHT LOADER] No weights file found - using defaults")
        return {}

def save_weights(weights: Dict[str, float]) -> bool:
    """
    Zapisz wagi do pliku JSON
    
    Args:
        weights: Dict z wagami do zapisania
        
    Returns:
        True jeśli zapisano pomyślnie
    """
    try:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        
        # Dodaj metadane
        output_data = {
            "last_updated": datetime.now().isoformat(),
            "total_labels": len(weights),
            "weights": weights
        }
        
        with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[WEIGHT SAVER] ✅ Saved {len(weights)} weights to {WEIGHTS_PATH}")
        return True
        
    except Exception as e:
        print(f"[WEIGHT SAVER ERROR] Failed to save weights: {e}")
        return False

def get_label_weight(label: str) -> float:
    """
    Pobierz wagę dla konkretnego labela
    
    Args:
        label: Label do sprawdzenia
        
    Returns:
        Waga dla labela (1.0 jako domyślna)
    """
    weights_data = load_current_weights()
    
    # Jeśli plik ma strukturę z metadanymi
    if isinstance(weights_data, dict) and "weights" in weights_data:
        weights = weights_data["weights"]
    else:
        weights = weights_data
    
    # Ensure weights is a dict
    if not isinstance(weights, dict):
        return 1.0
    
    return weights.get(label, 1.0)

def get_weight_statistics() -> Dict:
    """
    Pobierz statystyki wag dla raportowania
    
    Returns:
        Dict ze statystykami wag
    """
    weights_data = load_current_weights()
    
    if isinstance(weights_data, dict) and "weights" in weights_data:
        weights = weights_data["weights"]
        last_updated = weights_data.get("last_updated", "unknown")
    else:
        weights = weights_data
        last_updated = "unknown"
    
    # Ensure weights is a dict
    if not isinstance(weights, dict) or not weights:
        return {
            "total_labels": 0,
            "last_updated": last_updated,
            "weight_distribution": {},
            "performance_categories": {
                "excellent": [],
                "good": [],
                "poor": []
            }
        }
    
    # Kategoryzuj wagi według wydajności
    performance_categories = {
        "excellent": [],  # waga > 1.2
        "good": [],       # waga 1.0-1.2
        "poor": []        # waga < 0.6
    }
    
    for label, weight in weights.items():
        if isinstance(weight, (int, float)):
            if weight > 1.2:
                performance_categories["excellent"].append(f"{label} ({weight:.2f})")
            elif weight >= 1.0:
                performance_categories["good"].append(f"{label} ({weight:.2f})")
            elif weight < 0.6:
                performance_categories["poor"].append(f"{label} ({weight:.2f})")
    
    # Rozkład wag
    valid_weights = [w for w in weights.values() if isinstance(w, (int, float))]
    weight_distribution = {
        "above_1.5": len([w for w in valid_weights if w > 1.5]),
        "1.0_to_1.5": len([w for w in valid_weights if 1.0 <= w <= 1.5]),
        "0.5_to_1.0": len([w for w in valid_weights if 0.5 <= w < 1.0]),
        "below_0.5": len([w for w in valid_weights if w < 0.5])
    }
    
    return {
        "total_labels": len(weights),
        "last_updated": last_updated,
        "weight_distribution": weight_distribution,
        "performance_categories": performance_categories,
        "average_weight": sum(valid_weights) / len(valid_weights) if valid_weights else 1.0
    }

def print_weight_report():
    """
    Wydrukuj raport z wagami i statystykami
    """
    stats = get_weight_statistics()
    
    print("\n" + "="*50)
    print("📊 DYNAMIC LABEL WEIGHTS REPORT")
    print("="*50)
    print(f"Total labels: {stats['total_labels']}")
    print(f"Last updated: {stats['last_updated']}")
    print(f"Average weight: {stats['average_weight']:.3f}")
    
    print("\n📈 PERFORMANCE CATEGORIES:")
    print(f"Excellent (>1.2): {len(stats['performance_categories']['excellent'])}")
    for label in stats['performance_categories']['excellent']:
        print(f"  ✅ {label}")
    
    print(f"Good (1.0-1.2): {len(stats['performance_categories']['good'])}")
    for label in stats['performance_categories']['good']:
        print(f"  ⚡ {label}")
    
    print(f"Poor (<0.6): {len(stats['performance_categories']['poor'])}")
    for label in stats['performance_categories']['poor']:
        print(f"  ⚠️ {label}")
    
    print("\n📊 WEIGHT DISTRIBUTION:")
    dist = stats['weight_distribution']
    print(f"Above 1.5: {dist['above_1.5']}")
    print(f"1.0 to 1.5: {dist['1.0_to_1.5']}")
    print(f"0.5 to 1.0: {dist['0.5_to_1.0']}")
    print(f"Below 0.5: {dist['below_0.5']}")
    print("="*50)

# Test function
if __name__ == "__main__":
    print("Testing Dynamic Label Weight Adjustment System...")
    weights = adjust_label_weights()
    print_weight_report()