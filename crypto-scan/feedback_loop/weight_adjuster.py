"""
Weight Adjuster - Module 5 Feedback Loop
Dynamiczna korekta wag etykiet na podstawie historycznej skuteczności
"""

import json
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Ścieżka do pliku wag
WEIGHTS_PATH = "feedback_loop/label_weights.json"

# Domyślne wagi etykiet
DEFAULT_LABEL_WEIGHTS = {
    "pullback": 1.0,
    "pullback_continuation": 1.0,
    "breakout": 1.0,
    "breakout_pattern": 1.0,
    "momentum_follow": 1.0,
    "trend_continuation": 1.0,
    "reversal": 1.0,
    "reversal_pattern": 1.0,
    "range": 1.0,
    "consolidation": 1.0,
    "consolidation_squeeze": 1.0,
    "chaos": 1.0,
    "exhaustion": 1.0,
    "support_bounce": 1.0,
    "resistance_rejection": 1.0,
    "pullback_in_trend": 1.0,
    "unknown": 0.8,
    "no_clear_pattern": 0.7,
    "setup_analysis": 0.5
}

def load_label_weights() -> Dict[str, float]:
    """
    Wczytuje aktualne wagi etykiet z pliku
    
    Returns:
        Słownik z wagami etykiet
    """
    try:
        if os.path.exists(WEIGHTS_PATH):
            with open(WEIGHTS_PATH, "r", encoding='utf-8') as f:
                weights = json.load(f)
            
            # Dodaj brakujące domyślne wagi
            for label, default_weight in DEFAULT_LABEL_WEIGHTS.items():
                if label not in weights:
                    weights[label] = default_weight
            
            return weights
        else:
            # Utwórz plik z domyślnymi wagami
            save_label_weights(DEFAULT_LABEL_WEIGHTS)
            return DEFAULT_LABEL_WEIGHTS.copy()
            
    except Exception as e:
        logging.error(f"[WEIGHT ADJUSTER ERROR] Failed to load weights: {e}")
        return DEFAULT_LABEL_WEIGHTS.copy()

def save_label_weights(weights: Dict[str, float]) -> bool:
    """
    Zapisuje wagi etykiet do pliku
    
    Args:
        weights: Słownik z wagami etykiet
        
    Returns:
        True jeśli zapisano pomyślnie
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        
        # Dodaj timestamp ostatniej aktualizacji
        weights_with_metadata = {
            "weights": weights,
            "last_updated": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        with open(WEIGHTS_PATH, "w", encoding='utf-8') as f:
            json.dump(weights_with_metadata, f, indent=2, ensure_ascii=False)
            
        logging.info(f"[WEIGHT ADJUSTER] Saved {len(weights)} label weights")
        return True
        
    except Exception as e:
        logging.error(f"[WEIGHT ADJUSTER ERROR] Failed to save weights: {e}")
        return False

def adjust_weight_single(label: str, success: bool, confidence: float = 1.0, 
                        learning_rate: float = 0.01) -> float:
    """
    Dostosowuje wagę pojedynczej etykiety na podstawie sukcesu
    
    Args:
        label: Nazwa etykiety
        success: Czy predykcja była skuteczna
        confidence: Poziom pewności predykcji (0.0-1.0)
        learning_rate: Szybkość uczenia (default 0.01)
        
    Returns:
        Nowa waga etykiety
    """
    weights = load_label_weights()
    current_weight = weights.get(label, 1.0)
    
    # Dostosowanie based on success and confidence
    if success:
        # Zwiększ wagę, ale z uwzględnieniem confidence
        adjustment = learning_rate * confidence
        new_weight = current_weight + adjustment
    else:
        # Zmniejsz wagę, penalty jest większy dla wysokiego confidence
        penalty = learning_rate * 2.0 * confidence  # Penalty x2 dla błędnych predykcji
        new_weight = current_weight - penalty
    
    # Bounds checking - wagi między 0.2 a 1.5
    new_weight = max(0.2, min(1.5, new_weight))
    
    weights[label] = round(new_weight, 4)
    save_label_weights(weights)
    
    logging.info(f"[WEIGHT ADJUSTER] {label}: {current_weight:.3f} → {new_weight:.3f} "
                f"({'SUCCESS' if success else 'FAILED'}, conf: {confidence:.2f})")
    
    return new_weight

def adjust_weights_batch(evaluation_results: List[Dict], learning_rate: float = 0.01) -> Dict[str, float]:
    """
    Dostosowuje wagi na podstawie batch wyników ewaluacji
    
    Args:
        evaluation_results: Lista wyników ewaluacji
        learning_rate: Szybkość uczenia
        
    Returns:
        Słownik z nowymi wagami
    """
    weights = load_label_weights()
    weight_adjustments = {}
    
    # Grupuj wyniki według labeli
    label_results = {}
    for result in evaluation_results:
        # Pobierz label z różnych możliwych źródeł
        label = None
        if 'original_prediction' in result:
            label = result['original_prediction'].get('label')
        elif 'label' in result:
            label = result['label']
        
        if not label or label == "unknown":
            continue
            
        if label not in label_results:
            label_results[label] = []
        
        label_results[label].append(result)
    
    # Dostosuj wagi per label
    for label, results in label_results.items():
        if not results:
            continue
            
        # Oblicz średnią skuteczność i confidence dla labela
        successful_count = sum(1 for r in results if r.get('successful', False))
        total_count = len(results)
        success_rate = successful_count / total_count
        
        avg_confidence = sum(r.get('confidence', 0.0) for r in results) / total_count
        
        current_weight = weights.get(label, 1.0)
        
        # Stronger adjustment dla większej liczby próbek
        sample_strength = min(1.0, total_count / 10.0)  # Max strength przy 10+ próbkach
        
        if success_rate > 0.6:  # Good performance
            adjustment = learning_rate * sample_strength * avg_confidence * (success_rate - 0.5)
            new_weight = current_weight + adjustment
        else:  # Poor performance
            penalty = learning_rate * 2.0 * sample_strength * avg_confidence * (0.5 - success_rate)
            new_weight = current_weight - penalty
        
        # Bounds checking
        new_weight = max(0.2, min(1.5, new_weight))
        
        weights[label] = round(new_weight, 4)
        weight_adjustments[label] = {
            "old_weight": current_weight,
            "new_weight": new_weight,
            "success_rate": round(success_rate, 3),
            "sample_count": total_count,
            "avg_confidence": round(avg_confidence, 3)
        }
        
        logging.info(f"[WEIGHT ADJUSTER BATCH] {label}: {current_weight:.3f} → {new_weight:.3f} "
                    f"(success: {success_rate:.1%}, samples: {total_count})")
    
    # Zapisz zaktualizowane wagi
    save_label_weights(weights)
    
    return weight_adjustments

def get_effective_score_adjustment(label: str, base_adjustment: float, confidence: float) -> float:
    """
    Oblicza efektywny adjustment score z uwzględnieniem dynamicznych wag
    
    Args:
        label: Nazwa etykiety
        base_adjustment: Bazowy adjustment z AI-EYE
        confidence: Poziom pewności
        
    Returns:
        Dostosowany score adjustment
    """
    weights = load_label_weights()
    weight = weights.get(label, 1.0)
    
    # Zastosuj wagę do base adjustment
    effective_adjustment = base_adjustment * weight * confidence
    
    # Bounds checking - adjustment między -0.20 a +0.20
    effective_adjustment = max(-0.20, min(0.20, effective_adjustment))
    
    return round(effective_adjustment, 4)

def analyze_label_performance() -> Dict[str, Dict]:
    """
    Analizuje skuteczność etykiet na podstawie wag i historii
    
    Returns:
        Analiza skuteczności per label
    """
    weights = load_label_weights()
    analysis = {}
    
    for label, weight in weights.items():
        # Kategoryzuj skuteczność na podstawie wagi
        if weight >= 1.2:
            performance = "excellent"
        elif weight >= 1.0:
            performance = "good"
        elif weight >= 0.8:
            performance = "average"
        elif weight >= 0.6:
            performance = "below_average"
        else:
            performance = "poor"
        
        # Rekomendacje na podstawie wagi
        if weight < 0.5:
            recommendation = "Consider filtering out this label"
        elif weight < 0.7:
            recommendation = "Monitor closely, may need attention"
        elif weight > 1.3:
            recommendation = "High-value label, prioritize"
        else:
            recommendation = "Standard performance"
        
        analysis[label] = {
            "weight": weight,
            "performance": performance,
            "recommendation": recommendation,
            "confidence_multiplier": round(weight, 2)
        }
    
    return analysis

def reset_weights_to_default() -> bool:
    """
    Resetuje wszystkie wagi do wartości domyślnych
    
    Returns:
        True jeśli zresetowano pomyślnie
    """
    try:
        success = save_label_weights(DEFAULT_LABEL_WEIGHTS.copy())
        if success:
            logging.info("[WEIGHT ADJUSTER] Reset all weights to default values")
        return success
        
    except Exception as e:
        logging.error(f"[WEIGHT ADJUSTER ERROR] Failed to reset weights: {e}")
        return False

def get_weight_statistics() -> Dict:
    """
    Zwraca statystyki wag etykiet
    
    Returns:
        Statystyki wag
    """
    try:
        weights = load_label_weights()
        
        if not weights:
            return {"error": "No weights available"}
        
        weight_values = list(weights.values())
        
        return {
            "total_labels": len(weights),
            "average_weight": round(sum(weight_values) / len(weight_values), 3),
            "min_weight": min(weight_values),
            "max_weight": max(weight_values),
            "high_performance_labels": len([w for w in weight_values if w > 1.1]),
            "low_performance_labels": len([w for w in weight_values if w < 0.8]),
            "weights_distribution": {
                "excellent (>1.2)": len([w for w in weight_values if w > 1.2]),
                "good (1.0-1.2)": len([w for w in weight_values if 1.0 <= w <= 1.2]),
                "average (0.8-1.0)": len([w for w in weight_values if 0.8 <= w < 1.0]),
                "below_average (0.6-0.8)": len([w for w in weight_values if 0.6 <= w < 0.8]),
                "poor (<0.6)": len([w for w in weight_values if w < 0.6])
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

# Globalny dostęp do wag dla łatwego importu
def get_current_weights() -> Dict[str, float]:
    """Szybki dostęp do aktualnych wag"""
    return load_label_weights()

# Export dla backward compatibility
LABEL_WEIGHTS = get_current_weights()