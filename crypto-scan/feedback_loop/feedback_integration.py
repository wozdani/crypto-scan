"""
Feedback Integration - Module 5 Feedback Loop
Integruje system feedback loop z głównym silnikiem TJDE
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
from .feedback_cache import save_prediction, get_pending_evaluations, mark_as_evaluated, cleanup_old_predictions
from .feedback_evaluator import evaluate_predictions_batch, calculate_label_performance
from .weight_adjuster import adjust_weights_batch, get_weight_statistics

def log_prediction_for_feedback(symbol: str, tjde_score: float, decision: str, setup_label: str, confidence: float, price: float):
    """
    Rejestruje predykcję dla systemu feedback loop z diagnostyką
    
    Args:
        symbol: Symbol trading
        tjde_score: Final TJDE score
        decision: Decyzja systemu
        setup_label: Label setupu z AI
        confidence: Confidence AI
        price: Aktualna cena
    """
    try:
        # Sprawdź czy powinna być logowana z diagnostyką
        if not should_log_prediction(tjde_score, decision, setup_label, confidence):
            print(f"[FEEDBACK SKIP] {symbol}: Skipped logging – Score: {tjde_score}, Decision: {decision}, Setup: {setup_label}, Confidence: {confidence}")
            return
        
        # Przygotuj dane AI label dla kompatybilności
        ai_label = {
            'label': setup_label,
            'confidence': confidence
        }
        
        # Zapisz predykcję
        timestamp = datetime.now().isoformat()
        market_phase = "basic_screening"  # Default phase
        success = save_prediction(symbol, timestamp, ai_label, price, 
                                tjde_score, decision, market_phase)
        
        if success:
            print(f"[FEEDBACK LOG] {symbol}: Logged prediction – Score: {tjde_score}, Decision: {decision}, Setup: {setup_label}, Confidence: {confidence}")
        else:
            print(f"[FEEDBACK ERROR] {symbol}: Failed to save prediction")
            
    except Exception as e:
        print(f"[FEEDBACK INTEGRATION ERROR] {symbol}: Failed to log prediction - {e}")

def should_log_prediction(tjde_score: float, decision: str, setup_label: str, confidence: float) -> bool:
    """
    Określa, czy predykcja powinna zostać zapisana do feedbacku na podstawie jej siły i jakości.
    """
    # Ignoruj nieistotne lub słabe dane
    if setup_label in ["unknown", "undefined", "error"]:
        return False
    if confidence < 0.4:
        return False
    if decision not in ["enter", "avoid"]:
        return False
    if tjde_score >= 0.65 and confidence >= 0.5:
        return True
    if 0.50 <= tjde_score < 0.65 and confidence >= 0.6:
        return True
    if "reversal" in setup_label and tjde_score >= 0.55:
        return True
    return False

async def run_feedback_evaluation_cycle() -> Dict:
    """
    Uruchamia cykl ewaluacji feedback loop
    
    Returns:
        Wyniki cyklu ewaluacji
    """
    try:
        logging.info("[FEEDBACK INTEGRATION] Starting feedback evaluation cycle")
        
        # 1. Pobierz predykcje oczekujące na ewaluację (starsze niż 2h)
        pending_predictions = get_pending_evaluations(hours_old=2)
        
        if not pending_predictions:
            logging.info("[FEEDBACK INTEGRATION] No predictions pending evaluation")
            return {
                "status": "success",
                "evaluated_count": 0,
                "weight_adjustments": {},
                "message": "No predictions to evaluate"
            }
        
        logging.info(f"[FEEDBACK INTEGRATION] Found {len(pending_predictions)} predictions to evaluate")
        
        # 2. Oceń predykcje
        evaluation_results = evaluate_predictions_batch(pending_predictions)
        
        if not evaluation_results:
            logging.warning("[FEEDBACK INTEGRATION] No successful evaluations")
            return {
                "status": "partial_success",
                "evaluated_count": 0,
                "weight_adjustments": {},
                "message": "Price fetch failed for all predictions"
            }
        
        # 3. Oznacz jako ocenione
        marked_count = 0
        for result in evaluation_results:
            success = mark_as_evaluated(
                result["symbol"], 
                result["timestamp"], 
                result
            )
            if success:
                marked_count += 1
        
        # 4. Dostosuj wagi na podstawie wyników
        weight_adjustments = adjust_weights_batch(evaluation_results, learning_rate=0.01)
        
        # 5. Oblicz statystyki skuteczności
        performance_stats = calculate_label_performance(evaluation_results)
        
        # 6. Wyczyść stare predykcje (starsze niż 30 dni)
        cleanup_count = cleanup_old_predictions(days_old=30)
        
        logging.info(f"[FEEDBACK INTEGRATION] Evaluation cycle complete: "
                    f"{len(evaluation_results)} evaluated, {len(weight_adjustments)} weights adjusted")
        
        return {
            "status": "success",
            "evaluated_count": len(evaluation_results),
            "marked_count": marked_count,
            "weight_adjustments": weight_adjustments,
            "performance_stats": performance_stats,
            "cleanup_count": cleanup_count,
            "message": f"Successfully evaluated {len(evaluation_results)} predictions"
        }
        
    except Exception as e:
        logging.error(f"[FEEDBACK INTEGRATION ERROR] Evaluation cycle failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "evaluated_count": 0,
            "weight_adjustments": {}
        }

def get_feedback_system_status() -> Dict:
    """
    Zwraca status systemu feedback loop
    
    Returns:
        Status systemu
    """
    try:
        # Pobierz podstawowe statystyki
        from .feedback_cache import get_cache_statistics
        cache_stats = get_cache_statistics()
        weight_stats = get_weight_statistics()
        
        # Sprawdź czy są predykcje oczekujące
        pending_2h = get_pending_evaluations(hours_old=2)
        pending_6h = get_pending_evaluations(hours_old=6)
        
        return {
            "system_active": True,
            "cache_statistics": cache_stats,
            "weight_statistics": weight_stats,
            "pending_evaluations": {
                "2h_old": len(pending_2h),
                "6h_old": len(pending_6h)
            },
            "last_check": datetime.now().isoformat(),
            "recommendations": generate_system_recommendations(cache_stats, weight_stats)
        }
        
    except Exception as e:
        return {
            "system_active": False,
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }

def generate_system_recommendations(cache_stats: Dict, weight_stats: Dict) -> List[str]:
    """
    Generuje rekomendacje dla systemu feedback loop
    
    Args:
        cache_stats: Statystyki cache
        weight_stats: Statystyki wag
        
    Returns:
        Lista rekomendacji
    """
    recommendations = []
    
    # Analiza cache
    total_predictions = cache_stats.get("total_predictions", 0)
    evaluated_predictions = cache_stats.get("evaluated_predictions", 0)
    success_rate = cache_stats.get("success_rate", 0)
    
    if total_predictions < 50:
        recommendations.append("Need more prediction data for reliable learning")
    
    if evaluated_predictions > 0 and success_rate < 40:
        recommendations.append("Low success rate - consider adjusting prediction criteria")
    elif success_rate > 75:
        recommendations.append("High success rate - system performing well")
    
    # Analiza wag
    low_performance_count = weight_stats.get("low_performance_labels", 0)
    high_performance_count = weight_stats.get("high_performance_labels", 0)
    
    if low_performance_count > 5:
        recommendations.append(f"Consider filtering {low_performance_count} low-performing labels")
    
    if high_performance_count > 3:
        recommendations.append(f"Focus on {high_performance_count} high-performing labels")
    
    # Sprawdź pending evaluations
    pending = cache_stats.get("pending_evaluations", 0)
    if pending > 20:
        recommendations.append("Many pending evaluations - run evaluation cycle")
    
    if not recommendations:
        recommendations.append("System operating normally")
    
    return recommendations

async def scheduled_feedback_maintenance():
    """
    Zaplanowana konserwacja systemu feedback loop
    """
    try:
        logging.info("[FEEDBACK INTEGRATION] Starting scheduled maintenance")
        
        # Uruchom cykl ewaluacji
        evaluation_result = await run_feedback_evaluation_cycle()
        
        # Wyczyść stare dane
        cleanup_count = cleanup_old_predictions(days_old=60)
        
        # Sprawdź status systemu
        system_status = get_feedback_system_status()
        
        logging.info(f"[FEEDBACK INTEGRATION] Maintenance complete: "
                    f"evaluated {evaluation_result.get('evaluated_count', 0)}, "
                    f"cleaned {cleanup_count} old predictions")
        
        return {
            "maintenance_completed": True,
            "evaluation_result": evaluation_result,
            "cleanup_count": cleanup_count,
            "system_status": system_status
        }
        
    except Exception as e:
        logging.error(f"[FEEDBACK INTEGRATION ERROR] Maintenance failed: {e}")
        return {
            "maintenance_completed": False,
            "error": str(e)
        }

def initialize_feedback_system() -> bool:
    """
    Inicializuje system feedback loop
    
    Returns:
        True jeśli inicjalizacja zakończona pomyślnie
    """
    try:
        from .weight_adjuster import load_label_weights
        
        # Wczytaj/utwórz wagi etykiet
        weights = load_label_weights()
        
        logging.info(f"[FEEDBACK INTEGRATION] Feedback system initialized with {len(weights)} label weights")
        return True
        
    except Exception as e:
        logging.error(f"[FEEDBACK INTEGRATION ERROR] Failed to initialize feedback system: {e}")
        return False

# Harmonogram automatyczny (można dodać do cron job lub scheduler)
def should_run_evaluation_cycle() -> bool:
    """
    Sprawdza czy powinien zostać uruchomiony cykl ewaluacji
    
    Returns:
        True jeśli powinien zostać uruchomiony
    """
    try:
        pending_predictions = get_pending_evaluations(hours_old=2)
        
        # Uruchom jeśli jest > 5 predykcji oczekujących
        return len(pending_predictions) > 5
        
    except Exception as e:
        logging.error(f"[FEEDBACK INTEGRATION ERROR] Failed to check evaluation cycle: {e}")
        return False


def update_label_weights_from_performance() -> Dict:
    """
    Aktualizuje wagi etykiet na podstawie historii skuteczności
    
    Returns:
        Słownik z aktualizacjami wag
    """
    try:
        from .feedback_cache import get_evaluation_history
        
        # Pobierz historię ewaluacji z ostatnich 30 dni
        history = get_evaluation_history(days=30)
        
        if len(history) < 10:
            print("[WEIGHT UPDATE] Insufficient evaluation history for weight adjustment")
            return {}
        
        # Grupuj wyniki według etykiet
        label_performance = {}
        for entry in history:
            label = entry.get("label", "unknown")
            evaluation_result = entry.get("evaluation_result", {})
            successful = evaluation_result.get("successful", False)
            
            if label not in label_performance:
                label_performance[label] = {
                    "total": 0,
                    "successful": 0,
                    "success_rate": 0.0
                }
            
            label_performance[label]["total"] += 1
            if successful:
                label_performance[label]["successful"] += 1
        
        # Oblicz success rate dla każdej etykiety
        for label, stats in label_performance.items():
            stats["success_rate"] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Aktualizuj wagi na podstawie skuteczności
        from .weight_adjuster import load_label_weights, save_label_weights
        current_weights = load_label_weights()
        updated_weights = {}
        weight_changes = {}
        
        for label, stats in label_performance.items():
            if stats["total"] >= 5:  # Minimum 5 predykcji dla wiarygodności
                success_rate = stats["success_rate"]
                
                # Nowa waga: 0.5 + success_rate (range 0.5-1.5)
                new_weight = 0.5 + success_rate
                old_weight = current_weights.get(label, 1.0)
                
                # Smooth transition: 70% nowa waga, 30% stara waga
                final_weight = 0.7 * new_weight + 0.3 * old_weight
                
                updated_weights[label] = round(final_weight, 3)
                weight_changes[label] = {
                    "old_weight": round(old_weight, 3),
                    "new_weight": round(final_weight, 3),
                    "success_rate": round(success_rate, 3),
                    "sample_size": stats["total"]
                }
                
                print(f"[WEIGHT UPDATE] {label}: {old_weight:.3f} → {final_weight:.3f} (success: {success_rate:.1%}, n={stats['total']})")
        
        # Zachowaj istniejące wagi dla etykiet bez wystarczającej historii
        for label, weight in current_weights.items():
            if label not in updated_weights:
                updated_weights[label] = weight
        
        # Zapisz zaktualizowane wagi
        if weight_changes:
            save_label_weights(updated_weights)
            print(f"[WEIGHT UPDATE] Updated {len(weight_changes)} label weights based on performance")
        
        return weight_changes
        
    except Exception as e:
        print(f"[WEIGHT UPDATE ERROR] Failed to update label weights: {e}")
        return {}


def get_label_performance_report() -> Dict:
    """
    Generuje raport skuteczności etykiet
    
    Returns:
        Raport z analizą skuteczności
    """
    try:
        from .feedback_cache import get_evaluation_history
        from .weight_adjuster import load_label_weights
        
        # Pobierz dane z ostatnich 30 dni
        history = get_evaluation_history(days=30)
        current_weights = load_label_weights()
        
        if not history:
            return {
                "status": "no_data",
                "message": "No evaluation history available"
            }
        
        # Analiza per etykiety
        label_stats = {}
        total_predictions = len(history)
        total_successful = 0
        
        for entry in history:
            label = entry.get("label", "unknown")
            evaluation_result = entry.get("evaluation_result", {})
            successful = evaluation_result.get("successful", False)
            confidence = entry.get("confidence", 0.0)
            tjde_score = entry.get("tjde_score", 0.0)
            
            if successful:
                total_successful += 1
            
            if label not in label_stats:
                label_stats[label] = {
                    "total": 0,
                    "successful": 0,
                    "success_rate": 0.0,
                    "avg_confidence": 0.0,
                    "avg_tjde_score": 0.0,
                    "current_weight": current_weights.get(label, 1.0),
                    "confidence_sum": 0.0,
                    "tjde_sum": 0.0
                }
            
            stats = label_stats[label]
            stats["total"] += 1
            if successful:
                stats["successful"] += 1
            stats["confidence_sum"] += confidence
            stats["tjde_sum"] += tjde_score
        
        # Oblicz końcowe statystyki
        for label, stats in label_stats.items():
            if stats["total"] > 0:
                stats["success_rate"] = round(stats["successful"] / stats["total"], 3)
                stats["avg_confidence"] = round(stats["confidence_sum"] / stats["total"], 3)
                stats["avg_tjde_score"] = round(stats["tjde_sum"] / stats["total"], 3)
                
                # Kategoria skuteczności
                if stats["success_rate"] >= 0.75:
                    stats["performance_category"] = "excellent"
                elif stats["success_rate"] >= 0.60:
                    stats["performance_category"] = "good"
                elif stats["success_rate"] >= 0.45:
                    stats["performance_category"] = "moderate"
                else:
                    stats["performance_category"] = "poor"
        
        # Sortuj według skuteczności
        sorted_labels = sorted(label_stats.items(), key=lambda x: x[1]["success_rate"], reverse=True)
        
        # Overall statistics
        overall_success_rate = total_successful / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "status": "success",
            "overall": {
                "total_predictions": total_predictions,
                "successful_predictions": total_successful,
                "success_rate": round(overall_success_rate, 3),
                "evaluation_period_days": 30
            },
            "label_performance": dict(sorted_labels),
            "top_performers": [label for label, stats in sorted_labels[:5] if stats["success_rate"] >= 0.60],
            "poor_performers": [label for label, stats in sorted_labels if stats["success_rate"] < 0.40 and stats["total"] >= 5],
            "report_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "report_timestamp": datetime.now().isoformat()
        }

def load_dynamic_weights(symbol: str = None) -> Dict[str, float]:
    """
    Load dynamic scoring weights from feedback loop system for TJDE v2 integration
    
    Args:
        symbol: Trading symbol (optional, for symbol-specific weights)
        
    Returns:
        Dict with dynamic TJDE component weights or empty dict if unavailable
    """
    try:
        from .weight_adjustment_system import load_current_weights
        
        weights_data = load_current_weights()
        
        # Handle structured weight file format
        if isinstance(weights_data, dict) and "weights" in weights_data:
            weights = weights_data["weights"]
        else:
            weights = weights_data
        
        # Ensure weights is a valid dict
        if not isinstance(weights, dict):
            print(f"[DYNAMIC WEIGHTS] Invalid weights format - using fallback")
            return {}
        
        # Map AI label weights to TJDE component weights
        tjde_weights = {}
        
        # Extract weights for TJDE components based on AI labels
        if weights:
            # Calculate average weight for different pattern types
            bullish_patterns = ["breakout_pattern", "momentum_follow", "trend_continuation", "pullback"]
            bearish_patterns = ["reversal_pattern", "exhaustion", "resistance_rejection"]
            neutral_patterns = ["consolidation", "range", "sideways"]
            
            # Get weights that exist in our data
            bullish_weights = [weights.get(p, 1.0) for p in bullish_patterns if p in weights]
            bearish_weights = [weights.get(p, 1.0) for p in bearish_patterns if p in weights]
            neutral_weights = [weights.get(p, 1.0) for p in neutral_patterns if p in weights]
            
            # Calculate averages with fallback to 1.0
            bullish_avg = sum(bullish_weights) / len(bullish_weights) if bullish_weights else 1.0
            bearish_avg = sum(bearish_weights) / len(bearish_weights) if bearish_weights else 1.0
            neutral_avg = sum(neutral_weights) / len(neutral_weights) if neutral_weights else 1.0
            
            # Map to TJDE components with base weights
            tjde_weights = {
                "trend": bullish_avg * 0.3,           # Base trend weight adjusted by bullish pattern success
                "volume": (bullish_avg + neutral_avg) / 2 * 0.2,  # Volume weight from mixed patterns
                "momentum": bullish_avg * 0.2,        # Momentum from bullish patterns
                "orderbook": neutral_avg * 0.1,       # Orderbook from neutral patterns
                "price_change": (bullish_avg + bearish_avg) / 2 * 0.2  # Price change from all patterns
            }
            
            print(f"[DYNAMIC WEIGHTS] Loaded for {symbol or 'global'}: trend={tjde_weights['trend']:.3f}, "
                  f"volume={tjde_weights['volume']:.3f}, momentum={tjde_weights['momentum']:.3f}, "
                  f"orderbook={tjde_weights['orderbook']:.3f}, price_change={tjde_weights['price_change']:.3f}")
        
        return tjde_weights
        
    except ImportError:
        print(f"[DYNAMIC WEIGHTS] Weight adjustment module not available - using fallback")
        return {}
    except Exception as e:
        print(f"[DYNAMIC WEIGHTS ERROR] Failed to load weights: {e}")
        return {}