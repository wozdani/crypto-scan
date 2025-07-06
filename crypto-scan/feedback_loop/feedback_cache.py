"""
Feedback Cache - Module 5 Feedback Loop
Zapisuje i odczytuje predykcje z przeszłości dla samouczenia systemu
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Ścieżki do plików cache
CACHE_PATH = "feedback_loop/history.json"
WEIGHTS_PATH = "feedback_loop/label_weights.json"

def save_prediction(symbol: str, timestamp: str, ai_label: Dict, current_price: float, 
                   tjde_score: float, decision: str, market_phase: str = "basic_screening") -> bool:
    """
    Zapisuje predykcję do cache dla późniejszej ewaluacji
    
    Args:
        symbol: Symbol trading
        timestamp: Timestamp predykcji
        ai_label: AI label z confidence
        current_price: Aktualna cena
        tjde_score: Final TJDE score
        decision: Decyzja systemu
        market_phase: Faza rynku
        
    Returns:
        True jeśli zapisano pomyślnie
    """
    try:
        entry = {
            "symbol": symbol,
            "timestamp": timestamp,
            "price": current_price,
            "label": ai_label.get("label", "unknown"),
            "confidence": ai_label.get("confidence", 0.0),
            "tjde_score": tjde_score,
            "decision": decision,
            "market_phase": market_phase,
            "evaluated": False,
            "evaluation_result": None,
            "price_after_2h": None,
            "price_after_6h": None
        }

        # Wczytaj istniejące dane
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r", encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []

        # Dodaj nowy wpis
        data.append(entry)
        
        # Zachowaj tylko ostatnie 1000 predykcji
        if len(data) > 1000:
            data = data[-1000:]

        # Zapisz do pliku
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logging.info(f"[FEEDBACK CACHE] Saved prediction for {symbol}: {ai_label.get('label')} (score: {tjde_score:.3f})")
        return True
        
    except Exception as e:
        logging.error(f"[FEEDBACK CACHE ERROR] Failed to save prediction for {symbol}: {e}")
        return False

def get_pending_evaluations(hours_old: int = 2) -> List[Dict]:
    """
    Pobiera predykcje oczekujące na ewaluację (starsze niż określona liczba godzin)
    
    Args:
        hours_old: Minimalna liczba godzin od predykcji
        
    Returns:
        Lista predykcji do ewaluacji
    """
    try:
        if not os.path.exists(CACHE_PATH):
            return []

        with open(CACHE_PATH, "r", encoding='utf-8') as f:
            data = json.load(f)

        # Filtruj predykcje oczekujące na ewaluację
        pending = []
        cutoff_time = datetime.now() - timedelta(hours=hours_old)
        
        for entry in data:
            if entry.get("evaluated", False):
                continue
                
            try:
                # Parse timestamp
                pred_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                if pred_time.tzinfo:
                    pred_time = pred_time.replace(tzinfo=None)
                
                if pred_time < cutoff_time:
                    pending.append(entry)
                    
            except Exception as e:
                logging.warning(f"[FEEDBACK CACHE] Invalid timestamp format: {entry.get('timestamp')} - {e}")
                continue

        logging.info(f"[FEEDBACK CACHE] Found {len(pending)} predictions pending evaluation (>{hours_old}h old)")
        return pending
        
    except Exception as e:
        logging.error(f"[FEEDBACK CACHE ERROR] Failed to get pending evaluations: {e}")
        return []

def mark_as_evaluated(symbol: str, timestamp: str, evaluation_result: Dict) -> bool:
    """
    Oznacza predykcję jako ocenioną z wynikiem
    
    Args:
        symbol: Symbol trading
        timestamp: Timestamp predykcji
        evaluation_result: Wynik ewaluacji
        
    Returns:
        True jeśli zaktualizowano pomyślnie
    """
    try:
        if not os.path.exists(CACHE_PATH):
            return False

        with open(CACHE_PATH, "r", encoding='utf-8') as f:
            data = json.load(f)

        # Znajdź i zaktualizuj wpis
        updated = False
        for entry in data:
            if entry["symbol"] == symbol and entry["timestamp"] == timestamp:
                entry["evaluated"] = True
                entry["evaluation_result"] = evaluation_result
                entry["price_after_2h"] = evaluation_result.get("price_after_2h")
                entry["price_after_6h"] = evaluation_result.get("price_after_6h")
                updated = True
                break

        if updated:
            with open(CACHE_PATH, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f"[FEEDBACK CACHE] Marked {symbol} prediction as evaluated")
            return True
        else:
            logging.warning(f"[FEEDBACK CACHE] Prediction not found: {symbol} @ {timestamp}")
            return False
            
    except Exception as e:
        logging.error(f"[FEEDBACK CACHE ERROR] Failed to mark as evaluated: {e}")
        return False

def get_evaluation_history(label: str = None, days: int = 30) -> List[Dict]:
    """
    Pobiera historię ewaluacji dla analizy skuteczności
    
    Args:
        label: Opcjonalnie filtruj po label (np. "pullback_continuation")
        days: Liczba dni wstecz
        
    Returns:
        Lista ocenionych predykcji
    """
    try:
        if not os.path.exists(CACHE_PATH):
            return []

        with open(CACHE_PATH, "r", encoding='utf-8') as f:
            data = json.load(f)

        # Filtruj ocenione predykcje
        evaluated = []
        cutoff_time = datetime.now() - timedelta(days=days)
        
        for entry in data:
            if not entry.get("evaluated", False):
                continue
                
            if label and entry.get("label") != label:
                continue
                
            try:
                pred_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                if pred_time.tzinfo:
                    pred_time = pred_time.replace(tzinfo=None)
                
                if pred_time >= cutoff_time:
                    evaluated.append(entry)
                    
            except Exception as e:
                logging.warning(f"[FEEDBACK CACHE] Invalid timestamp in history: {e}")
                continue

        logging.info(f"[FEEDBACK CACHE] Retrieved {len(evaluated)} evaluated predictions")
        return evaluated
        
    except Exception as e:
        logging.error(f"[FEEDBACK CACHE ERROR] Failed to get evaluation history: {e}")
        return []

def cleanup_old_predictions(days_old: int = 60) -> int:
    """
    Usuwa stare predykcje starsze niż określona liczba dni
    
    Args:
        days_old: Liczba dni do zachowania
        
    Returns:
        Liczba usuniętych predykcji
    """
    try:
        if not os.path.exists(CACHE_PATH):
            return 0

        with open(CACHE_PATH, "r", encoding='utf-8') as f:
            data = json.load(f)

        original_count = len(data)
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        # Zachowaj tylko nowsze predykcje
        filtered_data = []
        for entry in data:
            try:
                pred_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                if pred_time.tzinfo:
                    pred_time = pred_time.replace(tzinfo=None)
                
                if pred_time >= cutoff_time:
                    filtered_data.append(entry)
                    
            except Exception as e:
                logging.warning(f"[FEEDBACK CACHE] Invalid timestamp during cleanup: {e}")
                continue

        # Zapisz oczyszczone dane
        if len(filtered_data) != original_count:
            with open(CACHE_PATH, "w", encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
                
        removed_count = original_count - len(filtered_data)
        if removed_count > 0:
            logging.info(f"[FEEDBACK CACHE] Cleaned up {removed_count} old predictions")
            
        return removed_count
        
    except Exception as e:
        logging.error(f"[FEEDBACK CACHE ERROR] Failed to cleanup old predictions: {e}")
        return 0

def get_cache_statistics() -> Dict:
    """
    Zwraca statystyki cache predykcji
    
    Returns:
        Słownik ze statystykami
    """
    try:
        if not os.path.exists(CACHE_PATH):
            return {
                "total_predictions": 0,
                "evaluated_predictions": 0,
                "pending_evaluations": 0,
                "success_rate": 0.0
            }

        with open(CACHE_PATH, "r", encoding='utf-8') as f:
            data = json.load(f)

        total = len(data)
        evaluated = sum(1 for entry in data if entry.get("evaluated", False))
        pending = total - evaluated
        
        # Oblicz success rate z ocenionych predykcji
        successful = 0
        for entry in data:
            if entry.get("evaluated", False):
                result = entry.get("evaluation_result", {})
                if result.get("successful", False):
                    successful += 1
        
        success_rate = (successful / evaluated * 100) if evaluated > 0 else 0.0
        
        return {
            "total_predictions": total,
            "evaluated_predictions": evaluated,
            "pending_evaluations": pending,
            "success_rate": round(success_rate, 2)
        }
        
    except Exception as e:
        logging.error(f"[FEEDBACK CACHE ERROR] Failed to get statistics: {e}")
        return {
            "total_predictions": 0,
            "evaluated_predictions": 0,
            "pending_evaluations": 0,
            "success_rate": 0.0,
            "error": str(e)
        }

def load_dynamic_weights(category: str) -> Dict[str, float]:
    """
    Load dynamic weights by category for TJDE v2 integration
    
    Args:
        category: Weight category ('setup_weights', 'phase_weights', 'clip_weights')
        
    Returns:
        Dictionary with category-specific weights
    """
    try:
        from .weight_adjustment_system import load_current_weights
        
        # Load main weights data
        weights_data = load_current_weights()
        
        # Handle structured weight file format
        if isinstance(weights_data, dict) and "weights" in weights_data:
            all_weights = weights_data["weights"]
        else:
            all_weights = weights_data or {}
        
        # Map AI labels to categories
        if category == "setup_weights":
            # Setup-specific patterns
            setup_patterns = {
                "breakout_pattern": all_weights.get("breakout_pattern", 1.0),
                "momentum_follow": all_weights.get("momentum_follow", 1.0),
                "trend_continuation": all_weights.get("trend_continuation", 1.0),
                "pullback": all_weights.get("pullback", 1.0),
                "reversal_pattern": all_weights.get("reversal_pattern", 1.0),
                "consolidation": all_weights.get("consolidation", 1.0),
                "unknown": 0.8  # Lower weight for unknown patterns
            }
            return setup_patterns
            
        elif category == "phase_weights":
            # Market phase patterns
            phase_patterns = {
                "trend-following": all_weights.get("trend_continuation", 1.0),
                "breakout-continuation": all_weights.get("breakout_pattern", 1.0),
                "consolidation": all_weights.get("consolidation", 1.0),
                "pullback-in-trend": all_weights.get("pullback", 1.0),
                "range": all_weights.get("consolidation", 0.8),
                "basic_screening": 0.7,  # Lower weight for basic screening
                "unknown": 0.6
            }
            return phase_patterns
            
        elif category == "clip_weights":
            # CLIP-specific patterns
            clip_patterns = {
                "trend_continuation": all_weights.get("trend_continuation", 1.0),
                "breakout_pattern": all_weights.get("breakout_pattern", 1.0),
                "momentum_follow": all_weights.get("momentum_follow", 1.0),
                "consolidation": all_weights.get("consolidation", 1.0),
                "pullback": all_weights.get("pullback", 1.0),
                "reversal_pattern": all_weights.get("reversal_pattern", 1.0),
                "unknown": 0.5  # Lowest weight for unknown CLIP patterns
            }
            return clip_patterns
            
        else:
            logging.warning(f"[DYNAMIC WEIGHTS] Unknown category: {category}")
            return {}
            
    except ImportError:
        logging.warning(f"[DYNAMIC WEIGHTS] Weight adjustment module not available")
        return {}
    except Exception as e:
        logging.error(f"[DYNAMIC WEIGHTS ERROR] Failed to load {category}: {e}")
        return {}