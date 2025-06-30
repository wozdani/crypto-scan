"""
Feedback Result Logging Module
Automatyczne zapisywanie wyników alertów do pliku feedback_results.json
dla późniejszej analizy przez feedback_loop.py
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

FEEDBACK_LOG = "data/feedback_results.json"

def log_feedback_result(
    symbol: str, 
    score_components: Dict[str, float], 
    phase: str, 
    was_successful: bool,
    entry_price: float = None,
    exit_price: float = None,
    profit_loss_pct: float = None,
    alert_time: str = None,
    additional_data: Dict[str, Any] = None
):
    """
    Zapisuje wynik konkretnego alertu do feedback logu,
    aby można było dostroić scoring później.
    
    Args:
        symbol: Symbol tokena (np. PEPEUSDT)
        score_components: Dict z wagami komponentów z simulate_trader_decision
        phase: Faza rynku ("pre-pump", "trend", "breakout", "consolidation")
        was_successful: True jeśli alert był skuteczny, False jeśli nie
        entry_price: Cena wejścia (opcjonalna)
        exit_price: Cena wyjścia (opcjonalna)
        profit_loss_pct: Procent zysku/straty (opcjonalna)
        alert_time: Czas alertu ISO format (opcjonalny)
        additional_data: Dodatkowe dane (opcjonalne)
    """
    
    # Przygotuj wpis
    result_entry = {
        "symbol": symbol,
        "phase": phase,
        "score_components": score_components,
        "was_successful": was_successful,
        "recorded_at": datetime.now().isoformat(),
        "alert_time": alert_time or datetime.now().isoformat(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "profit_loss_pct": profit_loss_pct
    }
    
    # Dodaj dodatkowe dane jeśli są
    if additional_data:
        result_entry.update(additional_data)

    # Upewnij się, że katalog istnieje
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)

    # Wczytaj istniejący log
    feedback_data = []
    if os.path.exists(FEEDBACK_LOG):
        try:
            with open(FEEDBACK_LOG, "r") as f:
                feedback_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            feedback_data = []

    # Dopisz nowy wpis
    feedback_data.append(result_entry)

    # Zapisz z backup
    try:
        with open(FEEDBACK_LOG, "w") as f:
            json.dump(feedback_data, f, indent=2)
        
        print(f"[FEEDBACK LOG] Recorded result: {symbol} {phase} {'SUCCESS' if was_successful else 'FAILURE'}")
        return True
        
    except Exception as e:
        print(f"[FEEDBACK LOG ERROR] Failed to save: {e}")
        return False

def load_feedback_results() -> list:
    """
    Wczytuje wszystkie zapisane wyniki feedback
    
    Returns:
        Lista wyników feedback
    """
    if not os.path.exists(FEEDBACK_LOG):
        return []
        
    try:
        with open(FEEDBACK_LOG, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def get_feedback_stats() -> Dict[str, Any]:
    """
    Zwraca statystyki feedback
    
    Returns:
        Dict ze statystykami
    """
    results = load_feedback_results()
    
    if not results:
        return {"total": 0, "successful": 0, "success_rate": 0.0}
    
    total = len(results)
    successful = sum(1 for r in results if r.get("was_successful", False))
    success_rate = (successful / total) * 100 if total > 0 else 0.0
    
    # Statystyki per phase
    phases = {}
    for result in results:
        phase = result.get("phase", "unknown")
        if phase not in phases:
            phases[phase] = {"total": 0, "successful": 0}
        phases[phase]["total"] += 1
        if result.get("was_successful", False):
            phases[phase]["successful"] += 1
    
    # Dodaj success rate per phase
    for phase in phases:
        phases[phase]["success_rate"] = (phases[phase]["successful"] / phases[phase]["total"]) * 100
    
    return {
        "total": total,
        "successful": successful,
        "success_rate": success_rate,
        "phases": phases
    }

def clear_old_feedback_results(days_old: int = 30):
    """
    Usuwa stare wyniki feedback starsze niż określona liczba dni
    
    Args:
        days_old: Liczba dni - wyniki starsze będą usunięte
    """
    results = load_feedback_results()
    if not results:
        return
    
    from datetime import timedelta
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    # Filtruj wyniki
    filtered_results = []
    for result in results:
        try:
            recorded_at = datetime.fromisoformat(result.get("recorded_at", ""))
            if recorded_at >= cutoff_date:
                filtered_results.append(result)
        except (ValueError, TypeError):
            # Keep results without valid timestamp
            filtered_results.append(result)
    
    # Zapisz przefiltrowane wyniki
    try:
        with open(FEEDBACK_LOG, "w") as f:
            json.dump(filtered_results, f, indent=2)
        
        removed_count = len(results) - len(filtered_results)
        print(f"[FEEDBACK CLEANUP] Removed {removed_count} old feedback results")
        
    except Exception as e:
        print(f"[FEEDBACK CLEANUP ERROR] Failed to clean: {e}")

# Przykład użycia
def test_feedback_logging():
    """Test funkcji logowania feedback"""
    
    # Przykładowe score_components
    test_components = {
        "pre_breakout_structure": 0.25,
        "volume_structure": 0.20,
        "liquidity_behavior": 0.15,
        "clip_confidence": 0.10,
        "gpt_label_match": 0.10,
        "heatmap_window": 0.10,
        "orderbook_setup": 0.05,
        "market_phase_modifier": 0.05
    }
    
    # Test successful alert
    log_feedback_result(
        symbol="TESTUSDT",
        score_components=test_components,
        phase="pre-pump",
        was_successful=True,
        entry_price=1.234,
        exit_price=1.295,
        profit_loss_pct=4.9
    )
    
    # Test unsuccessful alert
    log_feedback_result(
        symbol="FAILUSDT",
        score_components=test_components,
        phase="trend",
        was_successful=False,
        entry_price=0.567,
        exit_price=0.534,
        profit_loss_pct=-5.8
    )
    
    # Pokaż statystyki
    stats = get_feedback_stats()
    print(f"[FEEDBACK TEST] Stats: {stats}")

if __name__ == "__main__":
    test_feedback_logging()