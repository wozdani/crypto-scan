"""
Memory Feedback Loop for Token Performance Evaluation
Updates token memory with actual results after trading decisions
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from utils.token_memory import load_token_memory, save_token_memory, set_result_for_last_entry

def evaluate_recent_decisions(lookback_hours: int = 2):
    """
    Evaluate recent trading decisions and update token memory with results
    
    Args:
        lookback_hours: How many hours back to evaluate decisions
    """
    try:
        memory_data = load_token_memory()
        
        if not memory_data:
            logging.info("No token memory data to evaluate")
            return
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        evaluated_count = 0
        
        for symbol, entries in memory_data.items():
            for entry in entries:
                # Skip already evaluated entries
                if entry.get("result_after_2h") is not None:
                    continue
                
                entry_time = datetime.fromisoformat(entry["timestamp"])
                
                # Check if entry is old enough to evaluate (at least 2 hours)
                if entry_time < cutoff_time:
                    result = _evaluate_decision_outcome(symbol, entry)
                    if result:
                        entry["result_after_2h"] = result
                        entry["evaluated_at"] = datetime.now(timezone.utc).isoformat()
                        evaluated_count += 1
        
        if evaluated_count > 0:
            save_token_memory(memory_data)
            logging.info(f"Evaluated {evaluated_count} decisions in memory feedback loop")
        
    except Exception as e:
        logging.error(f"Error in memory feedback loop: {e}")

def _evaluate_decision_outcome(symbol: str, entry: Dict) -> Optional[str]:
    """
    Evaluate whether a trading decision was successful
    
    Args:
        symbol: Trading symbol
        entry: Memory entry to evaluate
        
    Returns:
        "success", "fail", or None if cannot determine
    """
    try:
        decision = entry.get("decision", "")
        tjde_score = entry.get("tjde_score", 0)
        entry_time = datetime.fromisoformat(entry["timestamp"])
        
        # Simple heuristic evaluation based on decision type and score
        if decision in ["consider_entry", "join_trend"]:
            # For entry decisions, success is determined by score threshold
            # and lack of immediate reversal signals
            
            if tjde_score >= 0.75:
                # High confidence decisions are more likely successful
                return "success" if _check_price_movement_positive(symbol, entry_time) else "fail"
            elif tjde_score >= 0.50:
                # Medium confidence - neutral evaluation
                return "success" if _check_no_major_reversal(symbol, entry_time) else "fail"
            else:
                # Low confidence decisions are more likely to fail
                return "fail"
                
        elif decision == "avoid":
            # Avoid decisions are successful if we avoided a loss
            return "success" if not _check_price_movement_positive(symbol, entry_time) else "fail"
        
        return None
        
    except Exception as e:
        logging.debug(f"Error evaluating decision for {symbol}: {e}")
        return None

def _check_price_movement_positive(symbol: str, entry_time: datetime) -> bool:
    """
    Check if price movement was positive after entry time
    This is a simplified version - in production would use actual price data
    """
    # Placeholder implementation - would integrate with price data
    # For now, use a simple heuristic based on time patterns
    
    try:
        # Load recent scan results to check price movement
        results_path = "data/latest_scan_results.json"
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                data = json.load(f)
                
            # Find symbol in recent results
            for result in data:
                if result.get('symbol') == symbol:
                    # Simple heuristic - assume positive if found in recent scans
                    return True
                    
        return False
        
    except Exception:
        return False

def _check_no_major_reversal(symbol: str, entry_time: datetime) -> bool:
    """Check if there was no major price reversal after entry"""
    # Simplified implementation - would use actual price data in production
    return True  # Assume no major reversal for now

def run_memory_feedback_evaluation():
    """Run the complete memory feedback evaluation process"""
    try:
        print("[MEMORY_FEEDBACK] Starting token memory evaluation...")
        
        # Evaluate decisions from last 2-8 hours
        evaluate_recent_decisions(lookback_hours=8)
        
        # Get statistics
        from utils.token_memory import get_memory_stats
        stats = get_memory_stats()
        
        print(f"[MEMORY_FEEDBACK] Completed evaluation")
        print(f"[MEMORY_STATS] Total symbols: {stats.get('total_symbols', 0)}")
        print(f"[MEMORY_STATS] Total entries: {stats.get('total_entries', 0)}")
        print(f"[MEMORY_STATS] Evaluated: {stats.get('evaluated_entries', 0)}")
        print(f"[MEMORY_STATS] Evaluation rate: {stats.get('evaluation_rate', 0):.1%}")
        
        return stats
        
    except Exception as e:
        logging.error(f"Error in memory feedback evaluation: {e}")
        return None

def get_token_performance_summary(days: int = 4) -> Dict:
    """Get performance summary for all tokens over specified days"""
    try:
        memory_data = load_token_memory()
        
        summary = {
            "total_tokens": len(memory_data),
            "best_performers": [],
            "worst_performers": [],
            "most_active": []
        }
        
        token_performance = {}
        
        for symbol, entries in memory_data.items():
            if not entries:
                continue
                
            evaluated_entries = [e for e in entries if e.get("result_after_2h") is not None]
            if not evaluated_entries:
                continue
                
            success_count = len([e for e in evaluated_entries if e.get("result_after_2h") == "success"])
            success_rate = success_count / len(evaluated_entries)
            avg_score = sum(e.get("tjde_score", 0) for e in entries) / len(entries)
            
            token_performance[symbol] = {
                "success_rate": success_rate,
                "total_decisions": len(entries),
                "evaluated_decisions": len(evaluated_entries),
                "avg_score": avg_score
            }
        
        # Sort by performance
        sorted_by_success = sorted(token_performance.items(), 
                                 key=lambda x: x[1]["success_rate"], reverse=True)
        sorted_by_activity = sorted(token_performance.items(), 
                                  key=lambda x: x[1]["total_decisions"], reverse=True)
        
        summary["best_performers"] = sorted_by_success[:5]
        summary["worst_performers"] = sorted_by_success[-5:]
        summary["most_active"] = sorted_by_activity[:5]
        
        return summary
        
    except Exception as e:
        logging.error(f"Error getting performance summary: {e}")
        return {"error": str(e)}