"""
Log Optimization for Performance
Reduces excessive logging during async scans to improve performance
"""

import os
from typing import Set

# Global debug flag
DEBUG_MODE = os.getenv('DEBUG_SCAN', 'false').lower() == 'true'

# Log filtering for performance
SKIP_LOG_DECISIONS = {'avoid', 'insufficient', 'weak', 'no_signal'}
IMPORTANT_DECISIONS = {'consider_entry', 'strong_entry', 'join_trend'}

def should_log_decision(decision: str, tjde_score: float = 0) -> bool:
    """
    Determine if TJDE decision should be logged based on importance
    
    Args:
        decision: TJDE decision
        tjde_score: TJDE score for additional filtering
        
    Returns:
        True if should log, False to skip
    """
    if not DEBUG_MODE and decision in SKIP_LOG_DECISIONS:
        return False
        
    if decision in IMPORTANT_DECISIONS:
        return True
        
    if tjde_score >= 0.7:  # High score always logs
        return True
        
    return DEBUG_MODE

def log_tjde_decision(symbol: str, decision: str, tjde_score: float, details: str = ""):
    """Optimized TJDE decision logging"""
    if should_log_decision(decision, tjde_score):
        print(f"[TJDE] {symbol}: {decision.upper()} ({tjde_score:.3f}) {details}")

def log_scan_error(symbol: str, error_type: str, details: str = ""):
    """Log only critical scan errors"""
    critical_errors = ['contract_missing', 'bybit_error', 'data_corruption']
    if any(err in error_type.lower() for err in critical_errors):
        print(f"[SCAN ERROR] {symbol}: {error_type} - {details}")

def log_debug_if_enabled(message: str):
    """Log debug message only if debug mode enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def get_scan_metrics() -> dict:
    """Get current scan performance metrics"""
    return {
        'debug_mode': DEBUG_MODE,
        'skip_decisions': list(SKIP_LOG_DECISIONS),
        'important_decisions': list(IMPORTANT_DECISIONS)
    }