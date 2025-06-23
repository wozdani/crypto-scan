def get_alert_level(ppwcs_score: float, pure_accumulation: bool = False) -> str:
    """
    Determine alert level based on PPWCS score
    Updated threshold: Only perfect scores (100) trigger alerts
    Returns: 'strong', 'active', 'watchlist', or 'none'
    """
    if ppwcs_score >= 97:  # Perfect score (max possible is 97)
        return "strong"     # 🚨
    elif ppwcs_score >= 90:
        return "watchlist"  # 👀 (monitoring only)
    else:
        return "none"

def get_alert_level_text(alert_level: str) -> str:
    """
    Get formatted alert level text for messages
    """
    if alert_level == "strong":
        return "🚨 *STRONG ALERT*"
    elif alert_level == "active":
        return "⚠️ *PRE-PUMP ACTIVE*"
    elif alert_level == "watchlist":
        return "👀 *WATCHLIST SIGNAL*"
    else:
        return ""

def should_send_telegram_alert(alert_level: str) -> bool:
    """
    Determine if Telegram alert should be sent based on level
    """
    return alert_level in ["active", "strong"]

def should_request_gpt_analysis(alert_level: str) -> bool:
    """
    Determine if GPT analysis should be requested
    """
    return alert_level == "strong"


import json
import os
from datetime import datetime, timezone
from pathlib import Path


def ensure_logs_directory():
    """Ensure logs directory exists"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def log_alert_history(symbol, score, decision, breakdown=None, timestamp=None):
    """
    Log alert to history file for feedback loop analysis
    
    Args:
        symbol: Trading symbol
        score: Final TJDE/PPWCS score
        decision: Decision type (join_trend, consider_entry, etc.)
        breakdown: Score breakdown dict
        timestamp: Custom timestamp (optional)
    """
    try:
        ensure_logs_directory()
        
        log_entry = {
            "symbol": symbol,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "score": round(float(score), 4),
            "decision": decision,
            "score_breakdown": breakdown or {}
        }
        
        log_file = Path("logs/alerts_history.jsonl")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        print(f"[ALERT LOG] Zapisano historię alertu: {symbol} ({score:.3f}, {decision})")
        
    except Exception as e:
        print(f"[ALERT LOG] Błąd zapisu historii alertu {symbol}: {e}")