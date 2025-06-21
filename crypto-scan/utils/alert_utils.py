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