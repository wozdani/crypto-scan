def get_alert_level(ppwcs_score: float) -> str:
    """
    Determine alert level based on PPWCS score
    Returns: 'strong', 'active', 'watchlist', or 'none'
    """
    if ppwcs_score >= 80:
        return "strong"     # 🚨
    elif 70 <= ppwcs_score < 80:
        return "active"     # ⚠️
    elif 60 <= ppwcs_score < 70:
        return "watchlist"  # 👀
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