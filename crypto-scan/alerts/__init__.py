"""
DiamondWhale AI Alert System
Stage 4/7: Diamond Alert Telegram Notifications

Exports:
- send_diamond_alert_auto: Automatyczne wysyłanie Diamond Alerts
- get_diamond_alert_stats: Statystyki alertów
"""

from .telegram_notification import (
    send_diamond_alert,
    send_diamond_alert_auto,
    get_diamond_alert_stats,
    format_confidence_indicator,
    format_trigger_reasons,
    format_score_breakdown
)

__all__ = [
    'send_diamond_alert',
    'send_diamond_alert_auto', 
    'get_diamond_alert_stats',
    'format_confidence_indicator',
    'format_trigger_reasons',
    'format_score_breakdown'
]