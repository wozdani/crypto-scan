# utils/alerts.py

import os
from utils.telegram_bot import send_telegram_message
from utils.take_profit import forecast_take_profits  # jeśli używasz TP forecast


def send_alert(symbol,
               ppwcs_score,
               signals,
               alert_level,
               take_profit_data=None):
    """
    Wysyła alert na Telegram z informacjami o sygnale.
    """
    message = f"🚨 ALERT {alert_level.upper()} – {symbol}\n"
    message += f"PPWCS: {ppwcs_score}/100\n"
    message += "Sygnały:\n"
    for k, v in signals.items():
        message += f"• {k}: {'✅' if v else '❌'}\n"

    if take_profit_data:
        message += "\n🎯 Prognoza TP:\n"
        message += f"TP1: {take_profit_data.get('TP1', '-')}\n"
        message += f"TP2: {take_profit_data.get('TP2', '-')}\n"
        message += f"TP3: {take_profit_data.get('TP3', '-')}\n"
        message += f"Trailing TP: {take_profit_data.get('trailing', '-')}\n"

    send_telegram_message(message)
