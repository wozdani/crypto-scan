# utils/alerts.py

import os
from utils.telegram_bot import send_alert as send_telegram_message
from utils.take_profit import forecast_take_profits  # jeśli używasz TP forecast


def send_alert(symbol,
               ppwcs_score=None,
               signals=None,
               alert_level=None,
               take_profit_data=None,
               alert_type="pre_pump",
               trend_score=None,
               trend_reasons=None,
               message=None):
    """
    Wysyła alert na Telegram z informacjami o sygnale.
    Obsługuje zarówno pre-pump alerts jak i trend mode alerts.
    """
    if message:
        # Direct message mode
        send_telegram_message(message)
        return
        
    if alert_type == "trend_mode":
        # Trend Mode Alert
        alert_message = f"📈 TREND MODE ALERT – {symbol}\n"
        alert_message += f"Trend Score: {trend_score}/100+\n"
        if trend_reasons:
            alert_message += "Active Signals:\n"
            for reason in trend_reasons:
                alert_message += f"• {reason}\n"
    else:
        # Traditional Pre-Pump Alert
        alert_message = f"🚨 PRE-PUMP ALERT {alert_level.upper()} – {symbol}\n"
        alert_message += f"PPWCS: {ppwcs_score}/100\n"
        if signals:
            alert_message += "Sygnały:\n"
            for k, v in signals.items():
                alert_message += f"• {k}: {'✅' if v else '❌'}\n"

        if take_profit_data:
            alert_message += "\n🎯 Prognoza TP:\n"
            alert_message += f"TP1: {take_profit_data.get('TP1', '-')}\n"
            alert_message += f"TP2: {take_profit_data.get('TP2', '-')}\n"
            alert_message += f"TP3: {take_profit_data.get('TP3', '-')}\n"
            alert_message += f"Trailing TP: {take_profit_data.get('trailing', '-')}\n"

    send_telegram_message(alert_message)
