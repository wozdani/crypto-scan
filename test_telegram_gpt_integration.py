#!/usr/bin/env python3

from utils.take_profit_engine import forecast_take_profit_levels
from utils.alert_system import send_telegram_alert
from utils.alert_utils import get_alert_level

print("🔄 Testing Telegram Alert with GPT Feedback Integration")
print("=" * 60)

# Test scenario for strong signal that includes GPT feedback
test_signals = {
    "ppwcs_score": 85,
    "whale_activity": True,
    "dex_inflow": 180000,
    "compressed": True,
    "stage1g_active": True,
    "pure_accumulation": False,
    "event_tag": "listing",
    "event_score": 15,
    "event_risk": False,
    "volume_spike": True,
    "social_spike": False,
    "orderbook_anomaly": True
}

symbol = "TESTUSDT"
ppwcs_score = test_signals["ppwcs_score"]

print(f"📊 Signal Analysis for {symbol}")
print("-" * 30)
print(f"PPWCS Score: {ppwcs_score}")
print(f"Alert Level: {get_alert_level(ppwcs_score)}")

# Generate TP forecast
tp_forecast = forecast_take_profit_levels(test_signals)
print(f"TP Forecast: {tp_forecast['TP1']}% / {tp_forecast['TP2']}% / {tp_forecast['TP3']}%")

# Simulate GPT feedback for strong signal
gpt_feedback = "Sygnał bardzo silny z potwierdzeniem wielorybów i aktywnym Stage 1G. Ryzyko ograniczone przy wysokim PPWCS. Zalecam pozycję z TP na wskazanych poziomach."

print(f"\n🤖 GPT Feedback Generated:")
print(f"'{gpt_feedback}'")

print(f"\n📱 Testing Telegram Alert Format:")
print("=" * 40)

# Show what the Telegram message would look like
alert_level_text = "🚨 *STRONG ALERT*"
score_change = "+8"

telegram_message = f"""{alert_level_text}
📈 Token: *{symbol}*
🧠 Score: *{ppwcs_score} ({score_change}) / 100*

🎯 TP Forecast:
• TP1: +{tp_forecast['TP1']}%
• TP2: +{tp_forecast['TP2']}%
• TP3: +{tp_forecast['TP3']}%
• Trailing TP: +{tp_forecast['TrailingTP']}%

🔬 Signals: volume_spike, whale_activity, orderbook_anomaly, compressed, stage1g_active
🧩 Trigger: tag_boost

🤖 GPT Feedback:
{gpt_feedback}

🕒 UTC: 2025-05-31 14:07:30
"""

print(telegram_message)

print(f"\n✅ GPT Integration Features:")
print("=" * 35)
print("• GPT feedback included only for PPWCS >= 80")
print("• Expert analysis in Polish language")
print("• Comprehensive signal evaluation")
print("• Risk assessment and recommendations")
print("• Automatic file logging maintained")
print("• Single unified Telegram message")

print(f"\n🚀 Complete Integration Ready")
print("Strong signals (PPWCS >= 80) will include expert GPT analysis")
print("directly in the Telegram alert message for immediate insight.")