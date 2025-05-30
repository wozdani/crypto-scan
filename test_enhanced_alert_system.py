#!/usr/bin/env python3

from utils.alert_utils import get_alert_level, get_alert_level_text, should_send_telegram_alert, should_request_gpt_analysis
from utils.scoring import compute_ppwcs
from utils.take_profit_engine import forecast_take_profit_levels
from utils.alert_system import process_alert

print("🔄 Testing Enhanced Alert Level System")
print("=" * 50)

# Test different PPWCS score levels
test_scores = [45, 65, 75, 85, 95]

for score in test_scores:
    alert_level = get_alert_level(score)
    alert_text = get_alert_level_text(alert_level)
    should_telegram = should_send_telegram_alert(alert_level)
    should_gpt = should_request_gpt_analysis(alert_level)
    
    print(f"\n📊 Score: {score}")
    print(f"   Level: {alert_level}")
    print(f"   Text: {alert_text}")
    print(f"   Telegram Alert: {'✅' if should_telegram else '❌'}")
    print(f"   GPT Analysis: {'✅' if should_gpt else '❌'}")

print(f"\n🧪 Testing Alert Processing Logic")
print("-" * 30)

# Simulate signals for strong alert scenario
test_signals = {
    'volume_spike': True,
    'whale_activity': True,
    'social_spike': False,
    'orderbook_anomaly': True,
    'dex_inflow': True,
    'event_tag': 'listing',
    'event_score': 15,
    'event_risk': False,
    'stage1g_active': True,
    'stage1g_trigger_type': 'tag_boost',
    'compressed': True
}

# Calculate score and forecast
ppwcs_score = compute_ppwcs(test_signals)
tp_forecast = forecast_take_profit_levels(test_signals)

print(f"Test Token: TESTUSDT")
print(f"PPWCS Score: {ppwcs_score}")
print(f"Alert Level: {get_alert_level(ppwcs_score)}")
print(f"TP Levels: {tp_forecast['TP1']}% / {tp_forecast['TP2']}% / {tp_forecast['TP3']}%")

# Show what the alert system would do
alert_level = get_alert_level(ppwcs_score)
print(f"\nSystem Actions:")
print(f"- Telegram Alert: {'Yes' if should_send_telegram_alert(alert_level) else 'No'}")
print(f"- GPT Analysis: {'Yes' if should_request_gpt_analysis(alert_level) else 'No'}")
print(f"- Cooldown Applied: {'Yes' if should_send_telegram_alert(alert_level) else 'No'}")

print(f"\n✅ Enhanced Alert System Ready")
print("System now uses standardized alert levels with proper utility functions")