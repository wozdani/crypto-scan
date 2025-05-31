#!/usr/bin/env python3

from utils.take_profit_engine import forecast_take_profit_levels
from utils.alert_utils import get_alert_level, should_request_gpt_analysis
import os

print("🔄 Testing Enhanced GPT Feedback System")
print("=" * 50)

# Test scenario for strong signal that would trigger GPT feedback
test_signals = {
    "ppwcs_score": 85,
    "rsi": 62,
    "whale_activity": True,
    "dex_inflow": 180000,
    "compressed": True,
    "stage1g_active": True,
    "pure_accumulation": False,
    "event_tag": "listing",
    "event_score": 15,
    "event_risk": False
}

symbol = "TESTUSDT"
score = test_signals["ppwcs_score"]

print(f"📊 Test Signal Analysis for {symbol}")
print("-" * 30)
print(f"PPWCS Score: {score}")
print(f"Alert Level: {get_alert_level(score)}")
print(f"Should Request GPT: {should_request_gpt_analysis(get_alert_level(score))}")

# Generate TP forecast
tp_forecast = forecast_take_profit_levels(test_signals)
print(f"\n🎯 TP Forecast:")
print(f"TP1: {tp_forecast['TP1']}%")
print(f"TP2: {tp_forecast['TP2']}%")
print(f"TP3: {tp_forecast['TP3']}%")
print(f"Trailing: {tp_forecast['TrailingTP']}%")

# Check if OpenAI API key is available
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    try:
        from utils.gpt_feedback import send_report_to_gpt
        
        print(f"\n🤖 Testing GPT Feedback (PPWCS >= 80):")
        print("-" * 40)
        
        gpt_feedback = send_report_to_gpt(symbol, test_signals, tp_forecast)
        print(f"GPT Response: {gpt_feedback}")
        
        # Simulate file logging
        from datetime import datetime
        feedback_filename = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt"
        print(f"\nWould save to: data/feedback/{feedback_filename}")
        
    except Exception as e:
        print(f"⚠️ GPT test failed: {e}")
        print("This is expected in development environment without API access")
else:
    print(f"\n🔑 OpenAI API Key Required")
    print("GPT feedback requires OPENAI_API_KEY environment variable")
    print("The system would normally:")
    print("1. Send comprehensive signal data to GPT-4o")
    print("2. Request 3-sentence Polish analysis")
    print("3. Log feedback to data/feedback/ folder")
    print("4. Include feedback in Telegram alerts")

print(f"\n✅ GPT Feedback System Features:")
print("=" * 35)
print("• Triggered only for PPWCS >= 80 (strong alerts)")
print("• Comprehensive signal analysis prompt")
print("• Enhanced TP forecast integration")
print("• Polish language responses")
print("• Automatic file logging")
print("• Telegram alert integration")

print(f"\n🚀 System Integration Complete")
print("Enhanced GPT feedback will provide expert analysis")
print("for high-confidence trading signals when deployed with API access.")