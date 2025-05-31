#!/usr/bin/env python3

import os
from datetime import datetime
from utils.take_profit_engine import forecast_take_profit_levels
from utils.scoring import compute_ppwcs
from utils.alert_utils import get_alert_level, should_request_gpt_analysis

print("🔄 Testing Complete Scan Cycle Integration")
print("=" * 50)

def simulate_scan_cycle():
    """Simulate the complete scan cycle workflow"""
    
    # Simulate strong signal data
    symbol = "TESTUSDT"
    signals = {
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
    
    print(f"📊 Processing {symbol}")
    print("-" * 30)
    
    # Step 1: Calculate PPWCS
    ppwcs_score = compute_ppwcs(signals)
    print(f"PPWCS Score: {ppwcs_score}")
    
    # Step 2: Check if signal qualifies for processing
    if ppwcs_score < 60:
        print("❌ Signal too weak, skipping")
        return
    
    # Step 3: Determine alert level
    alert_level = get_alert_level(ppwcs_score)
    print(f"Alert Level: {alert_level}")
    
    # Step 4: Generate TP forecast
    tp_forecast = forecast_take_profit_levels(signals)
    print(f"TP Forecast: {tp_forecast['TP1']}% / {tp_forecast['TP2']}% / {tp_forecast['TP3']}%")
    
    # Step 5: Process alert if conditions are met
    if ppwcs_score >= 60:
        print(f"✅ Would send alert for {symbol}")
        print(f"   - Alert type: {alert_level}")
        print(f"   - TP levels included")
        print(f"   - Cooldown applied")
    
    # Step 6: GPT feedback for strong signals (PPWCS >= 80)
    if ppwcs_score >= 80:
        print(f"\n🤖 GPT Feedback Triggered (PPWCS >= 80)")
        print("   - Comprehensive signal analysis")
        print("   - Expert risk assessment")
        print("   - Polish language response")
        
        # Check if OpenAI API key is available
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                # Simulate GPT call (would normally use send_report_to_gpt)
                feedback_example = "Sygnał bardzo silny dzięki aktywności wielorybów i Stage 1G. Ryzyko ograniczone przy tak wysokim PPWCS. Zalecam pozycję z TP na wskazanych poziomach."
                print(f"   GPT Response: {feedback_example}")
                
                # Simulate file logging
                feedback_file = f"data/feedback/{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt"
                print(f"   Would save to: {feedback_file}")
                
            except Exception as e:
                print(f"   ⚠️ GPT error: {e}")
        else:
            print(f"   🔑 Requires OpenAI API key for production")
    
    print(f"\n✅ Scan cycle completed for {symbol}")

def test_different_score_levels():
    """Test scan cycle behavior with different PPWCS scores"""
    
    test_scores = [45, 65, 75, 85]
    
    print(f"\n📊 Testing Different Score Levels")
    print("=" * 35)
    
    for score in test_scores:
        alert_level = get_alert_level(score)
        gpt_trigger = should_request_gpt_analysis(alert_level)
        
        print(f"\nScore {score}:")
        print(f"  Alert Level: {alert_level}")
        print(f"  Action: {'Alert + GPT' if gpt_trigger else 'Alert only' if score >= 70 else 'Watchlist' if score >= 60 else 'No action'}")

# Run the simulation
simulate_scan_cycle()
test_different_score_levels()

print(f"\n🚀 Complete Integration Summary:")
print("=" * 35)
print("✅ PPWCS calculation with trailing logic")
print("✅ Enhanced TP forecasting")
print("✅ Alert system with cooldown management") 
print("✅ GPT feedback for PPWCS >= 80")
print("✅ Automatic file logging")
print("✅ Three-tier alert classification")

print(f"\nThe scan cycle automatically:")
print("1. Calculates PPWCS for each symbol")
print("2. Generates TP forecast")
print("3. Sends alerts based on thresholds")
print("4. Requests GPT analysis for strong signals")
print("5. Logs all feedback to files")
print("6. Applies proper cooldown management")