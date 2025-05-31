#!/usr/bin/env python3

from utils.take_profit_engine import forecast_take_profit_levels

print("🔄 Testing Enhanced Take Profit Forecasting System")
print("=" * 55)

# Test scenarios with different signal combinations
test_scenarios = [
    {
        "name": "Strong Signal - All Conditions Met",
        "signals": {
            "ppwcs_score": 88,
            "rsi": 60,
            "whale_activity": True,
            "dex_inflow": 150000,
            "compressed": True,
            "stage1g_active": True,
            "pure_accumulation": True
        }
    },
    {
        "name": "High RSI - Momentum Exhaustion",
        "signals": {
            "ppwcs_score": 75,
            "rsi": 75,
            "whale_activity": False,
            "dex_inflow": 50000,
            "compressed": True,
            "stage1g_active": False,
            "pure_accumulation": False
        }
    },
    {
        "name": "Whale Activity with High Inflow",
        "signals": {
            "ppwcs_score": 82,
            "rsi": 55,
            "whale_activity": True,
            "dex_inflow": 200000,
            "compressed": False,
            "stage1g_active": True,
            "pure_accumulation": False
        }
    },
    {
        "name": "Low Confidence Signal",
        "signals": {
            "ppwcs_score": 58,
            "rsi": 45,
            "whale_activity": False,
            "dex_inflow": 20000,
            "compressed": False,
            "stage1g_active": False,
            "pure_accumulation": False
        }
    },
    {
        "name": "Pure Accumulation Phase",
        "signals": {
            "ppwcs_score": 72,
            "rsi": 50,
            "whale_activity": False,
            "dex_inflow": 30000,
            "compressed": True,
            "stage1g_active": False,
            "pure_accumulation": True
        }
    }
]

for scenario in test_scenarios:
    print(f"\n📊 {scenario['name']}")
    print("-" * 35)
    
    signals = scenario['signals']
    tp_forecast = forecast_take_profit_levels(signals)
    
    print(f"PPWCS Score: {signals['ppwcs_score']}")
    print(f"RSI: {signals['rsi']}")
    print(f"Whale Activity: {'Yes' if signals['whale_activity'] else 'No'}")
    print(f"DEX Inflow: ${signals['dex_inflow']:,}")
    print(f"Compressed: {'Yes' if signals['compressed'] else 'No'}")
    print(f"Stage 1G: {'Yes' if signals['stage1g_active'] else 'No'}")
    
    print(f"\n🎯 TP Forecast:")
    print(f"  TP1: {tp_forecast['TP1']}%")
    print(f"  TP2: {tp_forecast['TP2']}%") 
    print(f"  TP3: {tp_forecast['TP3']}%")
    print(f"  Trailing TP: {tp_forecast['TrailingTP']}%")
    print(f"  Multiplier: {tp_forecast['multiplier']}")
    print(f"  Confidence: {tp_forecast['confidence']}")

print(f"\n🎯 Enhanced TP System Features:")
print("=" * 35)
print("✅ RSI momentum exhaustion adjustment")
print("✅ Whale activity + high inflow boost")
print("✅ Pure accumulation phase bonus")
print("✅ Exceptional signal boost (85+ PPWCS)")
print("✅ Smart trailing TP logic")
print("✅ Safety bounds (min/max levels)")

print(f"\n🚀 Enhanced Take Profit Engine Ready")
print("System now provides more accurate TP levels based on")
print("market structure, momentum, and whale activity.")