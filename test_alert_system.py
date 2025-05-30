from utils.alert_system import process_alert, determine_alert_level, format_alert_message
from utils.take_profit_engine import forecast_take_profit_levels, calculate_risk_reward_ratio

# Test Alert System with different PPWCS scenarios

print("🚨 Testing Comprehensive Alert System")
print("=" * 60)

# Test scenarios with different PPWCS scores
test_scenarios = [
    {
        "name": "Watchlist Only (65 score)",
        "token": "TESTUSDT",
        "ppwcs_score": 65,
        "signals": {
            "whale_activity": True,
            "dex_inflow": False,
            "volume_spike": True,
            "orderbook_anomaly": False,
            "social_spike": False,
            "stage1g_active": False,
            "event_tag": None,
        }
    },
    {
        "name": "Pre-Pump Active (75 score)",
        "token": "PUMPUSDT", 
        "ppwcs_score": 75,
        "signals": {
            "whale_activity": True,
            "dex_inflow": True,
            "volume_spike": True,
            "orderbook_anomaly": False,
            "social_spike": False,
            "stage1g_active": True,
            "stage1g_trigger_type": "classic",
            "event_tag": "listing",
        }
    },
    {
        "name": "Strong Alert (85 score)",
        "token": "STRONGUSDT",
        "ppwcs_score": 85,
        "signals": {
            "whale_activity": True,
            "dex_inflow": True,
            "volume_spike": True,
            "orderbook_anomaly": True,
            "social_spike": True,
            "stage1g_active": True,
            "stage1g_trigger_type": "tag_boost",
            "event_tag": "partnership",
        }
    }
]

def test_alert_scenario(scenario):
    name = scenario["name"]
    token = scenario["token"]
    ppwcs_score = scenario["ppwcs_score"]
    signals = scenario["signals"]
    
    print(f"\n📊 Testing: {name}")
    print("-" * 40)
    
    # Determine alert level
    alert_level, alert_emoji = determine_alert_level(ppwcs_score)
    print(f"Alert Level: {alert_level} {alert_emoji}")
    
    if ppwcs_score >= 70:
        # Generate TP forecast for alerts
        tp_forecast = forecast_take_profit_levels(signals)
        risk_reward = calculate_risk_reward_ratio(signals, tp_forecast)
        
        print(f"TP Levels: {tp_forecast['TP1']}% / {tp_forecast['TP2']}% / {tp_forecast['TP3']}%")
        print(f"Risk/Reward TP2: {risk_reward['RR_TP2']}")
        print(f"Position Size: {risk_reward['recommended_position_size']}")
        
        # Show formatted alert message
        alert_message = format_alert_message(token, ppwcs_score, signals, tp_forecast, risk_reward)
        print("\nFormatted Alert Message:")
        print("-" * 25)
        print(alert_message)
        
    else:
        print("Action: Log to watchlist only (CSV)")
    
    print("\n" + "="*60)

# Test all scenarios
for scenario in test_scenarios:
    test_alert_scenario(scenario)

# Test cooldown logic
print("\n🕒 Testing Cooldown Logic")
print("-" * 30)

from utils.alert_system import check_cooldown, update_cooldown

test_token = "COOLDOWNTEST"
print(f"Initial cooldown check for {test_token}: {check_cooldown(test_token)}")

# Simulate updating cooldown
update_cooldown(test_token)
print(f"After update, cooldown check for {test_token}: {check_cooldown(test_token)}")

print(f"\nCooldown period: 60 minutes")
print("Tokens in cooldown will not generate new alerts")