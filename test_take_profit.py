from utils.take_profit_engine import forecast_take_profit_levels, calculate_risk_reward_ratio

# Test Take Profit Forecast Engine with different scenarios

print("🎯 Testing Take Profit Forecast Engine")
print("=" * 50)

# Scenario 1: High quality signal with strong momentum
signals_high_quality = {
    "ppwcs_score": 82,
    "rsi": 71,
    "delta_flow_strength": 1.8,
    "bookmap_heatmap_score": 1.3,
    "orderbook_imbalance": 1.6,
    "type_of_breakout": "squeeze_breakout",
    "time_tag": "after_15"
}

# Scenario 2: Medium quality signal
signals_medium = {
    "ppwcs_score": 68,
    "rsi": 58,
    "delta_flow_strength": 1.2,
    "bookmap_heatmap_score": 0.8,
    "orderbook_imbalance": 1.1,
    "type_of_breakout": "unknown",
    "time_tag": "other"
}

# Scenario 3: Lower quality signal
signals_low = {
    "ppwcs_score": 45,
    "rsi": 42,
    "delta_flow_strength": 0.6,
    "bookmap_heatmap_score": 0.5,
    "orderbook_imbalance": 0.8,
    "type_of_breakout": "reject_impulse",
    "time_tag": "night"
}

# Scenario 4: Exceptional signal (very high PPWCS)
signals_exceptional = {
    "ppwcs_score": 93,
    "rsi": 75,
    "delta_flow_strength": 2.2,
    "bookmap_heatmap_score": 1.8,
    "orderbook_imbalance": 2.1,
    "type_of_breakout": "squeeze_breakout",
    "time_tag": "before_15"
}

def test_scenario(name, signals):
    print(f"\n📊 {name}")
    print("-" * 30)
    
    tp_forecast = forecast_take_profit_levels(signals)
    risk_reward = calculate_risk_reward_ratio(signals, tp_forecast)
    
    print(f"PPWCS Score: {signals.get('ppwcs_score', 0)}")
    print(f"Multiplier: {tp_forecast['multiplier']}")
    print(f"Confidence: {tp_forecast['confidence']}")
    print()
    print(f"TP1: {tp_forecast['TP1']}% (R:R {risk_reward['RR_TP1']})")
    print(f"TP2: {tp_forecast['TP2']}% (R:R {risk_reward['RR_TP2']})")
    print(f"TP3: {tp_forecast['TP3']}% (R:R {risk_reward['RR_TP3']})")
    print(f"Trailing TP: {tp_forecast['TrailingTP']}%")
    print(f"Stop Loss: {risk_reward['stop_loss']}%")
    print(f"Position Size: {risk_reward['recommended_position_size']}")

test_scenario("High Quality Signal", signals_high_quality)
test_scenario("Medium Quality Signal", signals_medium)
test_scenario("Lower Quality Signal", signals_low)
test_scenario("Exceptional Signal", signals_exceptional)

print("\n🎯 Summary Comparison:")
print("=" * 50)
results = []
for name, signals in [
    ("High Quality", signals_high_quality),
    ("Medium Quality", signals_medium),
    ("Lower Quality", signals_low),
    ("Exceptional", signals_exceptional)
]:
    tp = forecast_take_profit_levels(signals)
    rr = calculate_risk_reward_ratio(signals, tp)
    results.append((name, tp['TP2'], rr['RR_TP2'], tp['confidence']))

for name, tp2, rr, conf in results:
    print(f"{name:15} | TP2: {tp2:5.1f}% | R:R: {rr:4.1f} | Confidence: {conf}")