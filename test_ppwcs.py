from utils.scoring import compute_ppwcs

# Test PPWCS 2.5 scoring with different scenarios

# Scenario 1: PEPEUSDT with listing tag
signals_pepe = {
    'social_spike': False,
    'whale_activity': False,
    'orderbook_anomaly': False,
    'volume_spike': False,
    'dex_inflow': False,
    'event_tag': 'listing',
    'event_score': 10,
    'event_risk': False,
    'stage1g_active': False,
    'stage1g_trigger_type': None
}

# Scenario 2: Token with whale activity + DEX inflow (pure accumulation)
signals_accumulation = {
    'social_spike': False,
    'whale_activity': True,
    'orderbook_anomaly': False,
    'volume_spike': True,
    'dex_inflow': True,
    'event_tag': None,
    'event_score': 0,
    'event_risk': False,
    'stage1g_active': False,
    'stage1g_trigger_type': None
}

# Scenario 3: Full signal detection with Stage 1G classic
signals_full = {
    'social_spike': True,
    'whale_activity': True,
    'orderbook_anomaly': True,
    'volume_spike': True,
    'dex_inflow': True,
    'event_tag': 'partnership',
    'event_score': 10,
    'event_risk': False,
    'stage1g_active': True,
    'stage1g_trigger_type': 'classic'
}

# Scenario 4: Risk token with exploit tag
signals_risk = {
    'social_spike': False,
    'whale_activity': False,
    'orderbook_anomaly': False,
    'volume_spike': False,
    'dex_inflow': False,
    'event_tag': 'exploit',
    'event_score': -15,
    'event_risk': True,
    'stage1g_active': False,
    'stage1g_trigger_type': None
}

print("🧪 Testing PPWCS 2.5 Scoring System")
print("=" * 50)

print(f"\n📊 Scenario 1 - PEPEUSDT (listing tag):")
score1 = compute_ppwcs(signals_pepe)
print(f"Score: {score1}")

print(f"\n📊 Scenario 2 - Pure accumulation (whale + DEX inflow):")
score2 = compute_ppwcs(signals_accumulation)
print(f"Score: {score2}")

print(f"\n📊 Scenario 3 - Full detection + Stage 1G classic:")
signals_full['compressed'] = True
score3 = compute_ppwcs(signals_full)
print(f"Score: {score3}")

print(f"\n📊 Scenario 4 - Risk token (exploit):")
score4 = compute_ppwcs(signals_risk)
print(f"Score: {score4}")

print(f"\n🎯 Summary:")
print(f"Listing tag only: {score1}")
print(f"Pure accumulation: {score2}")
print(f"Full detection: {score3}")
print(f"Risk token: {score4}")