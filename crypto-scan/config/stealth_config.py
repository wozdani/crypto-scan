"""
Stealth Engine Configuration
Centralized configuration for all stealth engine parameters
"""

STEALTH = {
    # Smart Money - Dynamic whale thresholds based on volume
    "SM_TRUST_MIN": 0.55,
    "SM_PRED_MIN": 5,
    "SM_REPEAT_MIN_7D": 2,
    "SM_WHALE_MIN_USD": 50_000,
    "SM_EXCLUDE_EXCHANGES": True,
    "SM_MAX_BOOST": 0.5,   # dodawane do base_strength; clamp do 1.0
    "WHALE_THRESHOLD_MULTIPLIER": 0.02,  # Dynamic: min_usd = volume_24h * multiplier
    "WHALE_THRESHOLD_MIN": 25_000,       # Absolute minimum threshold
    "WHALE_THRESHOLD_MAX": 200_000,      # Maximum threshold even for high volume

    # Orderbook (tylko realny L2)
    "OB_ENABLE_ON_SYNTHETIC": False,
    "OB_DEPTH_TOP10_MIN_USD": 200_000,
    "OB_WALL_MIN_USD": 50_000,
    "OB_OFI_MIN": 2.0,
    "OB_QUEUE_IMB_MIN": 0.6,

    # Agregacja
    "DEFAULT_WEIGHT": 0.10,
    "ALLOWED_SIGNALS": [
        "whale_ping", "dex_inflow", "repeated_address_boost",
        "velocity_boost", "inflow_momentum_boost", "multi_address_group_activity",
        "diamond_ai", "californium_ai", "spoofing_layers", "large_bid_walls",
        "orderbook_imbalance", "bid_ask_spread_tightening", "volume_spike",
        "volume_accumulation", "volume_slope", "ghost_orders",
        "liquidity_absorption", "event_tag", "spoofing_detected",
        "ask_wall_removal", "cross_token_activity_boost"
    ],

    # Decyzje - Hard gating logic
    "ALERT_TAU": 0.72,  # próg p stealtu (po kalibracji)
    "USE_CONSENSUS_GATE": True,   # BUY => ALERT, HOLD/AVOID => WATCHLIST (bez wysyłki alertu)
    "HARD_GATING": True,          # Alert only when whale>=0.8 AND dex>=0.8 AND p>=τ
    "MIN_WHALE_STRENGTH": 0.8,    # Minimum whale strength for alert
    "MIN_DEX_STRENGTH": 0.8,      # Minimum DEX inflow strength for alert
    "REMOVE_SCORE_FALLBACK": True,  # Remove "score≥0.7→alert" fallback logic
    
    # DEX Inflow - BSC specific
    "DEX_INFLOW_BSC_ENABLED": True,
    "DEX_INFLOW_UNKNOWN_ON_FAIL": True,  # Return UNKNOWN instead of 0 when uncertain
}