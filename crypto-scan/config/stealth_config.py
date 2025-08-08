"""
Stealth Engine Configuration
Centralized configuration for all stealth engine parameters
"""

STEALTH = {
    # Smart Money
    "SM_TRUST_MIN": 0.55,
    "SM_PRED_MIN": 5,
    "SM_REPEAT_MIN_7D": 2,
    "SM_WHALE_MIN_USD": 50_000,
    "SM_EXCLUDE_EXCHANGES": True,
    "SM_MAX_BOOST": 0.5,   # dodawane do base_strength; clamp do 1.0

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

    # Decyzje
    "ALERT_TAU": 0.72,  # próg p stealtu (po kalibracji)
    "USE_CONSENSUS_GATE": True,   # BUY => ALERT, HOLD/AVOID => WATCHLIST (bez wysyłki alertu)
}