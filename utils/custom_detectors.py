"""
Custom detectors for PPWCS 2.6 system
Contains stealth_acc, RSI_flatline and other specialized detectors
"""

def detect_stealth_acc(signals):
    """
    Stealth Accumulation Detector
    Wykrywa cichą akumulację: whale + DEX inflow bez social spike
    """
    if (
        signals.get("whale_activity") and
        signals.get("dex_inflow") and
        not signals.get("social_spike")
    ):
        return True
    return False

def detect_rsi_flatline(rsi_value, signals):
    """
    RSI Flatline Detector
    Wykrywa płaską linię RSI (45-55) z whale/DEX aktywnością
    """
    if rsi_value is None:
        return False
        
    if 45 <= rsi_value <= 55:
        if signals.get("dex_inflow") or signals.get("whale_activity"):
            return True
    return False

def get_rsi_from_data(data):
    """
    Extract RSI value from market data
    Returns RSI value or None if not available
    """
    if not isinstance(data, dict):
        return None
        
    # Try to get RSI from various possible fields
    rsi_fields = ["rsi", "rsi_14", "RSI", "rsi_value"]
    for field in rsi_fields:
        if field in data and data[field] is not None:
            try:
                return float(data[field])
            except (ValueError, TypeError):
                continue
                
    # If no RSI available, return None
    return None