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
    RSI Flatline Detector - USUNIĘTO WARUNEK RSI (45-55)
    Teraz wykrywa tylko kombinację whale/DEX aktywności (bez RSI ograniczenia)
    """
    print(f"[DEBUG] detect_rsi_flatline - RSI: {rsi_value}, signals: {signals}")
    
    # Usuń warunek RSI - sprawdzaj tylko whale/DEX aktywność
    whale_activity = signals.get("whale_activity", False)
    dex_inflow = signals.get("dex_inflow", False)
    
    result = whale_activity or dex_inflow
    print(f"[DEBUG] RSI flatline result: {result} (whale: {whale_activity}, dex: {dex_inflow})")
    
    return result

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