"""
Uptrend 15M Detector
Replaces old chaotic flow detector with proper trend detection
"""

def is_uptrend_15m(prices_15m: list[float]) -> bool:
    """
    Wykrywa czy mamy uptrend na 15-minutowych świecach
    
    Args:
        prices_15m: Lista cen z ostatnich 15-minutowych świec
        
    Returns:
        bool: True jeśli wykryty uptrend
    """
    if len(prices_15m) < 6:
        return False
    
    # Sprawdź postęp w górę (ostatnia cena wyższa niż 4 okresy wcześniej)
    upward_progress = prices_15m[-1] > prices_15m[-4]
    
    # Policz ile z ostatnich 5 okresów było wzrostowych
    up_count = sum(1 for i in range(-5, 0) if prices_15m[i] > prices_15m[i-1])
    dominance = up_count >= 3  # Dominacja wzrostów
    
    # Sprawdź czy korekty są małe (< 1%)
    corrections_ok = all(
        (abs(prices_15m[i] - prices_15m[i-1]) / prices_15m[i-1]) < 0.01
        for i in range(-3, 0)
        if prices_15m[i] < prices_15m[i-1]
    )
    
    return upward_progress and dominance and corrections_ok


def is_absorption_entry_trigger(prices_5m: list[float], volumes_ask: list[float], orderbook: dict) -> bool:
    """
    Wykrywa moment wejścia na absorpcji - gdy presja sprzedających słabnie
    
    Args:
        prices_5m: Lista cen 5-minutowych
        volumes_ask: Lista wolumenów ask
        orderbook: Dane orderbooka
        
    Returns:
        bool: True jeśli wykryty trigger wejścia
    """
    if len(prices_5m) < 5 or len(volumes_ask) < 3:
        return False
    
    # Sprawdź czy były minimum 2 czerwone świece z ostatnich 3
    red_count = sum(1 for i in range(-3, 0) if prices_5m[i] < prices_5m[i-1])
    if red_count < 2:
        return False
    
    # Sprawdź czy wolumen ask maleje (presja sprzedających słabnie)
    ask_trend_down = volumes_ask[-3] > volumes_ask[-2] > volumes_ask[-1]
    
    # Sprawdź presję bid/ask w orderbooku
    bids = orderbook.get("bids", [])[:5]
    asks = orderbook.get("asks", [])[:5]
    
    if not bids or not asks:
        return False
    
    bid_vol = sum([float(b[1]) for b in bids])
    ask_vol = sum([float(a[1]) for a in asks])
    
    bid_pressure = (bid_vol / ask_vol) > 1.4 if ask_vol > 0 else False
    
    # Sprawdź czy ostatnia świeca była odbiciem
    last_rebound = prices_5m[-1] > prices_5m[-2]
    
    return ask_trend_down and bid_pressure and last_rebound


def detect_trend_entry_signal(prices_15m: list[float], prices_5m: list[float], 
                             volumes_ask: list[float], orderbook: dict) -> tuple[bool, str, dict]:
    """
    Główna funkcja detektora wejścia w trend
    Zastępuje stary detektor chaotic flow
    
    Args:
        prices_15m: Ceny 15-minutowe
        prices_5m: Ceny 5-minutowe
        volumes_ask: Wolumeny ask
        orderbook: Dane orderbooka
        
    Returns:
        tuple: (detected, description, details)
    """
    uptrend_15m = is_uptrend_15m(prices_15m)
    absorption_trigger = is_absorption_entry_trigger(prices_5m, volumes_ask, orderbook)
    
    details = {
        "uptrend_15m": uptrend_15m,
        "absorption_trigger": absorption_trigger,
        "prices_15m_count": len(prices_15m),
        "prices_5m_count": len(prices_5m),
        "orderbook_available": bool(orderbook)
    }
    
    if uptrend_15m and absorption_trigger:
        description = "Trend Mode Entry: Uptrend 15M + Trigger 5M (Absorption)"
        return True, description, details
    elif uptrend_15m:
        description = "Uptrend 15M detected, waiting for absorption trigger"
        return False, description, details
    else:
        description = "No uptrend pattern on 15M timeframe"
        return False, description, details