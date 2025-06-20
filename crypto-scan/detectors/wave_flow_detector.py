"""
Wave Flow Detector - Rytmiczny Ruch Ceny
Wykrywa falowy rytm trendu bez użycia klasycznej analizy świec
Bazuje na zachowaniu ceny w czasie jak intuicyjny trader
"""

import requests

def get_price_series_bybit(symbol: str) -> list[float]:
    """
    Pobiera dane 5-minutowe z ostatnich 3h (36 punktów) z Bybit API
    
    Args:
        symbol: Symbol trading pair (np. 'BTCUSDT')
        
    Returns:
        list: seria cen zamknięcia z ostatnich 3h
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "interval": "5",
        "limit": 36,
        "category": "linear",
        "symbol": symbol
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data["retCode"] != 0:
            print(f"❌ Bybit API error for {symbol}: {data['retMsg']}")
            return []

        # używamy ceny close z każdej świecy
        prices = [float(item[4]) for item in data["result"]["list"]]
        return prices

    except Exception as e:
        print(f"❌ Exception fetching price series for {symbol}: {e}")
        return []


def detect_rhythmic_price_flow(prices: list[float]) -> tuple[bool, str, dict]:
    """
    Wykrywa rytmiczny flow ceny: podejście → cofka → podejście
    
    Args:
        prices: lista cen z ostatnich 3h, np. co 5 minut (≈36 punktów)
        
    Returns:
        tuple: (bool, str, dict) - (wykryto, opis, szczegóły)
    """
    if len(prices) < 20:
        return False, "Za mało punktów cenowych do analizy rytmu", {}
    
    # Oblicz ruchy procentowe między kolejnymi cenami
    moves = []
    for i in range(1, len(prices)):
        delta = (prices[i] - prices[i-1]) / prices[i-1]
        moves.append(round(delta, 5))  # np. 0.00123 = +0.123%
    
    detected_patterns = []
    strongest_pattern = {"strength": 0, "position": 0}
    
    # Szukaj wzorców: leg1 (podejście) → pullback (cofka) → leg2 (podejście)
    for i in range(5, len(moves) - 8):
        leg1 = sum(moves[i:i+3])        # 3 ruchy w górę
        pullback = sum(moves[i+3:i+5])  # 2 ruchy w dół 
        leg2 = sum(moves[i+5:i+8])      # 3 ruchy w górę
        
        # Warunki rytmicznego flow
        leg1_valid = leg1 > 0.001  # min +0.10% wzrost
        pullback_valid = pullback < 0  # spadek
        pullback_proportional = abs(pullback) < leg1 * 0.7  # cofka max 70% podejścia
        leg2_valid = leg2 > 0.001  # kolejny wzrost
        
        if leg1_valid and pullback_valid and pullback_proportional and leg2_valid:
            pattern_strength = (leg1 + leg2) / abs(pullback) if pullback != 0 else 0
            
            detected_patterns.append({
                "position": i,
                "leg1": round(leg1 * 100, 3),      # % wzrost
                "pullback": round(pullback * 100, 3),  # % spadek  
                "leg2": round(leg2 * 100, 3),      # % wzrost
                "strength": round(pattern_strength, 2)
            })
            
            if pattern_strength > strongest_pattern["strength"]:
                strongest_pattern = {
                    "strength": pattern_strength,
                    "position": i,
                    "leg1": leg1,
                    "pullback": pullback,
                    "leg2": leg2
                }
    
    if detected_patterns:
        return True, f"Wykryto rytmiczny flow ceny ({len(detected_patterns)} wzorców)", {
            "patterns_count": len(detected_patterns),
            "strongest_pattern": strongest_pattern,
            "all_patterns": detected_patterns[-3:],  # ostatnie 3 wzorce
            "flow_quality": "high" if strongest_pattern["strength"] > 2.0 else "medium"
        }
    
    return False, "Brak rytmicznego flow - cena chaotyczna lub jednokierunkowa", {}


def analyze_price_rhythm_detailed(prices: list[float]) -> dict:
    """
    Szczegółowa analiza rytmu cenowego dla debugowania i optymalizacji
    
    Args:
        prices: lista cen historycznych
        
    Returns:
        dict: szczegółowe dane o rytmie ceny
    """
    if len(prices) < 10:
        return {"error": "Za mało danych cenowych"}
    
    # Podstawowe statystyki
    price_change_total = (prices[-1] - prices[0]) / prices[0]
    max_price = max(prices)
    min_price = min(prices)
    volatility = (max_price - min_price) / prices[0]
    
    # Analiza ruchów
    moves = []
    positive_moves = 0
    negative_moves = 0
    
    for i in range(1, len(prices)):
        delta = (prices[i] - prices[i-1]) / prices[i-1]
        moves.append(delta)
        
        if delta > 0:
            positive_moves += 1
        elif delta < 0:
            negative_moves += 1
    
    # Momentum analysis
    recent_moves = moves[-10:] if len(moves) >= 10 else moves
    recent_momentum = sum(recent_moves)
    
    # Trend consistency
    trend_direction = "up" if price_change_total > 0 else "down"
    trend_strength = abs(price_change_total)
    
    return {
        "total_periods": len(prices),
        "price_change_pct": round(price_change_total * 100, 3),
        "volatility_pct": round(volatility * 100, 3),
        "positive_moves": positive_moves,
        "negative_moves": negative_moves,
        "move_ratio": round(positive_moves / (positive_moves + negative_moves), 3) if (positive_moves + negative_moves) > 0 else 0,
        "recent_momentum": round(recent_momentum * 100, 3),
        "trend_direction": trend_direction,
        "trend_strength": round(trend_strength * 100, 3),
        "rhythm_quality": "smooth" if volatility < 0.05 else "volatile"
    }


def get_price_series_from_candles(candles: list, lookback_minutes: int = 180) -> list[float]:
    """
    Konwertuje świece na serię cen dla analizy rytmu
    
    Args:
        candles: lista świec OHLCV
        lookback_minutes: ile minut wstecz analizować
        
    Returns:
        list: seria cen zamknięcia
    """
    if not candles:
        return []
    
    # Dla świec 15-minutowych, 180 minut = 12 świec
    # Ale pobieramy close price z każdej świecy jako punkt cenowy
    max_candles = lookback_minutes // 15 if lookback_minutes >= 15 else len(candles)
    recent_candles = candles[-max_candles:] if len(candles) >= max_candles else candles
    
    # Wyciągnij ceny zamknięcia
    prices = [float(candle[4]) for candle in recent_candles]  # candle[4] = close price
    
    return prices


def calculate_rhythmic_score(detection_result: tuple) -> int:
    """
    Oblicza punktację dla rytmicznego flow ceny
    
    Args:
        detection_result: wynik z detect_rhythmic_price_flow()
        
    Returns:
        int: punkty do dodania (0-25)
    """
    detected, description, details = detection_result
    
    if not detected:
        return 0
    
    patterns_count = details.get("patterns_count", 0)
    flow_quality = details.get("flow_quality", "medium")
    strongest_pattern = details.get("strongest_pattern", {})
    strength = strongest_pattern.get("strength", 0)
    
    # Bazowy score za wykrycie
    base_score = 15
    
    # Bonus za jakość flow
    if flow_quality == "high":
        base_score += 7
    elif flow_quality == "medium":
        base_score += 3
    
    # Bonus za siłę wzorca
    if strength > 2.5:
        base_score += 3
    elif strength > 2.0:
        base_score += 2
    
    # Ogranicz do maksymalnych 25 punktów
    return min(base_score, 25)