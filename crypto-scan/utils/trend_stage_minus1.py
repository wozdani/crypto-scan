"""
Stage -1 (Trend Mode) - Rhythm and Market Tension Detection
Wykrywa moment napięcia w rynku bez użycia scoringu czy wskaźników technicznych
Oparty na rytmie i harmonii świec - 'czy rynek trzyma oddech'
"""

def detect_stage_minus1(data):
    """
    Stage –1 (Trend Mode) – oparty na rytmie i napięciu rynku.
    Nie stosuje scoringu. Działa na zasadzie 'czy rynek trzyma oddech'.
    
    Args:
        data: Lista świec OHLCV [timestamp, open, high, low, close, volume]
        
    Returns:
        tuple: (bool, str) - (czy_wykryto, opis_sytuacji)
    """
    if len(data) < 6:
        return False, "Za mało świec do analizy rytmu"

    def body(c): 
        """Wysokość ciała świecy"""
        return abs(c[4] - c[1])
    
    def wick_down(c): 
        """Długość dolnego knota"""
        return c[1] - c[3]
    
    def total_range(c): 
        """Całkowity zakres świecy"""
        return c[2] - c[3]

    last = data[-1]
    prev = data[-2]
    three_back = data[-4]

    # 1. Czy rynek zatrzymał się, ale nie odwrócił?
    if last[4] < three_back[4]:
        return False, "Cena cofnęła się za głęboko względem poprzedniego impulsu"

    # 2. Czy świeca wygląda na spokojną, bez paniki?
    if last[4] < last[1]:  # świeca czerwona
        wick_ratio = wick_down(last) / total_range(last) if total_range(last) else 0
        if wick_ratio < 0.2:
            return False, "Korekta bez odbicia – możliwa panika"

    # 3. Czy rytm świec jest spójny?
    body_now = body(last)
    body_prev = body(three_back)
    if body_now == 0 or body_prev == 0:
        return False, "Brak wyraźnych ciał świec"

    similarity = min(body_now, body_prev) / max(body_now, body_prev)
    if similarity < 0.6:
        return False, "Zaburzony rytm ciał świec – brak harmonii"

    # 4. Czy trend nie został zniszczony? (close[-1] > close[-4])
    if last[4] < data[-4][4]:
        return False, "Trend się załamał – kierunek nieutrzymany"

    # ✅ Wszystko wygląda jak zatrzymanie, nie odwrót
    return True, "Ruch wstrzymany, rytm utrzymany, brak paniki – możliwe wybicie"

def analyze_market_rhythm(data):
    """
    Dodatkowa analiza rytmu rynku dla głębszego zrozumienia
    
    Args:
        data: Lista świec OHLCV
        
    Returns:
        dict: Szczegółowe dane o rytmie rynku
    """
    if len(data) < 6:
        return {"error": "Za mało danych"}
        
    def body(c): return abs(c[4] - c[1])
    def wick_down(c): return c[1] - c[3]
    def wick_up(c): return c[2] - c[4]
    def total_range(c): return c[2] - c[3]
    
    last_3_candles = data[-3:]
    
    # Analiza ciał świec
    bodies = [body(candle) for candle in last_3_candles]
    avg_body = sum(bodies) / len(bodies) if bodies else 0
    body_consistency = min(bodies) / max(bodies) if max(bodies) > 0 else 0
    
    # Analiza knotów
    upper_wicks = [wick_up(candle) for candle in last_3_candles]
    lower_wicks = [wick_down(candle) for candle in last_3_candles]
    
    # Trend direction analysis
    trend_direction = "up" if data[-1][4] > data[-4][4] else "down"
    
    # Volatility analysis
    ranges = [total_range(candle) for candle in last_3_candles]
    avg_range = sum(ranges) / len(ranges) if ranges else 0
    
    return {
        "body_consistency": body_consistency,
        "avg_body_size": avg_body,
        "trend_direction": trend_direction,
        "avg_volatility": avg_range,
        "upper_wick_avg": sum(upper_wicks) / len(upper_wicks) if upper_wicks else 0,
        "lower_wick_avg": sum(lower_wicks) / len(lower_wicks) if lower_wicks else 0,
        "rhythm_score": body_consistency * 100,  # 0-100 scale
    }

def get_stage_minus1_details(data):
    """
    Pobiera szczegółowe informacje o Stage -1 dla logging i debugging
    
    Args:
        data: Lista świec OHLCV
        
    Returns:
        dict: Szczegółowe dane diagnostyczne
    """
    if len(data) < 6:
        return {"error": "Insufficient data"}
        
    detected, message = detect_stage_minus1(data)
    rhythm_data = analyze_market_rhythm(data)
    
    return {
        "stage_minus1_detected": detected,
        "detection_message": message,
        "rhythm_analysis": rhythm_data,
        "candles_analyzed": len(data),
        "last_close": data[-1][4],
        "reference_close": data[-4][4],
        "price_progression": data[-1][4] > data[-4][4]
    }