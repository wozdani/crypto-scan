"""
Pulse Delay Detector - Detektor Kontrolowanych Pauzy
Wykrywa schemat: flat → impuls → flat → impuls (pauzy między ruchami)
Identyfikuje trendy prowadzone przez dużych graczy z kontrolowanymi pauzami
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

        return [float(item[4]) for item in data["result"]["list"]]
    except Exception as e:
        print(f"❌ Exception fetching price series for {symbol}: {e}")
        return []


def detect_pulse_delay(prices: list[float]) -> tuple[bool, str, dict]:
    """
    Wykrywa schemat: flat -> wzrost -> flat -> wzrost (pauzy między impulsami).
    Analizuje serie cen z interwałem 5m.
    
    Args:
        prices: lista cen z ostatnich 3h
        
    Returns:
        tuple: (bool, str, dict) - (wykryto_pulse_delay, opis, szczegóły)
    """
    if len(prices) < 20:
        return False, "Za mało punktów cenowych do analizy pulse delay", {}

    # Oblicz deltas (zmiany procentowe)
    deltas = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    pattern_count = 0
    detected_patterns = []
    
    i = 0
    while i < len(deltas) - 6:
        # Sprawdź wzorzec: flat1 -> impulse1 -> flat2
        flat1 = abs(sum(deltas[i:i+2])) < 0.0006  # flat max ~±0.06%
        impulse1 = sum(deltas[i+2:i+4]) > 0.001   # impuls ≥ +0.1%
        flat2 = abs(sum(deltas[i+4:i+6])) < 0.0006
        
        if flat1 and impulse1 and flat2:
            pattern_count += 1
            
            # Zapisz szczegóły wzorca
            pattern_info = {
                "start_index": i,
                "flat1_change": sum(deltas[i:i+2]) * 100,  # w procentach
                "impulse_change": sum(deltas[i+2:i+4]) * 100,
                "flat2_change": sum(deltas[i+4:i+6]) * 100,
                "pattern_strength": sum(deltas[i+2:i+4])  # siła impulsu
            }
            detected_patterns.append(pattern_info)
            
            i += 6  # skip forward to avoid overlap
        else:
            i += 1
    
    # Oblicz dodatkowe metryki
    total_price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
    avg_impulse_strength = sum(p["pattern_strength"] for p in detected_patterns) / len(detected_patterns) if detected_patterns else 0
    
    details = {
        "pattern_count": pattern_count,
        "detected_patterns": detected_patterns,
        "total_price_change_pct": round(total_price_change * 100, 3),
        "avg_impulse_strength": round(avg_impulse_strength * 100, 3),
        "periods_analyzed": len(prices),
        "pattern_density": round(pattern_count / (len(deltas) / 6), 3) if len(deltas) > 0 else 0
    }
    
    pulse_delay_detected = pattern_count >= 1  # przynajmniej jedna sekwencja
    
    if pulse_delay_detected:
        if pattern_count >= 3:
            description = f"Silny pulse delay - {pattern_count} wzorców kontrolowanych pauzy"
        elif pattern_count == 2:
            description = f"Umiarkowany pulse delay - {pattern_count} wzorce kontrolowanych pauzy"
        else:
            description = f"Słaby pulse delay - {pattern_count} wzorzec kontrolowanej pauzy"
    else:
        description = "Brak pulse delay - chaotyczny ruch bez kontrolowanych pauzy"
    
    return pulse_delay_detected, description, details


def calculate_pulse_delay_score(detection_result: tuple) -> int:
    """
    Oblicza punktację dla Pulse Delay Detection
    
    Args:
        detection_result: wynik z detect_pulse_delay()
        
    Returns:
        int: punkty do dodania (0-15)
    """
    detected, description, details = detection_result
    
    if not detected:
        return 0
    
    pattern_count = details.get("pattern_count", 0)
    avg_impulse_strength = details.get("avg_impulse_strength", 0)
    
    # Bazowy score za wykrycie pulse delay
    base_score = 10
    
    # Bonus za liczbę wzorców
    if pattern_count >= 3:
        base_score += 5  # Silny pulse delay
    elif pattern_count == 2:
        base_score += 3  # Umiarkowany pulse delay
    
    # Bonus za siłę impulsów (jeśli impulsy są mocne, to lepiej)
    if avg_impulse_strength > 0.15:  # >0.15% średnia siła impulsu
        base_score += 2
    
    return min(base_score, 15)  # Maksymalnie 15 punktów


def analyze_pulse_delay_detailed(prices: list[float]) -> dict:
    """
    Szczegółowa analiza pulse delay dla debugowania
    
    Args:
        prices: lista cen historycznych
        
    Returns:
        dict: szczegółowe dane o pulse delay
    """
    if len(prices) < 20:
        return {"error": "Za mało danych cenowych"}
    
    # Pobierz wyniki podstawowej detekcji
    detected, description, basic_details = detect_pulse_delay(prices)
    
    # Dodatkowe analizy
    deltas = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    
    # Analiza volatilności w segmentach
    segment_size = 6
    segment_volatilities = []
    
    for i in range(0, len(deltas) - segment_size, segment_size):
        segment = deltas[i:i + segment_size]
        volatility = sum(abs(d) for d in segment) / len(segment)
        segment_volatilities.append(volatility)
    
    # Analiza trendów w segmentach
    segment_trends = []
    for i in range(0, len(deltas) - segment_size, segment_size):
        segment = deltas[i:i + segment_size]
        trend = sum(segment)
        segment_trends.append(trend)
    
    return {
        **basic_details,
        "description": description,
        "detected": detected,
        "segment_volatilities": [round(v * 100, 3) for v in segment_volatilities],
        "segment_trends": [round(t * 100, 3) for t in segment_trends],
        "avg_segment_volatility": round(sum(segment_volatilities) / len(segment_volatilities) * 100, 3) if segment_volatilities else 0,
        "trend_consistency": len([t for t in segment_trends if t > 0]) / len(segment_trends) if segment_trends else 0
    }


def get_pulse_delay_summary(prices: list[float]) -> str:
    """
    Generuje krótkie podsumowanie pulse delay
    
    Args:
        prices: lista cen
        
    Returns:
        str: podsumowanie pulse delay
    """
    if len(prices) < 20:
        return "insufficient data"
    
    detected, description, details = detect_pulse_delay(prices)
    pattern_count = details.get("pattern_count", 0)
    
    if detected:
        if pattern_count >= 3:
            return "strong controlled flow"
        elif pattern_count == 2:
            return "moderate pulse delay"
        else:
            return "weak pulse pattern"
    else:
        return "chaotic no delays"


def test_pulse_delay_with_synthetic_data():
    """
    Test funkcji z syntetycznymi danymi
    """
    print("🧪 Testing Pulse Delay Detector with synthetic data\n")
    
    # Test 1: Perfect pulse delay pattern
    perfect_pulse = [100.0]
    price = 100.0
    
    # Wzorzec: flat -> impulse -> flat -> impulse -> flat -> impulse
    for cycle in range(3):
        # Flat period (2 periods, ~±0.05% change)
        price += 0.02
        perfect_pulse.append(price)
        price -= 0.01
        perfect_pulse.append(price)
        
        # Impulse period (2 periods, >0.1% change)
        price += 0.08
        perfect_pulse.append(price)
        price += 0.06
        perfect_pulse.append(price)
        
        # Another flat period
        price += 0.01
        perfect_pulse.append(price)
        price -= 0.02
        perfect_pulse.append(price)
    
    # Wypełnij do 25 punktów
    while len(perfect_pulse) < 25:
        price += 0.01
        perfect_pulse.append(price)
    
    detected1, desc1, details1 = detect_pulse_delay(perfect_pulse)
    score1 = calculate_pulse_delay_score((detected1, desc1, details1))
    
    print(f"Perfect pulse delay: {detected1}")
    print(f"Description: {desc1}")
    print(f"Patterns found: {details1['pattern_count']}")
    print(f"Score: {score1}/15")
    
    # Test 2: Chaotic movement (no pulse delay)
    chaotic_data = [100.0]
    price = 100.0
    
    for i in range(24):
        change = 0.05 * ((-1) ** i) * (1 + i * 0.1)  # chaotyczne zmiany
        price += change
        chaotic_data.append(price)
    
    detected2, desc2, details2 = detect_pulse_delay(chaotic_data)
    score2 = calculate_pulse_delay_score((detected2, desc2, details2))
    
    print(f"\nChaotic movement: {detected2}")
    print(f"Description: {desc2}")
    print(f"Patterns found: {details2['pattern_count']}")
    print(f"Score: {score2}/15")
    
    print("\n✅ Pulse Delay Detector tests completed!")


if __name__ == "__main__":
    test_pulse_delay_with_synthetic_data()