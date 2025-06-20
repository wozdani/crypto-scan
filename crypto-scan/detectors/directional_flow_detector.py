"""
Directional Flow Detector - Naturalny Kierunkowy Ruch Ceny
Wykrywa czy cena porusza się naturalnie i spójnie kierunkowo
czy chaotycznie i szarpanie (memowe tokeny)
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


def detect_directional_flow(prices: list[float]) -> tuple[bool, str, dict]:
    """
    Wykrywa, czy ruch ceny jest spójny kierunkowo (niechaotyczny).
    Jeśli flow jest chaotyczny – zwraca False.
    
    Args:
        prices: lista cen z ostatnich 3h
        
    Returns:
        tuple: (bool, str, dict) - (spójny_flow, opis, szczegóły)
    """
    if len(prices) < 20:
        return False, "Za mało punktów cenowych do analizy kierunku", {}

    direction_changes = 0
    last_dir = 0
    total_moves = 0
    up_moves = 0
    down_moves = 0
    
    # Analiza zmian kierunku
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i-1]
        
        if abs(delta) > 0:  # Tylko znaczące ruchy
            total_moves += 1
            
            if delta > 0:
                up_moves += 1
                if last_dir != 1:
                    direction_changes += 1
                    last_dir = 1
            elif delta < 0:
                down_moves += 1
                if last_dir != -1:
                    direction_changes += 1
                    last_dir = -1

    # Oblicz wskaźniki spójności
    chaos_ratio = direction_changes / len(prices) if len(prices) > 0 else 1.0
    move_ratio = up_moves / total_moves if total_moves > 0 else 0.5
    
    # Dodatowe metryki
    price_change_total = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
    volatility = (max(prices) - min(prices)) / prices[0] if prices[0] != 0 else 0
    
    # Kryteria naturalnego flow
    is_natural = chaos_ratio < 0.4  # mniej niż 40% zmian kierunku
    
    # Określ jakość flow
    if chaos_ratio < 0.25:
        flow_quality = "excellent"
    elif chaos_ratio < 0.35:
        flow_quality = "good"
    elif chaos_ratio < 0.45:
        flow_quality = "acceptable"
    else:
        flow_quality = "chaotic"
    
    details = {
        "chaos_ratio": round(chaos_ratio, 3),
        "direction_changes": direction_changes,
        "total_periods": len(prices),
        "up_moves": up_moves,
        "down_moves": down_moves,
        "move_ratio": round(move_ratio, 3),
        "price_change_pct": round(price_change_total * 100, 3),
        "volatility_pct": round(volatility * 100, 3),
        "flow_quality": flow_quality
    }
    
    if is_natural:
        description = f"Naturalny kierunkowy flow - {flow_quality} spójność ({round((1-chaos_ratio)*100, 1)}%)"
    else:
        description = f"Chaotyczny flow - zbyt dużo zmian kierunku ({round(chaos_ratio*100, 1)}%)"
    
    return is_natural, description, details


def calculate_directional_score(detection_result: tuple) -> int:
    """
    Oblicza punktację dla kierunkowego flow ceny
    
    Args:
        detection_result: wynik z detect_directional_flow()
        
    Returns:
        int: punkty do dodania/odjęcia (+25 do -10)
    """
    detected, description, details = detection_result
    
    if not detected:
        return -10  # Kara za chaotyczny ruch
    
    chaos_ratio = details.get("chaos_ratio", 1.0)
    flow_quality = details.get("flow_quality", "acceptable")
    
    # Bazowy score za wykrycie naturalnego flow
    base_score = 15
    
    # Bonus za jakość flow
    if flow_quality == "excellent":
        base_score += 10  # Maksymalny bonus
    elif flow_quality == "good":
        base_score += 7
    elif flow_quality == "acceptable":
        base_score += 3
    
    # Ogranicz do maksymalnych 25 punktów
    return min(base_score, 25)


def analyze_directional_trends(prices: list[float]) -> dict:
    """
    Szczegółowa analiza trendów kierunkowych
    
    Args:
        prices: lista cen historycznych
        
    Returns:
        dict: szczegółowe dane o trendach kierunkowych
    """
    if len(prices) < 10:
        return {"error": "Za mało danych cenowych"}
    
    # Podziel na segmenty i analizuj trendy
    segments = []
    segment_size = max(5, len(prices) // 4)  # 4 segmenty
    
    for i in range(0, len(prices) - segment_size, segment_size):
        segment = prices[i:i + segment_size]
        if len(segment) >= 3:
            trend = "up" if segment[-1] > segment[0] else "down"
            change = (segment[-1] - segment[0]) / segment[0] if segment[0] != 0 else 0
            segments.append({
                "start_idx": i,
                "trend": trend,
                "change_pct": round(change * 100, 3)
            })
    
    # Dominujący kierunek
    up_segments = len([s for s in segments if s["trend"] == "up"])
    down_segments = len([s for s in segments if s["trend"] == "down"])
    
    if up_segments > down_segments:
        dominant_trend = "bullish"
    elif down_segments > up_segments:
        dominant_trend = "bearish"
    else:
        dominant_trend = "sideways"
    
    # Consistency score
    consistency = max(up_segments, down_segments) / len(segments) if segments else 0
    
    return {
        "segments_analyzed": len(segments),
        "up_segments": up_segments,
        "down_segments": down_segments,
        "dominant_trend": dominant_trend,
        "trend_consistency": round(consistency, 3),
        "segment_details": segments
    }


def get_flow_summary(prices: list[float]) -> str:
    """
    Generuje krótkie podsumowanie charakteru flow ceny
    
    Args:
        prices: lista cen
        
    Returns:
        str: podsumowanie flow
    """
    if len(prices) < 10:
        return "insufficient data"
    
    detected, description, details = detect_directional_flow(prices)
    flow_quality = details.get("flow_quality", "unknown")
    chaos_ratio = details.get("chaos_ratio", 1.0)
    
    if detected:
        if flow_quality == "excellent":
            return "smooth trend flow"
        elif flow_quality == "good":
            return "consistent direction"
        else:
            return "acceptable trend"
    else:
        if chaos_ratio > 0.6:
            return "highly chaotic"
        else:
            return "moderately choppy"