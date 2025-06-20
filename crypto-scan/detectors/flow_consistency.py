"""
Flow Consistency Index (FCI) - Wskaźnik Spójności Kierunku
Mierzy spójność kierunku ceny na podstawie sekwencji mikro-ruchów
Nie analizuje formacji - tylko zachowanie ceny w czasie
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


def compute_flow_consistency(prices: list[float]) -> float:
    """
    Zwraca stosunek dominujących ruchów do ogólnych.
    Np. 0.72 = 72% ruchów w tym samym kierunku.
    
    Args:
        prices: lista cen z ostatnich 3h
        
    Returns:
        float: wskaźnik spójności 0.0-1.0
    """
    if len(prices) < 10:
        return 0.0

    ups = 0
    downs = 0

    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            ups += 1
        elif delta < 0:
            downs += 1

    total = ups + downs
    if total == 0:
        return 0.0

    dominant = max(ups, downs)
    return dominant / total  # wynik 0–1


def analyze_flow_consistency_detailed(prices: list[float]) -> dict:
    """
    Szczegółowa analiza spójności flow dla debugowania
    
    Args:
        prices: lista cen historycznych
        
    Returns:
        dict: szczegółowe dane o spójności flow
    """
    if len(prices) < 10:
        return {"error": "Za mało danych cenowych"}
    
    ups = 0
    downs = 0
    unchanged = 0
    move_sizes = []
    
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            ups += 1
            move_sizes.append(abs(delta))
        elif delta < 0:
            downs += 1
            move_sizes.append(abs(delta))
        else:
            unchanged += 1
    
    total_moves = ups + downs
    consistency = max(ups, downs) / total_moves if total_moves > 0 else 0.0
    
    # Określ dominujący kierunek
    if ups > downs:
        dominant_direction = "bullish"
        directional_strength = ups / total_moves if total_moves > 0 else 0.0
    elif downs > ups:
        dominant_direction = "bearish"
        directional_strength = downs / total_moves if total_moves > 0 else 0.0
    else:
        dominant_direction = "sideways"
        directional_strength = 0.5
    
    # Oblicz średni rozmiar ruchu
    avg_move_size = sum(move_sizes) / len(move_sizes) if move_sizes else 0.0
    
    # Oblicz zmianę ceny całkowita
    total_price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
    
    return {
        "consistency": round(consistency, 3),
        "up_moves": ups,
        "down_moves": downs,
        "unchanged_moves": unchanged,
        "total_moves": total_moves,
        "dominant_direction": dominant_direction,
        "directional_strength": round(directional_strength, 3),
        "avg_move_size": round(avg_move_size, 6),
        "total_price_change_pct": round(total_price_change * 100, 3),
        "periods_analyzed": len(prices)
    }


def calculate_flow_consistency_score(consistency: float) -> tuple[int, str]:
    """
    Oblicza punktację dla Flow Consistency Index
    
    Args:
        consistency: wynik z compute_flow_consistency() (0.0-1.0)
        
    Returns:
        tuple: (punkty, opis)
    """
    if consistency >= 0.75:
        return 15, f"wysoka spójność {round(consistency*100)}% – kierunkowy trend"
    elif consistency >= 0.65:
        return 8, f"dobra spójność {round(consistency*100)}% – umiarkowany trend"
    elif consistency >= 0.55:
        return 0, f"neutralna spójność {round(consistency*100)}% – brak bonusu"
    else:
        return -10, f"niska spójność {round(consistency*100)}% – kara za chaos"


def get_flow_consistency_summary(prices: list[float]) -> str:
    """
    Generuje krótkie podsumowanie spójności flow
    
    Args:
        prices: lista cen
        
    Returns:
        str: podsumowanie spójności
    """
    if len(prices) < 10:
        return "insufficient data"
    
    consistency = compute_flow_consistency(prices)
    
    if consistency >= 0.75:
        return "high consistency trend"
    elif consistency >= 0.65:
        return "moderate consistency"
    elif consistency >= 0.55:
        return "neutral flow"
    else:
        return "chaotic inconsistent"


def test_flow_consistency_with_synthetic_data():
    """
    Test funkcji z syntetycznymi danymi dla weryfikacji
    """
    print("🧪 Testing Flow Consistency Index with synthetic data\n")
    
    # Test 1: Silny trend wzrostowy (26 up, 10 down)
    strong_uptrend = [100.0]
    price = 100.0
    
    # 26 ruchów w górę
    for i in range(26):
        price += 0.1
        strong_uptrend.append(price)
    
    # 10 ruchów w dół
    for i in range(10):
        price -= 0.05
        strong_uptrend.append(price)
    
    consistency1 = compute_flow_consistency(strong_uptrend)
    score1, desc1 = calculate_flow_consistency_score(consistency1)
    
    print(f"Strong uptrend: consistency={consistency1:.3f}, score={score1}, {desc1}")
    
    # Test 2: Chaos (up-down-up-down)
    chaotic_data = [100.0]
    price = 100.0
    
    for i in range(17):  # 17 cykli up-down
        price += 0.1 if i % 2 == 0 else -0.1
        chaotic_data.append(price)
    
    consistency2 = compute_flow_consistency(chaotic_data)
    score2, desc2 = calculate_flow_consistency_score(consistency2)
    
    print(f"Chaotic pattern: consistency={consistency2:.3f}, score={score2}, {desc2}")
    
    # Test 3: Umiarkowany trend
    moderate_trend = [100.0]
    price = 100.0
    
    # 20 up, 16 down = 55.6% consistency
    for i in range(20):
        price += 0.08
        moderate_trend.append(price)
    
    for i in range(16):
        price -= 0.04
        moderate_trend.append(price)
    
    consistency3 = compute_flow_consistency(moderate_trend)
    score3, desc3 = calculate_flow_consistency_score(consistency3)
    
    print(f"Moderate trend: consistency={consistency3:.3f}, score={score3}, {desc3}")
    
    print("\n✅ Flow Consistency Index tests completed!")


if __name__ == "__main__":
    test_flow_consistency_with_synthetic_data()