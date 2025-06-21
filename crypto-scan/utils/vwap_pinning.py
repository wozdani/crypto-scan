from utils.data_fetchers import get_all_data

def detect_vwap_pinning(symbol, data):
    """
    VWAP Pinning Detector - wykrywa kontrolowaną akumulację przez whales
    
    Warunki aktywacji:
    - Cena przez ostatnie świece nie oddala się od VWAP
    - Średnia różnica < 0.4% przez kilka świec
    - Oznacza kontrolę whales przed wybiciem
    
    Returns: bool - True jeśli wykryto VWAP pinning
    """

    if not isinstance(data, dict): return False, 0.0
    closes = data.get("recent_closes", [])
    vwaps = data.get("recent_vwaps", [])
       
    if len(closes) != len(vwaps) or len(closes) < 3:
        return False, 0.0

    # Oblicz odchylenia procentowe od VWAP
    deviations = []
    for close, vwap in zip(closes, vwaps):
        if vwap > 0:  # Unikaj dzielenia przez zero
            deviation = abs(close - vwap) / vwap
            deviations.append(deviation)
    
    if len(deviations) < 3:
        return False, 0.0
        
    avg_deviation = sum(deviations) / len(deviations)
    
    # Skalowane kryteria VWAP pinning (złagodzone)
    if avg_deviation < 0.005:
        print(f"[VWAP PINNING] {symbol}: Strong pinning {avg_deviation:.3f}% < 0.5%")
        return True, 1.0
    elif avg_deviation < 0.008:
        print(f"[VWAP PINNING] {symbol}: Medium pinning {avg_deviation:.3f}% < 0.8%")
        return True, 0.7
    elif avg_deviation < 0.012:
        print(f"[VWAP PINNING] {symbol}: Weak pinning {avg_deviation:.3f}% < 1.2%")
        return True, 0.4
    else:
        return False, 0.0

def calculate_vwap(prices, volumes):
    """
    Oblicza VWAP (Volume Weighted Average Price)
    
    Args:
        prices: lista cen (typically close prices)
        volumes: lista wolumenów
    
    Returns: float - VWAP value
    """
    if len(prices) != len(volumes) or len(prices) == 0:
        return 0.0
        
    try:
        total_pv = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        
        if total_volume == 0:
            return 0.0
            
        return total_pv / total_volume
    except Exception as e:
        print(f"❌ Error calculating VWAP: {e}")
        return 0.0

def get_recent_market_data(symbol, limit=5):
    """
    Pobiera ostatnie dane rynkowe potrzebne do analizy VWAP pinning
    
    Args:
        symbol: symbol kryptowaluty
        limit: liczba ostatnich świec do analizy
    
    Returns: dict with recent closes and VWAPs
    """
    try:
        # Pobierz dane rynkowe
        market_data = get_all_data(symbol)
        
        if not market_data:
            return {
                "recent_closes": [],
                "recent_vwaps": [],
                "data_available": False
            }
        
        # W środowisku produkcyjnym będą to rzeczywiste dane z API
        # Na razie zwracamy strukturę do testowania z przykładowymi danymi
        return {
            "recent_closes": [100.0, 100.2, 99.8, 100.1, 99.9],  # Przykładowe ceny zamknięcia
            "recent_vwaps": [100.1, 100.0, 99.9, 100.0, 100.0],   # Przykładowe VWAP
            "data_available": True
        }
        
    except Exception as e:
        print(f"❌ Error getting recent market data for {symbol}: {e}")
        return {
            "recent_closes": [],
            "recent_vwaps": [],
            "data_available": False
        }

def analyze_vwap_control(symbol):
    """
    Analizuje stopień kontroli ceny względem VWAP
    
    Returns: dict with VWAP analysis data
    """
    try:
        market_data = get_recent_market_data(symbol)

        if not isinstance(market_data, dict):
            print(f"❌ market_data nie jest dict: {market_data}")
            return {
                "vwap_pinned": False,
                "avg_deviation": 0.0,
                "control_strength": 0.0,
                "candles_analyzed": 0
            }

        if not market_data.get("data_available"):
            print(f"❌ Brak danych VWAP dla {symbol}")
            return {
                "vwap_pinned": False,
                "avg_deviation": 0.0,
                "control_strength": 0.0,
                "candles_analyzed": 0
            }

        closes = market_data["recent_closes"]
        vwaps = market_data["recent_vwaps"]
        
        # Oblicz statystyki odchyleń
        deviations = []
        for close, vwap in zip(closes, vwaps):
            if vwap > 0:
                deviation = abs(close - vwap) / vwap
                deviations.append(deviation)
        
        if len(deviations) == 0:
            return {
                "vwap_pinned": False,
                "avg_deviation": 0.0,
                "control_strength": 0.0,
                "candles_analyzed": 0
            }
        
        avg_deviation = sum(deviations) / len(deviations)
        max_deviation = max(deviations) if deviations else 0.0
        
        # Siła kontroli (im mniejsze odchylenie, tym większa kontrola)
        control_strength = max(0.0, 1.0 - (avg_deviation / 0.01))  # Normalizacja do 0-1
        
        vwap_result = detect_vwap_pinning(symbol, market_data)
        vwap_pinned = vwap_result[0] if isinstance(vwap_result, tuple) else vwap_result
        
        return {
            "vwap_pinned": vwap_pinned,
            "avg_deviation": avg_deviation,
            "max_deviation": max_deviation,
            "control_strength": control_strength,
            "candles_analyzed": len(deviations)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing VWAP control for {symbol}: {e}")
        return {
            "vwap_pinned": False,
            "avg_deviation": 0.0,
            "control_strength": 0.0,
            "candles_analyzed": 0
        }

def get_vwap_pinning_score(vwap_data):
    """
    Oblicza score na podstawie stopnia VWAP pinning
    
    Returns: int (0-4 points)
    """
    score = 0
    
    # Podstawowy bonus za wykrycie VWAP pinning
    if vwap_data.get("vwap_pinned"):
        score += 4
    
    # Dodatkowy bonus za bardzo silną kontrolę
    control_strength = vwap_data.get("control_strength", 0.0)
    if control_strength > 0.8:  # Bardzo silna kontrola
        score += 1
    elif control_strength > 0.6:  # Silna kontrola
        score += 0.5
    
    # Bonus za konsystentność na większej liczbie świec
    candles_analyzed = vwap_data.get("candles_analyzed", 0)
    if candles_analyzed >= 5:
        score += 1
    
    return min(int(score), 5)  # Max 5 punktów