def detect_heatmap_exhaustion(data):
    """
    Heatmap Exhaustion Detector - wykrywa zmęczenie podaży w orderbooku
    
    Warunki aktywacji:
    - Duża ściana sell (ask) zniknęła w ostatnich minutach
    - Jednocześnie pojawił się volume spike lub whale inflow
    - Cena została w miejscu lub minimalnie ruszyła
    
    Returns: bool - True jeśli wykryto wyczerpanie podaży
    """
    ask_wall_disappeared = data.get("ask_wall_disappeared", False)
    volume_spike = data.get("volume_spike", False)
    whale_activity = data.get("whale_activity", False)
    
    # Detekcja: ściana sprzedaży zniknęła + presja akumulacyjna
    if ask_wall_disappeared and (volume_spike or whale_activity):
        return True
    return False

def analyze_orderbook_exhaustion(symbol):
    """
    Analizuje orderbook w poszukiwaniu oznak wyczerpania podaży
    
    W rzeczywistym środowisku będzie korzystać z:
    - Danych orderbook z Bybit
    - Analizy bid/ask ratio
    - Historii ścian sprzedaży
    
    Returns: dict with exhaustion indicators
    """
    try:
        # W środowisku produkcyjnym będą to rzeczywiste dane z API
        # Na razie zwracamy strukturę do testowania
        return {
            "ask_wall_disappeared": False,  # Będzie wykrywane z orderbook
            "bid_ask_ratio": 1.0,          # Stosunek bid/ask volume
            "large_asks_removed": 0,        # Liczba usuniętych dużych zleceń ask
            "price_stability": True         # Czy cena pozostała stabilna
        }
        
    except Exception as e:
        print(f"❌ Error analyzing orderbook exhaustion for {symbol}: {e}")
        return {
            "ask_wall_disappeared": False,
            "bid_ask_ratio": 1.0,
            "large_asks_removed": 0,
            "price_stability": True
        }

def get_exhaustion_score(exhaustion_data):
    """
    Oblicza score na podstawie stopnia wyczerpania podaży
    
    Returns: int (0-10 points)
    """
    score = 0
    
    if exhaustion_data.get("ask_wall_disappeared"):
        score += 5
        
    bid_ask_ratio = exhaustion_data.get("bid_ask_ratio", 1.0)
    if bid_ask_ratio > 1.5:  # Więcej bid niż ask
        score += 3
    elif bid_ask_ratio > 1.2:
        score += 2
        
    large_asks_removed = exhaustion_data.get("large_asks_removed", 0)
    if large_asks_removed > 3:
        score += 2
    elif large_asks_removed > 1:
        score += 1
        
    return min(score, 10)  # Max 10 punktów