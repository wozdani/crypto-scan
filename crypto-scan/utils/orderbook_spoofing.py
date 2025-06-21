import time
from datetime import datetime, timedelta, timezone

def detect_orderbook_spoofing(symbol):
    """
    Orderbook Spoofing Detector - wykrywa manipulacje orderbooka
    
    Warunki aktywacji:
    - Duża ściana ask pojawiła się i zniknęła w krótkim czasie
    - Czas życia ściany < 90 sekund (bait pattern)
    - Jednocześnie whale activity lub volume spike
    
    Returns: (bool, float) - True jeśli wykryto spoofing, score
    """

    try:
        # Sprawdź czy symbol jest string
        if not isinstance(symbol, str):
            return False, 0.0
            
        # Analizuj orderbook walls dla symbolu
        data = analyze_orderbook_walls(symbol)
        
        if not isinstance(data, dict):
            return False, 0.0
            
        ask_wall_appeared = data.get("ask_wall_appeared", False)
        ask_wall_disappeared = data.get("ask_wall_disappeared", False)
        wall_lifetime = data.get("ask_wall_lifetime_sec", 0)
        whale = data.get("whale_activity", False)
        volume_spike = data.get("volume_spike", False)

        # Detekcja: krótkotrwała ściana + presja akumulacyjna
        if ask_wall_appeared and ask_wall_disappeared and wall_lifetime < 90:
            if whale or volume_spike:
                print(f"[ORDERBOOK SPOOFING] {symbol}: Wall lifetime {wall_lifetime}s < 90s with whale/volume")
                return True, wall_lifetime
        return False, 0.0
        
    except Exception as e:
        print(f"❌ Error in detect_orderbook_spoofing for {symbol}: {e}")
        return False, 0.0

def analyze_orderbook_walls(symbol):
    """
    Analizuje historię ścian w orderbooku
    
    W środowisku produkcyjnym będzie korzystać z:
    - Historycznych danych orderbook z Bybit
    - Śledzenia lifecycle dużych zleceń ask/bid
    - Analizy timing ścian względem ruchów cenowych
    
    Returns: dict with wall analysis data
    """
    try:
        # W środowisku produkcyjnym będą to rzeczywiste dane z API
        # Na razie zwracamy strukturę do testowania
        return {
            "ask_wall_appeared": False,      # Czy pojawiła się duża ściana ask
            "ask_wall_disappeared": False,   # Czy ściana zniknęła
            "ask_wall_lifetime_sec": 0,      # Czas życia ściany w sekundach
            "wall_size_usd": 0,              # Rozmiar ściany w USD
            "walls_detected_count": 0,       # Liczba wykrytych ścian w ostatnich 15 min
            "bid_wall_strength": 0           # Siła ścian bid (support)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing orderbook walls for {symbol}: {e}")
        return {
            "ask_wall_appeared": False,
            "ask_wall_disappeared": False,
            "ask_wall_lifetime_sec": 0,
            "wall_size_usd": 0,
            "walls_detected_count": 0,
            "bid_wall_strength": 0
        }

def get_spoofing_score(spoofing_data):
    """
    Oblicza score na podstawie stopnia podejrzenia manipulacji
    
    Returns: int (0-8 points)
    """
    score = 0
    
    # Podstawowy bonus za wykrycie spoofingu
    if spoofing_data.get("spoofing_suspected"):
        score += 3
        
    # Bonus za rozmiar ściany
    wall_size = spoofing_data.get("wall_size_usd", 0)
    if wall_size > 100000:  # Ściana > $100k
        score += 3
    elif wall_size > 50000:  # Ściana > $50k
        score += 2
    elif wall_size > 25000:  # Ściana > $25k
        score += 1
        
    # Bonus za częstotliwość
    walls_count = spoofing_data.get("walls_detected_count", 0)
    if walls_count > 2:  # Wielokrotne ściany w 15 min
        score += 2
    elif walls_count > 1:
        score += 1
        
    # Penalty za silne bid walls (mniej podejrzane)
    bid_strength = spoofing_data.get("bid_wall_strength", 0)
    if bid_strength > 0.7:  # Silne wsparcie
        score = max(0, score - 1)
        
    return min(score, 8)  # Max 8 punktów

def track_wall_lifecycle(symbol, current_orderbook):
    """
    Śledzi lifecycle ścian orderbooka dla wykrywania spoofingu
    
    W produkcji będzie:
    - Zapisywać stan orderbooka co 5-10 sekund
    - Śledzić pojawianie/znikanie dużych zleceń
    - Mierzyć czas życia ścian
    
    Returns: dict with lifecycle data
    """
    # Placeholder dla rzeczywistej implementacji
    # W produkcji będzie zapisywać dane do pliku/bazy
    return {
        "wall_tracking_active": True,
        "last_update": datetime.now(timezone.utc).isoformat(),
        "tracked_walls": []
    }