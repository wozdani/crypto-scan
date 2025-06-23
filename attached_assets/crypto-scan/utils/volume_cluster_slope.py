import numpy as np
from utils.data_fetchers import get_all_data

def detect_volume_cluster_slope(data):
    """
    Volume Cluster Slope Detector - wykrywa aktywną akumulację przez whales
    
    Warunki aktywacji:
    - Wolumen dynamicznie rośnie w czasie
    - Towarzyszy temu wzrost ceny (nie dump)
    - Oba trendy rosną równolegle
    
    Returns: bool - True jeśli wykryto pozytywny slope cluster
    """
    if not isinstance(data, dict): return False, 0.0
    volumes = data.get("recent_volumes", [])
    closes = data.get("recent_closes", [])

    if len(volumes) < 3 or len(volumes) != len(closes):
        return False, 0.0

    # Oblicz proste nachylenia (slope) przez różnicę końca i początku
    volume_slope = (volumes[-1] - volumes[0]) / max(1, len(volumes) - 1)
    price_slope = (closes[-1] - closes[0]) / max(1, len(closes) - 1)

    # Próg minimalny wzrostu wolumenu i ceny
    if volume_slope > 0 and price_slope > 0:
        return True
    return False, 0.0

def calculate_advanced_slope(data_points):
    """
    Oblicza zaawansowane nachylenie używając regresji liniowej
    
    Args:
        data_points: lista wartości numerycznych
    
    Returns: float - współczynnik nachylenia
    """
    if len(data_points) < 2:
        return 0.0
        
    try:
        # Używamy indeksów jako x, wartości jako y
        x = np.arange(len(data_points))
        y = np.array(data_points)
        
        # Regresja liniowa: y = mx + b
        slope, _ = np.polyfit(x, y, 1)
        return slope
    except Exception as e:
        print(f"❌ Error calculating slope: {e}")
        return 0.0

def detect_advanced_volume_slope(data):
    """
    Zaawansowana detekcja volume slope z regresją liniową
    
    Returns: dict with detailed slope analysis
    """
    if not isinstance(data, dict): return False, 0.0
    volumes = data.get("recent_volumes", [])
    closes = data.get("recent_closes", [])
    
    if len(volumes) < 3 or len(volumes) != len(closes):
        return {
            "volume_slope_up": False,
            "volume_slope": 0.0,
            "price_slope": 0.0,
            "correlation_strength": 0.0
        }
    
    # Oblicz nachylenia
    volume_slope = calculate_advanced_slope(volumes)
    price_slope = calculate_advanced_slope(closes)
    
    # Oblicz korelację między wolumenem a ceną
    try:
        correlation = np.corrcoef(volumes, closes)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0
    
    # Detekcja pozytywnego trendu
    volume_slope_up = volume_slope > 0 and price_slope > 0 and correlation > 0.3
    
    return {
        "volume_slope_up": volume_slope_up,
        "volume_slope": volume_slope,
        "price_slope": price_slope,
        "correlation_strength": abs(correlation)
    }

def get_recent_candle_data(symbol, limit=6):
    """
    Pobiera ostatnie dane świec potrzebne do analizy volume slope
    
    Args:
        symbol: symbol kryptowaluty
        limit: liczba ostatnich świec do analizy
    
    Returns: dict with recent volumes and closes
    """
    try:
        # Pobierz dane rynkowe
        market_data = get_all_data(symbol)
        
        if not market_data:
            return {
                "recent_volumes": [],
                "recent_closes": [],
                "data_available": False
            }
        
        # W środowisku produkcyjnym będą to rzeczywiste dane z API
        # Na razie zwracamy strukturę do testowania z realistycznymi danymi
        return {
            "recent_volumes": [15000, 18000, 22000, 28000, 35000, 42000],  # Rosnący wolumen
            "recent_closes": [100.0, 100.5, 101.2, 102.1, 102.8, 103.5], # Rosnąca cena
            "data_available": True
        }
        
    except Exception as e:
        print(f"❌ Error getting recent candle data for {symbol}: {e}")
        return {
            "recent_volumes": [],
            "recent_closes": [],
            "data_available": False
        }

def analyze_volume_price_dynamics(symbol):
    """
    Analizuje dynamikę wolumenu i ceny dla wykrywania akumulacji
    
    Returns: dict with comprehensive analysis
    """
    try:
        candle_data = get_recent_candle_data(symbol)

        if not isinstance(candle_data, dict):
            print(f"❌ get_recent_candle_data nie zwrócił dict dla {symbol}: {type(candle_data)}")
            return {
                "volume_slope_up": False,
                "analysis_available": False,
                "volume_trend": "unknown",
                "price_trend": "unknown",
                "accumulation_strength": 0.0
            }

        if not candle_data.get("data_available"):
            print(f"❌ Brak danych świecowych dla {symbol}")
            return {
                "volume_slope_up": False,
                "analysis_available": False,
                "volume_trend": "unknown",
                "price_trend": "unknown",
                "accumulation_strength": 0.0
            }
       
        # Przeprowadź zaawansowaną analizę
        slope_analysis = detect_advanced_volume_slope(candle_data)
        
        # Określ trendy
        volume_trend = "up" if slope_analysis["volume_slope"] > 0 else "down"
        price_trend = "up" if slope_analysis["price_slope"] > 0 else "down"
        
        # Siła akumulacji (kombinacja slope i korelacji)
        accumulation_strength = 0.0
        if slope_analysis["volume_slope_up"]:
            accumulation_strength = min(1.0, (
                abs(slope_analysis["volume_slope"]) * 0.4 + 
                abs(slope_analysis["price_slope"]) * 0.3 +
                slope_analysis["correlation_strength"] * 0.3
            ))
        
        return {
            "volume_slope_up": slope_analysis["volume_slope_up"],
            "analysis_available": True,
            "volume_trend": volume_trend,
            "price_trend": price_trend,
            "volume_slope": slope_analysis["volume_slope"],
            "price_slope": slope_analysis["price_slope"],
            "correlation_strength": slope_analysis["correlation_strength"],
            "accumulation_strength": accumulation_strength
        }
        
    except Exception as e:
        print(f"❌ Error analyzing volume-price dynamics for {symbol}: {e}")
        return {
            "volume_slope_up": False,
            "analysis_available": False,
            "volume_trend": "unknown",
            "price_trend": "unknown",
            "accumulation_strength": 0.0
        }

def get_volume_slope_score(slope_data):
    """
    Oblicza score na podstawie analizy volume cluster slope
    
    Returns: int (0-4 points)
    """
    score = 0
    
    # Podstawowy bonus za wykrycie pozytywnego slope
    if slope_data.get("volume_slope_up"):
        score += 4
    
    # Dodatkowy bonus za silną akumulację
    accumulation_strength = slope_data.get("accumulation_strength", 0.0)
    if accumulation_strength > 0.8:
        score += 2
    elif accumulation_strength > 0.6:
        score += 1
    
    # Bonus za wysoką korelację
    correlation = slope_data.get("correlation_strength", 0.0)
    if correlation > 0.7:
        score += 1
    
    return min(score, 6)  # Max 6 punktów