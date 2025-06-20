"""
Calm Before The Trend Detector - 10th Layer of Flow Analysis
Wykrywa sytuację, gdy volatility jest bardzo niska, ale cena zaczyna się ruszać
mimo że jeszcze nic nie wskazuje na wybicie.
"""

import statistics
from typing import Tuple, Dict, List

def detect_calm_before_trend(prices: List[float]) -> Tuple[bool, str, Dict]:
    """
    Wykrywa: niska zmienność + początek ruchu = napięcie przed trendem
    
    Args:
        prices: Lista cen 5m z ostatnich 2h (~24 punktów)
        
    Returns:
        Tuple containing:
        - detected: bool
        - description: str (Polish)
        - details: dict with analysis data
    """
    details = {
        "sufficient_data": len(prices) >= 20,
        "volatility_zscore": 0.0,
        "recent_change": 0.0,
        "low_volatility": False,
        "price_moving": False,
        "pattern_strength": "none"
    }
    
    if len(prices) < 20:
        return False, "Niewystarczające dane (potrzeba ≥20 punktów)", details
    
    try:
        # Oblicz delty procentowe między kolejnymi cenami
        deltas = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        # Oblicz odchylenie standardowe (zmienność)
        std_dev = statistics.stdev(deltas)
        details["volatility_zscore"] = std_dev
        
        # Próg Z-score dla niskiej zmienności (~0.08%)
        z_score_threshold = 0.001
        details["low_volatility"] = std_dev < z_score_threshold
        
        # Zmiana z ostatnich 30 minut (6 punktów po 5 min)
        total_change = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        details["recent_change"] = total_change
        
        # Próg dla początkowego ruchu (0.3%)
        movement_threshold = 0.003
        details["price_moving"] = total_change > movement_threshold
        
        # Klasyfikacja siły wzorca
        if details["low_volatility"] and details["price_moving"]:
            if total_change > 0.005:  # >0.5%
                details["pattern_strength"] = "strong"
            elif total_change > 0.004:  # >0.4%
                details["pattern_strength"] = "moderate"
            else:  # 0.3-0.4%
                details["pattern_strength"] = "weak"
        
        # Główna logika detekcji
        calm_detected = details["low_volatility"] and details["price_moving"]
        
        if calm_detected:
            description = f"Cisza przed burzą - ultra niska volatility ({std_dev:.4f}) + wzrost {total_change:.1%} (siła: {details['pattern_strength']})"
        else:
            if not details["low_volatility"]:
                description = f"Brak calm pattern - volatility zbyt wysoka ({std_dev:.4f} > {z_score_threshold})"
            else:
                description = f"Brak calm pattern - cena nie rusza się ({total_change:.1%} ≤ {movement_threshold:.1%})"
        
        return calm_detected, description, details
        
    except Exception as e:
        error_description = f"Błąd w calm detection: {str(e)}"
        return False, error_description, details

def calculate_calm_before_trend_score(detection_result: Tuple[bool, str, Dict]) -> int:
    """
    Oblicza punktację dla Calm Before The Trend Detection
    
    Args:
        detection_result: Wynik z detect_calm_before_trend
        
    Returns:
        Score (0-10 points)
    """
    detected, _, details = detection_result
    
    if not detected:
        return 0
    
    # Bazowa punktacja za wykrycie
    base_score = 10
    
    # Bonus za siłę wzorca
    strength = details.get("pattern_strength", "none")
    if strength == "strong":
        return base_score  # 10 punktów
    elif strength == "moderate":
        return base_score - 1  # 9 punktów
    else:  # weak
        return base_score - 2  # 8 punktów

def create_mock_calm_prices() -> List[float]:
    """
    Tworzy syntetyczne dane dla testów - symuluje niską volatility + początek ruchu
    
    Returns:
        Lista 24 cen symulujących calm before trend pattern
    """
    base_price = 50000.0
    prices = []
    
    # Pierwsze 18 punktów - bardzo niska volatility (oscylacje ±0.05%)
    for i in range(18):
        # Mikro oscylacje wokół base_price
        variation = (i % 4 - 1.5) * base_price * 0.0005  # ±0.05%
        prices.append(base_price + variation)
    
    # Ostatnie 6 punktów - początek ruchu wzrostowego (0.4% total)
    growth_per_step = 0.0007  # 0.07% per step = 0.42% total
    for i in range(6):
        last_price = prices[-1]
        new_price = last_price * (1 + growth_per_step)
        prices.append(new_price)
    
    return prices

def test_calm_before_trend_detector():
    """Test funkcji calm before trend detection"""
    print("=== Test Calm Before The Trend Detector ===")
    
    # Test 1: Perfect calm pattern
    print("\nTest 1: Perfect calm pattern")
    perfect_prices = create_mock_calm_prices()
    result = detect_calm_before_trend(perfect_prices)
    score = calculate_calm_before_trend_score(result)
    print(f"Detected: {result[0]}, Score: {score}, Description: {result[1]}")
    
    # Test 2: High volatility (no calm)
    print("\nTest 2: High volatility")
    volatile_prices = [50000 + i * 200 for i in range(24)]  # Duże zmiany
    result = detect_calm_before_trend(volatile_prices)
    score = calculate_calm_before_trend_score(result)
    print(f"Detected: {result[0]}, Score: {score}, Description: {result[1]}")
    
    # Test 3: Calm but no movement
    print("\nTest 3: Calm but no movement")
    flat_prices = [50000.0] * 24  # Płaska cena
    result = detect_calm_before_trend(flat_prices)
    score = calculate_calm_before_trend_score(result)
    print(f"Detected: {result[0]}, Score: {score}, Description: {result[1]}")
    
    # Test 4: Insufficient data
    print("\nTest 4: Insufficient data")
    short_prices = [50000.0] * 10
    result = detect_calm_before_trend(short_prices)
    score = calculate_calm_before_trend_score(result)
    print(f"Detected: {result[0]}, Score: {score}, Description: {result[1]}")

if __name__ == "__main__":
    test_calm_before_trend_detector()