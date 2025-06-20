"""
VWAP Pinning Detector - Detektor Przyklejenia do VWAP
Wykrywa sytuacje gdy cena przez dłuższy czas utrzymuje się blisko VWAP
Identyfikuje manipulacyjny charakter konsolidacji kontrolowanej przez whales
"""

import requests
import json
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional

def get_kline_data_with_volume_bybit(symbol: str, interval: str = "5", limit: int = 50) -> List[Dict]:
    """
    Pobiera dane kline z wolumenem z Bybit API dla obliczenia VWAP
    
    Args:
        symbol: Symbol trading pair (np. 'BTCUSDT')
        interval: Interwał czasowy (default 5 minut)
        limit: Liczba świec (default 50 dla ~4h danych)
        
    Returns:
        List[Dict]: Lista świec z danymi OHLCV
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data["retCode"] != 0:
            print(f"❌ Bybit API error for {symbol}: {data['retMsg']}")
            return []

        klines = []
        for item in data["result"]["list"]:
            klines.append({
                "timestamp": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5])
            })
        
        # Reverse to get chronological order (Bybit returns newest first)
        return list(reversed(klines))
        
    except Exception as e:
        print(f"❌ Exception fetching kline data for {symbol}: {e}")
        return []


def calculate_vwap_values(klines: List[Dict]) -> List[float]:
    """
    Oblicza wartości VWAP (Volume Weighted Average Price) dla każdej świecy
    
    Args:
        klines: Lista świec z danymi OHLCV
        
    Returns:
        List[float]: Lista wartości VWAP
    """
    if not klines:
        return []
    
    vwap_values = []
    cumulative_volume = 0.0
    cumulative_pv = 0.0  # price * volume
    
    for candle in klines:
        # Typical price (HLC/3) * volume
        typical_price = (candle["high"] + candle["low"] + candle["close"]) / 3
        volume = candle["volume"]
        
        if volume > 0:
            cumulative_pv += typical_price * volume
            cumulative_volume += volume
            
            # VWAP = cumulative (price * volume) / cumulative volume
            vwap = cumulative_pv / cumulative_volume if cumulative_volume > 0 else typical_price
        else:
            # Jeśli brak wolumenu, użyj poprzedniej wartości VWAP lub typical price
            vwap = vwap_values[-1] if vwap_values else typical_price
        
        vwap_values.append(vwap)
    
    return vwap_values


def detect_vwap_pinning(prices: List[float], vwap_values: List[float], min_periods: int = 20) -> Tuple[bool, str, Dict]:
    """
    Wykrywa, czy cena przez dłuższy czas jest 'przyklejona' do VWAP.
    
    Args:
        prices: Lista cen zamknięcia
        vwap_values: Lista wartości VWAP
        min_periods: Minimalna liczba okresów do analizy (default 20 = ~100min)
        
    Returns:
        Tuple: (bool, str, dict) - (wykryto_pinning, opis, szczegóły)
    """
    if len(prices) < min_periods or len(vwap_values) != len(prices):
        return False, f"Za mało danych dla analizy VWAP pinning (wymagane {min_periods}, otrzymano {len(prices)})", {}

    # Filtruj zero/invalid values
    valid_pairs = [(p, v) for p, v in zip(prices, vwap_values) if p > 0 and v > 0]
    
    if len(valid_pairs) < min_periods:
        return False, "Za mało prawidłowych par cena-VWAP", {}

    # Oblicz odchylenia ceny od VWAP
    deviations = []
    absolute_deviations = []
    
    for price, vwap in valid_pairs:
        deviation_pct = ((price - vwap) / vwap) * 100  # odchylenie w %
        abs_deviation_pct = abs(deviation_pct)
        
        deviations.append(deviation_pct)
        absolute_deviations.append(abs_deviation_pct)
    
    # Oblicz metryki pinning
    avg_absolute_deviation = sum(absolute_deviations) / len(absolute_deviations)
    max_absolute_deviation = max(absolute_deviations)
    
    # Oblicz stabilność (ile % świec ma odchylenie <0.3%)
    stable_periods = sum(1 for dev in absolute_deviations if dev < 0.3)
    stability_ratio = stable_periods / len(absolute_deviations)
    
    # Oblicz kierunek bias (czy cena jest częściej powyżej czy poniżej VWAP)
    positive_deviations = sum(1 for dev in deviations if dev > 0)
    negative_deviations = sum(1 for dev in deviations if dev < 0)
    bias_ratio = positive_deviations / len(deviations) if deviations else 0.5
    
    # Warunki detekcji VWAP pinning:
    # 1. Średnie odchylenie <0.2% (bardzo blisko VWAP)
    # 2. Maksymalne odchylenie <0.5% (brak dużych wybić)
    # 3. Stabilność >70% (większość świec blisko VWAP)
    pinning_threshold = 0.2  # 0.2%
    max_deviation_threshold = 0.5  # 0.5%
    stability_threshold = 0.7  # 70%
    
    pinning_detected = (
        avg_absolute_deviation < pinning_threshold and
        max_absolute_deviation < max_deviation_threshold and
        stability_ratio >= stability_threshold
    )
    
    # Determine pinning strength
    if pinning_detected:
        if avg_absolute_deviation < 0.1 and stability_ratio >= 0.85:
            strength = "bardzo silne"
            strength_score = 15
        elif avg_absolute_deviation < 0.15 and stability_ratio >= 0.8:
            strength = "silne"
            strength_score = 15
        elif avg_absolute_deviation < 0.2 and stability_ratio >= 0.75:
            strength = "umiarkowane"
            strength_score = 12
        else:
            strength = "słabe"
            strength_score = 10
    else:
        strength = "brak"
        strength_score = 0
    
    # Określ kierunek bias
    if bias_ratio > 0.6:
        bias_direction = "bullish (cena powyżej VWAP)"
    elif bias_ratio < 0.4:
        bias_direction = "bearish (cena poniżej VWAP)"
    else:
        bias_direction = "neutralny (równowaga)"
    
    details = {
        "periods_analyzed": len(valid_pairs),
        "avg_absolute_deviation_pct": round(avg_absolute_deviation, 4),
        "max_absolute_deviation_pct": round(max_absolute_deviation, 4),
        "stability_ratio": round(stability_ratio, 3),
        "stable_periods": stable_periods,
        "bias_ratio": round(bias_ratio, 3),
        "bias_direction": bias_direction,
        "pinning_threshold_pct": pinning_threshold,
        "max_deviation_threshold_pct": max_deviation_threshold,
        "stability_threshold": stability_threshold,
        "strength": strength,
        "strength_score": strength_score,
        "price_range": {
            "min": round(min(prices), 6),
            "max": round(max(prices), 6)
        },
        "vwap_range": {
            "min": round(min(vwap_values), 6),
            "max": round(max(vwap_values), 6)
        }
    }
    
    if pinning_detected:
        description = f"{strength.capitalize()} VWAP pinning - cena przyklejona przez {len(valid_pairs)} okresów ({bias_direction})"
    else:
        if avg_absolute_deviation >= pinning_threshold:
            description = f"Brak VWAP pinning - za duże odchylenie ({avg_absolute_deviation:.3f}% > {pinning_threshold}%)"
        elif stability_ratio < stability_threshold:
            description = f"Brak VWAP pinning - za niska stabilność ({stability_ratio:.1%} < {stability_threshold:.1%})"
        else:
            description = f"Brak VWAP pinning - za duże maksymalne odchylenie ({max_absolute_deviation:.3f}%)"
    
    return pinning_detected, description, details


def calculate_vwap_pinning_score(detection_result: Tuple[bool, str, Dict]) -> int:
    """
    Oblicza punktację dla VWAP Pinning Detection
    
    Args:
        detection_result: wynik z detect_vwap_pinning()
        
    Returns:
        int: punkty do dodania (0-15)
    """
    detected, description, details = detection_result
    
    if not detected:
        return 0
    
    return details.get("strength_score", 15)  # 10-15 punktów w zależności od siły


def analyze_vwap_pinning_detailed(symbol: str, periods: int = 50) -> Dict:
    """
    Szczegółowa analiza VWAP pinning z pobraniem danych z API
    
    Args:
        symbol: Symbol trading pair
        periods: Liczba okresów do analizy
        
    Returns:
        Dict: szczegółowe dane o VWAP pinning
    """
    # Pobierz dane kline
    klines = get_kline_data_with_volume_bybit(symbol, interval="5", limit=periods)
    
    if not klines:
        return {"error": f"Nie udało się pobrać danych dla {symbol}"}
    
    # Wyciągnij ceny zamknięcia
    prices = [candle["close"] for candle in klines]
    
    # Oblicz VWAP
    vwap_values = calculate_vwap_values(klines)
    
    if not vwap_values or len(vwap_values) != len(prices):
        return {"error": "Błąd obliczania VWAP"}
    
    # Przeprowadź analizę pinning
    detected, description, basic_details = detect_vwap_pinning(prices, vwap_values)
    
    # Dodatkowe analizy
    latest_price = prices[-1]
    latest_vwap = vwap_values[-1]
    current_deviation = ((latest_price - latest_vwap) / latest_vwap) * 100
    
    # Analiza trendu VWAP
    vwap_trend = "rosnący" if vwap_values[-1] > vwap_values[0] else "malejący"
    vwap_change_pct = ((vwap_values[-1] - vwap_values[0]) / vwap_values[0]) * 100
    
    return {
        **basic_details,
        "symbol": symbol,
        "description": description,
        "detected": detected,
        "current_price": latest_price,
        "current_vwap": latest_vwap,
        "current_deviation_pct": round(current_deviation, 4),
        "vwap_trend": vwap_trend,
        "vwap_change_pct": round(vwap_change_pct, 3),
        "data_quality": {
            "klines_fetched": len(klines),
            "prices_valid": len(prices),
            "vwap_calculated": len(vwap_values)
        }
    }


def create_mock_vwap_pinning_data() -> Tuple[List[float], List[float]]:
    """
    Tworzy przykładowe dane dla testów VWAP pinning
    """
    # Symuluj cenę "przyklejoną" do VWAP przez 30 okresów
    base_vwap = 50000.0
    prices = []
    vwap_values = []
    
    for i in range(30):
        # VWAP lekko rośnie
        vwap = base_vwap + (i * 2)  # +2 USD na okres
        
        # Cena oscyluje bardzo blisko VWAP (±0.15%)
        price_deviation = (i % 3 - 1) * 0.0015  # -0.15%, 0%, +0.15%
        price = vwap * (1 + price_deviation)
        
        vwap_values.append(vwap)
        prices.append(price)
    
    return prices, vwap_values


def create_mock_volatile_data() -> Tuple[List[float], List[float]]:
    """
    Tworzy przykładowe dane bez VWAP pinning (wysoka zmienność)
    """
    base_vwap = 50000.0
    prices = []
    vwap_values = []
    
    for i in range(30):
        vwap = base_vwap + (i * 1)
        
        # Cena odchyla się znacznie od VWAP (±1.0%)
        price_deviation = ((i % 5) - 2) * 0.005  # -1%, -0.5%, 0%, +0.5%, +1%
        price = vwap * (1 + price_deviation)
        
        vwap_values.append(vwap)
        prices.append(price)
    
    return prices, vwap_values


def test_vwap_pinning_with_mock_data():
    """
    Test funkcji z przykładowymi danymi
    """
    print("🧪 Testing VWAP Pinning Detector with mock data\n")
    
    # Test 1: Perfect pinning scenario
    pinning_prices, pinning_vwap = create_mock_vwap_pinning_data()
    detected1, desc1, details1 = detect_vwap_pinning(pinning_prices, pinning_vwap)
    score1 = calculate_vwap_pinning_score((detected1, desc1, details1))
    
    print(f"📌 Perfect Pinning Test:")
    print(f"   Detected: {detected1}")
    print(f"   Description: {desc1}")
    print(f"   Avg deviation: {details1['avg_absolute_deviation_pct']:.4f}%")
    print(f"   Stability ratio: {details1['stability_ratio']:.3f}")
    print(f"   Bias direction: {details1['bias_direction']}")
    print(f"   Score: {score1}/15")
    
    # Test 2: Volatile scenario (no pinning)
    volatile_prices, volatile_vwap = create_mock_volatile_data()
    detected2, desc2, details2 = detect_vwap_pinning(volatile_prices, volatile_vwap)
    score2 = calculate_vwap_pinning_score((detected2, desc2, details2))
    
    print(f"\n📈 Volatile Market Test:")
    print(f"   Detected: {detected2}")
    print(f"   Description: {desc2}")
    print(f"   Avg deviation: {details2['avg_absolute_deviation_pct']:.4f}%")
    print(f"   Stability ratio: {details2['stability_ratio']:.3f}")
    print(f"   Bias direction: {details2['bias_direction']}")
    print(f"   Score: {score2}/15")
    
    print("\n✅ VWAP Pinning Detector tests completed!")


if __name__ == "__main__":
    test_vwap_pinning_with_mock_data()