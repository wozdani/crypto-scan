"""
Trend-Mode: Professional Trader Simulation Module

Symuluje myślenie profesjonalnego tradera, który dołącza do istniejącego trendu wzrostowego
w momencie korekty (pullbacku), a nie na breakoutcie czy odwróceniu.

Etapy decyzyjne:
1. Analiza Kontekstu Rynkowego (market_context)
2. Ocena Siły Trendu (trend_strength) 
3. Wykrycie Korekty (pullback_detection)
4. Reakcja na wsparcie (support_reaction)
5. Logika Decyzyjna Tradera (trader_decision_logic)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def determine_market_context(candles: List[List]) -> str:
    """
    🧩 Etap 1: Analiza Kontekstu Rynkowego
    
    Klasyfikuje aktualną strukturę rynku na podstawie:
    - slope (nachylenie)
    - zmienność 
    - układ świec
    - zmiany trendu na EMA
    
    Args:
        candles: Lista OHLCV candles [[timestamp, open, high, low, close, volume], ...]
        
    Returns:
        str: "impulse", "pullback", "range", "breakout", "redistribution"
    """
    if not candles or len(candles) < 20:
        return "range"
    
    try:
        # Wyciągnij dane cenowe
        closes = [float(c[4]) for c in candles[-20:]]  # Ostatnie 20 świec
        highs = [float(c[2]) for c in candles[-20:]]
        lows = [float(c[3]) for c in candles[-20:]]
        volumes = [float(c[5]) for c in candles[-20:]]
        
        # Oblicz EMA21 dla trendu
        ema21 = _calculate_ema(closes, 21)
        current_price = closes[-1]
        
        # 1. Sprawdź slope (nachylenie ceny)
        price_slope = _calculate_slope(closes[-10:])  # Ostatnie 10 świec
        ema_slope = _calculate_slope(ema21[-10:]) if len(ema21) >= 10 else 0
        
        # 2. Zmienność (ATR-like)
        volatility = _calculate_volatility(highs, lows, closes)
        
        # 3. Momentum (różnica ceny vs EMA)
        price_vs_ema = (current_price - ema21[-1]) / ema21[-1] * 100 if ema21 else 0
        
        # 4. Analiza układu świec (zielone vs czerwone)
        green_candles = sum(1 for i in range(-10, 0) if closes[i] > closes[i-1])
        green_ratio = green_candles / 10
        
        # 5. Wolumen trend (czy rośnie czy maleje)
        volume_slope = _calculate_slope(volumes[-5:])
        
        # === LOGIKA KLASYFIKACJI ===
        
        # IMPULSE: Silny trend wzrostowy z rosnącym wolumenem
        if (price_slope > 0.5 and ema_slope > 0.3 and 
            price_vs_ema > 1.0 and green_ratio >= 0.7 and volume_slope > 0):
            return "impulse"
        
        # PULLBACK: Korekta w trendzie wzrostowym
        if (ema_slope > 0.1 and price_vs_ema > -1.0 and price_vs_ema < 1.0 and
            price_slope < 0 and volatility < 3.0):
            return "pullback"
        
        # BREAKOUT: Wybicie z zakresu z wolumenem
        if (price_slope > 0.8 and volume_slope > 0.5 and volatility > 2.0):
            return "breakout"
        
        # REDISTRIBUTION: Wysokie wolumeny, brak kierunku
        if (volatility > 4.0 and abs(price_slope) < 0.2 and volume_slope > 0):
            return "redistribution"
        
        # RANGE: Brak wyraźnego trendu
        return "range"
        
    except Exception as e:
        print(f"⚠️ Error in determine_market_context: {e}")
        return "range"


def compute_trend_strength(candles: List[List]) -> float:
    """
    📈 Etap 2: Ocena Siły Trendu
    
    Oblicza score siły trendu (0.0 – 1.0) na podstawie:
    - % świec zielonych w ostatnich 20-40
    - nachylenie ceny (slope)
    - stabilność wybicia/momentum
    
    Args:
        candles: Lista OHLCV candles
        
    Returns:
        float: Score trendu 0.0-1.0
    """
    if not candles or len(candles) < 40:
        return 0.0
    
    try:
        closes = [float(c[4]) for c in candles[-40:]]  # Ostatnie 40 świec
        highs = [float(c[2]) for c in candles[-40:]]
        lows = [float(c[3]) for c in candles[-40:]]
        
        # 1. Procent zielonych świec (20 ostatnich)
        green_count_20 = sum(1 for i in range(-20, 0) if closes[i] > closes[i-1])
        green_ratio_20 = green_count_20 / 20
        
        # 2. Procent zielonych świec (40 ostatnich)  
        green_count_40 = sum(1 for i in range(-40, -1) if closes[i] > closes[i-1])
        green_ratio_40 = green_count_40 / 39
        
        # 3. Nachylenie ceny (slope ostatnich 20 świec)
        price_slope = _calculate_slope(closes[-20:])
        normalized_slope = min(max(price_slope / 2.0, 0), 1)  # Normalize 0-1
        
        # 4. Stabilność momentum (czy trend jest konsystentny?)
        ema21 = _calculate_ema(closes, 21)
        price_above_ema = sum(1 for i, price in enumerate(closes[-20:]) 
                             if i < len(ema21) and price > ema21[i-20])
        stability = price_above_ema / 20 if ema21 else 0
        
        # 5. Higher highs pattern (czy robimy wyższe szczyty?)
        recent_highs = highs[-20:]
        higher_highs = 0
        for i in range(5, len(recent_highs)):
            if recent_highs[i] > max(recent_highs[i-5:i]):
                higher_highs += 1
        hh_ratio = higher_highs / 15  # Ostatnie 15 porównań
        
        # === KOMBINACJA WSZYSTKICH CZYNNIKÓW ===
        trend_strength = (
            green_ratio_20 * 0.25 +      # 25% - ostatnie zielone świece
            green_ratio_40 * 0.15 +      # 15% - długoterminowe zielone
            normalized_slope * 0.25 +     # 25% - nachylenie trendu
            stability * 0.20 +            # 20% - stabilność nad EMA
            hh_ratio * 0.15               # 15% - pattern wyższych szczytów
        )
        
        return min(max(trend_strength, 0.0), 1.0)
        
    except Exception as e:
        print(f"⚠️ Error in compute_trend_strength: {e}")
        return 0.0


def detect_pullback(candles: List[List]) -> Dict:
    """
    🔁 Etap 3: Wykrycie Korekty
    
    Sprawdza, czy cena cofnęła się od lokalnego high o 1–2% (pullback)
    Dodatkowo sprawdza, czy wolumen maleje (brak dystrybucji)
    
    Args:
        candles: Lista OHLCV candles
        
    Returns:
        dict: {"detected": bool, "magnitude": float, "volume_declining": bool}
    """
    if not candles or len(candles) < 10:
        return {"detected": False, "magnitude": 0.0, "volume_declining": False}
    
    try:
        closes = [float(c[4]) for c in candles[-10:]]
        highs = [float(c[2]) for c in candles[-10:]]
        volumes = [float(c[5]) for c in candles[-10:]]
        
        current_price = closes[-1]
        
        # 1. Znajdź lokalne maximum w ostatnich 10 świecach
        local_high = max(highs)
        local_high_index = highs.index(local_high)
        
        # 2. Oblicz magnitude pullbacku
        pullback_magnitude = ((local_high - current_price) / local_high) * 100
        
        # 3. Sprawdź czy pullback jest w odpowiednim zakresie (0.5% - 4%)
        pullback_detected = 0.5 <= pullback_magnitude <= 4.0
        
        # 4. Sprawdź trend wolumenu (czy maleje podczas pullbacku?)
        if local_high_index < len(volumes) - 2:  # Musi być przynajmniej 2 świece po high
            volume_during_pullback = volumes[local_high_index+1:]
            volume_before_pullback = volumes[max(0, local_high_index-3):local_high_index+1]
            
            avg_volume_before = np.mean(volume_before_pullback) if volume_before_pullback else 0
            avg_volume_during = np.mean(volume_during_pullback) if volume_during_pullback else 0
            
            volume_declining = avg_volume_during < avg_volume_before * 0.8  # 20% spadek
        else:
            volume_declining = False
        
        # 5. Dodatkowe warunki jakości pullbacku
        # - Nie powinien być zbyt głęboki (max 4%)
        # - Nie powinien być zbyt płytki (min 0.5%)  
        # - Powinien trwać 2-5 świec
        pullback_duration = len(closes) - local_high_index - 1
        duration_ok = 1 <= pullback_duration <= 5
        
        quality_pullback = pullback_detected and duration_ok
        
        return {
            "detected": quality_pullback,
            "magnitude": pullback_magnitude,
            "volume_declining": volume_declining,
            "duration": pullback_duration,
            "local_high": local_high,
            "quality_score": 1.0 if quality_pullback and volume_declining else 0.5 if quality_pullback else 0.0
        }
        
    except Exception as e:
        print(f"⚠️ Error in detect_pullback: {e}")
        return {"detected": False, "magnitude": 0.0, "volume_declining": False}


def is_near_support(candles: List[List]) -> Dict:
    """
    📍 Etap 4: Reakcja na wsparcie
    
    Sprawdza, czy cena utrzymuje się nad dynamicznym wsparciem (EMA21 / VWAP)
    Obserwuje reakcję świec (czy była obrona poziomu?)
    
    Args:
        candles: Lista OHLCV candles
        
    Returns:
        dict: {"near_support": bool, "support_type": str, "reaction_strength": float}
    """
    if not candles or len(candles) < 21:
        return {"near_support": False, "support_type": "none", "reaction_strength": 0.0}
    
    try:
        closes = [float(c[4]) for c in candles[-21:]]
        lows = [float(c[3]) for c in candles[-21:]]
        highs = [float(c[2]) for c in candles[-21:]]
        volumes = [float(c[5]) for c in candles[-21:]]
        
        current_price = closes[-1]
        current_low = lows[-1]
        
        # 1. Oblicz EMA21 jako dynamiczne wsparcie
        ema21 = _calculate_ema(closes, 21)
        ema_support = ema21[-1] if ema21 else closes[-1]
        
        # 2. Oblicz VWAP jako alternatywne wsparcie
        vwap = _calculate_vwap(candles[-21:])
        
        # 3. Sprawdź odległość od wsparć
        distance_to_ema = abs(current_price - ema_support) / ema_support * 100
        distance_to_vwap = abs(current_price - vwap) / vwap * 100
        
        # 4. Określ typ wsparcia i czy jesteśmy blisko
        support_tolerance = 1.5  # 1.5% tolerancji
        
        near_ema = distance_to_ema <= support_tolerance and current_price >= ema_support * 0.99
        near_vwap = distance_to_vwap <= support_tolerance and current_price >= vwap * 0.99
        
        # 5. Sprawdź reakcję na wsparcie (czy były odbicia?)
        reaction_strength = 0.0
        support_type = "none"
        near_support = False
        
        if near_ema:
            # Sprawdź czy były niedawne testy EMA z odbiciami
            ema_tests = 0
            for i in range(-5, 0):  # Ostatnie 5 świec
                if (lows[i] <= ema21[i] * 1.01 and closes[i] > ema21[i]):  # Test i odbicie
                    ema_tests += 1
            
            reaction_strength = min(ema_tests / 3.0, 1.0)  # Max 3 testy = 100%
            support_type = "ema21"
            near_support = True
            
        elif near_vwap:
            # Sprawdź reakcję na VWAP
            vwap_tests = 0
            for i in range(-5, 0):
                if lows[i] <= vwap * 1.01 and closes[i] > vwap:
                    vwap_tests += 1
            
            reaction_strength = min(vwap_tests / 3.0, 1.0)
            support_type = "vwap"
            near_support = True
        
        # 6. Sprawdź siłę ostatniej świecy (czy była obrona?)
        last_candle_strength = 0.0
        if len(closes) >= 2:
            last_range = highs[-1] - lows[-1]
            close_position = (closes[-1] - lows[-1]) / last_range if last_range > 0 else 0
            last_candle_strength = close_position  # 0-1, gdzie 1 = zamknięcie przy high
        
        # 7. Kombinuj wszystkie czynniki
        total_reaction = (reaction_strength * 0.6 + last_candle_strength * 0.4)
        
        return {
            "near_support": near_support,
            "support_type": support_type,
            "reaction_strength": total_reaction,
            "distance_to_support": min(distance_to_ema, distance_to_vwap),
            "ema21_level": ema_support,
            "vwap_level": vwap,
            "last_candle_strength": last_candle_strength
        }
        
    except Exception as e:
        print(f"⚠️ Error in is_near_support: {e}")
        return {"near_support": False, "support_type": "none", "reaction_strength": 0.0}


def interpret_market_as_trader(symbol: str, candles: List[List]) -> Dict:
    """
    🧠 Etap 5: Logika Decyzyjna Tradera
    
    Łączy wszystko w jedno logiczne „myślenie" tradera:
    Jeśli trend_score wysoki + pullback aktywny + near_support = sygnał wejścia
    
    Args:
        symbol: Symbol trading pair (np. 'BTCUSDT')
        candles: Lista OHLCV candles
        
    Returns:
        dict: {
            "decision": "join_trend" / "wait" / "avoid",
            "confidence": 0.0-1.0,
            "reasons": ["reason1", "reason2", ...],
            "market_context": str,
            "trend_strength": float,
            "entry_quality": float
        }
    """
    if not candles or len(candles) < 40:
        return {
            "decision": "avoid",
            "confidence": 0.0,
            "reasons": ["insufficient_data"],
            "market_context": "unknown",
            "trend_strength": 0.0,
            "entry_quality": 0.0
        }
    
    try:
        # === WYKONAJ WSZYSTKIE ANALIZY ===
        
        # 1. Kontekst rynkowy
        market_context = determine_market_context(candles)
        
        # 2. Siła trendu
        trend_strength = compute_trend_strength(candles)
        
        # 3. Detekcja pullbacku
        pullback_data = detect_pullback(candles)
        
        # 4. Analiza wsparcia
        support_data = is_near_support(candles)
        
        # === LOGIKA DECYZYJNA TRADERA ===
        
        reasons = []
        decision = "wait"
        confidence = 0.0
        entry_quality = 0.0
        
        # Warunki podstawowe dla join_trend:
        # - Kontekst: pullback w trendzie wzrostowym
        # - Siła trendu: > 0.6
        # - Pullback: wykryty i jakościowy
        # - Wsparcie: blisko i z reakcją
        
        # 1. Sprawdź kontekst rynkowy
        context_score = 0.0
        if market_context == "pullback":
            context_score = 1.0
            reasons.append("clean_pullback_context")
        elif market_context == "impulse":
            context_score = 0.7  # Można dołączyć, ale mniej idealne
            reasons.append("impulse_momentum")
        elif market_context == "range":
            context_score = 0.0
            reasons.append("ranging_market")
        else:
            context_score = 0.3
            reasons.append(f"context_{market_context}")
        
        # 2. Sprawdź siłę trendu
        trend_score = 0.0
        if trend_strength >= 0.7:
            trend_score = 1.0
            reasons.append("strong_trend")
        elif trend_strength >= 0.5:
            trend_score = 0.7
            reasons.append("moderate_trend")
        elif trend_strength >= 0.3:
            trend_score = 0.3
            reasons.append("weak_trend")
        else:
            trend_score = 0.0
            reasons.append("no_trend")
        
        # 3. Sprawdź pullback
        pullback_score = 0.0
        if pullback_data["detected"]:
            if pullback_data["volume_declining"]:
                pullback_score = 1.0
                reasons.append("quality_pullback_low_volume")
            else:
                pullback_score = 0.6
                reasons.append("pullback_detected")
        else:
            pullback_score = 0.0
            if market_context != "impulse":  # W impulsie nie potrzebujemy pullbacku
                reasons.append("no_pullback")
        
        # 4. Sprawdź wsparcie
        support_score = 0.0
        if support_data["near_support"]:
            if support_data["reaction_strength"] >= 0.5:
                support_score = 1.0
                reasons.append("support_held_strong")
            else:
                support_score = 0.6
                reasons.append("near_support")
        else:
            support_score = 0.0
            reasons.append("no_support")
        
        # === KALKULACJA KOŃCOWA ===
        
        # Różne wagi dla różnych kontekstów
        if market_context == "pullback":
            # W pullbacku wszystkie czynniki ważne
            entry_quality = (
                context_score * 0.20 +
                trend_score * 0.30 +
                pullback_score * 0.25 +
                support_score * 0.25
            )
        elif market_context == "impulse":
            # W impulsie mniej ważny pullback, ważniejszy trend i momentum
            entry_quality = (
                context_score * 0.25 +
                trend_score * 0.50 +
                pullback_score * 0.10 +  # Mniej ważny
                support_score * 0.15
            )
        else:
            # Inne konteksty - ostrożnie
            entry_quality = (
                context_score * 0.40 +
                trend_score * 0.30 +
                pullback_score * 0.15 +
                support_score * 0.15
            )
        
        # === OSTATECZNA DECYZJA ===
        
        if entry_quality >= 0.75:
            decision = "join_trend"
            confidence = entry_quality
            reasons.append("high_probability_setup")
        elif entry_quality >= 0.50:
            decision = "wait"
            confidence = entry_quality
            reasons.append("wait_for_better_setup")
        else:
            decision = "avoid"
            confidence = entry_quality
            reasons.append("low_probability_setup")
        
        # Dodaj szczegółowe informacje dla debugowania
        debug_info = {
            "context_score": context_score,
            "trend_score": trend_score,
            "pullback_score": pullback_score,
            "support_score": support_score,
            "pullback_magnitude": pullback_data.get("magnitude", 0),
            "support_type": support_data.get("support_type", "none"),
            "reaction_strength": support_data.get("reaction_strength", 0)
        }
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasons": reasons,
            "market_context": market_context,
            "trend_strength": trend_strength,
            "entry_quality": entry_quality,
            "debug": debug_info
        }
        
    except Exception as e:
        print(f"⚠️ Error in interpret_market_as_trader for {symbol}: {e}")
        return {
            "decision": "avoid",
            "confidence": 0.0,
            "reasons": ["analysis_error"],
            "market_context": "error",
            "trend_strength": 0.0,
            "entry_quality": 0.0
        }


# === HELPER FUNCTIONS ===

def _calculate_ema(prices: List[float], period: int) -> List[float]:
    """Oblicz Exponential Moving Average"""
    if len(prices) < period:
        return []
    
    multiplier = 2.0 / (period + 1)
    ema = [sum(prices[:period]) / period]  # Pierwszy EMA to SMA
    
    for price in prices[period:]:
        ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
    
    return ema


def _calculate_slope(values: List[float]) -> float:
    """Oblicz nachylenie (slope) listy wartości"""
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    x = list(range(n))
    y = values
    
    # Linear regression slope: (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x_squared = sum(x[i] ** 2 for i in range(n))
    
    denominator = n * sum_x_squared - sum_x ** 2
    if denominator == 0:
        return 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope


def _calculate_volatility(highs: List[float], lows: List[float], closes: List[float]) -> float:
    """Oblicz zmienność (ATR-like)"""
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return 0.0
    
    true_ranges = []
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        true_ranges.append(max(tr1, tr2, tr3))
    
    if not true_ranges:
        return 0.0
    
    atr = sum(true_ranges) / len(true_ranges)
    return (atr / closes[-1]) * 100  # Procent volatility


def _calculate_vwap(candles: List[List]) -> float:
    """Oblicz Volume Weighted Average Price"""
    if not candles:
        return 0.0
    
    total_volume = 0
    total_price_volume = 0
    
    for candle in candles:
        high = float(candle[2])
        low = float(candle[3])
        close = float(candle[4])
        volume = float(candle[5])
        
        typical_price = (high + low + close) / 3
        total_price_volume += typical_price * volume
        total_volume += volume
    
    return total_price_volume / total_volume if total_volume > 0 else 0.0


# === MAIN INTEGRATION FUNCTION ===

def analyze_symbol_trend_mode(symbol: str, candles: List[List]) -> Dict:
    """
    Główna funkcja integracyjna dla Trend-Mode
    
    Args:
        symbol: Symbol trading pair
        candles: Lista OHLCV candles
        
    Returns:
        dict: Kompletna analiza Trend-Mode
    """
    result = interpret_market_as_trader(symbol, candles)
    
    # Dodaj timestamp i symbol do wyniku
    result.update({
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "trend_mode"
    })
    
    return result


if __name__ == "__main__":
    # Test z przykładowymi danymi
    print("🧪 Testing Trend-Mode module...")
    
    # Przykładowe dane (OHLCV format)
    test_candles = [
        [1640995200, 47000, 47500, 46800, 47200, 1000],  # Trend wzrostowy
        [1640995260, 47200, 47800, 47100, 47600, 1100],
        [1640995320, 47600, 48000, 47400, 47800, 1200],
        [1640995380, 47800, 48200, 47600, 48000, 1300],
        [1640995440, 48000, 48100, 47700, 47900, 900],   # Pullback start
        [1640995500, 47900, 48000, 47600, 47750, 800],   # Pullback continues
        [1640995560, 47750, 47950, 47650, 47850, 700],   # Recovery?
    ] * 10  # Powtórz dla większej próby
    
    analysis = analyze_symbol_trend_mode("BTCUSDT", test_candles)
    print(f"Decision: {analysis['decision']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Market Context: {analysis['market_context']}")
    print(f"Reasons: {', '.join(analysis['reasons'])}")