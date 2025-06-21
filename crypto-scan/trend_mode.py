"""
Trend-Mode: Advanced Professional Trader Simulation Module

Rozbudowany moduł symulujący analizę profesjonalnego tradera, który dołącza do silnego trendu
w czasie korekty (pullbacku), a nie wchodzi przypadkowo.

9 Etapów Analizy:
1. Analiza Kontekstu Rynkowego (market_context)
2. Ocena Siły Trendu (trend_strength) 
3. Detekcja Korekty (pullback_detection)
4. Reakcja na wsparcie (support_reaction)
5. Czas + dynamika (market_time_score)
6. Potwierdzenie bounce'a (detect_bounce_confirmation)
7. Scoring heurystyczny (compute_trend_score)
8. Trader logic (interpret_market_as_trader)
9. GPT-Feedback jako trader-asystent (opcjonalny)
"""

import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone
import json


def determine_market_context(candles: List[List], symbol: str = None) -> str:
    """
    🧩 Etap 1: Analiza Kontekstu Rynkowego
    
    Klasyfikuje aktualną strukturę rynku na podstawie:
    - slope (nachylenie)
    - zmienność 
    - układ świec
    - zmiany trendu na EMA
    
    Args:
        candles: Lista OHLCV candles [[timestamp, open, high, low, close, volume], ...]
        symbol: Symbol for debug logging
        
    Returns:
        str: "impulse", "pullback", "range", "breakout", "redistribution"
    """
    if not candles or len(candles) < 20:
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Market context = range (insufficient data: {len(candles)} candles)")
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
        
        context = "range"  # Default
        
        # IMPULSE: Silny trend wzrostowy z rosnącym wolumenem
        if (price_slope > 0.5 and ema_slope > 0.3 and 
            price_vs_ema > 1.0 and green_ratio >= 0.7 and volume_slope > 0):
            context = "impulse"
        
        # PULLBACK: Korekta w trendzie wzrostowym
        elif (ema_slope > 0.1 and price_vs_ema > -1.0 and price_vs_ema < 1.0 and
            price_slope < 0 and volatility < 3.0):
            context = "pullback"
        
        # BREAKOUT: Wybicie z zakresu z wolumenem
        elif (price_slope > 0.8 and volume_slope > 0.5 and volatility > 2.0):
            context = "breakout"
        
        # REDISTRIBUTION: Wysokie wolumeny, brak kierunku
        elif (volatility > 4.0 and abs(price_slope) < 0.2 and volume_slope > 0):
            context = "redistribution"
        
        # Debug logging
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Market context = {context} "
                  f"(price_slope={price_slope:.3f}, ema_slope={ema_slope:.3f}, "
                  f"volatility={volatility:.3f}, price_vs_ema={price_vs_ema:.2f}%, "
                  f"green_ratio={green_ratio:.2f})")
        
        return context
        
    except Exception as e:
        error_msg = f"determine_market_context failed: {str(e)}"
        if symbol:
            print(f"[TREND ERROR] {symbol} - {error_msg}")
            # Log to file
            try:
                with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "function": "determine_market_context",
                        "error": str(e),
                        "candles_count": len(candles) if candles else 0
                    }) + "\n")
            except:
                pass
        else:
            print(f"[TREND ERROR] {error_msg}")
        return "range"


def compute_trend_strength(candles: List[List], symbol: str = None) -> float:
    """
    📈 Etap 2: Ocena Siły Trendu
    
    Oblicza score siły trendu (0.0 – 1.0) na podstawie:
    - % świec zielonych w ostatnich 20-40
    - nachylenie ceny (slope)
    - stabilność wybicia/momentum
    
    Args:
        candles: Lista OHLCV candles
        symbol: Symbol for debug logging
        
    Returns:
        float: Score trendu 0.0-1.0
    """
    if not candles or len(candles) < 40:
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Trend strength = 0.0 (insufficient data: {len(candles)} candles)")
        return 0.0
    
    try:
        closes = [float(c[4]) for c in candles[-40:]]  # Ostatnie 40 świec
        highs = [float(c[2]) for c in candles[-40:]]
        lows = [float(c[3]) for c in candles[-40:]]
        
        # 1. Procent zielonych świec (20 ostatnich)
        green_count_20 = 0
        for i in range(len(closes)-20, len(closes)):
            if i > 0 and i < len(closes) and closes[i] > closes[i-1]:
                green_count_20 += 1
        green_ratio_20 = green_count_20 / 20
        
        # 2. Procent zielonych świec (40 ostatnich)  
        green_count_40 = 0
        for i in range(len(closes)-40, len(closes)):
            if i > 0 and i < len(closes) and closes[i] > closes[i-1]:
                green_count_40 += 1
        green_ratio_40 = green_count_40 / 39
        
        # 3. Nachylenie ceny (slope ostatnich 20 świec)
        price_slope = _calculate_slope(closes[-20:])
        normalized_slope = min(max(price_slope / 2.0, 0), 1)  # Normalize 0-1
        
        # 4. Stabilność momentum (czy trend jest konsystentny?)
        ema21 = _calculate_ema(closes, 21)
        stability = 0.0
        if ema21 and len(ema21) >= 20:
            price_above_ema = 0
            recent_closes = closes[-20:]
            recent_ema = ema21[-20:] if len(ema21) >= 20 else ema21
            for i in range(min(len(recent_closes), len(recent_ema))):
                if recent_closes[i] > recent_ema[i]:
                    price_above_ema += 1
            stability = price_above_ema / 20
        
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
        
        final_strength = min(max(trend_strength, 0.0), 1.0)
        
        # Debug logging
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Trend strength = {final_strength:.3f} "
                  f"(green_20={green_ratio_20:.2f}, green_40={green_ratio_40:.2f}, "
                  f"slope={price_slope:.3f}, stability={stability:.2f}, higher_highs={hh_ratio:.2f})")
        
        return final_strength
        
    except Exception as e:
        error_msg = f"compute_trend_strength failed: {str(e)}"
        if symbol:
            print(f"[TREND ERROR] {symbol} - {error_msg}")
            # Log to file
            try:
                with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "function": "compute_trend_strength",
                        "error": str(e),
                        "candles_count": len(candles) if candles else 0
                    }) + "\n")
            except:
                pass  # Don't fail on logging failure
        else:
            print(f"[TREND ERROR] {error_msg}")
        return 0.0


def detect_pullback(candles: List[List], symbol: str = None) -> Dict:
    """
    🔁 Etap 3: Wykrycie Korekty
    
    Sprawdza, czy cena cofnęła się od lokalnego high o 1–2% (pullback)
    Dodatkowo sprawdza, czy wolumen maleje (brak dystrybucji)
    
    Args:
        candles: Lista OHLCV candles
        symbol: Symbol for debug logging
        
    Returns:
        dict: {"detected": bool, "magnitude": float, "volume_declining": bool}
    """
    if not candles or len(candles) < 10:
        result = {"detected": False, "magnitude": 0.0, "volume_declining": False}
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Pullback = False (insufficient data: {len(candles)} candles)")
        return result
    
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
        volume_declining = False
        if local_high_index < len(volumes) - 2:  # Musi być przynajmniej 2 świece po high
            volume_during_pullback = volumes[local_high_index+1:]
            volume_before_pullback = volumes[max(0, local_high_index-3):local_high_index+1]
            
            avg_volume_before = np.mean(volume_before_pullback) if volume_before_pullback else 0
            avg_volume_during = np.mean(volume_during_pullback) if volume_during_pullback else 0
            
            volume_declining = avg_volume_during < avg_volume_before * 0.8  # 20% spadek
        
        # 5. Dodatkowe warunki jakości pullbacku
        # - Nie powinien być zbyt głęboki (max 4%)
        # - Nie powinien być zbyt płytki (min 0.5%)  
        # - Powinien trwać 2-5 świec
        pullback_duration = len(closes) - local_high_index - 1
        duration_ok = 1 <= pullback_duration <= 5
        
        quality_pullback = pullback_detected and duration_ok
        
        # Debug logging
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Pullback = {quality_pullback} "
                  f"(magnitude={pullback_magnitude:.2f}%, volume_declining={volume_declining}, "
                  f"duration={pullback_duration}, high_index={local_high_index})")
        
        return {
            "detected": quality_pullback,
            "magnitude": pullback_magnitude,
            "volume_declining": volume_declining,
            "duration": pullback_duration,
            "local_high": local_high,
            "quality_score": 1.0 if quality_pullback and volume_declining else 0.5 if quality_pullback else 0.0
        }
        
    except Exception as e:
        error_msg = f"detect_pullback failed: {str(e)}"
        if symbol:
            print(f"[TREND ERROR] {symbol} - {error_msg}")
            # Log to file
            try:
                with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "function": "detect_pullback",
                        "error": str(e),
                        "candles_count": len(candles) if candles else 0
                    }) + "\n")
            except:
                pass
        else:
            print(f"[TREND ERROR] {error_msg}")
        return {"detected": False, "magnitude": 0.0, "volume_declining": False, "error": True}


def detect_support_reaction(candles: List[List], symbol: str = None) -> Dict:
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
        ema_support = ema21[-1] if ema21 and len(ema21) > 0 else closes[-1]
        
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
        support_detected = False
        
        if near_ema:
            # Sprawdź czy były niedawne testy EMA z odbiciami
            ema_tests = 0
            if len(ema21) >= len(lows):
                for i in range(max(0, len(lows) - 5), len(lows)):  # Ostatnie 5 świec
                    if i < len(lows) and i < len(ema21) and i < len(closes):
                        if (lows[i] <= ema21[i] * 1.01 and closes[i] > ema21[i]):  # Test i odbicie
                            ema_tests += 1
            
            reaction_strength = min(ema_tests / 3.0, 1.0)  # Max 3 testy = 100%
            support_type = "ema21"
            near_support = True
            
        elif near_vwap:
            # Sprawdź reakcję na VWAP
            vwap_tests = 0
            for i in range(max(0, len(lows) - 5), len(lows)):
                if i < len(lows) and i < len(closes):
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
        
        # Check for additional patterns
        engulfing_pattern = _detect_engulfing_pattern(candles[-3:]) if len(candles) >= 3 else False
        wick_bounce = _detect_wick_bounce(candles[-2:]) if len(candles) >= 2 else False
        
        # Debug logging
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Support reaction = {support_detected} "
                  f"(type={support_type}, strength={total_reaction:.3f}, "
                  f"ema_distance={distance_to_ema:.2f}%, vwap_distance={distance_to_vwap:.2f}%, "
                  f"engulfing={engulfing_pattern}, wick_bounce={wick_bounce})")
        
        return {
            "support_detected": support_detected,
            "support_type": support_type,
            "reaction_strength": total_reaction,
            "distance_to_support": min(distance_to_ema, distance_to_vwap),
            "ema21_level": ema_support,
            "vwap_level": vwap,
            "last_candle_strength": last_candle_strength,
            "engulfing_pattern": engulfing_pattern,
            "wick_bounce": wick_bounce
        }
        
    except Exception as e:
        error_msg = f"detect_support_reaction failed: {str(e)}"
        if symbol:
            print(f"[TREND ERROR] {symbol} - {error_msg}")
            # Log to file
            try:
                with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "function": "detect_support_reaction",
                        "error": str(e),
                        "candles_count": len(candles) if candles else 0
                    }) + "\n")
            except:
                pass
        else:
            print(f"[TREND ERROR] {error_msg}")
        return {"support_detected": False, "support_type": "none", "reaction_strength": 0.0, "error": True}


def market_time_score(utc_hour: int, symbol: str = None) -> Dict:
    """
    ⏱️ Etap 5: Czas + dynamika
    
    Zwraca boost jeśli ruch odbywa się w czasie większej płynności
    
    Args:
        utc_hour: Godzina UTC (0-23)
        
    Returns:
        dict: {"time_boost": float, "session": str, "liquidity": str}
    """
    try:
        # Definicje sesji tradingowych (UTC)
        # Asian: 00:00-08:00 UTC (Tokyo 09:00-17:00 JST)
        # London: 08:00-16:00 UTC  
        # NY: 13:00-21:00 UTC
        # Overlap London/NY: 13:00-16:00 UTC (najlepsza płynność)
        
        result = {}
        
        if 13 <= utc_hour <= 16:  # London/NY overlap
            result = {
                "time_boost": 1.2,
                "session": "london_ny_overlap", 
                "liquidity": "highest"
            }
        elif 8 <= utc_hour <= 12:  # London morning
            result = {
                "time_boost": 1.1,
                "session": "london_morning",
                "liquidity": "high"
            }
        elif 17 <= utc_hour <= 21:  # NY afternoon
            result = {
                "time_boost": 1.1,
                "session": "ny_afternoon", 
                "liquidity": "high"
            }
        elif 1 <= utc_hour <= 7:  # Asian session
            result = {
                "time_boost": 0.9,
                "session": "asian",
                "liquidity": "medium"
            }
        else:  # Night/weekend (22:00-00:00)
            result = {
                "time_boost": 0.7,
                "session": "night",
                "liquidity": "low"
            }
        
        # Debug logging
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Time scoring at UTC {utc_hour} = {result['time_boost']:.2f} "
                  f"(session={result['session']}, liquidity={result['liquidity']})")
        
        return result
            
    except Exception as e:
        error_msg = f"market_time_score failed: {str(e)}"
        if symbol:
            print(f"[TREND ERROR] {symbol} - {error_msg}")
            # Log to file
            try:
                with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "function": "market_time_score",
                        "error": str(e),
                        "utc_hour": utc_hour
                    }) + "\n")
            except:
                pass
        else:
            print(f"[TREND ERROR] {error_msg}")
        return {"time_boost": 1.0, "session": "unknown", "liquidity": "medium", "error": True}


def detect_bounce_confirmation(candles: List[List], symbol: str = None) -> Dict:
    """
    🎥 Etap 6: Potwierdzenie bounce'a
    
    Obserwuje zakończenie pullbacku, mniejsze świece, odbicie od wsparcia
    
    Args:
        candles: Lista OHLCV candles
        
    Returns:
        dict: {"bounce_confirmed": bool, "bounce_strength": float, "pattern": str}
    """
    if not candles or len(candles) < 5:
        result = {"bounce_confirmed": False, "bounce_strength": 0.0, "pattern": "insufficient_data"}
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Bounce confirmation = False (insufficient data: {len(candles)} candles)")
        return result
    
    try:
        # Ostatnie 5 świec do analizy bounce'a
        recent_candles = candles[-5:]
        opens = [float(c[1]) for c in recent_candles]
        highs = [float(c[2]) for c in recent_candles]
        lows = [float(c[3]) for c in recent_candles]
        closes = [float(c[4]) for c in recent_candles]
        volumes = [float(c[5]) for c in recent_candles]
        
        # 1. Sprawdź czy ostatnie świece są mniejsze (konsolidacja po pullbacku)
        candle_sizes = [(highs[i] - lows[i]) for i in range(len(recent_candles))]
        avg_size_before = np.mean(candle_sizes[:-2]) if len(candle_sizes) > 2 else 0
        avg_size_recent = np.mean(candle_sizes[-2:]) if len(candle_sizes) >= 2 else 0
        
        smaller_candles = avg_size_recent < avg_size_before * 0.8 if avg_size_before > 0 else False
        
        # 2. Sprawdź czy była próba odbicia (higher lows pattern)
        higher_lows = 0
        for i in range(1, len(lows)):
            if lows[i] > lows[i-1]:
                higher_lows += 1
        
        higher_lows_pattern = higher_lows >= 2
        
        # 3. Sprawdź siłę ostatniej świecy (czy jest bullish?)
        last_candle_bullish = closes[-1] > opens[-1]
        last_candle_strength = (closes[-1] - lows[-1]) / (highs[-1] - lows[-1]) if highs[-1] > lows[-1] else 0
        
        # 4. Sprawdź wzrost wolumenu na bounce'ie
        volume_increase = False
        if len(volumes) >= 3:
            recent_volume = volumes[-1]
            avg_volume_before = np.mean(volumes[:-1])
            volume_increase = recent_volume > avg_volume_before * 1.2
        
        # 5. Detekcja specific patterns
        pattern = "none"
        if smaller_candles and higher_lows_pattern:
            pattern = "consolidation_bounce"
        elif last_candle_bullish and last_candle_strength > 0.7:
            pattern = "strong_reversal_candle"
        elif volume_increase and last_candle_bullish:
            pattern = "volume_bounce"
        elif higher_lows_pattern:
            pattern = "higher_lows"
        
        # 6. Kalkulacja bounce strength
        bounce_factors = [
            smaller_candles,
            higher_lows_pattern, 
            last_candle_bullish,
            volume_increase,
            last_candle_strength > 0.6
        ]
        
        bounce_strength = sum(bounce_factors) / len(bounce_factors)
        bounce_confirmed = bounce_strength >= 0.6 and pattern != "none"
        
        # Debug logging
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Bounce confirmation = {bounce_confirmed} "
                  f"(strength={bounce_strength:.3f}, pattern={pattern}, "
                  f"smaller_candles={smaller_candles}, higher_lows={higher_lows_pattern}, "
                  f"volume_increase={volume_increase}, last_candle_bullish={last_candle_bullish})")
        
        return {
            "bounce_confirmed": bounce_confirmed,
            "bounce_strength": bounce_strength,
            "pattern": pattern,
            "smaller_candles": smaller_candles,
            "higher_lows": higher_lows_pattern,
            "volume_increase": volume_increase,
            "last_candle_strength": last_candle_strength
        }
        
    except Exception as e:
        error_msg = f"detect_bounce_confirmation failed: {str(e)}"
        if symbol:
            print(f"[TREND ERROR] {symbol} - {error_msg}")
            # Log to file
            try:
                with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "function": "detect_bounce_confirmation",
                        "error": str(e),
                        "candles_count": len(candles) if candles else 0
                    }) + "\n")
            except:
                pass
        else:
            print(f"[TREND ERROR] {error_msg}")
        return {"bounce_confirmed": False, "bounce_strength": 0.0, "pattern": "error", "error": True}


def compute_trend_score(
    trend_strength: float,
    pullback_data: Dict,
    support_data: Dict, 
    bounce_data: Dict,
    time_data: Dict,
    symbol: str = None
) -> Dict:
    """
    📊 Etap 7: Scoring heurystyczny
    
    Łączy wszystkie czynniki w jeden score z wagami
    
    Args:
        trend_strength: Score siły trendu (0.0-1.0)
        pullback_data: Dane z detect_pullback()
        support_data: Dane z detect_support_reaction()
        bounce_data: Dane z detect_bounce_confirmation()
        time_data: Dane z market_time_score()
        
    Returns:
        dict: {"final_score": float, "weighted_scores": dict, "quality_grade": str}
    """
    try:
        # Wagi dla różnych czynników
        weights = {
            "trend_strength": 0.30,
            "pullback_quality": 0.20,
            "support_reaction": 0.20,
            "bounce_confirmation": 0.15,
            "time_boost": 0.15
        }
        
        # 1. Trend Strength Score (już jest 0.0-1.0)
        trend_score = trend_strength
        
        # 2. Pullback Quality Score
        pullback_score = 0.0
        if pullback_data.get("detected", False):
            magnitude = pullback_data.get("magnitude", 0)
            volume_declining = pullback_data.get("volume_declining", False)
            quality_score = pullback_data.get("quality_score", 0)
            
            # Idealne pullbacki: 1-3%, z malejącym wolumenem
            if 1.0 <= magnitude <= 3.0:
                pullback_score = 0.8
                if volume_declining:
                    pullback_score = 1.0
            elif 0.5 <= magnitude <= 4.0:
                pullback_score = 0.6
                if volume_declining:
                    pullback_score = 0.8
            else:
                pullback_score = quality_score
        
        # 3. Support Reaction Score
        support_score = 0.0
        if support_data.get("support_detected", False):
            reaction_strength = support_data.get("reaction_strength", 0)
            engulfing = support_data.get("engulfing_pattern", False)
            wick_bounce = support_data.get("wick_bounce", False)
            
            support_score = reaction_strength
            if engulfing:
                support_score += 0.2
            if wick_bounce:
                support_score += 0.1
            
            support_score = min(support_score, 1.0)
        
        # 4. Bounce Confirmation Score
        bounce_score = bounce_data.get("bounce_strength", 0.0)
        if bounce_data.get("bounce_confirmed", False):
            bounce_score += 0.2  # Bonus za potwierdzenie
            bounce_score = min(bounce_score, 1.0)
        
        # 5. Time Boost (może być > 1.0)
        time_boost = time_data.get("time_boost", 1.0)
        time_score = min((time_boost - 0.7) / 0.5, 1.0)  # Normalize 0.7-1.2 to 0.0-1.0
        
        # Kalkulacja weighted scores
        weighted_scores = {
            "trend_component": trend_score * weights["trend_strength"],
            "pullback_component": pullback_score * weights["pullback_quality"],
            "support_component": support_score * weights["support_reaction"],
            "bounce_component": bounce_score * weights["bounce_confirmation"],
            "time_component": time_score * weights["time_boost"]
        }
        
        # Final score (przed time boost)
        base_score = sum(weighted_scores.values())
        
        # Zastosuj time boost jako multiplier
        final_score = base_score * time_boost
        final_score = min(final_score, 1.0)  # Cap at 1.0
        
        # Quality grading
        if final_score >= 0.80:
            quality_grade = "excellent"
        elif final_score >= 0.65:
            quality_grade = "good"
        elif final_score >= 0.50:
            quality_grade = "average"
        elif final_score >= 0.35:
            quality_grade = "poor"
        else:
            quality_grade = "very_poor"
        
        # Debug logging
        if symbol:
            print(f"[TREND DEBUG] {symbol}: Final trend score = {final_score:.3f} "
                  f"(trend={trend_score:.2f}, pullback={pullback_score:.2f}, "
                  f"support={support_score:.2f}, bounce={bounce_score:.2f}, "
                  f"time={time_score:.2f}, grade={quality_grade})")
        
        return {
            "final_score": final_score,
            "base_score": base_score,
            "weighted_scores": weighted_scores,
            "quality_grade": quality_grade,
            "time_boost_applied": time_boost,
            "individual_scores": {
                "trend": trend_score,
                "pullback": pullback_score,
                "support": support_score,
                "bounce": bounce_score,
                "time": time_score
            }
        }
        
    except Exception as e:
        error_msg = f"compute_trend_score failed: {str(e)}"
        if symbol:
            print(f"[TREND ERROR] {symbol} - {error_msg}")
            # Log to file
            try:
                with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "function": "compute_trend_score",
                        "error": str(e),
                        "trend_strength": trend_strength,
                        "time_boost": time_data.get("time_boost", "unknown") if isinstance(time_data, dict) else "unknown"
                    }) + "\n")
            except:
                pass
        else:
            print(f"[TREND ERROR] {error_msg}")
        return {
            "final_score": 0.0,
            "base_score": 0.0,
            "weighted_scores": {},
            "quality_grade": "error",
            "time_boost_applied": 1.0,
            "error": True
        }


def gpt_infer_market_description(symbol: str, candles: List[List]) -> str:
    """
    💬 Etap 9A: Generuj opis rynku dla GPT
    
    Tworzy tekstowy opis wykresu do analizy przez GPT
    
    Args:
        symbol: Symbol trading pair
        candles: Lista OHLCV candles
        
    Returns:
        str: Opis rynku dla GPT
    """
    if not candles or len(candles) < 10:
        return f"{symbol}: Insufficient data for analysis"
    
    try:
        # Podstawowe dane
        closes = [float(c[4]) for c in candles[-20:]]
        highs = [float(c[2]) for c in candles[-10:]]
        lows = [float(c[3]) for c in candles[-10:]]
        volumes = [float(c[5]) for c in candles[-10:]]
        
        current_price = closes[-1]
        
        # EMA21 dla kontekstu trendu
        ema21 = _calculate_ema(closes, min(21, len(closes)))
        ema_current = ema21[-1] if ema21 else current_price
        
        # Trend direction
        price_change = ((closes[-1] - closes[-10]) / closes[-10]) * 100
        trend_direction = "uptrend" if price_change > 2 else "downtrend" if price_change < -2 else "sideways"
        
        # Pullback analysis
        recent_high = max(highs)
        pullback_pct = ((recent_high - current_price) / recent_high) * 100
        
        # Volume trend
        recent_vol = np.mean(volumes[-3:])
        earlier_vol = np.mean(volumes[-10:-3])
        volume_trend = "increasing" if recent_vol > earlier_vol * 1.1 else "decreasing" if recent_vol < earlier_vol * 0.9 else "stable"
        
        # EMA position
        ema_distance = ((current_price - ema_current) / ema_current) * 100
        ema_position = "above EMA21" if ema_distance > 0.5 else "below EMA21" if ema_distance < -0.5 else "near EMA21"
        
        # Last candle analysis
        last_candle = candles[-1]
        last_open, last_high, last_low, last_close = [float(x) for x in last_candle[1:5]]
        candle_type = "bullish" if last_close > last_open else "bearish"
        
        # Wick analysis
        upper_wick = last_high - max(last_open, last_close)
        lower_wick = min(last_open, last_close) - last_low
        body_size = abs(last_close - last_open)
        
        wick_info = ""
        if lower_wick > body_size * 2:
            wick_info = " with significant lower wick (potential support test)"
        elif upper_wick > body_size * 2:
            wick_info = " with significant upper wick (potential resistance)"
        
        # Construct description
        description = (
            f"{symbol}: {trend_direction.capitalize()} market, "
            f"price {ema_position} ({ema_distance:+.1f}%), "
            f"pulled back {pullback_pct:.1f}% from recent high, "
            f"{volume_trend} volume, "
            f"current candle is {candle_type}{wick_info}"
        )
        
        return description
        
    except Exception as e:
        print(f"⚠️ Error generating market description: {e}")
        return f"{symbol}: Error analyzing market structure"


def ask_gpt_trader_opinion(market_description: str) -> Dict:
    """
    💬 Etap 9B: GPT Trader Assistant
    
    Pyta GPT o opinię na temat entry point
    
    Args:
        market_description: Opis rynku z gpt_infer_market_description()
        
    Returns:
        dict: {"gpt_decision": str, "confidence": float, "explanation": str}
    """
    try:
        # Sprawdź czy mamy klucz API
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            return {
                "gpt_decision": "unavailable",
                "confidence": 0.0,
                "explanation": "OpenAI API key not configured"
            }
        
        try:
            import openai
            openai.api_key = openai_api_key
            
            # Prompt dla GPT jako doświadczonego tradera
            prompt = f"""You are an expert crypto trader analyzing pullback entries in uptrends.

Market situation: {market_description}

Based on this market structure, should I join this trend now?

Respond with JSON format:
{{
  "decision": "yes" / "wait" / "no",
  "confidence": 0.0-1.0,
  "explanation": "brief explanation of your reasoning"
}}

Focus on: trend quality, pullback depth, support levels, volume confirmation, entry timing."""

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency trader specializing in trend-following strategies. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            # Parse response
            gpt_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                import re
                json_match = re.search(r'\{.*\}', gpt_text, re.DOTALL)
                if json_match:
                    gpt_json = json.loads(json_match.group())
                    
                    decision = gpt_json.get("decision", "wait").lower()
                    confidence = float(gpt_json.get("confidence", 0.5))
                    explanation = gpt_json.get("explanation", "No explanation provided")
                    
                    # Map decision to standard format
                    if decision in ["yes", "join", "buy"]:
                        decision = "Yes – join trend"
                    elif decision in ["no", "avoid", "skip"]:
                        decision = "No – avoid entry"
                    else:
                        decision = "Wait – monitor"
                    
                    return {
                        "gpt_decision": decision,
                        "confidence": min(max(confidence, 0.0), 1.0),
                        "explanation": explanation
                    }
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as parse_error:
                # Fallback: simple text analysis
                gpt_lower = gpt_text.lower()
                if any(word in gpt_lower for word in ["yes", "good", "buy", "join"]):
                    decision = "Yes – join trend"
                    confidence = 0.7
                elif any(word in gpt_lower for word in ["no", "avoid", "bad"]):
                    decision = "No – avoid entry"  
                    confidence = 0.7
                else:
                    decision = "Wait – monitor"
                    confidence = 0.5
                
                return {
                    "gpt_decision": decision,
                    "confidence": confidence,
                    "explanation": gpt_text[:100] + "..." if len(gpt_text) > 100 else gpt_text
                }
                
        except ImportError:
            return {
                "gpt_decision": "unavailable",
                "confidence": 0.0,
                "explanation": "OpenAI library not installed"
            }
        except Exception as api_error:
            error_msg = f"OpenAI API error: {str(api_error)}"
            print(f"[TREND ERROR] GPT - {error_msg}")
            # Log GPT errors too
            try:
                with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "function": "ask_gpt_trader_opinion",
                        "error": str(api_error),
                        "error_type": type(api_error).__name__,
                        "market_description_length": len(market_description) if market_description else 0
                    }) + "\n")
            except:
                pass
            return {
                "gpt_decision": "error",
                "confidence": 0.0,
                "explanation": f"API error: {str(api_error)[:50]}"
            }
            
    except Exception as e:
        error_msg = f"ask_gpt_trader_opinion failed: {str(e)}"
        print(f"[TREND ERROR] GPT - {error_msg}")
        # Log GPT errors
        try:
            with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "function": "ask_gpt_trader_opinion",
                    "error": str(e),
                    "error_type": type(e).__name__
                }) + "\n")
        except:
            pass
        return {
            "gpt_decision": "error",
            "confidence": 0.0,
            "explanation": "Analysis error"
        }


def interpret_market_as_trader(symbol: str, candles: List[List], utc_hour: int = None, enable_gpt: bool = False) -> Dict:
    """
    🧠 Etap 8: Logika Decyzyjna Tradera
    
    Łączy wszystkie warstwy analizy w logiczną decyzję profesjonalnego tradera
    
    Args:
        symbol: Symbol trading pair (np. 'BTCUSDT')
        candles: Lista OHLCV candles
        utc_hour: Aktualna godzina UTC (0-23), None = auto-detect
        enable_gpt: Czy włączyć GPT trader assistant
        
    Returns:
        dict: Kompletna analiza i decyzja tradera
    """
    if not candles or len(candles) < 40:
        return {
            "decision": "avoid",
            "confidence": 0.0,
            "reasons": ["insufficient_data"],
            "market_context": "unknown",
            "trend_strength": 0.0,
            "entry_quality": 0.0,
            "analysis_complete": False
        }
    
    try:
        # Auto-detect UTC hour if not provided
        if utc_hour is None:
            utc_hour = datetime.now(timezone.utc).hour
        
        # === WYKONAJ WSZYSTKIE 9 ETAPÓW ANALIZY ===
        
        print(f"[TREND DEBUG] {symbol}: Starting 9-stage trend analysis...")
        
        # 1. Kontekst rynkowy
        market_context = determine_market_context(candles, symbol)
        
        # 2. Siła trendu  
        trend_strength = compute_trend_strength(candles, symbol)
        
        # 3. Detekcja pullbacku
        pullback_data = detect_pullback(candles, symbol)
        
        # 4. Reakcja na wsparcie
        support_data = detect_support_reaction(candles, symbol)
        
        # 5. Czas i dynamika
        time_data = market_time_score(utc_hour, symbol)
        
        # 6. Potwierdzenie bounce'a
        bounce_data = detect_bounce_confirmation(candles, symbol)
        
        # 7. Scoring heurystyczny
        scoring_data = compute_trend_score(
            trend_strength, pullback_data, support_data, bounce_data, time_data, symbol
        )
        
        # 9. GPT Analysis (opcjonalny)
        gpt_data = {}
        if enable_gpt:
            market_description = gpt_infer_market_description(symbol, candles)
            gpt_data = ask_gpt_trader_opinion(market_description)
            gpt_data["market_description"] = market_description
        
        # === ADVANCED TRADER DECISION LOGIC ===
        
        # Używaj scoring_data jako podstawy decyzji
        final_score = scoring_data.get("final_score", 0.0)
        quality_grade = scoring_data.get("quality_grade", "poor")
        time_boost = time_data.get("time_boost", 1.0)
        
        reasons = []
        decision = "wait"
        
        # Zbuduj powody na podstawie wszystkich analiz
        if market_context == "pullback":
            reasons.append("pullback_context")
        elif market_context == "impulse":
            reasons.append("impulse_momentum")
        else:
            reasons.append(f"{market_context}_market")
            
        if trend_strength >= 0.7:
            reasons.append("strong_trend")
        elif trend_strength >= 0.5:
            reasons.append("moderate_trend")
        else:
            reasons.append("weak_trend")
            
        if pullback_data.get("detected", False):
            if pullback_data.get("volume_declining", False):
                reasons.append("quality_pullback")
            else:
                reasons.append("pullback_detected")
                
        if support_data.get("support_detected", False):
            support_type = support_data.get("support_type", "")
            reasons.append(f"{support_type}_support")
            
        if bounce_data.get("bounce_confirmed", False):
            pattern = bounce_data.get("pattern", "")
            reasons.append(f"bounce_{pattern}")
            
        # Time boost info
        session = time_data.get("session", "unknown")
        if time_boost > 1.0:
            reasons.append(f"favorable_time_{session}")
        elif time_boost < 1.0:
            reasons.append(f"unfavorable_time_{session}")
            
        # === DECISION THRESHOLDS ===
        # Użyj final_score z compute_trend_score()
        
        if final_score >= 0.75:
            decision = "join_trend"
            reasons.append("excellent_setup")
        elif final_score >= 0.60:
            decision = "wait"  # Może zostać upgraded przez GPT
            reasons.append("good_setup_wait_confirmation")
        elif final_score >= 0.40:
            decision = "wait"
            reasons.append("average_setup")
        else:
            decision = "avoid"
            reasons.append("poor_setup")
            
        # === GPT OVERRIDE LOGIC ===
        # GPT może wpłynąć na decyzję jeśli jest włączony
        gpt_influenced = False
        if enable_gpt and gpt_data:
            gpt_decision = gpt_data.get("gpt_decision", "")
            gpt_confidence = gpt_data.get("confidence", 0.0)
            
            # GPT może upgrade'ować decyzję "wait" do "join_trend" jeśli jest bardzo pewny
            if decision == "wait" and "yes" in gpt_decision.lower() and gpt_confidence >= 0.8:
                decision = "join_trend"
                reasons.append("gpt_confirms_entry")
                gpt_influenced = True
                
            # GPT może downgrade'ować "join_trend" do "wait" jeśli widzi problemy
            elif decision == "join_trend" and "no" in gpt_decision.lower() and gpt_confidence >= 0.8:
                decision = "wait"
                reasons.append("gpt_advises_caution")
                gpt_influenced = True
                
        # Final confidence calculation
        confidence = final_score
        if gpt_influenced and enable_gpt:
            gpt_conf = gpt_data.get("confidence", 0.0)
            confidence = (confidence + gpt_conf) / 2  # Average of algo + GPT
        
        # Debug logging - Final decision
        print(f"[TREND DEBUG] {symbol}: TRADER DECISION = {decision} "
              f"(final_score={final_score:.3f}, confidence={confidence:.3f}, "
              f"quality_grade={quality_grade}, gpt_influenced={gpt_influenced})")
        print(f"[TREND DEBUG] {symbol}: Decision reasons: {reasons[:5]}")  # Top 5 reasons
        
        # === COMPREHENSIVE RESULT ===
        result = {
            "decision": decision,
            "confidence": confidence,
            "reasons": reasons,
            "market_context": market_context,
            "trend_strength": trend_strength,
            "entry_quality": final_score,
            "quality_grade": quality_grade,
            "analysis_complete": True,
            
            # Detailed component data
            "components": {
                "trend_strength": trend_strength,
                "pullback_data": pullback_data,
                "support_data": support_data,
                "bounce_data": bounce_data,
                "time_data": time_data,
                "scoring_data": scoring_data
            },
            
            # GPT data if enabled
            "gpt_analysis": gpt_data if enable_gpt else {},
            "gpt_influenced": gpt_influenced
        }
        
        return result
        
    except Exception as e:
        error_msg = f"interpret_market_as_trader failed: {str(e)}"
        print(f"[TREND ERROR] {symbol} - {error_msg}")
        
        # Log detailed error to file
        try:
            with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "function": "interpret_market_as_trader",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "candles_count": len(candles) if candles else 0,
                    "utc_hour": utc_hour,
                    "enable_gpt": enable_gpt
                }) + "\n")
        except:
            pass  # Don't fail on logging failure
        
        return {
            "decision": "error",
            "confidence": 0.0,
            "reasons": [f"ERROR: {str(e)}"],
            "market_context": "error",
            "trend_strength": 0.0,
            "entry_quality": 0.0,
            "quality_grade": "error",
            "analysis_complete": False,
            "error": True,
            "error_details": error_msg
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
    if not values or len(values) < 2:
        return 0.0
    
    try:
        # Filter out None values and convert to float
        clean_values = []
        for v in values:
            if v is not None and isinstance(v, (int, float)):
                clean_values.append(float(v))
        
        if len(clean_values) < 2:
            return 0.0
            
        n = len(clean_values)
        x = list(range(n))
        y = clean_values
        
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
    except (IndexError, TypeError, ZeroDivisionError, ValueError):
        return 0.0


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

def _detect_engulfing_pattern(candles: List[List]) -> bool:
    """Wykryj bullish engulfing pattern"""
    if len(candles) < 2:
        return False
    
    try:
        prev_candle = candles[-2]
        curr_candle = candles[-1]
        
        prev_open, prev_close = float(prev_candle[1]), float(prev_candle[4])
        curr_open, curr_close = float(curr_candle[1]), float(curr_candle[4])
        
        # Previous candle bearish, current bullish and engulfs previous
        prev_bearish = prev_close < prev_open
        curr_bullish = curr_close > curr_open
        engulfs = curr_open < prev_close and curr_close > prev_open
        
        return prev_bearish and curr_bullish and engulfs
    except:
        return False


def _detect_wick_bounce(candles: List[List]) -> bool:
    """Wykryj odbicie z długim dolnym knotem"""
    if len(candles) < 1:
        return False
    
    try:
        candle = candles[-1]
        open_price, high, low, close = [float(x) for x in candle[1:5]]
        
        body_size = abs(close - open_price)
        lower_wick = min(open_price, close) - low
        
        # Długi dolny knot (większy niż body)
        return lower_wick > body_size * 1.5 and close > open_price
    except:
        return False


def analyze_symbol_trend_mode(symbol: str, candles: List[List], enable_gpt: bool = False) -> Dict:
    """
    Główna funkcja integracyjna dla Advanced Trend-Mode
    
    Args:
        symbol: Symbol trading pair
        candles: Lista OHLCV candles  
        enable_gpt: Czy włączyć GPT trader assistant
        
    Returns:
        dict: Kompletna analiza Advanced Trend-Mode
    """
    # Auto-detect UTC hour
    utc_hour = datetime.now(timezone.utc).hour
    
    result = interpret_market_as_trader(symbol, candles, utc_hour, enable_gpt)
    
    # Dodaj timestamp i symbol do wyniku
    result.update({
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "advanced_trend_mode",
        "version": "2.0"
    })
    
    return result


if __name__ == "__main__":
    # Test Advanced Trend-Mode module
    print("🧪 Testing Advanced Trend-Mode module...")
    
    # Enhanced test data - realistic uptrend with pullback
    test_candles = []
    base_price = 50000
    
    # Build uptrend (50 candles - więcej danych dla analizy)
    for i in range(50):
        # Progressive uptrend with some noise
        trend_component = i * 60  # Base trend
        noise = (i % 3 - 1) * 20  # Small noise
        price = base_price + trend_component + noise
        
        candle = [
            1640995200 + i*900,  # 15min intervals
            price - 40,          # open
            price + 60,          # high
            price - 80,          # low  
            price,               # close
            1000 + i*30 + (50 if i % 2 == 0 else -20)  # volume with pattern
        ]
        test_candles.append(candle)
    
    # Add realistic pullback (4 candles)
    high_price = base_price + 49*60
    for i in range(4):
        pullback_drop = i * 40  # Gradual pullback
        price = high_price - pullback_drop
        
        candle = [
            1640995200 + (50+i)*900,
            price + 20,
            price + 30, 
            price - 60,
            price,
            800 - i*50  # Declining volume in pullback
        ]
        test_candles.append(candle)
    
    # Add bounce attempt (2 candles)
    bounce_price = high_price - 3*40
    for i in range(2):
        price = bounce_price + i*25
        candle = [
            1640995200 + (54+i)*900,
            price - 20,
            price + 50,
            price - 30,
            price,
            900 + i*100  # Increasing volume on bounce
        ]
        test_candles.append(candle)
    
    # Test complete analysis
    print("\n=== BASIC ANALYSIS ===")
    analysis = analyze_symbol_trend_mode("TESTUSDT", test_candles, enable_gpt=False)
    
    print(f"Decision: {analysis['decision']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Market Context: {analysis['market_context']}")
    print(f"Quality Grade: {analysis.get('quality_grade', 'unknown')}")
    print(f"Trend Strength: {analysis['trend_strength']:.2f}")
    print(f"Final Score: {analysis['entry_quality']:.2f}")
    print(f"Main Reasons: {', '.join(analysis['reasons'][:4])}")
    print(f"Analysis Complete: {analysis.get('analysis_complete', False)}")
    
    # Test individual components
    print(f"\n=== COMPONENT BREAKDOWN ===")
    components = analysis.get('components', {})
    
    pullback = components.get('pullback_data', {})
    if pullback.get('detected'):
        print(f"Pullback: {pullback['magnitude']:.1f}% (Volume declining: {pullback['volume_declining']})")
    
    support = components.get('support_data', {})
    if support.get('support_detected'):
        print(f"Support: {support['support_type']} (Reaction: {support['reaction_strength']:.2f})")
    
    bounce = components.get('bounce_data', {})
    if bounce.get('bounce_confirmed'):
        print(f"Bounce: {bounce['pattern']} (Strength: {bounce['bounce_strength']:.2f})")
    
    time_info = components.get('time_data', {})
    print(f"Time: {time_info.get('session', 'unknown')} session (Boost: {time_info.get('time_boost', 1.0):.1f}x)")
    
    scoring = components.get('scoring_data', {})
    if scoring:
        print(f"Score breakdown: {scoring.get('individual_scores', {})}")
    
    print(f"\n✅ Advanced Trend-Mode test completed!")
    print(f"Analysis type: {analysis.get('analysis_type', 'unknown')} v{analysis.get('version', '1.0')}")