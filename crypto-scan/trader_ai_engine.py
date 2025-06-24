"""
Trader AI Engine - Intelligent Market Analysis System

Symuluje myślenie doświadczonego tradera zamiast sztywnych reguł.
Analizuje kontekst rynkowy, zachowanie świec i orderbook jak prawdziwy trader.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# CLIP Integration - Market Phase Labels
CANDIDATE_PHASES = [
    "breakout-continuation",
    "pullback-in-trend", 
    "range-accumulation",
    "trend-reversal",
    "consolidation",
    "fake-breakout",
    "trending-up",
    "trending-down",
    "bullish-momentum",
    "bearish-momentum", 
    "exhaustion-pattern",
    "volume-backed-breakout"
]

# Import existing utilities
try:
    from utils.bybit_orderbook import get_orderbook_snapshot
except ImportError:
    get_orderbook_snapshot = None

# Import TJDE calculation functions with proper error handling
try:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from trend_mode_legacy import compute_trend_strength
    print("[TJDE IMPORT] compute_trend_strength imported successfully")
    TJDE_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"[TJDE ERROR] Failed to import from trend_mode_legacy.py: {e}")
    TJDE_FUNCTIONS_AVAILABLE = False
    
    # Enhanced fallback implementations that calculate meaningful values
    def compute_trend_strength(candles, symbol=None):
        """Enhanced trend strength calculation"""
        if not candles or len(candles) < 20:
            return 0.15  # Minimum baseline value
        
        try:
            closes = [float(c[4]) for c in candles[-20:]]
            if len(closes) < 10:
                return 0.15
                
            # Calculate multiple trend indicators
            # 1. Short vs long term comparison
            short_avg = sum(closes[-5:]) / 5
            long_avg = sum(closes[-15:-10]) / 5
            trend_direction = (short_avg - long_avg) / long_avg if long_avg > 0 else 0
            
            # 2. Momentum calculation
            price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            positive_changes = sum(1 for change in price_changes if change > 0)
            momentum_ratio = positive_changes / len(price_changes)
            
            # 3. Price volatility consideration
            price_range = max(closes) - min(closes)
            avg_price = sum(closes) / len(closes)
            volatility = price_range / avg_price if avg_price > 0 else 0
            
            # Combine indicators
            trend_strength = abs(trend_direction) * 0.4 + momentum_ratio * 0.4 + min(volatility * 2, 0.5) * 0.2
            trend_strength = min(max(trend_strength, 0.1), 1.0)
            
            if symbol:
                print(f"[TJDE CALC] {symbol}: trend_strength = {trend_strength:.3f} (direction: {trend_direction:.3f}, momentum: {momentum_ratio:.3f})")
            return trend_strength
        except Exception as e:
            if symbol:
                print(f"[TJDE ERROR] {symbol}: trend_strength calculation failed: {e}")
            return 0.15

# Enhanced TJDE component calculations
def compute_pullback_quality(candles, symbol=None):
    """Enhanced pullback quality calculation"""
    if not candles or len(candles) < 10:
        return 0.2
    try:
        recent_candles = candles[-10:]
        closes = [float(c[4]) for c in recent_candles]
        highs = [float(c[2]) for c in recent_candles]
        lows = [float(c[3]) for c in recent_candles]
        volumes = [float(c[5]) for c in recent_candles]
        
        # Detect pullback characteristics
        # 1. Find recent high and current position
        max_high = max(highs)
        current_price = closes[-1]
        pullback_depth = (max_high - current_price) / max_high if max_high > 0 else 0
        
        # 2. Volume during pullback (lower volume = healthier pullback)
        recent_volume = sum(volumes[-3:]) / 3
        avg_volume = sum(volumes[:-3]) / len(volumes[:-3]) if len(volumes) > 3 else recent_volume
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        volume_quality = max(0, 2.0 - volume_ratio) / 2.0  # Lower volume = higher quality
        
        # 3. Price stability during pullback
        recent_closes = closes[-3:]
        price_stability = 1.0 - (max(recent_closes) - min(recent_closes)) / max(recent_closes) if max(recent_closes) > 0 else 0
        
        # Combine factors
        pullback_quality = (pullback_depth * 0.4 + volume_quality * 0.3 + price_stability * 0.3)
        pullback_quality = min(max(pullback_quality, 0.1), 1.0)
        
        if symbol:
            print(f"[TJDE CALC] {symbol}: pullback_quality = {pullback_quality:.3f} (depth: {pullback_depth:.3f}, vol_qual: {volume_quality:.3f})")
        return pullback_quality
    except Exception as e:
        if symbol:
            print(f"[TJDE ERROR] {symbol}: pullback_quality calculation failed: {e}")
        return 0.2

def compute_support_reaction(candles, symbol=None):
    """Enhanced support reaction calculation"""
    if not candles or len(candles) < 15:
        return 0.25
    try:
        recent_candles = candles[-15:]
        closes = [float(c[4]) for c in recent_candles]
        lows = [float(c[3]) for c in recent_candles]
        highs = [float(c[2]) for c in recent_candles]
        volumes = [float(c[5]) for c in recent_candles]
        
        # Find support level (recent significant low)
        lowest_idx = lows.index(min(lows))
        lowest_price = lows[lowest_idx]
        current_price = closes[-1]
        
        # 1. Bounce strength from support
        bounce_strength = (current_price - lowest_price) / lowest_price if lowest_price > 0 else 0
        
        # 2. Volume confirmation on bounce
        bounce_volume = sum(volumes[lowest_idx:]) / len(volumes[lowest_idx:]) if lowest_idx < len(volumes) else volumes[-1]
        avg_volume = sum(volumes[:lowest_idx]) / len(volumes[:lowest_idx]) if lowest_idx > 0 else bounce_volume
        volume_confirmation = min(bounce_volume / avg_volume, 2.0) / 2.0 if avg_volume > 0 else 0.5
        
        # 3. Time factor (recent bounce is better)
        time_factor = (len(recent_candles) - lowest_idx) / len(recent_candles)
        recency_bonus = 1.0 - time_factor  # Recent bounce gets higher score
        
        # Combine factors
        support_reaction = bounce_strength * 0.5 + volume_confirmation * 0.3 + recency_bonus * 0.2
        support_reaction = min(max(support_reaction, 0.1), 1.0)
        
        if symbol:
            print(f"[TJDE CALC] {symbol}: support_reaction = {support_reaction:.3f} (bounce: {bounce_strength:.3f}, vol_conf: {volume_confirmation:.3f})")
        return support_reaction
    except Exception as e:
        if symbol:
            print(f"[TJDE ERROR] {symbol}: support_reaction calculation failed: {e}")
        return 0.25

def compute_volume_behavior_score(candles, symbol=None):
    """Enhanced volume behavior calculation"""
    if not candles or len(candles) < 10:
        return 0.3
    try:
        recent_candles = candles[-10:]
        volumes = [float(c[5]) for c in recent_candles]
        closes = [float(c[4]) for c in recent_candles]
        opens = [float(c[1]) for c in recent_candles]
        
        # 1. Recent volume vs historical average
        recent_vol = sum(volumes[-3:]) / 3
        historical_vol = sum(volumes[:-3]) / len(volumes[:-3]) if len(volumes) > 3 else recent_vol
        volume_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        # 2. Volume trend (increasing volume is positive)
        if len(volumes) >= 6:
            recent_trend = sum(volumes[-3:]) / 3
            older_trend = sum(volumes[-6:-3]) / 3
            volume_trend = recent_trend / older_trend if older_trend > 0 else 1.0
        else:
            volume_trend = 1.0
        
        # 3. Volume vs price action correlation
        green_candles = sum(1 for i in range(len(closes)) if closes[i] > opens[i])
        green_ratio = green_candles / len(closes)
        volume_price_sync = volume_ratio * green_ratio  # Higher volume on green candles = good
        
        # Combine factors
        volume_behavior = (min(volume_ratio / 2.0, 1.0) * 0.4 + 
                          min(volume_trend / 1.5, 1.0) * 0.3 + 
                          min(volume_price_sync, 1.0) * 0.3)
        volume_behavior = min(max(volume_behavior, 0.1), 1.0)
        
        if symbol:
            print(f"[TJDE CALC] {symbol}: volume_behavior_score = {volume_behavior:.3f} (ratio: {volume_ratio:.2f}, trend: {volume_trend:.2f})")
        return volume_behavior
    except Exception as e:
        if symbol:
            print(f"[TJDE ERROR] {symbol}: volume_behavior calculation failed: {e}")
        return 0.3

def compute_psych_score(candles, symbol=None):
    """Enhanced psychological score calculation"""
    if not candles or len(candles) < 8:
        return 0.4
    try:
        recent_candles = candles[-8:]
        closes = [float(c[4]) for c in recent_candles]
        opens = [float(c[1]) for c in recent_candles]
        highs = [float(c[2]) for c in recent_candles]
        lows = [float(c[3]) for c in recent_candles]
        
        # 1. Bullish candle pattern
        green_candles = sum(1 for i in range(len(closes)) if closes[i] > opens[i])
        green_ratio = green_candles / len(closes)
        
        # 2. Higher highs and higher lows pattern
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        trend_structure = (higher_highs + higher_lows) / (2 * (len(highs) - 1))
        
        # 3. Momentum consistency
        price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        positive_momentum = sum(1 for change in price_changes if change > 0)
        momentum_consistency = positive_momentum / len(price_changes)
        
        # 4. Recent strength vs weakness
        recent_strength = sum(closes[-3:]) / 3
        earlier_strength = sum(closes[:3]) / 3
        strength_improvement = recent_strength / earlier_strength if earlier_strength > 0 else 1.0
        strength_factor = min(strength_improvement, 1.5) / 1.5
        
        # Combine psychological factors
        psych_score = (green_ratio * 0.3 + 
                      trend_structure * 0.25 + 
                      momentum_consistency * 0.25 + 
                      strength_factor * 0.2)
        psych_score = min(max(psych_score, 0.1), 1.0)
        
        if symbol:
            print(f"[TJDE CALC] {symbol}: psych_score = {psych_score:.3f} (green: {green_ratio:.2f}, structure: {trend_structure:.2f}, momentum: {momentum_consistency:.2f})")
        return psych_score
    except Exception as e:
        if symbol:
            print(f"[TJDE ERROR] {symbol}: psych_score calculation failed: {e}")
        return 0.4


def analyze_market_structure(candles: List[List], symbol: str = None) -> str:
    """
    📊 Etap 1: Rozpoznanie struktury rynku
    
    Analizuje kontekst jak trader patrząc na wykres:
    - Czy to impulse, pullback, range, breakout czy dystrybucja?
    - Bazuje na nachyleniu, zmienności i relacji do EMA
    
    Args:
        candles: Lista OHLCV candles [[timestamp, open, high, low, close, volume], ...]
        symbol: Symbol for debug logging
        
    Returns:
        str: "impulse", "pullback", "range", "breakout", "distribution"
    """
    if not candles or len(candles) < 20:
        return "insufficient_data"
    
    try:
        closes = [float(c[4]) for c in candles[-20:]]
        highs = [float(c[2]) for c in candles[-20:]]
        lows = [float(c[3]) for c in candles[-20:]]
        volumes = [float(c[5]) for c in candles[-20:]]
        
        # Oblicz EMA21 dla trendu
        ema21 = _calculate_ema(closes, min(len(closes), 21))
        
        current_price = closes[-1]
        recent_high = max(highs[-10:])
        recent_low = min(lows[-10:])
        
        # 1. Analiza nachylenia (slope) - kluczowe dla tradera
        price_slope = _calculate_slope(closes[-10:]) if len(closes) >= 10 else 0
        ema_slope = _calculate_slope(ema21[-5:]) if len(ema21) >= 5 else 0
        
        # 2. Zmienność (volatility) - czy rynek jest spokojny czy chaotyczny
        atr = _calculate_atr(highs[-10:], lows[-10:], closes[-10:])
        avg_atr = np.mean([_calculate_atr(highs[i:i+5], lows[i:i+5], closes[i:i+5]) 
                          for i in range(len(highs)-10, len(highs)-5)]) if len(highs) >= 15 else atr
        volatility_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        
        # 3. Pozycja względem EMA - gdzie jesteśmy w trendzie
        price_vs_ema = ((current_price - ema21[-1]) / ema21[-1] * 100) if ema21 else 0
        
        # 4. Analiza wolumenu - czy widzimy absorpcję czy dystrybucję
        volume_trend = _calculate_slope(volumes[-5:]) if len(volumes) >= 5 else 0
        avg_volume = np.mean(volumes[-10:])
        recent_volume = np.mean(volumes[-3:])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # 5. Range analysis - czy jesteśmy w konsolidacji
        range_size = (recent_high - recent_low) / current_price * 100
        
        # === TRADER LOGIC - Jak myśli doświadczony trader ===
        
        # IMPULSE: Silny trend, cena oddalona od EMA, dobry slope
        if (abs(price_slope) > 0.002 and 
            abs(ema_slope) > 0.001 and
            abs(price_vs_ema) > 2.0 and
            price_slope * ema_slope > 0):  # Ten sam kierunek
            context = "impulse"
            
        # PULLBACK: Korekta w trendzie, cena wraca do EMA
        elif (abs(ema_slope) > 0.0005 and  # Trend nadal istnieje
              abs(price_vs_ema) < 3.0 and    # Blisko EMA
              price_slope * ema_slope < 0):   # Przeciwny kierunek (korekta)
            context = "pullback"
            
        # BREAKOUT: Wzrost volatility + volume + ruch poza range
        elif (volatility_ratio > 1.3 and
              volume_ratio > 1.2 and
              range_size < 3.0):  # Poprzedni range był wąski
            context = "breakout"
            
        # DISTRIBUTION: Wysokie volume + małe ruchy ceny + blisko highs
        elif (volume_ratio > 1.1 and
              abs(price_slope) < 0.001 and
              current_price > recent_high * 0.98):
            context = "distribution"
            
        # RANGE: Brak trendu, mała volatility, cena oscyluje
        elif (abs(price_slope) < 0.0005 and
              abs(ema_slope) < 0.0003 and
              volatility_ratio < 1.1):
            context = "range"
            
        else:
            context = "uncertain"
        
        # Enhanced debug logging
        if symbol:
            print(f"[TRADER DEBUG] {symbol}: Market structure = {context}")
            print(f"[TRADER DEBUG] {symbol}: price_slope={price_slope:.6f}, ema_slope={ema_slope:.6f}")
            print(f"[TRADER DEBUG] {symbol}: price_vs_ema={price_vs_ema:.2f}%, volatility_ratio={volatility_ratio:.2f}")
            print(f"[TRADER DEBUG] {symbol}: volume_ratio={volume_ratio:.2f}, range_size={range_size:.2f}%")
        
        return context
        
    except Exception as e:
        if symbol:
            print(f"[TRADER ERROR] {symbol} - analyze_market_structure failed: {e}")
        return "error"


def analyze_candle_behavior(candles: List[List], symbol: str = None) -> Dict:
    """
    📉 Etap 2: Analiza zachowania świec
    
    Ocenia jak trader patrząc na świece:
    - Czy widać agresję kupujących/sprzedających?
    - Czy to absorpcja, panika, czy równowaga?
    - Bullish wicks, małe korpusy, pattern
    
    Args:
        candles: Lista OHLCV candles
        symbol: Symbol for debug logging
        
    Returns:
        dict: {
            "shows_buy_pressure": bool,
            "pattern": str,
            "volume_behavior": str,
            "wick_analysis": str,
            "momentum": str
        }
    """
    if not candles or len(candles) < 5:
        return {
            "shows_buy_pressure": False,
            "pattern": "insufficient_data",
            "volume_behavior": "unknown",
            "wick_analysis": "none",
            "momentum": "neutral"
        }
    
    try:
        # Analizuj ostatnie 5 świec (jak trader patrzy na recent action)
        recent_candles = candles[-5:]
        
        buy_pressure_signals = 0
        pattern_signals = []
        volume_signals = []
        wick_signals = []
        momentum_signals = []
        
        for i, candle in enumerate(recent_candles):
            timestamp, open_price, high, low, close, volume = candle
            open_price, high, low, close, volume = float(open_price), float(high), float(low), float(close), float(volume)
            
            # 1. Analiza body vs wicks (jak trader ocenia siłę)
            body_size = abs(close - open_price)
            total_range = high - low
            
            upper_wick = high - max(close, open_price)
            lower_wick = min(close, open_price) - low
            
            if total_range > 0:
                body_ratio = body_size / total_range
                upper_wick_ratio = upper_wick / total_range
                lower_wick_ratio = lower_wick / total_range
            else:
                body_ratio = upper_wick_ratio = lower_wick_ratio = 0
            
            # 2. Bullish signals (kupujący kontrolują)
            if close > open_price:  # Green candle
                buy_pressure_signals += 1
                
                # Strong bullish body
                if body_ratio > 0.7:
                    momentum_signals.append("strong_bullish")
                
                # Bullish wick (odrzucenie niższych cen)
                if lower_wick_ratio > 0.3 and upper_wick_ratio < 0.1:
                    wick_signals.append("bullish_rejection")
                    buy_pressure_signals += 0.5
                    
            # 3. Absorpcja (buyer stepping in during sell-off)
            elif close < open_price and lower_wick_ratio > 0.4:
                wick_signals.append("absorbing_dip")
                buy_pressure_signals += 0.3
                
            # 4. Volume behavior (confirmation)
            if i > 0:
                prev_volume = float(recent_candles[i-1][5])
                volume_change = (volume - prev_volume) / prev_volume if prev_volume > 0 else 0
                
                if volume_change > 0.2:  # 20% więcej volume
                    if close > open_price:
                        volume_signals.append("volume_buying")
                        buy_pressure_signals += 0.2
                    else:
                        volume_signals.append("volume_selling")
        
        # === TRADER INTERPRETATION ===
        
        # Buy pressure assessment
        shows_buy_pressure = buy_pressure_signals >= 2.0
        
        # Pattern recognition
        if len([s for s in momentum_signals if "strong_bullish" in s]) >= 2:
            pattern = "momentum_building"
        elif "bullish_rejection" in wick_signals and "absorbing_dip" in wick_signals:
            pattern = "absorption_bounce"
        elif "absorbing_dip" in wick_signals:
            pattern = "absorbing_dip"
        elif len([s for s in wick_signals if "bullish" in s]) >= 2:
            pattern = "wick_support"
        else:
            pattern = "neutral_action"
        
        # Volume behavior
        if "volume_buying" in volume_signals and len(volume_signals) >= 2:
            volume_behavior = "buying_volume_increase"
        elif "volume_selling" in volume_signals:
            volume_behavior = "selling_pressure"
        else:
            volume_behavior = "normal_volume"
        
        # Wick analysis summary
        if len(wick_signals) >= 2:
            wick_analysis = "multiple_rejections"
        elif "bullish_rejection" in wick_signals:
            wick_analysis = "lower_rejection"
        elif "absorbing_dip" in wick_signals:
            wick_analysis = "absorption_wick"
        else:
            wick_analysis = "no_significant_wicks"
        
        # Momentum assessment
        if len(momentum_signals) >= 2:
            momentum = "building"
        elif buy_pressure_signals >= 2.5:
            momentum = "positive"
        elif buy_pressure_signals < 1.0:
            momentum = "negative"
        else:
            momentum = "neutral"
        
        result = {
            "shows_buy_pressure": shows_buy_pressure,
            "pattern": pattern,
            "volume_behavior": volume_behavior,
            "wick_analysis": wick_analysis,
            "momentum": momentum,
            "buy_pressure_score": round(buy_pressure_signals, 2)
        }
        
        # Enhanced debug logging
        if symbol:
            print(f"[TRADER DEBUG] {symbol}: Candle pattern = {pattern}, momentum = {momentum}")
            print(f"[TRADER DEBUG] {symbol}: buy_pressure={shows_buy_pressure}, volume_behavior={volume_behavior}")
            print(f"[TRADER DEBUG] {symbol}: wick_analysis={wick_analysis}, buy_pressure_score={round(buy_pressure_signals, 2)}")
        
        return result
        
    except Exception as e:
        if symbol:
            print(f"[TRADER ERROR] {symbol} - analyze_candle_behavior failed: {e}")
        return {
            "shows_buy_pressure": False,
            "pattern": "error",
            "volume_behavior": "error",
            "wick_analysis": "error",
            "momentum": "neutral"
        }


def interpret_orderbook(symbol: str, market_data: Dict = None) -> Dict:
    """
    🧾 Etap 3: Interpretacja orderbook
    
    Analizuje intencje z bid/ask jak doświadczony trader:
    - Czy są warstwy bidów (support)?
    - Spoofing detection?
    - Zmiana siły po jednej stronie?
    
    Args:
        symbol: Trading symbol
        market_data: Optional market data containing orderbook
        
    Returns:
        dict: {
            "bids_layered": bool,
            "spoofing_suspected": bool,
            "ask_pressure": str,
            "bid_strength": str,
            "imbalance": float
        }
    """
    try:
        # Try to get fresh orderbook data
        orderbook_data = None
        
        if get_orderbook_snapshot:
            try:
                orderbook_data = get_orderbook_snapshot(symbol)
            except:
                pass
        
        # Fallback to market_data if available
        if not orderbook_data and market_data:
            orderbook_data = market_data.get('orderbook', {})
        
        if not orderbook_data or not orderbook_data.get('bids') or not orderbook_data.get('asks'):
            return {
                "bids_layered": False,
                "spoofing_suspected": False,
                "ask_pressure": "unknown",
                "bid_strength": "unknown",
                "imbalance": 0.0,
                "data_available": False
            }
        
        bids = orderbook_data['bids'][:10]  # Top 10 bids
        asks = orderbook_data['asks'][:10]  # Top 10 asks
        
        # === TRADER ORDERBOOK ANALYSIS ===
        
        # 1. Bid layering analysis (support levels)
        bid_sizes = [float(bid[1]) for bid in bids]
        ask_sizes = [float(ask[1]) for ask in asks]
        
        total_bid_volume = sum(bid_sizes)
        total_ask_volume = sum(ask_sizes)
        
        # Check for layered bids (multiple large orders)
        large_bids = [size for size in bid_sizes if size > np.mean(bid_sizes) * 1.5]
        bids_layered = len(large_bids) >= 3
        
        # 2. Spoofing detection (unusually large orders that may be fake)
        max_bid = max(bid_sizes) if bid_sizes else 0
        avg_bid = np.mean(bid_sizes) if bid_sizes else 0
        max_ask = max(ask_sizes) if ask_sizes else 0
        avg_ask = np.mean(ask_sizes) if ask_sizes else 0
        
        # Spoofing suspected if one order is 5x+ larger than average
        spoofing_suspected = (max_bid > avg_bid * 5 or max_ask > avg_ask * 5) if avg_bid > 0 and avg_ask > 0 else False
        
        # 3. Order imbalance (trader edge detection)
        total_volume = total_bid_volume + total_ask_volume
        imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0.0
        
        # 4. Ask pressure assessment
        if total_ask_volume < total_bid_volume * 0.5:
            ask_pressure = "light"
        elif total_ask_volume > total_bid_volume * 1.5:
            ask_pressure = "heavy"
        else:
            ask_pressure = "balanced"
        
        # 5. Bid strength assessment
        if bids_layered and imbalance > 0.2:
            bid_strength = "strong_support"
        elif imbalance > 0.1:
            bid_strength = "decent_support"
        elif imbalance < -0.1:
            bid_strength = "weak"
        else:
            bid_strength = "neutral"
        
        result = {
            "bids_layered": bids_layered,
            "spoofing_suspected": spoofing_suspected,
            "ask_pressure": ask_pressure,
            "bid_strength": bid_strength,
            "imbalance": round(imbalance, 3),
            "data_available": True,
            "total_bid_volume": round(total_bid_volume, 2),
            "total_ask_volume": round(total_ask_volume, 2)
        }
        
        print(f"[TRADER DEBUG] {symbol}: Orderbook bid_strength={bid_strength}, ask_pressure={ask_pressure}")
        print(f"[TRADER DEBUG] {symbol}: imbalance={imbalance:.3f}, bids_layered={bids_layered}, spoofing={spoofing_suspected}")
        print(f"[TRADER DEBUG] {symbol}: bid_volume={round(total_bid_volume, 2)}, ask_volume={round(total_ask_volume, 2)}")
        
        return result
        
    except Exception as e:
        print(f"[TRADER ERROR] {symbol} - interpret_orderbook failed: {e}")
        return {
            "bids_layered": False,
            "spoofing_suspected": False,
            "ask_pressure": "unknown",
            "bid_strength": "unknown",
            "imbalance": 0.0,
            "data_available": False
        }


def compute_trader_score(features: Dict, symbol: str = None) -> Dict:
    """
    🧠 Inteligentny heurystyczny scoring jak trader
    
    Ważone punkty z kontekstem i adaptacją - nie sztywne reguły
    
    Args:
        features: Dict z cechami rynku {
            "trend_strength": 0.0-1.0,
            "pullback_quality": 0.0-1.0,  
            "support_reaction_strength": 0.0-1.0,
            "bounce_confirmation_strength": 0.0-1.0,
            "orderbook_alignment": 0.0-1.0,
            "time_boost": 0.0-1.0,
            "market_context": str
        }
        symbol: Symbol for logging
        
    Returns:
        dict: {
            "final_score": float,
            "quality_grade": str,
            "score_breakdown": dict,
            "weights_used": dict,
            "context_adjustment": str
        }
    """
    try:
        # Base weights (standardowe dla balanced market)
        base_weights = {
            "trend_strength": 0.25,
            "pullback_quality": 0.20,
            "support_reaction_strength": 0.20,
            "bounce_confirmation_strength": 0.15,
            "orderbook_alignment": 0.10,
            "time_boost": 0.10
        }
        
        # Context-aware weight adjustments (jak myśli trader)
        market_context = features.get("market_context", "neutral")
        weights = base_weights.copy()
        context_adjustment = "standard"
        
        if market_context == "impulse":
            # W impulsie liczy się trend strength + orderbook
            weights["trend_strength"] = 0.35
            weights["orderbook_alignment"] = 0.15
            weights["pullback_quality"] = 0.15
            weights["support_reaction_strength"] = 0.20
            weights["bounce_confirmation_strength"] = 0.10
            weights["time_boost"] = 0.05
            context_adjustment = "impulse_focused"
            
        elif market_context == "pullback":
            # W pullback kluczowe: support + bounce + quality
            weights["pullback_quality"] = 0.30
            weights["support_reaction_strength"] = 0.25
            weights["bounce_confirmation_strength"] = 0.20
            weights["trend_strength"] = 0.15
            weights["orderbook_alignment"] = 0.05
            weights["time_boost"] = 0.05
            context_adjustment = "pullback_focused"
            
        elif market_context == "breakout":
            # W breakout: volume + time + momentum
            weights["trend_strength"] = 0.30
            weights["time_boost"] = 0.20
            weights["orderbook_alignment"] = 0.20
            weights["bounce_confirmation_strength"] = 0.15
            weights["support_reaction_strength"] = 0.10
            weights["pullback_quality"] = 0.05
            context_adjustment = "breakout_focused"
            
        elif market_context == "range":
            # W range wszystko ma mniejsze znaczenie, liczy się patience
            weights = {k: v * 0.7 for k, v in weights.items()}  # Penalty za range
            weights["time_boost"] = 0.05  # Range nie liczy się timing
            context_adjustment = "range_penalty"
            
        elif market_context == "distribution":
            # Dystrybucja = avoid, heavy penalty
            weights = {k: v * 0.3 for k, v in weights.items()}
            context_adjustment = "distribution_avoid"
        
        # === CALCULATE ACTUAL TJDE COMPONENTS ===
        # Get candles from features if available
        candles_15m = features.get('candles_15m', [])
        candles_5m = features.get('candles_5m', [])
        symbol = features.get('symbol', 'UNKNOWN')
        
        # Calculate real TJDE component scores - ALWAYS execute calculations
        print(f"[TJDE CALC DEBUG] {symbol}: Starting calculations with {len(candles_15m)} candles")
        
        # Calculate with available data (even if less than 20 candles)
        trend_strength = compute_trend_strength(candles_15m, symbol)
        pullback_quality = compute_pullback_quality(candles_15m, symbol)
        support_reaction = compute_support_reaction(candles_15m, symbol)
        volume_behavior_score = compute_volume_behavior_score(candles_15m, symbol)
        psych_score = compute_psych_score(candles_15m, symbol)
        
        # Update features with calculated values
        features['trend_strength'] = trend_strength
        features['pullback_quality'] = pullback_quality
        features['support_reaction_strength'] = support_reaction
        features['volume_behavior_score'] = volume_behavior_score
        features['psych_score'] = psych_score
        
        print(f"[TJDE CALC] {symbol}: trend={trend_strength:.3f}, pullback={pullback_quality:.3f}, support={support_reaction:.3f}")
        print(f"[TJDE CALC] {symbol}: volume={volume_behavior_score:.3f}, psych={psych_score:.3f}")
        
        # Validate calculations were successful
        if all(val == 0.0 for val in [trend_strength, pullback_quality, support_reaction, volume_behavior_score, psych_score]):
            print(f"[TJDE CALC WARNING] {symbol}: All components returned 0.0 - calculation functions may have failed")
        
        # Calculate weighted score
        score_breakdown = {}
        total_score = 0.0
        
        for feature, weight in weights.items():
            feature_value = features.get(feature, 0.0)
            # Ensure value is in 0.0-1.0 range
            feature_value = max(0.0, min(1.0, feature_value))
            
            contribution = feature_value * weight
            score_breakdown[feature] = round(contribution, 4)
            total_score += contribution
        
        # Ensure final score is in 0.0-1.0 range
        final_score = max(0.0, min(1.0, total_score))
        
        # Quality grade assessment (jak trader ocenia setup)
        if final_score >= 0.85:
            quality_grade = "excellent"
        elif final_score >= 0.75:
            quality_grade = "strong"  
        elif final_score >= 0.65:
            quality_grade = "good"
        elif final_score >= 0.50:
            quality_grade = "neutral-watch"
        elif final_score >= 0.35:
            quality_grade = "weak"
        else:
            quality_grade = "very_poor"
        
        result = {
            "final_score": round(final_score, 3),
            "quality_grade": quality_grade,
            "score_breakdown": score_breakdown,
            "weights_used": {k: round(v, 3) for k, v in weights.items()},
            "context_adjustment": context_adjustment,
            "market_context": market_context
        }
        
        # Enhanced logging dla tradera
        if symbol:
            # Show individual component values from features (after calculation)
            calc_trend = features.get('trend_strength', 0.0)
            calc_pullback = features.get('pullback_quality', 0.0)
            calc_support = features.get('support_reaction_strength', 0.0)
            calc_volume = features.get('volume_behavior_score', 0.0)
            calc_psych = features.get('psych_score', 0.0)
            
            print(f"[TJDE SCORE] trend_strength={calc_trend:.2f}, pullback_quality={calc_pullback:.2f}, final_score={final_score:.2f}")
            print(f"[TJDE SCORE] volume_behavior_score={calc_volume:.1f}, psych_score={calc_psych:.1f}, support_reaction={calc_support:.1f}")
            
            breakdown_str = ", ".join([f"{k.split('_')[0]}={v:.3f}" for k, v in score_breakdown.items()])
            print(f"[TRADER SCORE] {symbol} → {final_score:.3f} ({quality_grade}) [{context_adjustment}]")
            print(f"[TRADER SCORE] {symbol}: {breakdown_str}")
        
        # Log to file dla analizy
        _log_trader_score(symbol, result, features)
        
        return result
        
    except Exception as e:
        if symbol:
            print(f"[TRADER ERROR] {symbol} - compute_trader_score failed: {e}")
        return {
            "final_score": 0.0,
            "quality_grade": "error",
            "score_breakdown": {},
            "weights_used": {},
            "context_adjustment": "error",
            "error": str(e)
        }


def simulate_trader_decision_advanced(features: dict) -> dict:
    """
    🧠 Nowa Generacja Adaptacyjnego Scoringu - Profesjonalna, kontekstowa wersja decyzyjna
    
    Wykorzystuje adaptacyjne ważenie scoringu oraz modyfikatory zależne od fazy rynku i psychologii.
    Bez sztywnych reguł - myśli jak profesjonalny trader oceniając sytuację wielowarstwowo.
    
    Args:
        features: Dict z cechami rynku i kontekstem
        
    Returns:
        dict: Adaptacyjna decyzja tradera z dynamicznym scoringiem
    """
    try:
        score_breakdown = {}
        context_modifiers = []
        
        # === ETAP 1: WYCIĄGANIE CECH Z WEJŚCIA ===
        trend_strength = features.get("trend_strength", 0.0)
        pullback_quality = features.get("pullback_quality", 0.0)
        support_reaction = features.get("support_reaction", 0.0)
        liquidity_pattern_score = features.get("liquidity_pattern_score", 0.0)
        psych_score = features.get("psych_score", 0.0)
        htf_supportive_score = features.get("htf_supportive_score", 0.0)
        market_phase = features.get("market_phase", "unknown")
        market_phase_modifier = features.get("market_phase_modifier", 0.0)
        price_action_pattern = features.get("price_action_pattern", "none")
        volume_behavior = features.get("volume_behavior", "neutral")
        htf_trend_match = features.get("htf_trend_match", False)
        
        print(f"[TRADER ADAPTIVE] Analyzing features: phase={market_phase}, trend={trend_strength:.3f}, pullback={pullback_quality:.3f}")
        
        # === ETAP 2: DYNAMIC TJDE WEIGHTS LOADING ===
        from utils.scoring import load_tjde_weights, apply_phase_adjustments
        
        # Load dynamic weights from JSON file with fallback to defaults
        base_weights = load_tjde_weights()
        
        # Apply intelligent phase-specific adjustments
        weights = apply_phase_adjustments(base_weights, market_phase)
        print(f"[TJDE WEIGHTS] Phase-adjusted weights applied for {market_phase}")
        
        # === ETAP 3: SCORING Z DYNAMICZNYMI WAGAMI ===
        score = (
            trend_strength * weights["trend_strength"] +
            pullback_quality * weights["pullback_quality"] +
            support_reaction * weights["support_reaction"] +
            liquidity_pattern_score * weights["liquidity_pattern_score"] +
            psych_score * weights["psych_score"] +
            htf_supportive_score * weights["htf_supportive_score"] +
            market_phase_modifier * weights["market_phase_modifier"]
        )
        
        print(f"[TRADER SCORE] Base score: {score:.3f}")
        
        # === ETAP 4: KONTEKSTOWE MODYFIKATORY SCORINGU ===
        if volume_behavior == "supporting" and price_action_pattern in ["impulse", "continuation"]:
            score += 0.07
            context_modifiers.append("volume_backed_breakout")
            print(f"[CONTEXT MODIFIER] +0.07 for volume_backed_breakout")
        
        if psych_score < 0.3:
            score -= 0.05
            context_modifiers.append("psych_noise_penalty")
            print(f"[CONTEXT MODIFIER] -0.05 for psych_noise_penalty")
        
        if market_phase == "exhaustion-pullback":
            score -= 0.08
            context_modifiers.append("phase_exhaustion_penalty")
            print(f"[CONTEXT MODIFIER] -0.08 for phase_exhaustion_penalty")
        
        if htf_trend_match and htf_supportive_score > 0.5:
            score += 0.05
            context_modifiers.append("htf_alignment_boost")
            print(f"[CONTEXT MODIFIER] +0.05 for htf_alignment_boost")
        
        # === ETAP 5: INTERPRETACJA KOŃCOWA ===
        if score >= 0.70:
            decision = "join_trend"
            grade = "strong"
        elif score >= 0.45:
            decision = "consider_entry"
            grade = "moderate"
        else:
            decision = "avoid"
            grade = "weak"
        
        symbol = features.get("symbol", "UNKNOWN")
        print(f"[TJDE DEBUG] Final decision for {symbol}: {decision}, Score: {score:.3f}")
        print(f"[TJDE DEBUG] Phase: {features.get('market_phase')} | CLIP Confidence: N/A")
        logging.debug(f"[TJDE DEBUG] Complete decision for {symbol}: decision={decision}, score={score:.3f}, grade={grade}, phase={features.get('market_phase')}")
        
        if context_modifiers:
            print(f"[CONTEXT MODIFIERS] Applied: {', '.join(context_modifiers)}")
            logging.debug(f"[TJDE DEBUG] Context modifiers for {symbol}: {context_modifiers}")
        
        # === ETAP 6: ADVANCED CLIP INTEGRATION WITH CONTEXTUAL BOOSTS ===
        clip_modifier = 0.0
        clip_predicted_phase = ""
        clip_info = {
            "predicted_phase": "N/A",
            "confidence": 0.0,
            "modifier": 0.0,
            "prediction_source": "unavailable"
        }
        clip_boost_applied = False
        cluster_enhanced = False
        cluster_modifier = 0.0
        cluster_info = {}
        ai_pattern_matched = False
        ai_pattern_info = {}
        original_decision = decision
        
        try:
            from utils.clip_prediction_loader import load_clip_prediction
            
            # Get symbol from features
            symbol = features.get("symbol", "UNKNOWN")
            
            # Load CLIP prediction
            clip_prediction = load_clip_prediction(symbol)
            
            if clip_prediction:
                prediction_str = clip_prediction.lower()
                
                # Apply contextual boosts based on CLIP prediction
                if "breakout-continuation" in prediction_str:
                    # Core boosts
                    trend_strength *= 1.2
                    liquidity_pattern_score *= 1.15
                    
                    # Contextual volume behavior boost
                    volume_behavior = features.get('volume_behavior', '')
                    if 'buying_volume_increase' in volume_behavior or 'volume_spike' in volume_behavior:
                        trend_strength *= 1.1
                        context_modifiers.append("CLIP: volume-backed breakout → trend boost")
                    
                    context_modifiers.append("CLIP: breakout-continuation → liquidity/trend boost")
                    clip_modifier = 0.08
                    
                elif "pullback-in-trend" in prediction_str:
                    # Core boosts
                    pullback_quality *= 1.2
                    support_reaction *= 1.1
                    
                    # Contextual psych_flags boost
                    psych_flags = features.get('psych_flags', {})
                    if psych_flags.get('fakeout_rejection') or psych_flags.get('bounce_confirmed'):
                        psych_score *= 1.2
                        context_modifiers.append("CLIP: rejection pullback → psych_score boost")
                    
                    context_modifiers.append("CLIP: pullback-in-trend → pullback/support boost")
                    clip_modifier = 0.05
                    
                elif "trend-reversal" in prediction_str:
                    # Reversal caution
                    market_phase_modifier *= -1.0
                    
                    # Contextual psych penalty
                    psych_flags = features.get('psych_flags', {})
                    if psych_flags.get('liquidity_grab') or psych_flags.get('trap_pattern'):
                        psych_score *= 0.5
                        context_modifiers.append("CLIP: liquidity trap in reversal → psych penalized")
                    
                    context_modifiers.append("CLIP: trend-reversal → caution bias")
                    clip_modifier = -0.08
                    
                elif "volume-backed breakout" in prediction_str:
                    # Strong volume breakout
                    trend_strength *= 1.3
                    liquidity_pattern_score *= 1.25
                    clip_modifier = 0.10
                    context_modifiers.append("CLIP: volume-backed breakout → strong boost")
                    
                elif "fake-breakout" in prediction_str or "fakeout" in prediction_str:
                    # Fakeout warning
                    trend_strength *= 0.7
                    support_reaction *= 0.8
                    clip_modifier = -0.12
                    context_modifiers.append("CLIP: fake-breakout → strong caution")
                    
                elif "exhaustion pattern" in prediction_str:
                    # Exhaustion warning
                    trend_strength *= 0.6
                    psych_score *= 0.7
                    clip_modifier = -0.10
                    context_modifiers.append("CLIP: exhaustion pattern → trend weakness")
                    
                elif "consolidation" in prediction_str or "range-accumulation" in prediction_str:
                    # Range bound
                    pullback_quality *= 1.1
                    support_reaction *= 1.05
                    clip_modifier = 0.03
                    context_modifiers.append("CLIP: consolidation → range trading bias")
                
                # Store CLIP info
                clip_info = {
                    "predicted_phase": prediction_str,
                    "confidence": 0.75,  # Default confidence from file
                    "modifier": clip_modifier,
                    "prediction_source": "file_loader",
                    "boosts_applied": [mod for mod in context_modifiers if "CLIP:" in mod]
                }
                
                clip_boost_applied = True
                
                print(f"[CLIP INTEGRATION] {symbol}: {prediction_str}")
                print(f"[CLIP BOOSTS] Applied: {clip_modifier:+.3f} with contextual modifiers")
                
            else:
                # Fallback to chart-based prediction
                try:
                    from ai.clip_predictor import predict_clip_chart
                    import glob
                except ImportError:
                    try:
                        from vision_ai.clip_predictor import predict_clip_chart
                        import glob
                    except ImportError:
                        print(f"[CLIP FALLBACK] No clip_predictor available for {symbol}")
                        clip_prediction = None
                
                chart_locations = [
                    f"charts/{symbol}_*.png",
                    f"exports/{symbol}_*.png", 
                    f"training_data/clip/{symbol}_*.png"
                ]
                
                chart_path = None
                for pattern in chart_locations:
                    matches = glob.glob(pattern)
                    if matches:
                        chart_path = sorted(matches, reverse=True)[0]
                        break
                
                if chart_path:
                    # Use global CANDIDATE_PHASES defined at module level
                    chart_prediction = predict_clip_chart(chart_path, CANDIDATE_PHASES)
                    
                    if chart_prediction and chart_prediction.get("success"):
                        clip_predicted_phase = chart_prediction["predicted_label"]
                        clip_confidence = chart_prediction["confidence"]
                        
                        if clip_confidence > 0.4:
                            clip_modifier = 0.05 * clip_confidence  # Conservative chart-based boost
                            clip_info = {
                                "predicted_phase": clip_predicted_phase,
                                "confidence": clip_confidence,
                                "modifier": clip_modifier,
                                "prediction_source": "chart_analysis"
                            }
                            context_modifiers.append(f"CLIP: chart-based {clip_predicted_phase}")
                            clip_boost_applied = True
                        else:
                            # Low confidence prediction
                            clip_info = {
                                "predicted_phase": clip_predicted_phase,
                                "confidence": clip_confidence,
                                "modifier": 0.0,
                                "prediction_source": "chart_analysis_low_confidence"
                            }
                    else:
                        # Handle failed prediction
                        confidence_val = chart_prediction.get("confidence", 0.0) if chart_prediction else 0.0
                        clip_info = {
                            "predicted_phase": chart_prediction.get("predicted_label", "no_prediction") if chart_prediction else "no_prediction",
                            "confidence": confidence_val,
                            "modifier": 0.0,
                            "prediction_source": "chart_analysis_failed",
                            "reason": chart_prediction.get("reason", "Unknown error") if chart_prediction else "No prediction returned"
                        }
                
                if not clip_boost_applied:
                    context_modifiers.append("CLIP: No prediction available")
                    clip_info = {
                        "predicted_phase": "no_prediction",
                        "confidence": 0.0,
                        "modifier": 0.0,
                        "prediction_source": "unavailable"
                    }
                
        except Exception as clip_error:
            print(f"[CLIP INTEGRATION ERROR] {symbol}: {clip_error}")
            context_modifiers.append("CLIP: Integration error")
        
        # Apply CLIP modifier to final score
        enhanced_score = score + clip_modifier
        
        # Update decision based on CLIP enhancement
        original_decision = decision
        if clip_modifier != 0:
            if enhanced_score >= 0.75 and decision != "join_trend":
                decision = "join_trend"
                grade = "excellent" if enhanced_score >= 0.85 else "strong"
                print(f"[CLIP BOOST] {symbol}: Decision upgraded from {original_decision.upper()} to JOIN_TREND")
            elif enhanced_score >= 0.65 and decision == "avoid":
                decision = "consider_entry"
                grade = "moderate"
                print(f"[CLIP RECOVERY] {symbol}: Decision upgraded from AVOID to CONSIDER_ENTRY")
            elif enhanced_score < 0.40 and decision == "consider_entry":
                decision = "avoid"
                grade = "weak"
                print(f"[CLIP WARNING] {symbol}: Decision downgraded to AVOID due to negative visual signals")
            elif enhanced_score < 0.25:
                decision = "avoid"
                grade = "very_poor"
                print(f"[CLIP DANGER] {symbol}: Strong avoid signal from visual analysis")
        
        # === ETAP 7: CLUSTER ANALYSIS INTEGRATION ===
        cluster_enhanced = False
        cluster_modifier = 0.0
        cluster_info = {}
        
        try:
            from cluster_integration import get_cluster_integration
            
            # Create temporary TJDE result for cluster analysis
            temp_tjde_result = {
                "symbol": features.get("symbol", "UNKNOWN"),
                "final_score": enhanced_score,
                "decision": decision,
                "score_breakdown": {
                    "trend_strength": trend_strength,
                    "pullback_quality": pullback_quality,
                    "support_reaction": support_reaction,
                    "liquidity_pattern_score": liquidity_pattern_score,
                    "psych_score": psych_score,
                    "htf_supportive_score": htf_supportive_score,
                    "market_phase_modifier": market_phase_modifier
                }
            }
            
            # Get cluster integration
            cluster_integration = get_cluster_integration()
            
            # Enhance with cluster analysis
            cluster_enhanced_result = cluster_integration.enhance_tjde_with_cluster(
                features.get("symbol", "UNKNOWN"),
                temp_tjde_result
            )
            
            if cluster_enhanced_result.get("cluster_enhanced"):
                # Update score and decision with cluster enhancement
                cluster_enhanced_score = cluster_enhanced_result.get("final_score", enhanced_score)
                cluster_enhanced_decision = cluster_enhanced_result.get("decision", decision)
                cluster_info = cluster_enhanced_result.get("cluster_info", {})
                
                cluster_modifier = cluster_info.get("score_modifier", 0.0)
                
                # Update final values
                enhanced_score = cluster_enhanced_score
                decision = cluster_enhanced_decision
                cluster_enhanced = True
                
                # Add cluster reasoning
                if cluster_modifier != 0:
                    cluster_reason = f"Cluster {cluster_info.get('cluster', -1)}: {cluster_info.get('recommendation', 'neutral')} ({cluster_modifier:+.3f})"
                    context_modifiers.append(cluster_reason)
                
                print(f"[CLUSTER] {features.get('symbol', 'UNKNOWN')}: Enhanced with cluster analysis")
                print(f"[CLUSTER] Modifier: {cluster_modifier:+.3f}, Quality: {cluster_info.get('quality_score', 0):.3f}")
            
        except Exception as cluster_error:
            print(f"[CLUSTER ERROR] {features.get('symbol', 'UNKNOWN')}: {cluster_error}")
        
        # === ETAP 8: AI HEURISTIC PATTERN CHECKING ===
        ai_pattern_matched = False
        ai_pattern_info = {}
        
        try:
            from utils.ai_heuristic_pattern_checker import check_known_success_patterns
            
            # Check for AI heuristic patterns that override low scoring
            heuristic_alert = check_known_success_patterns(features, enhanced_score)
            
            if heuristic_alert:
                # Override decision with heuristic pattern
                original_decision = decision
                decision = "heuristic_alert"
                ai_pattern_matched = True
                ai_pattern_info = heuristic_alert
                
                # Add AI pattern reasoning
                pattern_reason = f"AI Pattern: {heuristic_alert['label']} ({heuristic_alert['confidence']:.2f} confidence)"
                context_modifiers.append(pattern_reason)
                
                print(f"[AI PATTERN] {features.get('symbol', 'UNKNOWN')}: {heuristic_alert['label']}")
                print(f"[AI PATTERN] Confidence: {heuristic_alert['confidence']:.2f}, Score: {enhanced_score:.3f} -> HEURISTIC ALERT")
                print(f"[AI PATTERN] Features: {', '.join(heuristic_alert['features_matched'])}")
                
        except Exception as ai_error:
            print(f"[AI PATTERN ERROR] {features.get('symbol', 'UNKNOWN')}: {ai_error}")
        
        # === ETAP 9: FINAL RESULT ASSEMBLY WITH ALL ENHANCEMENTS ===
        
        # Store CLIP info in features for debug output after CLIP processing
        features["clip_confidence"] = clip_info.get("confidence", "N/A")
        features["clip_phase"] = clip_info.get("predicted_phase", "N/A")
        
        # Enhanced CLIP debug logging
        if clip_info.get("confidence", 0) > 0:
            print(f"[CLIP SUCCESS] Prediction: {clip_info.get('predicted_phase')} (confidence: {clip_info.get('confidence'):.3f})")
        else:
            print(f"[CLIP STATUS] No valid prediction available")
        
        # Prepare enhanced debug info for alerts and logging
        debug_info = {
            "base_score": round(score, 3),
            "enhanced_score": round(enhanced_score, 3),
            "clip_phase_prediction": clip_info.get("predicted_phase", "unknown"),
            "clip_confidence": round(clip_info.get("confidence", 0), 3),
            "clip_modifier": round(clip_modifier, 3),
            "clip_boost_applied": clip_boost_applied,
            "contextual_boosts": [mod for mod in context_modifiers if "CLIP:" in mod],
            "cluster_enhanced": cluster_enhanced,
            "cluster_modifier": round(cluster_modifier, 3),
            "cluster_info": cluster_info,
            "ai_pattern_matched": ai_pattern_matched,
            "ai_pattern_info": ai_pattern_info,
            "decision_change": original_decision != decision,
            "original_decision": original_decision
        }
        
        final_result = {
            "decision": decision,
            "final_score": round(enhanced_score, 3),
            "quality_grade": grade,
            "context_modifiers": context_modifiers,
            "weights": weights,
            "used_features": {
                "trend_strength": trend_strength,
                "pullback_quality": pullback_quality,
                "support_reaction": support_reaction,
                "liquidity_pattern_score": liquidity_pattern_score,
                "psych_score": psych_score,
                "htf_supportive_score": htf_supportive_score,
                "market_phase_modifier": market_phase_modifier
            },
            "score_breakdown": {
                "trend_strength": round(trend_strength, 3),
                "pullback_quality": round(pullback_quality, 3),
                "support_reaction": round(support_reaction, 3),
                "liquidity_pattern_score": round(liquidity_pattern_score, 3),
                "psych_score": round(psych_score, 3),
                "htf_supportive_score": round(htf_supportive_score, 3),
                "market_phase_modifier": round(market_phase_modifier, 3),
                "clip_prediction": clip_info.get("predicted_phase", "none")
            },
            "market_phase": market_phase,
            "confidence": min(enhanced_score * 1.2, 1.0),
            "clip_enhanced": clip_modifier != 0,
            "clip_info": clip_info,
            "clip_modifier": clip_modifier,
            "base_score_before_clip": score,
            "debug_info": debug_info
        }
        
        # Enhanced logging for CLIP integration
        if clip_modifier != 0:
            print(f"[CLIP INTEGRATION] {symbol} Final Impact:")
            print(f"   Base Score: {score:.3f} → Enhanced: {enhanced_score:.3f}")
            print(f"   Phase: {clip_info.get('predicted_phase', 'unknown')}")
            print(f"   Decision: {original_decision.upper()} → {decision.upper()}")
            
            # Log contextual boosts
            contextual_boosts = debug_info.get('contextual_boosts', [])
            if contextual_boosts:
                print(f"   Contextual Boosts Applied:")
                for boost in contextual_boosts:
                    print(f"     • {boost}")
        else:
            print(f"[CLIP INTEGRATION] {symbol}: No enhancement applied")
        
        # Enhanced logging for AI pattern matching
        if ai_pattern_matched:
            print(f"[AI PATTERN MATCH] {symbol} Heuristic Override:")
            print(f"   Pattern: {ai_pattern_info.get('label', 'unknown')}")
            print(f"   Confidence: {ai_pattern_info.get('confidence', 0):.2f}")
            print(f"   Description: {ai_pattern_info.get('description', 'N/A')}")
            print(f"   Original Score: {enhanced_score:.3f} (below {ai_pattern_info.get('min_score', 0):.2f} threshold)")
            print(f"   Decision Override: {original_decision.upper()} → HEURISTIC_ALERT")
        
        return final_result
        
        # Enhanced logging
        _log_advanced_trader_decision(symbol, result)
        
        return result
        
    except Exception as e:
        print(f"❌ [ADVANCED TRADER ERROR] {symbol}: {e}")
        return {
            "decision": "avoid",
            "confidence": 0.0,
            "final_score": 0.0,
            "reasons": [f"Advanced analysis error: {e}"],
            "quality_grade": "error",
            "analysis_quality": "error"
        }


def simulate_trader_decision(
    symbol: str,
    market_context: str,
    candle_behavior: Dict,
    orderbook_info: Dict,
    current_price: float = None
) -> Dict:
    """
    🧠 TraderWeightedDecisionEngine - Weighted Feature Scoring System
    
    Nowy system scoringu oparty na ważonych składnikach zamiast sztywnych reguł:
    - Każda cecha wnosi częściowy scoring z określoną wagą
    - Finalna decyzja to suma ważonych składników  
    - Symuluje podejście realnego tradera bez rigid if/then logic
    
    Args:
        symbol: Trading symbol
        market_context: Wynik z analyze_market_structure()
        candle_behavior: Wynik z analyze_candle_behavior()
        orderbook_info: Wynik z interpret_orderbook()
        current_price: Current price for context
        
    Returns:
        dict: {
            "decision": str,        # "join_trend", "consider_entry", "avoid"
            "confidence": float,    # 0.0-1.0
            "final_score": float,   # 0.0-1.0
            "reasons": List[str],   # Why this decision
            "quality_grade": str,   # "excellent", "strong", "good", etc.
            "score_breakdown": dict # Detailed weighted feature scoring
        }
    """
    try:
        # Extract raw features - handle None values safely
        buy_pressure = candle_behavior.get("shows_buy_pressure", False)
        pattern = candle_behavior.get("pattern", "neutral")
        volume_behavior = candle_behavior.get("volume_behavior", "average")
        momentum = candle_behavior.get("momentum", "neutral")
        wick_analysis = candle_behavior.get("wick_analysis", "neutral")
        
        bids_layered = orderbook_info.get("bids_layered", False)
        spoofing_suspected = orderbook_info.get("spoofing_suspected", False)
        imbalance = orderbook_info.get("imbalance", 0.0) or 0.0
        ask_pressure = orderbook_info.get("ask_pressure", "neutral")
        bid_strength = orderbook_info.get("bid_strength", "neutral")
        
        # 📊 TraderWeightedDecisionEngine - Weighted Feature Scoring
        trend_features = {}
        
        # 1. Green Ratio & Trend Strength (30%)
        if market_context == "impulse":
            green_ratio = 0.8 if buy_pressure else 0.4
        elif market_context == "breakout":
            green_ratio = 0.7 if buy_pressure else 0.3
        elif market_context == "pullback":
            green_ratio = 0.6 if buy_pressure else 0.2
        else:
            green_ratio = 0.3
        trend_features["green_ratio"] = green_ratio * 0.3
        
        # 2. Higher Highs & Pattern Strength (20%)
        if pattern in ["momentum_building", "absorption_bounce"]:
            higher_highs = 0.8
        elif pattern == "absorbing_dip":
            higher_highs = 0.7
        elif pattern == "consolidation":
            higher_highs = 0.5
        elif pattern == "reversal":
            higher_highs = 0.6
        else:
            higher_highs = 0.3
        trend_features["higher_highs_score"] = higher_highs * 0.2
        
        # 3. Pullback Quality (20%)
        if market_context == "pullback":
            if buy_pressure and momentum == "building":
                pullback_strength = 0.9
            elif buy_pressure:
                pullback_strength = 0.7
            elif momentum == "building":
                pullback_strength = 0.6
            else:
                pullback_strength = 0.3
        elif market_context == "impulse":
            pullback_strength = 0.4  # Still some value in impulse
        else:
            pullback_strength = 0.2
        trend_features["pullback_quality"] = pullback_strength * 0.2
        
        # 4. Bid Wall Strength (15%)
        if bid_strength == "strong_support" and bids_layered:
            bid_wall_score = 0.9
        elif bids_layered:
            bid_wall_score = 0.7
        elif bid_strength == "strong_support":
            bid_wall_score = 0.6
        elif bid_strength == "decent_support":
            bid_wall_score = 0.4
        elif not spoofing_suspected:
            bid_wall_score = 0.3
        else:
            bid_wall_score = 0.1
        trend_features["bid_wall_strength"] = bid_wall_score * 0.15
        
        # 5. Volume Decline Bonus (10%)
        if volume_behavior == "declining" and market_context == "pullback":
            volume_score = 0.1  # Bonus for healthy pullback
        elif volume_behavior == "increasing" and market_context in ["impulse", "breakout"]:
            volume_score = 0.05  # Slight bonus for momentum
        elif volume_behavior == "increasing" and market_context == "pullback":
            volume_score = -0.05  # Distribution concern
        else:
            volume_score = 0.0
        trend_features["volume_decline_bonus"] = volume_score
        
        # 6. Time of Day Boost (5%)
        utc_hour = datetime.now(timezone.utc).hour
        if 13 <= utc_hour <= 16:  # London/NY overlap
            time_score = 0.85
        elif 8 <= utc_hour <= 12:   # London session
            time_score = 0.70
        elif 17 <= utc_hour <= 21:  # NY session
            time_score = 0.65
        else:
            time_score = 0.40
        trend_features["time_of_day_boost"] = time_score * 0.05
        
        # 7. Wick Analysis Bonus (5%)
        if wick_analysis == "bullish_hammer":
            wick_bonus = 0.08
        elif wick_analysis == "rejection":
            wick_bonus = 0.05
        elif wick_analysis == "doji_indecision":
            wick_bonus = -0.03
        elif wick_analysis == "bearish_engulfing":
            wick_bonus = -0.05
        else:
            wick_bonus = 0.0
        trend_features["wick_analysis_bonus"] = wick_bonus
        
        # 8. Orderbook Imbalance (5%)
        if imbalance > 0.2:  # Strong bid imbalance
            imbalance_score = 0.05
        elif imbalance > 0.1:  # Moderate bid imbalance
            imbalance_score = 0.03
        elif imbalance < -0.2:  # Strong ask imbalance
            imbalance_score = -0.03
        else:
            imbalance_score = 0.0
        trend_features["orderbook_imbalance"] = imbalance_score
        
        # 📈 Final Score Calculation - Pure Weighted Sum
        final_score = sum(trend_features.values())
        final_score = max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
        
        # 🎯 Decision Logic - Weighted Thresholds (NO if/then complexity)
        if final_score >= 0.75:
            decision = "join_trend"
            confidence = min(0.95, final_score + 0.15)
            quality_grade = "excellent" if final_score >= 0.85 else "strong"
        elif final_score >= 0.55:
            decision = "consider_entry"
            confidence = final_score * 0.85
            quality_grade = "good" if final_score >= 0.65 else "neutral-watch"
        else:
            decision = "avoid"
            confidence = max(0.1, 1.0 - final_score)
            quality_grade = "weak" if final_score >= 0.3 else "very_poor"
        
        # 🔍 Build Comprehensive Reasons (feature-based)
        reasons = []
        
        # Primary decision reasons
        if final_score >= 0.75:
            reasons.append(f"High-quality weighted setup - score {final_score:.3f}")
        elif final_score >= 0.55:
            reasons.append(f"Moderate setup - weighted score {final_score:.3f}")
        else:
            reasons.append(f"Weak setup - weighted score {final_score:.3f} below threshold")
        
        # Feature-specific reasons (top contributors)
        feature_contributions = [(k, v) for k, v in trend_features.items() if v > 0.05]
        feature_contributions.sort(key=lambda x: x[1], reverse=True)
        
        for feature_name, contribution in feature_contributions[:3]:
            if contribution > 0.1:
                reasons.append(f"Strong {feature_name.replace('_', ' ')} ({contribution:.3f})")
            elif contribution > 0.05:
                reasons.append(f"Good {feature_name.replace('_', ' ')} ({contribution:.3f})")
        
        # Risk factors (negative contributors)
        risk_factors = [(k, v) for k, v in trend_features.items() if v < -0.02]
        for risk_name, risk_value in risk_factors:
            reasons.append(f"⚠️ {risk_name.replace('_', ' ')} concern ({risk_value:.3f})")
        
        # Context-specific insights
        if market_context == "impulse" and final_score < 0.5:
            reasons.append("Impulse lacks weighted conviction")
        if market_context == "pullback" and final_score >= 0.7:
            reasons.append("High-quality pullback with strong weighted components")
        if spoofing_suspected:
            reasons.append("⚠️ Orderbook spoofing detected in analysis")
        
        result = {
            "decision": decision,
            "confidence": confidence,
            "final_score": final_score,
            "reasons": reasons,
            "quality_grade": quality_grade,
            "score_breakdown": trend_features,
            "weights_used": {
                "green_ratio": 0.3,
                "higher_highs": 0.2,
                "pullback_quality": 0.2,
                "bid_wall_strength": 0.15,
                "volume_bonus": 0.1,
                "time_boost": 0.05,
                "wick_bonus": 0.05,
                "imbalance_bonus": 0.05
            },
            "context_adjustment": f"TraderWeightedDecisionEngine optimized for {market_context}"
        }
        
        # Enhanced logging with weighted breakdown
        print(f"[TRADER WEIGHTED] {symbol}: {decision.upper()} | Score: {final_score:.3f} | Confidence: {confidence:.3f} | Grade: {quality_grade}")
        
        # Score breakdown in terminal
        breakdown_items = [f"{k}:{v:+.3f}" for k, v in trend_features.items() if abs(v) > 0.005]
        if breakdown_items:
            print(f"[WEIGHTED BREAKDOWN] {symbol}: {' | '.join(breakdown_items[:5])}")
        
        if reasons:
            print(f"[WEIGHTED REASONS] {symbol}: {', '.join(reasons[:3])}")
        
        # Enhanced logging to file
        _log_trader_decision_weighted(symbol, result, market_context, candle_behavior, orderbook_info, trend_features)
        
        return result
        
    except Exception as e:
        print(f"❌ [TRADER WEIGHTED ERROR] {symbol}: {e}")
        return {
            "decision": "avoid",
            "confidence": 0.0,
            "final_score": 0.0,
            "reasons": [f"TraderWeightedDecisionEngine error: {e}"],
            "quality_grade": "error",
            "score_breakdown": {}
        }


def _calculate_advanced_weighted_score(
    symbol: str,
    market_context: str,
    candle_behavior: Dict,
    orderbook_info: Dict,
    phase_analysis: Dict,
    liquidity_analysis: Dict,
    psychology_analysis: Dict,
    htf_analysis: Dict
) -> Dict:
    """Calculate advanced weighted score with professional trader components"""
    try:
        weighted_features = {}
        
        # 1. Trend Strength (25%) - Basic market trend
        if market_context == "impulse":
            trend_score = 0.85
        elif market_context == "breakout":
            trend_score = 0.8
        elif market_context == "pullback":
            trend_score = 0.7
        else:
            trend_score = 0.4
        weighted_features["trend_strength"] = trend_score * 0.25
        
        # 2. Pullback Quality (20%) - Enhanced with phase analysis
        base_pullback = 0.5
        if candle_behavior.get("shows_buy_pressure", False):
            base_pullback += 0.2
        if phase_analysis.get("market_phase") == "retest-confirmation":
            base_pullback += 0.2
        elif phase_analysis.get("market_phase") == "pre-breakout":
            base_pullback += 0.1
        weighted_features["pullback_quality"] = min(1.0, base_pullback) * 0.2
        
        # 3. Support Reaction (15%) - Orderbook + structure
        support_score = 0.5
        if orderbook_info.get("bids_layered", False):
            support_score += 0.2
        if orderbook_info.get("bid_strength") == "strong_support":
            support_score += 0.2
        if phase_analysis.get("confidence", 0) > 0.7:
            support_score += 0.1
        weighted_features["support_reaction"] = min(1.0, support_score) * 0.15
        
        # 4. Liquidity Pattern Score (10%) - New component
        liquidity_score = liquidity_analysis.get("liquidity_pattern_score", 0.5)
        weighted_features["liquidity_pattern_score"] = liquidity_score * 0.1
        
        # 5. Psychology Score (10%) - Clean move vs manipulation
        psych_score = psychology_analysis.get("psych_score", 0.7)
        weighted_features["psych_score"] = psych_score * 0.1
        
        # 6. HTF Supportive Score (10%) - Higher timeframe confirmation
        htf_score = htf_analysis.get("htf_supportive_score", 0.5)
        weighted_features["htf_supportive_score"] = htf_score * 0.1
        
        # 7. Market Phase Modifier (10%) - Phase-specific boost
        phase_score = phase_analysis.get("phase_score", 0.5)
        phase_modifier = 1.0
        phase_name = phase_analysis.get("market_phase", "undefined")
        
        if phase_name == "breakout-continuation":
            phase_modifier = 1.2  # 20% boost
        elif phase_name == "pre-breakout":
            phase_modifier = 1.15  # 15% boost
        elif phase_name == "retest-confirmation":
            phase_modifier = 1.1   # 10% boost
        elif phase_name == "exhaustion-pullback":
            phase_modifier = 0.8   # 20% penalty
        
        weighted_features["market_phase_modifier"] = (phase_score * phase_modifier - phase_score) * 0.1
        
        # Calculate final score
        final_score = sum(weighted_features.values())
        final_score = max(0.0, min(1.0, final_score))
        
        return {
            "final_score": final_score,
            "weighted_features": weighted_features,
            "phase_modifier": phase_modifier,
            "component_count": len(weighted_features)
        }
        
    except Exception as e:
        print(f"❌ Advanced scoring error for {symbol}: {e}")
        return {"final_score": 0.5, "weighted_features": {}, "phase_modifier": 1.0}


def _make_advanced_trader_decision(advanced_result: Dict, symbol: str) -> Dict:
    """Make final decision based on advanced weighted scoring"""
    try:
        final_score = advanced_result.get("final_score", 0.5)
        weighted_features = advanced_result.get("weighted_features", {})
        
        # Enhanced decision thresholds
        if final_score >= 0.8:
            decision = "join_trend"
            confidence = min(0.95, final_score + 0.1)
            quality_grade = "excellent"
        elif final_score >= 0.7:
            decision = "join_trend"
            confidence = final_score * 0.9
            quality_grade = "strong"
        elif final_score >= 0.6:
            decision = "consider_entry"
            confidence = final_score * 0.8
            quality_grade = "good"
        elif final_score >= 0.45:
            decision = "consider_entry"
            confidence = final_score * 0.7
            quality_grade = "neutral-watch"
        else:
            decision = "avoid"
            confidence = max(0.1, 1.0 - final_score)
            quality_grade = "weak" if final_score >= 0.3 else "very_poor"
        
        # Build comprehensive reasons
        reasons = []
        
        # Primary decision reason
        if final_score >= 0.7:
            reasons.append(f"High-quality professional setup - advanced score {final_score:.3f}")
        elif final_score >= 0.6:
            reasons.append(f"Good setup with professional validation - score {final_score:.3f}")
        elif final_score >= 0.45:
            reasons.append(f"Moderate setup requiring careful entry - score {final_score:.3f}")
        else:
            reasons.append(f"Weak setup - advanced score {final_score:.3f} below threshold")
        
        # Feature-specific reasons (top contributors)
        sorted_features = sorted(weighted_features.items(), key=lambda x: x[1], reverse=True)
        for feature_name, contribution in sorted_features[:3]:
            if contribution > 0.08:  # Significant contribution
                clean_name = feature_name.replace('_', ' ').title()
                reasons.append(f"Strong {clean_name} ({contribution:.3f})")
        
        # Risk assessment
        psych_score = weighted_features.get("psych_score", 0.07)
        if psych_score < 0.05:  # Low psychology score indicates manipulation
            reasons.append("⚠️ Potential market manipulation detected")
        
        htf_score = weighted_features.get("htf_supportive_score", 0.05)
        if htf_score < 0.03:  # HTF not supportive
            reasons.append("⚠️ Higher timeframe not supportive")
        
        return {
            "decision": decision,
            "confidence": confidence,
            "final_score": final_score,
            "reasons": reasons,
            "quality_grade": quality_grade,
            "score_breakdown": weighted_features,
            "advanced_analysis": True
        }
        
    except Exception as e:
        print(f"❌ Advanced decision error for {symbol}: {e}")
        return {
            "decision": "avoid",
            "confidence": 0.0,
            "final_score": 0.0,
            "reasons": ["Advanced decision error"],
            "quality_grade": "error"
        }


def _log_advanced_trader_decision(symbol: str, result: Dict):
    """Enhanced logging for advanced trader decisions"""
    try:
        decision = result.get("decision", "unknown")
        confidence = result.get("confidence", 0.0)
        final_score = result.get("final_score", 0.0)
        grade = result.get("quality_grade", "unknown")
        
        # Enhanced terminal output
        print(f"[ADVANCED TRADER] {symbol}: {decision.upper()} | Score: {final_score:.3f} | Confidence: {confidence:.3f} | Grade: {grade}")
        
        # Component breakdown
        phase_analysis = result.get("phase_analysis", {})
        psychology_analysis = result.get("psychology_analysis", {})
        htf_analysis = result.get("htf_analysis", {})
        
        market_phase = phase_analysis.get("market_phase", "unknown")
        psych_flags = psychology_analysis.get("psychological_flags", [])
        htf_match = htf_analysis.get("htf_trend_match", False)
        
        print(f"[TREND DEBUG] {symbol}: market_phase={market_phase} | psych_flags={len(psych_flags)} | htf_match={htf_match}")
        
        # Score breakdown
        advanced_features = result.get("advanced_features", {})
        if advanced_features:
            breakdown_str = " | ".join([f"{k}:{v:+.3f}" for k, v in advanced_features.items() if abs(v) > 0.01])
            print(f"[ADVANCED BREAKDOWN] {symbol}: {breakdown_str}")
        
        # Comprehensive log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "engine_version": "AdvancedTraderWeightedDecisionEngine",
            "decision": decision,
            "final_score": final_score,
            "confidence": confidence,
            "quality_grade": grade,
            "advanced_features": advanced_features,
            "market_context": result.get("market_context", "unknown"),
            "phase_analysis": phase_analysis,
            "liquidity_analysis": result.get("liquidity_analysis", {}),
            "psychology_analysis": psychology_analysis,
            "htf_analysis": htf_analysis,
            "reasons": result.get("reasons", [])
        }
        
        # Save to advanced log
        os.makedirs("logs", exist_ok=True)
        with open("logs/advanced_trader_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(log_entry, ensure_ascii=False)}\n")
            
    except Exception as e:
        print(f"❌ Advanced log error for {symbol}: {e}")


def describe_setup_naturally(
    symbol: str,
    market_context: str,
    candle_behavior: Dict,
    orderbook_info: Dict,
    decision_result: Dict
) -> str:
    """
    🗣️ Etap 5: Naturalny opis setup'u
    
    Tworzy tekstowy opis sytuacji jak trader wyjaśniający setup:
    "Rynek utrzymuje trend wzrostowy z lekką korektą, wsparcie broni EMA21, widoczne warstwy bidów."
    
    Args:
        symbol: Trading symbol
        market_context: Market structure
        candle_behavior: Candle analysis result
        orderbook_info: Orderbook analysis result
        decision_result: Final decision
        
    Returns:
        str: Natural language description
    """
    try:
        parts = []
        
        # 1. Market context description
        context_map = {
            "impulse": "Rynek pokazuje silny impuls wzrostowy",
            "pullback": "Rynek koryguje się w ramach trendu wzrostowego",
            "breakout": "Cena przebija się powyżej kluczowego oporu",
            "range": "Rynek konsoliduje się w wąskim zakresie",
            "distribution": "Widoczna dystrybucja przy wysokich poziomach",
            "uncertain": "Struktura rynku pozostaje niejednoznaczna"
        }
        
        parts.append(context_map.get(market_context, f"Struktura rynku: {market_context}"))
        
        # 2. Candle behavior description
        pattern = candle_behavior.get("pattern", "")
        momentum = candle_behavior.get("momentum", "neutral")
        
        if pattern == "momentum_building":
            parts.append("momentum buduje się z każdą świecą")
        elif pattern == "absorption_bounce":
            parts.append("widoczne absorpcja słabości i odbicie")
        elif pattern == "absorbing_dip":
            parts.append("kupujący absorbują spadek")
        elif candle_behavior.get("shows_buy_pressure", False):
            parts.append("świece pokazują presję kupujących")
        
        # 3. Orderbook insights
        if orderbook_info.get("data_available", False):
            if orderbook_info.get("bids_layered", False):
                parts.append("warstwy bidów zapewniają wsparcie")
            
            bid_strength = orderbook_info.get("bid_strength", "")
            ask_pressure = orderbook_info.get("ask_pressure", "")
            
            if bid_strength == "strong_support":
                parts.append("silne wsparcie w orderbuku")
            elif ask_pressure == "light":
                parts.append("lekka presja sprzedaży")
        
        # 4. Decision reasoning
        decision = decision_result.get("decision", "wait")
        quality = decision_result.get("quality_grade", "medium")
        
        if decision == "join_trend":
            if quality == "premium":
                parts.append("Doskonały moment na wejście")
            elif quality == "high":
                parts.append("Dobry setup do wejścia")
            else:
                parts.append("Możliwość wejścia przy zarządzaniu ryzykiem")
        elif decision == "wait":
            parts.append("Lepiej poczekać na lepszy moment")
        else:
            parts.append("Brak wyraźnej przewagi rynkowej")
        
        # 5. Combine into natural sentence
        description = ". ".join(parts) + "."
        
        # Capitalize first letter
        description = description[0].upper() + description[1:] if description else ""
        
        return description
        
    except Exception as e:
        print(f"[TRADER ERROR] {symbol} - describe_setup_naturally failed: {e}")
        return f"Analiza {symbol}: {decision_result.get('decision', 'wait')} - score {decision_result.get('final_score', 0):.2f}"


def _log_trader_score(symbol: str, scoring_result: Dict, features: Dict):
    """Log detailed trader scoring for analysis"""
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "scoring_result": scoring_result,
            "input_features": features
        }
        
        log_file = "trader_score_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    except Exception as e:
        print(f"[TRADER ERROR] Failed to log score for {symbol}: {e}")


def _log_comprehensive_debug(
    symbol: str,
    decision_result: Dict,
    market_context: str,
    candle_behavior: Dict,
    orderbook_info: Dict,
    scoring_result: Dict,
    features: Dict
):
    """Log comprehensive debug information per analysis"""
    try:
        os.makedirs("logs", exist_ok=True)
        
        # Create comprehensive debug entry
        debug_entry = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_context": market_context,
            "trend_strength": features.get("trend_strength", 0.0),
            "pullback": {
                "quality": features.get("pullback_quality", 0.0),
                "active": candle_behavior.get("pattern", "") in ["absorbing_dip", "absorption_bounce"],
                "volume_behavior": candle_behavior.get("volume_behavior", "unknown")
            },
            "support_reaction": {
                "strength": features.get("support_reaction_strength", 0.0),
                "type": "support" if features.get("support_reaction_strength", 0) > 0.6 else "weak",
                "bounce_confirmation": features.get("bounce_confirmation_strength", 0.0)
            },
            "candle_behavior": {
                "pattern": candle_behavior.get("pattern", "none"),
                "shows_buy_pressure": candle_behavior.get("shows_buy_pressure", False),
                "momentum": candle_behavior.get("momentum", "neutral"),
                "volume_increase": candle_behavior.get("volume_behavior", "") == "buying_volume_increase",
                "wick_analysis": candle_behavior.get("wick_analysis", "none")
            },
            "orderbook_analysis": {
                "bids_layered": orderbook_info.get("bids_layered", False),
                "spoofing_suspected": orderbook_info.get("spoofing_suspected", False),
                "bid_strength": orderbook_info.get("bid_strength", "unknown"),
                "ask_pressure": orderbook_info.get("ask_pressure", "unknown"),
                "imbalance": orderbook_info.get("imbalance", 0.0),
                "data_available": orderbook_info.get("data_available", False)
            },
            "scoring_breakdown": scoring_result.get("score_breakdown", {}),
            "weights_used": scoring_result.get("weights_used", {}),
            "context_adjustment": scoring_result.get("context_adjustment", "unknown"),
            "final_score": decision_result.get("final_score", 0.0),
            "confidence": decision_result.get("confidence", 0.0),
            "quality_grade": decision_result.get("quality_grade", "unknown"),
            "decision": decision_result.get("decision", "unknown"),
            "reasons": decision_result.get("reasons", []),
            "red_flags": decision_result.get("red_flags", [])
        }
        
        # Write to comprehensive debug log
        debug_file = "logs/trader_debug_log.txt"
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(debug_entry) + "\n")
            
        # Log alerts separately if decision is join_trend
        if decision_result.get("decision") == "join_trend" and decision_result.get("final_score", 0) >= 0.75:
            alert_entry = {
                "symbol": symbol,
                "score": decision_result.get("final_score", 0.0),
                "confidence": decision_result.get("confidence", 0.0),
                "grade": decision_result.get("quality_grade", "unknown"),
                "time": datetime.now(timezone.utc).isoformat(),
                "reason_summary": ", ".join(decision_result.get("reasons", [])[:4]),
                "context": market_context,
                "scoring_breakdown": scoring_result.get("score_breakdown", {})
            }
            
            alerts_file = "logs/alerted_symbols_log.txt"
            with open(alerts_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_entry) + "\n")
                
            print(f"[ALERT LOG] {symbol}: High-quality setup logged (score={decision_result.get('final_score', 0):.3f})")
            
    except Exception as e:
        print(f"[DEBUG ERROR] Failed to log comprehensive debug for {symbol}: {e}")


def _log_trader_decision(
    symbol: str,
    decision_result: Dict,
    market_context: str,
    candle_behavior: Dict,
    orderbook_info: Dict
):
    """Log detailed trader decision for analysis"""
    try:
        # Create JSON-serializable copy
        safe_candle_behavior = {}
        for key, value in candle_behavior.items():
            if isinstance(value, (bool, str, int, float)):
                safe_candle_behavior[key] = value
            else:
                safe_candle_behavior[key] = str(value)
        
        safe_orderbook_info = {}
        for key, value in orderbook_info.items():
            if isinstance(value, (bool, str, int, float)):
                safe_orderbook_info[key] = value
            else:
                safe_orderbook_info[key] = str(value)
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "decision": decision_result,
            "market_context": market_context,
            "candle_behavior": safe_candle_behavior,
            "orderbook_info": safe_orderbook_info
        }
        
        log_file = "trader_decision_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    except Exception as e:
        print(f"[TRADER ERROR] Failed to log decision for {symbol}: {e}")


# === HELPER FUNCTIONS ===

def _calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    if not prices or len(prices) < period:
        return []
    
    try:
        ema = []
        multiplier = 2 / (period + 1)
        
        # First value is SMA
        sma = sum(prices[:period]) / period
        ema.append(sma)
        
        # Rest is EMA
        for price in prices[period:]:
            if price is not None and isinstance(price, (int, float)):
                ema_value = (price * multiplier) + (ema[-1] * (1 - multiplier))
                ema.append(ema_value)
        
        return ema
    except:
        return []


def _calculate_slope(values: List[float]) -> float:
    """Calculate slope of values"""
    if not values or len(values) < 2:
        return 0.0
    
    try:
        # Filter out None values
        clean_values = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
        
        if len(clean_values) < 2:
            return 0.0
            
        n = len(clean_values)
        x = list(range(n))
        
        # Linear regression slope
        x_mean = sum(x) / n
        y_mean = sum(clean_values) / n
        
        numerator = sum((x[i] - x_mean) * (clean_values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    except:
        return 0.0


def _calculate_atr(highs: List[float], lows: List[float], closes: List[float]) -> float:
    """Calculate Average True Range"""
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return 0.0
    
    try:
        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        return np.mean(true_ranges) if true_ranges else 0.0
    except:
        return 0.0


# === MAIN INTEGRATION FUNCTION ===

def analyze_symbol_with_trader_ai(
    symbol: str,
    candles: List[List],
    market_data: Dict = None,
    enable_description: bool = False
) -> Dict:
    """
    🚀 Główna funkcja integracyjna dla Trader AI Engine
    
    Przeprowadza pełną analizę symbolu przez AI tradera
    
    Args:
        symbol: Trading symbol
        candles: Lista OHLCV candles
        market_data: Optional market data with orderbook
        enable_description: Whether to generate natural description
        
    Returns:
        dict: Complete trader AI analysis
    """
    if not candles or len(candles) < 10:
        return {
            "decision": "avoid",
            "confidence": 0.0,
            "final_score": 0.0,
            "reasons": ["insufficient_data"],
            "quality_grade": "low",
            "analysis_complete": False
        }
    
    try:
        print(f"[TRADER AI] Starting analysis for {symbol}...")
        
        # Step 1: Analyze market structure
        market_context = analyze_market_structure(candles, symbol)
        
        # Step 2: Analyze candle behavior  
        candle_behavior = analyze_candle_behavior(candles, symbol)
        
        # Step 3: Interpret orderbook
        orderbook_info = interpret_orderbook(symbol, market_data)
        
        # Step 4: Make trader decision
        current_price = float(candles[-1][4]) if candles else None
        decision_result = simulate_trader_decision(
            symbol, market_context, candle_behavior, orderbook_info, current_price
        )
        
        # Step 5: Generate description if requested
        description = ""
        if enable_description:
            description = describe_setup_naturally(
                symbol, market_context, candle_behavior, orderbook_info, decision_result
            )
        
        # Combine all results
        complete_analysis = {
            **decision_result,
            "market_context": market_context,
            "candle_behavior": candle_behavior,
            "orderbook_info": orderbook_info,
            "description": description,
            "analysis_complete": True,
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"[TRADER SUMMARY] {symbol}: Analysis complete - {decision_result['decision']} ({decision_result['quality_grade']})")
        
        # Additional debug summary
        if decision_result.get('final_score', 0) >= 0.6:
            print(f"[TRADER QUALITY] {symbol}: Good setup detected - monitoring for improvements")
        elif decision_result.get('final_score', 0) >= 0.4:
            print(f"[TRADER QUALITY] {symbol}: Decent setup - waiting for better confluence")
        else:
            print(f"[TRADER QUALITY] {symbol}: Weak setup - avoiding until conditions improve")
        
        return complete_analysis
        
    except Exception as e:
        print(f"[TRADER ERROR] {symbol} - analyze_symbol_with_trader_ai failed: {e}")
        return {
            "decision": "avoid",
            "confidence": 0.0,
            "final_score": 0.0,
            "reasons": ["analysis_error"],
            "quality_grade": "low",
            "analysis_complete": False,
            "error": str(e)
        }