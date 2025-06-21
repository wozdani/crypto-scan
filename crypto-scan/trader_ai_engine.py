"""
Trader AI Engine - Intelligent Market Analysis System

Symuluje myślenie doświadczonego tradera zamiast sztywnych reguł.
Analizuje kontekst rynkowy, zachowanie świec i orderbook jak prawdziwy trader.
"""

import os
import json
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Import existing utilities
try:
    from utils.bybit_orderbook import get_orderbook_snapshot
except ImportError:
    get_orderbook_snapshot = None


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
        
        # Debug logging
        if symbol:
            print(f"[TRADER AI] {symbol}: Market structure = {context}")
            print(f"[TRADER AI] {symbol}: price_slope={price_slope:.6f}, ema_slope={ema_slope:.6f}")
            print(f"[TRADER AI] {symbol}: price_vs_ema={price_vs_ema:.2f}%, volatility_ratio={volatility_ratio:.2f}")
            print(f"[TRADER AI] {symbol}: volume_ratio={volume_ratio:.2f}, range_size={range_size:.2f}%")
        
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
        
        # Debug logging
        if symbol:
            print(f"[TRADER AI] {symbol}: Candle behavior = {pattern}")
            print(f"[TRADER AI] {symbol}: buy_pressure={shows_buy_pressure}, momentum={momentum}")
            print(f"[TRADER AI] {symbol}: volume_behavior={volume_behavior}, wick_analysis={wick_analysis}")
        
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
        
        print(f"[TRADER AI] {symbol}: Orderbook = {bid_strength}, ask_pressure={ask_pressure}")
        print(f"[TRADER AI] {symbol}: imbalance={imbalance:.3f}, bids_layered={bids_layered}")
        
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


def simulate_trader_decision(
    symbol: str,
    market_context: str,
    candle_behavior: Dict,
    orderbook_info: Dict,
    current_price: float = None
) -> Dict:
    """
    🧠 Etap 4: Główna funkcja decyzyjna tradera
    
    Scala wszystkie dane i podejmuje decyzję jak doświadczony trader:
    - Nie bazuje na checklistach, ale na intuicji i doświadczeniu
    - Szuka edge'u w rynku
    - Jeśli nie ma przewagi, milczy
    
    Args:
        symbol: Trading symbol
        market_context: Wynik z analyze_market_structure()
        candle_behavior: Wynik z analyze_candle_behavior()
        orderbook_info: Wynik z interpret_orderbook()
        current_price: Current price for context
        
    Returns:
        dict: {
            "decision": str,        # "join_trend", "wait", "avoid"
            "confidence": float,    # 0.0-1.0
            "final_score": float,   # 0.0-1.0
            "reasons": List[str],   # Why this decision
            "quality_grade": str    # "low", "medium", "high", "premium"
        }
    """
    try:
        reasons = []
        confidence_factors = []
        red_flags = []
        
        # === TRADER THINKING PROCESS ===
        
        # 1. Context Assessment - What's the big picture?
        context_score = 0.0
        
        if market_context == "pullback":
            context_score = 0.7
            reasons.append("pullback_in_trend")
            confidence_factors.append(0.15)
        elif market_context == "impulse":
            context_score = 0.6
            reasons.append("strong_impulse")
            confidence_factors.append(0.1)
        elif market_context == "breakout":
            context_score = 0.8
            reasons.append("breakout_detected")
            confidence_factors.append(0.2)
        elif market_context in ["range", "distribution"]:
            context_score = 0.2
            red_flags.append("choppy_conditions")
        elif market_context == "uncertain":
            context_score = 0.3
            red_flags.append("unclear_structure")
        
        # 2. Candle Behavior - What's price telling us?
        candle_score = 0.0
        
        if candle_behavior.get("shows_buy_pressure", False):
            candle_score += 0.3
            reasons.append("buy_pressure_visible")
            confidence_factors.append(0.1)
        
        pattern = candle_behavior.get("pattern", "")
        if pattern in ["momentum_building", "absorption_bounce"]:
            candle_score += 0.3
            reasons.append(f"pattern_{pattern}")
            confidence_factors.append(0.15)
        elif pattern == "absorbing_dip":
            candle_score += 0.2
            reasons.append("absorbing_weakness")
            confidence_factors.append(0.1)
        
        momentum = candle_behavior.get("momentum", "neutral")
        if momentum == "building":
            candle_score += 0.2
            confidence_factors.append(0.1)
        elif momentum == "negative":
            candle_score -= 0.2
            red_flags.append("negative_momentum")
        
        # 3. Orderbook Intelligence - What's the institutional view?
        orderbook_score = 0.0
        
        if orderbook_info.get("data_available", False):
            if orderbook_info.get("bids_layered", False):
                orderbook_score += 0.25
                reasons.append("bid_layering")
                confidence_factors.append(0.1)
            
            bid_strength = orderbook_info.get("bid_strength", "unknown")
            if bid_strength == "strong_support":
                orderbook_score += 0.25
                reasons.append("strong_bid_support")
                confidence_factors.append(0.15)
            elif bid_strength == "weak":
                orderbook_score -= 0.15
                red_flags.append("weak_bids")
            
            ask_pressure = orderbook_info.get("ask_pressure", "unknown")
            if ask_pressure == "light":
                orderbook_score += 0.15
                reasons.append("light_ask_pressure")
                confidence_factors.append(0.05)
            elif ask_pressure == "heavy":
                orderbook_score -= 0.15
                red_flags.append("heavy_asks")
            
            if orderbook_info.get("spoofing_suspected", False):
                orderbook_score -= 0.2
                red_flags.append("spoofing_detected")
        
        # 4. Trader's Final Assessment
        base_score = (context_score + candle_score + orderbook_score) / 3.0
        confidence = sum(confidence_factors)
        
        # Apply red flags penalty
        red_flag_penalty = len(red_flags) * 0.1
        final_score = max(0.0, base_score - red_flag_penalty)
        
        # === DECISION LOGIC (jak myśli trader) ===
        
        if final_score >= 0.75 and confidence >= 0.3 and len(red_flags) == 0:
            decision = "join_trend"
            quality_grade = "premium"
        elif final_score >= 0.65 and confidence >= 0.2 and len(red_flags) <= 1:
            decision = "join_trend"
            quality_grade = "high"
        elif final_score >= 0.5 and confidence >= 0.15:
            decision = "join_trend"
            quality_grade = "medium"
        elif final_score >= 0.35:
            decision = "wait"
            quality_grade = "low"
            reasons.append("insufficient_edge")
        else:
            decision = "avoid"
            quality_grade = "low"
            reasons.append("no_clear_edge")
        
        # Add red flags to reasons for transparency
        if red_flags:
            reasons.extend([f"concern_{flag}" for flag in red_flags])
        
        result = {
            "decision": decision,
            "confidence": round(confidence, 3),
            "final_score": round(final_score, 3),
            "reasons": reasons,
            "quality_grade": quality_grade,
            "context_score": round(context_score, 3),
            "candle_score": round(candle_score, 3),
            "orderbook_score": round(orderbook_score, 3),
            "red_flags": red_flags
        }
        
        # Enhanced logging
        print(f"[TRADER AI] {symbol}: DECISION = {decision} (score={final_score:.3f}, confidence={confidence:.3f})")
        print(f"[TRADER AI] {symbol}: Quality = {quality_grade}, Reasons = {', '.join(reasons[:3])}")
        if red_flags:
            print(f"[TRADER AI] {symbol}: Red flags = {', '.join(red_flags)}")
        
        # Log to file for analysis
        _log_trader_decision(symbol, result, market_context, candle_behavior, orderbook_info)
        
        return result
        
    except Exception as e:
        print(f"[TRADER ERROR] {symbol} - simulate_trader_decision failed: {e}")
        return {
            "decision": "avoid",
            "confidence": 0.0,
            "final_score": 0.0,
            "reasons": ["analysis_error"],
            "quality_grade": "low"
        }


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


def _log_trader_decision(
    symbol: str,
    decision_result: Dict,
    market_context: str,
    candle_behavior: Dict,
    orderbook_info: Dict
):
    """Log detailed trader decision for analysis"""
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "decision": decision_result,
            "market_context": market_context,
            "candle_behavior": candle_behavior,
            "orderbook_info": orderbook_info
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
        
        print(f"[TRADER AI] {symbol}: Analysis complete - {decision_result['decision']} ({decision_result['quality_grade']})")
        
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