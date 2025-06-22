#!/usr/bin/env python3
"""
Feature Extractor for New Adaptive Trader Decision Engine

Zbiera wszystkie cechy potrzebne dla simulate_trader_decision_advanced()
zastępując stary compute_trader_score() system.
"""

import numpy as np
from typing import List, Dict, Any


def extract_all_features_for_token(symbol: str, candles: List[List] = None, market_data: Dict = None) -> Dict[str, Any]:
    """
    Centralny ekstractor cech dla nowego systemu decyzyjnego
    
    Args:
        symbol: Trading symbol
        candles: Lista OHLCV candles
        market_data: Optional market data
        
    Returns:
        dict: Wszystkie cechy potrzebne dla simulate_trader_decision_advanced()
    """
    try:
        from trader_ai_engine import analyze_market_structure, analyze_candle_behavior, interpret_orderbook
        
        # === BASIC MARKET ANALYSIS ===
        if candles and len(candles) >= 10:
            market_context = analyze_market_structure(candles, symbol)
            candle_behavior = analyze_candle_behavior(candles, symbol)
            orderbook_info = interpret_orderbook(symbol, market_data)
        else:
            # Fallback for insufficient data
            market_context = "unknown"
            candle_behavior = {"shows_buy_pressure": False, "pattern": "neutral"}
            orderbook_info = {"bids_layered": False, "imbalance": 0.0}
        
        # === FEATURE EXTRACTION ===
        features = {
            # Core Features
            "trend_strength": _calculate_trend_strength(candles) if candles else 0.0,
            "pullback_quality": _analyze_pullback_quality(candles, market_context) if candles else 0.0,
            "support_reaction": _measure_support_reaction(candles, orderbook_info) if candles else 0.0,
            "liquidity_pattern_score": _score_liquidity_patterns(candles, market_data) if candles else 0.0,
            "psych_score": _detect_market_psychology(candles, candle_behavior) if candles else 0.5,
            "htf_supportive_score": _evaluate_htf_support(candles, symbol) if candles else 0.0,
            "market_phase_modifier": _get_phase_modifier(candles, market_context) if candles else 0.0,
            
            # Context Features
            "market_phase": _detect_market_phase(candles, market_context) if candles else "unknown",
            "price_action_pattern": _identify_price_action_pattern(candles, candle_behavior) if candles else "none",
            "volume_behavior": _analyze_volume_behavior(candles) if candles else "neutral",
            "htf_trend_match": _check_htf_trend_match(candles, market_context) if candles else False,
            
            # Meta
            "symbol": symbol,
            "candle_count": len(candles) if candles else 0,
            "data_quality": "good" if candles and len(candles) >= 20 else "limited"
        }
        
        print(f"[FEATURE EXTRACTOR] {symbol}: Extracted {len(features)} features, phase={features['market_phase']}")
        
        return features
        
    except Exception as e:
        print(f"❌ [FEATURE EXTRACTOR ERROR] {symbol}: {e}")
        return _get_fallback_features(symbol)


def _calculate_trend_strength(candles: List[List]) -> float:
    """Oblicz siłę trendu na podstawie nachylenia i konsystencji"""
    if not candles or len(candles) < 10:
        return 0.0
    
    try:
        closes = [float(c[4]) for c in candles[-20:]]
        
        # Simple slope calculation
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        
        # Normalize to 0-1 range
        price_range = max(closes) - min(closes)
        if price_range > 0:
            trend_strength = abs(slope) / (price_range / len(closes))
            return min(trend_strength, 1.0)
        
        return 0.0
        
    except Exception:
        return 0.0


def _analyze_pullback_quality(candles: List[List], market_context: str) -> float:
    """Analizuj jakość korekty - czy czysta czy chaotyczna"""
    if not candles or len(candles) < 10:
        return 0.0
    
    try:
        # Check for orderly pullback vs chaotic movement
        recent_candles = candles[-10:]
        closes = [float(c[4]) for c in recent_candles]
        
        # Calculate consistency of direction
        direction_changes = 0
        for i in range(1, len(closes)):
            if i > 1:
                prev_direction = closes[i-1] - closes[i-2]
                curr_direction = closes[i] - closes[i-1]
                if (prev_direction > 0) != (curr_direction > 0):
                    direction_changes += 1
        
        # Lower direction changes = higher quality pullback
        max_changes = len(closes) - 2
        if max_changes > 0:
            consistency = 1.0 - (direction_changes / max_changes)
            return max(0.0, consistency)
        
        return 0.5
        
    except Exception:
        return 0.0


def _measure_support_reaction(candles: List[List], orderbook_info: Dict) -> float:
    """Zmierz siłę reakcji na poziomie wsparcia"""
    if not candles or len(candles) < 5:
        return 0.0
    
    try:
        # Check for bounce from recent lows
        recent_lows = [float(c[3]) for c in candles[-10:]]  # Low prices
        recent_closes = [float(c[4]) for c in candles[-5:]]  # Recent closes
        
        min_low = min(recent_lows)
        current_close = recent_closes[-1]
        
        # Distance from low as support strength
        if min_low > 0:
            bounce_strength = (current_close - min_low) / min_low
            
            # Bonus for orderbook support
            orderbook_support = 0.1 if orderbook_info.get("bids_layered", False) else 0.0
            
            return min(bounce_strength + orderbook_support, 1.0)
        
        return 0.0
        
    except Exception:
        return 0.0


def _score_liquidity_patterns(candles: List[List], market_data: Dict) -> float:
    """Ocena wzorców płynności"""
    if not candles:
        return 0.0
    
    try:
        # Analyze volume patterns
        volumes = [float(c[5]) for c in candles[-10:] if len(c) > 5]
        
        if not volumes:
            return 0.0
        
        avg_volume = sum(volumes) / len(volumes)
        recent_volume = volumes[-1] if volumes else 0
        
        # Volume spike indicates liquidity interest
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Normalize to 0-1
        liquidity_score = min(volume_ratio / 2.0, 1.0)
        
        return liquidity_score
        
    except Exception:
        return 0.0


def _detect_market_psychology(candles: List[List], candle_behavior: Dict) -> float:
    """Wykryj psychologię rynku - manipulacje, panikę"""
    if not candles:
        return 0.5
    
    try:
        # Check for manipulation patterns
        manipulation_score = 0.0
        
        # Large wicks suggest manipulation
        recent_candles = candles[-5:]
        for candle in recent_candles:
            if len(candle) >= 5:
                high, low, open_p, close = float(candle[2]), float(candle[3]), float(candle[1]), float(candle[4])
                body = abs(close - open_p)
                total_range = high - low
                
                if total_range > 0:
                    wick_ratio = (total_range - body) / total_range
                    if wick_ratio > 0.7:  # Large wicks
                        manipulation_score += 0.2
        
        # Normalize
        manipulation_score = min(manipulation_score, 1.0)
        
        return manipulation_score
        
    except Exception:
        return 0.5


def _evaluate_htf_support(candles: List[List], symbol: str) -> float:
    """Oceń wsparcie z wyższych timeframe'ów"""
    if not candles:
        return 0.0
    
    try:
        # Simple HTF analysis based on longer trend
        if len(candles) >= 50:
            long_term_closes = [float(c[4]) for c in candles[-50:]]
            short_term_closes = [float(c[4]) for c in candles[-10:]]
            
            long_term_avg = sum(long_term_closes) / len(long_term_closes)
            short_term_avg = sum(short_term_closes) / len(short_term_closes)
            
            # HTF supportive if short term above long term
            if long_term_avg > 0:
                htf_support = (short_term_avg - long_term_avg) / long_term_avg
                return max(0.0, min(htf_support * 2, 1.0))  # Normalize
        
        return 0.0
        
    except Exception:
        return 0.0


def _get_phase_modifier(candles: List[List], market_context: str) -> float:
    """Oblicz modyfikator fazy rynku"""
    if not candles:
        return 0.0
    
    # Simple phase scoring based on market context
    phase_scores = {
        "impulse": 0.8,
        "pullback": 0.6,
        "range": 0.3,
        "breakout": 0.9,
        "distribution": 0.1,
        "unknown": 0.5
    }
    
    return phase_scores.get(market_context, 0.5)


def _detect_market_phase(candles: List[List], market_context: str) -> str:
    """Wykryj fazę rynku"""
    if not candles or len(candles) < 10:
        return "unknown"
    
    # Map market context to phases
    context_to_phase = {
        "impulse": "breakout-continuation",
        "pullback": "range-accumulation", 
        "range": "range-accumulation",
        "breakout": "breakout-continuation",
        "distribution": "exhaustion-pullback"
    }
    
    return context_to_phase.get(market_context, "unknown")


def _identify_price_action_pattern(candles: List[List], candle_behavior: Dict) -> str:
    """Identyfikuj wzorzec price action"""
    if not candles:
        return "none"
    
    # Simple pattern detection
    if candle_behavior.get("shows_buy_pressure", False):
        if candle_behavior.get("pattern") == "bullish":
            return "impulse"
        else:
            return "continuation"
    
    return "none"


def _analyze_volume_behavior(candles: List[List]) -> str:
    """Analizuj zachowanie wolumenu"""
    if not candles or len(candles) < 5:
        return "neutral"
    
    try:
        volumes = [float(c[5]) for c in candles[-5:] if len(c) > 5]
        if len(volumes) < 2:
            return "neutral"
        
        # Compare recent vs average
        recent_vol = volumes[-1]
        avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
        
        if recent_vol > avg_vol * 1.5:
            return "supporting"
        elif recent_vol < avg_vol * 0.7:
            return "declining"
        
        return "neutral"
        
    except Exception:
        return "neutral"


def _check_htf_trend_match(candles: List[List], market_context: str) -> bool:
    """Sprawdź zgodność z trendem HTF"""
    if not candles:
        return False
    
    # Simple HTF check - if context suggests uptrend
    uptrend_contexts = ["impulse", "breakout", "pullback"]
    return market_context in uptrend_contexts


def _get_fallback_features(symbol: str) -> Dict[str, Any]:
    """Fallback features when extraction fails"""
    return {
        "trend_strength": 0.0,
        "pullback_quality": 0.0,
        "support_reaction": 0.0,
        "liquidity_pattern_score": 0.0,
        "psych_score": 0.5,
        "htf_supportive_score": 0.0,
        "market_phase_modifier": 0.0,
        "market_phase": "unknown",
        "price_action_pattern": "none",
        "volume_behavior": "neutral",
        "htf_trend_match": False,
        "symbol": symbol,
        "candle_count": 0,
        "data_quality": "error"
    }


if __name__ == "__main__":
    # Test feature extraction
    print("🧪 Testing Feature Extractor...")
    
    # Mock test data
    test_candles = [
        [1640995200, 100.0, 102.0, 99.0, 101.0, 1000],
        [1640995260, 101.0, 103.0, 100.0, 102.0, 1200],
        [1640995320, 102.0, 104.0, 101.0, 103.0, 1100]
    ]
    
    features = extract_all_features_for_token("TESTUSDT", test_candles)
    
    print(f"📊 Extracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print(f"✅ Feature extraction test complete")