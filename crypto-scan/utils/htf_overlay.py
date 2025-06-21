#!/usr/bin/env python3
"""
HTF (Higher Time Frame) Confirmation Layer
Pobiera dane z wyższych timeframów i ocenia zgodność trendu
"""

import numpy as np
from typing import List, Dict, Optional
from utils.bybit_candles import get_candles


def get_htf_confirmation(symbol: str, current_timeframe: str = "15") -> Dict:
    """
    HTF Confirmation Layer
    
    Pobiera dane z 1h (lub 4h) i ocenia:
    - Slope EMA
    - Green ratio
    - Trend match z current timeframe
    
    Args:
        symbol: Trading symbol
        current_timeframe: Current timeframe being analyzed
        
    Returns:
        dict: {
            "htf_trend_match": bool,
            "htf_supportive_score": float,  # 0.0-1.0
            "htf_details": dict
        }
    """
    try:
        # Determine HTF based on current timeframe
        if current_timeframe in ["1", "3", "5"]:
            htf_interval = "15"  # Use 15m for lower timeframes
        elif current_timeframe in ["15", "30"]:
            htf_interval = "60"  # Use 1h for 15m/30m
        else:
            htf_interval = "240"  # Use 4h for 1h and above
        
        # Get HTF candles
        print(f"[HTF] Fetching {htf_interval}m candles for {symbol}")
        htf_candles = get_candles(symbol, interval=htf_interval, limit=50)
        
        if not htf_candles or len(htf_candles) < 20:
            return {
                "htf_trend_match": False,
                "htf_supportive_score": 0.5,
                "htf_details": {"error": "insufficient_htf_data"},
                "data_available": False
            }
        
        # Analyze HTF data
        htf_analysis = _analyze_htf_trend(htf_candles, htf_interval)
        
        # Calculate supportive score
        supportive_score = _calculate_htf_supportive_score(htf_analysis)
        
        # Determine trend match
        trend_match = _determine_htf_trend_match(htf_analysis)
        
        result = {
            "htf_trend_match": trend_match,
            "htf_supportive_score": round(supportive_score, 3),
            "htf_details": htf_analysis,
            "data_available": True,
            "htf_timeframe": htf_interval
        }
        
        print(f"[HTF] {symbol}: {htf_interval}m trend_match={trend_match} | supportive_score={supportive_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ HTF confirmation error for {symbol}: {e}")
        return {
            "htf_trend_match": False,
            "htf_supportive_score": 0.5,
            "htf_details": {"error": str(e)},
            "data_available": False
        }


def _analyze_htf_trend(candles: List[List], timeframe: str) -> Dict:
    """Analyze HTF trend characteristics"""
    try:
        # Extract price data
        opens = [float(candle[1]) for candle in candles]
        highs = [float(candle[2]) for candle in candles]
        lows = [float(candle[3]) for candle in candles]
        closes = [float(candle[4]) for candle in candles]
        volumes = [float(candle[5]) for candle in candles]
        
        # 1. EMA Slope Analysis
        ema_analysis = _analyze_htf_ema_slope(closes)
        
        # 2. Green/Red Ratio Analysis
        green_ratio_analysis = _analyze_htf_green_ratio(opens, closes)
        
        # 3. Higher Highs / Lower Lows Analysis
        structure_analysis = _analyze_htf_structure(highs, lows, closes)
        
        # 4. Volume Trend Analysis
        volume_analysis = _analyze_htf_volume_trend(volumes)
        
        # 5. Momentum Analysis
        momentum_analysis = _analyze_htf_momentum(closes)
        
        return {
            "timeframe": timeframe,
            "ema_analysis": ema_analysis,
            "green_ratio_analysis": green_ratio_analysis,
            "structure_analysis": structure_analysis,
            "volume_analysis": volume_analysis,
            "momentum_analysis": momentum_analysis,
            "candle_count": len(candles)
        }
        
    except Exception as e:
        return {"error": f"HTF analysis failed: {e}"}


def _analyze_htf_ema_slope(closes: List[float]) -> Dict:
    """Analyze EMA slope on HTF"""
    try:
        if len(closes) < 21:
            return {"slope": "insufficient_data", "slope_score": 0.5}
        
        # Calculate EMA21
        ema21 = _calculate_ema(closes, 21)
        if len(ema21) < 10:
            return {"slope": "insufficient_ema", "slope_score": 0.5}
        
        # Calculate slope over different periods
        recent_slope = (ema21[-1] - ema21[-5]) / ema21[-5]  # 5-period slope
        longer_slope = (ema21[-1] - ema21[-10]) / ema21[-10]  # 10-period slope
        
        # Determine slope direction and strength
        if recent_slope > 0.02:  # 2%+ slope
            slope_direction = "strong_rising"
            slope_score = 0.9
        elif recent_slope > 0.005:  # 0.5%+ slope
            slope_direction = "rising"
            slope_score = 0.7
        elif recent_slope > -0.005:  # Flat
            slope_direction = "flat"
            slope_score = 0.5
        elif recent_slope > -0.02:  # Declining
            slope_direction = "falling"
            slope_score = 0.3
        else:  # Strong decline
            slope_direction = "strong_falling"
            slope_score = 0.1
        
        return {
            "slope": slope_direction,
            "slope_score": slope_score,
            "recent_slope": round(recent_slope, 4),
            "longer_slope": round(longer_slope, 4),
            "current_ema": ema21[-1],
            "ema_change_5p": round((ema21[-1] - ema21[-5]) / ema21[-5] * 100, 2)
        }
        
    except Exception:
        return {"slope": "error", "slope_score": 0.5}


def _analyze_htf_green_ratio(opens: List[float], closes: List[float]) -> Dict:
    """Analyze green/red candle ratio on HTF"""
    try:
        if len(closes) < 10:
            return {"green_ratio": 0.5, "green_score": 0.5}
        
        # Count green/red candles in different periods
        recent_period = min(10, len(closes))
        longer_period = min(20, len(closes))
        
        # Recent period analysis
        recent_green = 0
        for i in range(-recent_period, 0):
            if i < len(closes) and closes[i] > opens[i]:
                recent_green += 1
        
        recent_green_ratio = recent_green / recent_period
        
        # Longer period analysis
        longer_green = 0
        for i in range(-longer_period, 0):
            if i < len(closes) and closes[i] > opens[i]:
                longer_green += 1
        
        longer_green_ratio = longer_green / longer_period
        
        # Score based on green ratio
        if recent_green_ratio >= 0.7:  # 70%+ green
            green_score = 0.9
        elif recent_green_ratio >= 0.6:  # 60%+ green
            green_score = 0.7
        elif recent_green_ratio >= 0.4:  # Balanced
            green_score = 0.5
        elif recent_green_ratio >= 0.3:  # More red
            green_score = 0.3
        else:  # Mostly red
            green_score = 0.1
        
        return {
            "green_ratio": round(recent_green_ratio, 3),
            "green_score": green_score,
            "recent_green_count": recent_green,
            "recent_period": recent_period,
            "longer_green_ratio": round(longer_green_ratio, 3),
            "longer_period": longer_period
        }
        
    except Exception:
        return {"green_ratio": 0.5, "green_score": 0.5}


def _analyze_htf_structure(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Analyze HTF market structure"""
    try:
        if len(closes) < 10:
            return {"structure": "insufficient_data", "structure_score": 0.5}
        
        # Look for higher highs and higher lows (uptrend)
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        higher_highs = 0
        higher_lows = 0
        lower_highs = 0
        lower_lows = 0
        
        # Count structure patterns
        for i in range(1, len(recent_highs)):
            if recent_highs[i] > recent_highs[i-1]:
                higher_highs += 1
            else:
                lower_highs += 1
            
            if recent_lows[i] > recent_lows[i-1]:
                higher_lows += 1
            else:
                lower_lows += 1
        
        # Determine structure
        if higher_highs >= 6 and higher_lows >= 6:
            structure = "strong_uptrend"
            structure_score = 0.9
        elif higher_highs >= 4 and higher_lows >= 4:
            structure = "uptrend"
            structure_score = 0.7
        elif lower_highs >= 6 and lower_lows >= 6:
            structure = "strong_downtrend"
            structure_score = 0.1
        elif lower_highs >= 4 and lower_lows >= 4:
            structure = "downtrend"
            structure_score = 0.3
        else:
            structure = "sideways"
            structure_score = 0.5
        
        return {
            "structure": structure,
            "structure_score": structure_score,
            "higher_highs": higher_highs,
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "lower_lows": lower_lows
        }
        
    except Exception:
        return {"structure": "error", "structure_score": 0.5}


def _analyze_htf_volume_trend(volumes: List[float]) -> Dict:
    """Analyze HTF volume trend"""
    try:
        if len(volumes) < 10:
            return {"volume_trend": "insufficient_data", "volume_score": 0.5}
        
        recent_volumes = volumes[-5:]
        earlier_volumes = volumes[-15:-5] if len(volumes) >= 15 else volumes[:-5]
        
        recent_avg = np.mean(recent_volumes)
        earlier_avg = np.mean(earlier_volumes) if earlier_volumes else recent_avg
        
        if earlier_avg > 0:
            volume_ratio = recent_avg / earlier_avg
        else:
            volume_ratio = 1.0
        
        # Determine volume trend
        if volume_ratio > 1.3:
            volume_trend = "increasing"
            volume_score = 0.7
        elif volume_ratio > 1.1:
            volume_trend = "moderate_increase"
            volume_score = 0.6
        elif volume_ratio < 0.8:
            volume_trend = "decreasing"
            volume_score = 0.4
        else:
            volume_trend = "stable"
            volume_score = 0.5
        
        return {
            "volume_trend": volume_trend,
            "volume_score": volume_score,
            "volume_ratio": round(volume_ratio, 3),
            "recent_avg": recent_avg,
            "earlier_avg": earlier_avg
        }
        
    except Exception:
        return {"volume_trend": "error", "volume_score": 0.5}


def _analyze_htf_momentum(closes: List[float]) -> Dict:
    """Analyze HTF momentum"""
    try:
        if len(closes) < 10:
            return {"momentum": "insufficient_data", "momentum_score": 0.5}
        
        # Calculate price changes over different periods
        short_change = (closes[-1] - closes[-3]) / closes[-3]  # 3-period change
        medium_change = (closes[-1] - closes[-7]) / closes[-7]  # 7-period change
        long_change = (closes[-1] - closes[-14]) / closes[-14] if len(closes) >= 14 else short_change
        
        # Determine momentum
        avg_momentum = (short_change + medium_change + long_change) / 3
        
        if avg_momentum > 0.05:  # 5%+ momentum
            momentum = "strong_bullish"
            momentum_score = 0.9
        elif avg_momentum > 0.02:  # 2%+ momentum
            momentum = "bullish"
            momentum_score = 0.7
        elif avg_momentum > -0.02:  # Neutral
            momentum = "neutral"
            momentum_score = 0.5
        elif avg_momentum > -0.05:  # Bearish
            momentum = "bearish"
            momentum_score = 0.3
        else:  # Strong bearish
            momentum = "strong_bearish"
            momentum_score = 0.1
        
        return {
            "momentum": momentum,
            "momentum_score": momentum_score,
            "short_change": round(short_change * 100, 2),
            "medium_change": round(medium_change * 100, 2),
            "long_change": round(long_change * 100, 2),
            "avg_momentum": round(avg_momentum * 100, 2)
        }
        
    except Exception:
        return {"momentum": "error", "momentum_score": 0.5}


def _calculate_htf_supportive_score(htf_analysis: Dict) -> float:
    """Calculate overall HTF supportive score"""
    try:
        # Extract scores from different analyses
        ema_score = htf_analysis.get("ema_analysis", {}).get("slope_score", 0.5)
        green_score = htf_analysis.get("green_ratio_analysis", {}).get("green_score", 0.5)
        structure_score = htf_analysis.get("structure_analysis", {}).get("structure_score", 0.5)
        volume_score = htf_analysis.get("volume_analysis", {}).get("volume_score", 0.5)
        momentum_score = htf_analysis.get("momentum_analysis", {}).get("momentum_score", 0.5)
        
        # Weighted average
        supportive_score = (
            ema_score * 0.3 +        # EMA slope most important
            structure_score * 0.25 +  # Market structure
            momentum_score * 0.2 +    # Momentum
            green_score * 0.15 +      # Green ratio
            volume_score * 0.1        # Volume trend
        )
        
        return supportive_score
        
    except Exception:
        return 0.5


def _determine_htf_trend_match(htf_analysis: Dict) -> bool:
    """Determine if HTF trend matches/supports current timeframe"""
    try:
        # Get key indicators
        ema_slope = htf_analysis.get("ema_analysis", {}).get("slope", "unknown")
        green_ratio = htf_analysis.get("green_ratio_analysis", {}).get("green_ratio", 0.5)
        structure = htf_analysis.get("structure_analysis", {}).get("structure", "unknown")
        momentum = htf_analysis.get("momentum_analysis", {}).get("momentum", "unknown")
        
        # Count bullish signals
        bullish_signals = 0
        
        if ema_slope in ["rising", "strong_rising"]:
            bullish_signals += 1
        if green_ratio >= 0.6:  # 60%+ green candles
            bullish_signals += 1
        if structure in ["uptrend", "strong_uptrend"]:
            bullish_signals += 1
        if momentum in ["bullish", "strong_bullish"]:
            bullish_signals += 1
        
        # Trend match if majority of signals are bullish
        trend_match = bullish_signals >= 3
        
        return trend_match
        
    except Exception:
        return False


def _calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    if not prices or len(prices) < period:
        return []
    
    ema = []
    multiplier = 2 / (period + 1)
    
    # Start with SMA for first value
    sma = sum(prices[:period]) / period
    ema.append(sma)
    
    # Calculate EMA for remaining values
    for i in range(period, len(prices)):
        ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
        ema.append(ema_value)
    
    return ema