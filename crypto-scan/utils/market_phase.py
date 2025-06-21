#!/usr/bin/env python3
"""
Market Structure Phase Detection
Analizuje fazę struktury rynku na podstawie HH/LL, EMA, volatility i compression
"""

import numpy as np
from typing import List, Dict, Tuple


def detect_market_phase(candles: List[List], symbol: str = None) -> Dict:
    """
    Market Structure Phase Detection
    
    Analizuje ostatnie 20-40 świec i klasyfikuje ruch jako:
    - pre-breakout: Kompresja przed wybiciem
    - breakout-continuation: Aktywne wybicie z kontynuacją
    - retest-confirmation: Test poziomów wsparcia/oporu
    - exhaustion-pullback: Wyczerpanie ruchu, pullback
    - range-accumulation: Konsolidacja w range
    
    Args:
        candles: Lista OHLCV candles [[timestamp, open, high, low, close, volume], ...]
        symbol: Symbol for logging
        
    Returns:
        dict: {
            "market_phase": str,
            "phase_score": float,  # 0.0-1.0
            "structure_details": dict,
            "confidence": float
        }
    """
    try:
        if not candles or len(candles) < 20:
            return {
                "market_phase": "insufficient_data",
                "phase_score": 0.0,
                "structure_details": {},
                "confidence": 0.0
            }
        
        # Extract price data
        closes = [float(candle[4]) for candle in candles[-40:]]  # Last 40 candles
        highs = [float(candle[2]) for candle in candles[-40:]]
        lows = [float(candle[3]) for candle in candles[-40:]]
        volumes = [float(candle[5]) for candle in candles[-40:]]
        
        # Calculate EMA21 for trend context
        ema21 = _calculate_ema(closes, 21)
        current_price = closes[-1]
        current_ema = ema21[-1] if ema21 else current_price
        
        # 1. Higher Highs / Lower Lows Analysis
        hh_ll_analysis = _analyze_hh_ll_structure(highs, lows, closes)
        
        # 2. Price vs EMA Analysis
        ema_analysis = _analyze_price_vs_ema(closes, ema21)
        
        # 3. Volatility & Compression Analysis
        volatility_analysis = _analyze_volatility_compression(highs, lows, closes)
        
        # 4. Volume Pattern Analysis
        volume_analysis = _analyze_volume_pattern(volumes, closes)
        
        # 5. Determine Market Phase
        phase_result = _determine_market_phase(
            hh_ll_analysis, ema_analysis, volatility_analysis, volume_analysis
        )
        
        # Calculate overall confidence
        confidence = _calculate_phase_confidence(
            hh_ll_analysis, ema_analysis, volatility_analysis, volume_analysis
        )
        
        result = {
            "market_phase": phase_result["phase"],
            "phase_score": phase_result["score"],
            "structure_details": {
                "hh_ll_pattern": hh_ll_analysis,
                "ema_relationship": ema_analysis,
                "volatility_state": volatility_analysis,
                "volume_behavior": volume_analysis,
                "current_price": current_price,
                "current_ema21": current_ema
            },
            "confidence": confidence
        }
        
        if symbol:
            print(f"[MARKET PHASE] {symbol}: {phase_result['phase']} | Score: {phase_result['score']:.3f} | Confidence: {confidence:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Market phase detection error for {symbol}: {e}")
        return {
            "market_phase": "error",
            "phase_score": 0.0,
            "structure_details": {},
            "confidence": 0.0
        }


def _analyze_hh_ll_structure(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Analyze Higher Highs / Lower Lows structure"""
    try:
        recent_highs = highs[-10:]  # Last 10 candles
        recent_lows = lows[-10:]
        
        # Count higher highs and lower lows
        higher_highs = 0
        lower_lows = 0
        
        for i in range(1, len(recent_highs)):
            if recent_highs[i] > recent_highs[i-1]:
                higher_highs += 1
            if recent_lows[i] < recent_lows[i-1]:
                lower_lows += 1
        
        # Determine trend structure
        if higher_highs >= 6:
            structure = "strong_uptrend"
            structure_score = 0.9
        elif higher_highs >= 4:
            structure = "uptrend"
            structure_score = 0.7
        elif lower_lows >= 6:
            structure = "strong_downtrend"
            structure_score = 0.1
        elif lower_lows >= 4:
            structure = "downtrend"
            structure_score = 0.3
        else:
            structure = "sideways"
            structure_score = 0.5
        
        return {
            "structure": structure,
            "structure_score": structure_score,
            "higher_highs_count": higher_highs,
            "lower_lows_count": lower_lows
        }
        
    except Exception:
        return {"structure": "unknown", "structure_score": 0.5, "higher_highs_count": 0, "lower_lows_count": 0}


def _analyze_price_vs_ema(closes: List[float], ema21: List[float]) -> Dict:
    """Analyze price relationship with EMA21"""
    try:
        if not ema21 or len(ema21) < 5:
            return {"ema_trend": "unknown", "ema_score": 0.5, "price_position": "unknown"}
        
        current_price = closes[-1]
        current_ema = ema21[-1]
        
        # EMA slope analysis
        ema_slope = (ema21[-1] - ema21[-5]) / ema21[-5]
        
        if ema_slope > 0.02:  # 2% increase over 5 periods
            ema_trend = "strong_rising"
            ema_score = 0.9
        elif ema_slope > 0.005:
            ema_trend = "rising"
            ema_score = 0.7
        elif ema_slope < -0.02:
            ema_trend = "strong_falling"
            ema_score = 0.1
        elif ema_slope < -0.005:
            ema_trend = "falling"
            ema_score = 0.3
        else:
            ema_trend = "flat"
            ema_score = 0.5
        
        # Price position relative to EMA
        price_ema_ratio = (current_price - current_ema) / current_ema
        
        if price_ema_ratio > 0.05:
            price_position = "well_above_ema"
        elif price_ema_ratio > 0.01:
            price_position = "above_ema"
        elif price_ema_ratio > -0.01:
            price_position = "near_ema"
        elif price_ema_ratio > -0.05:
            price_position = "below_ema"
        else:
            price_position = "well_below_ema"
        
        return {
            "ema_trend": ema_trend,
            "ema_score": ema_score,
            "price_position": price_position,
            "price_ema_ratio": price_ema_ratio,
            "ema_slope": ema_slope
        }
        
    except Exception:
        return {"ema_trend": "unknown", "ema_score": 0.5, "price_position": "unknown"}


def _analyze_volatility_compression(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Analyze volatility and compression patterns"""
    try:
        # Calculate recent ATR (Average True Range)
        recent_ranges = []
        for i in range(-10, 0):  # Last 10 candles
            if i < len(highs):
                candle_range = highs[i] - lows[i]
                recent_ranges.append(candle_range / closes[i])  # Normalized range
        
        current_atr = np.mean(recent_ranges) if recent_ranges else 0
        
        # Compare with longer-term ATR
        longer_ranges = []
        for i in range(-30, 0):  # Last 30 candles
            if i < len(highs):
                candle_range = highs[i] - lows[i]
                longer_ranges.append(candle_range / closes[i])
        
        longer_atr = np.mean(longer_ranges) if longer_ranges else current_atr
        
        # Compression detection
        if longer_atr > 0:
            compression_ratio = current_atr / longer_atr
        else:
            compression_ratio = 1.0
        
        if compression_ratio < 0.5:
            volatility_state = "high_compression"
            volatility_score = 0.8  # High score for pre-breakout potential
        elif compression_ratio < 0.7:
            volatility_state = "moderate_compression"
            volatility_score = 0.6
        elif compression_ratio > 1.5:
            volatility_state = "expansion"
            volatility_score = 0.7  # Good for continuation
        else:
            volatility_state = "normal"
            volatility_score = 0.5
        
        return {
            "volatility_state": volatility_state,
            "volatility_score": volatility_score,
            "compression_ratio": compression_ratio,
            "current_atr": current_atr,
            "longer_atr": longer_atr
        }
        
    except Exception:
        return {"volatility_state": "unknown", "volatility_score": 0.5, "compression_ratio": 1.0}


def _analyze_volume_pattern(volumes: List[float], closes: List[float]) -> Dict:
    """Analyze volume patterns for market phase detection"""
    try:
        recent_volumes = volumes[-10:]
        older_volumes = volumes[-30:-10] if len(volumes) >= 30 else volumes[:-10]
        
        recent_avg_volume = np.mean(recent_volumes)
        older_avg_volume = np.mean(older_volumes) if older_volumes else recent_avg_volume
        
        # Volume trend
        if older_avg_volume > 0:
            volume_ratio = recent_avg_volume / older_avg_volume
        else:
            volume_ratio = 1.0
        
        # Price-volume relationship
        recent_price_changes = []
        recent_volume_spikes = []
        
        for i in range(-5, 0):  # Last 5 candles
            if i < len(closes) - 1:
                price_change = (closes[i] - closes[i-1]) / closes[i-1]
                volume_spike = volumes[i] / recent_avg_volume if recent_avg_volume > 0 else 1.0
                recent_price_changes.append(price_change)
                recent_volume_spikes.append(volume_spike)
        
        # Determine volume behavior
        if volume_ratio > 1.5:
            volume_behavior = "increasing"
            volume_score = 0.8
        elif volume_ratio > 1.2:
            volume_behavior = "moderate_increase"
            volume_score = 0.6
        elif volume_ratio < 0.7:
            volume_behavior = "declining"
            volume_score = 0.4
        else:
            volume_behavior = "stable"
            volume_score = 0.5
        
        return {
            "volume_behavior": volume_behavior,
            "volume_score": volume_score,
            "volume_ratio": volume_ratio,
            "recent_avg_volume": recent_avg_volume
        }
        
    except Exception:
        return {"volume_behavior": "unknown", "volume_score": 0.5, "volume_ratio": 1.0}


def _determine_market_phase(hh_ll: Dict, ema: Dict, volatility: Dict, volume: Dict) -> Dict:
    """Determine the current market phase based on all analyses"""
    try:
        structure = hh_ll.get("structure", "unknown")
        volatility_state = volatility.get("volatility_state", "unknown")
        volume_behavior = volume.get("volume_behavior", "unknown")
        ema_trend = ema.get("ema_trend", "unknown")
        
        # Phase detection logic
        
        # Pre-breakout: High compression + sideways structure
        if (volatility_state == "high_compression" and 
            structure in ["sideways"] and 
            volume_behavior in ["declining", "stable"]):
            return {"phase": "pre-breakout", "score": 0.8}
        
        # Breakout-continuation: Expansion + strong trend + increasing volume
        if (volatility_state == "expansion" and 
            structure in ["strong_uptrend", "uptrend"] and 
            volume_behavior in ["increasing", "moderate_increase"] and
            ema_trend in ["strong_rising", "rising"]):
            return {"phase": "breakout-continuation", "score": 0.9}
        
        # Retest-confirmation: Moderate compression + trend structure + stable volume
        if (volatility_state in ["moderate_compression", "normal"] and 
            structure in ["uptrend", "strong_uptrend"] and 
            ema_trend in ["rising", "strong_rising"]):
            return {"phase": "retest-confirmation", "score": 0.7}
        
        # Exhaustion-pullback: Normal/expansion volatility + declining volume + weakening trend
        if (volume_behavior == "declining" and 
            ema_trend in ["flat", "falling"] and
            structure in ["sideways", "downtrend"]):
            return {"phase": "exhaustion-pullback", "score": 0.3}
        
        # Range-accumulation: Normal compression + sideways + stable volume
        if (structure == "sideways" and 
            volatility_state in ["normal", "moderate_compression"] and 
            ema_trend == "flat"):
            return {"phase": "range-accumulation", "score": 0.5}
        
        # Default case
        return {"phase": "undefined", "score": 0.4}
        
    except Exception:
        return {"phase": "error", "score": 0.0}


def _calculate_phase_confidence(hh_ll: Dict, ema: Dict, volatility: Dict, volume: Dict) -> float:
    """Calculate confidence level for phase detection"""
    try:
        # Base confidence from individual component scores
        structure_confidence = hh_ll.get("structure_score", 0.5)
        ema_confidence = abs(ema.get("ema_score", 0.5) - 0.5) * 2  # Distance from neutral
        volatility_confidence = volatility.get("volatility_score", 0.5)
        volume_confidence = volume.get("volume_score", 0.5)
        
        # Weighted average confidence
        confidence = (structure_confidence * 0.3 + 
                     ema_confidence * 0.3 + 
                     volatility_confidence * 0.2 + 
                     volume_confidence * 0.2)
        
        return min(0.95, max(0.1, confidence))
        
    except Exception:
        return 0.5


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