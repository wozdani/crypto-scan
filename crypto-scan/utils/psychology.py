#!/usr/bin/env python3
"""
Psychological Traps Detector
Wykrywa sygnały manipulacji rynku i psychologiczne pułapki
"""

import numpy as np
from typing import List, Dict, Tuple


def detect_psychological_traps(candles: List[List], symbol: str = None) -> Dict:
    """
    Psychological Traps Detector
    
    Wykrywa sygnały:
    - Liquidity grab / SL sweep (long wick z szybkim powrotem)
    - Fake breakout (wybicie bez kontynuacji)
    - Choppy behavior (chaotyczne zachowanie ceny)
    
    Args:
        candles: Lista OHLCV candles [[timestamp, open, high, low, close, volume], ...]
        symbol: Symbol for logging
        
    Returns:
        dict: {
            "psychological_flags": List[str],
            "psych_score": float,  # 0.0-1.0 (higher = cleaner move)
            "trap_details": dict
        }
    """
    try:
        if not candles or len(candles) < 10:
            return {
                "psychological_flags": ["insufficient_data"],
                "psych_score": 0.5,
                "trap_details": {}
            }
        
        # Extract price data
        opens = [float(candle[1]) for candle in candles[-20:]]
        highs = [float(candle[2]) for candle in candles[-20:]]
        lows = [float(candle[3]) for candle in candles[-20:]]
        closes = [float(candle[4]) for candle in candles[-20:]]
        volumes = [float(candle[5]) for candle in candles[-20:]]
        
        psychological_flags = []
        trap_details = {}
        
        # 1. Liquidity Grab / Stop Loss Sweep Detection
        liquidity_grab_result = _detect_liquidity_grabs(opens, highs, lows, closes)
        if liquidity_grab_result["detected"]:
            psychological_flags.append("liquidity_grab")
        trap_details["liquidity_grab"] = liquidity_grab_result
        
        # 2. Fake Breakout Detection
        fake_breakout_result = _detect_fake_breakouts(highs, lows, closes, volumes)
        if fake_breakout_result["detected"]:
            psychological_flags.append("fake_breakout")
        trap_details["fake_breakout"] = fake_breakout_result
        
        # 3. Choppy Behavior Detection
        choppy_result = _detect_choppy_behavior(opens, highs, lows, closes)
        if choppy_result["detected"]:
            psychological_flags.append("choppy_behavior")
        trap_details["choppy_behavior"] = choppy_result
        
        # 4. Pinning Detection (price manipulation around key levels)
        pinning_result = _detect_pinning_behavior(highs, lows, closes)
        if pinning_result["detected"]:
            psychological_flags.append("pinning_detected")
        trap_details["pinning"] = pinning_result
        
        # 5. Calculate Psychological Score (higher = cleaner, more trustworthy move)
        psych_score = _calculate_psychological_score(
            liquidity_grab_result, fake_breakout_result, choppy_result, pinning_result
        )
        
        result = {
            "psychological_flags": psychological_flags,
            "psych_score": round(psych_score, 3),
            "trap_details": trap_details
        }
        
        if symbol and psychological_flags:
            flags_str = ", ".join(psychological_flags)
            print(f"[PSYCHOLOGY] {symbol}: Flags: {flags_str} | Clean Score: {psych_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Psychological analysis error for {symbol}: {e}")
        return {
            "psychological_flags": ["analysis_error"],
            "psych_score": 0.5,
            "trap_details": {}
        }


def _detect_liquidity_grabs(opens: List[float], highs: List[float], 
                           lows: List[float], closes: List[float]) -> Dict:
    """Detect liquidity grabs / stop loss sweeps"""
    try:
        liquidity_grabs = 0
        grab_details = []
        
        # Look for long wicks with quick reversal in last 10 candles
        for i in range(-10, 0):
            if i < len(highs):
                open_price = opens[i]
                high_price = highs[i]
                low_price = lows[i]
                close_price = closes[i]
                
                body_size = abs(close_price - open_price)
                upper_wick = high_price - max(open_price, close_price)
                lower_wick = min(open_price, close_price) - low_price
                candle_range = high_price - low_price
                
                if candle_range > 0:
                    upper_wick_ratio = upper_wick / candle_range
                    lower_wick_ratio = lower_wick / candle_range
                    body_ratio = body_size / candle_range
                    
                    # Upper liquidity grab (long upper wick, small body, quick reversal down)
                    if (upper_wick_ratio > 0.4 and body_ratio < 0.3 and 
                        close_price < open_price):  # Bearish close after spike
                        liquidity_grabs += 1
                        grab_details.append({
                            "type": "upper_grab",
                            "candle_index": i,
                            "wick_ratio": upper_wick_ratio
                        })
                    
                    # Lower liquidity grab (long lower wick, small body, quick reversal up)
                    if (lower_wick_ratio > 0.4 and body_ratio < 0.3 and 
                        close_price > open_price):  # Bullish close after dip
                        liquidity_grabs += 1
                        grab_details.append({
                            "type": "lower_grab",
                            "candle_index": i,
                            "wick_ratio": lower_wick_ratio
                        })
        
        detected = liquidity_grabs >= 1
        confidence = min(0.9, liquidity_grabs * 0.3)
        
        return {
            "detected": detected,
            "confidence": confidence,
            "grabs_count": liquidity_grabs,
            "grab_details": grab_details
        }
        
    except Exception:
        return {"detected": False, "confidence": 0.0, "grabs_count": 0, "grab_details": []}


def _detect_fake_breakouts(highs: List[float], lows: List[float], 
                          closes: List[float], volumes: List[float]) -> Dict:
    """Detect fake breakouts"""
    try:
        if len(closes) < 10:
            return {"detected": False, "confidence": 0.0}
        
        # Calculate recent high/low levels
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        recent_closes = closes[-10:]
        recent_volumes = volumes[-10:] if len(volumes) >= 10 else [1] * 10
        
        # Find potential resistance/support levels
        resistance_level = max(recent_highs[:-2])  # Exclude last 2 candles
        support_level = min(recent_lows[:-2])
        
        fake_breakouts = 0
        fake_details = []
        
        # Check last 3 candles for fake breakouts
        for i in range(-3, 0):
            if i < len(closes):
                high = highs[i]
                low = lows[i]
                close = closes[i]
                volume = recent_volumes[i] if i < len(recent_volumes) else 1
                avg_volume = np.mean(recent_volumes) if recent_volumes else 1
                
                # Fake upside breakout (break resistance but close below)
                if (high > resistance_level * 1.005 and  # 0.5% above resistance
                    close < resistance_level and  # Close back below
                    volume < avg_volume * 0.8):  # Low volume breakout
                    fake_breakouts += 1
                    fake_details.append({
                        "type": "fake_upside_breakout",
                        "candle_index": i,
                        "resistance_level": resistance_level,
                        "high_reached": high,
                        "close_price": close
                    })
                
                # Fake downside breakout (break support but close above)
                if (low < support_level * 0.995 and  # 0.5% below support
                    close > support_level and  # Close back above
                    volume < avg_volume * 0.8):  # Low volume breakdown
                    fake_breakouts += 1
                    fake_details.append({
                        "type": "fake_downside_breakout",
                        "candle_index": i,
                        "support_level": support_level,
                        "low_reached": low,
                        "close_price": close
                    })
        
        detected = fake_breakouts >= 1
        confidence = min(0.8, fake_breakouts * 0.4)
        
        return {
            "detected": detected,
            "confidence": confidence,
            "fake_breakouts_count": fake_breakouts,
            "fake_details": fake_details,
            "resistance_level": resistance_level,
            "support_level": support_level
        }
        
    except Exception:
        return {"detected": False, "confidence": 0.0}


def _detect_choppy_behavior(opens: List[float], highs: List[float], 
                           lows: List[float], closes: List[float]) -> Dict:
    """Detect choppy price behavior"""
    try:
        if len(closes) < 8:
            return {"detected": False, "confidence": 0.0}
        
        # Calculate directional changes
        directional_changes = 0
        price_ranges = []
        
        for i in range(1, len(closes)):
            current_change = closes[i] - closes[i-1]
            previous_change = closes[i-1] - closes[i-2] if i > 1 else 0
            
            # Count direction reversals
            if (current_change > 0 and previous_change < 0) or (current_change < 0 and previous_change > 0):
                directional_changes += 1
            
            # Calculate normalized price range
            price_range = (highs[i] - lows[i]) / closes[i]
            price_ranges.append(price_range)
        
        # Choppy behavior indicators
        change_ratio = directional_changes / max(1, len(closes) - 2)  # Frequency of direction changes
        avg_range = np.mean(price_ranges) if price_ranges else 0
        range_variance = np.var(price_ranges) if len(price_ranges) > 1 else 0
        
        # Calculate net movement vs total movement
        net_movement = abs(closes[-1] - closes[0])
        total_movement = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        movement_efficiency = net_movement / total_movement if total_movement > 0 else 0
        
        # Choppy if: high direction changes, low movement efficiency, high range variance
        choppy_signals = 0
        if change_ratio > 0.6:  # More than 60% direction changes
            choppy_signals += 1
        if movement_efficiency < 0.3:  # Less than 30% efficiency
            choppy_signals += 1
        if range_variance > avg_range * 0.5:  # High range variance
            choppy_signals += 1
        
        detected = choppy_signals >= 2
        confidence = min(0.8, choppy_signals * 0.3)
        
        return {
            "detected": detected,
            "confidence": confidence,
            "choppy_signals": choppy_signals,
            "change_ratio": round(change_ratio, 3),
            "movement_efficiency": round(movement_efficiency, 3),
            "avg_range": round(avg_range, 4),
            "range_variance": round(range_variance, 6)
        }
        
    except Exception:
        return {"detected": False, "confidence": 0.0}


def _detect_pinning_behavior(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Detect price pinning around key levels"""
    try:
        if len(closes) < 10:
            return {"detected": False, "confidence": 0.0}
        
        # Calculate potential key levels (round numbers, previous highs/lows)
        current_price = closes[-1]
        
        # Find round number levels
        round_levels = []
        price_magnitude = len(str(int(current_price)))
        
        if price_magnitude >= 3:  # For prices > 100
            round_increment = 10 ** (price_magnitude - 2)  # e.g., 10 for $1000, 1 for $100
        else:
            round_increment = 0.1 if current_price > 1 else 0.01
        
        # Generate nearby round levels
        base_level = round(current_price / round_increment) * round_increment
        for offset in [-2, -1, 0, 1, 2]:
            round_levels.append(base_level + (offset * round_increment))
        
        # Count price touches near round levels
        pinning_count = 0
        pinning_details = []
        
        for level in round_levels:
            touches = 0
            tolerance = level * 0.002  # 0.2% tolerance
            
            for i in range(-10, 0):  # Last 10 candles
                if i < len(closes):
                    high = highs[i]
                    low = lows[i]
                    close = closes[i]
                    
                    # Check if price touched this level
                    if (abs(high - level) <= tolerance or 
                        abs(low - level) <= tolerance or 
                        abs(close - level) <= tolerance):
                        touches += 1
            
            if touches >= 3:  # 3+ touches = potential pinning
                pinning_count += 1
                pinning_details.append({
                    "level": level,
                    "touches": touches,
                    "tolerance": tolerance
                })
        
        detected = pinning_count >= 1
        confidence = min(0.7, pinning_count * 0.3)
        
        return {
            "detected": detected,
            "confidence": confidence,
            "pinning_levels": len(pinning_details),
            "pinning_details": pinning_details,
            "current_price": current_price
        }
        
    except Exception:
        return {"detected": False, "confidence": 0.0}


def _calculate_psychological_score(liquidity_grab: Dict, fake_breakout: Dict, 
                                 choppy: Dict, pinning: Dict) -> float:
    """Calculate overall psychological score (higher = cleaner move)"""
    try:
        # Start with neutral score
        base_score = 0.7
        
        # Penalize for psychological traps (reduce score)
        if liquidity_grab.get("detected", False):
            penalty = liquidity_grab.get("confidence", 0) * 0.2
            base_score -= penalty
        
        if fake_breakout.get("detected", False):
            penalty = fake_breakout.get("confidence", 0) * 0.25
            base_score -= penalty
        
        if choppy.get("detected", False):
            penalty = choppy.get("confidence", 0) * 0.3
            base_score -= penalty
        
        if pinning.get("detected", False):
            penalty = pinning.get("confidence", 0) * 0.15
            base_score -= penalty
        
        # Clamp score to valid range
        final_score = max(0.1, min(0.9, base_score))
        
        return final_score
        
    except Exception:
        return 0.5