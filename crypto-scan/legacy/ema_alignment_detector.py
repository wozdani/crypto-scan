"""
EMA Alignment Detector - Classic Bull Trend Formation
Wykrywa układ świec, w którym EMA10 > EMA20 > EMA50 oraz cena > EMA20
"""

import numpy as np


def calculate_ema(prices, period):
    """
    Calculate Exponential Moving Average manually
    
    Args:
        prices: list of close prices
        period: EMA period (10, 20, 50)
    
    Returns:
        float: Current EMA value
    """
    if len(prices) < period:
        return None
    
    # Use Simple Moving Average for first value
    sma = sum(prices[:period]) / period
    
    # Calculate multiplier
    multiplier = 2 / (period + 1)
    
    # Start with SMA, then calculate EMA
    ema = sma
    
    # Calculate EMA for remaining values
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def detect_ema_alignment(candle_data, min_candles=30):
    """
    Detect EMA alignment for bull trend confirmation
    
    Args:
        candle_data: OHLCV candle data (list of [timestamp, open, high, low, close, volume])
        min_candles: minimum number of candles required (default 30)
    
    Returns:
        dict: {
            "ema_alignment": bool,
            "ema_values": dict,
            "alignment_strength": str,
            "price_above_ema20": bool,
            "trend_quality": str
        }
    """
    print(f"[EMA ALIGNMENT] Analyzing EMA alignment with {len(candle_data)} candles")
    
    # Validate input data
    if not candle_data or len(candle_data) < min_candles:
        return {
            "ema_alignment": False,
            "ema_values": {"ema10": None, "ema20": None, "ema50": None},
            "alignment_strength": "insufficient_data",
            "price_above_ema20": False,
            "trend_quality": "invalid"
        }
    
    # Extract close prices from candle data
    try:
        close_prices = [float(candle[4]) for candle in candle_data]
        current_price = close_prices[-1]
    except (IndexError, ValueError, TypeError) as e:
        print(f"[EMA ALIGNMENT] Error extracting close prices: {e}")
        return {
            "ema_alignment": False,
            "ema_values": {"ema10": None, "ema20": None, "ema50": None},
            "alignment_strength": "data_error",
            "price_above_ema20": False,
            "trend_quality": "invalid"
        }
    
    # Calculate EMAs
    ema10 = calculate_ema(close_prices, 10)
    ema20 = calculate_ema(close_prices, 20)
    ema50 = calculate_ema(close_prices, 50)
    
    # Validate EMA calculations
    if ema10 is None or ema20 is None or ema50 is None:
        return {
            "ema_alignment": False,
            "ema_values": {"ema10": ema10, "ema20": ema20, "ema50": ema50},
            "alignment_strength": "calculation_error",
            "price_above_ema20": False,
            "trend_quality": "invalid"
        }
    
    # Check alignment conditions
    # Main condition: EMA10 > EMA20 > EMA50
    ema_sequence_correct = ema10 > ema20 > ema50
    
    # Secondary condition: Current price > EMA20
    price_above_ema20 = current_price > ema20
    
    # Both conditions must be true for alignment
    ema_alignment = ema_sequence_correct and price_above_ema20
    
    # Calculate alignment strength
    alignment_strength = "none"
    trend_quality = "weak"
    
    if ema_alignment:
        # Calculate percentage differences for strength assessment
        ema10_ema20_diff = ((ema10 - ema20) / ema20) * 100
        ema20_ema50_diff = ((ema20 - ema50) / ema50) * 100
        price_ema20_diff = ((current_price - ema20) / ema20) * 100
        
        # Determine alignment strength based on gaps between EMAs
        total_separation = ema10_ema20_diff + ema20_ema50_diff
        
        if total_separation >= 3.0:  # Strong separation (>3%)
            alignment_strength = "strong"
            trend_quality = "excellent"
        elif total_separation >= 1.5:  # Moderate separation (1.5-3%)
            alignment_strength = "moderate"
            trend_quality = "good"
        elif total_separation >= 0.5:  # Weak separation (0.5-1.5%)
            alignment_strength = "weak"
            trend_quality = "fair"
        else:  # Very weak separation (<0.5%)
            alignment_strength = "minimal"
            trend_quality = "weak"
        
        print(f"[EMA ALIGNMENT] ✅ Bull alignment confirmed!")
        print(f"[EMA ALIGNMENT]   EMA10: {ema10:.6f}")
        print(f"[EMA ALIGNMENT]   EMA20: {ema20:.6f}")
        print(f"[EMA ALIGNMENT]   EMA50: {ema50:.6f}")
        print(f"[EMA ALIGNMENT]   Price: {current_price:.6f}")
        print(f"[EMA ALIGNMENT]   Strength: {alignment_strength} ({total_separation:.2f}% separation)")
        
    elif ema_sequence_correct and not price_above_ema20:
        alignment_strength = "ema_aligned_but_price_below"
        trend_quality = "weak"
        print(f"[EMA ALIGNMENT] ⚠️ EMAs aligned but price below EMA20")
        
    elif not ema_sequence_correct and price_above_ema20:
        alignment_strength = "price_above_but_emas_misaligned"
        trend_quality = "weak"
        print(f"[EMA ALIGNMENT] ⚠️ Price above EMA20 but EMAs not aligned")
        
    else:
        print(f"[EMA ALIGNMENT] ❌ No bull alignment detected")
        print(f"[EMA ALIGNMENT]   EMA Sequence: {ema10:.6f} > {ema20:.6f} > {ema50:.6f} = {ema_sequence_correct}")
        print(f"[EMA ALIGNMENT]   Price > EMA20: {current_price:.6f} > {ema20:.6f} = {price_above_ema20}")
    
    return {
        "ema_alignment": ema_alignment,
        "ema_values": {
            "ema10": round(ema10, 6),
            "ema20": round(ema20, 6),
            "ema50": round(ema50, 6),
            "current_price": round(current_price, 6)
        },
        "alignment_strength": alignment_strength,
        "price_above_ema20": price_above_ema20,
        "trend_quality": trend_quality,
        "separation_percentage": round(((ema10 - ema50) / ema50) * 100, 2) if ema_alignment else 0
    }


def get_ema_trend_score_boost(alignment_result):
    """
    Calculate score boost based on EMA alignment quality
    
    Args:
        alignment_result: result from detect_ema_alignment()
    
    Returns:
        int: score boost (0-7 points)
    """
    if not alignment_result.get("ema_alignment", False):
        return 0
    
    # Base score for any EMA alignment
    base_score = 7
    
    # No additional scoring - keeping it simple as requested
    # Could be enhanced later with strength-based bonuses
    
    return base_score


def analyze_ema_trend_momentum(candle_data):
    """
    Additional analysis of EMA trend momentum and sustainability
    
    Args:
        candle_data: OHLCV candle data
    
    Returns:
        dict: Advanced EMA momentum analysis
    """
    if len(candle_data) < 50:
        return {"momentum_analysis": "insufficient_data"}
    
    close_prices = [float(candle[4]) for candle in candle_data]
    
    # Calculate EMAs for different periods
    current_ema20 = calculate_ema(close_prices, 20)
    previous_ema20 = calculate_ema(close_prices[:-5], 20)  # 5 candles ago
    
    if current_ema20 is None or previous_ema20 is None:
        return {"momentum_analysis": "calculation_error"}
    
    # EMA20 slope analysis
    ema20_slope = ((current_ema20 - previous_ema20) / previous_ema20) * 100
    
    # Momentum classification
    if ema20_slope > 1.0:
        momentum = "strong_up"
    elif ema20_slope > 0.3:
        momentum = "moderate_up"
    elif ema20_slope > 0:
        momentum = "weak_up"
    elif ema20_slope > -0.3:
        momentum = "sideways"
    else:
        momentum = "declining"
    
    return {
        "momentum_analysis": momentum,
        "ema20_slope_percent": round(ema20_slope, 3),
        "momentum_score": max(0, min(5, int(ema20_slope * 2))) if ema20_slope > 0 else 0
    }


def compute_ema_alignment_boost(candle_data, symbol=None):
    """
    Main function to compute EMA alignment boost for Trend Mode
    
    Args:
        candle_data: OHLCV candle data
        symbol: token symbol (optional)
    
    Returns:
        dict: {
            "ema_boost": int (0-7),
            "ema_analysis": dict,
            "summary_text": str
        }
    """
    if symbol:
        print(f"[EMA ALIGNMENT] Computing EMA alignment for {symbol}")
    
    # Detect EMA alignment
    alignment_result = detect_ema_alignment(candle_data)
    
    # Calculate score boost
    ema_boost = get_ema_trend_score_boost(alignment_result)
    
    # Generate summary text
    summary_text = ""
    if alignment_result.get("ema_alignment", False):
        strength = alignment_result.get("alignment_strength", "unknown")
        summary_text = f"EMA alignment confirmed ({strength})"
    else:
        summary_text = "No EMA alignment"
    
    # Add momentum analysis for aligned EMAs
    if alignment_result.get("ema_alignment", False):
        momentum_analysis = analyze_ema_trend_momentum(candle_data)
        alignment_result["momentum"] = momentum_analysis
    
    return {
        "ema_boost": ema_boost,
        "ema_analysis": alignment_result,
        "summary_text": summary_text
    }