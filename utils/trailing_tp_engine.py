"""
Trailing TP Engine - Advanced Take Profit Forecasting
Prognozowanie TP1/TP2/TP3 w oparciu o RSI, momentum, VWAP, delta
"""

import numpy as np
from datetime import datetime, timezone


def calculate_momentum_delta(data, period=5):
    """Calculate momentum delta based on price and volume changes"""
    if len(data) < period:
        return 0
    
    recent_candles = data[-period:]
    
    # Price momentum (weighted by volume)
    price_changes = []
    volume_weights = []
    
    for i in range(1, len(recent_candles)):
        price_change = (recent_candles[i][4] - recent_candles[i-1][4]) / recent_candles[i-1][4]
        volume_weight = recent_candles[i][5]
        
        price_changes.append(price_change)
        volume_weights.append(volume_weight)
    
    if not price_changes:
        return 0
    
    # Volume-weighted momentum
    total_volume = sum(volume_weights)
    if total_volume == 0:
        return np.mean(price_changes)
    
    weighted_momentum = sum(pc * vw for pc, vw in zip(price_changes, volume_weights)) / total_volume
    return weighted_momentum


def calculate_vwap_distance(data, current_price):
    """Calculate distance from VWAP as momentum indicator"""
    if len(data) < 20:
        return 0
    
    # Calculate VWAP from last 20 candles
    volumes = np.array([candle[5] for candle in data[-20:]])
    typical_prices = np.array([(candle[2] + candle[3] + candle[4]) / 3 for candle in data[-20:]])
    
    total_volume = np.sum(volumes)
    if total_volume == 0:
        return 0
    
    vwap = np.sum(typical_prices * volumes) / total_volume
    
    # Distance as percentage
    vwap_distance = (current_price - vwap) / vwap if vwap > 0 else 0
    return vwap_distance


def calculate_rsi_projection(data, period=14):
    """Calculate RSI and project potential continuation"""
    if len(data) < period + 1:
        return 50, 0
    
    closes = np.array([candle[4] for candle in data[-(period+1):]])
    deltas = np.diff(closes)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    # Project RSI continuation potential
    rsi_momentum = 0
    if rsi > 70:
        rsi_momentum = min(rsi - 70, 20) / 20  # 0-1 scale for overbought momentum
    elif rsi > 50:
        rsi_momentum = (rsi - 50) / 20  # 0-1 scale for bullish momentum
    
    return rsi, rsi_momentum


def compute_trailing_tp_levels(data, symbol=None, trend_score=0):
    """
    Advanced TP level computation based on trend strength and technical indicators
    
    Args:
        data: OHLCV candle data
        symbol: token symbol
        trend_score: current trend score (0-50)
    
    Returns:
        dict: {
            "TP1": float,     # Conservative target
            "TP2": float,     # Moderate target  
            "TP3": float,     # Aggressive target
            "trailing_stop": float,  # Dynamic trailing stop
            "confidence": float,     # 0-100 confidence level
            "time_horizon": str      # Expected timeframe
        }
    """
    print(f"[TRAILING TP] Computing advanced TP levels for {symbol}")
    
    if len(data) < 20:
        return {
            "TP1": 3.0, "TP2": 8.0, "TP3": 15.0,
            "trailing_stop": 2.0, "confidence": 30,
            "time_horizon": "4-8 hours"
        }
    
    current_price = data[-1][4]
    
    # Calculate technical indicators
    rsi, rsi_momentum = calculate_rsi_projection(data)
    momentum_delta = calculate_momentum_delta(data)
    vwap_distance = calculate_vwap_distance(data, current_price)
    
    # Base TP levels based on trend score
    base_tp1 = 2.5 + (trend_score / 50) * 2.5  # 2.5-5%
    base_tp2 = 6.0 + (trend_score / 50) * 6.0  # 6-12%
    base_tp3 = 12.0 + (trend_score / 50) * 8.0  # 12-20%
    
    # RSI momentum adjustments
    rsi_multiplier = 1.0 + (rsi_momentum * 0.5)  # Up to 50% boost
    
    # VWAP distance adjustments
    vwap_multiplier = 1.0
    if vwap_distance > 0.02:  # Above VWAP by 2%+
        vwap_multiplier = 1.0 + min(vwap_distance * 10, 0.3)  # Up to 30% boost
    
    # Momentum delta adjustments
    momentum_multiplier = 1.0
    if momentum_delta > 0:
        momentum_multiplier = 1.0 + min(momentum_delta * 20, 0.4)  # Up to 40% boost
    
    # Apply all multipliers
    total_multiplier = rsi_multiplier * vwap_multiplier * momentum_multiplier
    
    tp1 = base_tp1 * total_multiplier
    tp2 = base_tp2 * total_multiplier
    tp3 = base_tp3 * total_multiplier
    
    # Trailing stop calculation
    trailing_stop = max(1.5, tp1 * 0.6)  # 60% of TP1, minimum 1.5%
    
    # Confidence calculation
    confidence = min(100, 40 + trend_score + (rsi_momentum * 20) + (momentum_delta * 100))
    
    # Time horizon based on momentum
    if momentum_delta > 0.05:
        time_horizon = "2-4 hours"
    elif momentum_delta > 0.02:
        time_horizon = "4-8 hours"
    else:
        time_horizon = "8-16 hours"
    
    result = {
        "TP1": round(tp1, 2),
        "TP2": round(tp2, 2), 
        "TP3": round(tp3, 2),
        "trailing_stop": round(trailing_stop, 2),
        "confidence": round(confidence, 1),
        "time_horizon": time_horizon
    }
    
    print(f"[TRAILING TP] {symbol}: TP1={result['TP1']}%, TP2={result['TP2']}%, TP3={result['TP3']}% (Confidence: {result['confidence']}%)")
    
    return result


def update_trailing_stop(symbol, entry_price, current_price, tp_levels):
    """
    Dynamic trailing stop adjustment based on price movement
    
    Args:
        symbol: token symbol
        entry_price: original entry price
        current_price: current market price
        tp_levels: TP levels from compute_trailing_tp_levels
    
    Returns:
        dict: Updated trailing stop information
    """
    if not entry_price or not current_price or entry_price <= 0:
        return {"error": "Invalid price data"}
    
    # Calculate current profit percentage
    profit_pct = ((current_price - entry_price) / entry_price) * 100
    
    # Dynamic trailing stop based on profit level
    if profit_pct >= tp_levels["TP2"]:
        # Above TP2: trail at TP1 level
        trailing_stop_pct = tp_levels["TP1"]
    elif profit_pct >= tp_levels["TP1"]:
        # Above TP1: trail at 50% of TP1
        trailing_stop_pct = tp_levels["TP1"] * 0.5
    else:
        # Below TP1: use original trailing stop
        trailing_stop_pct = tp_levels["trailing_stop"]
    
    stop_price = entry_price * (1 + trailing_stop_pct / 100)
    
    return {
        "trailing_stop_pct": trailing_stop_pct,
        "stop_price": stop_price,
        "current_profit_pct": profit_pct,
        "recommendation": "HOLD" if current_price > stop_price else "EXIT"
    }