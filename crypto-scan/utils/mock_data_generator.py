"""
Mock Data Generator for Chart Export Testing
Generates realistic cryptocurrency candle data for Computer Vision training
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import random


def generate_realistic_candles(
    symbol: str,
    count: int = 96,
    base_price: float = None,
    pattern: str = "trending_up"
) -> List[List]:
    """
    Generate realistic cryptocurrency candle data
    
    Args:
        symbol: Trading symbol for reference
        count: Number of candles to generate
        base_price: Starting price (auto-detected from symbol if None)
        pattern: Market pattern to simulate
        
    Returns:
        List of candles in format [timestamp, open, high, low, close, volume]
    """
    
    # Auto-detect base price from symbol
    if base_price is None:
        if 'BTC' in symbol:
            base_price = random.uniform(45000, 55000)
        elif 'ETH' in symbol:
            base_price = random.uniform(2800, 3200)
        elif 'BNB' in symbol:
            base_price = random.uniform(300, 400)
        elif 'ADA' in symbol:
            base_price = random.uniform(0.4, 0.6)
        elif 'SOL' in symbol:
            base_price = random.uniform(80, 120)
        else:
            base_price = random.uniform(1, 100)
    
    candles = []
    current_time = datetime.now()
    current_price = base_price
    
    # Pattern-specific parameters
    if pattern == "trending_up":
        trend_strength = 0.002  # 0.2% upward bias per candle
        volatility = 0.008      # 0.8% volatility
    elif pattern == "trending_down":
        trend_strength = -0.002
        volatility = 0.008
    elif pattern == "consolidation":
        trend_strength = 0.0
        volatility = 0.005
    elif pattern == "breakout":
        trend_strength = 0.001
        volatility = 0.015      # Higher volatility for breakout
    elif pattern == "pullback":
        trend_strength = -0.001
        volatility = 0.006
    else:
        trend_strength = 0.0
        volatility = 0.008
    
    for i in range(count):
        # Calculate timestamp (15-minute intervals, going backwards)
        timestamp = int((current_time - timedelta(minutes=15 * (count - 1 - i))).timestamp() * 1000)
        
        # Generate price movement
        trend_move = current_price * trend_strength
        random_move = np.random.normal(0, current_price * volatility)
        price_change = trend_move + random_move
        
        # Create OHLC for this candle
        open_price = current_price
        close_price = open_price + price_change
        
        # Generate realistic high/low based on open/close
        high_range = abs(close_price - open_price) * random.uniform(0.5, 2.0)
        low_range = abs(close_price - open_price) * random.uniform(0.5, 2.0)
        
        high_price = max(open_price, close_price) + high_range
        low_price = min(open_price, close_price) - low_range
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume (higher volume on bigger moves)
        base_volume = random.uniform(500, 2000)
        move_multiplier = 1 + abs(price_change / current_price) * 5
        volume = base_volume * move_multiplier
        
        candles.append([
            timestamp,
            round(open_price, 6),
            round(high_price, 6),
            round(low_price, 6),
            round(close_price, 6),
            round(volume, 2)
        ])
        
        current_price = close_price
        
        # Add some noise to trend for realism
        if i % 10 == 0:  # Every 10 candles, slightly modify trend
            trend_strength += random.uniform(-0.0005, 0.0005)
    
    return candles


def generate_pattern_specific_data(pattern_type: str, symbol: str = "MOCKUSDT") -> List[List]:
    """Generate candles for specific chart patterns"""
    
    if pattern_type == "breakout_continuation":
        # Consolidation followed by breakout
        consolidation = generate_realistic_candles(symbol, 60, pattern="consolidation")
        breakout = generate_realistic_candles(symbol, 36, 
                                            base_price=consolidation[-1][4], 
                                            pattern="breakout")
        return consolidation + breakout
        
    elif pattern_type == "pullback_setup":
        # Uptrend, pullback, then continuation
        uptrend = generate_realistic_candles(symbol, 40, pattern="trending_up")
        pullback = generate_realistic_candles(symbol, 30, 
                                            base_price=uptrend[-1][4], 
                                            pattern="pullback")
        continuation = generate_realistic_candles(symbol, 26,
                                                base_price=pullback[-1][4],
                                                pattern="trending_up")
        return uptrend + pullback + continuation
        
    elif pattern_type == "trend_exhaustion":
        # Strong trend followed by consolidation
        trend = generate_realistic_candles(symbol, 50, pattern="trending_up")
        exhaustion = generate_realistic_candles(symbol, 46,
                                              base_price=trend[-1][4],
                                              pattern="consolidation")
        return trend + exhaustion
        
    elif pattern_type == "fakeout_pattern":
        # False breakout followed by reversal
        buildup = generate_realistic_candles(symbol, 40, pattern="consolidation")
        fake_breakout = generate_realistic_candles(symbol, 20,
                                                 base_price=buildup[-1][4],
                                                 pattern="breakout")
        reversal = generate_realistic_candles(symbol, 36,
                                            base_price=fake_breakout[-1][4],
                                            pattern="trending_down")
        return buildup + fake_breakout + reversal
    
    else:
        # Default random pattern
        return generate_realistic_candles(symbol, 96, pattern="trending_up")


def create_training_dataset(num_charts_per_pattern: int = 5) -> dict:
    """Create a complete training dataset with various patterns"""
    
    patterns = [
        "breakout_continuation",
        "pullback_setup", 
        "trend_exhaustion",
        "fakeout_pattern"
    ]
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
    
    dataset = {}
    
    for pattern in patterns:
        dataset[pattern] = []
        
        for i in range(num_charts_per_pattern):
            symbol = symbols[i % len(symbols)]
            mock_symbol = f"{pattern.upper()}_{symbol}_{i+1}"
            
            candles = generate_pattern_specific_data(pattern, mock_symbol)
            dataset[pattern].append({
                "symbol": mock_symbol,
                "pattern": pattern,
                "candles": candles,
                "label": pattern.replace("_", "-")
            })
    
    return dataset