"""
HTF Support/Resistance Detection - Higher Timeframe Level Analysis
Optional enhancement module for detecting key S/R levels on higher timeframes
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def detect_htf_levels(htf_candles: List[Dict], current_price: float, 
                     timeframe: str = "1H") -> Dict:
    """
    Detect key support and resistance levels on higher timeframe
    
    Args:
        htf_candles: HTF candle data
        current_price: Current market price
        timeframe: Timeframe identifier
        
    Returns:
        dict: {
            "key_resistance": List[float],
            "key_support": List[float],
            "nearest_resistance": float,
            "nearest_support": float,
            "price_position": str,     # "at_resistance", "at_support", "middle"
            "level_strength": Dict,   # Strength rating for each level
            "breakout_potential": float # 0.0-1.0 breakout probability
        }
    """
    
    try:
        if not htf_candles or len(htf_candles) < 20:
            logger.warning(f"[HTF S/R] Insufficient HTF data: {len(htf_candles) if htf_candles else 0}")
            return _default_sr_levels(current_price)
            
        # Format candles
        formatted_candles = _format_htf_candles(htf_candles)
        
        if len(formatted_candles) < 20:
            return _default_sr_levels(current_price)
            
        # Use more candles for S/R detection (up to 100)
        analysis_window = min(100, len(formatted_candles))
        candles = formatted_candles[-analysis_window:]
        
        logger.info(f"[HTF S/R] Analyzing {len(candles)} {timeframe} candles for S/R levels")
        
        # Extract price arrays
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        closes = np.array([c['close'] for c in candles])
        
        # === LEVEL DETECTION ===
        
        # 1. Swing highs and lows
        swing_levels = _detect_swing_levels(highs, lows, window=5)
        
        # 2. Volume-weighted levels (if volume available)
        volume_levels = _detect_volume_levels(candles)
        
        # 3. Round number levels
        round_levels = _detect_round_numbers(current_price)
        
        # 4. Fibonacci retracement levels
        fib_levels = _detect_fibonacci_levels(highs, lows, current_price)
        
        # === LEVEL CONSOLIDATION ===
        
        all_resistance = []
        all_support = []
        
        # Combine all resistance levels
        all_resistance.extend(swing_levels['resistance'])
        all_resistance.extend(volume_levels['resistance'])
        all_resistance.extend([r for r in round_levels if r > current_price])
        all_resistance.extend(fib_levels['resistance'])
        
        # Combine all support levels
        all_support.extend(swing_levels['support'])
        all_support.extend(volume_levels['support'])
        all_support.extend([s for s in round_levels if s < current_price])
        all_support.extend(fib_levels['support'])
        
        # === LEVEL FILTERING AND RANKING ===
        
        # Filter and rank levels
        key_resistance = _filter_and_rank_levels(
            all_resistance, current_price, direction='resistance'
        )
        
        key_support = _filter_and_rank_levels(
            all_support, current_price, direction='support'
        )
        
        # === LEVEL ANALYSIS ===
        
        # Find nearest levels
        nearest_resistance = min(key_resistance) if key_resistance else current_price * 1.1
        nearest_support = max(key_support) if key_support else current_price * 0.9
        
        # Determine price position
        price_position = _determine_price_position(
            current_price, nearest_resistance, nearest_support
        )
        
        # Calculate level strengths
        level_strength = _calculate_level_strengths(
            key_resistance + key_support, candles, current_price
        )
        
        # Calculate breakout potential
        breakout_potential = _calculate_breakout_potential(
            current_price, nearest_resistance, nearest_support, 
            level_strength, closes
        )
        
        result = {
            'key_resistance': sorted(key_resistance, reverse=True)[:5],  # Top 5
            'key_support': sorted(key_support, reverse=True)[:5],        # Top 5
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'price_position': price_position,
            'level_strength': level_strength,
            'breakout_potential': round(breakout_potential, 3),
            'timeframe': timeframe,
            'analysis_window': len(candles),
            'current_price': current_price
        }
        
        logger.info(f"[HTF S/R] {timeframe}: Found {len(key_resistance)} resistance, "
                   f"{len(key_support)} support levels")
        logger.info(f"[HTF S/R] Price position: {price_position}, "
                   f"Breakout potential: {breakout_potential:.2f}")
        
        return result
        
    except Exception as e:
        logger.error(f"[HTF S/R ERROR] Level detection failed: {e}")
        return _default_sr_levels(current_price)

def _format_htf_candles(candles: List) -> List[Dict]:
    """Format HTF candles to standardized dict format"""
    formatted = []
    
    for candle in candles:
        try:
            if isinstance(candle, dict):
                if all(key in candle for key in ['open', 'high', 'low', 'close']):
                    formatted.append(candle)
            elif isinstance(candle, (list, tuple)) and len(candle) >= 5:
                formatted.append({
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]) if len(candle) > 5 else 1000000
                })
        except (ValueError, TypeError, IndexError):
            continue
            
    return formatted

def _detect_swing_levels(highs: np.ndarray, lows: np.ndarray, window: int = 5) -> Dict:
    """Detect swing high and low levels"""
    
    resistance_levels = []
    support_levels = []
    
    # Find swing highs
    for i in range(window, len(highs) - window):
        if highs[i] == np.max(highs[i-window:i+window+1]):
            resistance_levels.append(highs[i])
            
    # Find swing lows
    for i in range(window, len(lows) - window):
        if lows[i] == np.min(lows[i-window:i+window+1]):
            support_levels.append(lows[i])
    
    return {
        'resistance': resistance_levels,
        'support': support_levels
    }

def _detect_volume_levels(candles: List[Dict]) -> Dict:
    """Detect levels with high volume activity"""
    
    resistance_levels = []
    support_levels = []
    
    # Group candles by price ranges and find high-volume areas
    try:
        prices = [c['close'] for c in candles]
        volumes = [c.get('volume', 1000000) for c in candles]
        
        if not prices or not volumes:
            return {'resistance': [], 'support': []}
            
        min_price = min(prices)
        max_price = max(prices)
        
        # Create price buckets
        num_buckets = 20
        bucket_size = (max_price - min_price) / num_buckets
        
        volume_by_bucket = {}
        
        for price, volume in zip(prices, volumes):
            bucket = int((price - min_price) / bucket_size)
            bucket = min(bucket, num_buckets - 1)  # Ensure within bounds
            
            if bucket not in volume_by_bucket:
                volume_by_bucket[bucket] = {'total_volume': 0, 'price_sum': 0, 'count': 0}
                
            volume_by_bucket[bucket]['total_volume'] += volume
            volume_by_bucket[bucket]['price_sum'] += price
            volume_by_bucket[bucket]['count'] += 1
        
        # Find high-volume buckets
        avg_volume = np.mean([b['total_volume'] for b in volume_by_bucket.values()])
        
        for bucket, data in volume_by_bucket.items():
            if data['total_volume'] > avg_volume * 1.5:  # 50% above average
                avg_price = data['price_sum'] / data['count']
                
                # Classify as support or resistance based on recent price action
                recent_price = prices[-1]
                if avg_price > recent_price:
                    resistance_levels.append(avg_price)
                else:
                    support_levels.append(avg_price)
                    
    except Exception as e:
        logger.warning(f"[VOLUME LEVELS] Detection failed: {e}")
        
    return {
        'resistance': resistance_levels,
        'support': support_levels
    }

def _detect_round_numbers(current_price: float) -> List[float]:
    """Detect psychologically important round number levels"""
    
    round_levels = []
    
    # Determine appropriate round number intervals based on price
    if current_price > 100000:
        intervals = [10000, 5000, 1000]
    elif current_price > 10000:
        intervals = [1000, 500, 100]
    elif current_price > 1000:
        intervals = [100, 50, 10]
    elif current_price > 100:
        intervals = [10, 5, 1]
    elif current_price > 10:
        intervals = [1, 0.5, 0.1]
    else:
        intervals = [0.1, 0.05, 0.01]
    
    # Generate round numbers within reasonable range
    for interval in intervals:
        # Find round numbers above and below current price
        lower_bound = current_price * 0.8
        upper_bound = current_price * 1.2
        
        # Round current price down to nearest interval
        base = int(current_price / interval) * interval
        
        # Generate levels
        for i in range(-3, 4):  # 7 levels around current price
            level = base + (i * interval)
            if lower_bound <= level <= upper_bound and level != current_price:
                round_levels.append(level)
    
    return round_levels

def _detect_fibonacci_levels(highs: np.ndarray, lows: np.ndarray, 
                           current_price: float) -> Dict:
    """Detect Fibonacci retracement levels"""
    
    try:
        # Find recent significant swing high and low
        recent_window = min(50, len(highs))
        recent_highs = highs[-recent_window:]
        recent_lows = lows[-recent_window:]
        
        swing_high = np.max(recent_highs)
        swing_low = np.min(recent_lows)
        
        # Calculate Fibonacci levels
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        range_size = swing_high - swing_low
        
        fib_levels = []
        for ratio in fib_ratios:
            level = swing_low + (range_size * ratio)
            fib_levels.append(level)
        
        # Also add extension levels
        extension_ratios = [1.272, 1.618]
        for ratio in extension_ratios:
            level = swing_high + (range_size * (ratio - 1))
            fib_levels.append(level)
        
        # Classify as support or resistance
        resistance_levels = [l for l in fib_levels if l > current_price]
        support_levels = [l for l in fib_levels if l < current_price]
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
        
    except Exception as e:
        logger.warning(f"[FIBONACCI] Level calculation failed: {e}")
        return {'resistance': [], 'support': []}

def _filter_and_rank_levels(levels: List[float], current_price: float, 
                           direction: str) -> List[float]:
    """Filter and rank S/R levels by proximity and significance"""
    
    if not levels:
        return []
    
    # Remove duplicates and near-duplicates
    unique_levels = []
    sorted_levels = sorted(levels)
    
    for level in sorted_levels:
        # Check if this level is significantly different from existing ones
        is_unique = True
        for existing in unique_levels:
            if abs(level - existing) / current_price < 0.01:  # Within 1%
                is_unique = False
                break
        
        if is_unique:
            unique_levels.append(level)
    
    # Filter by proximity to current price
    if direction == 'resistance':
        # Keep levels above current price within reasonable range
        filtered = [l for l in unique_levels 
                   if current_price < l <= current_price * 1.15]
        filtered.sort()  # Nearest first
    else:  # support
        # Keep levels below current price within reasonable range
        filtered = [l for l in unique_levels 
                   if current_price * 0.85 <= l < current_price]
        filtered.sort(reverse=True)  # Nearest first
    
    # Return top 5 most significant levels
    return filtered[:5]

def _determine_price_position(current_price: float, nearest_resistance: float, 
                            nearest_support: float) -> str:
    """Determine price position relative to key levels"""
    
    resistance_distance = (nearest_resistance - current_price) / current_price
    support_distance = (current_price - nearest_support) / current_price
    
    if resistance_distance < 0.02:  # Within 2% of resistance
        return "at_resistance"
    elif support_distance < 0.02:  # Within 2% of support
        return "at_support"
    elif resistance_distance < support_distance:
        return "near_resistance"
    elif support_distance < resistance_distance:
        return "near_support"
    else:
        return "middle_range"

def _calculate_level_strengths(levels: List[float], candles: List[Dict], 
                             current_price: float) -> Dict:
    """Calculate strength rating for each S/R level"""
    
    level_strength = {}
    
    for level in levels:
        touches = 0
        volume_at_level = 0
        
        # Count touches and volume at this level
        for candle in candles:
            high = candle['high']
            low = candle['low']
            volume = candle.get('volume', 1000000)
            
            # Check if candle touched this level (within 1%)
            level_range = level * 0.01
            if low <= level + level_range and high >= level - level_range:
                touches += 1
                volume_at_level += volume
        
        # Calculate strength score
        touch_score = min(touches / 5, 1.0)  # Normalize to max 5 touches
        volume_score = min(volume_at_level / 10000000, 1.0)  # Normalize volume
        
        # Distance penalty (farther levels are less significant)
        distance = abs(level - current_price) / current_price
        distance_penalty = max(0, 1 - distance * 5)  # Penalty for distance > 20%
        
        strength = (touch_score * 0.5 + volume_score * 0.3 + distance_penalty * 0.2)
        level_strength[level] = round(strength, 3)
    
    return level_strength

def _calculate_breakout_potential(current_price: float, nearest_resistance: float,
                                nearest_support: float, level_strength: Dict,
                                closes: np.ndarray) -> float:
    """Calculate probability of breakout from current range"""
    
    try:
        # 1. Price momentum
        recent_momentum = (closes[-1] / closes[-10] - 1) if len(closes) >= 10 else 0
        momentum_score = min(abs(recent_momentum) * 5, 1.0)  # Normalize momentum
        
        # 2. Distance to levels
        resistance_distance = (nearest_resistance - current_price) / current_price
        support_distance = (current_price - nearest_support) / current_price
        
        min_distance = min(resistance_distance, support_distance)
        proximity_score = max(0, 1 - min_distance * 20)  # Higher score for closer levels
        
        # 3. Level strength (weaker levels more likely to break)
        resistance_strength = level_strength.get(nearest_resistance, 0.5)
        support_strength = level_strength.get(nearest_support, 0.5)
        
        # Average strength (lower strength = higher breakout potential)
        avg_strength = (resistance_strength + support_strength) / 2
        weakness_score = 1 - avg_strength
        
        # 4. Volatility (higher volatility = higher breakout potential)
        if len(closes) >= 20:
            volatility = np.std(closes[-20:]) / np.mean(closes[-20:])
            volatility_score = min(volatility * 10, 1.0)
        else:
            volatility_score = 0.5
        
        # Combine factors
        breakout_potential = (
            momentum_score * 0.3 +
            proximity_score * 0.25 +
            weakness_score * 0.25 +
            volatility_score * 0.2
        )
        
        return max(0.0, min(1.0, breakout_potential))
        
    except Exception as e:
        logger.warning(f"[BREAKOUT CALC] Calculation failed: {e}")
        return 0.5

def _default_sr_levels(current_price: float) -> Dict:
    """Return default S/R levels when analysis fails"""
    return {
        'key_resistance': [current_price * 1.05],
        'key_support': [current_price * 0.95],
        'nearest_resistance': current_price * 1.05,
        'nearest_support': current_price * 0.95,
        'price_position': 'middle_range',
        'level_strength': {},
        'breakout_potential': 0.5,
        'timeframe': 'unknown',
        'analysis_window': 0,
        'current_price': current_price
    }

def test_htf_sr_detection():
    """Test HTF S/R detection with sample data"""
    
    # Create sample data with clear S/R levels
    test_candles = []
    base_price = 50000
    
    # Create data with resistance around 52000 and support around 48000
    for i in range(60):
        if i < 20:
            # First phase: range between 48000-52000
            price = 48000 + (i % 10) * 400  # Oscillate in range
        elif i < 40:
            # Second phase: break above resistance
            price = 52000 + (i - 20) * 100  # Gradual rise
        else:
            # Third phase: pullback to test old resistance as support
            price = 53000 - (i - 40) * 50  # Pullback
        
        test_candles.append({
            'timestamp': 1640995200 + i * 3600,
            'open': price - 50,
            'high': price + 200,
            'low': price - 200,
            'close': price + 50,
            'volume': 1000000 + i * 10000
        })
    
    current_price = 52500
    
    print("🧪 Testing HTF S/R Detection:")
    print("=" * 40)
    
    result = detect_htf_levels(test_candles, current_price, "1H")
    
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Nearest Resistance: ${result['nearest_resistance']:,.2f}")
    print(f"Nearest Support: ${result['nearest_support']:,.2f}")
    print(f"Price Position: {result['price_position']}")
    print(f"Breakout Potential: {result['breakout_potential']:.2f}")
    print(f"Key Resistance Levels: {[f'${r:,.0f}' for r in result['key_resistance']]}")
    print(f"Key Support Levels: {[f'${s:,.0f}' for s in result['key_support']]}")

if __name__ == "__main__":
    test_htf_sr_detection()