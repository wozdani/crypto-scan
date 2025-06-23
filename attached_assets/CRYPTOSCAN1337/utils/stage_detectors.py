import json
import math
from datetime import datetime, timedelta

def detect_stage_minus2(symbol, data):
    """
    Stage -2: Early accumulation detection
    Analyzes volume patterns, price stability, and market cap changes
    Returns: (stage2_pass, signals, inflow)
    """
    try:
        if not data or not data.get('price'):
            return False, {}, 0
            
        price_data = data['price']
        volume_24h = price_data.get('volume_24h', 0)
        price_change_24h = price_data.get('price_change_24h', 0)
        market_cap = price_data.get('market_cap', 0)
        
        signals = {
            'volume_spike': False,
            'price_stability': False,
            'accumulation_pattern': False,
            'market_cap_growth': False
        }
        
        # Volume spike detection (volume > average expected)
        if volume_24h > 0:
            # Simplified volume spike detection
            # In production, this would compare against historical averages
            signals['volume_spike'] = volume_24h > (market_cap * 0.1) if market_cap > 0 else False
            
        # Price stability (low volatility but positive trend)
        if abs(price_change_24h) < 5 and price_change_24h > -2:
            signals['price_stability'] = True
            
        # Accumulation pattern (consistent buying pressure)
        high_24h = price_data.get('high_24h', 0)
        low_24h = price_data.get('low_24h', 0)
        current_price = price_data.get('price', 0)
        
        if high_24h > 0 and low_24h > 0:
            price_position = (current_price - low_24h) / (high_24h - low_24h)
            signals['accumulation_pattern'] = price_position > 0.6  # Trading in upper 40%
            
        # Market cap growth assessment
        if market_cap > 1000000:  # At least $1M market cap
            signals['market_cap_growth'] = True
            
        # Calculate inflow estimation
        inflow = estimate_money_inflow(data)
        
        # Stage -2 passes if at least 2 signals are positive
        stage2_pass = sum(signals.values()) >= 2
        
        return stage2_pass, signals, inflow
        
    except Exception as e:
        print(f"❌ Error in stage -2 detection for {symbol}: {e}")
        return False, {}, 0

def detect_stage_minus2_2(symbol):
    """
    Stage -2.2: Advanced pattern recognition
    Detects specific chart patterns and technical indicators
    Returns: event_tags list
    """
    try:
        event_tags = []
        
        # In a production system, this would analyze:
        # - Chart patterns (cup and handle, ascending triangles)
        # - Technical indicators (RSI, MACD, Bollinger Bands)
        # - Social sentiment changes
        # - Whale wallet activities
        
        # Simplified implementation with random patterns for demonstration
        import random
        random.seed(hash(symbol) % 1000)  # Deterministic randomness based on symbol
        
        patterns = [
            'ascending_triangle',
            'volume_breakout',
            'support_bounce',
            'whale_accumulation',
            'social_buzz_increase'
        ]
        
        # Simulate pattern detection
        for pattern in patterns:
            if random.random() > 0.7:  # 30% chance of detecting each pattern
                event_tags.append(pattern)
                
        return event_tags
        
    except Exception as e:
        print(f"❌ Error in stage -2.2 detection for {symbol}: {e}")
        return []

def detect_stage_minus1(signals):
    """
    Stage -1: Pre-breakout confirmation
    Compresses and validates signals from previous stages
    Returns: compressed_signals dict
    """
    try:
        compressed = {
            'strength': 0,
            'confidence': 0,
            'momentum': False,
            'volume_confirmed': False,
            'technical_alignment': False
        }
        
        if not signals:
            return compressed
            
        # Calculate signal strength
        positive_signals = sum(1 for v in signals.values() if v)
        total_signals = len(signals)
        
        if total_signals > 0:
            compressed['strength'] = (positive_signals / total_signals) * 100
            
        # Confidence calculation
        key_signals = ['volume_spike', 'accumulation_pattern']
        key_positive = sum(1 for key in key_signals if signals.get(key, False))
        compressed['confidence'] = (key_positive / len(key_signals)) * 100
        
        # Momentum detection
        compressed['momentum'] = compressed['strength'] > 60
        
        # Volume confirmation
        compressed['volume_confirmed'] = signals.get('volume_spike', False)
        
        # Technical alignment
        compressed['technical_alignment'] = (
            signals.get('price_stability', False) and 
            signals.get('accumulation_pattern', False)
        )
        
        return compressed
        
    except Exception as e:
        print(f"❌ Error in stage -1 detection: {e}")
        return {'strength': 0, 'confidence': 0, 'momentum': False, 
                'volume_confirmed': False, 'technical_alignment': False}

def detect_stage_1g(symbol, data, event_tags):
    """
    Stage 1G: Breakout initiation
    Detects the beginning of significant price movement
    Returns: stage1g_active boolean
    """
    try:
        if not data or not data.get('price'):
            return False
            
        price_data = data['price']
        price_change_24h = price_data.get('price_change_24h', 0)
        volume_24h = price_data.get('volume_24h', 0)
        market_cap = price_data.get('market_cap', 0)
        
        conditions = []
        
        # Price breakout condition
        if price_change_24h > 5:  # 5% increase
            conditions.append(True)
        else:
            conditions.append(False)
            
        # Volume surge condition
        if volume_24h > 0 and market_cap > 0:
            volume_ratio = volume_24h / market_cap
            conditions.append(volume_ratio > 0.15)  # High volume relative to market cap
        else:
            conditions.append(False)
            
        # Event tags confirmation
        strong_patterns = ['ascending_triangle', 'volume_breakout', 'whale_accumulation']
        has_strong_pattern = any(tag in event_tags for tag in strong_patterns)
        conditions.append(has_strong_pattern)
        
        # Trend consistency
        high_24h = price_data.get('high_24h', 0)
        current_price = price_data.get('price', 0)
        if high_24h > 0:
            near_high = (current_price / high_24h) > 0.95  # Trading near 24h high
            conditions.append(near_high)
        else:
            conditions.append(False)
            
        # Stage 1G is active if at least 3 conditions are met
        stage1g_active = sum(conditions) >= 3
        
        return stage1g_active
        
    except Exception as e:
        print(f"❌ Error in stage 1G detection for {symbol}: {e}")
        return False

def estimate_money_inflow(data):
    """
    Estimate money inflow based on volume and price action
    Returns: estimated inflow in USD
    """
    try:
        price_data = data.get('price', {})
        volume_24h = price_data.get('volume_24h', 0)
        price_change_24h = price_data.get('price_change_24h', 0)
        
        # Simple inflow estimation
        # Positive price change with volume suggests buying pressure
        if price_change_24h > 0 and volume_24h > 0:
            # Estimate net inflow as percentage of volume based on price performance
            inflow_ratio = min(price_change_24h / 100, 0.5)  # Cap at 50%
            estimated_inflow = volume_24h * inflow_ratio
            return estimated_inflow
        
        return 0
        
    except Exception as e:
        print(f"❌ Error estimating money inflow: {e}")
        return 0

def analyze_market_structure(data):
    """
    Analyze overall market structure for context
    Returns: market_structure dict
    """
    try:
        structure = {
            'trend': 'neutral',
            'strength': 0,
            'support_level': 0,
            'resistance_level': 0
        }
        
        if not data or not data.get('price'):
            return structure
            
        price_data = data['price']
        current_price = price_data.get('price', 0)
        high_24h = price_data.get('high_24h', 0)
        low_24h = price_data.get('low_24h', 0)
        price_change_24h = price_data.get('price_change_24h', 0)
        
        # Determine trend
        if price_change_24h > 2:
            structure['trend'] = 'bullish'
        elif price_change_24h < -2:
            structure['trend'] = 'bearish'
        else:
            structure['trend'] = 'neutral'
            
        # Calculate trend strength
        structure['strength'] = min(abs(price_change_24h), 20) / 20 * 100
        
        # Estimate support and resistance
        if high_24h > 0 and low_24h > 0:
            structure['support_level'] = low_24h
            structure['resistance_level'] = high_24h
            
        return structure
        
    except Exception as e:
        print(f"❌ Error analyzing market structure: {e}")
        return {'trend': 'neutral', 'strength': 0, 'support_level': 0, 'resistance_level': 0}
