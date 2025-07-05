"""
Scenario Evaluator - Module 4 Future Scenario Mapping
Analyzes current candle behavior to predict future price scenarios (bull/bear/neutral)
"""

from typing import Dict, List, Optional
import logging

def evaluate_future_scenario(candle: Dict, ema: float = None, current_price: float = None) -> Dict:
    """
    Evaluates possible scenario for current candle: bull_case, bear_case, neutral
    
    Args:
        candle: Current candle data with OHLC and volume
        ema: EMA reference level for position analysis
        current_price: Current/live price for real-time analysis
        
    Returns:
        Dictionary with scenario analysis results
    """
    try:
        # Extract candle data with safe fallbacks
        high = float(candle.get("high", 0))
        low = float(candle.get("low", 0))
        open_price = float(candle.get("open", 0))
        close = float(candle.get("close", 0))
        volume = float(candle.get("volume", 0))
        
        # Use current price if available for real-time analysis
        if current_price:
            close = current_price
        
        # Validate data
        if high <= 0 or low <= 0 or open_price <= 0 or close <= 0:
            return {
                'scenario': 'neutral',
                'confidence': 0.0,
                'reasoning': 'Invalid candle data',
                'bull_signals': [],
                'bear_signals': [],
                'neutral_signals': ['invalid_data']
            }
        
        # Calculate candle structure metrics
        midpoint = (high + low) / 2
        wick_up = high - max(open_price, close)
        wick_down = min(open_price, close) - low
        body = abs(close - open_price)
        full_range = high - low + 1e-9  # Avoid division by zero
        
        # Calculate ratios
        wick_up_ratio = wick_up / full_range
        wick_down_ratio = wick_down / full_range
        body_ratio = body / full_range
        
        # Initialize signal tracking
        bull_signals = []
        bear_signals = []
        neutral_signals = []
        
        # Analyze candle structure
        if body_ratio > 0.4:
            if close > open_price:
                bull_signals.append(f'strong_green_body_{body_ratio:.2f}')
            else:
                bear_signals.append(f'strong_red_body_{body_ratio:.2f}')
        else:
            neutral_signals.append(f'small_body_{body_ratio:.2f}')
        
        # Analyze wick patterns
        if wick_up_ratio < 0.2:
            bull_signals.append(f'minimal_upper_wick_{wick_up_ratio:.2f}')
        elif wick_up_ratio > 0.4:
            bear_signals.append(f'long_upper_wick_{wick_up_ratio:.2f}')
        
        if wick_down_ratio < 0.2:
            bear_signals.append(f'minimal_lower_wick_{wick_down_ratio:.2f}')
        elif wick_down_ratio > 0.4:
            bull_signals.append(f'long_lower_wick_{wick_down_ratio:.2f}')
        
        # EMA position analysis if available
        if ema and ema > 0:
            if close > ema:
                distance_ratio = (close - ema) / ema
                if distance_ratio > 0.02:  # More than 2% above EMA
                    bull_signals.append(f'strong_above_ema_{distance_ratio:.3f}')
                else:
                    bull_signals.append(f'above_ema_{distance_ratio:.3f}')
            elif close < ema:
                distance_ratio = (ema - close) / ema
                if distance_ratio > 0.02:  # More than 2% below EMA
                    bear_signals.append(f'strong_below_ema_{distance_ratio:.3f}')
                else:
                    bear_signals.append(f'below_ema_{distance_ratio:.3f}')
            else:
                neutral_signals.append('at_ema')
        
        # Volume analysis (if available)
        if volume > 0:
            # We'd need historical volume for comparison, but we can analyze relative to candle size
            volume_intensity = volume / (full_range + 1e-9)  # Volume per price unit
            if volume_intensity > 1000000:  # High volume relative to range
                if close > open_price:
                    bull_signals.append(f'high_volume_green_{volume_intensity:.0f}')
                else:
                    bear_signals.append(f'high_volume_red_{volume_intensity:.0f}')
        
        # Determine scenario based on signal strength
        bull_strength = len(bull_signals)
        bear_strength = len(bear_signals)
        neutral_strength = len(neutral_signals)
        
        # Enhanced scenario determination with EMA context
        if ema and ema > 0:
            # EMA-based scenario logic
            if close > ema and wick_up_ratio < 0.2 and body_ratio > 0.4:
                scenario = 'bull_case'
                confidence = min(0.9, 0.6 + bull_strength * 0.1)
                reasoning = f'Strong candle above EMA: body_ratio={body_ratio:.2f}, wick_up={wick_up_ratio:.2f}'
            elif close < ema and wick_down_ratio < 0.2 and body_ratio > 0.4:
                scenario = 'bear_case'
                confidence = min(0.9, 0.6 + bear_strength * 0.1)
                reasoning = f'Strong candle below EMA: body_ratio={body_ratio:.2f}, wick_down={wick_down_ratio:.2f}'
            else:
                scenario = 'neutral'
                confidence = 0.3 + neutral_strength * 0.1
                reasoning = f'Mixed signals or weak structure: bull={bull_strength}, bear={bear_strength}'
        else:
            # Pure candle structure analysis without EMA
            if bull_strength > bear_strength and bull_strength > 1:
                scenario = 'bull_case'
                confidence = min(0.8, 0.5 + bull_strength * 0.1)
                reasoning = f'Bull signals dominate: {bull_strength} vs {bear_strength}'
            elif bear_strength > bull_strength and bear_strength > 1:
                scenario = 'bear_case'
                confidence = min(0.8, 0.5 + bear_strength * 0.1)
                reasoning = f'Bear signals dominate: {bear_strength} vs {bull_strength}'
            else:
                scenario = 'neutral'
                confidence = 0.3
                reasoning = f'Balanced or weak signals: bull={bull_strength}, bear={bear_strength}'
        
        return {
            'scenario': scenario,
            'confidence': round(confidence, 3),
            'reasoning': reasoning,
            'bull_signals': bull_signals,
            'bear_signals': bear_signals,
            'neutral_signals': neutral_signals,
            'candle_metrics': {
                'body_ratio': round(body_ratio, 3),
                'wick_up_ratio': round(wick_up_ratio, 3),
                'wick_down_ratio': round(wick_down_ratio, 3),
                'ema_position': 'above' if ema and close > ema else 'below' if ema and close < ema else 'no_ema'
            }
        }
        
    except Exception as e:
        return {
            'scenario': 'neutral',
            'confidence': 0.0,
            'reasoning': f'Evaluation error: {str(e)}',
            'bull_signals': [],
            'bear_signals': [],
            'neutral_signals': ['evaluation_error'],
            'candle_metrics': {}
        }

def analyze_candle_momentum(candles: List[Dict], lookback: int = 3) -> Dict:
    """
    Analyzes momentum from recent candles to support scenario evaluation
    
    Args:
        candles: List of recent candle data
        lookback: Number of candles to analyze for momentum
        
    Returns:
        Dictionary with momentum analysis
    """
    try:
        if not candles or len(candles) < 2:
            return {
                'momentum': 'neutral',
                'strength': 0.0,
                'reasoning': 'Insufficient candle data'
            }
        
        # Use last 'lookback' candles
        recent_candles = candles[-lookback:] if len(candles) >= lookback else candles
        
        # Calculate price momentum
        first_close = float(recent_candles[0].get('close', 0))
        last_close = float(recent_candles[-1].get('close', 0))
        
        if first_close <= 0 or last_close <= 0:
            return {
                'momentum': 'neutral',
                'strength': 0.0,
                'reasoning': 'Invalid price data'
            }
        
        price_change = (last_close - first_close) / first_close
        
        # Analyze consecutive moves
        green_count = 0
        red_count = 0
        
        for candle in recent_candles:
            open_price = float(candle.get('open', 0))
            close = float(candle.get('close', 0))
            if close > open_price:
                green_count += 1
            elif close < open_price:
                red_count += 1
        
        # Determine momentum
        if price_change > 0.02 and green_count > red_count:
            momentum = 'bullish'
            strength = min(0.8, abs(price_change) * 10 + green_count * 0.1)
        elif price_change < -0.02 and red_count > green_count:
            momentum = 'bearish'
            strength = min(0.8, abs(price_change) * 10 + red_count * 0.1)
        else:
            momentum = 'neutral'
            strength = 0.2
        
        return {
            'momentum': momentum,
            'strength': round(strength, 3),
            'reasoning': f'Price change: {price_change:.3f}, Green: {green_count}, Red: {red_count}',
            'price_change': round(price_change, 4),
            'green_candles': green_count,
            'red_candles': red_count
        }
        
    except Exception as e:
        return {
            'momentum': 'neutral',
            'strength': 0.0,
            'reasoning': f'Momentum analysis error: {str(e)}'
        }

def enhanced_scenario_evaluation(candles: List[Dict], ema: float = None, current_price: float = None) -> Dict:
    """
    Enhanced scenario evaluation combining current candle analysis with momentum context
    
    Args:
        candles: List of candle data (last candle is current)
        ema: EMA reference level
        current_price: Current/live price
        
    Returns:
        Comprehensive scenario analysis
    """
    try:
        if not candles:
            return {
                'scenario': 'neutral',
                'confidence': 0.0,
                'reasoning': 'No candle data available'
            }
        
        # Analyze current candle
        current_candle = candles[-1]
        current_scenario = evaluate_future_scenario(current_candle, ema, current_price)
        
        # Analyze momentum context
        momentum_analysis = analyze_candle_momentum(candles, lookback=3)
        
        # Combine analyses for enhanced prediction
        base_scenario = current_scenario['scenario']
        base_confidence = current_scenario['confidence']
        
        # Momentum adjustment
        momentum_boost = 0.0
        if momentum_analysis['momentum'] == 'bullish' and base_scenario == 'bull_case':
            momentum_boost = min(0.15, momentum_analysis['strength'] * 0.2)
        elif momentum_analysis['momentum'] == 'bearish' and base_scenario == 'bear_case':
            momentum_boost = min(0.15, momentum_analysis['strength'] * 0.2)
        elif momentum_analysis['momentum'] != 'neutral':
            # Momentum conflicts with scenario
            momentum_boost = -0.05
        
        final_confidence = min(0.95, max(0.1, base_confidence + momentum_boost))
        
        enhanced_reasoning = f"{current_scenario['reasoning']}; Momentum: {momentum_analysis['momentum']} ({momentum_analysis['strength']:.2f})"
        
        return {
            'scenario': base_scenario,
            'confidence': round(final_confidence, 3),
            'reasoning': enhanced_reasoning,
            'current_candle': current_scenario,
            'momentum_context': momentum_analysis,
            'momentum_boost': round(momentum_boost, 3)
        }
        
    except Exception as e:
        return {
            'scenario': 'neutral',
            'confidence': 0.0,
            'reasoning': f'Enhanced evaluation error: {str(e)}'
        }