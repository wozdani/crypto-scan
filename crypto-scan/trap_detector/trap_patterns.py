"""
Trap Detector - Pattern Recognition Module
Wykrywa fałszywe wybicia, bull/bear trapy i pułapki FOMO

Module 3 of Advanced Vision-AI System
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_fake_breakout(candles: List[Dict], lookback: int = 5) -> Dict:
    """
    Wykrywa świecę z długim knotem górnym i dużym wolumenem – klasyczny bull trap
    
    Args:
        candles: Lista świec w formacie dict z kluczami: open, high, low, close, volume
        lookback: Ile świec wstecz sprawdzać dla kontekstu wolumenu
        
    Returns:
        Dict z informacjami o fake breakout:
        {
            'detected': bool,
            'confidence': float (0.0-1.0),
            'wick_ratio': float,
            'volume_ratio': float,
            'reasoning': str
        }
    """
    
    if not candles or len(candles) < lookback:
        return {
            'detected': False,
            'confidence': 0.0,
            'wick_ratio': 0.0,
            'volume_ratio': 0.0,
            'reasoning': 'Insufficient candle data'
        }
    
    try:
        current_candle = candles[-1]
        
        # Validate candle data
        required_keys = ['open', 'high', 'low', 'close', 'volume']
        if not all(key in current_candle for key in required_keys):
            return {
                'detected': False,
                'confidence': 0.0,
                'wick_ratio': 0.0,
                'volume_ratio': 0.0,
                'reasoning': 'Invalid candle data format'
            }
        
        # Calculate candle metrics
        open_price = float(current_candle['open'])
        high_price = float(current_candle['high'])
        low_price = float(current_candle['low'])
        close_price = float(current_candle['close'])
        volume = float(current_candle['volume'])
        
        # Calculate body and wick sizes
        body_size = abs(close_price - open_price)
        upper_wick = high_price - max(close_price, open_price)
        lower_wick = min(close_price, open_price) - low_price
        total_range = high_price - low_price + 1e-9  # Avoid division by zero
        
        # Calculate ratios
        wick_ratio = upper_wick / total_range
        body_ratio = body_size / total_range
        
        # Calculate average volume from previous candles
        if len(candles) >= lookback:
            prev_volumes = [float(c['volume']) for c in candles[-lookback-1:-1] if 'volume' in c]
            avg_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else 1.0
        else:
            avg_volume = 1.0
            
        volume_ratio = volume / (avg_volume + 1e-9)
        
        # Fake breakout detection criteria
        fake_breakout_detected = False
        confidence = 0.0
        reasoning_parts = []
        
        # Primary criteria: Long upper wick with high volume
        if wick_ratio > 0.5 and volume_ratio > 1.5:
            fake_breakout_detected = True
            confidence += 0.6
            reasoning_parts.append(f"Long upper wick ({wick_ratio:.2f}) with elevated volume ({volume_ratio:.1f}x)")
        
        # Secondary criteria: Small body relative to range
        if body_ratio < 0.3 and wick_ratio > 0.4:
            confidence += 0.2
            reasoning_parts.append(f"Small body ({body_ratio:.2f}) with significant wick")
        
        # Tertiary criteria: Volume spike without follow-through
        if volume_ratio > 2.0 and close_price < (high_price + low_price) / 2:
            confidence += 0.2
            reasoning_parts.append(f"Volume spike ({volume_ratio:.1f}x) without bullish close")
        
        # Ensure confidence doesn't exceed 1.0
        confidence = min(confidence, 1.0)
        
        # Final detection logic
        if not fake_breakout_detected and confidence >= 0.4:
            fake_breakout_detected = True
            
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No fake breakout pattern detected"
        
        return {
            'detected': fake_breakout_detected,
            'confidence': confidence,
            'wick_ratio': wick_ratio,
            'volume_ratio': volume_ratio,
            'body_ratio': body_ratio,
            'reasoning': reasoning
        }
        
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error in detect_fake_breakout: {e}")
        return {
            'detected': False,
            'confidence': 0.0,
            'wick_ratio': 0.0,
            'volume_ratio': 0.0,
            'reasoning': f"Error processing candle data: {e}"
        }

def detect_failed_breakout(candles: List[Dict], lookback: int = 4) -> Dict:
    """
    Wykrywa sytuację gdy cena przebiła szczyt i wróciła – podejrzany failed breakout
    
    Args:
        candles: Lista świec
        lookback: Ile świec wstecz sprawdzać dla ustalenia poprzedniego szczytu
        
    Returns:
        Dict z informacjami o failed breakout
    """
    
    if not candles or len(candles) < lookback + 1:
        return {
            'detected': False,
            'confidence': 0.0,
            'breakout_level': 0.0,
            'close_below_ratio': 0.0,
            'reasoning': 'Insufficient candle data for failed breakout detection'
        }
    
    try:
        current_candle = candles[-1]
        context_candles = candles[-lookback-1:-1]  # Previous candles for context
        
        # Find the highest high in context period
        prev_highs = [float(c['high']) for c in context_candles if 'high' in c]
        if not prev_highs:
            return {
                'detected': False,
                'confidence': 0.0,
                'breakout_level': 0.0,
                'close_below_ratio': 0.0,
                'reasoning': 'No valid highs found in context candles'
            }
            
        prev_high = max(prev_highs)
        current_high = float(current_candle['high'])
        current_close = float(current_candle['close'])
        
        # Check if there was a breakout
        breakout_occurred = current_high > prev_high
        breakout_margin = (current_high - prev_high) / prev_high if prev_high > 0 else 0
        
        # Check if close is below the breakout level
        close_below_breakout = current_close < prev_high
        close_below_ratio = (prev_high - current_close) / prev_high if prev_high > 0 else 0
        
        # Failed breakout detection
        failed_breakout_detected = breakout_occurred and close_below_breakout
        
        # Calculate confidence based on multiple factors
        confidence = 0.0
        reasoning_parts = []
        
        if failed_breakout_detected:
            # Base confidence for failed breakout
            confidence += 0.5
            reasoning_parts.append(f"Breakout above {prev_high:.6f} failed, close at {current_close:.6f}")
            
            # Higher confidence if significant breakout that failed
            if breakout_margin > 0.02:  # 2% breakout
                confidence += 0.2
                reasoning_parts.append(f"Significant breakout margin ({breakout_margin:.1%}) failed")
            
            # Higher confidence if close is well below breakout level
            if close_below_ratio > 0.01:  # 1% below
                confidence += 0.2
                reasoning_parts.append(f"Close significantly below breakout level ({close_below_ratio:.1%})")
            
            # Check for volume spike during breakout (if volume data available)
            if 'volume' in current_candle:
                # Get average volume from context
                context_volumes = [float(c.get('volume', 0)) for c in context_candles]
                avg_volume = sum(context_volumes) / len(context_volumes) if context_volumes else 1.0
                current_volume = float(current_candle['volume'])
                volume_ratio = current_volume / (avg_volume + 1e-9)
                
                if volume_ratio > 1.5:
                    confidence += 0.1
                    reasoning_parts.append(f"Elevated volume during failed breakout ({volume_ratio:.1f}x)")
        
        confidence = min(confidence, 1.0)
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No failed breakout detected"
        
        return {
            'detected': failed_breakout_detected,
            'confidence': confidence,
            'breakout_level': prev_high,
            'current_high': current_high,
            'current_close': current_close,
            'breakout_margin': breakout_margin,
            'close_below_ratio': close_below_ratio,
            'reasoning': reasoning
        }
        
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error in detect_failed_breakout: {e}")
        return {
            'detected': False,
            'confidence': 0.0,
            'breakout_level': 0.0,
            'close_below_ratio': 0.0,
            'reasoning': f"Error processing breakout data: {e}"
        }

def detect_bear_trap(candles: List[Dict], lookback: int = 5) -> Dict:
    """
    Wykrywa bear trap - długie dolne knoty z wysokim wolumenem sugerujące fałszywy breakdown
    
    Args:
        candles: Lista świec
        lookback: Ile świec wstecz sprawdzać dla kontekstu
        
    Returns:
        Dict z informacjami o bear trap
    """
    
    if not candles or len(candles) < lookback:
        return {
            'detected': False,
            'confidence': 0.0,
            'lower_wick_ratio': 0.0,
            'volume_ratio': 0.0,
            'reasoning': 'Insufficient candle data for bear trap detection'
        }
    
    try:
        current_candle = candles[-1]
        
        # Calculate candle metrics
        open_price = float(current_candle['open'])
        high_price = float(current_candle['high'])
        low_price = float(current_candle['low'])
        close_price = float(current_candle['close'])
        volume = float(current_candle.get('volume', 0))
        
        # Calculate body and wick sizes
        body_size = abs(close_price - open_price)
        lower_wick = min(close_price, open_price) - low_price
        total_range = high_price - low_price + 1e-9
        
        # Calculate ratios
        lower_wick_ratio = lower_wick / total_range
        body_ratio = body_size / total_range
        
        # Calculate volume context
        if len(candles) >= lookback:
            prev_volumes = [float(c.get('volume', 0)) for c in candles[-lookback-1:-1]]
            avg_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else 1.0
        else:
            avg_volume = 1.0
            
        volume_ratio = volume / (avg_volume + 1e-9)
        
        # Bear trap detection criteria
        bear_trap_detected = False
        confidence = 0.0
        reasoning_parts = []
        
        # Primary criteria: Long lower wick with high volume
        if lower_wick_ratio > 0.4 and volume_ratio > 1.5:
            bear_trap_detected = True
            confidence += 0.6
            reasoning_parts.append(f"Long lower wick ({lower_wick_ratio:.2f}) with elevated volume ({volume_ratio:.1f}x)")
        
        # Secondary criteria: Recovery close above midpoint
        midpoint = (high_price + low_price) / 2
        if close_price > midpoint and lower_wick_ratio > 0.3:
            confidence += 0.2
            reasoning_parts.append(f"Recovery close above midpoint with significant lower wick")
        
        # Tertiary criteria: Volume spike with bullish close
        if volume_ratio > 2.0 and close_price > open_price:
            confidence += 0.2
            reasoning_parts.append(f"Volume spike ({volume_ratio:.1f}x) with bullish close")
        
        confidence = min(confidence, 1.0)
        
        if not bear_trap_detected and confidence >= 0.4:
            bear_trap_detected = True
            
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No bear trap pattern detected"
        
        return {
            'detected': bear_trap_detected,
            'confidence': confidence,
            'lower_wick_ratio': lower_wick_ratio,
            'volume_ratio': volume_ratio,
            'body_ratio': body_ratio,
            'reasoning': reasoning
        }
        
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error in detect_bear_trap: {e}")
        return {
            'detected': False,
            'confidence': 0.0,
            'lower_wick_ratio': 0.0,
            'volume_ratio': 0.0,
            'reasoning': f"Error processing bear trap data: {e}"
        }

def detect_exhaustion_spike(candles: List[Dict], lookback: int = 10) -> Dict:
    """
    Wykrywa exhaustion spike - nagły ruch z bardzo wysokim wolumenem ale brak kontynuacji
    
    Args:
        candles: Lista świec
        lookback: Ile świec sprawdzać dla trendu i kontekstu
        
    Returns:
        Dict z informacjami o exhaustion spike
    """
    
    if not candles or len(candles) < lookback:
        return {
            'detected': False,
            'confidence': 0.0,
            'volume_spike_ratio': 0.0,
            'price_rejection': 0.0,
            'reasoning': 'Insufficient data for exhaustion spike detection'
        }
    
    try:
        current_candle = candles[-1]
        context_candles = candles[-lookback-1:-1]
        
        # Calculate volume metrics
        current_volume = float(current_candle.get('volume', 0))
        context_volumes = [float(c.get('volume', 0)) for c in context_candles]
        avg_volume = sum(context_volumes) / len(context_volumes) if context_volumes else 1.0
        max_volume = max(context_volumes) if context_volumes else 1.0
        
        volume_spike_ratio = current_volume / (avg_volume + 1e-9)
        volume_vs_max_ratio = current_volume / (max_volume + 1e-9)
        
        # Calculate price metrics
        open_price = float(current_candle['open'])
        high_price = float(current_candle['high'])
        low_price = float(current_candle['low'])
        close_price = float(current_candle['close'])
        
        # Price movement and rejection
        price_range = high_price - low_price
        price_movement = abs(close_price - open_price)
        
        # Calculate rejection from extremes
        if close_price > open_price:  # Bullish candle
            rejection_from_high = (high_price - close_price) / price_range if price_range > 0 else 0
            price_rejection = rejection_from_high
        else:  # Bearish candle
            rejection_from_low = (close_price - low_price) / price_range if price_range > 0 else 0
            price_rejection = rejection_from_low
        
        # Exhaustion spike detection
        exhaustion_detected = False
        confidence = 0.0
        reasoning_parts = []
        
        # Primary criteria: Extreme volume spike with price rejection
        if volume_spike_ratio > 3.0 and price_rejection > 0.3:
            exhaustion_detected = True
            confidence += 0.7
            reasoning_parts.append(f"Extreme volume spike ({volume_spike_ratio:.1f}x) with significant price rejection ({price_rejection:.1%})")
        
        # Secondary criteria: New volume high but failed to hold extreme
        if volume_vs_max_ratio > 1.2 and price_rejection > 0.4:
            confidence += 0.2
            reasoning_parts.append(f"New volume high with failed price follow-through")
        
        # Tertiary criteria: Large range but small body
        body_ratio = price_movement / (price_range + 1e-9)
        if volume_spike_ratio > 2.0 and body_ratio < 0.3:
            confidence += 0.1
            reasoning_parts.append(f"High volume with small body relative to range")
        
        confidence = min(confidence, 1.0)
        
        if not exhaustion_detected and confidence >= 0.5:
            exhaustion_detected = True
            
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No exhaustion spike detected"
        
        return {
            'detected': exhaustion_detected,
            'confidence': confidence,
            'volume_spike_ratio': volume_spike_ratio,
            'price_rejection': price_rejection,
            'body_ratio': body_ratio,
            'reasoning': reasoning
        }
        
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error in detect_exhaustion_spike: {e}")
        return {
            'detected': False,
            'confidence': 0.0,
            'volume_spike_ratio': 0.0,
            'price_rejection': 0.0,
            'reasoning': f"Error processing exhaustion spike data: {e}"
        }

def comprehensive_trap_analysis(candles: List[Dict]) -> Dict:
    """
    Przeprowadza kompletną analizę wszystkich typów pułapek
    
    Args:
        candles: Lista świec do analizy
        
    Returns:
        Dict ze wszystkimi wynikami analizy trap detection
    """
    
    if not candles:
        return {
            'any_trap_detected': False,
            'max_confidence': 0.0,
            'trap_types': [],
            'summary': 'No candle data provided'
        }
    
    # Run all trap detectors
    fake_breakout = detect_fake_breakout(candles)
    failed_breakout = detect_failed_breakout(candles)
    bear_trap = detect_bear_trap(candles)
    exhaustion_spike = detect_exhaustion_spike(candles)
    
    # Aggregate results
    all_results = {
        'fake_breakout': fake_breakout,
        'failed_breakout': failed_breakout,
        'bear_trap': bear_trap,
        'exhaustion_spike': exhaustion_spike
    }
    
    # Find detected traps
    detected_traps = [trap_type for trap_type, result in all_results.items() if result['detected']]
    max_confidence = max([result['confidence'] for result in all_results.values()], default=0.0)
    any_trap_detected = len(detected_traps) > 0
    
    # Create summary
    if any_trap_detected:
        summary = f"Detected {len(detected_traps)} trap(s): {', '.join(detected_traps)} (max confidence: {max_confidence:.2f})"
    else:
        summary = "No trap patterns detected"
    
    return {
        'any_trap_detected': any_trap_detected,
        'max_confidence': max_confidence,
        'detected_trap_count': len(detected_traps),
        'trap_types': detected_traps,
        'detailed_results': all_results,
        'summary': summary
    }