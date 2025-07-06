#!/usr/bin/env python3
"""
TJDE Basic Engine - Phase 1 Analysis
Fast scoring without AI-EYE dependencies for all tokens
Used in new two-phase architecture
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time


def simulate_trader_decision_basic(
    symbol: str,
    candles_15m: List,
    candles_5m: List = None,
    volume_24h: float = 0,
    price_change_24h: float = 0,
    current_price: float = 0
) -> Dict[str, Any]:
    """
    PHASE 1: Basic TJDE scoring without AI-EYE dependency
    
    Args:
        symbol: Trading symbol
        candles_15m: 15-minute candle data
        candles_5m: 5-minute candle data (optional)
        volume_24h: 24-hour volume
        price_change_24h: 24-hour price change percentage
        current_price: Current price
        
    Returns:
        Basic scoring result with score and decision
    """
    
    try:
        if not candles_15m or len(candles_15m) < 20:
            return {
                'score': 0.0,
                'decision': 'skip',
                'reason': 'insufficient_data',
                'breakdown': {'data_quality': 'insufficient'}
            }
            
        # Basic trend analysis
        trend_score = analyze_basic_trend(candles_15m)
        
        # Volume analysis
        volume_score = analyze_volume_pattern(candles_15m, volume_24h)
        
        # Momentum analysis
        momentum_score = analyze_basic_momentum(candles_15m)
        
        # Price change factor
        price_change_score = analyze_price_change(price_change_24h)
        
        # Orderbook basic analysis
        orderbook_score = analyze_basic_orderbook(candles_15m)
        
        # Combine scores with weights
        final_score = (
            trend_score * 0.25 +
            volume_score * 0.25 + 
            momentum_score * 0.2 +
            price_change_score * 0.15 +
            orderbook_score * 0.15
        )
        
        # Normalize to 0-1 range
        final_score = max(0.0, min(1.0, final_score))
        
        # Decision logic
        if final_score >= 0.6:
            decision = 'consider'
        elif final_score >= 0.4:
            decision = 'consider'
        elif final_score >= 0.2:
            decision = 'wait'
        else:
            decision = 'avoid'
            
        return {
            'score': final_score,
            'decision': decision,
            'breakdown': {
                'trend': trend_score,
                'volume': volume_score,
                'momentum': momentum_score,
                'price_change': price_change_score,
                'orderbook': orderbook_score,
                'method': 'basic_phase1'
            }
        }
        
    except Exception as e:
        return {
            'score': 0.0,
            'decision': 'error',
            'reason': f'basic_analysis_error: {e}',
            'breakdown': {'error': str(e)}
        }


def analyze_basic_trend(candles_15m: List) -> float:
    """Basic trend analysis using price action"""
    try:
        if len(candles_15m) < 10:
            return 0.1
            
        # Extract close prices
        closes = []
        for candle in candles_15m[-20:]:
            if isinstance(candle, dict):
                close = candle.get('close', candle.get('4', 0))
            else:
                close = candle[4] if len(candle) > 4 else 0
            closes.append(float(close))
            
        if not closes or all(c == 0 for c in closes):
            return 0.1
            
        # Simple trend calculation
        recent_avg = np.mean(closes[-5:])
        older_avg = np.mean(closes[-15:-10])
        
        if older_avg == 0:
            return 0.1
            
        trend_ratio = recent_avg / older_avg
        
        # Normalize around 1.0
        if trend_ratio > 1.0:
            score = min(0.8, (trend_ratio - 1.0) * 4.0 + 0.1)
        else:
            score = max(0.0, trend_ratio * 0.1)
            
        return score
        
    except Exception:
        return 0.1


def analyze_volume_pattern(candles_15m: List, volume_24h: float) -> float:
    """Basic volume pattern analysis"""
    try:
        if len(candles_15m) < 10:
            return 0.1
            
        # Extract volumes
        volumes = []
        for candle in candles_15m[-10:]:
            if isinstance(candle, dict):
                volume = candle.get('volume', candle.get('5', 0))
            else:
                volume = candle[5] if len(candle) > 5 else 0
            volumes.append(float(volume))
            
        if not volumes or all(v == 0 for v in volumes):
            return 0.1
            
        # Recent vs historical volume
        recent_volume = np.mean(volumes[-3:])
        hist_volume = np.mean(volumes[:-3])
        
        if hist_volume == 0:
            return 0.2
            
        volume_ratio = recent_volume / hist_volume
        
        # Higher recent volume is positive
        score = min(0.8, max(0.1, volume_ratio * 0.3))
        
        # Boost for high 24h volume
        if volume_24h > 1000000:  # High volume threshold
            score += 0.1
            
        return min(0.8, score)
        
    except Exception:
        return 0.1


def analyze_basic_momentum(candles_15m: List) -> float:
    """Basic momentum analysis"""
    try:
        if len(candles_15m) < 15:
            return 0.1
            
        # Extract OHLC
        ohlc_data = []
        for candle in candles_15m[-15:]:
            if isinstance(candle, dict):
                ohlc = [
                    candle.get('open', candle.get('1', 0)),
                    candle.get('high', candle.get('2', 0)),
                    candle.get('low', candle.get('3', 0)),
                    candle.get('close', candle.get('4', 0))
                ]
            else:
                ohlc = [candle[1], candle[2], candle[3], candle[4]]
            ohlc_data.append([float(x) for x in ohlc])
            
        if not ohlc_data:
            return 0.1
            
        # Calculate simple momentum indicators
        closes = [ohlc[3] for ohlc in ohlc_data]
        
        # Price momentum (recent vs older)
        recent_price = np.mean(closes[-3:])
        older_price = np.mean(closes[-10:-7])
        
        if older_price == 0:
            return 0.1
            
        price_momentum = recent_price / older_price
        
        # Volatility (range expansion)
        recent_ranges = [(ohlc[1] - ohlc[2]) / ohlc[3] for ohlc in ohlc_data[-5:] if ohlc[3] > 0]
        hist_ranges = [(ohlc[1] - ohlc[2]) / ohlc[3] for ohlc in ohlc_data[-15:-10] if ohlc[3] > 0]
        
        volatility_ratio = 1.0
        if recent_ranges and hist_ranges:
            volatility_ratio = np.mean(recent_ranges) / max(np.mean(hist_ranges), 0.001)
            
        # Combine factors
        momentum_score = (price_momentum - 1.0) * 2.0 + volatility_ratio * 0.2
        
        return max(0.0, min(0.8, momentum_score))
        
    except Exception:
        return 0.1


def analyze_price_change(price_change_24h: float) -> float:
    """Analyze 24h price change impact"""
    try:
        if price_change_24h == 0:
            return 0.1
            
        # Positive changes are favorable
        if price_change_24h > 0:
            # Scale positive changes 0-20% to 0.1-0.8
            score = min(0.8, 0.1 + (price_change_24h / 20.0) * 0.7)
        else:
            # Slight negative penalty
            score = max(0.0, 0.1 + price_change_24h / 100.0)
            
        return score
        
    except Exception:
        return 0.1


def analyze_basic_orderbook(candles_15m: List) -> float:
    """Basic orderbook analysis from candle patterns"""
    try:
        if len(candles_15m) < 5:
            return 0.1
            
        # Analyze recent candle patterns
        pattern_scores = []
        
        for candle in candles_15m[-5:]:
            if isinstance(candle, dict):
                open_price = float(candle.get('open', candle.get('1', 0)))
                high = float(candle.get('high', candle.get('2', 0)))
                low = float(candle.get('low', candle.get('3', 0)))
                close = float(candle.get('close', candle.get('4', 0)))
            else:
                open_price = float(candle[1])
                high = float(candle[2])
                low = float(candle[3])
                close = float(candle[4])
                
            if close == 0 or high == low:
                pattern_scores.append(0.1)
                continue
                
            # Body size relative to range
            body_size = abs(close - open_price)
            total_range = high - low
            
            body_ratio = body_size / total_range if total_range > 0 else 0
            
            # Direction bias
            direction = 1.0 if close > open_price else 0.5
            
            # Pattern score
            pattern_score = body_ratio * direction * 0.5 + 0.1
            pattern_scores.append(pattern_score)
            
        return min(0.8, np.mean(pattern_scores))
        
    except Exception:
        return 0.1


if __name__ == "__main__":
    # Test basic engine
    test_candles = [
        [1625097600, 100, 105, 98, 103, 1000],
        [1625101200, 103, 108, 102, 107, 1200],
        [1625104800, 107, 110, 105, 109, 1500],
        [1625108400, 109, 112, 108, 111, 1300],
        [1625112000, 111, 115, 110, 114, 1800]
    ]
    
    result = simulate_trader_decision_basic(
        symbol="TESTUSDT",
        candles_15m=test_candles,
        volume_24h=50000000,
        price_change_24h=5.2,
        current_price=114
    )
    
    print("Basic Engine Test Result:")
    print(f"Score: {result['score']:.3f}")
    print(f"Decision: {result['decision']}")
    print(f"Breakdown: {result['breakdown']}")