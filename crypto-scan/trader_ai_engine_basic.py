#!/usr/bin/env python3
"""
Basic TJDE Engine - Lightweight Initial Screening
Used for initial token filtering before advanced analysis
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import logging

def simulate_trader_decision_basic(
    symbol: str,
    current_price: float,
    candles_15m: List,
    candles_5m: List = None,
    orderbook_data: Dict = None,
    volume_24h: float = None,
    price_change_24h: float = None,
    **kwargs
) -> Dict:
    """
    Basic TJDE scoring for initial filtering - NO ai_label or htf_candles required
    Uses only 15M/5M candles + orderbook for quick screening
    
    Returns lightweight scoring suitable for TOP 5 selection
    """
    try:
        # Validate basic inputs
        if not candles_15m or len(candles_15m) < 20:
            return {
                'final_score': 0.0,
                'decision': 'skip',
                'confidence': 0.0,
                'reason': 'insufficient_15m_candles',
                'scoring_type': 'basic'
            }
        
        if current_price <= 0:
            return {
                'final_score': 0.0,
                'decision': 'skip', 
                'confidence': 0.0,
                'reason': 'invalid_price',
                'scoring_type': 'basic'
            }
        
        # Basic trend analysis using 15M candles
        trend_score = _compute_basic_trend_strength(candles_15m, current_price)
        
        # Basic volume analysis
        volume_score = _compute_basic_volume_behavior(candles_15m, volume_24h)
        
        # Basic momentum analysis
        momentum_score = _compute_basic_momentum(candles_15m, current_price)
        
        # Basic orderbook pressure (if available)
        orderbook_score = _compute_basic_orderbook_pressure(orderbook_data, current_price) if orderbook_data else 0.0
        
        # Price change context
        price_change_score = _compute_basic_price_change_context(price_change_24h) if price_change_24h else 0.0
        
        # Weighted basic scoring (conservative weights for initial filtering)
        component_scores = {
            'trend': trend_score,
            'volume': volume_score, 
            'momentum': momentum_score,
            'orderbook': orderbook_score,
            'price_change': price_change_score
        }
        
        # Conservative weighting for basic screening
        weights = {
            'trend': 0.35,
            'volume': 0.25,
            'momentum': 0.20,
            'orderbook': 0.10,
            'price_change': 0.10
        }
        
        final_score = sum(component_scores[key] * weights[key] for key in weights.keys())
        
        # Basic decision logic (lower thresholds for initial screening)
        if final_score >= 0.30:
            decision = 'consider'  # Candidate for advanced analysis
            confidence = min(0.8, final_score * 2.0)  # Cap basic confidence
        elif final_score >= 0.15:
            decision = 'wait'
            confidence = final_score * 1.5
        else:
            decision = 'avoid'
            confidence = max(0.1, final_score)
        
        return {
            'final_score': round(final_score, 4),
            'decision': decision,
            'confidence': round(confidence, 3),
            'components': component_scores,
            'weights_used': weights,
            'scoring_type': 'basic',
            'reason': f'basic_analysis_complete'
        }
        
    except Exception as e:
        logging.error(f"[BASIC TJDE ERROR] {symbol}: {e}")
        return {
            'final_score': 0.0,
            'decision': 'skip',
            'confidence': 0.0,
            'reason': f'basic_analysis_error: {str(e)}',
            'scoring_type': 'basic'
        }

def _compute_basic_trend_strength(candles_15m: List, current_price: float) -> float:
    """Basic trend strength using EMA and price position"""
    try:
        closes = []
        for candle in candles_15m[-20:]:  # Last 20 candles
            if isinstance(candle, dict):
                close = candle.get('close', candle.get('4', 0))
            elif isinstance(candle, (list, tuple)) and len(candle) > 4:
                close = candle[4]
            else:
                continue
            
            if close and close > 0:
                closes.append(float(close))
        
        if len(closes) < 10:
            return 0.0
        
        # Simple EMA calculation
        ema_20 = closes[0]
        alpha = 2.0 / (20 + 1)
        for price in closes[1:]:
            ema_20 = alpha * price + (1 - alpha) * ema_20
        
        # Price position relative to EMA
        price_vs_ema = (current_price - ema_20) / ema_20 if ema_20 > 0 else 0
        
        # Recent price momentum
        recent_change = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 and closes[-5] > 0 else 0
        
        # Combine factors
        trend_score = (price_vs_ema * 0.6) + (recent_change * 0.4)
        
        # Normalize to 0-1 range with more lenient scoring for initial screening
        # Allow some positive scoring even for declining trends if there's volatility
        normalized_score = abs(trend_score) * 2.0 + 0.1  # Base 0.1 for any movement
        return max(0.0, min(1.0, normalized_score))
        
    except Exception:
        return 0.0

def _compute_basic_volume_behavior(candles_15m: List, volume_24h: float = None) -> float:
    """Basic volume trend analysis"""
    try:
        volumes = []
        for candle in candles_15m[-10:]:  # Last 10 candles
            if isinstance(candle, dict):
                volume = candle.get('volume', candle.get('5', 0))
            elif isinstance(candle, (list, tuple)) and len(candle) > 5:
                volume = candle[5]
            else:
                continue
            
            if volume and volume > 0:
                volumes.append(float(volume))
        
        if len(volumes) < 5:
            return 0.0
        
        # Recent vs older volume comparison
        recent_avg = np.mean(volumes[-3:])
        older_avg = np.mean(volumes[:3])
        
        if older_avg > 0:
            volume_trend = (recent_avg - older_avg) / older_avg
        else:
            volume_trend = 0
        
        # Normalize with base scoring for any volume activity
        # Use absolute value to capture both increase and decrease patterns
        volume_score = max(0.1, min(1.0, abs(volume_trend) * 3.0 + 0.2))
        
        return volume_score
        
    except Exception:
        return 0.0

def _compute_basic_momentum(candles_15m: List, current_price: float) -> float:
    """Basic momentum using price changes"""
    try:
        closes = []
        for candle in candles_15m[-15:]:
            if isinstance(candle, dict):
                close = candle.get('close', candle.get('4', 0))
            elif isinstance(candle, (list, tuple)) and len(candle) > 4:
                close = candle[4]
            else:
                continue
            
            if close and close > 0:
                closes.append(float(close))
        
        if len(closes) < 10:
            return 0.0
        
        # Multiple timeframe momentum
        momentum_1h = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 and closes[-4] > 0 else 0  # 1H momentum
        momentum_3h = (closes[-1] - closes[-12]) / closes[-12] if len(closes) >= 12 and closes[-12] > 0 else 0  # 3H momentum
        
        # Combine momentum signals
        combined_momentum = (momentum_1h * 0.7) + (momentum_3h * 0.3)
        
        # Normalize with base scoring for any momentum activity
        # Use absolute value to capture both positive and negative momentum
        momentum_score = max(0.1, min(1.0, abs(combined_momentum) * 5.0 + 0.2))
        
        return momentum_score
        
    except Exception:
        return 0.0

def _compute_basic_orderbook_pressure(orderbook_data: Dict, current_price: float) -> float:
    """Basic bid/ask pressure analysis"""
    try:
        if not orderbook_data or 'bids' not in orderbook_data or 'asks' not in orderbook_data:
            return 0.0
        
        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])
        
        if not bids or not asks:
            return 0.0
        
        # Calculate bid/ask volume in nearby price levels
        bid_volume = sum(float(bid[1]) for bid in bids[:5])  # Top 5 bids
        ask_volume = sum(float(ask[1]) for ask in asks[:5])  # Top 5 asks
        
        if bid_volume + ask_volume == 0:
            return 0.0
        
        # Bid/ask ratio (>0.5 = more bids, <0.5 = more asks)
        pressure_ratio = bid_volume / (bid_volume + ask_volume)
        
        # Convert to -1 to +1 scale, then normalize to 0-1
        pressure_score = (pressure_ratio - 0.5) * 2.0  # -1 to +1
        pressure_score = max(0.0, min(1.0, pressure_score * 0.5 + 0.5))  # 0 to 1
        
        return pressure_score
        
    except Exception:
        return 0.0

def _compute_basic_price_change_context(price_change_24h: float) -> float:
    """Basic 24h price change context"""
    try:
        if price_change_24h is None:
            return 0.0
        
        # Normalize 24h change to score
        # Positive changes get higher scores, but cap extreme moves
        if price_change_24h > 0:
            # Positive change: 0% = 0.5, 10% = 0.8, 20%+ = 1.0
            change_score = min(1.0, 0.5 + (price_change_24h / 100.0) * 2.5)
        else:
            # Negative change: 0% = 0.5, -10% = 0.2, -20%+ = 0.0
            change_score = max(0.0, 0.5 + (price_change_24h / 100.0) * 2.5)
        
        return change_score
        
    except Exception:
        return 0.0

# Test function
def test_basic_scoring():
    """Test basic scoring with sample data"""
    # Sample 15M candles (timestamp, open, high, low, close, volume)
    sample_candles = [
        [1640995200, 100.0, 102.0, 99.0, 101.0, 1000],
        [1640996100, 101.0, 103.0, 100.0, 102.0, 1200],
        [1640997000, 102.0, 104.0, 101.0, 103.5, 1500],
        [1640997900, 103.5, 105.0, 102.0, 104.0, 1800],
        [1640998800, 104.0, 106.0, 103.0, 105.5, 2000],
    ] * 4  # Repeat to get 20 candles
    
    result = simulate_trader_decision_basic(
        symbol="TESTUSDT",
        current_price=105.5,
        candles_15m=sample_candles,
        volume_24h=50000,
        price_change_24h=5.5
    )
    
    print(f"[BASIC TEST] Score: {result['final_score']}, Decision: {result['decision']}")
    print(f"[BASIC TEST] Components: {result['components']}")
    
    return result

if __name__ == "__main__":
    test_basic_scoring()