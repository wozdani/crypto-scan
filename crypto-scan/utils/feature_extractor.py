#!/usr/bin/env python3
"""
TJDE v2 Stage 4 - Feature Extraction Engine
Extracts trading features from market data like a professional trader

Features extracted:
- trend_strength: Price slope/EMA, directional stability
- pullback_quality: Depth and regularity of pullbacks in trend
- volume_behavior_score: Whether volume supports price direction
- psych_score: Hidden pressure interpretation (wicks, false signals)
- support_reaction: Price reaction at local support/resistance levels
"""

import numpy as np
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(market_data: Dict) -> Optional[Dict]:
    """
    Extract trading features from market data
    
    Args:
        market_data: Dictionary containing candles_15m, orderbook, price_usd, volume_24h
        
    Returns:
        Dictionary with extracted features or None if extraction fails
    """
    try:
        # Validate required data
        if not market_data or "candles_15m" not in market_data:
            logger.warning("[FEATURE ERROR] Missing candles_15m in market data")
            return None
            
        candles = market_data["candles_15m"]
        if len(candles) < 20:
            logger.warning(f"[FEATURE ERROR] Insufficient candles: {len(candles)} < 20")
            return None
            
        # Extract basic price data from last 20 candles
        recent_candles = candles[-20:]
        
        highs = []
        lows = []
        closes = []
        volumes = []
        
        for candle in recent_candles:
            if isinstance(candle, dict):
                highs.append(float(candle.get("high", 0)))
                lows.append(float(candle.get("low", 0)))
                closes.append(float(candle.get("close", 0)))
                volumes.append(float(candle.get("volume", 0)))
            elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                # [timestamp, open, high, low, close, volume]
                highs.append(float(candle[2]))
                lows.append(float(candle[3]))
                closes.append(float(candle[4]))
                volumes.append(float(candle[5]))
            else:
                logger.warning(f"[FEATURE ERROR] Invalid candle format: {candle}")
                return None
        
        # Validate extracted data
        if len(closes) < 20 or sum(volumes) == 0:
            logger.warning("[FEATURE ERROR] Invalid extracted data")
            return None
            
        # Extract features
        features = {}
        
        # 1. Trend Strength - Price slope and directional stability
        features["trend_strength"] = calculate_trend_strength(closes)
        
        # 2. Pullback Quality - Depth and regularity of corrections
        features["pullback_quality"] = calculate_pullback_quality(closes, highs, lows)
        
        # 3. Volume Behavior - Volume supporting price direction
        features["volume_behavior_score"] = calculate_volume_behavior(closes, volumes)
        
        # 4. Psychology Score - Hidden pressure from wicks and false signals
        features["psych_score"] = calculate_psych_score(recent_candles)
        
        # 5. Support Reaction - Price reaction at key levels
        features["support_reaction"] = calculate_support_reaction(closes, volumes, highs, lows)
        
        # Add additional features for comprehensive analysis
        features["liquidity_pattern_score"] = calculate_liquidity_pattern(market_data)
        features["htf_supportive_score"] = calculate_htf_support(closes)
        
        logger.info(f"[FEATURE EXTRACT] Successfully extracted {len(features)} features")
        return features
        
    except Exception as e:
        logger.error(f"[FEATURE ERROR] Feature extraction failed: {e}")
        return None


def calculate_trend_strength(closes: List[float]) -> float:
    """
    Calculate trend strength based on price slope and EMA alignment
    
    Args:
        closes: List of closing prices
        
    Returns:
        Trend strength score (0.0 - 1.0)
    """
    try:
        if len(closes) < 10:
            return 0.0
            
        # Calculate price slope (linear regression)
        x = np.arange(len(closes))
        coefficients = np.polyfit(x, closes, 1)
        slope = coefficients[0]
        
        # Normalize slope by price range
        price_range = max(closes) - min(closes)
        if price_range == 0:
            return 0.0
            
        normalized_slope = abs(slope) / (price_range / len(closes))
        
        # Calculate EMA alignment (short vs long EMA)
        ema_short = calculate_ema(closes, 5)
        ema_long = calculate_ema(closes, 10)
        
        if ema_long == 0:
            ema_alignment = 0.0
        else:
            ema_alignment = abs(ema_short - ema_long) / ema_long
        
        # Combine slope and EMA alignment
        trend_strength = min(1.0, (normalized_slope * 0.6 + ema_alignment * 0.4) * 2)
        
        return round(trend_strength, 3)
        
    except Exception as e:
        logger.warning(f"[TREND CALC ERROR] {e}")
        return 0.0


def calculate_pullback_quality(closes: List[float], highs: List[float], lows: List[float]) -> float:
    """
    Calculate pullback quality - depth and regularity of corrections in trend
    
    Args:
        closes: List of closing prices
        highs: List of high prices
        lows: List of low prices
        
    Returns:
        Pullback quality score (0.0 - 1.0)
    """
    try:
        if len(closes) < 10:
            return 0.0
            
        # Identify trend direction
        trend_up = closes[-1] > closes[0]
        
        if trend_up:
            # For uptrend, measure pullbacks from highs
            peak_price = max(highs)
            current_price = closes[-1]
            pullback_depth = (peak_price - current_price) / peak_price if peak_price > 0 else 0
        else:
            # For downtrend, measure bounces from lows
            trough_price = min(lows)
            current_price = closes[-1]
            pullback_depth = (current_price - trough_price) / current_price if current_price > 0 else 0
        
        # Healthy pullback is 30-70% of recent move
        if 0.3 <= pullback_depth <= 0.7:
            quality_score = 1.0 - abs(pullback_depth - 0.5) * 2  # Peak quality at 50% pullback
        elif pullback_depth < 0.3:
            quality_score = pullback_depth / 0.3  # Shallow pullback
        else:
            quality_score = max(0.0, 1.0 - (pullback_depth - 0.7) / 0.3)  # Deep pullback
        
        return round(quality_score, 3)
        
    except Exception as e:
        logger.warning(f"[PULLBACK CALC ERROR] {e}")
        return 0.0


def calculate_volume_behavior(closes: List[float], volumes: List[float]) -> float:
    """
    Calculate volume behavior - whether volume supports price direction
    
    Args:
        closes: List of closing prices
        volumes: List of volumes
        
    Returns:
        Volume behavior score (0.0 - 1.0)
    """
    try:
        if len(closes) < 5 or len(volumes) < 5:
            return 0.0
            
        # Calculate recent price change
        price_change = closes[-1] - closes[-5]
        
        # Calculate volume trend
        recent_volume = np.mean(volumes[-3:])
        older_volume = np.mean(volumes[-10:-3]) if len(volumes) >= 10 else np.mean(volumes[:-3])
        
        if older_volume == 0:
            return 0.5
            
        volume_ratio = recent_volume / older_volume
        
        # Volume should increase with strong price moves
        if abs(price_change) > 0:
            # Strong price move with volume increase = good
            if volume_ratio > 1.2:
                volume_score = min(1.0, volume_ratio / 2.0)
            # Weak volume on price move = concerning
            elif volume_ratio < 0.8:
                volume_score = max(0.3, volume_ratio)
            else:
                volume_score = 0.6
        else:
            # Sideways price with high volume = accumulation/distribution
            volume_score = min(0.8, volume_ratio / 1.5)
        
        return round(volume_score, 3)
        
    except Exception as e:
        logger.warning(f"[VOLUME CALC ERROR] {e}")
        return 0.5


def calculate_psych_score(candles: List) -> float:
    """
    Calculate psychology score - hidden pressure from wicks and false signals
    
    Args:
        candles: List of candle data
        
    Returns:
        Psychology score (0.0 - 1.0)
    """
    try:
        if len(candles) < 5:
            return 0.0
            
        psych_signals = []
        
        for candle in candles[-5:]:
            if isinstance(candle, dict):
                high = float(candle.get("high", 0))
                low = float(candle.get("low", 0))
                open_price = float(candle.get("open", 0))
                close = float(candle.get("close", 0))
            elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                high = float(candle[2])
                low = float(candle[3])
                open_price = float(candle[1])
                close = float(candle[4])
            else:
                continue
                
            if high == low:  # Avoid division by zero
                continue
                
            # Calculate wick sizes
            body_size = abs(close - open_price)
            upper_wick = high - max(close, open_price)
            lower_wick = min(close, open_price) - low
            total_range = high - low
            
            # Psychology indicators
            wick_dominance = (upper_wick + lower_wick) / total_range
            
            # Doji or hammer patterns (small body, large wicks)
            if body_size / total_range < 0.3 and wick_dominance > 0.5:
                psych_signals.append(0.8)  # Indecision/reversal signal
            # Strong directional move with small wicks
            elif body_size / total_range > 0.7 and wick_dominance < 0.2:
                psych_signals.append(0.9)  # Strong conviction
            # Normal price action
            else:
                psych_signals.append(0.5)
        
        if not psych_signals:
            return 0.0
            
        psych_score = np.mean(psych_signals)
        return round(psych_score, 3)
        
    except Exception as e:
        logger.warning(f"[PSYCH CALC ERROR] {e}")
        return 0.0


def calculate_support_reaction(closes: List[float], volumes: List[float], 
                             highs: List[float], lows: List[float]) -> float:
    """
    Calculate support reaction - price reaction at key levels
    
    Args:
        closes: List of closing prices
        volumes: List of volumes
        highs: List of high prices
        lows: List of low prices
        
    Returns:
        Support reaction score (0.0 - 1.0)
    """
    try:
        if len(closes) < 10:
            return 0.0
            
        # Identify recent support/resistance levels
        recent_lows = sorted(lows[-10:])
        recent_highs = sorted(highs[-10:], reverse=True)
        
        current_price = closes[-1]
        prev_price = closes[-2] if len(closes) >= 2 else current_price
        recent_volume = volumes[-1] if volumes else 0
        avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else recent_volume
        
        reaction_score = 0.0
        
        # Check reaction from support levels
        for support_level in recent_lows[:3]:  # Check top 3 support levels
            if support_level > 0:
                distance_to_support = abs(current_price - support_level) / support_level
                
                # Price near support (within 2%)
                if distance_to_support < 0.02:
                    # Strong bounce with volume
                    if current_price > prev_price and recent_volume > avg_volume * 1.2:
                        reaction_score = max(reaction_score, 0.9)
                    # Weak bounce
                    elif current_price > prev_price:
                        reaction_score = max(reaction_score, 0.6)
                    # Breaking support
                    elif current_price < support_level:
                        reaction_score = max(reaction_score, 0.2)
        
        # Check reaction from resistance levels
        for resistance_level in recent_highs[:3]:  # Check top 3 resistance levels
            if resistance_level > 0:
                distance_to_resistance = abs(current_price - resistance_level) / resistance_level
                
                # Price near resistance (within 2%)
                if distance_to_resistance < 0.02:
                    # Strong breakout with volume
                    if current_price > resistance_level and recent_volume > avg_volume * 1.2:
                        reaction_score = max(reaction_score, 0.95)
                    # Rejection at resistance
                    elif current_price < prev_price:
                        reaction_score = max(reaction_score, 0.4)
        
        return round(reaction_score, 3)
        
    except Exception as e:
        logger.warning(f"[SUPPORT CALC ERROR] {e}")
        return 0.0


def calculate_liquidity_pattern(market_data: Dict) -> float:
    """
    Calculate liquidity pattern score from orderbook and volume
    
    Args:
        market_data: Market data including orderbook
        
    Returns:
        Liquidity pattern score (0.0 - 1.0)
    """
    try:
        # Basic liquidity analysis from volume
        if "volume_24h" in market_data:
            volume_24h = float(market_data["volume_24h"])
            # Higher volume generally indicates better liquidity
            # Normalize to 0-1 range (assuming 1M+ is good liquidity)
            liquidity_score = min(1.0, volume_24h / 1000000)
            return round(liquidity_score, 3)
        
        return 0.5  # Default moderate liquidity
        
    except Exception as e:
        logger.warning(f"[LIQUIDITY CALC ERROR] {e}")
        return 0.5


def calculate_htf_support(closes: List[float]) -> float:
    """
    Calculate higher timeframe support score
    
    Args:
        closes: List of closing prices
        
    Returns:
        HTF support score (0.0 - 1.0)
    """
    try:
        if len(closes) < 20:
            return 0.0
            
        # Calculate longer-term trend (20-period)
        long_term_slope = (closes[-1] - closes[0]) / len(closes)
        short_term_slope = (closes[-1] - closes[-5]) / 5 if len(closes) >= 5 else long_term_slope
        
        # HTF support when short-term aligns with long-term trend
        if long_term_slope != 0:
            alignment = 1.0 - abs(short_term_slope - long_term_slope) / abs(long_term_slope)
            htf_score = max(0.0, min(1.0, alignment))
        else:
            htf_score = 0.5
            
        return round(htf_score, 3)
        
    except Exception as e:
        logger.warning(f"[HTF CALC ERROR] {e}")
        return 0.0


def calculate_ema(prices: List[float], period: int) -> float:
    """
    Calculate Exponential Moving Average
    
    Args:
        prices: List of prices
        period: EMA period
        
    Returns:
        EMA value
    """
    try:
        if len(prices) < period:
            return np.mean(prices) if prices else 0.0
            
        multiplier = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
            
        return ema
        
    except Exception as e:
        logger.warning(f"[EMA CALC ERROR] {e}")
        return 0.0


def validate_features(features: Dict) -> bool:
    """
    Validate extracted features
    
    Args:
        features: Dictionary of features
        
    Returns:
        True if features are valid
    """
    required_features = [
        "trend_strength", "pullback_quality", "volume_behavior_score",
        "psych_score", "support_reaction", "liquidity_pattern_score", "htf_supportive_score"
    ]
    
    for feature in required_features:
        if feature not in features:
            logger.warning(f"[FEATURE VALIDATION] Missing feature: {feature}")
            return False
            
        value = features[feature]
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            logger.warning(f"[FEATURE VALIDATION] Invalid feature value {feature}: {value}")
            return False
    
    return True


def get_feature_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all extracted features
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return {
        "trend_strength": "Price slope and EMA alignment indicating directional momentum",
        "pullback_quality": "Depth and regularity of corrections within the trend",
        "volume_behavior_score": "Whether volume pattern supports price direction",
        "psych_score": "Hidden market pressure from wicks and false signals",
        "support_reaction": "Price reaction quality at key support/resistance levels",
        "liquidity_pattern_score": "Orderbook depth and liquidity availability",
        "htf_supportive_score": "Higher timeframe trend alignment and support"
    }


if __name__ == "__main__":
    # Test feature extraction
    test_data = {
        "candles_15m": [
            [1625097600000, 1.0, 1.1, 0.9, 1.05, 1000],
            [1625098500000, 1.05, 1.15, 1.0, 1.1, 1200],
            [1625099400000, 1.1, 1.2, 1.05, 1.15, 1100],
            # Add more test candles...
        ] + [[1625097600000 + i*900000, 1.0+i*0.01, 1.1+i*0.01, 0.9+i*0.01, 1.05+i*0.01, 1000+i*10] for i in range(3, 25)],
        "volume_24h": 5000000,
        "price_usd": 1.25
    }
    
    features = extract_features(test_data)
    if features:
        print("✅ Feature extraction test successful:")
        for feature, value in features.items():
            print(f"  {feature}: {value}")
    else:
        print("❌ Feature extraction test failed")