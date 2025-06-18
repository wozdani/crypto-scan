#!/usr/bin/env python3
"""
VWAP Pinning Detector
Detects when price closely follows VWAP indicating potential accumulation
Transferred from crypto-scan for pump-analysis independence
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def detect_vwap_pinning(df: pd.DataFrame, tolerance: float = 0.005) -> Tuple[bool, float, str]:
    """
    Detect VWAP pinning pattern
    
    Args:
        df: DataFrame with OHLCV data and VWAP column
        tolerance: Maximum distance from VWAP as percentage (0.005 = 0.5%)
    
    Returns:
        tuple: (is_pinning, avg_distance, pattern_strength)
    """
    try:
        if df is None or len(df) < 10:
            return False, 0.0, "insufficient_data"
        
        # Calculate VWAP if not present
        if 'vwap' not in df.columns:
            df = calculate_vwap(df)
        
        # Get recent data (last 10 candles)
        recent_df = df.tail(10).copy()
        
        if len(recent_df) < 5:
            return False, 0.0, "insufficient_recent_data"
        
        # Calculate distance from VWAP for close prices
        vwap_distances = []
        for _, row in recent_df.iterrows():
            if pd.isna(row['vwap']) or row['vwap'] <= 0:
                continue
            
            distance = abs(row['close'] - row['vwap']) / row['vwap']
            vwap_distances.append(distance)
        
        if not vwap_distances:
            return False, 0.0, "no_valid_vwap_data"
        
        avg_distance = np.mean(vwap_distances)
        max_distance = max(vwap_distances)
        
        # Check pinning criteria
        is_pinning = False
        pattern_strength = "weak"
        
        # Strong pinning: most candles very close to VWAP
        tight_candles = sum(1 for d in vwap_distances if d <= tolerance)
        pinning_ratio = tight_candles / len(vwap_distances)
        
        if pinning_ratio >= 0.7 and avg_distance <= tolerance:
            is_pinning = True
            if avg_distance <= tolerance * 0.5:
                pattern_strength = "very_strong"
            else:
                pattern_strength = "strong"
        elif pinning_ratio >= 0.5 and avg_distance <= tolerance * 1.5:
            is_pinning = True
            pattern_strength = "moderate"
        
        # Additional check for volume confirmation
        if is_pinning and 'volume' in df.columns:
            recent_volume = recent_df['volume'].mean()
            historical_volume = df['volume'].tail(50).mean()
            
            if recent_volume > historical_volume * 1.2:
                if pattern_strength == "moderate":
                    pattern_strength = "strong"
                elif pattern_strength == "strong":
                    pattern_strength = "very_strong"
        
        logger.debug(f"VWAP pinning analysis: avg_distance={avg_distance:.4f}, "
                    f"pinning_ratio={pinning_ratio:.2f}, strength={pattern_strength}")
        
        return is_pinning, avg_distance, pattern_strength
        
    except Exception as e:
        logger.error(f"Error in VWAP pinning detection: {e}")
        return False, 0.0, "error"

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with VWAP column added
    """
    try:
        df = df.copy()
        
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate volume * typical price
        df['volume_price'] = df['typical_price'] * df['volume']
        
        # Calculate cumulative sums
        df['cum_volume_price'] = df['volume_price'].cumsum()
        df['cum_volume'] = df['volume'].cumsum()
        
        # Calculate VWAP
        df['vwap'] = df['cum_volume_price'] / df['cum_volume']
        
        # Clean up temporary columns
        df.drop(['typical_price', 'volume_price', 'cum_volume_price', 'cum_volume'], 
                axis=1, inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating VWAP: {e}")
        return df

def analyze_vwap_trend(df: pd.DataFrame, period: int = 20) -> dict:
    """
    Analyze VWAP trend and price relationship
    
    Args:
        df: DataFrame with OHLCV data and VWAP
        period: Number of periods to analyze
    
    Returns:
        Dictionary with VWAP analysis results
    """
    try:
        if 'vwap' not in df.columns:
            df = calculate_vwap(df)
        
        recent_df = df.tail(period).copy()
        
        if len(recent_df) < 5:
            return {"error": "insufficient_data"}
        
        # VWAP trend analysis
        vwap_values = recent_df['vwap'].dropna()
        if len(vwap_values) < 3:
            return {"error": "insufficient_vwap_data"}
        
        vwap_trend = "neutral"
        if vwap_values.iloc[-1] > vwap_values.iloc[0]:
            vwap_trend = "rising"
        elif vwap_values.iloc[-1] < vwap_values.iloc[0]:
            vwap_trend = "falling"
        
        # Price vs VWAP position
        current_price = recent_df['close'].iloc[-1]
        current_vwap = vwap_values.iloc[-1]
        
        price_position = "on_vwap"
        if current_price > current_vwap * 1.002:
            price_position = "above_vwap"
        elif current_price < current_vwap * 0.998:
            price_position = "below_vwap"
        
        # Support/Resistance analysis
        touches = 0
        bounces = 0
        
        for i in range(1, len(recent_df) - 1):
            row = recent_df.iloc[i]
            if pd.isna(row['vwap']):
                continue
            
            # Check for VWAP touch
            low_distance = abs(row['low'] - row['vwap']) / row['vwap']
            high_distance = abs(row['high'] - row['vwap']) / row['vwap']
            
            if min(low_distance, high_distance) <= 0.003:  # 0.3% tolerance
                touches += 1
                
                # Check for bounce
                next_row = recent_df.iloc[i + 1]
                if not pd.isna(next_row['close']):
                    if (row['low'] <= row['vwap'] <= row['high'] and 
                        next_row['close'] > row['vwap']):
                        bounces += 1
        
        support_strength = "weak"
        if touches >= 3:
            bounce_ratio = bounces / touches if touches > 0 else 0
            if bounce_ratio >= 0.6:
                support_strength = "strong"
            elif bounce_ratio >= 0.4:
                support_strength = "moderate"
        
        return {
            "vwap_trend": vwap_trend,
            "price_position": price_position,
            "current_distance": abs(current_price - current_vwap) / current_vwap,
            "vwap_touches": touches,
            "vwap_bounces": bounces,
            "support_strength": support_strength,
            "current_vwap": current_vwap,
            "current_price": current_price
        }
        
    except Exception as e:
        logger.error(f"Error in VWAP trend analysis: {e}")
        return {"error": str(e)}