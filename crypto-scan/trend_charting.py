"""
Vision-AI Optimized Chart Generation for CLIP Training - DISABLED
🚫 ALL MATPLOTLIB FUNCTIONS DISABLED - TradingView-only system active
All functions in this module now return None and redirect to TradingView screenshot system
"""

import os
from datetime import datetime
from typing import Optional, Dict, List


def plot_chart_with_context(symbol, candles, alert_indices=None, alert_index=None, score=None, decision=None, phase=None, setup=None, save_path="chart.png", context_days=2):
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        candles: List of candle data
        alert_indices: List of alert indices
        alert_index: Single alert index
        score: TJDE score
        decision: Trading decision
        phase: Market phase
        setup: Setup description
        save_path: Path to save chart
        context_days: Days of context
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Chart generation disabled, using TradingView-only system")
    return None


def prepare_ohlcv_dataframes(candles):
    """
    Prepare OHLCV dataframes with validation but no matplotlib processing
    
    Args:
        candles: List of candle data
        
    Returns:
        Dict with validation results
    """
    if not candles:
        return {"valid": False, "reason": "No candles provided"}
    
    valid_candles = 0
    for candle in candles[:10]:  # Check first 10 candles
        try:
            if isinstance(candle, dict):
                # Dictionary format validation
                required_keys = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if all(key in candle for key in required_keys):
                    valid_candles += 1
            elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                # Array format validation: [timestamp, open, high, low, close, volume]
                if all(isinstance(candle[i], (int, float)) for i in range(6)):
                    valid_candles += 1
        except (ValueError, TypeError, IndexError):
            continue
    
    validation_result = {
        "valid": valid_candles >= 5,
        "valid_candles": valid_candles,
        "total_candles": len(candles),
        "reason": f"Found {valid_candles} valid candles out of {len(candles)}"
    }
    
    return validation_result


def plot_chart_vision_ai(symbol, candles, alert_index=None, score=None, decision=None, phase=None, setup=None, save_path=None):
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        candles: List of candle data
        alert_index: Alert index
        score: TJDE score
        decision: Trading decision
        phase: Market phase
        setup: Setup description
        save_path: Path to save chart
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Vision-AI chart generation disabled, using TradingView-only system")
    return None


def create_dark_theme_chart(symbol, candles, alerts=None, output_path=None):
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        candles: Candle data
        alerts: Alert data
        output_path: Output path
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Dark theme chart disabled, using TradingView-only system")
    return None


def generate_training_chart(symbol, market_data, tjde_result, output_dir="training_data/charts"):
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        market_data: Market data
        tjde_result: TJDE result
        output_dir: Output directory
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Training chart disabled, using TradingView-only system")
    return None


def validate_chart_requirements(candles, min_candles=10):
    """
    Validate chart requirements without matplotlib processing
    
    Args:
        candles: List of candle data
        min_candles: Minimum required candles
        
    Returns:
        Dict with validation status
    """
    if not candles:
        return {
            "valid": False,
            "reason": "No candles provided",
            "candle_count": 0
        }
    
    if len(candles) < min_candles:
        return {
            "valid": False,
            "reason": f"Insufficient candles: {len(candles)} < {min_candles}",
            "candle_count": len(candles)
        }
    
    # Validate data format
    validation = prepare_ohlcv_dataframes(candles)
    
    return {
        "valid": validation["valid"],
        "reason": validation["reason"],
        "candle_count": len(candles),
        "valid_candles": validation.get("valid_candles", 0)
    }


# Export main functions for compatibility
__all__ = [
    'plot_chart_with_context',
    'prepare_ohlcv_dataframes',
    'plot_chart_vision_ai',
    'create_dark_theme_chart',
    'generate_training_chart',
    'validate_chart_requirements'
]


if __name__ == "__main__":
    print("[TEST] All trend charting matplotlib functions disabled")
    print("[TEST] Use TradingView screenshot system for trend charts")