"""
Vision-AI Chart Generator - DISABLED
🚫 ALL MATPLOTLIB FUNCTIONS DISABLED - TradingView-only system active
All functions in this module now return None and redirect to TradingView screenshot system
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime


def plot_chart_vision_ai(
    symbol: str, 
    candles: List = None, 
    alert_index: int = None, 
    alert_indices: List = None, 
    score: float = None, 
    decision: str = None, 
    phase: str = None, 
    setup: str = None,
    save_path: str = None, 
    context_days: int = 2, 
    force_fresh: bool = True
) -> Optional[str]:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        candles: List of candle data
        alert_index: Single alert index
        alert_indices: Multiple alert indices
        score: TJDE score
        decision: Trading decision
        phase: Market phase
        setup: Setup description
        save_path: Custom save path
        context_days: Days of context
        force_fresh: Force fresh data fetch
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Vision-AI chart generation disabled, using TradingView-only system")
    return None


def generate_vision_ai_chart(
    symbol: str,
    market_data: Dict,
    tjde_result: Dict,
    output_dir: str = "training_data/charts"
) -> Optional[str]:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        market_data: Market data dictionary
        tjde_result: TJDE analysis result
        output_dir: Output directory
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Vision-AI chart generation disabled, using TradingView-only system")
    return None


def create_vision_training_chart(
    symbol: str,
    candles: List,
    alert_moment: int,
    tjde_data: Dict
) -> Optional[str]:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        candles: Candle data
        alert_moment: Alert moment index
        tjde_data: TJDE data
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Vision-AI training chart disabled, using TradingView-only system")
    return None


def validate_chart_data(candles: List) -> bool:
    """
    Validate chart data without generating matplotlib charts
    
    Args:
        candles: List of candle data
        
    Returns:
        bool: True if data is valid for TradingView screenshot
    """
    if not candles:
        return False
    
    if len(candles) < 10:
        return False
    
    # Basic validation without matplotlib processing
    valid_count = 0
    for candle in candles[:5]:  # Check first 5 candles
        try:
            if isinstance(candle, dict):
                required_keys = ['open', 'high', 'low', 'close']
                if all(key in candle for key in required_keys):
                    valid_count += 1
            elif isinstance(candle, list) and len(candle) >= 5:
                if all(isinstance(candle[i], (int, float)) for i in range(1, 5)):
                    valid_count += 1
        except:
            continue
    
    return valid_count >= 3  # At least 3 valid candles


# Export main functions for compatibility
__all__ = [
    'plot_chart_vision_ai',
    'generate_vision_ai_chart',
    'create_vision_training_chart',
    'validate_chart_data'
]


if __name__ == "__main__":
    print("[TEST] All Vision-AI matplotlib chart generation functions disabled")
    print("[TEST] Use TradingView screenshot system for Vision-AI charts")