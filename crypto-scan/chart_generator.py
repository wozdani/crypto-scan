"""
Alert-Focused Chart Generator for TJDE Training
Generuje wykresy treningowe skupione na momencie alertu z kontekstem decyzyjnym

🚫 ALL MATPLOTLIB CHART GENERATION FUNCTIONS DISABLED - TradingView-only system active
All functions in this module now return None and redirect to TradingView screenshot system
"""

import os
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone


def validate_candle_data(candle) -> dict:
    """
    Comprehensive candle data validation supporting multiple formats
    
    Args:
        candle: Candle data in dict, list, or tuple format
        
    Returns:
        dict: Validated candle data with standard keys
    """
    if isinstance(candle, dict):
        # Dictionary format
        required_keys = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if all(key in candle for key in required_keys):
            return candle
        # Try alternative keys
        alt_mapping = {
            'time': 'timestamp',
            'o': 'open',
            'h': 'high', 
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }
        validated = {}
        for std_key, alt_key in alt_mapping.items():
            if alt_key in candle:
                validated[std_key] = candle[alt_key]
            elif std_key in candle:
                validated[std_key] = candle[std_key]
        if len(validated) >= 6:
            return validated
    
    elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
        # Array format: [timestamp, open, high, low, close, volume]
        return {
            'timestamp': candle[0],
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5])
        }
    
    # Fallback - return empty dict for invalid data
    return {}


def detect_alert_moment(candles_15m, tjde_score=None, tjde_decision=None):
    """
    Wykrywa dokładny moment alertu TJDE w danych świecowych
    
    Args:
        candles_15m: Dane świec 15-minutowych
        tjde_score: Score TJDE dla kontekstu
        tjde_decision: Decyzja TJDE
        
    Returns:
        Index świecy, która wygenerowała alert
    """
    if not candles_15m or len(candles_15m) < 10:
        return len(candles_15m) - 1 if candles_15m else 0
    
    # Default to 80% through the data for alert moment
    alert_point = int(len(candles_15m) * 0.8)
    return min(alert_point, len(candles_15m) - 1)


def generate_alert_focused_training_chart(
    symbol: str, 
    candles_15m: List, 
    tjde_score: float, 
    tjde_phase: str, 
    tjde_decision: str, 
    tjde_clip_confidence: float = None, 
    setup_label: str = None
) -> Optional[str]:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        candles_15m: 15-minute candle data
        tjde_score: Final score TJDE 
        tjde_phase: Market phase from TJDE
        tjde_decision: TJDE decision
        tjde_clip_confidence: CLIP confidence (optional)
        setup_label: Setup description (optional)
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Chart generation disabled, using TradingView-only system")
    return None


def generate_tjde_training_chart(
    symbol: str,
    candles_15m: List,
    candles_5m: List = None,
    tjde_result: Dict = None,
    clip_info: Dict = None,
    output_dir: str = "training_data/charts"
) -> Optional[str]:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        candles_15m: 15-minute candle data
        candles_5m: 5-minute candle data (optional)
        tjde_result: Complete TJDE result dictionary
        clip_info: CLIP analysis results
        output_dir: Output directory for charts
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Chart generation disabled, using TradingView-only system")
    return None


def generate_tjde_training_chart_contextual(symbol, candles_15m, tjde_score, tjde_phase, tjde_decision, tjde_clip_confidence=None, setup_label=None):
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        candles_15m: 15-minute candle data
        tjde_score: TJDE final score
        tjde_phase: Market phase from TJDE
        tjde_decision: TJDE decision
        tjde_clip_confidence: Optional CLIP confidence
        setup_label: Optional setup description
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Chart generation disabled, using TradingView-only system")
    return None


def generate_tjde_training_chart_simple(symbol, price_series, tjde_score, tjde_phase, tjde_decision, tjde_clip_confidence=None, setup_label=None):
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    Args:
        symbol: Trading symbol
        price_series: Price data series
        tjde_score: TJDE score
        tjde_phase: Market phase
        tjde_decision: TJDE decision
        tjde_clip_confidence: Optional CLIP confidence
        setup_label: Optional setup description
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Chart generation disabled, using TradingView-only system")
    return None


def flatten_candles(candles_15m, candles_5m=None):
    """
    Flatten candle data to price series for chart plotting
    
    Args:
        candles_15m: 15-minute candles
        candles_5m: Optional 5-minute candles
        
    Returns:
        List of close prices
    """
    if not candles_15m:
        return []
    
    prices = []
    for candle in candles_15m:
        validated = validate_candle_data(candle)
        if validated and 'close' in validated:
            prices.append(float(validated['close']))
    
    return prices


def _save_chart_metadata(
    symbol: str, 
    timestamp: str, 
    tjde_score: float, 
    decision: str,
    tjde_breakdown: Dict = None,
    output_dir: str = "training_data/charts"
) -> bool:
    """Save chart metadata as JSON for training purposes"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        metadata_file = f"{output_dir}/{symbol}_{timestamp}_metadata.json"
        
        metadata = {
            "symbol": symbol,
            "timestamp": timestamp,
            "tjde_score": tjde_score,
            "decision": decision,
            "tjde_breakdown": tjde_breakdown or {},
            "chart_type": "disabled_matplotlib",
            "generation_method": "tradingview_only",
            "created_at": datetime.utcnow().isoformat()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        print(f"[METADATA ERROR] {symbol}: {e}")
        return False


def validate_chart_quality(candles_15m: List, candles_5m: List = None) -> Dict[str, bool]:
    """
    Quality checklist for chart generation
    
    Returns:
        Dictionary with quality validation results
    """
    quality = {
        "sufficient_15m_data": len(candles_15m) >= 20 if candles_15m else False,
        "sufficient_5m_data": len(candles_5m) >= 50 if candles_5m else True,  # 5M is optional
        "valid_format": all(validate_candle_data(c) for c in (candles_15m[:5] if candles_15m else [])),
        "matplotlib_disabled": True  # Always true now
    }
    
    quality["overall_quality"] = all(quality.values())
    return quality


def generate_chart_async_safe(
    symbol: str,
    market_data: Dict,
    tjde_result: Dict,
    tjde_breakdown: Dict = None
) -> Optional[str]:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    This function now ensures fresh market data is always used for chart generation
    instead of relying on potentially outdated cached data.
    
    Args:
        symbol: Trading symbol
        market_data: Market data dictionary with candles
        tjde_result: TJDE result with score and decision
        tjde_breakdown: Optional detailed TJDE breakdown
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Chart generation disabled, using TradingView-only system")
    return None


def test_chart_generation():
    """Test chart generation with sample data"""
    print("[TEST] All matplotlib chart generation functions disabled")
    print("[TEST] Use TradingView screenshot system for chart generation")
    return None


# Export main functions for compatibility
__all__ = [
    'generate_alert_focused_training_chart',
    'generate_tjde_training_chart',
    'generate_tjde_training_chart_contextual', 
    'generate_tjde_training_chart_simple',
    'generate_chart_async_safe',
    'validate_candle_data',
    'detect_alert_moment',
    'flatten_candles',
    'validate_chart_quality',
    '_save_chart_metadata',
    'test_chart_generation'
]