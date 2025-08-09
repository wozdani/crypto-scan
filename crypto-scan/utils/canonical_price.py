"""
Canonical Price System - Eliminates price mismatch chaos
Single source of truth per token per scan round
"""

from typing import Dict, Any, Tuple, Optional

def canonical_price(market_data: Dict[str, Any]) -> Tuple[float, str]:
    """
    Extract canonical price - ONE source of truth per token per scan round
    
    Args:
        market_data: Market data dict with price sources
        
    Returns:
        tuple: (price, source) where source is "ticker"|"candle_15m"|"candle_5m"|"failed"
    """
    
    symbol = market_data.get("symbol", "UNKNOWN")
    
    # Priority 1: Real-time ticker price (preferred)
    ticker_price = market_data.get("price_usd", 0)
    if ticker_price and ticker_price > 0:
        return ticker_price, "ticker"
    
    # Priority 2: 15M candle close price (most reliable fallback)
    candles_15m = market_data.get("candles_15m", [])
    if candles_15m and len(candles_15m) > 0:
        try:
            last_candle = candles_15m[-1]  # Most recent candle
            
            # Handle dict format {"close": price}
            if isinstance(last_candle, dict) and "close" in last_candle:
                candle_price = float(last_candle["close"])
                if candle_price > 0:
                    return candle_price, "candle_15m"
            
            # Handle OHLCV array format [open, high, low, close, volume]
            elif isinstance(last_candle, (list, tuple)) and len(last_candle) >= 5:
                candle_price = float(last_candle[4])  # Close price index
                if candle_price > 0:
                    return candle_price, "candle_15m"
                    
        except (ValueError, IndexError, TypeError) as e:
            print(f"[CANONICAL PRICE] {symbol}: 15M candle price extraction failed: {e}")
    
    # Priority 3: 5M candle close price (last resort)
    candles_5m = market_data.get("candles_5m", [])
    if candles_5m and len(candles_5m) > 0:
        try:
            last_candle = candles_5m[-1]  # Most recent candle
            
            # Handle dict format
            if isinstance(last_candle, dict) and "close" in last_candle:
                candle_price = float(last_candle["close"])
                if candle_price > 0:
                    return candle_price, "candle_5m"
            
            # Handle OHLCV array format
            elif isinstance(last_candle, (list, tuple)) and len(last_candle) >= 5:
                candle_price = float(last_candle[4])
                if candle_price > 0:
                    return candle_price, "candle_5m"
                    
        except (ValueError, IndexError, TypeError) as e:
            print(f"[CANONICAL PRICE] {symbol}: 5M candle price extraction failed: {e}")
    
    # All sources failed
    print(f"[CANONICAL PRICE] {symbol}: All price sources failed - no valid price available")
    return 0.0, "failed"


def freeze_canonical_price(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Freeze canonical price in market_data to prevent multiple fallbacks per round
    
    Args:
        market_data: Original market data
        
    Returns:
        Updated market_data with canonical_price and canonical_price_source
    """
    
    symbol = market_data.get("symbol", "UNKNOWN")
    
    # Check if already frozen
    if "canonical_price" in market_data and "canonical_price_source" in market_data:
        return market_data  # Already processed
    
    # Extract canonical price
    price, source = canonical_price(market_data)
    
    # Freeze in market_data
    market_data["canonical_price"] = price
    market_data["canonical_price_source"] = source
    
    # Update price_usd to canonical for compatibility
    market_data["price_usd"] = price
    
    print(f"[CANONICAL PRICE] {symbol}: Frozen price=${price:.6f} source={source}")
    
    return market_data


def get_canonical_price_log(market_data: Dict[str, Any]) -> str:
    """
    Generate log string for canonical price debugging
    
    Args:
        market_data: Market data with canonical price
        
    Returns:
        Formatted log string
    """
    
    symbol = market_data.get("symbol", "UNKNOWN")
    price = market_data.get("canonical_price", 0)
    source = market_data.get("canonical_price_source", "unknown")
    
    return f"{symbol}: ${price:.6f} ({source})"


def validate_canonical_price(market_data: Dict[str, Any]) -> bool:
    """
    Validate that canonical price is available and valid
    
    Args:
        market_data: Market data to validate
        
    Returns:
        True if canonical price is valid, False otherwise
    """
    
    price = market_data.get("canonical_price", 0)
    source = market_data.get("canonical_price_source", "unknown")
    
    if price <= 0 or source == "failed":
        return False
        
    return True


# Export functions for easy import
__all__ = [
    "canonical_price",
    "freeze_canonical_price", 
    "get_canonical_price_log",
    "validate_canonical_price"
]