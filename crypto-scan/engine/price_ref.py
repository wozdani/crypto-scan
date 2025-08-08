"""
Unified Price Reference Resolution
Single price_ref for entire run cycle
"""

from utils.decorators import once_per_scan


@once_per_scan("FEATURES", "engine")
def resolve_price_ref(ticker_price, candle_price):
    """
    Resolve price reference using priority: ticker > candle > error
    Cached for entire scan cycle with @once_per_scan
    
    Args:
        ticker_price: Price from ticker data
        candle_price: Price from candle data
        
    Returns:
        float: Valid price reference
        
    Raises:
        ValueError: If no valid price available
    """
    p_t = float(ticker_price or 0.0)
    p_c = float(candle_price or 0.0)
    p = p_t if p_t > 0 else p_c
    if p <= 0: 
        raise ValueError("No valid price_ref")
    return p