"""
Unified Price Reference Resolution
Single price_ref for entire run cycle
"""

def resolve_price_ref(ticker_price, candle_price):
    """
    Resolve price reference using priority: ticker > candle > error
    
    Args:
        ticker_price: Price from ticker data
        candle_price: Price from candle data
        
    Returns:
        float: Valid price reference
        
    Raises:
        ValueError: If no valid price available
    """
    p_t = ticker_price or 0.0
    p_c = candle_price or 0.0
    
    if p_t > 0:
        return p_t
    if p_c > 0:
        return p_c
    
    raise ValueError("No valid price_ref")