"""
Price Reference Module
Provides consistent price reference for entire scan run
"""

def resolve_price_ref(ticker_price: Optional[float], candle_price: Optional[float]) -> float:
    """
    Return one price_ref for entire scan. Prefer ticker if >0,
    otherwise use candle. Raise ValueError if both None/0.
    
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
    
    raise ValueError("No valid price_ref available (both ticker and candle prices are 0 or None)")