"""
Labeler Module
Triple-barrier labeling for price movements within 6 hours
"""

from typing import List, Tuple, Dict, Optional

def label_triple_barrier(
    prices: List[Tuple[float, float]], 
    t0_idx: int = 0,
    tp: float = 0.04, 
    sl: float = 0.02, 
    ttl_min: int = 360
) -> Dict:
    """
    Apply triple-barrier labeling to price series
    
    Args:
        prices: List of (timestamp, price) tuples at 1-5 minute intervals
        t0_idx: Index of starting point
        tp: Take profit threshold (e.g., 0.04 for 4%)
        sl: Stop loss threshold (e.g., 0.02 for 2%)
        ttl_min: Time to live in minutes (e.g., 360 for 6 hours)
    
    Returns:
        Dict with labeling results:
        - y_hit_6h: 1 if hit take profit within 6h, 0 otherwise
        - max_return_6h: Maximum return achieved within 6h
        - time_to_peak_min: Time to peak in minutes
        - triple_barrier: Dict with label (+1, 0, -1), tp, sl, ttl_min
    """
    if not prices or t0_idx >= len(prices):
        return {
            "y_hit_6h": 0,
            "max_return_6h": 0.0,
            "time_to_peak_min": 0,
            "triple_barrier": {"label": 0, "tp": tp, "sl": sl, "ttl_min": ttl_min}
        }
    
    p0 = prices[t0_idx][1]
    t0 = prices[t0_idx][0]
    
    if p0 <= 0:
        return {
            "y_hit_6h": 0,
            "max_return_6h": 0.0,
            "time_to_peak_min": 0,
            "triple_barrier": {"label": 0, "tp": tp, "sl": sl, "ttl_min": ttl_min}
        }
    
    tp_px = p0 * (1 + tp)
    sl_px = p0 * (1 - sl)
    
    peak = p0
    t_peak = t0
    
    for ts, px in prices[t0_idx + 1:]:
        # Track peak
        if px > peak:
            peak = px
            t_peak = ts
        
        # Check take profit
        if px >= tp_px:
            return {
                "y_hit_6h": 1,
                "max_return_6h": (peak / p0 - 1),
                "time_to_peak_min": int((t_peak - t0) / 60),
                "triple_barrier": {"label": 1, "tp": tp, "sl": sl, "ttl_min": ttl_min}
            }
        
        # Check stop loss
        if px <= sl_px:
            return {
                "y_hit_6h": 0,
                "max_return_6h": (peak / p0 - 1),
                "time_to_peak_min": int((t_peak - t0) / 60),
                "triple_barrier": {"label": -1, "tp": tp, "sl": sl, "ttl_min": ttl_min}
            }
        
        # Check time limit
        if (ts - t0) / 60 >= ttl_min:
            break
    
    # Time expired - check if peak exceeded take profit
    return {
        "y_hit_6h": int((peak / p0 - 1) >= tp),
        "max_return_6h": (peak / p0 - 1),
        "time_to_peak_min": int((t_peak - t0) / 60),
        "triple_barrier": {"label": 0, "tp": tp, "sl": sl, "ttl_min": ttl_min}
    }