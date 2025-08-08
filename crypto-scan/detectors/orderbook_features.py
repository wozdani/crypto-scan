"""
Orderbook Features Detector
Implements USD-based calculations and real L2 orderbook validation
"""

from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.stealth_config import STEALTH

def ob_valid(ob: Dict) -> bool:
    """
    Check if orderbook is valid (real L2 with sufficient depth)
    
    Args:
        ob: Orderbook metadata dictionary
    
    Returns:
        bool: True if orderbook is valid for analysis
    """
    # Check if it's real orderbook (not synthetic)
    is_real = ob.get("source", "synthetic") == "real"
    
    # Check minimum depth in USD
    depth_usd = ob.get("top10_depth_usd", 0.0)
    has_depth = depth_usd >= STEALTH["OB_DEPTH_TOP10_MIN_USD"]
    
    if not is_real:
        print(f"[ORDERBOOK] Synthetic orderbook detected - skipping OB signals")
    elif not has_depth:
        print(f"[ORDERBOOK] Insufficient depth ${depth_usd:,.0f} < ${STEALTH['OB_DEPTH_TOP10_MIN_USD']:,.0f}")
    
    return is_real and has_depth

def compute_ob_signals(bids: List, asks: List, price_ref: float, meta: Dict) -> Dict:
    """
    Compute orderbook signals with USD-based thresholds
    
    Args:
        bids: List of bid orders
        asks: List of ask orders
        price_ref: Reference price for USD calculations
        meta: Orderbook metadata
    
    Returns:
        Dict with enabled status and detected signals
    """
    # Validate orderbook first
    if not ob_valid(meta):
        return {"enabled": False, "signals": {}, "reason": "invalid_orderbook"}
    
    # Helper function to convert to USD
    def to_usd(level):
        if isinstance(level, dict):
            return level.get("size", 0) * price_ref
        elif isinstance(level, list) and len(level) >= 2:
            return float(level[1]) * price_ref
        return 0.0
    
    # Get top levels
    top_bids = bids[:10] if bids else []
    top_asks = asks[:10] if asks else []
    
    # Calculate max order sizes in USD
    max_bid_usd = max((to_usd(x) for x in top_bids), default=0.0)
    max_ask_usd = max((to_usd(x) for x in top_asks), default=0.0)
    
    # Calculate spread percentage
    spread_pct = 0.0
    if top_bids and top_asks:
        if isinstance(top_bids[0], dict):
            bid_price = top_bids[0].get("price", 0)
            ask_price = top_asks[0].get("price", 0)
        elif isinstance(top_bids[0], list) and len(top_bids[0]) >= 1:
            bid_price = float(top_bids[0][0])
            ask_price = float(top_asks[0][0]) if isinstance(top_asks[0], list) else 0
        else:
            bid_price = ask_price = 0
        
        if bid_price > 0 and ask_price > 0:
            spread_pct = abs(ask_price - bid_price) / price_ref

    signals = {}
    
    # Large bid walls detection (USD-based)
    if max_bid_usd >= STEALTH["OB_WALL_MIN_USD"]:
        signals["large_bid_walls"] = {
            "active": True, 
            "strength": min(1.0, max_bid_usd / (STEALTH["OB_WALL_MIN_USD"] * 2)),
            "usd_value": max_bid_usd
        }
        print(f"[ORDERBOOK] Large bid wall detected: ${max_bid_usd:,.0f}")
    
    # Large ask walls detection (USD-based)
    if max_ask_usd >= STEALTH["OB_WALL_MIN_USD"]:
        signals["large_ask_walls"] = {
            "active": True,
            "strength": min(1.0, max_ask_usd / (STEALTH["OB_WALL_MIN_USD"] * 2)),
            "usd_value": max_ask_usd
        }
        print(f"[ORDERBOOK] Large ask wall detected: ${max_ask_usd:,.0f}")
    
    # Order Flow Imbalance (OFI)
    ofi = meta.get("ofi", 0.0)
    if ofi >= STEALTH["OB_OFI_MIN"]:
        signals["orderbook_ofi"] = {
            "active": True,
            "strength": min(1.0, ofi / (STEALTH["OB_OFI_MIN"] * 2)),
            "value": ofi
        }
    
    # Queue Imbalance
    queue_imb = meta.get("queue_imb", 0.0)
    if queue_imb >= STEALTH["OB_QUEUE_IMB_MIN"]:
        signals["orderbook_queue_imb"] = {
            "active": True,
            "strength": min(1.0, queue_imb / STEALTH["OB_QUEUE_IMB_MIN"]),
            "value": queue_imb
        }
    
    # Spoofing detection based on rapid changes
    if meta.get("rapid_changes", 0) > 5:
        signals["spoofing_detected"] = {
            "active": True,
            "strength": 0.7,
            "changes": meta.get("rapid_changes")
        }
    
    return {
        "enabled": True,
        "signals": signals,
        "spread_pct": spread_pct,
        "max_bid_usd": max_bid_usd,
        "max_ask_usd": max_ask_usd
    }