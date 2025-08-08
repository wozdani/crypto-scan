"""
Whale Ping Detector with CEX Address Exclusion
Detects whale transactions while filtering out known exchange addresses
"""

def is_cex(addr: str, known_exchanges: set[str]) -> bool:
    """
    Check if address is a known exchange address
    
    Args:
        addr: Address to check
        known_exchanges: Set of known exchange addresses (lowercase)
        
    Returns:
        True if address is a known exchange
    """
    a = addr.lower()
    return a in known_exchanges


def detect_whales(transfers, vol_24h_usd, known_exchanges):
    """
    Detect whale transfers excluding CEX addresses
    
    Args:
        transfers: List of transfers to analyze
        vol_24h_usd: 24h volume in USD for threshold calculation
        known_exchanges: Set of known exchange addresses
        
    Returns:
        Dict with active status, strength, and metadata
    """
    # Dynamic threshold: 2% of daily volume, clamped between $25k-$150k
    th = max(25_000.0, min(150_000.0, 0.02 * vol_24h_usd))
    
    whales = []
    for t in transfers:
        # Only count transfers above threshold that don't involve CEX addresses
        if (t["amount_usd"] >= th and 
            not is_cex(t["from"], known_exchanges) and 
            not is_cex(t["to"], known_exchanges)):
            whales.append(t)
    
    s = 1.0 if whales else 0.0
    return {
        "active": bool(whales), 
        "strength": s, 
        "meta": {"th_usd": th, "hits": len(whales)}
    }