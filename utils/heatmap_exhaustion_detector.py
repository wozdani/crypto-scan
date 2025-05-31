def detect_heatmap_exhaustion(data):
    """
    Detect orderbook supply exhaustion (heatmap exhaustion)
    
    Conditions for activation:
    - Large ask wall disappeared recently
    - Volume spike or whale activity present
    - Price remained stable or moved minimally
    
    Returns: True if heatmap exhaustion detected, False otherwise
    """
    ask_wall_disappeared = data.get("ask_wall_disappeared", False)
    volume_spike = data.get("volume_spike", False)
    whale_activity = data.get("whale_activity", False)
    price_stable = data.get("price_stable", True)  # Assume stable unless specified
    
    # Core detection logic: ask wall disappeared + accumulation pressure
    if ask_wall_disappeared and (volume_spike or whale_activity):
        # Additional validation: price should be stable (not already pumping)
        if price_stable:
            return True
    
    return False

def analyze_orderbook_pressure(symbol):
    """
    Analyze orderbook for signs of supply exhaustion
    This is a simplified implementation - in production would use real orderbook data
    """
    try:
        # Placeholder for orderbook analysis
        # In production, this would connect to exchange APIs for real orderbook data
        
        # Simulate orderbook conditions based on available market data
        orderbook_data = {
            "ask_wall_disappeared": False,
            "bid_ask_ratio": 1.0,
            "depth_imbalance": 0.0,
            "price_stable": True
        }
        
        # In real implementation, would analyze:
        # - Recent ask wall sizes and disappearances
        # - Bid/ask ratio changes
        # - Depth imbalance across price levels
        # - Price stability during accumulation
        
        return orderbook_data
        
    except Exception as e:
        print(f"❌ Error analyzing orderbook for {symbol}: {e}")
        return {
            "ask_wall_disappeared": False,
            "bid_ask_ratio": 1.0,
            "depth_imbalance": 0.0,
            "price_stable": True
        }

def get_heatmap_exhaustion_score(data):
    """
    Calculate scoring impact of heatmap exhaustion
    """
    if detect_heatmap_exhaustion(data):
        # Base score for heatmap exhaustion
        base_score = 5
        
        # Additional points for strong conditions
        if data.get("whale_activity") and data.get("volume_spike"):
            base_score += 2  # Both whale activity and volume spike
            
        if data.get("bid_ask_ratio", 1.0) > 1.5:
            base_score += 1  # Strong bid pressure
            
        return base_score
    
    return 0