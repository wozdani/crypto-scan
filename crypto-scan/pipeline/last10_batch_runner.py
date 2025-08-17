"""
Batch runner for last 10 tokens processing
"""
import json
import hashlib
from typing import List, Dict, Any
from state.last10_store import get_last10_store
from features.compact_features import compact_for_detector, DET_SHORT

def build_items_from_last10(min_trust: float = 0.0, min_liq_usd: float = 0.0) -> List[Dict[str, Any]]:
    """
    Build items list from last 10 tokens only
    
    Args:
        min_trust: Minimum trust score filter
        min_liq_usd: Minimum liquidity USD filter
    
    Returns:
        List of items for LLM processing
    """
    store = get_last10_store()
    last10_data = store.get_last10()
    
    items = []
    
    for symbol, token_data in last10_data.items():
        ts = token_data["ts"]
        detectors = token_data["detectors"]
        feats = token_data["feats"]
        
        # Create item for each active detector
        for detector in detectors:
            if detector not in DET_SHORT:
                continue  # Skip unknown detectors
            
            detector_feats = feats.get(detector, {})
            compact_feats = compact_for_detector(detector, detector_feats)
            
            # Apply filters
            if compact_feats.get("tr", 0.0) < min_trust:
                continue
            if compact_feats.get("lq", 0.0) < min_liq_usd:
                continue
            
            # Create item
            item = {
                "s": symbol,
                "det": DET_SHORT[detector],
                "ts": ts,
                "f": compact_feats
            }
            
            items.append(item)
    
    return items

def _cache_key(item: Dict[str, Any]) -> str:
    """
    Generate cache key for item based on (symbol, detector, features_hash)
    
    Args:
        item: Item dictionary
    
    Returns:
        Cache key string
    """
    symbol = item["s"]
    detector = item["det"]
    features_str = json.dumps(item["f"], sort_keys=True)
    features_hash = hashlib.md5(features_str.encode()).hexdigest()[:8]
    
    return f"{symbol}_{detector}_{features_hash}"