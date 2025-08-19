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
    
    print(f"[LAST10 BATCH] Building items from {len(last10_data)} tokens in store")
    
    items = []
    token_count = 0
    
    for symbol, token_data in last10_data.items():
        ts = token_data["ts"]
        detectors = token_data["detectors"]
        feats = token_data["feats"]
        
        token_count += 1
        print(f"[LAST10 BATCH] Token {token_count}: {symbol} with detectors: {detectors}")
        
        # Create item for each active detector
        for detector in detectors:
            if detector not in DET_SHORT:
                print(f"[LAST10 BATCH] Skipping unknown detector: {detector} for {symbol}")
                continue  # Skip unknown detectors
            
            detector_feats = feats.get(detector, {})
            compact_feats = compact_for_detector(detector, detector_feats)
            
            # Skip filters when min thresholds are 0 (Multi-Agent should analyze all data)
            if min_trust > 0.0 and compact_feats.get("tr", 0.0) < min_trust:
                print(f"[LAST10 BATCH] Filtered {symbol}/{detector}: trust={compact_feats.get('tr', 0.0)} < {min_trust}")
                continue
            if min_liq_usd > 0.0 and compact_feats.get("lq", 0.0) < min_liq_usd:
                print(f"[LAST10 BATCH] Filtered {symbol}/{detector}: liquidity={compact_feats.get('lq', 0.0)} < {min_liq_usd}")
                continue
            
            # Create item
            item = {
                "s": symbol,
                "det": DET_SHORT[detector],
                "ts": ts,
                "f": compact_feats
            }
            
            items.append(item)
            print(f"[LAST10 BATCH] Added item: {symbol}/{DET_SHORT[detector]} (detector={detector})")
    
    print(f"[LAST10 BATCH] Total items built: {len(items)} from {token_count} tokens")
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