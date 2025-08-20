"""
Context packer - reduces and normalizes token inputs for efficient batch processing
"""
from typing import Dict, Any, List

def pack_token_context(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pack token context to reduce API payload size and processing time
    """
    # 1) Detectors: top-4 by strength (ranking without thresholds) + cut the rest
    detector_breakdown = raw.get("detector_breakdown", {})
    top_detectors = dict(sorted(detector_breakdown.items(), key=lambda kv: kv[1], reverse=True)[:4])
    
    # 2) History: max 6 most recent events + deduplicate event types
    history = raw.get("history", {})
    events = history.get("events", [])
    if isinstance(events, list):
        # Sort by timestamp (most recent first) and limit to 6
        events = sorted(events, key=lambda e: e.get("timestamp", e.get("t", 0)), reverse=True)[:6]
        
        # Deduplicate by event type
        seen_types = set()
        unique_events = []
        for event in events:
            event_type = event.get("type", event.get("event_type", "unknown"))
            if event_type not in seen_types:
                seen_types.add(event_type)
                unique_events.append(event)
        events = unique_events[:6]  # Ensure max 6 after deduplication
    
    # 3) Meta/trust/perf: only essential fields, round values to 2 decimal places
    def round_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """Round numeric values in dictionary to 2 decimal places"""
        result = {}
        for k, v in d.items():
            if isinstance(v, (int, float)):
                result[k] = round(float(v), 2)
            elif isinstance(v, dict):
                result[k] = round_dict(v)
            else:
                result[k] = v
        return result
    
    # Extract essential meta fields
    meta_raw = raw.get("meta", {})
    essential_meta_fields = [
        "price", "volume_24h", "price_change_24h", "spread", "market_cap",
        "liquidity_tier", "is_perp", "exchange"
    ]
    meta = round_dict({k: meta_raw[k] for k in essential_meta_fields if k in meta_raw})
    
    # Extract essential trust fields
    trust_raw = raw.get("trust", {})
    essential_trust_fields = [
        "whale_addresses", "trust_score", "dex_inflow", "address_reputation"
    ]
    trust = {}
    for field in essential_trust_fields:
        if field in trust_raw:
            if field == "whale_addresses" and isinstance(trust_raw[field], list):
                # Limit whale addresses to top 5
                trust[field] = trust_raw[field][:5]
            else:
                trust[field] = trust_raw[field]
    trust = round_dict(trust)
    
    # Extract and compress performance data
    perf_raw = raw.get("perf", {})
    perf = {}
    for k, v in perf_raw.items():
        if isinstance(v, dict):
            perf[k] = round_dict(v)
        elif isinstance(v, (int, float)):
            perf[k] = round(float(v), 2)
        else:
            perf[k] = v
    
    return {
        "token_id": raw["token_id"],
        "detector_breakdown": top_detectors,
        "meta": meta,
        "trust": trust,
        "history": {"events": events},
        "perf": perf
    }

def calculate_context_size(packed: Dict[str, Any]) -> int:
    """Estimate context size for load balancing"""
    import json
    return len(json.dumps(packed, ensure_ascii=False))

def compress_for_emergency(packed: Dict[str, Any]) -> Dict[str, Any]:
    """Emergency compression for timeout situations"""
    # Ultra-minimal context for emergency single-token processing
    detector_breakdown = packed.get("detector_breakdown", {})
    top_2_detectors = dict(sorted(detector_breakdown.items(), key=lambda kv: kv[1], reverse=True)[:2])
    
    meta = packed.get("meta", {})
    essential_meta = {k: meta[k] for k in ["price", "volume_24h"] if k in meta}
    
    return {
        "token_id": packed["token_id"],
        "detector_breakdown": top_2_detectors,
        "meta": essential_meta,
        "trust": {"trust_score": packed.get("trust", {}).get("trust_score", 0.0)},
        "history": {"events": []},
        "perf": {}
    }