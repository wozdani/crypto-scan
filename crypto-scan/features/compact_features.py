"""
Feature compaction with short keys for efficient LLM processing
"""
from typing import Dict, Any

# Detector name shortcuts
DET_SHORT = {
    "whale_ping": "WP",
    "dex_inflow": "DX", 
    "orderbook_anomaly": "OBA",
    "whaleclip": "WCL",
    "diamond": "DIA",
    "californium": "CAL",
    "mastermind_tracing": "MMT"
}

def compact_for_detector(det: str, feats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact features for detector with short keys
    Always includes tr=trust, lq=liquidity_1pct_usd
    
    Args:
        det: Detector name
        feats: Full feature dictionary
    
    Returns:
        Compacted feature dict with short keys
    """
    # Base fields (always included)
    compact = {
        "tr": float(feats.get("trust_score", 0.0)),
        "lq": float(feats.get("liquidity_1pct_usd", 0.0))
    }
    
    # Detector-specific fields (3-5 per detector)
    if det == "whale_ping":
        compact.update({
            "wp": float(feats.get("whale_ping_strength", 0.0)),
            "rw": float(feats.get("repeated_whale_boost", 0.0)),
            "sd": float(feats.get("smart_money_detection", 0.0))
        })
    
    elif det == "dex_inflow":
        compact.update({
            "dx5": float(feats.get("dex_inflow_5m", 0.0)),
            "dx15": float(feats.get("dex_inflow_15m", 0.0)),
            "rx": float(feats.get("reliability_score", 0.0))
        })
    
    elif det == "orderbook_anomaly":
        compact.update({
            "bw": float(feats.get("bid_wall_strength", 0.0)),
            "oba": float(feats.get("orderbook_anomaly_score", 0.0)),
            "imb": float(feats.get("imbalance_ratio", 0.0)),
            "spr": float(feats.get("spread_tightening", 0.0))
        })
    
    elif det == "whaleclip":
        compact.update({
            "vis": float(feats.get("vision_score", 0.0)),
            "pat": float(feats.get("pattern_match", 0.0)),
            "conf": float(feats.get("confidence_score", 0.0))
        })
    
    elif det == "diamond":
        compact.update({
            "dia": float(feats.get("diamond_score", 0.0)),
            "tmp": float(feats.get("temporal_score", 0.0)),
            "grph": float(feats.get("graph_score", 0.0))
        })
    
    elif det == "californium":
        compact.update({
            "cal": float(feats.get("californium_score", 0.0)),
            "ai": float(feats.get("ai_confidence", 0.0)),
            "sig": float(feats.get("signal_strength", 0.0))
        })
    
    elif det == "mastermind_tracing":
        compact.update({
            "mmt": float(feats.get("mastermind_score", 0.0)),
            "trc": float(feats.get("trace_confidence", 0.0)),
            "net": float(feats.get("network_score", 0.0))
        })
    
    # Convert all values to numbers or booleans only
    result = {}
    for k, v in compact.items():
        if isinstance(v, (int, float)):
            result[k] = round(float(v), 4)  # Round to 4 decimal places
        elif isinstance(v, bool):
            result[k] = v
        else:
            # Convert strings/other to 0
            result[k] = 0.0
    
    return result