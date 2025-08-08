"""
Explore to Training Data Converter
Converts explore mode JSON files to training-ready JSONL format
"""

import json
import pathlib
from typing import Dict, List, Tuple, Callable, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from engine.resolve_price import resolve_price_ref
from engine.agg_utils import logit, whale_dex_synergy
from engine.labeler import label_triple_barrier

ALLOWED_SIGNAL_KEYS = {
    "whale_ping",
    "dex_inflow", 
    "repeated_address_boost",
    "velocity_boost",
    "diamond_ai",
    "californium_ai"
}

def convert_one(
    explore_path: str,
    price_loader: Callable,
    weights: Dict[str, float],
    ohlcv_loader: Callable
) -> Dict:
    """
    Convert single explore mode record to training sample
    
    Args:
        explore_path: Path to explore mode JSON file
        price_loader: Function (symbol, ts) -> (ticker_price, candle_price, vol_24h_usd)
        weights: Dict of signal weights
        ohlcv_loader: Function (symbol, ts, horizon_min) -> list[(ts, px)] at 1-5m intervals
    
    Returns:
        Dict: Training-ready sample in specified format
    """
    # Load explore mode record
    rec = json.loads(pathlib.Path(explore_path).read_text())
    
    symbol = rec.get("symbol", "UNKNOWN")
    ts = rec.get("timestamp", datetime.utcnow().isoformat() + "Z")
    
    # 1) Get price_ref and volume
    ticker_price, candle_price, vol_24h_usd = price_loader(symbol, ts)
    
    try:
        price_ref = resolve_price_ref(ticker_price, candle_price)
    except ValueError:
        # Fallback to explore data if available
        price_ref = rec.get("price", 0.0)
        if price_ref <= 0:
            raise ValueError(f"No valid price for {symbol} at {ts}")
    
    # 2) Extract and structure signals
    sig = {}
    
    # On-chain core signals
    sW = float(rec.get("whale_ping_strength", 0.0))
    sD_usd = float(rec.get("dex_inflow_usd", 0.0))
    
    # Heuristic normalization for DEX inflow to strength [0,1]
    sD = min(1.0, sD_usd / 150_000.0) if sD_usd > 0 else 0.0
    
    sig["whale_ping"] = {
        "strength": sW,
        "weight": weights.get("whale_ping", 0.22),
        "contrib": 0.0,
        "repeats_7d": rec.get("whale_repeats_7d", 0),
        "sum_usd_60m": rec.get("whale_sum_usd_60m", 0.0)
    }
    
    sig["dex_inflow"] = {
        "strength": sD,
        "weight": weights.get("dex_inflow", 0.20),
        "contrib": 0.0,
        "inflow_usd_60m": sD_usd,
        "ratio_vs_base": rec.get("dex_ratio_vs_base", 0.0)
    }
    
    # Address-based boosts
    sig["repeated_address_boost"] = {
        "strength": float(rec.get("repeated_address_boost_strength", 0.0)),
        "weight": weights.get("repeated_address_boost", 0.25),
        "contrib": 0.0
    }
    
    sig["velocity_boost"] = {
        "strength": float(rec.get("velocity_boost_strength", 0.0)),
        "weight": weights.get("velocity_boost", 0.18),
        "contrib": 0.0
    }
    
    # AI detectors
    confidence_sources = rec.get("confidence_sources", {})
    dw = float(confidence_sources.get("diamondwhale_ai", 0.0))
    cf = float(confidence_sources.get("californiumwhale_ai", 0.0))
    
    sig["diamond_ai"] = {
        "score": dw,
        "weight": weights.get("diamond_ai", 0.30),
        "contrib": 0.0
    }
    
    sig["californium_ai"] = {
        "score": cf,
        "weight": weights.get("californium_ai", 0.25),
        "contrib": 0.0
    }
    
    # 3) Calculate contributions in logit space
    z_raw = 0.0
    
    for k, v in sig.items():
        if k in ["diamond_ai", "californium_ai"]:
            # AI detectors use score field
            s = max(0.0, min(1.0, v["score"]))
        else:
            # Other signals use strength field
            s = max(0.0, min(1.0, v["strength"]))
        
        w = float(v["weight"])
        
        if w > 0 and 0 < s < 1:
            contrib = w * logit(s)
            v["contrib"] = contrib
            z_raw += contrib
        else:
            v["contrib"] = 0.0
    
    # Calculate synergy
    phi = whale_dex_synergy(
        sig["whale_ping"]["strength"],
        sig["dex_inflow"]["strength"]
    )
    
    synergy_weight = weights.get("whale_dex_synergy", 0.35)
    synergy_contrib = synergy_weight * phi
    z_raw += synergy_contrib
    
    # 4) Orderbook metadata
    orderbook_data = rec.get("orderbook", {})
    ob_source = "real" if orderbook_data.get("source") == "real" else "synthetic"
    
    ob = {
        "source": ob_source,
        "depth_top10_usd": orderbook_data.get("depth_top10_usd", 0.0),
        "spread_pct": orderbook_data.get("spread_pct", 0.0),
        "ofi": orderbook_data.get("ofi", 0.0),
        "queue_imb": orderbook_data.get("queue_imb", 0.0),
        "quality_flag": 0 if ob_source == "synthetic" else 1
    }
    
    # 5) Label within 6h (triple-barrier)
    prices = ohlcv_loader(symbol, ts, horizon_min=360)
    
    if prices:
        lab = label_triple_barrier(prices, t0_idx=0, tp=0.04, sl=0.02, ttl_min=360)
    else:
        # No future price data available
        lab = {
            "y_hit_6h": 0,
            "max_return_6h": 0.0,
            "time_to_peak_min": 0,
            "triple_barrier": {"label": 0, "tp": 0.04, "sl": 0.02, "ttl_min": 360}
        }
    
    # 6) Missing flags
    graph_features = rec.get("graph_features", {})
    gf_missing = int(all(
        (v == 0 or v is None) for v in graph_features.values()
    )) if graph_features else 1
    
    # 7) Calculate final probabilities
    try:
        from math import exp
        p_raw = 1.0 / (1.0 + exp(-z_raw))
    except OverflowError:
        p_raw = 1.0 if z_raw > 0 else 0.0
    
    # 8) Determine sample weight (lower if no on-chain core)
    has_onchain = (
        sig["whale_ping"]["strength"] > 0.0 or 
        sig["dex_inflow"]["strength"] > 0.0
    )
    sample_weight = 1.0 if has_onchain else 0.5
    
    # Build output record
    out = {
        "sample_id": f"{symbol}_{ts[:19]}Z",
        "symbol": symbol,
        "ts": ts,
        "price_ref": price_ref,
        "vol_24h_usd": vol_24h_usd,
        "signals": sig,
        "synergy": {
            "whale_dex": phi,
            "weight": synergy_weight,
            "contrib": synergy_contrib
        },
        "orderbook": ob,
        "regime": {
            "atrp": rec.get("regime", {}).get("atrp", 0.0),
            "dOI": rec.get("regime", {}).get("dOI", 0.0),
            "funding": rec.get("regime", {}).get("funding", 0.0),
            "basis": rec.get("regime", {}).get("basis", 0.0),
            "btc_d": rec.get("regime", {}).get("btc_d", 0.0)
        },
        "aggregator": {
            "z_raw": z_raw,
            "p_raw": p_raw,
            "p_calib": rec.get("p_calib", 0.0)
        },
        "label": lab,
        "missing_flags": {
            "graph_features_missing": gf_missing,
            "orderbook_missing": 1 if ob["quality_flag"] == 0 else 0
        },
        "meta": {
            "engine_version": rec.get("engine_version", "stealth_v2.0"),
            "weights_version": rec.get("weights_version", "20250105"),
            "lookback_min": 90,
            "consensus": rec.get("consensus_decision", "UNKNOWN"),
            "explore_confidence": rec.get("explore_confidence", 0.0)
        },
        "sample_weight": sample_weight
    }
    
    return out

def batch_convert(
    explore_dir: str,
    output_path: str,
    price_loader: Callable,
    weights: Dict[str, float],
    ohlcv_loader: Callable,
    max_files: Optional[int] = None
) -> int:
    """
    Convert all explore mode files in directory to training JSONL
    
    Args:
        explore_dir: Directory containing *_explore.json files
        output_path: Path to output JSONL file
        price_loader: Function to load price data
        weights: Signal weights dictionary
        ohlcv_loader: Function to load OHLCV data
        max_files: Maximum number of files to process (for testing)
    
    Returns:
        int: Number of records successfully converted
    """
    explore_path = pathlib.Path(explore_dir)
    output_file = pathlib.Path(output_path)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    converted = 0
    errors = 0
    
    # Find all explore files
    explore_files = list(explore_path.glob("*_explore.json"))
    
    if max_files:
        explore_files = explore_files[:max_files]
    
    print(f"[CONVERTER] Found {len(explore_files)} explore files to process")
    
    with open(output_file, 'w') as f:
        for explore_file in explore_files:
            try:
                sample = convert_one(
                    str(explore_file),
                    price_loader,
                    weights,
                    ohlcv_loader
                )
                
                # Write as single line JSON
                f.write(json.dumps(sample) + '\n')
                converted += 1
                
                if converted % 100 == 0:
                    print(f"[CONVERTER] Processed {converted} records...")
                    
            except Exception as e:
                errors += 1
                print(f"[CONVERTER ERROR] Failed to convert {explore_file}: {e}")
                
                if errors > 10:
                    print(f"[CONVERTER] Too many errors, stopping...")
                    break
    
    print(f"[CONVERTER] Complete: {converted} converted, {errors} errors")
    print(f"[CONVERTER] Output saved to {output_file}")
    
    return converted

if __name__ == "__main__":
    # Example usage - requires actual price/OHLCV loaders
    def mock_price_loader(symbol, ts):
        """Mock price loader for testing"""
        return (0.1234, 0.1235, 1_000_000.0)  # ticker, candle, volume
    
    def mock_ohlcv_loader(symbol, ts, horizon_min):
        """Mock OHLCV loader for testing"""
        # Return fake price series for testing
        base_price = 0.1234
        prices = []
        base_ts = 1700000000  # Fake timestamp
        
        for i in range(horizon_min // 5):  # 5-minute intervals
            ts = base_ts + i * 300
            px = base_price * (1 + 0.001 * i)  # Gradual increase
            prices.append((ts, px))
        
        return prices
    
    # Default weights
    weights = {
        "whale_ping": 0.22,
        "dex_inflow": 0.20,
        "repeated_address_boost": 0.25,
        "velocity_boost": 0.18,
        "diamond_ai": 0.30,
        "californium_ai": 0.25,
        "whale_dex_synergy": 0.35
    }
    
    # Convert single file for testing
    import sys
    if len(sys.argv) > 1:
        explore_file = sys.argv[1]
        sample = convert_one(explore_file, mock_price_loader, weights, mock_ohlcv_loader)
        print(json.dumps(sample, indent=2))