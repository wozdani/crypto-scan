"""
Signal Standardization Module
Ensures active_signals are always returned in consistent format for Explore mode
"""

from typing import Dict, List, Any, Optional
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.stealth_config import STEALTH
from utils.dynamic_whale_thresholds import get_whale_context_info
from utils.bsc_dex_enhanced import detect_bsc_dex_inflow

def standardize_stealth_result(
    symbol: str,
    stealth_score: float,
    signals: Dict = None,
    volume_24h_usd: float = 0.0,
    consensus_decision: str = "UNKNOWN",
    whale_transactions_usd: List[float] = None,
    token_contract: str = None
) -> Dict:
    """
    Standardize stealth engine result with EDUUSDT hotfix applied
    Ensures active_signals are always returned, fixing Explore mode guessing
    
    Args:
        symbol: Token symbol
        stealth_score: Final stealth score
        signals: Raw signals dictionary
        volume_24h_usd: 24h volume for dynamic thresholds
        consensus_decision: Multi-agent consensus
        whale_transactions_usd: Actual whale transaction values
        token_contract: Contract address for DEX detection
        
    Returns:
        Dict with standardized stealth result
    """
    
    # Ensure active_signals is always a list of objects
    active_signals = _extract_active_signals(signals or {})
    
    # Enhanced whale analysis with dynamic thresholds
    whale_context = None
    if whale_transactions_usd and volume_24h_usd > 0:
        whale_context = get_whale_context_info(symbol, whale_transactions_usd, volume_24h_usd)
    
    # Enhanced DEX inflow for BSC tokens
    dex_inflow_result = None
    if token_contract and STEALTH.get("DEX_INFLOW_BSC_ENABLED", True):
        # Try enhanced BSC detection
        try:
            dex_inflow_result = detect_bsc_dex_inflow(token_contract)
        except Exception as e:
            print(f"[DEX INFLOW BSC] Error for {symbol}: {e}")
            dex_inflow_result = {"dex_inflow_usd": 0.0, "status": "ERROR"}
    
    # Extract key strengths for hard gating
    whale_strength = _extract_whale_strength(signals, whale_context)
    dex_strength = _extract_dex_strength(signals, dex_inflow_result)
    
    # Build standardized result
    result = {
        "symbol": symbol,
        "stealth_score": stealth_score,
        "consensus_decision": consensus_decision,
        "active_signals": active_signals,  # Always list of objects
        "active_signals_count": len(active_signals),
        "whale_strength": whale_strength,
        "dex_inflow_strength": dex_strength,
        "volume_24h_usd": volume_24h_usd,
        "hotfix_applied": True,
        "standardized": True
    }
    
    # Add enhanced analysis if available
    if whale_context:
        result["whale_context"] = whale_context
    
    if dex_inflow_result:
        result["dex_inflow_analysis"] = dex_inflow_result
    
    # Add signal details for transparency
    if signals:
        result["raw_signals"] = signals
    
    print(f"[SIGNAL STANDARDIZER] {symbol}: {len(active_signals)} active signals, whale={whale_strength:.3f}, dex={dex_strength:.3f}")
    
    return result

def _extract_active_signals(signals: Dict) -> List[Dict]:
    """Extract active signals as list of objects"""
    active_signals = []
    
    for name, payload in signals.items():
        try:
            if isinstance(payload, dict):
                is_active = payload.get("active", False)
                strength = payload.get("strength", 0.0)
                
                if is_active and strength > 0:
                    active_signals.append({
                        "name": name,
                        "strength": float(strength),
                        "active": True,
                        "category": _get_signal_category(name),
                        "payload": payload
                    })
            elif payload and payload > 0:  # Handle legacy numeric format
                active_signals.append({
                    "name": name,
                    "strength": float(payload),
                    "active": True,
                    "category": _get_signal_category(name),
                    "payload": {"strength": payload, "active": True}
                })
        except Exception as e:
            print(f"[SIGNAL STANDARDIZER] Error processing signal {name}: {e}")
    
    return active_signals

def _extract_whale_strength(signals: Dict, whale_context: Dict = None) -> float:
    """Extract whale strength for hard gating"""
    if whale_context:
        return whale_context.get("whale_strength", 0.0)
    
    # Fallback to signals
    whale_signal = signals.get("whale_ping", {}) if signals else {}
    if isinstance(whale_signal, dict):
        return whale_signal.get("strength", 0.0)
    
    return 0.0

def _extract_dex_strength(signals: Dict, dex_result: Dict = None) -> float:
    """Extract DEX inflow strength for hard gating"""
    if dex_result:
        dex_usd = dex_result.get("dex_inflow_usd", 0.0)
        if dex_usd is None:  # UNKNOWN case
            return 0.5  # Neutral for UNKNOWN
        elif dex_usd > 0:
            # Normalize DEX inflow to strength [0,1]
            # Using same logic as explore_to_train.py
            return min(1.0, dex_usd / 150_000.0)
    
    # Fallback to signals
    dex_signal = signals.get("dex_inflow", {}) if signals else {}
    if isinstance(dex_signal, dict):
        return dex_signal.get("strength", 0.0)
    
    return 0.0

def _get_signal_category(signal_name: str) -> str:
    """Get signal category for classification"""
    categories = {
        "whale_ping": "whale",
        "dex_inflow": "dex", 
        "orderbook_imbalance": "orderbook",
        "large_bid_walls": "orderbook",
        "volume_spike": "volume",
        "spoofing_detected": "manipulation",
        "diamond_ai": "ai_detector",
        "californium_ai": "ai_detector"
    }
    
    return categories.get(signal_name, "unknown")