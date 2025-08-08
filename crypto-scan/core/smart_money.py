"""
Smart Money Detection Module
Implements strict criteria for identifying smart money addresses
"""

from typing import Set, Optional
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.stealth_config import STEALTH
from utils.dynamic_whale_thresholds import calculate_dynamic_whale_threshold, validate_whale_strength

def is_exchange(addr: str, known_exchanges: Set[str]) -> bool:
    """Check if address is a known exchange"""
    return addr.lower() in known_exchanges

def is_smart_money(addr: str, trust: float, preds: int, usd: float, 
                   repeats_7d: int, known_exchanges: Set[str]) -> bool:
    """
    Determine if address qualifies as smart money
    
    Args:
        addr: Address to check
        trust: Trust score (0-1)
        preds: Number of predictions
        usd: USD value of transaction
        repeats_7d: Number of repeat transactions in 7 days
        known_exchanges: Set of known exchange addresses
    
    Returns:
        bool: True if address is smart money
    """
    # Exclude exchange addresses if configured
    if STEALTH["SM_EXCLUDE_EXCHANGES"] and is_exchange(addr, known_exchanges):
        return False
    
    # Criteria 1: High trust and sufficient predictions
    if trust >= STEALTH["SM_TRUST_MIN"] and preds >= STEALTH["SM_PRED_MIN"]:
        return True
    
    # Criteria 2: Repeat whale with sufficient value
    if repeats_7d >= STEALTH["SM_REPEAT_MIN_7D"] and usd >= STEALTH["SM_WHALE_MIN_USD"]:
        return True
    
    return False

def apply_smart_money_boost(base_strength: float, trust: float, preds: int, 
                           usd: float, repeats_7d: int, addr: str, 
                           known_exchanges: Set[str]) -> float:
    """
    Apply boost only for real smart money. Safe clamp to [0,1].
    
    Args:
        base_strength: Base signal strength
        trust: Address trust score
        preds: Number of predictions
        usd: Transaction USD value
        repeats_7d: Repeat count in 7 days
        addr: Address string
        known_exchanges: Set of known exchange addresses
    
    Returns:
        float: Boosted strength, clamped to [0,1]
    """
    # Ensure base strength is in valid range
    s = max(0.0, min(1.0, base_strength))
    
    # Apply boost only if address qualifies as smart money
    if is_smart_money(addr, trust, preds, usd, repeats_7d, known_exchanges):
        s = min(1.0, s + STEALTH["SM_MAX_BOOST"])
        print(f"[SMART MONEY] Applied boost +{STEALTH['SM_MAX_BOOST']} to {addr[:10]}... (trust={trust:.2f}, preds={preds})")
    else:
        print(f"[SMART MONEY] No boost for {addr[:10]}... (trust={trust:.2f}, preds={preds}, not qualified)")
    
    return s