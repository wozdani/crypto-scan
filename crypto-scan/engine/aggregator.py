"""
Signal Aggregator Module
Implements whitelist-based weight aggregation with logit transformation
"""

from math import log
from typing import Dict
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.stealth_config import STEALTH

EPS = 1e-6

def logit(p: float) -> float:
    """
    Logit transformation for probability
    
    Args:
        p: Probability value [0,1]
    
    Returns:
        float: Logit transformed value
    """
    p = max(EPS, min(1.0 - EPS, p))
    return log(p/(1.0-p))

def aggregate(signals: Dict, weights: Dict) -> Dict:
    """
    Aggregate signals with whitelisted weights
    
    Args:
        signals: {name: {"active": bool, "strength": float}}
        weights: {name: float}
    
    Returns:
        Dict with base_score, final_score, and contributions
    
    Key features:
    - Only whitelisted signals are processed
    - Unknown signals have weight 0.0
    - Strength clamped to [0,1]
    - No "active rules bonus"
    """
    z = 0.0
    contrib = {}
    active_count = 0
    
    for name, payload in signals.items():
        # Skip if signal not in whitelist
        if name not in STEALTH["ALLOWED_SIGNALS"]:
            print(f"[AGGREGATOR] Skipping unknown signal: {name}")
            continue
        
        # Extract and clamp strength
        if isinstance(payload, dict):
            is_active = payload.get("active", False)
            strength = payload.get("strength", 0.0)
        else:
            # Handle legacy format where payload might be just a float
            is_active = True if payload else False
            strength = float(payload) if payload else 0.0
        
        strength = max(0.0, min(1.0, strength))
        
        # Get weight (default to DEFAULT_WEIGHT if not specified)
        w = float(weights.get(name, STEALTH["DEFAULT_WEIGHT"]))
        
        # Skip if weight is 0
        if w == 0.0:
            continue
        
        # Only contribute if signal is active
        if is_active and strength > 0:
            contribution = w * logit(strength)
            z += contribution
            active_count += 1
            contrib[name] = {
                "strength": strength,
                "weight": w,
                "contribution": contribution,
                "logit_strength": logit(strength)
            }
            print(f"[AGGREGATOR] {name}: strength={strength:.3f}, weight={w:.3f}, contribution={contribution:.3f}")
    
    # Calculate final scores
    base_score = z
    
    # Apply sigmoid to get back to probability space for final score
    try:
        from math import exp
        final_score = 1.0 / (1.0 + exp(-base_score))
    except OverflowError:
        # Handle extreme values
        final_score = 1.0 if base_score > 0 else 0.0
    
    print(f"[AGGREGATOR] Total: {active_count} active signals, base_score={base_score:.3f}, final_score={final_score:.3f}")
    
    return {
        "base_score": base_score,
        "final_score": final_score,
        "contrib": contrib,
        "active_count": active_count
    }