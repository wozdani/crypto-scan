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

# Note: No longer importing STEALTH config for whitelist

EPS = 1e-6
ACTIVE_RULES_BONUS = 0.0   # OFF

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
    Aggregate signals using logit transformation without active rules bonus
    
    Args:
        signals: {name: {"active": bool, "strength": float}}
        weights: {name: float}
    
    Returns:
        Dict with z_raw, p_raw, and contrib
    
    Key features:
    - z_raw can be >1, p_raw ∈ (0,1)
    - No active rules bonus (ACTIVE_RULES_BONUS = 0.0)
    - Logit transformation for all active signals
    """
    z = 0.0
    contrib = {}
    
    for name, payload in signals.items():
        if not payload.get("active"):
            continue
            
        s = max(0.0, min(1.0, float(payload.get("strength", 0.0))))
        w = float(weights.get(name, 0.0))
        
        if w == 0.0:
            continue
            
        val = w * logit(s)  # logit() handles edge cases with EPS, no need for 0.0 < s < 1.0 restriction
        z += val
        contrib[name] = {"s": s, "w": w, "v": val}
    
    return {
        "z_raw": z, 
        "p_raw": 1/(1+pow(2.718281828, -z)), 
        "contrib": contrib
    }