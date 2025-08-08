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

def logit(p):
    """PUNKT 7 FIX: z-logit transformation with clamp"""
    p = max(EPS, min(1-EPS, p))
    return log(p/(1-p))

def aggregate(signals: dict, weights: dict):
    """PUNKT 7 FIX: z-logit + clamp, bez bonusów"""
    z = 0.0
    contrib = {}
    
    for name, s in signals.items():
        if not s.get("active"):
            continue
        
        w = float(weights.get(name, 0.0))
        x = float(s.get("strength", 0.0))
        
        if w == 0.0 or x <= 0.0 or x >= 1.0:
            contrib[name] = 0.0
            continue
            
        val = w * logit(x)
        z += val
        contrib[name] = val
    
    p = 1 / (1 + 2.718281828**(-z))
    return {"z_raw": z, "p_raw": p, "contrib": contrib}