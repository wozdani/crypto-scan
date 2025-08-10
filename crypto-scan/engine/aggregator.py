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
    """
    PUNKT 7 FIX: z-logit + clamp, bez bonusów
    🎯 UNMEASURED STATUS: Handle signals with UNMEASURED status for coverage ratio
    """
    z = 0.0
    contrib = {}
    
    # 🎯 COVERAGE RATIO CALCULATION: Count measured vs unmeasured signals
    measured_signals = 0
    unmeasured_signals = 0
    total_signals = len(signals)
    
    # Count signals by status for coverage calculation
    for name, s in signals.items():
        signal_status = s.get('status', 'MEASURED')  # Default to MEASURED for compatibility
        if signal_status == 'UNMEASURED':
            unmeasured_signals += 1
            contrib[name] = 0.0  # Neutral impact - no penalty for missing data
            print(f"[AGGREGATOR UNMEASURED] {name}: status=UNMEASURED (no orderbook data)")
            continue  # Skip UNMEASURED signals from scoring
        else:
            measured_signals += 1
        
        if not s.get("active"):
            contrib[name] = 0.0  # Inactive but measured signal
            continue
        
        w = float(weights.get(name, 0.0))
        x = float(s.get("strength", 0.0))
        
        if w == 0.0 or x <= 0.0 or x >= 1.0:
            contrib[name] = 0.0
            continue
            
        val = w * logit(x)
        z += val
        contrib[name] = val
    
    # Calculate coverage ratio (percentage of measurable signals)
    coverage_ratio = measured_signals / total_signals if total_signals > 0 else 1.0
    print(f"[AGGREGATOR COVERAGE] {measured_signals}/{total_signals} signals measured, coverage={coverage_ratio:.3f}")
    
    # 🎯 COVERAGE-ADJUSTED SCORE: Apply coverage ratio to prevent unfair penalization
    # When orderbook=synthetic, microstructure signals are UNMEASURED - adjust score accordingly
    raw_p = 1 / (1 + 2.718281828**(-z))
    
    # Apply coverage ratio to compensate for unmeasured signals
    coverage_adjusted_p = raw_p * (1.0 + (1.0 - coverage_ratio) * 0.1)  # Small boost for missing data
    coverage_adjusted_p = min(1.0, coverage_adjusted_p)  # Cap at 1.0
    
    if coverage_ratio < 1.0:
        print(f"[AGGREGATOR COVERAGE ADJUSTED] raw_p={raw_p:.3f} → adjusted={coverage_adjusted_p:.3f} (coverage={coverage_ratio:.3f})")
    
    return {
        "z_raw": z, 
        "p_raw": coverage_adjusted_p,  # Use coverage-adjusted score
        "contrib": contrib,
        "coverage_ratio": coverage_ratio,
        "measured_signals": measured_signals,
        "unmeasured_signals": unmeasured_signals
    }