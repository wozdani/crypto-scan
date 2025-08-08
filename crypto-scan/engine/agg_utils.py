"""
Aggregation Utilities Module
Logit transformation and synergy calculations
"""

from math import log

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
    return log(p / (1.0 - p))

def whale_dex_synergy(s_whale: float, s_dex: float) -> float:
    """
    Calculate synergy between whale and DEX signals
    
    Args:
        s_whale: Whale signal strength [0,1]
        s_dex: DEX signal strength [0,1]
    
    Returns:
        float: Synergy value (0 if either signal is 0)
    """
    if s_whale <= 0 or s_dex <= 0:
        return 0.0
    
    return logit(s_whale) * logit(s_dex)