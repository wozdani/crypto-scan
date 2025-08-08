"""
Unified Decision Module
Single decision pathway for all alert types
"""

from typing import Dict, Optional
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.stealth_config import STEALTH

def make_decision(stealth_p: float, consensus: Optional[str] = None) -> Dict:
    """
    Unified decision maker for alerts
    
    Args:
        stealth_p: Stealth probability from aggregator (0-1)
        consensus: Optional consensus decision (BUY/HOLD/AVOID)
    
    Returns:
        Dict with decision, send_alert flag, and reasoning
    """
    result = {
        "decision": "WATCHLIST",
        "send_alert": False,
        "reason": "",
        "stealth_p": stealth_p,
        "consensus": consensus
    }
    
    # Check stealth threshold
    above_threshold = stealth_p >= STEALTH["ALERT_TAU"]
    
    if above_threshold:
        if STEALTH["USE_CONSENSUS_GATE"] and consensus:
            # Use consensus gate - only BUY triggers alert
            if consensus == "BUY":
                result["decision"] = "ALERT"
                result["send_alert"] = True
                result["reason"] = f"Stealth p={stealth_p:.3f} > τ={STEALTH['ALERT_TAU']:.3f} AND Consensus=BUY"
                print(f"[DECISION] ALERT: {result['reason']}")
            else:
                result["decision"] = "WATCHLIST"
                result["send_alert"] = False
                result["reason"] = f"Stealth p={stealth_p:.3f} > τ but Consensus={consensus} (not BUY)"
                print(f"[DECISION] WATCHLIST: {result['reason']}")
        else:
            # No consensus gate - stealth alone decides
            result["decision"] = "ALERT"
            result["send_alert"] = True
            result["reason"] = f"Stealth p={stealth_p:.3f} > τ={STEALTH['ALERT_TAU']:.3f}"
            print(f"[DECISION] ALERT: {result['reason']}")
    else:
        # Below threshold - always watchlist
        result["decision"] = "WATCHLIST"
        result["send_alert"] = False
        result["reason"] = f"Stealth p={stealth_p:.3f} < τ={STEALTH['ALERT_TAU']:.3f}"
        print(f"[DECISION] WATCHLIST: {result['reason']}")
    
    return result