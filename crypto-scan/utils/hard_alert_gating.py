"""
Hard Alert Gating System
Implements strict gating logic: Alert only when whale>=0.8 AND dex>=0.8 AND p>=τ
Removes conflicting score-based fallback logic
"""

from typing import Dict, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.stealth_config import STEALTH

def should_trigger_alert(
    whale_strength: float,
    dex_inflow_strength: float, 
    final_probability: float,
    consensus_decision: str = "UNKNOWN",
    symbol: str = "",
    active_signals: list = None
) -> Tuple[bool, str, Dict]:
    """
    Hard gating logic for alert triggering
    
    Args:
        whale_strength: Whale detection strength [0,1]
        dex_inflow_strength: DEX inflow strength [0,1] 
        final_probability: Final calibrated probability [0,1]
        consensus_decision: Multi-agent consensus (BUY/HOLD/AVOID/UNKNOWN)
        symbol: Token symbol for logging
        active_signals: List of active signal objects
        
    Returns:
        Tuple of (should_alert, reason, details)
    """
    if not STEALTH.get("HARD_GATING", True):
        # If hard gating disabled, use legacy logic
        return _legacy_gating_logic(final_probability, consensus_decision, symbol)
    
    # Hard gating requirements
    min_whale = STEALTH["MIN_WHALE_STRENGTH"]
    min_dex = STEALTH["MIN_DEX_STRENGTH"]
    min_prob = STEALTH["ALERT_TAU"]
    
    # Check consensus first (if enabled)
    if STEALTH.get("USE_CONSENSUS_GATE", True):
        if consensus_decision != "BUY":
            return False, f"consensus_block_{consensus_decision.lower()}", {
                "consensus_decision": consensus_decision,
                "whale_strength": whale_strength,
                "dex_strength": dex_inflow_strength,
                "final_probability": final_probability,
                "active_signals_count": len(active_signals) if active_signals else 0
            }
    
    # Check whale requirement
    if whale_strength < min_whale:
        return False, f"whale_insufficient_{whale_strength:.3f}<{min_whale}", {
            "whale_strength": whale_strength,
            "required_whale": min_whale,
            "dex_strength": dex_inflow_strength,
            "final_probability": final_probability
        }
    
    # Check DEX inflow requirement  
    if dex_inflow_strength < min_dex:
        return False, f"dex_insufficient_{dex_inflow_strength:.3f}<{min_dex}", {
            "dex_strength": dex_inflow_strength,
            "required_dex": min_dex,
            "whale_strength": whale_strength,
            "final_probability": final_probability
        }
    
    # Check probability threshold
    if final_probability < min_prob:
        return False, f"probability_insufficient_{final_probability:.3f}<{min_prob}", {
            "final_probability": final_probability,
            "required_prob": min_prob,
            "whale_strength": whale_strength,
            "dex_strength": dex_inflow_strength
        }
    
    # All requirements met
    print(f"[HARD GATING] {symbol} PASSED: whale={whale_strength:.3f}≥{min_whale}, dex={dex_inflow_strength:.3f}≥{min_dex}, p={final_probability:.3f}≥{min_prob}")
    
    return True, "hard_gating_passed", {
        "whale_strength": whale_strength,
        "dex_strength": dex_inflow_strength,
        "final_probability": final_probability,
        "consensus_decision": consensus_decision,
        "active_signals_count": len(active_signals) if active_signals else 0,
        "gating_mode": "hard"
    }

def _legacy_gating_logic(final_probability: float, consensus_decision: str, symbol: str) -> Tuple[bool, str, Dict]:
    """Legacy gating logic (when hard gating disabled)"""
    min_prob = STEALTH["ALERT_TAU"]
    
    if consensus_decision == "BUY" and final_probability >= min_prob:
        return True, "legacy_consensus_buy", {
            "final_probability": final_probability,
            "consensus_decision": consensus_decision,
            "gating_mode": "legacy"
        }
    elif final_probability >= min_prob:
        return True, "legacy_probability", {
            "final_probability": final_probability,
            "consensus_decision": consensus_decision,
            "gating_mode": "legacy"
        }
    else:
        return False, f"legacy_insufficient_{final_probability:.3f}<{min_prob}", {
            "final_probability": final_probability,
            "required_prob": min_prob,
            "consensus_decision": consensus_decision,
            "gating_mode": "legacy"
        }

def check_conflicting_fallbacks(alert_system_name: str, score: float) -> bool:
    """
    Check for and block conflicting score-based fallback alerts
    
    Args:
        alert_system_name: Name of alert system trying to trigger
        score: Score being used for fallback
        
    Returns:
        bool: True if fallback should be blocked
    """
    if not STEALTH.get("REMOVE_SCORE_FALLBACK", True):
        return False  # Allow fallbacks if not configured to remove them
    
    # Block common problematic fallback patterns
    problematic_patterns = [
        "score≥0.7",
        "fallback_alert",
        "no_consensus_but_score",
        "score_override"
    ]
    
    for pattern in problematic_patterns:
        if pattern in alert_system_name.lower():
            print(f"[FALLBACK BLOCK] Blocked conflicting fallback: {alert_system_name} with score {score:.3f}")
            return True
    
    return False

def ensure_active_signals_format(signals_data) -> list:
    """
    Ensure active_signals is always returned as a list of objects
    So Explore mode doesn't have to guess
    
    Args:
        signals_data: Raw signals data in various formats
        
    Returns:
        list: Properly formatted active signals list
    """
    if signals_data is None:
        return []
    
    if isinstance(signals_data, list):
        # Already in correct format
        return signals_data
    
    if isinstance(signals_data, dict):
        # Convert dict format to list of objects
        active_signals = []
        for name, payload in signals_data.items():
            if isinstance(payload, dict):
                is_active = payload.get("active", False)
                strength = payload.get("strength", 0.0)
                
                if is_active and strength > 0:
                    active_signals.append({
                        "name": name,
                        "strength": strength,
                        "active": True,
                        "payload": payload
                    })
        
        return active_signals
    
    # Unknown format, return empty
    print(f"[ACTIVE SIGNALS] Unknown signals format: {type(signals_data)}")
    return []

def get_gating_summary(symbol: str, result: Tuple[bool, str, Dict]) -> str:
    """
    Generate human-readable gating summary for debugging
    
    Args:
        symbol: Token symbol
        result: Result from should_trigger_alert()
        
    Returns:
        str: Human-readable summary
    """
    should_alert, reason, details = result
    
    if should_alert:
        return f"✅ {symbol} ALERT APPROVED: {reason}"
    else:
        return f"❌ {symbol} ALERT BLOCKED: {reason} - {details}"