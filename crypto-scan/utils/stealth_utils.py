"""
Stealth Engine Utilities
Utility functions for stealth signal detection and analysis
"""

from typing import Dict, List, Any, Optional

def is_cold_start(token_data: Dict[str, Any]) -> bool:
    """
    Returns True if token has no historical trust, feedback or contract presence.
    
    Args:
        token_data: Dictionary containing token analysis data
        
    Returns:
        True if token meets cold start criteria
    """
    trust = token_data.get("trust_addresses", 0)
    feedbacks = token_data.get("feedback_history", [])
    contract_found = token_data.get("contract_found", True)
    
    # Additional cold start indicators
    whale_memory = token_data.get("whale_memory_entries", 0)
    historical_alerts = token_data.get("historical_alerts", [])
    
    return (trust == 0 and 
            len(feedbacks) < 1 and 
            not contract_found and
            whale_memory < 3 and
            len(historical_alerts) < 2)

def should_explore_mode_trigger(token_data: Dict[str, Any], final_score: float) -> tuple[bool, str]:
    """
    Check if token should trigger explore mode experimental alert.
    
    Args:
        token_data: Token analysis data
        final_score: Final stealth score
        
    Returns:
        Tuple of (should_trigger, reason)
    """
    explore_score_threshold = 2.8
    
    # Must be cold start and high score
    if not is_cold_start(token_data) or final_score < explore_score_threshold:
        return False, "not_cold_start_or_low_score"
    
    # Check for high-quality core signals
    active_signals = set(token_data.get("active_signals", []))
    core_signals = {'whale_ping', 'dex_inflow', 'orderbook_anomaly', 'spoofing_layers'}
    
    # Require at least 2 core signals
    core_signal_count = len(active_signals.intersection(core_signals))
    if core_signal_count < 2:
        return False, f"insufficient_core_signals_{core_signal_count}"
    
    # Additional quality checks
    whale_ping_strength = token_data.get("whale_ping_strength", 0.0)
    dex_inflow_value = token_data.get("dex_inflow_usd", 0.0)
    
    # High-quality whale ping (> 0.7 strength)
    if whale_ping_strength > 0.7:
        return True, f"high_quality_whale_ping_{whale_ping_strength:.3f}"
    
    # Significant DEX inflow (> $10k)
    if dex_inflow_value > 10000:
        return True, f"significant_dex_inflow_{dex_inflow_value:.0f}"
    
    # Strong multi-signal combination
    if core_signal_count >= 3:
        return True, f"strong_multi_signal_{core_signal_count}_signals"
    
    return False, "quality_threshold_not_met"

def calculate_explore_mode_confidence(token_data: Dict[str, Any], final_score: float) -> float:
    """
    Calculate synthetic confidence boost for explore mode.
    
    Args:
        token_data: Token analysis data
        final_score: Final stealth score
        
    Returns:
        Synthetic confidence score
    """
    base_confidence = 1.5  # Base explore mode confidence
    
    # Boost based on signal quality
    whale_ping_strength = token_data.get("whale_ping_strength", 0.0)
    dex_inflow_value = token_data.get("dex_inflow_usd", 0.0)
    active_signals = set(token_data.get("active_signals", []))
    
    # Whale ping quality boost
    if whale_ping_strength > 0.8:
        base_confidence += 0.3
    elif whale_ping_strength > 0.7:
        base_confidence += 0.2
    
    # DEX inflow boost
    if dex_inflow_value > 20000:
        base_confidence += 0.4
    elif dex_inflow_value > 10000:
        base_confidence += 0.2
    
    # Multi-signal boost
    signal_count = len(active_signals)
    if signal_count >= 4:
        base_confidence += 0.3
    elif signal_count >= 3:
        base_confidence += 0.2
    
    # Score-based boost
    if final_score > 3.5:
        base_confidence += 0.2
    elif final_score > 3.0:
        base_confidence += 0.1
    
    return round(base_confidence, 3)

def format_explore_mode_reason(token_data: Dict[str, Any], trigger_reason: str) -> str:
    """
    Format explore mode reasoning for transparency.
    
    Args:
        token_data: Token analysis data
        trigger_reason: Reason for explore mode trigger
        
    Returns:
        Formatted reason string
    """
    active_signals = token_data.get("active_signals", [])
    core_signals = {'whale_ping', 'dex_inflow', 'orderbook_anomaly', 'spoofing_layers'}
    active_core = [sig for sig in active_signals if sig in core_signals]
    
    reason_parts = [
        f"EXPLORE MODE: {trigger_reason}",
        f"Core signals: {', '.join(active_core)}",
        f"Trust: {token_data.get('trust_addresses', 0)}",
        f"Contract: {'not found' if not token_data.get('contract_found', True) else 'found'}",
        f"Feedback history: {len(token_data.get('feedback_history', []))}"
    ]
    
    return " | ".join(reason_parts)

def log_explore_mode_feedback(symbol: str, final_score: float, confidence: float, 
                            decision: str, explore_mode: bool, token_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log explore mode feedback for learning.
    
    Args:
        symbol: Token symbol
        final_score: Final stealth score
        confidence: Confidence level
        decision: Decision made
        explore_mode: Whether in explore mode
        token_data: Token analysis data
        
    Returns:
        Log entry dictionary
    """
    import datetime
    
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "token": symbol,
        "score": final_score,
        "confidence": confidence,
        "decision": decision,
        "mode": "explore" if explore_mode else "normal",
        "cold_start": is_cold_start(token_data),
        "active_signals": list(token_data.get("active_signals", [])),
        "trust_addresses": token_data.get("trust_addresses", 0),
        "contract_found": token_data.get("contract_found", True),
        "whale_ping_strength": token_data.get("whale_ping_strength", 0.0),
        "dex_inflow_usd": token_data.get("dex_inflow_usd", 0.0),
        "feedback_history_count": len(token_data.get("feedback_history", [])),
        "explore_trigger_reason": token_data.get("explore_trigger_reason", "unknown")
    }
    
    return log_entry

def get_cold_start_statistics(token_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about cold start tokens.
    
    Args:
        token_data_list: List of token analysis data
        
    Returns:
        Cold start statistics
    """
    cold_start_tokens = [token for token in token_data_list if is_cold_start(token)]
    total_tokens = len(token_data_list)
    cold_start_count = len(cold_start_tokens)
    
    if cold_start_count == 0:
        return {
            "total_tokens": total_tokens,
            "cold_start_count": 0,
            "cold_start_percentage": 0.0,
            "average_score": 0.0,
            "explore_eligible": 0
        }
    
    # Calculate explore mode eligibility
    explore_eligible = sum(1 for token in cold_start_tokens 
                          if should_explore_mode_trigger(token, token.get("final_score", 0.0))[0])
    
    # Average score of cold start tokens
    avg_score = sum(token.get("final_score", 0.0) for token in cold_start_tokens) / cold_start_count
    
    return {
        "total_tokens": total_tokens,
        "cold_start_count": cold_start_count,
        "cold_start_percentage": round((cold_start_count / total_tokens) * 100, 2),
        "average_score": round(avg_score, 3),
        "explore_eligible": explore_eligible,
        "explore_eligible_percentage": round((explore_eligible / cold_start_count) * 100, 2) if cold_start_count > 0 else 0.0
    }