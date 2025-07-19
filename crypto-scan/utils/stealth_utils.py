"""
Stealth Engine Utilities
Utility functions for stealth signal detection and analysis
"""

from typing import Dict, List, Any, Optional

def is_cold_start(token_data: Dict[str, Any]) -> bool:
    """
    Returns True if token has no historical trust or feedback data.
    
    Args:
        token_data: Dictionary containing token analysis data
        
    Returns:
        True if token meets cold start criteria
    """
    trust = token_data.get("trust_addresses", 0)
    feedbacks = token_data.get("feedback_history", [])
    
    # Simple cold start criteria - only trust and feedback
    return (trust == 0 and               # No trust addresses
            len(feedbacks) == 0)         # No feedback history

def should_explore_mode_trigger(token_data: Dict[str, Any]) -> bool:
    """
    Check if token should trigger explore mode experimental alert.
    
    Args:
        token_data: Token analysis data
        
    Returns:
        True if should trigger explore mode
    """
    final_score = token_data.get("final_score", 0.0)
    explore_score_threshold = 1.8  # Lowered from 2.8 to 1.8 for more triggers
    
    # Additional quality checks
    whale_ping_strength = token_data.get("whale_ping_strength", 0.0)
    dex_inflow_value = token_data.get("dex_inflow_usd", 0.0)
    active_signals = set(token_data.get("active_signals", []))
    core_signals = {'whale_ping', 'dex_inflow', 'orderbook_anomaly', 'spoofing_layers'}
    core_signal_count = len(active_signals.intersection(core_signals))
    
    # Enhanced debug logging dla explore mode analysis
    is_cold = is_cold_start(token_data)
    print(f"[EXPLORE MODE DEBUG] ====== EXPLORE MODE EVALUATION ======")
    print(f"[EXPLORE MODE DEBUG] Score check: {final_score:.3f} >= {explore_score_threshold}? {final_score >= explore_score_threshold}")
    print(f"[EXPLORE MODE DEBUG] Cold start status: {is_cold}")
    print(f"[EXPLORE MODE DEBUG] Trust addresses: {token_data.get('trust_addresses', 0)}")
    print(f"[EXPLORE MODE DEBUG] Whale memory entries: {token_data.get('whale_memory_entries', 0)}")
    print(f"[EXPLORE MODE DEBUG] Historical alerts count: {len(token_data.get('historical_alerts', []))}")
    print(f"[EXPLORE MODE DEBUG] Active signals: {list(active_signals)}")
    print(f"[EXPLORE MODE DEBUG] Core signals found: {core_signal_count} out of {len(core_signals)}")
    print(f"[EXPLORE MODE DEBUG] Core signal intersection: {active_signals.intersection(core_signals)}")
    print(f"[EXPLORE MODE DEBUG] Whale ping strength: {whale_ping_strength:.3f}")
    print(f"[EXPLORE MODE DEBUG] DEX inflow value: ${dex_inflow_value:.2f}")
    print(f"[EXPLORE MODE DEBUG] Contract found: {token_data.get('contract_found', 'unknown')}")
    
    # Relaxed scoring requirement - either cold start OR decent signals
    score_sufficient = final_score >= explore_score_threshold
    
    print(f"[EXPLORE MODE DEBUG] ----- DECISION LOGIC EVALUATION -----")
    
    if not score_sufficient:
        print(f"[EXPLORE MODE DECISION] ❌ REJECTED: Score too low ({final_score:.3f} < {explore_score_threshold})")
        print(f"[EXPLORE MODE DEBUG] ====== EXPLORE MODE REJECTED ======")
        return False
    
    print(f"[EXPLORE MODE DECISION] ✅ Score sufficient: {final_score:.3f} >= {explore_score_threshold}")
    
    # Require at least 1 core signal (lowered from 2)
    if core_signal_count < 1:
        print(f"[EXPLORE MODE DECISION] ❌ REJECTED: Insufficient core signals ({core_signal_count} < 1)")
        print(f"[EXPLORE MODE DEBUG] ====== EXPLORE MODE REJECTED ======")
        return False
    
    print(f"[EXPLORE MODE DECISION] ✅ Core signals sufficient: {core_signal_count} >= 1")
    
    # High-quality whale ping (lowered threshold to > 0.5)
    if whale_ping_strength > 0.5:
        print(f"[EXPLORE MODE DECISION] ✅ TRIGGERED: Quality whale ping detected ({whale_ping_strength:.3f} > 0.5)")
        print(f"[EXPLORE MODE DEBUG] ====== EXPLORE MODE TRIGGERED ======")
        return True
    
    print(f"[EXPLORE MODE DECISION] ⚠️ Whale ping not strong enough: {whale_ping_strength:.3f} <= 0.5")
    
    # Significant DEX inflow (lowered to > $5k)
    if dex_inflow_value > 5000:
        print(f"[EXPLORE MODE DECISION] ✅ TRIGGERED: Significant DEX inflow detected (${dex_inflow_value:.0f} > $5,000)")
        print(f"[EXPLORE MODE DEBUG] ====== EXPLORE MODE TRIGGERED ======")
        return True
    
    print(f"[EXPLORE MODE DECISION] ⚠️ DEX inflow not significant enough: ${dex_inflow_value:.0f} <= $5,000")
    
    # Multi-signal combination (lowered to >= 2)
    if core_signal_count >= 2:
        return True
    
    # Any decent signal + cold start
    if is_cold and core_signal_count >= 1:
        return True
    
    return False

def calculate_explore_mode_confidence(token_data: Dict[str, Any]) -> float:
    """
    Calculate synthetic confidence boost for explore mode.
    
    Args:
        token_data: Token analysis data
        
    Returns:
        Synthetic confidence score
    """
    base_confidence = 1.5  # Base explore mode confidence
    final_score = token_data.get("final_score", 0.0)
    
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

def format_explore_mode_reason(token_data: Dict[str, Any]) -> str:
    """
    Format explore mode reasoning for transparency.
    
    Args:
        token_data: Token analysis data
        
    Returns:
        Formatted reason string
    """
    active_signals = token_data.get("active_signals", [])
    core_signals = {'whale_ping', 'dex_inflow', 'orderbook_anomaly', 'spoofing_layers'}
    active_core = [sig for sig in active_signals if sig in core_signals]
    final_score = token_data.get("final_score", 0.0)
    
    # Generate trigger reason based on available data
    trigger_reason = "experimental_cold_start"
    if token_data.get("whale_ping_strength", 0) > 0.5:
        trigger_reason = "quality_whale_ping"
    elif token_data.get("dex_inflow_usd", 0) > 5000:
        trigger_reason = "significant_dex_inflow"
    elif len(active_core) >= 2:
        trigger_reason = "multi_signal_combination"
    
    reason_parts = [
        f"EXPLORE MODE: {trigger_reason}",
        f"Score: {final_score:.3f}",
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
                          if should_explore_mode_trigger(token))
    
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