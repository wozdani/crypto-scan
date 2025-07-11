#!/usr/bin/env python3
"""
Dual Engine Decision System - Independent TJDE & Stealth Scoring
🔧 Architectural Separation: TJDE (Trend Mode) + Stealth Engine

This module implements completely separated decision engines:
- TJDE Engine: Trend-following analysis for active trend entry
- Stealth Engine: Smart money detection for hidden signals
- Hybrid Logic: Combined decisions when both engines activate
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def compute_dual_engine_decision(symbol: str, market_data: Dict, tjde_result: Dict, stealth_result: Dict) -> Dict:
    """
    🎯 Dual Engine Decision - Complete separation of TJDE and Stealth scoring
    
    Args:
        symbol: Trading symbol
        market_data: Market data dictionary
        tjde_result: TJDE analysis result
        stealth_result: Stealth engine result
        
    Returns:
        Complete decision with separated scoring and hybrid logic
    """
    print(f"[DUAL ENGINE] {symbol}: Processing independent TJDE + Stealth decisions")
    
    # === TJDE DECISION ENGINE ===
    tjde_score = tjde_result.get('final_score', 0.0)
    tjde_confidence = tjde_result.get('confidence', 0.0)
    
    # TJDE Decision Logic
    if tjde_score >= 0.7:
        tjde_decision = "trend_alert"
        tjde_priority = "high"
    elif tjde_score >= 0.4:
        tjde_decision = "watch"
        tjde_priority = "medium"
    else:
        tjde_decision = "wait"
        tjde_priority = "low"
    
    print(f"[TJDE ENGINE] {symbol}: Score={tjde_score:.3f}, Decision={tjde_decision}")
    
    # === STEALTH DECISION ENGINE ===
    stealth_score = stealth_result.get('stealth_score', 0.0)
    stealth_signals = stealth_result.get('active_signals', [])
    
    # Stealth Decision Logic
    if stealth_score >= 0.75:
        stealth_decision = "stealth_alert"
        stealth_priority = "high"
    elif stealth_score >= 0.5:
        stealth_decision = "stealth_watch"
        stealth_priority = "medium"
    else:
        stealth_decision = "none"
        stealth_priority = "low"
    
    print(f"[STEALTH ENGINE] {symbol}: Score={stealth_score:.3f}, Decision={stealth_decision}")
    
    # === HYBRID DECISION LOGIC ===
    final_decision, alert_type, combined_priority = compute_hybrid_decision(
        tjde_decision, stealth_decision, tjde_priority, stealth_priority
    )
    
    print(f"[HYBRID LOGIC] {symbol}: Final={final_decision}, Type={alert_type}")
    
    # === COMBINED PRIORITY BOOST ===
    priority_boost = 0.0
    if tjde_decision == "trend_alert" and stealth_decision == "stealth_alert":
        priority_boost = 0.3  # Strong boost for hybrid alerts
        print(f"[PRIORITY BOOST] {symbol}: Hybrid alert boost: +{priority_boost}")
    elif tjde_decision == "trend_alert":
        priority_boost = 0.15  # Moderate boost for trend alerts
    elif stealth_decision == "stealth_alert":
        priority_boost = 0.2   # Higher boost for stealth (rarer but valuable)
    
    # Build comprehensive result
    result = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        
        # TJDE Results
        "tjde_score": tjde_score,
        "tjde_decision": tjde_decision,
        "tjde_confidence": tjde_confidence,
        "tjde_priority": tjde_priority,
        "tjde_breakdown": tjde_result.get('score_breakdown', {}),
        
        # Stealth Results
        "stealth_score": stealth_score,
        "stealth_decision": stealth_decision,
        "stealth_signals": stealth_signals,
        "stealth_priority": stealth_priority,
        
        # Hybrid Results
        "final_decision": final_decision,
        "alert_type": alert_type,
        "combined_priority": combined_priority,
        "priority_boost": priority_boost,
        
        # Decision Context
        "reasoning": build_decision_reasoning(tjde_result, stealth_result, final_decision),
        "market_phase": tjde_result.get('market_phase', 'unknown'),
        "setup_type": tjde_result.get('setup_type', 'unknown')
    }
    
    # Log decision for monitoring
    log_dual_engine_decision(result)
    
    return result


def compute_hybrid_decision(tjde_decision: str, stealth_decision: str, 
                          tjde_priority: str, stealth_priority: str) -> tuple:
    """
    Compute hybrid decision logic combining both engines
    
    Returns:
        tuple: (final_decision, alert_type, combined_priority)
    """
    
    # Hybrid Alert - Both engines active
    if tjde_decision == "trend_alert" and stealth_decision == "stealth_alert":
        return "hybrid_alert", "[💣 HYBRID ALERT]", "critical"
    
    # Trend Alert - TJDE dominant
    elif tjde_decision == "trend_alert":
        return "trend_alert", "[🔥 TREND ALERT]", "high"
    
    # Stealth Alert - Stealth dominant
    elif stealth_decision == "stealth_alert":
        return "stealth_alert", "[🕵️ STEALTH ALERT]", "high"
    
    # Watch - Either engine showing interest
    elif tjde_decision == "watch" or stealth_decision == "stealth_watch":
        if tjde_decision == "watch" and stealth_decision == "stealth_watch":
            return "dual_watch", "[👀 DUAL WATCH]", "medium"
        elif tjde_decision == "watch":
            return "trend_watch", "[📈 TREND WATCH]", "medium"
        else:
            return "stealth_watch", "[🔍 STEALTH WATCH]", "medium"
    
    # Wait - No significant signals
    else:
        return "wait", "[⏳ WAIT]", "low"


def build_decision_reasoning(tjde_result: Dict, stealth_result: Dict, final_decision: str) -> List[str]:
    """Build comprehensive reasoning for decision"""
    reasons = []
    
    # TJDE Reasoning
    tjde_score = tjde_result.get('final_score', 0.0)
    if tjde_score >= 0.7:
        reasons.append(f"Strong trend setup - TJDE {tjde_score:.3f}")
    elif tjde_score >= 0.4:
        reasons.append(f"Moderate trend potential - TJDE {tjde_score:.3f}")
    else:
        reasons.append(f"Weak trend signals - TJDE {tjde_score:.3f}")
    
    # Stealth Reasoning
    stealth_score = stealth_result.get('stealth_score', 0.0)
    active_signals = stealth_result.get('active_signals', [])
    
    if stealth_score >= 0.75:
        reasons.append(f"Strong smart money activity - Stealth {stealth_score:.3f}")
        if active_signals:
            top_signals = sorted(active_signals, key=lambda x: x.get('strength', 0), reverse=True)[:2]
            signal_names = [s.get('name', 'unknown') for s in top_signals]
            reasons.append(f"Key signals: {', '.join(signal_names)}")
    elif stealth_score >= 0.5:
        reasons.append(f"Moderate smart money signals - Stealth {stealth_score:.3f}")
    
    # Hybrid reasoning
    if final_decision == "hybrid_alert":
        reasons.append("🎯 EXCEPTIONAL: Both trend and smart money aligned")
    
    return reasons


def log_dual_engine_decision(result: Dict):
    """Log decision to dual engine decision log"""
    try:
        log_file = "data/dual_engine_decisions.jsonl"
        os.makedirs("data", exist_ok=True)
        
        log_entry = {
            "symbol": result["symbol"],
            "timestamp": result["timestamp"],
            "tjde_score": result["tjde_score"],
            "tjde_decision": result["tjde_decision"],
            "stealth_score": result["stealth_score"],
            "stealth_decision": result["stealth_decision"],
            "final_decision": result["final_decision"],
            "alert_type": result["alert_type"],
            "priority_boost": result["priority_boost"]
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    except Exception as e:
        logger.error(f"Failed to log dual engine decision: {e}")


def get_alert_message_format(result: Dict) -> Dict:
    """
    Generate formatted alert message for Telegram
    
    Args:
        result: Dual engine decision result
        
    Returns:
        Dictionary with formatted message components
    """
    alert_type = result["alert_type"]
    symbol = result["symbol"]
    tjde_score = result["tjde_score"]
    stealth_score = result["stealth_score"]
    final_decision = result["final_decision"]
    
    # Build alert header
    if final_decision == "hybrid_alert":
        header = f"{alert_type} {symbol}"
        priority_emoji = "🚨"
    elif final_decision == "trend_alert":
        header = f"{alert_type} {symbol}"
        priority_emoji = "🔥"
    elif final_decision == "stealth_alert":
        header = f"{alert_type} {symbol}"
        priority_emoji = "🕵️"
    else:
        header = f"{alert_type} {symbol}"
        priority_emoji = "👀"
    
    # Build score display
    score_line = f"TJDE: {tjde_score:.3f} | Stealth: {stealth_score:.3f}"
    
    # Build reasoning
    reasons = result.get("reasoning", [])
    reasoning_text = "\n".join([f"• {reason}" for reason in reasons[:3]])
    
    message_parts = {
        "header": f"{priority_emoji} {header}",
        "scores": score_line,
        "reasoning": reasoning_text,
        "market_phase": result.get("market_phase", "unknown"),
        "priority": result.get("combined_priority", "low")
    }
    
    return message_parts


# === GLOBAL CONVENIENCE FUNCTIONS ===

def analyze_with_dual_engines(symbol: str, market_data: Dict, tjde_result: Dict, stealth_result: Dict) -> Dict:
    """Convenience function for dual engine analysis"""
    return compute_dual_engine_decision(symbol, market_data, tjde_result, stealth_result)


def is_hybrid_alert(result: Dict) -> bool:
    """Check if result is a hybrid alert"""
    return result.get("final_decision") == "hybrid_alert"


def get_primary_engine(result: Dict) -> str:
    """Get primary engine for the decision"""
    final_decision = result.get("final_decision", "wait")
    
    if final_decision == "hybrid_alert":
        return "hybrid"
    elif final_decision in ["trend_alert", "trend_watch"]:
        return "tjde"
    elif final_decision in ["stealth_alert", "stealth_watch"]:
        return "stealth"
    else:
        return "none"


def get_decision_priority_score(result: Dict) -> float:
    """Calculate priority score for sorting alerts"""
    base_priority = {
        "hybrid_alert": 10.0,
        "stealth_alert": 8.0,
        "trend_alert": 7.0,
        "dual_watch": 5.0,
        "stealth_watch": 4.0,
        "trend_watch": 3.0,
        "wait": 1.0
    }
    
    decision = result.get("final_decision", "wait")
    priority = base_priority.get(decision, 1.0)
    priority_boost = result.get("priority_boost", 0.0)
    
    return priority + priority_boost * 10.0  # Scale boost for sorting


if __name__ == "__main__":
    # Test dual engine system
    print("Dual Engine Decision System - Architectural Separation Complete")
    print("✅ TJDE Engine: Independent trend analysis")
    print("✅ Stealth Engine: Independent smart money detection") 
    print("✅ Hybrid Logic: Combined decision making")
    print("✅ Priority System: Enhanced alert sorting")