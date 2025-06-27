"""
Alert Threshold Fix - Critical Logic for High TJDE Scores
Addresses the core issue where high TJDE scores (0.6+) don't generate alerts
"""
import logging

def fix_alert_thresholds(symbol: str, tjde_score: float, decision: str, clip_confidence: float = 0.0, gpt_commentary: str = "") -> dict:
    """
    Critical fix for alert threshold logic
    
    Args:
        symbol: Trading symbol
        tjde_score: TJDE score (0.0-1.0)
        decision: Current decision (avoid/consider_entry/join_trend)
        clip_confidence: CLIP confidence (0.0-1.0)
        gpt_commentary: GPT commentary text
        
    Returns:
        dict: Enhanced alert decision with reasoning
    """
    
    # Original thresholds are too restrictive
    original_decision = decision
    alert_generated = False
    alert_level = 0
    reasoning = []
    
    # Critical fix: Lower thresholds for high TJDE scores
    if tjde_score >= 0.7:
        # High TJDE score should generate alerts
        if decision == "avoid":
            decision = "consider_entry"
            reasoning.append(f"TJDE {tjde_score:.3f} too high for avoid → consider_entry")
        
        alert_generated = True
        alert_level = 2  # Medium alert level
        reasoning.append(f"High TJDE {tjde_score:.3f} generates alert")
    
    if tjde_score >= 0.7:
        # Very high TJDE should be strong alerts
        if decision != "join_trend":
            decision = "join_trend"
            reasoning.append(f"TJDE {tjde_score:.3f} too high for {original_decision} → join_trend")
        
        alert_level = 3  # High alert level
        reasoning.append(f"Very high TJDE {tjde_score:.3f} generates strong alert")
    
    # CLIP confidence boosts
    if clip_confidence > 0.6:
        if tjde_score >= 0.5 and decision == "avoid":
            decision = "consider_entry" 
            reasoning.append(f"CLIP {clip_confidence:.3f} + TJDE {tjde_score:.3f} → consider_entry")
            alert_generated = True
            alert_level = max(alert_level, 2)
        
        if clip_confidence > 0.75 and tjde_score >= 0.7:
            alert_level = 3
            reasoning.append(f"High CLIP {clip_confidence:.3f} + high TJDE → strong alert")
    
    # GPT commentary enhancement
    if gpt_commentary:
        commentary_lower = gpt_commentary.lower()
        bullish_terms = ["pullback", "squeeze", "support", "bounce", "breakout", "momentum", "strong"]
        bullish_count = sum(1 for term in bullish_terms if term in commentary_lower)
        
        if bullish_count >= 2 and tjde_score >= 0.5:
            if decision == "avoid":
                decision = "consider_entry"
                reasoning.append(f"GPT bullish ({bullish_count} signals) + TJDE {tjde_score:.3f} → consider_entry")
            alert_generated = True
            alert_level = max(alert_level, 2)
    
    # Safety net: Any score >0.75 should generate some alert
    if tjde_score >= 0.75 and not alert_generated:
        alert_generated = True
        alert_level = 2
        reasoning.append(f"Safety net: TJDE {tjde_score:.3f} > 0.75 forces alert")
    
    # Final decision validation
    if tjde_score >= 0.7 and decision == "avoid":
        decision = "consider_entry"
        reasoning.append("Final check: High TJDE cannot be avoid")
    
    return {
        "symbol": symbol,
        "original_decision": original_decision,
        "enhanced_decision": decision,
        "alert_generated": alert_generated,
        "alert_level": alert_level,
        "tjde_score": tjde_score,
        "clip_confidence": clip_confidence,
        "reasoning": reasoning,
        "decision_changed": decision != original_decision
    }

def should_generate_alert(tjde_score: float, decision: str, clip_confidence: float = 0.0) -> bool:
    """
    Simple boolean check if alert should be generated
    
    Returns:
        bool: True if alert should be generated
    """
    
    # Critical thresholds (much lower than current system)
    if tjde_score >= 0.7:
        return True
    
    if tjde_score >= 0.65 and clip_confidence > 0.6:
        return True
    
    if tjde_score >= 0.65 and decision in ["consider_entry", "join_trend"]:
        return True
    
    return False

def calculate_alert_level(tjde_score: float, decision: str, clip_confidence: float = 0.0) -> int:
    """
    Calculate alert level (1-3) based on score and confidence
    
    Returns:
        int: Alert level (1=low, 2=medium, 3=high)
    """
    
    if tjde_score >= 0.85 or (tjde_score >= 0.75 and clip_confidence > 0.7):
        return 3  # High alert
    
    if tjde_score >= 0.75 or (tjde_score >= 0.65 and clip_confidence > 0.6):
        return 2  # Medium alert
    
    if tjde_score >= 0.7 or (tjde_score >= 0.65 and clip_confidence > 0.5):
        return 1  # Low alert
    
    return 0  # No alert

def log_alert_fix(symbol: str, fix_result: dict):
    """Log alert threshold fix for debugging"""
    if fix_result.get("decision_changed", False):
        logging.info(f"[ALERT FIX] {symbol}: {fix_result['original_decision']} → {fix_result['enhanced_decision']}")
        logging.info(f"[ALERT FIX] {symbol}: Reasoning: {'; '.join(fix_result['reasoning'])}")
    
    if fix_result.get("alert_generated", False):
        logging.info(f"[ALERT FIX] {symbol}: Generated Level {fix_result['alert_level']} alert (TJDE: {fix_result['tjde_score']:.3f})")