#!/usr/bin/env python3
"""
Stage 7 - Final Decision Classification
TJDE v2 Final Decision Engine: LONG / WAIT / AVOID

Transforms TJDE scores into clear trading decisions with dynamic thresholds
based on CLIP visual confidence and market conditions.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def classify_final_decision(
    adjusted_score: float, 
    clip_confidence: Optional[float] = None,
    market_phase: str = "trend-following",
    support_reaction: Optional[float] = None
) -> Tuple[str, Dict]:
    """
    Stage 7: Classify final trading decision based on adjusted score and confidence
    
    Args:
        adjusted_score: Final TJDE score after market phase modifier
        clip_confidence: Visual confidence from CLIP model (0.0-1.0)
        market_phase: Current market phase for phase-specific thresholds
        support_reaction: Support/resistance reaction quality
        
    Returns:
        Tuple of (decision, classification_info)
        decision: "long", "wait", "avoid"
        classification_info: Dict with thresholds and reasoning
    """
    
    # Base thresholds
    base_threshold_long = 0.70
    base_threshold_wait = 0.55
    
    # Phase-specific threshold adjustments
    phase_adjustments = {
        "pre-pump": {"long": -0.05, "wait": -0.03},  # More aggressive for pre-pump
        "breakout": {"long": -0.03, "wait": -0.02},  # Slightly more aggressive
        "trend-following": {"long": 0.0, "wait": 0.0},  # Standard thresholds
        "consolidation": {"long": +0.08, "wait": +0.05}  # More conservative (increased further)
    }
    
    # Apply phase adjustments
    phase_adj = phase_adjustments.get(market_phase, {"long": 0.0, "wait": 0.0})
    threshold_long = base_threshold_long + phase_adj["long"]
    threshold_wait = base_threshold_wait + phase_adj["wait"]
    
    # CLIP confidence adjustments
    clip_adjustment_long = 0.0
    clip_adjustment_wait = 0.0
    
    if clip_confidence is not None:
        if clip_confidence >= 0.90:
            # Very high visual confidence - aggressive threshold reduction
            clip_adjustment_long = -0.05
            clip_adjustment_wait = -0.03
        elif clip_confidence >= 0.75:
            # High visual confidence - lower thresholds
            clip_adjustment_long = -0.03
            clip_adjustment_wait = -0.02
        elif clip_confidence >= 0.60:
            # Medium-high confidence - slight threshold reduction
            clip_adjustment_long = -0.02
            clip_adjustment_wait = -0.01
        elif clip_confidence < 0.30:
            # Low visual confidence - raise thresholds (more conservative)
            clip_adjustment_long = +0.02
            clip_adjustment_wait = +0.01
    
    # Apply CLIP adjustments
    threshold_long += clip_adjustment_long
    threshold_wait += clip_adjustment_wait
    
    # Support reaction override (safety mechanism)
    support_override = False
    if support_reaction is not None and support_reaction < 0.2:
        # Very weak support reaction - force more conservative decision
        threshold_long += 0.15  # Very strong penalty for weak support
        threshold_wait += 0.08
        support_override = True
    
    # Make final decision
    if adjusted_score >= threshold_long:
        decision = "long"
        confidence_level = "high" if adjusted_score >= (threshold_long + 0.1) else "medium"
    elif adjusted_score >= threshold_wait:
        decision = "wait"
        confidence_level = "medium" if adjusted_score >= (threshold_wait + 0.05) else "low"
    else:
        decision = "avoid"
        confidence_level = "high"  # High confidence to avoid
    
    # Classification info for debugging and analysis
    classification_info = {
        "adjusted_score": adjusted_score,
        "threshold_long": threshold_long,
        "threshold_wait": threshold_wait,
        "base_threshold_long": base_threshold_long,
        "base_threshold_wait": base_threshold_wait,
        "phase_adjustment": phase_adj,
        "clip_confidence": clip_confidence,
        "clip_adjustment_long": clip_adjustment_long,
        "clip_adjustment_wait": clip_adjustment_wait,
        "support_override": support_override,
        "confidence_level": confidence_level,
        "market_phase": market_phase,
        "reasoning": _get_decision_reasoning(
            decision, adjusted_score, threshold_long, threshold_wait,
            clip_confidence, support_override, market_phase
        )
    }
    
    return decision, classification_info


def _get_decision_reasoning(
    decision: str, 
    score: float, 
    threshold_long: float, 
    threshold_wait: float,
    clip_confidence: Optional[float],
    support_override: bool,
    market_phase: str
) -> str:
    """Generate human-readable reasoning for the decision"""
    
    reasons = []
    
    if decision == "long":
        reasons.append(f"Score {score:.3f} exceeds LONG threshold {threshold_long:.3f}")
        if clip_confidence and clip_confidence >= 0.60:
            reasons.append(f"High visual confidence ({clip_confidence:.2f}) supports entry")
        if market_phase == "pre-pump":
            reasons.append("Pre-pump phase allows aggressive entry")
    
    elif decision == "wait":
        reasons.append(f"Score {score:.3f} in WAIT range ({threshold_wait:.3f}-{threshold_long:.3f})")
        if support_override:
            reasons.append("Weak support reaction suggests caution")
        if market_phase == "consolidation":
            reasons.append("Consolidation phase requires patience")
    
    else:  # avoid
        reasons.append(f"Score {score:.3f} below WAIT threshold {threshold_wait:.3f}")
        if clip_confidence and clip_confidence < 0.30:
            reasons.append(f"Low visual confidence ({clip_confidence:.2f}) confirms avoidance")
    
    return " | ".join(reasons)


def log_decision_classification(symbol: str, decision: str, classification_info: Dict):
    """Log detailed decision classification for debugging"""
    
    info = classification_info
    
    print(f"[TJDE STAGE 7] {symbol}: Final Decision Classification")
    print(f"[DECISION] {symbol}: {decision.upper()} (confidence: {info['confidence_level']})")
    print(f"[THRESHOLDS] LONG: {info['threshold_long']:.3f}, WAIT: {info['threshold_wait']:.3f}")
    print(f"[SCORE] Adjusted: {info['adjusted_score']:.3f}")
    
    if info['clip_confidence'] is not None:
        print(f"[CLIP] Confidence: {info['clip_confidence']:.2f}, Adjustment: {info['clip_adjustment_long']:+.3f}")
    
    if info['support_override']:
        print(f"[OVERRIDE] Weak support reaction triggered conservative adjustment")
    
    print(f"[REASONING] {info['reasoning']}")
    print()


def get_decision_statistics(decisions: list) -> Dict:
    """Calculate statistics for a batch of decisions"""
    
    if not decisions:
        return {}
    
    total = len(decisions)
    long_count = sum(1 for d in decisions if d == "long")
    wait_count = sum(1 for d in decisions if d == "wait")
    avoid_count = sum(1 for d in decisions if d == "avoid")
    
    return {
        "total_decisions": total,
        "long_count": long_count,
        "wait_count": wait_count,
        "avoid_count": avoid_count,
        "long_percentage": (long_count / total) * 100,
        "wait_percentage": (wait_count / total) * 100,
        "avoid_percentage": (avoid_count / total) * 100,
        "actionable_percentage": ((long_count + wait_count) / total) * 100
    }


def test_decision_classifier():
    """Test the decision classifier with various scenarios"""
    
    test_cases = [
        # (score, clip_confidence, phase, expected_decision)
        (0.75, 0.8, "trend-following", "long"),  # High score + high confidence
        (0.68, 0.7, "pre-pump", "long"),        # Medium score + pre-pump phase
        (0.65, 0.3, "consolidation", "wait"),   # Medium score + low confidence
        (0.45, 0.6, "trend-following", "avoid"), # Low score
        (0.72, None, "trend-following", "long"), # High score without CLIP
        (0.58, 0.8, "consolidation", "wait"),   # Conservative phase
    ]
    
    print("🧪 Testing Final Decision Classifier")
    print("=" * 50)
    
    for i, (score, clip_conf, phase, expected) in enumerate(test_cases, 1):
        decision, info = classify_final_decision(score, clip_conf, phase)
        
        status = "✅ PASS" if decision == expected else "❌ FAIL"
        
        print(f"Test {i}: {status}")
        print(f"  Score: {score}, CLIP: {clip_conf}, Phase: {phase}")
        print(f"  Expected: {expected}, Got: {decision}")
        print(f"  Reasoning: {info['reasoning']}")
        print()
    
    print("🎯 Decision Classifier Test Complete")


if __name__ == "__main__":
    test_decision_classifier()