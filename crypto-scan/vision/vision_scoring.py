"""
Vision Scoring - AI Label Integration
Converts AI vision analysis into scoring adjustments for TJDE decision engine
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def score_from_ai_label(ai_label: Dict, market_phase: str = None) -> float:
    """
    Calculate scoring adjustment from AI vision analysis with dynamic feedback loop weights
    
    Args:
        ai_label: Dictionary containing label, phase, confidence from AI analysis
        market_phase: Current market phase for context-aware scoring
        
    Returns:
        Score adjustment (-0.20 to +0.20 range)
    """
    try:
        if not ai_label or not isinstance(ai_label, dict):
            return 0.0
            
        # Extract AI analysis components
        label = ai_label.get("label", "").lower().replace("-", "_")  # Normalize hyphens to underscores
        phase = ai_label.get("phase", "").lower()
        confidence = float(ai_label.get("confidence", 0.0))
        
        # Validate confidence
        if confidence <= 0.0 or confidence > 1.0:
            return 0.0
        
        # Base scoring for different patterns
        base_adjustments = {
            # Bullish patterns
            "pullback": 0.12,
            "pullback_continuation": 0.15,
            "breakout": 0.18,
            "breakout_pattern": 0.16,
            "momentum_follow": 0.14,
            "trend_continuation": 0.13,
            "trend_following": 0.13,  # FIX: Added missing trend-following pattern
            "support_bounce": 0.11,
            "pullback_in_trend": 0.12,
            "early_trend": 0.10,
            "accumulation": 0.06,
            "retest": 0.05,
            
            # Bearish/Caution patterns  
            "reversal": -0.15,
            "reversal_pattern": -0.17,
            "exhaustion": -0.12,
            "resistance_rejection": -0.13,
            "bear_trap": -0.10,
            
            # Neutral/Range patterns
            "range": -0.05,
            "consolidation": -0.03,
            "consolidation_squeeze": 0.02,
            "sideways": -0.04,
            
            # Uncertain patterns
            "chaos": -0.08,
            "unknown": -0.10,
            "no_clear_pattern": -0.12,
            "setup_analysis": -0.15
        }
        
        base_adjustment = base_adjustments.get(label, 0.0)
        
        # Apply dynamic weight from feedback loop
        try:
            from feedback_loop.weight_adjuster import get_effective_score_adjustment
            effective_adjustment = get_effective_score_adjustment(label, base_adjustment, confidence)
            
            logger.info(f"[VISION SCORE DYNAMIC] {label}: base {base_adjustment:.3f} → effective {effective_adjustment:.3f} "
                       f"(conf: {confidence:.2f})")
            
        except ImportError:
            # Fallback to static scoring if feedback loop not available
            effective_adjustment = base_adjustment * confidence
            logger.info(f"[VISION SCORE STATIC] {label}: {base_adjustment:.3f} × {confidence:.2f} = {effective_adjustment:.3f}")
        
        # Market phase context adjustments
        phase_modifier = 0.0
        phase_source = market_phase or phase  # Use market_phase if provided, otherwise AI phase
        
        if phase_source:
            phase_lower = phase_source.lower()
            
            # Enhance bullish patterns in uptrending phases
            if base_adjustment > 0 and phase_lower in ["trend-following", "uptrend", "breakout", "trend"]:
                phase_modifier = 0.02
                logger.info(f"[VISION SCORE] Bull pattern enhanced in {phase_lower} phase (+0.02)")
            
            # Enhance bearish patterns in downtrending phases  
            elif base_adjustment < 0 and phase_lower in ["downtrend", "bearish", "distribution"]:
                phase_modifier = -0.02
                logger.info(f"[VISION SCORE] Bear pattern enhanced in {phase_lower} phase (-0.02)")
            
            # Reduce pattern strength in consolidation
            elif phase_lower in ["consolidation", "range"]:
                phase_modifier = effective_adjustment * -0.1  # 10% reduction
                logger.info(f"[VISION SCORE] Pattern reduced in {phase_lower} phase ({phase_modifier:.3f})")
        
        final_adjustment = effective_adjustment + phase_modifier
        
        # Bounds checking
        final_adjustment = max(-0.20, min(0.20, final_adjustment))
        
        logger.info(f"[VISION SCORE] Final adjustment: {final_adjustment:.3f} "
                   f"(pattern: {label}, confidence: {confidence:.2f}, phase: {phase_source})")
        
        return round(final_adjustment, 4)
        
    except Exception as e:
        logger.error(f"[VISION SCORE ERROR] Scoring failed: {e}")
        return 0.0

def analyze_vision_confidence(ai_label: Dict) -> str:
    """
    Analyze overall confidence level from AI vision system
    
    Args:
        ai_label: AI analysis dictionary
        
    Returns:
        Confidence category: 'high', 'medium', 'low', 'very_low'
    """
    try:
        confidence = float(ai_label.get("confidence", 0.0))
        
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.70:
            return "medium"
        elif confidence >= 0.50:
            return "low"
        else:
            return "very_low"
            
    except Exception:
        return "very_low"

def get_pattern_strength(ai_label: Dict) -> float:
    """
    Get pattern strength indicator (0.0 - 1.0)
    
    Args:
        ai_label: AI analysis dictionary
        
    Returns:
        Pattern strength score
    """
    try:
        label = ai_label.get("label", "").lower()
        confidence = float(ai_label.get("confidence", 0.0))
        
        # Strong pattern multipliers
        strong_patterns = {
            "breakout": 1.0,
            "pullback": 0.9,
            "early_trend": 0.8,
            "accumulation": 0.7,
            "retest": 0.6,
            "exhaustion": 0.8,  # Strong bearish
            "range": 0.4,
            "chaos": 0.2
        }
        
        pattern_multiplier = strong_patterns.get(label, 0.5)
        strength = confidence * pattern_multiplier
        
        return round(strength, 3)
        
    except Exception:
        return 0.0

def create_vision_summary(ai_label: Dict) -> Dict:
    """
    Create comprehensive vision analysis summary
    
    Args:
        ai_label: AI analysis dictionary
        
    Returns:
        Summary dictionary with scoring and analysis
    """
    try:
        score_adjustment = score_from_ai_label(ai_label)
        confidence_level = analyze_vision_confidence(ai_label)
        pattern_strength = get_pattern_strength(ai_label)
        
        return {
            "score_adjustment": score_adjustment,
            "confidence_level": confidence_level,
            "pattern_strength": pattern_strength,
            "label": ai_label.get("label", "unknown"),
            "phase": ai_label.get("phase", "unknown"),
            "confidence": ai_label.get("confidence", 0.0),
            "reasoning": ai_label.get("reasoning", "No reasoning provided")
        }
        
    except Exception as e:
        logger.error(f"[VISION SUMMARY] ❌ Creation failed: {e}")
        return {
            "score_adjustment": 0.0,
            "confidence_level": "very_low",
            "pattern_strength": 0.0,
            "label": "error",
            "phase": "unknown",
            "confidence": 0.0,
            "reasoning": f"Analysis failed: {e}"
        }

def test_vision_scoring():
    """Test vision scoring functionality"""
    
    # Test cases
    test_cases = [
        {
            "label": "breakout",
            "phase": "trend", 
            "confidence": 0.85,
            "reasoning": "Strong breakout with high volume"
        },
        {
            "label": "pullback",
            "phase": "trend",
            "confidence": 0.70,
            "reasoning": "Healthy pullback in uptrend"
        },
        {
            "label": "exhaustion",
            "phase": "distribution",
            "confidence": 0.75,
            "reasoning": "Signs of trend exhaustion"
        },
        {
            "label": "chaos",
            "phase": "consolidation",
            "confidence": 0.40,
            "reasoning": "No clear pattern visible"
        }
    ]
    
    print("🧪 Testing Vision Scoring System:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['label'].upper()}")
        
        score = score_from_ai_label(test_case)
        summary = create_vision_summary(test_case)
        
        print(f"  Score Adjustment: {score:+.3f}")
        print(f"  Confidence Level: {summary['confidence_level']}")
        print(f"  Pattern Strength: {summary['pattern_strength']:.3f}")
        print(f"  Reasoning: {test_case['reasoning']}")

if __name__ == "__main__":
    test_vision_scoring()