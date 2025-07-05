"""
Vision Scoring - AI Label Integration
Converts AI vision analysis into scoring adjustments for TJDE decision engine
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def score_from_ai_label(ai_label: Dict) -> float:
    """
    Calculate scoring adjustment from AI vision analysis
    
    Args:
        ai_label: Dictionary containing label, phase, confidence from AI analysis
        
    Returns:
        Score adjustment (-0.20 to +0.20 range)
    """
    try:
        if not ai_label or not isinstance(ai_label, dict):
            return 0.0
            
        # Extract AI analysis components
        label = ai_label.get("label", "").lower()
        phase = ai_label.get("phase", "").lower()
        confidence = float(ai_label.get("confidence", 0.0))
        
        # Base score adjustment
        score_adjustment = 0.0
        
        # === BULLISH PATTERNS ===
        bullish_patterns = {
            "pullback": 0.08,      # Strong bullish - correction in uptrend
            "breakout": 0.12,      # Very strong - momentum breakout
            "early_trend": 0.10,   # Strong - early trend formation
            "accumulation": 0.06,  # Moderate - building position
            "retest": 0.05         # Moderate - support confirmation
        }
        
        # === BEARISH PATTERNS ===
        bearish_patterns = {
            "exhaustion": -0.12,   # Very bearish - trend exhaustion
            "chaos": -0.08,        # Bearish - unclear direction
            "range": -0.04         # Slightly bearish - sideways movement
        }
        
        # Get base pattern score
        if label in bullish_patterns:
            score_adjustment = bullish_patterns[label]
            logger.info(f"[VISION SCORE] 📈 Bullish pattern detected: {label} (+{score_adjustment})")
            
        elif label in bearish_patterns:
            score_adjustment = bearish_patterns[label]
            logger.info(f"[VISION SCORE] 📉 Bearish pattern detected: {label} ({score_adjustment})")
            
        else:
            logger.info(f"[VISION SCORE] ➡️ Neutral pattern: {label} (0.0)")
            return 0.0
            
        # === CONFIDENCE MODULATION ===
        # Only apply adjustment if confidence is sufficient
        confidence_threshold = 0.60  # Minimum confidence to apply adjustment
        
        if confidence < confidence_threshold:
            # Reduce adjustment for low confidence
            confidence_factor = confidence / confidence_threshold
            score_adjustment *= confidence_factor
            logger.info(f"[VISION SCORE] ⚠️ Low confidence {confidence:.2f} - reduced to {score_adjustment:.3f}")
            
        elif confidence >= 0.80:
            # Boost for high confidence
            high_confidence_boost = 1.2
            score_adjustment *= high_confidence_boost
            logger.info(f"[VISION SCORE] 🎯 High confidence {confidence:.2f} - boosted to {score_adjustment:.3f}")
            
        # === PHASE CONTEXT MODULATION ===
        phase_modifiers = {
            "trend": 1.1,          # Amplify in trending markets
            "accumulation": 1.05,  # Slight boost in accumulation
            "reversal": 0.9,       # Reduce in reversal (uncertainty)
            "distribution": 0.8,   # Reduce in distribution phase
            "consolidation": 0.85  # Reduce in consolidation
        }
        
        if phase in phase_modifiers:
            phase_factor = phase_modifiers[phase]
            original_adjustment = score_adjustment
            score_adjustment *= phase_factor
            
            logger.info(f"[VISION SCORE] 🔄 Phase '{phase}' modifier: {original_adjustment:.3f} → {score_adjustment:.3f}")
            
        # === FINAL BOUNDS CHECKING ===
        # Clamp to reasonable range
        max_adjustment = 0.20
        min_adjustment = -0.20
        
        score_adjustment = max(min_adjustment, min(max_adjustment, score_adjustment))
        
        logger.info(f"[VISION SCORE] ✅ Final AI adjustment: {score_adjustment:.3f} "
                   f"(pattern: {label}, confidence: {confidence:.2f}, phase: {phase})")
        
        return round(score_adjustment, 3)
        
    except Exception as e:
        logger.error(f"[VISION SCORE] ❌ Scoring failed: {e}")
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