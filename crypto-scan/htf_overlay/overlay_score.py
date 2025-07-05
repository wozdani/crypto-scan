"""
HTF Overlay Scoring - Higher Timeframe Context Scoring
Provides scoring adjustments based on HTF phase alignment with AI-EYE patterns
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def score_from_htf_overlay(htf_phase: Dict, ai_label: Dict) -> Dict:
    """
    Calculate scoring adjustment based on HTF phase and AI-EYE pattern alignment
    
    Args:
        htf_phase: HTF phase detection result from detect_htf_phase()
        ai_label: AI-EYE label result from prepare_ai_label()
        
    Returns:
        dict: {
            "adjustment": float,         # -0.20 to +0.20 scoring adjustment
            "reasoning": str,            # Explanation of the adjustment
            "alignment": str,            # "supportive", "neutral", "conflicting"
            "htf_context": str,          # HTF phase summary
            "confidence": float,         # Confidence in the adjustment
            "components": dict           # Detailed component analysis
        }
    """
    
    try:
        logger.info(f"[HTF OVERLAY] Starting HTF-AI alignment analysis")
        
        # Extract HTF phase information
        phase = htf_phase.get('phase', 'range')
        direction = htf_phase.get('direction', 'neutral')
        strength = htf_phase.get('strength', 0.5)
        confidence = htf_phase.get('confidence', 0.5)
        trend_quality = htf_phase.get('trend_quality', 'weak')
        
        # Extract AI-EYE information
        ai_pattern = ai_label.get('label', 'unknown')
        ai_confidence = ai_label.get('confidence', 0.0)
        ai_phase = ai_label.get('phase', 'unknown')
        
        logger.info(f"[HTF OVERLAY] HTF: {phase} ({direction}, strength: {strength:.2f}) | "
                   f"AI: {ai_pattern} (conf: {ai_confidence:.2f})")
        
        # === ALIGNMENT ANALYSIS ===
        
        alignment_result = _analyze_htf_ai_alignment(
            phase, direction, strength, trend_quality,
            ai_pattern, ai_confidence, ai_phase
        )
        
        # === SCORING CALCULATION ===
        
        base_adjustment = alignment_result['base_adjustment']
        
        # Apply confidence multipliers
        htf_confidence_multiplier = _calculate_confidence_multiplier(confidence, strength)
        ai_confidence_multiplier = _calculate_ai_confidence_multiplier(ai_confidence)
        
        # Final adjustment with bounds checking
        final_adjustment = base_adjustment * htf_confidence_multiplier * ai_confidence_multiplier
        final_adjustment = max(-0.20, min(0.20, final_adjustment))
        
        # Build result
        result = {
            'adjustment': round(final_adjustment, 3),
            'reasoning': alignment_result['reasoning'],
            'alignment': alignment_result['alignment'],
            'htf_context': f"{phase}_{direction}_{trend_quality}",
            'confidence': round((confidence + ai_confidence) / 2, 3),
            'components': {
                'htf_phase': {
                    'phase': phase,
                    'direction': direction,
                    'strength': strength,
                    'quality': trend_quality,
                    'confidence': confidence
                },
                'ai_pattern': {
                    'label': ai_pattern,
                    'confidence': ai_confidence,
                    'phase': ai_phase
                },
                'scoring': {
                    'base_adjustment': base_adjustment,
                    'htf_multiplier': htf_confidence_multiplier,
                    'ai_multiplier': ai_confidence_multiplier,
                    'final_adjustment': final_adjustment
                }
            }
        }
        
        logger.info(f"[HTF OVERLAY] Final adjustment: {final_adjustment:+.3f} "
                   f"({alignment_result['alignment']})")
        
        return result
        
    except Exception as e:
        logger.error(f"[HTF OVERLAY ERROR] Scoring failed: {e}")
        return _default_overlay_score()

def _analyze_htf_ai_alignment(phase: str, direction: str, strength: float, 
                             trend_quality: str, ai_pattern: str, 
                             ai_confidence: float, ai_phase: str) -> Dict:
    """Analyze alignment between HTF phase and AI pattern"""
    
    # === UPTREND SCENARIOS ===
    if phase == "uptrend" and direction == "bullish":
        
        if ai_pattern in ["pullback", "pullback_in_trend", "trend_continuation"] and ai_confidence >= 0.7:
            # Perfect pullback in uptrend
            if trend_quality == "strong":
                return {
                    'base_adjustment': 0.15,
                    'alignment': 'supportive',
                    'reasoning': f"Strong pullback in HTF uptrend - excellent entry opportunity"
                }
            else:
                return {
                    'base_adjustment': 0.10,
                    'alignment': 'supportive', 
                    'reasoning': f"Pullback in HTF uptrend - good entry setup"
                }
                
        elif ai_pattern in ["breakout", "breakout_pattern", "momentum_follow"] and ai_confidence >= 0.7:
            # Breakout continuation in uptrend
            return {
                'base_adjustment': 0.08,
                'alignment': 'supportive',
                'reasoning': f"Breakout continuation in HTF uptrend - trend acceleration"
            }
            
        elif ai_pattern in ["early_trend", "bullish_momentum"] and ai_confidence >= 0.6:
            # Early trend signals in uptrend
            return {
                'base_adjustment': 0.12,
                'alignment': 'supportive',
                'reasoning': f"Early bullish signals aligned with HTF uptrend"
            }
            
        elif ai_pattern in ["exhaustion", "reversal_pattern", "distribution"]:
            # Warning signals in uptrend
            return {
                'base_adjustment': -0.08,
                'alignment': 'conflicting',
                'reasoning': f"Reversal signals in HTF uptrend - potential trend change"
            }
            
    # === DOWNTREND SCENARIOS ===
    elif phase == "downtrend" and direction == "bearish":
        
        if ai_pattern in ["pullback", "retest"] and ai_confidence >= 0.7:
            # Pullback in downtrend (bearish continuation)
            return {
                'base_adjustment': -0.12,
                'alignment': 'conflicting',
                'reasoning': f"Bear market pullback - likely continuation lower"
            }
            
        elif ai_pattern in ["breakdown", "bearish_momentum"] and ai_confidence >= 0.7:
            # Breakdown in downtrend
            return {
                'base_adjustment': -0.10,
                'alignment': 'conflicting',
                'reasoning': f"Breakdown in HTF downtrend - avoid long positions"
            }
            
        elif ai_pattern in ["reversal_pattern", "oversold_bounce"] and ai_confidence >= 0.8:
            # Strong reversal in downtrend
            if strength < 0.7:  # Weak downtrend
                return {
                    'base_adjustment': 0.06,
                    'alignment': 'neutral',
                    'reasoning': f"Potential reversal in weak HTF downtrend"
                }
            else:
                return {
                    'base_adjustment': -0.05,
                    'alignment': 'conflicting',
                    'reasoning': f"Reversal attempt in strong HTF downtrend - risky"
                }
                
    # === RANGE/CONSOLIDATION SCENARIOS ===
    elif phase in ["range", "consolidation"]:
        
        if ai_pattern in ["range", "consolidation", "accumulation"] and ai_confidence >= 0.6:
            # Range-bound patterns in range market
            return {
                'base_adjustment': 0.05,
                'alignment': 'supportive',
                'reasoning': f"Range pattern in HTF consolidation - scalping opportunity"
            }
            
        elif ai_pattern in ["breakout", "breakout_pattern"] and ai_confidence >= 0.8:
            # Strong breakout from range
            return {
                'base_adjustment': 0.12,
                'alignment': 'supportive',
                'reasoning': f"Strong breakout from HTF consolidation - trend initiation"
            }
            
        elif ai_pattern in ["breakout", "breakout_pattern"] and ai_confidence < 0.65:
            # Weak breakout from range (likely fakeout)
            return {
                'base_adjustment': -0.08,
                'alignment': 'conflicting',
                'reasoning': f"Weak breakout from HTF range - potential fakeout"
            }
            
        elif ai_pattern in ["pullback", "retest"]:
            # Pullback in range
            return {
                'base_adjustment': 0.03,
                'alignment': 'neutral',
                'reasoning': f"Pullback in HTF range - mean reversion play"
            }
    
    # === DEFAULT/UNCERTAIN CASES ===
    
    # High confidence AI signals get some weight regardless
    if ai_confidence >= 0.85:
        return {
            'base_adjustment': 0.05,
            'alignment': 'neutral',
            'reasoning': f"High confidence AI signal despite unclear HTF alignment"
        }
    
    # Low confidence in both HTF and AI
    if ai_confidence < 0.5 and strength < 0.5:
        return {
            'base_adjustment': -0.05,
            'alignment': 'conflicting',
            'reasoning': f"Low confidence in both HTF and AI analysis"
        }
    
    # Default neutral case
    return {
        'base_adjustment': 0.0,
        'alignment': 'neutral',
        'reasoning': f"Neutral HTF-AI alignment: {phase} vs {ai_pattern}"
    }

def _calculate_confidence_multiplier(htf_confidence: float, htf_strength: float) -> float:
    """Calculate confidence multiplier for HTF analysis"""
    
    # Base multiplier from HTF confidence
    confidence_multiplier = 0.5 + (htf_confidence * 0.5)  # 0.5 to 1.0
    
    # Strength adjustment
    strength_multiplier = 0.7 + (htf_strength * 0.3)  # 0.7 to 1.0
    
    # Combined multiplier
    combined = confidence_multiplier * strength_multiplier
    
    return max(0.3, min(1.2, combined))  # Bounds: 0.3 to 1.2

def _calculate_ai_confidence_multiplier(ai_confidence: float) -> float:
    """Calculate confidence multiplier for AI analysis"""
    
    if ai_confidence >= 0.8:
        return 1.1  # Boost for high confidence
    elif ai_confidence >= 0.6:
        return 1.0  # Full weight for good confidence
    elif ai_confidence >= 0.4:
        return 0.8  # Reduced weight for medium confidence
    else:
        return 0.5  # Heavily reduced for low confidence

def create_htf_overlay_summary(htf_overlay_result: Dict) -> str:
    """Create human-readable summary of HTF overlay analysis"""
    
    try:
        adjustment = htf_overlay_result.get('adjustment', 0.0)
        alignment = htf_overlay_result.get('alignment', 'neutral')
        reasoning = htf_overlay_result.get('reasoning', 'No specific reasoning')
        htf_context = htf_overlay_result.get('htf_context', 'unknown')
        
        # Format adjustment with sign
        adj_str = f"{adjustment:+.3f}" if adjustment != 0 else "0.000"
        
        # Create summary
        summary = f"HTF Overlay: {adj_str} ({alignment}) - {reasoning}"
        
        # Add context if available
        if htf_context != 'unknown':
            summary += f" | Context: {htf_context}"
            
        return summary
        
    except Exception as e:
        logger.error(f"[HTF SUMMARY ERROR] Failed to create summary: {e}")
        return "HTF Overlay: Analysis failed"

def _default_overlay_score() -> Dict:
    """Return default overlay score when analysis fails"""
    return {
        'adjustment': 0.0,
        'reasoning': 'HTF overlay analysis failed - using neutral adjustment',
        'alignment': 'neutral',
        'htf_context': 'analysis_failed',
        'confidence': 0.0,
        'components': {
            'error': 'HTF overlay scoring failed'
        }
    }

def test_htf_overlay_scoring():
    """Test HTF overlay scoring with various scenarios"""
    
    print("🧪 Testing HTF Overlay Scoring:")
    print("=" * 50)
    
    # Test Case 1: Pullback in strong uptrend
    htf_phase_1 = {
        'phase': 'uptrend',
        'direction': 'bullish',
        'strength': 0.85,
        'confidence': 0.80,
        'trend_quality': 'strong'
    }
    
    ai_label_1 = {
        'label': 'pullback_in_trend',
        'confidence': 0.78,
        'phase': 'trend'
    }
    
    result_1 = score_from_htf_overlay(htf_phase_1, ai_label_1)
    print(f"Test 1 - Strong Pullback in Uptrend:")
    print(f"  Adjustment: {result_1['adjustment']:+.3f}")
    print(f"  Alignment: {result_1['alignment']}")
    print(f"  Reasoning: {result_1['reasoning']}")
    print()
    
    # Test Case 2: Weak breakout in range
    htf_phase_2 = {
        'phase': 'range',
        'direction': 'neutral',
        'strength': 0.3,
        'confidence': 0.65,
        'trend_quality': 'choppy'
    }
    
    ai_label_2 = {
        'label': 'breakout_pattern',
        'confidence': 0.55,
        'phase': 'breakout'
    }
    
    result_2 = score_from_htf_overlay(htf_phase_2, ai_label_2)
    print(f"Test 2 - Weak Breakout in Range:")
    print(f"  Adjustment: {result_2['adjustment']:+.3f}")
    print(f"  Alignment: {result_2['alignment']}")
    print(f"  Reasoning: {result_2['reasoning']}")
    print()
    
    # Test Case 3: High confidence AI in downtrend
    htf_phase_3 = {
        'phase': 'downtrend', 
        'direction': 'bearish',
        'strength': 0.75,
        'confidence': 0.70,
        'trend_quality': 'strong'
    }
    
    ai_label_3 = {
        'label': 'pullback',
        'confidence': 0.85,
        'phase': 'trend'
    }
    
    result_3 = score_from_htf_overlay(htf_phase_3, ai_label_3)
    print(f"Test 3 - Pullback in Strong Downtrend:")
    print(f"  Adjustment: {result_3['adjustment']:+.3f}")
    print(f"  Alignment: {result_3['alignment']}")
    print(f"  Reasoning: {result_3['reasoning']}")

if __name__ == "__main__":
    test_htf_overlay_scoring()