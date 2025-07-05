"""
Trap Detector - Scoring Module
System scoringu penalizujący pułapki i fałszywe wybicia

Module 3 of Advanced Vision-AI System
"""

import logging
from typing import Dict, List, Optional
from .trap_patterns import (
    detect_fake_breakout,
    detect_failed_breakout,
    detect_bear_trap,
    detect_exhaustion_spike,
    comprehensive_trap_analysis
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def score_from_trap_detector(candles: List[Dict], ai_label: Dict, market_phase: str = "unknown") -> Dict:
    """
    Główna funkcja scoringu Trap Detector
    Penalizuje setupy które wyglądają jak okazje ale są prawdopodobnie pułapkami
    
    Args:
        candles: Lista świec do analizy
        ai_label: AI-EYE label z pattern i confidence
        market_phase: Faza rynku dla kontekstu
        
    Returns:
        Dict z adjustment i szczegółami analizy:
        {
            'adjustment': float (-0.20 to 0.0),
            'trap_detected': bool,
            'trap_types': List[str],
            'confidence': float,
            'reasoning': str,
            'trap_analysis': Dict
        }
    """
    
    if not candles or not ai_label:
        return {
            'adjustment': 0.0,
            'trap_detected': False,
            'trap_types': [],
            'confidence': 0.0,
            'reasoning': 'Insufficient data for trap analysis',
            'trap_analysis': {}
        }
    
    try:
        # Extract AI label information
        ai_pattern = ai_label.get('label', '').lower()
        ai_confidence = ai_label.get('confidence', 0.0)
        
        # Run comprehensive trap analysis
        trap_analysis = comprehensive_trap_analysis(candles)
        
        # Initialize scoring variables
        adjustment = 0.0
        penalty_applied = False
        reasoning_parts = []
        
        # === SCORING LOGIC FOR DIFFERENT AI PATTERNS ===
        
        # 1. BREAKOUT PATTERNS - Most vulnerable to bull traps
        if ai_pattern in ['breakout_pattern', 'breakout', 'bullish_breakout'] and ai_confidence >= 0.6:
            # Check for fake breakout (long upper wick + volume)
            fake_breakout = trap_analysis['detailed_results']['fake_breakout']
            if fake_breakout['detected']:
                penalty = -0.12 * fake_breakout['confidence']
                adjustment += penalty
                penalty_applied = True
                reasoning_parts.append(f"Fake breakout detected ({fake_breakout['confidence']:.2f}) - {penalty:.3f}")
            
            # Check for failed breakout (price returned below breakout level)
            failed_breakout = trap_analysis['detailed_results']['failed_breakout']
            if failed_breakout['detected']:
                penalty = -0.08 * failed_breakout['confidence']
                adjustment += penalty
                penalty_applied = True
                reasoning_parts.append(f"Failed breakout detected ({failed_breakout['confidence']:.2f}) - {penalty:.3f}")
        
        # 2. PULLBACK/CONTINUATION PATTERNS - Vulnerable to trend exhaustion
        elif ai_pattern in ['pullback_in_trend', 'trend_continuation', 'pullback_continuation'] and ai_confidence >= 0.7:
            # Check for exhaustion spike
            exhaustion = trap_analysis['detailed_results']['exhaustion_spike']
            if exhaustion['detected']:
                penalty = -0.10 * exhaustion['confidence']
                adjustment += penalty
                penalty_applied = True
                reasoning_parts.append(f"Exhaustion spike detected ({exhaustion['confidence']:.2f}) - {penalty:.3f}")
            
            # Additional check for fake breakout in continuation
            fake_breakout = trap_analysis['detailed_results']['fake_breakout']
            if fake_breakout['detected'] and fake_breakout['confidence'] > 0.7:
                penalty = -0.06 * fake_breakout['confidence']
                adjustment += penalty
                penalty_applied = True
                reasoning_parts.append(f"Fake continuation breakout - {penalty:.3f}")
        
        # 3. REVERSAL PATTERNS - Vulnerable to bear traps
        elif ai_pattern in ['reversal_pattern', 'trend_reversal', 'bottom_reversal'] and ai_confidence >= 0.6:
            # Check for bear trap (failed breakdown with recovery)
            bear_trap = trap_analysis['detailed_results']['bear_trap']
            if bear_trap['detected']:
                # For reversal patterns, bear trap might actually be positive (true reversal)
                # But we still penalize if it's unclear
                if bear_trap['confidence'] < 0.7:
                    penalty = -0.05 * bear_trap['confidence']
                    adjustment += penalty
                    penalty_applied = True
                    reasoning_parts.append(f"Uncertain bear trap in reversal - {penalty:.3f}")
        
        # 4. SUPPORT/RESISTANCE PATTERNS - Check for failed levels
        elif ai_pattern in ['support_test', 'resistance_break', 'support_break'] and ai_confidence >= 0.6:
            # Check for failed breakout at S/R levels
            failed_breakout = trap_analysis['detailed_results']['failed_breakout']
            if failed_breakout['detected']:
                penalty = -0.09 * failed_breakout['confidence']
                adjustment += penalty
                penalty_applied = True
                reasoning_parts.append(f"Failed S/R breakout - {penalty:.3f}")
        
        # === MARKET PHASE CONTEXT ADJUSTMENTS ===
        
        # In consolidation phase, breakouts are more likely to be false
        if market_phase in ['consolidation', 'range'] and 'breakout' in ai_pattern:
            if trap_analysis['any_trap_detected']:
                # Additional penalty for breakouts in ranging market
                context_penalty = -0.03
                adjustment += context_penalty
                reasoning_parts.append(f"Breakout in consolidation phase - {context_penalty:.3f}")
        
        # In strong trend, continuation setups get less penalty
        elif market_phase in ['trend-following', 'uptrend', 'downtrend'] and 'continuation' in ai_pattern:
            if trap_analysis['any_trap_detected'] and adjustment < 0:
                # Reduce penalty by 30% in strong trending markets
                adjustment *= 0.7
                reasoning_parts.append("Reduced trap penalty in trending market")
        
        # === HIGH CONFIDENCE AI PROTECTION ===
        
        # If AI is very confident (>0.85), reduce trap penalties
        if ai_confidence > 0.85 and adjustment < 0:
            original_adjustment = adjustment
            adjustment *= 0.6  # Reduce penalty by 40%
            reasoning_parts.append(f"High AI confidence protection: {original_adjustment:.3f} → {adjustment:.3f}")
        
        # === FINALIZE SCORING ===
        
        # Ensure adjustment is within bounds
        adjustment = max(adjustment, -0.20)  # Maximum penalty
        adjustment = min(adjustment, 0.0)    # No positive adjustments
        
        # Create final reasoning
        if reasoning_parts:
            reasoning = "; ".join(reasoning_parts)
        else:
            reasoning = "No significant trap patterns detected"
        
        # Determine overall confidence
        overall_confidence = trap_analysis['max_confidence'] if trap_analysis['any_trap_detected'] else 0.0
        
        return {
            'adjustment': round(adjustment, 3),
            'trap_detected': trap_analysis['any_trap_detected'],
            'trap_types': trap_analysis['trap_types'],
            'confidence': overall_confidence,
            'reasoning': reasoning,
            'trap_analysis': trap_analysis,
            'penalty_applied': penalty_applied,
            'ai_pattern_analyzed': ai_pattern,
            'market_phase_context': market_phase
        }
        
    except Exception as e:
        logger.error(f"Error in score_from_trap_detector: {e}")
        return {
            'adjustment': 0.0,
            'trap_detected': False,
            'trap_types': [],
            'confidence': 0.0,
            'reasoning': f"Error in trap analysis: {e}",
            'trap_analysis': {},
            'penalty_applied': False,
            'ai_pattern_analyzed': ai_pattern if 'ai_pattern' in locals() else 'unknown',
            'market_phase_context': market_phase
        }

def create_trap_detector_summary(trap_result: Dict) -> Dict:
    """
    Tworzy czytelne podsumowanie wyników Trap Detector
    
    Args:
        trap_result: Wynik z score_from_trap_detector
        
    Returns:
        Dict z podsumowaniem dla UI/logging
    """
    
    if not trap_result:
        return {
            'status': 'error',
            'message': 'No trap detection results available'
        }
    
    trap_detected = trap_result.get('trap_detected', False)
    adjustment = trap_result.get('adjustment', 0.0)
    trap_types = trap_result.get('trap_types', [])
    confidence = trap_result.get('confidence', 0.0)
    
    if trap_detected:
        if len(trap_types) == 1:
            trap_description = trap_types[0].replace('_', ' ').title()
        else:
            trap_description = f"{len(trap_types)} trap patterns"
        
        if abs(adjustment) >= 0.10:
            severity = "HIGH"
        elif abs(adjustment) >= 0.05:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        status_message = f"{severity} RISK: {trap_description} detected (confidence: {confidence:.2f})"
        
    else:
        status_message = "No trap patterns detected - setup appears clean"
        severity = "NONE"
        trap_description = "None"
    
    return {
        'status': 'trap_detected' if trap_detected else 'clean',
        'severity': severity,
        'message': status_message,
        'adjustment': adjustment,
        'trap_description': trap_description,
        'confidence': confidence,
        'trap_count': len(trap_types)
    }

def validate_trap_scoring_input(candles: List[Dict], ai_label: Dict) -> bool:
    """
    Waliduje dane wejściowe dla trap scoring
    
    Args:
        candles: Lista świec
        ai_label: AI label data
        
    Returns:
        bool: True jeśli dane są prawidłowe
    """
    
    if not candles or not isinstance(candles, list):
        return False
    
    if len(candles) < 3:  # Minimum 3 candles needed
        return False
    
    if not ai_label or not isinstance(ai_label, dict):
        return False
    
    # Check if candles have required fields
    required_candle_fields = ['open', 'high', 'low', 'close']
    for candle in candles[-3:]:  # Check last 3 candles
        if not all(field in candle for field in required_candle_fields):
            return False
    
    return True