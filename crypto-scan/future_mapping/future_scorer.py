"""
Future Scorer - Module 4 Future Scenario Mapping
Probabilistic scoring based on predicted future price paths
"""

from typing import Dict, List, Optional
import logging
from .scenario_evaluator import evaluate_future_scenario, enhanced_scenario_evaluation

def score_from_future_mapping(candles: List[Dict], ema: float = None, current_price: float = None, 
                            ai_label: Dict = None, market_phase: str = None) -> Dict:
    """
    Analyzes current candle and assigns score based on future scenario prediction
    
    Args:
        candles: List of candle data (last candle is current/forming)
        ema: EMA50 or other reference level for position analysis
        current_price: Current/live price for real-time analysis
        ai_label: AI pattern analysis for context
        market_phase: Current market phase for scenario weighting
        
    Returns:
        Dictionary with future mapping score and analysis
    """
    try:
        if not candles:
            return {
                'adjustment': 0.0,
                'scenario': 'neutral',
                'confidence': 0.0,
                'reasoning': 'No candle data available for future mapping',
                'analysis_completed': False
            }
        
        # Enhanced scenario evaluation with momentum context
        scenario_result = enhanced_scenario_evaluation(candles, ema, current_price)
        
        scenario = scenario_result['scenario']
        confidence = scenario_result['confidence']
        
        # Base scoring according to specification
        base_adjustment = 0.0
        if scenario == 'bull_case':
            base_adjustment = +0.07
        elif scenario == 'bear_case':
            base_adjustment = -0.07
        else:  # neutral
            base_adjustment = 0.0
        
        # Apply confidence scaling
        confidence_scaled_adjustment = base_adjustment * confidence
        
        # Context-aware adjustments
        context_modifier = 0.0
        context_reasons = []
        
        # AI label context enhancement
        if ai_label and isinstance(ai_label, dict):
            ai_pattern = ai_label.get('label', '').lower()
            ai_confidence = ai_label.get('confidence', 0.0)
            
            # Enhance bull scenarios with bullish AI patterns
            if scenario == 'bull_case' and any(pattern in ai_pattern for pattern in [
                'breakout', 'momentum_follow', 'trend_continuation', 'bullish'
            ]):
                context_modifier += 0.02 * ai_confidence
                context_reasons.append(f'bullish_ai_pattern_{ai_pattern}')
            
            # Enhance bear scenarios with bearish AI patterns
            elif scenario == 'bear_case' and any(pattern in ai_pattern for pattern in [
                'reversal', 'breakdown', 'bearish', 'exhaustion'
            ]):
                context_modifier += 0.02 * ai_confidence
                context_reasons.append(f'bearish_ai_pattern_{ai_pattern}')
            
            # Conflict detection
            elif scenario == 'bull_case' and any(pattern in ai_pattern for pattern in [
                'reversal', 'breakdown', 'bearish'
            ]):
                context_modifier -= 0.01
                context_reasons.append('ai_pattern_conflict')
            
            elif scenario == 'bear_case' and any(pattern in ai_pattern for pattern in [
                'breakout', 'momentum_follow', 'bullish'
            ]):
                context_modifier -= 0.01
                context_reasons.append('ai_pattern_conflict')
        
        # Market phase context enhancement
        if market_phase:
            phase = market_phase.lower()
            
            # Bull scenarios in trending markets
            if scenario == 'bull_case' and phase in ['trend-following', 'uptrend']:
                context_modifier += 0.01
                context_reasons.append(f'bull_trend_alignment_{phase}')
            
            # Bear scenarios in bearish phases
            elif scenario == 'bear_case' and phase in ['downtrend', 'bearish']:
                context_modifier += 0.01
                context_reasons.append(f'bear_trend_alignment_{phase}')
            
            # Consolidation reduces extreme scenarios
            elif phase in ['consolidation', 'range']:
                if abs(base_adjustment) > 0.05:
                    context_modifier -= 0.015
                    context_reasons.append('consolidation_dampening')
        
        # Momentum context from scenario analysis
        momentum_info = scenario_result.get('momentum_context', {})
        momentum = momentum_info.get('momentum', 'neutral')
        momentum_strength = momentum_info.get('strength', 0.0)
        
        if momentum == 'bullish' and scenario == 'bull_case':
            momentum_boost = min(0.02, momentum_strength * 0.05)
            context_modifier += momentum_boost
            context_reasons.append(f'momentum_alignment_{momentum_strength:.2f}')
        elif momentum == 'bearish' and scenario == 'bear_case':
            momentum_boost = min(0.02, momentum_strength * 0.05)
            context_modifier += momentum_boost
            context_reasons.append(f'momentum_alignment_{momentum_strength:.2f}')
        
        # Calculate final adjustment
        final_adjustment = confidence_scaled_adjustment + context_modifier
        
        # Bounds checking according to module specifications
        final_adjustment = max(-0.10, min(0.10, final_adjustment))
        
        # Detailed reasoning
        reasoning_parts = [
            f"Scenario: {scenario} (confidence: {confidence:.2f})",
            f"Base adjustment: {base_adjustment:+.3f}",
            f"Confidence scaled: {confidence_scaled_adjustment:+.3f}"
        ]
        
        if context_modifier != 0:
            reasoning_parts.append(f"Context modifier: {context_modifier:+.3f}")
            reasoning_parts.extend([f"  - {reason}" for reason in context_reasons])
        
        reasoning = "; ".join(reasoning_parts)
        
        return {
            'adjustment': round(final_adjustment, 3),
            'scenario': scenario,
            'confidence': confidence,
            'reasoning': reasoning,
            'analysis_completed': True,
            'scenario_details': scenario_result,
            'base_adjustment': base_adjustment,
            'context_modifier': round(context_modifier, 3),
            'context_reasons': context_reasons
        }
        
    except Exception as e:
        return {
            'adjustment': 0.0,
            'scenario': 'neutral',
            'confidence': 0.0,
            'reasoning': f'Future mapping error: {str(e)}',
            'analysis_completed': False
        }

def calculate_ema(candles: List[Dict], period: int = 50) -> Optional[float]:
    """
    Calculate EMA for use in future scenario mapping
    
    Args:
        candles: List of candle data
        period: EMA period (default 50)
        
    Returns:
        EMA value or None if insufficient data
    """
    try:
        if not candles or len(candles) < period:
            return None
        
        closes = []
        for candle in candles[-period:]:
            close = float(candle.get('close', 0))
            if close > 0:
                closes.append(close)
        
        if len(closes) < period:
            return None
        
        # Calculate EMA
        multiplier = 2 / (period + 1)
        ema = closes[0]  # Start with first close
        
        for close in closes[1:]:
            ema = (close * multiplier) + (ema * (1 - multiplier))
        
        return round(ema, 6)
        
    except Exception as e:
        logging.warning(f"EMA calculation error: {e}")
        return None

def get_scenario_probability_distribution(candles: List[Dict], ema: float = None) -> Dict:
    """
    Gets probability distribution across different scenarios for risk assessment
    
    Args:
        candles: List of candle data
        ema: EMA reference level
        
    Returns:
        Dictionary with probability distribution
    """
    try:
        if not candles:
            return {
                'bull_probability': 0.33,
                'bear_probability': 0.33,
                'neutral_probability': 0.34,
                'uncertainty': 1.0
            }
        
        scenario_result = enhanced_scenario_evaluation(candles, ema)
        confidence = scenario_result['confidence']
        scenario = scenario_result['scenario']
        
        # Convert confidence to probability distribution
        if scenario == 'bull_case':
            bull_prob = 0.33 + (confidence * 0.4)
            bear_prob = max(0.1, 0.33 - (confidence * 0.2))
            neutral_prob = 1.0 - bull_prob - bear_prob
        elif scenario == 'bear_case':
            bear_prob = 0.33 + (confidence * 0.4)
            bull_prob = max(0.1, 0.33 - (confidence * 0.2))
            neutral_prob = 1.0 - bull_prob - bear_prob
        else:  # neutral
            neutral_prob = 0.34 + (confidence * 0.3)
            bull_prob = (1.0 - neutral_prob) / 2
            bear_prob = (1.0 - neutral_prob) / 2
        
        uncertainty = 1.0 - confidence
        
        return {
            'bull_probability': round(bull_prob, 3),
            'bear_probability': round(bear_prob, 3),
            'neutral_probability': round(neutral_prob, 3),
            'uncertainty': round(uncertainty, 3),
            'dominant_scenario': scenario,
            'confidence': confidence
        }
        
    except Exception as e:
        return {
            'bull_probability': 0.33,
            'bear_probability': 0.33,
            'neutral_probability': 0.34,
            'uncertainty': 1.0,
            'error': str(e)
        }

def create_future_mapping_summary(candles: List[Dict], ema: float = None, 
                                 current_price: float = None, ai_label: Dict = None,
                                 market_phase: str = None) -> Dict:
    """
    Creates comprehensive summary of future mapping analysis
    
    Args:
        candles: Candle data
        ema: EMA reference
        current_price: Current price
        ai_label: AI pattern analysis
        market_phase: Market phase
        
    Returns:
        Comprehensive analysis summary
    """
    try:
        # Get scoring result
        score_result = score_from_future_mapping(candles, ema, current_price, ai_label, market_phase)
        
        # Get probability distribution
        prob_dist = get_scenario_probability_distribution(candles, ema)
        
        # Extract key metrics
        scenario = score_result['scenario']
        confidence = score_result['confidence']
        adjustment = score_result['adjustment']
        
        # Risk assessment
        risk_level = 'low'
        if abs(adjustment) > 0.05:
            risk_level = 'medium'
        if abs(adjustment) > 0.08:
            risk_level = 'high'
        
        return {
            'module': 'Future Scenario Mapping (Module 4)',
            'scenario': scenario,
            'adjustment': adjustment,
            'confidence': confidence,
            'risk_level': risk_level,
            'probability_distribution': prob_dist,
            'detailed_analysis': score_result,
            'analysis_quality': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low'
        }
        
    except Exception as e:
        return {
            'module': 'Future Scenario Mapping (Module 4)',
            'scenario': 'neutral',
            'adjustment': 0.0,
            'confidence': 0.0,
            'error': str(e)
        }