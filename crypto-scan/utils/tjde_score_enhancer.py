"""
TJDE Score Enhancement System
Addresses scoring cap at ~0.66 and implements nonlinear boosters for strong signals
"""

import math
from typing import Dict, Any, Optional

class TJDEScoreEnhancer:
    """
    Enhances TJDE scoring system to break through ~0.66 ceiling
    Implements nonlinear boosters for exceptional signals
    """
    
    def __init__(self):
        self.max_boost = 0.35  # Maximum score boost possible
        self.elite_threshold = 0.60  # Threshold for elite signal detection
        
    def enhance_tjde_score(self, base_score: float, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply nonlinear enhancement to break scoring ceiling
        
        Args:
            base_score: Original TJDE score (0.0-1.0)
            signal_data: Complete signal analysis data
            
        Returns:
            Enhanced score with breakdown
        """
        try:
            if base_score < 0.1:
                return {"enhanced_score": base_score, "boost": 0.0, "reason": "weak_signal"}
            
            # Extract signal components
            ai_score = signal_data.get("ai_eye_score", 0.0)
            htf_score = signal_data.get("htf_overlay_score", 0.0)
            volume_surge = signal_data.get("volume_behavior_score", 0.0)
            momentum = signal_data.get("trend_strength", 0.0)
            confidence = signal_data.get("ai_confidence", 0.0)
            
            # Detect exceptional signal patterns
            boost_factors = self._calculate_boost_factors(
                ai_score, htf_score, volume_surge, momentum, confidence
            )
            
            # Apply nonlinear boost
            total_boost = self._compute_nonlinear_boost(boost_factors)
            enhanced_score = min(1.0, base_score + total_boost)
            
            return {
                "enhanced_score": enhanced_score,
                "base_score": base_score,
                "boost": total_boost,
                "boost_factors": boost_factors,
                "reason": self._get_boost_reason(boost_factors)
            }
            
        except Exception as e:
            print(f"[SCORE ENHANCER ERROR] {e}")
            return {"enhanced_score": base_score, "boost": 0.0, "reason": "error"}
    
    def _calculate_boost_factors(self, ai_score: float, htf_score: float, 
                                volume_surge: float, momentum: float, 
                                confidence: float) -> Dict[str, float]:
        """Calculate individual boost factors for different signal types"""
        
        factors = {}
        
        # AI-EYE Visual Pattern Boost (0.0-0.15)
        if ai_score > 0.08 and confidence > 0.7:
            factors["ai_visual"] = min(0.15, ai_score * confidence * 1.5)
        
        # HTF Macro Alignment Boost (0.0-0.12)
        if htf_score > 0.05:
            factors["macro_alignment"] = min(0.12, htf_score * 2.0)
        
        # Volume Explosion Boost (0.0-0.10)
        if volume_surge > 0.7:
            factors["volume_explosion"] = min(0.10, (volume_surge - 0.7) * 0.33)
        
        # Momentum Breakout Boost (0.0-0.08)
        if momentum > 0.8:
            factors["momentum_breakout"] = min(0.08, (momentum - 0.8) * 0.4)
        
        # Combined Signal Synergy Boost (0.0-0.05)
        # Count both strong signals (>0.5) and moderate AI signals (>0.05)
        strong_signals = sum(1 for score in [volume_surge, momentum] if score > 0.5)
        moderate_ai_signals = sum(1 for score in [ai_score, htf_score] if score > 0.05)
        
        # Synergy if 2+ strong signals OR 1 strong + 2 moderate AI signals
        if strong_signals >= 2 or (strong_signals >= 1 and moderate_ai_signals >= 2):
            total_synergy_strength = strong_signals + moderate_ai_signals * 0.5
            factors["synergy"] = min(0.05, total_synergy_strength * 0.015)
        
        return factors
    
    def _compute_nonlinear_boost(self, boost_factors: Dict[str, float]) -> float:
        """Apply nonlinear transformation to boost factors"""
        
        total_linear = sum(boost_factors.values())
        
        if total_linear == 0:
            return 0.0
        
        # Nonlinear amplification for exceptional signals
        if total_linear > 0.25:
            # Exponential boost for truly exceptional signals
            nonlinear_factor = 1.0 + math.log(total_linear * 4) * 0.1
        elif total_linear > 0.15:
            # Moderate boost for strong signals
            nonlinear_factor = 1.0 + (total_linear - 0.15) * 0.5
        else:
            # Linear for normal signals
            nonlinear_factor = 1.0
        
        boosted = total_linear * nonlinear_factor
        return min(self.max_boost, boosted)
    
    def _get_boost_reason(self, boost_factors: Dict[str, float]) -> str:
        """Generate human-readable reason for score boost"""
        
        if not boost_factors:
            return "no_enhancement"
        
        strongest = max(boost_factors.items(), key=lambda x: x[1])
        reason_map = {
            "ai_visual": "exceptional_visual_pattern",
            "macro_alignment": "strong_macro_alignment", 
            "volume_explosion": "volume_surge_detected",
            "momentum_breakout": "momentum_breakout",
            "synergy": "multi_signal_convergence"
        }
        
        return reason_map.get(strongest[0], "enhanced_signal")
    
    def should_apply_enhancement(self, base_score: float, signal_data: Dict[str, Any]) -> bool:
        """Determine if score enhancement should be applied"""
        
        # Don't enhance very weak signals
        if base_score < 0.3:
            return False
        
        # Always enhance signals above elite threshold
        if base_score >= self.elite_threshold:
            return True
        
        # Enhance if multiple strong components present
        strong_components = 0
        for component in ["ai_eye_score", "htf_overlay_score", "volume_behavior_score", "trend_strength"]:
            if signal_data.get(component, 0.0) > 0.6:
                strong_components += 1
        
        return strong_components >= 2

# Global enhancer instance
score_enhancer = TJDEScoreEnhancer()

def enhance_tjde_score(base_score: float, signal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for score enhancement"""
    return score_enhancer.enhance_tjde_score(base_score, signal_data)

def should_apply_enhancement(base_score: float, signal_data: Dict[str, Any]) -> bool:
    """Convenience function for enhancement check"""
    return score_enhancer.should_apply_enhancement(base_score, signal_data)