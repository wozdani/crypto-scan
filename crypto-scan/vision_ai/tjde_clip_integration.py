"""
TJDE CLIP Integration
Integruje predykcje CLIP z systemem simulate_trader_decision_advanced
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TJDEClipIntegration:
    """Integrates CLIP predictions with TJDE decision engine"""
    
    def __init__(self):
        # Score modifiers based on CLIP predictions
        self.clip_score_modifiers = {
            "breakout-continuation": 0.08,     # Strong bullish signal
            "volume-backed breakout": 0.10,    # Very strong signal
            "pullback-in-trend": 0.06,         # Good entry opportunity
            "range-accumulation": 0.04,        # Potential setup building
            "consolidation": 0.02,             # Neutral/slight positive
            "trend-reversal": -0.05,           # Caution signal
            "exhaustion pattern": -0.08,       # Bearish signal
            "fakeout": -0.12,                  # Strong bearish signal
            "no-trend noise": 0.00             # No adjustment
        }
        
        # Confidence thresholds
        self.min_confidence = 0.3   # Minimum confidence to apply modifier
        self.high_confidence = 0.7  # High confidence threshold
        
        # Confidence multipliers
        self.confidence_multipliers = {
            "very_high": 1.5,    # >0.8 confidence
            "high": 1.2,         # 0.7-0.8 confidence
            "medium": 1.0,       # 0.5-0.7 confidence
            "low": 0.6           # 0.3-0.5 confidence
        }
    
    def get_confidence_level(self, confidence: float) -> str:
        """Determine confidence level category"""
        if confidence >= 0.8:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
    
    def calculate_clip_adjustment(self, clip_result: Dict) -> Dict:
        """
        Calculate score adjustment based on CLIP prediction
        
        Args:
            clip_result: CLIP prediction result
            
        Returns:
            Adjustment details
        """
        try:
            predicted_label = clip_result.get("predicted_label", "no-trend noise")
            confidence = clip_result.get("confidence", 0.0)
            
            # Skip if confidence too low
            if confidence < self.min_confidence:
                return {
                    "adjustment": 0.0,
                    "reason": f"CLIP confidence too low ({confidence:.3f} < {self.min_confidence})",
                    "confidence_level": "insufficient"
                }
            
            # Base adjustment from label
            base_adjustment = self.clip_score_modifiers.get(predicted_label, 0.0)
            
            # Apply confidence multiplier
            confidence_level = self.get_confidence_level(confidence)
            confidence_multiplier = self.confidence_multipliers[confidence_level]
            
            final_adjustment = base_adjustment * confidence_multiplier
            
            # Cap adjustments to reasonable range
            final_adjustment = max(-0.15, min(0.15, final_adjustment))
            
            return {
                "adjustment": final_adjustment,
                "base_adjustment": base_adjustment,
                "confidence_multiplier": confidence_multiplier,
                "confidence_level": confidence_level,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "reason": f"CLIP detected '{predicted_label}' with {confidence_level} confidence"
            }
            
        except Exception as e:
            logger.error(f"Error calculating CLIP adjustment: {e}")
            return {
                "adjustment": 0.0,
                "reason": f"CLIP adjustment calculation error: {e}",
                "confidence_level": "error"
            }
    
    def enhance_tjde_with_clip(self, symbol: str, tjde_result: Dict, chart_path: Optional[str] = None) -> Dict:
        """
        Enhance TJDE result with CLIP visual analysis
        
        Args:
            symbol: Trading symbol
            tjde_result: Original TJDE analysis result
            chart_path: Optional chart path (auto-detected if None)
            
        Returns:
            Enhanced TJDE result with CLIP integration
        """
        try:
            from vision_ai.clip_predictor import predict_chart_for_symbol, predict_chart_clip
            
            # Get CLIP prediction
            if chart_path and os.path.exists(chart_path):
                # Use provided chart path
                predicted_label, confidence = predict_chart_clip(chart_path)
                clip_result = {
                    "success": True,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "image_path": chart_path
                }
            else:
                # Auto-detect chart for symbol
                clip_result = predict_chart_for_symbol(symbol)
            
            # Calculate adjustment
            adjustment_info = self.calculate_clip_adjustment(clip_result)
            
            # Apply adjustment to TJDE score
            original_score = tjde_result.get("final_score", 0.0)
            clip_adjustment = adjustment_info.get("adjustment", 0.0)
            enhanced_score = original_score + clip_adjustment
            
            # Create enhanced result
            enhanced_result = tjde_result.copy()
            enhanced_result.update({
                "final_score": enhanced_score,
                "clip_enhanced": True,
                "clip_prediction": {
                    "label": clip_result.get("predicted_label", "no-trend noise"),
                    "confidence": clip_result.get("confidence", 0.0),
                    "success": clip_result.get("success", False)
                },
                "clip_adjustment": clip_adjustment,
                "clip_adjustment_info": adjustment_info,
                "original_score_before_clip": original_score
            })
            
            # Add CLIP insight to decision reasons
            if "decision_reasons" not in enhanced_result:
                enhanced_result["decision_reasons"] = []
            
            if clip_adjustment != 0:
                clip_reason = f"Visual analysis: {adjustment_info.get('reason', 'CLIP adjustment applied')}"
                enhanced_result["decision_reasons"].append(clip_reason)
            
            # Update score breakdown
            if "score_breakdown" not in enhanced_result:
                enhanced_result["score_breakdown"] = {}
            
            enhanced_result["score_breakdown"]["clip_visual"] = clip_adjustment
            
            logger.info(f"CLIP enhanced {symbol}: {original_score:.3f} → {enhanced_score:.3f} (adjustment: {clip_adjustment:+.3f})")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error enhancing TJDE with CLIP for {symbol}: {e}")
            # Return original result if enhancement fails
            return tjde_result
    
    def save_clip_prediction_to_results(self, symbol: str, clip_result: Dict, tjde_results_path: str = "data/tjde_results"):
        """
        Save CLIP prediction to TJDE results file
        
        Args:
            symbol: Trading symbol
            clip_result: CLIP prediction result
            tjde_results_path: Path to TJDE results directory
        """
        try:
            results_dir = Path(tjde_results_path)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Find latest TJDE results file for symbol
            pattern = f"*{symbol}*.json"
            result_files = list(results_dir.glob(pattern))
            
            if result_files:
                # Update most recent file
                latest_file = sorted(result_files, reverse=True)[0]
                
                with open(latest_file, 'r') as f:
                    tjde_data = json.load(f)
                
                # Add CLIP prediction
                tjde_data["clip_prediction"] = {
                    "label": clip_result.get("predicted_label", "no-trend noise"),
                    "confidence": clip_result.get("confidence", 0.0),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save updated data
                with open(latest_file, 'w') as f:
                    json.dump(tjde_data, f, indent=2)
                
                logger.info(f"Added CLIP prediction to {latest_file.name}")
            
        except Exception as e:
            logger.error(f"Error saving CLIP prediction for {symbol}: {e}")


def integrate_clip_with_tjde(symbol: str, tjde_result: Dict, chart_path: Optional[str] = None) -> Dict:
    """
    Convenience function for CLIP-TJDE integration
    
    Args:
        symbol: Trading symbol
        tjde_result: TJDE analysis result
        chart_path: Optional chart path
        
    Returns:
        Enhanced TJDE result with CLIP
    """
    integrator = TJDEClipIntegration()
    return integrator.enhance_tjde_with_clip(symbol, tjde_result, chart_path)

def test_tjde_clip_integration():
    """Test TJDE CLIP integration"""
    print("Testing TJDE CLIP Integration")
    print("=" * 40)
    
    # Mock TJDE result for testing
    mock_tjde = {
        "symbol": "BTCUSDT",
        "final_score": 0.65,
        "decision": "consider_entry",
        "confidence": 0.58,
        "decision_reasons": ["Support holding", "Volume increasing"],
        "score_breakdown": {
            "trend_strength": 0.70,
            "pullback_quality": 0.60
        }
    }
    
    integrator = TJDEClipIntegration()
    
    print(f"Original TJDE score: {mock_tjde['final_score']:.3f}")
    
    # Test enhancement
    enhanced = integrator.enhance_tjde_with_clip("BTCUSDT", mock_tjde)
    
    if enhanced.get("clip_enhanced"):
        print(f"Enhanced score: {enhanced['final_score']:.3f}")
        print(f"CLIP adjustment: {enhanced.get('clip_adjustment', 0):+.3f}")
        print(f"CLIP prediction: {enhanced['clip_prediction']['label']}")
        print(f"CLIP confidence: {enhanced['clip_prediction']['confidence']:.3f}")
        print("✅ Integration successful")
        return True
    else:
        print("⚠️ No CLIP enhancement applied")
        return False

if __name__ == "__main__":
    test_tjde_clip_integration()