"""
CLIP Integration with simulate_trader_decision_advanced()
Enhances trader AI decision engine with visual pattern recognition
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CLIPTraderIntegration:
    """Integrates CLIP visual analysis with trader decision engine"""
    
    def __init__(self):
        self.vision_ai_dir = Path("vision_ai")
        self.charts_dir = Path("charts")
        self.enabled = self._check_clip_availability()
        
        # CLIP prediction confidence thresholds
        self.min_confidence = 0.3
        self.high_confidence = 0.7
        
        # Score adjustments based on CLIP predictions
        self.clip_score_adjustments = {
            # Bullish patterns
            "breakout-continuation": 0.15,
            "pullback-in-trend": 0.12,
            "support-bounce": 0.10,
            "trending-up": 0.08,
            "accumulation": 0.06,
            
            # Bearish patterns
            "fakeout": -0.20,
            "failed-breakout": -0.15,
            "exhaustion": -0.12,
            "distribution": -0.10,
            "resistance-rejection": -0.08,
            
            # Neutral patterns
            "consolidation": 0.00,
            "range-bound": 0.00,
            
            # Uncertain patterns
            "unknown": 0.00
        }
        
        # Decision confidence modifiers
        self.confidence_multipliers = {
            "high": 1.5,    # >0.7 confidence
            "medium": 1.0,  # 0.5-0.7 confidence  
            "low": 0.5      # 0.3-0.5 confidence
        }
    
    def _check_clip_availability(self) -> bool:
        """Check if CLIP components are available"""
        try:
            # Check if CLIP prediction module exists
            clip_predictor = self.vision_ai_dir / "predict_clip_similarity.py"
            embeddings_file = Path("data/clip/clip_embeddings.pt")
            
            if clip_predictor.exists():
                logger.info("CLIP predictor module found")
                if embeddings_file.exists():
                    logger.info("CLIP embeddings found - full functionality available")
                    return True
                else:
                    logger.info("CLIP predictor available but no embeddings - zero-shot only")
                    return True
            else:
                logger.warning("CLIP predictor module not found")
                return False
                
        except Exception as e:
            logger.error(f"Error checking CLIP availability: {e}")
            return False
    
    def get_chart_path_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Get most recent chart path for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Path to chart image or None
        """
        try:
            # Look for recent charts in charts directory
            pattern = f"{symbol}_*.png"
            chart_files = list(self.charts_dir.glob(pattern))
            
            if chart_files:
                # Return most recent chart (by filename timestamp)
                latest_chart = sorted(chart_files, reverse=True)[0]
                return str(latest_chart)
            else:
                # Check in exports directory as fallback
                exports_dir = Path("exports")
                if exports_dir.exists():
                    export_charts = list(exports_dir.glob(pattern))
                    if export_charts:
                        latest_export = sorted(export_charts, reverse=True)[0]
                        return str(latest_export)
                
                return None
                
        except Exception as e:
            logger.error(f"Error finding chart for {symbol}: {e}")
            return None
    
    def analyze_chart_with_clip(self, symbol: str, chart_path: Optional[str] = None) -> Dict:
        """
        Analyze chart using CLIP and return vision insights
        
        Args:
            symbol: Trading symbol
            chart_path: Path to chart (auto-detected if None)
            
        Returns:
            CLIP analysis results
        """
        try:
            if not self.enabled:
                return {"enabled": False, "error": "CLIP not available"}
            
            # Get chart path
            if chart_path is None:
                chart_path = self.get_chart_path_for_symbol(symbol)
            
            if chart_path is None:
                return {"enabled": True, "error": "No chart found", "symbol": symbol}
            
            # Import and use CLIP predictor
            from vision_ai.predict_clip_similarity import predict_chart_pattern
            
            # Try similarity-based prediction first (if embeddings available)
            results = predict_chart_pattern(chart_path, method="similarity")
            
            # Fallback to zero-shot if similarity fails
            if not results.get("success"):
                results = predict_chart_pattern(chart_path, method="zero_shot")
            
            if results.get("success"):
                # Extract key information
                analysis = {
                    "enabled": True,
                    "symbol": symbol,
                    "chart_path": chart_path,
                    "phase": results.get("phase", "unknown"),
                    "setup": results.get("setup", "unknown"),
                    "confidence": results.get("confidence", 0),
                    "phase_description": results.get("phase_description", ""),
                    "method": "similarity" if "similar_examples" in results else "zero_shot"
                }
                
                # Add top predictions if available
                if "all_predictions" in results:
                    analysis["top_predictions"] = results["all_predictions"][:5]
                
                return analysis
            else:
                return {
                    "enabled": True,
                    "error": results.get("error", "Analysis failed"),
                    "symbol": symbol,
                    "chart_path": chart_path
                }
                
        except Exception as e:
            logger.error(f"CLIP analysis error for {symbol}: {e}")
            return {
                "enabled": True,
                "error": str(e),
                "symbol": symbol
            }
    
    def calculate_clip_score_adjustment(self, clip_analysis: Dict) -> Dict:
        """
        Calculate score adjustments based on CLIP analysis
        
        Args:
            clip_analysis: CLIP analysis results
            
        Returns:
            Score adjustment details
        """
        try:
            if not clip_analysis.get("enabled") or "error" in clip_analysis:
                return {"adjustment": 0.0, "reason": "CLIP not available or failed"}
            
            phase = clip_analysis.get("phase", "unknown")
            setup = clip_analysis.get("setup", "unknown")
            confidence = clip_analysis.get("confidence", 0)
            
            # Base adjustment from phase/setup
            base_adjustment = 0.0
            
            # Check phase adjustment
            if phase in self.clip_score_adjustments:
                base_adjustment += self.clip_score_adjustments[phase]
            
            # Check setup adjustment (if different from phase)
            if setup != phase and setup in self.clip_score_adjustments:
                base_adjustment += self.clip_score_adjustments[setup] * 0.5
            
            # Apply confidence multiplier
            confidence_level = "low"
            if confidence >= self.high_confidence:
                confidence_level = "high"
            elif confidence >= 0.5:
                confidence_level = "medium"
            
            confidence_multiplier = self.confidence_multipliers[confidence_level]
            final_adjustment = base_adjustment * confidence_multiplier
            
            # Cap adjustments
            final_adjustment = max(-0.25, min(0.25, final_adjustment))
            
            return {
                "adjustment": final_adjustment,
                "base_adjustment": base_adjustment,
                "confidence_multiplier": confidence_multiplier,
                "confidence_level": confidence_level,
                "phase": phase,
                "setup": setup,
                "confidence": confidence,
                "reason": f"CLIP detected {phase} pattern with {confidence_level} confidence"
            }
            
        except Exception as e:
            logger.error(f"Error calculating CLIP adjustment: {e}")
            return {"adjustment": 0.0, "reason": f"Calculation error: {e}"}
    
    def enhance_trader_decision(self, symbol: str, base_analysis: Dict, chart_path: Optional[str] = None) -> Dict:
        """
        Enhance trader decision with CLIP visual analysis
        
        Args:
            symbol: Trading symbol
            base_analysis: Base trader analysis results
            chart_path: Optional chart path
            
        Returns:
            Enhanced analysis with CLIP integration
        """
        try:
            # Get CLIP analysis
            clip_analysis = self.analyze_chart_with_clip(symbol, chart_path)
            
            # Calculate score adjustment
            score_adjustment_info = self.calculate_clip_score_adjustment(clip_analysis)
            
            # Apply adjustment to base score
            base_score = base_analysis.get("final_score", 0)
            clip_adjustment = score_adjustment_info.get("adjustment", 0)
            enhanced_score = base_score + clip_adjustment
            
            # Create enhanced analysis
            enhanced_analysis = base_analysis.copy()
            enhanced_analysis.update({
                "final_score": enhanced_score,
                "clip_enhanced": True,
                "clip_analysis": clip_analysis,
                "clip_adjustment": clip_adjustment,
                "clip_adjustment_info": score_adjustment_info,
                "base_score_before_clip": base_score
            })
            
            # Add CLIP insights to decision reasons
            if "decision_reasons" not in enhanced_analysis:
                enhanced_analysis["decision_reasons"] = []
            
            if clip_adjustment != 0:
                clip_reason = f"Visual pattern analysis: {score_adjustment_info.get('reason', 'CLIP adjustment applied')}"
                enhanced_analysis["decision_reasons"].append(clip_reason)
            
            # Update score breakdown
            if "score_breakdown" not in enhanced_analysis:
                enhanced_analysis["score_breakdown"] = {}
            
            enhanced_analysis["score_breakdown"]["clip_visual"] = clip_adjustment
            
            logger.info(f"Enhanced {symbol}: {base_score:.3f} → {enhanced_score:.3f} (CLIP: {clip_adjustment:+.3f})")
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Error enhancing trader decision for {symbol}: {e}")
            # Return original analysis if enhancement fails
            return base_analysis


def integrate_clip_with_trader_analysis(symbol: str, trader_analysis: Dict, chart_path: Optional[str] = None) -> Dict:
    """
    Convenience function for CLIP integration
    
    Args:
        symbol: Trading symbol
        trader_analysis: Original trader analysis
        chart_path: Optional chart path
        
    Returns:
        Enhanced analysis with CLIP integration
    """
    integrator = CLIPTraderIntegration()
    return integrator.enhance_trader_decision(symbol, trader_analysis, chart_path)


def main():
    """Demo CLIP integration functionality"""
    print("🎯 CLIP Trader Integration Demo")
    print("=" * 50)
    
    try:
        integrator = CLIPTraderIntegration()
        
        print(f"CLIP Integration Status: {'✅ Enabled' if integrator.enabled else '❌ Disabled'}")
        
        # Demo with mock trader analysis
        mock_analysis = {
            "symbol": "BTCUSDT",
            "final_score": 0.65,
            "decision": "consider_entry",
            "confidence": 0.58,
            "decision_reasons": ["Support level holding", "Volume increasing"],
            "score_breakdown": {
                "trend_strength": 0.70,
                "pullback_quality": 0.60
            }
        }
        
        print(f"\n📊 Original Analysis:")
        print(f"   Score: {mock_analysis['final_score']}")
        print(f"   Decision: {mock_analysis['decision']}")
        
        # Enhance with CLIP
        enhanced = integrator.enhance_trader_decision("BTCUSDT", mock_analysis)
        
        print(f"\n🎯 Enhanced Analysis:")
        print(f"   Score: {enhanced['final_score']:.3f} (change: {enhanced.get('clip_adjustment', 0):+.3f})")
        
        if enhanced.get("clip_enhanced"):
            clip_info = enhanced.get("clip_adjustment_info", {})
            print(f"   CLIP Pattern: {clip_info.get('phase', 'unknown')}")
            print(f"   CLIP Confidence: {clip_info.get('confidence', 0):.3f}")
            print(f"   Adjustment Reason: {clip_info.get('reason', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main()