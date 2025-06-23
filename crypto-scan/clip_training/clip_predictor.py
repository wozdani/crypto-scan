"""
CLIP Predictor for Real-time Chart Analysis
Wykorzystuje wytrenowany model CLIP do predykcji faz rynku i setupów
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import os

from .clip_model import get_clip_model

logger = logging.getLogger(__name__)

class CLIPChartPredictor:
    """CLIP predictor for chart pattern analysis"""
    
    def __init__(self):
        self.model = get_clip_model()
        
        # Predefined market labels for prediction
        self.market_labels = [
            "bullish breakout continuation",
            "bearish trend reversal",
            "sideways consolidation", 
            "pullback in uptrend",
            "pullback in downtrend",
            "fakeout pattern",
            "volume-backed breakout",
            "exhaustion pattern",
            "accumulation phase",
            "distribution phase",
            "range-bound trading",
            "momentum building",
            "trend exhaustion",
            "support bounce",
            "resistance rejection"
        ]
        
        # Mapping for scoring integration
        self.scoring_modifiers = {
            "breakout": 0.08,
            "bullish": 0.06,
            "continuation": 0.05,
            "volume-backed": 0.07,
            "accumulation": 0.04,
            "momentum": 0.05,
            "support": 0.03,
            
            "fakeout": -0.10,
            "bearish": -0.06,
            "reversal": -0.05,
            "exhaustion": -0.08,
            "distribution": -0.06,
            "resistance": -0.04
        }
        
        logger.info("CLIP Chart Predictor initialized")
    
    def predict_clip_labels(self, chart_image_path: str) -> Dict:
        """
        Predict market labels for chart image
        
        Args:
            chart_image_path: Path to chart image
            
        Returns:
            Prediction results with labels and confidence
        """
        if not self.model.is_initialized:
            return {
                "success": False,
                "error": "CLIP model not initialized",
                "labels": [],
                "confidence": 0.0
            }
        
        if not os.path.exists(chart_image_path):
            return {
                "success": False,
                "error": f"Chart image not found: {chart_image_path}",
                "labels": [],
                "confidence": 0.0
            }
        
        try:
            # Get predictions from model
            prediction = self.model.predict(chart_image_path, self.market_labels)
            
            if not prediction.get("success"):
                return {
                    "success": False,
                    "error": prediction.get("error", "Prediction failed"),
                    "labels": [],
                    "confidence": 0.0
                }
            
            # Extract key information
            predicted_text = prediction["predicted_text"]
            confidence = prediction["confidence"]
            all_predictions = prediction.get("all_predictions", [])
            
            # Extract individual labels from predicted text
            labels = self._extract_labels_from_text(predicted_text)
            
            # Get top predictions with high confidence
            high_confidence_labels = []
            for pred in all_predictions:
                if pred["confidence"] > 0.3:  # Confidence threshold
                    pred_labels = self._extract_labels_from_text(pred["text"])
                    high_confidence_labels.extend(pred_labels)
            
            # Remove duplicates while preserving order
            all_labels = list(dict.fromkeys(labels + high_confidence_labels))
            
            return {
                "success": True,
                "primary_prediction": predicted_text,
                "labels": all_labels,
                "confidence": confidence,
                "all_predictions": all_predictions[:5],  # Top 5 only
                "training_step": prediction.get("training_step", 0)
            }
            
        except Exception as e:
            logger.error(f"Error predicting labels for {chart_image_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "labels": [],
                "confidence": 0.0
            }
    
    def _extract_labels_from_text(self, text: str) -> List[str]:
        """Extract meaningful labels from prediction text"""
        text_lower = text.lower()
        found_labels = []
        
        # Look for key patterns
        key_patterns = [
            "breakout", "pullback", "fakeout", "reversal", "continuation",
            "bullish", "bearish", "sideways", "consolidation", "accumulation",
            "distribution", "exhaustion", "momentum", "volume-backed",
            "support", "resistance", "range-bound"
        ]
        
        for pattern in key_patterns:
            if pattern in text_lower:
                found_labels.append(pattern)
        
        return found_labels
    
    def calculate_clip_modifier(self, labels: List[str], confidence: float) -> Dict:
        """
        Calculate scoring modifier based on CLIP predictions
        
        Args:
            labels: Predicted labels
            confidence: Prediction confidence
            
        Returns:
            Scoring modifier information
        """
        if confidence < 0.3:  # Too low confidence
            return {
                "modifier": 0.0,
                "reason": f"CLIP confidence too low ({confidence:.3f})",
                "labels_found": labels
            }
        
        # Calculate base modifier
        base_modifier = 0.0
        applied_labels = []
        
        for label in labels:
            if label in self.scoring_modifiers:
                modifier_value = self.scoring_modifiers[label]
                base_modifier += modifier_value
                applied_labels.append(f"{label}({modifier_value:+.3f})")
        
        # Apply confidence weighting
        confidence_weight = min(confidence * 1.2, 1.0)  # Boost confidence slightly
        final_modifier = base_modifier * confidence_weight
        
        # Cap modifier to reasonable range
        final_modifier = max(-0.15, min(0.15, final_modifier))
        
        return {
            "modifier": final_modifier,
            "base_modifier": base_modifier,
            "confidence_weight": confidence_weight,
            "confidence": confidence,
            "labels_found": labels,
            "applied_labels": applied_labels,
            "reason": f"CLIP detected {', '.join(labels)} with {confidence:.3f} confidence"
        }
    
    def find_chart_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Find most recent chart for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Path to chart or None
        """
        # Check multiple chart locations
        chart_locations = [
            Path("charts"),
            Path("exports"),
            Path("data/charts"),
            Path("data/vision_ai/train_data/charts")
        ]
        
        for chart_dir in chart_locations:
            if chart_dir.exists():
                # Look for charts with symbol
                pattern = f"{symbol}_*.png"
                chart_files = list(chart_dir.glob(pattern))
                
                if chart_files:
                    # Return most recent (by filename timestamp)
                    latest_chart = sorted(chart_files, reverse=True)[0]
                    return str(latest_chart)
        
        return None
    
    def predict_for_symbol(self, symbol: str) -> Dict:
        """
        Predict labels for symbol's most recent chart
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Prediction results
        """
        chart_path = self.find_chart_for_symbol(symbol)
        
        if not chart_path:
            return {
                "success": False,
                "error": f"No chart found for {symbol}",
                "symbol": symbol,
                "labels": [],
                "confidence": 0.0
            }
        
        result = self.predict_clip_labels(chart_path)
        result["symbol"] = symbol
        result["chart_path"] = chart_path
        
        return result


# Global predictor instance
_global_predictor = None

def get_clip_predictor() -> CLIPChartPredictor:
    """Get global CLIP predictor instance"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = CLIPChartPredictor()
    return _global_predictor

def predict_clip_labels(chart_image_path: str) -> Dict:
    """
    Convenience function for CLIP prediction
    
    Args:
        chart_image_path: Path to chart image
        
    Returns:
        Prediction results
    """
    predictor = get_clip_predictor()
    return predictor.predict_clip_labels(chart_image_path)

def main():
    """Test CLIP predictor functionality"""
    print("Testing CLIP Chart Predictor")
    print("=" * 40)
    
    predictor = CLIPChartPredictor()
    
    if predictor.model.is_initialized:
        print("✅ CLIP predictor initialized")
        print(f"   Model training step: {predictor.model.training_step}")
        print(f"   Available labels: {len(predictor.market_labels)}")
        
        # Test with available charts
        from pathlib import Path
        test_charts = list(Path("data/training/charts").glob("*.png"))
        
        if test_charts:
            test_chart = str(test_charts[0])
            print(f"\nTesting prediction with: {Path(test_chart).name}")
            
            result = predictor.predict_clip_labels(test_chart)
            
            if result["success"]:
                print(f"   Primary prediction: {result['primary_prediction']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Labels found: {result['labels']}")
                
                # Test scoring modifier
                modifier_info = predictor.calculate_clip_modifier(
                    result['labels'], 
                    result['confidence']
                )
                print(f"   Score modifier: {modifier_info['modifier']:+.3f}")
                print(f"   Modifier reason: {modifier_info['reason']}")
            else:
                print(f"   ❌ Prediction failed: {result['error']}")
        else:
            print("\n⚠️ No test charts available")
    else:
        print("❌ CLIP predictor initialization failed")

if __name__ == "__main__":
    main()