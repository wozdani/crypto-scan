"""
CLIP Visual Predictor for Chart Pattern Recognition
Integruje model CLIP (OpenAI) do analizy wykresów cenowych
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Lista etykiet CLIP do rozpoznawania wzorców
CLIP_LABELS = [
    "breakout-continuation",
    "pullback-in-trend", 
    "range-accumulation",
    "trend-reversal",
    "consolidation",
    "fakeout",
    "volume-backed breakout",
    "exhaustion pattern",
    "no-trend noise"
]

class CLIPChartPredictor:
    """CLIP-based chart pattern predictor using OpenAI ViT-B/32 model"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP predictor
        
        Args:
            model_name: CLIP model name from HuggingFace
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model initialization
        self.model = None
        self.processor = None
        self.initialized = False
        
        # Initialize model on first use
        self._lazy_init()
    
    def _lazy_init(self):
        """Lazy initialization of CLIP model"""
        try:
            if not self.initialized:
                logger.info(f"Initializing CLIP model: {self.model_name}")
                
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                
                self.model.eval()
                self.initialized = True
                
                logger.info(f"CLIP model initialized successfully on {self.device}")
                
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            self.initialized = False
    
    def predict_chart_clip(self, image_path: str) -> Tuple[str, float]:
        """
        Predict chart pattern using CLIP
        
        Args:
            image_path: Path to chart image
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        try:
            if not self.initialized:
                self._lazy_init()
            
            if not self.initialized:
                return "no-trend noise", 0.0
            
            if not os.path.exists(image_path):
                logger.warning(f"Chart image not found: {image_path}")
                return "no-trend noise", 0.0
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process inputs
            inputs = self.processor(text=CLIP_LABELS, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Get top prediction
            top_index = probs.argmax()
            predicted_label = CLIP_LABELS[top_index]
            confidence = float(probs[top_index])
            
            logger.info(f"CLIP prediction for {Path(image_path).name}: {predicted_label} ({confidence:.3f})")
            
            return predicted_label, confidence
            
        except Exception as e:
            logger.error(f"CLIP prediction error for {image_path}: {e}")
            return "no-trend noise", 0.0
    
    def predict_with_details(self, image_path: str) -> Dict:
        """
        Predict chart pattern with detailed results
        
        Args:
            image_path: Path to chart image
            
        Returns:
            Detailed prediction results
        """
        try:
            predicted_label, confidence = self.predict_chart_clip(image_path)
            
            # Get all predictions for analysis
            if not self.initialized:
                all_predictions = []
            else:
                image = Image.open(image_path).convert('RGB')
                
                inputs = self.processor(text=CLIP_LABELS, images=image, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                
                all_predictions = [
                    {"label": label, "confidence": float(prob)}
                    for label, prob in zip(CLIP_LABELS, probs)
                ]
                all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "success": True,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "image_path": image_path,
                "all_predictions": all_predictions[:5],  # Top 5
                "model_info": {
                    "model_name": self.model_name,
                    "device": self.device
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path,
                "predicted_label": "no-trend noise",
                "confidence": 0.0
            }
    
    def find_chart_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Find most recent chart for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Path to chart or None
        """
        # Check multiple possible chart locations
        chart_locations = [
            Path("charts"),
            Path("exports"),
            Path("data/charts"),
            Path("vision_ai/charts")
        ]
        
        for chart_dir in chart_locations:
            if chart_dir.exists():
                # Look for recent charts with symbol
                pattern = f"{symbol}_*.png"
                chart_files = list(chart_dir.glob(pattern))
                
                if chart_files:
                    # Return most recent (by filename timestamp)
                    latest_chart = sorted(chart_files, reverse=True)[0]
                    return str(latest_chart)
        
        return None


# Global predictor instance (lazy-loaded)
_global_predictor = None

def get_clip_predictor() -> CLIPChartPredictor:
    """Get global CLIP predictor instance"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = CLIPChartPredictor()
    return _global_predictor

def predict_chart_clip(image_path: str) -> Tuple[str, float]:
    """
    Convenience function for CLIP prediction
    
    Args:
        image_path: Path to chart image
        
    Returns:
        Tuple of (predicted_label, confidence)
    """
    predictor = get_clip_predictor()
    return predictor.predict_chart_clip(image_path)

def predict_chart_for_symbol(symbol: str) -> Dict:
    """
    Predict chart pattern for trading symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Prediction results
    """
    predictor = get_clip_predictor()
    
    # Find chart for symbol
    chart_path = predictor.find_chart_for_symbol(symbol)
    
    if chart_path:
        return predictor.predict_with_details(chart_path)
    else:
        return {
            "success": False,
            "error": f"No chart found for {symbol}",
            "predicted_label": "no-trend noise",
            "confidence": 0.0,
            "symbol": symbol
        }

def test_clip_prediction():
    """Test CLIP prediction functionality"""
    print("Testing CLIP Chart Prediction")
    print("=" * 40)
    
    predictor = CLIPChartPredictor()
    
    if not predictor.initialized:
        print("❌ CLIP model failed to initialize")
        return False
    
    # Test with available charts
    charts_dir = Path("data/training/charts")
    if charts_dir.exists():
        chart_files = list(charts_dir.glob("*.png"))
        
        if chart_files:
            test_chart = str(chart_files[0])
            print(f"Testing with: {Path(test_chart).name}")
            
            result = predictor.predict_with_details(test_chart)
            
            if result["success"]:
                print(f"✅ Prediction: {result['predicted_label']}")
                print(f"✅ Confidence: {result['confidence']:.3f}")
                
                print("\nTop predictions:")
                for i, pred in enumerate(result["all_predictions"][:3], 1):
                    print(f"  {i}. {pred['label']}: {pred['confidence']:.3f}")
                
                return True
            else:
                print(f"❌ Prediction failed: {result['error']}")
                return False
        else:
            print("❌ No test charts found")
            return False
    else:
        print("❌ Charts directory not found")
        return False

if __name__ == "__main__":
    test_clip_prediction()