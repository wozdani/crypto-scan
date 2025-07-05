"""
CLIP Predictor - Visual Pattern Recognition
Analyzes TradingView charts using CLIP model to identify trading patterns
"""

import torch
from PIL import Image
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

# Trading pattern labels for CLIP classification
TRADING_LABELS = [
    "pullback in uptrend",
    "breakout pattern", 
    "consolidation range",
    "parabolic pump movement",
    "bull trap reversal",
    "exhaustion pattern",
    "accumulation phase",
    "support retest",
    "early trend formation",
    "choppy sideways movement"
]

class CLIPPredictor:
    """CLIP-based trading pattern predictor"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_loaded = False
        
    def _load_model(self):
        """Load CLIP model on first use"""
        if self.model_loaded:
            return True
            
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info("[CLIP] Loading CLIP model for pattern recognition...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model_loaded = True
            logger.info("[CLIP] ✅ Model loaded successfully")
            return True
            
        except ImportError:
            logger.warning("[CLIP] ❌ transformers library not available - install with: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"[CLIP] ❌ Failed to load model: {e}")
            return False
    
    def predict_pattern(self, image_path: str) -> Tuple[str, float]:
        """
        Predict trading pattern from chart image
        
        Args:
            image_path: Path to TradingView chart image
            
        Returns:
            Tuple of (pattern_label, confidence_score)
        """
        try:
            # Load model if needed
            if not self._load_model():
                return "unknown_pattern", 0.0
                
            # Check if image exists
            if not os.path.exists(image_path):
                logger.warning(f"[CLIP] ❌ Image not found: {image_path}")
                return "image_not_found", 0.0
                
            # Load and process image
            logger.info(f"[CLIP] 🔍 Analyzing chart: {os.path.basename(image_path)}")
            image = Image.open(image_path)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Process with CLIP
            inputs = self.processor(
                text=TRADING_LABELS, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
                
            # Find best match
            best_idx = torch.argmax(probs).item()
            best_label = TRADING_LABELS[best_idx]
            confidence = round(probs[best_idx].item(), 4)
            
            logger.info(f"[CLIP] 📊 Pattern detected: {best_label} (confidence: {confidence})")
            
            # Convert to simplified label format
            simplified_label = self._simplify_label(best_label)
            
            return simplified_label, confidence
            
        except Exception as e:
            logger.error(f"[CLIP] ❌ Prediction failed: {e}")
            return "prediction_error", 0.0
    
    def _simplify_label(self, full_label: str) -> str:
        """Convert full CLIP label to simplified trading pattern"""
        label_mapping = {
            "pullback in uptrend": "pullback",
            "breakout pattern": "breakout", 
            "consolidation range": "range",
            "parabolic pump movement": "parabolic",
            "bull trap reversal": "bull_trap",
            "exhaustion pattern": "exhaustion",
            "accumulation phase": "accumulation",
            "support retest": "retest",
            "early trend formation": "early_trend",
            "choppy sideways movement": "chaos"
        }
        
        return label_mapping.get(full_label, "unknown")

# Global predictor instance
_clip_predictor = CLIPPredictor()

def get_clip_prediction(image_path: str) -> Tuple[str, float]:
    """
    Main function to get CLIP prediction for trading chart
    
    Args:
        image_path: Path to TradingView chart image
        
    Returns:
        Tuple of (pattern_label, confidence_score)
    """
    return _clip_predictor.predict_pattern(image_path)

def test_clip_prediction():
    """Test CLIP prediction functionality"""
    # Test with dummy data
    result = get_clip_prediction("test_chart.png")
    print(f"Test result: {result}")

if __name__ == "__main__":
    test_clip_prediction()