# ai/clip_predictor.py

import os
import torch
from PIL import Image
import logging
from typing import Dict, Any, Optional

# Use transformers CLIP implementation for better compatibility
try:
    from transformers import CLIPModel, CLIPProcessor
    TRANSFORMERS_AVAILABLE = True
    print("[CLIP INIT] Using transformers CLIP implementation")
except ImportError:
    try:
        import clip
        TRANSFORMERS_AVAILABLE = False
        print("[CLIP INIT] Using OpenAI CLIP implementation")
    except ImportError:
        print("[CLIP ERROR] No CLIP implementation available")
        TRANSFORMERS_AVAILABLE = None

logger = logging.getLogger(__name__)

class CLIPPredictor:
    """CLIP-based chart pattern predictor with enhanced debugging"""
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.preprocess = None
        
        print(f"[CLIP INIT] Using device: {self.device}")
        
        if TRANSFORMERS_AVAILABLE is None:
            print("[CLIP INIT] ❌ No CLIP implementation available")
            return
            
        try:
            self._load_model()
            if self.model is not None:
                print("[CLIP INIT] ✅ Model loaded successfully")
            else:
                print("[CLIP INIT] ❌ Model failed to load")
        except Exception as e:
            print(f"[CLIP INIT ERROR] Failed to load model: {e}")
    
    def _load_model(self):
        """Load CLIP model with enhanced error handling"""
        try:
            if TRANSFORMERS_AVAILABLE:
                print("[CLIP MODEL] Loading openai/clip-vit-base-patch32...")
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
                
                if self.device == "cuda":
                    self.model = self.model.cuda()
                    print("[CLIP MODEL] Moved to CUDA")
                    
                self.model.eval()
                print("[CLIP MODEL] ✅ transformers CLIP ready")
            else:
                print("[CLIP MODEL] Loading ViT-B/32...")
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                
                self.label_texts = [
                    "breakout-continuation", "pullback-in-trend", "trend-reversal",
                    "range-accumulation", "consolidation", "fake-breakout", 
                    "volume-backed breakout", "exhaustion pattern", "no-trend noise"
                ]
                self.tokenized_labels = clip.tokenize(self.label_texts).to(self.device)
                print("[CLIP MODEL] ✅ OpenAI CLIP ready")
                
        except Exception as e:
            print(f"[CLIP MODEL ERROR] {e}")
            print("[CLIP MODEL] Setting model to None - CLIP predictions will return N/A")
            self.model = None
            self.processor = None
            self.preprocess = None

    def predict_chart_setup(self, image_path: str) -> Optional[Dict]:
        """
        Predict chart setup using CLIP with comprehensive debugging
        
        Args:
            image_path: Path to chart image
            
        Returns:
            Dictionary with label and confidence or None
        """
        print(f"[CLIP PREDICT] Starting prediction for: {image_path}")
        
        # Check if model is loaded
        if not self.model:
            print(f"[CLIP PREDICT] ❌ Model not loaded")
            return None
            
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"[CLIP PREDICT] ❌ Image file does not exist: {image_path}")
            return None
            
        try:
            print(f"[CLIP PREDICT] Loading image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            print(f"[CLIP PREDICT] Image loaded, size: {image.size}")
            
            if TRANSFORMERS_AVAILABLE and self.processor:
                return self._predict_with_transformers(image)
            elif self.preprocess:
                return self._predict_with_openai_clip(image)
            else:
                print(f"[CLIP PREDICT] ❌ No valid prediction method available")
                return None
                
        except Exception as e:
            print(f"[CLIP PREDICT ERROR] {e}")
            import traceback
            print(f"[CLIP PREDICT ERROR] Traceback: {traceback.format_exc()}")
            return None
    
    def _predict_with_transformers(self, image):
        """Predict using transformers CLIP"""
        candidate_labels = [
            "breakout-continuation", "pullback-in-trend", "range-accumulation",
            "trend-reversal", "consolidation", "fake-breakout", 
            "volume-backed breakout", "exhaustion pattern", "no-trend noise"
        ]
        print(f"[CLIP PREDICT] Using {len(candidate_labels)} candidate labels")
        
        # Process inputs
        print(f"[CLIP PREDICT] Processing inputs...")
        inputs = self.processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
            print(f"[CLIP PREDICT] Moved inputs to CUDA")
        
        # Get predictions
        print(f"[CLIP PREDICT] Running model inference...")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get best prediction
        best_idx = probs.argmax().item()
        confidence = probs[0][best_idx].item()
        label = candidate_labels[best_idx]
        
        print(f"[CLIP PREDICT] ✅ Prediction: {label} (confidence: {confidence:.3f})")
        
        result = {
            "label": label,
            "confidence": confidence,
            "all_predictions": {
                candidate_labels[i]: probs[0][i].item() 
                for i in range(len(candidate_labels))
            }
        }
        
        # Debug: show top 3 predictions
        sorted_preds = sorted(result["all_predictions"].items(), key=lambda x: x[1], reverse=True)
        print(f"[CLIP PREDICT] Top 3: {sorted_preds[:3]}")
        
        return result
    
    def _predict_with_openai_clip(self, image):
        """Predict using OpenAI CLIP"""
        print(f"[CLIP PREDICT] Using OpenAI CLIP with {len(self.label_texts)} labels")
        
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(self.tokenized_labels)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).squeeze(0)

        probs = similarities.softmax(dim=0)
        top_index = torch.argmax(probs).item()
        confidence = probs[top_index].item()
        label = self.label_texts[top_index]
        
        print(f"[CLIP PREDICT] ✅ Prediction: {label} (confidence: {confidence:.3f})")

        return {
            "label": label,
            "confidence": confidence,
            "all_predictions": {self.label_texts[i]: probs[i].item() for i in range(len(self.label_texts))}
        }
    
    # Alias for compatibility
    def predict(self, image_path):
        """Compatibility alias for predict_chart_setup"""
        result = self.predict_chart_setup(image_path)
        if result:
            # Convert to old format for compatibility
            result["all_scores"] = result.get("all_predictions", {})
        return result


def predict_clip_chart(image_path: str, candidate_phases=None) -> Dict[str, Any]:
    """
    Global function for CLIP chart prediction with fallback handling
    
    Args:
        image_path: Path to chart image
        candidate_phases: List of candidate phase labels (optional)
        
    Returns:
        Prediction result dictionary
    """
    try:
        predictor = CLIPPredictor()
        
        if predictor.model is None:
            print(f"[CLIP FALLBACK] Model not available, using fallback for: {image_path}")
            return _fallback_prediction(image_path)
        
        result = predictor.predict_chart_setup(image_path)
        
        if result and result.get("confidence", 0) >= 0.3:
            return {
                "success": True,
                "predicted_label": result["label"],
                "confidence": result["confidence"],
                "method": "clip_model",
                "all_predictions": result.get("all_predictions", {})
            }
        else:
            confidence_val = result.get("confidence", 0) if result else 0
            print(f"[CLIP FALLBACK] Low confidence ({confidence_val:.3f}), using fallback")
            return _fallback_prediction(image_path)
            
    except Exception as e:
        print(f"[CLIP ERROR] Error in prediction: {e}")
        return _fallback_prediction(image_path)


def _fallback_prediction(chart_path: str) -> Dict[str, Any]:
    """
    Fallback prediction when CLIP is unavailable
    
    Args:
        chart_path: Path to chart image
        
    Returns:
        Basic prediction result
    """
    # Extract basic info from filename if possible
    filename = os.path.basename(chart_path)
    
    # Simple heuristics based on filename patterns
    if "BTCUSDT" in filename or "ETHUSDT" in filename:
        phase = "trending-up"
        confidence = 0.4
    elif "breakdown" in filename.lower():
        phase = "trend-reversal"
        confidence = 0.4
    elif "pullback" in filename.lower():
        phase = "pullback-in-trend"
        confidence = 0.4
    else:
        phase = "consolidation"
        confidence = 0.3
        
    return {
        "success": True,
        "predicted_label": phase,
        "confidence": confidence,
        "method": "fallback_heuristic",
        "all_predictions": {phase: confidence}
    }


def main():
    """Test CLIP chart prediction"""
    import glob
    
    # Test with existing charts
    chart_locations = [
        "charts/",
        "exports/", 
        "data/charts/",
        "training_data/charts/",
        "training_charts/"
    ]
    
    found_charts = []
    for location in chart_locations:
        if os.path.exists(location):
            charts = glob.glob(f"{location}*.png")
            found_charts.extend(charts)
            
    if found_charts:
        test_chart = found_charts[0]
        print(f"Testing CLIP prediction with: {test_chart}")
        
        result = predict_clip_chart(test_chart)
        if result and result.get("success"):
            print(f"Prediction: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Method: {result['method']}")
        else:
            print("Prediction failed")
    else:
        print("No chart files found for testing")


if __name__ == "__main__":
    main()

# Global predictor instance
_global_predictor = None

def get_clip_predictor():
    """Get global CLIP predictor instance"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = CLIPPredictor()
    return _global_predictor

def predict_clip_chart(chart_path: str, candidate_phases: list = None, confidence_threshold: float = 0.3) -> Optional[Dict[str, Any]]:
    """
    Predict market phase from chart image using CLIP model
    
    Args:
        chart_path: Path to chart image
        candidate_phases: List of phase labels to predict (optional, uses default if None)
        confidence_threshold: Minimum confidence for valid prediction
        
    Returns:
        Dictionary with prediction results or None if failed
    """
    try:
        # Extract symbol from chart path for debugging
        symbol = "UNKNOWN"
        if "/" in chart_path:
            filename = chart_path.split("/")[-1]
            if "_" in filename:
                symbol = filename.split("_")[0]
        
        print(f"[CLIP DEBUG] Predicting phase/setup for {symbol}...")
        logging.debug(f"[CLIP DEBUG] Starting prediction for {symbol} using chart: {chart_path}")
        
        # Use provided candidate phases or default from CLIPPredictor
        predictor = get_clip_predictor()
        if candidate_phases:
            # Override predictor's labels temporarily
            original_labels = predictor.label_texts
            predictor.label_texts = candidate_phases
            predictor.tokenized_labels = clip.tokenize(candidate_phases).to(predictor.device)
        
        result = predictor.predict(chart_path)
        
        # Restore original labels if they were overridden
        if candidate_phases:
            predictor.label_texts = original_labels
            predictor.tokenized_labels = clip.tokenize(original_labels).to(predictor.device)
        
        print(f"[CLIP DEBUG] CLIP prediction: {result['label']}, confidence: {result['confidence']:.4f}")
        logging.debug(f"[CLIP DEBUG] Full prediction result for {symbol}: {result}")
        
        # Apply confidence threshold
        if result["confidence"] < confidence_threshold:
            print(f"[CLIP DEBUG] Low confidence prediction rejected: {result['confidence']:.3f} < {confidence_threshold}")
            logging.debug(f"[CLIP DEBUG] Low confidence prediction for {symbol}: {result['confidence']:.3f} < {confidence_threshold}")
            return {
                "success": False,
                "predicted_label": result["label"],
                "confidence": result["confidence"],
                "reason": f"Low confidence: {result['confidence']:.3f} < {confidence_threshold}",
                "symbol": symbol,
                "chart_path": chart_path,
                "method": "clip_optimized"
            }
            
        print(f"[CLIP DEBUG] Valid prediction accepted for {symbol}: {result['label']} ({result['confidence']:.3f})")
        logging.debug(f"[CLIP DEBUG] Accepted prediction for {symbol}: {result['label']} with confidence {result['confidence']:.3f}")
        
        return {
            "success": True,
            "predicted_label": result["label"],
            "confidence": result["confidence"],
            "symbol": symbol,
            "chart_path": chart_path,
            "method": "clip_optimized",
            "all_predictions": result.get("all_scores", [])
        }
        
    except Exception as e:
        print(f"[CLIP DEBUG] Error in CLIP prediction: {e}")
        logging.error(f"[CLIP DEBUG] Prediction error for chart {chart_path}: {e}")
        return _fallback_prediction(chart_path)


def _fallback_prediction(chart_path: str) -> Dict[str, Any]:
    """
    Fallback prediction when CLIP is unavailable
    
    Args:
        chart_path: Path to chart image
        
    Returns:
        Basic prediction result
    """
    # Extract basic info from filename if possible
    filename = os.path.basename(chart_path)
    
    # Simple heuristics based on filename patterns
    if "BTCUSDT" in filename or "ETHUSDT" in filename:
        phase = "trending-up"
        confidence = 0.4
    elif "breakdown" in filename.lower():
        phase = "trend-reversal"
        confidence = 0.4
    elif "pullback" in filename.lower():
        phase = "pullback-in-trend"
        confidence = 0.4
    else:
        phase = "consolidation"
        confidence = 0.3
        
    return {
        "label": phase,
        "confidence": confidence,
        "chart_path": chart_path,
        "method": "fallback_heuristic",
        "all_scores": {}
    }


def batch_predict_charts(chart_directory: str, pattern: str = "*.png") -> Dict[str, Dict[str, Any]]:
    """
    Predict phases for multiple charts in directory
    
    Args:
        chart_directory: Directory containing chart images
        pattern: File pattern to match
        
    Returns:
        Dictionary mapping chart paths to predictions
    """
    results = {}
    chart_dir = Path(chart_directory)
    
    if not chart_dir.exists():
        logger.warning(f"Chart directory not found: {chart_directory}")
        return results
        
    # Find all matching chart files
    chart_files = list(chart_dir.glob(pattern))
    
    if not chart_files:
        logger.warning(f"No chart files found in {chart_directory}")
        return results
        
    logger.info(f"Processing {len(chart_files)} charts from {chart_directory}")
    
    for chart_file in chart_files:
        try:
            prediction = predict_clip_chart(str(chart_file))
            if prediction:
                results[str(chart_file)] = prediction
                
        except Exception as e:
            logger.error(f"Error processing {chart_file}: {e}")
            continue
            
    logger.info(f"Successfully processed {len(results)}/{len(chart_files)} charts")
    return results


def get_phase_modifiers() -> Dict[str, float]:
    """
    Get score modifiers for different market phases
    
    Returns:
        Dictionary mapping phases to score modifiers
    """
    return {
        "breakout-continuation": 0.08,
        "volume-backed-breakout": 0.10,
        "pullback-in-trend": 0.05,
        "bullish-momentum": 0.06,
        "trending-up": 0.04,
        "range-accumulation": 0.02,
        "consolidation": 0.0,
        "trending-down": -0.04,
        "bearish-momentum": -0.06,
        "exhaustion-pattern": -0.08,
        "trend-reversal": -0.08,
        "fake-breakout": -0.10
    }


def main():
    """Test CLIP chart prediction"""
    import glob
    
    # Test with existing charts
    chart_locations = [
        "charts/",
        "exports/", 
        "data/charts/",
        "training_data/charts/"
    ]
    
    found_charts = []
    for location in chart_locations:
        if os.path.exists(location):
            charts = glob.glob(f"{location}*.png")
            found_charts.extend(charts)
            
    if found_charts:
        test_chart = found_charts[0]
        print(f"Testing CLIP prediction with: {test_chart}")
        
        result = predict_clip_chart(test_chart)
        if result:
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Method: {result['method']}")
        else:
            print("Prediction failed")
    else:
        print("No chart files found for testing")


if __name__ == "__main__":
    main()