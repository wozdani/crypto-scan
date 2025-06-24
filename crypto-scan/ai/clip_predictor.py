# ai/clip_predictor.py

import os
import torch
from PIL import Image
import clip
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CLIPPredictor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.label_texts = [
            "breakout-continuation",
            "pullback-in-trend",
            "trend-reversal",
            "range-accumulation",
            "consolidation",
            "fakeout",
            "volume-backed",
            "low-volume",
            "trending-up",
            "trending-down"
        ]
        self.tokenized_labels = clip.tokenize(self.label_texts).to(self.device)

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.tokenized_labels)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).squeeze(0)

        probs = similarities.softmax(dim=0)
        top_index = torch.argmax(probs).item()

        return {
            "label": self.label_texts[top_index],
            "confidence": round(probs[top_index].item(), 4),
            "all_scores": dict(zip(self.label_texts, [round(p.item(), 4) for p in probs]))
        }

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