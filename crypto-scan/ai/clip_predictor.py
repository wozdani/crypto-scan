"""
CLIP Chart Predictor for AI Module
Provides chart phase prediction using CLIP model integration
"""

import os
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Market phase labels for CLIP prediction
CANDIDATE_PHASES = [
    "breakout-continuation",
    "pullback-in-trend", 
    "range-accumulation",
    "trend-reversal",
    "consolidation",
    "fake-breakout",
    "trending-up",
    "trending-down",
    "bullish-momentum",
    "bearish-momentum",
    "exhaustion-pattern",
    "volume-backed-breakout"
]

def predict_clip_chart(chart_path: str, confidence_threshold: float = 0.3) -> Optional[Dict[str, Any]]:
    """
    Predict market phase from chart image using CLIP model
    
    Args:
        chart_path: Path to chart image
        confidence_threshold: Minimum confidence for valid prediction
        
    Returns:
        Dictionary with prediction results or None if failed
    """
    try:
        # Import CLIP model from local ai module
        from ai.clip_model import CLIPWrapper
        
        # Initialize CLIP model
        clip_model = CLIPWrapper()
        
        # Validate chart file exists
        if not os.path.exists(chart_path):
            logger.warning(f"Chart file not found: {chart_path}")
            return None
            
        # Load and process image
        try:
            image_embedding = clip_model.get_image_embedding(chart_path)
            if image_embedding is None:
                logger.warning(f"Failed to generate image embedding for {chart_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing image {chart_path}: {e}")
            return None
            
        # Generate text embeddings for all candidate phases
        text_embeddings = []
        for phase in CANDIDATE_PHASES:
            text_prompt = f"A cryptocurrency chart showing {phase.replace('-', ' ')} pattern"
            text_embedding = clip_model.get_text_embedding(text_prompt)
            if text_embedding is not None:
                text_embeddings.append(text_embedding)
            else:
                text_embeddings.append(np.zeros_like(image_embedding))
                
        if not text_embeddings:
            logger.warning("Failed to generate any text embeddings")
            return None
            
        # Calculate similarities
        text_embeddings = np.array(text_embeddings)
        similarities = []
        
        for text_emb in text_embeddings:
            # Normalize embeddings
            image_norm = image_embedding / np.linalg.norm(image_embedding)
            text_norm = text_emb / np.linalg.norm(text_emb)
            
            # Calculate cosine similarity
            similarity = np.dot(image_norm, text_norm)
            similarities.append(similarity)
            
        # Find best match
        similarities = np.array(similarities)
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_phase = CANDIDATE_PHASES[best_idx]
        
        # Apply confidence threshold
        if best_similarity < confidence_threshold:
            logger.info(f"Low confidence prediction: {best_similarity:.3f} < {confidence_threshold}")
            return {
                "label": "no-clear-pattern",
                "confidence": float(best_similarity),
                "chart_path": chart_path,
                "method": "clip_zero_shot",
                "all_scores": {phase: float(sim) for phase, sim in zip(CANDIDATE_PHASES, similarities)}
            }
            
        logger.info(f"CLIP prediction: {best_phase} (confidence: {best_similarity:.3f})")
        
        return {
            "label": best_phase,
            "confidence": float(best_similarity),
            "chart_path": chart_path,
            "method": "clip_zero_shot",
            "all_scores": {phase: float(sim) for phase, sim in zip(CANDIDATE_PHASES, similarities)}
        }
        
    except ImportError as e:
        logger.warning(f"CLIP model not available: {e}")
        return _fallback_prediction(chart_path)
        
    except Exception as e:
        logger.error(f"Error in CLIP prediction: {e}")
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