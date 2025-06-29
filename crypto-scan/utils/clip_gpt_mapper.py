"""
CLIP-GPT Label Mapper
Maps unknown CLIP predictions to known labels using GPT commentary fallback
"""

import os
import json
import glob
from typing import Optional, Dict, Any

# Known CLIP labels for mapping
KNOWN_CLIP_LABELS = [
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
    "volume-backed-breakout",
    "support-bounce",
    "resistance-test",
    "squeeze-pattern"
]

# GPT keywords to CLIP label mapping
GPT_TO_CLIP_MAPPING = {
    "squeeze": "squeeze-pattern",
    "pullback": "pullback-in-trend",
    "retest": "resistance-test",
    "bounce": "support-bounce",
    "breakout": "breakout-continuation",
    "consolidation": "consolidation", 
    "accumulation": "range-accumulation",
    "distribution": "trend-reversal",
    "exhaustion": "exhaustion-pattern",
    "momentum": "bullish-momentum",
    "reversal": "trend-reversal",
    "continuation": "breakout-continuation",
    "support": "support-bounce",
    "resistance": "resistance-test",
    "trend": "trending-up",
    "volume": "volume-backed-breakout"
}

def map_clip_label_with_gpt_fallback(symbol: str, clip_prediction: Dict[str, Any], confidence_threshold: float = 0.4) -> Dict[str, Any]:
    """
    Map CLIP prediction to known label, using GPT commentary as fallback for unknown labels
    
    Args:
        symbol: Trading symbol
        clip_prediction: CLIP prediction result
        confidence_threshold: Minimum confidence for using CLIP prediction
        
    Returns:
        Enhanced prediction with mapped label
    """
    try:
        # Extract CLIP prediction details
        predicted_label = clip_prediction.get("predicted_label", "unknown")
        confidence = clip_prediction.get("confidence", 0.0)
        
        # If CLIP prediction is good and known, use it directly
        if confidence >= confidence_threshold and predicted_label in KNOWN_CLIP_LABELS:
            return {
                "predicted_label": predicted_label,
                "confidence": confidence,
                "mapping_method": "direct_clip",
                "success": True
            }
        
        # If CLIP label is unknown or low confidence, try GPT fallback
        gpt_label = get_gpt_fallback_label(symbol)
        if gpt_label:
            return {
                "predicted_label": gpt_label,
                "confidence": max(confidence, 0.45),  # Boost confidence slightly for GPT mapping
                "mapping_method": "gpt_fallback",
                "original_clip_label": predicted_label,
                "success": True
            }
        
        # Last resort: return best available with warning
        return {
            "predicted_label": predicted_label if predicted_label != "unknown" else "consolidation",
            "confidence": max(confidence, 0.3),
            "mapping_method": "fallback_default",
            "original_clip_label": predicted_label,
            "success": False,
            "warning": "No suitable mapping found"
        }
        
    except Exception as e:
        print(f"[CLIP MAPPER ERROR] {symbol}: {e}")
        return {
            "predicted_label": "consolidation",
            "confidence": 0.3,
            "mapping_method": "error_fallback",
            "success": False,
            "error": str(e)
        }

def get_gpt_fallback_label(symbol: str) -> Optional[str]:
    """
    Extract label from GPT commentary files for given symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Mapped label from GPT commentary or None
    """
    try:
        # Search for GPT commentary files for this symbol
        gpt_patterns = [
            f"training_data/charts/{symbol}_*.gpt.json",
            f"data/gpt_analysis/{symbol}_*.json",
            f"metadata/{symbol}_*.json"
        ]
        
        for pattern in gpt_patterns:
            gpt_files = glob.glob(pattern)
            for gpt_file in sorted(gpt_files, reverse=True):  # Most recent first
                try:
                    with open(gpt_file, 'r') as f:
                        gpt_data = json.load(f)
                    
                    # Extract GPT commentary text
                    commentary = ""
                    if isinstance(gpt_data, dict):
                        commentary = gpt_data.get("commentary", gpt_data.get("analysis", gpt_data.get("description", "")))
                    elif isinstance(gpt_data, str):
                        commentary = gpt_data
                    
                    if commentary:
                        mapped_label = map_gpt_keywords_to_label(commentary.lower())
                        if mapped_label:
                            print(f"[GPT MAPPER] {symbol}: '{commentary[:50]}...' → {mapped_label}")
                            return mapped_label
                            
                except Exception as file_error:
                    continue
        
        return None
        
    except Exception as e:
        print(f"[GPT FALLBACK ERROR] {symbol}: {e}")
        return None

def map_gpt_keywords_to_label(commentary_text: str) -> Optional[str]:
    """
    Map GPT commentary keywords to known CLIP labels
    
    Args:
        commentary_text: GPT commentary text (lowercase)
        
    Returns:
        Mapped label or None
    """
    # Score each potential label based on keyword presence
    label_scores = {}
    
    for keyword, label in GPT_TO_CLIP_MAPPING.items():
        if keyword in commentary_text:
            label_scores[label] = label_scores.get(label, 0) + 1
    
    # Return label with highest score
    if label_scores:
        best_label = max(label_scores.items(), key=lambda x: x[1])[0]
        return best_label
    
    return None

def enhance_clip_prediction_with_mapping(symbol: str, tjde_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance TJDE result with improved CLIP label mapping
    
    Args:
        symbol: Trading symbol  
        tjde_result: Original TJDE result with CLIP prediction
        
    Returns:
        Enhanced TJDE result with mapped CLIP label
    """
    try:
        # Extract existing CLIP prediction
        clip_info = tjde_result.get("clip_prediction", {})
        if not clip_info:
            return tjde_result
        
        # Apply label mapping
        mapped_prediction = map_clip_label_with_gpt_fallback(symbol, clip_info)
        
        # Update TJDE result with mapped prediction
        tjde_result["clip_prediction"] = mapped_prediction
        tjde_result["clip_enhanced"] = True
        
        # Log mapping result
        method = mapped_prediction.get("mapping_method", "unknown")
        label = mapped_prediction.get("predicted_label", "unknown")
        confidence = mapped_prediction.get("confidence", 0.0)
        
        print(f"[CLIP ENHANCED] {symbol}: {label} (conf: {confidence:.3f}, method: {method})")
        
        return tjde_result
        
    except Exception as e:
        print(f"[CLIP ENHANCEMENT ERROR] {symbol}: {e}")
        return tjde_result

def test_label_mapping():
    """Test label mapping functionality"""
    test_cases = [
        {
            "symbol": "TESTUSDT",
            "clip_prediction": {"predicted_label": "unknown", "confidence": 0.2},
            "expected_fallback": True
        },
        {
            "symbol": "TESTUSDT", 
            "clip_prediction": {"predicted_label": "pullback-in-trend", "confidence": 0.6},
            "expected_fallback": False
        }
    ]
    
    for test in test_cases:
        result = map_clip_label_with_gpt_fallback(
            test["symbol"], 
            test["clip_prediction"]
        )
        print(f"Test {test['symbol']}: {result}")

if __name__ == "__main__":
    test_label_mapping()