"""
CLIP Prediction Loader
Ładuje predykcje CLIP z folderów data/clip_predictions/ dla integracji z TJDE
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def load_clip_prediction(symbol: str, max_age_hours: int = 2) -> Optional[str]:
    """
    Load CLIP prediction for symbol from data/clip_predictions/
    
    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        max_age_hours: Maximum age of prediction file in hours
        
    Returns:
        CLIP prediction string or None if not found
    """
    try:
        # Check multiple prediction directories
        prediction_dirs = [
            Path("data/clip_predictions"),
            Path("data/vision_ai/predictions"),
            Path("clip_predictions"),
            Path("logs")  # Fallback to logs directory
        ]
        
        for pred_dir in prediction_dirs:
            if pred_dir.exists():
                # Look for prediction files with symbol
                patterns = [
                    f"{symbol}_*.txt",
                    f"{symbol}_*.json",
                    f"*{symbol}*.txt"
                ]
                
                for pattern in patterns:
                    prediction_files = list(pred_dir.glob(pattern))
                    
                    if prediction_files:
                        # Get most recent prediction file
                        latest_file = sorted(prediction_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                        
                        # Check file age
                        file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
                        
                        if file_age.total_seconds() / 3600 <= max_age_hours:
                            # Load prediction content
                            with open(latest_file, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                            
                            logger.info(f"Loaded CLIP prediction for {symbol} from {latest_file.name}")
                            return content
                        else:
                            logger.warning(f"CLIP prediction for {symbol} too old: {file_age}")
        
        # If no files found, try to get from recent session history
        return load_from_session_history(symbol, max_age_hours)
        
    except Exception as e:
        logger.error(f"Error loading CLIP prediction for {symbol}: {e}")
        return None

def load_from_session_history(symbol: str, max_age_hours: int = 2) -> Optional[str]:
    """
    Load CLIP prediction from auto_label_session_history.json
    
    Args:
        symbol: Trading symbol
        max_age_hours: Maximum age in hours
        
    Returns:
        CLIP prediction string or None
    """
    try:
        history_file = Path("logs/auto_label_session_history.json")
        
        if not history_file.exists():
            return None
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # Find recent prediction for symbol
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for entry in reversed(history):  # Start from most recent
            if entry.get("symbol") == symbol:
                try:
                    # Check timestamp if available
                    if "timestamp" in entry:
                        entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                        if entry_time.replace(tzinfo=None) < cutoff_time:
                            continue
                    
                    prediction = entry.get("label", "")
                    if prediction:
                        logger.info(f"Loaded CLIP prediction for {symbol} from session history")
                        return prediction
                        
                except Exception:
                    continue
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading from session history for {symbol}: {e}")
        return None

def save_clip_prediction(symbol: str, prediction: str, confidence: float = 0.0) -> bool:
    """
    Save CLIP prediction for symbol
    
    Args:
        symbol: Trading symbol
        prediction: CLIP prediction string
        confidence: Prediction confidence
        
    Returns:
        True if saved successfully
    """
    try:
        # Create predictions directory
        pred_dir = Path("data/clip_predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        pred_file = pred_dir / f"{symbol}_{timestamp}.txt"
        
        # Save prediction
        with open(pred_file, 'w', encoding='utf-8') as f:
            f.write(prediction)
        
        # Also save as JSON with metadata
        json_file = pred_dir / f"{symbol}_{timestamp}.json"
        prediction_data = {
            "symbol": symbol,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "source": "clip_predictor"
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, indent=2)
        
        logger.info(f"Saved CLIP prediction for {symbol}: {prediction}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving CLIP prediction for {symbol}: {e}")
        return False

def get_prediction_stats() -> Dict[str, Any]:
    """Get statistics about available CLIP predictions"""
    try:
        stats = {
            "total_predictions": 0,
            "recent_predictions": 0,
            "symbols_with_predictions": set(),
            "oldest_prediction": None,
            "newest_prediction": None
        }
        
        # Check all prediction directories
        prediction_dirs = [
            Path("data/clip_predictions"),
            Path("data/vision_ai/predictions"),
            Path("clip_predictions")
        ]
        
        for pred_dir in prediction_dirs:
            if pred_dir.exists():
                prediction_files = list(pred_dir.glob("*.txt")) + list(pred_dir.glob("*.json"))
                
                for pred_file in prediction_files:
                    try:
                        # Extract symbol from filename
                        symbol = pred_file.stem.split('_')[0]
                        stats["symbols_with_predictions"].add(symbol)
                        stats["total_predictions"] += 1
                        
                        # Check file age
                        file_time = datetime.fromtimestamp(pred_file.stat().st_mtime)
                        file_age = datetime.now() - file_time
                        
                        if file_age.total_seconds() / 3600 <= 24:  # Recent = last 24h
                            stats["recent_predictions"] += 1
                        
                        # Track oldest and newest
                        if stats["oldest_prediction"] is None or file_time < stats["oldest_prediction"]:
                            stats["oldest_prediction"] = file_time
                        
                        if stats["newest_prediction"] is None or file_time > stats["newest_prediction"]:
                            stats["newest_prediction"] = file_time
                            
                    except Exception:
                        continue
        
        stats["symbols_with_predictions"] = len(stats["symbols_with_predictions"])
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting prediction stats: {e}")
        return {"error": str(e)}

def main():
    """Test CLIP prediction loader"""
    print("Testing CLIP Prediction Loader")
    print("=" * 40)
    
    # Get prediction statistics
    stats = get_prediction_stats()
    
    print("Prediction Statistics:")
    for key, value in stats.items():
        if key not in ["oldest_prediction", "newest_prediction"]:
            print(f"   {key}: {value}")
    
    if stats.get("oldest_prediction"):
        print(f"   oldest_prediction: {stats['oldest_prediction'].strftime('%Y-%m-%d %H:%M')}")
    if stats.get("newest_prediction"):
        print(f"   newest_prediction: {stats['newest_prediction'].strftime('%Y-%m-%d %H:%M')}")
    
    # Test loading predictions for common symbols
    test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    
    print(f"\nTesting prediction loading:")
    for symbol in test_symbols:
        prediction = load_clip_prediction(symbol)
        if prediction:
            print(f"   {symbol}: {prediction[:50]}...")
        else:
            print(f"   {symbol}: No prediction found")

if __name__ == "__main__":
    main()