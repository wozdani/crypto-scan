"""
Local Candle Cache System for Trend-Mode
Enables fallback charts for Vision-AI when API doesn't return candles
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict

CACHE_DIR = "data/candles_cache"

def save_candles_to_cache(symbol: str, candles: list, interval: str = "15m"):
    """
    Save candles to local cache for future fallback usage
    
    Args:
        symbol: Trading symbol
        candles: List of candle data
        interval: Time interval (15m, 5m, etc.)
    """
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
        # Add metadata for cache validation
        cache_data = {
            "symbol": symbol,
            "interval": interval,
            "cached_at": datetime.now().isoformat(),
            "candle_count": len(candles),
            "candles": candles
        }
        
        path = f"{CACHE_DIR}/{symbol}_{interval}.json"
        with open(path, "w") as f:
            json.dump(cache_data, f, indent=2)
            
        logging.debug(f"Cached {len(candles)} candles for {symbol} ({interval})")
        
    except Exception as e:
        logging.error(f"Failed to cache candles for {symbol}: {e}")

def load_candles_from_cache(symbol: str, interval: str = "15m") -> Optional[List]:
    """
    Load candles from local cache
    
    Args:
        symbol: Trading symbol
        interval: Time interval
        
    Returns:
        List of candles or None if not found/invalid
    """
    try:
        path = f"{CACHE_DIR}/{symbol}_{interval}.json"
        if not os.path.exists(path):
            return None
            
        with open(path, "r") as f:
            cache_data = json.load(f)
            
        # Validate cache structure
        if isinstance(cache_data, list):
            # Legacy format - just candles
            return cache_data
        elif isinstance(cache_data, dict) and "candles" in cache_data:
            # New format with metadata
            candles = cache_data["candles"]
            cached_at = cache_data.get("cached_at")
            
            # Check if cache is not too old (optional validation)
            if cached_at:
                cache_time = datetime.fromisoformat(cached_at)
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                
                # Log cache age for debugging
                logging.debug(f"Cache for {symbol} is {age_hours:.1f} hours old")
                
                # For now, accept any cache age - user can tune this later
                
            return candles
        else:
            logging.warning(f"Invalid cache format for {symbol}")
            return None
            
    except Exception as e:
        logging.error(f"Failed to load cache for {symbol}: {e}")
        return None

def get_cache_stats() -> Dict:
    """Get statistics about cached candles"""
    try:
        if not os.path.exists(CACHE_DIR):
            return {"total_files": 0, "cache_dir": CACHE_DIR}
            
        files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
        
        stats = {
            "total_files": len(files),
            "cache_dir": CACHE_DIR,
            "symbols": []
        }
        
        for file in files[:10]:  # Show first 10 for brevity
            try:
                with open(os.path.join(CACHE_DIR, file), 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict):
                    stats["symbols"].append({
                        "file": file,
                        "symbol": data.get("symbol", "unknown"),
                        "candle_count": data.get("candle_count", 0),
                        "cached_at": data.get("cached_at", "unknown")
                    })
                else:
                    stats["symbols"].append({
                        "file": file,
                        "symbol": file.split('_')[0],
                        "candle_count": len(data) if isinstance(data, list) else 0,
                        "cached_at": "legacy_format"
                    })
                    
            except Exception as e:
                logging.debug(f"Error reading cache file {file}: {e}")
                
        return stats
        
    except Exception as e:
        logging.error(f"Error getting cache stats: {e}")
        return {"error": str(e)}

def cleanup_old_cache(max_age_days: int = 7):
    """
    Clean up cache files older than specified days
    
    Args:
        max_age_days: Maximum age in days before cleanup
    """
    try:
        if not os.path.exists(CACHE_DIR):
            return
            
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        for file in os.listdir(CACHE_DIR):
            if not file.endswith('.json'):
                continue
                
            file_path = os.path.join(CACHE_DIR, file)
            
            try:
                # Check file modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if mod_time < cutoff_time:
                    os.remove(file_path)
                    cleaned_count += 1
                    logging.debug(f"Cleaned old cache file: {file}")
                    
            except Exception as e:
                logging.debug(f"Error cleaning cache file {file}: {e}")
                
        if cleaned_count > 0:
            logging.info(f"Cleaned {cleaned_count} old cache files")
            
    except Exception as e:
        logging.error(f"Error during cache cleanup: {e}")