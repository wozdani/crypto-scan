"""
Candle Data Fallback System for Vision-AI Chart Generation
Implements alternative data sources and cache loading for insufficient candle data
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_safe_candles(symbol: str, interval: str = "15m", try_alt_sources: bool = True) -> Optional[List]:
    """
    Enhanced candle fetching with fallback mechanisms
    
    Args:
        symbol: Trading symbol
        interval: Time interval (15m, 5m, etc.)
        try_alt_sources: Enable alternative data sources
        
    Returns:
        List of candles or None
    """
    try:
        # Primary source - try current scan results
        candles = _load_from_scan_results(symbol)
        if candles and len(candles) >= 48:
            return candles
            
        if try_alt_sources:
            # Fallback 1: Cache files
            candles = load_candles_from_cache(symbol, interval)
            if candles and len(candles) >= 48:
                return candles
                
            # Fallback 2: Historical data files
            candles = _load_from_historical_data(symbol, interval)
            if candles and len(candles) >= 48:
                return candles
                
        logging.debug(f"get_safe_candles: {symbol} insufficient data from all sources")
        return None
        
    except Exception as e:
        logging.error(f"get_safe_candles error for {symbol}: {e}")
        return None

def load_candles_from_cache(symbol: str, interval: str = "15m") -> Optional[List]:
    """Load candles from various cache locations"""
    cache_locations = [
        f"data/cache/{symbol}_candles_{interval}.json",
        f"data/async_results/{symbol}_async.json",
        f"data/charts/{symbol}_*.json",
        f"data/results/{symbol}_*.json"
    ]
    
    for cache_path in cache_locations:
        if '*' in cache_path:
            # Handle wildcard patterns
            import glob
            files = glob.glob(cache_path)
            if files:
                cache_path = max(files, key=os.path.getmtime)  # Most recent
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract candles from various data structures
                candles = None
                if isinstance(data, list):
                    candles = data
                elif isinstance(data, dict):
                    candles = (data.get('candles_15m') or 
                             data.get('candles') or 
                             data.get('market_data', {}).get('candles', []))
                
                if candles and len(candles) >= 48:
                    logging.debug(f"load_candles_from_cache: {symbol} loaded {len(candles)} candles from {cache_path}")
                    return candles
                    
            except Exception as e:
                logging.debug(f"load_candles_from_cache: Failed to load {cache_path}: {e}")
                continue
                
    return None

def _load_from_scan_results(symbol: str) -> Optional[List]:
    """Load candles from latest scan results"""
    try:
        results_files = [
            "data/latest_scan_results.json",
            "data/async_results/latest.json"
        ]
        
        for file_path in results_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    for entry in data:
                        if entry.get('symbol') == symbol:
                            market_data = entry.get('market_data', {})
                            candles = market_data.get('candles_15m') or market_data.get('candles', [])
                            if candles:
                                return candles
                                
    except Exception as e:
        logging.debug(f"_load_from_scan_results error: {e}")
        
    return None

def _load_from_historical_data(symbol: str, interval: str) -> Optional[List]:
    """Load from historical data directory"""
    try:
        hist_path = f"data/historical/{symbol}_{interval}.json"
        if os.path.exists(hist_path):
            with open(hist_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.debug(f"_load_from_historical_data error: {e}")
        
    return None

def plot_empty_chart(symbol: str, output_dir: str = "training_charts") -> Optional[str]:
    """
    Generate placeholder chart when candle data is unavailable
    
    Args:
        symbol: Trading symbol
        output_dir: Output directory
        
    Returns:
        Path to generated placeholder chart
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        # Placeholder content
        ax.text(0.5, 0.6, f"{symbol}", 
               ha='center', va='center', fontsize=24, fontweight='bold', color='white')
        ax.text(0.5, 0.4, "Insufficient Candle Data", 
               ha='center', va='center', fontsize=16, color='orange')
        ax.text(0.5, 0.3, "Chart will be generated when data becomes available", 
               ha='center', va='center', fontsize=12, color='gray')
        
        # Styling
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timestamp}_placeholder.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        
        logging.info(f"plot_empty_chart: Generated placeholder for {symbol}")
        return filepath
        
    except Exception as e:
        logging.error(f"plot_empty_chart error for {symbol}: {e}")
        return None