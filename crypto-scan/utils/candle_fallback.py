# utils/candle_fallback.py

import requests
import json
import os
from typing import List, Dict, Optional
import time

def get_safe_candles(symbol: str, interval: str = "15m", limit: int = 96, try_alt_sources: bool = True) -> List:
    """
    Enhanced candle fetching with comprehensive fallback system
    
    Args:
        symbol: Trading symbol
        interval: Candle interval (15m, 5m, etc.)
        limit: Number of candles to fetch
        try_alt_sources: Whether to try alternative data sources
        
    Returns:
        List of candle data
    """
    candles = []
    
    # Source 1: Try async results
    if try_alt_sources:
        async_file = f"data/async_results/{symbol}.json"
        if os.path.exists(async_file):
            try:
                with open(async_file, 'r') as f:
                    data = json.load(f)
                    candles = data.get("candles", [])
                    if len(candles) >= 20:
                        print(f"[CANDLE FALLBACK] {symbol}: Using {len(candles)} async candles")
                        return candles
            except Exception as e:
                print(f"[CANDLE FALLBACK] {symbol}: Async source error - {e}")
    
    # Source 2: Try candle cache
    if try_alt_sources and len(candles) < 20:
        cache_file = f"data/candles_cache/{symbol}_{interval}.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    candles = data.get("candles", [])
                    if len(candles) >= 20:
                        print(f"[CANDLE FALLBACK] {symbol}: Using {len(candles)} cached candles")
                        return candles
            except Exception as e:
                print(f"[CANDLE FALLBACK] {symbol}: Cache source error - {e}")
    
    # Source 3: Direct API fetch
    if len(candles) < 20:
        try:
            print(f"[CANDLE FALLBACK] {symbol}: Fetching from Bybit API...")
            
            bybit_interval = interval.replace("m", "")  # "15m" -> "15"
            response = requests.get(
                "https://api.bybit.com/v5/market/kline",
                params={
                    'category': 'linear',
                    'symbol': symbol,
                    'interval': bybit_interval,
                    'limit': str(limit)
                },
                timeout=15
            )
            
            if response.status_code == 200:
                api_data = response.json()
                if api_data.get('retCode') == 0:
                    candles = api_data.get('result', {}).get('list', [])
                    if len(candles) >= 20:
                        print(f"[CANDLE FALLBACK] {symbol}: Fetched {len(candles)} fresh candles")
                        
                        # Save to cache for future use
                        save_candles_to_cache(symbol, interval, candles)
                        return candles
                else:
                    print(f"[CANDLE FALLBACK] {symbol}: API error - {api_data.get('retMsg', 'Unknown')}")
            else:
                print(f"[CANDLE FALLBACK] {symbol}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"[CANDLE FALLBACK] {symbol}: API fetch error - {e}")
    
    print(f"[CANDLE FALLBACK] {symbol}: All sources failed, returning {len(candles)} candles")
    return candles

def save_candles_to_cache(symbol: str, interval: str, candles: List) -> bool:
    """Save candles to cache for future use"""
    try:
        os.makedirs("data/candles_cache", exist_ok=True)
        cache_file = f"data/candles_cache/{symbol}_{interval}.json"
        
        cache_data = {
            "symbol": symbol,
            "interval": interval,
            "candles": candles,
            "timestamp": time.time()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        print(f"[CANDLE CACHE] {symbol}: Saved {len(candles)} candles to cache")
        return True
        
    except Exception as e:
        print(f"[CANDLE CACHE ERROR] {symbol}: {e}")
        return False

def generate_synthetic_candles(symbol: str, base_price: float = 1.0, count: int = 50) -> List:
    """
    Generate synthetic candle data for testing purposes only
    Note: This should only be used for development/testing
    """
    print(f"[SYNTHETIC CANDLES] {symbol}: Generating {count} synthetic candles for testing")
    
    candles = []
    current_price = base_price
    
    for i in range(count):
        # Small random price movements
        import random
        price_change = random.uniform(-0.02, 0.02)  # ±2% movement
        new_price = current_price * (1 + price_change)
        
        # Generate OHLCV
        high = new_price * random.uniform(1.001, 1.01)
        low = new_price * random.uniform(0.99, 0.999)
        close = random.uniform(low, high)
        volume = random.uniform(1000, 10000)
        
        # Bybit format: [timestamp, open, high, low, close, volume]
        timestamp = str(int(time.time() * 1000) - (count - i) * 900000)  # 15min intervals
        candle = [timestamp, str(current_price), str(high), str(low), str(close), str(volume)]
        candles.append(candle)
        
        current_price = close
    
    return candles

def plot_empty_chart(symbol: str, save_path: str, message: str = "Insufficient Data") -> Optional[str]:
    """
    Generate placeholder chart when candle data is unavailable
    For development use only
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f"{symbol}\n{message}", 
                ha='center', va='center', fontsize=16,
                transform=ax.transAxes)
        ax.set_title(f"{symbol} - Chart Generation Failed")
        
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"[PLACEHOLDER CHART] {symbol}: Generated placeholder at {save_path}")
        return save_path
        
    except Exception as e:
        print(f"[PLACEHOLDER ERROR] {symbol}: {e}")
        return None