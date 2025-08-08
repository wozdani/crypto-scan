"""
Real Data Loaders for Explore to Training Converter
Integrates with existing Bybit data fetchers
"""

import json
import sys
import os
from typing import Tuple, List, Optional
from datetime import datetime, timedelta

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_fetchers import fetch_klines, get_all_data
from utils.robust_api import robust_api

def get_price_and_volume(symbol: str, ts: str) -> Tuple[Optional[float], Optional[float], float]:
    """
    Get ticker price, candle price and 24h volume for a symbol
    
    Args:
        symbol: Trading symbol (e.g., "BIOUSDT")
        ts: Timestamp (not used currently, gets latest data)
    
    Returns:
        Tuple of (ticker_price, candle_price, volume_24h_usd)
    """
    try:
        # Try to get ticker data from Bybit
        ticker_data = robust_api.get_bybit_ticker_v5(symbol)
        
        if ticker_data and ticker_data.get("result"):
            result = ticker_data["result"]["list"][0] if ticker_data["result"]["list"] else {}
            ticker_price = float(result.get("lastPrice", 0)) if result else None
            volume_24h = float(result.get("volume24h", 0)) if result else 0
        else:
            ticker_price = None
            volume_24h = 0
        
        # Get candle price as fallback
        candles = robust_api.get_bybit_kline_v5(symbol, "15", 2)
        if candles and len(candles) > 0:
            # Candle format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
            candle_price = float(candles[0][4])  # Close price of latest candle
            
            # If no ticker volume, estimate from candles
            if volume_24h == 0 and len(candles) > 0:
                # Sum last 96 15-minute candles (24 hours)
                candles_24h = robust_api.get_bybit_kline_v5(symbol, "15", 96)
                if candles_24h:
                    volume_24h = sum(float(c[5]) * float(c[4]) for c in candles_24h)  # volume * price
        else:
            candle_price = None
        
        return ticker_price, candle_price, volume_24h
        
    except Exception as e:
        print(f"[PRICE LOADER ERROR] {symbol}: {e}")
        return None, None, 0

def get_future_prices(symbol: str, ts: str, horizon_min: int = 360) -> List[Tuple[float, float]]:
    """
    Get future price data for labeling
    
    Args:
        symbol: Trading symbol
        ts: Starting timestamp
        horizon_min: How many minutes into the future to get (default 360 = 6 hours)
    
    Returns:
        List of (timestamp, price) tuples at 5-minute intervals
    """
    try:
        # Calculate number of 5-minute candles needed
        num_candles = horizon_min // 5
        
        # Get 5-minute candles
        candles = robust_api.get_bybit_kline_v5(symbol, "5", num_candles + 1)
        
        if not candles:
            print(f"[OHLCV LOADER] No candle data for {symbol}")
            return []
        
        # Convert to (timestamp, price) tuples
        prices = []
        for candle in reversed(candles):  # Reverse to get chronological order
            # Candle format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
            timestamp = int(candle[0]) / 1000  # Convert ms to seconds
            close_price = float(candle[4])
            prices.append((timestamp, close_price))
        
        return prices[:num_candles]  # Return only requested number
        
    except Exception as e:
        print(f"[OHLCV LOADER ERROR] {symbol}: {e}")
        return []

def load_weights_from_cache() -> dict:
    """
    Load current weights from stealth weights cache
    
    Returns:
        Dict of signal weights
    """
    try:
        weights_path = "crypto-scan/cache/stealth_weights.json"
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                cached = json.load(f)
                
                # Map to our expected names
                weights = {
                    "whale_ping": cached.get("whale_ping", 0.22),
                    "dex_inflow": cached.get("dex_inflow", 0.20),
                    "repeated_address_boost": cached.get("repeated_address_boost", 0.25),
                    "velocity_boost": cached.get("velocity_boost", 0.18),
                    "diamond_ai": cached.get("diamond_whale_detection", 0.30),
                    "californium_ai": cached.get("californium_whale_detection", 0.25),
                    "whale_dex_synergy": 0.35  # Not in cache, use default
                }
                return weights
        else:
            print("[WEIGHTS] Cache not found, using defaults")
    except Exception as e:
        print(f"[WEIGHTS ERROR] Failed to load weights: {e}")
    
    # Return defaults if loading fails
    return {
        "whale_ping": 0.22,
        "dex_inflow": 0.20,
        "repeated_address_boost": 0.25,
        "velocity_boost": 0.18,
        "diamond_ai": 0.30,
        "californium_ai": 0.25,
        "whale_dex_synergy": 0.35
    }

if __name__ == "__main__":
    # Test the loaders
    print("Testing real data loaders...")
    
    # Test price loader
    symbol = "BTCUSDT"
    ticker, candle, volume = get_price_and_volume(symbol, "")
    print(f"\n{symbol} prices:")
    print(f"  Ticker: ${ticker}")
    print(f"  Candle: ${candle}")
    print(f"  Volume: ${volume:,.0f}")
    
    # Test OHLCV loader
    prices = get_future_prices(symbol, "", 30)  # Get 30 minutes of data
    print(f"\n{symbol} future prices (first 5):")
    for ts, px in prices[:5]:
        print(f"  {datetime.fromtimestamp(ts)}: ${px:,.2f}")
    
    # Test weights loader
    weights = load_weights_from_cache()
    print(f"\nLoaded weights:")
    for k, v in weights.items():
        print(f"  {k}: {v}")