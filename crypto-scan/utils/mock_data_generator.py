"""
Mock Data Generator for Development Environment
Generates realistic market data when Bybit API is unavailable (HTTP 403 in Replit)
"""
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

def generate_realistic_candles(symbol: str, interval: str = "15", limit: int = 96) -> List[List]:
    """
    Generate realistic candle data for development/testing
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Candle interval in minutes
        limit: Number of candles to generate
        
    Returns:
        List of candles in Bybit format [timestamp, open, high, low, close, volume]
    """
    # Base prices for different symbols
    base_prices = {
        'BTCUSDT': 50000,
        'ETHUSDT': 3000,
        'ADAUSDT': 0.5,
        'DOTUSDT': 8.0,
        'SOLUSDT': 100,
        'LINKUSDT': 15,
        'AVAXUSDT': 25,
        'MATICUSDT': 0.8,
        'ATOMUSDT': 12,
        'ALGOUSDT': 0.3
    }
    
    base_price = base_prices.get(symbol, 1.0)
    
    candles = []
    current_time = int(time.time() * 1000)
    interval_ms = int(interval) * 60 * 1000
    
    current_price = base_price
    
    for i in range(limit):
        timestamp = current_time - (limit - i) * interval_ms
        
        # Generate realistic price movement (±2% per candle)
        price_change = random.uniform(-0.02, 0.02)
        current_price *= (1 + price_change)
        
        # Generate OHLC around current price
        volatility = random.uniform(0.001, 0.005)  # 0.1-0.5% volatility
        
        open_price = current_price
        close_price = current_price * (1 + price_change)
        
        high_price = max(open_price, close_price) * (1 + volatility)
        low_price = min(open_price, close_price) * (1 - volatility)
        
        # Generate realistic volume
        base_volume = 1000000 if 'BTC' in symbol or 'ETH' in symbol else 500000
        volume = base_volume * random.uniform(0.5, 2.0)
        
        candle = [
            str(timestamp),
            f"{open_price:.6f}",
            f"{high_price:.6f}",
            f"{low_price:.6f}",
            f"{close_price:.6f}",
            f"{volume:.2f}"
        ]
        
        candles.append(candle)
        current_price = close_price
    
    return candles

def generate_realistic_ticker(symbol: str, last_candle: List = None) -> Dict:
    """
    Generate realistic ticker data
    
    Args:
        symbol: Trading symbol
        last_candle: Last candle data to extract price from
        
    Returns:
        Realistic ticker data
    """
    if last_candle and len(last_candle) >= 5:
        last_price = float(last_candle[4])  # close price
        high_24h = float(last_candle[2]) * 1.05  # ~5% higher
        low_24h = float(last_candle[3]) * 0.95   # ~5% lower
    else:
        base_prices = {
            'BTCUSDT': 50000,
            'ETHUSDT': 3000,
            'ADAUSDT': 0.5,
            'DOTUSDT': 8.0,
            'SOLUSDT': 100,
            'LINKUSDT': 15,
            'AVAXUSDT': 25,
            'MATICUSDT': 0.8,
            'ATOMUSDT': 12,
            'ALGOUSDT': 0.3
        }
        last_price = base_prices.get(symbol, 1.0)
        high_24h = last_price * 1.05
        low_24h = last_price * 0.95
    
    # Generate realistic 24h change
    change_24h = random.uniform(-5.0, 5.0)
    
    # Generate realistic volume
    base_volume = 1000000 if 'BTC' in symbol or 'ETH' in symbol else 500000
    volume_24h = base_volume * random.uniform(0.8, 1.5)
    
    return {
        "symbol": symbol,
        "lastPrice": f"{last_price:.6f}",
        "priceChangePercent": f"{change_24h:.2f}",
        "highPrice24h": f"{high_24h:.6f}",
        "lowPrice24h": f"{low_24h:.6f}",
        "volume24h": f"{volume_24h:.2f}",
        "turnover24h": f"{volume_24h * last_price:.2f}"
    }

def generate_realistic_orderbook(symbol: str, current_price: float = None) -> Dict:
    """
    Generate realistic orderbook data
    
    Args:
        symbol: Trading symbol
        current_price: Current price for orderbook generation
        
    Returns:
        Realistic orderbook data
    """
    if not current_price:
        base_prices = {
            'BTCUSDT': 50000,
            'ETHUSDT': 3000,
            'ADAUSDT': 0.5,
            'DOTUSDT': 8.0,
            'SOLUSDT': 100,
            'LINKUSDT': 15,
            'AVAXUSDT': 25,
            'MATICUSDT': 0.8,
            'ATOMUSDT': 12,
            'ALGOUSDT': 0.3
        }
        current_price = base_prices.get(symbol, 1.0)
    
    # Generate bid/ask spread (0.01-0.1%)
    spread_pct = random.uniform(0.0001, 0.001)
    bid_price = current_price * (1 - spread_pct/2)
    ask_price = current_price * (1 + spread_pct/2)
    
    # Generate orderbook levels
    bids = []
    asks = []
    
    for i in range(25):  # 25 levels each side
        # Bids (decreasing prices)
        bid_level_price = bid_price * (1 - i * 0.0001)
        bid_quantity = random.uniform(0.1, 10.0)
        bids.append([f"{bid_level_price:.6f}", f"{bid_quantity:.4f}"])
        
        # Asks (increasing prices)
        ask_level_price = ask_price * (1 + i * 0.0001)
        ask_quantity = random.uniform(0.1, 10.0)
        asks.append([f"{ask_level_price:.6f}", f"{ask_quantity:.4f}"])
    
    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "ts": str(int(time.time() * 1000)),
        "u": random.randint(1000000, 9999999)
    }

def should_use_mock_data() -> bool:
    """
    Determine if mock data should be used based on environment
    
    Returns:
        True if in development environment (Replit) where API returns 403
    """
    # In Replit environment, Bybit API returns 403
    # This is a development environment indicator
    return True  # Always use mock data in this environment

def get_mock_data_for_symbol(symbol: str) -> Dict:
    """
    Get complete mock data package for a symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Complete data package with candles, ticker, and orderbook
    """
    candles_15m = generate_realistic_candles(symbol, "15", 96)
    candles_5m = generate_realistic_candles(symbol, "5", 200)
    
    last_candle = candles_15m[-1] if candles_15m else None
    ticker = generate_realistic_ticker(symbol, last_candle)
    
    current_price = float(ticker["lastPrice"])
    orderbook = generate_realistic_orderbook(symbol, current_price)
    
    return {
        "candles_15m": candles_15m,
        "candles_5m": candles_5m,
        "ticker": ticker,
        "orderbook": orderbook,
        "source": "mock_data_generator",
        "timestamp": int(time.time())
    }

def log_mock_data_usage(symbol: str, data_types: List[str]):
    """
    Log when mock data is used for transparency
    
    Args:
        symbol: Symbol using mock data
        data_types: Types of data mocked (ticker, candles, orderbook)
    """
    data_str = ", ".join(data_types)
    print(f"[MOCK DATA] {symbol} → Using realistic mock data for: {data_str}")