#!/usr/bin/env python3
"""
API Utilities for Crypto Scanner
Provides unified API access for market data from various exchanges
"""

import requests
import json
import time
from typing import List, Dict, Optional
from datetime import datetime


def get_bybit_candles(symbol: str, interval: str = "15", limit: int = 200) -> Optional[List[List]]:
    """
    Get candlestick data from Bybit API
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Timeframe ('1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D')
        limit: Number of candles to fetch (max 1000)
        
    Returns:
        List of OHLCV candle data or None if failed
    """
    try:
        base_url = "https://api.bybit.com/v5/market/kline"
        
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("retCode") == 0 and "result" in data:
                candles = data["result"]["list"]
                
                # Convert Bybit format to standard OHLCV format
                # Bybit returns: [startTime, open, high, low, close, volume, turnover]
                # We need: [timestamp, open, high, low, close, volume]
                formatted_candles = []
                
                for candle in candles:
                    formatted_candle = [
                        int(candle[0]),      # timestamp
                        float(candle[1]),    # open
                        float(candle[2]),    # high
                        float(candle[3]),    # low
                        float(candle[4]),    # close
                        float(candle[5])     # volume
                    ]
                    formatted_candles.append(formatted_candle)
                
                # Reverse to get chronological order (oldest first)
                formatted_candles.reverse()
                
                return formatted_candles
            else:
                print(f"[API] Bybit API error for {symbol}: {data.get('retMsg', 'Unknown error')}")
                return None
        else:
            print(f"[API] HTTP {response.status_code} for {symbol}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"[API] Timeout fetching {symbol} data")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[API] Request error for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"[API] Unexpected error for {symbol}: {e}")
        return None


def get_ticker_data(symbol: str) -> Optional[Dict]:
    """
    Get 24h ticker data from Bybit
    
    Args:
        symbol: Trading pair symbol
        
    Returns:
        Ticker data dict or None if failed
    """
    try:
        base_url = "https://api.bybit.com/v5/market/tickers"
        
        params = {
            "category": "linear",
            "symbol": symbol
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("retCode") == 0 and "result" in data:
                ticker_list = data["result"]["list"]
                
                if ticker_list:
                    ticker = ticker_list[0]
                    
                    return {
                        "symbol": ticker.get("symbol"),
                        "price": float(ticker.get("lastPrice", 0)),
                        "price_change_24h": float(ticker.get("price24hPcnt", 0)) * 100,
                        "volume_24h": float(ticker.get("volume24h", 0)),
                        "high_24h": float(ticker.get("highPrice24h", 0)),
                        "low_24h": float(ticker.get("lowPrice24h", 0)),
                        "turnover_24h": float(ticker.get("turnover24h", 0))
                    }
        
        return None
        
    except Exception as e:
        print(f"[API] Error fetching ticker for {symbol}: {e}")
        return None


def get_orderbook(symbol: str, limit: int = 25) -> Optional[Dict]:
    """
    Get orderbook data from Bybit
    
    Args:
        symbol: Trading pair symbol
        limit: Depth limit (5, 10, 25, 50, 100, 200)
        
    Returns:
        Orderbook data dict or None if failed
    """
    try:
        base_url = "https://api.bybit.com/v5/market/orderbook"
        
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("retCode") == 0 and "result" in data:
                result = data["result"]
                
                return {
                    "symbol": symbol,
                    "bids": [[float(bid[0]), float(bid[1])] for bid in result.get("b", [])],
                    "asks": [[float(ask[0]), float(ask[1])] for ask in result.get("a", [])],
                    "timestamp": int(result.get("ts", 0))
                }
        
        return None
        
    except Exception as e:
        print(f"[API] Error fetching orderbook for {symbol}: {e}")
        return None


def test_api_connection() -> Dict:
    """Test API connection and return status"""
    try:
        # Test with a common symbol
        test_symbol = "BTCUSDT"
        
        # Test candles endpoint
        candles = get_bybit_candles(test_symbol, "15", 10)
        candles_ok = candles is not None and len(candles) > 0
        
        # Test ticker endpoint
        ticker = get_ticker_data(test_symbol)
        ticker_ok = ticker is not None and "price" in ticker
        
        # Test orderbook endpoint
        orderbook = get_orderbook(test_symbol, 10)
        orderbook_ok = orderbook is not None and "bids" in orderbook
        
        return {
            "candles": candles_ok,
            "ticker": ticker_ok,
            "orderbook": orderbook_ok,
            "overall": candles_ok and ticker_ok and orderbook_ok,
            "test_symbol": test_symbol,
            "test_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "overall": False,
            "test_time": datetime.now().isoformat()
        }


def get_market_data(symbol: str) -> Optional[Dict]:
    """
    Get comprehensive market data for a symbol
    
    Args:
        symbol: Trading pair symbol
        
    Returns:
        Combined market data dict or None if failed
    """
    try:
        # Get candles (last 50 for analysis)
        candles = get_bybit_candles(symbol, "15", 50)
        
        # Get ticker data
        ticker = get_ticker_data(symbol)
        
        # Get orderbook
        orderbook = get_orderbook(symbol, 25)
        
        if not candles or not ticker:
            return None
        
        # Combine data
        market_data = {
            "symbol": symbol,
            "candles": candles,
            "ticker": ticker,
            "orderbook": orderbook,
            "timestamp": datetime.now().isoformat()
        }
        
        return market_data
        
    except Exception as e:
        print(f"[API] Error fetching market data for {symbol}: {e}")
        return None


def main():
    """Test API utilities"""
    print("Testing API utilities...")
    
    # Test connection
    status = test_api_connection()
    print(f"API Status: {status}")
    
    if status.get("overall"):
        print("API connection successful")
        
        # Test market data
        market_data = get_market_data("BTCUSDT")
        
        if market_data:
            print(f"Market data fetched: {len(market_data['candles'])} candles")
            print(f"Current price: ${market_data['ticker']['price']:.2f}")
        else:
            print("Failed to fetch market data")
    else:
        print("API connection failed")


if __name__ == "__main__":
    main()