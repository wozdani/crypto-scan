"""
Async Data Fetchers - aiohttp implementation for parallel I/O
Replaces synchronous requests.get() with async aiohttp calls
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import os

load_dotenv()

class AsyncDataFetcher:
    """Async data fetcher with connection pooling and rate limiting"""
    
    def __init__(self, max_concurrent: int = 20):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=8, connect=3)
        connector = aiohttp.TCPConnector(
            limit=100,           # Total connection pool size
            limit_per_host=20,   # Max connections per host
            ttl_dns_cache=300,   # DNS cache TTL
            use_dns_cache=True,
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": "CryptoScanner/1.0"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_bybit_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch ticker data from Bybit V5 API"""
        async with self.semaphore:
            try:
                url = "https://api.bybit.com/v5/market/tickers"
                params = {"category": "spot", "symbol": symbol}
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    if not data.get("result", {}).get("list"):
                        return None
                    
                    ticker = data["result"]["list"][0]
                    return {
                        "symbol": symbol,
                        "price": float(ticker.get("lastPrice", 0)),
                        "volume_24h": float(ticker.get("volume24h", 0)),
                        "price_change_24h": float(ticker.get("price24hPcnt", 0)),
                        "high_24h": float(ticker.get("highPrice24h", 0)),
                        "low_24h": float(ticker.get("lowPrice24h", 0)),
                        "bid": float(ticker.get("bid1Price", 0)),
                        "ask": float(ticker.get("ask1Price", 0))
                    }
                    
            except Exception as e:
                print(f"Error fetching ticker for {symbol}: {e}")
                return None
    
    async def fetch_bybit_candles(self, symbol: str, interval: str = "15", limit: int = 20) -> Optional[List[Dict]]:
        """Fetch candlestick data from Bybit V5 API"""
        async with self.semaphore:
            try:
                url = "https://api.bybit.com/v5/market/kline"
                params = {
                    "category": "spot",
                    "symbol": symbol,
                    "interval": interval,
                    "limit": str(limit)
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    if not data.get("result", {}).get("list"):
                        return None
                    
                    candles = []
                    for candle in data["result"]["list"]:
                        candles.append({
                            "timestamp": int(candle[0]),
                            "open": float(candle[1]),
                            "high": float(candle[2]),
                            "low": float(candle[3]),
                            "close": float(candle[4]),
                            "volume": float(candle[5]),
                            "turnover": float(candle[6]) if len(candle) > 6 else 0
                        })
                    
                    return candles
                    
            except Exception as e:
                print(f"Error fetching candles for {symbol}: {e}")
                return None
    
    async def fetch_bybit_orderbook(self, symbol: str, depth: int = 10) -> Optional[Dict]:
        """Fetch orderbook data from Bybit V5 API"""
        async with self.semaphore:
            try:
                url = "https://api.bybit.com/v5/market/orderbook"
                params = {
                    "category": "spot",
                    "symbol": symbol,
                    "limit": str(depth)
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    result = data.get("result", {})
                    
                    bids = []
                    asks = []
                    
                    if result.get("b"):
                        for bid in result["b"]:
                            bids.append({
                                "price": float(bid[0]),
                                "size": float(bid[1])
                            })
                    
                    if result.get("a"):
                        for ask in result["a"]:
                            asks.append({
                                "price": float(ask[0]),
                                "size": float(ask[1])
                            })
                    
                    return {
                        "symbol": symbol,
                        "bids": bids,
                        "asks": asks,
                        "best_bid": bids[0]["price"] if bids else 0,
                        "best_ask": asks[0]["price"] if asks else 0,
                        "spread": asks[0]["price"] - bids[0]["price"] if bids and asks else 0
                    }
                    
            except Exception as e:
                print(f"Error fetching orderbook for {symbol}: {e}")
                return None
    
    async def fetch_symbol_complete_data(self, symbol: str) -> Optional[Dict]:
        """Fetch complete market data for symbol (ticker + candles + orderbook)"""
        try:
            # Parallel fetch all data types
            ticker_task = self.fetch_bybit_ticker(symbol)
            candles_task = self.fetch_bybit_candles(symbol, "15", 20)
            orderbook_task = self.fetch_bybit_orderbook(symbol, 10)
            
            ticker, candles, orderbook = await asyncio.gather(
                ticker_task, candles_task, orderbook_task,
                return_exceptions=True
            )
            
            # Check for errors
            if isinstance(ticker, Exception) or not ticker:
                return None
            if isinstance(candles, Exception):
                candles = []
            if isinstance(orderbook, Exception):
                orderbook = {"bids": [], "asks": [], "best_bid": 0, "best_ask": 0}
            
            # Combine all data
            complete_data = {
                "symbol": symbol,
                "success": True,
                "price_usd": ticker["price"],
                "volume": ticker["volume_24h"],
                "volume_24h": ticker["volume_24h"],
                "price_change_24h": ticker["price_change_24h"],
                "high_24h": ticker["high_24h"],
                "low_24h": ticker["low_24h"],
                "best_bid": orderbook["best_bid"] or ticker["bid"],
                "best_ask": orderbook["best_ask"] or ticker["ask"],
                "spread": orderbook.get("spread", 0),
                "candles": candles,
                "orderbook": orderbook,
                "recent_volumes": [c["volume"] for c in candles[-7:]] if candles else [],
                "compressed": False  # For compatibility with existing code
            }
            
            return complete_data
            
        except Exception as e:
            print(f"Error fetching complete data for {symbol}: {e}")
            return None

# Async wrapper function for compatibility with existing code
async def async_get_market_data(symbol: str) -> Tuple[bool, Dict, float, bool]:
    """Async version of get_market_data() function"""
    try:
        async with AsyncDataFetcher(max_concurrent=20) as fetcher:
            data = await fetcher.fetch_symbol_complete_data(symbol)
            
            if not data:
                return False, {}, 0.0, False
            
            return True, data, data["price_usd"], False
            
    except Exception as e:
        print(f"Error in async_get_market_data for {symbol}: {e}")
        return False, {}, 0.0, False

# Batch processing function
async def async_get_market_data_batch(symbols: List[str], max_concurrent: int = 25) -> Dict[str, Tuple]:
    """Fetch market data for multiple symbols concurrently"""
    results = {}
    
    async with AsyncDataFetcher(max_concurrent=max_concurrent) as fetcher:
        tasks = [fetcher.fetch_symbol_complete_data(symbol) for symbol in symbols]
        
        symbol_data_pairs = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, data in zip(symbols, symbol_data_pairs):
            if isinstance(data, Exception) or not data:
                results[symbol] = (False, {}, 0.0, False)
            else:
                results[symbol] = (True, data, data["price_usd"], False)
    
    return results