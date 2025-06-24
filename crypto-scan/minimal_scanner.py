#!/usr/bin/env python3
"""
Minimal Scanner - eliminates ALL duplicate API calls
Single API call per token, no complex dependencies
"""

import asyncio
import aiohttp
import time
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# Import only essential modules without complex dependencies
from utils.bybit_cache_manager import get_bybit_symbols_cached

class MinimalScanner:
    """Minimal scanner with zero duplicate API calls"""
    
    def __init__(self, max_concurrent: int = 20):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=5, connect=2)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=15)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scan_single_token(self, symbol: str) -> Optional[Dict]:
        """Single API call per token - no duplicates"""
        async with self.semaphore:
            try:
                start_time = time.time()
                
                # Single API call only
                url = "https://api.bybit.com/v5/market/tickers"
                params = {"category": "spot", "symbol": symbol}
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    if not data.get("result", {}).get("list"):
                        return None
                    
                    ticker = data["result"]["list"][0]
                    price = float(ticker.get("lastPrice", 0))
                    volume = float(ticker.get("volume24h", 0))
                    
                    if price <= 0 or volume < 100_000:
                        return None
                    
                    # Simple scoring without external dependencies
                    score = 0
                    if volume > 1_000_000:
                        score += 20
                    if volume > 5_000_000:
                        score += 30
                    
                    price_change = float(ticker.get("price24hPcnt", 0))
                    if abs(price_change) > 5:
                        score += 25
                    if price_change > 10:
                        score += 35
                    
                    scan_time = time.time() - start_time
                    
                    # Save minimal result to file
                    self.save_result(symbol, score, price, volume)
                    
                    return {
                        "symbol": symbol,
                        "score": score,
                        "price": price,
                        "volume": volume,
                        "scan_time": scan_time
                    }
                    
            except Exception:
                return None
    
    def save_result(self, symbol: str, score: float, price: float, volume: float):
        """Save result without complex dependencies"""
        try:
            result = {
                "symbol": symbol,
                "score": score,
                "price": price,
                "volume": volume,
                "timestamp": datetime.now().isoformat()
            }
            
            os.makedirs("data/results", exist_ok=True)
            with open(f"data/results/{symbol}_latest.json", "w") as f:
                import json
                json.dump(result, f)
                
        except Exception:
            pass
    
    async def scan_all(self, symbols: List[str]) -> List[Dict]:
        """Scan all symbols with progress tracking"""
        print(f"Scanning {len(symbols)} symbols with {self.max_concurrent} concurrent connections")
        
        start_time = time.time()
        tasks = [self.scan_single_token(symbol) for symbol in symbols]
        
        results = []
        completed = 0
        
        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            
            if result:
                results.append(result)
                print(f"[{completed}/{len(symbols)}] {result['symbol']}: Score {result['score']:.1f} ({result['scan_time']:.3f}s)")
            else:
                print(f"[{completed}/{len(symbols)}] Skipped")
        
        total_time = time.time() - start_time
        print(f"Scan completed: {len(results)}/{len(symbols)} in {total_time:.1f}s")
        print(f"Performance: {len(symbols)/total_time:.1f} tokens/second")
        
        return results

async def minimal_scan_cycle():
    """Minimal scan cycle"""
    print(f"Starting minimal scan at {datetime.now().strftime('%H:%M:%S')}")
    
    symbols = get_bybit_symbols_cached()
    print(f"Fetched {len(symbols)} symbols")
    
    async with MinimalScanner(max_concurrent=20) as scanner:
        results = await scanner.scan_all(symbols)
    
    return results

def wait_for_next_candle():
    """Wait for next candle"""
    now = datetime.now(timezone.utc)
    next_candle = now.replace(second=5, microsecond=0) + timedelta(minutes=15 - (now.minute % 15))
    wait_seconds = (next_candle - now).total_seconds()
    
    if wait_seconds > 0:
        print(f"Waiting {wait_seconds:.1f}s for next candle")
        time.sleep(wait_seconds)

async def main():
    """Main loop"""
    print("Starting Minimal Crypto Scanner")
    
    try:
        while True:
            try:
                await minimal_scan_cycle()
                wait_for_next_candle()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(60)
                
    except KeyboardInterrupt:
        print("Scanner stopped")

if __name__ == "__main__":
    asyncio.run(main())