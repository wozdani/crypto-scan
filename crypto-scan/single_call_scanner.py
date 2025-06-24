#!/usr/bin/env python3
"""
Single Call Scanner - Guarantees exactly ONE API call per token
No external dependencies, no duplicate calls
"""

import asyncio
import aiohttp
import time
import json
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

class SingleCallScanner:
    """Scanner that guarantees exactly one API call per token"""
    
    def __init__(self, max_concurrent: int = 15):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.api_calls_made = 0
        self.processed_symbols = set()
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=4, connect=1.5)
        connector = aiohttp.TCPConnector(limit=30, limit_per_host=10)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        print(f"Total API calls made: {self.api_calls_made}")
    
    async def fetch_ticker_only(self, symbol: str) -> Optional[Dict]:
        """Single ticker API call - NOTHING ELSE"""
        if symbol in self.processed_symbols:
            print(f"WARNING: {symbol} already processed - skipping duplicate")
            return None
            
        self.processed_symbols.add(symbol)
        
        async with self.semaphore:
            try:
                url = "https://api.bybit.com/v5/market/tickers"
                params = {"category": "spot", "symbol": symbol}
                
                # This is the ONLY API call for this symbol
                self.api_calls_made += 1
                print(f"[API CALL #{self.api_calls_made}] Fetching {symbol}")
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        print(f"[{symbol}] API returned {response.status}")
                        return None
                    
                    data = await response.json()
                    if not data.get("result", {}).get("list"):
                        print(f"[{symbol}] No ticker data")
                        return None
                    
                    ticker = data["result"]["list"][0]
                    price = float(ticker.get("lastPrice", 0))
                    volume = float(ticker.get("volume24h", 0))
                    
                    if price <= 0:
                        print(f"[{symbol}] Invalid price: {price}")
                        return None
                    
                    if volume < 50_000:  # Lower threshold for testing
                        print(f"[{symbol}] Low volume: ${volume:,.0f}")
                        return None
                    
                    print(f"[{symbol}] SUCCESS: ${price:.6f}, Volume: ${volume:,.0f}")
                    
                    return {
                        "symbol": symbol,
                        "price": price,
                        "volume": volume,
                        "price_change_24h": float(ticker.get("price24hPcnt", 0)),
                        "high_24h": float(ticker.get("highPrice24h", price)),
                        "low_24h": float(ticker.get("lowPrice24h", price))
                    }
                    
            except asyncio.TimeoutError:
                print(f"[{symbol}] Timeout")
                return None
            except Exception as e:
                print(f"[{symbol}] Error: {e}")
                return None
    
    def calculate_simple_score(self, data: Dict) -> float:
        """Calculate score from fetched data only"""
        score = 0.0
        volume = data["volume"]
        price_change = abs(data["price_change_24h"])
        
        # Volume scoring
        if volume > 5_000_000:
            score += 40
        elif volume > 1_000_000:
            score += 25
        elif volume > 500_000:
            score += 15
        
        # Price movement scoring
        if price_change > 15:
            score += 35
        elif price_change > 10:
            score += 25
        elif price_change > 5:
            score += 15
        
        # Volatility scoring
        price_range = (data["high_24h"] - data["low_24h"]) / data["price"]
        if price_range > 0.2:
            score += 20
        elif price_range > 0.1:
            score += 10
        
        return min(100, score)
    
    async def scan_symbol_complete(self, symbol: str) -> Optional[Dict]:
        """Complete scan with single API call"""
        start_time = time.time()
        
        # Single API call
        ticker_data = await self.fetch_ticker_only(symbol)
        if not ticker_data:
            return None
        
        # Score calculation
        score = self.calculate_simple_score(ticker_data)
        
        # Save result
        self.save_result_simple(symbol, score, ticker_data)
        
        scan_time = time.time() - start_time
        
        return {
            "symbol": symbol,
            "score": score,
            "price": ticker_data["price"],
            "volume": ticker_data["volume"],
            "scan_time": scan_time
        }
    
    def save_result_simple(self, symbol: str, score: float, ticker_data: Dict):
        """Save without complex dependencies"""
        try:
            result = {
                "symbol": symbol,
                "final_score": score,
                "price_usd": ticker_data["price"],
                "volume_24h": ticker_data["volume"],
                "timestamp": datetime.now().isoformat(),
                "scan_method": "single_call"
            }
            
            os.makedirs("data/scores", exist_ok=True)
            with open(f"data/scores/{symbol}_single.json", "w") as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Save error for {symbol}: {e}")

async def run_single_call_scan():
    """Run scan with guaranteed single API calls"""
    print(f"Starting single-call scan at {datetime.now().strftime('%H:%M:%S')}")
    
    # Hardcoded test symbols to avoid cache dependencies
    test_symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT",
        "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT",
        "LTCUSDT", "DOTUSDT", "UNIUSDT", "AAVEUSDT", "ATOMUSDT"
    ]
    
    print(f"Scanning {len(test_symbols)} symbols with single API calls")
    
    async with SingleCallScanner(max_concurrent=10) as scanner:
        start_time = time.time()
        
        tasks = [scanner.scan_symbol_complete(symbol) for symbol in test_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[{i+1}/{len(test_symbols)}] {test_symbols[i]}: Error - {result}")
            elif result:
                successful_results.append(result)
                print(f"[{i+1}/{len(test_symbols)}] {result['symbol']}: Score {result['score']:.1f} ({result['scan_time']:.3f}s)")
            else:
                print(f"[{i+1}/{len(test_symbols)}] {test_symbols[i]}: Skipped")
        
        total_time = time.time() - start_time
        print(f"\nScan Summary:")
        print(f"- Processed: {len(successful_results)}/{len(test_symbols)}")
        print(f"- Total time: {total_time:.1f}s")
        print(f"- Performance: {len(test_symbols)/total_time:.1f} tokens/second")
        print(f"- API calls made: {scanner.api_calls_made}")
        print(f"- Calls per symbol: {scanner.api_calls_made/len(test_symbols):.1f}")
        
        return successful_results

async def main():
    """Main function"""
    print("Starting Single Call Scanner Test")
    
    try:
        results = await run_single_call_scan()
        
        if results:
            print(f"\nTop performers:")
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            for i, result in enumerate(sorted_results[:5], 1):
                print(f"{i}. {result['symbol']}: {result['score']:.1f} points")
        
        print("\nTest completed - checking for duplicate API calls...")
        
    except KeyboardInterrupt:
        print("Scan interrupted")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())