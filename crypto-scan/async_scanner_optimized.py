#!/usr/bin/env python3
"""
Optimized Async Scanner - eliminates duplicate API calls
Fetches data once and processes all analysis with that data
"""

import asyncio
import aiohttp
import time
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# Import essential modules
from utils.bybit_cache_manager import get_bybit_symbols_cached
from utils.scoring import compute_ppwcs, compute_checklist_score, get_alert_level, save_score, log_ppwcs_score
from utils.alert_system import process_alert
from utils.coingecko import build_coingecko_cache
from utils.whale_priority import prioritize_whale_tokens

class OptimizedAsyncScanner:
    """Optimized async scanner - fetch once, analyze once"""
    
    def __init__(self, max_concurrent: int = 30):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=6, connect=2)
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=25,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": "CryptoScanner/2.0"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_single_token_data(self, symbol: str) -> Optional[Dict]:
        """Fetch all required data for token in one call - no duplicates"""
        async with self.semaphore:
            try:
                # Single API call to get ticker data
                url = "https://api.bybit.com/v5/market/tickers"
                params = {"category": "spot", "symbol": symbol}
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    if not data.get("result", {}).get("list"):
                        return None
                    
                    ticker = data["result"]["list"][0]
                    price_usd = float(ticker.get("lastPrice", 0))
                    volume_24h = float(ticker.get("volume24h", 0))
                    
                    # Basic validation
                    if price_usd <= 0 or volume_24h < 100_000:
                        return None
                    
                    # Return processed data ready for analysis
                    return {
                        "symbol": symbol,
                        "price_usd": price_usd,
                        "volume_24h": volume_24h,
                        "high_24h": float(ticker.get("highPrice24h", price_usd)),
                        "low_24h": float(ticker.get("lowPrice24h", price_usd)),
                        "price_change_24h": float(ticker.get("price24hPcnt", 0)),
                        "bid": float(ticker.get("bid1Price", 0)),
                        "ask": float(ticker.get("ask1Price", 0)),
                        "volume": volume_24h,
                        "best_bid": float(ticker.get("bid1Price", price_usd * 0.999)),
                        "best_ask": float(ticker.get("ask1Price", price_usd * 1.001))
                    }
                    
            except Exception as e:
                return None
    
    def analyze_token_fast(self, token_data: Dict) -> Dict:
        """Fast token analysis without external API calls"""
        symbol = token_data["symbol"]
        price_usd = token_data["price_usd"]
        volume_24h = token_data["volume_24h"]
        
        # Create signals based on fetched data
        signals = {
            "symbol": symbol,
            "price_usd": price_usd,
            "volume_24h": volume_24h,
            "volume": volume_24h,
            "high_24h": token_data["high_24h"],
            "low_24h": token_data["low_24h"],
            "price_change_24h": token_data["price_change_24h"],
            "best_bid": token_data["best_bid"],
            "best_ask": token_data["best_ask"],
            
            # Fast signal calculations
            "volume_spike": volume_24h > 1_000_000,
            "price_momentum": abs(token_data["price_change_24h"]) > 5.0,
            "high_volatility": (token_data["high_24h"] - token_data["low_24h"]) / price_usd > 0.1,
            "spread_normal": abs(token_data["ask"] - token_data["bid"]) / price_usd < 0.02,
            
            # Volume analysis
            "volume_tier": "high" if volume_24h > 5_000_000 else "medium" if volume_24h > 1_000_000 else "low",
            "liquidity_score": min(100, volume_24h / 50_000),  # 0-100 based on volume
            
            # Price action signals
            "near_24h_high": price_usd > token_data["high_24h"] * 0.95,
            "near_24h_low": price_usd < token_data["low_24h"] * 1.05,
            "mid_range": 0.3 < (price_usd - token_data["low_24h"]) / (token_data["high_24h"] - token_data["low_24h"]) < 0.7,
            
            # Basic momentum indicators
            "bullish_momentum": token_data["price_change_24h"] > 3.0,
            "bearish_momentum": token_data["price_change_24h"] < -3.0,
            "consolidation": abs(token_data["price_change_24h"]) < 2.0
        }
        
        return signals
    
    async def scan_token_optimized(self, symbol: str) -> Optional[Dict]:
        """Optimized token scan - single API call, fast analysis"""
        try:
            start_time = time.time()
            
            # Single data fetch
            token_data = await self.fetch_single_token_data(symbol)
            if not token_data:
                return None
            
            # Fast analysis without external calls
            signals = self.analyze_token_fast(token_data)
            
            # Fast scoring
            final_score = compute_ppwcs(signals, symbol)
            if isinstance(final_score, tuple):
                final_score = final_score[0]
            final_score = float(final_score) if final_score else 0.0
            
            checklist_score = compute_checklist_score(signals)
            if isinstance(checklist_score, tuple):
                checklist_score = checklist_score[0]
            checklist_score = float(checklist_score) if checklist_score else 0.0
            
            # Save and alert
            save_score(symbol, final_score, signals)
            log_ppwcs_score(symbol, final_score, signals)
            
            alert_level = get_alert_level(final_score, checklist_score)
            if alert_level >= 2:
                process_alert(symbol, final_score, signals, None)
            
            scan_time = time.time() - start_time
            
            return {
                "symbol": symbol,
                "final_score": final_score,
                "checklist_score": checklist_score,
                "signals": signals,
                "scan_time": scan_time,
                "price_usd": token_data["price_usd"],
                "volume_24h": token_data["volume_24h"]
            }
            
        except Exception as e:
            return None
    
    async def scan_batch_optimized(self, symbols: List[str]) -> List[Dict]:
        """Optimized batch scanning"""
        print(f"Starting optimized async scan of {len(symbols)} symbols with max {self.max_concurrent} concurrent")
        
        start_time = time.time()
        
        # Create tasks
        tasks = [self.scan_token_optimized(symbol) for symbol in symbols]
        
        # Execute with progress tracking
        results = []
        completed = 0
        
        # Process in smaller chunks for better memory management
        chunk_size = self.max_concurrent
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i + chunk_size]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            for result in chunk_results:
                completed += 1
                if isinstance(result, Exception):
                    print(f"[{completed}/{len(symbols)}] Error: {result}")
                elif result:
                    results.append(result)
                    print(f"[{completed}/{len(symbols)}] {result['symbol']}: Score {result['final_score']:.1f} ({result['scan_time']:.3f}s)")
                else:
                    print(f"[{completed}/{len(symbols)}] Skipped")
        
        total_time = time.time() - start_time
        tokens_per_second = len(symbols) / total_time if total_time > 0 else 0
        
        print(f"Optimized scan completed: {len(results)}/{len(symbols)} tokens in {total_time:.1f}s")
        print(f"Performance: {tokens_per_second:.1f} tokens/second, {total_time/len(symbols):.3f}s avg per token")
        
        return results

async def optimized_scan_cycle():
    """Optimized scan cycle"""
    print(f"\nStarting optimized scan cycle at {datetime.now().strftime('%H:%M:%S')}")
    
    # Get symbols
    symbols = get_bybit_symbols_cached()
    print(f"Scanning {len(symbols)} symbols with optimized async scanner")
    
    # Build cache (synchronous)
    build_coingecko_cache()
    
    # Whale priority (synchronous)
    symbols, priority_symbols, priority_info = prioritize_whale_tokens(symbols)
    
    # Optimized async scanning
    async with OptimizedAsyncScanner(max_concurrent=30) as scanner:
        results = await scanner.scan_batch_optimized(symbols)
    
    return results

def wait_for_next_candle():
    """Wait for next 15-minute candle"""
    now = datetime.now(timezone.utc)
    next_candle = now.replace(second=5, microsecond=0)
    
    if now.second >= 5:
        next_candle += timedelta(minutes=15 - (now.minute % 15))
    else:
        next_candle += timedelta(minutes=15 - (now.minute % 15))
    
    wait_seconds = (next_candle - now).total_seconds()
    
    if wait_seconds > 0:
        print(f"Waiting {wait_seconds:.1f}s for next candle at {next_candle.strftime('%H:%M:%S')} UTC")
        time.sleep(wait_seconds)

async def main():
    """Main optimized async loop"""
    print("Starting Optimized Async Crypto Scanner")
    
    try:
        while True:
            try:
                await optimized_scan_cycle()
                wait_for_next_candle()
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Scan error: {e}")
                await asyncio.sleep(60)
                
    except KeyboardInterrupt:
        print("Optimized scanner stopped.")

if __name__ == "__main__":
    asyncio.run(main())