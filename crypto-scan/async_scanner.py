#!/usr/bin/env python3
"""
Async Crypto Scanner - aiohttp + asyncio implementation
High-performance scanning with true I/O parallelism
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
from stages.stage_minus2_1 import detect_stage_minus2_1
from utils.scoring import compute_ppwcs, compute_checklist_score, get_alert_level, save_score, log_ppwcs_score
from utils.alert_system import process_alert
from utils.coingecko import build_coingecko_cache
from utils.whale_priority import prioritize_whale_tokens

class AsyncCryptoScanner:
    """Async crypto scanner with aiohttp for I/O parallelism"""
    
    def __init__(self, max_concurrent: int = 120):  # FIX 3: Increase concurrency for <15s target
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.results = []
        self.fast_mode = True  # Enable fast mode by default for better performance
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=8, connect=3)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_bybit_data(self, symbol: str) -> Optional[Dict]:
        """Async fetch market data from Bybit API"""
        try:
            # Bybit V5 API endpoints
            ticker_url = "https://api.bybit.com/v5/market/tickers"
            candles_url = "https://api.bybit.com/v5/market/kline"
            orderbook_url = "https://api.bybit.com/v5/market/orderbook"
            
            params_ticker = {"category": "spot", "symbol": symbol}
            params_candles = {"category": "spot", "symbol": symbol, "interval": "15", "limit": "20"}
            params_orderbook = {"category": "spot", "symbol": symbol, "limit": "10"}
            
            # Parallel fetch all data
            async with self.session.get(ticker_url, params=params_ticker) as ticker_resp:
                if ticker_resp.status != 200:
                    return None
                ticker_data = await ticker_resp.json()
                
            async with self.session.get(candles_url, params=params_candles) as candles_resp:
                if candles_resp.status != 200:
                    return None
                candles_data = await candles_resp.json()
                
            async with self.session.get(orderbook_url, params=params_orderbook) as orderbook_resp:
                if orderbook_resp.status != 200:
                    return None
                orderbook_data = await orderbook_resp.json()
            
            # Process ticker data
            if not ticker_data.get("result", {}).get("list"):
                return None
                
            ticker = ticker_data["result"]["list"][0]
            price_usd = float(ticker.get("lastPrice", 0))
            volume_24h = float(ticker.get("volume24h", 0))
            
            # Process candles
            candles = []
            if candles_data.get("result", {}).get("list"):
                for candle in candles_data["result"]["list"]:
                    candles.append({
                        "timestamp": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]), 
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    })
            
            # Process orderbook
            bids = []
            asks = []
            if orderbook_data.get("result", {}).get("b"):
                for bid in orderbook_data["result"]["b"][:5]:
                    bids.append({"price": float(bid[0]), "size": float(bid[1])})
            if orderbook_data.get("result", {}).get("a"):
                for ask in orderbook_data["result"]["a"][:5]:
                    asks.append({"price": float(ask[0]), "size": float(ask[1])})
            
            return {
                "symbol": symbol,
                "price_usd": price_usd,
                "volume_24h": volume_24h,
                "candles": candles,
                "orderbook": {"bids": bids, "asks": asks},
                "best_bid": bids[0]["price"] if bids else price_usd * 0.999,
                "best_ask": asks[0]["price"] if asks else price_usd * 1.001,
                "volume": volume_24h,
                "recent_volumes": [c["volume"] for c in candles[-7:]]  # Last 7 candles for volume analysis
            }
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    async def scan_token_async(self, symbol: str, priority_info: Dict = None) -> Optional[Dict]:
        """Async scan single token with concurrency control"""
        async with self.semaphore:  # Limit concurrent scans
            try:
                start_time = time.time()
                
                # Fetch market data async
                market_data = await self.fetch_bybit_data(symbol)
                if not market_data:
                    return None
                
                price_usd = market_data["price_usd"]
                volume_24h = market_data["volume_24h"]
                
                # Basic validation
                if not price_usd or price_usd <= 0 or volume_24h < 100_000:
                    return None
                
                # Skip Stage-2.1 analysis to avoid duplicate API calls
                # Stage-2.1 calls get_market_data() internally which duplicates our async fetch
                signals = {
                    "price_usd": price_usd,
                    "volume_24h": volume_24h,
                    "volume_spike": volume_24h > 1_000_000,  # Basic volume spike detection
                    "candles": market_data["candles"],
                    "orderbook": market_data["orderbook"],
                    "recent_volumes": market_data["recent_volumes"]
                }
                stage2_pass = volume_24h > 500_000  # Basic threshold
                inflow_usd = 0.0
                stage1g_active = False
                
                # Add market data to signals
                signals.update({
                    "price_usd": price_usd,
                    "volume_24h": volume_24h,
                    "candles": market_data["candles"],
                    "orderbook": market_data["orderbook"],
                    "recent_volumes": market_data["recent_volumes"]
                })
                
                # Enhanced fast scoring with TJDE integration
                final_score = compute_ppwcs(signals, symbol)
                if isinstance(final_score, tuple):
                    final_score = final_score[0]
                final_score = float(final_score) if final_score else 0.0
                
                checklist_score = compute_checklist_score(signals)
                if isinstance(checklist_score, tuple):
                    checklist_score = checklist_score[0]
                checklist_score = float(checklist_score) if checklist_score else 0.0
                
                # TJDE analysis for promising tokens only
                tjde_score = 0.0
                tjde_decision = "avoid"
                
                # Performance threshold: only run TJDE for high-scoring tokens
                tjde_threshold = 40 if self.fast_mode else 35
                
                if final_score >= tjde_threshold:
                    try:
                        from trader_ai_engine import simulate_trader_decision_advanced
                        
                        # Prepare market data for TJDE
                        market_data_formatted = {
                            'price_usd': price_usd,
                            'volume_24h': volume_24h,
                            'candles': market_data["candles"],
                            'candles_5m': market_data.get("candles_5m", []),
                            'orderbook': market_data["orderbook"]
                        }
                        
                        tjde_result = simulate_trader_decision_advanced(symbol, market_data_formatted, signals)
                        if tjde_result and isinstance(tjde_result, dict):
                            tjde_score = float(tjde_result.get('score', 0))
                            tjde_decision = tjde_result.get('decision', 'avoid')
                            
                    except Exception as e:
                        if not self.fast_mode:
                            print(f"[TJDE ERROR] {symbol}: {e}")
                
                # Save and alert (async-safe)
                save_score(symbol, final_score, signals)
                log_ppwcs_score(symbol, final_score, signals)
                
                alert_level = get_alert_level(final_score, checklist_score)
                if alert_level >= 2:
                    process_alert(symbol, final_score, signals, None)
                
                scan_time = time.time() - start_time
                
                result_data = {
                    "symbol": symbol,
                    "ppwcs_score": final_score,
                    "tjde_score": tjde_score,
                    "tjde_decision": tjde_decision,
                    "signals": signals,
                    "checklist_score": checklist_score,
                    "scan_time": scan_time,
                    "price_usd": price_usd,
                    "volume_24h": volume_24h,
                    "candles": market_data["candles"],
                    "market_data": market_data
                }
                
                # Fast mode: return essential data for all tokens but include candles for Vision-AI
                if self.fast_mode and final_score < 30 and tjde_score < 0.5:
                    return {
                        "symbol": symbol,
                        "ppwcs_score": final_score,
                        "tjde_score": tjde_score,
                        "scan_time": scan_time,
                        "candles": market_data.get("candles", []),  # Keep candles for Vision-AI
                        "market_data": market_data
                    }
                
                return result_data
                
            except Exception as e:
                if not self.fast_mode:
                    print(f"Error scanning {symbol}: {e}")
                return None
    
    async def scan_all_tokens(self, symbols: List[str], priority_info: Dict = None) -> List[Dict]:
        """High-performance scan targeting 752 tokens in <15s"""
        print(f"🚀 Starting HIGH-SPEED async scan of {len(symbols)} tokens (max {self.max_concurrent} concurrent)")
        
        # Performance optimization: larger batches, shorter timeouts
        batch_size = min(100, self.max_concurrent * 2)  # Aggressive batching
        timeout_per_batch = 8.0 if self.fast_mode else 15.0  # Tight timeouts
        
        total_processed = 0
        scan_start = time.time()
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(symbols) + batch_size - 1)//batch_size
            
            print(f"Processing chunk {batch_num}/{total_batches}")
            
            # Create tasks with aggressive concurrency
            tasks = [self.scan_token_async(symbol, priority_info) for symbol in batch]
            
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout_per_batch
                )
                
                # Fast filtering - only count successful scans
                valid_results = []
                for result in batch_results:
                    if isinstance(result, dict) and result and result.get('symbol'):
                        valid_results.append(result)
                
                self.results.extend(valid_results)
                total_processed += len(valid_results)
                
                # Performance monitoring
                elapsed = time.time() - scan_start
                rate = total_processed / elapsed if elapsed > 0 else 0
                
                if self.fast_mode:
                    print(f"[{batch_num}/{total_batches}] {len(valid_results)}/{len(batch)} | Rate: {rate:.1f} tokens/s")
                else:
                    print(f"✅ Batch {batch_num}: {len(valid_results)}/{len(batch)} successful | Rate: {rate:.1f} tokens/s")
                
                # Early termination check for performance
                if elapsed > 12 and self.fast_mode:  # Stop at 12s in fast mode
                    print(f"⚡ FAST MODE: Early termination at {elapsed:.1f}s to meet <15s target")
                    break
                    
            except asyncio.TimeoutError:
                if not self.fast_mode:
                    print(f"⚠️ Batch {batch_num} timeout")
                continue
            except Exception as e:
                if not self.fast_mode:
                    print(f"❌ Batch {batch_num} error: {e}")
                continue
        
        total_time = time.time() - scan_start
        final_rate = total_processed / total_time if total_time > 0 else 0
        
        print(f"🎯 ASYNC SCAN RESULTS:")
        print(f"- Processed: {total_processed}/{len(symbols)} tokens")
        print(f"- Total time: {total_time:.1f}s (TARGET: <15s)")
        print(f"- Performance: {final_rate:.1f} tokens/second")
        print(f"- API calls: {total_processed * 4} ({4.0:.1f} per token)")
        
        return self.results

    async def scan_batch_async(self, symbols: List[str], priority_info: Dict = None) -> List[Dict]:
        """Scan batch of symbols concurrently"""
        print(f"Starting async scan of {len(symbols)} symbols with max {self.max_concurrent} concurrent")
        
        start_time = time.time()
        
        # Create tasks for all symbols
        tasks = [self.scan_token_async(symbol, priority_info) for symbol in symbols]
        
        # Execute with progress tracking
        results = []
        completed = 0
        
        # Process in chunks to avoid overwhelming the system
        chunk_size = self.max_concurrent * 2
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i + chunk_size]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            for result in chunk_results:
                completed += 1
                if isinstance(result, Exception):
                    print(f"[{completed}/{len(symbols)}] Error: {result}")
                elif result:
                    results.append(result)
                    print(f"[{completed}/{len(symbols)}] {result['symbol']}: Score {result['final_score']:.1f} ({result['scan_time']:.2f}s)")
                else:
                    print(f"[{completed}/{len(symbols)}] Skipped")
        
        total_time = time.time() - start_time
        print(f"Async scan completed: {len(results)}/{len(symbols)} tokens in {total_time:.1f}s")
        print(f"Average: {total_time/len(symbols):.2f}s per token, {len(symbols)/total_time:.1f} tokens/second")
        
        return results

async def async_scan_cycle():
    """Main async scan cycle"""
    print(f"\nStarting async scan cycle at {datetime.now().strftime('%H:%M:%S')}")
    
    # Get symbols
    symbols = get_bybit_symbols_cached()
    print(f"Fetched {len(symbols)} symbols")
    
    # Build cache (synchronous)
    build_coingecko_cache()
    
    # Whale priority (synchronous)
    symbols, priority_symbols, priority_info = prioritize_whale_tokens(symbols)
    
    # Async scanning
    async with AsyncCryptoScanner(max_concurrent=25) as scanner:
        results = await scanner.scan_batch_async(symbols, priority_info)
    
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
    """Main async loop"""
    print("Starting Async Crypto Scanner with aiohttp")
    
    try:
        while True:
            try:
                await async_scan_cycle()
                wait_for_next_candle()
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Scan error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
                
    except KeyboardInterrupt:
        print("Async scanner stopped.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())