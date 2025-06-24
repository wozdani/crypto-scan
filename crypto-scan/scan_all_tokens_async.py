#!/usr/bin/env python3
"""
Async All Tokens Scanner - Mass parallel scanning with semaphore control
Orchestrates scan_token_async() for hundreds of tokens simultaneously
Target: <15 seconds for 500+ tokens
"""

import asyncio
import aiohttp
import time
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scan_token_async import scan_token_async
from utils.bybit_cache_manager import get_bybit_symbols_cached
from utils.coingecko import build_coingecko_cache
from utils.whale_priority import prioritize_whale_tokens

class AsyncTokenScanner:
    """High-performance async scanner for all tokens"""
    
    def __init__(self, max_concurrent: int = 20):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.total_api_calls = 0
        self.successful_scans = 0
        
    async def __aenter__(self):
        """Setup async session"""
        timeout = aiohttp.ClientTimeout(total=8, connect=3)
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": "CryptoScanner/3.0"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async session"""
        if self.session:
            await self.session.close()
        print(f"Session closed. Total API calls: {self.total_api_calls}")

    async def limited_scan(self, symbol: str, priority_info: Dict = None) -> Optional[Dict]:
        """Scan single token with semaphore control"""
        async with self.semaphore:
            self.total_api_calls += 4  # ticker + 15m + 5m + orderbook
            result = await scan_token_async(symbol, self.session, priority_info)
            if result:
                self.successful_scans += 1
            return result

    async def scan_all_tokens(self, symbols: List[str], priority_info: Dict = None) -> List[Dict]:
        """
        Scan all tokens with parallel execution and progress tracking
        Core function replacing sequential scanning
        """
        print(f"🚀 Starting async scan of {len(symbols)} tokens (max {self.max_concurrent} concurrent)")
        start_time = time.time()
        
        # Create tasks for all symbols
        tasks = [self.limited_scan(symbol, priority_info) for symbol in symbols]
        
        # Process with progress tracking
        results = []
        completed = 0
        
        # Process in chunks for memory efficiency and progress display
        chunk_size = min(50, len(tasks))
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i + chunk_size]
            chunk_symbols = symbols[i:i + chunk_size]
            
            print(f"Processing chunk {i//chunk_size + 1}/{(len(tasks) + chunk_size - 1)//chunk_size}")
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            for j, result in enumerate(chunk_results):
                completed += 1
                symbol = chunk_symbols[j]
                
                if isinstance(result, Exception):
                    print(f"[{completed}/{len(symbols)}] {symbol}: ERROR - {result}")
                elif result:
                    results.append(result)
                    # Compact progress display
                    if completed % 10 == 0 or result.get("ppwcs_score", 0) >= 40:
                        print(f"[{completed}/{len(symbols)}] {symbol}: PPWCS {result['ppwcs_score']:.1f}, TJDE {result['tjde_decision']}")
                else:
                    if completed % 20 == 0:  # Less frequent logging for skipped tokens
                        print(f"[{completed}/{len(symbols)}] {symbol}: Skipped")
        
        total_time = time.time() - start_time
        tokens_per_second = len(symbols) / total_time if total_time > 0 else 0
        
        # Performance summary
        print(f"\n🎯 ASYNC SCAN RESULTS:")
        print(f"- Processed: {self.successful_scans}/{len(symbols)} tokens")
        print(f"- Total time: {total_time:.1f}s (TARGET: <15s)")
        print(f"- Performance: {tokens_per_second:.1f} tokens/second")
        print(f"- API calls: {self.total_api_calls} ({self.total_api_calls/len(symbols):.1f} per token)")
        
        # Top performers
        if results:
            sorted_results = sorted(results, key=lambda x: x.get('ppwcs_score', 0), reverse=True)
            print(f"\n🔥 TOP 10 PERFORMERS:")
            for i, result in enumerate(sorted_results[:10], 1):
                print(f"{i:2d}. {result['symbol']:12} PPWCS {result['ppwcs_score']:5.1f} TJDE {result['tjde_decision']:12} Vol ${result['volume_24h']:>12,.0f}")
        
        return results

async def async_scan_cycle():
    """Main async scan cycle - replaces sequential scan_cycle()"""
    print(f"Starting async scan cycle at {datetime.now().strftime('%H:%M:%S')}")
    
    # Prepare symbols and cache
    symbols = get_bybit_symbols_cached()
    print(f"Fetched {len(symbols)} symbols from cache")
    
    # Build CoinGecko cache
    build_coingecko_cache()
    
    # Whale prioritization
    symbols, priority_symbols, priority_info = prioritize_whale_tokens(symbols)
    if priority_symbols:
        print(f"Whale priority: {len(priority_symbols)} symbols flagged")
    
    # Execute async scan
    async with AsyncTokenScanner(max_concurrent=20) as scanner:
        results = await scanner.scan_all_tokens(symbols, priority_info)
    
    # Post-scan processing
    if results:
        # Save summary
        save_scan_summary(results)
        
        # Generate training charts for TOP 5 TJDE tokens only
        generate_top_tjde_charts(results)
        
        # Check for high-value setups
        high_score_setups = [r for r in results if r.get('ppwcs_score', 0) >= 50 or r.get('tjde_score', 0) >= 0.8]
        if high_score_setups:
            print(f"\n⚡ {len(high_score_setups)} HIGH-VALUE SETUPS DETECTED")
            for setup in high_score_setups:
                print(f"   {setup['symbol']}: PPWCS {setup['ppwcs_score']:.1f}, TJDE {setup['tjde_decision']}")
    
    return results

def generate_top_tjde_charts(results: List[Dict]):
    """Generate training charts and Vision-AI data for TOP 5 TJDE tokens only"""
    try:
        # Import Vision-AI pipeline
        from vision_ai_pipeline import generate_vision_ai_training_data
        
        # Generate comprehensive Vision-AI training data
        training_pairs = generate_vision_ai_training_data(results)
        
        if training_pairs > 0:
            print(f"\n🎯 VISION-AI PIPELINE: Generated {training_pairs} training pairs")
        else:
            print("\n🎯 VISION-AI PIPELINE: No training data generated")
        
        # Original chart generation logic (kept for compatibility)
        valid_results = [r for r in results if r.get('tjde_score', 0) > 0]
        
        if not valid_results:
            print("[CHART GEN] No valid TJDE results for chart generation")
            return
            
        # Sort by TJDE score descending and take TOP 5
        top_5_tjde = sorted(valid_results, key=lambda x: x.get('tjde_score', 0), reverse=True)[:5]
        
        print(f"\n📊 GENERATING CHARTS FOR TOP 5 TJDE TOKENS:")
        
        chart_count = 0
        for i, entry in enumerate(top_5_tjde, 1):
            symbol = entry.get('symbol', 'UNKNOWN')
            tjde_score = entry.get('tjde_score', 0)
            tjde_decision = entry.get('tjde_decision', 'unknown')
            market_data = entry.get('market_data', {})
            
            print(f"{i}. {symbol}: TJDE {tjde_score:.3f} ({tjde_decision})")
            
            try:
                from chart_generator import generate_alert_focused_training_chart
                
                # Extract candle data from market_data
                candles_15m = market_data.get('candles', [])
                if not candles_15m or len(candles_15m) < 20:
                    print(f"   [SKIP] {symbol}: Insufficient candle data")
                    continue
                
                # Generate chart
                chart_path = generate_alert_focused_training_chart(
                    symbol=symbol,
                    candles_15m=candles_15m,
                    tjde_score=tjde_score,
                    tjde_phase=entry.get('market_phase', 'unknown'),
                    tjde_decision=tjde_decision,
                    tjde_clip_confidence=entry.get('clip_confidence', None),
                    setup_label=entry.get('setup_type', None)
                )
                
                if chart_path:
                    chart_count += 1
                    print(f"   ✅ Chart generated: {chart_path}")
                else:
                    print(f"   ❌ Chart generation failed")
                    
            except Exception as chart_e:
                print(f"   ❌ Chart error for {symbol}: {chart_e}")
        
        print(f"📊 Generated {chart_count}/{len(top_5_tjde)} training charts for TOP TJDE tokens")
        
    except Exception as e:
        print(f"[CHART GEN ERROR] {e}")

def save_scan_summary(results: List[Dict]):
    """Save scan summary for dashboard"""
    try:
        summary = {
            "scan_time": datetime.now().isoformat(),
            "total_processed": len(results),
            "scan_method": "async_parallel",
            "top_performers": sorted(results, key=lambda x: x.get('ppwcs_score', 0), reverse=True)[:20],
            "high_tjde_scores": [r for r in results if r.get('tjde_score', 0) >= 0.7],
            "alerts_sent": len([r for r in results if r.get('alert_sent', False)])
        }
        
        os.makedirs("data/scan_summaries", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"data/scan_summaries/async_scan_{timestamp}.json", "w") as f:
            import json
            json.dump(summary, f, indent=2)
            
    except Exception as e:
        print(f"Error saving scan summary: {e}")

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
    """Main async scanning loop"""
    print("🚀 ASYNC TOKEN SCANNER STARTED")
    print("Target: <15 seconds for 500+ tokens")
    print("Replacing sequential scanning with parallel async execution")
    
    try:
        while True:
            try:
                await async_scan_cycle()
                wait_for_next_candle()
            except KeyboardInterrupt:
                print("\nShutting down async scanner...")
                break
            except Exception as e:
                print(f"Scan cycle error: {e}")
                await asyncio.sleep(60)
                
    except KeyboardInterrupt:
        print("Async scanner stopped.")

# Direct execution functions for integration
async def scan_symbols_async(symbols: List[str], max_concurrent: int = 15) -> List[Dict]:
    """Direct function to scan list of symbols asynchronously with configurable concurrency"""
    scan_start_time = time.time()
    try:
        async with AsyncTokenScanner(max_concurrent=max_concurrent) as scanner:
            results = await scanner.scan_all_tokens(symbols)
        
        scan_duration = time.time() - scan_start_time
        print(f"scan_symbols_async completed in {scan_duration:.1f}s")
        return results
        
    except Exception as e:
        scan_duration = time.time() - scan_start_time
        print(f"scan_symbols_async error after {scan_duration:.1f}s: {e}")
        return []

def run_async_scan(symbols: List[str]) -> List[Dict]:
    """Synchronous wrapper for async scanning"""
    return asyncio.run(scan_symbols_async(symbols))

if __name__ == "__main__":
    asyncio.run(main())