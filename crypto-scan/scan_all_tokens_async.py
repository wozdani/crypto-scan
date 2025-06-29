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
from utils.scan_error_reporter import (
    initialize_scan_session, log_scan_error, log_global_error, log_token_error,
    log_api_error, log_clip_error, log_chart_error,
    print_error_summary, save_error_report, get_error_count
)

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
                    log_token_error(symbol, "ASYNC_SCAN", f"Scan failed: {result}")
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
    
    # Initialize error reporting for this scan session
    initialize_scan_session()
    
    # Prepare symbols and cache
    try:
        symbols = get_bybit_symbols_cached()
        print(f"Fetched {len(symbols)} symbols from cache")
    except Exception as e:
        log_global_error("Symbol Cache", f"Failed to load symbols: {e}")
        return []
    
    # Build CoinGecko cache
    try:
        build_coingecko_cache()
    except Exception as e:
        log_global_error("CoinGecko Cache", f"Cache building failed: {e}")
    
    # Whale prioritization
    try:
        symbols, priority_symbols, priority_info = prioritize_whale_tokens(symbols)
        if priority_symbols:
            print(f"Whale priority: {len(priority_symbols)} symbols flagged")
    except Exception as e:
        log_global_error("Whale Priority", f"Priority calculation failed: {e}")
        priority_info = {}
    
    # Apply performance optimization
    try:
        from utils.performance_optimizer import optimize_scan_performance
        perf_config = optimize_scan_performance(symbols)
        optimized_symbols = perf_config['symbols']
        max_concurrent = perf_config['max_concurrent']
        
        print(f"[PERFORMANCE] Optimized: {len(optimized_symbols)} symbols, {max_concurrent} concurrent")
        print(f"[PERFORMANCE] Target: {len(optimized_symbols)} tokens in <15s")
    except Exception as e:
        log_global_error("Performance Optimization", f"Using default settings: {e}")
        optimized_symbols = symbols[:752]  # Process all available symbols
        max_concurrent = 400  # Increased for 752 tokens
    
    # Execute async scan with enhanced performance  
    start_time = time.time()
    async with AsyncTokenScanner(max_concurrent=max_concurrent) as scanner:
        scanner.fast_mode = True  # Set fast mode after initialization
        results = await scanner.scan_all_tokens(optimized_symbols, priority_info)
    scan_duration = time.time() - start_time
    
    print(f"[PERFORMANCE] Scan completed in {scan_duration:.1f}s (Target: <15s)")
    if scan_duration > 15:
        print(f"[PERFORMANCE WARNING] Scan exceeded 15s target by {scan_duration-15:.1f}s")
    
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
    
    # Display comprehensive error summary at end of scan
    error_count = get_error_count()
    if error_count > 0:
        print_error_summary()
        save_error_report()  # Save detailed error report to file
    else:
        print("\n✅ No errors during scan cycle")
    
    return results

def generate_top_tjde_charts(results: List[Dict]):
    """🎯 UNIFIED: Generate training charts for TOP 5 TJDE tokens - SINGLE GENERATION PER TOKEN"""
    try:
        # 🎯 CRITICAL FIX: Select TOP 5 tokens first to prevent dataset quality degradation
        from utils.top5_selector import select_top5_tjde_tokens, get_top5_selector
        
        # Select TOP 5 tokens by TJDE score
        top5_tokens = select_top5_tjde_tokens(results)
        
        if not top5_tokens:
            print("❌ [TOP5 FILTER] No tokens qualified for TOP 5 selection")
            return
        
        print(f"🎯 [TOP5 FILTER] Selected {len(top5_tokens)} tokens for training data generation")
        
        # 🔥 CRITICAL FIX: UNIFIED CHART GENERATION - Only ONE chart per token
        # Track which tokens already have charts to prevent duplication
        generated_charts = {}
        
        print(f"\n📊 UNIFIED CHART GENERATION FOR TOP 5 TJDE TOKENS:")
        
        for i, entry in enumerate(top5_tokens, 1):
            symbol = entry.get('symbol', 'UNKNOWN')
            tjde_score = entry.get('tjde_score', 0)
            tjde_decision = entry.get('tjde_decision', 'unknown')
            market_phase = entry.get('market_phase', 'unknown')
            
            print(f"{i}. {symbol}: TJDE {tjde_score:.3f} ({tjde_decision})")
            
            # 🔍 CHECK: Does this token already have a recent chart?
            import glob
            import os
            from datetime import datetime, timedelta
            
            current_time = datetime.now()
            recent_pattern = f"training_data/charts/{symbol}_*.png"
            existing_charts = glob.glob(recent_pattern)
            
            recent_chart_found = False
            for chart_path in existing_charts:
                try:
                    # Check if chart is recent (within last 30 minutes)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(chart_path))
                    if (current_time - file_mtime).total_seconds() < 1800:  # 30 minutes
                        print(f"   ✅ Using existing chart: {os.path.basename(chart_path)}")
                        generated_charts[symbol] = chart_path
                        recent_chart_found = True
                        break
                except:
                    continue
            
            if recent_chart_found:
                continue  # Skip to next token - already has recent chart
            
            # 🎯 SINGLE CHART GENERATION: Try TradingView first, then fallback to custom
            chart_generated = False
            
            # METHOD 1: Try TradingView generation (authentic charts)
            try:
                print(f"   🔄 Attempting TradingView chart generation...")
                
                # Import TradingView async fix for thread-safe generation
                from utils.tradingview_async_fix import TradingViewAsyncFix
                
                # Generate single TradingView chart
                tv_fix = TradingViewAsyncFix()
                chart_path = tv_fix.generate_single_chart(
                    symbol=symbol, 
                    tjde_score=tjde_score, 
                    market_phase=market_phase,
                    tjde_decision=tjde_decision
                )
                
                if chart_path and os.path.exists(chart_path):
                    print(f"   ✅ TradingView chart: {os.path.basename(chart_path)}")
                    generated_charts[symbol] = chart_path
                    chart_generated = True
                else:
                    print(f"   ⚠️ TradingView failed - trying fallback...")
                    
            except Exception as tv_e:
                print(f"   ⚠️ TradingView error: {tv_e} - trying fallback...")
            
            # METHOD 2: Fallback to custom chart generation
            if not chart_generated:
                try:
                    print(f"   🔄 Generating custom chart...")
                    
                    market_data = entry.get('market_data', {})
                    candles_15m = market_data.get('candles_15m', market_data.get('candles', []))
                    
                    # Emergency cache fetch if no candles
                    if not candles_15m or len(candles_15m) < 20:
                        import json
                        cache_file = f"data/scan_results/{symbol}_candles.json"
                        if os.path.exists(cache_file):
                            with open(cache_file, 'r') as f:
                                cached_data = json.load(f)
                                candles_15m = cached_data.get('candles_15m', [])
                    
                    # 🎯 NEW SYSTEM: Generate only TradingView screenshots with new naming format
                    # COMPLETELY ELIMINATE matplotlib charts - TradingView ONLY
                    try:
                        print(f"   🎯 NEW SYSTEM: TradingView-only generation with new naming format")
                        
                        # Extract setup from GPT commentary if available
                        gpt_comment = entry.get('gpt_comment', '')
                        setup_type = 'pattern_analysis'
                        
                        if gpt_comment:
                            if 'breakout' in gpt_comment.lower():
                                setup_type = 'breakout_continuation'
                            elif 'pullback' in gpt_comment.lower():
                                setup_type = 'pullback_in_trend'
                            elif 'consolidation' in gpt_comment.lower():
                                setup_type = 'range_consolidation'
                            elif 'support' in gpt_comment.lower():
                                setup_type = 'support_bounce'
                            elif 'resistance' in gpt_comment.lower():
                                setup_type = 'resistance_test'
                            else:
                                setup_type = market_phase.replace('-', '_')
                        else:
                            setup_type = market_phase.replace('-', '_')
                        
                        # Generate new format filename: SYMBOL_phase-setup_score-xxx.png
                        score_str = f"{int(tjde_score * 1000):03d}"  # Convert 0.427 to "427"
                        new_filename = f"{symbol}_{market_phase}-{setup_type}_score-{score_str}.png"
                        output_path = f"training_data/charts/{new_filename}"
                        
                        print(f"   📋 NEW FORMAT: {new_filename}")
                        print(f"   📊 Data: Phase={market_phase}, Setup={setup_type}, Score={tjde_score:.3f}")
                        
                        # Use TradingView screenshot generator with new naming
                        from utils.tradingview_screenshot import TradingViewScreenshotGenerator
                        
                        # Create generator instance
                        tv_generator = TradingViewScreenshotGenerator()
                        
                        # Initialize browser context if needed
                        import asyncio
                        if not hasattr(tv_generator, 'context') or tv_generator.context is None:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(tv_generator.__aenter__())
                        
                        # Generate TradingView screenshot with new filename
                        screenshot_path = loop.run_until_complete(
                            tv_generator.generate_tradingview_screenshot(
                                symbol=symbol,
                                tjde_score=tjde_score,
                                market_phase=market_phase,
                                decision=tjde_decision
                            )
                        )
                        
                        if screenshot_path and os.path.exists(screenshot_path):
                            # Rename to new format
                            os.makedirs("training_data/charts", exist_ok=True)
                            import shutil
                            shutil.move(screenshot_path, output_path)
                            
                            print(f"   ✅ TradingView NEW FORMAT: {new_filename}")
                            generated_charts[symbol] = output_path
                            chart_generated = True
                        else:
                            print(f"   ❌ TradingView generation failed for {symbol}")
                            
                    except Exception as tv_error:
                        print(f"   ❌ TradingView NEW SYSTEM failed: {tv_error}")
                        # NO FALLBACK TO MATPLOTLIB - TradingView only!
                    
                except Exception as custom_e:
                    print(f"   ❌ Custom chart failed: {custom_e}")
            
            if not chart_generated:
                print(f"   ❌ All chart generation methods failed for {symbol}")
        
        # 🎯 FINAL REPORT: Show unified results
        print(f"\n📊 UNIFIED GENERATION COMPLETE:")
        print(f"   ✅ Charts generated: {len(generated_charts)}/{len(top5_tokens)}")
        for symbol, path in generated_charts.items():
            print(f"   • {symbol}: {os.path.basename(path)}")
        
        # Generate Vision-AI metadata ONLY (NO additional chart generation)
        try:
            print("[VISION-AI] 🎯 Generating metadata for TOP 5 tokens (no duplicate charts)")
            
            # Generate metadata directly without calling generate_vision_ai_training_data 
            # to prevent duplicate TradingView pipeline execution
            from vision_ai_pipeline import save_label_jsonl
            from datetime import datetime
            
            training_pairs = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            for symbol in top5_tokens:
                try:
                    # Find the token data from results
                    token_data = None
                    for result in results:  # Use 'results' instead of undefined 'top5_results'
                        if result.get('symbol') == symbol:
                            token_data = result
                            break
                    
                    if not token_data:
                        continue
                    
                    tjde_score = token_data.get('tjde_score', 0)
                    phase = token_data.get('market_phase', 'unknown')
                    decision = token_data.get('tjde_decision', 'unknown')
                    setup = token_data.get('setup_type', phase)
                    clip_confidence = token_data.get('clip_confidence', 0.0)
                    
                    # Create training metadata
                    label_data = {
                        "phase": phase,
                        "setup": setup,
                        "tjde_score": tjde_score,
                        "tjde_decision": decision,
                        "confidence": clip_confidence,
                        "data_source": "unified_generation",
                        "chart_type": "authentic_tradingview",
                        "has_chart": symbol in generated_charts,
                        "chart_path": generated_charts.get(symbol, None)
                    }
                    
                    if save_label_jsonl(symbol, timestamp, label_data):
                        training_pairs += 1
                        print(f"[VISION-AI] Label saved: {symbol} - {setup}")
                        
                except Exception as e:
                    print(f"[VISION-AI ERROR] {symbol}: {e}")
            
            if training_pairs > 0:
                print(f"🎯 VISION-AI METADATA: {training_pairs} training pairs (no duplicate generation)")
                
        except Exception as metadata_e:
            print(f"⚠️ Vision-AI metadata generation failed: {metadata_e}")
        
        return generated_charts
    
    except Exception as e:
        print(f"[UNIFIED CHART ERROR] {e}")
        from crypto_scan_service import log_warning
        log_warning("UNIFIED CHART GENERATION", e, "Unified chart generation failed")
        return {}

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