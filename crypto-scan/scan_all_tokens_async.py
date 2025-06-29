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
    
    # 🎯 CRITICAL FIXES: Initialize performance tracking and memory validation
    try:
        from utils.performance_critical_fixes import (
            validate_memory_files, track_scan_performance, performance_fixes
        )
        
        # Validate all memory files before starting scan
        memory_valid = validate_memory_files()
        print(f"[CRITICAL FIX] Memory validation: {'✅ VALID' if memory_valid else '⚠️ REPAIRED'}")
        
        # Reset performance tracker
        performance_fixes.training_worker_active = False
        
    except Exception as e:
        log_global_error("Critical Fixes", f"Failed to initialize critical fixes: {e}")
    
    # Execute async scan with enhanced performance  
    start_time = time.time()
    async with AsyncTokenScanner(max_concurrent=max_concurrent) as scanner:
        scanner.fast_mode = True  # Set fast mode after initialization
        results = await scanner.scan_all_tokens(optimized_symbols, priority_info)
    scan_duration = time.time() - start_time
    
    # 🎯 CRITICAL FIX: Track performance and apply optimizations
    try:
        performance_analysis = track_scan_performance(start_time, len(results) if results else 0)
        
        print(f"[PERFORMANCE] Scan completed in {scan_duration:.1f}s (Target: <15s)")
        if scan_duration > 15:
            print(f"[PERFORMANCE WARNING] Scan exceeded 15s target by {scan_duration-15:.1f}s")
            print(f"[PERFORMANCE ANALYSIS] Recommendations:")
            for rec in performance_analysis.get('recommendations', []):
                print(f"  • {rec}")
        else:
            print(f"[PERFORMANCE SUCCESS] ✅ Target achieved!")
            
    except Exception as e:
        log_global_error("Performance Tracking", f"Failed to track performance: {e}")
        print(f"[PERFORMANCE] Scan completed in {scan_duration:.1f}s (Target: <15s)")
        if scan_duration > 15:
            print(f"[PERFORMANCE WARNING] Scan exceeded 15s target by {scan_duration-15:.1f}s")
    
    # Post-scan processing
    if results:
        # Save summary
        save_scan_summary(results)
        
        # Generate training charts for TOP 5 TJDE tokens only
        await generate_top_tjde_charts(results)
        
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

async def generate_top_tjde_charts(results: List[Dict]):
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
            
            # 🚫 FORCE REGENERATION FOR TOP 5 TJDE TOKENS
            # Vision-AI requires FRESH screenshots - market phases change every 15-60 minutes
            # Never reuse existing charts for TOP 5 tokens as they need real-time analysis
            print(f"   🔄 FORCE REGENERATION: TOP 5 TJDE token requires fresh TradingView screenshot")
            
            # Smart cleanup: Remove only stale current chart while preserving 72h history
            import glob
            import os
            from datetime import datetime, timedelta
            
            cleanup_pattern = f"training_data/charts/{symbol}_*.png"
            existing_charts = glob.glob(cleanup_pattern)
            
            # Sort charts by modification time (newest first)
            chart_files = []
            for chart_path in existing_charts:
                if os.path.exists(chart_path):
                    mtime = os.path.getmtime(chart_path)
                    chart_files.append((chart_path, mtime))
            
            chart_files.sort(key=lambda x: x[1], reverse=True)
            
            cleaned_count = 0
            current_time = datetime.now().timestamp()
            
            # Strategy: Remove only the most recent chart if it's stale (>30 minutes)
            # Keep all historical charts within 72 hours for trend analysis
            if chart_files:
                newest_chart, newest_mtime = chart_files[0]
                age_minutes = (current_time - newest_mtime) / 60
                
                if age_minutes > 30:  # Only remove if stale
                    try:
                        os.remove(newest_chart)
                        cleaned_count += 1
                        
                        # Remove associated metadata
                        metadata_path = newest_chart.replace('.png', '_metadata.json')
                        if os.path.exists(metadata_path):
                            os.remove(metadata_path)
                            
                    except Exception as cleanup_e:
                        pass  # Continue even if cleanup fails
                
                # Archive old charts beyond 72 hours (but keep some for historical reference)
                keep_count = 0
                for chart_path, mtime in chart_files[1:]:  # Skip the newest chart
                    age_hours = (current_time - mtime) / 3600
                    
                    if age_hours > 72 and keep_count > 20:  # Keep max 20 historical charts
                        try:
                            os.remove(chart_path)
                            cleaned_count += 1
                            
                            # Remove associated metadata
                            metadata_path = chart_path.replace('.png', '_metadata.json')
                            if os.path.exists(metadata_path):
                                os.remove(metadata_path)
                                
                        except Exception as cleanup_e:
                            pass
                    else:
                        keep_count += 1
            
            if cleaned_count > 0:
                print(f"   🧹 Cleaned {cleaned_count} old charts for {symbol} to ensure fresh generation")
            
            # 🎯 SINGLE CHART GENERATION: Try TradingView first, then fallback to custom
            chart_generated = False
            
            # METHOD 1: Try TradingView generation (authentic charts) with multi-exchange fallback
            try:
                print(f"   🔄 Attempting TradingView chart generation...")
                
                # Use robust TradingView generator
                from utils.tradingview_robust import RobustTradingViewGenerator
                
                # Generate TradingView chart using robust generator
                async with RobustTradingViewGenerator() as generator:
                    chart_path = await generator.generate_screenshot(
                        symbol=symbol, 
                        tjde_score=tjde_score,
                        decision=entry.get('decision', 'unknown')
                    )
                
                # Handle invalid symbol case - try alternative exchanges
                if chart_path == "INVALID_SYMBOL":
                    print(f"   ⚠️ Invalid symbol detected - trying alternative exchanges...")
                    
                    # Try alternative exchanges using multi-exchange resolver
                    from utils.multi_exchange_resolver import get_multi_exchange_resolver
                    resolver = get_multi_exchange_resolver()
                    
                    if resolver:
                        # Get all possible exchange combinations
                        alternative_results = resolver.get_all_possible_exchanges(symbol)
                        
                        # Get the original failed exchange to skip it
                        original_resolution = resolver.resolve_tradingview_symbol(symbol)
                        original_exchange = original_resolution[1] if original_resolution else "UNKNOWN"
                        
                        for i, (tv_symbol, exchange) in enumerate(alternative_results[:3]):  # Try up to 3 alternatives
                            if exchange != original_exchange:  # Skip the one that already failed
                                print(f"   🔄 Trying alternative: {tv_symbol} ({exchange})")
                                
                                # Temporarily override resolver result
                                resolver.cache[symbol] = (tv_symbol, exchange)
                                
                                # Try generation with alternative exchange
                                alt_chart_path = await generator.generate_screenshot(
                                    symbol=symbol, 
                                    tjde_score=tjde_score,
                                    decision=entry.get('decision', 'unknown')
                                )
                                
                                if alt_chart_path and alt_chart_path != "INVALID_SYMBOL" and os.path.exists(alt_chart_path):
                                    print(f"   ✅ Alternative success: {os.path.basename(alt_chart_path)}")
                                    generated_charts[symbol] = alt_chart_path
                                    chart_generated = True
                                    break
                                elif alt_chart_path == "INVALID_SYMBOL":
                                    print(f"   ❌ Alternative {exchange} also invalid")
                                else:
                                    print(f"   ❌ Alternative {exchange} failed: {alt_chart_path}")
                    
                    if not chart_generated:
                        print(f"   ❌ All exchanges failed for {symbol}")
                
                elif chart_path and os.path.exists(chart_path):
                    print(f"   ✅ TradingView chart: {os.path.basename(chart_path)}")
                    generated_charts[symbol] = chart_path
                    chart_generated = True
                else:
                    print(f"   ⚠️ TradingView failed - trying fallback...")
                    
            except Exception as tv_e:
                print(f"   ⚠️ TradingView error: {tv_e} - trying fallback...")
            
            # METHOD 2: Use fallback eliminator instead of matplotlib fallback
            if not chart_generated:
                try:
                    print(f"   🔄 Creating TradingView FAILED placeholder...")
                    
                    # Use the fallback eliminator for proper placeholder creation
                    from utils.tradingview_fallback_eliminator import handle_tradingview_failure_safe
                    
                    placeholder_path = handle_tradingview_failure_safe(
                        symbol=symbol,
                        error_reason="sync_wrapper_failed",
                        phase=market_phase or 'unknown',
                        setup=setup_type,
                        score=tjde_score
                    )
                    
                    if placeholder_path:
                        print(f"   ⚠️ TradingView FAILED - Created placeholder: {os.path.basename(placeholder_path)}")
                        print(f"   🚫 NO MATPLOTLIB FALLBACK - TradingView-only system enforced")
                    else:
                        print(f"   ❌ Placeholder creation also failed")
                    
                except Exception as placeholder_e:
                    print(f"   ❌ Placeholder creation failed: {placeholder_e}")
            
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