#!/usr/bin/env python3
"""
Async All Tokens Scanner - TJDE v3 Architecture Implementation
Uses new two-phase architecture to fix AI-EYE circular dependency
Target: <15 seconds for 500+ tokens with enhanced intelligence
"""

import asyncio
import aiohttp
import json
import time
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# TJDE v3 PIPELINE - New architecture
from scan_pipeline_v3 import scan_with_tjde_v3
from scan_token_async import scan_token_async  # Fallback for compatibility
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
        """Scan single token with semaphore control - TJDE v3 Enhanced"""
        async with self.semaphore:
            self.total_api_calls += 4  # ticker + 15m + 5m + orderbook
            
            # Try TJDE v3 pipeline for enhanced analysis
            try:
                # Single symbol pipeline through v3
                results = await scan_with_tjde_v3([symbol], priority_info)
                if results and len(results) > 0:
                    self.successful_scans += 1
                    return results[0]  # Return first (and only) result
            except Exception as e:
                print(f"[TJDE v3 ERROR] {symbol}: {e} - falling back to legacy")
            
            # Fallback to legacy scan if v3 fails
            result = await scan_token_async(symbol, self.session, priority_info)
            if result:
                self.successful_scans += 1
            return result

    async def scan_all_tokens(self, symbols: List[str], priority_info: Dict = None) -> List[Dict]:
        """
        Scan all tokens with TJDE v3 batch processing + parallel fallback
        Core function replacing sequential scanning
        """
        print(f"🚀 Starting TJDE v3 batch scan of {len(symbols)} tokens")
        start_time = time.time()
        
        # FALLBACK TO LEGACY INDIVIDUAL SCANNING FIRST (for data gathering)
        print(f"🔄 Using individual token scanning for data gathering (max {self.max_concurrent} concurrent)")
        
        # Create tasks for all symbols
        tasks = [self.limited_scan(symbol, priority_info) for symbol in symbols]
        
        # Process with progress tracking
        legacy_results = []
        completed = 0
        
        # Process in chunks for memory efficiency and progress display
        chunk_size = min(50, len(tasks)) if len(tasks) > 0 else 1
        chunk_size = max(1, chunk_size)
        
        if len(tasks) == 0:
            print("No tasks to process")
            return []
            
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
                    legacy_results.append(result)
                    # Compact progress display
                    if completed % 10 == 0 or result.get("tjde_score", 0) >= 0.7:
                        print(f"[{completed}/{len(symbols)}] {symbol}: TJDE {result['tjde_score']:.3f} ({result['tjde_decision']})")
                else:
                    if completed % 20 == 0:  # Less frequent logging for skipped tokens
                        print(f"[{completed}/{len(symbols)}] {symbol}: Skipped")
        
        print(f"[LEGACY COMPLETE] Gathered data for {len(legacy_results)} tokens")
        
        # NOW TRY TJDE v3 PIPELINE WITH PRE-FETCHED DATA
        try:
            from scan_pipeline_v3 import scan_with_tjde_v3_from_data
            
            print(f"[TJDE v3 FROM DATA] Processing {len(legacy_results)} pre-fetched results...")
            print(f"[PIPELINE FLOW] Phase 2: TOP 20 selection → Phase 3: Charts → Phase 4: CLIP → Phase 5: Advanced")
            
            tjde_v3_results = await scan_with_tjde_v3_from_data(legacy_results, priority_info)
            
            if tjde_v3_results and len(tjde_v3_results) > 0:
                # TJDE v3 SUCCESS - return enhanced results
                self.successful_scans = len(tjde_v3_results)
                total_time = time.time() - start_time
                
                print(f"✅ TJDE v3 FROM DATA SUCCESS: {len(tjde_v3_results)} TOP tokens with advanced analysis in {total_time:.1f}s")
                print(f"[CRITICAL SUCCESS] TOP 20 selection and advanced AI-EYE analysis completed!")
                return tjde_v3_results
                
        except Exception as e:
            print(f"[TJDE v3 FROM DATA ERROR] Failed: {e}")
            import traceback
            print(f"[TJDE v3 ERROR DETAILS] {traceback.format_exc()}")
            print(f"[FALLBACK] Returning legacy scan results...")
            
            # Return legacy results as fallback
            self.successful_scans = len(legacy_results)
            total_time = time.time() - start_time
            print(f"[LEGACY FALLBACK] Returning {len(legacy_results)} results in {total_time:.1f}s")
            return legacy_results

        # Performance summary for legacy results
        total_time = time.time() - start_time
        tokens_per_second = len(symbols) / total_time if total_time > 0 else 0
        
        print(f"\n🎯 ASYNC SCAN RESULTS:")
        print(f"- Processed: {self.successful_scans}/{len(symbols)} tokens")
        print(f"- Total time: {total_time:.1f}s (TARGET: <15s)")
        print(f"- Performance: {tokens_per_second:.1f} tokens/second")
        api_per_token = self.total_api_calls/len(symbols) if len(symbols) > 0 else 0
        print(f"- API calls: {self.total_api_calls} ({api_per_token:.1f} per token)")
        
        # Top performers
        if legacy_results:
            sorted_results = sorted(legacy_results, key=lambda x: x.get('tjde_score', 0), reverse=True)
            print(f"\n🔥 TOP 10 PERFORMERS:")
            for i, result in enumerate(sorted_results[:10], 1):
                print(f"{i:2d}. {result['symbol']:12} TJDE {result['tjde_score']:5.3f} ({result['tjde_decision']:12}) Vol ${result['volume_24h']:>12,.0f}")
        
        return legacy_results

async def async_scan_cycle():
    """Main async scan cycle with unified scoring deduplication"""
    print(f"Starting async scan cycle at {datetime.now().strftime('%H:%M:%S')}")
    
    # Initialize error reporting for this scan session
    initialize_scan_session()
    
    # 🔒 DEDUPLICATION: Start new unified scoring cycle
    try:
        from utils.score_unification import start_unified_scan_cycle
        cycle_id = start_unified_scan_cycle()
        print(f"[SCORE UNIFY] Started unified scoring cycle: {cycle_id}")
    except Exception as e:
        print(f"[SCORE UNIFY ERROR] Failed to start unified cycle: {e}")
        cycle_id = None
    
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
    
    # 🛡 INTELLIGENT TOKEN VALIDATION: Smart processing for geographical restrictions
    try:
        from utils.token_validator import filter_tokens_by_completeness
        
        # On production servers: full validation to avoid retCode 10001
        # In geographical restricted environments (Replit): skip validation, use mock data fallback
        print(f"🛡 Testing token validation capability...")
        complete_tokens, partial_tokens, invalid_tokens = await filter_tokens_by_completeness(symbols[:5])  # Quick test
        
        # If validation fails completely (HTTP 403), we're in restricted environment
        if not complete_tokens and not partial_tokens:
            print(f"🌍 Geographical API restrictions detected - using all {len(symbols)} symbols with mock data fallback")
        else:
            print(f"✅ Complete tokens (15M+5M): {len(complete_tokens)}")
            print(f"⚠️ Partial tokens (15M only): {len(partial_tokens)}")
            print(f"❌ Invalid tokens: {len(invalid_tokens)}")
            # On production server with working API, we could filter more aggressively
            # For now, keep all symbols to maintain full processing capability
            print(f"🎯 Using all {len(symbols)} symbols (production flexibility)")
        
    except Exception as e:
        log_global_error("Token Validation", f"Token filtering failed: {e}")
        print(f"⚠️ Token validation error - using all {len(symbols)} symbols")
    
    # Whale prioritization
    try:
        symbols, priority_symbols, priority_info = prioritize_whale_tokens(symbols)
        if priority_symbols:
            print(f"Whale priority: {len(priority_symbols)} symbols flagged")
    except Exception as e:
        log_global_error("Whale Priority", f"Priority calculation failed: {e}")
        priority_info = {}
    
    # Etap 4: Dynamic Token Priority Sorting - Tokeny z repeat whales skanowane jako pierwsze
    try:
        from utils.token_priority_manager import sort_tokens_by_priority, get_priority_statistics
        
        # Pobierz statystyki przed sortowaniem
        priority_stats = get_priority_statistics()
        
        if priority_stats.get('total_tokens', 0) > 0:
            print(f"[TOKEN PRIORITY] Sorting {len(symbols)} tokens by priority...")
            print(f"[TOKEN PRIORITY] High priority tokens: {priority_stats['high_priority_tokens']}")
            print(f"[TOKEN PRIORITY] Max priority: {priority_stats['max_priority']:.1f}")
            
            # Pokaż TOP 5 priorytetowych tokenów
            for i, (token, priority) in enumerate(priority_stats['top_tokens'][:5]):
                print(f"[TOKEN PRIORITY]   {i+1}. {token}: {priority:.1f}")
            
            # Sortuj tokeny według priorytetu (najwyższy priorytet pierwszy)
            symbols = sort_tokens_by_priority(symbols)
            print(f"[TOKEN PRIORITY] ✅ Tokens sorted - high priority tokens will be scanned first")
        else:
            print(f"[TOKEN PRIORITY] No priority data available - using default order")
            
    except Exception as e:
        log_global_error("Token Priority", f"Priority sorting failed: {e}")
        print(f"[TOKEN PRIORITY] Error sorting tokens - using default order")
    
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
        optimized_symbols = symbols[:9999]  # Process all available symbols
        max_concurrent = 400  # Increased for 9999 tokens
    
    # 🔍 TOKEN VALIDATION: DISABLED - Process all tokens with mock data fallback
    print(f"[TOKEN VALIDATOR] DISABLED for 752 token processing")
    print(f"[PRODUCTION MODE] Processing all {len(optimized_symbols)} symbols with API/mock fallback")
    print(f"[GEOGRAPHICAL BYPASS] System will use mock data when API restricted (HTTP 403)")
    
    # REMOVED: Token filtering that was limiting to 31 tokens
    # On production server with 752 symbols, this allows full processing
    # In development (Replit), mock data fallback ensures operation continues
    
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
        high_score_setups = [r for r in results if r.get('tjde_score', 0) >= 0.8]
        if high_score_setups:
            print(f"\n⚡ {len(high_score_setups)} HIGH-VALUE SETUPS DETECTED")
            for setup in high_score_setups:
                print(f"   {setup['symbol']}: TJDE {setup['tjde_score']:.3f} ({setup['tjde_decision']})")
    
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
        
        # 🔥 CRITICAL FIX: UNIFIED CHART GENERATION WITH DEDUPLICATION
        # Track which tokens already have charts to prevent duplication
        generated_charts = {}
        already_processed = set()  # 🔒 DEDUPLICATION: Track processed symbols per cycle
        
        print(f"\n📊 UNIFIED CHART GENERATION FOR TOP 5 TJDE TOKENS:")
        
        for i, entry in enumerate(top5_tokens, 1):
            symbol = entry.get('symbol', 'UNKNOWN')
            
            # 🔒 DEDUPLICATION CHECK: Skip if symbol already processed this cycle
            if symbol in already_processed:
                print(f"{i}. {symbol}: ⚠️ SKIPPED - Already processed this cycle (preventing duplication)")
                continue
            
            # Mark symbol as being processed
            already_processed.add(symbol)
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
            
            # 🎯 CONDITIONAL CHART GENERATION: Check if this token deserves a chart
            from vision.chart_exporter import should_generate_chart, get_chart_generation_reason
            
            decision = entry.get('tjde_decision', 'unknown')  # FIX: Use correct field name
            clip_confidence = entry.get('clip_confidence', 0.0)
            
            should_generate = should_generate_chart(symbol, decision, tjde_score, clip_confidence)
            generation_reason = get_chart_generation_reason(symbol, decision, tjde_score, clip_confidence)
            
            if not should_generate:
                print(f"   🚫 Chart generation skipped: {generation_reason}")
                print(f"   📊 Vision-AI optimization: Avoiding low-value signal for training")
                continue  # Skip this token - no chart generation needed
            
            print(f"   ✅ Chart generation approved: {generation_reason}")
            
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
                
                # 🔒 CRITICAL VALIDATION: Handle invalid symbol detection (both pre and post OCR)
                if chart_path in ["INVALID_SYMBOL", "INVALID_SYMBOL_OCR"]:
                    validation_method = "page content" if chart_path == "INVALID_SYMBOL" else "OCR analysis"
                    print(f"   🚨 [CHART ERROR] Invalid symbol detected via {validation_method}: {symbol}")
                    print(f"   🚫 [INVALID SYMBOL] {symbol}: BLOCKED from TOP5 and CLIP training")
                    print(f"   🔒 [SAFETY CAP] Removing {symbol} from TOP5 processing to prevent contamination")
                    
                    # Mark as invalid and block from further processing
                    entry['invalid_symbol'] = True
                    entry['tjde_score'] = 0.0
                    entry['tjde_decision'] = 'avoid'
                    entry['blocked_reason'] = f'TradingView Invalid symbol detected via {validation_method}'
                    entry['validation_method'] = validation_method
                    
                    # Do NOT try alternative exchanges for invalid symbols
                    # Invalid symbols should be completely blocked from TOP5 processing
                    continue  # Skip to next token
                
                elif chart_path and os.path.exists(chart_path):
                    print(f"   ✅ TradingView chart: {os.path.basename(chart_path)}")
                    
                    # 🧠 GPT ANALYSIS: Analyze chart and add setup label for CLIP training
                    try:
                        from utils.gpt_chart_analyzer import analyze_and_label_chart
                        print(f"   🧠 Starting GPT analysis for setup identification...")
                        
                        labeled_chart_path = analyze_and_label_chart(
                            image_path=chart_path,
                            symbol=symbol,
                            tjde_score=tjde_score
                        )
                        
                        if labeled_chart_path and labeled_chart_path != chart_path:
                            print(f"   🏷️ Chart labeled: {os.path.basename(labeled_chart_path)}")
                            generated_charts[symbol] = labeled_chart_path
                        else:
                            print(f"   ⚠️ GPT labeling failed - using original chart")
                            generated_charts[symbol] = chart_path
                        
                    except Exception as gpt_e:
                        print(f"   ⚠️ GPT analysis error: {gpt_e} - using original chart")
                        generated_charts[symbol] = chart_path
                    
                    chart_generated = True
                else:
                    print(f"   ⚠️ TradingView failed - trying fallback...")
                    
            except Exception as tv_e:
                print(f"   ⚠️ TradingView error: {tv_e} - trying fallback...")
            
            # SKIP TOKEN WITHOUT CHART: No placeholder creation - token is skipped
            if not chart_generated:
                print(f"   🚫 TradingView failed - SKIPPING {symbol} (no chart = no training data)")
                print(f"   ✅ NO PLACEHOLDER - TradingView-only system enforced")
                # Token is completely skipped - no chart, no training data, no placeholder
                continue
        
        # 🎯 FINAL REPORT: Show unified results
        print(f"\n📊 UNIFIED GENERATION COMPLETE:")
        print(f"   ✅ Charts generated: {len(generated_charts)}/{len(top5_tokens)}")
        for symbol, path in generated_charts.items():
            print(f"   • {symbol}: {os.path.basename(path)}")
        
        # 📊 MARKET HEALTH MONITORING: Record scan health and check for alerts
        try:
            from utils.market_health_monitor import log_scan_health
            
            print(f"[MARKET MONITOR] Recording health stats for {len(results)} scan results")
            health_stats = log_scan_health(results)
            
            # Display market health summary
            condition = health_stats.get("market_condition", "unknown")
            max_score = health_stats.get("score_stats", {}).get("max", 0.0)
            chart_eligible = health_stats.get("chart_generation_eligible", 0)
            
            print(f"[MARKET HEALTH] Condition: {condition} | Max Score: {max_score:.3f} | Chart Eligible: {chart_eligible}")
            
            # Save score histogram for dynamic threshold training
            if "score_histogram" in health_stats:
                histogram_file = "data/score_histogram.json"
                os.makedirs(os.path.dirname(histogram_file), exist_ok=True)
                
                with open(histogram_file, 'w') as f:
                    json.dump({
                        "timestamp": health_stats["timestamp"], 
                        "histogram": health_stats["score_histogram"],
                        "stats": health_stats["score_stats"],
                        "condition": condition
                    }, f, indent=2)
                
                print(f"[SCORE HISTOGRAM] Saved to {histogram_file}")
            
            # Display alert if present
            if "alert" in health_stats:
                alert = health_stats["alert"]
                print(f"🚨 [MARKET ALERT] {alert['type']}: {alert['message']}")
                print(f"   💡 Recommendation: {alert['recommendation']}")
            
        except Exception as monitor_e:
            print(f"[MARKET MONITOR ERROR] {monitor_e}")

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
            "top_performers": sorted(results, key=lambda x: x.get('tjde_score', 0), reverse=True)[:20],
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