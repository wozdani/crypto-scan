#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time
import json
import random as system_random
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# Global scan warnings system
SCAN_WARNINGS = []

def log_warning(label, exception=None, additional_info=None):
    """
    Bezpieczne logowanie ostrzeżeń podczas skanu
    Args:
        label: Etykieta błędu (np. "TRADINGVIEW SCREENSHOT ERROR")
        exception: Opcjonalny obiekt wyjątku
        additional_info: Dodatkowe informacje kontekstowe
    """
    msg = f"[{label}]"
    if exception:
        msg += f" {str(exception)}"
    if additional_info:
        msg += f" - {additional_info}"
    
    print(f"⚠️ {msg}")
    SCAN_WARNINGS.append(msg)

def clear_scan_warnings():
    """Wyczyść listę ostrzeżeń na początku nowego skanu"""
    global SCAN_WARNINGS
    SCAN_WARNINGS = []

def report_scan_warnings():
    """Pokaż podsumowanie ostrzeżeń na końcu skanu"""
    if SCAN_WARNINGS:
        print("\n🚨 SCAN COMPLETED WITH WARNINGS:")
        for warning in SCAN_WARNINGS:
            print(warning)
        print(f"📊 Total warnings: {len(SCAN_WARNINGS)}")
    else:
        print("\n✅ No errors during scan cycle")

# Import only essential modules
from utils.bybit_cache_manager import get_bybit_symbols_cached
from stages.stage_minus2_1 import detect_stage_minus2_1
from utils.scoring import compute_ppwcs, compute_checklist_score, get_alert_level, save_score, log_ppwcs_score
from utils.data_fetchers import get_market_data
from utils.alert_system import process_alert
from utils.coingecko import build_coingecko_cache
from utils.whale_priority import prioritize_whale_tokens
from utils.training_data_manager import TrainingDataManager

def scan_single_token(symbol):
    """Simple, fast token scan"""
    try:
        # Get market data
        data = get_market_data(symbol)
        if not data or len(data) != 4:
            return None
            
        success, market_data, price_usd, compressed = data
        if not success:
            return None

        # Basic Stage-2.1 analysis only
        stage2_pass, signals, inflow_usd, stage1g_active = detect_stage_minus2_1(symbol, price_usd)
        
        # Simple scoring
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

        # Training chart generation for quality setups - RESTRICTED TO TOP 5 ONLY
        if final_score >= 40 or checklist_score >= 35:
            # 🎯 CRITICAL FIX: Check if token is in TOP 5 before generating training data
            try:
                from utils.top5_selector import should_generate_training_data, warn_about_non_top5_generation
                
                if should_generate_training_data(symbol):
                    print(f"[TRAINING] {symbol} → Generating training chart (PPWCS: {final_score}, Checklist: {checklist_score}) - TOP 5 token")
                    try:
                        # Initialize training data manager
                        training_manager = TrainingDataManager()
                        
                        # Prepare context features for training
                        context_features = {
                            "ppwcs_score": final_score,
                            "checklist_score": checklist_score,
                            "alert_level": alert_level,
                            "price": signals.get("price", 0),
                            "volume_24h": signals.get("volume_24h", 0),
                            "market_phase": "simple_scan",
                            "scan_timestamp": datetime.now().isoformat(),
                            "signals_count": len([k for k, v in signals.items() if v and k != 'price' and k != 'volume_24h'])
                        }
                        
                        # Generate and save training chart
                        chart_id = training_manager.collect_from_scan(symbol, context_features)
                        if chart_id:
                            print(f"[TRAINING] {symbol} → Chart saved: {chart_id}")
                        else:
                            print(f"[TRAINING] {symbol} → Chart generation failed")
                            
                    except Exception as e:
                        print(f"[TRAINING ERROR] {symbol} → {e}")
                else:
                    # Token qualifies but is not in TOP 5 - log warning about dataset quality
                    warn_about_non_top5_generation(symbol, f"PPWCS scan - score {final_score}")
                    
            except ImportError:
                # TOP5 selector not available - skip training to avoid dataset degradation
                print(f"[TRAINING SKIP] {symbol} → TOP5 selector not available, skipping to maintain dataset quality")
            except Exception as e:
                print(f"[TRAINING TOP5 ERROR] {symbol} → {e}")

        return {
            'symbol': symbol,
            'final_score': final_score,
            'signals': signals,
            'checklist_score': checklist_score
        }
        
    except Exception as e:
        print(f"Error scanning {symbol}: {e}")
        return None

def scan_cycle():
    """Enhanced scan cycle with async integration and comprehensive error tracking"""
    print(f"\nStarting scan cycle at {datetime.now().strftime('%H:%M:%S')}")
    
    # Clear warnings from previous scan
    clear_scan_warnings()
    
    start_time = time.time()
    
    # Get symbols with error tracking
    try:
        symbols = get_bybit_symbols_cached()
        print(f"Scanning {len(symbols)} symbols...")
    except Exception as e:
        log_warning("SYMBOL FETCH ERROR", e)
        symbols = []
    
    # Build cache with error tracking
    try:
        build_coingecko_cache()
    except Exception as e:
        log_warning("COINGECKO CACHE BUILD ERROR", e)
    
    # Use enhanced async scan with TJDE and chart generation
    try:
        print("Running enhanced async scan with TJDE analysis...")
        
        # Import and run full async scan cycle with TJDE
        import asyncio
        from scan_all_tokens_async import async_scan_cycle
        
        # Run enhanced async scan cycle with event loop error tracking
        try:
            result = asyncio.run(async_scan_cycle())
            # Result is a list of scan results, not an integer
            processed_count = len(result) if isinstance(result, list) else 0
            
            print(f"Enhanced async scan processed {processed_count} tokens with TJDE analysis")
            print("Async scan completed successfully")
            
            # CHART CLEANUP: Run after successful scan (5% chance per cycle)
            if system_random.random() < 0.05:  # 5% chance
                try:
                    from utils.chart_cleanup import cleanup_with_size_report
                    print("🧹 Running automatic chart cleanup...")
                    cleanup_stats = cleanup_with_size_report(max_age_hours=72, dry_run=False)
                    print(f"🧹 Chart cleanup: {cleanup_stats.get('deleted_files', 0)} files deleted, {cleanup_stats.get('space_saved_gb', 0):.2f} GB saved")
                except Exception as cleanup_error:
                    log_warning("CHART CLEANUP ERROR", cleanup_error)
            
            # Check for low processing count
            if processed_count == 0:
                log_warning("ASYNC SCAN PROCESSING", None, "No tokens processed successfully")
                
        except RuntimeError as asyncio_error:
            if "cannot be called from a running event loop" in str(asyncio_error):
                log_warning("ASYNCIO EVENT LOOP CONFLICT", asyncio_error)
            else:
                log_warning("ASYNCIO RUNTIME ERROR", asyncio_error)
        except Exception as async_error:
            log_warning("ASYNC SCAN EXECUTION ERROR", async_error)
        
        # Flush results with error tracking
        try:
            from scan_token_async import flush_async_results
            flush_async_results()
            print("[FLUSH] Async results flushed successfully")
        except Exception as flush_e:
            log_warning("RESULT FLUSH ERROR", flush_e)
        
    except ImportError as import_error:
        log_warning("ASYNC MODULE IMPORT ERROR", import_error)
        # Fallback to simple scan
        simple_scan_fallback(symbols)
    except Exception as e:
        log_warning("ASYNC SCAN SETUP ERROR", e)
        # Fallback to async batch scan with reduced concurrency
        simple_scan_fallback(symbols)
    
    # Show timing
    elapsed = time.time() - start_time
    print(f"Scan cycle completed in {elapsed:.1f}s")
    
    # Report all warnings collected during scan
    report_scan_warnings()
    
    # Run memory feedback evaluation periodically
    try:
        from utils.memory_feedback_loop import run_memory_feedback_evaluation
        
        # Run feedback evaluation every few cycles
        if system_random.random() < 0.3:  # 30% chance each cycle
            run_memory_feedback_evaluation()
    except ImportError as import_error:
        log_warning("MEMORY FEEDBACK MODULE IMPORT ERROR", import_error)
    except Exception as memory_error:
        log_warning("MEMORY FEEDBACK EVALUATION ERROR", memory_error)
    
    # Run Phase 2 memory outcome updates periodically
    try:
        from token_context_memory import update_historical_outcomes_loop
        
        # Update historical outcomes every few cycles
        if system_random.random() < 0.2:  # 20% chance each cycle
            update_historical_outcomes_loop()
    except ImportError as import_error:
        log_warning("PHASE 2 MEMORY MODULE IMPORT ERROR", import_error)
    except Exception as phase2_error:
        log_warning("PHASE 2 MEMORY UPDATE ERROR", phase2_error)
    
    # Run Phase 3 Vision-AI evaluation periodically
    try:
        from evaluate_model_accuracy import VisionAIEvaluator
        
        # Run Vision-AI evaluation less frequently (once per day worth of cycles)
        if system_random.random() < 0.05:  # 5% chance each cycle
            evaluator = VisionAIEvaluator()
            evaluator.run_complete_evaluation(days_back=3)
            print("[PHASE 3] Vision-AI feedback loop evaluation completed")
    except ImportError as import_error:
        log_warning("PHASE 3 VISION-AI MODULE IMPORT ERROR", import_error)
    except Exception as phase3_error:
        log_warning("PHASE 3 VISION-AI EVALUATION ERROR", phase3_error)
    
    # Run Phase 4 Embedding processing periodically
    try:
        from hybrid_embedding_system import process_training_charts_for_embeddings
        
        # Process training charts for embeddings (very low frequency)
        if system_random.random() < 0.02:  # 2% chance each cycle
            processed = process_training_charts_for_embeddings()
            if processed > 0:
                print(f"[PHASE 4] Processed {processed} charts for hybrid embeddings")
    except ImportError as import_error:
        log_warning("PHASE 4 EMBEDDING MODULE IMPORT ERROR", import_error)
    except Exception as phase4_error:
        log_warning("PHASE 4 EMBEDDING PROCESSING ERROR", phase4_error)
    
    # Run Phase 5 Reinforcement Learning periodically
    try:
        from reinforce_embeddings import run_periodic_reinforcement_learning
        
        # Run RL cycle very infrequently (daily learning)
        if system_random.random() < 0.01:  # 1% chance each cycle
            report = run_periodic_reinforcement_learning()
            if report:
                performance = report["overall_performance"]
                print(f"[PHASE 5] RL cycle completed: {performance['avg_success_rate']:.1%} success rate")
            
    except ImportError as import_error:
        log_warning("PHASE 5 REINFORCEMENT LEARNING MODULE IMPORT ERROR", import_error)
    except Exception as phase5_error:
        log_warning("PHASE 5 REINFORCEMENT LEARNING ERROR", phase5_error)
    
    # Run TJDE Feedback Integration - evaluate +6h alerts and trigger learning
    try:
        from utils.feedback_integration import run_feedback_evaluation
        
        # Run feedback evaluation every 6 hours worth of cycles (~24 cycles assuming 15min intervals)
        if system_random.random() < 0.1:  # 10% chance each cycle for frequent evaluation
            evaluated_count = run_feedback_evaluation()
            if evaluated_count > 0:
                print(f"[FEEDBACK INTEGRATION] Evaluated {evaluated_count} alerts and triggered adaptive learning")
            
    except ImportError as import_error:
        log_warning("FEEDBACK INTEGRATION MODULE IMPORT ERROR", import_error)
    except Exception as feedback_error:
        log_warning("FEEDBACK INTEGRATION ERROR", feedback_error)

def simple_scan_fallback(symbols):
    """Fallback using async batch scan with lower concurrency"""
    print("Running async batch fallback with reduced concurrency...")
    log_warning("FALLBACK SCAN TRIGGERED", None, f"Processing {len(symbols)} symbols with reduced concurrency")
    
    try:
        # Use async batch scanning even in fallback but with lower concurrency
        import asyncio
        from scan_all_tokens_async import scan_symbols_async
        
        # Run with lower concurrency for stability but ALL symbols
        results = asyncio.run(scan_symbols_async(symbols, max_concurrent=8))
        print(f"Async batch fallback processed {len(results)} tokens")
        
        if len(results) == 0:
            log_warning("ASYNC BATCH FALLBACK ZERO RESULTS", None, "No tokens processed in async batch mode")
        
        return results
        
    except RuntimeError as runtime_error:
        if "cannot be called from a running event loop" in str(runtime_error):
            log_warning("FALLBACK ASYNCIO EVENT LOOP CONFLICT", runtime_error)
        else:
            log_warning("FALLBACK ASYNC RUNTIME ERROR", runtime_error)
    except ImportError as import_error:
        log_warning("FALLBACK ASYNC MODULE IMPORT ERROR", import_error)
    except Exception as e:
        log_warning("ASYNC BATCH FALLBACK ERROR", e)
        
    print("Using simple sequential scan as last resort...")
    log_warning("SEQUENTIAL FALLBACK TRIGGERED", None, "All async methods failed, using basic sequential scan")
    
    # Ultimate fallback - simple sequential for core functionality only
    processed = 0
    results = []
    failed_tokens = 0
    
    for symbol in symbols:  # Process ALL symbols in fallback
        try:
            result = scan_single_token(symbol)
            if result:
                results.append(result)
                processed += 1
                if processed % 10 == 0:  # Progress every 10 tokens
                    print(f"[{processed}/{len(symbols)}] Progress: {symbol}: PPWCS {result.get('final_score', 0):.1f}")
            else:
                failed_tokens += 1
        except Exception as token_error:
            failed_tokens += 1
            if failed_tokens <= 5:  # Log only first 5 token errors to avoid spam
                log_warning("SEQUENTIAL SCAN TOKEN ERROR", token_error, f"Failed to process {symbol}")
    
    if failed_tokens > 5:
        log_warning("SEQUENTIAL SCAN MULTIPLE FAILURES", None, f"Additional {failed_tokens - 5} tokens failed")
    
    print(f"Sequential fallback processed {processed}/{len(symbols)} tokens")
    
    if processed == 0:
        log_warning("SEQUENTIAL FALLBACK ZERO RESULTS", None, "No tokens processed successfully in final fallback")
    
    return results

def wait_for_next_candle():
    """Wait for next 15-minute candle"""
    now = datetime.now(timezone.utc)
    next_candle = now.replace(second=5, microsecond=0)
    
    # Add 15 minutes if we're past the 5-second mark
    if now.second >= 5:
        next_candle += timedelta(minutes=15 - (now.minute % 15))
    else:
        next_candle += timedelta(minutes=15 - (now.minute % 15))
    
    wait_seconds = (next_candle - now).total_seconds()
    
    if wait_seconds > 0:
        print(f"Waiting {wait_seconds:.1f}s for next candle at {next_candle.strftime('%H:%M:%S')} UTC")
        time.sleep(wait_seconds)

def main():
    """Main scanning loop"""
    print("Starting Crypto Scanner Service (Simple Version)")
    
    try:
        while True:
            try:
                clear_scan_warnings()  # Clear warnings at start of each cycle
                scan_cycle()
                report_scan_warnings()  # Report warnings at end of cycle
                wait_for_next_candle()
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as scan_error:
                log_warning("MAIN SCAN CYCLE ERROR", scan_error)
                print(f"Scan error: {scan_error}")
                time.sleep(60)  # Wait 1 minute before retry
                
    except KeyboardInterrupt:
        print("Service stopped.")
    except Exception as main_error:
        log_warning("MAIN SERVICE ERROR", main_error)
        print(f"Service error: {main_error}")

if __name__ == "__main__":
    main()