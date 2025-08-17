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

# Sanity-check for required dependencies
try:
    import pytesseract
    print("✅ pytesseract dependency check passed")
except ImportError:
    raise RuntimeError("🚨 pytesseract is required for TradingView chart validation – please install it.")

# Import explore mode cleanup funkcji
try:
    from utils.stealth_utils import cleanup_old_explore_mode_data
    EXPLORE_CLEANUP_AVAILABLE = True
except ImportError:
    EXPLORE_CLEANUP_AVAILABLE = False
    print("[EXPLORE CLEANUP] Cleanup function not available")

# Import Last10Runner for single LLM call after scan cycle
try:
    from pipeline.last10_runner import get_last10_runner
    LAST10_RUNNER_AVAILABLE = True
    print("[LAST10 RUNNER] Single LLM call system for last 10 tokens loaded")
except ImportError:
    LAST10_RUNNER_AVAILABLE = False
    print("[LAST10 RUNNER] Last10 runner system not available")

# Global scan warnings system
SCAN_WARNINGS = []

# Global adaptive learning availability flag
try:
    from feedback_loop.adaptive_threshold_integration import run_adaptive_threshold_maintenance, get_adaptive_system_status
    ADAPTIVE_LEARNING_AVAILABLE = True
    print("[ADAPTIVE LEARNING] Adaptive threshold learning system loaded globally")
except ImportError:
    ADAPTIVE_LEARNING_AVAILABLE = False
    print("[ADAPTIVE LEARNING] Adaptive threshold learning system not available")

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

def display_top5_stealth_tokens():
    """Wyświetl TOP 5 tokenów z najwyższymi stealth score - NOWY SYSTEM LOGOWANIA"""
    try:
        # Import nowego systemu logowania
        from utils.stealth_logger import print_top5_stealth_tokens_enhanced
        from stealth_engine.priority_scheduler import AlertQueueManager
        
        priority_manager = AlertQueueManager()
        
        # Pobierz TOP 5 tokenów z najwyższymi stealth scores
        top_tokens = priority_manager.get_top_priority_tokens(limit=5)
        
        if top_tokens and len(top_tokens) > 0:
            # Użyj nowego systemu logowania z breakdown detektorów
            print_top5_stealth_tokens_enhanced(top_tokens)
            
        else:
            print("\n⚠️ TOP 5 STEALTH TOKENS: Brak danych stealth score")
            
            # Fallback: sprawdź bezpośrednio cache file
            try:
                import json
                from pathlib import Path
                
                stealth_cache_file = Path("cache/stealth_last_scores.json")
                if stealth_cache_file.exists():
                    with open(stealth_cache_file, 'r') as f:
                        stealth_scores = json.load(f)
                    
                    if stealth_scores:
                        # Konwertuj cache format na nowy format
                        converted_tokens = []
                        sorted_tokens = sorted(
                            stealth_scores.items(), 
                            key=lambda x: x[1].get('score', 0.0), 
                            reverse=True
                        )[:5]
                        
                        for token, data in sorted_tokens:
                            token_data = {
                                'token': token,
                                'base_score': data.get('score', 0.0),
                                'early_score': data.get('score', 0.0),
                                'dex_inflow': data.get('dex_inflow', 0.0),
                                'whale_ping': data.get('whale_ping', 0.0),
                                'trust_boost': data.get('trust_boost', 0.0),
                                'identity_boost': data.get('identity_boost', 0.0),
                                'stealth_signals': data.get('stealth_signals', []),
                                # Dodaj AI detektory
                                'diamond_score': data.get('diamond_score', 0.0),
                                'californium_score': data.get('californium_score', 0.0),
                                'whaleclip_score': data.get('whaleclip_score', 0.0),
                                'stealth_engine_score': data.get('stealth_engine_score', data.get('score', 0.0)),
                                # Używamy nowych pól consensus
                                'consensus_decision': data.get('consensus_decision', 'UNKNOWN'),
                                'consensus_votes': data.get('consensus_votes', []),
                                'consensus_score': data.get('consensus_score', 0.0),
                                'consensus_confidence': data.get('consensus_confidence', 0.0),
                                'consensus_detectors': data.get('consensus_detectors', []),
                                'feedback_adjust': data.get('feedback_adjust', 0.0)
                            }
                            converted_tokens.append(token_data)
                        
                        # Użyj nowego systemu logowania
                        print_top5_stealth_tokens_enhanced(converted_tokens)
                    else:
                        print("⚠️ Cache stealth scores pusty")
                else:
                    print("⚠️ Brak cache file stealth scores")
                    
            except Exception as cache_error:
                print(f"⚠️ Błąd odczytu cache stealth scores: {cache_error}")
        
    except ImportError as import_error:
        print(f"⚠️ TOP 5 STEALTH TOKENS: Import error - {import_error}")
    except Exception as e:
        print(f"⚠️ TOP 5 STEALTH TOKENS ERROR: {e}")

# Import only essential modules
from utils.bybit_cache_manager import get_bybit_symbols_cached
from stages.stage_minus2_1 import detect_stage_minus2_1
# Legacy PPWCS/checklist functions removed - using TJDE v2 only
from utils.data_fetchers import get_market_data
from utils.alert_system import process_alert
from utils.coingecko import build_coingecko_cache
from utils.whale_priority import prioritize_whale_tokens
from utils.training_data_manager import TrainingDataManager

# Daily RL Training System Imports
try:
    from daily_rl_train_job import train_rl_v3_from_feedback
    from visual_weights_evolution import visualize_weight_evolution, generate_all_charts
    DAILY_RL_TRAINING_AVAILABLE = True
    print("✅ Daily RL Training System loaded")
except ImportError as e:
    DAILY_RL_TRAINING_AVAILABLE = False
    print(f"ℹ️ Daily RL Training System not available: {e}")

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
        
        # Legacy PPWCS/checklist removed - using TJDE v2 only
        # Alerts are now handled by TJDE and Stealth Engine

        # Training chart generation removed - handled by TJDE v2

        return {
            'symbol': symbol,
            'signals': signals
        }
        
    except Exception as e:
        print(f"Error scanning {symbol}: {e}")
        return None

def scan_cycle():
    """Enhanced scan cycle with async integration and comprehensive error tracking"""
    print(f"\nStarting scan cycle at {datetime.now().strftime('%H:%M:%S')}")
    
    # Clear warnings from previous scan
    clear_scan_warnings()
    
    # Automatyczne czyszczenie explore mode (co 10% skanów)
    if EXPLORE_CLEANUP_AVAILABLE and system_random.random() < 0.1:  # 10% szans na wywołanie
        try:
            cleanup_stats = cleanup_old_explore_mode_data()
            if cleanup_stats["removed_entries"] > 0:
                print(f"[EXPLORE CLEANUP] Removed {cleanup_stats['removed_entries']} old explore mode entries (>3 days)")
        except Exception as e:
            print(f"[EXPLORE CLEANUP ERROR] {e}")
    
    start_time = time.time()
    
    # Get symbols with error tracking
    try:
        symbols = get_bybit_symbols_cached()
        print(f"Retrieved {len(symbols)} symbols from Bybit...")
    except Exception as e:
        log_warning("SYMBOL FETCH ERROR", e)
        symbols = []
    
    # Build cache with error tracking
    try:
        build_coingecko_cache()
    except Exception as e:
        log_warning("COINGECKO CACHE BUILD ERROR", e)
    
    # Filter symbols by CoinGecko cache
    if symbols:
        try:
            from utils.coingecko import filter_tokens_by_coingecko_cache
            filtered_symbols, invalid_symbols = filter_tokens_by_coingecko_cache(symbols)
            
            print(f"[COINGECKO FILTER] DISABLED - skanowanie wszystkich {len(symbols)} tokenów:")
            print(f"[COINGECKO FILTER] ✅ Valid: {len(filtered_symbols)}")
            print(f"[COINGECKO FILTER] ℹ️ Invalid (skanowane): {len(invalid_symbols)}")
            
            # Use only valid symbols for scanning
            symbols = filtered_symbols
            
        except Exception as e:
            log_warning("COINGECKO FILTER ERROR", e)
            # Continue with original symbols if filtering fails
    
    print(f"Scanning {len(symbols)} valid symbols...")
    
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
    
    # Display TOP 5 stealth score tokens at end of scan
    display_top5_stealth_tokens()
    
    # === LAST10 RUNNER INTEGRATION ===
    # Run single LLM call for last 10 tokens after full scan cycle
    if LAST10_RUNNER_AVAILABLE:
        try:
            print("\n[LAST10 ANALYSIS] Running single LLM analysis for last 10 tokens...")
            last10_runner = get_last10_runner()
            results = last10_runner.run_one_call_for_last10(min_trust=0.2, min_liq_usd=50000)
            
            if results:
                print(f"[LAST10 SUCCESS] Analyzed {len(results)} detector items from last 10 tokens")
                
                # Aggregate results per token (optional)
                aggregated = last10_runner.aggregate_per_token(results)
                
                # Display summary
                buy_tokens = [token for token, data in aggregated.items() if data["decision"] == "BUY"]
                hold_tokens = [token for token, data in aggregated.items() if data["decision"] == "HOLD"]
                avoid_tokens = [token for token, data in aggregated.items() if data["decision"] == "AVOID"]
                
                print(f"[LAST10 SUMMARY] BUY: {len(buy_tokens)}, HOLD: {len(hold_tokens)}, AVOID: {len(avoid_tokens)}")
                
                if buy_tokens:
                    print(f"[LAST10 BUY SIGNALS] {', '.join(buy_tokens)}")
                    
            else:
                print("[LAST10 INFO] No items from last 10 tokens to analyze")
                
        except Exception as last10_error:
            print(f"[LAST10 ERROR] Failed to run last10 analysis: {last10_error}")
    else:
        print("[LAST10 INFO] Last10 runner not available")
    
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
    
    # Run Module 5: Dynamic Weight Adjustment every 4 cycles (~1 hour)
    try:
        from feedback_loop.weight_adjustment_system import adjust_label_weights, print_weight_report
        
        # Module-level cycle counter for weight adjustment tracking
        global cycle_counter
        try:
            cycle_counter += 1
        except NameError:
            cycle_counter = 1
        
        if cycle_counter % 4 == 0:  # Every 4th cycle
            print(f"\n🧠 [MODULE 5] Running Dynamic Weight Adjustment (Cycle {cycle_counter})")
            updated_weights = adjust_label_weights()
            print_weight_report()
            print(f"🧠 [MODULE 5] Weight adjustment complete - {len(updated_weights)} labels processed")
    except ImportError as import_error:
        log_warning("WEIGHT ADJUSTMENT MODULE IMPORT ERROR", import_error)
    except Exception as weight_error:
        log_warning("WEIGHT ADJUSTMENT ERROR", weight_error)
    
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
    # UWAGA: Na prywatnym serwerze ten proces może blokować skany przez długi czas!
    # Jeśli masz dużo plików w training_data/charts, zmień prawdopodobieństwo na mniejsze
    try:
        from hybrid_embedding_system import process_training_charts_for_embeddings
        
        # Process training charts for embeddings (very low frequency)
        # ZMNIEJSZONO z 0.02 (2%) na 0.0005 (0.05%) aby uniknąć blokowania skanów
        if system_random.random() < 0.0005:  # 0.05% chance - bardzo rzadko (raz na ~2000 skanów)
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
                    print(f"[{processed}/{len(symbols)}] Progress: {symbol}: Processed")
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

def run_feedback_evaluation_if_needed():
    """
    Sprawdza i uruchamia ewaluację feedback loop jeśli potrzeba
    """
    try:
        from feedback_loop.feedback_integration import should_run_evaluation_cycle, run_feedback_evaluation_cycle, update_label_weights_from_performance
        
        # Adaptive threshold learning system already imported globally
        import asyncio
        
        # Sprawdź czy należy uruchomić ewaluację
        if should_run_evaluation_cycle():
            print("[FEEDBACK SYSTEM] Starting automatic evaluation cycle...")
            
            # Uruchom ewaluację asynchronicznie
            result = asyncio.run(run_feedback_evaluation_cycle())
            
            if result.get("status") == "success":
                evaluated_count = result.get("evaluated_count", 0)
                print(f"[FEEDBACK SYSTEM] ✅ Evaluation completed: {evaluated_count} predictions evaluated")
                
                # Aktualizuj wagi etykiet na podstawie wyników
                if evaluated_count > 0:
                    weight_updates = update_label_weights_from_performance()
                    if weight_updates:
                        print(f"[FEEDBACK SYSTEM] ✅ Updated {len(weight_updates)} label weights based on performance")
                    else:
                        print("[FEEDBACK SYSTEM] ℹ️ No weight updates needed yet")
            else:
                error = result.get("error", "Unknown error")
                print(f"[FEEDBACK SYSTEM] ⚠️ Evaluation failed: {error}")
        else:
            print("[FEEDBACK SYSTEM] ℹ️ No evaluation needed - insufficient pending predictions")
            
    except Exception as e:
        print(f"[FEEDBACK SYSTEM ERROR] Failed to run evaluation: {e}")

def daily_self_train_at_night(last_run_file="cache/last_train.txt"):
    """
    Automated daily RL training at 02:00 UTC with duplicate prevention
    Trains RLAgentV3 and generates weight evolution charts
    """
    if not DAILY_RL_TRAINING_AVAILABLE:
        return
    
    try:
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")

        # Run only between 02:00 and 02:10 UTC
        if now.hour == 2 and now.minute < 10:
            # Check if already trained today
            if os.path.exists(last_run_file):
                try:
                    with open(last_run_file, "r") as f:
                        last_run_date = f.read().strip()
                        if last_run_date == today_str:
                            return  # Already trained today
                except Exception:
                    pass  # File corrupted, proceed with training

            print(f"[RL SELF-TRAIN] 🧠 Starting daily training at {now.strftime('%H:%M:%S')} UTC...")
            
            # Train RLAgentV3 from feedback data
            training_result = train_rl_v3_from_feedback()
            
            if training_result.get("status") == "success":
                updates = training_result.get("total_updates", 0)
                success_rate = training_result.get("success_rate", 0.0)
                weights = training_result.get("weights", {})
                
                print(f"[RL SELF-TRAIN] ✅ Training completed:")
                print(f"   • Updates processed: {updates}")
                print(f"   • Success rate: {success_rate}%")
                print(f"   • Updated weights: {weights}")
                
                # Generate weight evolution charts
                print("[RL SELF-TRAIN] 📊 Generating weight evolution charts...")
                chart_results = generate_all_charts()
                
                charts_generated = chart_results.get("total_generated", 0)
                print(f"[RL SELF-TRAIN] ✅ Generated {charts_generated} visualization charts")
                
                # Mark training as completed for today
                os.makedirs(os.path.dirname(last_run_file), exist_ok=True)
                with open(last_run_file, "w") as f:
                    f.write(today_str)
                
                print(f"[RL SELF-TRAIN] 🎯 Daily training cycle completed successfully")
                
            elif training_result.get("status") == "no_data":
                print("[RL SELF-TRAIN] ℹ️ No feedback data available for training")
            else:
                error = training_result.get("error", "Unknown error")
                print(f"[RL SELF-TRAIN] ❌ Training failed: {error}")
            
    except Exception as e:
        print(f"[RL SELF-TRAIN ERROR] Failed to execute daily training: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main scanning loop with integrated feedback evaluation"""
    global ADAPTIVE_LEARNING_AVAILABLE  # Declare global access to ADAPTIVE_LEARNING_AVAILABLE
    print("Starting Crypto Scanner Service (Enhanced with Feedback Loop)")
    
    # Initialize detector learning system at startup
    try:
        from stealth_engine.detector_learning_system import get_detector_learning_system
        learning_system = get_detector_learning_system()
        print(f"[SYSTEM INIT] ✅ Detector Learning System initialized successfully")
    except Exception as e:
        print(f"[SYSTEM INIT ERROR] ❌ Failed to initialize Detector Learning System: {e}")
    
    # Initialize threshold-aware learning scheduler
    try:
        from utils.threshold_scheduler import start_threshold_learning
        start_threshold_learning()
        print(f"[SYSTEM INIT] ✅ Threshold-Aware Learning System started successfully")
    except Exception as e:
        print(f"[SYSTEM INIT ERROR] ❌ Failed to start Threshold-Aware Learning System: {e}")
    
    # 🔍 COMPREHENSIVE STAGE 1-7 DIAGNOSTIC CHECK
    print("\n🔍 === STAGE 1-7 COMPREHENSIVE DIAGNOSTIC CHECK ===")
    
    # STAGE 1: CaliforniumWhale AI Temporal Graph + QIRL Detector
    try:
        from stealth.californium.californium_whale_detect import CaliforniumTGN, QIRLAgent
        print("✅ [STAGE 1/7] CaliforniumWhale AI Temporal Graph + QIRL Detector - OPERATIONAL")
    except ImportError as e:
        print(f"❌ [STAGE 1/7] CaliforniumWhale AI import failed: {e}")
    except Exception as e:
        print(f"⚠️ [STAGE 1/7] CaliforniumWhale AI error: {e}")
    
    # STAGE 2: DiamondWhale AI Integration
    try:
        from stealth_engine.diamond_detector import run_diamond_detector
        print("✅ [STAGE 2/7] DiamondWhale AI Stealth Engine Integration - OPERATIONAL")
    except ImportError as e:
        print(f"❌ [STAGE 2/7] DiamondWhale AI import failed: {e}")
    except Exception as e:
        print(f"⚠️ [STAGE 2/7] DiamondWhale AI error: {e}")
    
    # STAGE 3: Diamond Decision Engine
    try:
        from stealth_engine.decision import simulate_diamond_decision
        print("✅ [STAGE 3/7] Diamond Decision Engine - OPERATIONAL")
    except ImportError as e:
        print(f"❌ [STAGE 3/7] Diamond Decision Engine import failed: {e}")
    except Exception as e:
        print(f"⚠️ [STAGE 3/7] Diamond Decision Engine error: {e}")
    
    # STAGE 4: Diamond Alert Telegram System
    try:
        from alerts.telegram_notification import send_diamond_alert_auto
        print("✅ [STAGE 4/7] Diamond Alert Telegram System - OPERATIONAL")
    except ImportError as e:
        print(f"❌ [STAGE 4/7] Diamond Alert System import failed: {e}")
    except Exception as e:
        print(f"⚠️ [STAGE 4/7] Diamond Alert System error: {e}")
    
    # STAGE 5: Fusion Engine Multi-Detector
    try:
        from stealth_engine.fusion_layer import FusionEngine
        print("✅ [STAGE 5/7] Fusion Engine Multi-Detector - OPERATIONAL")
    except ImportError as e:
        print(f"❌ [STAGE 5/7] Fusion Engine import failed: {e}")
    except Exception as e:
        print(f"⚠️ [STAGE 5/7] Fusion Engine error: {e}")
    
    # STAGE 6: Diamond Scheduler (DETAILED DIAGNOSTIC)
    print("\n🔍 [STAGE 6/7] DETAILED DIAMOND SCHEDULER DIAGNOSTIC:")
    try:
        # Test individual imports
        print("  🔍 Testing scheduler module import...")
        import scheduler
        print(f"  ✅ Scheduler module imported: {scheduler.__file__}")
        
        print("  🔍 Testing scheduler functions...")
        from scheduler import start_diamond_scheduler_thread, manual_run
        print(f"  ✅ Scheduler functions imported: start_diamond_scheduler_thread, manual_run")
        
        print("  🔍 Testing feedback loop import...")
        from feedback.feedback_loop_diamond import get_diamond_feedback_loop
        print(f"  ✅ Feedback loop imported successfully")
        
        print("  🔍 Testing scheduler execution...")
        diamond_scheduler_thread = start_diamond_scheduler_thread()
        print("✅ [STAGE 6/7] DiamondWhale AI Scheduler started - daily training automation active")
        print("   • Daily feedback loop: 02:00 UTC")
        print("   • Model checkpoints: 02:15 UTC") 
        print("   • Hourly pending checks: every hour at :30")
        
    except ImportError as e:
        print(f"❌ [STAGE 6/7] Diamond Scheduler import failed: {e}")
        print(f"  🔍 Import error details: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"  🔍 Full traceback:")
        traceback.print_exc()
    except Exception as diamond_error:
        print(f"⚠️ [STAGE 6/7] Diamond Scheduler execution error: {diamond_error}")
        print(f"  🔍 Error details: {type(diamond_error).__name__}: {str(diamond_error)}")
        import traceback
        print(f"  🔍 Full traceback:")
        traceback.print_exc()
    
    # STAGE 7: RLAgentV4 RL Fusion Agent
    try:
        from stealth_engine.rl.fusion_rl_agent import get_rl_fusion_agent
        print("✅ [STAGE 7/7] RLAgentV4 RL Fusion Agent - OPERATIONAL")
    except ImportError as e:
        print(f"❌ [STAGE 7/7] RLAgentV4 import failed: {e}")
    except Exception as e:
        print(f"⚠️ [STAGE 7/7] RLAgentV4 error: {e}")
    
    print("=" * 60)
    
    # 🎯 ETAP 10 - URUCHOMIENIE TELEGRAM ALERT MANAGER
    try:
        from stealth_engine.telegram_alert_manager import get_telegram_manager
        telegram_manager = get_telegram_manager()
        telegram_manager.start_processing_loop(interval=10)
        print("✅ [STAGE 10] Telegram Alert Manager uruchomiony - kolejka alertów aktywna")
    except ImportError:
        print("ℹ️ [STAGE 10] Telegram Alert Manager niedostępny")
    except Exception as telegram_error:
        print(f"⚠️ [STAGE 10] Błąd uruchamiania Telegram Alert Manager: {telegram_error}")
    
    # 🧠 ETAP 11 - URUCHOMIENIE PRIORITY LEARNING MEMORY SYSTEM
    try:
        from stealth_engine.priority_learning import get_priority_learning_memory
        from stealth_engine.stealth_scanner import get_stealth_scanner
        
        priority_memory = get_priority_learning_memory()
        stealth_scanner = get_stealth_scanner()
        
        print("✅ [STAGE 11] Priority Learning Memory system uruchomiony - inteligentne priorytetowanie tokenów aktywne")
        
        # 🎓 PUMP VERIFICATION SCHEDULER - Agent Learning System
        try:
            from agent_learning.pump_verification_scheduler import start_pump_verification_scheduler
            start_pump_verification_scheduler()
            print("✅ [STAGE 11+] Pump Verification Scheduler uruchomiony - agent learning z pump verification co 6h")
        except ImportError:
            print("ℹ️ [STAGE 11+] Pump Verification Scheduler niedostępny")
        except Exception as pump_error:
            print(f"⚠️ [STAGE 11+] Błąd uruchamiania Pump Verification Scheduler: {pump_error}")
        
        # Pokaż statystyki uczenia
        learning_stats = priority_memory.get_learning_statistics()
        print(f"ℹ️ [STAGE 11] Pamięć uczenia: {learning_stats['total_entries']} wpisów, "
              f"{learning_stats.get('success_rate', learning_stats.get('overall_success_rate', 0.0)):.1%} skuteczność")
              
    except ImportError:
        print("ℹ️ [STAGE 11] Priority Learning Memory system niedostępny")
    except Exception as learning_error:
        print(f"⚠️ [STAGE 11] Błąd uruchamiania Priority Learning: {learning_error}")
    
    # Licznik cykli dla harmonogramowania feedback
    cycle_count = 0
    
    try:
        while True:
            try:
                clear_scan_warnings()  # Clear warnings at start of each cycle
                
                # 🧠 Daily RL Training System - Check for 02:00 UTC training window
                daily_self_train_at_night()
                
                scan_cycle()
                report_scan_warnings()  # Report warnings at end of cycle
                
                # Uruchom feedback evaluation co 4 cykle (co ~1 godzinę)
                cycle_count += 1
                if cycle_count % 4 == 0:
                    run_feedback_evaluation_if_needed()
                    
                    # 🧠 ETAP 11 - PRIORITY LEARNING FEEDBACK PROCESSING
                    try:
                        from stealth_engine.stealth_scanner import process_stealth_learning_feedback
                        import asyncio
                        
                        # Process stealth-ready tokens dla learning memory (6h evaluation)
                        processed_tokens = asyncio.run(process_stealth_learning_feedback([], hours=6))
                        if processed_tokens > 0:
                            print(f"[STAGE 11] ✅ Processed {processed_tokens} stealth-ready tokens for priority learning")
                        else:
                            print("[STAGE 11] ℹ️ No stealth-ready tokens pending evaluation")
                            
                    except ImportError:
                        print("[STAGE 11] ℹ️ Priority learning feedback system not available")
                    except Exception as learning_feedback_error:
                        print(f"[STAGE 11 ERROR] Priority learning feedback: {learning_feedback_error}")
                    
                    # 🎯 ADAPTIVE THRESHOLD MAINTENANCE
                    if ADAPTIVE_LEARNING_AVAILABLE:
                        try:
                            import asyncio
                            evaluated_threshold_count = asyncio.run(run_adaptive_threshold_maintenance())
                            if evaluated_threshold_count > 0:
                                print(f"[ADAPTIVE LEARNING] ✅ Evaluated {evaluated_threshold_count} tokens for threshold learning")
                            else:
                                print("[ADAPTIVE LEARNING] ℹ️ No threshold evaluation needed")
                        except Exception as adaptive_error:
                            print(f"[ADAPTIVE LEARNING ERROR] {adaptive_error}")
                
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