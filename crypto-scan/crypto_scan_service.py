#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

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

        # Training chart generation for quality setups
        if final_score >= 40 or checklist_score >= 35:
            print(f"[TRAINING] {symbol} → Generating training chart (PPWCS: {final_score}, Checklist: {checklist_score})")
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
    """Enhanced scan cycle with async integration"""
    print(f"\nStarting scan cycle at {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    # Get symbols
    symbols = get_bybit_symbols_cached()
    print(f"Scanning {len(symbols)} symbols...")
    
    # Build cache
    build_coingecko_cache()
    
    # Use enhanced async scan with TJDE and chart generation
    try:
        print("Running enhanced async scan with TJDE analysis...")
        
        # Import and run full async scan cycle with TJDE
        import asyncio
        from scan_all_tokens_async import async_scan_cycle
        
        # Run enhanced async scan cycle
        result = asyncio.run(async_scan_cycle())
        processed_count = result if isinstance(result, int) else 0
        
        print(f"Enhanced async scan processed {processed_count} tokens with TJDE analysis")
        print("Async scan completed successfully")
        
        # Flush results
        try:
            from scan_token_async import flush_async_results
            flush_async_results()
        except Exception as flush_e:
            print(f"Flush error: {flush_e}")
        
    except Exception as e:
        print(f"Async scan failed ({e}), falling back to simple scan...")
        
        # Fallback to async batch scan with reduced concurrency
        simple_scan_fallback(symbols)
    
    # Show timing
    elapsed = time.time() - start_time
    print(f"Scan cycle completed in {elapsed:.1f}s")
    
    # Run memory feedback evaluation periodically
    try:
        from utils.memory_feedback_loop import run_memory_feedback_evaluation
        
        # Run feedback evaluation every few cycles
        import random
        if random.random() < 0.3:  # 30% chance each cycle
            run_memory_feedback_evaluation()
    except ImportError:
        pass
    
    # Run Phase 2 memory outcome updates periodically
    try:
        from token_context_memory import update_historical_outcomes_loop
        
        # Update historical outcomes every few cycles
        if random.random() < 0.2:  # 20% chance each cycle
            update_historical_outcomes_loop()
            
    except Exception as feedback_e:
        print(f"[MEMORY_FEEDBACK_ERROR] {feedback_e}")

def simple_scan_fallback(symbols):
    """Fallback using async batch scan with lower concurrency"""
    print("Running async batch fallback with reduced concurrency...")
    
    try:
        # Use async batch scanning even in fallback but with lower concurrency
        import asyncio
        from scan_all_tokens_async import scan_symbols_async
        
        # Run with lower concurrency for stability
        results = asyncio.run(scan_symbols_async(symbols[:100], max_concurrent=8))
        print(f"Async batch fallback processed {len(results)} tokens")
        return results
        
    except Exception as e:
        print(f"Async batch fallback failed: {e}")
        print("Using simple sequential scan as last resort...")
        
        # Ultimate fallback - simple sequential for core functionality only
        processed = 0
        results = []
        
        for symbol in symbols[:10]:  # Very limited fallback
            result = scan_single_token(symbol)
            if result:
                results.append(result)
                processed += 1
                print(f"[{processed}/10] {symbol}: PPWCS {result.get('final_score', 0):.1f}")
        
        print(f"Sequential fallback processed {processed} tokens")
        return results
        return []

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
                scan_cycle()
                wait_for_next_candle()
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Scan error: {e}")
                time.sleep(60)  # Wait 1 minute before retry
                
    except KeyboardInterrupt:
        print("Service stopped.")

if __name__ == "__main__":
    main()