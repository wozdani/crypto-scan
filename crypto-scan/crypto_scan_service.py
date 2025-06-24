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
    """Simple sequential scan"""
    print(f"\nStarting scan cycle at {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    # Get symbols
    symbols = get_bybit_symbols_cached()
    print(f"Scanning {len(symbols)} symbols...")
    
    # Build cache
    build_coingecko_cache()
    
    # Whale priority
    symbols, priority_symbols, priority_info = prioritize_whale_tokens(symbols)
    
    # Async scan using scan_all_tokens_async
    import asyncio
    from scan_all_tokens_async import scan_symbols_async
    
    try:
        results = asyncio.run(scan_symbols_async(symbols))
        duration = time.time() - start_time
        processed = len(results) if results else 0
        print(f"Async scan completed in {duration:.1f}s, processed {processed} tokens")
        
        # Flush any remaining async results to storage
        try:
            from scan_token_async import flush_async_results
            asyncio.run(flush_async_results())
        except:
            pass
        
        return results
    except Exception as e:
        print(f"Async scan error: {e}")
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