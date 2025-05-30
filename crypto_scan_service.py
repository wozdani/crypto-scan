# === crypto_scan_service.py ===
import os
import time
import glob
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# === Load ENV ===
load_dotenv()
from utils.telegram_bot import send_alert, format_alert
from utils.data_fetchers import get_symbols_cached, get_market_data
from stages.stage_minus2_1 import detect_stage_minus2_1
from utils.stage_detectors import detect_stage_minus1
from utils.scoring import compute_ppwcs, should_alert, log_ppwcs_score, get_previous_score, save_score
from utils.gpt_feedback import send_report_to_chatgpt
from utils.alert_system import process_alert
from utils.reports import save_stage_signal, save_conditional_reports, compress_reports_to_zip

# === Wait for next 15m candle ===
def wait_for_next_candle():
    now = datetime.utcnow()
    next_quarter = (now.minute // 15 + 1) * 15 % 60
    next_time = now.replace(minute=next_quarter, second=5, microsecond=0)
    if next_quarter == 0:
        next_time += timedelta(hours=1)
    wait_seconds = (next_time - now).total_seconds()
    if wait_seconds > 0:
        print(f"⏳ Waiting {wait_seconds:.1f}s for next candle at {next_time.strftime('%H:%M:%S')} UTC")
        time.sleep(wait_seconds)

# === Main scan cycle ===
def scan_cycle():
    print(f"\n🔁 Start scan: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    symbols = get_symbols_cached()
    scan_results = []
    
    for symbol in symbols:
        try:
            data = get_market_data(symbol)
            if not data:
                continue
                
            stage2_pass, signals, inflow, stage1g_active = detect_stage_minus2_1(symbol)
            compressed = detect_stage_minus1(signals) if signals else False
            
            # Get previous score for trailing logic
            previous_score = get_previous_score(symbol)
            score = compute_ppwcs(signals, previous_score)
            
            # Save current score for future trailing logic
            save_score(symbol, score)
            
            log_ppwcs_score(symbol, score)
            save_stage_signal(symbol, score, stage2_pass, compressed, stage1g_active)
            
            scan_results.append({
                'symbol': symbol,
                'score': score,
                'timestamp': datetime.utcnow().isoformat(),
                'stage2_pass': stage2_pass,
                'compressed': compressed,
                'stage1g_active': stage1g_active
            })
            
            # Process alerts using the comprehensive alert system
            from utils.alert_utils import get_alert_level, should_request_gpt_analysis
            
            gpt_analysis = None
            alert_level = get_alert_level(score)
            
            if should_request_gpt_analysis(alert_level):
                try:
                    # Get GPT analysis for strong alerts only
                    event_tag = signals.get('event_tag')
                    event_tags: list[str] = [str(event_tag)] if event_tag else []
                    gpt_analysis = send_report_to_chatgpt(symbol, event_tags, score, bool(compressed), stage1g_active)
                except Exception as gpt_error:
                    print(f"⚠️ GPT analysis failed for {symbol}: {gpt_error}")
            
            # Process alert with the new system
            process_alert(symbol, score, signals, gpt_analysis)
                        
        except Exception as e:
            print(f"❌ Error scanning {symbol}: {e}")
            
    save_conditional_reports()
    compress_reports_to_zip()
    print(f"✅ Scan completed. Processed {len(scan_results)} symbols.")
    return scan_results

# === Main execution ===
if __name__ == "__main__":
    print("🚀 Crypto Pre-Pump Detection System Starting...")
    print("Press Ctrl+C to stop")
    
    try:
        # Initial setup
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("data/alerts", exist_ok=True)
        os.makedirs("data/scores", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        # === Loop forever ===
        while True:
            try:
                scan_cycle()
                wait_for_next_candle()
            except KeyboardInterrupt:
                print("\n🛑 Shutting down gracefully...")
                break
            except Exception as e:
                print(f"💥 Critical error in main loop: {e}")
                print("🔄 Restarting in 60 seconds...")
                time.sleep(60)
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
