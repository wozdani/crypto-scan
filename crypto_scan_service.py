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
from utils.stage_detectors import (
    detect_stage_minus2,
    detect_stage_minus2_2,
    detect_stage_minus1,
    detect_stage_1g,
)
from utils.scoring import compute_ppwcs, should_alert, log_ppwcs_score
from utils.gpt_feedback import send_report_to_chatgpt
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
                
            stage2_pass, signals, inflow = detect_stage_minus2(symbol, data)
            event_tags = detect_stage_minus2_2(symbol)
            compressed = detect_stage_minus1(signals)
            stage1g_active = detect_stage_1g(symbol, data, event_tags)
            score = compute_ppwcs(symbol, signals, compressed, stage1g_active, event_tags)
            
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
            
            if should_alert(symbol, score):
                gpt_analysis = None
                
                # Get GPT analysis for high scores first
                if score >= 80:
                    try:
                        gpt_analysis = send_report_to_chatgpt(symbol, signals, score)
                        print(f"🤖 ChatGPT analysis completed for {symbol}")
                    except Exception as gpt_error:
                        print(f"⚠️ ChatGPT analysis failed for {symbol}: {gpt_error}")
                
                # Send alert with GPT analysis if available
                alert_message = format_alert(symbol, score, event_tags, compressed, stage1g_active, gpt_analysis)
                send_alert(alert_message)
                print(f"📢 Alert sent for {symbol} with score {score}")
                        
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
