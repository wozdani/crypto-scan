import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import glob
import json
import openai
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# === Load ENV ===
load_dotenv()
from utils.telegram_bot import send_alert, format_alert
from utils.data_fetchers import get_symbols_cached, get_market_data
from stages.stage_minus2_1 import detect_stage_minus2_1
from utils.stage_detectors import detect_stage_minus1
from utils.scoring import compute_ppwcs, should_alert, log_ppwcs_score, get_previous_score, save_score
from utils.gpt_feedback import send_report_to_chatgpt, score_gpt_feedback, categorize_feedback_score
from utils.alert_system import process_alert
from utils.reports import save_stage_signal, save_conditional_reports, compress_reports_to_zip

def send_report_to_gpt(symbol, data, tp_forecast, alert_level):
    """Send comprehensive signal report to GPT for analysis"""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    ppwcs = data.get("ppwcs_score", 0)
    whale = data.get("whale_activity", False)
    inflow = data.get("dex_inflow", 0)
    compressed = data.get("compressed", False)
    stage1g = data.get("stage1g_active", False)
    pure_acc = data.get("pure_accumulation", False)
    # social_spike removed - handled by Stage -2.2 tags
    heatmap_exhaustion = data.get("heatmap_exhaustion", False)
    sector_cluster = data.get("sector_clustered", False)
    spoofing = data.get("spoofing_suspected", False)
    vwap_pinned = data.get("vwap_pinned", False)
    vol_slope = data.get("volume_slope_up", False)

    timestamp = datetime.now(timezone.utc).strftime("%H:%M UTC")

    prompt = f"""You are an expert crypto analyst. Evaluate the following pre-pump signal:

Token: ${symbol.upper()}
PPWCS: {ppwcs}
Alert Level: {alert_level}
Detected at: {timestamp}

Stage –2.1:
• Whale Activity: {whale}
• DEX Inflow (USD): {inflow}
• News/Tag Analysis: {data.get("event_tag", "none")}
• Sector Time Clustering Active: {sector_cluster}

Stage –1:
• Compressed Structure: {compressed}

Stage 1g:
• Active: {stage1g}
• Pure Accumulation (No Social): {pure_acc}

Structural Detectors:
• Heatmap Exhaustion: {heatmap_exhaustion}
• Spoofing Suspected: {spoofing}
• VWAP Pinned: {vwap_pinned}
• Volume Slope Up: {vol_slope}

TP Forecast:
• TP1: +{tp_forecast['TP1']}%
• TP2: +{tp_forecast['TP2']}%
• TP3: +{tp_forecast['TP3']}%
• Trailing: {tp_forecast['TrailingTP']}%

Evaluate the quality and strength of this signal. Provide a confident but concise assessment in **3 short sentences**, including **any risk factors** and **probability of continuation** in Polish."""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a crypto signal quality evaluator. Respond in Polish."},
                {"role": "user", "content": prompt}
            ],
            timeout=15
        )
        return response.choices[0].message.content.strip() if response.choices[0].message.content else "No response"
    except Exception as e:
        return f"[GPT ERROR] {str(e)}"

# === Wait for next 15m candle ===
def wait_for_next_candle():
    now = datetime.now(timezone.utc)
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
    print(f"\n🔁 Start scan: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    symbols = get_symbols_cached()
    
    # Automatyczne pobieranie brakujących kontraktów z CoinGecko
    from utils.coingecko_mapper import bulk_update_contracts
    bulk_update_contracts(symbols)
    
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
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stage2_pass': stage2_pass,
                'compressed': compressed,
                'stage1g_active': stage1g_active
            })
            
            # Process alerts using the comprehensive alert system
            from utils.alert_utils import get_alert_level, should_request_gpt_analysis
            
            gpt_analysis = None
            alert_level = get_alert_level(score)
            
            # Generate TP forecast for all qualifying signals
            from utils.take_profit_engine import forecast_take_profit_levels
            tp_forecast = forecast_take_profit_levels(signals)
            
            # Enhanced GPT feedback for strong signals (PPWCS >= 80)
            gpt_feedback = None
            feedback_score = None
            if score >= 80:
                try:
                    # Pass alert level to GPT function
                    alert_level_text = get_alert_level(score)
                    gpt_feedback = send_report_to_gpt(symbol, signals, tp_forecast, alert_level_text)
                    
                    # Auto-score the GPT feedback
                    feedback_score = score_gpt_feedback(gpt_feedback)
                    category, description, emoji = categorize_feedback_score(feedback_score)
                    
                    print(f"[GPT FEEDBACK] {symbol}: {gpt_feedback}")
                    print(f"[FEEDBACK SCORE] {symbol}: {feedback_score}/100 ({category}) {emoji}")
                    
                    # Add feedback score to signals data
                    signals["feedback_score"] = feedback_score
                    signals["feedback_category"] = category
                    
                    # Create feedback directory and save report
                    os.makedirs("data/feedback", exist_ok=True)
                    feedback_file = f"data/feedback/{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.txt"
                    with open(feedback_file, "w", encoding="utf-8") as f:
                        f.write(f"Token: {symbol}\n")
                        f.write(f"PPWCS: {score}\n")
                        f.write(f"Alert Level: {alert_level_text}\n")
                        f.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
                        f.write(f"Signals: {signals}\n")
                        f.write(f"TP Forecast: {tp_forecast}\n")
                        f.write(f"Feedback Score: {feedback_score}/100\n")
                        f.write(f"Feedback Category: {category} ({description})\n")
                        f.write(f"GPT Feedback:\n{gpt_feedback}\n")
                        
                except Exception as gpt_error:
                    print(f"⚠️ GPT feedback failed for {symbol}: {gpt_error}")
            
            # Send alert with GPT feedback (if available) for all qualifying signals
            if score >= 60:  # Minimum threshold for any action
                from utils.alert_system import process_alert
                process_alert(symbol, score, signals, gpt_feedback)
                        
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
