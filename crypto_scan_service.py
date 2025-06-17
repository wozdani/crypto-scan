import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import time
import glob
import json
import openai
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables at the start
load_dotenv()
from utils.contracts import get_contract
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.coingecko import build_coingecko_cache
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# === Load ENV ===
load_dotenv()
from utils.telegram_bot import send_alert, format_alert
from utils.data_fetchers import get_symbols_cached, get_market_data
from utils.data_fetchers import build_bybit_symbol_cache_all_categories as build_bybit_symbol_cache, is_bybit_cache_expired

if is_bybit_cache_expired():
    print("🕒 Cache symboli Bybit jest przestarzały – buduję ponownie...")
    build_bybit_symbol_cache()

from stages.stage_minus2_1 import detect_stage_minus2_1
from utils.stage_detectors import detect_stage_minus1
from utils.scoring import compute_ppwcs, should_alert, log_ppwcs_score, get_previous_score, save_score
from utils.gpt_feedback import send_report_to_chatgpt, score_gpt_feedback, categorize_feedback_score
from utils.alert_system import process_alert
from utils.reports import save_stage_signal, save_conditional_reports, compress_reports_to_zip

def check_momentum_kill_switch(symbol, candle_data, signals):
    """
    Momentum Kill-Switch - Pre-Pump 1.0 Integration
    Anulowanie sygnału, jeśli po wybiciu nie ma kontynuacji
    """
    try:
        if not candle_data or not signals:
            return False
            
        # Sprawdź czy Stage 1g był aktywny
        if not signals.get("stage1g_active"):
            return False
            
        # Pobierz ostatnią świeczkę
        if isinstance(candle_data, list) and len(candle_data) > 0:
            last_candle = candle_data[-1]
        elif isinstance(candle_data, dict):
            last_candle = candle_data
        else:
            return False
            
        # Pobierz dane świecy
        open_price = float(last_candle.get('open', 0))
        close_price = float(last_candle.get('close', 0))
        high_price = float(last_candle.get('high', 0))
        low_price = float(last_candle.get('low', 0))
        volume = float(last_candle.get('volume', 0))
        
        if any(x <= 0 for x in [open_price, close_price, high_price, low_price]):
            return False
            
        # Oblicz body ratio
        candle_range = high_price - low_price
        body_size = abs(close_price - open_price)
        body_ratio = body_size / candle_range if candle_range > 0 else 0
        
        # Oblicz RSI z ostatnich 5 świec (uproszczone)
        if isinstance(candle_data, list) and len(candle_data) >= 5:
            recent_closes = [float(c.get('close', 0)) for c in candle_data[-5:]]
            gains = []
            losses = []
            for i in range(1, len(recent_closes)):
                change = recent_closes[i] - recent_closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0.0001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50  # Neutralny RSI jako fallback
            
        # Warunki kill-switch
        weak_body = body_ratio < 0.4
        weak_rsi = rsi < 60
        low_volume = volume < signals.get('avg_volume', volume * 1.5) * 1.5
        
        momentum_killed = weak_body and weak_rsi and low_volume
        
        if momentum_killed:
            print(f"[MOMENTUM KILL-SWITCH] Alert cancelled for {symbol}: body_ratio={body_ratio:.2f}, RSI={rsi:.1f}")
            return True
            
        return False
        
    except Exception as e:
        print(f"[MOMENTUM KILL-SWITCH] Error for {symbol}: {e}")
        return False

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
    from utils.cache_utils import should_rebuild_cache
 
    if should_rebuild_cache():
        print("🛠 Cache CoinGecko wygasł - buduję cache...")
        build_coingecko_cache()
    else:
        print("✅ Cache CoinGecko jest aktualny - pomijam rebuild")

    symbols = get_symbols_cached()
    symbols_bybit = symbols  # Używaj tych samych symboli
        
        
    scan_results = []

    def run_detect_stage(symbol, price_usd):
        try:
            print(f"🧪 Wywołanie detect_stage_minus2_1({symbol})")
            stage2_pass, signals, inflow_usd, stage1g_active = detect_stage_minus2_1(symbol, price_usd=price_usd)
            print(f"[DEBUG] {symbol} signals: {signals}")
            return symbol, (stage2_pass, signals, inflow_usd, stage1g_active)
        except Exception as e:
            print(f"❌ Error in detect_stage_minus2_1 for {symbol}: {e}")
            return symbol, (False, {}, 0.0, False)

    futures = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        for symbol in symbols:
            print(f"🔍 Skanuję {symbol}...")
            if symbol not in symbols_bybit:
                print(f"⚠️ Pomijam {symbol} – nie znajduje się na Bybit (USDT perp)")
                continue

            try:
                success, data, price_usd, is_valid = get_market_data(symbol)

                if not success or not data or not isinstance(data, dict) or price_usd is None:
                    logger.warning(f"⚠️ Pomiń {symbol} – niepoprawne dane z get_market_data(): {data}")
                    continue

                # 📉 Filtrowanie tokenów: zbyt tanie, niska płynność, szeroki spread
                volume_usdt = data.get("volume")
                best_ask = data.get("best_ask")
                best_bid = data.get("best_bid")

                if not price_usd or price_usd <= 0:
                    print(f"⚠️ Pominięto {symbol} – nieprawidłowa cena: {price_usd}")
                    continue

                if not volume_usdt or volume_usdt < 100_000:
                    print(f"⚠️ Pominięto {symbol} – zbyt niski wolumen: ${volume_usdt}")
                    continue

                if best_ask and best_bid:
                    spread = (best_ask - best_bid) / best_ask
                    if spread > 0.02:
                        print(f"⚠️ Pominięto {symbol} – zbyt szeroki spread: {spread*100:.2f}%")
                        continue

                # ✅ Token przeszedł wszystkie filtry
                futures.append(executor.submit(run_detect_stage, symbol, price_usd))

            except Exception as e:
                print(f"❌ Błąd przy analizie {symbol}: {e}")
                continue

    for future in as_completed(futures):
        symbol, (stage2_pass, signals, inflow, stage1g_active) = future.result()
        try:
            compressed = detect_stage_minus1(signals) if signals else False
            previous_score = get_previous_score(symbol)
            final_score, ppwcs_structure, ppwcs_quality = compute_ppwcs(signals, previous_score)
            
            # Integrate checklist scoring with scan cycle
            from utils.scoring import compute_checklist_score
            checklist_score, checklist_summary = compute_checklist_score(signals)
            
            # Add checklist data to signals for downstream processing
            signals["checklist_score"] = checklist_score
            signals["checklist_summary"] = checklist_summary
            
            save_score(symbol, final_score)
            log_ppwcs_score(symbol, final_score, signals)
            save_stage_signal(symbol, final_score, stage2_pass, compressed, stage1g_active, checklist_score, checklist_summary)

            scan_results.append({
                'symbol': symbol,
                'score': final_score,
                'checklist_score': checklist_score,
                'checklist_summary': checklist_summary,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stage2_pass': stage2_pass,
                'compressed': compressed,
                'stage1g_active': stage1g_active
            })

            from utils.take_profit_engine import forecast_take_profit_levels
            tp_forecast = forecast_take_profit_levels(signals)

            # Enhanced 3-Tier Alert Confidence System with Checklist Integration
            allow_alert = False
            alert_tier = None
            is_high_confidence = False
            structure_ok = False
            
            # Check checklist scoring conditions
            if checklist_score >= 50 and final_score >= 70:
                is_high_confidence = True
                print(f"[HIGH CONFIDENCE] {symbol}: PPWCS={final_score}, Checklist={checklist_score}")
            
            if checklist_score >= 60:
                structure_ok = True
                print(f"[STRUCTURE OK] {symbol}: Checklist score {checklist_score}/100")
            
            # Stage 1g quality boost allows alerts at lower PPWCS
            if signals.get("stage1g_active") and ppwcs_quality >= 12:
                allow_alert = True
            
            # Enhanced alert tiers with checklist integration
            if is_high_confidence and final_score >= 80 and ppwcs_quality >= 14 and compressed:
                alert_tier = "🔴 Urgent Alert [HIGH CONFIDENCE]"
                allow_alert = True
            elif final_score >= 80 and ppwcs_quality >= 14 and compressed:
                alert_tier = "🔴 Urgent Alert"
                allow_alert = True
            elif is_high_confidence and final_score >= 70:
                alert_tier = "🟠 Pre-pump Active [HIGH CONFIDENCE]"
                allow_alert = True
            elif final_score >= 70 or ppwcs_quality >= 12:
                alert_tier = "🟠 Pre-pump Active"
                allow_alert = True
            elif 60 <= final_score <= 69 and ppwcs_quality < 10:
                alert_tier = "🟡 Watchlist"
            
            # Add structure tag to alert tier if applicable
            if structure_ok and alert_tier and "[HIGH CONFIDENCE]" not in alert_tier:
                alert_tier = alert_tier.replace("🟠 Pre-pump Active", "🟠 Pre-pump Active [STRUCTURE OK]")
                alert_tier = alert_tier.replace("🔴 Urgent Alert", "🔴 Urgent Alert [STRUCTURE OK]")

            gpt_feedback = None
            # Enhanced GPT conditions to include high confidence alerts
            gpt_alert_conditions = [
                "🔴 Urgent Alert", "🟠 Pre-pump Active", 
                "🔴 Urgent Alert [HIGH CONFIDENCE]", "🟠 Pre-pump Active [HIGH CONFIDENCE]",
                "🔴 Urgent Alert [STRUCTURE OK]", "🟠 Pre-pump Active [STRUCTURE OK]"
            ]
            
            if allow_alert and any(tier in alert_tier for tier in gpt_alert_conditions if alert_tier):
                try:
                    gpt_feedback = send_report_to_gpt(symbol, signals, tp_forecast, alert_tier)
                    feedback_score = score_gpt_feedback(gpt_feedback)
                    category, description, emoji = categorize_feedback_score(feedback_score)
                    print(f"[GPT FEEDBACK] {symbol}: {gpt_feedback}")
                    print(f"[FEEDBACK SCORE] {symbol}: {feedback_score}/100 ({category}) {emoji}")
                    signals["feedback_score"] = feedback_score
                    signals["feedback_category"] = category
                    os.makedirs("data/feedback", exist_ok=True)
                    feedback_file = f"data/feedback/{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.txt"
                    with open(feedback_file, "w", encoding="utf-8") as f:
                        f.write(f"Token: {symbol}\nPPWCS: {final_score} (Structure: {ppwcs_structure}, Quality: {ppwcs_quality})\n")
                        f.write(f"Checklist Score: {checklist_score}/100 ({len(checklist_summary)}/20 conditions)\n")
                        f.write(f"Alert Tier: {alert_tier}\nTimestamp: {datetime.now(timezone.utc).isoformat()}\n")
                        f.write(f"Signals: {signals}\nTP Forecast: {tp_forecast}\nFeedback Score: {feedback_score}/100\n")
                        f.write(f"Feedback Category: {category} ({description})\nGPT Feedback:\n{gpt_feedback}\n")
                except Exception as gpt_error:
                    print(f"⚠️ GPT feedback failed for {symbol}: {gpt_error}")

            if allow_alert and alert_tier:
                print(f"✅ Alert triggered: {symbol} - {alert_tier} (Score: {final_score}, Quality: {ppwcs_quality})")
                
                # Use enhanced send_alert function with checklist integration
                from utils.alert_system import send_alert
                alert_success = send_alert(
                    symbol=symbol,
                    ppwcs=final_score,
                    checklist_score=checklist_score,
                    checklist_summary=checklist_summary,
                    signals=signals
                )
                
                if alert_success:
                    print(f"📢 Enhanced alert sent for {symbol} with checklist integration")
                else:
                    print(f"❌ Failed to send enhanced alert for {symbol}")
                
                # Fallback to legacy system for additional features (TP forecast, GPT)
                from utils.alert_system import process_alert
                process_alert(symbol, final_score, signals, gpt_feedback)

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
