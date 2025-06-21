import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.stdout.reconfigure(encoding='utf-8')  # Commented out for compatibility
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
# === CONFIGURATION ===

if is_bybit_cache_expired():
    print("🕒 Cache symboli Bybit jest przestarzały – buduję ponownie...")
    build_bybit_symbol_cache()

from stages.stage_minus2_1 import detect_stage_minus2_1

from utils.scoring import compute_ppwcs, should_alert, log_ppwcs_score, get_previous_score, save_score
from utils.datetime_utils import get_utc_hour, get_utc_timestamp
from utils.trend_alert_cache import trend_alert_cache
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
    
    # 🐋 WHALE PRIORITY SYSTEM - Priorytetowanie na podstawie wcześniejszej aktywności whale
    from utils.whale_priority import prioritize_whale_tokens, save_priority_report, get_whale_boost_for_symbol
    
    symbols, priority_symbols, priority_info = prioritize_whale_tokens(symbols)
    
    # Zapisz raport priorytetowania
    if priority_info:
        save_priority_report(priority_info)
        
    scan_results = []

    def run_detect_stage(symbol, price_usd):
        try:
            # Run pre-pump analysis
            stage2_pass, signals, inflow_usd, stage1g_active = detect_stage_minus2_1(symbol, price_usd=price_usd)
            
            # === PARALLEL TREND-MODE ANALYSIS ===
            # Run trend-mode analysis simultaneously with pre-pump PPWCS
            try:
                print(f"[TREND DEBUG] {symbol}: Attempting Trend-Mode analysis...")
                # Use market data that was already fetched for PPWCS
                candles_15m = market_data.get('candles', []) if market_data else []
                
                print(f"[TREND DEBUG] {symbol}: Got {len(candles_15m) if candles_15m else 0} candles for trend analysis")
                
                if candles_15m and len(candles_15m) >= 10:
                    from trend_mode import analyze_symbol_trend_mode
                    trend_analysis = analyze_symbol_trend_mode(
                        symbol=symbol,
                        candles=candles_15m,
                        enable_gpt=False  # GPT disabled for speed
                    )
                    
                    # Extract trend data
                    decision = trend_analysis.get('decision', 'avoid')
                    confidence = trend_analysis.get('confidence', 0.0)
                    entry_quality = trend_analysis.get('entry_quality', 0.0)
                    quality_grade = trend_analysis.get('quality_grade', 'poor')
                    market_context = trend_analysis.get('market_context', 'unknown')
                    trend_strength = trend_analysis.get('trend_strength', 0.0)
                    
                    # Add trend data to signals for PPWCS integration
                    signals.update({
                        'trend_mode_decision': decision,
                        'trend_mode_confidence': confidence,
                        'trend_mode_context': market_context,
                        'trend_mode_strength': trend_strength,
                        'trend_mode_quality': entry_quality,
                        'trend_mode_grade': quality_grade
                    })
                    
                    # Send Trend-Mode alert if high-quality setup
                    if decision == "join_trend" and entry_quality >= 0.75:
                        from utils.trend_alert_cache import should_send_trend_alert, add_trend_alert
                        
                        if should_send_trend_alert(symbol):
                            alert_message = (
                                f"🎯 TREND-MODE ALERT\n"
                                f"Symbol: {symbol}\n"
                                f"Decision: {decision.upper()}\n"
                                f"Quality: {quality_grade} ({entry_quality:.2f})\n"
                                f"Context: {market_context}\n"
                                f"Confidence: {confidence:.2f}\n"
                                f"Trend Strength: {trend_strength:.2f}"
                            )
                            
                            try:
                                from utils.telegram_bot import send_alert
                                send_alert(alert_message)
                                add_trend_alert(symbol, decision, entry_quality, quality_grade)
                                print(f"📈 [TREND ALERT] {symbol}: {quality_grade} entry opportunity")
                            except Exception as telegram_error:
                                print(f"⚠️ Trend alert send failed for {symbol}: {telegram_error}")
                    
                    # Compact logging for trend decisions (only for interesting decisions)
                    if decision == "join_trend":
                        print(f"✅ [TREND] {symbol}: JOIN ({quality_grade}, {entry_quality:.2f})")
                    elif decision == "wait":
                        print(f"⏳ [TREND] {symbol}: WAIT ({quality_grade}, {entry_quality:.2f})")
                else:
                    print(f"[TREND DEBUG] {symbol}: Insufficient candles for trend analysis ({len(candles_15m) if candles_15m else 0}/10 required)")
                
            except Exception as trend_error:
                # Log trend errors but don't fail pre-pump analysis
                print(f"[TREND ERROR] {symbol} - Trend analysis failed: {str(trend_error)}")
                
                # Add error fallback values to signals
                signals.update({
                    'trend_mode_decision': 'error',
                    'trend_mode_confidence': 0.0,
                    'trend_mode_context': 'error',
                    'trend_mode_strength': 0.0,
                    'trend_mode_quality': 0.0,
                    'trend_mode_grade': 'error'
                })
            
            return symbol, (stage2_pass, signals, inflow_usd, stage1g_active)
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
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
            # === STAGE -1 (TREND MODE) - NEW RHYTHM DETECTION ===
            stage_minus1_detected = False
            stage_minus1_description = "Brak detekcji"
            
            try:
                # Pobierz dane świec dla Stage -1 rhythm analysis
                success, market_data, price_usd, is_valid = get_market_data(symbol)
                if success and market_data and isinstance(market_data, dict) and 'candles' in market_data:
                    candles_data = market_data.get('candles')
                    if candles_data and isinstance(candles_data, list) and len(candles_data) >= 6:
                        stage_minus1_detected, stage_minus1_description = detect_stage_minus1(candles_data)
                        if stage_minus1_detected:
                            print(f"🎵 {symbol}: Stage -1 DETECTED - {stage_minus1_description}")
                            # Zapisz alert Stage -1
                            stage_minus1_alert = {
                                'symbol': symbol,
                                'market_tension': 'WYKRYTE',
                                'rhythm_description': stage_minus1_description,
                                'tension_level': 'WYSOKIE' if 'napięcie' in stage_minus1_description.lower() else 'STANDARD',
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            }
                            
                            # Zapisz do pliku alerts
                            os.makedirs("data", exist_ok=True)
                            alerts_file = "data/stage_minus1_alerts.json"
                            
                            if os.path.exists(alerts_file):
                                with open(alerts_file, 'r') as f:
                                    existing_alerts = json.load(f)
                            else:
                                existing_alerts = []
                            
                            existing_alerts.append(stage_minus1_alert)
                            
                            # Zachowaj tylko ostatnie 50 alertów
                            if len(existing_alerts) > 50:
                                existing_alerts = existing_alerts[-50:]
                            
                            with open(alerts_file, 'w') as f:
                                json.dump(existing_alerts, f, indent=2)
            except Exception as stage_minus1_error:
                print(f"⚠️ Stage -1 analysis failed for {symbol}: {stage_minus1_error}")
            
            # Update signals with Stage -1 results
            signals["stage_minus1_detected"] = stage_minus1_detected
            signals["stage_minus1_description"] = stage_minus1_description
            
            # Legacy compressed detection (keeping for compatibility)
            compressed = stage_minus1_detected
            
            previous_score = get_previous_score(symbol)
            final_score, ppwcs_structure, ppwcs_quality = compute_ppwcs(signals, previous_score)
            
            # 🐋 WHALE BOOST SCORING - Dodaj punkty za priorytet whale
            whale_boost = get_whale_boost_for_symbol(symbol, priority_info)
            if whale_boost > 0:
                final_score += whale_boost
                print(f"🔥 {symbol}: Whale boost +{whale_boost} points (total: {final_score})")
            
            # Integrate checklist scoring with scan cycle
            from utils.scoring import compute_checklist_score
            checklist_score, checklist_summary = compute_checklist_score(signals)
            
            # Add checklist data to signals for downstream processing
            signals["checklist_score"] = checklist_score
            signals["checklist_summary"] = checklist_summary
            signals["whale_boost"] = whale_boost
            signals["whale_priority"] = symbol in priority_symbols
            
            save_score(symbol, final_score)
            log_ppwcs_score(symbol, final_score, signals)


            scan_results.append({
                'symbol': symbol,
                'score': final_score,
                'checklist_score': checklist_score,
                'checklist_summary': checklist_summary,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stage2_pass': stage2_pass,
                'compressed': compressed,
                'stage1g_active': stage1g_active,

            })

            from utils.take_profit_engine import forecast_take_profit_levels
            tp_forecast = forecast_take_profit_levels(signals)

            # New Alert Level System - Updated for PPWCS 0-65 and Checklist 0-85 ranges
            from utils.scoring import get_alert_level
            
            alert_level = get_alert_level(final_score, checklist_score)
            allow_alert = False
            alert_tier = None
            is_high_confidence = False
            structure_ok = False
            
            # Enhanced conditions based on weighted checklist scoring (new system)
            if checklist_score >= 25 and final_score >= 45:  # Adjusted for weighted scoring (max 41)
                is_high_confidence = True
            
            if checklist_score >= 15:  # Quality threshold instead of count-based ≥3
                structure_ok = True
            
            # Map alert levels to tiers and determine if alert should be sent
            if alert_level == 3:  # Strong alert
                alert_tier = "🔴 Strong Alert" + (" [HIGH CONFIDENCE]" if is_high_confidence else "")
                alert_tier += " [STRUCTURE OK]" if structure_ok and not is_high_confidence else ""
                allow_alert = True
            elif alert_level == 2:  # Pre-pump active
                alert_tier = "🟠 Pre-pump Active" + (" [HIGH CONFIDENCE]" if is_high_confidence else "")
                alert_tier += " [STRUCTURE OK]" if structure_ok and not is_high_confidence else ""
                allow_alert = True
            elif alert_level == 1:  # Watchlist
                alert_tier = "🟡 Watchlist" + (" [STRUCTURE OK]" if structure_ok else "")
                allow_alert = False  # Watchlist items don't send alerts, only logged
            


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

                    signals["feedback_score"] = feedback_score
                    signals["feedback_category"] = category
                    os.makedirs("data/feedback", exist_ok=True)
                    feedback_file = f"data/feedback/{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.txt"
                    with open(feedback_file, "w", encoding="utf-8") as f:
                        f.write(f"Token: {symbol}\nPPWCS: {final_score} (Structure: {ppwcs_structure}, Quality: {ppwcs_quality})\n")
                        f.write(f"Checklist Score: {checklist_score}/41 ({len(checklist_summary)}/20 conditions)\n")

                        f.write(f"Alert Tier: {alert_tier}\nTimestamp: {datetime.now(timezone.utc).isoformat()}\n")
                        f.write(f"Signals: {signals}\nTP Forecast: {tp_forecast}\nFeedback Score: {feedback_score}/100\n")
                        f.write(f"Feedback Category: {category} ({description})\nGPT Feedback:\n{gpt_feedback}\n")
                except Exception as gpt_error:
                    print(f"⚠️ GPT feedback failed for {symbol}: {gpt_error}")

            if allow_alert and alert_tier:
                # Use comprehensive alert system with all features (checklist, TP forecast, GPT)
                from utils.alert_system import process_alert
                alert_success = process_alert(symbol, final_score, signals, gpt_feedback)




            # === ADVANCED TREND-MODE ANALYSIS & ALERTS ===
            try:
                from trend_mode import interpret_market_as_trader
                
                # Get UTC hour for session analysis
                utc_hour = get_utc_hour()
                candles_15m = market_data.get('candles', [])
                
                # Run Advanced Trend-Mode analysis
                trend_result = interpret_market_as_trader(
                    symbol, 
                    candles_15m, 
                    utc_hour, 
                    enable_gpt=False  # GPT disabled for performance - can be enabled for high-confidence
                )
                
                if trend_result and trend_result.get('analysis_complete', False):
                    decision = trend_result.get('decision', 'wait')
                    confidence = trend_result.get('confidence', 0.0)
                    market_context = trend_result.get('market_context', 'unknown')
                    trend_strength = trend_result.get('trend_strength', 0.0)
                    entry_quality = trend_result.get('entry_quality', 0.0)
                    quality_grade = trend_result.get('quality_grade', 'unknown')
                    reasons = trend_result.get('reasons', [])
                    
                    # Update signals z Trend-Mode wynikami
                    signals.update({
                        'trend_mode_decision': decision,
                        'trend_mode_confidence': confidence,
                        'trend_mode_context': market_context,
                        'trend_mode_strength': trend_strength,
                        'trend_mode_quality': entry_quality,
                        'trend_mode_grade': quality_grade
                    })
                    
                    # === TREND-MODE ALERT LOGIC ===
                    # Alert conditions: decision="join_trend" AND entry_quality >= 0.75
                    if (decision == "join_trend" and entry_quality >= 0.75 and 
                        not trend_alert_cache.already_alerted_recently(symbol, "trend_mode")):
                        
                        # Prepare alert message
                        alert_text = f'''📈 [Trend Mode Alert] {symbol}
✅ Trader AI Decision: JOIN TREND
🎯 Confidence: {confidence*100:.1f}%
📊 Quality: {quality_grade.upper()} ({entry_quality:.2f})
🧠 Reasons:
- {chr(10).join(f"  {reason}" for reason in reasons[:5])}

💡 Context: {market_context} | Trend Strength: {trend_strength:.2f}
⏰ Session: {utc_hour:02d}:XX UTC'''

                        # Check if GPT feedback is available
                        gpt_data = trend_result.get('gpt_analysis', {})
                        if gpt_data and gpt_data.get('gpt_decision'):
                            alert_text += f'''

🤖 GPT Analysis: {gpt_data["gpt_decision"]}
📝 {gpt_data.get("explanation", "No explanation")}'''

                        # Send Telegram alert
                        try:
                            from utils.telegram_bot import send_alert
                            alert_sent = send_alert(symbol, alert_text)
                            
                            if alert_sent:
                                trend_alert_cache.mark_alert_sent(symbol, "trend_mode")
                                print(f"📢 [TREND-MODE ALERT] Sent for {symbol} - Quality: {quality_grade}")
                                
                                # Log alert to file
                                alert_log_file = f"data/alerts/trend_mode_{symbol}_{get_utc_timestamp()[:10]}.txt"
                                os.makedirs("data/alerts", exist_ok=True)
                                with open(alert_log_file, "w", encoding="utf-8") as f:
                                    f.write(f"Timestamp: {get_utc_timestamp()}\n")
                                    f.write(f"Symbol: {symbol}\n")
                                    f.write(f"Decision: {decision}\n")
                                    f.write(f"Confidence: {confidence:.3f}\n")
                                    f.write(f"Entry Quality: {entry_quality:.3f}\n")
                                    f.write(f"Quality Grade: {quality_grade}\n")
                                    f.write(f"Market Context: {market_context}\n")
                                    f.write(f"Trend Strength: {trend_strength:.3f}\n")
                                    f.write(f"Reasons: {reasons}\n")
                                    f.write(f"Components: {trend_result.get('components', {})}\n")
                                    f.write(f"\nAlert Text:\n{alert_text}\n")
                            else:
                                print(f"❌ [TREND-MODE ALERT] Failed to send for {symbol}")
                                
                        except Exception as alert_error:
                            print(f"⚠️ Trend-Mode alert error for {symbol}: {alert_error}")
                    
                    # Enhanced logging for all decisions
                    if decision == "join_trend":
                        if entry_quality >= 0.75:
                            cooldown_status = trend_alert_cache.get_cooldown_status(symbol, "trend_mode")
                            if cooldown_status["on_cooldown"]:
                                print(f"🎯 [TREND-MODE] {symbol}: JOIN TREND - Quality: {quality_grade} (COOLDOWN: {cooldown_status['minutes_remaining']}min)")
                            else:
                                print(f"🎯 [TREND-MODE] {symbol}: JOIN TREND - Quality: {quality_grade} (ALERT SENT)")
                        else:
                            print(f"🎯 [TREND-MODE] {symbol}: JOIN TREND - Quality: {quality_grade} (below alert threshold)")
                        print(f"   Context: {market_context}, Confidence: {confidence:.2f}")
                        print(f"   Top reasons: {', '.join(reasons[:3])}")
                    elif decision == "wait":
                        print(f"⏳ [TREND-MODE] {symbol}: WAIT - Quality: {quality_grade} ({entry_quality:.2f})")
                    else:
                        print(f"❌ [TREND-MODE] {symbol}: AVOID - {quality_grade} setup ({market_context})")
                
            except Exception as trend_error:
                error_msg = f"Trend-Mode integration failed: {str(trend_error)}"
                print(f"[TREND ERROR] {symbol} - {error_msg}")
                
                # Log integration errors
                try:
                    with open("trend_error_log.txt", "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "timestamp": get_utc_timestamp(),
                            "symbol": symbol,
                            "function": "crypto_scan_service_trend_integration",
                            "error": str(trend_error),
                            "error_type": type(trend_error).__name__,
                            "candles_available": len(candles_15m) if candles_15m else 0
                        }) + "\n")
                except:
                    pass
                
                # Fallback values
                signals.update({
                    'trend_mode_decision': 'error',
                    'trend_mode_confidence': 0.0,
                    'trend_mode_context': 'error',
                    'trend_mode_strength': 0.0,
                    'trend_mode_quality': 0.0,
                    'trend_mode_grade': 'error'
                })

            # Save complete stage signal data 
            save_stage_signal(symbol, final_score, stage2_pass, compressed, stage1g_active, 
                            checklist_score, checklist_summary)

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
