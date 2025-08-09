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
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Load environment variables at the start
load_dotenv()
# from utils.contracts import get_contract  # Module not available
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
from utils.data_fetchers import get_market_data
from utils.bybit_cache_manager import get_bybit_symbols_cached
# === CONFIGURATION ===

# Sprawdzanie i odświeżanie cache symboli Bybit
print("🔍 Sprawdzanie cache symboli Bybit...")

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
            model="gpt-5",
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

    symbols = get_bybit_symbols_cached()
    symbols_bybit = symbols  # Używaj symboli z Bybit cache manager
    
    # 🐋 WHALE PRIORITY SYSTEM - Priorytetowanie na podstawie wcześniejszej aktywności whale
    from utils.whale_priority import prioritize_whale_tokens, save_priority_report, get_whale_boost_for_symbol
    
    symbols, priority_symbols, priority_info = prioritize_whale_tokens(symbols)
    
    # Zapisz raport priorytetowania
    if priority_info:
        save_priority_report(priority_info)
        
    # Dynamiczne zarządzanie wątkami - min(CPU cores, liczba symboli, max 16)
    max_workers = min(multiprocessing.cpu_count(), len(symbols), 16)
    print(f"🚀 [PARALLEL SCAN] Using {max_workers} workers for {len(symbols)} symbols")
    
    scan_results = []

    def scan_single_token(symbol):
        """Scan pojedynczego tokena - kompletna analiza w jednym wątku"""
        try:
            print(f"🔍 Skanuję {symbol}...")
            
            # Podstawowa walidacja symbolu
            if symbol not in symbols_bybit:
                print(f"⚠️ Pomijam {symbol} – nie znajduje się na Bybit (USDT perp)")
                return None

            # Pobieranie danych rynkowych z timeoutami
            success, data, price_usd, is_valid = get_market_data(symbol)

            if not success or not data or not isinstance(data, dict) or price_usd is None:
                print(f"⚠️ Pomiń {symbol} – niepoprawne dane z get_market_data(): {data}")
                return None

            # Filtry płynności i jakości
            volume_usdt = data.get("volume")
            best_ask = data.get("best_ask")
            best_bid = data.get("best_bid")

            if not price_usd or price_usd <= 0:
                print(f"⚠️ Pominięto {symbol} – nieprawidłowa cena: {price_usd}")
                return None

            if not volume_usdt or volume_usdt < 100_000:
                print(f"⚠️ Pominięto {symbol} – zbyt niski wolumen: ${volume_usdt}")
                return None

            if best_ask and best_bid:
                spread = (best_ask - best_bid) / best_ask
                if spread > 0.02:
                    print(f"⚠️ Pominięto {symbol} – zbyt szeroki spread: {spread*100:.2f}%")
                    return None

            # Run pre-pump analysis
            stage2_pass, signals, inflow_usd, stage1g_active = detect_stage_minus2_1(symbol, price_usd=price_usd)
            
            # === PARALLEL TREND-MODE ANALYSIS ===
            # Run trend-mode analysis simultaneously with pre-pump PPWCS
            try:
                print(f"[TREND DEBUG] {symbol}: Attempting Trend-Mode analysis...")
                
                # Safe candle fetching with validation
                from utils.safe_candles import safe_trend_analysis_check
                candles_15m, is_ready = safe_trend_analysis_check(symbol, market_data)
                
                if is_ready and candles_15m:
                    from trend_mode import analyze_trend_opportunity
                    trend_analysis = analyze_trend_opportunity(
                        symbol=symbol,
                        candles=candles_15m
                    )
                    
                    # Extract TJDE trend data
                    decision = trend_analysis.get('decision', 'avoid')
                    confidence = trend_analysis.get('confidence', 0.0)
                    final_score = trend_analysis.get('final_score', 0.0)
                    quality_grade = trend_analysis.get('grade', 'poor')
                    market_context = trend_analysis.get('market_context', {}).get('session', 'unknown')
                    trend_strength = trend_analysis.get('trend_strength', 0.0)
                    
                    # Add TJDE data to signals for PPWCS integration
                    signals.update({
                        'tjde_decision': decision,
                        'tjde_confidence': confidence,
                        'tjde_final_score': final_score,
                        'tjde_context': market_context,
                        'tjde_strength': trend_strength,
                        'tjde_grade': quality_grade
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
                    # Symbol skipped - insufficient or invalid candle data
                    print(f"[TREND DEBUG] {symbol}: Trend-Mode analysis skipped - data quality check failed")
                
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
            
            # === STAGE -1 (TREND MODE) - NEW RHYTHM DETECTION ===
            stage_minus1_detected = False
            stage_minus1_description = "Brak detekcji"
            
            try:
                from stages.stage_minus1 import detect_stage_minus1
                candles_data = data.get('candles')
                if candles_data and isinstance(candles_data, list) and len(candles_data) >= 6:
                    stage_minus1_detected, stage_minus1_description = detect_stage_minus1(candles_data)
                    if stage_minus1_detected:
                        print(f"🎵 {symbol}: Stage -1 DETECTED - {stage_minus1_description}")
            except Exception as stage_minus1_error:
                print(f"⚠️ Stage -1 analysis failed for {symbol}: {stage_minus1_error}")
            
            # Update signals with Stage -1 results
            signals["stage_minus1_detected"] = stage_minus1_detected
            signals["stage_minus1_description"] = stage_minus1_description
            
            # Legacy compressed detection (keeping for compatibility)
            compressed = stage_minus1_detected
            
            # PPWCS Scoring
            previous_score = get_previous_score(symbol)
            try:
                ppwcs_result = compute_ppwcs(signals, symbol)
                if isinstance(ppwcs_result, tuple) and len(ppwcs_result) == 3:
                    final_score, ppwcs_structure, ppwcs_quality = ppwcs_result
                else:
                    final_score = float(ppwcs_result) if ppwcs_result else 0.0
                    ppwcs_structure = {}
                    ppwcs_quality = "basic"
            except Exception as ppwcs_error:
                print(f"❌ PPWCS error for {symbol}: {ppwcs_error}")
                final_score = 0.0
                ppwcs_structure = {}
                ppwcs_quality = "error"
            
            # Whale boost scoring
            whale_boost = get_whale_boost_for_symbol(symbol, priority_info)
            if whale_boost > 0:
                final_score += whale_boost
                print(f"🔥 {symbol}: Whale boost +{whale_boost} points (total: {final_score})")
            
            # Checklist scoring
            try:
                from utils.scoring import compute_checklist_score
                checklist_result = compute_checklist_score(signals)
                if isinstance(checklist_result, tuple) and len(checklist_result) == 2:
                    checklist_score, checklist_summary = checklist_result
                else:
                    checklist_score = float(checklist_result) if checklist_result else 0.0
                    checklist_summary = {}
            except Exception as checklist_error:
                print(f"❌ Checklist error for {symbol}: {checklist_error}")
                checklist_score = 0.0
                checklist_summary = {}
            
            # Add checklist data to signals
            signals["checklist_score"] = checklist_score
            signals["checklist_summary"] = checklist_summary
            signals["whale_boost"] = whale_boost
            signals["whale_priority"] = symbol in priority_symbols
            
            # Save results
            save_score(symbol, final_score, signals)
            log_ppwcs_score(symbol, final_score, signals)

            # Alert processing
            from utils.take_profit_engine import forecast_take_profit_levels
            tp_forecast = forecast_take_profit_levels(signals)

            from utils.scoring import get_alert_level
            alert_level = get_alert_level(final_score, checklist_score)
            
            # Process alerts if needed
            if alert_level >= 2:  # Level 2 or 3 alerts
                try:
                    from utils.alert_system import process_alert
                    alert_success = process_alert(symbol, final_score, signals, None)
                    if alert_success:
                        print(f"📢 [ALERT] {symbol}: Level {alert_level} alert sent")
                except Exception as alert_error:
                    print(f"⚠️ Alert processing failed for {symbol}: {alert_error}")

            return {
                'symbol': symbol,
                'score': final_score,
                'checklist_score': checklist_score,
                'checklist_summary': checklist_summary,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stage2_pass': stage2_pass,
                'compressed': compressed,
                'stage1g_active': stage1g_active,
                'alert_level': alert_level,
                'signals': signals
            }
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            return None

    # Równoległe skanowanie wszystkich symboli
    scan_start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"⚡ [PARALLEL] Submitting {len(symbols)} tokens for concurrent analysis...")
        future_to_symbol = {executor.submit(scan_single_token, symbol): symbol for symbol in symbols}
        
        completed_count = 0
        for future in as_completed(future_to_symbol):
            completed_count += 1
            symbol = future_to_symbol[future]
            
            try:
                result = future.result(timeout=30)  # 30s timeout per token
                if result:
                    scan_results.append(result)
                    print(f"✅ [{completed_count}/{len(symbols)}] {symbol}: Score {result['score']:.1f}")
                else:
                    print(f"⚠️ [{completed_count}/{len(symbols)}] {symbol}: Skipped")
            except Exception as e:
                print(f"❌ [{completed_count}/{len(symbols)}] {symbol}: Error - {e}")
    
    scan_duration = time.time() - scan_start_time
    print(f"🏁 [PARALLEL SCAN] Completed in {scan_duration:.1f}s, processed {len(scan_results)} tokens")

    # Save stage signal data and handle special alerts
    for result in scan_results:
        try:
            symbol = result['symbol']
            signals = result['signals']
            
            # Save stage signal data
            save_stage_signal(symbol, result['score'], result.get('stage2_pass', False), 
                            result.get('compressed', False), result.get('stage1g_active', False), 
                            result.get('checklist_score', 0), result.get('checklist_summary', {}))
        except Exception as e:
            print(f"❌ Error saving stage signal for {result.get('symbol', 'unknown')}: {e}")

    # Post-scan processing and reports
    print("📊 Conditional reports generated")
    save_conditional_reports()
    compress_reports_to_zip()
    
    print(f"✅ Scan completed. Processed {len(scan_results)} symbols.")
    
    # Advanced feedback system integration  
    try:
        from feedback.feedback_loop_v2 import run_feedback_analysis_v2
        feedback_results = run_feedback_analysis_v2()
        if feedback_results.get("adjustments_made", 0) > 0:
            print(f"📊 Feedback loop adjusted {feedback_results['adjustments_made']} weights")
    except Exception as feedback_error:
        print(f"📊 No feedback analysis data available for this scan")
    except Exception as e:
        print(f"📊 No feedback analysis data available for this scan")
    
    return scan_results
                                
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
    
    # Send enhanced TJDE alerts to Telegram after scan completion
    try:
        from utils.alerts import send_tjde_telegram_summary
        from utils.feedback_integration import run_periodic_feedback_analysis
        from trend_mode import analyze_trend_opportunity
        from utils.training_data_manager import training_manager
        
        # Collect TJDE results from this scan and gather training data
        tjde_results = []
        training_samples_collected = 0
        
        for result in scan_results:
            symbol = result.get('symbol')
            if symbol:
                try:
                    trend_result = analyze_trend_opportunity(symbol)
                    if trend_result and trend_result.get("final_score", 0) > 0:
                        trend_result["symbol"] = symbol
                        tjde_results.append(trend_result)
                        
                        # Collect training data for Vision-AI (selective collection)
                        if (trend_result.get("final_score", 0) > 0.6 or 
                            trend_result.get("decision") in ["join_trend", "consider_entry"]):
                            
                            # Get candles for this symbol
                            from utils.api_utils import get_bybit_candles
                            candles = get_bybit_candles(symbol, "15", 100)
                            
                            if candles and len(candles) >= 50:
                                # Collect training sample
                                sample_id = training_manager.collect_sample_during_scan(
                                    symbol=symbol,
                                    candles=candles,
                                    scoring_context=result,
                                    trend_decision=trend_result
                                )
                                
                                if sample_id:
                                    training_samples_collected += 1
                                    print(f"[TRAINING] Collected sample: {sample_id}")
                                
                                # Generate CV prediction for high-quality signals
                                try:
                                    from utils.chart_exporter import save_candlestick_chart
                                    from vision_ai.predict_cv_setup import predict_setup_from_chart
                                    
                                    # Export chart for CV analysis
                                    chart_path = save_candlestick_chart(symbol, candles, "temp_cv_charts")
                                    
                                    if chart_path:
                                        # Generate CV prediction
                                        cv_prediction = predict_setup_from_chart(chart_path, symbol)
                                        
                                        if cv_prediction and "error" not in cv_prediction:
                                            print(f"[CV PREDICTION] {symbol}: {cv_prediction.get('setup', 'unknown')} ({cv_prediction.get('confidence', 0):.2f})")
                                        
                                        # Clean up temp chart
                                        import os
                                        try:
                                            os.remove(chart_path)
                                        except:
                                            pass
                                            
                                except Exception as cv_error:
                                    print(f"[CV PREDICTION] Failed for {symbol}: {cv_error}")
                        
                except Exception as e:
                    print(f"[TJDE] Error analyzing {symbol}: {e}")
                    continue
        
        # Run periodic feedback analysis first
        feedback_data = run_periodic_feedback_analysis()
        
        # Display feedback loop score changes table if available
        if feedback_data and feedback_data.get("success"):
            try:
                from feedback.feedback_loop_v2 import print_adjustment_summary
                weights_before = feedback_data.get("weights_before", {})
                weights_after = feedback_data.get("weights_after", {})
                adjustments = feedback_data.get("adjustments", {})
                performance = feedback_data.get("performance", {})
                
                if weights_before and weights_after:
                    print_adjustment_summary(weights_before, weights_after, adjustments, {
                        "success_rate": performance.get("success_rate", 0),
                        "analysis_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    })
                else:
                    print("📊 Feedback analysis completed but no weight changes made")
            except Exception as table_error:
                print(f"[FEEDBACK TABLE] Error displaying changes table: {table_error}")
        else:
            print("📊 No feedback analysis data available for this scan")
        
        if tjde_results:
            # Filter for meaningful decisions and sort by score
            valid_results = [
                r for r in tjde_results 
                if r.get("decision") in ["join_trend", "consider_entry"] 
                and r.get("final_score", 0) > 0.3
            ]
            
            if valid_results:
                # Sort by final_score and get top 5
                top5_results = sorted(valid_results, key=lambda x: x.get("final_score", 0), reverse=True)[:5]
                
                print(f"[TJDE ALERTS] Sending TOP {len(top5_results)} enhanced alerts")
                send_tjde_telegram_summary(top5_results, feedback_data)
            else:
                print("[TJDE ALERTS] No qualifying tokens found for alerts")
                # Send feedback summary only if weight changes occurred
                if feedback_data and feedback_data.get("success"):
                    from utils.alerts import send_feedback_summary
                    send_feedback_summary(feedback_data)
        else:
            print("[TJDE ALERTS] No TJDE results found")
        
        # Log training data collection summary
        if training_samples_collected > 0:
            print(f"[TRAINING] Collected {training_samples_collected} training samples this scan")
            
            # Get training stats
            stats = training_manager.get_training_stats()
            print(f"[TRAINING] Total dataset: {stats.get('total_samples', 0)} samples")
            
            # Run CV feedback analysis for recent predictions
            try:
                from vision_ai.feedback_loop_cv import CVFeedbackLoop
                
                feedback_loop = CVFeedbackLoop()
                analysis = feedback_loop.analyze_pending_predictions(hours_back=12)
                
                if "error" not in analysis:
                    summary = analysis["summary"]
                    if summary["predictions_analyzed"] > 0:
                        print(f"[CV FEEDBACK] Analyzed {summary['predictions_analyzed']} predictions, {summary['success_rate']:.1%} success rate")
                        
                        # Update model weights if we have enough data
                        if len(analysis["results"]) >= 3:
                            weight_updates = feedback_loop.update_cv_model_weights(analysis["results"])
                            if "error" not in weight_updates:
                                print(f"[CV FEEDBACK] Model performance analysis completed")
                                
            except Exception as feedback_error:
                print(f"[CV FEEDBACK] Feedback analysis failed: {feedback_error}")
        
        # Save TJDE results for auto-labeling system
        if tjde_results:
            try:
                import json
                from pathlib import Path
                
                # Save current TJDE results
                results_dir = Path("data/tjde_results")
                results_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                results_file = results_dir / f"tjde_results_{timestamp}.json"
                
                with open(results_file, 'w') as f:
                    json.dump({
                        "timestamp": timestamp,
                        "scan_completed": datetime.now(timezone.utc).isoformat(),
                        "total_results": len(tjde_results),
                        "results": tjde_results
                    }, f, indent=2)
                
                print(f"[TJDE RESULTS] Saved {len(tjde_results)} results to {results_file.name}")
                
                # Run auto-labeling for top tokens
                try:
                    from vision_ai.auto_label_runner import run_auto_labeling_for_tjde
                    
                    # Process top 5 tokens for training data
                    auto_label_summary = run_auto_labeling_for_tjde(top_count=5)
                    
                    if "error" not in auto_label_summary:
                        pairs_created = auto_label_summary["processing_stats"]["training_pairs_created"]
                        if pairs_created > 0:
                            print(f"[AUTO LABEL] Created {pairs_created} training pairs from top tokens")
                    
                except Exception as auto_label_error:
                    print(f"[AUTO LABEL] Auto-labeling failed: {auto_label_error}")
                
            except Exception as save_error:
                print(f"[TJDE RESULTS] Failed to save results: {save_error}")
            
    except Exception as e:
        print(f"❌ [TJDE ALERTS] Error in enhanced alert system: {e}")
    
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
