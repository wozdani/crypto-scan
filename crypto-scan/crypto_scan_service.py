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
        if not candle_data or len(candle_data) < 2:
            return False
            
        latest_candle = candle_data[-1]
        
        # Analiza siły ostatniej świecy
        open_price = float(latest_candle.get('open', 0))
        close_price = float(latest_candle.get('close', 0))
        high_price = float(latest_candle.get('high', 0))
        low_price = float(latest_candle.get('low', 0))
        volume = float(latest_candle.get('volume', 0))
        
        if not all([open_price, close_price, high_price, low_price]):
            return False
            
        # Body ratio (stosunek korpusu do całego zakresu)
        total_range = high_price - low_price
        body_size = abs(close_price - open_price)
        body_ratio = body_size / total_range if total_range > 0 else 0
        
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

Advanced Detectors:
• Pure Accumulation Pattern: {pure_acc}
• Heatmap Exhaustion: {heatmap_exhaustion}
• Orderbook Spoofing: {spoofing}
• VWAP Pinning: {vwap_pinned}
• Volume Slope Rising: {vol_slope}

Market Conditions:
• Stage 1G Active: {stage1g}
• Price Compression: {compressed}

TP Forecast: {tp_forecast}

Provide brief analysis (max 100 words) with risk level (low/medium/high), confidence (1-100), and entry recommendation (immediate/wait/avoid)."""

    try:
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
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
    
    # 🔍 SYMBOL VALIDATION - Filter out inactive/delisted symbols
    try:
        from utils.symbol_validator import validate_symbols_before_scan, get_validator_stats
        
        print(f"🔍 [VALIDATOR] Validating {len(symbols)} symbols for health...")
        healthy_symbols = validate_symbols_before_scan(symbols)
        
        # Log validation results
        validator_stats = get_validator_stats()
        print(f"✅ [VALIDATOR] {len(healthy_symbols)}/{len(symbols)} symbols healthy ({validator_stats.get('healthy_rate', 0):.1f}%)")
        
        symbols = healthy_symbols
        symbols_bybit = healthy_symbols
        
    except Exception as validator_error:
        print(f"⚠️ [VALIDATOR] Symbol validation failed, using all symbols: {validator_error}")
    
    # 🐋 WHALE PRIORITY SYSTEM - Priorytetowanie na podstawie wcześniejszej aktywności whale
    from utils.whale_priority import prioritize_whale_tokens, save_priority_report, get_whale_boost_for_symbol
    
    symbols, priority_symbols, priority_info = prioritize_whale_tokens(symbols)
    
    # Zapisz raport priorytetowania
    if priority_info:
        save_priority_report(priority_info)
        
    # Handle empty symbols list
    if not symbols:
        print("⚠️ [SCAN] No symbols available - using essential fallback")
        essential_symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT",
            "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT", "LTCUSDT", "UNIUSDT",
            "NEARUSDT", "AAVEUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "SUIUSDT"
        ]
        symbols = essential_symbols
        symbols_bybit = essential_symbols
        priority_symbols = []
        priority_info = {}
    
    # Dynamiczne zarządzanie wątkami - min(CPU cores, liczba symboli, max 16)
    max_workers = min(multiprocessing.cpu_count(), len(symbols), 16)
    max_workers = max(1, max_workers)  # Ensure at least 1 worker
    print(f"🚀 [PARALLEL SCAN] Using {max_workers} workers for {len(symbols)} symbols")
    
    scan_results = []

    def scan_single_token(symbol):
        """Optimized single token scan with parallel sub-analysis"""
        token_start = time.time()
        try:
            # Podstawowa walidacja symbolu
            if symbol not in symbols_bybit:
                return None

            # Pobieranie danych rynkowych z timeoutami (najwolniejsza operacja)
            api_start = time.time()
            success, data, price_usd, is_valid = get_market_data(symbol)
            api_time = time.time() - api_start

            if not success or not data or not isinstance(data, dict) or price_usd is None:
                return None

            # Quick validation filters
            volume_usdt = data.get("volume", 0)
            if not price_usd or price_usd <= 0 or volume_usdt < 100_000:
                return None

            best_ask = data.get("best_ask")
            best_bid = data.get("best_bid")
            if best_ask and best_bid:
                spread = (best_ask - best_bid) / best_ask
                if spread > 0.02:
                    return None

            # Parallel analysis execution using ThreadPoolExecutor for sub-tasks
            analysis_start = time.time()
            
            with ThreadPoolExecutor(max_workers=3) as sub_executor:
                # Submit parallel analysis tasks
                stage2_future = sub_executor.submit(detect_stage_minus2_1, symbol, price_usd=price_usd)
                
                # TJDE trend analysis (if candles available)
                trend_future = None
                try:
                    from utils.safe_candles import safe_trend_analysis_check
                    candles_15m, is_ready = safe_trend_analysis_check(symbol, data)
                    if is_ready and candles_15m:
                        from trend_mode import analyze_trend_opportunity
                        trend_future = sub_executor.submit(analyze_trend_opportunity, symbol, candles_15m)
                except:
                    pass
                
                # Stage -1 analysis
                stage1_future = None
                candles_data = data.get('candles')
                if candles_data and isinstance(candles_data, list) and len(candles_data) >= 6:
                    from stages.stage_minus1 import detect_stage_minus1
                    stage1_future = sub_executor.submit(detect_stage_minus1, candles_data)
                
                # Collect results with timeout
                try:
                    stage2_pass, signals, inflow_usd, stage1g_active = stage2_future.result(timeout=10)
                except:
                    stage2_pass, signals, inflow_usd, stage1g_active = False, {}, 0, False
                
                # Collect trend analysis
                trend_analysis = None
                if trend_future:
                    try:
                        trend_analysis = trend_future.result(timeout=8)
                    except:
                        pass
                
                # Collect stage -1 results  
                stage_minus1_detected, stage_minus1_description = False, "Brak detekcji"
                if stage1_future:
                    try:
                        stage_minus1_detected, stage_minus1_description = stage1_future.result(timeout=5)
                    except:
                        pass
            
            analysis_time = time.time() - analysis_start
            
            # Process trend analysis results (if available)
            if trend_analysis:
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
                
                # Handle alerts for high-quality setups
                entry_quality = trend_analysis.get('entry_quality', 0.0)
                if decision == "join_trend" and entry_quality >= 0.75:
                    try:
                        from utils.trend_alert_cache import should_send_trend_alert, add_trend_alert
                        if should_send_trend_alert(symbol):
                            alert_message = f"TREND-MODE: {symbol} {decision.upper()} - {quality_grade} ({entry_quality:.2f})"
                            from utils.telegram_bot import send_alert
                            send_alert(alert_message)
                            add_trend_alert(symbol, decision, entry_quality, quality_grade)
                    except:
                        pass
            
            # Update signals with Stage -1 results  
            signals["stage_minus1_detected"] = stage_minus1_detected
            signals["stage_minus1_description"] = stage_minus1_description
            compressed = stage_minus1_detected
            
            # Fast scoring pipeline with parallel execution
            scoring_start = time.time()
            
            with ThreadPoolExecutor(max_workers=4) as scoring_executor:
                # Submit parallel scoring tasks
                ppwcs_future = scoring_executor.submit(compute_ppwcs, signals, symbol)
                checklist_future = scoring_executor.submit(lambda: compute_checklist_score(signals))
                whale_future = scoring_executor.submit(get_whale_boost_for_symbol, symbol, priority_info)
                
                # Collect scoring results with timeouts
                try:
                    ppwcs_result = ppwcs_future.result(timeout=5)
                    if isinstance(ppwcs_result, tuple) and len(ppwcs_result) == 3:
                        final_score, ppwcs_structure, ppwcs_quality = ppwcs_result
                    else:
                        final_score = float(ppwcs_result) if ppwcs_result else 0.0
                        ppwcs_structure, ppwcs_quality = {}, "basic"
                except:
                    final_score, ppwcs_structure, ppwcs_quality = 0.0, {}, "error"
                
                try:
                    checklist_result = checklist_future.result(timeout=3)
                    if isinstance(checklist_result, tuple) and len(checklist_result) == 2:
                        checklist_score, checklist_summary = checklist_result
                    else:
                        checklist_score, checklist_summary = float(checklist_result or 0), {}
                except:
                    checklist_score, checklist_summary = 0.0, {}
                
                try:
                    whale_boost = whale_future.result(timeout=2)
                    if whale_boost > 0:
                        final_score += whale_boost
                except:
                    whale_boost = 0
            
            scoring_time = time.time() - scoring_start
            
            # Add metadata to signals
            signals.update({
                "checklist_score": checklist_score,
                "checklist_summary": checklist_summary,
                "whale_boost": whale_boost,
                "whale_priority": symbol in priority_symbols
            })
            
            # Quick save and alert processing
            save_score(symbol, final_score, signals)
            log_ppwcs_score(symbol, final_score, signals)

            from utils.scoring import get_alert_level
            alert_level = get_alert_level(final_score, checklist_score)
            
            # Fast alert processing (no blocking operations)
            if alert_level >= 2:
                try:
                    from utils.alert_system import process_alert
                    process_alert(symbol, final_score, signals, None)
                except:
                    pass

            token_time = time.time() - token_start
            
            # Performance logging for optimization
            if token_time > 2.0:  # Log slow tokens
                print(f"⏱️ [SLOW] {symbol}: {token_time:.1f}s (API: {api_time:.1f}s, Analysis: {analysis_time:.1f}s, Scoring: {scoring_time:.1f}s)")

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
                'signals': signals,
                'processing_time': token_time
            }
            
        except Exception as e:
            token_time = time.time() - token_start
            print(f"❌ Error processing {symbol} in {token_time:.1f}s: {e}")
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