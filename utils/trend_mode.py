"""
Trend Mode v1.0 - Professional Trend Continuation Analysis
Based on techniques from Minervini, Raschke, Grimes, SMB Capital

Detects trend continuation opportunities rather than breakouts.
Separate from pre-pump detection with independent scoring (max 50 points).
"""

import numpy as np
from datetime import datetime, timezone, timedelta
import os
import json


def calculate_atr(data, period=14):
    """Calculate Average True Range for volatility measurement"""
    if len(data) < period:
        return 0
    
    high_low = np.array([candle[2] - candle[3] for candle in data[-period:]])  # high - low
    high_close = np.array([abs(candle[2] - data[i-1][4]) for i, candle in enumerate(data[-period:]) if i > 0])  # high - prev_close
    low_close = np.array([abs(candle[3] - data[i-1][4]) for i, candle in enumerate(data[-period:]) if i > 0])  # low - prev_close
    
    true_ranges = np.maximum(high_low[1:], np.maximum(high_close, low_close))
    return np.mean(true_ranges) if len(true_ranges) > 0 else 0


def calculate_vwap(data):
    """Calculate Volume Weighted Average Price"""
    if len(data) < 20:
        return 0
    
    volumes = np.array([candle[5] for candle in data[-20:]])
    typical_prices = np.array([(candle[2] + candle[3] + candle[4]) / 3 for candle in data[-20:]])
    
    total_volume = np.sum(volumes)
    if total_volume == 0:
        return 0
    
    return np.sum(typical_prices * volumes) / total_volume


def check_trend_activation(data):
    """
    Check mandatory trend activation conditions:
    1. RSI > 68
    2. Price > VWAP for last 3 candles (15M)
    3. Volume rising for 3 consecutive candles
    
    Returns: (trend_active, activation_details)
    """
    if len(data) < 20:
        return False, "Insufficient data for trend analysis"
    
    # Calculate RSI for last candle
    rsi = calculate_rsi(data)
    if rsi <= 68:
        return False, f"RSI too low: {rsi:.1f} (need >68)"
    
    # Check price > VWAP for last 3 candles
    vwap = calculate_vwap(data)
    last_3_closes = [candle[4] for candle in data[-3:]]  # close prices
    
    above_vwap_count = sum(1 for close in last_3_closes if close > vwap)
    if above_vwap_count < 3:
        return False, f"Price above VWAP: {above_vwap_count}/3 candles (need 3/3)"
    
    # Check volume rising for 3 consecutive candles
    last_3_volumes = [candle[5] for candle in data[-3:]]
    volume_rising = all(last_3_volumes[i] > last_3_volumes[i-1] for i in range(1, 3))
    
    if not volume_rising:
        return False, "Volume not rising consistently for 3 candles"
    
    return True, "All trend activation conditions met"


def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    if len(data) < period + 1:
        return 50
    
    closes = np.array([candle[4] for candle in data[-(period+1):]])
    deltas = np.diff(closes)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def detect_orderbook_pressure(symbol):
    """
    Detect orderbook bid pressure indicating buying interest
    Returns: (bid_pressure, pressure_ratio)
    """
    try:
        # Placeholder for orderbook analysis
        # In production, this would analyze real orderbook data
        return False, 1.0
    except Exception:
        return False, 1.0


def detect_social_burst(symbol):
    """
    Detect social media burst during trend (different from pre-pump social)
    This indicates momentum confirmation, not early detection
    """
    try:
        # Placeholder for social analysis
        # In production, would check Twitter/Reddit activity spikes
        return False
    except Exception:
        return False


def compute_ppwcs_t_trend_boost(data, symbol=None):
    """
    PPWCS-T 2.0 - Trend mode wzmacniacz przy stabilnym wzroście bez breakoutów
    
    Args:
        data: OHLCV candle data (list of [timestamp, open, high, low, close, volume])
        symbol: token symbol for additional analysis
    
    Returns:
        dict: {
            "trend_boost": int (0-20),
            "trend_mode_active": bool,
            "boost_details": list of reasons for boost
        }
    """
    if len(data) < 20:
        return {"trend_boost": 0, "trend_mode_active": False, "boost_details": []}
    
    trend_boost = 0
    boost_details = []
    
    # Calculate required metrics
    rsi_last = calculate_rsi(data)
    vwap = calculate_vwap(data)
    
    last_candle = data[-1]
    last_3_candles = data[-3:]
    last_5_candles = data[-5:]
    
    # 1. RSI w zakresie "trendowej akumulacji" (50-60)
    if 50 < rsi_last < 60:
        trend_boost += 5
        boost_details.append(f"RSI trendowa akumulacja ({rsi_last:.1f})")
        print(f"[PPWCS-T 2.0] ✅ RSI trendowa akumulacja: +5 (RSI={rsi_last:.1f})")
    
    # 2. VWAP pinning - cena nie odkleja się od VWAP
    vwap_pinning = detect_vwap_pinning(data, vwap)
    if vwap_pinning:
        trend_boost += 5
        boost_details.append("VWAP pinning")
        print(f"[PPWCS-T 2.0] ✅ VWAP pinning: +5")
    
    # 3. Volume slope up - narastająca aktywność bez pumpy
    volume_slope_up = detect_volume_slope_up(data)
    if volume_slope_up:
        trend_boost += 5
        boost_details.append("Volume slope up")
        print(f"[PPWCS-T 2.0] ✅ Volume slope up: +5")
    
    # 4. Liquidity box + higher lows - stabilna konsolidacja w strukturze wzrostowej
    liquidity_box = detect_liquidity_box_trend(data)
    higher_lows = detect_higher_lows(data)
    
    if liquidity_box and higher_lows:
        trend_boost += 5
        boost_details.append("Stabilna konsolidacja wzrostowa")
        print(f"[PPWCS-T 2.0] ✅ Liquidity box + higher lows: +5")
    
    # 5. Sprawdź czy nie ma breakoutu (wykluczenie dużych ruchów)
    breakout_detected = detect_breakout_exclusion(data)
    
    # Aktywacja trend mode jeśli boost >= 10 i brak breakoutu
    if trend_boost >= 10 and not breakout_detected:
        trend_mode_active = True
        print(f"[PPWCS-T 2.0] ✅ Trend mode aktywny: {trend_boost} punktów (bez breakoutu)")
    else:
        trend_mode_active = False
        if breakout_detected:
            print(f"[PPWCS-T 2.0] ❌ Trend mode nieaktywny: wykryto breakout")
        else:
            print(f"[PPWCS-T 2.0] ❌ Trend mode nieaktywny: boost {trend_boost} < 10")
    
    return {
        "trend_boost": trend_boost,
        "trend_mode_active": trend_mode_active,
        "boost_details": boost_details
    }


def detect_vwap_pinning(data, vwap):
    """Detect if price stays close to VWAP (not detaching)"""
    if len(data) < 5 or vwap == 0:
        return False
    
    last_5_closes = [candle[4] for candle in data[-5:]]
    
    # Check if all closes are within 2% of VWAP
    vwap_distances = [abs(close - vwap) / vwap for close in last_5_closes]
    
    # All prices within 2% of VWAP = pinning
    return all(distance < 0.02 for distance in vwap_distances)


def detect_volume_slope_up(data):
    """Detect gradually increasing volume without pump spikes"""
    if len(data) < 10:
        return False
    
    last_10_volumes = [candle[5] for candle in data[-10:]]
    
    # Calculate linear regression slope for volume
    x = np.arange(len(last_10_volumes))
    slope = np.polyfit(x, last_10_volumes, 1)[0]
    
    # Check if slope is positive but not too steep (no pump)
    avg_volume = np.mean(last_10_volumes)
    max_volume = max(last_10_volumes)
    
    # Volume increasing but no single candle > 3x average
    return slope > 0 and max_volume < (avg_volume * 3)


def detect_liquidity_box_trend(data):
    """Detect sideways consolidation with tight range"""
    if len(data) < 10:
        return False
    
    last_10_candles = data[-10:]
    highs = [candle[2] for candle in last_10_candles]
    lows = [candle[3] for candle in last_10_candles]
    
    price_range = max(highs) - min(lows)
    avg_price = (max(highs) + min(lows)) / 2
    
    # Range < 3% of average price = liquidity box
    return (price_range / avg_price) < 0.03


def detect_higher_lows(data):
    """Detect pattern of higher lows over last 15 candles"""
    if len(data) < 15:
        return False
    
    last_15_lows = [candle[3] for candle in data[-15:]]
    
    # Find local minima (lows)
    local_mins = []
    for i in range(1, len(last_15_lows) - 1):
        if last_15_lows[i] < last_15_lows[i-1] and last_15_lows[i] < last_15_lows[i+1]:
            local_mins.append(last_15_lows[i])
    
    # Need at least 2 lows to compare
    if len(local_mins) < 2:
        return False
    
    # Check if latest low is higher than previous
    return local_mins[-1] > local_mins[-2]


def detect_breakout_exclusion(data):
    """Detect if there was a significant breakout in last 5 candles"""
    if len(data) < 10:
        return False
    
    atr = calculate_atr(data)
    if atr == 0:
        return False
    
    last_5_candles = data[-5:]
    
    for candle in last_5_candles:
        candle_range = candle[2] - candle[3]  # high - low
        
        # If any candle > 2.5x ATR = breakout detected
        if candle_range > (2.5 * atr):
            return True
    
    return False


def compute_trend_score(data, symbol=None, enable_extensions=True):
    """
    Trend Mode v1.0 + PPWCS-T 2.0 - Professional trend continuation scoring with extensions
    
    Args:
        data: OHLCV candle data (list of [timestamp, open, high, low, close, volume])
        symbol: token symbol for additional analysis
        enable_extensions: whether to use advanced extension modules
    
    Returns:
        dict: {
            "trend_score": int (0-77),
            "trend_mode_active": bool,
            "trend_summary": list of active signals,
            "activation_details": str,
            "extensions": dict (if enabled),
            "ppwcs_t_boost": dict (PPWCS-T 2.0 results)
        }
    """
    print(f"[TREND MODE v1.0 + PPWCS-T 2.0] Analyzing trend for {symbol}")
    
    # Step 1: PPWCS-T 2.0 Trend Boost Analysis (independent of legacy activation)
    ppwcs_t_result = compute_ppwcs_t_trend_boost(data, symbol)
    ppwcs_t_boost = ppwcs_t_result["trend_boost"]
    ppwcs_t_active = ppwcs_t_result["trend_mode_active"]
    
    print(f"[PPWCS-T 2.0] Boost: {ppwcs_t_boost} points, Active: {ppwcs_t_active}")
    
    # Step 2: Check mandatory activation conditions (legacy system)
    trend_active, activation_details = check_trend_activation(data)
    
    # PPWCS-T 2.0 can override legacy activation if strong enough
    final_trend_active = trend_active or ppwcs_t_active
    
    if not final_trend_active:
        print(f"[TREND MODE] ❌ Both systems failed - Legacy: {activation_details}, PPWCS-T: {ppwcs_t_boost} boost")
        return {
            "trend_score": ppwcs_t_boost,  # Still return PPWCS-T boost even if not fully active
            "trend_mode_active": False,
            "trend_summary": ppwcs_t_result["boost_details"],
            "activation_details": f"Legacy: {activation_details}, PPWCS-T: {ppwcs_t_boost}/10 boost",
            "extensions": {},
            "ppwcs_t_boost": ppwcs_t_result
        }
    
    if trend_active:
        print(f"[TREND MODE] ✅ Legacy activation confirmed: {activation_details}")
    else:
        print(f"[TREND MODE] ⚠️ Legacy activation failed, but PPWCS-T 2.0 override active")
    
    # Step 3: Calculate basic trend continuation score (max 50 points)
    score = ppwcs_t_boost  # Start with PPWCS-T 2.0 boost
    summary = ppwcs_t_result["boost_details"].copy()  # Include PPWCS-T details
    
    # Calculate required metrics
    rsi = calculate_rsi(data)
    atr = calculate_atr(data)
    vwap = calculate_vwap(data)
    
    if len(data) < 3:
        return {
            "trend_score": 0,
            "trend_mode_active": False,
            "trend_summary": [],
            "activation_details": "Insufficient data",
            "extensions": {}
        }
    
    last_candle = data[-1]
    last_3_candles = data[-3:]
    
    # 1. RSI > 68 (+10 points) - already verified in activation
    if rsi > 68:
        score += 10
        summary.append(f"RSI strong ({rsi:.1f})")
        print(f"[TREND] ✅ RSI strong: +10 ({rsi:.1f})")
    
    # 2. Candle > 2x ATR (+10 points) - strong momentum candle
    candle_range = last_candle[2] - last_candle[3]  # high - low
    if atr > 0 and candle_range > (2 * atr):
        score += 10
        summary.append("2x ATR candle")
        print(f"[TREND] ✅ Large momentum candle: +10 (range={candle_range:.4f}, 2xATR={2*atr:.4f})")
    
    # 3. Volume rising (3 candles) (+10 points) - already verified in activation
    last_3_volumes = [candle[5] for candle in last_3_candles]
    volume_rising = all(last_3_volumes[i] > last_3_volumes[i-1] for i in range(1, 3))
    if volume_rising:
        score += 10
        summary.append("Volume rising")
        print(f"[TREND] ✅ Volume rising trend: +10")
    
    # 4. 3 consecutive closes higher (+5 points)
    last_3_closes = [candle[4] for candle in last_3_candles]
    closes_rising = all(last_3_closes[i] > last_3_closes[i-1] for i in range(1, 3))
    if closes_rising:
        score += 5
        summary.append("3x close up")
        print(f"[TREND] ✅ Rising closes: +5")
    
    # 5. Price > VWAP for 3 candles (+5 points) - already verified in activation
    last_3_closes = [candle[4] for candle in last_3_candles]
    above_vwap_count = sum(1 for close in last_3_closes if close > vwap)
    if above_vwap_count >= 3:
        score += 5
        summary.append("Above VWAP")
        print(f"[TREND] ✅ Above VWAP: +5")
    
    # 6. Orderbook bid pressure (+5 points)
    bid_pressure, pressure_ratio = detect_orderbook_pressure(symbol)
    if bid_pressure:
        score += 5
        summary.append("Orderbook bid pressure")
        print(f"[TREND] ✅ Bid pressure: +5")
    
    # 7. Social burst during trend (+5 points)
    social_burst = detect_social_burst(symbol)
    if social_burst:
        score += 5
        summary.append("Social burst")
        print(f"[TREND] ✅ Social momentum: +5")
    
    # 8. EMA Alignment Detection (+7 points) - NEW MODULE
    try:
        from utils.ema_alignment_detector import compute_ema_alignment_boost
        ema_result = compute_ema_alignment_boost(data, symbol)
        
        ema_boost = ema_result.get("ema_boost", 0)
        if ema_boost > 0:
            score += ema_boost
            summary.append(ema_result.get("summary_text", "EMA alignment confirmed"))
            print(f"[TREND] ✅ EMA alignment: +{ema_boost} ({ema_result.get('summary_text', 'confirmed')})")
        else:
            print(f"[TREND] ❌ EMA alignment: not detected")
            
    except Exception as ema_error:
        print(f"[TREND] ⚠️ EMA alignment error: {ema_error}")
    
    base_score = score
    base_summary = summary.copy()
    
    print(f"[TREND MODE] Base score: {base_score}/57 points")
    
    # Step 3: Apply Extension Modules (if enabled)
    extensions = {}
    
    if enable_extensions and symbol:
        print(f"[TREND EXTENSIONS] Running advanced modules for {symbol}")
        
        try:
            # Extension 1: Trailing TP Engine
            from utils.trailing_tp_engine import compute_trailing_tp_levels
            tp_levels = compute_trailing_tp_levels(data, symbol, base_score)
            extensions["trailing_tp"] = tp_levels
            
            # Extension 2: Breakout Cluster Scoring
            from utils.breakout_cluster_scoring import compute_cluster_boost
            cluster_result = compute_cluster_boost(symbol, base_score, base_summary)
            extensions["cluster_scoring"] = cluster_result
            
            # Add cluster boost to score (max 25 additional points)
            cluster_boost = cluster_result.get("cluster_boost", 0)
            if cluster_boost > 0:
                score = min(50, score + min(cluster_boost, 10))  # Cap boost at 10 for trend mode
                summary.append(f"Cluster boost (+{min(cluster_boost, 10)})")
                print(f"[TREND EXTENSIONS] Cluster boost: +{min(cluster_boost, 10)} points")
            
            # Extension 3: VWAP Anchoring Tracker
            from utils.vwap_anchoring_tracker import compute_vwap_anchoring_score
            vwap_result = compute_vwap_anchoring_score(data, symbol, "up")
            extensions["vwap_anchoring"] = vwap_result
            
            # Add VWAP anchoring boost (max 15 additional points)
            vwap_boost = vwap_result.get("anchoring_score", 0)
            if vwap_boost >= 8:  # Only if anchoring is active
                boost_points = min(8, int(vwap_boost * 0.3))  # 30% of anchoring score, max 8
                score = min(50, score + boost_points)
                summary.append(f"VWAP anchoring (+{boost_points})")
                print(f"[TREND EXTENSIONS] VWAP anchoring: +{boost_points} points")
            
            # Extension 4: Trend Confirmation GPT (for high scores only)
            if base_score >= 35:  # Only for strong trends
                from utils.trend_confirmation_gpt import compute_gpt_trend_boost
                gpt_result = compute_gpt_trend_boost(symbol, base_score, base_summary, data, tp_levels)
                extensions["gpt_confirmation"] = gpt_result
                
                # Add GPT boost (max 10 additional points)
                gpt_boost = gpt_result.get("gpt_boost", 0)
                if gpt_boost > 0:
                    score = min(50, score + gpt_boost)
                    summary.append(f"GPT confirmation (+{gpt_boost})")
                    print(f"[TREND EXTENSIONS] GPT boost: +{gpt_boost} points")
            else:
                extensions["gpt_confirmation"] = {"gpt_boost": 0, "gpt_analysis": {"gpt_assessment": "Score below GPT threshold (35)"}}
            
        except Exception as e:
            print(f"[TREND EXTENSIONS] Error in extensions: {e}")
            extensions["error"] = str(e)
    
    print(f"[TREND MODE] Final score: {score}/77 points (base: {base_score}, PPWCS-T: {ppwcs_t_boost})")
    print(f"[TREND MODE] Active signals: {len(summary)}")
    
    # Final activation decision
    final_activation = score >= 35 or ppwcs_t_active  # 35 points threshold or PPWCS-T override
    
    return {
        "trend_score": score,
        "trend_mode_active": final_activation,
        "trend_summary": summary,
        "activation_details": f"Legacy: {activation_details}, PPWCS-T: {ppwcs_t_boost} boost",
        "extensions": extensions,
        "ppwcs_t_boost": ppwcs_t_result
    }


def check_trend_cooldown(symbol, cooldown_minutes=45):
    """
    Check if symbol is in trend mode cooldown (independent from pre-pump)
    Returns: (in_cooldown, remaining_minutes)
    """
    try:
        cooldown_file = os.path.join("data", "trend_cooldown.json")
        
        if not os.path.exists(cooldown_file):
            return False, 0
        
        with open(cooldown_file, 'r') as f:
            cooldowns = json.load(f)
        
        if symbol not in cooldowns:
            return False, 0
        
        last_alert_time = datetime.fromisoformat(cooldowns[symbol])
        time_diff = datetime.now(timezone.utc) - last_alert_time
        
        if time_diff.total_seconds() < (cooldown_minutes * 60):
            remaining = cooldown_minutes - (time_diff.total_seconds() / 60)
            return True, remaining
        
        return False, 0
        
    except Exception as e:
        print(f"[TREND MODE] Cooldown check error: {e}")
        return False, 0


def update_trend_cooldown(symbol):
    """Update trend mode cooldown timestamp"""
    try:
        cooldown_file = os.path.join("data", "trend_cooldown.json")
        
        cooldowns = {}
        if os.path.exists(cooldown_file):
            with open(cooldown_file, 'r') as f:
                cooldowns = json.load(f)
        
        cooldowns[symbol] = datetime.now(timezone.utc).isoformat()
        
        with open(cooldown_file, 'w') as f:
            json.dump(cooldowns, f, indent=2)
            
        print(f"[TREND MODE] Cooldown updated for {symbol}")
        
    except Exception as e:
        print(f"[TREND MODE] Cooldown update error: {e}")


def get_trend_alert_message(symbol, trend_score, trend_summary, trend_data):
    """
    Generate Polish alert message for trend mode
    Separate formatting from pre-pump alerts
    """
    activation_details = trend_data.get("activation_details", "")
    
    # Determine trend strength
    if trend_score >= 45:
        strength = "🔥 SILNY TREND"
        confidence = "Bardzo wysoka"
    elif trend_score >= 40:
        strength = "🚀 MOCNY TREND"
        confidence = "Wysoka"
    elif trend_score >= 35:
        strength = "📈 TREND"
        confidence = "Średnia"
    else:
        strength = "📊 SŁABY TREND"
        confidence = "Niska"
    
    message = f"""
{strength} | {symbol}
🎯 Kontynuacja trendu: {trend_score}/50 pkt
📊 Pewność: {confidence}

✅ Aktywacja:
{activation_details}

🔍 Sygnały ({len(trend_summary)}/7):
"""
    
    for signal in trend_summary:
        message += f"• {signal}\n"
    
    message += f"""
⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}
🤖 Trend Mode v1.0 by SMB Capital methods
"""
    
    return message.strip()


def save_trend_alert(symbol, trend_score, trend_summary, trend_data):
    """Save trend mode alert to data files"""
    try:
        alert_data = {
            "symbol": symbol,
            "trend_score": trend_score,
            "trend_summary": trend_summary,
            "trend_data": trend_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_type": "trend_mode"
        }
        
        # Save to trend alerts file
        alerts_file = os.path.join("data", "trend_alerts.json")
        
        alerts = []
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        
        alerts.append(alert_data)
        
        # Keep only last 100 alerts
        alerts = alerts[-100:]
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        print(f"[TREND MODE] Alert saved for {symbol}")
        
    except Exception as e:
        print(f"[TREND MODE] Save alert error: {e}")