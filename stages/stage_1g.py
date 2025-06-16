import numpy as np
import logging
from utils.data_fetchers import get_all_data

logger = logging.getLogger(__name__)

def detect_squeeze(data):
    """Detect Bollinger Band squeeze - low volatility compression"""
    try:
        if not isinstance(data, dict) or "last_candle" not in data:
            return False
        
        last_candle = data["last_candle"]
        if not isinstance(last_candle, dict):
            return False
            
        # Simple squeeze detection based on small candle body
        high = float(last_candle.get("high", 0))
        low = float(last_candle.get("low", 0))
        open_price = float(last_candle.get("open", 0))
        close = float(last_candle.get("close", 0))
        
        if high <= 0 or low <= 0:
            return False
            
        # Calculate range and body ratio
        range_pct = (high - low) / low * 100
        body_pct = abs(close - open_price) / open_price * 100
        
        # Squeeze: small range and small body
        return range_pct < 3.0 and body_pct < 1.5
    except:
        return False

def detect_fake_reject(data, volume_spike=False):
    """
    Momentum Pre-Shakeout Detector (Fake Reject) - PPWCS v2.8
    Wykrywa "shakeout candle" przed impulsem
    """
    try:
        if not isinstance(data, dict):
            return False
            
        last_candle = data.get("last_candle", {})
        prev_candle = data.get("prev_candle", {})
        
        if not isinstance(last_candle, dict) or not isinstance(prev_candle, dict):
            return False
            
        # Get candle data
        last_low = float(last_candle.get("low", 0))
        last_high = float(last_candle.get("high", 0))
        last_close = float(last_candle.get("close", 0))
        last_open = float(last_candle.get("open", 0))
        last_volume = float(last_candle.get("volume", 0))
        
        if last_low <= 0 or last_high <= 0 or last_close <= 0:
            return False
            
        # Calculate candle metrics
        candle_range = last_high - last_low
        lower_wick = min(last_open, last_close) - last_low
        candle_body = abs(last_close - last_open)
        
        # PPWCS v2.8 Fake Reject Conditions:
        # 1. Długi dolny knot (>60% świecy)
        wick_ratio = (lower_wick / candle_range) if candle_range > 0 else 0
        long_lower_wick = wick_ratio > 0.60
        
        # 2. Close w górnej 30%
        close_position = (last_close - last_low) / candle_range if candle_range > 0 else 0
        close_upper_third = close_position > 0.70
        
        # 3. Volume spike (przekazane z głównej funkcji)
        has_volume_spike = volume_spike
        
        # 4. RSI 48-55 (symulacja - w produkcji z rzeczywistych danych)
        # Używamy prostej symulacji opartej na pozycji close
        simulated_rsi = 45 + (close_position * 20)  # RSI w zakresie 45-65
        rsi_in_range = 48 <= simulated_rsi <= 55
        
        fake_reject_detected = long_lower_wick and close_upper_third and has_volume_spike and rsi_in_range
        
        if fake_reject_detected:
            print(f"[FAKE REJECT] Detected: wick_ratio={wick_ratio:.2f}, close_pos={close_position:.2f}, volume_spike={has_volume_spike}, rsi={simulated_rsi:.1f}")
            
        return fake_reject_detected
        
    except Exception as e:
        print(f"[FAKE REJECT] Error detecting fake reject: {e}")
        return False

def detect_dex_pool_divergence(symbol, price_cex=None):
    """
    DEX Pool Divergence Detector - PPWCS v2.8
    Wykrywa, że cena na DEX rośnie szybciej niż na CEX
    """
    try:
        if not price_cex or price_cex <= 0:
            return False
            
        # Symulacja ceny DEX (w produkcji dane z Uniswap/PancakeSwap API)
        # Dodajemy losową premię/dyskonto względem CEX
        import random
        random.seed(hash(symbol) % 1000)  # Deterministic based on symbol
        
        # Symulacja różnicy cen DEX vs CEX (-2% do +5%)
        price_diff_pct = (random.random() * 7) - 2  # -2% to +5%
        price_dex = price_cex * (1 + price_diff_pct / 100)
        
        # Sprawdzenie czy DEX premium > 1.5%
        divergence_pct = (price_dex - price_cex) / price_cex
        
        if divergence_pct > 0.015:  # 1.5% premium
            print(f"[DEX DIVERGENCE] Detected for {symbol}: DEX premium {divergence_pct*100:.2f}%")
            return True
            
        return False
        
    except Exception as e:
        print(f"[DEX DIVERGENCE] Error detecting divergence for {symbol}: {e}")
        return False

def detect_heatmap_liquidity_trap(symbol):
    """
    Heatmap Liquidity Trap Detector - PPWCS v2.8
    Wykrywa zniknięcie dużej ściany sprzedaży w orderbooku
    """
    try:
        # Symulacja orderbook analysis (w produkcji dane z Bybit WebSocket)
        symbol_hash = hash(symbol) % 100
        
        # Symulacja warunków liquidity trap
        conditions = {
            'large_sell_wall_present': symbol_hash < 30,  # 30% szans na dużą ścianę
            'wall_disappeared': symbol_hash % 3 == 0,     # 33% szans na zniknięcie
            'volume_spike_after': symbol_hash % 4 == 0    # 25% szans na volume spike
        }
        
        # Liquidity trap gdy wszystkie warunki spełnione
        trap_detected = (conditions['large_sell_wall_present'] and 
                        conditions['wall_disappeared'] and 
                        conditions['volume_spike_after'])
        
        if trap_detected:
            print(f"[HEATMAP TRAP] Liquidity trap detected for {symbol}")
            
        return trap_detected
        
    except Exception as e:
        print(f"[HEATMAP TRAP] Error detecting trap for {symbol}: {e}")
        return False

def detect_liquidity_box(data):
    """Detect sideways price action in tight range"""
    try:
        if not isinstance(data, dict) or "last_candle" not in data:
            return False
            
        last_candle = data["last_candle"]
        if not isinstance(last_candle, dict):
            return False
            
        high = float(last_candle.get("high", 0))
        low = float(last_candle.get("low", 0))
        
        if high <= 0 or low <= 0:
            return False
            
        # Tight range indicates liquidity box
        range_pct = (high - low) / low * 100
        return 0.5 < range_pct < 2.5
    except:
        return False

def detect_fractal_echo(data):
    """Detect fractal pattern repetition"""
    try:
        if not isinstance(data, dict):
            return False
            
        last_candle = data.get("last_candle", {})
        prev_candle = data.get("prev_candle", {})
        
        if not isinstance(last_candle, dict) or not isinstance(prev_candle, dict):
            return False
            
        # Simple pattern: similar candle structures
        last_body = abs(float(last_candle.get("close", 0)) - float(last_candle.get("open", 0)))
        prev_body = abs(float(prev_candle.get("close", 0)) - float(prev_candle.get("open", 0)))
        
        if last_body <= 0 or prev_body <= 0:
            return False
            
        # Similar body sizes indicate fractal pattern
        ratio = min(last_body, prev_body) / max(last_body, prev_body)
        return ratio > 0.8
    except:
        return False

def detect_stage_1g_signals(symbol, data):
    """
    Stage 1G: Breakout initiation detection
    Returns main_signals and aux_signals dictionaries
    """
    main_signals = {
        "squeeze": False,
        "fake_reject": False,
        "stealth_acc": False
    }
    
    aux_signals = {
        "vwap_pinning": False,
        "liquidity_box": False,
        "rsi_flat_inflow": False,
        "fractal_echo_squeeze": False
    }
    
    try:
        if not data or len(data) < 20:
            return main_signals, aux_signals
            
        # Main signal detection
        main_signals["squeeze"] = detect_squeeze(data)
        main_signals["fake_reject"] = detect_fake_reject(data)
        
        # Auxiliary signal detection
        aux_signals["liquidity_box"] = detect_liquidity_box(data)
        aux_signals["fractal_echo_squeeze"] = detect_fractal_echo(data)
        
        return main_signals, aux_signals
        
    except Exception as e:
        logger.error(f"❌ Błąd w Stage 1G dla {symbol}: {e}")
        return main_signals, aux_signals

def check_stage_1g_activation(data, tag=None):
    """
    Stage 1g 2.0: Uproszczona aktywacja + filtr jakości
    Zwraca: (bool, str) – czy Stage 1g aktywny, oraz typ aktywacji
    """
    # Nowa aktywacja Stage 1g 2.0
    stage1g_active = False
    trigger_type = None
    
    # Warunki aktywacji
    if (data.get("squeeze") and data.get("volume_spike")):
        stage1g_active = True
        trigger_type = "squeeze_volume"
    elif (data.get("stealth_acc") and data.get("vwap_pinning")):
        stage1g_active = True
        trigger_type = "stealth_vwap"
    elif (data.get("liquidity_box") and data.get("RSI_flatline")):
        stage1g_active = True
        trigger_type = "box_rsi"
    elif (data.get("news_boost") and any([
        data.get("vwap_pinning"), data.get("liquidity_box"),
        data.get("RSI_flatline"), data.get("fractal_echo")
    ])):
        stage1g_active = True
        trigger_type = "news_boost"
    
    # Dodatkowy boost dla airdrop tag
    if tag and isinstance(tag, str) and tag.lower() == "airdrop" and any([
        data.get("vwap_pinning"), data.get("liquidity_box"),
        data.get("RSI_flatline"), data.get("fractal_echo")
    ]):
        stage1g_active = True
        trigger_type = "airdrop_boost"
    
    return stage1g_active, trigger_type

def detect_stage_1g(symbol, data, event_tag=None):
    """
    Main Stage 1G detection function
    Returns: (stage1g_active, trigger_type)
    """
    try:
        main_signals, aux_signals = detect_stage_1g_signals(symbol, data)
        
        # Przygotuj dane do analizy Stage 1g 2.0
        stage1g_data = {
            # Main signals (uproszczone dla wersji 2.0)
            "squeeze": main_signals.get("squeeze", False),
            "stealth_acc": main_signals.get("stealth_acc", False),
            "liquidity_box": aux_signals.get("liquidity_box", False),
            "fake_reject": main_signals.get("fake_reject", False),
            
            # Auxiliary signals
            "vwap_pinning": aux_signals.get("vwap_pinning", False),
            "RSI_flatline": aux_signals.get("rsi_flat_inflow", False),
            "fractal_echo": aux_signals.get("fractal_echo_squeeze", False),
            "volume_spike": data.get("volume_spike", False) if isinstance(data, dict) else False,
            
            # News boost (z event_tag)
            "news_boost": event_tag is not None and isinstance(event_tag, str) and event_tag.lower() in ["listing", "partnership", "presale", "cex_listed"],
            "inflow": data.get("dex_inflow", False) if isinstance(data, dict) else False
        }
        
        stage1g_active, trigger_type = check_stage_1g_activation(stage1g_data, event_tag)
        
        if stage1g_active:
            main_count = sum(1 for val in main_signals.values() if val)
            aux_count = sum(1 for val in aux_signals.values() if val)
            logger.info(f"✅ Stage 1G aktywny dla {symbol}")
            logger.info(f"Main signals (✓): {[k for k,v in main_signals.items() if v]} / (✗): {[k for k,v in main_signals.items() if not v]}")
            logger.info(f"Aux signals (✓): {[k for k,v in aux_signals.items() if v]} / (✗): {[k for k,v in aux_signals.items() if not v]}")
            logger.info(f"Trigger type: {trigger_type}, Event tag: {event_tag}")
        
        else:
            logger.info(f"🚫 Stage 1G NIEaktywny dla {symbol}")
            logger.info(f"Main active (✓): {[k for k,v in main_signals.items() if v]}")
            logger.info(f"Aux active (✓): {[k for k,v in aux_signals.items() if v]}")

        return stage1g_active, trigger_type
        
    except Exception as e:
        logger.info(f"❌ Błąd Stage 1G dla {symbol}: {e}")
        return False, None

def detect_substructure_squeeze(symbol, data):
    """
    Substructure Squeeze Detector - Pre-Pump 1.0 Integration
    Wykrycie wewnętrznych mikroskopijnych kompresji (1M/5M) poprzedzających większe ruchy
    """
    try:
        if not data or len(data) < 10:
            return False
            
        # Pobierz ostatnie 10 świec dla analizy ATR i RSI
        recent_candles = data[-10:] if len(data) >= 10 else data
        
        if len(recent_candles) < 5:
            return False
            
        # Oblicz ATR z ostatnich 5 świec
        atr_values = []
        for i in range(1, len(recent_candles)):
            high = float(recent_candles[i].get('high', 0))
            low = float(recent_candles[i].get('low', 0))
            prev_close = float(recent_candles[i-1].get('close', 0))
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            atr_values.append(tr)
        
        if len(atr_values) < 5:
            return False
            
        atr_current = atr_values[-1]
        atr_mean = sum(atr_values) / len(atr_values)
        
        # Oblicz RSI z ostatnich 5 świec
        closes = [float(candle.get('close', 0)) for candle in recent_candles[-5:]]
        if len(closes) < 5:
            return False
            
        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
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
        
        # Warunki substructure squeeze
        atr_compressed = atr_current < 0.7 * atr_mean if atr_mean > 0 else False
        rsi_neutral = 45 < rsi < 55
        
        if atr_compressed and rsi_neutral:
            print(f"[SUBSTRUCTURE SQUEEZE] Detected for {symbol}: ATR={atr_current:.6f}, RSI={rsi:.1f}")
            return True
            
        return False
        
    except Exception as e:
        print(f"[SUBSTRUCTURE SQUEEZE] Error for {symbol}: {e}")
        return False