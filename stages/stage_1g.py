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

def detect_fake_reject(data):
    """Detect fake rejection - wick followed by recovery"""
    try:
        if not isinstance(data, dict):
            return False
            
        last_candle = data.get("last_candle", {})
        prev_candle = data.get("prev_candle", {})
        
        if not isinstance(last_candle, dict) or not isinstance(prev_candle, dict):
            return False
            
        # Get candle data
        last_low = float(last_candle.get("low", 0))
        last_close = float(last_candle.get("close", 0))
        last_open = float(last_candle.get("open", 0))
        
        prev_close = float(prev_candle.get("close", 0))
        
        if last_low <= 0 or last_close <= 0 or prev_close <= 0:
            return False
            
        # Fake reject: significant wick below open/close, but recovery
        wick_size = min(last_open, last_close) - last_low
        wick_pct = wick_size / last_low * 100
        
        # Current close above previous close despite the wick
        recovery = last_close > prev_close
        
        return wick_pct > 2.0 and recovery
    except:
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