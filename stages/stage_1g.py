import numpy as np
import logging
from utils.data_fetchers import get_all_data

logger = logging.getLogger(__name__)

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
            
        # Main signal detection (simplified for now)
        # TODO: Implement actual detection algorithms
        
        # Auxiliary signal detection (simplified for now)
        # TODO: Implement actual detection algorithms
        
        return main_signals, aux_signals
        
    except Exception as e:
        logger.error(f"❌ Błąd w Stage 1G dla {symbol}: {e}")
        return main_signals, aux_signals

def check_stage_1g_activation(main_signals, aux_signals, tag=None):
    """
    Zwraca: (bool, str) – czy Stage 1g aktywny, oraz typ aktywacji
    """
    main_count = sum(1 for val in main_signals.values() if val)
    aux_count = sum(1 for val in aux_signals.values() if val)

    boost_tags = {"listing", "partnership", "cex_listed", "presale", "airdrop"}

    if main_count >= 2:
        return True, "classic"
    if main_count >= 1 and aux_count >= 2:
        return True, "classic"

    if tag and tag.lower() in boost_tags and aux_count >= 1:
        logger.info(f"🚀 Stage 1G aktywowany przez tag '{tag}' + {aux_count} aux sygnałów")
        return True, "tag_boost"

    return False, None

def detect_stage_1g(symbol, data, event_tag=None):
    """
    Main Stage 1G detection function
    Returns: (stage1g_active, trigger_type)
    """
    try:
        main_signals, aux_signals = detect_stage_1g_signals(symbol, data)
        
        stage1g_active, trigger_type = check_stage_1g_activation(main_signals, aux_signals, event_tag)
        
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