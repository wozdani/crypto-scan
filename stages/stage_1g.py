import numpy as np
from utils.data_fetchers import get_all_data

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
        print(f"❌ Błąd w Stage 1G dla {symbol}: {e}")
        return main_signals, aux_signals

def check_stage_1g_activation(main_signals, aux_signals, tag=None):
    """
    Hierarchical logic for Stage 1G activation
    """
    main_count = sum(main_signals.values())
    aux_count = sum(aux_signals.values())

    # Primary conditions
    if main_count >= 2:
        return True
    if main_count >= 1 and aux_count >= 2:
        return True

    # News tag boost condition
    boost_tags = {"listing", "partnership", "cex_listed", "presale", "airdrop"}
    if tag and tag.lower() in boost_tags and aux_count >= 1:
        print(f"🚀 Stage 1G aktywowany przez tag '{tag}' + {aux_count} aux sygnałów")
        return True

    return False

def detect_stage_1g(symbol, data, event_tag=None):
    """
    Main Stage 1G detection function
    Returns: stage1g_active boolean
    """
    try:
        main_signals, aux_signals = detect_stage_1g_signals(symbol, data)
        
        stage1g_active = check_stage_1g_activation(main_signals, aux_signals, event_tag)
        
        if stage1g_active:
            main_count = sum(main_signals.values())
            aux_count = sum(aux_signals.values())
            print(f"✅ Stage 1G aktywny dla {symbol}: main={main_count}, aux={aux_count}, tag={event_tag}")
        
        return stage1g_active
        
    except Exception as e:
        print(f"❌ Błąd Stage 1G dla {symbol}: {e}")
        return False