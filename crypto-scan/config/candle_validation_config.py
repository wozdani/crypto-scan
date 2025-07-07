"""
Candle Validation Configuration
Centralized configuration for minimum candle requirements
"""

# 🎯 CANDLE VALIDATION THRESHOLDS
# These values determine minimum candle history required for analysis

# Minimum 15M candles (each candle = 15 minutes)
MIN_15M_CANDLES = 20  # ~5 hours of trading history
# Reasoning: 20 candles provide sufficient data for:
# - Basic trend analysis
# - EMA calculations (EMA20 needs ~20 periods)
# - Support/resistance identification
# - Volume pattern analysis

# Minimum 5M candles (each candle = 5 minutes) 
MIN_5M_CANDLES = 60   # ~5 hours of trading history
# Reasoning: 60 candles provide sufficient data for:
# - Detailed price action analysis  
# - Micro-trend detection
# - Entry/exit timing precision
# - Intraday pattern recognition

# Alternative configurations for different market conditions
STRICT_VALIDATION = {
    "MIN_15M_CANDLES": 40,  # ~10 hours
    "MIN_5M_CANDLES": 120,  # ~10 hours
    "description": "Strict validation for high-quality analysis"
}

RELAXED_VALIDATION = {
    "MIN_15M_CANDLES": 10,  # ~2.5 hours
    "MIN_5M_CANDLES": 30,   # ~2.5 hours  
    "description": "Relaxed validation for emerging tokens"
}

# Current active configuration
ACTIVE_CONFIG = "STANDARD"  # Options: STANDARD, STRICT, RELAXED

def get_candle_thresholds(config_type: str = None) -> dict:
    """
    Get candle validation thresholds for specified configuration
    
    Args:
        config_type: Configuration type (STANDARD, STRICT, RELAXED)
        
    Returns:
        Dictionary with MIN_15M_CANDLES and MIN_5M_CANDLES
    """
    if config_type == "STRICT":
        return STRICT_VALIDATION
    elif config_type == "RELAXED":
        return RELAXED_VALIDATION
    else:
        # STANDARD configuration
        return {
            "MIN_15M_CANDLES": MIN_15M_CANDLES,
            "MIN_5M_CANDLES": MIN_5M_CANDLES,
            "description": "Standard validation for balanced analysis"
        }

def should_skip_token(candles_15m_count: int, candles_5m_count: int, config_type: str = None) -> tuple:
    """
    Determine if token should be skipped based on candle validation
    
    Args:
        candles_15m_count: Number of 15M candles available
        candles_5m_count: Number of 5M candles available  
        config_type: Configuration type to use
        
    Returns:
        Tuple: (should_skip: bool, reason: str)
    """
    thresholds = get_candle_thresholds(config_type or ACTIVE_CONFIG)
    
    min_15m = thresholds["MIN_15M_CANDLES"]
    min_5m = thresholds["MIN_5M_CANDLES"]
    
    if candles_15m_count < min_15m:
        return True, f"Insufficient 15M candles ({candles_15m_count}/{min_15m})"
        
    if candles_5m_count < min_5m:
        return True, f"Insufficient 5M candles ({candles_5m_count}/{min_5m})"
        
    return False, f"Candle validation passed (15M: {candles_15m_count}, 5M: {candles_5m_count})"

# Export commonly used values
__all__ = [
    'MIN_15M_CANDLES',
    'MIN_5M_CANDLES', 
    'get_candle_thresholds',
    'should_skip_token',
    'ACTIVE_CONFIG'
]