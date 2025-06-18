"""
GPT Generated Detector Function
Generated: 2025-06-18T17:10:23.348349
Symbol: TESTUSDT
Pump Date: 20250618
Active Signals: volume_spike, compression
Pump Increase: N/A%
Duration: N/A minutes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def detect_pre_pump_testusdt_20250618(data: pd.DataFrame) -> Dict:
    """
    Test detector function for TESTUSDT
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Dict with detection results
    """
    
    # Simple detection logic for testing
    if len(data) < 10:
        return {
            'signal_detected': False,
            'confidence': 0.0,
            'active_signals': [],
            'detection_reason': 'Insufficient data'
        }
    
    # Check for volume spike
    recent_volume = data['volume'].tail(3).mean()
    avg_volume = data['volume'].head(-3).mean() if len(data) > 6 else recent_volume
    
    volume_spike = recent_volume > avg_volume * 2.0 if avg_volume > 0 else False
    
    # Check for price compression
    price_range = data['high'].tail(5).max() - data['low'].tail(5).min()
    avg_range = (data['high'] - data['low']).head(-5).mean() if len(data) > 10 else price_range
    
    compression = price_range < avg_range * 0.8 if avg_range > 0 else False
    
    # Combine signals
    signals = []
    if volume_spike:
        signals.append('volume_spike')
    if compression:
        signals.append('compression')
    
    signal_detected = len(signals) >= 2
    confidence = len(signals) / 2.0  # Max 2 signals = 100% confidence
    
    return {
        'signal_detected': signal_detected,
        'confidence': confidence,
        'active_signals': signals,
        'detection_reason': f"Detected {len(signals)} signals: {', '.join(signals)}"
    }

# Metadata for learning system
FUNCTION_METADATA = {
    "function_name": "detect_pre_pump_testusdt_20250618",
    "symbol": "TESTUSDT",
    "pump_date": "20250618",
    "active_signals": ["volume_spike", "compression"],
    "generated_timestamp": "2025-06-18T17:00:00",
    "version": 1
}


# Metadata for learning system
FUNCTION_METADATA = {
    "function_name": "detect_pre_pump_testusdt_20250618",
    "symbol": "TESTUSDT",
    "pump_date": "20250618",
    "active_signals": ['volume_spike', 'compression'],
    "generated_timestamp": "2025-06-18T17:10:23.348362",
    "pump_increase_pct": 0,
    "pump_duration_minutes": 0,
    "version": 1
}
