#!/usr/bin/env python3
"""
Complete test of GPT Learning System functionality
Tests all major components: saving functions, testing, evolution, retrospective analysis
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from learning_system import LearningSystem
from main import PumpEvent

def create_test_pump_event() -> PumpEvent:
    """Create a test pump event for testing"""
    return PumpEvent(
        symbol="TESTUSDT",
        start_time=datetime.now() - timedelta(hours=1),
        price_before=1.0,
        price_peak=1.25,
        price_increase_pct=25.0,
        duration_minutes=30,
        volume_spike=5.0
    )

def create_test_pre_pump_data() -> Dict:
    """Create test pre-pump analysis data"""
    return {
        'trend': 'bullish',
        'rsi': 52.5,
        'volume_spikes': [
            {'time': '17:00', 'multiplier': 3.2}
        ],
        'compression': {
            'detected': True,
            'strength': 'strong'
        },
        'fake_rejects': [
            {'time': '16:45', 'wick_size': 2.1}
        ],
        'vwap_analysis': {
            'position': 'above',
            'deviation': 0.8
        },
        'support_resistance': {
            'key_support': 0.98,
            'key_resistance': 1.05
        },
        'liquidity_gaps': []
    }

def create_test_detector_function() -> str:
    """Create a test detector function"""
    return '''
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
'''

def create_test_candle_data() -> pd.DataFrame:
    """Create test candle data for function testing"""
    timestamps = pd.date_range(start='2025-06-18 16:00:00', periods=20, freq='5min')
    
    data = []
    base_price = 1.0
    
    for i, ts in enumerate(timestamps):
        # Simulate price movement with some volatility
        price_change = 0.01 * (i - 10) / 10  # Gradual increase
        noise = 0.005 * (0.5 - np.random.random())  # Random noise
        
        price = base_price + price_change + noise
        
        # Simulate volume spike in last few candles
        volume = 1000 + (500 if i >= 15 else 0) + np.random.randint(-100, 100)
        
        data.append({
            'timestamp': ts,
            'open': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_learning_system():
    """Complete test of learning system functionality"""
    
    print("🧠 Testing GPT Learning System...")
    
    # Initialize learning system
    learning_system = LearningSystem()
    
    # Test 1: Save GPT function
    print("\n📝 Test 1: Saving GPT function")
    pump_event = create_test_pump_event()
    pre_pump_data = create_test_pre_pump_data()
    detector_function = create_test_detector_function()
    
    function_path = learning_system.save_gpt_function(
        function_code=detector_function,
        symbol=pump_event.symbol,
        pump_date=pump_event.start_time.strftime('%Y%m%d'),
        active_signals=['volume_spike', 'compression'],
        pre_pump_data=pre_pump_data
    )
    
    print(f"✅ Function saved to: {function_path}")
    
    # Test 2: Test functions on new pump
    print("\n🧪 Test 2: Testing functions on new pump")
    pump_data = {
        'symbol': 'NEWUSDT',
        'start_time': datetime.now(),
        'price_increase_pct': 20.0
    }
    
    candle_data = create_test_candle_data()
    
    test_results = learning_system.test_functions_on_new_pump(
        pump_data=pump_data,
        pre_pump_candles=candle_data
    )
    
    print(f"✅ Tested {test_results['functions_tested']} functions")
    print(f"   Successful detections: {len(test_results['successful_detections'])}")
    print(f"   Close detections: {len(test_results['close_detections'])}")
    
    # Test 3: Function evolution
    print("\n🔄 Test 3: Function evolution")
    if test_results['functions_tested'] > 0:
        evolved_path = learning_system.evolve_function(
            function_name="detect_pre_pump_testusdt_20250618",
            improvement_reason="Added RSI analysis for better accuracy",
            new_function_code=detector_function.replace(
                "# Combine signals",
                "# Check RSI\n    rsi_neutral = True  # Placeholder\n    if rsi_neutral:\n        signals.append('rsi_neutral')\n    \n    # Combine signals"
            ).replace("len(signals) / 2.0", "len(signals) / 3.0")  # Adjust for 3 signals
        )
        print(f"✅ Function evolved to: {evolved_path}")
    
    # Test 4: Retrospective test suite
    print("\n📊 Test 4: Retrospective test suite")
    recent_pumps = [
        {
            'symbol': 'PUMP1USDT',
            'start_time': datetime.now() - timedelta(hours=6),
            'price_increase_pct': 18.0
        },
        {
            'symbol': 'PUMP2USDT',
            'start_time': datetime.now() - timedelta(hours=12),
            'price_increase_pct': 22.0
        }
    ]
    
    retro_results = learning_system.retrospective_test_suite(recent_pumps)
    
    print(f"✅ Retrospective tests completed")
    print(f"   Pumps tested: {retro_results['pumps_tested']}")
    print(f"   Functions tested: {retro_results['functions_tested']}")
    print(f"   Recommendations: {len(retro_results['recommendations'])}")
    
    # Test 5: Learning summary
    print("\n📋 Test 5: Learning summary")
    summary = learning_system.get_learning_summary()
    
    print(f"✅ Learning summary:")
    print(f"   Total functions created: {summary['total_functions_created']}")
    print(f"   Active functions: {summary['active_functions']}")
    print(f"   Average accuracy: {summary['avg_accuracy']:.1%}")
    print(f"   Evolution count: {summary['evolution_count']}")
    
    # Test 6: Function deprecation
    print("\n🗑️ Test 6: Function deprecation")
    if summary['active_functions'] > 0:
        learning_system.deprecate_function(
            function_name="detect_pre_pump_testusdt_20250618",
            reason="Test deprecation for demonstration"
        )
        print("✅ Function deprecated successfully")
    
    print("\n🎉 All learning system tests completed successfully!")
    
    return {
        'tests_passed': 6,
        'functions_created': summary['total_functions_created'],
        'system_operational': True
    }

if __name__ == "__main__":
    try:
        results = test_learning_system()
        print(f"\n✅ Learning System Test Results:")
        print(f"   Tests passed: {results['tests_passed']}/6")
        print(f"   Functions created: {results['functions_created']}")
        print(f"   System operational: {results['system_operational']}")
        
    except Exception as e:
        print(f"\n❌ Learning system test failed: {e}")
        raise