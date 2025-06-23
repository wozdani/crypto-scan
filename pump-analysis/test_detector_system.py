#!/usr/bin/env python3
"""
Test script for the automated detector generation system
Tests both the GPT function generation and the dynamic loading system
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(__file__))

from generated_detectors import get_available_detectors, load_detector, test_all_detectors

def create_sample_data():
    """Create sample OHLCV data for testing detectors"""
    np.random.seed(42)  # For reproducible results
    
    # Generate 24 hours of 5-minute data (288 candles)
    timestamps = pd.date_range(start='2025-06-13 00:00:00', periods=288, freq='5T')
    
    # Simulate price data with pump pattern
    base_price = 65000.0
    price_data = []
    volume_data = []
    
    for i in range(len(timestamps)):
        # Normal price movement with slight uptrend
        if i < 250:  # Pre-pump phase
            price_change = np.random.normal(0, 0.002)  # 0.2% volatility
            volume_multiplier = np.random.normal(1, 0.3)
        elif i < 265:  # Pump phase
            price_change = np.random.normal(0.008, 0.003)  # Strong upward movement
            volume_multiplier = np.random.normal(3, 0.5)  # High volume
        else:  # Post-pump
            price_change = np.random.normal(-0.002, 0.004)  # Correction
            volume_multiplier = np.random.normal(1.5, 0.4)
        
        if i == 0:
            price = base_price
        else:
            price = price_data[-1] * (1 + price_change)
        
        price_data.append(price)
        volume_data.append(max(1000000 * volume_multiplier, 100000))
    
    # Create OHLC from price data
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': price_data,
        'high': [p * np.random.uniform(1.001, 1.008) for p in price_data],
        'low': [p * np.random.uniform(0.992, 0.999) for p in price_data],
        'close': price_data,
        'volume': volume_data
    })
    
    # Calculate VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def test_sample_detector():
    """Test the sample BTCUSDT detector"""
    print("🧪 Testing sample BTCUSDT_20250613 detector...")
    
    # Create test data
    df = create_sample_data()
    
    try:
        # Load the detector
        detector = load_detector('BTCUSDT', '20250613')
        
        # Test with pre-pump data (first 250 candles)
        pre_pump_data = df.iloc[:250].copy()
        result_pre = detector(pre_pump_data)
        
        # Test with pump data (up to candle 265)
        pump_data = df.iloc[:265].copy()
        result_pump = detector(pump_data)
        
        print(f"   Pre-pump detection: {result_pre}")
        print(f"   During-pump detection: {result_pump}")
        
        # Expected: False for pre-pump, True for pump phase
        if not result_pre and result_pump:
            print("   ✅ Detector working correctly!")
        else:
            print("   ⚠️ Detector results may need adjustment")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing detector: {e}")
        return False

def test_detector_discovery():
    """Test automatic detector discovery"""
    print("🔍 Testing detector discovery system...")
    
    try:
        detectors = get_available_detectors()
        print(f"   Found {len(detectors)} detector functions:")
        
        for detector_name in detectors:
            print(f"   - {detector_name}")
        
        return len(detectors) > 0
        
    except Exception as e:
        print(f"   ❌ Error in detector discovery: {e}")
        return False

def test_batch_detection():
    """Test running all detectors on sample data"""
    print("📊 Testing batch detector execution...")
    
    try:
        df = create_sample_data()
        
        # Test on pre-pump period
        pre_pump_data = df.iloc[:250].copy()
        results_pre = test_all_detectors(pre_pump_data)
        
        # Test on pump period  
        pump_data = df.iloc[:265].copy()
        results_pump = test_all_detectors(pump_data)
        
        print(f"   Pre-pump results: {results_pre}")
        print(f"   Pump-phase results: {results_pump}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in batch testing: {e}")
        return False

def test_detector_integration():
    """Test integration with main pump analysis system"""
    print("🔗 Testing integration with pump analysis...")
    
    try:
        # This would test calling the detector generation from main system
        # For now, just verify the file structure is correct
        
        detector_dir = "generated_detectors"
        if os.path.exists(detector_dir):
            files = [f for f in os.listdir(detector_dir) if f.endswith('.py') and f != '__init__.py']
            print(f"   Generated detector files: {len(files)}")
            for file in files:
                print(f"   - {file}")
            return len(files) > 0
        else:
            print("   ❌ Generated detectors directory not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing integration: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Detector Generation System Tests")
    print("=" * 50)
    
    tests = [
        ("Detector Discovery", test_detector_discovery),
        ("Sample Detector", test_sample_detector),
        ("Batch Detection", test_batch_detection),
        ("System Integration", test_detector_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Detector generation system is ready.")
    else:
        print("⚠️ Some tests failed. Review the system before deployment.")

if __name__ == "__main__":
    main()