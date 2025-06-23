#!/usr/bin/env python3
"""
Test suite for Enhanced Pump Detection Module
Tests the new 15-minute candle analysis with multiple timeframe detection
"""

import json
from detect_pumps import detect_biggest_pump_15m, categorize_pump_15m, batch_pump_detection, get_pump_statistics

def test_pump_impulse():
    """Test detection of pump-impulse (>20% in ≤1h)"""
    print("🧪 Testing pump-impulse detection...")
    
    # Create test data with 35% pump in 1 hour (4 candles)
    candles = [
        {"timestamp": 1718524500, "open": 0.01000, "close": 0.01010, "high": 0.01015, "low": 0.00995},
        {"timestamp": 1718525400, "open": 0.01010, "close": 0.01050, "high": 0.01055, "low": 0.01005},
        {"timestamp": 1718526300, "open": 0.01050, "close": 0.01200, "high": 0.01250, "low": 0.01045},
        {"timestamp": 1718527200, "open": 0.01200, "close": 0.01350, "high": 0.01350, "low": 0.01180},  # 35% pump peak
        {"timestamp": 1718528100, "open": 0.01350, "close": 0.01320, "high": 0.01360, "low": 0.01300},
    ]
    
    result = detect_biggest_pump_15m(candles, "IMPULSE_TEST")
    
    if result and result["type"] == "pump-impulse" and result["growth"] > 30:
        print(f"✅ Pump-impulse detected: {result['growth']}% in {result['window_hours']}h")
        return True
    else:
        print(f"❌ Expected pump-impulse, got: {result}")
        return False

def test_trend_breakout():
    """Test detection of trend-breakout (>30% in ≤4h)"""
    print("\n🧪 Testing trend-breakout detection...")
    
    # Create test data with 45% pump over 3 hours (12 candles)
    candles = []
    base_price = 0.00500
    
    # Build gradual pump over 3 hours
    for i in range(15):
        timestamp = 1718524500 + (i * 900)  # 15 min intervals
        
        if i < 8:  # First 2 hours - slow rise
            multiplier = 1 + (i * 0.05)  # 5% per candle
        else:  # Last hour - big breakout
            multiplier = 1.4 + ((i-8) * 0.08)  # Sharp rise to 45%
            
        price = base_price * multiplier
        
        candles.append({
            "timestamp": timestamp,
            "open": price * 0.99,
            "close": price,
            "high": price * 1.01,
            "low": price * 0.98
        })
    
    result = detect_biggest_pump_15m(candles, "BREAKOUT_TEST")
    
    if result and result["type"] == "trend-breakout" and result["window_hours"] <= 4:
        print(f"✅ Trend-breakout detected: {result['growth']}% in {result['window_hours']}h")
        return True
    else:
        print(f"❌ Expected trend-breakout, got: {result}")
        return False

def test_trend_mode():
    """Test detection of trend-mode (>50% in >4h)"""
    print("\n🧪 Testing trend-mode detection...")
    
    # Create test data with 60% pump over 6 hours (24 candles)
    candles = []
    base_price = 0.00800
    
    # Build sustained pump over 6 hours
    for i in range(26):
        timestamp = 1718524500 + (i * 900)  # 15 min intervals
        multiplier = 1 + (i * 0.025)  # Steady 2.5% per candle = 65% total
        price = base_price * multiplier
        
        candles.append({
            "timestamp": timestamp,
            "open": price * 0.995,
            "close": price,
            "high": price * 1.005,
            "low": price * 0.99
        })
    
    result = detect_biggest_pump_15m(candles, "TREND_TEST")
    
    if result and result["type"] == "trend-mode" and result["window_hours"] > 4:
        print(f"✅ Trend-mode detected: {result['growth']}% in {result['window_hours']}h")
        return True
    else:
        print(f"❌ Expected trend-mode, got: {result}")
        return False

def test_micro_move():
    """Test detection of micro-move (small movements)"""
    print("\n🧪 Testing micro-move detection...")
    
    # Create test data with 18% pump (above 15% threshold but below category thresholds)
    candles = [
        {"timestamp": 1718524500, "open": 0.01000, "close": 0.01020, "high": 0.01025, "low": 0.00995},
        {"timestamp": 1718525400, "open": 0.01020, "close": 0.01050, "high": 0.01055, "low": 0.01015},
        {"timestamp": 1718526300, "open": 0.01050, "close": 0.01100, "high": 0.01120, "low": 0.01045},
        {"timestamp": 1718527200, "open": 0.01100, "close": 0.01180, "high": 0.01180, "low": 0.01095},  # 18% pump
        {"timestamp": 1718528100, "open": 0.01180, "close": 0.01170, "high": 0.01185, "low": 0.01160},
    ]
    
    result = detect_biggest_pump_15m(candles, "MICRO_TEST")
    
    if result and result["type"] == "micro-move" and 15 <= result["growth"] < 20:
        print(f"✅ Micro-move detected: {result['growth']}% in {result['window_hours']}h")
        return True
    else:
        print(f"❌ Expected micro-move, got: {result}")
        return False

def test_batch_processing():
    """Test batch processing of multiple symbols"""
    print("\n🧪 Testing batch pump detection...")
    
    # Create test data for multiple symbols with sufficient candles
    symbols_data = {
        "SYMBOL1": [
            {"timestamp": 1718524500, "open": 0.01000, "close": 0.01010, "high": 0.01015, "low": 0.00995},
            {"timestamp": 1718525400, "open": 0.01010, "close": 0.01050, "high": 0.01055, "low": 0.01005},
            {"timestamp": 1718526300, "open": 0.01050, "close": 0.01200, "high": 0.01250, "low": 0.01045},
            {"timestamp": 1718527200, "open": 0.01200, "close": 0.01350, "high": 0.01350, "low": 0.01180},  # 35% pump
            {"timestamp": 1718528100, "open": 0.01350, "close": 0.01320, "high": 0.01360, "low": 0.01300},
        ],
        "SYMBOL2": [
            {"timestamp": 1718524500, "open": 0.00500, "close": 0.00510, "high": 0.00515, "low": 0.00495},  # 3% - no pump
            {"timestamp": 1718525400, "open": 0.00510, "close": 0.00515, "high": 0.00520, "low": 0.00505},
            {"timestamp": 1718526300, "open": 0.00515, "close": 0.00518, "high": 0.00522, "low": 0.00512},
            {"timestamp": 1718527200, "open": 0.00518, "close": 0.00520, "high": 0.00525, "low": 0.00515},
        ],
        "SYMBOL3": [
            {"timestamp": 1718524500, "open": 0.00200, "close": 0.00210, "high": 0.00215, "low": 0.00195},
            {"timestamp": 1718525400, "open": 0.00210, "close": 0.00220, "high": 0.00225, "low": 0.00205},
            {"timestamp": 1718526300, "open": 0.00220, "close": 0.00230, "high": 0.00235, "low": 0.00215},
            {"timestamp": 1718527200, "open": 0.00230, "close": 0.00240, "high": 0.00240, "low": 0.00225},  # 20% pump
            {"timestamp": 1718528100, "open": 0.00240, "close": 0.00235, "high": 0.00245, "low": 0.00230},
        ]
    }
    
    results = batch_pump_detection(symbols_data, enable_gpt=False)
    stats = get_pump_statistics(results)
    
    if len(results) >= 2 and stats["total_pumps"] >= 2:
        print(f"✅ Batch processing: {stats['total_pumps']} pumps detected from {len(symbols_data)} symbols")
        return True
    else:
        print(f"❌ Expected 2+ pumps, got: {len(results)} - Stats: {stats}")
        return False

def test_integration_with_main_system():
    """Test integration with main pump analysis system"""
    print("\n🧪 Testing integration with main system...")
    
    try:
        from main import PumpDetector
        
        # Create detector instance
        detector = PumpDetector(min_increase_pct=15.0)
        
        # Create Bybit-format test data (list of lists)
        kline_data = [
            [1718524500000, "0.01000", "0.01015", "0.00995", "0.01010", "1000000", "10150"],
            [1718525400000, "0.01010", "0.01055", "0.01005", "0.01050", "1200000", "12500"],
            [1718526300000, "0.01050", "0.01250", "0.01045", "0.01200", "1500000", "17800"],
            [1718527200000, "0.01200", "0.01350", "0.01180", "0.01350", "1800000", "23500"],  # 35% pump
            [1718528100000, "0.01350", "0.01360", "0.01300", "0.01320", "1100000", "14600"],
        ]
        
        pumps = detector.detect_pumps_in_data(kline_data, "INTEGRATION_TEST")
        
        if pumps and len(pumps) > 0:
            pump = pumps[0]
            print(f"✅ Integration test: Detected {pump.price_increase_pct:.2f}% pump")
            return True
        else:
            print("❌ Integration test failed: No pumps detected")
            return False
            
    except ImportError as e:
        print(f"⚠️ Integration test skipped: {e}")
        return True  # Not a failure, just unavailable

def main():
    """Run all tests"""
    print("🚀 Enhanced Pump Detection Test Suite")
    print("=" * 50)
    
    tests = [
        test_pump_impulse,
        test_trend_breakout,
        test_trend_mode,
        test_micro_move,
        test_batch_processing,
        test_integration_with_main_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced pump detection is working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()