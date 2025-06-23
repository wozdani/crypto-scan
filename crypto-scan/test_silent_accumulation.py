#!/usr/bin/env python3
"""
Test Silent Accumulation Detector
Validates the new detector functionality with synthetic test data
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from stages.stage_minus2_1 import detect_silent_accumulation

def create_test_data():
    """Create test data that matches Silent Accumulation criteria"""
    
    # Test case 1: Perfect Silent Accumulation pattern
    perfect_pattern = {
        "market_data": [
            {"open": 100.0, "close": 100.1, "high": 100.5, "low": 99.5},  # body: 0.1, range: 1.0, ratio: 0.1 (10%)
            {"open": 100.1, "close": 100.0, "high": 100.6, "low": 99.4},  # body: 0.1, range: 1.2, ratio: 0.08 (8%)
            {"open": 100.0, "close": 100.15, "high": 100.7, "low": 99.3}, # body: 0.15, range: 1.4, ratio: 0.11 (11%)
            {"open": 100.15, "close": 100.05, "high": 100.8, "low": 99.2}, # body: 0.1, range: 1.6, ratio: 0.06 (6%)
            {"open": 100.05, "close": 100.2, "high": 100.9, "low": 99.1},  # body: 0.15, range: 1.8, ratio: 0.08 (8%)
            {"open": 100.2, "close": 100.1, "high": 101.0, "low": 99.0},   # body: 0.1, range: 2.0, ratio: 0.05 (5%)
            {"open": 100.1, "close": 100.25, "high": 101.1, "low": 98.9},  # body: 0.15, range: 2.2, ratio: 0.07 (7%)
            {"open": 100.25, "close": 100.15, "high": 101.2, "low": 98.8}  # body: 0.1, range: 2.4, ratio: 0.04 (4%)
        ],
        "rsi_series": [50.0, 48.0, 52.0, 49.0, 51.0, 47.0, 53.0, 50.0],
        "expected": True,
        "description": "Perfect flat RSI (45-55), small bodies, minimal wicks"
    }
    
    # Test case 2: RSI too high - should fail
    high_rsi_pattern = {
        "market_data": perfect_pattern["market_data"],
        "rsi_series": [65.0, 68.0, 70.0, 72.0, 75.0, 73.0, 71.0, 69.0],
        "expected": False,
        "description": "RSI too high (above 55 range)"
    }
    
    # Test case 3: Large candle bodies - should fail
    large_bodies_pattern = {
        "market_data": [
            {"open": 100.0, "close": 105.0, "high": 106.0, "low": 99.0},  # Large body
            {"open": 105.0, "close": 102.0, "high": 105.5, "low": 101.0},  # Large body
            {"open": 102.0, "close": 108.0, "high": 109.0, "low": 101.0},  # Large body
            {"open": 108.0, "close": 104.0, "high": 109.0, "low": 103.0},  # Large body
            {"open": 104.0, "close": 110.0, "high": 111.0, "low": 103.0},  # Large body
            {"open": 110.0, "close": 106.0, "high": 111.0, "low": 105.0},  # Large body
            {"open": 106.0, "close": 112.0, "high": 113.0, "low": 105.0},  # Large body
            {"open": 112.0, "close": 108.0, "high": 113.0, "low": 107.0}   # Large body
        ],
        "rsi_series": [50.0, 48.0, 52.0, 49.0, 51.0, 47.0, 53.0, 50.0],
        "expected": False,
        "description": "Large candle bodies (>30% of range)"
    }
    
    # Test case 4: Large wicks - should fail
    large_wicks_pattern = {
        "market_data": [
            {"open": 100.0, "close": 100.2, "high": 115.0, "low": 85.0},  # Large wicks
            {"open": 100.2, "close": 100.1, "high": 120.0, "low": 80.0},  # Large wicks
            {"open": 100.1, "close": 100.3, "high": 118.0, "low": 82.0},  # Large wicks
            {"open": 100.3, "close": 100.0, "high": 125.0, "low": 75.0},  # Large wicks
            {"open": 100.0, "close": 100.4, "high": 122.0, "low": 78.0},  # Large wicks
            {"open": 100.4, "close": 100.1, "high": 130.0, "low": 70.0},  # Large wicks
            {"open": 100.1, "close": 100.2, "high": 128.0, "low": 72.0},  # Large wicks
            {"open": 100.2, "close": 100.0, "high": 135.0, "low": 65.0}   # Large wicks
        ],
        "rsi_series": [50.0, 48.0, 52.0, 49.0, 51.0, 47.0, 53.0, 50.0],
        "expected": False,
        "description": "Large upper/lower wicks (>10% of price)"
    }
    
    # Test case 5: Insufficient data - should fail
    insufficient_data = {
        "market_data": [
            {"open": 100.0, "close": 100.2, "high": 100.5, "low": 99.8},
            {"open": 100.2, "close": 100.1, "high": 100.4, "low": 100.0}
        ],
        "rsi_series": [50.0, 48.0],
        "expected": False,
        "description": "Insufficient data (less than 8 candles)"
    }
    
    return [
        perfect_pattern,
        high_rsi_pattern,
        large_bodies_pattern,
        large_wicks_pattern,
        insufficient_data
    ]

def run_tests():
    """Run all Silent Accumulation detector tests"""
    
    print("🧪 Testing Silent Accumulation Detector")
    print("=" * 50)
    
    test_cases = create_test_data()
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['description']}")
        
        try:
            result = detect_silent_accumulation(
                symbol=f"TEST{i}USDT",
                market_data=test_case["market_data"],
                rsi_series=test_case["rsi_series"]
            )
            
            expected = test_case["expected"]
            
            if result == expected:
                print(f"✅ PASSED: Expected {expected}, got {result}")
                passed += 1
            else:
                print(f"❌ FAILED: Expected {expected}, got {result}")
                
        except Exception as e:
            print(f"💥 ERROR: {str(e)}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Silent Accumulation detector is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the detector logic.")
        return False

def test_integration():
    """Test integration with main detection system"""
    
    print("\n🔗 Testing Integration with Main System")
    print("=" * 50)
    
    # Check if data directory exists for alert caching
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"✅ Created data directory: {data_dir}")
    else:
        print(f"✅ Data directory exists: {data_dir}")
    
    # Test cache file creation
    cache_file = os.path.join(data_dir, "silent_accumulation_alerts.json")
    print(f"📁 Alert cache file location: {cache_file}")
    
    # Test with perfect pattern to trigger alert cache
    perfect_data = create_test_data()[0]
    
    try:
        result = detect_silent_accumulation(
            symbol="INTEGRATION_TEST_USDT",
            market_data=perfect_data["market_data"],
            rsi_series=perfect_data["rsi_series"]
        )
        
        if result:
            print("✅ Integration test passed - detector triggered successfully")
            
            # Check if cache file was created
            if os.path.exists(cache_file):
                print("✅ Alert cache file created successfully")
                
                # Read and display cache content
                import json
                with open(cache_file, 'r') as f:
                    alerts = json.load(f)
                
                print(f"📋 Cache contains {len(alerts)} alert(s)")
                if alerts:
                    latest_alert = alerts[-1]
                    print(f"🔍 Latest alert: {latest_alert['symbol']} (PPWCS: {latest_alert['ppwcs_score']})")
            else:
                print("⚠️ Alert cache file not created")
        else:
            print("❌ Integration test failed - detector did not trigger")
            
    except Exception as e:
        print(f"💥 Integration test error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Silent Accumulation Detector Test Suite")
    print("=" * 60)
    
    # Run unit tests
    unit_tests_passed = run_tests()
    
    # Run integration tests
    test_integration()
    
    print("\n" + "=" * 60)
    if unit_tests_passed:
        print("🎯 Test suite completed successfully!")
        print("Silent Accumulation detector is ready for production use.")
    else:
        print("🔧 Test suite found issues that need fixing.")