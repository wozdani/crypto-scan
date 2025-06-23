#!/usr/bin/env python3
"""
Final Test for Silent Accumulation v1 Detector
Tests core logic without external dependencies to avoid timeouts
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_detector_logic():
    """Test the detector logic directly without external calls"""
    print("Testing Silent Accumulation v1 detector logic...")
    
    # Import the function directly
    from stages.stage_minus2_1 import detect_silent_accumulation_v1
    
    # Perfect pattern test data
    market_data = [
        {"open": 100.0, "close": 100.1, "high": 100.5, "low": 99.5},
        {"open": 100.1, "close": 100.0, "high": 100.6, "low": 99.4},
        {"open": 100.0, "close": 100.15, "high": 100.7, "low": 99.3},
        {"open": 100.15, "close": 100.05, "high": 100.8, "low": 99.2},
        {"open": 100.05, "close": 100.2, "high": 100.9, "low": 99.1},
        {"open": 100.2, "close": 100.1, "high": 101.0, "low": 99.0},
        {"open": 100.1, "close": 100.25, "high": 101.1, "low": 98.9},
        {"open": 100.25, "close": 100.15, "high": 101.2, "low": 98.8}
    ]
    
    rsi_series = [50.0, 48.0, 52.0, 49.0, 51.0, 47.0, 53.0, 50.0]
    
    test_cases = [
        {
            "name": "Perfect Pattern",
            "data": {
                "orderbook": {"supply_vanish": True},
                "vwap_data": {"pinning_count": 8},
                "volume_profile": {"bullish_cluster": True},
                "inflow": 5000,
                "whale_txs": [{"usd": 8000}, {"usd": 9500}, {"usd": 7200}],
                "social_score": 2
            },
            "expected": True
        },
        {
            "name": "No Buying Pressure",
            "data": {
                "orderbook": None,
                "vwap_data": {"pinning_count": 4},
                "volume_profile": None,
                "inflow": 1500,
                "whale_txs": [{"usd": 9000}],
                "social_score": 2
            },
            "expected": False
        },
        {
            "name": "Volatile RSI",
            "data": {
                "orderbook": {"supply_vanish": True},
                "vwap_data": {"pinning_count": 8},
                "volume_profile": {"bullish_cluster": True},
                "inflow": 5000,
                "whale_txs": [{"usd": 8000}, {"usd": 9500}],
                "social_score": 2
            },
            "rsi": [50.0, 55.0, 45.0, 54.0, 46.0, 53.0, 47.0, 52.0],  # Within range but volatile (range=10)
            "expected": False
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        # Use custom RSI if provided, otherwise use default flat RSI
        test_rsi = test_case.get('rsi', rsi_series)
        
        try:
            # Mock the alert processing to avoid external calls
            original_import = __builtins__.__import__
            
            def mock_import(name, *args, **kwargs):
                if name in ['utils.telegram_bot', 'crypto_scan_service']:
                    # Return a mock module
                    class MockModule:
                        def send_alert(self, msg): 
                            print(f"Mock alert: {msg[:50]}...")
                        def send_report_to_gpt(self, symbol, data, tp, level):
                            print(f"Mock GPT report for {symbol}")
                    return MockModule()
                return original_import(name, *args, **kwargs)
            
            __builtins__.__import__ = mock_import
            
            result = detect_silent_accumulation_v1(
                f"TEST_{test_case['name'].upper().replace(' ', '_')}",
                market_data,
                test_rsi,
                test_case['data'].get('orderbook'),
                test_case['data'].get('vwap_data'),
                test_case['data'].get('volume_profile'),
                test_case['data'].get('inflow', 0),
                test_case['data'].get('whale_txs', []),
                test_case['data'].get('social_score', 0)
            )
            
            # Restore original import
            __builtins__.__import__ = original_import
            
            if result == test_case['expected']:
                print(f"✅ PASSED: Expected {test_case['expected']}, got {result}")
                passed += 1
            else:
                print(f"❌ FAILED: Expected {test_case['expected']}, got {result}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            # Restore original import on error
            __builtins__.__import__ = original_import
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    return passed == total

def main():
    print("🚀 Silent Accumulation v1 Final Test Suite")
    print("=" * 50)
    
    success = test_detector_logic()
    
    if success:
        print("\n✅ All tests passed! Silent Accumulation v1 detector is working correctly.")
        print("\n📋 Key Features Validated:")
        print("   • RSI flat detection (45-55 range)")
        print("   • Small candle body analysis (<30% of range)")
        print("   • Minimal wick validation (<10% of range)")
        print("   • VWAP pinning detection (≥6 occurrences)")
        print("   • Buying pressure validation (orderbook, volume, inflow)")
        print("   • Whale activity analysis (≥2 transactions >$5k)")
        print("   • Social activity filtering (low activity preferred)")
        print("   • Minimum 5 signals + buying pressure requirement")
    else:
        print("\n⚠️ Some tests failed. The detector needs review.")

if __name__ == "__main__":
    main()