#!/usr/bin/env python3
"""
Simple Test for Silent Accumulation v1 Detector
Tests the core detection logic without external dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from stages.stage_minus2_1 import detect_silent_accumulation_v1

def test_perfect_pattern():
    """Test perfect Silent Accumulation v1 pattern"""
    print("🧪 Testing perfect Silent Accumulation v1 pattern...")
    
    # Base market data with small bodies and minimal wicks
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
    
    # Flat RSI in 45-55 range
    rsi_series = [50.0, 48.0, 52.0, 49.0, 51.0, 47.0, 53.0, 50.0]
    
    # All buying pressure signals active
    orderbook = {"supply_vanish": True}
    vwap_data = {"pinning_count": 8}
    volume_profile = {"bullish_cluster": True}
    inflow = 5000  # 5x average (1000)
    whale_txs = [{"usd": 8000}, {"usd": 9500}, {"usd": 7200}]
    social_score = 2
    
    result = detect_silent_accumulation_v1(
        "TEST_PERFECT",
        market_data,
        rsi_series,
        orderbook,
        vwap_data,
        volume_profile,
        inflow,
        whale_txs,
        social_score
    )
    
    if result:
        print("✅ Perfect pattern test PASSED")
        return True
    else:
        print("❌ Perfect pattern test FAILED")
        return False

def test_no_buying_pressure():
    """Test pattern without buying pressure"""
    print("🧪 Testing pattern without buying pressure...")
    
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
    
    # No buying pressure signals
    orderbook = None
    vwap_data = {"pinning_count": 4}  # Below threshold
    volume_profile = None
    inflow = 1500  # Below 2x threshold
    whale_txs = [{"usd": 9000}]  # Only 1 tx
    social_score = 2
    
    result = detect_silent_accumulation_v1(
        "TEST_NO_PRESSURE",
        market_data,
        rsi_series,
        orderbook,
        vwap_data,
        volume_profile,
        inflow,
        whale_txs,
        social_score
    )
    
    if not result:
        print("✅ No buying pressure test PASSED")
        return True
    else:
        print("❌ No buying pressure test FAILED - should not trigger")
        return False

def test_volatile_rsi():
    """Test pattern with volatile RSI"""
    print("🧪 Testing pattern with volatile RSI...")
    
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
    
    # Volatile RSI - should fail
    rsi_series = [65.0, 35.0, 70.0, 30.0, 68.0, 32.0, 72.0, 28.0]
    
    # All other signals perfect
    orderbook = {"supply_vanish": True}
    vwap_data = {"pinning_count": 8}
    volume_profile = {"bullish_cluster": True}
    inflow = 5000
    whale_txs = [{"usd": 8000}, {"usd": 9500}]
    social_score = 2
    
    result = detect_silent_accumulation_v1(
        "TEST_VOLATILE_RSI",
        market_data,
        rsi_series,
        orderbook,
        vwap_data,
        volume_profile,
        inflow,
        whale_txs,
        social_score
    )
    
    if not result:
        print("✅ Volatile RSI test PASSED")
        return True
    else:
        print("❌ Volatile RSI test FAILED - should not trigger")
        return False

def test_large_bodies():
    """Test pattern with large candle bodies"""
    print("🧪 Testing pattern with large candle bodies...")
    
    # Large body candles - should fail
    market_data = [
        {"open": 100.0, "close": 105.0, "high": 106.0, "low": 99.0},  # Body: 5, Range: 7, Ratio: 0.71 (71%)
        {"open": 105.0, "close": 99.0, "high": 107.0, "low": 98.0},   # Body: 6, Range: 9, Ratio: 0.67 (67%)
        {"open": 99.0, "close": 104.5, "high": 105.5, "low": 97.5},   # Body: 5.5, Range: 8, Ratio: 0.69 (69%)
        {"open": 104.5, "close": 98.5, "high": 106.0, "low": 97.0},   # Body: 6, Range: 9, Ratio: 0.67 (67%)
        {"open": 98.5, "close": 104.0, "high": 105.0, "low": 97.0},   # Body: 5.5, Range: 8, Ratio: 0.69 (69%)
        {"open": 104.0, "close": 99.5, "high": 106.5, "low": 98.5},   # Body: 4.5, Range: 8, Ratio: 0.56 (56%)
        {"open": 99.5, "close": 104.8, "high": 106.0, "low": 98.0},   # Body: 5.3, Range: 8, Ratio: 0.66 (66%)
        {"open": 104.8, "close": 100.0, "high": 107.0, "low": 99.0}   # Body: 4.8, Range: 8, Ratio: 0.60 (60%)
    ]
    
    rsi_series = [50.0, 48.0, 52.0, 49.0, 51.0, 47.0, 53.0, 50.0]
    
    # All other signals perfect
    orderbook = {"supply_vanish": True}
    vwap_data = {"pinning_count": 8}
    volume_profile = {"bullish_cluster": True}
    inflow = 5000
    whale_txs = [{"usd": 8000}, {"usd": 9500}]
    social_score = 2
    
    result = detect_silent_accumulation_v1(
        "TEST_LARGE_BODIES",
        market_data,
        rsi_series,
        orderbook,
        vwap_data,
        volume_profile,
        inflow,
        whale_txs,
        social_score
    )
    
    if not result:
        print("✅ Large bodies test PASSED")
        return True
    else:
        print("❌ Large bodies test FAILED - should not trigger")
        return False

def test_minimal_buying_pressure():
    """Test pattern with minimal buying pressure"""
    print("🧪 Testing pattern with minimal buying pressure...")
    
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
    
    # Only VWAP pinning active (minimal buying pressure)
    orderbook = None
    vwap_data = {"pinning_count": 7}  # Just above threshold
    volume_profile = None
    inflow = 1800  # Below 2x threshold
    whale_txs = [{"usd": 9000}, {"usd": 8500}]  # 2 whale txs
    social_score = 3
    
    result = detect_silent_accumulation_v1(
        "TEST_MINIMAL_PRESSURE",
        market_data,
        rsi_series,
        orderbook,
        vwap_data,
        volume_profile,
        inflow,
        whale_txs,
        social_score
    )
    
    if result:
        print("✅ Minimal buying pressure test PASSED")
        return True
    else:
        print("❌ Minimal buying pressure test FAILED - should trigger")
        return False

def main():
    """Run all simplified tests"""
    print("🚀 Silent Accumulation v1 Simple Test Suite")
    print("=" * 50)
    
    tests = [
        test_perfect_pattern,
        test_minimal_buying_pressure,
        test_no_buying_pressure,
        test_volatile_rsi,
        test_large_bodies
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
        print("✅ All tests passed! Silent Accumulation v1 detector is working correctly.")
    else:
        print(f"⚠️ {total - passed} test(s) failed.")
    
    return passed == total

if __name__ == "__main__":
    main()