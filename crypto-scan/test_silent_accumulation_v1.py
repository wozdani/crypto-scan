#!/usr/bin/env python3
"""
Test Silent Accumulation v1 Detector
Validates the enhanced detector functionality with comprehensive test scenarios
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from stages.stage_minus2_1 import detect_silent_accumulation_v1, inflow_avg

def create_test_scenarios():
    """Create comprehensive test scenarios for Silent Accumulation v1"""
    
    # Base market data with small bodies and minimal wicks
    base_market_data = [
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
    flat_rsi = [50.0, 48.0, 52.0, 49.0, 51.0, 47.0, 53.0, 50.0]
    
    scenarios = {
        "perfect_pattern": {
            "market_data": base_market_data,
            "rsi_series": flat_rsi,
            "orderbook": {"supply_vanish": True},
            "vwap_data": {"pinning_count": 8},
            "volume_profile": {"bullish_cluster": True},
            "inflow": 5000,  # 5x average (1000)
            "whale_txs": [
                {"usd": 8000}, {"usd": 9500}, {"usd": 7200}
            ],
            "social_score": 2,
            "expected": True,
            "description": "Perfect Silent Accumulation v1 - all criteria met"
        },
        
        "minimal_buying_pressure": {
            "market_data": base_market_data,
            "rsi_series": flat_rsi,
            "orderbook": None,
            "vwap_data": {"pinning_count": 7},  # Only VWAP pinning
            "volume_profile": None,
            "inflow": 1800,  # Below 2x threshold
            "whale_txs": [{"usd": 9000}, {"usd": 8500}],
            "social_score": 3,
            "expected": True,
            "description": "Minimal buying pressure - only VWAP pinning active"
        },
        
        "no_buying_pressure": {
            "market_data": base_market_data,
            "rsi_series": flat_rsi,
            "orderbook": None,
            "vwap_data": {"pinning_count": 4},  # Below threshold
            "volume_profile": None,
            "inflow": 1500,  # Below 2x threshold
            "whale_txs": [{"usd": 9000}],  # Only 1 tx
            "social_score": 2,
            "expected": False,
            "description": "No buying pressure detected"
        },
        
        "high_social_activity": {
            "market_data": base_market_data,
            "rsi_series": flat_rsi,
            "orderbook": {"supply_vanish": True},
            "vwap_data": {"pinning_count": 8},
            "volume_profile": {"bullish_cluster": True},
            "inflow": 3000,
            "whale_txs": [{"usd": 8000}, {"usd": 9500}],
            "social_score": 15,  # High social activity
            "expected": True,  # Still should trigger due to other signals
            "description": "High social activity but other signals strong"
        },
        
        "volatile_rsi": {
            "market_data": base_market_data,
            "rsi_series": [65.0, 35.0, 70.0, 30.0, 68.0, 32.0, 72.0, 28.0],  # Volatile RSI
            "orderbook": {"supply_vanish": True},
            "vwap_data": {"pinning_count": 8},
            "volume_profile": {"bullish_cluster": True},
            "inflow": 5000,
            "whale_txs": [{"usd": 8000}, {"usd": 9500}],
            "social_score": 2,
            "expected": False,
            "description": "Volatile RSI - should fail"
        },
        
        "large_candle_bodies": {
            "market_data": [
                {"open": 100.0, "close": 105.0, "high": 106.0, "low": 99.0},  # Body: 5, Range: 7, Ratio: 0.71 (71%)
                {"open": 105.0, "close": 99.0, "high": 107.0, "low": 98.0},   # Body: 6, Range: 9, Ratio: 0.67 (67%)
                {"open": 99.0, "close": 104.5, "high": 105.5, "low": 97.5},   # Body: 5.5, Range: 8, Ratio: 0.69 (69%)
                {"open": 104.5, "close": 98.5, "high": 106.0, "low": 97.0},   # Body: 6, Range: 9, Ratio: 0.67 (67%)
                {"open": 98.5, "close": 104.0, "high": 105.0, "low": 97.0},   # Body: 5.5, Range: 8, Ratio: 0.69 (69%)
                {"open": 104.0, "close": 99.5, "high": 106.5, "low": 98.5},   # Body: 4.5, Range: 8, Ratio: 0.56 (56%)
                {"open": 99.5, "close": 104.8, "high": 106.0, "low": 98.0},   # Body: 5.3, Range: 8, Ratio: 0.66 (66%)
                {"open": 104.8, "close": 100.0, "high": 107.0, "low": 99.0}   # Body: 4.8, Range: 8, Ratio: 0.60 (60%)
            ],
            "rsi_series": flat_rsi,
            "orderbook": {"supply_vanish": True},
            "vwap_data": {"pinning_count": 8},
            "volume_profile": {"bullish_cluster": True},
            "inflow": 5000,
            "whale_txs": [{"usd": 8000}, {"usd": 9500}],
            "social_score": 2,
            "expected": False,
            "description": "Large candle bodies - should fail"
        },
        
        "large_wicks": {
            "market_data": [
                {"open": 100.0, "close": 100.1, "high": 102.0, "low": 98.0},   # Large wicks: 1.9 upper, 2.0 lower vs 4.0 range = 47.5%
                {"open": 100.1, "close": 100.0, "high": 102.5, "low": 97.5},   # Large wicks: 2.4 upper, 2.5 lower vs 5.0 range = 49%
                {"open": 100.0, "close": 100.15, "high": 103.0, "low": 97.0},  # Large wicks: 2.85 upper, 3.0 lower vs 6.0 range = 49%
                {"open": 100.15, "close": 100.05, "high": 103.5, "low": 96.5}, # Large wicks: 3.35 upper, 3.55 lower vs 7.0 range = 49%
                {"open": 100.05, "close": 100.2, "high": 104.0, "low": 96.0},  # Large wicks: 3.8 upper, 4.05 lower vs 8.0 range = 49%
                {"open": 100.2, "close": 100.1, "high": 104.5, "low": 95.5},   # Large wicks: 4.3 upper, 4.6 lower vs 9.0 range = 49%
                {"open": 100.1, "close": 100.25, "high": 105.0, "low": 95.0},  # Large wicks: 4.75 upper, 5.1 lower vs 10.0 range = 49%
                {"open": 100.25, "close": 100.15, "high": 105.5, "low": 94.5}  # Large wicks: 5.25 upper, 5.65 lower vs 11.0 range = 49%
            ],
            "rsi_series": flat_rsi,
            "orderbook": {"supply_vanish": True},
            "vwap_data": {"pinning_count": 8},
            "volume_profile": {"bullish_cluster": True},
            "inflow": 5000,
            "whale_txs": [{"usd": 8000}, {"usd": 9500}],
            "social_score": 2,
            "expected": False,
            "description": "Large wicks - should fail"
        }
    }
    
    return scenarios

def run_tests():
    """Run all Silent Accumulation v1 detector tests"""
    print("🧪 Silent Accumulation v1 Detector Test Suite")
    print("=" * 60)
    
    scenarios = create_test_scenarios()
    passed = 0
    total = len(scenarios)
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n📋 Test: {scenario['description']}")
        
        result = detect_silent_accumulation_v1(
            symbol=f"TEST_{scenario_name.upper()}",
            market_data=scenario["market_data"],
            rsi_series=scenario["rsi_series"],
            orderbook=scenario.get("orderbook"),
            vwap_data=scenario.get("vwap_data"),
            volume_profile=scenario.get("volume_profile"),
            inflow=scenario.get("inflow", 0),
            whale_txs=scenario.get("whale_txs", []),
            social_score=scenario.get("social_score", 0)
        )
        
        expected = scenario["expected"]
        
        if result == expected:
            print(f"✅ PASSED: Expected {expected}, got {result}")
            passed += 1
        else:
            print(f"❌ FAILED: Expected {expected}, got {result}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Silent Accumulation v1 detector is working correctly.")
    else:
        print(f"⚠️ {total - passed} test(s) failed. Please review the detector logic.")
    
    return passed == total

def test_inflow_avg_function():
    """Test the inflow_avg helper function"""
    print("\n🔧 Testing inflow_avg function")
    print("=" * 40)
    
    # Test with non-existent symbol
    avg1 = inflow_avg("NONEXISTENT")
    print(f"📊 Default average for non-existent symbol: {avg1}")
    
    # Test basic functionality
    avg2 = inflow_avg("BTCUSDT")
    print(f"📊 Average for BTCUSDT: {avg2}")
    
    if avg1 == 1000.0 and avg2 >= 0:
        print("✅ inflow_avg function working correctly")
        return True
    else:
        print("❌ inflow_avg function has issues")
        return False

def test_integration():
    """Test integration with main system"""
    print("\n🔗 Testing Integration with Main System")
    print("=" * 50)
    
    # Test with perfect pattern
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
    
    orderbook = {"supply_vanish": True}
    vwap_data = {"pinning_count": 8}
    volume_profile = {"bullish_cluster": True}
    inflow = 5000
    whale_txs = [{"usd": 8000}, {"usd": 9500}, {"usd": 7200}]
    social_score = 2
    
    result = detect_silent_accumulation_v1(
        "INTEGRATION_TEST_V1",
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
        print("✅ Integration test passed - detector triggered successfully")
        
        # Check if alert cache file was created
        cache_file = "data/silent_accumulation_v1_alerts.json"
        if os.path.exists(cache_file):
            print("✅ Alert cache file created successfully")
            import json
            with open(cache_file, 'r') as f:
                alerts = json.load(f)
            print(f"📋 Cache contains {len(alerts)} alert(s)")
        else:
            print("⚠️ Alert cache file not found (may be using main system functions)")
        
        return True
    else:
        print("❌ Integration test failed - detector did not trigger")
        return False

def main():
    """Main test function"""
    print("🚀 Silent Accumulation v1 Detector Test Suite")
    print("=" * 60)
    
    # Run all tests
    basic_tests_passed = run_tests()
    inflow_test_passed = test_inflow_avg_function()
    integration_test_passed = test_integration()
    
    print("\n" + "=" * 60)
    print("🎯 Test suite completed!")
    
    if basic_tests_passed and inflow_test_passed and integration_test_passed:
        print("✅ All tests passed! Silent Accumulation v1 detector is ready for production use.")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
    
    print("\n📌 Silent Accumulation v1 Key Features:")
    print("   • Enhanced pattern recognition with 8+ criteria")
    print("   • Buying pressure detection (VWAP, orderbook, inflow, volume)")
    print("   • Micro whale activity analysis")
    print("   • Social activity filtering")
    print("   • Independent alert and GPT analysis system")
    print("   • Minimum 5 signals + buying pressure requirement")

if __name__ == "__main__":
    main()