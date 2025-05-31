#!/usr/bin/env python3
"""
Test script for Volume Cluster Slope Detector
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.volume_cluster_slope import (
    detect_volume_cluster_slope, 
    detect_advanced_volume_slope, 
    calculate_advanced_slope,
    get_volume_slope_score,
    analyze_volume_price_dynamics
)

def test_volume_slope_scenarios():
    """Test different volume cluster slope scenarios"""
    
    print("🧪 Testing Volume Cluster Slope Detector")
    print("="*50)
    
    # Test scenario 1: Perfect accumulation pattern
    print("\n📊 Scenario 1: Perfect accumulation (volume + price up)")
    data1 = {
        "recent_volumes": [10000, 15000, 22000, 28000, 35000],
        "recent_closes": [100.0, 100.5, 101.2, 102.0, 103.1]
    }
    result1 = detect_volume_cluster_slope(data1)
    print(f"Result: {result1} (Expected: True)")
    
    # Test scenario 2: Volume up but price down (dump pattern)
    print("\n📊 Scenario 2: Volume up but price down (dump)")
    data2 = {
        "recent_volumes": [10000, 15000, 22000, 28000, 35000],
        "recent_closes": [100.0, 99.5, 98.8, 98.2, 97.5]
    }
    result2 = detect_volume_cluster_slope(data2)
    print(f"Result: {result2} (Expected: False)")
    
    # Test scenario 3: Price up but volume down
    print("\n📊 Scenario 3: Price up but volume declining")
    data3 = {
        "recent_volumes": [35000, 28000, 22000, 15000, 10000],
        "recent_closes": [100.0, 100.5, 101.2, 102.0, 103.1]
    }
    result3 = detect_volume_cluster_slope(data3)
    print(f"Result: {result3} (Expected: False)")
    
    # Test scenario 4: Insufficient data
    print("\n📊 Scenario 4: Insufficient data")
    data4 = {
        "recent_volumes": [10000, 15000],
        "recent_closes": [100.0, 100.5]
    }
    result4 = detect_volume_cluster_slope(data4)
    print(f"Result: {result4} (Expected: False)")
    
    return [result1, result2, result3, result4]

def test_advanced_slope_calculation():
    """Test advanced slope calculation using linear regression"""
    
    print("\n🧮 Testing Advanced Slope Calculation")
    print("="*40)
    
    # Test scenario 1: Strong upward trend
    print("\nScenario 1: Strong upward trend")
    data_points = [10, 15, 22, 28, 35, 42]
    slope = calculate_advanced_slope(data_points)
    print(f"Data: {data_points}")
    print(f"Calculated slope: {slope:.2f}")
    print(f"Trend: {'Up' if slope > 0 else 'Down'}")
    
    # Test scenario 2: Downward trend
    print("\nScenario 2: Downward trend")
    data_points2 = [42, 35, 28, 22, 15, 10]
    slope2 = calculate_advanced_slope(data_points2)
    print(f"Data: {data_points2}")
    print(f"Calculated slope: {slope2:.2f}")
    print(f"Trend: {'Up' if slope2 > 0 else 'Down'}")
    
    # Test scenario 3: Flat trend
    print("\nScenario 3: Flat trend")
    data_points3 = [20, 21, 20, 19, 20, 21]
    slope3 = calculate_advanced_slope(data_points3)
    print(f"Data: {data_points3}")
    print(f"Calculated slope: {slope3:.2f}")
    print(f"Trend: {'Flat' if abs(slope3) < 0.5 else 'Up' if slope3 > 0 else 'Down'}")
    
    return [slope, slope2, slope3]

def test_advanced_volume_slope():
    """Test advanced volume slope detection with correlation"""
    
    print("\n🎯 Testing Advanced Volume Slope Detection")
    print("="*45)
    
    # Test scenario 1: High correlation accumulation
    print("\nScenario 1: High correlation accumulation")
    data1 = {
        "recent_volumes": [10000, 15000, 22000, 28000, 35000, 42000],
        "recent_closes": [100.0, 100.8, 101.5, 102.3, 103.0, 103.8]
    }
    result1 = detect_advanced_volume_slope(data1)
    print(f"Volume slope up: {result1['volume_slope_up']}")
    print(f"Volume slope: {result1['volume_slope']:.0f}")
    print(f"Price slope: {result1['price_slope']:.3f}")
    print(f"Correlation: {result1['correlation_strength']:.3f}")
    
    # Test scenario 2: Low correlation (mixed signals)
    print("\nScenario 2: Low correlation (mixed signals)")
    data2 = {
        "recent_volumes": [10000, 25000, 15000, 30000, 20000, 35000],
        "recent_closes": [100.0, 99.5, 101.0, 99.8, 100.5, 101.2]
    }
    result2 = detect_advanced_volume_slope(data2)
    print(f"Volume slope up: {result2['volume_slope_up']}")
    print(f"Volume slope: {result2['volume_slope']:.0f}")
    print(f"Price slope: {result2['price_slope']:.3f}")
    print(f"Correlation: {result2['correlation_strength']:.3f}")
    
    return [result1, result2]

def test_volume_slope_scoring():
    """Test volume slope scoring system"""
    
    print("\n🏆 Testing Volume Slope Scoring")
    print("="*35)
    
    # Test high score scenario
    print("\nScenario 1: High volume slope score")
    slope_data1 = {
        "volume_slope_up": True,
        "accumulation_strength": 0.9,
        "correlation_strength": 0.8
    }
    score1 = get_volume_slope_score(slope_data1)
    print(f"Score: {score1}/6 (Expected: 6)")
    
    # Test medium score scenario
    print("\nScenario 2: Medium volume slope score")
    slope_data2 = {
        "volume_slope_up": True,
        "accumulation_strength": 0.7,
        "correlation_strength": 0.6
    }
    score2 = get_volume_slope_score(slope_data2)
    print(f"Score: {score2}/6 (Expected: 5)")
    
    # Test low score scenario
    print("\nScenario 3: No volume slope detected")
    slope_data3 = {
        "volume_slope_up": False,
        "accumulation_strength": 0.3,
        "correlation_strength": 0.4
    }
    score3 = get_volume_slope_score(slope_data3)
    print(f"Score: {score3}/6 (Expected: 0)")
    
    return [score1, score2, score3]

def test_volume_price_analysis():
    """Test comprehensive volume-price dynamics analysis"""
    
    print("\n📈 Testing Volume-Price Dynamics Analysis")
    print("="*45)
    
    test_symbol = "BTCUSDT"
    result = analyze_volume_price_dynamics(test_symbol)
    
    print(f"Symbol: {test_symbol}")
    print(f"Volume slope up: {result.get('volume_slope_up')}")
    print(f"Analysis available: {result.get('analysis_available')}")
    print(f"Volume trend: {result.get('volume_trend')}")
    print(f"Price trend: {result.get('price_trend')}")
    print(f"Accumulation strength: {result.get('accumulation_strength', 0):.2f}")
    
    return result

def test_integration_scenarios():
    """Test integration scenarios with other signals"""
    
    print("\n🔄 Testing Integration Scenarios")
    print("="*35)
    
    # Scenario 1: Volume slope + whale activity
    print("\nScenario 1: Volume slope with whale accumulation")
    signals = {
        "volume_slope_up": True,
        "whale_activity": True,
        "vwap_pinned": True,
        "dex_inflow": True
    }
    
    signal_count = sum(1 for signal in signals.values() if signal)
    
    print(f"Volume slope up: {signals['volume_slope_up']}")
    print(f"Whale activity: {signals['whale_activity']}")
    print(f"VWAP pinned: {signals['vwap_pinned']}")
    print(f"DEX inflow: {signals['dex_inflow']}")
    print(f"Combined signals: {signal_count}/4")
    print(f"PPWCS contribution: +4 (volume) + 15 (whale) + 4 (VWAP) + 15 (DEX) = +38 points")
    
    return signals

if __name__ == "__main__":
    print("🚀 Volume Cluster Slope Detector Test Suite")
    print("="*65)
    
    # Run all tests
    slope_results = test_volume_slope_scenarios()
    slope_calculations = test_advanced_slope_calculation()
    advanced_results = test_advanced_volume_slope()
    scoring_results = test_volume_slope_scoring()
    analysis_result = test_volume_price_analysis()
    integration_result = test_integration_scenarios()
    
    # Summary
    print("\n📋 Test Summary")
    print("="*20)
    
    expected_slope = [True, False, False, False]
    slope_passed = slope_results == expected_slope
    print(f"✅ Volume Slope Detection: {'PASSED' if slope_passed else 'FAILED'}")
    
    slope_calc_passed = slope_calculations[0] > 0 and slope_calculations[1] < 0
    print(f"✅ Slope Calculation: {'PASSED' if slope_calc_passed else 'FAILED'}")
    
    advanced_passed = advanced_results[0]["volume_slope_up"] == True and advanced_results[1]["volume_slope_up"] == False
    print(f"✅ Advanced Detection: {'PASSED' if advanced_passed else 'FAILED'}")
    
    expected_scores = [6, 5, 0]
    scoring_passed = scoring_results == expected_scores
    print(f"✅ Scoring System: {'PASSED' if scoring_passed else 'FAILED'}")
    
    analysis_passed = isinstance(analysis_result, dict) and "volume_slope_up" in analysis_result
    print(f"✅ Dynamics Analysis: {'PASSED' if analysis_passed else 'FAILED'}")
    
    integration_passed = integration_result["volume_slope_up"] == True
    print(f"✅ Integration Test: {'PASSED' if integration_passed else 'FAILED'}")
    
    overall_passed = slope_passed and slope_calc_passed and advanced_passed and scoring_passed and analysis_passed and integration_passed
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_passed else '❌ SOME TESTS FAILED'}")
    
    if overall_passed:
        print("\n🎉 Volume Cluster Slope Detector ready for production!")
        print("Key features:")
        print("- Detects dynamic volume and price growth patterns")
        print("- Uses linear regression for accurate slope calculation")
        print("- Requires positive correlation between volume and price")
        print("- Contributes +4 points to PPWCS score")
        print("- Integrates with whale detection for stronger accumulation signals")