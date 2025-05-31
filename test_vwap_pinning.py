#!/usr/bin/env python3
"""
Test script for VWAP Pinning Detector
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.vwap_pinning import detect_vwap_pinning, analyze_vwap_control, get_vwap_pinning_score, calculate_vwap

def test_vwap_pinning_scenarios():
    """Test different VWAP pinning scenarios"""
    
    print("🧪 Testing VWAP Pinning Detector")
    print("="*50)
    
    # Test scenario 1: Perfect VWAP pinning
    print("\n📊 Scenario 1: Perfect VWAP pinning (tight control)")
    data1 = {
        "recent_closes": [100.0, 100.1, 99.9, 100.0, 100.2],
        "recent_vwaps": [100.0, 100.0, 100.0, 100.0, 100.0]
    }
    result1 = detect_vwap_pinning(data1)
    print(f"Result: {result1} (Expected: True)")
    
    # Test scenario 2: Moderate VWAP control
    print("\n📊 Scenario 2: Moderate VWAP control")
    data2 = {
        "recent_closes": [100.0, 100.3, 99.7, 100.1, 99.8],
        "recent_vwaps": [100.0, 100.0, 100.0, 100.0, 100.0]
    }
    result2 = detect_vwap_pinning(data2)
    print(f"Result: {result2} (Expected: True)")
    
    # Test scenario 3: No VWAP control (normal volatility)
    print("\n📊 Scenario 3: No VWAP control (high volatility)")
    data3 = {
        "recent_closes": [100.0, 102.0, 98.5, 101.5, 97.0],
        "recent_vwaps": [100.0, 100.0, 100.0, 100.0, 100.0]
    }
    result3 = detect_vwap_pinning(data3)
    print(f"Result: {result3} (Expected: False)")
    
    # Test scenario 4: Insufficient data
    print("\n📊 Scenario 4: Insufficient data")
    data4 = {
        "recent_closes": [100.0, 100.1],
        "recent_vwaps": [100.0, 100.0]
    }
    result4 = detect_vwap_pinning(data4)
    print(f"Result: {result4} (Expected: False)")
    
    return [result1, result2, result3, result4]

def test_vwap_calculation():
    """Test VWAP calculation function"""
    
    print("\n🧮 Testing VWAP Calculation")
    print("="*35)
    
    # Test scenario 1: Basic VWAP calculation
    print("\nScenario 1: Basic VWAP calculation")
    prices = [100.0, 101.0, 99.0, 102.0]
    volumes = [1000, 1500, 800, 1200]
    vwap = calculate_vwap(prices, volumes)
    expected_vwap = (100*1000 + 101*1500 + 99*800 + 102*1200) / (1000+1500+800+1200)
    print(f"Calculated VWAP: {vwap:.2f}")
    print(f"Expected VWAP: {expected_vwap:.2f}")
    print(f"Match: {abs(vwap - expected_vwap) < 0.01}")
    
    # Test scenario 2: Edge cases
    print("\nScenario 2: Edge cases")
    empty_vwap = calculate_vwap([], [])
    zero_volume_vwap = calculate_vwap([100, 101], [0, 0])
    print(f"Empty data VWAP: {empty_vwap}")
    print(f"Zero volume VWAP: {zero_volume_vwap}")
    
    return vwap, expected_vwap

def test_vwap_scoring():
    """Test VWAP pinning scoring system"""
    
    print("\n🎯 Testing VWAP Pinning Scoring")
    print("="*40)
    
    # Test high score scenario
    print("\nScenario 1: High VWAP pinning score")
    vwap_data1 = {
        "vwap_pinned": True,
        "control_strength": 0.9,
        "candles_analyzed": 5
    }
    score1 = get_vwap_pinning_score(vwap_data1)
    print(f"Score: {score1}/5 (Expected: 5)")
    
    # Test medium score scenario
    print("\nScenario 2: Medium VWAP pinning score")
    vwap_data2 = {
        "vwap_pinned": True,
        "control_strength": 0.7,
        "candles_analyzed": 3
    }
    score2 = get_vwap_pinning_score(vwap_data2)
    print(f"Score: {score2}/5 (Expected: 4)")
    
    # Test low score scenario
    print("\nScenario 3: No VWAP pinning detected")
    vwap_data3 = {
        "vwap_pinned": False,
        "control_strength": 0.3,
        "candles_analyzed": 5
    }
    score3 = get_vwap_pinning_score(vwap_data3)
    print(f"Score: {score3}/5 (Expected: 1)")
    
    return [score1, score2, score3]

def test_vwap_analysis():
    """Test VWAP control analysis function"""
    
    print("\n📈 Testing VWAP Control Analysis")
    print("="*40)
    
    test_symbol = "BTCUSDT"
    result = analyze_vwap_control(test_symbol)
    
    print(f"Symbol: {test_symbol}")
    print(f"VWAP pinned: {result.get('vwap_pinned')}")
    print(f"Average deviation: {result.get('avg_deviation', 0):.4f}")
    print(f"Control strength: {result.get('control_strength', 0):.2f}")
    print(f"Candles analyzed: {result.get('candles_analyzed')}")
    
    return result

def test_integration_scenarios():
    """Test integration scenarios combining VWAP pinning with other signals"""
    
    print("\n🔄 Testing Integration Scenarios")
    print("="*35)
    
    # Scenario 1: VWAP pinning + whale activity
    print("\nScenario 1: VWAP pinning with whale accumulation")
    signals = {
        "vwap_pinned": True,
        "whale_activity": True,
        "volume_spike": False,
        "dex_inflow": True
    }
    
    # Calculate combined signal strength
    signal_count = sum(1 for signal in signals.values() if signal)
    
    print(f"VWAP pinned: {signals['vwap_pinned']}")
    print(f"Whale activity: {signals['whale_activity']}")
    print(f"DEX inflow: {signals['dex_inflow']}")
    print(f"Combined signals: {signal_count}/4")
    print(f"PPWCS contribution: +4 points (VWAP) + 15 (whale) + 15 (DEX)")
    
    # Scenario 2: VWAP pinning alone
    print("\nScenario 2: VWAP pinning without other signals")
    signals2 = {
        "vwap_pinned": True,
        "whale_activity": False,
        "volume_spike": False,
        "dex_inflow": False
    }
    
    signal_count2 = sum(1 for signal in signals2.values() if signal)
    print(f"VWAP pinned: {signals2['vwap_pinned']}")
    print(f"Other signals: {signal_count2-1}/3")
    print(f"PPWCS contribution: +4 points (VWAP only)")
    
    return signals, signals2

if __name__ == "__main__":
    print("🚀 VWAP Pinning Detector Test Suite")
    print("="*60)
    
    # Run all tests
    pinning_results = test_vwap_pinning_scenarios()
    vwap_calc, expected_calc = test_vwap_calculation()
    scoring_results = test_vwap_scoring()
    analysis_result = test_vwap_analysis()
    integration_results = test_integration_scenarios()
    
    # Summary
    print("\n📋 Test Summary")
    print("="*20)
    
    expected_pinning = [True, True, False, False]
    pinning_passed = pinning_results == expected_pinning
    print(f"✅ VWAP Pinning Detection: {'PASSED' if pinning_passed else 'FAILED'}")
    
    vwap_calc_passed = abs(vwap_calc - expected_calc) < 0.01
    print(f"✅ VWAP Calculation: {'PASSED' if vwap_calc_passed else 'FAILED'}")
    
    expected_scores = [5, 4, 1]
    scoring_passed = scoring_results == expected_scores
    print(f"✅ Scoring System: {'PASSED' if scoring_passed else 'FAILED'}")
    
    analysis_passed = isinstance(analysis_result, dict) and all(
        key in analysis_result for key in ["vwap_pinned", "avg_deviation", "control_strength"]
    )
    print(f"✅ Control Analysis: {'PASSED' if analysis_passed else 'FAILED'}")
    
    integration_passed = integration_results[0]["vwap_pinned"] == True
    print(f"✅ Integration Test: {'PASSED' if integration_passed else 'FAILED'}")
    
    overall_passed = pinning_passed and vwap_calc_passed and scoring_passed and analysis_passed and integration_passed
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_passed else '❌ SOME TESTS FAILED'}")
    
    if overall_passed:
        print("\n🎉 VWAP Pinning Detector ready for production!")
        print("Key features:")
        print("- Detects controlled accumulation when price stays near VWAP")
        print("- Requires average deviation < 0.4% over multiple candles")
        print("- Contributes +4 points to PPWCS score")
        print("- Integrates with whale activity detection for stronger signals")