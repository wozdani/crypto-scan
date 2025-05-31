#!/usr/bin/env python3
"""
Test script for Orderbook Spoofing Detector
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.orderbook_spoofing import detect_orderbook_spoofing, analyze_orderbook_walls, get_spoofing_score

def test_spoofing_detection_scenarios():
    """Test different orderbook spoofing scenarios"""
    
    print("🧪 Testing Orderbook Spoofing Detector")
    print("="*50)
    
    # Test scenario 1: Classic spoofing pattern
    print("\n📊 Scenario 1: Short-lived ask wall + whale activity")
    data1 = {
        "ask_wall_appeared": True,
        "ask_wall_disappeared": True,
        "ask_wall_lifetime_sec": 45,  # Short-lived
        "whale_activity": True,
        "volume_spike": False
    }
    result1 = detect_orderbook_spoofing(data1)
    print(f"Result: {result1} (Expected: True)")
    
    # Test scenario 2: Spoofing with volume spike
    print("\n📊 Scenario 2: Short-lived ask wall + volume spike")
    data2 = {
        "ask_wall_appeared": True,
        "ask_wall_disappeared": True,
        "ask_wall_lifetime_sec": 60,  # Short-lived
        "whale_activity": False,
        "volume_spike": True
    }
    result2 = detect_orderbook_spoofing(data2)
    print(f"Result: {result2} (Expected: True)")
    
    # Test scenario 3: Long-lived wall (not spoofing)
    print("\n📊 Scenario 3: Long-lived ask wall (legitimate)")
    data3 = {
        "ask_wall_appeared": True,
        "ask_wall_disappeared": True,
        "ask_wall_lifetime_sec": 120,  # Too long for spoofing
        "whale_activity": True,
        "volume_spike": False
    }
    result3 = detect_orderbook_spoofing(data3)
    print(f"Result: {result3} (Expected: False)")
    
    # Test scenario 4: Wall without disappearing
    print("\n📊 Scenario 4: Wall appeared but didn't disappear")
    data4 = {
        "ask_wall_appeared": True,
        "ask_wall_disappeared": False,
        "ask_wall_lifetime_sec": 45,
        "whale_activity": True,
        "volume_spike": False
    }
    result4 = detect_orderbook_spoofing(data4)
    print(f"Result: {result4} (Expected: False)")
    
    # Test scenario 5: No activity signals
    print("\n📊 Scenario 5: Short wall but no whale/volume activity")
    data5 = {
        "ask_wall_appeared": True,
        "ask_wall_disappeared": True,
        "ask_wall_lifetime_sec": 30,
        "whale_activity": False,
        "volume_spike": False
    }
    result5 = detect_orderbook_spoofing(data5)
    print(f"Result: {result5} (Expected: False)")
    
    return [result1, result2, result3, result4, result5]

def test_spoofing_scoring():
    """Test spoofing scoring system"""
    
    print("\n🎯 Testing Spoofing Scoring System")
    print("="*40)
    
    # Test high score scenario
    print("\nScenario 1: High spoofing score")
    spoofing_data1 = {
        "spoofing_suspected": True,
        "wall_size_usd": 150000,
        "walls_detected_count": 3,
        "bid_wall_strength": 0.3
    }
    score1 = get_spoofing_score(spoofing_data1)
    print(f"Score: {score1}/8 (Expected: 8)")
    
    # Test medium score scenario
    print("\nScenario 2: Medium spoofing score")
    spoofing_data2 = {
        "spoofing_suspected": True,
        "wall_size_usd": 75000,
        "walls_detected_count": 1,
        "bid_wall_strength": 0.5
    }
    score2 = get_spoofing_score(spoofing_data2)
    print(f"Score: {score2}/8 (Expected: 4)")
    
    # Test low score scenario with strong bid support
    print("\nScenario 3: Reduced score due to strong bid support")
    spoofing_data3 = {
        "spoofing_suspected": True,
        "wall_size_usd": 30000,
        "walls_detected_count": 0,
        "bid_wall_strength": 0.8  # Strong support reduces suspicion
    }
    score3 = get_spoofing_score(spoofing_data3)
    print(f"Score: {score3}/8 (Expected: 2)")
    
    return [score1, score2, score3]

def test_wall_analysis():
    """Test orderbook wall analysis function"""
    
    print("\n📈 Testing Orderbook Wall Analysis")
    print("="*40)
    
    test_symbol = "BTCUSDT"
    result = analyze_orderbook_walls(test_symbol)
    
    print(f"Symbol: {test_symbol}")
    print(f"Ask wall appeared: {result.get('ask_wall_appeared')}")
    print(f"Ask wall disappeared: {result.get('ask_wall_disappeared')}")
    print(f"Wall lifetime (sec): {result.get('ask_wall_lifetime_sec')}")
    print(f"Wall size (USD): {result.get('wall_size_usd')}")
    print(f"Walls detected count: {result.get('walls_detected_count')}")
    print(f"Bid wall strength: {result.get('bid_wall_strength')}")
    
    return result

def test_integration_scenarios():
    """Test integration scenarios combining multiple signals"""
    
    print("\n🔄 Testing Integration Scenarios")
    print("="*35)
    
    # Scenario 1: Perfect spoofing setup
    print("\nScenario 1: Perfect spoofing pattern")
    data = {
        "ask_wall_appeared": True,
        "ask_wall_disappeared": True,
        "ask_wall_lifetime_sec": 30,
        "whale_activity": True,
        "volume_spike": True
    }
    spoofing_detected = detect_orderbook_spoofing(data)
    
    spoofing_score_data = {
        "spoofing_suspected": spoofing_detected,
        "wall_size_usd": 200000,
        "walls_detected_count": 2,
        "bid_wall_strength": 0.2
    }
    score = get_spoofing_score(spoofing_score_data)
    
    print(f"Spoofing detected: {spoofing_detected}")
    print(f"Spoofing score: {score}/8")
    print(f"PPWCS contribution: +3 points (if detected)")
    
    return spoofing_detected, score

if __name__ == "__main__":
    print("🚀 Orderbook Spoofing Detector Test Suite")
    print("="*60)
    
    # Run all tests
    spoofing_results = test_spoofing_detection_scenarios()
    scoring_results = test_spoofing_scoring()
    wall_result = test_wall_analysis()
    integration_results = test_integration_scenarios()
    
    # Summary
    print("\n📋 Test Summary")
    print("="*20)
    
    expected_spoofing = [True, True, False, False, False]
    spoofing_passed = spoofing_results == expected_spoofing
    print(f"✅ Spoofing Detection: {'PASSED' if spoofing_passed else 'FAILED'}")
    
    expected_scores = [8, 4, 2]
    scoring_passed = scoring_results == expected_scores
    print(f"✅ Scoring System: {'PASSED' if scoring_passed else 'FAILED'}")
    
    wall_passed = isinstance(wall_result, dict) and all(
        key in wall_result for key in ["ask_wall_appeared", "ask_wall_disappeared", "ask_wall_lifetime_sec"]
    )
    print(f"✅ Wall Analysis: {'PASSED' if wall_passed else 'FAILED'}")
    
    integration_passed = integration_results[0] == True and integration_results[1] >= 6
    print(f"✅ Integration Test: {'PASSED' if integration_passed else 'FAILED'}")
    
    overall_passed = spoofing_passed and scoring_passed and wall_passed and integration_passed
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_passed else '❌ SOME TESTS FAILED'}")
    
    if overall_passed:
        print("\n🎉 Orderbook Spoofing Detector ready for production!")
        print("Key features:")
        print("- Detects short-lived ask walls (< 90 seconds)")
        print("- Requires whale activity or volume spike confirmation")
        print("- Contributes +3 points to PPWCS score")
        print("- Integrates with existing Stage -2.1 pipeline")