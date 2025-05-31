#!/usr/bin/env python3
"""
Test script for Heatmap Exhaustion Detector
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.heatmap_exhaustion import detect_heatmap_exhaustion, analyze_orderbook_exhaustion, get_exhaustion_score

def test_heatmap_exhaustion_scenarios():
    """Test different heatmap exhaustion scenarios"""
    
    print("🧪 Testing Heatmap Exhaustion Detector")
    print("="*50)
    
    # Test scenario 1: Strong exhaustion signal
    print("\n📊 Scenario 1: Ask wall disappeared + Volume spike")
    data1 = {
        "ask_wall_disappeared": True,
        "volume_spike": True,
        "whale_activity": False
    }
    result1 = detect_heatmap_exhaustion(data1)
    print(f"Result: {result1} (Expected: True)")
    
    # Test scenario 2: Strong exhaustion with whale activity
    print("\n📊 Scenario 2: Ask wall disappeared + Whale activity")
    data2 = {
        "ask_wall_disappeared": True,
        "volume_spike": False,
        "whale_activity": True
    }
    result2 = detect_heatmap_exhaustion(data2)
    print(f"Result: {result2} (Expected: True)")
    
    # Test scenario 3: No exhaustion - only volume spike
    print("\n📊 Scenario 3: Volume spike only (no ask wall disappeared)")
    data3 = {
        "ask_wall_disappeared": False,
        "volume_spike": True,
        "whale_activity": False
    }
    result3 = detect_heatmap_exhaustion(data3)
    print(f"Result: {result3} (Expected: False)")
    
    # Test scenario 4: No signals
    print("\n📊 Scenario 4: No signals")
    data4 = {
        "ask_wall_disappeared": False,
        "volume_spike": False,
        "whale_activity": False
    }
    result4 = detect_heatmap_exhaustion(data4)
    print(f"Result: {result4} (Expected: False)")
    
    return [result1, result2, result3, result4]

def test_exhaustion_scoring():
    """Test exhaustion scoring system"""
    
    print("\n🎯 Testing Exhaustion Scoring System")
    print("="*40)
    
    # Test high score scenario
    print("\nScenario 1: High exhaustion score")
    exhaustion_data1 = {
        "ask_wall_disappeared": True,
        "bid_ask_ratio": 2.0,
        "large_asks_removed": 5,
        "price_stability": True
    }
    score1 = get_exhaustion_score(exhaustion_data1)
    print(f"Score: {score1}/10 (Expected: 10)")
    
    # Test medium score scenario
    print("\nScenario 2: Medium exhaustion score")
    exhaustion_data2 = {
        "ask_wall_disappeared": True,
        "bid_ask_ratio": 1.3,
        "large_asks_removed": 2,
        "price_stability": True
    }
    score2 = get_exhaustion_score(exhaustion_data2)
    print(f"Score: {score2}/10 (Expected: 8)")
    
    # Test low score scenario
    print("\nScenario 3: Low exhaustion score")
    exhaustion_data3 = {
        "ask_wall_disappeared": False,
        "bid_ask_ratio": 1.1,
        "large_asks_removed": 0,
        "price_stability": True
    }
    score3 = get_exhaustion_score(exhaustion_data3)
    print(f"Score: {score3}/10 (Expected: 2)")
    
    return [score1, score2, score3]

def test_orderbook_analysis():
    """Test orderbook analysis function"""
    
    print("\n📈 Testing Orderbook Analysis")
    print("="*35)
    
    test_symbol = "BTCUSDT"
    result = analyze_orderbook_exhaustion(test_symbol)
    
    print(f"Symbol: {test_symbol}")
    print(f"Ask wall disappeared: {result.get('ask_wall_disappeared')}")
    print(f"Bid/Ask ratio: {result.get('bid_ask_ratio')}")
    print(f"Large asks removed: {result.get('large_asks_removed')}")
    print(f"Price stability: {result.get('price_stability')}")
    
    return result

if __name__ == "__main__":
    print("🚀 Heatmap Exhaustion Detector Test Suite")
    print("="*60)
    
    # Run all tests
    exhaustion_results = test_heatmap_exhaustion_scenarios()
    scoring_results = test_exhaustion_scoring()
    orderbook_result = test_orderbook_analysis()
    
    # Summary
    print("\n📋 Test Summary")
    print("="*20)
    
    expected_exhaustion = [True, True, False, False]
    exhaustion_passed = exhaustion_results == expected_exhaustion
    print(f"✅ Exhaustion Detection: {'PASSED' if exhaustion_passed else 'FAILED'}")
    
    expected_scores = [10, 8, 2]
    scoring_passed = scoring_results == expected_scores
    print(f"✅ Scoring System: {'PASSED' if scoring_passed else 'FAILED'}")
    
    orderbook_passed = isinstance(orderbook_result, dict) and all(
        key in orderbook_result for key in ["ask_wall_disappeared", "bid_ask_ratio", "large_asks_removed", "price_stability"]
    )
    print(f"✅ Orderbook Analysis: {'PASSED' if orderbook_passed else 'FAILED'}")
    
    overall_passed = exhaustion_passed and scoring_passed and orderbook_passed
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_passed else '❌ SOME TESTS FAILED'}")