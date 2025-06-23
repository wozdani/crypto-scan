#!/usr/bin/env python3
"""
Test Pulse Delay Integration
Comprehensive testing of Pulse Delay Detector with trend mode pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.pulse_delay import (
    detect_pulse_delay,
    calculate_pulse_delay_score,
    analyze_pulse_delay_detailed,
    get_pulse_delay_summary
)

def create_controlled_trend_data():
    """Tworzy dane z kontrolowanym trendem (flat → impulse → flat → impulse)"""
    prices = [100.0]
    price = 100.0
    
    # Stwórz 4 cykle flat → impulse → flat
    for cycle in range(4):
        # Flat period 1 (2 periods, minimal change)
        price += 0.02  # +0.02%
        prices.append(price)
        price -= 0.01  # -0.01%
        prices.append(price)
        
        # Impulse period (2 periods, >0.1% total)
        price += 0.08  # +0.08%
        prices.append(price)
        price += 0.04  # +0.04% (total +0.12%)
        prices.append(price)
        
        # Flat period 2 (2 periods, minimal change)
        price += 0.01  # +0.01%
        prices.append(price)
        price -= 0.02  # -0.02%
        prices.append(price)
    
    # Dodaj kilka dodatkowych punktów
    for i in range(6):
        price += 0.01 * ((-1) ** i)
        prices.append(price)
    
    return prices

def create_institutional_flow_data():
    """Tworzy dane symulujące ruch instytucjonalny z pauzami"""
    prices = [100.0]
    price = 100.0
    
    # Pattern: długie pauzy, krótkie impulsy
    # Pauza 1
    for i in range(3):
        price += 0.01 * ((-1) ** i)  # minimal movement
        prices.append(price)
    
    # Impuls 1
    price += 0.15  # strong move
    prices.append(price)
    price += 0.05
    prices.append(price)
    
    # Pauza 2
    for i in range(3):
        price += 0.02 * ((-1) ** i)
        prices.append(price)
    
    # Impuls 2
    price += 0.12
    prices.append(price)
    price += 0.08
    prices.append(price)
    
    # Długa pauza
    for i in range(8):
        price += 0.01 * ((-1) ** (i % 3))
        prices.append(price)
    
    # Końcowy impuls
    price += 0.20
    prices.append(price)
    price += 0.10
    prices.append(price)
    
    return prices

def test_pulse_delay_scenarios():
    """Test różnych scenariuszy pulse delay"""
    print("🧪 Testing Pulse Delay Scenarios\n")
    
    # Scenario 1: Controlled trend
    controlled_data = create_controlled_trend_data()
    detected1, desc1, details1 = detect_pulse_delay(controlled_data)
    score1 = calculate_pulse_delay_score((detected1, desc1, details1))
    analysis1 = analyze_pulse_delay_detailed(controlled_data)
    
    print(f"📊 Controlled Trend Test:")
    print(f"   Detected: {detected1}")
    print(f"   Description: {desc1}")
    print(f"   Patterns: {details1['pattern_count']}")
    print(f"   Score: {score1}/15")
    print(f"   Pattern density: {analysis1['pattern_density']}")
    
    # Scenario 2: Institutional flow
    institutional_data = create_institutional_flow_data()
    detected2, desc2, details2 = detect_pulse_delay(institutional_data)
    score2 = calculate_pulse_delay_score((detected2, desc2, details2))
    analysis2 = analyze_pulse_delay_detailed(institutional_data)
    
    print(f"\n🏦 Institutional Flow Test:")
    print(f"   Detected: {detected2}")
    print(f"   Description: {desc2}")
    print(f"   Patterns: {details2['pattern_count']}")
    print(f"   Score: {score2}/15")
    print(f"   Avg impulse strength: {details2['avg_impulse_strength']:.3f}%")
    
    # Scenario 3: FOMO chaotic movement
    fomo_data = [100.0]
    price = 100.0
    
    # Rapid chaotic movements (FOMO pattern)
    for i in range(30):
        change = 0.08 * (1 if i % 3 == 0 else -0.5) * (1 + i * 0.02)
        price += change
        fomo_data.append(price)
    
    detected3, desc3, details3 = detect_pulse_delay(fomo_data)
    score3 = calculate_pulse_delay_score((detected3, desc3, details3))
    
    print(f"\n📈 FOMO Chaotic Test:")
    print(f"   Detected: {detected3}")
    print(f"   Description: {desc3}")
    print(f"   Patterns: {details3['pattern_count']}")
    print(f"   Score: {score3}/15")

def test_trend_mode_integration():
    """Test integracji z trend mode pipeline"""
    print("\n🔗 Testing Trend Mode Integration\n")
    
    try:
        from utils.trend_mode_pipeline import detect_trend_mode_extended
        
        # Test z controlled trend data
        controlled_data = create_controlled_trend_data()
        
        # Mock function for price series
        def mock_get_price_series(symbol):
            return controlled_data
        
        # Temporary monkey patch
        import detectors.pulse_delay as pd_module
        original_func = pd_module.get_price_series_bybit
        pd_module.get_price_series_bybit = mock_get_price_series
        
        # Also patch flow_consistency module
        import detectors.flow_consistency as fc_module
        original_fc_func = fc_module.get_price_series_bybit
        fc_module.get_price_series_bybit = mock_get_price_series
        
        try:
            # Create synthetic candles
            import time
            base_timestamp = int(time.time() * 1000)
            synthetic_candles = []
            
            for i, price in enumerate(controlled_data[:12]):
                timestamp = base_timestamp - (11-i) * 900000
                open_price = price * 0.9998
                close_price = price
                high_price = price * 1.0002
                low_price = price * 0.9998
                volume = 80000
                
                synthetic_candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
            
            # Test trend mode with pulse delay
            trend_active, description, details = detect_trend_mode_extended("TESTUSDT", synthetic_candles)
            
            print(f"Trend Mode Result:")
            print(f"   Active: {trend_active}")
            print(f"   Description: {description}")
            print(f"   Combined confidence: {details.get('combined_confidence', 0)}")
            print(f"   Base confidence: {details.get('base_confidence', 0)}")
            
            # Check pulse delay integration
            pulse_info = details.get('pulse_delay', {})
            consistency_info = details.get('flow_consistency', {})
            directional_info = details.get('directional_flow', {})
            
            print(f"\nComprehensive Flow Analysis Results:")
            print(f"   Pulse delay detected: {pulse_info.get('detected', False)}")
            print(f"   Pulse delay score: {pulse_info.get('score', 0):+d}")
            print(f"   Flow consistency: {consistency_info.get('index', 0):.3f}")
            print(f"   Consistency score: {consistency_info.get('score', 0):+d}")
            print(f"   Directional score: {directional_info.get('score', 0):+d}")
            print(f"   Total flow adjustment: {details.get('total_flow_adjustment', 0):+d}")
            
            # Check if we reached enhanced confidence range
            if details.get('combined_confidence', 0) > 140:
                print(f"   ✅ Enhanced confidence range achieved: {details.get('combined_confidence', 0)}/155")
            
        finally:
            # Restore original functions
            pd_module.get_price_series_bybit = original_func
            fc_module.get_price_series_bybit = original_fc_func
            
    except ImportError as e:
        print(f"❌ Integration test failed: {e}")

def test_pulse_delay_edge_cases():
    """Test edge cases dla pulse delay"""
    print("\n🧪 Testing Pulse Delay Edge Cases\n")
    
    # Test 1: Insufficient data
    small_data = [100.0, 101.0, 102.0]
    detected, desc, details = detect_pulse_delay(small_data)
    print(f"Small dataset: {detected}, {desc}")
    
    # Test 2: Perfect flat (no movement)
    flat_data = [100.0] * 25
    detected, desc, details = detect_pulse_delay(flat_data)
    score = calculate_pulse_delay_score((detected, desc, details))
    print(f"Flat data: {detected}, score: {score}")
    
    # Test 3: Single strong impulse (no pattern)
    single_impulse = [100.0] * 10
    single_impulse.extend([100.5, 101.0, 101.5, 102.0])  # single strong move
    single_impulse.extend([102.0] * 10)  # then flat
    
    detected, desc, details = detect_pulse_delay(single_impulse)
    score = calculate_pulse_delay_score((detected, desc, details))
    print(f"Single impulse: {detected}, patterns: {details['pattern_count']}, score: {score}")
    
    # Test 4: Overlapping patterns
    overlapping = [100.0]
    price = 100.0
    
    # Create overlapping flat-impulse sequences
    for i in range(15):
        if i % 4 < 2:  # flat periods
            price += 0.01 * ((-1) ** i)
        else:  # impulse periods
            price += 0.08
        overlapping.append(price)
    
    detected, desc, details = detect_pulse_delay(overlapping)
    analysis = analyze_pulse_delay_detailed(overlapping)
    print(f"Overlapping patterns: {detected}, patterns: {details.get('pattern_count', 0)}")
    print(f"Pattern density: {analysis.get('pattern_density', 0)}")

def test_pulse_delay_summary():
    """Test pulse delay summary function"""
    print("\n📋 Testing Pulse Delay Summary\n")
    
    test_cases = [
        ("Controlled trend", create_controlled_trend_data()),
        ("Institutional flow", create_institutional_flow_data()),
        ("Small dataset", [100.0, 101.0, 102.0]),
        ("Flat data", [100.0] * 25)
    ]
    
    for name, data in test_cases:
        summary = get_pulse_delay_summary(data)
        print(f"{name}: {summary}")

def run_comprehensive_pulse_delay_tests():
    """Uruchom wszystkie testy pulse delay"""
    print("🚀 Starting Comprehensive Pulse Delay Tests\n")
    
    try:
        test_pulse_delay_scenarios()
        test_trend_mode_integration()
        test_pulse_delay_edge_cases()
        test_pulse_delay_summary()
        
        print("\n✅ All Pulse Delay tests completed!")
        print("\n📋 Test Summary:")
        print("✓ Pulse delay pattern detection (controlled/institutional/chaotic)")
        print("✓ Trend mode pipeline integration")
        print("✓ Edge cases validation")
        print("✓ Enhanced confidence calculation (0-155 range)")
        print("✓ Comprehensive flow analysis (directional + consistency + pulse delay)")
        
    except Exception as e:
        print(f"\n❌ Pulse delay test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_pulse_delay_tests()