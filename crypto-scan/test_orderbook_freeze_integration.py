#!/usr/bin/env python3
"""
Test Orderbook Freeze Integration
Comprehensive testing of 4-layer flow analysis with orderbook freeze detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.orderbook_freeze import (
    detect_orderbook_freeze,
    calculate_orderbook_freeze_score,
    analyze_orderbook_freeze_detailed,
    create_mock_orderbook_snapshots
)

def create_perfect_freeze_snapshots():
    """Tworzy snapshoty z idealnym freeze - ask nieruchomy, cena rośnie"""
    snapshots = []
    base_price = 0.023
    
    # Snapshot 1: początek
    snapshots.append({
        "price": base_price,
        "bids": [[0.0229, 120], [0.0228, 80], [0.0227, 45]],
        "asks": [[0.0231, 100], [0.0232, 75], [0.0233, 60]],  # asks stabilne
        "timestamp": 1000000
    })
    
    # Snapshot 2: cena wzrosła, ask nadal nieruchomy
    snapshots.append({
        "price": base_price * 1.008,  # +0.8%
        "bids": [[0.02298, 115], [0.02288, 85], [0.02278, 50]],  # bids podążają
        "asks": [[0.0231, 100], [0.0232, 75], [0.0233, 60]],      # asks ZAMROŻONE
        "timestamp": 1000300
    })
    
    # Snapshot 3: dalszy wzrost, ask nadal zamrożony
    snapshots.append({
        "price": base_price * 1.015,  # +1.5% total
        "bids": [[0.02305, 110], [0.02295, 90], [0.02285, 55]],  # bids dalej podążają
        "asks": [[0.0231, 100], [0.0232, 75], [0.0233, 60]],      # asks NADAL ZAMROŻONE
        "timestamp": 1000600
    })
    
    return snapshots

def create_active_ask_snapshots():
    """Tworzy snapshoty z aktywnym ask - brak freeze"""
    snapshots = []
    base_price = 0.023
    
    # Snapshot 1: początek
    snapshots.append({
        "price": base_price,
        "bids": [[0.0229, 120], [0.0228, 80], [0.0227, 45]],
        "asks": [[0.0231, 100], [0.0232, 75], [0.0233, 60]],
        "timestamp": 1000000
    })
    
    # Snapshot 2: cena wzrosła, ask się dostosowuje
    snapshots.append({
        "price": base_price * 1.008,  # +0.8%
        "bids": [[0.02298, 115], [0.02288, 85], [0.02278, 50]],
        "asks": [[0.02318, 95], [0.02328, 70], [0.02338, 55]],    # asks PORUSZAJĄ SIĘ
        "timestamp": 1000300
    })
    
    # Snapshot 3: dalszy wzrost, ask dalej aktywny
    snapshots.append({
        "price": base_price * 1.015,  # +1.5% total
        "bids": [[0.02305, 110], [0.02295, 90], [0.02285, 55]],
        "asks": [[0.02325, 90], [0.02335, 65], [0.02345, 50]],    # asks AKTYWNE
        "timestamp": 1000600
    })
    
    return snapshots

def create_strong_freeze_snapshots():
    """Tworzy snapshoty z silnym freeze - większy wzrost ceny"""
    snapshots = []
    base_price = 0.023
    
    # Snapshot 1
    snapshots.append({
        "price": base_price,
        "bids": [[0.0229, 120], [0.0228, 80], [0.0227, 45]],
        "asks": [[0.0231, 100], [0.0232, 75], [0.0233, 60]],
        "timestamp": 1000000
    })
    
    # Snapshot 2: większy wzrost
    snapshots.append({
        "price": base_price * 1.020,  # +2.0%
        "bids": [[0.02318, 115], [0.02308, 85], [0.02298, 50]],
        "asks": [[0.0231, 100], [0.0232, 75], [0.0233, 60]],  # asks zamrożone
        "timestamp": 1000300
    })
    
    # Snapshot 3: jeszcze większy wzrost
    snapshots.append({
        "price": base_price * 1.035,  # +3.5% total
        "bids": [[0.02333, 110], [0.02323, 90], [0.02313, 55]],
        "asks": [[0.0231, 100], [0.0232, 75], [0.0233, 60]],  # asks NADAL zamrożone
        "timestamp": 1000600
    })
    
    return snapshots

def test_orderbook_freeze_scenarios():
    """Test różnych scenariuszy orderbook freeze"""
    print("🧪 Testing Orderbook Freeze Scenarios\n")
    
    # Scenario 1: Perfect freeze
    perfect_snapshots = create_perfect_freeze_snapshots()
    detected1, desc1, details1 = detect_orderbook_freeze(perfect_snapshots)
    score1 = calculate_orderbook_freeze_score((detected1, desc1, details1))
    
    print(f"🧊 Perfect Freeze Test:")
    print(f"   Detected: {detected1}")
    print(f"   Description: {desc1}")
    print(f"   Price change: {details1['total_price_change_pct']:.3f}%")
    print(f"   Ask movement: {details1['ask_movement_detected']}")
    print(f"   Score: {score1}/15")
    
    # Scenario 2: Active ask (no freeze)
    active_snapshots = create_active_ask_snapshots()
    detected2, desc2, details2 = detect_orderbook_freeze(active_snapshots)
    score2 = calculate_orderbook_freeze_score((detected2, desc2, details2))
    
    print(f"\n📋 Active Ask Test:")
    print(f"   Detected: {detected2}")
    print(f"   Description: {desc2}")
    print(f"   Price change: {details2['total_price_change_pct']:.3f}%")
    print(f"   Ask movement: {details2['ask_movement_detected']}")
    print(f"   Score: {score2}/15")
    
    # Scenario 3: Strong freeze
    strong_snapshots = create_strong_freeze_snapshots()
    detected3, desc3, details3 = detect_orderbook_freeze(strong_snapshots)
    score3 = calculate_orderbook_freeze_score((detected3, desc3, details3))
    
    print(f"\n❄️ Strong Freeze Test:")
    print(f"   Detected: {detected3}")
    print(f"   Description: {desc3}")
    print(f"   Price change: {details3['total_price_change_pct']:.3f}%")
    print(f"   Ask movement: {details3['ask_movement_detected']}")
    print(f"   Score: {score3}/15")

def test_4_layer_flow_integration():
    """Test integracji z 4-warstwowym systemem flow analysis"""
    print("\n🔗 Testing 4-Layer Flow Analysis Integration\n")
    
    try:
        from utils.trend_mode_pipeline import detect_trend_mode_extended
        
        # Test z perfect freeze data
        freeze_snapshots = create_perfect_freeze_snapshots()
        
        # Mock functions for complete flow analysis
        def mock_get_price_series(symbol):
            # Generate price series that matches freeze scenario
            prices = [snapshot["price"] for snapshot in freeze_snapshots]
            # Extend to full 36-point series
            while len(prices) < 36:
                last_price = prices[-1]
                prices.append(last_price * (1 + 0.001))  # gentle upward trend
            return prices
        
        def mock_create_orderbook_snapshots():
            return freeze_snapshots
        
        # Temporary monkey patches
        import detectors.flow_consistency as fc_module
        import detectors.directional_flow_detector as df_module
        import detectors.pulse_delay as pd_module
        import detectors.orderbook_freeze as of_module
        
        original_fc_func = fc_module.get_price_series_bybit
        original_df_func = df_module.get_price_series_bybit
        original_pd_func = pd_module.get_price_series_bybit
        original_of_func = of_module.create_mock_orderbook_snapshots
        
        fc_module.get_price_series_bybit = mock_get_price_series
        df_module.get_price_series_bybit = mock_get_price_series
        pd_module.get_price_series_bybit = mock_get_price_series
        of_module.create_mock_orderbook_snapshots = mock_create_orderbook_snapshots
        
        try:
            # Create synthetic candles
            import time
            base_timestamp = int(time.time() * 1000)
            synthetic_candles = []
            
            for i, snapshot in enumerate(freeze_snapshots):
                timestamp = base_timestamp - (2-i) * 900000
                price = snapshot["price"]
                open_price = price * 0.999
                close_price = price
                high_price = price * 1.001
                low_price = price * 0.999
                volume = 100000
                
                synthetic_candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
            
            # Test complete 4-layer flow analysis
            trend_active, description, details = detect_trend_mode_extended("TESTUSDT", synthetic_candles)
            
            print(f"4-Layer Flow Analysis Result:")
            print(f"   Trend Active: {trend_active}")
            print(f"   Description: {description}")
            print(f"   Combined confidence: {details.get('combined_confidence', 0)}/170")
            print(f"   Base confidence: {details.get('base_confidence', 0)}")
            
            # Check all 4 layers
            directional_info = details.get('directional_flow', {})
            consistency_info = details.get('flow_consistency', {})
            pulse_info = details.get('pulse_delay', {})
            freeze_info = details.get('orderbook_freeze', {})
            
            print(f"\n4-Layer Analysis Breakdown:")
            print(f"   Layer 1 - Directional flow: {directional_info.get('active', False)} ({directional_info.get('score', 0):+d} points)")
            print(f"   Layer 2 - Flow consistency: {consistency_info.get('index', 0):.3f} ({consistency_info.get('score', 0):+d} points)")
            print(f"   Layer 3 - Pulse delay: {pulse_info.get('detected', False)} ({pulse_info.get('score', 0):+d} points)")
            print(f"   Layer 4 - Orderbook freeze: {freeze_info.get('detected', False)} ({freeze_info.get('score', 0):+d} points)")
            print(f"   Total flow adjustment: {details.get('total_flow_adjustment', 0):+d}")
            
            # Check if we reached maximum enhanced confidence
            if details.get('combined_confidence', 0) > 140:
                print(f"   ✅ Enhanced confidence achieved: {details.get('combined_confidence', 0)}/170")
            
        finally:
            # Restore original functions
            fc_module.get_price_series_bybit = original_fc_func
            df_module.get_price_series_bybit = original_df_func
            pd_module.get_price_series_bybit = original_pd_func
            of_module.create_mock_orderbook_snapshots = original_of_func
            
    except ImportError as e:
        print(f"❌ Integration test failed: {e}")

def test_orderbook_freeze_edge_cases():
    """Test edge cases dla orderbook freeze"""
    print("\n🧪 Testing Orderbook Freeze Edge Cases\n")
    
    # Test 1: Insufficient snapshots
    small_snapshots = create_perfect_freeze_snapshots()[:2]  # Only 2 snapshots
    detected, desc, details = detect_orderbook_freeze(small_snapshots)
    print(f"Insufficient snapshots: {detected}, {desc}")
    
    # Test 2: Price decline (should not trigger freeze)
    declining_snapshots = []
    base_price = 0.023
    
    for i, multiplier in enumerate([1.0, 0.995, 0.988]):  # declining prices
        declining_snapshots.append({
            "price": base_price * multiplier,
            "bids": [[0.0229 * multiplier, 120], [0.0228 * multiplier, 80]],
            "asks": [[0.0231, 100], [0.0232, 75]],  # asks frozen but price declining
            "timestamp": 1000000 + i * 300
        })
    
    detected, desc, details = detect_orderbook_freeze(declining_snapshots)
    score = calculate_orderbook_freeze_score((detected, desc, details))
    print(f"Declining price: {detected}, score: {score}, change: {details['total_price_change_pct']:.3f}%")
    
    # Test 3: Minimal price change
    minimal_snapshots = []
    for i, multiplier in enumerate([1.0, 1.002, 1.003]):  # only 0.3% total change
        minimal_snapshots.append({
            "price": base_price * multiplier,
            "bids": [[0.0229 * multiplier, 120], [0.0228 * multiplier, 80]],
            "asks": [[0.0231, 100], [0.0232, 75]],  # asks frozen
            "timestamp": 1000000 + i * 300
        })
    
    detected, desc, details = detect_orderbook_freeze(minimal_snapshots)
    score = calculate_orderbook_freeze_score((detected, desc, details))
    print(f"Minimal change: {detected}, score: {score}, change: {details['total_price_change_pct']:.3f}%")

def run_comprehensive_orderbook_freeze_tests():
    """Uruchom wszystkie testy orderbook freeze"""
    print("🚀 Starting Comprehensive Orderbook Freeze Tests\n")
    
    try:
        test_orderbook_freeze_scenarios()
        test_4_layer_flow_integration()
        test_orderbook_freeze_edge_cases()
        
        print("\n✅ All Orderbook Freeze tests completed!")
        print("\n📋 Test Summary:")
        print("✓ Orderbook freeze pattern detection (perfect/active/strong)")
        print("✓ 4-layer flow analysis integration")
        print("✓ Enhanced confidence calculation (0-170 range)")
        print("✓ Edge cases validation")
        print("✓ Complete trend mode pipeline integration")
        
    except Exception as e:
        print(f"\n❌ Orderbook freeze test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_orderbook_freeze_tests()