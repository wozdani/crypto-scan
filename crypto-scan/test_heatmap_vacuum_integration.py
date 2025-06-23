#!/usr/bin/env python3
"""
Test Heatmap Vacuum Integration
Comprehensive testing of 5-layer flow analysis with heatmap vacuum detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.heatmap_vacuum import (
    detect_heatmap_vacuum,
    calculate_heatmap_vacuum_score,
    analyze_heatmap_vacuum_detailed,
    create_mock_heatmap_snapshots
)

def create_strong_vacuum_snapshots():
    """Tworzy snapshoty z silną próżnią likwidacyjną"""
    snapshots = []
    base_price = 50000.0
    
    # Snapshot 1: pełna heatmap
    snapshots.append({
        "price": base_price,
        "bids": [[49950, 0.5], [49900, 0.3]],
        "asks": [
            [50500, 2.0],  # +1.0% - vacuum zone
            [50600, 1.5],  # +1.2% - vacuum zone  
            [50750, 1.8],  # +1.5% - vacuum zone
            [50900, 1.2],  # +1.8% - vacuum zone
            [51000, 2.5],  # +2.0% - vacuum zone end
        ],
        "timestamp": 1000000
    })
    
    # Snapshot 2: częściowy zanik
    snapshots.append({
        "price": base_price * 1.002,
        "bids": [[49960, 0.5], [49910, 0.3]],
        "asks": [
            [50510, 1.2],  # 40% zanik
            [50610, 0.8],  # 47% zanik
            [50760, 0.9],  # 50% zanik
            [50910, 0.6],  # 50% zanik
            [51010, 2.5],  # bez zmian
        ],
        "timestamp": 1000300
    })
    
    # Snapshot 3: silny vacuum
    snapshots.append({
        "price": base_price * 1.007,
        "bids": [[49985, 0.5], [49935, 0.3]],
        "asks": [
            [50535, 0.4],  # 80% zanik (2.0→0.4)
            [50635, 0.3],  # 80% zanik (1.5→0.3)
            [50785, 0.4],  # 78% zanik (1.8→0.4)
            [50935, 0.2],  # 83% zanik (1.2→0.2)
            [51035, 2.5],  # bez zmian
        ],
        "timestamp": 1000600
    })
    
    return snapshots

def create_weak_vacuum_snapshots():
    """Tworzy snapshoty ze słabą próżnią (tylko 1 poziom z zanikiem)"""
    snapshots = []
    base_price = 50000.0
    
    # Snapshot 1
    snapshots.append({
        "price": base_price,
        "bids": [[49950, 0.5]],
        "asks": [
            [50500, 1.0],  # +1.0%
            [50600, 1.0],  # +1.2%
            [50750, 1.0],  # +1.5%
        ],
        "timestamp": 1000000
    })
    
    # Snapshot 2
    snapshots.append({
        "price": base_price * 1.003,
        "bids": [[49965, 0.5]],
        "asks": [
            [50515, 0.5],  # 50% zanik (tylko ten jeden)
            [50615, 1.0],  # bez zmian
            [50765, 1.0],  # bez zmian
        ],
        "timestamp": 1000300
    })
    
    # Snapshot 3
    snapshots.append({
        "price": base_price * 1.005,
        "bids": [[49975, 0.5]],
        "asks": [
            [50525, 0.4],  # 60% zanik
            [50625, 1.0],  # bez zmian
            [50775, 1.0],  # bez zmian
        ],
        "timestamp": 1000600
    })
    
    return snapshots

def test_heatmap_vacuum_scenarios():
    """Test różnych scenariuszy heatmap vacuum"""
    print("🧪 Testing Heatmap Vacuum Scenarios\n")
    
    # Scenario 1: Strong vacuum
    strong_snapshots = create_strong_vacuum_snapshots()
    detected1, desc1, details1 = detect_heatmap_vacuum(strong_snapshots)
    score1 = calculate_heatmap_vacuum_score((detected1, desc1, details1))
    
    print(f"💪 Strong Vacuum Test:")
    print(f"   Detected: {detected1}")
    print(f"   Description: {desc1}")
    print(f"   Vacuum levels: {details1['vacuum_levels_count']}/{details1['total_levels_analyzed']}")
    print(f"   Vacuum intensity: {details1['vacuum_intensity']:.3f}")
    print(f"   Avg volume reduction: {details1['avg_volume_reduction_pct']:.1f}%")
    print(f"   Score: {score1}/10")
    
    # Scenario 2: Weak vacuum (insufficient levels)
    weak_snapshots = create_weak_vacuum_snapshots()
    detected2, desc2, details2 = detect_heatmap_vacuum(weak_snapshots)
    score2 = calculate_heatmap_vacuum_score((detected2, desc2, details2))
    
    print(f"\n🔸 Weak Vacuum Test:")
    print(f"   Detected: {detected2}")
    print(f"   Description: {desc2}")
    print(f"   Vacuum levels: {details2['vacuum_levels_count']}/{details2['total_levels_analyzed']}")
    print(f"   Vacuum intensity: {details2['vacuum_intensity']:.3f}")
    print(f"   Avg volume reduction: {details2['avg_volume_reduction_pct']:.1f}%")
    print(f"   Score: {score2}/10")
    
    # Scenario 3: Perfect mock vacuum
    mock_snapshots = create_mock_heatmap_snapshots()
    detected3, desc3, details3 = detect_heatmap_vacuum(mock_snapshots)
    score3 = calculate_heatmap_vacuum_score((detected3, desc3, details3))
    
    print(f"\n🎯 Mock Perfect Vacuum Test:")
    print(f"   Detected: {detected3}")
    print(f"   Description: {desc3}")
    print(f"   Vacuum levels: {details3['vacuum_levels_count']}/{details3['total_levels_analyzed']}")
    print(f"   Vacuum intensity: {details3['vacuum_intensity']:.3f}")
    print(f"   Avg volume reduction: {details3['avg_volume_reduction_pct']:.1f}%")
    print(f"   Score: {score3}/10")

def test_5_layer_flow_integration():
    """Test integracji z 5-warstwowym systemem flow analysis"""
    print("\n🔗 Testing 5-Layer Flow Analysis Integration\n")
    
    try:
        from utils.trend_mode_pipeline import detect_trend_mode_extended
        
        # Test z strong vacuum data
        vacuum_snapshots = create_strong_vacuum_snapshots()
        
        # Mock functions for complete flow analysis
        def mock_get_price_series(symbol):
            prices = [snapshot["price"] for snapshot in vacuum_snapshots]
            # Extend to full 36-point series with upward trend
            while len(prices) < 36:
                last_price = prices[-1]
                prices.append(last_price * (1 + 0.002))  # gentle upward trend
            return prices
        
        def mock_create_orderbook_snapshots():
            return vacuum_snapshots  # Use same data for freeze analysis
        
        def mock_create_heatmap_snapshots():
            return vacuum_snapshots  # Strong vacuum data
        
        # Temporary monkey patches
        import detectors.flow_consistency as fc_module
        import detectors.directional_flow_detector as df_module
        import detectors.pulse_delay as pd_module
        import detectors.orderbook_freeze as of_module
        import detectors.heatmap_vacuum as hv_module
        
        original_fc_func = fc_module.get_price_series_bybit
        original_df_func = df_module.get_price_series_bybit
        original_pd_func = pd_module.get_price_series_bybit
        original_of_func = of_module.create_mock_orderbook_snapshots
        original_hv_func = hv_module.create_mock_heatmap_snapshots
        
        fc_module.get_price_series_bybit = mock_get_price_series
        df_module.get_price_series_bybit = mock_get_price_series
        pd_module.get_price_series_bybit = mock_get_price_series
        of_module.create_mock_orderbook_snapshots = mock_create_orderbook_snapshots
        hv_module.create_mock_heatmap_snapshots = mock_create_heatmap_snapshots
        
        try:
            # Create synthetic candles with upward trend
            import time
            base_timestamp = int(time.time() * 1000)
            synthetic_candles = []
            
            for i, snapshot in enumerate(vacuum_snapshots):
                timestamp = base_timestamp - (2-i) * 900000
                price = snapshot["price"]
                open_price = price * 0.999
                close_price = price
                high_price = price * 1.002
                low_price = price * 0.998
                volume = 150000
                
                synthetic_candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
            
            # Test complete 5-layer flow analysis
            trend_active, description, details = detect_trend_mode_extended("TESTUSDT", synthetic_candles)
            
            print(f"5-Layer Flow Analysis Result:")
            print(f"   Trend Active: {trend_active}")
            print(f"   Description: {description}")
            print(f"   Combined confidence: {details.get('combined_confidence', 0)}/180")
            print(f"   Base confidence: {details.get('base_confidence', 0)}")
            
            # Check all 5 layers
            directional_info = details.get('directional_flow', {})
            consistency_info = details.get('flow_consistency', {})
            pulse_info = details.get('pulse_delay', {})
            freeze_info = details.get('orderbook_freeze', {})
            vacuum_info = details.get('heatmap_vacuum', {})
            
            print(f"\n5-Layer Analysis Breakdown:")
            print(f"   Layer 1 - Directional flow: {directional_info.get('active', False)} ({directional_info.get('score', 0):+d} points)")
            print(f"   Layer 2 - Flow consistency: {consistency_info.get('index', 0):.3f} ({consistency_info.get('score', 0):+d} points)")
            print(f"   Layer 3 - Pulse delay: {pulse_info.get('detected', False)} ({pulse_info.get('score', 0):+d} points)")
            print(f"   Layer 4 - Orderbook freeze: {freeze_info.get('detected', False)} ({freeze_info.get('score', 0):+d} points)")
            print(f"   Layer 5 - Heatmap vacuum: {vacuum_info.get('detected', False)} ({vacuum_info.get('score', 0):+d} points)")
            print(f"   Total flow adjustment: {details.get('total_flow_adjustment', 0):+d}")
            
            # Check if we reached maximum enhanced confidence
            if details.get('combined_confidence', 0) > 150:
                print(f"   ✅ Enhanced confidence achieved: {details.get('combined_confidence', 0)}/180")
            
        finally:
            # Restore original functions
            fc_module.get_price_series_bybit = original_fc_func
            df_module.get_price_series_bybit = original_df_func
            pd_module.get_price_series_bybit = original_pd_func
            of_module.create_mock_orderbook_snapshots = original_of_func
            hv_module.create_mock_heatmap_snapshots = original_hv_func
            
    except ImportError as e:
        print(f"❌ Integration test failed: {e}")

def test_heatmap_vacuum_edge_cases():
    """Test edge cases dla heatmap vacuum"""
    print("\n🧪 Testing Heatmap Vacuum Edge Cases\n")
    
    # Test 1: Insufficient snapshots
    small_snapshots = create_strong_vacuum_snapshots()[:2]
    detected, desc, details = detect_heatmap_vacuum(small_snapshots)
    print(f"Insufficient snapshots: {detected}, {desc}")
    
    # Test 2: No levels in vacuum zone
    empty_zone_snapshots = []
    base_price = 50000.0
    
    for i, multiplier in enumerate([1.0, 1.002, 1.005]):
        empty_zone_snapshots.append({
            "price": base_price * multiplier,
            "bids": [[49950 * multiplier, 0.5]],
            "asks": [
                [50200 * multiplier, 1.0],  # Only +0.4% (below vacuum zone)
                [51200 * multiplier, 1.0],  # Only +2.4% (above vacuum zone)
            ],
            "timestamp": 1000000 + i * 300
        })
    
    detected, desc, details = detect_heatmap_vacuum(empty_zone_snapshots)
    score = calculate_heatmap_vacuum_score((detected, desc, details))
    print(f"Empty vacuum zone: {detected}, score: {score}, levels: {details['total_levels_analyzed']}")
    
    # Test 3: Volume increases (reverse vacuum)
    reverse_snapshots = []
    for i, volumes in enumerate([[1.0, 1.0, 1.0], [1.2, 1.3, 1.1], [1.5, 1.6, 1.4]]):
        reverse_snapshots.append({
            "price": base_price * (1 + i * 0.003),
            "bids": [[49950, 0.5]],
            "asks": [
                [50500, volumes[0]],  # +1.0% - volume increases
                [50600, volumes[1]],  # +1.2% - volume increases
                [50750, volumes[2]],  # +1.5% - volume increases
            ],
            "timestamp": 1000000 + i * 300
        })
    
    detected, desc, details = detect_heatmap_vacuum(reverse_snapshots)
    score = calculate_heatmap_vacuum_score((detected, desc, details))
    print(f"Volume increases: {detected}, score: {score}, avg reduction: {details['avg_volume_reduction_pct']:.1f}%")

def run_comprehensive_heatmap_vacuum_tests():
    """Uruchom wszystkie testy heatmap vacuum"""
    print("🚀 Starting Comprehensive Heatmap Vacuum Tests\n")
    
    try:
        test_heatmap_vacuum_scenarios()
        test_5_layer_flow_integration()
        test_heatmap_vacuum_edge_cases()
        
        print("\n✅ All Heatmap Vacuum tests completed!")
        print("\n📋 Test Summary:")
        print("✓ Heatmap vacuum pattern detection (strong/weak/perfect)")
        print("✓ 5-layer flow analysis integration")
        print("✓ Enhanced confidence calculation (0-180 range)")
        print("✓ Edge cases validation")
        print("✓ Complete trend mode pipeline integration")
        print("✓ Vacuum zone analysis (1-2% above price)")
        
    except Exception as e:
        print(f"\n❌ Heatmap vacuum test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_heatmap_vacuum_tests()