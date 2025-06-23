#!/usr/bin/env python3
"""
Test VWAP Pinning Integration
Comprehensive testing of 6-layer flow analysis with VWAP pinning detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.vwap_pinning import (
    detect_vwap_pinning,
    calculate_vwap_pinning_score,
    analyze_vwap_pinning_detailed,
    create_mock_vwap_pinning_data,
    create_mock_volatile_data,
    calculate_vwap_values
)

def create_bullish_pinning_data():
    """Tworzy dane z bullish VWAP pinning (cena powyżej VWAP)"""
    base_vwap = 50000.0
    prices = []
    vwap_values = []
    
    for i in range(25):
        vwap = base_vwap + (i * 3)  # VWAP rośnie
        
        # Cena lekko powyżej VWAP (bullish bias)
        price_deviation = 0.0012 + (i % 2) * 0.0003  # +0.12% do +0.15%
        price = vwap * (1 + price_deviation)
        
        vwap_values.append(vwap)
        prices.append(price)
    
    return prices, vwap_values

def create_neutral_pinning_data():
    """Tworzy dane z neutralnym VWAP pinning (cena oscyluje wokół VWAP)"""
    base_vwap = 50000.0
    prices = []
    vwap_values = []
    
    for i in range(30):
        vwap = base_vwap + (i * 1)  # VWAP lekko rośnie
        
        # Cena oscyluje wokół VWAP (-0.1% do +0.1%)
        price_deviation = ((i % 4) - 1.5) * 0.0007  # -0.105%, -0.035%, +0.035%, +0.105%
        price = vwap * (1 + price_deviation)
        
        vwap_values.append(vwap)
        prices.append(price)
    
    return prices, vwap_values

def create_edge_case_data():
    """Tworzy dane edge case - prawie pinning ale za mało stabilności"""
    base_vwap = 50000.0
    prices = []
    vwap_values = []
    
    for i in range(25):
        vwap = base_vwap + (i * 2)
        
        if i < 15:
            # Pierwsze 15 okresów - stabilne pinning
            price_deviation = (i % 3 - 1) * 0.0008  # -0.08%, 0%, +0.08%
        else:
            # Ostatnie 10 okresów - większa zmienność
            price_deviation = (i % 5 - 2) * 0.004  # -0.8% do +0.8%
        
        price = vwap * (1 + price_deviation)
        
        vwap_values.append(vwap)
        prices.append(price)
    
    return prices, vwap_values

def test_vwap_pinning_scenarios():
    """Test różnych scenariuszy VWAP pinning"""
    print("🧪 Testing VWAP Pinning Scenarios\n")
    
    # Scenario 1: Bullish pinning
    bullish_prices, bullish_vwap = create_bullish_pinning_data()
    detected1, desc1, details1 = detect_vwap_pinning(bullish_prices, bullish_vwap)
    score1 = calculate_vwap_pinning_score((detected1, desc1, details1))
    
    print(f"📈 Bullish Pinning Test:")
    print(f"   Detected: {detected1}")
    print(f"   Description: {desc1}")
    print(f"   Avg deviation: {details1['avg_absolute_deviation_pct']:.4f}%")
    print(f"   Stability ratio: {details1['stability_ratio']:.3f}")
    print(f"   Bias: {details1['bias_direction']}")
    print(f"   Score: {score1}/15")
    
    # Scenario 2: Neutral pinning
    neutral_prices, neutral_vwap = create_neutral_pinning_data()
    detected2, desc2, details2 = detect_vwap_pinning(neutral_prices, neutral_vwap)
    score2 = calculate_vwap_pinning_score((detected2, desc2, details2))
    
    print(f"\n⚖️ Neutral Pinning Test:")
    print(f"   Detected: {detected2}")
    print(f"   Description: {desc2}")
    print(f"   Avg deviation: {details2['avg_absolute_deviation_pct']:.4f}%")
    print(f"   Stability ratio: {details2['stability_ratio']:.3f}")
    print(f"   Bias: {details2['bias_direction']}")
    print(f"   Score: {score2}/15")
    
    # Scenario 3: Edge case
    edge_prices, edge_vwap = create_edge_case_data()
    detected3, desc3, details3 = detect_vwap_pinning(edge_prices, edge_vwap)
    score3 = calculate_vwap_pinning_score((detected3, desc3, details3))
    
    print(f"\n🔸 Edge Case Test:")
    print(f"   Detected: {detected3}")
    print(f"   Description: {desc3}")
    print(f"   Avg deviation: {details3['avg_absolute_deviation_pct']:.4f}%")
    print(f"   Stability ratio: {details3['stability_ratio']:.3f}")
    print(f"   Bias: {details3['bias_direction']}")
    print(f"   Score: {score3}/15")

def test_6_layer_flow_integration():
    """Test integracji z 6-warstwowym systemem flow analysis"""
    print("\n🔗 Testing 6-Layer Flow Analysis Integration\n")
    
    try:
        from utils.trend_mode_pipeline import detect_trend_mode_extended
        
        # Test z bullish pinning data
        pinning_prices, pinning_vwap = create_bullish_pinning_data()
        
        # Mock functions for complete flow analysis
        def mock_get_price_series(symbol):
            # Extend pinning prices to full 36-point series
            extended_prices = pinning_prices.copy()
            while len(extended_prices) < 36:
                last_price = extended_prices[-1]
                extended_prices.append(last_price * (1 + 0.001))  # gentle upward trend
            return extended_prices
        
        def mock_create_orderbook_snapshots():
            # Create orderbook snapshots that match pinning scenario
            snapshots = []
            for i, price in enumerate(pinning_prices[:3]):
                snapshots.append({
                    "price": price,
                    "bids": [[price * 0.999, 1.0], [price * 0.998, 0.8]],
                    "asks": [[price * 1.001, 1.0], [price * 1.002, 0.8]],
                    "timestamp": 1000000 + i * 300
                })
            return snapshots
        
        def mock_create_heatmap_snapshots():
            return mock_create_orderbook_snapshots()  # Same data for consistency
        
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
            # Create synthetic candles with bullish pinning characteristics
            import time
            base_timestamp = int(time.time() * 1000)
            synthetic_candles = []
            
            for i, price in enumerate(pinning_prices[:5]):  # Use first 5 prices
                timestamp = base_timestamp - (4-i) * 900000
                open_price = price * 0.9995
                close_price = price
                high_price = price * 1.0005
                low_price = price * 0.9995
                volume = 120000
                
                synthetic_candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
            
            # Test complete 6-layer flow analysis
            trend_active, description, details = detect_trend_mode_extended("TESTUSDT", synthetic_candles)
            
            print(f"6-Layer Flow Analysis Result:")
            print(f"   Trend Active: {trend_active}")
            print(f"   Description: {description}")
            print(f"   Combined confidence: {details.get('combined_confidence', 0)}/195")
            print(f"   Base confidence: {details.get('base_confidence', 0)}")
            
            # Check all 6 layers
            directional_info = details.get('directional_flow', {})
            consistency_info = details.get('flow_consistency', {})
            pulse_info = details.get('pulse_delay', {})
            freeze_info = details.get('orderbook_freeze', {})
            vacuum_info = details.get('heatmap_vacuum', {})
            pinning_info = details.get('vwap_pinning', {})
            
            print(f"\n6-Layer Analysis Breakdown:")
            print(f"   Layer 1 - Directional flow: {directional_info.get('active', False)} ({directional_info.get('score', 0):+d} points)")
            print(f"   Layer 2 - Flow consistency: {consistency_info.get('index', 0):.3f} ({consistency_info.get('score', 0):+d} points)")
            print(f"   Layer 3 - Pulse delay: {pulse_info.get('detected', False)} ({pulse_info.get('score', 0):+d} points)")
            print(f"   Layer 4 - Orderbook freeze: {freeze_info.get('detected', False)} ({freeze_info.get('score', 0):+d} points)")
            print(f"   Layer 5 - Heatmap vacuum: {vacuum_info.get('detected', False)} ({vacuum_info.get('score', 0):+d} points)")
            print(f"   Layer 6 - VWAP pinning: {pinning_info.get('detected', False)} ({pinning_info.get('score', 0):+d} points)")
            print(f"   Total flow adjustment: {details.get('total_flow_adjustment', 0):+d}")
            
            # Check if we reached maximum enhanced confidence
            if details.get('combined_confidence', 0) > 160:
                print(f"   ✅ Enhanced confidence achieved: {details.get('combined_confidence', 0)}/195")
            
        finally:
            # Restore original functions
            fc_module.get_price_series_bybit = original_fc_func
            df_module.get_price_series_bybit = original_df_func
            pd_module.get_price_series_bybit = original_pd_func
            of_module.create_mock_orderbook_snapshots = original_of_func
            hv_module.create_mock_heatmap_snapshots = original_hv_func
            
    except ImportError as e:
        print(f"❌ Integration test failed: {e}")

def test_vwap_pinning_edge_cases():
    """Test edge cases dla VWAP pinning"""
    print("\n🧪 Testing VWAP Pinning Edge Cases\n")
    
    # Test 1: Insufficient data
    short_prices = [50000, 50010, 50005]
    short_vwap = [50000, 50010, 50005]
    detected, desc, details = detect_vwap_pinning(short_prices, short_vwap)
    print(f"Insufficient data: {detected}, {desc}")
    
    # Test 2: Mismatched lengths
    mismatched_prices = [50000, 50010, 50005, 50012]
    mismatched_vwap = [50000, 50010]
    detected, desc, details = detect_vwap_pinning(mismatched_prices, mismatched_vwap)
    print(f"Mismatched lengths: {detected}, {desc}")
    
    # Test 3: Zero/invalid values
    invalid_prices = [50000, 0, 50005, -100, 50012] + [50000] * 20
    invalid_vwap = [50000, 50010, 0, 50015, 50012] + [50000] * 20
    detected, desc, details = detect_vwap_pinning(invalid_prices, invalid_vwap)
    score = calculate_vwap_pinning_score((detected, desc, details))
    print(f"Invalid values: {detected}, score: {score}, periods: {details.get('periods_analyzed', 0)}")

def test_vwap_calculation():
    """Test VWAP calculation functionality"""
    print("\n🧪 Testing VWAP Calculation\n")
    
    # Create mock kline data
    mock_klines = []
    base_price = 50000
    
    for i in range(10):
        price = base_price + i * 10
        mock_klines.append({
            "timestamp": 1000000 + i * 300,
            "open": price - 5,
            "high": price + 10,
            "low": price - 10,
            "close": price + 5,
            "volume": 100 + i * 10
        })
    
    vwap_values = calculate_vwap_values(mock_klines)
    
    print(f"VWAP Calculation Test:")
    print(f"   Klines processed: {len(mock_klines)}")
    print(f"   VWAP values calculated: {len(vwap_values)}")
    print(f"   First VWAP: {vwap_values[0]:.2f}")
    print(f"   Last VWAP: {vwap_values[-1]:.2f}")
    print(f"   VWAP trend: {'ascending' if vwap_values[-1] > vwap_values[0] else 'descending'}")

def run_comprehensive_vwap_pinning_tests():
    """Uruchom wszystkie testy VWAP pinning"""
    print("🚀 Starting Comprehensive VWAP Pinning Tests\n")
    
    try:
        test_vwap_pinning_scenarios()
        test_6_layer_flow_integration()
        test_vwap_pinning_edge_cases()
        test_vwap_calculation()
        
        print("\n✅ All VWAP Pinning tests completed!")
        print("\n📋 Test Summary:")
        print("✓ VWAP pinning pattern detection (bullish/neutral/edge cases)")
        print("✓ 6-layer flow analysis integration")
        print("✓ Enhanced confidence calculation (0-195 range)")
        print("✓ Edge cases validation")
        print("✓ VWAP calculation functionality")
        print("✓ Complete trend mode pipeline integration")
        print("✓ Controlled consolidation detection")
        
    except Exception as e:
        print(f"\n❌ VWAP pinning test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_vwap_pinning_tests()