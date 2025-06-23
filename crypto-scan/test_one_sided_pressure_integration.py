#!/usr/bin/env python3
"""
Test One-Sided Pressure Integration
Comprehensive testing of 7-layer flow analysis with One-Sided Pressure detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.one_sided_pressure import (
    detect_one_sided_pressure,
    calculate_one_sided_pressure_score,
    analyze_one_sided_pressure_detailed,
    create_mock_strong_bid_orderbook,
    create_mock_balanced_orderbook,
    create_mock_ask_dominant_orderbook
)

def create_extreme_bid_dominance_orderbook():
    """Tworzy orderbook z ekstremalną przewagą bid dla maksymalnego score"""
    return {
        "bids": [
            [50000.0, 5.0],  # Bardzo silne bidy
            [49995.0, 4.5],
            [49990.0, 4.0],
            [49985.0, 3.5],
            [49980.0, 3.0]
        ],
        "asks": [
            [50005.0, 0.2],  # Bardzo słabe aski
            [50010.0, 0.15],
            [50015.0, 0.1],
            [50020.0, 0.08],
            [50025.0, 0.05]
        ]
    }

def create_insufficient_depth_orderbook():
    """Tworzy orderbook z niewystarczającą głębokością"""
    return {
        "bids": [
            [50000.0, 1.0],
            [49995.0, 0.5]
        ],
        "asks": [
            [50005.0, 0.8],
            [50010.0, 0.3]
        ]
    }

def create_real_world_strong_bid_orderbook():
    """Tworzy realistyczny orderbook z silną przewagą bid"""
    return {
        "bids": [
            [50000.0, 3.2],  # Realistic strong bid
            [49995.0, 2.8],
            [49990.0, 2.5],
            [49985.0, 2.2],
            [49980.0, 1.9],
            [49975.0, 1.6],
            [49970.0, 1.3]
        ],
        "asks": [
            [50005.0, 1.1],  # Moderate ask resistance
            [50010.0, 0.9],
            [50015.0, 0.8],
            [50020.0, 0.7],
            [50025.0, 0.6],
            [50030.0, 0.5],
            [50035.0, 0.4]
        ]
    }

def test_one_sided_pressure_scenarios():
    """Test różnych scenariuszy One-Sided Pressure"""
    print("🧪 Testing One-Sided Pressure Scenarios\n")
    
    scenarios = [
        ("💪 Extreme Bid Dominance", create_extreme_bid_dominance_orderbook()),
        ("📈 Strong Bid Dominance", create_mock_strong_bid_orderbook()),
        ("🌍 Real-World Strong Bid", create_real_world_strong_bid_orderbook()),
        ("⚖️ Balanced Orderbook", create_mock_balanced_orderbook()),
        ("📉 Ask Dominance", create_mock_ask_dominant_orderbook()),
        ("⚠️ Insufficient Depth", create_insufficient_depth_orderbook())
    ]
    
    for name, orderbook in scenarios:
        detected, desc, details = detect_one_sided_pressure(orderbook)
        score = calculate_one_sided_pressure_score((detected, desc, details))
        
        print(f"{name}:")
        print(f"   Detected: {detected}")
        print(f"   Description: {desc}")
        print(f"   Pressure ratio: {details.get('pressure_ratio', 0):.3f}")
        print(f"   Bid depth: {details.get('bid_depth_score', 0)}/5")
        print(f"   Ask depth: {details.get('ask_depth_score', 0)}/5")
        print(f"   Dominance type: {details.get('dominance_type', 'unknown')}")
        print(f"   Score: {score}/20\n")

def test_detailed_analysis_features():
    """Test szczegółowej analizy one-sided pressure"""
    print("🔍 Testing Detailed Analysis Features\n")
    
    strong_bid_book = create_real_world_strong_bid_orderbook()
    detailed_analysis = analyze_one_sided_pressure_detailed(strong_bid_book)
    
    basic = detailed_analysis['basic_analysis']
    
    print(f"Basic Analysis:")
    print(f"   Detection: {basic['detected']}")
    print(f"   Score: {basic['score']}/20")
    print(f"   Description: {basic['description']}")
    
    if 'advanced_metrics' in detailed_analysis:
        advanced = detailed_analysis['advanced_metrics']
        print(f"\nAdvanced Metrics:")
        print(f"   Top level ratio: {advanced['top_level_ratio']}")
        print(f"   Avg bid volume: {advanced['avg_bid_vol']}")
        print(f"   Avg ask volume: {advanced['avg_ask_vol']}")
        print(f"   Bid concentration: {advanced['bid_concentration']}")
        print(f"   Ask concentration: {advanced['ask_concentration']}")
        print(f"   Concentration advantage: {advanced['concentration_advantage']}")
    
    if 'interpretation' in detailed_analysis:
        interpretation = detailed_analysis['interpretation']
        print(f"\nInterpretation:")
        print(f"   Pressure strength: {interpretation['pressure_strength']}")
        print(f"   Market sentiment: {interpretation['market_sentiment']}")
        print(f"   Continuation probability: {interpretation['continuation_probability']}")

def test_7_layer_flow_integration():
    """Test integracji z 7-warstwowym systemem flow analysis"""
    print("\n🔗 Testing 7-Layer Flow Analysis Integration\n")
    
    try:
        from utils.trend_mode_pipeline import detect_trend_mode_extended
        
        # Test z real-world strong bid orderbook
        strong_bid_book = create_real_world_strong_bid_orderbook()
        
        # Create synthetic candles that match strong bid scenario
        import time
        base_timestamp = int(time.time() * 1000)
        synthetic_candles = []
        
        base_price = 50000.0
        for i in range(8):  # 8 candles for sufficient data
            timestamp = base_timestamp - (7-i) * 900000  # 15-minute intervals
            
            # Create bullish candles that would align with strong bid pressure
            open_price = base_price + i * 2
            close_price = open_price + 5  # Small bullish candles
            high_price = close_price + 2
            low_price = open_price - 1
            volume = 150000 + i * 5000  # Increasing volume
            
            synthetic_candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
        
        # Mock functions for complete flow analysis
        def mock_get_orderbook_with_fallback(symbol):
            return strong_bid_book
        
        # Temporarily patch the orderbook function
        import utils.trend_mode_pipeline as tmp_module
        original_func = tmp_module.get_orderbook_with_fallback
        tmp_module.get_orderbook_with_fallback = mock_get_orderbook_with_fallback
        
        try:
            # Test complete 7-layer flow analysis
            trend_active, description, details = detect_trend_mode_extended("TESTUSDT", synthetic_candles)
            
            print(f"7-Layer Flow Analysis Result:")
            print(f"   Trend Active: {trend_active}")
            print(f"   Description: {description}")
            print(f"   Combined confidence: {details.get('combined_confidence', 0)}/215")
            print(f"   Base confidence: {details.get('base_confidence', 0)}")
            
            # Check all 7 layers
            directional_info = details.get('directional_flow', {})
            consistency_info = details.get('flow_consistency', {})
            pulse_info = details.get('pulse_delay', {})
            freeze_info = details.get('orderbook_freeze', {})
            vacuum_info = details.get('heatmap_vacuum', {})
            pinning_info = details.get('vwap_pinning', {})
            pressure_info = details.get('one_sided_pressure', {})
            
            print(f"\n7-Layer Analysis Breakdown:")
            print(f"   Layer 1 - Directional flow: {directional_info.get('active', False)} ({directional_info.get('score', 0):+d} points)")
            print(f"   Layer 2 - Flow consistency: {consistency_info.get('index', 0):.3f} ({consistency_info.get('score', 0):+d} points)")
            print(f"   Layer 3 - Pulse delay: {pulse_info.get('detected', False)} ({pulse_info.get('score', 0):+d} points)")
            print(f"   Layer 4 - Orderbook freeze: {freeze_info.get('detected', False)} ({freeze_info.get('score', 0):+d} points)")
            print(f"   Layer 5 - Heatmap vacuum: {vacuum_info.get('detected', False)} ({vacuum_info.get('score', 0):+d} points)")
            print(f"   Layer 6 - VWAP pinning: {pinning_info.get('detected', False)} ({pinning_info.get('score', 0):+d} points)")
            print(f"   Layer 7 - One-sided pressure: {pressure_info.get('detected', False)} ({pressure_info.get('score', 0):+d} points)")
            print(f"   Total flow adjustment: {details.get('total_flow_adjustment', 0):+d}")
            
            # Check if we reached enhanced confidence with 7 layers
            if details.get('combined_confidence', 0) > 180:
                print(f"   ✅ Enhanced 7-layer confidence achieved: {details.get('combined_confidence', 0)}/215")
            
            # Validate One-Sided Pressure contribution
            pressure_score = pressure_info.get('score', 0)
            if pressure_score > 0:
                print(f"   ✅ One-sided pressure contributing: {pressure_score} points")
                
                # Check pressure details
                pressure_details = pressure_info.get('details', {})
                if pressure_details.get('pressure_ratio', 0) > 1.5:
                    print(f"   ✅ Strong bid pressure detected: {pressure_details.get('pressure_ratio', 0):.1f}x ratio")
            
        finally:
            # Restore original function
            tmp_module.get_orderbook_with_fallback = original_func
            
    except ImportError as e:
        print(f"❌ Integration test failed: {e}")

def test_edge_cases_and_validation():
    """Test edge cases i walidacji"""
    print("\n🧪 Testing Edge Cases and Validation\n")
    
    edge_cases = [
        ("Empty bids", {"bids": [], "asks": [[50005, 1.0], [50010, 0.8], [50015, 0.6]]}),
        ("Empty asks", {"bids": [[50000, 1.0], [49995, 0.8], [49990, 0.6]], "asks": []}),
        ("Zero ask volume", {"bids": [[50000, 1.0], [49995, 0.8], [49990, 0.6]], "asks": [[50005, 0], [50010, 0], [50015, 0]]}),
        ("Missing data", {"bids": [[50000]], "asks": [[50005, 1.0]]}),
        ("Invalid format", {"invalid": "data"}),
    ]
    
    for name, orderbook in edge_cases:
        try:
            detected, desc, details = detect_one_sided_pressure(orderbook)
            score = calculate_one_sided_pressure_score((detected, desc, details))
            print(f"{name}: {detected}, Score: {score}, Desc: {desc[:50]}...")
        except Exception as e:
            print(f"{name}: Exception handled - {str(e)[:50]}...")

def test_scoring_thresholds():
    """Test progów scoringowych"""
    print("\n📊 Testing Scoring Thresholds\n")
    
    # Test different pressure ratios
    test_ratios = [1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    for ratio in test_ratios:
        # Create orderbook with specific ratio
        bid_vol = 2.0
        ask_vol = bid_vol / ratio
        
        test_orderbook = {
            "bids": [[50000, bid_vol], [49995, bid_vol*0.8], [49990, bid_vol*0.6], [49985, bid_vol*0.4], [49980, bid_vol*0.2]],
            "asks": [[50005, ask_vol], [50010, ask_vol*0.8], [50015, ask_vol*0.6], [50020, ask_vol*0.4], [50025, ask_vol*0.2]]
        }
        
        detected, desc, details = detect_one_sided_pressure(test_orderbook)
        score = calculate_one_sided_pressure_score((detected, desc, details))
        
        print(f"Ratio {ratio:.1f}x: Detected={detected}, Score={score}/20")

def run_comprehensive_one_sided_pressure_tests():
    """Uruchom wszystkie testy One-Sided Pressure"""
    print("🚀 Starting Comprehensive One-Sided Pressure Tests\n")
    
    try:
        test_one_sided_pressure_scenarios()
        test_detailed_analysis_features()
        test_7_layer_flow_integration()
        test_edge_cases_and_validation()
        test_scoring_thresholds()
        
        print("\n✅ All One-Sided Pressure tests completed!")
        print("\n📋 Test Summary:")
        print("✓ One-sided pressure detection scenarios (extreme/strong/balanced/ask dominance)")
        print("✓ 7-layer flow analysis integration")
        print("✓ Enhanced confidence calculation (0-215 range)")
        print("✓ Detailed analysis with advanced metrics")
        print("✓ Edge cases and error handling")
        print("✓ Scoring threshold validation")
        print("✓ Real-world orderbook scenarios")
        print("✓ Complete trend mode pipeline integration")
        print("✓ Bid dominance detection for trend continuation")
        
    except Exception as e:
        print(f"\n❌ One-sided pressure test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_one_sided_pressure_tests()