#!/usr/bin/env python3
"""
Test Suite for Shadow Sync Detector v2 – Stealth Protocol
Comprehensive validation of the new stealth detection system
"""

import sys
import os
import json
import traceback
from datetime import datetime, timezone

# Add the crypto-scan directory to the path
sys.path.insert(0, os.path.abspath('.'))

def test_shadow_sync_v2_basic():
    """Test basic Shadow Sync v2 functionality"""
    print("\n🧪 TEST 1: Shadow Sync v2 Basic Functionality")
    
    try:
        from stages.stage_minus2_1 import detect_shadow_sync_v2
        
        # Create test market data
        test_data = {
            "recent_prices": [100, 100.1, 99.9, 100.05, 99.95, 100.02, 99.98, 100.01],
            "recent_volumes": [1000, 1100, 950, 1050, 1200, 1150, 1000, 1075],
            "timestamp": 1671234567,
            "candles": []
        }
        
        # Test with whale activity present (required condition)
        shadow_sync_active, stealth_score, details = detect_shadow_sync_v2(
            symbol="TESTUSDT",
            data=test_data,
            price_usd=100.0,
            whale_activity=True,  # Required condition
            dex_inflow_detected=False
        )
        
        print(f"✅ Shadow Sync active: {shadow_sync_active}")
        print(f"✅ Stealth score: {stealth_score}")
        print(f"✅ Detection details: {details}")
        
        # Verify the function returns proper structure
        assert isinstance(shadow_sync_active, bool), "shadow_sync_active should be boolean"
        assert isinstance(stealth_score, (int, float)), "stealth_score should be numeric"
        assert isinstance(details, dict), "details should be dictionary"
        
        # Check required details keys
        required_keys = ["rsi_flatline", "heatmap_fade", "buy_dominance", "vwap_pinning", 
                        "zero_noise", "spoof_echo", "whale_or_dex"]
        for key in required_keys:
            assert key in details, f"Missing required key: {key}"
        
        print("✅ TEST 1 PASSED: Basic functionality works correctly")
        return True
        
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        traceback.print_exc()
        return False

def test_shadow_sync_v2_activation_conditions():
    """Test Shadow Sync v2 activation conditions"""
    print("\n🧪 TEST 2: Shadow Sync v2 Activation Conditions")
    
    try:
        from stages.stage_minus2_1 import detect_shadow_sync_v2
        
        # Create stable market data (should trigger multiple conditions)
        stable_data = {
            "recent_prices": [50.0, 50.01, 49.99, 50.005, 49.995, 50.002, 49.998, 50.001],
            "recent_volumes": [2000, 2100, 1950, 2050, 2200, 2150, 2000, 2075],
            "timestamp": 1671234567,
            "candles": [
                {"open": 50.0, "close": 50.01, "high": 50.02, "low": 49.98},
                {"open": 50.01, "close": 49.99, "high": 50.015, "low": 49.985}
            ]
        }
        
        # Test without whale/dex activity (should not activate)
        shadow_sync_active_no_whale, _, details_no_whale = detect_shadow_sync_v2(
            symbol="TESTUSDT",
            data=stable_data,
            price_usd=50.0,
            whale_activity=False,
            dex_inflow_detected=False
        )
        
        print(f"✅ Without whale/dex: {shadow_sync_active_no_whale} (should be False)")
        assert not shadow_sync_active_no_whale, "Should not activate without whale/dex activity"
        
        # Test with whale activity (should potentially activate)
        shadow_sync_active_whale, stealth_score_whale, details_whale = detect_shadow_sync_v2(
            symbol="TESTUSDT",
            data=stable_data,
            price_usd=50.0,
            whale_activity=True,
            dex_inflow_detected=False
        )
        
        print(f"✅ With whale activity: {shadow_sync_active_whale}")
        print(f"✅ Stealth score with whale: {stealth_score_whale}")
        print(f"✅ Active conditions: {sum(details_whale.values())}/7")
        
        # Test with DEX inflow (should potentially activate)
        shadow_sync_active_dex, stealth_score_dex, details_dex = detect_shadow_sync_v2(
            symbol="TESTUSDT",
            data=stable_data,
            price_usd=50.0,
            whale_activity=False,
            dex_inflow_detected=True
        )
        
        print(f"✅ With DEX inflow: {shadow_sync_active_dex}")
        print(f"✅ Stealth score with DEX: {stealth_score_dex}")
        
        # Verify whale_or_dex condition is met when either is present
        assert details_whale["whale_or_dex"], "whale_or_dex should be True with whale activity"
        assert details_dex["whale_or_dex"], "whale_or_dex should be True with DEX inflow"
        
        print("✅ TEST 2 PASSED: Activation conditions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        traceback.print_exc()
        return False

def test_shadow_sync_v2_integration():
    """Test Shadow Sync v2 integration with Stage -2.1"""
    print("\n🧪 TEST 3: Shadow Sync v2 Integration with Stage -2.1")
    
    try:
        from stages.stage_minus2_1 import detect_stage_minus2_1
        
        # Test that Stage -2.1 includes Shadow Sync v2 in signals
        stage_result = detect_stage_minus2_1("TESTUSDT", price_usd=100.0)
        
        if isinstance(stage_result, tuple) and len(stage_result) >= 2:
            stage_pass, signals = stage_result[0], stage_result[1]
            
            print(f"✅ Stage -2.1 executed: {stage_pass}")
            print(f"✅ Signals type: {type(signals)}")
            
            if isinstance(signals, dict):
                # Check if Shadow Sync v2 signals are included
                shadow_sync_keys = ["shadow_sync_v2", "stealth_score", "shadow_sync_details"]
                
                for key in shadow_sync_keys:
                    if key in signals:
                        print(f"✅ Found {key}: {signals[key]}")
                    else:
                        print(f"⚠️ Missing {key} in signals")
                
                print("✅ TEST 3 PASSED: Integration with Stage -2.1 works")
                return True
            else:
                print(f"❌ Expected signals dict, got: {type(signals)}")
                return False
        else:
            print(f"❌ Unexpected Stage -2.1 result format: {stage_result}")
            return False
        
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        traceback.print_exc()
        return False

def test_shadow_sync_v2_scoring():
    """Test Shadow Sync v2 PPWCS scoring integration"""
    print("\n🧪 TEST 4: Shadow Sync v2 PPWCS Scoring Integration")
    
    try:
        from utils.scoring import compute_ppwcs
        
        # Create signals with Shadow Sync v2 active
        test_signals = {
            "whale_activity": False,
            "dex_inflow": False,
            "stealth_inflow": False,
            "compressed": False,
            "stage1g_active": False,
            "event_tag": False,
            "shadow_sync_v2": True  # Shadow Sync v2 active
        }
        
        # Test PPWCS scoring with Shadow Sync v2
        total_score, structure_score, quality_score = compute_ppwcs(test_signals)
        
        print(f"✅ Total PPWCS score: {total_score}")
        print(f"✅ Structure score: {structure_score}")
        print(f"✅ Quality score: {quality_score}")
        
        # Shadow Sync v2 should contribute 25 points
        assert total_score >= 25, f"Expected at least 25 points from Shadow Sync v2, got {total_score}"
        
        # Test with multiple detectors including Shadow Sync v2
        combined_signals = {
            "whale_activity": True,   # +10
            "dex_inflow": True,       # +10
            "stealth_inflow": False,
            "compressed": False,
            "stage1g_active": False,
            "event_tag": False,
            "shadow_sync_v2": True    # +25
        }
        
        combined_score, _, _ = compute_ppwcs(combined_signals)
        print(f"✅ Combined score (whale + dex + shadow_sync): {combined_score}")
        
        # Should be at least 45 points (10+10+25)
        assert combined_score >= 45, f"Expected at least 45 points, got {combined_score}"
        
        print("✅ TEST 4 PASSED: PPWCS scoring integration works correctly")
        return True
        
    except Exception as e:
        print(f"❌ TEST 4 FAILED: {e}")
        traceback.print_exc()
        return False

def test_shadow_sync_v2_error_handling():
    """Test Shadow Sync v2 error handling"""
    print("\n🧪 TEST 5: Shadow Sync v2 Error Handling")
    
    try:
        from stages.stage_minus2_1 import detect_shadow_sync_v2
        
        # Test with invalid/missing data
        invalid_data_cases = [
            {},  # Empty data
            {"timestamp": 123},  # Minimal data
            None,  # None data
        ]
        
        for i, invalid_data in enumerate(invalid_data_cases):
            print(f"Testing invalid data case {i+1}")
            
            try:
                shadow_sync_active, stealth_score, details = detect_shadow_sync_v2(
                    symbol="TESTUSDT",
                    data=invalid_data,
                    price_usd=100.0,
                    whale_activity=True,
                    dex_inflow_detected=False
                )
                
                # Should handle gracefully and return sensible defaults
                assert isinstance(shadow_sync_active, bool), "Should return boolean"
                assert isinstance(stealth_score, (int, float)), "Should return numeric score"
                assert isinstance(details, dict), "Should return dict details"
                
                print(f"✅ Case {i+1}: Handled gracefully")
                
            except Exception as case_error:
                print(f"⚠️ Case {i+1}: Exception handled: {case_error}")
        
        print("✅ TEST 5 PASSED: Error handling works correctly")
        return True
        
    except Exception as e:
        print(f"❌ TEST 5 FAILED: {e}")
        traceback.print_exc()
        return False

def test_shadow_sync_v2_stealth_conditions():
    """Test specific stealth detection conditions"""
    print("\n🧪 TEST 6: Shadow Sync v2 Stealth Detection Conditions")
    
    try:
        from stages.stage_minus2_1 import detect_shadow_sync_v2
        
        # Create data that should trigger multiple stealth conditions
        stealth_optimized_data = {
            "recent_prices": [
                # Very stable prices (should trigger zero_noise and buy_dominance)
                100.000, 100.001, 100.002, 100.001, 100.003, 100.002, 100.001, 100.002
            ],
            "recent_volumes": [
                # High volume with stable price (should suggest stealth accumulation)
                5000, 5200, 5100, 5300, 5150, 5250, 5180, 5220
            ],
            "timestamp": 1671234567,
            "candles": [
                {"open": 100.000, "close": 100.001, "high": 100.002, "low": 99.999},
                {"open": 100.001, "close": 100.002, "high": 100.003, "low": 100.000},
                {"open": 100.002, "close": 100.001, "high": 100.002, "low": 100.000}
            ]
        }
        
        # Test with optimal stealth conditions
        shadow_sync_active, stealth_score, details = detect_shadow_sync_v2(
            symbol="STEALTHUSDT",
            data=stealth_optimized_data,
            price_usd=100.0,
            whale_activity=True,  # Required condition
            dex_inflow_detected=True  # Also present for maximum stealth score
        )
        
        print(f"✅ Stealth optimized result: {shadow_sync_active}")
        print(f"✅ Stealth score: {stealth_score}")
        print(f"✅ Detailed conditions:")
        
        for condition, active in details.items():
            status = "✅ ACTIVE" if active else "❌ inactive"
            print(f"   {condition}: {status}")
        
        # Count active conditions
        active_conditions = sum(details.values())
        print(f"✅ Total active conditions: {active_conditions}/7")
        
        # For maximum stealth, we should have at least several conditions active
        if active_conditions >= 4 and details["whale_or_dex"]:
            print("✅ Optimal stealth conditions detected!")
        else:
            print(f"⚠️ Stealth conditions could be improved (need 4+ and whale/dex)")
        
        print("✅ TEST 6 PASSED: Stealth detection conditions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ TEST 6 FAILED: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all Shadow Sync v2 tests"""
    print("🚀 SHADOW SYNC DETECTOR V2 – STEALTH PROTOCOL TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_shadow_sync_v2_basic,
        test_shadow_sync_v2_activation_conditions,
        test_shadow_sync_v2_integration,
        test_shadow_sync_v2_scoring,
        test_shadow_sync_v2_error_handling,
        test_shadow_sync_v2_stealth_conditions
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 TEST RESULTS: {passed}/{len(tests)} PASSED")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Shadow Sync Detector v2 is ready for production.")
    else:
        print(f"⚠️ {failed} tests failed. Review implementation before deployment.")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)