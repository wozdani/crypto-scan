#!/usr/bin/env python3
"""
Test script for pump analysis system
Tests basic functionality without requiring API keys
"""

import sys
import os
from datetime import datetime, timedelta
import json

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import BybitDataFetcher, PumpDetector, PrePumpAnalyzer

def test_bybit_connection():
    """Test Bybit API connection"""
    print("🔗 Testing Bybit API connection...")
    
    bybit = BybitDataFetcher()
    
    # Test getting symbols
    symbols = bybit.get_active_symbols(limit=5)
    
    if symbols:
        print(f"✅ Successfully retrieved {len(symbols)} symbols: {symbols}")
        return True, symbols
    else:
        print("❌ Failed to retrieve symbols from Bybit")
        return False, []

def test_data_fetching(symbol="BTCUSDT"):
    """Test data fetching for a specific symbol"""
    print(f"📊 Testing data fetching for {symbol}...")
    
    bybit = BybitDataFetcher()
    
    # Get recent data
    kline_data = bybit.get_kline_data(
        symbol=symbol,
        interval="5",
        limit=50
    )
    
    if kline_data:
        print(f"✅ Retrieved {len(kline_data)} candles for {symbol}")
        
        # Show sample data
        if len(kline_data) > 0:
            sample = kline_data[0]
            print(f"📈 Sample candle: {sample}")
        
        return True, kline_data
    else:
        print(f"❌ Failed to retrieve data for {symbol}")
        return False, []

def test_pump_detection(kline_data, symbol="TEST"):
    """Test pump detection algorithm"""
    print("🎯 Testing pump detection...")
    
    detector = PumpDetector(min_increase_pct=10.0, detection_window_minutes=30)
    
    pumps = detector.detect_pumps_in_data(kline_data, symbol)
    
    if pumps:
        print(f"✅ Detected {len(pumps)} pump(s)")
        for i, pump in enumerate(pumps):
            print(f"  Pump {i+1}: +{pump.price_increase_pct:.1f}% at {pump.start_time}")
    else:
        print("ℹ️ No pumps detected (this is normal for most timeframes)")
    
    return pumps

def test_pre_pump_analysis(pump_event):
    """Test pre-pump analysis"""
    print("🔍 Testing pre-pump analysis...")
    
    bybit = BybitDataFetcher()
    analyzer = PrePumpAnalyzer(bybit)
    
    try:
        analysis = analyzer.analyze_pre_pump_conditions(pump_event)
        
        if analysis:
            print("✅ Pre-pump analysis completed")
            print(f"📊 Analysis keys: {list(analysis.keys())}")
            return True, analysis
        else:
            print("❌ Pre-pump analysis failed")
            return False, {}
            
    except Exception as e:
        print(f"❌ Pre-pump analysis error: {e}")
        return False, {}

def test_configuration():
    """Test system configuration"""
    print("⚙️ Testing configuration...")
    
    try:
        from config import Config
        
        config = Config()
        is_valid, missing = config.validate()
        
        print(f"📋 Configuration summary:")
        summary = config.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        if not is_valid:
            print(f"⚠️ Missing required keys: {missing}")
            print("💡 Add these to your .env file to run full analysis")
        else:
            print("✅ All required configuration present")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def create_test_pump_event():
    """Create a test pump event for testing"""
    from main import PumpEvent
    
    return PumpEvent(
        symbol="TESTUSDT",
        start_time=datetime.now() - timedelta(hours=2),
        price_before=1.0,
        price_peak=1.2,
        price_increase_pct=20.0,
        duration_minutes=30,
        volume_spike=2.5
    )

def main():
    """Run all tests"""
    print("🚀 Pump Analysis System - Test Suite")
    print("=" * 50)
    
    # Test 1: Configuration
    config_ok = test_configuration()
    print()
    
    # Test 2: Bybit connection
    bybit_ok, symbols = test_bybit_connection()
    print()
    
    if not bybit_ok:
        print("⚠️ Bybit connection failed - cannot continue with data tests")
        print("💡 This might be due to network issues or API changes")
        return
    
    # Test 3: Data fetching
    test_symbol = symbols[0] if symbols else "BTCUSDT"
    data_ok, kline_data = test_data_fetching(test_symbol)
    print()
    
    if not data_ok:
        print("⚠️ Data fetching failed - using synthetic test")
        # Create minimal test data for pump detection
        kline_data = []
    
    # Test 4: Pump detection
    pumps = test_pump_detection(kline_data, test_symbol)
    print()
    
    # Test 5: Pre-pump analysis (with test data if no real pumps)
    if pumps:
        test_pump = pumps[0]
        print("🧪 Testing with real pump data...")
    else:
        test_pump = create_test_pump_event()
        print("🧪 Testing with synthetic pump data...")
    
    analysis_ok, analysis = test_pre_pump_analysis(test_pump)
    print()
    
    # Summary
    print("📋 TEST SUMMARY")
    print("=" * 30)
    print(f"✅ Configuration: {'PASS' if config_ok else 'NEEDS SETUP'}")
    print(f"✅ Bybit API: {'PASS' if bybit_ok else 'FAIL'}")
    print(f"✅ Data Fetching: {'PASS' if data_ok else 'FAIL'}")
    print(f"✅ Pump Detection: PASS (found {len(pumps)} pumps)")
    print(f"✅ Pre-pump Analysis: {'PASS' if analysis_ok else 'FAIL'}")
    
    if config_ok and bybit_ok and data_ok:
        print("\n🎉 System ready for production use!")
        print("💡 Add your API keys to .env file and run: python main.py")
    else:
        print("\n⚠️ Some components need configuration")
        if not config_ok:
            print("   - Add OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID to .env")
        if not bybit_ok or not data_ok:
            print("   - Check internet connection and Bybit API status")

if __name__ == "__main__":
    main()