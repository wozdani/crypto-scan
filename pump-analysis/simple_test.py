#!/usr/bin/env python3
"""
Simple test script to verify pump analysis components work
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if required API keys are configured
def check_api_keys():
    """Check if API keys are properly configured"""
    required_keys = ['OPENAI_API_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key) or os.getenv(key) == f'your_{key.lower()}_here':
            missing_keys.append(key)
    
    return len(missing_keys) == 0, missing_keys

def test_basic_functionality():
    """Test basic system functionality without external APIs"""
    print("Testing pump analysis system components...")
    
    # Test 1: Configuration loading
    try:
        from config import Config
        config = Config()
        print("✅ Configuration module loaded successfully")
        
        summary = config.get_summary()
        print(f"Configuration: {summary['min_pump_increase_pct']}% minimum pump threshold")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # Test 2: Data structures
    try:
        from main import PumpEvent
        from datetime import datetime
        
        test_pump = PumpEvent(
            symbol="TESTUSDT",
            start_time=datetime.now(),
            price_before=1.0,
            price_peak=1.2,
            price_increase_pct=20.0,
            duration_minutes=30,
            volume_spike=2.5
        )
        print("✅ PumpEvent data structure working")
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False
    
    # Test 3: Analysis components
    try:
        from main import PumpDetector
        detector = PumpDetector(min_increase_pct=15.0)
        print("✅ PumpDetector initialized successfully")
        
    except Exception as e:
        print(f"❌ PumpDetector test failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🔧 Pump Analysis System - Quick Test")
    print("=" * 40)
    
    # Check API configuration
    api_configured, missing = check_api_keys()
    
    if api_configured:
        print("✅ All API keys configured")
        print("💡 System ready for full analysis")
    else:
        print(f"⚠️  Missing API keys: {', '.join(missing)}")
        print("💡 Add these to your .env file for full functionality")
    
    print()
    
    # Test basic functionality
    if test_basic_functionality():
        print()
        print("✅ All basic components working correctly")
        
        if api_configured:
            print("🚀 Ready to run: python main.py")
        else:
            print("🔧 Configure API keys in .env file first")
    else:
        print()
        print("❌ Some components need fixing")
    
    print()
    print("📝 Next steps:")
    print("1. Add your API keys to .env file")
    print("2. Run: python main.py")
    print("3. Check pump_data/ folder for results")

if __name__ == "__main__":
    main()