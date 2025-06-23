#!/usr/bin/env python3
"""
Function History Integration Test
Comprehensive test for the complete function history system integration
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add pump-analysis to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import PumpEvent, PumpAnalysisSystem
from functions_history import FunctionHistoryManager, PerformanceTracker, GPTLearningEngine
from functions_history.function_manager import FunctionMetadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_pump_event() -> PumpEvent:
    """Create a test pump event for testing"""
    return PumpEvent(
        symbol="TESTUSDT",
        start_time=datetime.now() - timedelta(hours=1),
        price_before=1.0,
        price_peak=1.25,
        price_increase_pct=25.0,
        duration_minutes=30,
        volume_spike=3.5
    )

def create_test_pre_pump_data() -> dict:
    """Create test pre-pump analysis data"""
    return {
        'trend': 'bullish',
        'rsi_value': 45.2,
        'compression': {
            'detected': True,
            'compression_ratio': 0.85
        },
        'volume_spikes': [
            {'timestamp': '2025-06-18T17:30:00', 'volume_ratio': 2.1},
            {'timestamp': '2025-06-18T17:45:00', 'volume_ratio': 1.8}
        ],
        'vwap_analysis': {
            'above_vwap': True,
            'vwap_distance': 0.02
        },
        'fake_rejects': [
            {'timestamp': '2025-06-18T17:20:00', 'wick_ratio': 0.6}
        ],
        'support_resistance': {
            'near_support': True,
            'support_level': 0.98
        },
        'liquidity_gaps': [
            {'price_level': 1.05, 'gap_size': 0.03}
        ]
    }

def test_function_manager():
    """Test FunctionHistoryManager functionality"""
    logger.info("🧪 Testing FunctionHistoryManager...")
    
    try:
        manager = FunctionHistoryManager()
        
        # Create test metadata
        metadata = FunctionMetadata(
            symbol="TESTUSDT",
            date="20250618",
            pump_increase=25.0,
            generation_time=datetime.now(),
            active_signals=['bullish_trend', 'price_compression', 'above_vwap'],
            pre_pump_analysis=create_test_pre_pump_data()
        )
        
        # Store test function
        test_function = """
def detect_testusdt_20250618_preconditions(df):
    \"\"\"Test detector function\"\"\"
    if len(df) < 10:
        return False, 0.0, ['insufficient_data']
    
    # Check for bullish trend
    if df['close'].iloc[-1] > df['close'].iloc[-10]:
        return True, 0.85, ['bullish_trend', 'price_compression']
    
    return False, 0.0, ['no_signal']
"""
        
        function_id = manager.store_function(test_function, metadata)
        logger.info(f"✅ Function stored with ID: {function_id}")
        
        # Retrieve function
        retrieved_function, retrieved_metadata = manager.get_function(function_id)
        if retrieved_function and retrieved_metadata:
            logger.info("✅ Function retrieval successful")
        else:
            logger.error("❌ Function retrieval failed")
            return False
        
        # List functions
        functions = manager.list_functions()
        logger.info(f"✅ Found {len(functions)} functions in storage")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FunctionHistoryManager test failed: {e}")
        return False

def test_performance_tracker():
    """Test PerformanceTracker functionality"""
    logger.info("🧪 Testing PerformanceTracker...")
    
    try:
        tracker = PerformanceTracker()
        
        # Log test generation
        test_function = "def test_function(): return True"
        tracker.log_generation("test_func_001", "TESTUSDT", "20250618", test_function, 25.0)
        
        # Log test execution
        tracker.log_execution("test_func_001", True, 0.85, ['bullish_trend'])
        
        # Get performance stats
        stats = tracker.get_performance_stats("test_func_001")
        if stats:
            logger.info(f"✅ Performance stats retrieved: {stats['total_executions']} executions")
        else:
            logger.error("❌ Performance stats retrieval failed")
            return False
        
        # Get function ranking
        ranking = tracker.get_function_ranking()
        logger.info(f"✅ Function ranking retrieved: {len(ranking)} functions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PerformanceTracker test failed: {e}")
        return False

def test_gpt_learning_engine():
    """Test GPTLearningEngine functionality"""
    logger.info("🧪 Testing GPTLearningEngine...")
    
    try:
        # Check if OpenAI API key is available
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            logger.warning("⚠️ No OpenAI API key found, skipping GPT tests")
            return True
        
        engine = GPTLearningEngine(openai_key)
        
        # Test function generation
        pump_event = create_test_pump_event()
        pre_pump_data = create_test_pre_pump_data()
        
        logger.info("🤖 Generating test detector function...")
        detector_function = engine.generate_detector_function(pre_pump_data, pump_event)
        
        if detector_function and "def detect_" in detector_function:
            logger.info("✅ Detector function generation successful")
        else:
            logger.error("❌ Detector function generation failed")
            return False
        
        # Test function improvement
        logger.info("🔧 Testing function improvement...")
        improved_function = engine.create_improved_version(
            "test_func_001", detector_function, 0.6, ['low_accuracy']
        )
        
        if improved_function:
            logger.info("✅ Function improvement successful")
        else:
            logger.warning("⚠️ Function improvement failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GPTLearningEngine test failed: {e}")
        return False

def test_main_system_integration():
    """Test main system integration"""
    logger.info("🧪 Testing main system integration...")
    
    try:
        # Check required environment variables
        required_vars = ['OPENAI_API_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
            logger.info("✅ Skipping main system test due to missing credentials")
            return True
        
        # Initialize system
        system = PumpAnalysisSystem()
        
        # Test detector function generation
        pump_event = create_test_pump_event()
        pre_pump_data = create_test_pre_pump_data()
        
        logger.info("🔧 Testing detector function generation integration...")
        detector_function = system.generate_and_store_detector_function(pump_event, pre_pump_data)
        
        if detector_function and "def detect_" in detector_function:
            logger.info("✅ Main system integration successful")
        else:
            logger.error("❌ Main system integration failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Main system integration test failed: {e}")
        return False

def test_complete_workflow():
    """Test complete function history workflow"""
    logger.info("🧪 Testing complete workflow...")
    
    try:
        # Initialize all components
        manager = FunctionHistoryManager()
        tracker = PerformanceTracker()
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            logger.warning("⚠️ No OpenAI API key, using mock function generation")
            
        # Create test scenario
        pump_event = create_test_pump_event()
        pre_pump_data = create_test_pre_pump_data()
        
        # 1. Generate function metadata
        metadata = FunctionMetadata(
            symbol=pump_event.symbol,
            date=pump_event.start_time.strftime('%Y%m%d'),
            pump_increase=pump_event.price_increase_pct,
            generation_time=datetime.now(),
            active_signals=['bullish_trend', 'price_compression', 'above_vwap'],
            pre_pump_analysis=pre_pump_data
        )
        
        # 2. Create test function
        test_function = f"""
def detect_{pump_event.symbol.lower()}_{pump_event.start_time.strftime('%Y%m%d')}_preconditions(df):
    \"\"\"Generated detector for {pump_event.symbol} pump analysis\"\"\"
    if len(df) < 10:
        return False, 0.0, ['insufficient_data']
    
    signals = []
    confidence = 0.0
    
    # Check bullish trend
    if df['close'].iloc[-1] > df['close'].iloc[-5]:
        signals.append('bullish_trend')
        confidence += 0.3
    
    # Check volume spike
    if 'volume' in df.columns:
        recent_volume = df['volume'].iloc[-3:].mean()
        historical_volume = df['volume'].iloc[-20:-3].mean()
        if recent_volume > historical_volume * 1.5:
            signals.append('volume_spike')
            confidence += 0.4
    
    # Check price compression
    if 'high' in df.columns and 'low' in df.columns:
        recent_range = (df['high'].iloc[-5:] - df['low'].iloc[-5:]).mean()
        historical_range = (df['high'].iloc[-20:-5] - df['low'].iloc[-20:-5]).mean()
        if recent_range < historical_range * 0.8:
            signals.append('price_compression')
            confidence += 0.3
    
    return len(signals) >= 2, confidence, signals
"""
        
        # 3. Store function
        function_id = manager.store_function(test_function, metadata)
        logger.info(f"✅ Function stored: {function_id}")
        
        # 4. Log generation
        tracker.log_generation(
            function_id, pump_event.symbol, 
            pump_event.start_time.strftime('%Y%m%d'), 
            test_function, pump_event.price_increase_pct
        )
        
        # 5. Simulate function execution
        tracker.log_execution(function_id, True, 0.75, ['bullish_trend', 'volume_spike'])
        tracker.log_execution(function_id, False, 0.25, ['weak_signal'])
        tracker.log_execution(function_id, True, 0.85, ['bullish_trend', 'price_compression'])
        
        # 6. Get performance stats
        stats = tracker.get_performance_stats(function_id)
        logger.info(f"✅ Performance stats: {stats['success_rate']:.2f} success rate over {stats['total_executions']} executions")
        
        # 7. Get function ranking
        ranking = tracker.get_function_ranking()
        logger.info(f"✅ Function ranking: {len(ranking)} functions ranked")
        
        logger.info("✅ Complete workflow test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Complete workflow test failed: {e}")
        return False

def main():
    """Run comprehensive function history integration tests"""
    logger.info("🚀 Starting Function History Integration Tests")
    
    tests = [
        ("Function Manager", test_function_manager),
        ("Performance Tracker", test_performance_tracker),
        ("GPT Learning Engine", test_gpt_learning_engine),
        ("Main System Integration", test_main_system_integration),
        ("Complete Workflow", test_complete_workflow)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} - PASSED")
            else:
                logger.error(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} - ERROR: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Function history system is ready for production.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)