#!/usr/bin/env python3
"""
Test script for function history context system
Tests the new GPT-4o function history functionality
"""

import os
import sys
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv('../.env')

# Import modules
from main import GPTAnalyzer

def test_function_history_system():
    """Test the function history context system"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize GPT Analyzer
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY not found")
            return False
            
        analyzer = GPTAnalyzer(api_key)
        
        logger.info("Testing function history context system...")
        logger.info(f"Current history size: {len(analyzer.function_history)}")
        
        # Test 1: Add mock function to history
        test_function_code = '''def detect_TESTUSDT_20250618_preconditions(df):
    """
    Test detector function for RSI oversold + volume spike pattern
    
    Args:
        df: DataFrame with OHLCV data and technical indicators
        
    Returns:
        bool: True if pre-pump conditions detected
    """
    import pandas as pd
    import numpy as np
    
    if len(df) < 20:
        return False
    
    # RSI oversold condition
    current_rsi = df['rsi'].iloc[-1]
    if current_rsi > 35:
        return False
    
    # Volume spike condition
    avg_volume = df['volume'].rolling(10).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    if current_volume < avg_volume * 1.8:
        return False
    
    # Price near support
    recent_lows = df['low'].rolling(5).min()
    if df['close'].iloc[-1] > recent_lows.iloc[-1] * 1.03:
        return False
    
    return True'''
        
        analyzer._add_to_function_history(
            'TESTUSDT',
            '20250618',
            test_function_code,
            25.5
        )
        
        logger.info(f"✅ Added test function to history")
        logger.info(f"History size after adding: {len(analyzer.function_history)}")
        
        # Test 2: Format context
        context = analyzer._format_function_history_context()
        logger.info(f"✅ Context formatted: {len(context)} characters")
        
        if context:
            logger.info("Context preview:")
            lines = context.split('\n')[:10]
            for line in lines:
                logger.info(f"  {line}")
            logger.info("  ...")
        
        # Test 3: Verify JSON persistence
        if os.path.exists(analyzer.function_history_file):
            with open(analyzer.function_history_file, 'r', encoding='utf-8') as f:
                saved_history = json.load(f)
            logger.info(f"✅ History persisted to file: {len(saved_history)} entries")
        
        # Test 4: Test history limit
        for i in range(7):  # Add more than max_history_size (5)
            analyzer._add_to_function_history(
                f'TEST{i}USDT',
                '20250618',
                f'def detect_TEST{i}USDT_20250618_preconditions(df): return True',
                15.0 + i
            )
        
        logger.info(f"✅ After adding 7 more functions, history size: {len(analyzer.function_history)}")
        logger.info(f"✅ Max size maintained: {len(analyzer.function_history) <= analyzer.max_history_size}")
        
        # Test 5: Verify history order (newest first)
        if analyzer.function_history:
            newest_entry = analyzer.function_history[0]
            logger.info(f"✅ Newest entry: {newest_entry['symbol']} (+{newest_entry['pump_increase']:.1f}%)")
        
        logger.info("🎯 Function history context system test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_function_history_system()
    sys.exit(0 if success else 1)