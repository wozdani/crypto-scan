#!/usr/bin/env python3
"""
Automated Testing Module for Generated Detector Functions

This module tests all generated detector functions against real pump data
to validate their accuracy and effectiveness. Each function is tested on:
1. Its own pre-pump case (should return True)
2. Random other cases from the same timeframe (should return False)
3. Cross-validation against different market conditions

Results are logged and can be used for:
- Quality assessment of generated detectors
- ML training data preparation 
- PPWCS/trend mode enhancement
- Pattern recognition validation
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from generated_detectors import get_available_detectors, load_detector

class DetectorTester:
    """Main class for testing generated detector functions"""
    
    def __init__(self):
        self.test_results = {}
        self.pump_data_dir = current_dir / "pump_data"
        self.results_dir = current_dir / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize Bybit data fetcher for real data loading
        try:
            from main import BybitDataFetcher
            self.data_fetcher = BybitDataFetcher()
            logger.info("✅ Bybit data fetcher initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Bybit fetcher: {e}")
            self.data_fetcher = None
    
    def load_pump_analysis_files(self) -> List[Dict]:
        """Load all saved pump analysis files to get pump details"""
        pump_cases = []
        
        if not self.pump_data_dir.exists():
            logger.warning("📁 Pump data directory not found")
            return pump_cases
        
        for file_path in self.pump_data_dir.glob("*_pump_analysis.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    pump_cases.append(data)
                    logger.debug(f"📄 Loaded pump case from {file_path.name}")
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
        
        logger.info(f"📊 Loaded {len(pump_cases)} pump analysis cases")
        return pump_cases
    
    def create_test_dataframe(self, kline_data: List, add_indicators: bool = True) -> pd.DataFrame:
        """Convert Bybit kline data to DataFrame with required columns"""
        if not kline_data:
            return pd.DataFrame()
        
        # Convert kline data to DataFrame
        df = pd.DataFrame(kline_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Convert to proper types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        if add_indicators:
            # Calculate VWAP
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Calculate RSI
            df['rsi'] = self.calculate_rsi(df['close'])
        
        # Reorder columns to match expected format
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'rsi']
        df = df[expected_cols]
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def load_pre_pump_data(self, symbol: str, pump_start_time: str, 
                          hours_before: int = 1) -> Optional[pd.DataFrame]:
        """
        Load real market data for specified time before pump
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            pump_start_time: ISO format timestamp of pump start
            hours_before: Hours of data to load before pump
            
        Returns:
            DataFrame with pre-pump data or None if failed
        """
        if not self.data_fetcher:
            logger.error("❌ Data fetcher not available")
            return None
        
        try:
            # Parse pump start time
            pump_time = datetime.fromisoformat(pump_start_time.replace('Z', '+00:00'))
            
            # Calculate time range (60 minutes before pump)
            end_time = pump_time
            start_time = end_time - timedelta(hours=hours_before)
            
            # Convert to milliseconds for Bybit API
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            logger.info(f"📊 Loading data for {symbol}: {start_time} to {end_time}")
            
            # Fetch kline data (5-minute intervals)
            kline_data = self.data_fetcher.get_kline_data(
                symbol=symbol,
                interval="5",
                start_time=start_ms,
                limit=int(hours_before * 12)  # 12 candles per hour for 5min intervals
            )
            
            if not kline_data:
                logger.warning(f"⚠️ No kline data received for {symbol}")
                return None
            
            # Convert to DataFrame
            df = self.create_test_dataframe(kline_data)
            
            # Filter to exact time range
            df = df[
                (df['timestamp'] >= start_time) & 
                (df['timestamp'] <= end_time)
            ].copy()
            
            logger.info(f"✅ Loaded {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading pre-pump data for {symbol}: {e}")
            return None
    
    def test_detector_on_own_case(self, detector_name: str, pump_case: Dict) -> Dict:
        """Test detector on its own pump case"""
        logger.info(f"🎯 Testing {detector_name} on its own case...")
        
        try:
            # Extract symbol and date from detector name
            parts = detector_name.replace('detect_', '').replace('_preconditions', '').split('_')
            if len(parts) < 2:
                return {"error": "Invalid detector name format"}
                
            symbol = '_'.join(parts[:-1])
            date = parts[-1]
            
            # Load the detector function
            detector_func = load_detector(symbol, date)
            
            # Get pump details
            pump_event = pump_case['pump_event']
            pump_start_time = pump_event['start_time']
            
            # Load pre-pump data
            pre_pump_df = self.load_pre_pump_data(symbol, pump_start_time)
            
            if pre_pump_df is None or len(pre_pump_df) == 0:
                return {
                    "result": None,
                    "error": "Could not load pre-pump data",
                    "data_points": 0
                }
            
            # Run detector
            detection_result = detector_func(pre_pump_df)
            
            result = {
                "result": bool(detection_result),
                "expected": True,  # Should detect its own case
                "correct": bool(detection_result) == True,
                "data_points": len(pre_pump_df),
                "pump_increase": pump_event['price_increase_pct'],
                "test_type": "own_case"
            }
            
            logger.info(f"   Result: {detection_result} (Expected: True)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error testing {detector_name}: {e}")
            return {"error": str(e), "test_type": "own_case"}
    
    def test_detector_cross_validation(self, detector_name: str, 
                                     other_cases: List[Dict], max_tests: int = 5) -> List[Dict]:
        """Test detector on other pump cases (should return False)"""
        logger.info(f"🔄 Cross-validating {detector_name} on {min(len(other_cases), max_tests)} other cases...")
        
        results = []
        
        try:
            # Extract symbol and date from detector name
            parts = detector_name.replace('detect_', '').replace('_preconditions', '').split('_')
            if len(parts) < 2:
                return [{"error": "Invalid detector name format"}]
                
            symbol = '_'.join(parts[:-1])
            date = parts[-1]
            
            # Load the detector function
            detector_func = load_detector(symbol, date)
            
            # Test on other cases
            tested_count = 0
            for other_case in other_cases[:max_tests]:
                try:
                    other_pump = other_case['pump_event']
                    other_symbol = other_pump['symbol']
                    other_start_time = other_pump['start_time']
                    
                    # Skip same symbol/date combination
                    if other_symbol == symbol and other_start_time.startswith(date[:4]):
                        continue
                    
                    # Load pre-pump data for other case
                    other_pre_pump_df = self.load_pre_pump_data(other_symbol, other_start_time)
                    
                    if other_pre_pump_df is None or len(other_pre_pump_df) == 0:
                        continue
                    
                    # Run detector on other case
                    detection_result = detector_func(other_pre_pump_df)
                    
                    result = {
                        "result": bool(detection_result),
                        "expected": False,  # Should NOT detect other cases
                        "correct": bool(detection_result) == False,
                        "tested_symbol": other_symbol,
                        "tested_date": other_start_time[:10],
                        "data_points": len(other_pre_pump_df),
                        "test_type": "cross_validation"
                    }
                    
                    results.append(result)
                    tested_count += 1
                    
                    logger.debug(f"   {other_symbol}: {detection_result} (Expected: False)")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error testing on {other_case.get('pump_event', {}).get('symbol', 'unknown')}: {e}")
                    continue
            
            logger.info(f"   Completed {tested_count} cross-validation tests")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in cross-validation for {detector_name}: {e}")
            return [{"error": str(e), "test_type": "cross_validation"}]
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive test on all available detectors"""
        logger.info("🚀 Starting comprehensive detector testing...")
        
        # Get available detectors
        detectors = get_available_detectors()
        if not detectors:
            logger.warning("⚠️ No detectors found")
            return {}
        
        logger.info(f"🔍 Found {len(detectors)} detectors to test")
        
        # Load pump cases
        pump_cases = self.load_pump_analysis_files()
        if not pump_cases:
            logger.warning("⚠️ No pump analysis files found for testing")
            return {}
        
        logger.info(f"📊 Found {len(pump_cases)} pump cases for testing")
        
        # Test each detector
        all_results = {}
        
        for i, detector_name in enumerate(detectors):
            logger.info(f"\n📋 Testing detector {i+1}/{len(detectors)}: {detector_name}")
            
            detector_results = {
                "detector_name": detector_name,
                "timestamp": datetime.now().isoformat(),
                "own_case_test": {},
                "cross_validation_tests": []
            }
            
            # Find corresponding pump case
            corresponding_case = None
            detector_symbol = detector_name.split('_')[1]  # Extract symbol from detector name
            
            for case in pump_cases:
                if case['pump_event']['symbol'] == detector_symbol:
                    corresponding_case = case
                    break
            
            if corresponding_case:
                # Test on own case
                own_case_result = self.test_detector_on_own_case(detector_name, corresponding_case)
                detector_results["own_case_test"] = own_case_result
                
                # Cross-validation on other cases
                other_cases = [case for case in pump_cases if case != corresponding_case]
                cross_val_results = self.test_detector_cross_validation(detector_name, other_cases)
                detector_results["cross_validation_tests"] = cross_val_results
            else:
                logger.warning(f"⚠️ No corresponding pump case found for {detector_name}")
                detector_results["own_case_test"] = {"error": "No corresponding pump case found"}
            
            all_results[detector_name] = detector_results
        
        # Save results
        self.save_test_results(all_results)
        
        # Generate summary
        summary = self.generate_test_summary(all_results)
        logger.info(f"\n📊 Test Summary:\n{summary}")
        
        return all_results
    
    def save_test_results(self, results: Dict):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f"detector_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"💾 Test results saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Error saving test results: {e}")
    
    def generate_test_summary(self, results: Dict) -> str:
        """Generate human-readable test summary"""
        if not results:
            return "No test results available"
        
        summary_lines = []
        summary_lines.append("=" * 50)
        summary_lines.append("DETECTOR TESTING SUMMARY")
        summary_lines.append("=" * 50)
        
        total_detectors = len(results)
        own_case_correct = 0
        cross_val_correct = 0
        total_cross_val = 0
        
        for detector_name, detector_results in results.items():
            summary_lines.append(f"\n🔍 {detector_name}")
            
            # Own case test
            own_test = detector_results.get("own_case_test", {})
            if "correct" in own_test:
                if own_test["correct"]:
                    summary_lines.append("   ✅ Own case: PASS")
                    own_case_correct += 1
                else:
                    summary_lines.append("   ❌ Own case: FAIL")
            else:
                summary_lines.append(f"   ⚠️ Own case: ERROR - {own_test.get('error', 'Unknown')}")
            
            # Cross-validation tests
            cross_val_tests = detector_results.get("cross_validation_tests", [])
            if cross_val_tests:
                correct_cross_val = sum(1 for test in cross_val_tests if test.get("correct", False))
                total_tests = len(cross_val_tests)
                total_cross_val += total_tests
                cross_val_correct += correct_cross_val
                
                accuracy = (correct_cross_val / total_tests) * 100 if total_tests > 0 else 0
                summary_lines.append(f"   📊 Cross-validation: {correct_cross_val}/{total_tests} correct ({accuracy:.1f}%)")
            else:
                summary_lines.append("   ⚠️ Cross-validation: No tests completed")
        
        # Overall statistics
        summary_lines.append("\n" + "=" * 30)
        summary_lines.append("OVERALL STATISTICS")
        summary_lines.append("=" * 30)
        summary_lines.append(f"Total detectors tested: {total_detectors}")
        summary_lines.append(f"Own case accuracy: {own_case_correct}/{total_detectors} ({(own_case_correct/total_detectors)*100:.1f}%)")
        
        if total_cross_val > 0:
            overall_cross_val_accuracy = (cross_val_correct / total_cross_val) * 100
            summary_lines.append(f"Cross-validation accuracy: {cross_val_correct}/{total_cross_val} ({overall_cross_val_accuracy:.1f}%)")
        
        return "\n".join(summary_lines)

def main():
    """Main function to run detector tests"""
    logger.info("🧪 Starting Detector Testing System")
    
    try:
        tester = DetectorTester()
        results = tester.run_comprehensive_test()
        
        if results:
            logger.info("✅ Detector testing completed successfully")
        else:
            logger.warning("⚠️ No tests were completed")
            
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        raise

if __name__ == "__main__":
    main()