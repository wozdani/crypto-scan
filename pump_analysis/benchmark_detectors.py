#!/usr/bin/env python3
"""
Benchmark Testing Module for Generated Detector Functions

Simple benchmark system that tests generated detectors on synthetic data
to validate their logic and thresholds. Creates controlled test scenarios:
1. Positive cases with pump-like patterns
2. Negative cases with normal market conditions
3. Edge cases with specific pattern variations

This provides immediate validation without requiring external API access.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
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

class DetectorBenchmark:
    """Benchmark testing for detector functions using controlled scenarios"""
    
    def __init__(self):
        self.test_scenarios = {}
        
    def create_pump_pattern_data(self, base_price: float = 50000.0, 
                                candles: int = 72) -> pd.DataFrame:
        """Create DataFrame with pump-like pre-conditions"""
        np.random.seed(42)
        
        timestamps = pd.date_range(start='2025-06-13 10:00:00', periods=candles, freq='5min')
        
        # Generate pump-like pattern
        prices = []
        volumes = []
        
        for i in range(candles):
            if i < 60:  # Pre-pump compression phase
                # Tight price range with low volatility
                price_change = np.random.normal(0, 0.001)  # 0.1% volatility
                volume_multiplier = np.random.normal(1, 0.2)
            else:  # Early pump signals
                # Slight increase in volatility and volume
                price_change = np.random.normal(0.002, 0.002)  # 0.2% upward bias
                volume_multiplier = np.random.normal(2.5, 0.5)  # Volume spike
                
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + price_change)
                
            prices.append(price)
            volumes.append(max(1000000 * volume_multiplier, 100000))
        
        # Create OHLC data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'close': prices,
            'volume': volumes
        })
        
        # Add small variations for high/low
        df['high'] = df['close'] * np.random.uniform(1.001, 1.005, len(df))
        df['low'] = df['close'] * np.random.uniform(0.995, 0.999, len(df))
        
        # Add fake reject pattern in last few candles
        for i in range(-3, 0):
            df.iloc[i, df.columns.get_loc('low')] = df.iloc[i]['close'] * 0.97  # 3% wick down
        
        # Calculate indicators
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Adjust RSI to accumulation zone (50-60)
        df.loc[df.index[-12:], 'rsi'] = np.random.uniform(52, 58, 12)
        
        return df
    
    def create_normal_market_data(self, base_price: float = 50000.0,
                                 candles: int = 72) -> pd.DataFrame:
        """Create DataFrame with normal market conditions (no pump)"""
        np.random.seed(123)  # Different seed for different pattern
        
        timestamps = pd.date_range(start='2025-06-13 10:00:00', periods=candles, freq='5min')
        
        prices = []
        volumes = []
        
        for i in range(candles):
            # Normal market volatility
            price_change = np.random.normal(0, 0.003)  # 0.3% volatility
            volume_multiplier = np.random.normal(1, 0.4)
                
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + price_change)
                
            prices.append(price)
            volumes.append(max(1000000 * volume_multiplier, 100000))
        
        # Create OHLC data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'close': prices,
            'volume': volumes
        })
        
        df['high'] = df['close'] * np.random.uniform(1.001, 1.008, len(df))
        df['low'] = df['close'] * np.random.uniform(0.992, 0.999, len(df))
        
        # Calculate indicators
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        return df
    
    def create_compression_only_data(self, base_price: float = 50000.0,
                                    candles: int = 72) -> pd.DataFrame:
        """Create data with price compression but no other pump signals"""
        np.random.seed(456)
        
        timestamps = pd.date_range(start='2025-06-13 10:00:00', periods=candles, freq='5min')
        
        # Very tight price range but normal volume and RSI
        prices = [base_price + np.random.uniform(-50, 50) for _ in range(candles)]
        volumes = [1000000 * np.random.normal(1, 0.3) for _ in range(candles)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'close': prices,
            'volume': [max(v, 100000) for v in volumes]
        })
        
        df['high'] = df['close'] + np.random.uniform(10, 30, len(df))
        df['low'] = df['close'] - np.random.uniform(10, 30, len(df))
        
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def create_test_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Create all test scenarios"""
        scenarios = {
            'pump_pattern': self.create_pump_pattern_data(),
            'normal_market': self.create_normal_market_data(),
            'compression_only': self.create_compression_only_data(),
            'high_volume_no_compression': self.create_normal_market_data(base_price=48000),
            'low_rsi_compression': self.create_compression_only_data(base_price=52000)
        }
        
        # Modify specific scenarios
        # High volume scenario
        scenarios['high_volume_no_compression']['volume'] *= 3
        
        # Low RSI scenario
        scenarios['low_rsi_compression']['rsi'] = 35  # Below accumulation zone
        
        return scenarios
    
    def test_detector_on_scenarios(self, detector_name: str) -> Dict:
        """Test a detector on all scenarios"""
        logger.info(f"Testing {detector_name} on benchmark scenarios...")
        
        try:
            # Extract symbol and date for loading
            parts = detector_name.replace('detect_', '').replace('_preconditions', '').split('_')
            if len(parts) < 2:
                return {"error": "Invalid detector name format"}
                
            symbol = '_'.join(parts[:-1])
            date = parts[-1]
            
            # Load detector function
            detector_func = load_detector(symbol, date)
            
            # Create test scenarios
            scenarios = self.create_test_scenarios()
            
            results = {}
            
            for scenario_name, scenario_data in scenarios.items():
                try:
                    detection_result = detector_func(scenario_data)
                    
                    # Expected results based on scenario
                    expected_results = {
                        'pump_pattern': True,  # Should detect pump pattern
                        'normal_market': False,  # Should not detect normal market
                        'compression_only': False,  # Compression alone shouldn't trigger
                        'high_volume_no_compression': False,  # Volume alone shouldn't trigger
                        'low_rsi_compression': False  # Wrong RSI shouldn't trigger
                    }
                    
                    expected = expected_results.get(scenario_name, False)
                    
                    results[scenario_name] = {
                        'result': bool(detection_result),
                        'expected': expected,
                        'correct': bool(detection_result) == expected,
                        'data_points': len(scenario_data)
                    }
                    
                    status = "✅" if bool(detection_result) == expected else "❌"
                    logger.info(f"  {scenario_name}: {detection_result} (expected: {expected}) {status}")
                    
                except Exception as e:
                    results[scenario_name] = {
                        'error': str(e),
                        'result': None,
                        'expected': expected_results.get(scenario_name, False),
                        'correct': False
                    }
                    logger.error(f"  {scenario_name}: Error - {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing {detector_name}: {e}")
            return {"error": str(e)}
    
    def run_benchmark_suite(self) -> Dict:
        """Run complete benchmark on all detectors"""
        logger.info("Starting detector benchmark suite...")
        
        # Get available detectors
        detectors = get_available_detectors()
        if not detectors:
            logger.warning("No detectors found")
            return {}
        
        logger.info(f"Found {len(detectors)} detectors to benchmark")
        
        all_results = {}
        
        for i, detector_name in enumerate(detectors):
            logger.info(f"\nBenchmark {i+1}/{len(detectors)}: {detector_name}")
            
            detector_results = self.test_detector_on_scenarios(detector_name)
            all_results[detector_name] = {
                'detector_name': detector_name,
                'timestamp': datetime.now().isoformat(),
                'scenario_results': detector_results
            }
        
        # Generate benchmark summary
        summary = self.generate_benchmark_summary(all_results)
        logger.info(f"\n{summary}")
        
        # Save results
        self.save_benchmark_results(all_results)
        
        return all_results
    
    def generate_benchmark_summary(self, results: Dict) -> str:
        """Generate benchmark summary"""
        if not results:
            return "No benchmark results available"
        
        summary_lines = []
        summary_lines.append("=" * 50)
        summary_lines.append("DETECTOR BENCHMARK SUMMARY")
        summary_lines.append("=" * 50)
        
        total_detectors = len(results)
        total_tests = 0
        total_correct = 0
        
        for detector_name, detector_data in results.items():
            scenario_results = detector_data.get('scenario_results', {})
            
            if 'error' in scenario_results:
                summary_lines.append(f"\n❌ {detector_name}: ERROR - {scenario_results['error']}")
                continue
            
            detector_correct = 0
            detector_total = 0
            
            summary_lines.append(f"\n🔍 {detector_name}:")
            
            for scenario_name, scenario_result in scenario_results.items():
                if 'correct' in scenario_result:
                    detector_total += 1
                    total_tests += 1
                    
                    if scenario_result['correct']:
                        detector_correct += 1
                        total_correct += 1
                        status = "✅"
                    else:
                        status = "❌"
                    
                    result = scenario_result['result']
                    expected = scenario_result['expected']
                    summary_lines.append(f"   {scenario_name}: {result} (exp: {expected}) {status}")
                else:
                    summary_lines.append(f"   {scenario_name}: ERROR - {scenario_result.get('error', 'Unknown')}")
            
            if detector_total > 0:
                accuracy = (detector_correct / detector_total) * 100
                summary_lines.append(f"   Accuracy: {detector_correct}/{detector_total} ({accuracy:.1f}%)")
        
        # Overall statistics
        summary_lines.append("\n" + "=" * 30)
        summary_lines.append("OVERALL BENCHMARK RESULTS")
        summary_lines.append("=" * 30)
        summary_lines.append(f"Total detectors: {total_detectors}")
        summary_lines.append(f"Total tests: {total_tests}")
        
        if total_tests > 0:
            overall_accuracy = (total_correct / total_tests) * 100
            summary_lines.append(f"Overall accuracy: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
        
        return "\n".join(summary_lines)
    
    def save_benchmark_results(self, results: Dict):
        """Save benchmark results to file"""
        results_dir = current_dir / "test_results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = results_dir / f"detector_benchmark_{timestamp}.json"
        
        try:
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Benchmark results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving benchmark results: {e}")

def main():
    """Main function to run benchmark"""
    logger.info("Starting Detector Benchmark System")
    
    try:
        benchmark = DetectorBenchmark()
        results = benchmark.run_benchmark_suite()
        
        if results:
            logger.info("Benchmark completed successfully")
        else:
            logger.warning("No benchmark tests were completed")
            
    except Exception as e:
        logger.error(f"Benchmark system error: {e}")
        raise

if __name__ == "__main__":
    main()