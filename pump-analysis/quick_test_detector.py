#!/usr/bin/env python3
"""
Quick Testing Tool for Individual Detector Functions

Simple script to test a specific detector function with custom data or scenarios.
Useful for debugging detector logic and threshold tuning.

Usage:
    python quick_test_detector.py BTCUSDT 20250613
    python quick_test_detector.py SONUSDT 20250614 --scenario pump
"""

import argparse
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from generated_detectors import load_detector
from benchmark_detectors import DetectorBenchmark

def create_custom_test_data(scenario: str = "pump") -> pd.DataFrame:
    """Create test data based on scenario type"""
    benchmark = DetectorBenchmark()
    
    scenarios = {
        "pump": benchmark.create_pump_pattern_data(),
        "normal": benchmark.create_normal_market_data(),
        "compression": benchmark.create_compression_only_data()
    }
    
    return scenarios.get(scenario, benchmark.create_pump_pattern_data())

def analyze_detector_logic(detector_func, test_data: pd.DataFrame) -> dict:
    """Analyze what conditions the detector is checking"""
    
    # Extract recent data (last 12 candles like most detectors expect)
    recent_data = test_data.tail(12).copy()
    
    analysis = {
        "data_points": len(recent_data),
        "price_range_pct": ((recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()) * 100,
        "current_rsi": recent_data['rsi'].iloc[-1] if 'rsi' in recent_data.columns else None,
        "vwap_premium_pct": ((recent_data['close'].iloc[-1] - recent_data['vwap'].iloc[-1]) / recent_data['vwap'].iloc[-1]) * 100 if 'vwap' in recent_data.columns else None,
        "volume_spike_detected": False,
        "fake_reject_detected": False
    }
    
    # Check for volume spikes
    if len(recent_data) >= 6:
        recent_volumes = recent_data['volume'].values
        avg_volume = np.mean(recent_volumes[:-3])
        recent_spikes = recent_volumes[-3:] / avg_volume
        analysis["max_volume_multiplier"] = float(np.max(recent_spikes))
        analysis["volume_spike_detected"] = np.any(recent_spikes > 2.0)
    
    # Check for fake reject patterns
    fake_rejects = 0
    for i in range(-3, 0):
        if abs(i) <= len(recent_data):
            candle = recent_data.iloc[i]
            body_size = abs(candle['close'] - candle['open'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            
            if lower_wick > body_size * 1.5:  # Significant lower wick
                fake_rejects += 1
    
    analysis["fake_reject_count"] = fake_rejects
    analysis["fake_reject_detected"] = fake_rejects > 0
    
    return analysis

def test_single_detector(symbol: str, date: str, scenario: str = "pump", 
                        verbose: bool = True) -> dict:
    """Test a single detector function"""
    
    try:
        # Load detector function
        detector_func = load_detector(symbol, date)
        detector_name = f"detect_{symbol}_{date}_preconditions"
        
        if verbose:
            print(f"Testing detector: {detector_name}")
            print("=" * 50)
        
        # Create test data
        test_data = create_custom_test_data(scenario)
        
        if verbose:
            print(f"Test scenario: {scenario}")
            print(f"Data points: {len(test_data)}")
            print()
        
        # Analyze data characteristics
        analysis = analyze_detector_logic(detector_func, test_data)
        
        if verbose:
            print("Data Analysis:")
            print(f"  Price range: {analysis['price_range_pct']:.2f}%")
            print(f"  Current RSI: {analysis['current_rsi']:.1f}" if analysis['current_rsi'] else "  RSI: Not available")
            print(f"  VWAP premium: {analysis['vwap_premium_pct']:.2f}%" if analysis['vwap_premium_pct'] else "  VWAP: Not available")
            print(f"  Max volume multiplier: {analysis.get('max_volume_multiplier', 'N/A')}")
            print(f"  Volume spike detected: {analysis['volume_spike_detected']}")
            print(f"  Fake reject patterns: {analysis['fake_reject_count']}")
            print()
        
        # Run detector
        result = detector_func(test_data)
        
        if verbose:
            print("Detection Result:")
            status = "✅ DETECTED" if result else "❌ NOT DETECTED"
            print(f"  {status}")
            print()
            
            # Provide interpretation
            print("Interpretation:")
            if result:
                print("  The detector identified pre-pump conditions in this data.")
                print("  Key factors likely contributing to detection:")
                
                if analysis['price_range_pct'] < 3.0:
                    print("    - Price compression detected")
                if analysis['current_rsi'] and 50 <= analysis['current_rsi'] <= 65:
                    print("    - RSI in accumulation zone")
                if analysis['vwap_premium_pct'] and analysis['vwap_premium_pct'] > 1:
                    print("    - Price above VWAP")
                if analysis['volume_spike_detected']:
                    print("    - Volume spike detected")
                if analysis['fake_reject_detected']:
                    print("    - Fake reject pattern found")
                    
            else:
                print("  The detector did not find sufficient pre-pump conditions.")
                print("  Missing or insufficient factors:")
                
                if analysis['price_range_pct'] > 4.0:
                    print("    - Price range too wide (no compression)")
                if analysis['current_rsi'] and (analysis['current_rsi'] < 45 or analysis['current_rsi'] > 70):
                    print("    - RSI outside accumulation zone")
                if analysis['vwap_premium_pct'] and analysis['vwap_premium_pct'] < 0:
                    print("    - Price below VWAP")
                if not analysis['volume_spike_detected']:
                    print("    - No significant volume spike")
                if not analysis['fake_reject_detected']:
                    print("    - No fake reject patterns")
        
        return {
            "detector_name": detector_name,
            "scenario": scenario,
            "result": bool(result),
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except FileNotFoundError:
        error_msg = f"Detector not found: {symbol}_{date}.py"
        if verbose:
            print(f"❌ Error: {error_msg}")
        return {"error": error_msg}
        
    except Exception as e:
        error_msg = f"Error testing detector: {str(e)}"
        if verbose:
            print(f"❌ Error: {error_msg}")
        return {"error": error_msg}

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Quick test for individual detector functions")
    parser.add_argument("symbol", help="Symbol (e.g., BTCUSDT)")
    parser.add_argument("date", help="Date in YYYYMMDD format (e.g., 20250613)")
    parser.add_argument("--scenario", choices=["pump", "normal", "compression"], 
                       default="pump", help="Test scenario type")
    parser.add_argument("--quiet", "-q", action="store_true", 
                       help="Quiet mode - minimal output")
    
    args = parser.parse_args()
    
    # Test the detector
    result = test_single_detector(
        symbol=args.symbol,
        date=args.date,
        scenario=args.scenario,
        verbose=not args.quiet
    )
    
    # Exit with appropriate code
    if "error" in result:
        sys.exit(1)
    elif result.get("result", False):
        sys.exit(0)  # Detection successful
    else:
        sys.exit(2)  # No detection

if __name__ == "__main__":
    main()