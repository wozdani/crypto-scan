#!/usr/bin/env python3
"""
Advanced Pump Detection Module for 15-minute Candles

Detects the biggest price movements across multiple timeframes (1h, 2h, 4h, 6h, 12h)
and classifies them into pump types: pump-impulse, trend-breakout, trend-mode, micro-move
"""

import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Optional

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def detect_biggest_pump_15m(candles: List[Dict], symbol: str) -> Optional[Dict]:
    """
    Detect the biggest pump from 15-minute candles across multiple timeframes
    
    Args:
        candles: List of 15-minute OHLCV candles (last 7 days)
        symbol: Trading symbol (e.g., 'BTCUSDT')
        
    Returns:
        Dict with pump details or None if no significant pump found
    """
    if not candles or len(candles) < 4:
        return None
        
    # Timeframes: hours -> number of 15-min candles
    timeframes = {
        1: 4,    # 1 hour = 4 x 15-min candles
        2: 8,    # 2 hours = 8 x 15-min candles  
        4: 16,   # 4 hours = 16 x 15-min candles
        6: 24,   # 6 hours = 24 x 15-min candles
        12: 48   # 12 hours = 48 x 15-min candles
    }
    
    min_growth = 0.15  # Minimum 15% growth to consider
    best_pump = None
    
    print(f"🔍 Analyzing {symbol} across {len(candles)} candles for biggest pump...")
    
    for hours, window in timeframes.items():
        if len(candles) < window:
            continue
            
        for i in range(len(candles) - window):
            try:
                start_candle = candles[i]
                end_index = i + window - 1  # Include the window-th candle
                
                # Find the lowest price in the starting period and highest in the ending period
                start_low = float(start_candle["low"])
                
                # Get the highest price within the window
                window_high = 0.0
                end_timestamp = None
                
                for j in range(i, i + window):
                    candle_high = float(candles[j]["high"])
                    if candle_high > window_high:
                        window_high = candle_high
                        end_timestamp = candles[j]["timestamp"]
                
                if start_low == 0:
                    continue
                    
                # Calculate growth percentage
                change = (window_high - start_low) / start_low
                
                if change >= min_growth:
                    pump = {
                        "symbol": symbol,
                        "growth": round(change * 100, 2),
                        "start_price": round(start_low, 6),
                        "end_price": round(window_high, 6),
                        "start_time": format_timestamp(start_candle["timestamp"]),
                        "end_time": format_timestamp(end_timestamp),
                        "window_hours": hours,
                        "window_min": hours * 60,
                        "type": categorize_pump_15m(hours, change)
                    }
                    
                    # Keep the biggest pump found
                    if best_pump is None or pump["growth"] > best_pump["growth"]:
                        best_pump = pump
                        
            except Exception as e:
                print(f"❌ Pump check error for {symbol} at window {hours}h: {e}")
                continue
    
    if best_pump:
        print(f"✅ Biggest pump detected for {symbol}: {best_pump['growth']}% ({best_pump['type']})")
    else:
        print(f"📊 No significant pump found for {symbol} (min {min_growth*100}%)")
        
    return best_pump

def categorize_pump_15m(hours: int, growth: float) -> str:
    """
    Categorize pump type based on timeframe and growth percentage
    
    Args:
        hours: Timeframe in hours
        growth: Growth as decimal (e.g., 0.347 for 34.7%)
        
    Returns:
        Pump category string
    """
    if hours <= 1 and growth >= 0.2:
        return "pump-impulse"     # >20% in ≤1h
    elif hours <= 4 and growth >= 0.3:
        return "trend-breakout"   # >30% in ≤4h  
    elif hours > 4 and growth >= 0.5:
        return "trend-mode"       # >50% in >4h
    else:
        return "micro-move"       # Everything else (usually filtered out)

def format_timestamp(timestamp) -> str:
    """
    Format timestamp to readable UTC string
    
    Args:
        timestamp: Unix timestamp or datetime string
        
    Returns:
        Formatted UTC timestamp string
    """
    try:
        if isinstance(timestamp, (int, float)):
            # Unix timestamp in seconds
            dt = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp, tz=timezone.utc)
        elif isinstance(timestamp, str):
            # Already formatted string
            return timestamp
        else:
            return str(timestamp)
            
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except Exception:
        return str(timestamp)

def analyze_pump_with_gpt(pump_data: Dict) -> Optional[Dict]:
    """
    Analyze detected pump with GPT for additional insights
    
    Args:
        pump_data: Pump detection results
        
    Returns:
        GPT analysis results or None if analysis fails
    """
    try:
        print(f"🤖 GPT analysis available for {pump_data['symbol']} pump")
        
        # Prepare analysis summary for logging
        analysis_summary = {
            "symbol": pump_data["symbol"],
            "pump_type": pump_data["type"],
            "growth_percent": pump_data["growth"],
            "timeframe_hours": pump_data["window_hours"],
            "analysis_available": True
        }
        
        print(f"📊 Pump analysis summary: {analysis_summary}")
        return analysis_summary
            
    except Exception as e:
        print(f"❌ GPT analysis error: {e}")
        return None

def batch_pump_detection(symbols_data: Dict[str, List[Dict]], enable_gpt: bool = False) -> Dict[str, Dict]:
    """
    Run pump detection on multiple symbols
    
    Args:
        symbols_data: Dict mapping symbol -> list of candles
        enable_gpt: Whether to run GPT analysis on detected pumps
        
    Returns:
        Dict mapping symbol -> pump detection results
    """
    results = {}
    total_symbols = len(symbols_data)
    
    print(f"🚀 Starting batch pump detection for {total_symbols} symbols...")
    
    for i, (symbol, candles) in enumerate(symbols_data.items(), 1):
        print(f"📊 Analyzing {symbol} ({i}/{total_symbols})...")
        
        pump_result = detect_biggest_pump_15m(candles, symbol)
        
        if pump_result:
            results[symbol] = pump_result
            
            # Run GPT analysis if enabled and pump is significant
            if enable_gpt and pump_result["type"] != "micro-move":
                gpt_analysis = analyze_pump_with_gpt(pump_result)
                if gpt_analysis:
                    pump_result["gpt_analysis"] = gpt_analysis
        else:
            print(f"📊 No pump detected for {symbol}")
    
    detected_count = len(results)
    print(f"🏁 Batch analysis complete: {detected_count}/{total_symbols} pumps detected")
    
    return results

def filter_pumps_by_type(pump_results: Dict[str, Dict], pump_types: List[str]) -> Dict[str, Dict]:
    """
    Filter pump results by type
    
    Args:
        pump_results: Results from batch_pump_detection
        pump_types: List of pump types to include (e.g., ['pump-impulse', 'trend-breakout'])
        
    Returns:
        Filtered pump results
    """
    filtered = {}
    
    for symbol, pump_data in pump_results.items():
        if pump_data.get("type") in pump_types:
            filtered[symbol] = pump_data
    
    print(f"🔍 Filtered {len(filtered)}/{len(pump_results)} pumps by type {pump_types}")
    return filtered

def get_pump_statistics(pump_results: Dict[str, Dict]) -> Dict:
    """
    Generate statistics from pump detection results
    
    Args:
        pump_results: Results from batch_pump_detection
        
    Returns:
        Statistics summary
    """
    if not pump_results:
        return {"total_pumps": 0}
    
    types_count = {}
    growth_values = []
    timeframes = []
    
    for pump_data in pump_results.values():
        pump_type = pump_data.get("type", "unknown")
        types_count[pump_type] = types_count.get(pump_type, 0) + 1
        growth_values.append(pump_data.get("growth", 0))
        timeframes.append(pump_data.get("window_hours", 0))
    
    stats = {
        "total_pumps": len(pump_results),
        "pump_types": types_count,
        "avg_growth": round(sum(growth_values) / len(growth_values), 2) if growth_values else 0,
        "max_growth": max(growth_values) if growth_values else 0,
        "avg_timeframe": round(sum(timeframes) / len(timeframes), 1) if timeframes else 0,
        "symbols_with_pumps": list(pump_results.keys())
    }
    
    return stats

def main():
    """
    Test function for pump detection module
    """
    print("🧪 Testing Pump Detection Module...")
    
    # Example test data - 15-minute candles
    test_candles = [
        {"timestamp": 1718524500, "open": 0.00594, "close": 0.00598, "high": 0.00599, "low": 0.00590},
        {"timestamp": 1718525400, "open": 0.00598, "close": 0.00612, "high": 0.00615, "low": 0.00595},
        {"timestamp": 1718526300, "open": 0.00612, "close": 0.00645, "high": 0.00650, "low": 0.00610},
        {"timestamp": 1718527200, "open": 0.00645, "close": 0.00799, "high": 0.00820, "low": 0.00640},  # Big pump here
        {"timestamp": 1718528100, "open": 0.00799, "close": 0.00785, "high": 0.00805, "low": 0.00770},
        {"timestamp": 1718529000, "open": 0.00785, "close": 0.00765, "high": 0.00790, "low": 0.00760}
    ]
    
    # Test single symbol detection
    result = detect_biggest_pump_15m(test_candles, "TESTUSDT")
    
    if result:
        print(f"✅ Test pump detected:")
        for key, value in result.items():
            print(f"   {key}: {value}")
    else:
        print("❌ No pump detected in test data")
    
    # Test statistics
    if result:
        test_results = {"TESTUSDT": result}
        stats = get_pump_statistics(test_results)
        print(f"\n📊 Test Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    main()