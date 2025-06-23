#!/usr/bin/env python3
"""
Chart Exporter for Vision-AI Training
Exports chart snapshots for automatic training data collection
"""

import os
import sys
import pandas as pd
import mplfinance as mpf
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_chart_snapshot(
    symbol: str, 
    candles: List = None,
    timeframe: str = "15m",
    output_dir: str = "data/charts"
) -> Optional[str]:
    """
    Save chart snapshot for training data collection
    
    Args:
        symbol: Trading symbol
        candles: OHLCV candle data
        timeframe: Chart timeframe
        output_dir: Output directory for charts
        
    Returns:
        Path to saved chart or None if failed
    """
    try:
        # Create output directory
        chart_dir = Path(output_dir)
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Get candles if not provided
        if not candles:
            from utils.api_utils import get_bybit_candles
            candles = get_bybit_candles(symbol, timeframe, 100)
        
        if not candles or len(candles) < 20:
            print(f"[CHART EXPORT] Insufficient data for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Convert to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        
        # Generate filename: SYMBOL_YYYYMMDD_HHMM.png
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{symbol}_{timestamp}.png"
        chart_path = chart_dir / filename
        
        # Create professional chart for training
        mpf.plot(
            df,
            type='candle',
            volume=True,
            style='charles',
            title=f'{symbol} - {timeframe} Chart Analysis',
            ylabel='Price (USDT)',
            ylabel_lower='Volume',
            figsize=(12, 8),
            savefig=dict(
                fname=str(chart_path), 
                dpi=150, 
                bbox_inches='tight',
                facecolor='white'
            )
        )
        
        print(f"[CHART EXPORT] Saved: {filename}")
        return str(chart_path)
        
    except Exception as e:
        print(f"[CHART EXPORT] Error saving chart for {symbol}: {e}")
        return None


def export_charts_for_symbols(
    symbols: List[str], 
    timeframe: str = "15m",
    output_dir: str = "data/charts"
) -> Dict:
    """
    Export charts for multiple symbols
    
    Args:
        symbols: List of trading symbols
        timeframe: Chart timeframe
        output_dir: Output directory
        
    Returns:
        Export results summary
    """
    results = {
        "exported": [],
        "failed": [],
        "total": len(symbols)
    }
    
    print(f"[CHART EXPORT] Exporting charts for {len(symbols)} symbols...")
    
    for symbol in symbols:
        try:
            chart_path = save_chart_snapshot(symbol, timeframe=timeframe, output_dir=output_dir)
            
            if chart_path:
                results["exported"].append({
                    "symbol": symbol,
                    "chart_path": chart_path,
                    "filename": os.path.basename(chart_path)
                })
            else:
                results["failed"].append(symbol)
                
        except Exception as e:
            print(f"[CHART EXPORT] Failed to export {symbol}: {e}")
            results["failed"].append(symbol)
    
    print(f"[CHART EXPORT] Completed: {len(results['exported'])}/{results['total']} charts exported")
    return results


def get_chart_export_stats(charts_dir: str = "data/charts") -> Dict:
    """Get statistics about exported charts"""
    try:
        chart_path = Path(charts_dir)
        
        if not chart_path.exists():
            return {"total_charts": 0, "chart_files": []}
        
        chart_files = list(chart_path.glob("*.png"))
        
        # Sort by modification time (newest first)
        chart_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        total_size = sum(f.stat().st_size for f in chart_files)
        
        return {
            "total_charts": len(chart_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "charts_dir": str(chart_path),
            "latest_charts": [f.name for f in chart_files[:10]]  # Latest 10
        }
        
    except Exception as e:
        return {"error": str(e)}


def cleanup_old_charts(charts_dir: str = "data/charts", keep_count: int = 100) -> int:
    """Clean up old chart files, keeping only the most recent ones"""
    try:
        chart_path = Path(charts_dir)
        
        if not chart_path.exists():
            return 0
        
        chart_files = list(chart_path.glob("*.png"))
        
        if len(chart_files) <= keep_count:
            return 0
        
        # Sort by modification time (oldest first)
        chart_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest files
        files_to_remove = chart_files[:-keep_count]
        removed_count = 0
        
        for file_to_remove in files_to_remove:
            try:
                file_to_remove.unlink()
                removed_count += 1
            except Exception as e:
                print(f"[CLEANUP] Failed to remove {file_to_remove.name}: {e}")
        
        if removed_count > 0:
            print(f"[CLEANUP] Removed {removed_count} old chart files")
        
        return removed_count
        
    except Exception as e:
        print(f"[CLEANUP] Chart cleanup failed: {e}")
        return 0


def main():
    """Test chart export functionality"""
    print("📊 Chart Exporter Test")
    print("=" * 30)
    
    # Test single chart export
    test_symbol = "BTCUSDT"
    chart_path = save_chart_snapshot(test_symbol)
    
    if chart_path:
        print(f"✅ Test chart exported: {os.path.basename(chart_path)}")
    else:
        print("❌ Test chart export failed")
    
    # Show statistics
    stats = get_chart_export_stats()
    print(f"\n📈 Chart Statistics:")
    print(f"  Total charts: {stats.get('total_charts', 0)}")
    print(f"  Total size: {stats.get('total_size_mb', 0)} MB")
    
    if stats.get('latest_charts'):
        print(f"  Latest charts:")
        for chart in stats['latest_charts'][:3]:
            print(f"    {chart}")


if __name__ == "__main__":
    main()