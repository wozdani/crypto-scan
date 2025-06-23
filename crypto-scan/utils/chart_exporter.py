"""
Chart Exporter for Vision-AI Training
Exports candlestick charts during token scans for Computer Vision model training
"""

import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict


def save_candlestick_chart(symbol: str, candles: List, output_dir: str = "charts") -> Optional[str]:
    """
    Export candlestick chart with volume for Vision-AI training
    
    Args:
        symbol: Trading symbol
        candles: OHLCV candle data
        output_dir: Output directory for charts
        
    Returns:
        Path to saved chart or None if failed
    """
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not candles or len(candles) < 10:
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
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"{symbol}_{timestamp}.png"
        full_path = os.path.join(output_dir, fname)
        
        # Create chart with mplfinance
        mpf.plot(
            df,
            type='candle',
            volume=True,
            style='charles',
            title=f'{symbol} - Chart Analysis',
            ylabel='Price (USDT)',
            ylabel_lower='Volume',
            figsize=(12, 8),
            savefig=dict(fname=full_path, dpi=150, bbox_inches='tight')
        )
        
        print(f"[CHART EXPORT] Saved: {fname}")
        return full_path
        
    except Exception as e:
        print(f"[CHART EXPORT] Error saving chart for {symbol}: {e}")
        return None


def export_chart_with_indicators(
    symbol: str, 
    candles: List, 
    indicators: Dict = None,
    output_dir: str = "charts"
) -> Optional[str]:
    """
    Export chart with technical indicators for enhanced training data
    
    Args:
        symbol: Trading symbol
        candles: OHLCV candle data
        indicators: Technical indicators to display
        output_dir: Output directory
        
    Returns:
        Path to saved chart
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not candles or len(candles) < 20:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Convert to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        
        # Add moving averages
        addplot = []
        if len(df) >= 20:
            df['EMA20'] = df['close'].ewm(span=20).mean()
            addplot.append(mpf.make_addplot(df['EMA20'], color='blue', width=1))
        
        if len(df) >= 50:
            df['EMA50'] = df['close'].ewm(span=50).mean()
            addplot.append(mpf.make_addplot(df['EMA50'], color='orange', width=1))
        
        # Generate enhanced filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"{symbol}_{timestamp}_enhanced.png"
        full_path = os.path.join(output_dir, fname)
        
        # Create enhanced chart
        mpf.plot(
            df,
            type='candle',
            volume=True,
            addplot=addplot if addplot else None,
            style='charles',
            title=f'{symbol} - Enhanced Analysis',
            ylabel='Price (USDT)',
            ylabel_lower='Volume',
            figsize=(12, 8),
            savefig=dict(fname=full_path, dpi=150, bbox_inches='tight')
        )
        
        print(f"[CHART EXPORT] Enhanced chart saved: {fname}")
        return full_path
        
    except Exception as e:
        print(f"[CHART EXPORT] Error saving enhanced chart for {symbol}: {e}")
        return None


def batch_export_charts(symbols_data: Dict, output_dir: str = "charts") -> Dict:
    """
    Export charts for multiple symbols
    
    Args:
        symbols_data: Dict with symbol -> candles mapping
        output_dir: Output directory
        
    Returns:
        Export results
    """
    results = {
        "exported": [],
        "failed": [],
        "total": len(symbols_data)
    }
    
    for symbol, candles in symbols_data.items():
        try:
            chart_path = save_candlestick_chart(symbol, candles, output_dir)
            if chart_path:
                results["exported"].append({
                    "symbol": symbol,
                    "path": chart_path
                })
            else:
                results["failed"].append(symbol)
        except Exception as e:
            print(f"[BATCH EXPORT] Failed {symbol}: {e}")
            results["failed"].append(symbol)
    
    print(f"[BATCH EXPORT] Exported {len(results['exported'])}/{results['total']} charts")
    return results


def get_export_stats(charts_dir: str = "charts") -> Dict:
    """Get statistics about exported charts"""
    try:
        charts_path = Path(charts_dir)
        if not charts_path.exists():
            return {"total_files": 0, "total_size_mb": 0}
        
        chart_files = list(charts_path.glob("*.png"))
        total_size = sum(f.stat().st_size for f in chart_files)
        
        return {
            "total_files": len(chart_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "charts_dir": str(charts_path),
            "latest_charts": [f.name for f in sorted(chart_files, key=lambda x: x.stat().st_mtime)[-5:]]
        }
    except Exception as e:
        return {"error": str(e)}