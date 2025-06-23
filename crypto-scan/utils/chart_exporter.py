"""
Chart Exporter Module - Export candlestick charts as training data for Computer Vision AI

Generates high-quality chart images for pattern recognition training.
Supports various timeframes and customizable chart parameters.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# Import safe candles function
try:
    from utils.safe_candles import get_candles
except ImportError:
    # Fallback to API utils
    try:
        from utils.api_utils import get_bybit_candles
        def get_candles(symbol, timeframe="15m", limit=100):
            print(f"Using API fallback for {symbol}")
            return get_bybit_candles(symbol, timeframe, limit)
    except ImportError:
        def get_candles(symbol, timeframe="15m", limit=100):
            print(f"Warning: No candle data source available for {symbol}")
            return None

EXPORT_FOLDER = "data/chart_exports"
os.makedirs(EXPORT_FOLDER, exist_ok=True)


def export_chart_image(
    symbol: str, 
    timeframe: str = "15m", 
    limit: int = 96, 
    save_as: str = "auto",
    include_volume: bool = True,
    include_ema: bool = True,
    chart_style: str = "professional"
) -> Optional[str]:
    """
    Export candlestick chart as PNG image for Computer Vision training
    
    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        timeframe: Chart timeframe ("15m", "1h", "4h", "1d")
        limit: Number of candles to include
        save_as: Filename or "auto" for timestamp-based naming
        include_volume: Whether to include volume subplot
        include_ema: Whether to include EMA line
        chart_style: Chart style ("professional", "clean", "detailed")
        
    Returns:
        Full path to saved image or None if failed
    """
    try:
        print(f"[CHART EXPORT] Exporting {symbol} chart ({timeframe}, {limit} candles)...")
        
        # Get candle data using correct parameter format
        candles = get_candles(symbol, timeframe, limit)
        if not candles or len(candles) < limit:
            print(f"[CHART EXPORT] ❌ Insufficient data for {symbol} (got {len(candles) if candles else 0}, need {limit})")
            return None

        # Process candle data
        times = [datetime.fromtimestamp(c[0] / 1000) for c in candles]
        opens = [float(c[1]) for c in candles]
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        closes = [float(c[4]) for c in candles]
        volumes = [float(c[5]) for c in candles]

        # Create figure based on style
        if chart_style == "professional":
            return _create_professional_chart(
                symbol, timeframe, times, opens, highs, lows, closes, volumes,
                save_as, include_volume, include_ema
            )
        elif chart_style == "clean":
            return _create_clean_chart(
                symbol, timeframe, times, opens, highs, lows, closes, volumes,
                save_as, include_volume, include_ema
            )
        elif chart_style == "detailed":
            return _create_detailed_chart(
                symbol, timeframe, times, opens, highs, lows, closes, volumes,
                save_as, include_volume, include_ema
            )
        else:
            print(f"[CHART EXPORT] ❌ Unknown chart style: {chart_style}")
            return None

    except Exception as e:
        print(f"[CHART EXPORT] ❌ Error exporting {symbol}: {e}")
        return None


def _create_professional_chart(
    symbol: str, timeframe: str, times: List, opens: List, highs: List, 
    lows: List, closes: List, volumes: List, save_as: str, 
    include_volume: bool, include_ema: bool
) -> Optional[str]:
    """Create professional-style chart for CV training"""
    try:
        # Figure setup
        height_ratios = [4, 1] if include_volume else [1]
        fig, axes = plt.subplots(
            len(height_ratios), 1, 
            figsize=(12, 8), 
            height_ratios=height_ratios,
            facecolor='white'
        )
        
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        
        ax1 = axes[0]
        fig.subplots_adjust(hspace=0.1)

        # Draw candlesticks with professional styling
        for i in range(len(times)):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'  # Teal/Red
            
            # Candlestick body
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            
            # Wicks
            ax1.plot([times[i], times[i]], [lows[i], highs[i]], 
                    color=color, linewidth=1, alpha=0.8)
            
            # Body (rectangle)
            ax1.bar(times[i], body_height, bottom=body_bottom, 
                   color=color, width=0.6, alpha=0.9)

        # Add EMA if requested
        if include_ema and len(closes) >= 20:
            ema_period = 20
            ema_values = _calculate_ema(closes, ema_period)
            if ema_values:
                ax1.plot(times[ema_period-1:], ema_values, 
                        label="EMA20", color='#2196f3', linewidth=2, alpha=0.8)
                ax1.legend(loc='upper left', fontsize=10)

        # Chart styling
        ax1.set_title(f"{symbol} - {timeframe.upper()} Chart Analysis", 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
        
        # Remove top and right spines for cleaner look
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Volume subplot
        if include_volume and len(axes) > 1:
            ax2 = axes[1]
            volume_colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' 
                           for i in range(len(closes))]
            
            ax2.bar(times, volumes, color=volume_colors, alpha=0.7, width=0.6)
            ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Save the chart
        if save_as == "auto":
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timeframe}_{timestamp}_professional.png"
        else:
            filename = save_as

        full_path = os.path.join(EXPORT_FOLDER, filename)
        plt.savefig(full_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"[CHART EXPORT] ✅ Professional chart saved: {full_path}")
        return full_path

    except Exception as e:
        print(f"[CHART EXPORT] ❌ Professional chart creation failed: {e}")
        plt.close('all')
        return None


def _create_clean_chart(
    symbol: str, timeframe: str, times: List, opens: List, highs: List, 
    lows: List, closes: List, volumes: List, save_as: str, 
    include_volume: bool, include_ema: bool
) -> Optional[str]:
    """Create clean minimalist chart for CV training"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

        # Simple candlesticks - black and white for pattern focus
        for i in range(len(times)):
            color = 'black' if closes[i] >= opens[i] else 'gray'
            
            # Wicks
            ax.plot([times[i], times[i]], [lows[i], highs[i]], 
                   color=color, linewidth=1)
            
            # Bodies - simple lines
            ax.plot([times[i], times[i]], [opens[i], closes[i]], 
                   color=color, linewidth=3)

        # Minimal styling
        ax.set_title(f"{symbol} - Clean Chart", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        # Remove all spines for ultra-clean look
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Minimal axis formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)

        # Save
        if save_as == "auto":
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timeframe}_{timestamp}_clean.png"
        else:
            filename = save_as

        full_path = os.path.join(EXPORT_FOLDER, filename)
        plt.savefig(full_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"[CHART EXPORT] ✅ Clean chart saved: {full_path}")
        return full_path

    except Exception as e:
        print(f"[CHART EXPORT] ❌ Clean chart creation failed: {e}")
        plt.close('all')
        return None


def _create_detailed_chart(
    symbol: str, timeframe: str, times: List, opens: List, highs: List, 
    lows: List, closes: List, volumes: List, save_as: str, 
    include_volume: bool, include_ema: bool
) -> Optional[str]:
    """Create detailed chart with multiple indicators for CV training"""
    try:
        height_ratios = [5, 2, 1] if include_volume else [5, 2]
        fig, axes = plt.subplots(
            len(height_ratios), 1, 
            figsize=(14, 10), 
            height_ratios=height_ratios,
            facecolor='white'
        )
        
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        
        ax1 = axes[0]  # Price chart
        fig.subplots_adjust(hspace=0.2)

        # Enhanced candlesticks
        for i in range(len(times)):
            color = '#00e676' if closes[i] >= opens[i] else '#ff1744'  # Bright green/red
            
            # Enhanced wicks
            ax1.plot([times[i], times[i]], [lows[i], highs[i]], 
                    color=color, linewidth=1.5, alpha=0.9)
            
            # Enhanced bodies
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            ax1.bar(times[i], body_height, bottom=body_bottom, 
                   color=color, width=0.8, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Multiple EMAs
        if include_ema and len(closes) >= 50:
            ema20 = _calculate_ema(closes, 20)
            ema50 = _calculate_ema(closes, 50)
            
            if ema20:
                ax1.plot(times[19:], ema20, label="EMA20", color='blue', linewidth=2)
            if ema50:
                ax1.plot(times[49:], ema50, label="EMA50", color='orange', linewidth=2)
            
            ax1.legend(loc='upper left', fontsize=12)

        # Price chart styling
        ax1.set_title(f"{symbol} - {timeframe.upper()} Detailed Analysis", 
                     fontsize=18, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Price (USDT)', fontsize=14, fontweight='bold')

        # RSI subplot
        if len(axes) > 1:
            ax2 = axes[1]
            rsi_values = _calculate_rsi(closes, 14)
            if rsi_values:
                ax2.plot(times[14:], rsi_values, color='purple', linewidth=2)
                ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
                ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
                ax2.set_ylabel('RSI', fontsize=12, fontweight='bold')
                ax2.set_ylim(0, 100)
                ax2.grid(True, alpha=0.3)

        # Volume subplot
        if include_volume and len(axes) > 2:
            ax3 = axes[2]
            volume_colors = ['#00e676' if closes[i] >= opens[i] else '#ff1744' 
                           for i in range(len(closes))]
            
            ax3.bar(times, volumes, color=volume_colors, alpha=0.8, width=0.8)
            ax3.set_ylabel('Volume', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)

        # Format all x-axes
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Save
        if save_as == "auto":
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timeframe}_{timestamp}_detailed.png"
        else:
            filename = save_as

        full_path = os.path.join(EXPORT_FOLDER, filename)
        plt.savefig(full_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"[CHART EXPORT] ✅ Detailed chart saved: {full_path}")
        return full_path

    except Exception as e:
        print(f"[CHART EXPORT] ❌ Detailed chart creation failed: {e}")
        plt.close('all')
        return None


def _calculate_ema(prices: List[float], period: int) -> Optional[List[float]]:
    """Calculate Exponential Moving Average"""
    try:
        if len(prices) < period:
            return None
        
        ema = []
        multiplier = 2 / (period + 1)
        ema.append(sum(prices[:period]) / period)  # SMA for first value
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_value)
        
        return ema
    except Exception:
        return None


def _calculate_rsi(prices: List[float], period: int = 14) -> Optional[List[float]]:
    """Calculate Relative Strength Index"""
    try:
        if len(prices) < period + 1:
            return None
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    except Exception:
        return None


def export_multiple_charts(
    symbols: List[str], 
    timeframes: List[str] = ["15m", "1h"], 
    styles: List[str] = ["professional", "clean"],
    limit: int = 96
) -> Dict[str, List[str]]:
    """
    Export charts for multiple symbols in different styles for CV training
    
    Args:
        symbols: List of trading symbols
        timeframes: List of timeframes to export
        styles: List of chart styles to generate
        limit: Number of candles per chart
        
    Returns:
        Dictionary mapping symbols to list of exported file paths
    """
    results = {}
    
    for symbol in symbols:
        symbol_results = []
        
        for timeframe in timeframes:
            for style in styles:
                try:
                    file_path = export_chart_image(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=limit,
                        chart_style=style,
                        include_volume=True,
                        include_ema=True
                    )
                    
                    if file_path:
                        symbol_results.append(file_path)
                        
                except Exception as e:
                    print(f"[CHART EXPORT] ❌ Failed {symbol} {timeframe} {style}: {e}")
        
        results[symbol] = symbol_results
        print(f"[CHART EXPORT] Completed {symbol}: {len(symbol_results)} charts exported")
    
    return results


def get_export_stats() -> Dict:
    """Get statistics about exported charts"""
    try:
        export_path = Path(EXPORT_FOLDER)
        if not export_path.exists():
            return {"total_files": 0, "total_size_mb": 0}
        
        files = list(export_path.glob("*.png"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "export_folder": str(export_path),
            "latest_exports": [f.name for f in sorted(files, key=lambda x: x.stat().st_mtime)[-5:]]
        }
    except Exception as e:
        return {"error": str(e)}