"""
Chart Generator for Computer Vision Training
Generates realistic chart patterns for ML training when market data is unavailable
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Optional
import os
from pathlib import Path


def create_pattern_chart(
    pattern_type: str,
    symbol: str = "TRAINING",
    save_path: str = None,
    style: str = "professional"
) -> Optional[str]:
    """
    Create chart with specific pattern for Computer Vision training
    
    Args:
        pattern_type: Type of pattern to generate
        symbol: Symbol name for the chart
        save_path: Custom save path
        style: Chart style
        
    Returns:
        Path to saved chart or None if failed
    """
    try:
        # Generate pattern-specific data
        candles = _generate_pattern_data(pattern_type, 96)
        
        if not candles:
            return None
        
        # Create chart image
        return _create_chart_image(candles, pattern_type, symbol, save_path, style)
        
    except Exception as e:
        print(f"[CHART GEN] Error creating {pattern_type} chart: {e}")
        return None


def _generate_pattern_data(pattern_type: str, count: int) -> List[List]:
    """Generate realistic OHLCV data for specific patterns"""
    
    base_price = 50000
    candles = []
    current_time = datetime.now()
    
    for i in range(count):
        timestamp = int((current_time - timedelta(minutes=15 * (count - 1 - i))).timestamp() * 1000)
        
        # Pattern-specific price movement
        if pattern_type == "breakout_continuation":
            if i < 40:  # Consolidation phase
                price_change = np.random.normal(0, base_price * 0.002)
            elif i < 60:  # Breakout phase
                price_change = np.random.normal(base_price * 0.005, base_price * 0.004)
            else:  # Continuation phase
                price_change = np.random.normal(base_price * 0.003, base_price * 0.003)
                
        elif pattern_type == "pullback_setup":
            if i < 30:  # Initial uptrend
                price_change = np.random.normal(base_price * 0.003, base_price * 0.002)
            elif i < 60:  # Pullback phase
                price_change = np.random.normal(-base_price * 0.002, base_price * 0.003)
            else:  # Recovery phase
                price_change = np.random.normal(base_price * 0.004, base_price * 0.003)
                
        elif pattern_type == "trend_exhaustion":
            if i < 50:  # Strong trend
                price_change = np.random.normal(base_price * 0.004, base_price * 0.002)
            else:  # Exhaustion/sideways
                price_change = np.random.normal(0, base_price * 0.001)
                
        else:  # Default trending pattern
            price_change = np.random.normal(base_price * 0.001, base_price * 0.003)
        
        # Create OHLC
        open_price = base_price
        close_price = open_price + price_change
        high_price = max(open_price, close_price) + abs(np.random.normal(0, base_price * 0.001))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, base_price * 0.001))
        volume = np.random.uniform(800, 2500)
        
        candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
        base_price = close_price
    
    return candles


def _create_chart_image(
    candles: List[List],
    pattern_type: str,
    symbol: str,
    save_path: str = None,
    style: str = "professional"
) -> str:
    """Create chart image from candle data"""
    
    # Process data
    times = [datetime.fromtimestamp(c[0] / 1000) for c in candles]
    opens = [float(c[1]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    volumes = [float(c[5]) for c in candles]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[4, 1], facecolor='white')
    
    # Draw candlesticks
    for i in range(len(times)):
        color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
        
        # Wicks
        ax1.plot([times[i], times[i]], [lows[i], highs[i]], color=color, linewidth=1, alpha=0.8)
        
        # Bodies
        body_height = abs(closes[i] - opens[i])
        body_bottom = min(opens[i], closes[i])
        ax1.bar(times[i], body_height, bottom=body_bottom, color=color, width=0.6, alpha=0.9)
    
    # Add EMA
    if len(closes) >= 20:
        ema = _calculate_ema(closes, 20)
        if ema:
            ax1.plot(times[19:], ema, label="EMA20", color='#2196f3', linewidth=2, alpha=0.8)
            ax1.legend(loc='upper left')
    
    # Style price chart
    ax1.set_title(f"{symbol} - {pattern_type.replace('_', ' ').title()}", fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
    
    # Volume chart
    volume_colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' for i in range(len(closes))]
    ax2.bar(times, volumes, color=volume_colors, alpha=0.7, width=0.6)
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Save chart
    if not save_path:
        export_dir = Path("data/chart_exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = export_dir / f"pattern_{pattern_type}_{symbol}_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"[CHART GEN] Pattern chart saved: {save_path}")
    return str(save_path)


def _calculate_ema(prices: List[float], period: int) -> Optional[List[float]]:
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return None
    
    ema = []
    multiplier = 2 / (period + 1)
    ema.append(sum(prices[:period]) / period)
    
    for i in range(period, len(prices)):
        ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
        ema.append(ema_value)
    
    return ema


def create_training_charts(patterns: List[str], charts_per_pattern: int = 3) -> List[str]:
    """Create multiple training charts for different patterns"""
    
    created_charts = []
    
    for pattern in patterns:
        for i in range(charts_per_pattern):
            chart_path = create_pattern_chart(
                pattern_type=pattern,
                symbol=f"CV_{pattern.upper()}_{i+1}",
                style="professional"
            )
            
            if chart_path:
                created_charts.append(chart_path)
    
    return created_charts