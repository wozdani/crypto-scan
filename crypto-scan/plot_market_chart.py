"""
Professional Market Chart Generator - TradingView/Bybit Style
Zastępuje obecny moduł generowania wykresów z poprawnymi proporcjami OHLC
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import datetime
import os
import json
from typing import List, Dict, Optional

# Set non-interactive backend for production
import matplotlib
matplotlib.use('Agg')

def plot_market_chart(candles: List, alert_info: Dict, filename: str = "chart.png") -> bool:
    """
    Generate professional market chart with TradingView/Bybit styling
    
    Args:
        candles: List of [timestamp_ms, open, high, low, close, volume]
        alert_info: Dictionary with symbol, phase, setup, score, decision, clip
        filename: Output filename
        
    Returns:
        bool: Success status
    """
    try:
        if not candles or len(candles) < 2:
            print(f"[MARKET CHART] ❌ Insufficient candle data: {len(candles) if candles else 0}")
            return False
            
        # Convert timestamps and OHLCV data
        times = []
        opens, highs, lows, closes, volumes = [], [], [], [], []
        
        for i, candle in enumerate(candles):
            try:
                # Handle different candle formats
                if isinstance(candle, dict):
                    timestamp = candle.get('timestamp', candle.get('time', 0))
                    open_price = float(candle.get('open', 0))
                    high_price = float(candle.get('high', 0))
                    low_price = float(candle.get('low', 0))
                    close_price = float(candle.get('close', 0))
                    volume = float(candle.get('volume', 0))
                elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                    timestamp = candle[0]
                    open_price = float(candle[1])
                    high_price = float(candle[2])
                    low_price = float(candle[3])
                    close_price = float(candle[4])
                    volume = float(candle[5])
                else:
                    continue
                    
                # Convert timestamp to datetime
                if timestamp > 1e12:  # milliseconds
                    dt = datetime.datetime.utcfromtimestamp(timestamp / 1000)
                else:  # seconds
                    dt = datetime.datetime.utcfromtimestamp(timestamp)
                    
                times.append(dt)
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                volumes.append(volume)
                
            except (ValueError, TypeError, KeyError) as e:
                print(f"[MARKET CHART] Warning: Skipping invalid candle {i}: {e}")
                continue
        
        if len(times) < 2:
            print(f"[MARKET CHART] ❌ No valid candles processed")
            return False
            
        # Create figure with professional styling
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, 
                                      gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor('#0e1621')  # TradingView dark background
        fig.subplots_adjust(hspace=0.1)
        
        # Calculate time intervals for proper candle width
        if len(times) > 1:
            time_diff = (times[1] - times[0]).total_seconds() / 60  # minutes
            candle_width = datetime.timedelta(minutes=time_diff * 0.8)
        else:
            candle_width = datetime.timedelta(minutes=12)  # 15M default
        
        # Plot candlesticks with proper OHLC representation
        for i in range(len(times)):
            # Determine candle colors (TradingView style)
            if closes[i] >= opens[i]:
                color = '#26a69a'  # Green for bullish
                edge_color = '#26a69a'
            else:
                color = '#ef5350'  # Red for bearish  
                edge_color = '#ef5350'
            
            # Draw high-low line (wick)
            ax1.plot([times[i], times[i]], [lows[i], highs[i]], 
                    color=edge_color, linewidth=1.5, alpha=0.9)
            
            # Draw OHLC body (rectangle)
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            
            if body_height > 0:  # Normal candle
                rect = Rectangle(
                    (times[i] - candle_width/2, body_bottom),
                    candle_width,
                    body_height,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=0.8,
                    alpha=0.9
                )
                ax1.add_patch(rect)
            else:  # Doji candle
                ax1.plot([times[i] - candle_width/2, times[i] + candle_width/2], 
                        [opens[i], opens[i]], color=edge_color, linewidth=2)
        
        # Mark alert candle (LAST candle as requested)
        alert_index = len(times) - 1
        ax1.axvline(times[alert_index], color='#00ff00', linestyle='--', 
                   linewidth=2, alpha=0.8, label='Alert')
        ax1.scatter(times[alert_index], closes[alert_index], 
                   color='#00ff00', s=100, marker='o', zorder=10, 
                   edgecolors='white', linewidth=2)
        
        # Professional info box (TradingView style)
        info_text = (
            f"PHASE: {alert_info.get('phase', 'unknown').upper()}\n"
            f"SETUP: {alert_info.get('setup', 'unknown').upper()}\n"
            f"TJDE: {alert_info.get('score', 0.0):.3f}\n"
            f"DECISION: {alert_info.get('decision', 'N/A').upper()}\n"
            f"CLIP: {alert_info.get('clip', 0.0):.3f}"
        )
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#1e2329', 
                         edgecolor='#434651', alpha=0.95),
                color='white')
        
        # Strict Y-axis scaling (±1% around median as requested)
        all_prices = highs + lows
        mid_price = (max(all_prices) + min(all_prices)) / 2
        price_range = mid_price * 0.01  # ±1% range
        ax1.set_ylim(mid_price - price_range, mid_price + price_range)
        
        # Professional title with all context
        title = (f"{alert_info.get('symbol', 'SYMBOL')} | "
                f"{alert_info.get('phase', '').upper()} | "
                f"TJDE: {alert_info.get('score', 0.0):.3f} | "
                f"{alert_info.get('setup', '').upper()}")
        ax1.set_title(title, fontsize=14, color='white', fontweight='bold', pad=20)
        
        # Volume histogram (lower panel)
        volume_colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' 
                        for i in range(len(times))]
        ax2.bar(times, volumes, width=candle_width, color=volume_colors, 
               alpha=0.7, edgecolor='none')
        
        # Volume alert marker
        ax2.axvline(times[alert_index], color='#00ff00', linestyle='--', 
                   linewidth=2, alpha=0.8)
        
        # Styling for both axes
        ax1.set_ylabel("Price (USDT)", color='white', fontsize=12)
        ax2.set_ylabel("Volume", color='white', fontsize=12)
        
        # Professional grid (TradingView style)
        ax1.grid(True, linestyle='-', alpha=0.1, color='white')
        ax2.grid(True, linestyle='-', alpha=0.1, color='white')
        
        # Time formatting
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
        
        # Axis colors
        ax1.tick_params(colors='white')
        ax2.tick_params(colors='white')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['right'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['right'].set_color('white')
        ax2.spines['left'].set_color('white')
        
        plt.xlabel("Time (Alert Context)", color='white', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save with high quality
        plt.savefig(filename, dpi=200, bbox_inches='tight', 
                   facecolor='#0e1621', edgecolor='none')
        plt.close()
        
        # Generate metadata JSON
        metadata = {
            "symbol": alert_info.get('symbol', 'UNKNOWN'),
            "timestamp": datetime.datetime.now().isoformat(),
            "chart_type": "professional_market_chart",
            "phase": alert_info.get('phase', 'unknown'),
            "setup": alert_info.get('setup', 'unknown'),
            "decision": alert_info.get('decision', 'unknown'),
            "tjde_score": alert_info.get('score', 0.0),
            "clip_confidence": alert_info.get('clip', 0.0),
            "candles_count": len(times),
            "alert_candle_index": alert_index,
            "price_range": {
                "high": max(all_prices),
                "low": min(all_prices),
                "mid": mid_price,
                "range_percent": 1.0
            },
            "volume_stats": {
                "total": sum(volumes),
                "average": sum(volumes) / len(volumes) if volumes else 0,
                "max": max(volumes) if volumes else 0
            },
            "chart_style": "tradingview_professional",
            "alert_position": "last_candle"
        }
        
        # Save metadata
        json_filename = filename.replace('.png', '.json')
        with open(json_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        file_size = os.path.getsize(filename)
        print(f"[MARKET CHART] ✅ Professional chart saved: {filename} ({file_size} bytes)")
        return True
        
    except Exception as e:
        print(f"[MARKET CHART] ❌ Generation failed: {e}")
        return False

def test_market_chart():
    """Test the professional market chart generator"""
    # Sample data
    import time
    current_time = int(time.time() * 1000)
    
    candles = []
    base_price = 1.234567
    
    for i in range(48):  # 48 x 15M = 12 hours
        timestamp = current_time - (47-i) * 15 * 60 * 1000
        price_change = (i - 24) * 0.001  # Gradual trend
        
        open_price = base_price + price_change
        close_price = open_price + ((i % 3 - 1) * 0.002)
        high_price = max(open_price, close_price) * 1.005
        low_price = min(open_price, close_price) * 0.995
        volume = 1000 + (i * 100)
        
        candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
    
    alert_info = {
        "symbol": "TESTUSDT",
        "phase": "trend-following",
        "setup": "continuation",
        "score": 0.483,
        "decision": "avoid", 
        "clip": 0.539
    }
    
    return plot_market_chart(candles, alert_info, "training_charts/test_professional_chart.png")

if __name__ == "__main__":
    success = test_market_chart()
    print(f"Test result: {'SUCCESS' if success else 'FAILED'}")