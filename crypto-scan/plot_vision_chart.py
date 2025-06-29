#!/usr/bin/env python3
"""
Professional Vision-AI Chart Generator - TradingView Style
Complete replacement for orderbook heatmap charts with candlestick + volume charts
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import os
import json
from datetime import datetime, timezone
from typing import Optional


def plot_vision_chart(
    df: pd.DataFrame,
    symbol: str,
    setup: str,
    market_phase: str,
    decision: str,
    clip_label: str,
    clip_confidence: float,
    tjde_score: float,
    output_path: str,
    timestamp: str = None,
) -> bool:
    """
    Generates and saves a candlestick + volume chart for Vision-AI + GPT labeling.

    Parameters:
        df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        symbol: e.g. "HIPPOUSDT"
        setup: e.g. "pullback", "breakout"
        market_phase: e.g. "uptrend", "consolidation"
        decision: "entry" or "avoid"
        clip_label: CLIP prediction label
        clip_confidence: float between 0.0 and 1.0
        tjde_score: float, score from 0 to 1
        output_path: path to save PNG
        timestamp: optional string used in filename
        
    Returns:
        bool: True if chart was successfully generated and saved
    """
    try:
        # Input validation
        if df is None or df.empty:
            print(f"[VISION CHART ERROR] {symbol}: Empty or invalid DataFrame")
            return False
            
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"[VISION CHART ERROR] {symbol}: Missing columns: {missing_columns}")
            return False

        # Prepare data
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date_num'] = mdates.date2num(df['timestamp'])

        # Validate OHLC data
        df = df.dropna()
        if len(df) < 10:
            print(f"[VISION CHART ERROR] {symbol}: Insufficient data after cleanup - {len(df)} candles")
            return False

        ohlc = df[['date_num', 'open', 'high', 'low', 'close']].values
        volume = df['volume'].values

        # Clear any existing plots
        plt.close('all')

        # Setup figure with professional styling
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={'height_ratios': [4, 1]},
            facecolor='#0e1621'
        )

        # Candlestick chart with enhanced styling
        candlestick_ohlc(ax1, ohlc, width=0.0008, colorup='#26a69a', colordown='#ef5350', alpha=0.9)
        ax1.set_ylabel("Price (USDT)", color='#d1d4dc', fontsize=12)
        ax1.grid(True, alpha=0.2, color='#2a2e39')
        ax1.set_facecolor('#0e1621')
        ax1.tick_params(colors='#d1d4dc')

        # Dynamic title with comprehensive information
        title = (
            f"{symbol} | Setup: {setup.upper()} | Phase: {market_phase.upper()} | "
            f"TJDE: {tjde_score:.3f} | Decision: {decision.upper()}"
        )
        if clip_label != "unknown":
            title += f" | CLIP: {clip_label} ({clip_confidence:.2f})"
        
        ax1.set_title(title, fontsize=12, color='#d1d4dc', pad=20, weight='bold')

        # Volume chart with color coding
        volume_colors = ['#26a69a' if close >= open_val else '#ef5350' 
                        for close, open_val in zip(df['close'], df['open'])]
        
        ax2.bar(df['date_num'], volume, width=0.0008, color=volume_colors, alpha=0.7)
        ax2.set_ylabel("Volume", color='#d1d4dc', fontsize=12)
        ax2.grid(True, alpha=0.2, color='#2a2e39')
        ax2.set_facecolor('#0e1621')
        ax2.tick_params(colors='#d1d4dc')

        # X-axis formatting
        ax2.xaxis_date()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, color='#d1d4dc', fontsize=10)

        # Style spines
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_color('#2a2e39')
                spine.set_linewidth(1)

        # Add price information box
        current_price = df['close'].iloc[-1]
        price_change = ((current_price - df['open'].iloc[0]) / df['open'].iloc[0]) * 100
        
        info_text = f"Price: ${current_price:.6f}\nChange: {price_change:+.2f}%\nCandles: {len(df)}"
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                fontsize=10, color='#d1d4dc', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='#2a2e39', alpha=0.8), verticalalignment='top')

        # Add TJDE score visualization
        score_color = '#26a69a' if tjde_score >= 0.7 else '#ffa726' if tjde_score >= 0.5 else '#ef5350'
        ax1.text(0.98, 0.98, f"TJDE: {tjde_score:.3f}", transform=ax1.transAxes,
                fontsize=12, color=score_color, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=score_color, alpha=0.2),
                horizontalalignment='right', verticalalignment='top')

        plt.tight_layout()

        # Create folder if missing
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save file with high quality
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0e1621', 
                   edgecolor='none', format='png')
        plt.close('all')

        # Validate saved file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 5120:  # At least 5KB
            print(f"[VISION CHART] ✅ Professional chart saved: {output_path} ({os.path.getsize(output_path)} bytes)")
            
            # Create comprehensive metadata
            metadata = {
                "symbol": symbol,
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                "chart_type": "candlestick_volume",
                "setup": setup,
                "market_phase": market_phase,
                "decision": decision,
                "tjde_score": tjde_score,
                "clip_label": clip_label,
                "clip_confidence": clip_confidence,
                "candles_count": len(df),
                "timeframe": "15M",
                "price_range": {
                    "high": float(df['high'].max()),
                    "low": float(df['low'].min()),
                    "current": float(current_price),
                    "change_percent": float(price_change)
                },
                "volume_stats": {
                    "total": float(df['volume'].sum()),
                    "average": float(df['volume'].mean()),
                    "max": float(df['volume'].max())
                },
                "chart_path": output_path,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "vision_ai_optimized": True,
                "tradingview_style": True
            }
            
            # Save metadata
            metadata_path = output_path.replace('.png', '.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        else:
            print(f"[VISION CHART ERROR] {symbol}: Generated file too small or missing")
            return False

    except Exception as e:
        print(f"[VISION CHART ERROR] {symbol}: Failed to generate chart: {e}")
        plt.close('all')
        return False


def convert_candles_to_dataframe(candles, symbol: str) -> Optional[pd.DataFrame]:
    """
    Convert various candle formats to DataFrame for plot_vision_chart
    
    Args:
        candles: List of candle data (dict or list format)
        symbol: Trading symbol for logging
        
    Returns:
        DataFrame with required columns or None if conversion fails
    """
    try:
        df_data = []
        
        for i, candle in enumerate(candles):
            try:
                if isinstance(candle, dict):
                    # Dictionary format
                    timestamp = candle.get('timestamp', candle.get('time', i * 900000))
                    open_price = float(candle.get('open', candle.get('Open', 1.0)))
                    high_price = float(candle.get('high', candle.get('High', 1.0)))
                    low_price = float(candle.get('low', candle.get('Low', 1.0)))
                    close_price = float(candle.get('close', candle.get('Close', 1.0)))
                    volume = float(candle.get('volume', candle.get('Volume', 1000.0)))
                elif isinstance(candle, list) and len(candle) >= 6:
                    # List format: [timestamp, open, high, low, close, volume]
                    timestamp = int(candle[0])
                    open_price = float(candle[1])
                    high_price = float(candle[2])
                    low_price = float(candle[3])
                    close_price = float(candle[4])
                    volume = float(candle[5])
                else:
                    continue
                
                # Convert timestamp to datetime
                if timestamp > 1e12:  # Milliseconds
                    dt = pd.to_datetime(timestamp, unit='ms')
                else:  # Seconds
                    dt = pd.to_datetime(timestamp, unit='s')
                
                # Validate OHLC data
                if all(p > 0 for p in [open_price, high_price, low_price, close_price]):
                    if (low_price <= high_price and 
                        min(open_price, close_price) >= low_price and 
                        max(open_price, close_price) <= high_price):
                        
                        df_data.append({
                            'timestamp': dt,
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': volume
                        })
            except (ValueError, TypeError, IndexError) as e:
                print(f"[CANDLE CONVERT] {symbol}: Skipping invalid candle {i}: {e}")
                continue
        
        if len(df_data) < 10:
            print(f"[CANDLE CONVERT ERROR] {symbol}: Insufficient valid data - got {len(df_data)} valid candles")
            return None
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp')
        print(f"[CANDLE CONVERT] {symbol}: Successfully converted {len(df)} candles to DataFrame")
        return df
        
    except Exception as e:
        print(f"[CANDLE CONVERT ERROR] {symbol}: Failed to convert candles: {e}")
        return None


if __name__ == "__main__":
    # Test the new chart generator
    print("Testing new Vision-AI chart generator...")
    
    # Create sample data
    import numpy as np
    
    dates = pd.date_range(start='2025-01-01', periods=50, freq='15min')
    base_price = 50000.0
    
    test_data = []
    for i, date in enumerate(dates):
        # Simulate realistic price movement
        price_change = np.random.normal(0, 0.002)  # 0.2% std deviation
        base_price *= (1 + price_change)
        
        # Generate realistic OHLC
        open_price = base_price
        close_price = base_price * (1 + np.random.normal(0, 0.001))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0005)))
        volume = np.random.uniform(1000, 10000)
        
        test_data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df_test = pd.DataFrame(test_data)
    
    # Test chart generation
    result = plot_vision_chart(
        df=df_test,
        symbol="TESTUSDT",
        setup="breakout",
        market_phase="uptrend",
        decision="entry",
        clip_label="trend_continuation",
        clip_confidence=0.85,
        tjde_score=0.72,
        output_path="training_data/charts/TESTUSDT_20250627_vision_chart.png"
    )
    
    if result:
        print("✅ Test chart generated successfully")
    else:
        print("❌ Test chart generation failed")