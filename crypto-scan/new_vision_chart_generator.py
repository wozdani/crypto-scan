#!/usr/bin/env python3
"""
New Vision-AI Chart Generator - TradingView Style Candlestick Charts
Complete replacement for orderbook heatmap charts with professional candlestick + volume charts
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Union


def plot_vision_chart(symbol: str, candles: List[Dict], setup: str = "unknown", 
                     decision: str = "unknown", clip_label: str = "unknown", 
                     clip_confidence: float = 0.0, tjde_score: float = 0.0, 
                     save_path: str = None, context_days: int = 2) -> Optional[str]:
    """
    Generate professional TradingView-style candlestick chart for Vision-AI training
    
    Args:
        symbol: Trading symbol (e.g. "BTCUSDT")
        candles: List of OHLCV candle data
        setup: Market setup (breakout, pullback, consolidation, etc.)
        decision: Trading decision (entry, avoid, consider)
        clip_label: CLIP prediction label
        clip_confidence: CLIP confidence score (0.0-1.0)
        tjde_score: TJDE score (0.0-1.0)
        save_path: Custom save path
        context_days: Days of context (default 2 = 48 candles of 15M)
        
    Returns:
        Path to saved chart file or None if failed
    """
    try:
        # Input validation
        if not candles or len(candles) < 20:
            print(f"[VISION CHART ERROR] {symbol}: Insufficient candles - got {len(candles)} (need ≥20)")
            return None
        
        # Determine candle count based on context
        max_candles = context_days * 96  # 96 candles per day (15M)
        use_candles = candles[-min(max_candles, len(candles)):]
        
        print(f"[VISION CHART] {symbol}: Processing {len(use_candles)} candles for {context_days} days context")
        
        # Convert candles to DataFrame with proper timestamp handling
        df_data = []
        for i, candle in enumerate(use_candles):
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
                print(f"[VISION CHART] {symbol}: Skipping invalid candle {i}: {e}")
                continue
        
        if len(df_data) < 20:
            print(f"[VISION CHART ERROR] {symbol}: Insufficient valid data - got {len(df_data)} valid candles")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp')
        df['date_num'] = mdates.date2num(df['timestamp'])
        
        # Prepare OHLC data for candlestick_ohlc
        ohlc = df[['date_num', 'open', 'high', 'low', 'close']].values
        
        # Generate save path
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = f"training_charts/{symbol}_{timestamp}_vision_chart.png"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Clear any existing plots
        plt.close('all')
        
        # Create figure with TradingView styling
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                      gridspec_kw={'height_ratios': [4, 1]}, 
                                      facecolor='#0e1621')
        
        # Plot candlesticks
        candlestick_ohlc(ax1, ohlc, width=0.0008, colorup='#26a69a', colordown='#ef5350')
        
        # Configure price chart styling
        ax1.set_facecolor('#0e1621')
        ax1.grid(True, alpha=0.2, color='#2a2e39', linestyle='-', linewidth=0.5)
        ax1.tick_params(colors='#d1d4dc', labelsize=10)
        
        # Professional title with all context
        title = f"{symbol} | Setup: {setup.upper()} | TJDE: {tjde_score:.3f} | Decision: {decision.upper()}"
        if clip_label != "unknown":
            title += f" | CLIP: {clip_label} ({clip_confidence:.2f})"
        
        ax1.set_title(title, fontsize=14, color='#d1d4dc', pad=20, weight='bold')
        
        # Format x-axis for price chart
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df_data)//10)))
        
        # Plot volume with color coding
        volume_colors = ['#26a69a' if close >= open_val else '#ef5350' 
                        for close, open_val in zip(df['close'], df['open'])]
        
        ax2.bar(df['date_num'], df['volume'], width=0.0008, color=volume_colors, 
                alpha=0.8, edgecolor='none')
        
        # Configure volume chart styling
        ax2.set_facecolor('#0e1621')
        ax2.grid(True, alpha=0.2, color='#2a2e39', linestyle='-', linewidth=0.5)
        ax2.tick_params(colors='#d1d4dc', labelsize=10)
        ax2.set_ylabel('Volume', color='#d1d4dc', fontsize=12)
        
        # Format x-axis for volume chart
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df_data)//10)))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, fontsize=9)
        
        # Style spines
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_color('#2a2e39')
                spine.set_linewidth(1)
        
        # Add price information box
        current_price = df['close'].iloc[-1]
        price_change = ((current_price - df['open'].iloc[0]) / df['open'].iloc[0]) * 100
        
        info_text = f"Price: ${current_price:.6f}\nChange: {price_change:+.2f}%\nCandles: {len(df_data)}"
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                fontsize=10, color='#d1d4dc', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='#2a2e39', alpha=0.8), verticalalignment='top')
        
        # Add TJDE score visualization
        score_color = '#26a69a' if tjde_score >= 0.7 else '#ffa726' if tjde_score >= 0.5 else '#ef5350'
        ax1.text(0.98, 0.98, f"TJDE: {tjde_score:.3f}", transform=ax1.transAxes,
                fontsize=12, color=score_color, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=score_color, alpha=0.2),
                horizontalalignment='right', verticalalignment='top')
        
        # Save chart
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#0e1621', 
                   edgecolor='none', format='png')
        plt.close('all')
        
        # Validate saved file
        if os.path.exists(save_path) and os.path.getsize(save_path) > 5120:  # At least 5KB
            print(f"[VISION CHART] ✅ Generated professional chart: {save_path} ({os.path.getsize(save_path)} bytes)")
        else:
            print(f"[VISION CHART ERROR] {symbol}: Generated file too small or missing")
            return None
        
        # Create comprehensive metadata
        metadata = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chart_type": "candlestick_volume",
            "setup": setup,
            "decision": decision,
            "tjde_score": tjde_score,
            "clip_label": clip_label,
            "clip_confidence": clip_confidence,
            "candles_count": len(df_data),
            "timeframe": "15M",
            "context_days": context_days,
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
            "chart_path": save_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vision_ai_optimized": True,
            "tradingview_style": True
        }
        
        metadata_path = save_path.replace('.png', '.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[VISION CHART] {symbol}: Metadata saved: {metadata_path}")
        return save_path
        
    except Exception as e:
        print(f"[VISION CHART ERROR] {symbol}: Failed to generate chart: {e}")
        plt.close('all')
        return None


def generate_enhanced_vision_charts(tjde_results: List[Dict], max_charts: int = 5) -> int:
    """
    Generate enhanced Vision-AI charts for top TJDE tokens
    
    Args:
        tjde_results: List of scan results with TJDE analysis
        max_charts: Maximum number of charts to generate
        
    Returns:
        Number of charts successfully generated
    """
    charts_generated = 0
    
    # Filter and sort results
    valid_results = [r for r in tjde_results if r.get('tjde_score', 0) > 0]
    if not valid_results:
        print("[VISION CHARTS] No valid TJDE results for chart generation")
        return 0
    
    # Sort by TJDE score descending
    top_results = sorted(valid_results, key=lambda x: x.get('tjde_score', 0), reverse=True)[:max_charts]
    
    print(f"[VISION CHARTS] Generating enhanced charts for TOP {len(top_results)} tokens")
    
    for i, result in enumerate(top_results, 1):
        try:
            symbol = result.get('symbol', 'UNKNOWN')
            tjde_score = result.get('tjde_score', 0)
            market_data = result.get('market_data', {})
            
            # Get candles from market_data
            candles = market_data.get('candles_15m', market_data.get('candles', []))
            if not candles:
                print(f"[VISION CHART SKIP] {symbol}: No candle data available")
                continue
            
            # Extract chart parameters
            setup = result.get('market_phase', 'unknown')
            decision = result.get('tjde_decision', 'unknown')
            clip_label = result.get('clip_label', 'unknown')
            clip_confidence = result.get('clip_confidence', 0.0)
            
            # Generate chart
            chart_path = plot_vision_chart(
                symbol=symbol,
                candles=candles,
                setup=setup,
                decision=decision,
                clip_label=clip_label,
                clip_confidence=clip_confidence,
                tjde_score=tjde_score
            )
            
            if chart_path:
                charts_generated += 1
                print(f"[VISION CHART] {i}/{len(top_results)} ✅ {symbol}: Chart generated successfully")
            else:
                print(f"[VISION CHART] {i}/{len(top_results)} ❌ {symbol}: Chart generation failed")
                
        except Exception as e:
            print(f"[VISION CHART ERROR] {symbol}: {e}")
    
    print(f"[VISION CHARTS] Successfully generated {charts_generated}/{len(top_results)} charts")
    return charts_generated


if __name__ == "__main__":
    # Test the new chart generator
    print("Testing new Vision-AI chart generator...")
    
    # Create sample candle data
    test_candles = []
    base_timestamp = int(datetime.now().timestamp()) - (50 * 900)  # 50 candles back
    base_price = 50000.0
    
    for i in range(50):
        timestamp = (base_timestamp + i * 900) * 1000  # 15 min intervals in ms
        
        # Simulate realistic price movement
        price_change = np.random.normal(0, 0.002)  # 0.2% std deviation
        base_price *= (1 + price_change)
        
        # Generate realistic OHLC
        open_price = base_price
        close_price = base_price * (1 + np.random.normal(0, 0.001))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0005)))
        volume = np.random.uniform(1000, 10000)
        
        test_candles.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    # Test chart generation
    result = plot_vision_chart(
        symbol="TESTUSDT",
        candles=test_candles,
        setup="breakout",
        decision="entry",
        clip_label="trend_continuation",
        clip_confidence=0.85,
        tjde_score=0.72
    )
    
    if result:
        print(f"✅ Test chart generated successfully: {result}")
    else:
        print("❌ Test chart generation failed")