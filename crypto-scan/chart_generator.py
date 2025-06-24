#!/usr/bin/env python3
"""
Chart Generator for Trend Mode Training
Generates professional charts with TJDE results for Vision-AI training
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def generate_trend_mode_chart(
    symbol: str, 
    candles_15m: List, 
    candles_5m: List, 
    tjde_score: float, 
    decision: str,
    tjde_breakdown: Dict = None,
    output_dir: str = "training_charts"
) -> Optional[str]:
    """
    Generate professional trend mode chart with TJDE results
    
    Args:
        symbol: Trading symbol
        candles_15m: 15-minute candle data
        candles_5m: 5-minute candle data  
        tjde_score: TJDE final score
        decision: TJDE decision (long/short/avoid/consider_entry)
        tjde_breakdown: Optional detailed score breakdown
        output_dir: Output directory for charts
        
    Returns:
        Path to generated chart or None if failed
    """
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Validate candle data
        if not candles_15m or len(candles_15m) < 10:
            print(f"[CHART ERROR] {symbol}: Insufficient 15M candles ({len(candles_15m) if candles_15m else 0})")
            return None
            
        # Use last 96 candles (24 hours at 15M intervals)
        candles_to_plot = candles_15m[-96:] if len(candles_15m) >= 96 else candles_15m
        
        # Extract data for plotting
        times = [datetime.fromtimestamp(c[0] / 1000) for c in candles_to_plot]
        opens = [float(c[1]) for c in candles_to_plot]
        highs = [float(c[2]) for c in candles_to_plot]
        lows = [float(c[3]) for c in candles_to_plot]
        closes = [float(c[4]) for c in candles_to_plot]
        volumes = [float(c[5]) for c in candles_to_plot]
        
        # Create figure with professional layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                      gridspec_kw={'height_ratios': [4, 1]},
                                      facecolor='white')
        
        # Price chart with candlesticks
        for i in range(len(times)):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            
            # Candlestick wicks
            ax1.plot([times[i], times[i]], [lows[i], highs[i]], 
                    color=color, linewidth=1.5, alpha=0.8)
            
            # Candlestick bodies
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            ax1.bar(times[i], body_height, bottom=body_bottom, 
                   color=color, width=0.0008, alpha=0.9)
        
        # Add trend line for context
        ax1.plot(times, closes, color='#2196f3', linewidth=1, alpha=0.6, label='Close Price')
        
        # Format decision color
        decision_colors = {
            'join_trend': '#4caf50',
            'consider_entry': '#ff9800', 
            'avoid': '#f44336',
            'long': '#4caf50',
            'short': '#f44336'
        }
        decision_color = decision_colors.get(decision.lower(), '#757575')
        
        # Chart title with TJDE results
        title = f"{symbol} | TJDE: {tjde_score:.3f} ({decision.upper()})"
        if tjde_breakdown:
            title += f" | Trend: {tjde_breakdown.get('trend_strength', 0):.2f}"
        
        ax1.set_title(title, fontsize=14, fontweight='bold', color=decision_color)
        ax1.set_ylabel("Price (USDT)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Format time axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Volume chart
        volume_colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' for i in range(len(times))]
        ax2.bar(times, volumes, color=volume_colors, alpha=0.7, width=0.0008)
        ax2.set_ylabel("Volume", fontsize=10)
        ax2.set_xlabel("Time (UTC)", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format volume axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add TJDE score annotation
        score_text = f"TJDE Score: {tjde_score:.3f}\nDecision: {decision.upper()}"
        if tjde_breakdown:
            score_text += f"\nTrend: {tjde_breakdown.get('trend_strength', 0):.2f}"
            score_text += f"\nPullback: {tjde_breakdown.get('pullback_quality', 0):.2f}"
        
        ax1.text(0.02, 0.98, score_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor=decision_color, alpha=0.1),
                verticalalignment='top', fontsize=10)
        
        # Save chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{timestamp}_chart.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"[CHART SUCCESS] {symbol}: Professional chart saved to {filepath}")
        
        # Save metadata JSON
        _save_chart_metadata(symbol, timestamp, tjde_score, decision, tjde_breakdown, output_dir)
        
        return filepath
        
    except Exception as e:
        print(f"[CHART ERROR] {symbol}: Failed to generate chart - {e}")
        return None


def _save_chart_metadata(
    symbol: str, 
    timestamp: str, 
    tjde_score: float, 
    decision: str,
    tjde_breakdown: Dict = None,
    output_dir: str = "training_charts"
) -> bool:
    """Save chart metadata as JSON for training purposes"""
    try:
        metadata = {
            "symbol": symbol,
            "timestamp": timestamp,
            "chart_file": f"{symbol}_{timestamp}_chart.png",
            "tjde_score": tjde_score,
            "decision": decision,
            "tjde_breakdown": tjde_breakdown or {},
            "created_at": datetime.now().isoformat(),
            "chart_type": "trend_mode_training",
            "candle_timeframe": "15M",
            "quality_check": {
                "min_candles": True,
                "valid_ohlcv": True,
                "tjde_calculated": tjde_score is not None
            }
        }
        
        metadata_file = os.path.join(output_dir, f"{symbol}_{timestamp}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"[METADATA] {symbol}: Saved metadata to {metadata_file}")
        return True
        
    except Exception as e:
        print(f"[METADATA ERROR] {symbol}: {e}")
        return False


def validate_chart_quality(candles_15m: List, candles_5m: List = None) -> Dict[str, bool]:
    """
    Quality checklist for chart generation
    
    Returns:
        Dictionary with quality validation results
    """
    quality = {
        "sufficient_candles_15m": len(candles_15m) >= 20 if candles_15m else False,
        "sufficient_candles_5m": len(candles_5m) >= 60 if candles_5m else True,  # Optional
        "valid_ohlcv_data": False,
        "no_nan_values": False,
        "price_variance": False
    }
    
    if candles_15m and len(candles_15m) >= 10:
        try:
            # Check OHLCV data validity
            sample_candle = candles_15m[-1]
            if len(sample_candle) >= 6:
                quality["valid_ohlcv_data"] = True
                
            # Check for NaN values
            closes = [float(c[4]) for c in candles_15m[-10:]]
            quality["no_nan_values"] = all(not (c != c) for c in closes)  # NaN check
            
            # Check price variance (not flat line)
            if closes:
                price_std = (max(closes) - min(closes)) / max(closes) if max(closes) > 0 else 0
                quality["price_variance"] = price_std > 0.001  # At least 0.1% price movement
                
        except Exception:
            pass
    
    return quality


def generate_chart_async_safe(
    symbol: str,
    market_data: Dict,
    tjde_result: Dict,
    tjde_breakdown: Dict = None
) -> Optional[str]:
    """
    Async-safe chart generation for integration with scan_token_async
    
    Args:
        symbol: Trading symbol
        market_data: Market data dictionary with candles
        tjde_result: TJDE result with score and decision
        tjde_breakdown: Optional detailed TJDE breakdown
        
    Returns:
        Path to generated chart or None
    """
    try:
        candles_15m = market_data.get("candles", [])
        candles_5m = market_data.get("candles_5m", [])
        
        # Quality validation
        quality = validate_chart_quality(candles_15m, candles_5m)
        if not quality["sufficient_candles_15m"]:
            print(f"[CHART SKIP] {symbol}: Insufficient candles for quality chart")
            return None
            
        # Extract TJDE results
        tjde_score = tjde_result.get("final_score", 0.0)
        decision = tjde_result.get("decision", "unknown")
        
        # Generate chart
        chart_path = generate_trend_mode_chart(
            symbol=symbol,
            candles_15m=candles_15m,
            candles_5m=candles_5m,
            tjde_score=tjde_score,
            decision=decision,
            tjde_breakdown=tjde_breakdown,
            output_dir="training_charts"
        )
        
        return chart_path
        
    except Exception as e:
        print(f"[CHART ASYNC ERROR] {symbol}: {e}")
        return None


def test_chart_generation():
    """Test chart generation with sample data"""
    print("Testing chart generation...")
    
    # Generate sample candle data
    import time
    base_time = int(time.time() * 1000)
    base_price = 100.0
    
    sample_candles = []
    for i in range(96):
        timestamp = base_time - (96 - i) * 15 * 60 * 1000  # 15 minutes intervals
        price_change = (i % 10 - 5) * 0.5  # Small price movements
        
        open_price = base_price + price_change
        close_price = open_price + (i % 3 - 1) * 0.2
        high_price = max(open_price, close_price) + 0.1
        low_price = min(open_price, close_price) - 0.1
        volume = 1000 + (i % 20) * 50
        
        sample_candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
    
    # Test chart generation
    result = generate_trend_mode_chart(
        symbol="TESTUSDT",
        candles_15m=sample_candles,
        candles_5m=[],
        tjde_score=0.75,
        decision="consider_entry",
        tjde_breakdown={
            "trend_strength": 0.8,
            "pullback_quality": 0.7,
            "support_reaction": 0.6
        }
    )
    
    if result:
        print(f"✅ Test chart generated: {result}")
        # Clean up test file
        try:
            os.remove(result)
            metadata_file = result.replace("_chart.png", "_metadata.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            print("✅ Test files cleaned up")
        except:
            pass
    else:
        print("❌ Test chart generation failed")


if __name__ == "__main__":
    test_chart_generation()