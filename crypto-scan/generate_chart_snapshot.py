#!/usr/bin/env python3
"""
Chart Snapshot Generator for CV Training
Generates charts in standardized format for machine learning pipeline
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from utils.chart_generator import create_pattern_chart
# Mock data generator removed - using authentic market data only
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


# Chart training directory structure
CHART_TRAINING_DIR = Path("data/chart_training")
CHARTS_DIR = CHART_TRAINING_DIR / "charts"
LABELS_DIR = CHART_TRAINING_DIR / "labels"
TRAINING_DATA_DIR = CHART_TRAINING_DIR / "training_data"

# Create directories
for directory in [CHARTS_DIR, LABELS_DIR, TRAINING_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def generate_chart_snapshot(
    symbol: str,
    candles: list = None,
    market_features: Dict = None,
    timestamp: str = None
) -> Optional[str]:
    """
    Generate standardized chart snapshot for CV training
    
    Args:
        symbol: Trading symbol
        candles: OHLCV candle data (will generate if None)
        market_features: Market analysis features
        timestamp: Custom timestamp (will generate if None)
        
    Returns:
        Path to generated chart file
    """
    try:
        # Generate timestamp if not provided
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Generate filename in required format: SYMBOL_TIMESTAMP.png
        filename = f"{symbol}_{timestamp}.png"
        chart_path = CHARTS_DIR / filename
        
        # Real data only - no mock candles generated
        if not candles:
            print(f"[SNAPSHOT] ❌ No candle data provided for {symbol} - authentic data required")
            return None
        
        if not candles or len(candles) < 20:
            print(f"[SNAPSHOT] ❌ Insufficient candle data for {symbol}")
            return None
        
        # Create standardized chart
        success = _create_standardized_chart(candles, symbol, str(chart_path))
        
        if success:
            print(f"[SNAPSHOT] ✅ Generated: {filename}")
            
            # Save market features if provided
            if market_features:
                _save_market_features(symbol, timestamp, market_features)
            
            return str(chart_path)
        else:
            return None
            
    except Exception as e:
        print(f"[SNAPSHOT] ❌ Error generating chart for {symbol}: {e}")
        return None


def _create_standardized_chart(candles: list, symbol: str, save_path: str) -> bool:
    """Create standardized chart format for consistent CV training"""
    try:
        # Process candle data
        times = [datetime.fromtimestamp(c[0] / 1000) for c in candles]
        opens = [float(c[1]) for c in candles]
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        closes = [float(c[4]) for c in candles]
        volumes = [float(c[5]) for c in candles]
        
        # Create figure with fixed size for consistent training
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[4, 1], facecolor='white')
        
        # Price chart with candlesticks
        for i in range(len(times)):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            
            # Wicks
            ax1.plot([times[i], times[i]], [lows[i], highs[i]], 
                    color=color, linewidth=1, alpha=0.8)
            
            # Bodies
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            ax1.bar(times[i], body_height, bottom=body_bottom, 
                   color=color, width=0.6, alpha=0.9)
        
        # Add EMA20 for pattern context
        if len(closes) >= 20:
            ema20 = _calculate_ema(closes, 20)
            if ema20:
                ax1.plot(times[19:], ema20, label="EMA20", color='#2196f3', linewidth=2)
                ax1.legend(loc='upper left')
        
        # Chart styling
        ax1.set_title(f"{symbol} - Chart Pattern Analysis", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylabel('Price', fontsize=12)
        
        # Remove spines for cleaner look
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Volume chart
        volume_colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' 
                        for i in range(len(closes))]
        ax2.bar(times, volumes, color=volume_colors, alpha=0.7, width=0.6)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Save with consistent settings
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"[SNAPSHOT] ❌ Chart creation failed: {e}")
        plt.close('all')
        return False


def _calculate_ema(prices: list, period: int) -> Optional[list]:
    """Calculate Exponential Moving Average"""
    try:
        if len(prices) < period:
            return None
        
        ema = []
        multiplier = 2 / (period + 1)
        ema.append(sum(prices[:period]) / period)
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_value)
        
        return ema
    except Exception:
        return None


def _save_market_features(symbol: str, timestamp: str, features: Dict):
    """Save market features for reference"""
    try:
        features_file = LABELS_DIR / f"{symbol}_{timestamp}_features.json"
        
        import json
        with open(features_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        print(f"[SNAPSHOT] 💾 Saved features: {features_file.name}")
        
    except Exception as e:
        print(f"[SNAPSHOT] ⚠️ Failed to save features: {e}")


def generate_training_batch(
    symbols: list = None,
    patterns: list = None,
    count_per_pattern: int = 5
) -> Dict:
    """
    Generate batch of training charts with different patterns
    
    Args:
        symbols: List of symbols to use
        patterns: List of patterns to generate
        count_per_pattern: Number of charts per pattern
        
    Returns:
        Dictionary with generation results
    """
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
    
    if not patterns:
        patterns = ["breakout_continuation", "pullback_setup", "trend_exhaustion", 
                   "consolidation", "fakeout_pattern"]
    
    results = {
        "generated_charts": [],
        "failed_charts": [],
        "total_generated": 0
    }
    
    print(f"[BATCH] Generating {len(patterns)} patterns x {count_per_pattern} charts...")
    
    for pattern in patterns:
        for i in range(count_per_pattern):
            symbol = symbols[i % len(symbols)]
            
            try:
                # Real data only - no pattern-specific generation
                print(f"[BATCH] ❌ {symbol}_{pattern}_{i}: Mock data generation disabled - authentic data required")
                candles = None
                
                # Create market features based on pattern
                features = _create_pattern_features(pattern, i)
                
                # Generate chart
                timestamp = datetime.now().strftime(f"%Y%m%d_%H%M{i:02d}")
                chart_path = generate_chart_snapshot(
                    symbol=symbol,
                    candles=candles,
                    market_features=features,
                    timestamp=timestamp
                )
                
                if chart_path:
                    results["generated_charts"].append({
                        "path": chart_path,
                        "symbol": symbol,
                        "pattern": pattern,
                        "features": features
                    })
                    results["total_generated"] += 1
                else:
                    results["failed_charts"].append(f"{symbol}_{pattern}_{i}")
                    
            except Exception as e:
                print(f"[BATCH] ❌ Failed {symbol}_{pattern}_{i}: {e}")
                results["failed_charts"].append(f"{symbol}_{pattern}_{i}")
    
    print(f"[BATCH] ✅ Generated {results['total_generated']} charts")
    print(f"[BATCH] ❌ Failed {len(results['failed_charts'])} charts")
    
    return results


def _create_pattern_features(pattern: str, index: int) -> Dict:
    """Create realistic market features for pattern"""
    base_features = {
        "trend_strength": 0.6,
        "pullback_quality": 0.5,
        "liquidity_score": 0.7,
        "htf_trend_match": True,
        "phase_score": 0.6
    }
    
    # Pattern-specific adjustments
    if pattern == "breakout_continuation":
        base_features.update({
            "trend_strength": 0.8 + (index * 0.02),
            "phase_score": 0.85,
            "liquidity_score": 0.9
        })
    elif pattern == "pullback_setup":
        base_features.update({
            "pullback_quality": 0.9 + (index * 0.01),
            "trend_strength": 0.7,
            "phase_score": 0.75
        })
    elif pattern == "trend_exhaustion":
        base_features.update({
            "trend_strength": 0.3 - (index * 0.02),
            "phase_score": 0.4,
            "htf_trend_match": False
        })
    
    return base_features


def main():
    """Main function for chart snapshot generation"""
    print("📸 Chart Snapshot Generator")
    print("=" * 40)
    
    # Generate training batch
    results = generate_training_batch(count_per_pattern=3)
    
    print(f"\n📊 Generation Summary:")
    print(f"  Charts Directory: {CHARTS_DIR}")
    print(f"  Labels Directory: {LABELS_DIR}")
    print(f"  Generated: {results['total_generated']} charts")
    print(f"  Failed: {len(results['failed_charts'])}")
    
    # List generated files
    chart_files = list(CHARTS_DIR.glob("*.png"))
    print(f"\n📁 Available Charts: {len(chart_files)}")
    for chart_file in chart_files[-5:]:  # Show last 5
        print(f"  {chart_file.name}")
    
    print(f"\n✅ Chart snapshots ready for GPT labeling")


if __name__ == "__main__":
    main()