"""
Create Demo Training Data for CLIP Model Testing
Generates sample chart images and labels for testing the CLIP system
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

def create_demo_chart(symbol: str, pattern_type: str, timestamp: str) -> str:
    """
    Create a demo chart with specific pattern
    
    Args:
        symbol: Trading symbol
        pattern_type: Type of pattern to create
        timestamp: Timestamp for filename
        
    Returns:
        Path to created chart
    """
    # Generate synthetic price data based on pattern
    days = 30
    x = np.arange(days)
    
    # Base price trend
    base_price = 50000 if 'BTC' in symbol else 2000 if 'ETH' in symbol else 100
    
    if pattern_type == "trending-up":
        trend = np.linspace(0, 0.3, days)
        noise = np.random.normal(0, 0.02, days)
        prices = base_price * (1 + trend + noise)
        volume_pattern = "increasing"
        
    elif pattern_type == "pullback-in-trend":
        trend = np.linspace(0, 0.25, days)
        # Add pullback in the middle
        pullback = np.zeros(days)
        pullback[15:22] = -0.08
        noise = np.random.normal(0, 0.015, days)
        prices = base_price * (1 + trend + pullback + noise)
        volume_pattern = "spike_on_pullback"
        
    elif pattern_type == "breakout-continuation":
        # Consolidation then breakout
        consolidation = np.zeros(20)
        breakout = np.linspace(0, 0.15, 10)
        trend = np.concatenate([consolidation, breakout])
        noise = np.random.normal(0, 0.01, days)
        prices = base_price * (1 + trend + noise)
        volume_pattern = "breakout_volume"
        
    elif pattern_type == "fakeout":
        # False breakout then reversal
        fake_break = np.zeros(15)
        fake_break = np.concatenate([fake_break, np.linspace(0, 0.08, 5)])
        reversal = np.linspace(0.08, -0.05, 10)
        trend = np.concatenate([fake_break, reversal])
        noise = np.random.normal(0, 0.02, days)
        prices = base_price * (1 + trend + noise)
        volume_pattern = "decreasing"
        
    elif pattern_type == "accumulation":
        # Sideways with slight upward bias
        trend = np.random.normal(0, 0.01, days)
        trend = np.cumsum(trend * 0.5)
        noise = np.random.normal(0, 0.015, days)
        prices = base_price * (1 + trend + noise)
        volume_pattern = "low_consistent"
        
    else:  # consolidation
        trend = np.random.normal(0, 0.01, days)
        noise = np.random.normal(0, 0.02, days)
        prices = base_price * (1 + trend + noise)
        volume_pattern = "low_consistent"
    
    # Generate volume data
    base_volume = np.random.uniform(1000, 5000, days)
    
    if volume_pattern == "increasing":
        volume_trend = np.linspace(1, 2.5, days)
        volume = base_volume * volume_trend
    elif volume_pattern == "spike_on_pullback":
        volume = base_volume.copy()
        volume[15:22] *= 3  # Volume spike during pullback
    elif volume_pattern == "breakout_volume":
        volume = base_volume.copy()
        volume[20:] *= 2.5  # Volume increase on breakout
    else:
        volume = base_volume
    
    # Create chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Price chart
    ax1.plot(x, prices, linewidth=2, color='#2E8B57' if prices[-1] > prices[0] else '#DC143C')
    ax1.fill_between(x, prices, alpha=0.1, color='#2E8B57' if prices[-1] > prices[0] else '#DC143C')
    
    # Add support/resistance levels for some patterns
    if pattern_type in ["pullback-in-trend", "breakout-continuation"]:
        support_level = np.min(prices) * 1.02
        ax1.axhline(y=support_level, color='green', linestyle='--', alpha=0.7, label='Support')
    
    if pattern_type == "fakeout":
        resistance_level = np.max(prices) * 0.98
        ax1.axhline(y=resistance_level, color='red', linestyle='--', alpha=0.7, label='Resistance')
    
    ax1.set_title(f"{symbol} - {pattern_type.replace('-', ' ').title()}", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend() if pattern_type in ["pullback-in-trend", "breakout-continuation", "fakeout"] else None
    
    # Volume chart
    colors = ['green' if prices[i] >= prices[i-1] else 'red' for i in range(1, len(prices))]
    colors.insert(0, 'green')
    ax2.bar(x, volume, color=colors, alpha=0.6)
    ax2.set_ylabel('Volume', fontweight='bold')
    ax2.set_xlabel('Time Period', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    charts_dir = Path("data/training/charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{symbol}_{timestamp}.png"
    filepath = charts_dir / filename
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)

def create_demo_label(symbol: str, pattern_type: str, timestamp: str) -> str:
    """
    Create demo label file for chart
    
    Args:
        symbol: Trading symbol
        pattern_type: Type of pattern
        timestamp: Timestamp for filename
        
    Returns:
        Path to created label file
    """
    # Define label patterns
    label_templates = {
        "trending-up": [
            "trending-up | uptrend-continuation | volume-backed",
            "trending-up | momentum-strong | higher-highs",
            "trending-up | bullish-trend | volume-confirmation"
        ],
        "pullback-in-trend": [
            "pullback-in-trend | trending-up | support-bounce",
            "pullback-in-trend | retracement | volume-spike",
            "pullback-in-trend | correction | trend-continuation"
        ],
        "breakout-continuation": [
            "breakout-continuation | trending-up | volume-breakout",
            "breakout-continuation | resistance-break | momentum-shift",
            "breakout-continuation | consolidation-exit | volume-backed"
        ],
        "fakeout": [
            "fakeout | failed-breakout | reversal-pattern",
            "fakeout | false-signal | volume-decline",
            "fakeout | trap-pattern | momentum-failure"
        ],
        "accumulation": [
            "accumulation | base-building | consolidation",
            "accumulation | sideways | preparation-phase",
            "accumulation | range-bound | distribution-phase"
        ],
        "consolidation": [
            "consolidation | range-bound | sideways-movement",
            "consolidation | neutral | equilibrium",
            "consolidation | indecision | low-volume"
        ]
    }
    
    # Select random label from template
    labels = label_templates.get(pattern_type, ["unknown | neutral | standard"])
    selected_label = random.choice(labels)
    
    # Save label file
    labels_dir = Path("data/training/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{symbol}_{timestamp}.txt"
    filepath = labels_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(selected_label)
    
    return str(filepath)

def create_demo_training_dataset(num_samples: int = 20) -> dict:
    """
    Create complete demo training dataset
    
    Args:
        num_samples: Number of training samples to create
        
    Returns:
        Summary of created dataset
    """
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT", "MATICUSDT"]
    patterns = ["trending-up", "pullback-in-trend", "breakout-continuation", "fakeout", "accumulation", "consolidation"]
    
    created_charts = []
    created_labels = []
    
    print(f"Creating {num_samples} demo training samples...")
    
    for i in range(num_samples):
        # Random selection
        symbol = random.choice(symbols)
        pattern = random.choice(patterns)
        
        # Generate timestamp (last 30 days)
        base_date = datetime.now() - timedelta(days=random.randint(1, 30))
        timestamp = base_date.strftime("%Y%m%d_%H%M")
        
        # Create chart and label
        try:
            chart_path = create_demo_chart(symbol, pattern, timestamp)
            label_path = create_demo_label(symbol, pattern, timestamp)
            
            created_charts.append(chart_path)
            created_labels.append(label_path)
            
            print(f"  {i+1:2d}. {symbol} - {pattern} -> {Path(chart_path).name}")
            
        except Exception as e:
            print(f"  Error creating sample {i+1}: {e}")
            continue
    
    # Create dataset summary
    summary = {
        "created_timestamp": datetime.now().isoformat(),
        "total_samples": len(created_charts),
        "charts_created": len(created_charts),
        "labels_created": len(created_labels),
        "patterns_used": patterns,
        "symbols_used": symbols,
        "charts_directory": "data/training/charts",
        "labels_directory": "data/training/labels"
    }
    
    # Save summary
    summary_file = Path("data/training/dataset_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Create demo training dataset"""
    print("🎯 Creating Demo Training Dataset for CLIP")
    print("=" * 50)
    
    try:
        # Create demo dataset
        summary = create_demo_training_dataset(24)
        
        print(f"\n✅ Demo dataset created successfully!")
        print(f"   Total samples: {summary['total_samples']}")
        print(f"   Charts: {summary['charts_created']}")
        print(f"   Labels: {summary['labels_created']}")
        print(f"   Patterns: {', '.join(summary['patterns_used'])}")
        
        # Verify files
        charts_dir = Path("data/training/charts")
        labels_dir = Path("data/training/labels")
        
        chart_count = len(list(charts_dir.glob("*.png")))
        label_count = len(list(labels_dir.glob("*.txt")))
        
        print(f"\n📊 Verification:")
        print(f"   Chart files: {chart_count}")
        print(f"   Label files: {label_count}")
        
        if chart_count > 0 and label_count > 0:
            print(f"\n🚀 Ready for CLIP training!")
            print(f"   Run: python vision_ai/train_clip_model.py")
        else:
            print(f"\n❌ No files created - check for errors")
    
    except Exception as e:
        print(f"❌ Error creating demo dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()