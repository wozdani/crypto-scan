"""
RLAgentV4 Weight Evolution Visualization
Wizualizacja ewolucji wag neural network dla detektorów fusion
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_weight_history(log_path="logs/rl_weight_history.jsonl"):
    """
    Load weight evolution history from JSONL log file
    
    Returns:
        tuple: (weights, timestamps) where weights is list of [californium, diamond, whaleclip]
    """
    weights = []
    timestamps = []
    
    if not os.path.exists(log_path):
        print(f"📂 No weight history found at {log_path}")
        return [], []

    try:
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    entry = json.loads(line)
                    weights.append([
                        entry.get("californium", 0.0),
                        entry.get("diamond", 0.0), 
                        entry.get("whaleclip", 0.0)
                    ])
                    timestamp = entry.get("timestamp")
                    if timestamp:
                        timestamps.append(datetime.fromisoformat(timestamp.replace('Z', '+00:00')))
                    
        print(f"📊 Loaded {len(weights)} weight history entries")
        return weights, timestamps
        
    except Exception as e:
        print(f"❌ Error loading weight history: {e}")
        return [], []

def plot_weights_evolution(save_path="crypto-scan/visualization/rl_weights_evolution.png"):
    """
    Generate and save weight evolution chart
    
    Args:
        save_path: Path to save the PNG chart
    """
    weights, timestamps = load_weight_history()
    
    if not weights:
        print("⚠️ No weight data available for visualization")
        return False

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Transpose weights for plotting
    weights = list(zip(*weights))
    californium_weights = weights[0]
    diamond_weights = weights[1]
    whaleclip_weights = weights[2]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # Plot weight evolution lines
    plt.plot(timestamps, californium_weights, 
             label="CaliforniumWhale AI", 
             color='#FF6B35', linewidth=2.5, marker='o', markersize=4)
    plt.plot(timestamps, diamond_weights, 
             label="DiamondWhale AI", 
             color='#4ECDC4', linewidth=2.5, marker='s', markersize=4)
    plt.plot(timestamps, whaleclip_weights, 
             label="WhaleCLIP", 
             color='#45B7D1', linewidth=2.5, marker='^', markersize=4)
    
    # Formatting
    plt.title("📈 RLAgentV4 Neural Network - Ewolucja Wag Detektorów", 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Data i Czas", fontsize=12)
    plt.ylabel("Waga Detektora", fontsize=12)
    
    # Format x-axis
    if len(timestamps) > 1:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add statistics text box
    if weights:
        current_weights = [californium_weights[-1], diamond_weights[-1], whaleclip_weights[-1]]
        stats_text = f"Aktualne wagi:\nCalifornium: {current_weights[0]:.3f}\nDiamond: {current_weights[1]:.3f}\nWhaleCLIP: {current_weights[2]:.3f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ RLAgentV4 weight evolution chart saved: {save_path}")
    return True

def plot_weight_distribution(save_path="crypto-scan/visualization/rl_weights_distribution.png"):
    """
    Generate weight distribution pie chart for latest weights
    
    Args:
        save_path: Path to save the PNG chart
    """
    weights, timestamps = load_weight_history()
    
    if not weights:
        print("⚠️ No weight data available for distribution chart")
        return False

    # Get latest weights
    latest_weights = weights[-1]
    labels = ['CaliforniumWhale AI', 'DiamondWhale AI', 'WhaleCLIP']
    colors = ['#FF6B35', '#4ECDC4', '#45B7D1']
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(latest_weights, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90, 
                                      explode=(0.05, 0.05, 0.05))
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    
    plt.title("🎯 RLAgentV4 - Aktualny Rozkład Wag Detektorów", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add timestamp
    if timestamps:
        plt.figtext(0.5, 0.02, f"Ostatnia aktualizacja: {timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')}", 
                   ha='center', fontsize=10, style='italic')
    
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ RLAgentV4 weight distribution chart saved: {save_path}")
    return True

def generate_training_summary():
    """
    Generate training summary statistics
    
    Returns:
        dict: Training summary statistics
    """
    weights, timestamps = load_weight_history()
    
    if not weights:
        return {}
    
    weights_array = np.array(weights)
    
    summary = {
        "total_updates": len(weights),
        "training_period_days": (timestamps[-1] - timestamps[0]).days if len(timestamps) > 1 else 0,
        "latest_weights": {
            "californium": weights[-1][0],
            "diamond": weights[-1][1], 
            "whaleclip": weights[-1][2]
        },
        "weight_evolution": {
            "californium_change": weights[-1][0] - weights[0][0] if len(weights) > 1 else 0,
            "diamond_change": weights[-1][1] - weights[0][1] if len(weights) > 1 else 0,
            "whaleclip_change": weights[-1][2] - weights[0][2] if len(weights) > 1 else 0
        },
        "dominant_detector": ["CaliforniumWhale", "DiamondWhale", "WhaleCLIP"][np.argmax(weights[-1])],
        "training_stability": np.std(weights_array, axis=0).tolist()
    }
    
    return summary

def main():
    """
    Main function to generate all RLAgentV4 visualization charts
    """
    print("🎨 Generating RLAgentV4 Weight Evolution Visualizations...")
    
    # Generate evolution chart
    evolution_success = plot_weights_evolution()
    
    # Generate distribution chart
    distribution_success = plot_weight_distribution()
    
    # Generate summary
    summary = generate_training_summary()
    
    if summary:
        print(f"\n📊 RLAgentV4 Training Summary:")
        print(f"   🔄 Total updates: {summary['total_updates']}")
        print(f"   📅 Training period: {summary['training_period_days']} days")
        print(f"   🎯 Dominant detector: {summary['dominant_detector']}")
        print(f"   ⚖️ Latest weights: C:{summary['latest_weights']['californium']:.3f}, "
              f"D:{summary['latest_weights']['diamond']:.3f}, W:{summary['latest_weights']['whaleclip']:.3f}")
    
    print(f"\n🎉 Visualization generation {'completed successfully!' if (evolution_success or distribution_success) else 'failed - no data available'}")

if __name__ == "__main__":
    main()