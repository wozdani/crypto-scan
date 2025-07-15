#!/usr/bin/env python3
"""
Visual Weights Evolution - RLAgentV3 Booster Weight Visualization
Generates comprehensive charts showing the evolution of RLAgentV3 booster weights over time
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_weight_evolution(
    log_path: str = "cache/training_log.jsonl", 
    output_path: str = "plots/booster_weight_evolution.png",
    show_effectiveness: bool = True,
    title_suffix: str = ""
) -> bool:
    """
    Visualize the evolution of RLAgentV3 booster weights over time
    
    Args:
        log_path: Path to training log JSONL file
        output_path: Path to save the generated chart
        show_effectiveness: Whether to include effectiveness metrics
        title_suffix: Additional text for chart title
        
    Returns:
        True if chart was generated successfully, False otherwise
    """
    logger.info(f"[WYKRES WAG] Starting weight evolution visualization")
    
    try:
        # Check if log file exists
        if not os.path.exists(log_path):
            logger.warning(f"[WYKRES WAG] Training log not found: {log_path}")
            return False
        
        timestamps = []
        weight_series = {}
        effectiveness_series = {}
        update_counts = []
        
        # Read training log
        with open(log_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                    
                    # Extract weights
                    weights = entry.get("weights", {})
                    for booster, weight in weights.items():
                        if booster not in weight_series:
                            weight_series[booster] = []
                        weight_series[booster].append(float(weight))
                    
                    # Extract effectiveness data if available
                    if show_effectiveness and "booster_effectiveness" in entry:
                        effectiveness = entry["booster_effectiveness"]
                        for booster, eff_data in effectiveness.items():
                            if booster not in effectiveness_series:
                                effectiveness_series[booster] = []
                            # Extract effectiveness percentage or use weight as fallback
                            eff_value = eff_data.get("effectiveness", weight_series.get(booster, [0])[-1])
                            effectiveness_series[booster].append(float(eff_value))
                    
                    # Track update counts
                    update_counts.append(entry.get("total_updates", 0))
                    
                    # Ensure all series have same length (fill missing values)
                    for booster in weight_series:
                        if len(weight_series[booster]) < len(timestamps):
                            # Fill with last known value or 1.0 as default
                            last_value = weight_series[booster][-1] if weight_series[booster] else 1.0
                            weight_series[booster].append(last_value)
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"[WYKRES WAG] JSON error in line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"[WYKRES WAG] Error processing line {line_num}: {e}")
        
        if not timestamps or not weight_series:
            logger.warning("[WYKRES WAG] No valid data found for visualization")
            return False
        
        # Create the chart
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Define colors for consistent booster representation
        colors = {
            'gnn': '#2E8B57',      # Sea Green
            'whaleClip': '#4169E1', # Royal Blue  
            'dexInflow': '#DC143C', # Crimson
            'volumeSpike': '#FF8C00', # Dark Orange
            'default': '#708090'    # Slate Gray
        }
        
        # Top subplot: Weight Evolution
        ax1.set_title(f"Ewolucja wag boosterów RLAgentV3{title_suffix}", fontsize=14, fontweight='bold')
        
        for booster, values in weight_series.items():
            color = colors.get(booster, colors['default'])
            ax1.plot(timestamps, values, 
                    label=f"{booster.upper()}", 
                    linewidth=2.5, 
                    color=color, 
                    marker='o', 
                    markersize=4,
                    alpha=0.8)
        
        ax1.set_xlabel("Data")
        ax1.set_ylabel("Waga Boostera", fontweight='bold')
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        
        # Add horizontal reference line at weight = 1.0
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0)')
        
        # Bottom subplot: Training Progress
        ax2.set_title("Postęp treningu - liczba aktualizacji", fontsize=12, fontweight='bold')
        ax2.plot(timestamps, update_counts, 
                color='#9932CC', 
                linewidth=2, 
                marker='s', 
                markersize=3,
                label='Aktualizacje na sesję')
        
        ax2.set_xlabel("Data", fontweight='bold')
        ax2.set_ylabel("Liczba aktualizacji", fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        
        # Add statistics text box
        if timestamps and weight_series:
            latest_weights = {booster: values[-1] for booster, values in weight_series.items()}
            total_sessions = len(timestamps)
            total_updates = sum(update_counts)
            
            stats_text = f"Sesje treningowe: {total_sessions}\n"
            stats_text += f"Łączne aktualizacje: {total_updates}\n"
            stats_text += f"Ostatnie wagi:\n"
            for booster, weight in latest_weights.items():
                stats_text += f"  • {booster}: {weight:.3f}\n"
            
            ax1.text(0.02, 0.98, stats_text, 
                    transform=ax1.transAxes, 
                    fontsize=9, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the chart
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"[WYKRES WAG] Chart saved successfully: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"[WYKRES WAG] Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_effectiveness_comparison(
    log_path: str = "cache/training_log.jsonl",
    output_path: str = "plots/booster_effectiveness_comparison.png"
) -> bool:
    """
    Generate a comparison chart of booster effectiveness over time
    
    Args:
        log_path: Path to training log JSONL file
        output_path: Path to save the effectiveness chart
        
    Returns:
        True if chart was generated successfully, False otherwise
    """
    logger.info("[EFFECTIVENESS CHART] Generating booster effectiveness comparison")
    
    try:
        if not os.path.exists(log_path):
            logger.warning(f"[EFFECTIVENESS CHART] Log file not found: {log_path}")
            return False
        
        effectiveness_data = {}
        timestamps = []
        
        with open(log_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    if "booster_effectiveness" not in entry:
                        continue
                    
                    timestamp = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                    
                    for booster, eff_data in entry["booster_effectiveness"].items():
                        if booster not in effectiveness_data:
                            effectiveness_data[booster] = []
                        
                        effectiveness = eff_data.get("effectiveness", 50.0)  # Default 50%
                        effectiveness_data[booster].append(float(effectiveness))
                        
                except Exception as e:
                    logger.warning(f"[EFFECTIVENESS CHART] Error parsing entry: {e}")
        
        if not effectiveness_data or not timestamps:
            logger.warning("[EFFECTIVENESS CHART] No effectiveness data found")
            return False
        
        # Create effectiveness chart
        plt.figure(figsize=(12, 8))
        
        colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC']
        
        for i, (booster, values) in enumerate(effectiveness_data.items()):
            color = colors[i % len(colors)]
            plt.plot(timestamps, values, 
                    label=f"{booster.upper()}", 
                    linewidth=2.5, 
                    color=color, 
                    marker='o', 
                    markersize=4)
        
        plt.title("Porównanie skuteczności boosterów RLAgentV3", fontsize=14, fontweight='bold')
        plt.xlabel("Data", fontweight='bold')
        plt.ylabel("Skuteczność (%)", fontweight='bold')
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Format dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        
        # Add reference line at 50%
        plt.axhline(y=50.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (50%)')
        
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"[EFFECTIVENESS CHART] Chart saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"[EFFECTIVENESS CHART] Error: {e}")
        return False

def generate_all_charts(
    log_path: str = "cache/training_log.jsonl",
    output_dir: str = "plots"
) -> Dict[str, bool]:
    """
    Generate all available charts for RLAgentV3 analysis
    
    Args:
        log_path: Path to training log file
        output_dir: Directory to save all charts
        
    Returns:
        Dictionary with chart generation results
    """
    logger.info("[ALL CHARTS] Generating complete chart suite")
    
    results = {}
    
    # Generate weight evolution chart
    weight_chart_path = os.path.join(output_dir, "booster_weight_evolution.png")
    results["weight_evolution"] = visualize_weight_evolution(log_path, weight_chart_path)
    
    # Generate effectiveness comparison chart
    effectiveness_chart_path = os.path.join(output_dir, "booster_effectiveness_comparison.png")
    results["effectiveness_comparison"] = generate_effectiveness_comparison(log_path, effectiveness_chart_path)
    
    # Add timestamp to results
    results["generation_timestamp"] = datetime.utcnow().isoformat()
    results["total_generated"] = sum(1 for v in results.values() if isinstance(v, bool) and v)
    
    logger.info(f"[ALL CHARTS] Generated {results['total_generated']} charts successfully")
    return results

def main():
    """Main function for manual testing"""
    print("📊 RLAgentV3 Weight Evolution Visualizer")
    print("=" * 50)
    
    # Generate all charts
    results = generate_all_charts()
    
    print(f"Chart generation results:")
    for chart_type, success in results.items():
        if isinstance(success, bool):
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  • {chart_type}: {status}")
    
    return results.get("total_generated", 0) > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)