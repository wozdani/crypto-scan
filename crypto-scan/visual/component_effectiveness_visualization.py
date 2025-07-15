#!/usr/bin/env python3
"""
🎯 COMPONENT EFFECTIVENESS VISUALIZATION V4 - Heatmapa skuteczności komponentów
===============================================================================

Component Effectiveness Visualization - Wizualizacja skuteczności komponentów
(per detektor) zgodnie z user specification dla Component-Aware Feedback Loop V4.

Funkcjonalności:
✅ Heatmapa skuteczności komponentów per detektor
✅ Trend wag: jak zmieniała się wartość clip_weight, diamond_weight w czasie
✅ Component effectiveness percentages visualization
✅ Dynamic weight evolution charts
✅ Multi-detector comparison analysis
✅ Historical trend analysis z JSONL data

Obsługiwane detektory:
- ClassicStealth (dex, whale, trust, id)
- DiamondWhale AI (diamond, whale, trust)
- CaliforniumWhale AI (californium, trust, whale)
- WhaleCLIP (clip, whale, trust)
- GraphGNN (gnn, whale, trust)
- MultiAgentConsensus (consensus, all components)
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentEffectivenessVisualizer:
    """
    Visualizer dla component effectiveness i weight evolution analysis
    """
    
    def __init__(self, feedback_dir: str = "feedback_loop", output_dir: str = "visual_output"):
        self.feedback_dir = Path(feedback_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure directories exist
        self.feedback_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.component_memory_file = self.feedback_dir / "component_score_memory.jsonl"
        self.component_weights_file = self.feedback_dir / "component_dynamic_weights.json"
        
        # Styling configuration
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"[COMPONENT VIZ] Initialized with feedback_dir={feedback_dir}, output_dir={output_dir}")
    
    def create_component_effectiveness_heatmap(self, days_back: int = 14) -> str:
        """
        Tworzy heatmapę skuteczności komponentów per detektor
        
        Args:
            days_back: Liczba dni wstecz do analizy
            
        Returns:
            str: Ścieżka do wygenerowanego pliku PNG
        """
        try:
            if not self.component_memory_file.exists():
                logger.warning(f"[COMPONENT VIZ] No component memory file found")
                return ""
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            # Collect component effectiveness data
            detector_component_data = {}
            
            with open(self.component_memory_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                        
                        if entry_time < cutoff_date:
                            continue
                        
                        detector = entry.get("detector", "Unknown")
                        scores = entry.get("scores", {})
                        was_successful = entry.get("was_successful", False)
                        
                        if detector not in detector_component_data:
                            detector_component_data[detector] = {}
                        
                        for component, score in scores.items():
                            if isinstance(score, (int, float)) and score > 0:
                                if component not in detector_component_data[detector]:
                                    detector_component_data[detector][component] = {
                                        "total_uses": 0,
                                        "successful_uses": 0
                                    }
                                
                                detector_component_data[detector][component]["total_uses"] += 1
                                if was_successful:
                                    detector_component_data[detector][component]["successful_uses"] += 1
                    
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            
            if not detector_component_data:
                logger.warning(f"[COMPONENT VIZ] No data available for heatmap")
                return ""
            
            # Calculate effectiveness percentages
            effectiveness_matrix = {}
            all_components = set()
            all_detectors = list(detector_component_data.keys())
            
            for detector, components in detector_component_data.items():
                effectiveness_matrix[detector] = {}
                for component, data in components.items():
                    if data["total_uses"] > 0:
                        effectiveness = (data["successful_uses"] / data["total_uses"]) * 100
                        effectiveness_matrix[detector][component] = effectiveness
                        all_components.add(component)
            
            # Create DataFrame for heatmap
            all_components = sorted(list(all_components))
            heatmap_data = []
            
            for detector in all_detectors:
                row = []
                for component in all_components:
                    effectiveness = effectiveness_matrix.get(detector, {}).get(component, 0.0)
                    row.append(effectiveness)
                heatmap_data.append(row)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Use DataFrame for better labeling
            df = pd.DataFrame(heatmap_data, index=all_detectors, columns=all_components)
            
            # Create heatmap with custom colormap
            sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                       cbar_kws={'label': 'Effectiveness %'}, ax=ax)
            
            ax.set_title(f'Component Effectiveness Heatmap (Last {days_back} Days)\nComponent Success Rate by Detector',
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Components', fontsize=12, fontweight='bold')
            ax.set_ylabel('Detectors', fontsize=12, fontweight='bold')
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Add timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            plt.figtext(0.02, 0.02, f"Generated: {timestamp}", fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            
            # Save heatmap
            heatmap_file = self.output_dir / f"component_effectiveness_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[COMPONENT VIZ] Generated effectiveness heatmap: {heatmap_file}")
            return str(heatmap_file)
            
        except Exception as e:
            logger.error(f"[COMPONENT VIZ] Error creating effectiveness heatmap: {e}")
            return ""
    
    def create_weight_evolution_trends(self, days_back: int = 30) -> str:
        """
        Tworzy wykresy trend wag: jak zmieniała się wartość clip_weight, diamond_weight w czasie
        
        Args:
            days_back: Liczba dni wstecz do analizy
            
        Returns:
            str: Ścieżka do wygenerowanego pliku PNG
        """
        try:
            if not self.component_memory_file.exists():
                logger.warning(f"[COMPONENT VIZ] No component memory file found")
                return ""
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            # Collect weight evolution data (simulated from feedback)
            weight_history = {}
            daily_aggregates = {}
            
            with open(self.component_memory_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                        
                        if entry_time < cutoff_date:
                            continue
                        
                        # Group by day for trend analysis
                        day_key = entry_time.strftime("%Y-%m-%d")
                        
                        if day_key not in daily_aggregates:
                            daily_aggregates[day_key] = {
                                "success_count": 0,
                                "total_count": 0,
                                "component_usage": {}
                            }
                        
                        daily_aggregates[day_key]["total_count"] += 1
                        if entry.get("was_successful", False):
                            daily_aggregates[day_key]["success_count"] += 1
                        
                        # Track component usage
                        scores = entry.get("scores", {})
                        for component, score in scores.items():
                            if isinstance(score, (int, float)) and score > 0:
                                if component not in daily_aggregates[day_key]["component_usage"]:
                                    daily_aggregates[day_key]["component_usage"][component] = []
                                daily_aggregates[day_key]["component_usage"][component].append(score)
                    
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            
            if not daily_aggregates:
                logger.warning(f"[COMPONENT VIZ] No weight evolution data available")
                return ""
            
            # Calculate simulated weight evolution based on performance
            dates = sorted(daily_aggregates.keys())
            component_trends = {}
            
            # Initialize components with baseline weight 1.0
            key_components = ["dex", "whale", "trust", "id", "diamond", "californium", "clip", "gnn"]
            
            for component in key_components:
                component_trends[component] = []
                current_weight = 1.0
                
                for date in dates:
                    day_data = daily_aggregates[date]
                    component_usage = day_data["component_usage"].get(component, [])
                    
                    if component_usage:
                        # Simulate weight adjustment based on daily performance
                        avg_score = np.mean(component_usage)
                        success_rate = day_data["success_count"] / max(day_data["total_count"], 1)
                        
                        # Adjust weight based on performance (learning simulation)
                        if success_rate > 0.6 and avg_score > 0.5:
                            current_weight = min(current_weight * 1.02, 2.0)  # Increase weight
                        elif success_rate < 0.4:
                            current_weight = max(current_weight * 0.98, 0.1)  # Decrease weight
                    
                    component_trends[component].append(current_weight)
            
            # Create weight evolution chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Convert dates to datetime objects
            date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
            
            # Plot 1: Main components (Classic Stealth)
            classic_components = ["dex", "whale", "trust", "id"]
            for component in classic_components:
                if len(component_trends[component]) > 0:
                    ax1.plot(date_objects, component_trends[component], 
                            marker='o', linewidth=2, label=f"{component.upper()}", markersize=4)
            
            ax1.set_title('Classic Stealth Component Weight Evolution', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Weight Factor', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0)')
            
            # Plot 2: AI Detector components
            ai_components = ["diamond", "californium", "clip", "gnn"]
            for component in ai_components:
                if len(component_trends[component]) > 0:
                    ax2.plot(date_objects, component_trends[component], 
                            marker='s', linewidth=2, label=f"{component.upper()}", markersize=4)
            
            ax2.set_title('AI Detector Component Weight Evolution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Weight Factor', fontsize=12)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0)')
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            plt.figtext(0.02, 0.02, f"Generated: {timestamp} | Period: {days_back} days", fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            
            # Save trends chart
            trends_file = self.output_dir / f"component_weight_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(trends_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[COMPONENT VIZ] Generated weight evolution trends: {trends_file}")
            return str(trends_file)
            
        except Exception as e:
            logger.error(f"[COMPONENT VIZ] Error creating weight evolution trends: {e}")
            return ""
    
    def create_detector_comparison_chart(self, days_back: int = 7) -> str:
        """
        Tworzy wykres porównawczy detektorów z component breakdown
        
        Args:
            days_back: Liczba dni wstecz do analizy
            
        Returns:
            str: Ścieżka do wygenerowanego pliku PNG
        """
        try:
            if not self.component_memory_file.exists():
                logger.warning(f"[COMPONENT VIZ] No component memory file found")
                return ""
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            # Collect detector performance data
            detector_stats = {}
            
            with open(self.component_memory_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                        
                        if entry_time < cutoff_date:
                            continue
                        
                        detector = entry.get("detector", "Unknown")
                        was_successful = entry.get("was_successful", False)
                        scores = entry.get("scores", {})
                        
                        if detector not in detector_stats:
                            detector_stats[detector] = {
                                "total_alerts": 0,
                                "successful_alerts": 0,
                                "component_contributions": {}
                            }
                        
                        detector_stats[detector]["total_alerts"] += 1
                        if was_successful:
                            detector_stats[detector]["successful_alerts"] += 1
                        
                        # Track component contributions
                        for component, score in scores.items():
                            if isinstance(score, (int, float)) and score > 0:
                                if component not in detector_stats[detector]["component_contributions"]:
                                    detector_stats[detector]["component_contributions"][component] = []
                                detector_stats[detector]["component_contributions"][component].append(score)
                    
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            
            if not detector_stats:
                logger.warning(f"[COMPONENT VIZ] No detector comparison data available")
                return ""
            
            # Create comparison chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Chart 1: Success Rates
            detectors = list(detector_stats.keys())
            success_rates = []
            
            for detector in detectors:
                stats = detector_stats[detector]
                rate = (stats["successful_alerts"] / max(stats["total_alerts"], 1)) * 100
                success_rates.append(rate)
            
            bars1 = ax1.bar(detectors, success_rates, color=sns.color_palette("viridis", len(detectors)))
            ax1.set_title('Detector Success Rates', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Success Rate (%)', fontsize=12)
            ax1.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, rate in zip(bars1, success_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Chart 2: Alert Volume
            alert_volumes = [detector_stats[detector]["total_alerts"] for detector in detectors]
            bars2 = ax2.bar(detectors, alert_volumes, color=sns.color_palette("plasma", len(detectors)))
            ax2.set_title('Alert Volume by Detector', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Total Alerts', fontsize=12)
            
            for bar, volume in zip(bars2, alert_volumes):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{volume}', ha='center', va='bottom', fontweight='bold')
            
            # Chart 3: Component Usage Distribution
            all_components = set()
            for detector_data in detector_stats.values():
                all_components.update(detector_data["component_contributions"].keys())
            
            component_usage_matrix = []
            for detector in detectors:
                detector_contributions = detector_stats[detector]["component_contributions"]
                row = []
                for component in sorted(all_components):
                    avg_contribution = np.mean(detector_contributions.get(component, [0]))
                    row.append(avg_contribution)
                component_usage_matrix.append(row)
            
            if component_usage_matrix:
                im = ax3.imshow(component_usage_matrix, cmap='YlOrRd', aspect='auto')
                ax3.set_title('Average Component Contributions', fontsize=14, fontweight='bold')
                ax3.set_xticks(range(len(sorted(all_components))))
                ax3.set_xticklabels(sorted(all_components), rotation=45, ha='right')
                ax3.set_yticks(range(len(detectors)))
                ax3.set_yticklabels(detectors)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax3)
                cbar.set_label('Avg Score', fontsize=10)
            
            # Chart 4: Performance Summary Table
            ax4.axis('tight')
            ax4.axis('off')
            
            # Create performance summary data
            table_data = []
            headers = ['Detector', 'Success Rate', 'Total Alerts', 'Top Component']
            
            for detector in detectors:
                stats = detector_stats[detector]
                success_rate = (stats["successful_alerts"] / max(stats["total_alerts"], 1)) * 100
                
                # Find top component
                top_component = "None"
                max_avg = 0
                for component, scores in stats["component_contributions"].items():
                    avg_score = np.mean(scores)
                    if avg_score > max_avg:
                        max_avg = avg_score
                        top_component = f"{component} ({avg_score:.2f})"
                
                table_data.append([
                    detector,
                    f"{success_rate:.1f}%",
                    str(stats["total_alerts"]),
                    top_component
                ])
            
            table = ax4.table(cellText=table_data, colLabels=headers,
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
            
            # Rotate x-axis labels for better readability
            for ax in [ax1, ax2]:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            plt.figtext(0.02, 0.02, f"Generated: {timestamp} | Period: {days_back} days", fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            
            # Save comparison chart
            comparison_file = self.output_dir / f"detector_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[COMPONENT VIZ] Generated detector comparison chart: {comparison_file}")
            return str(comparison_file)
            
        except Exception as e:
            logger.error(f"[COMPONENT VIZ] Error creating detector comparison chart: {e}")
            return ""
    
    def generate_all_visualizations(self, days_back: int = 14) -> Dict[str, str]:
        """
        Generuje wszystkie wizualizacje Component Effectiveness V4
        
        Args:
            days_back: Liczba dni wstecz do analizy
            
        Returns:
            Dict[str, str]: Ścieżki do wygenerowanych plików
        """
        try:
            results = {}
            
            print(f"[COMPONENT VIZ] Generating component effectiveness visualizations...")
            
            # Generate effectiveness heatmap
            heatmap_file = self.create_component_effectiveness_heatmap(days_back)
            if heatmap_file:
                results["effectiveness_heatmap"] = heatmap_file
            
            # Generate weight evolution trends
            trends_file = self.create_weight_evolution_trends(days_back * 2)  # Longer period for trends
            if trends_file:
                results["weight_evolution_trends"] = trends_file
            
            # Generate detector comparison
            comparison_file = self.create_detector_comparison_chart(days_back)
            if comparison_file:
                results["detector_comparison"] = comparison_file
            
            logger.info(f"[COMPONENT VIZ] Generated {len(results)} visualizations")
            return results
            
        except Exception as e:
            logger.error(f"[COMPONENT VIZ] Error generating visualizations: {e}")
            return {}


# Global instance and convenience functions
_visualizer = None

def get_component_visualizer() -> ComponentEffectivenessVisualizer:
    """Get global component visualizer instance"""
    global _visualizer
    if _visualizer is None:
        _visualizer = ComponentEffectivenessVisualizer()
    return _visualizer

def generate_component_effectiveness_charts(days_back: int = 14) -> Dict[str, str]:
    """Generate all component effectiveness visualizations"""
    return get_component_visualizer().generate_all_visualizations(days_back)

def create_effectiveness_heatmap(days_back: int = 14) -> str:
    """Create component effectiveness heatmap"""
    return get_component_visualizer().create_component_effectiveness_heatmap(days_back)

def create_weight_trends_chart(days_back: int = 30) -> str:
    """Create weight evolution trends chart"""
    return get_component_visualizer().create_weight_evolution_trends(days_back)


if __name__ == "__main__":
    # Test Component Effectiveness Visualization
    print("=== COMPONENT EFFECTIVENESS VISUALIZATION V4 TEST ===")
    
    visualizer = ComponentEffectivenessVisualizer()
    
    print("Testing component effectiveness heatmap...")
    heatmap_file = visualizer.create_component_effectiveness_heatmap(14)
    print(f"Heatmap generated: {heatmap_file}")
    
    print("Testing weight evolution trends...")
    trends_file = visualizer.create_weight_evolution_trends(30)
    print(f"Trends chart generated: {trends_file}")
    
    print("Testing detector comparison...")
    comparison_file = visualizer.create_detector_comparison_chart(7)
    print(f"Comparison chart generated: {comparison_file}")
    
    print("Testing all visualizations...")
    all_charts = visualizer.generate_all_visualizations(14)
    print(f"All visualizations: {all_charts}")
    
    print("=== TEST COMPLETE ===")