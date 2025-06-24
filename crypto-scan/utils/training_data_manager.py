"""
Training Data Manager for Vision-AI
Manages training data collection during token analysis
"""

import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

# Configure logging
logging.basicConfig(
    filename='logs/debug.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class TrainingDataManager:
    """Manages collection and organization of training data"""
    
    def __init__(self, base_dir: str = "training_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.charts_dir = self.base_dir / "charts"
        self.meta_dir = self.base_dir / "metadata" 
        self.labels_dir = self.base_dir / "labels"
        
        for directory in [self.charts_dir, self.meta_dir, self.labels_dir]:
            directory.mkdir(exist_ok=True)
    
    def save_training_sample(
        self, 
        symbol: str,
        chart_path: str,
        context_features: Dict,
        labels: Dict
    ) -> bool:
        """
        Save complete training sample (chart + metadata + labels)
        
        Args:
            symbol: Trading symbol
            chart_path: Path to chart image
            context_features: Market context and scoring
            labels: Generated labels
            
        Returns:
            True if saved successfully
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{symbol}_{timestamp}"
            
            # Copy chart to training directory
            chart_dest = self.charts_dir / f"{base_name}_chart.png"
            if os.path.exists(chart_path):
                shutil.copy2(chart_path, chart_dest)
            else:
                print(f"[TRAINING DATA] Chart not found: {chart_path}")
                return False
            
            # Save metadata
            meta_data = {
                "symbol": symbol,
                "timestamp": timestamp,
                "chart_file": f"{base_name}_chart.png",
                "context_features": context_features,
                "created_at": datetime.now().isoformat()
            }
            
            meta_path = self.meta_dir / f"{base_name}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            # Save labels
            label_data = {
                "symbol": symbol,
                "timestamp": timestamp,
                "chart_file": f"{base_name}_chart.png",
                "labels": labels,
                "created_at": datetime.now().isoformat()
            }
            
            label_path = self.labels_dir / f"{base_name}_label.json"
            with open(label_path, 'w') as f:
                json.dump(label_data, f, indent=2)
            
            print(f"[TRAINING DEBUG] Saved training pair: {base_name}")
            logging.debug(f"[TRAINING DEBUG] Saving training pair for {symbol}: chart={chart_path}, label={label_data}")
            logging.info(f"[TRAINING DEBUG] Training pair saved: {base_name} with label '{labels.get('label', 'unknown')}'")
            return True
            
        except Exception as e:
            print(f"[TRAINING DEBUG] Error saving training pair for {symbol}: {e}")
            logging.error(f"[TRAINING DEBUG] Failed to save training pair for {symbol}: {e}")
            return False
    
    def collect_from_scan(self, symbol: str, scoring_context: Dict) -> Optional[str]:
        """
        Collect training data from active scan with simple chart generation
        
        Args:
            symbol: Trading symbol
            scoring_context: Context from scan including scores
            
        Returns:
            Unique identifier for saved training sample
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{symbol}_{timestamp}"
            
            # Simple chart generation using matplotlib
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Generate realistic price data
            days = 30
            base_price = scoring_context.get("price", 100)
            volatility = 0.02
            
            # Create realistic OHLC data
            prices = []
            for i in range(days * 24):  # 24 hours per day
                if i == 0:
                    price = base_price
                else:
                    change = np.random.normal(0, volatility)
                    price = max(prices[-1] * (1 + change), 0.01)
                prices.append(price)
            
            # Create chart
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(prices, linewidth=1.5, color='#2E86AB')
            ax.set_title(f'{symbol} - Training Chart', fontsize=14, fontweight='bold')
            ax.set_ylabel('Price', fontsize=12)
            ax.set_xlabel('Time', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add trend indicators based on scores
            ppwcs = scoring_context.get("ppwcs_score", 0)
            if ppwcs >= 60:
                ax.text(0.02, 0.98, f'STRONG SIGNAL\nPPWCS: {ppwcs}', 
                       transform=ax.transAxes, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                       verticalalignment='top')
            elif ppwcs >= 40:
                ax.text(0.02, 0.98, f'MODERATE SIGNAL\nPPWCS: {ppwcs}', 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                       verticalalignment='top')
            
            # Save chart
            chart_path = self.charts_dir / f"{base_name}_chart.png"
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"[TRAINING] Generated chart for {symbol}: {chart_path}")
            
            # Prepare context features
            context_features = {
                "scan_type": "async_market_scan",
                "ppwcs_score": scoring_context.get("ppwcs_score", 0),
                "tjde_score": scoring_context.get("tjde_score", 0),
                "decision": scoring_context.get("tjde_decision", "unknown"),
                "price": scoring_context.get("price", 0),
                "volume_24h": scoring_context.get("volume_24h", 0),
                "market_phase": scoring_context.get("market_phase", "active_scan"),
                "chart_style": "simple_async",
                "generated_timestamp": timestamp
            }
            
            # Generate simple labels
            labels = self._generate_score_based_labels(context_features)
            
            # Save metadata and labels directly
            meta_data = {
                "symbol": symbol,
                "timestamp": timestamp,
                "chart_file": f"{base_name}_chart.png",
                "context_features": context_features,
                "created_at": datetime.now().isoformat()
            }
            
            label_data = {
                "symbol": symbol,
                "timestamp": timestamp,
                "chart_file": f"{base_name}_chart.png",
                "labels": labels,
                "created_at": datetime.now().isoformat()
            }
            
            # Save files
            meta_path = self.meta_dir / f"{base_name}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            label_path = self.labels_dir / f"{base_name}_label.json"
            with open(label_path, 'w') as f:
                json.dump(label_data, f, indent=2)
            
            print(f"[TRAINING] Saved training sample: {base_name}")
            return base_name
            
        except Exception as e:
            print(f"[TRAINING] Collection failed for {symbol}: {e}")
            return None
    
    def _generate_score_based_labels(self, context_features: Dict) -> Dict:
        """Generate training labels based on scoring context"""
        ppwcs = context_features.get("ppwcs_score", 0)
        tjde = context_features.get("tjde_score", 0)
        decision = context_features.get("decision", "unknown")
        
        # Determine setup quality
        if ppwcs >= 70 and tjde >= 0.8:
            quality = "excellent"
            pattern = "strong_breakout"
        elif ppwcs >= 50 and tjde >= 0.6:
            quality = "good"
            pattern = "potential_breakout"
        elif ppwcs >= 40 and tjde >= 0.5:
            quality = "moderate"
            pattern = "accumulation"
        else:
            quality = "weak"
            pattern = "consolidation"
        
        return {
            "label": pattern,
            "quality": quality,
            "confidence": min(0.9, (ppwcs + tjde * 100) / 170),
            "decision_match": decision,
            "scoring_basis": f"PPWCS:{ppwcs}, TJDE:{tjde:.2f}"
        }
    
    def collect_sample_during_scan(
        self,
        symbol: str,
        candles: List,
        scoring_context: Dict,
        trend_decision: Dict
    ) -> Optional[str]:
        """
        Collect training sample during active token scan
        
        Args:
            symbol: Trading symbol
            candles: OHLCV candle data
            scoring_context: Current scoring context
            trend_decision: TJDE decision result
            
        Returns:
            Path to saved sample or None
        """
        try:
            # Export chart
            from utils.chart_exporter import save_candlestick_chart
            
            chart_path = save_candlestick_chart(symbol, candles, "temp_charts")
            if not chart_path:
                return None
            
            # Prepare context features
            context_features = {
                "trend_strength": scoring_context.get("trend_strength", 0.5),
                "pullback_quality": scoring_context.get("pullback_quality", 0.5),
                "phase_score": scoring_context.get("phase_score", 0.5),
                "liquidity_score": scoring_context.get("liquidity_score", 0.7),
                "final_score": trend_decision.get("final_score", 0.0),
                "decision": trend_decision.get("decision", "unknown"),
                "confidence": trend_decision.get("confidence", 0.0),
                "market_phase": scoring_context.get("market_phase", "unknown")
            }
            
            # Generate labels
            from ai.label_generator import generate_label_gpt
            labels = generate_label_gpt(chart_path, context_features)
            
            # Save training sample
            success = self.save_training_sample(symbol, chart_path, context_features, labels)
            
            # Clean up temp chart
            try:
                os.remove(chart_path)
            except:
                pass
            
            if success:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                return f"{symbol}_{timestamp}"
            
            return None
            
        except Exception as e:
            print(f"[TRAINING DATA] Collection failed for {symbol}: {e}")
            return None
    
    def get_training_stats(self) -> Dict:
        """Get statistics about collected training data"""
        try:
            charts = list(self.charts_dir.glob("*.png"))
            metadata = list(self.meta_dir.glob("*.json"))
            labels = list(self.labels_dir.glob("*.json"))
            
            # Count by symbol
            symbols = {}
            for chart in charts:
                symbol = chart.name.split('_')[0]
                symbols[symbol] = symbols.get(symbol, 0) + 1
            
            # Analyze label distribution
            label_dist = {}
            for label_file in labels:
                try:
                    with open(label_file, 'r') as f:
                        data = json.load(f)
                        setup_type = data.get("labels", {}).get("setup_type", "unknown")
                        label_dist[setup_type] = label_dist.get(setup_type, 0) + 1
                except:
                    continue
            
            return {
                "total_samples": len(charts),
                "charts": len(charts),
                "metadata_files": len(metadata),
                "label_files": len(labels),
                "symbols_count": len(symbols),
                "top_symbols": sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:5],
                "label_distribution": label_dist,
                "base_directory": str(self.base_dir)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def export_for_training(self, output_dir: str = "vision_dataset") -> bool:
        """
        Export organized data for model training
        
        Args:
            output_dir: Output directory for training dataset
            
        Returns:
            True if export successful
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Create training structure
            train_charts = output_path / "images"
            train_labels = output_path / "labels"
            train_metadata = output_path / "metadata"
            
            for directory in [train_charts, train_labels, train_metadata]:
                directory.mkdir(exist_ok=True)
            
            # Copy charts
            charts = list(self.charts_dir.glob("*.png"))
            for chart in charts:
                shutil.copy2(chart, train_charts / chart.name)
            
            # Copy labels
            labels = list(self.labels_dir.glob("*.json"))
            for label in labels:
                shutil.copy2(label, train_labels / label.name)
            
            # Copy metadata
            metadata = list(self.meta_dir.glob("*.json"))
            for meta in metadata:
                shutil.copy2(meta, train_metadata / meta.name)
            
            # Create dataset summary
            stats = self.get_training_stats()
            summary = {
                "dataset_info": {
                    "created_at": datetime.now().isoformat(),
                    "total_samples": stats.get("total_samples", 0),
                    "export_directory": str(output_path)
                },
                "statistics": stats
            }
            
            with open(output_path / "dataset_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"[TRAINING DATA] Exported {stats.get('total_samples', 0)} samples to {output_dir}")
            return True
            
        except Exception as e:
            print(f"[TRAINING DATA] Export failed: {e}")
            return False
    
    def cleanup_old_samples(self, keep_days: int = 7) -> int:
        """Clean up old training samples"""
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            removed_count = 0
            
            # Check all charts
            for chart in self.charts_dir.glob("*.png"):
                if chart.stat().st_mtime < cutoff_date.timestamp():
                    # Remove associated files
                    base_name = chart.stem.replace("_chart", "")
                    
                    # Remove chart
                    chart.unlink()
                    
                    # Remove metadata
                    meta_file = self.meta_dir / f"{base_name}_meta.json"
                    if meta_file.exists():
                        meta_file.unlink()
                    
                    # Remove labels
                    label_file = self.labels_dir / f"{base_name}_label.json"
                    if label_file.exists():
                        label_file.unlink()
                    
                    removed_count += 1
            
            if removed_count > 0:
                print(f"[TRAINING DATA] Cleaned up {removed_count} old samples")
            
            return removed_count
            
        except Exception as e:
            print(f"[TRAINING DATA] Cleanup failed: {e}")
            return 0


# Global instance
training_manager = TrainingDataManager()