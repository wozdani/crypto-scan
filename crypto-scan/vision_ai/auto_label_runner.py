#!/usr/bin/env python3
"""
Auto Label Runner for Vision-AI Training
Automatically processes top tokens from TJDE scans and creates training data
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_ai.chart_exporter import save_chart_snapshot, export_charts_for_symbols
from vision_ai.gpt_chart_labeler import label_chart


class AutoLabelRunner:
    """Automatically creates training data from TJDE scan results"""
    
    def __init__(self):
        # Directory setup
        self.charts_dir = Path("data/charts")
        self.train_data_dir = Path("data/vision_ai/train_data")
        self.results_dir = Path("data/tjde_results")
        
        # Create directories
        for directory in [self.charts_dir, self.train_data_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Processing stats
        self.session_stats = {
            "processed_symbols": 0,
            "charts_exported": 0,
            "labels_generated": 0,
            "training_pairs_created": 0,
            "errors": 0
        }
    
    def get_top_tjde_tokens(self, top_count: int = 5) -> List[Dict]:
        """
        Get top tokens from latest TJDE scan results
        
        Args:
            top_count: Number of top tokens to retrieve
            
        Returns:
            List of top TJDE results
        """
        try:
            # Look for recent TJDE results
            tjde_files = list(self.results_dir.glob("tjde_results_*.json"))
            
            if not tjde_files:
                print("[AUTO LABEL] No TJDE results found")
                return []
            
            # Get latest results file
            latest_file = max(tjde_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                tjde_data = json.load(f)
            
            # Extract and sort results
            if "results" in tjde_data:
                results = tjde_data["results"]
            elif isinstance(tjde_data, list):
                results = tjde_data
            else:
                print("[AUTO LABEL] Invalid TJDE data format")
                return []
            
            # Sort by final_score
            sorted_results = sorted(
                results, 
                key=lambda x: x.get("final_score", 0), 
                reverse=True
            )
            
            top_tokens = sorted_results[:top_count]
            
            print(f"[AUTO LABEL] Found {len(top_tokens)} top tokens from {latest_file.name}")
            return top_tokens
            
        except Exception as e:
            print(f"[AUTO LABEL] Failed to get TJDE tokens: {e}")
            return []
    
    def process_token(self, token_data: Dict) -> Dict:
        """
        Process single token: export chart and generate label
        
        Args:
            token_data: TJDE analysis result for token
            
        Returns:
            Processing result
        """
        try:
            symbol = token_data.get("symbol", "UNKNOWN")
            
            print(f"[AUTO LABEL] Processing {symbol}...")
            
            self.session_stats["processed_symbols"] += 1
            
            # Step 1: Export chart
            chart_path = save_chart_snapshot(
                symbol=symbol,
                timeframe="15m",
                output_dir=str(self.charts_dir)
            )
            
            if not chart_path:
                self.session_stats["errors"] += 1
                return {"symbol": symbol, "error": "Chart export failed"}
            
            self.session_stats["charts_exported"] += 1
            
            # Step 2: Generate GPT label
            label_result = label_chart(chart_path, token_data)
            
            if "error" in label_result:
                self.session_stats["errors"] += 1
                return {"symbol": symbol, "error": f"Labeling failed: {label_result['error']}"}
            
            self.session_stats["labels_generated"] += 1
            
            # Step 3: Create training data pair
            training_pair = self.create_training_pair(
                symbol=symbol,
                chart_path=chart_path,
                label_path=label_result.get("text_path"),
                token_data=token_data
            )
            
            if training_pair:
                self.session_stats["training_pairs_created"] += 1
            
            result = {
                "symbol": symbol,
                "chart_path": chart_path,
                "label_path": label_result.get("text_path"),
                "description": label_result.get("description"),
                "training_pair": training_pair,
                "success": True
            }
            
            print(f"[AUTO LABEL] ✅ {symbol}: {label_result.get('description', 'labeled')}")
            return result
            
        except Exception as e:
            self.session_stats["errors"] += 1
            print(f"[AUTO LABEL] ❌ Failed to process {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def create_training_pair(
        self, 
        symbol: str,
        chart_path: str,
        label_path: str,
        token_data: Dict
    ) -> Optional[str]:
        """
        Create training data pair in train_data directory
        
        Args:
            symbol: Trading symbol
            chart_path: Path to chart image
            label_path: Path to label text file
            token_data: TJDE analysis data
            
        Returns:
            Path to created training pair or None if failed
        """
        try:
            if not chart_path or not label_path:
                return None
            
            if not os.path.exists(chart_path) or not os.path.exists(label_path):
                return None
            
            # Generate training pair filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            base_name = f"{symbol}_{timestamp}"
            
            training_chart = self.train_data_dir / f"{base_name}.png"
            training_label = self.train_data_dir / f"{base_name}.txt"
            
            # Copy chart to training directory
            import shutil
            shutil.copy2(chart_path, training_chart)
            
            # Copy and enhance label file
            with open(label_path, 'r', encoding='utf-8') as f:
                label_content = f.read()
            
            # Add TJDE context to label
            enhanced_content = label_content + "\n\n# TJDE Analysis Context:\n"
            
            for key, value in token_data.items():
                if key in ["symbol", "final_score", "decision", "confidence"]:
                    enhanced_content += f"# {key}: {value}\n"
            
            with open(training_label, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)
            
            print(f"[AUTO LABEL] 💾 Training pair created: {base_name}")
            return str(training_chart)
            
        except Exception as e:
            print(f"[AUTO LABEL] Failed to create training pair for {symbol}: {e}")
            return None
    
    def run_auto_labeling(self, top_count: int = 5) -> Dict:
        """
        Run complete auto-labeling process
        
        Args:
            top_count: Number of top tokens to process
            
        Returns:
            Processing summary
        """
        try:
            print(f"🤖 Auto Label Runner - Processing Top {top_count} Tokens")
            print("=" * 60)
            
            # Reset session stats
            self.session_stats = {key: 0 for key in self.session_stats}
            
            # Get top tokens from TJDE
            top_tokens = self.get_top_tjde_tokens(top_count)
            
            if not top_tokens:
                return {"error": "No TJDE tokens found"}
            
            # Process each token
            results = []
            
            for token_data in top_tokens:
                result = self.process_token(token_data)
                results.append(result)
            
            # Generate summary
            summary = {
                "timestamp": datetime.now().isoformat(),
                "tokens_requested": top_count,
                "tokens_found": len(top_tokens),
                "processing_stats": self.session_stats.copy(),
                "results": results,
                "success_rate": (
                    self.session_stats["training_pairs_created"] / 
                    self.session_stats["processed_symbols"]
                ) if self.session_stats["processed_symbols"] > 0 else 0.0
            }
            
            # Save session summary
            self.save_session_summary(summary)
            
            # Print summary
            self.print_summary(summary)
            
            return summary
            
        except Exception as e:
            print(f"[AUTO LABEL] Auto-labeling failed: {e}")
            return {"error": str(e)}
    
    def save_session_summary(self, summary: Dict):
        """Save session summary to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.results_dir / f"auto_label_session_{timestamp}.json"
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"[AUTO LABEL] Session summary saved: {summary_file.name}")
            
        except Exception as e:
            print(f"[AUTO LABEL] Failed to save summary: {e}")
    
    def print_summary(self, summary: Dict):
        """Print processing summary"""
        stats = summary["processing_stats"]
        
        print(f"\n📊 Auto-Labeling Session Summary:")
        print(f"  Tokens processed: {stats['processed_symbols']}")
        print(f"  Charts exported: {stats['charts_exported']}")
        print(f"  Labels generated: {stats['labels_generated']}")
        print(f"  Training pairs created: {stats['training_pairs_created']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        
        # Show successful results
        successful_results = [r for r in summary["results"] if r.get("success")]
        
        if successful_results:
            print(f"\n✅ Successfully Processed:")
            for result in successful_results:
                symbol = result["symbol"]
                desc = result.get("description", "labeled")
                print(f"  {symbol}: {desc}")
    
    def get_training_data_stats(self) -> Dict:
        """Get statistics about training data"""
        try:
            chart_files = list(self.train_data_dir.glob("*.png"))
            text_files = list(self.train_data_dir.glob("*.txt"))
            
            # Count matching pairs
            pairs_count = 0
            for chart_file in chart_files:
                text_file = chart_file.with_suffix('.txt')
                if text_file.exists():
                    pairs_count += 1
            
            # Analyze symbols
            symbols = set()
            for chart_file in chart_files:
                symbol = chart_file.name.split('_')[0]
                symbols.add(symbol)
            
            return {
                "total_charts": len(chart_files),
                "total_labels": len(text_files),
                "complete_pairs": pairs_count,
                "unique_symbols": len(symbols),
                "symbols_list": sorted(list(symbols)),
                "train_data_dir": str(self.train_data_dir)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup_old_training_data(self, keep_count: int = 50) -> int:
        """Clean up old training data, keeping most recent pairs"""
        try:
            chart_files = list(self.train_data_dir.glob("*.png"))
            
            if len(chart_files) <= keep_count:
                return 0
            
            # Sort by modification time (oldest first)
            chart_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest files and their pairs
            files_to_remove = chart_files[:-keep_count]
            removed_count = 0
            
            for chart_file in files_to_remove:
                try:
                    # Remove chart
                    chart_file.unlink()
                    
                    # Remove corresponding text file
                    text_file = chart_file.with_suffix('.txt')
                    if text_file.exists():
                        text_file.unlink()
                    
                    removed_count += 1
                    
                except Exception as e:
                    print(f"[CLEANUP] Failed to remove {chart_file.name}: {e}")
            
            if removed_count > 0:
                print(f"[CLEANUP] Removed {removed_count} old training pairs")
            
            return removed_count
            
        except Exception as e:
            print(f"[CLEANUP] Training data cleanup failed: {e}")
            return 0


# Global runner instance
auto_runner = AutoLabelRunner()


def run_auto_labeling_for_tjde(top_count: int = 5) -> Dict:
    """
    Main function to run auto-labeling from TJDE results
    
    Args:
        top_count: Number of top tokens to process
        
    Returns:
        Processing summary
    """
    return auto_runner.run_auto_labeling(top_count)


def main():
    """Test auto label runner"""
    print("🤖 Auto Label Runner Test")
    print("=" * 40)
    
    # Show current training data stats
    stats = auto_runner.get_training_data_stats()
    print(f"📊 Current Training Data:")
    print(f"  Complete pairs: {stats.get('complete_pairs', 0)}")
    print(f"  Unique symbols: {stats.get('unique_symbols', 0)}")
    
    if stats.get('symbols_list'):
        print(f"  Symbols: {', '.join(stats['symbols_list'][:5])}")
    
    # Test getting TJDE tokens
    top_tokens = auto_runner.get_top_tjde_tokens(3)
    
    if top_tokens:
        print(f"\n🎯 Found {len(top_tokens)} top TJDE tokens:")
        for token in top_tokens:
            symbol = token.get("symbol", "UNKNOWN")
            score = token.get("final_score", 0)
            print(f"  {symbol}: {score:.3f}")
    else:
        print("\n⚠️ No TJDE tokens found - run crypto scan first")
    
    print(f"\n✅ Auto Label Runner ready")


if __name__ == "__main__":
    main()