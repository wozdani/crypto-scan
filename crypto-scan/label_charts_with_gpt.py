#!/usr/bin/env python3
"""
GPT Chart Labeling with Automatic File Renaming
Iterates through charts, labels with GPT Vision, and renames files with labels
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.chart_labeler import ChartLabeler


class GPTChartRenamer:
    """GPT-based chart labeling with automatic file renaming"""
    
    def __init__(self):
        self.labeler = ChartLabeler()
        self.charts_dir = Path("data/chart_training/charts")
        self.labels_dir = Path("data/chart_training/labels")
        
        # Ensure directories exist
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported labels for this system
        self.supported_labels = [
            'breakout_continuation',
            'pullback_setup', 
            'range',
            'fakeout',
            'exhaustion',
            'retest_confirmation'
        ]
    
    def create_gpt_prompt(self, features: Dict) -> str:
        """Create specialized prompt for market phase classification"""
        
        features_text = []
        for key, value in features.items():
            if isinstance(value, bool):
                features_text.append(f"- {key}: {value}")
            elif isinstance(value, float):
                features_text.append(f"- {key}: {value:.2f}")
            else:
                features_text.append(f"- {key}: {value}")
        
        features_str = "\n".join(features_text)
        labels_str = ", ".join([f"'{label}'" for label in self.supported_labels])
        
        prompt = f"""Podaj tylko jedną z etykiet: [{labels_str}]

Na podstawie wykresu i kontekstu scoringu:
{features_str}

Oceń fazę rynku i odpowiedz tylko i wyłącznie nazwą klasy. Bez żadnych dodatków."""

        return prompt
    
    def label_single_chart(self, chart_path: Path, features: Dict = None) -> Optional[Dict]:
        """
        Label single chart with GPT and return result
        
        Args:
            chart_path: Path to chart image
            features: Market features for context
            
        Returns:
            Dict with label and confidence or None if failed
        """
        try:
            if not features:
                features = self._extract_features_from_filename(chart_path.name)
            
            print(f"[GPT LABEL] 🔍 Analyzing: {chart_path.name}")
            
            # Use custom prompt for this system
            original_prompt_method = self.labeler.create_vision_prompt
            self.labeler.create_vision_prompt = lambda f: self.create_gpt_prompt(f)
            
            # Temporarily update supported classes
            original_classes = self.labeler.pattern_classes
            self.labeler.pattern_classes = self.supported_labels
            
            try:
                # Get label from GPT
                label = self.labeler.label_chart_image(str(chart_path), features)
                
                # Restore original methods
                self.labeler.create_vision_prompt = original_prompt_method
                self.labeler.pattern_classes = original_classes
                
                if label in self.supported_labels:
                    # Estimate confidence based on label specificity
                    confidence = self._estimate_confidence(label, features)
                    
                    result = {
                        "label": label,
                        "confidence": confidence
                    }
                    
                    print(f"[GPT LABEL] ✅ Classified as: {label} (confidence: {confidence:.2f})")
                    return result
                else:
                    print(f"[GPT LABEL] ⚠️ Invalid label: {label}")
                    return None
                    
            finally:
                # Always restore original methods
                self.labeler.create_vision_prompt = original_prompt_method
                self.labeler.pattern_classes = original_classes
            
        except Exception as e:
            print(f"[GPT LABEL] ❌ Error labeling {chart_path.name}: {e}")
            return None
    
    def save_label_json(self, chart_path: Path, label_result: Dict) -> bool:
        """Save label as JSON file"""
        try:
            # Create JSON filename based on chart name
            json_filename = chart_path.stem + "_label.json"
            json_path = self.labels_dir / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(label_result, f, indent=2)
            
            print(f"[LABEL JSON] 💾 Saved: {json_filename}")
            return True
            
        except Exception as e:
            print(f"[LABEL JSON] ❌ Failed to save JSON: {e}")
            return False
    
    def rename_chart_with_label(self, chart_path: Path, label: str) -> Optional[Path]:
        """
        Rename chart file to include label
        Format: SYMBOL_TIMESTAMP_LABEL.png
        
        Args:
            chart_path: Current chart path
            label: Classified label
            
        Returns:
            New path if renamed, None if failed or already labeled
        """
        try:
            # Check if already labeled
            if any(supported_label in chart_path.name for supported_label in self.supported_labels):
                print(f"[RENAME] ⏭️ Already labeled: {chart_path.name}")
                return chart_path
            
            # Create new filename with label
            name_parts = chart_path.stem.split('_')
            if len(name_parts) >= 2:
                # Format: SYMBOL_TIMESTAMP_LABEL.png
                new_name = f"{name_parts[0]}_{name_parts[1]}_{label}.png"
            else:
                # Fallback: add label to existing name
                new_name = f"{chart_path.stem}_{label}.png"
            
            new_path = chart_path.parent / new_name
            
            # Rename file
            shutil.move(str(chart_path), str(new_path))
            
            print(f"[RENAME] ✅ Renamed: {chart_path.name} → {new_name}")
            return new_path
            
        except Exception as e:
            print(f"[RENAME] ❌ Failed to rename {chart_path.name}: {e}")
            return None
    
    def process_charts_directory(self) -> Dict:
        """
        Process all PNG files in charts directory
        
        Returns:
            Processing results summary
        """
        try:
            # Find all PNG files
            chart_files = list(self.charts_dir.glob("*.png"))
            
            if not chart_files:
                print(f"[PROCESS] ⚠️ No PNG files found in {self.charts_dir}")
                return {"processed": 0, "labeled": 0, "renamed": 0, "errors": 0}
            
            print(f"[PROCESS] 📊 Found {len(chart_files)} charts to process")
            
            results = {
                "processed": 0,
                "labeled": 0,
                "renamed": 0,
                "errors": 0,
                "processed_files": []
            }
            
            for chart_path in chart_files:
                try:
                    results["processed"] += 1
                    
                    # Extract features from filename or use defaults
                    features = self._extract_features_from_filename(chart_path.name)
                    
                    # Label with GPT
                    label_result = self.label_single_chart(chart_path, features)
                    
                    if label_result:
                        results["labeled"] += 1
                        
                        # Save JSON label
                        self.save_label_json(chart_path, label_result)
                        
                        # Rename file with label
                        new_path = self.rename_chart_with_label(chart_path, label_result["label"])
                        
                        if new_path and new_path != chart_path:
                            results["renamed"] += 1
                        
                        results["processed_files"].append({
                            "original": chart_path.name,
                            "label": label_result["label"],
                            "confidence": label_result["confidence"],
                            "renamed": new_path.name if new_path else chart_path.name
                        })
                    else:
                        results["errors"] += 1
                        
                except Exception as e:
                    print(f"[PROCESS] ❌ Error processing {chart_path.name}: {e}")
                    results["errors"] += 1
            
            return results
            
        except Exception as e:
            print(f"[PROCESS] ❌ Directory processing failed: {e}")
            return {"processed": 0, "labeled": 0, "renamed": 0, "errors": 1}
    
    def _extract_features_from_filename(self, filename: str) -> Dict:
        """Extract or generate features from filename patterns"""
        features = {
            "trend_strength": 0.6,
            "phase_score": 0.6,
            "pullback_quality": 0.5,
            "liquidity_score": 0.7,
            "htf_trend_match": True
        }
        
        # Pattern detection from filename
        filename_lower = filename.lower()
        
        if "breakout" in filename_lower:
            features.update({
                "trend_strength": 0.8,
                "phase_score": 0.85
            })
        elif "pullback" in filename_lower:
            features.update({
                "pullback_quality": 0.9,
                "phase_score": 0.75
            })
        elif "exhaustion" in filename_lower:
            features.update({
                "trend_strength": 0.3,
                "phase_score": 0.4
            })
        elif "range" in filename_lower or "consolidation" in filename_lower:
            features.update({
                "trend_strength": 0.4,
                "phase_score": 0.5
            })
        
        return features
    
    def _estimate_confidence(self, label: str, features: Dict) -> float:
        """Estimate confidence based on label and features alignment"""
        base_confidence = 0.75
        
        # Adjust based on feature alignment
        if label == "breakout_continuation" and features.get("trend_strength", 0) > 0.7:
            base_confidence += 0.1
        elif label == "pullback_setup" and features.get("pullback_quality", 0) > 0.8:
            base_confidence += 0.1
        elif label == "exhaustion" and features.get("trend_strength", 0) < 0.4:
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about processed charts"""
        try:
            chart_files = list(self.charts_dir.glob("*.png"))
            label_files = list(self.labels_dir.glob("*_label.json"))
            
            # Count labeled charts (those with labels in filename)
            labeled_charts = [f for f in chart_files 
                            if any(label in f.name for label in self.supported_labels)]
            
            # Count by label type
            label_counts = {}
            for label in self.supported_labels:
                count = len([f for f in chart_files if label in f.name])
                if count > 0:
                    label_counts[label] = count
            
            return {
                "total_charts": len(chart_files),
                "labeled_charts": len(labeled_charts),
                "json_labels": len(label_files),
                "label_distribution": label_counts,
                "charts_dir": str(self.charts_dir),
                "labels_dir": str(self.labels_dir)
            }
            
        except Exception as e:
            print(f"[STATS] ❌ Error getting stats: {e}")
            return {"error": str(e)}


def main():
    """Main function for GPT chart labeling and renaming"""
    print("🤖 GPT Chart Labeling with Automatic Renaming")
    print("=" * 50)
    
    # Initialize renamer
    renamer = GPTChartRenamer()
    
    # Check if OpenAI is available
    if not renamer.labeler.client:
        print("❌ OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        return
    
    # Get initial stats
    initial_stats = renamer.get_processing_stats()
    print(f"📊 Initial Statistics:")
    print(f"  Charts Found: {initial_stats.get('total_charts', 0)}")
    print(f"  Already Labeled: {initial_stats.get('labeled_charts', 0)}")
    
    if initial_stats.get('total_charts', 0) == 0:
        print("⚠️ No charts found. Run generate_chart_snapshot.py first.")
        return
    
    # Process all charts
    results = renamer.process_charts_directory()
    
    # Summary
    print(f"\n📈 Processing Results:")
    print(f"  Processed: {results['processed']}")
    print(f"  Successfully Labeled: {results['labeled']}")
    print(f"  Files Renamed: {results['renamed']}")
    print(f"  Errors: {results['errors']}")
    
    # Show processed files
    if results.get('processed_files'):
        print(f"\n🏷️ Labeled Charts:")
        for file_info in results['processed_files'][-5:]:  # Show last 5
            print(f"  {file_info['renamed']} ({file_info['label']}, {file_info['confidence']:.2f})")
    
    # Final stats
    final_stats = renamer.get_processing_stats()
    print(f"\n📊 Final Statistics:")
    print(f"  Total Charts: {final_stats.get('total_charts', 0)}")
    print(f"  Labeled Charts: {final_stats.get('labeled_charts', 0)}")
    
    label_dist = final_stats.get('label_distribution', {})
    if label_dist:
        print(f"  Label Distribution:")
        for label, count in label_dist.items():
            print(f"    {label}: {count}")
    
    print(f"\n✅ GPT labeling and renaming completed!")
    print(f"📁 Charts: {final_stats.get('charts_dir', 'Unknown')}")
    print(f"🏷️ Labels: {final_stats.get('labels_dir', 'Unknown')}")


if __name__ == "__main__":
    main()