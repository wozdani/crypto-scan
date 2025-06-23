#!/usr/bin/env python3
"""
Label Existing Charts with OpenAI Vision
Automatically label all existing charts in the exports directory
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.chart_labeler import chart_labeler
from utils.chart_exporter import get_export_stats
from pathlib import Path


def main():
    """Label all existing charts in the export directory"""
    
    print("🏷️ Labeling Existing Charts with OpenAI Vision")
    print("=" * 50)
    
    # Check if OpenAI is available
    if not chart_labeler.client:
        print("❌ OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        return
    
    # Get export directory stats
    stats = get_export_stats()
    export_dir = stats.get('export_folder', 'data/chart_exports')
    
    print(f"📁 Export Directory: {export_dir}")
    print(f"📊 Available Charts: {stats.get('total_files', 0)}")
    
    if stats.get('total_files', 0) == 0:
        print("⚠️ No charts found to label")
        return
    
    # Process all charts
    results = chart_labeler.batch_label_charts(export_dir)
    
    # Summary
    total_processed = len(results)
    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    failed = total_processed - successful
    
    print(f"\n📈 Labeling Results:")
    print(f"  Total Processed: {total_processed}")
    print(f"  Successfully Labeled: {successful}")
    print(f"  Failed: {failed}")
    
    # Show label distribution
    label_counts = {}
    for result in results.values():
        if result.get('status') == 'success':
            label = result.get('label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
    
    if label_counts:
        print(f"\n🏷️ Label Distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")
    
    # Dataset statistics
    dataset_stats = chart_labeler.get_dataset_stats()
    print(f"\n💾 Training Dataset:")
    print(f"  Total Entries: {dataset_stats.get('total_entries', 0)}")
    print(f"  Dataset File: {dataset_stats.get('dataset_file', 'Unknown')}")
    
    print(f"\n✅ Chart labeling completed!")
    print(f"📚 Training dataset ready for Computer Vision model training")


if __name__ == "__main__":
    main()