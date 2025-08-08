#!/usr/bin/env python3
"""
Batch converter for explore mode files to training data
Uses real data loaders from Bybit API
"""

import json
import sys
import os
import glob as glob_module
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.explore_to_train import convert_one, batch_convert
from tools.real_data_loaders import get_price_and_volume, get_future_prices, load_weights_from_cache

def find_explore_files(base_dir: str = ".", pattern: str = "*_explore.json") -> list:
    """
    Find all explore mode files in the data directory
    
    Args:
        base_dir: Base directory to search
        pattern: File pattern to match
    
    Returns:
        List of file paths
    """
    explore_files = []
    
    # Search in common locations
    search_paths = [
        os.path.join(base_dir, pattern),
        os.path.join(base_dir, "cache", "explore_mode", pattern),
        os.path.join(base_dir, "data", pattern),
        os.path.join(base_dir, "exports", pattern),
        os.path.join("cache", "explore_mode", pattern),
        os.path.join("data", pattern),
        os.path.join("exports", pattern)
    ]
    
    for search_path in search_paths:
        files = glob_module.glob(search_path)
        explore_files.extend(files)
    
    # Remove duplicates
    explore_files = list(set(explore_files))
    
    print(f"[CONVERTER] Found {len(explore_files)} explore files")
    for f in explore_files[:5]:  # Show first 5
        print(f"  - {f}")
    if len(explore_files) > 5:
        print(f"  ... and {len(explore_files) - 5} more")
    
    return explore_files

def convert_batch_with_real_data(
    explore_files: list = None,
    output_dir: str = "crypto-scan/training_data",
    max_files: int = None
):
    """
    Convert explore files to training data using real Bybit data
    
    Args:
        explore_files: List of explore files to convert (or None to find automatically)
        output_dir: Output directory for JSONL files
        max_files: Maximum number of files to process
    """
    # Find files if not provided
    if explore_files is None:
        explore_files = find_explore_files()
    
    if not explore_files:
        print("[CONVERTER] No explore files found")
        return 0
    
    # Load current weights
    weights = load_weights_from_cache()
    print(f"[CONVERTER] Loaded weights: {list(weights.keys())}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"stealth_train_{timestamp}.jsonl"
    
    # Process files
    converted = 0
    errors = 0
    
    if max_files:
        explore_files = explore_files[:max_files]
    
    print(f"[CONVERTER] Processing {len(explore_files)} files...")
    print(f"[CONVERTER] Output: {output_file}")
    
    with open(output_file, 'w') as f:
        for i, explore_file in enumerate(explore_files):
            try:
                print(f"[CONVERTER] [{i+1}/{len(explore_files)}] Processing {explore_file}...")
                
                # Convert using real data loaders
                sample = convert_one(
                    explore_file,
                    price_loader=get_price_and_volume,
                    weights=weights,
                    ohlcv_loader=get_future_prices
                )
                
                # Write to JSONL
                f.write(json.dumps(sample) + '\n')
                converted += 1
                
                # Progress update
                if converted % 10 == 0:
                    print(f"[CONVERTER] Progress: {converted}/{len(explore_files)} converted")
                
            except Exception as e:
                errors += 1
                print(f"[CONVERTER ERROR] Failed on {explore_file}: {e}")
                
                if errors > len(explore_files) * 0.2:  # Stop if >20% errors
                    print(f"[CONVERTER] Too many errors ({errors}), stopping...")
                    break
    
    print(f"\n[CONVERTER] ✅ Complete!")
    print(f"  - Converted: {converted} files")
    print(f"  - Errors: {errors} files")
    print(f"  - Output: {output_file}")
    print(f"  - Size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Show sample of output
    if converted > 0:
        with open(output_file, 'r') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            print(f"\n[CONVERTER] Sample output:")
            print(f"  - Symbol: {sample['symbol']}")
            print(f"  - Price: ${sample['price_ref']:.6f}")
            print(f"  - Label: {sample['label']['y_hit_6h']}")
            print(f"  - Signals: {list(sample['signals'].keys())}")
    
    return converted

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert explore mode files to training data")
    parser.add_argument("--max", type=int, help="Maximum files to process")
    parser.add_argument("--output", default="crypto-scan/training_data", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Test mode - process only 3 files")
    
    args = parser.parse_args()
    
    if args.test:
        print("[CONVERTER] Running in test mode (max 3 files)...")
        args.max = 3
    
    # Run conversion
    converted = convert_batch_with_real_data(
        output_dir=args.output,
        max_files=args.max
    )
    
    return 0 if converted > 0 else 1

if __name__ == "__main__":
    sys.exit(main())