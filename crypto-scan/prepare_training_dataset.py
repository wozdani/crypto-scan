#!/usr/bin/env python3
"""
Training Dataset Preparation for Vision-AI
Prepares comprehensive dataset for Computer Vision model training
"""

import os
import sys
import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def prepare_vision_dataset(
    input_dir: str = "training_data",
    output_dir: str = "vision_dataset",
    format_type: str = "pytorch"
) -> bool:
    """
    Prepare training dataset for Computer Vision model
    
    Args:
        input_dir: Directory containing raw training data
        output_dir: Output directory for organized dataset
        format_type: Dataset format (pytorch, tensorflow, csv)
        
    Returns:
        True if preparation successful
    """
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return False
        
        # Create output structure
        output_path.mkdir(exist_ok=True)
        
        if format_type == "pytorch":
            return _prepare_pytorch_dataset(input_path, output_path)
        elif format_type == "tensorflow":
            return _prepare_tensorflow_dataset(input_path, output_path)
        elif format_type == "csv":
            return _prepare_csv_dataset(input_path, output_path)
        else:
            print(f"❌ Unknown format: {format_type}")
            return False
            
    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        return False


def _prepare_pytorch_dataset(input_path: Path, output_path: Path) -> bool:
    """Prepare PyTorch-style dataset"""
    try:
        print("📊 Preparing PyTorch dataset...")
        
        # Create PyTorch structure
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        splits_dir = output_path / "splits"
        
        for directory in [images_dir, labels_dir, splits_dir]:
            directory.mkdir(exist_ok=True)
        
        # Collect all samples
        charts_dir = input_path / "charts"
        labels_input_dir = input_path / "labels"
        meta_dir = input_path / "metadata"
        
        if not all([charts_dir.exists(), labels_input_dir.exists()]):
            print("❌ Required directories not found in input")
            return False
        
        # Process samples
        samples = []
        charts = list(charts_dir.glob("*.png"))
        
        print(f"🔍 Processing {len(charts)} chart samples...")
        
        for chart_file in charts:
            # Find corresponding label file
            base_name = chart_file.stem.replace("_chart", "")
            label_file = labels_input_dir / f"{base_name}_label.json"
            meta_file = meta_dir / f"{base_name}_meta.json" if meta_dir.exists() else None
            
            if not label_file.exists():
                print(f"⚠️ No label found for {chart_file.name}")
                continue
            
            # Load label data
            try:
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                
                labels_info = label_data.get("labels", {})
                setup_type = labels_info.get("setup_type", "unknown")
                phase_type = labels_info.get("phase_type", "unknown")
                confidence = labels_info.get("confidence", 0.0)
                
                # Load metadata if available
                metadata = {}
                if meta_file and meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                
                # Copy image to output
                output_image = images_dir / chart_file.name
                shutil.copy2(chart_file, output_image)
                
                # Create sample record
                sample = {
                    "image_file": chart_file.name,
                    "setup_type": setup_type,
                    "phase_type": phase_type,
                    "confidence": confidence,
                    "symbol": label_data.get("symbol", "unknown"),
                    "timestamp": label_data.get("timestamp", ""),
                    "metadata": metadata.get("context_features", {})
                }
                
                samples.append(sample)
                
            except Exception as e:
                print(f"⚠️ Error processing {chart_file.name}: {e}")
                continue
        
        # Save dataset annotations
        annotations_file = output_path / "annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump({
                "samples": samples,
                "dataset_info": {
                    "total_samples": len(samples),
                    "format": "pytorch",
                    "created_at": pd.Timestamp.now().isoformat(),
                    "classes": {
                        "setup_types": list(set(s["setup_type"] for s in samples)),
                        "phase_types": list(set(s["phase_type"] for s in samples))
                    }
                }
            }, f, indent=2)
        
        # Create train/val split
        _create_data_splits(samples, splits_dir)
        
        # Generate class mappings
        _generate_class_mappings(samples, output_path)
        
        print(f"✅ PyTorch dataset prepared: {len(samples)} samples")
        print(f"📁 Output directory: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch dataset preparation failed: {e}")
        return False


def _prepare_csv_dataset(input_path: Path, output_path: Path) -> bool:
    """Prepare CSV-style dataset for analysis"""
    try:
        print("📊 Preparing CSV dataset...")
        
        # Collect data
        charts_dir = input_path / "charts"
        labels_dir = input_path / "labels"
        meta_dir = input_path / "metadata"
        
        dataset_records = []
        
        for label_file in labels_dir.glob("*.json"):
            try:
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                
                base_name = label_file.stem.replace("_label", "")
                chart_file = charts_dir / f"{base_name}_chart.png"
                meta_file = meta_dir / f"{base_name}_meta.json" if meta_dir.exists() else None
                
                # Load metadata
                metadata = {}
                if meta_file and meta_file.exists():
                    with open(meta_file, 'r') as f:
                        meta_data = json.load(f)
                        metadata = meta_data.get("context_features", {})
                
                # Create record
                labels_info = label_data.get("labels", {})
                
                record = {
                    "symbol": label_data.get("symbol", "unknown"),
                    "timestamp": label_data.get("timestamp", ""),
                    "chart_file": chart_file.name if chart_file.exists() else "",
                    "setup_type": labels_info.get("setup_type", "unknown"),
                    "phase_type": labels_info.get("phase_type", "unknown"),
                    "confidence": labels_info.get("confidence", 0.0),
                    "labeler": labels_info.get("labeler", "unknown"),
                    # Add metadata features
                    **{f"feature_{k}": v for k, v in metadata.items() if isinstance(v, (int, float, bool))}
                }
                
                dataset_records.append(record)
                
            except Exception as e:
                print(f"⚠️ Error processing {label_file.name}: {e}")
                continue
        
        # Create DataFrame and save
        df = pd.DataFrame(dataset_records)
        
        if len(df) > 0:
            csv_path = output_path / "vision_dataset.csv"
            df.to_csv(csv_path, index=False)
            
            # Generate summary statistics
            summary = {
                "total_samples": len(df),
                "unique_symbols": df["symbol"].nunique(),
                "setup_type_distribution": df["setup_type"].value_counts().to_dict(),
                "phase_type_distribution": df["phase_type"].value_counts().to_dict(),
                "confidence_stats": {
                    "mean": float(df["confidence"].mean()),
                    "std": float(df["confidence"].std()),
                    "min": float(df["confidence"].min()),
                    "max": float(df["confidence"].max())
                }
            }
            
            with open(output_path / "dataset_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"✅ CSV dataset created: {len(df)} records")
            print(f"📊 Setup types: {list(df['setup_type'].unique())}")
            print(f"📊 Phase types: {list(df['phase_type'].unique())}")
            
            return True
        else:
            print("❌ No valid records found")
            return False
            
    except Exception as e:
        print(f"❌ CSV dataset preparation failed: {e}")
        return False


def _create_data_splits(samples: List[Dict], splits_dir: Path, train_ratio: float = 0.8):
    """Create train/validation splits"""
    try:
        import random
        random.shuffle(samples)
        
        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # Save splits
        with open(splits_dir / "train.json", 'w') as f:
            json.dump(train_samples, f, indent=2)
        
        with open(splits_dir / "val.json", 'w') as f:
            json.dump(val_samples, f, indent=2)
        
        print(f"📊 Data splits: {len(train_samples)} train, {len(val_samples)} val")
        
    except Exception as e:
        print(f"⚠️ Failed to create data splits: {e}")


def _generate_class_mappings(samples: List[Dict], output_path: Path):
    """Generate class to index mappings"""
    try:
        setup_types = sorted(set(s["setup_type"] for s in samples))
        phase_types = sorted(set(s["phase_type"] for s in samples))
        
        mappings = {
            "setup_type_to_idx": {setup: idx for idx, setup in enumerate(setup_types)},
            "phase_type_to_idx": {phase: idx for idx, phase in enumerate(phase_types)},
            "idx_to_setup_type": {idx: setup for idx, setup in enumerate(setup_types)},
            "idx_to_phase_type": {idx: phase for idx, phase in enumerate(phase_types)}
        }
        
        with open(output_path / "class_mappings.json", 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print(f"📋 Class mappings: {len(setup_types)} setup types, {len(phase_types)} phase types")
        
    except Exception as e:
        print(f"⚠️ Failed to generate class mappings: {e}")


def main():
    """Main function for dataset preparation"""
    parser = argparse.ArgumentParser(description="Prepare Vision-AI training dataset")
    parser.add_argument("--input", default="training_data", help="Input directory")
    parser.add_argument("--output", default="vision_dataset", help="Output directory")
    parser.add_argument("--format", choices=["pytorch", "tensorflow", "csv"], 
                       default="pytorch", help="Dataset format")
    
    args = parser.parse_args()
    
    print("🧠 Vision-AI Dataset Preparation")
    print("=" * 40)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    
    success = prepare_vision_dataset(args.input, args.output, args.format)
    
    if success:
        print("\n✅ Dataset preparation completed!")
        
        # Show final statistics
        output_path = Path(args.output)
        if (output_path / "dataset_summary.json").exists():
            with open(output_path / "dataset_summary.json", 'r') as f:
                summary = json.load(f)
            
            print(f"📊 Final Dataset Stats:")
            for key, value in summary.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
    else:
        print("\n❌ Dataset preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()