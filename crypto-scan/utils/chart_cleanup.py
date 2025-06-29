"""
Automated Chart Screenshot Cleanup System
Automatically removes old chart screenshots (>72h) that have been processed for training
"""

import os
import time
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set
import sys

def log_cleanup(message: str):
    """Log cleanup operations with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[CHART CLEANUP {timestamp}] {message}")

def is_screen_processed(screen_filename: str) -> bool:
    """
    Check if screenshot has been processed and saved in training datasets
    
    Args:
        screen_filename: Name of the screenshot file
        
    Returns:
        True if file has been processed and can be safely deleted
    """
    
    # Check 1: Training data labels.csv
    label_path = Path("training_data/labels.csv")
    if label_path.exists():
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                content = f.read()
                if screen_filename in content:
                    return True
        except Exception as e:
            log_cleanup(f"Error reading labels.csv: {e}")
    
    # Check 2: Training dataset JSONL
    dataset_path = Path("training_data/training_dataset.jsonl")
    if dataset_path.exists():
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if screen_filename in line:
                        return True
        except Exception as e:
            log_cleanup(f"Error reading training_dataset.jsonl: {e}")
    
    # Check 3: Embeddings directory
    base_name = screen_filename.replace(".png", "").replace(".webp", "").replace(".jpg", "")
    embedding_patterns = [
        f"data/embeddings/{base_name}.npy",
        f"data/embeddings/{base_name}_embedding.npy",
        f"data/embeddings/visual/{base_name}.npy"
    ]
    
    for pattern in embedding_patterns:
        if Path(pattern).exists():
            return True
    
    # Check 4: Training pairs directory
    training_pair_patterns = [
        f"data/training_pairs/{screen_filename}",
        f"training_pairs/{screen_filename}",
        f"data/vision_training/{screen_filename}"
    ]
    
    for pattern in training_pair_patterns:
        if Path(pattern).exists():
            return True
    
    # Check 5: Metadata files (JSON companions)
    metadata_patterns = [
        screen_filename.replace(".png", ".json"),
        screen_filename.replace(".webp", ".json"),
        screen_filename.replace(".jpg", ".json")
    ]
    
    for metadata_file in metadata_patterns:
        metadata_path = Path("training_data/charts") / metadata_file
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    # If metadata exists and has training flags, consider it processed
                    if metadata.get("used_in_training", False) or metadata.get("embedding_generated", False):
                        return True
            except Exception as e:
                log_cleanup(f"Error reading metadata {metadata_file}: {e}")
    
    return False

def get_chart_age_hours(file_path: str) -> float:
    """Get age of chart file in hours"""
    try:
        file_mtime = os.path.getmtime(file_path)
        age_seconds = time.time() - file_mtime
        return age_seconds / 3600.0
    except Exception:
        return 0.0

def safe_cleanup_old_screens(
    folder_paths: List[str] = None, 
    max_age_hours: int = 72,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Safely cleanup old chart screenshots that have been processed
    
    Args:
        folder_paths: List of folders to clean (default: training_data/charts)
        max_age_hours: Maximum age in hours before cleanup (default: 72)
        dry_run: If True, only report what would be deleted
        
    Returns:
        Dictionary with cleanup statistics
    """
    
    if folder_paths is None:
        folder_paths = ["training_data/charts", "data/charts", "screenshots"]
    
    stats = {
        "total_checked": 0,
        "old_files": 0,
        "processed_files": 0,
        "deleted_files": 0,
        "preserved_files": 0,
        "errors": 0
    }
    
    log_cleanup(f"Starting cleanup - Max age: {max_age_hours}h, Dry run: {dry_run}")
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            log_cleanup(f"Folder not found: {folder_path}")
            continue
            
        log_cleanup(f"Scanning folder: {folder_path}")
        
        # Get all image files
        image_patterns = ["*.png", "*.webp", "*.jpg", "*.jpeg"]
        image_files = []
        
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        
        for file_path in image_files:
            stats["total_checked"] += 1
            filename = os.path.basename(file_path)
            
            try:
                # Check file age
                age_hours = get_chart_age_hours(file_path)
                
                if age_hours > max_age_hours:
                    stats["old_files"] += 1
                    
                    # Check if file has been processed
                    if is_screen_processed(filename):
                        stats["processed_files"] += 1
                        
                        if dry_run:
                            log_cleanup(f"[DRY RUN] Would delete: {filename} (age: {age_hours:.1f}h)")
                        else:
                            try:
                                os.remove(file_path)
                                stats["deleted_files"] += 1
                                log_cleanup(f"✅ Deleted: {filename} (age: {age_hours:.1f}h)")
                                
                                # Also remove companion JSON metadata if exists
                                json_path = file_path.replace(".png", ".json").replace(".webp", ".json").replace(".jpg", ".json")
                                if os.path.exists(json_path):
                                    os.remove(json_path)
                                    log_cleanup(f"✅ Deleted metadata: {os.path.basename(json_path)}")
                                    
                            except Exception as e:
                                stats["errors"] += 1
                                log_cleanup(f"❌ Error deleting {filename}: {e}")
                    else:
                        stats["preserved_files"] += 1
                        log_cleanup(f"⚠️ Preserved (not in dataset): {filename} (age: {age_hours:.1f}h)")
                        
            except Exception as e:
                stats["errors"] += 1
                log_cleanup(f"❌ Error processing {filename}: {e}")
    
    # Summary
    log_cleanup(f"✅ Cleanup complete:")
    log_cleanup(f"   Total files checked: {stats['total_checked']}")
    log_cleanup(f"   Old files found: {stats['old_files']}")
    log_cleanup(f"   Processed files: {stats['processed_files']}")
    log_cleanup(f"   Files deleted: {stats['deleted_files']}")
    log_cleanup(f"   Files preserved: {stats['preserved_files']}")
    log_cleanup(f"   Errors: {stats['errors']}")
    
    return stats

def cleanup_with_size_report(max_age_hours: int = 72, dry_run: bool = False) -> Dict[str, any]:
    """
    Cleanup with disk space analysis
    
    Args:
        max_age_hours: Maximum age before cleanup
        dry_run: Only report, don't delete
        
    Returns:
        Detailed cleanup report with size information
    """
    
    # Calculate current disk usage
    total_size_before = 0
    folder_paths = ["training_data/charts", "data/charts", "screenshots"]
    
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('.png', '.webp', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        try:
                            total_size_before += os.path.getsize(file_path)
                        except:
                            pass
    
    log_cleanup(f"💾 Current chart storage: {total_size_before / (1024*1024*1024):.2f} GB")
    
    # Perform cleanup
    stats = safe_cleanup_old_screens(folder_paths, max_age_hours, dry_run)
    
    # Calculate size after cleanup (estimate for dry run)
    if not dry_run:
        total_size_after = 0
        for folder_path in folder_paths:
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith(('.png', '.webp', '.jpg', '.jpeg')):
                            file_path = os.path.join(root, file)
                            try:
                                total_size_after += os.path.getsize(file_path)
                            except:
                                pass
        
        space_saved = total_size_before - total_size_after
        log_cleanup(f"💾 Space saved: {space_saved / (1024*1024*1024):.2f} GB")
        log_cleanup(f"💾 Remaining storage: {total_size_after / (1024*1024*1024):.2f} GB")
        
        stats["space_saved_gb"] = space_saved / (1024*1024*1024)
        stats["remaining_size_gb"] = total_size_after / (1024*1024*1024)
    
    stats["initial_size_gb"] = total_size_before / (1024*1024*1024)
    
    return stats

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cleanup old chart screenshots")
    parser.add_argument("--max-age", type=int, default=72, help="Maximum age in hours (default: 72)")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be deleted")
    parser.add_argument("--folders", nargs="+", default=["training_data/charts"], help="Folders to clean")
    
    args = parser.parse_args()
    
    log_cleanup("🧹 Chart Cleanup System Starting")
    
    stats = cleanup_with_size_report(args.max_age, args.dry_run)
    
    if args.dry_run:
        log_cleanup("📊 Dry run completed - no files were deleted")
    else:
        log_cleanup("🎯 Cleanup completed successfully")
    
    return stats

if __name__ == "__main__":
    main()