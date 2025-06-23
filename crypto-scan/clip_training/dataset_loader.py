"""
Dataset Loader for CLIP Training
Ładuje pary (obraz, opis) z auto_labels/ dla trenowania CLIP
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from PIL import Image
import torch

logger = logging.getLogger(__name__)

class CLIPDatasetLoader:
    """Loads image-text pairs for CLIP training"""
    
    def __init__(self, auto_labels_dir: str = "data/vision_ai/train_data"):
        """
        Initialize dataset loader
        
        Args:
            auto_labels_dir: Directory containing auto-labeled data
        """
        self.auto_labels_dir = Path(auto_labels_dir)
        self.charts_dir = self.auto_labels_dir / "charts"
        self.labels_dir = self.auto_labels_dir / "labels"
        
        # Alternative paths to check
        self.alternative_paths = [
            Path("auto_labels"),
            Path("data/auto_labels"),
            Path("exports"),
            Path("charts")
        ]
        
        logger.info(f"Initialized dataset loader for {self.auto_labels_dir}")
    
    def find_data_directories(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Find charts and labels directories"""
        # Check primary path
        if self.charts_dir.exists() and self.labels_dir.exists():
            return self.charts_dir, self.labels_dir
        
        # Check alternative paths
        for alt_path in self.alternative_paths:
            if alt_path.exists():
                charts_path = alt_path / "charts" if (alt_path / "charts").exists() else alt_path
                labels_path = alt_path / "labels" if (alt_path / "labels").exists() else alt_path
                
                # Check if we have both images and labels
                has_images = any(charts_path.glob("*.png"))
                has_labels = any(labels_path.glob("*.txt")) or any(labels_path.glob("*.json"))
                
                if has_images and has_labels:
                    return charts_path, labels_path
        
        return None, None
    
    def load_training_pairs(self, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Load image-text pairs for training
        
        Args:
            limit: Maximum number of pairs to load
            
        Returns:
            List of (image_path, text_description) tuples
        """
        charts_dir, labels_dir = self.find_data_directories()
        
        if not charts_dir or not labels_dir:
            logger.warning("No training data directories found")
            return []
        
        logger.info(f"Loading training pairs from {charts_dir} and {labels_dir}")
        
        training_pairs = []
        
        # Get all image files
        image_files = list(charts_dir.glob("*.png"))
        
        if limit:
            image_files = image_files[:limit]
        
        for image_path in image_files:
            # Find corresponding label file
            base_name = image_path.stem
            
            # Try different label file patterns
            label_patterns = [
                labels_dir / f"{base_name}.txt",
                labels_dir / f"{base_name}.json",
                labels_dir / f"{base_name}_label.txt",
                labels_dir / f"{base_name}_gpt.txt"
            ]
            
            label_path = None
            for pattern in label_patterns:
                if pattern.exists():
                    label_path = pattern
                    break
            
            if label_path:
                text_description = self._load_text_description(label_path)
                if text_description:
                    training_pairs.append((str(image_path), text_description))
        
        logger.info(f"Loaded {len(training_pairs)} training pairs")
        return training_pairs
    
    def _load_text_description(self, label_path: Path) -> Optional[str]:
        """Load text description from label file"""
        try:
            if label_path.suffix == '.json':
                with open(label_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract description from various JSON formats
                if isinstance(data, dict):
                    description = (data.get('description') or 
                                 data.get('label') or 
                                 data.get('pattern') or
                                 data.get('phase'))
                    
                    if isinstance(description, str):
                        return description
                    
                    # Try to construct from components
                    phase = data.get('phase', '')
                    setup = data.get('setup', '')
                    confidence = data.get('confidence', '')
                    
                    if phase or setup:
                        parts = [p for p in [phase, setup, confidence] if p]
                        return ' | '.join(parts)
                
                return str(data) if data else None
            
            else:  # .txt file
                with open(label_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    return content if content else None
                    
        except Exception as e:
            logger.error(f"Error loading label from {label_path}: {e}")
            return None
    
    def load_recent_pairs(self, hours: int = 24, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Load recent training pairs from last N hours
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of pairs
            
        Returns:
            List of recent (image_path, text_description) pairs
        """
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        all_pairs = self.load_training_pairs()
        
        recent_pairs = []
        
        for image_path, text_desc in all_pairs:
            try:
                # Get file modification time
                file_time = datetime.fromtimestamp(Path(image_path).stat().st_mtime)
                
                if file_time >= cutoff_time:
                    recent_pairs.append((image_path, text_desc))
                    
            except Exception:
                # If we can't get file time, include it anyway
                recent_pairs.append((image_path, text_desc))
        
        # Sort by modification time (newest first)
        recent_pairs.sort(key=lambda x: Path(x[0]).stat().st_mtime, reverse=True)
        
        if limit:
            recent_pairs = recent_pairs[:limit]
        
        logger.info(f"Found {len(recent_pairs)} recent training pairs from last {hours} hours")
        return recent_pairs
    
    def prepare_batch_for_training(self, pairs: List[Tuple[str, str]], clip_model) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Prepare batch of image-text pairs for CLIP training
        
        Args:
            pairs: List of (image_path, text) pairs
            clip_model: CLIP model with preprocess function
            
        Returns:
            Tuple of (processed_images, text_descriptions)
        """
        processed_images = []
        text_descriptions = []
        
        for image_path, text_desc in pairs:
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                processed_image = clip_model.preprocess(image)
                
                processed_images.append(processed_image)
                text_descriptions.append(text_desc)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        return processed_images, text_descriptions
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about available training data"""
        charts_dir, labels_dir = self.find_data_directories()
        
        if not charts_dir or not labels_dir:
            return {"error": "No data directories found"}
        
        image_count = len(list(charts_dir.glob("*.png")))
        label_count = len(list(labels_dir.glob("*.txt"))) + len(list(labels_dir.glob("*.json")))
        
        # Count matching pairs
        pairs = self.load_training_pairs()
        
        # Get recent data stats
        recent_pairs_24h = self.load_recent_pairs(hours=24)
        recent_pairs_6h = self.load_recent_pairs(hours=6)
        
        return {
            "charts_directory": str(charts_dir),
            "labels_directory": str(labels_dir),
            "total_images": image_count,
            "total_labels": label_count,
            "valid_pairs": len(pairs),
            "recent_pairs_24h": len(recent_pairs_24h),
            "recent_pairs_6h": len(recent_pairs_6h),
            "data_available": len(pairs) > 0
        }


def main():
    """Test dataset loader functionality"""
    print("Testing CLIP Dataset Loader")
    print("=" * 40)
    
    loader = CLIPDatasetLoader()
    
    # Get dataset statistics
    stats = loader.get_dataset_stats()
    
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    if stats.get("data_available"):
        # Load some training pairs
        pairs = loader.load_training_pairs(limit=5)
        
        print(f"\nSample training pairs:")
        for i, (image_path, text_desc) in enumerate(pairs[:3], 1):
            print(f"   {i}. {Path(image_path).name}")
            print(f"      Text: {text_desc}")
    else:
        print("\nNo training data available")

if __name__ == "__main__":
    main()