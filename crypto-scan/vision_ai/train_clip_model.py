"""
CLIP Model Training for Chart Pattern Recognition
Trains CLIP embeddings using chart images and GPT-generated labels
"""

import os
import torch
import clip
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPTrainer:
    """CLIP model trainer for crypto chart pattern recognition"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize CLIP trainer
        
        Args:
            model_name: CLIP model variant to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            logger.info(f"Loaded CLIP model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
        
        # Create directories
        self.data_dir = Path("data")
        self.training_dir = self.data_dir / "training"
        self.charts_dir = self.training_dir / "charts"
        self.labels_dir = self.training_dir / "labels"
        self.clip_dir = self.data_dir / "clip"
        
        for dir_path in [self.training_dir, self.charts_dir, self.labels_dir, self.clip_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self) -> Tuple[List[torch.Tensor], List[str], List[str]]:
        """
        Load chart images and corresponding text labels
        
        Returns:
            Tuple of (processed_images, text_labels, filenames)
        """
        images, texts, filenames = [], [], []
        
        # Get all PNG files in charts directory
        chart_files = list(self.charts_dir.glob("*.png"))
        
        if not chart_files:
            logger.warning("No chart files found in training directory")
            return images, texts, filenames
        
        logger.info(f"Found {len(chart_files)} chart files")
        
        for chart_file in tqdm(chart_files, desc="Loading training data"):
            # Construct corresponding label file path
            base_name = chart_file.stem
            label_file = self.labels_dir / f"{base_name}.txt"
            
            if not label_file.exists():
                logger.warning(f"No label file found for {chart_file.name}")
                continue
            
            try:
                # Load and preprocess image
                image = Image.open(chart_file).convert('RGB')
                processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
                
                # Load text label
                with open(label_file, 'r', encoding='utf-8') as f:
                    text_label = f.read().strip()
                
                if text_label:
                    images.append(processed_image)
                    texts.append(text_label)
                    filenames.append(base_name)
                
            except Exception as e:
                logger.error(f"Error processing {chart_file.name}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(images)} training pairs")
        return images, texts, filenames
    
    def train_clip_embeddings(self, save_metadata: bool = True) -> Dict:
        """
        Generate CLIP embeddings for training data
        
        Args:
            save_metadata: Whether to save training metadata
            
        Returns:
            Training results summary
        """
        logger.info("Starting CLIP embedding generation...")
        
        # Load training data
        images, texts, filenames = self.load_training_data()
        
        if not images:
            logger.error("No training data found")
            return {"success": False, "error": "No training data available"}
        
        try:
            # Combine images into single tensor
            image_input = torch.cat(images, dim=0)
            
            # Tokenize texts
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                logger.info("Generating image embeddings...")
                image_features = self.model.encode_image(image_input)
                
                logger.info("Generating text embeddings...")
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Save embeddings
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            embedding_data = {
                "image_embeddings": image_features.cpu(),
                "text_embeddings": text_features.cpu(),
                "text_labels": texts,
                "filenames": filenames,
                "timestamp": timestamp,
                "model_name": "ViT-B/32",
                "num_samples": len(images)
            }
            
            torch.save(embedding_data, self.clip_dir / "clip_embeddings.pt")
            logger.info(f"Saved embeddings to {self.clip_dir / 'clip_embeddings.pt'}")
            
            # Save metadata if requested
            if save_metadata:
                metadata = {
                    "training_timestamp": timestamp,
                    "num_samples": len(images),
                    "model_name": "ViT-B/32",
                    "device": self.device,
                    "text_labels": texts,
                    "filenames": filenames,
                    "embedding_shapes": {
                        "image": list(image_features.shape),
                        "text": list(text_features.shape)
                    }
                }
                
                with open(self.clip_dir / "training_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info("Saved training metadata")
            
            # Calculate similarity statistics
            similarities = torch.cosine_similarity(image_features, text_features, dim=1)
            avg_similarity = similarities.mean().item()
            
            results = {
                "success": True,
                "num_samples": len(images),
                "avg_similarity": avg_similarity,
                "timestamp": timestamp,
                "embedding_file": str(self.clip_dir / "clip_embeddings.pt")
            }
            
            logger.info(f"Training completed: {len(images)} samples, avg similarity: {avg_similarity:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_embeddings(self) -> Dict:
        """
        Validate generated embeddings by running similarity tests
        
        Returns:
            Validation results
        """
        try:
            embedding_file = self.clip_dir / "clip_embeddings.pt"
            
            if not embedding_file.exists():
                return {"success": False, "error": "No embeddings file found"}
            
            # Load embeddings
            data = torch.load(embedding_file, map_location=self.device)
            image_embeddings = data["image_embeddings"].to(self.device)
            text_embeddings = data["text_embeddings"].to(self.device)
            text_labels = data["text_labels"]
            
            # Calculate similarity matrix
            similarity_matrix = torch.cosine_similarity(
                image_embeddings.unsqueeze(1), 
                text_embeddings.unsqueeze(0), 
                dim=2
            )
            
            # Find best matches for each image
            best_matches = torch.argmax(similarity_matrix, dim=1)
            correct_matches = (best_matches == torch.arange(len(best_matches)).to(self.device)).sum().item()
            
            accuracy = correct_matches / len(best_matches)
            avg_similarity = torch.diagonal(similarity_matrix).mean().item()
            
            validation_results = {
                "success": True,
                "accuracy": accuracy,
                "avg_similarity": avg_similarity,
                "total_samples": len(text_labels),
                "correct_matches": correct_matches
            }
            
            logger.info(f"Validation: {accuracy:.1%} accuracy, {avg_similarity:.4f} avg similarity")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"success": False, "error": str(e)}


def main():
    """Main training function"""
    print("🎯 CLIP Model Training for Chart Pattern Recognition")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = CLIPTrainer()
        
        # Check for training data
        charts_count = len(list(trainer.charts_dir.glob("*.png")))
        labels_count = len(list(trainer.labels_dir.glob("*.txt")))
        
        print(f"📊 Found {charts_count} chart files and {labels_count} label files")
        
        if charts_count == 0 or labels_count == 0:
            print("❌ No training data found. Please ensure:")
            print(f"   - Chart images are in: {trainer.charts_dir}")
            print(f"   - Label files are in: {trainer.labels_dir}")
            return
        
        # Train embeddings
        print("\n🔄 Generating CLIP embeddings...")
        results = trainer.train_clip_embeddings()
        
        if results["success"]:
            print(f"✅ Training completed successfully!")
            print(f"   - Samples processed: {results['num_samples']}")
            print(f"   - Average similarity: {results['avg_similarity']:.4f}")
            print(f"   - Embeddings saved to: {results['embedding_file']}")
            
            # Run validation
            print("\n🔍 Validating embeddings...")
            validation = trainer.validate_embeddings()
            
            if validation["success"]:
                print(f"✅ Validation completed!")
                print(f"   - Accuracy: {validation['accuracy']:.1%}")
                print(f"   - Correct matches: {validation['correct_matches']}/{validation['total_samples']}")
            else:
                print(f"❌ Validation failed: {validation['error']}")
        else:
            print(f"❌ Training failed: {results['error']}")
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()