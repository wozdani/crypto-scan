#!/usr/bin/env python3
"""
Vision-AI Model Training with CLIP Embeddings
Trains a model to understand chart patterns using image-text embeddings
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import clip
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available. Install with: pip install clip-by-openai transformers")


class VisionAITrainer:
    """Trainer for Vision-AI model using CLIP embeddings"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        # Directory structure
        self.vision_dir = Path("data/vision_ai")
        self.embeddings_dir = self.vision_dir / "embeddings"
        self.predictions_dir = self.vision_dir / "predictions"
        
        # Create directories
        for directory in [self.vision_dir, self.embeddings_dir, self.predictions_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize model if available
        if CLIP_AVAILABLE:
            self.load_clip_model()
    
    def load_clip_model(self):
        """Load CLIP model and processor"""
        try:
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=True)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[VISION AI] Loaded CLIP model: {self.model_name}")
            return True
            
        except Exception as e:
            print(f"[VISION AI] Failed to load CLIP model: {e}")
            return False
    
    def generate_setup_description(self, labels: Dict, context: Dict) -> str:
        """Generate natural language description of setup"""
        setup_type = labels.get("setup_type", "unknown")
        phase_type = labels.get("phase_type", "unknown")
        
        # Context features
        trend_strength = context.get("trend_strength", 0.5)
        pullback_quality = context.get("pullback_quality", 0.5)
        
        # Generate descriptive text
        descriptions = {
            "breakout_with_pullback": f"Strong breakout setup with quality pullback in trending market, trend strength {trend_strength:.2f}",
            "trend_continuation": f"Trend continuation pattern with good momentum, pullback quality {pullback_quality:.2f}",
            "exhaustion": f"Market exhaustion pattern showing weakness after extended move",
            "fakeout": f"False breakout pattern - likely trap setup with weak follow-through",
            "range_accumulation": f"Sideways consolidation pattern within defined range boundaries",
            "support_retest": f"Price testing support level with potential bounce setup",
            "resistance_rejection": f"Price rejected at resistance with reversal potential"
        }
        
        base_desc = descriptions.get(setup_type, f"Market pattern showing {setup_type} characteristics")
        
        # Add phase context
        if phase_type != "unknown":
            phase_context = {
                "breakout-continuation": "in breakout continuation phase",
                "exhaustion-pullback": "in exhaustion pullback phase", 
                "consolidation-range": "in consolidation range phase",
                "trend-acceleration": "in trend acceleration phase"
            }
            
            phase_desc = phase_context.get(phase_type, f"in {phase_type} phase")
            base_desc += f" {phase_desc}"
        
        return base_desc
    
    def encode_image_text_pair(self, image_path: str, description: str) -> Tuple[np.ndarray, np.ndarray]:
        """Encode image-text pair using CLIP"""
        if not self.model or not self.processor:
            return None, None
        
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Process inputs
            inputs = self.processor(
                text=[description],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embedding = outputs.image_embeds.cpu().numpy()[0]
                text_embedding = outputs.text_embeds.cpu().numpy()[0]
            
            return image_embedding, text_embedding
            
        except Exception as e:
            print(f"[VISION AI] Encoding failed for {image_path}: {e}")
            return None, None
    
    def train_on_dataset(self, training_data_dir: str = "training_data") -> Dict:
        """Train model on collected dataset"""
        training_path = Path(training_data_dir)
        
        if not training_path.exists():
            print(f"[VISION AI] Training data not found: {training_data_dir}")
            return {"error": "No training data"}
        
        # Collect training samples
        charts_dir = training_path / "charts"
        labels_dir = training_path / "labels"
        meta_dir = training_path / "metadata"
        
        if not all([charts_dir.exists(), labels_dir.exists()]):
            print(f"[VISION AI] Required directories missing")
            return {"error": "Incomplete training data"}
        
        print(f"[VISION AI] Starting training on dataset...")
        
        # Process all samples
        embeddings_data = []
        similarities = []
        
        chart_files = list(charts_dir.glob("*.png"))
        processed_count = 0
        
        for chart_file in chart_files:
            # Find corresponding files
            base_name = chart_file.stem.replace("_chart", "")
            label_file = labels_dir / f"{base_name}_label.json"
            meta_file = meta_dir / f"{base_name}_meta.json" if meta_dir.exists() else None
            
            if not label_file.exists():
                continue
            
            try:
                # Load labels and metadata
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                
                metadata = {}
                if meta_file and meta_file.exists():
                    with open(meta_file, 'r') as f:
                        meta_data = json.load(f)
                        metadata = meta_data.get("context_features", {})
                
                # Generate description
                labels = label_data.get("labels", {})
                description = self.generate_setup_description(labels, metadata)
                
                # Encode image-text pair
                image_emb, text_emb = self.encode_image_text_pair(str(chart_file), description)
                
                if image_emb is not None and text_emb is not None:
                    # Calculate similarity
                    similarity = np.dot(image_emb, text_emb) / (np.linalg.norm(image_emb) * np.linalg.norm(text_emb))
                    similarities.append(similarity)
                    
                    # Store embedding data
                    embedding_record = {
                        "symbol": label_data.get("symbol", "unknown"),
                        "timestamp": label_data.get("timestamp", ""),
                        "chart_file": chart_file.name,
                        "description": description,
                        "setup_type": labels.get("setup_type", "unknown"),
                        "phase_type": labels.get("phase_type", "unknown"),
                        "confidence": labels.get("confidence", 0.0),
                        "similarity": float(similarity),
                        "image_embedding": image_emb.tolist(),
                        "text_embedding": text_emb.tolist(),
                        "context_features": metadata,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    embeddings_data.append(embedding_record)
                    processed_count += 1
                    
                    print(f"[VISION AI] Processed {chart_file.name}: {similarity:.3f} similarity")
                
            except Exception as e:
                print(f"[VISION AI] Error processing {chart_file.name}: {e}")
                continue
        
        # Save embeddings dataset
        if embeddings_data:
            embeddings_file = self.embeddings_dir / f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            
            with open(embeddings_file, 'w') as f:
                for record in embeddings_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Save training summary
            training_summary = {
                "training_completed": datetime.now().isoformat(),
                "samples_processed": processed_count,
                "average_similarity": float(np.mean(similarities)) if similarities else 0.0,
                "similarity_std": float(np.std(similarities)) if similarities else 0.0,
                "embeddings_file": str(embeddings_file),
                "setup_types": list(set(r["setup_type"] for r in embeddings_data)),
                "phase_types": list(set(r["phase_type"] for r in embeddings_data))
            }
            
            summary_file = self.vision_dir / "training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(training_summary, f, indent=2)
            
            print(f"[VISION AI] Training completed:")
            print(f"  Processed: {processed_count} samples")
            print(f"  Avg similarity: {training_summary['average_similarity']:.3f}")
            print(f"  Setup types: {len(training_summary['setup_types'])}")
            print(f"  Embeddings saved: {embeddings_file}")
            
            return training_summary
        else:
            print(f"[VISION AI] No valid samples processed")
            return {"error": "No valid training samples"}
    
    def save_prediction_embeddings(self, symbol: str, image_path: str, description: str) -> Optional[str]:
        """Save prediction embeddings for a symbol"""
        try:
            image_emb, text_emb = self.encode_image_text_pair(image_path, description)
            
            if image_emb is None or text_emb is None:
                return None
            
            # Calculate similarity with training embeddings (if available)
            similarity_scores = self.calculate_similarity_with_training(image_emb, text_emb)
            
            prediction_data = {
                "symbol": symbol,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "image_path": image_path,
                "description": description,
                "image_embedding": image_emb.tolist(),
                "text_embedding": text_emb.tolist(),
                "similarity_scores": similarity_scores,
                "created_at": datetime.now().isoformat()
            }
            
            # Save prediction
            pred_file = self.predictions_dir / f"{symbol}_{prediction_data['timestamp']}.json"
            with open(pred_file, 'w') as f:
                json.dump(prediction_data, f, indent=2)
            
            print(f"[VISION AI] Saved prediction embeddings: {pred_file.name}")
            return str(pred_file)
            
        except Exception as e:
            print(f"[VISION AI] Failed to save prediction embeddings: {e}")
            return None
    
    def calculate_similarity_with_training(self, image_emb: np.ndarray, text_emb: np.ndarray) -> Dict:
        """Calculate similarity with training embeddings"""
        try:
            # Load latest embeddings file
            embedding_files = list(self.embeddings_dir.glob("embeddings_*.jsonl"))
            
            if not embedding_files:
                return {"status": "no_training_data"}
            
            latest_file = max(embedding_files, key=lambda x: x.stat().st_mtime)
            
            setup_similarities = {}
            
            with open(latest_file, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        
                        # Calculate image similarity
                        train_img_emb = np.array(record["image_embedding"])
                        img_similarity = np.dot(image_emb, train_img_emb) / (
                            np.linalg.norm(image_emb) * np.linalg.norm(train_img_emb)
                        )
                        
                        setup_type = record["setup_type"]
                        if setup_type not in setup_similarities:
                            setup_similarities[setup_type] = []
                        
                        setup_similarities[setup_type].append(float(img_similarity))
            
            # Calculate average similarities per setup type
            avg_similarities = {}
            for setup_type, similarities in setup_similarities.items():
                avg_similarities[setup_type] = {
                    "avg_similarity": float(np.mean(similarities)),
                    "max_similarity": float(np.max(similarities)),
                    "sample_count": len(similarities)
                }
            
            return {
                "status": "calculated",
                "setup_similarities": avg_similarities,
                "training_file": str(latest_file)
            }
            
        except Exception as e:
            print(f"[VISION AI] Similarity calculation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_training_stats(self) -> Dict:
        """Get statistics about training data and embeddings"""
        try:
            stats = {
                "embeddings_files": len(list(self.embeddings_dir.glob("*.jsonl"))),
                "prediction_files": len(list(self.predictions_dir.glob("*.json"))),
                "model_loaded": self.model is not None,
                "device": str(self.device),
                "model_name": self.model_name
            }
            
            # Load latest training summary
            summary_file = self.vision_dir / "training_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    stats["last_training"] = summary
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main training function"""
    print("🧠 Vision-AI Model Training with CLIP Embeddings")
    print("=" * 50)
    
    if not CLIP_AVAILABLE:
        print("❌ CLIP not available. Install required packages first.")
        return
    
    # Initialize trainer
    trainer = VisionAITrainer()
    
    # Check if model loaded successfully
    if not trainer.model:
        print("❌ Failed to load CLIP model")
        return
    
    # Train on available dataset
    print("Starting training on available dataset...")
    results = trainer.train_on_dataset()
    
    if "error" in results:
        print(f"❌ Training failed: {results['error']}")
        print("Run the training data collection pipeline first.")
    else:
        print("✅ Training completed successfully!")
        
        # Show statistics
        stats = trainer.get_training_stats()
        print(f"\n📊 Training Statistics:")
        print(f"  Model: {stats.get('model_name', 'unknown')}")
        print(f"  Device: {stats.get('device', 'unknown')}")
        print(f"  Embeddings files: {stats.get('embeddings_files', 0)}")
        
        if "last_training" in stats:
            training_info = stats["last_training"]
            print(f"  Samples processed: {training_info.get('samples_processed', 0)}")
            print(f"  Average similarity: {training_info.get('average_similarity', 0):.3f}")
            print(f"  Setup types: {len(training_info.get('setup_types', []))}")


if __name__ == "__main__":
    main()