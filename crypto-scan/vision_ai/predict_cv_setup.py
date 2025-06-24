#!/usr/bin/env python3
"""
Computer Vision Setup Prediction
Predicts market setup and phase from chart images using trained embeddings
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class CVSetupPredictor:
    """Predicts setup type and market phase from chart images"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        # Directory paths
        self.vision_dir = Path("data/vision_ai")
        self.embeddings_dir = self.vision_dir / "embeddings"
        self.predictions_dir = self.vision_dir / "predictions"
        
        # Create predictions directory
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        if CLIP_AVAILABLE:
            self.load_model()
        
        # Load training embeddings
        self.training_embeddings = self.load_training_embeddings()
    
    def load_model(self):
        """Load CLIP model for predictions"""
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=True)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[CV PREDICTOR] Loaded model: {self.model_name}")
            return True
            
        except Exception as e:
            print(f"[CV PREDICTOR] Failed to load model: {e}")
            return False
    
    def load_training_embeddings(self) -> List[Dict]:
        """Load training embeddings for comparison"""
        try:
            embeddings = []
            
            # Find latest embeddings file
            embedding_files = list(self.embeddings_dir.glob("embeddings_*.jsonl"))
            
            if not embedding_files:
                print("[CV PREDICTOR] No training embeddings found")
                return []
            
            latest_file = max(embedding_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                for line in f:
                    if line.strip():
                        embeddings.append(json.loads(line))
            
            print(f"[CV PREDICTOR] Loaded {len(embeddings)} training embeddings")
            return embeddings
            
        except Exception as e:
            print(f"[CV PREDICTOR] Failed to load training embeddings: {e}")
            return []
    
    def encode_chart_image(self, image_path: str) -> Optional[np.ndarray]:
        """Encode chart image to embedding vector"""
        if not self.model or not self.processor:
            return None
        
        try:
            import torch
            
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            print(f"[CV PREDICTOR] Image encoding failed: {e}")
            return None
    
    def predict_setup(self, image_path: str, symbol: str = None) -> Dict:
        """
        Predict setup type and market phase from chart image
        
        Args:
            image_path: Path to chart image
            symbol: Trading symbol (optional)
            
        Returns:
            Prediction results with setup, phase, and confidence
        """
        try:
            if not os.path.exists(image_path):
                return {"error": "Image file not found"}
            
            if not self.training_embeddings:
                return {"error": "No training data available"}
            
            # Encode chart image
            image_embedding = self.encode_chart_image(image_path)
            
            if image_embedding is None:
                return {"error": "Failed to encode image"}
            
            # Calculate similarities with training samples
            similarities = []
            
            for train_sample in self.training_embeddings:
                train_embedding = np.array(train_sample["image_embedding"])
                
                # Cosine similarity
                similarity = np.dot(image_embedding, train_embedding) / (
                    np.linalg.norm(image_embedding) * np.linalg.norm(train_embedding)
                )
                
                similarities.append({
                    "similarity": float(similarity),
                    "setup_type": train_sample["setup_type"],
                    "phase_type": train_sample["phase_type"],
                    "confidence": train_sample.get("confidence", 0.0),
                    "symbol": train_sample.get("symbol", "unknown")
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Get top matches
            top_matches = similarities[:5]
            
            # Predict setup type (most common among top matches)
            setup_votes = {}
            phase_votes = {}
            
            for match in top_matches:
                setup = match["setup_type"]
                phase = match["phase_type"]
                weight = match["similarity"]
                
                setup_votes[setup] = setup_votes.get(setup, 0) + weight
                phase_votes[phase] = phase_votes.get(phase, 0) + weight
            
            # Determine best predictions
            best_setup = max(setup_votes.items(), key=lambda x: x[1])
            best_phase = max(phase_votes.items(), key=lambda x: x[1])
            
            # Calculate confidence based on similarity and consensus
            avg_top_similarity = np.mean([m["similarity"] for m in top_matches])
            setup_consensus = best_setup[1] / sum(setup_votes.values())
            confidence = float(avg_top_similarity * setup_consensus)
            
            prediction = {
                "setup": best_setup[0],
                "phase": best_phase[0],
                "confidence": round(confidence, 3),
                "avg_similarity": round(float(avg_top_similarity), 3),
                "setup_consensus": round(float(setup_consensus), 3),
                "top_matches": top_matches[:3],  # Include top 3 for debugging
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path
            }
            
            # Save prediction
            if symbol:
                self.save_prediction(symbol, prediction)
            
            print(f"[CV PREDICTOR] Predicted: {best_setup[0]} / {best_phase[0]} (conf: {confidence:.3f})")
            return prediction
            
        except Exception as e:
            print(f"[CV PREDICTOR] Prediction failed: {e}")
            return {"error": str(e)}
    
    def save_prediction(self, symbol: str, prediction: Dict) -> str:
        """Save prediction to predictions directory"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp}.json"
            
            prediction_file = self.predictions_dir / filename
            
            with open(prediction_file, 'w') as f:
                json.dump(prediction, f, indent=2)
            
            print(f"[CV PREDICTOR] Saved prediction: {filename}")
            return str(prediction_file)
            
        except Exception as e:
            print(f"[CV PREDICTOR] Failed to save prediction: {e}")
            return ""
    
    def load_cv_prediction(self, symbol: str) -> Optional[Dict]:
        """Load latest CV prediction for symbol"""
        try:
            # Find latest prediction file for symbol
            prediction_files = list(self.predictions_dir.glob(f"{symbol}_*.json"))
            
            if not prediction_files:
                return None
            
            latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                prediction = json.load(f)
            
            return prediction
            
        except Exception as e:
            print(f"[CV PREDICTOR] Failed to load prediction for {symbol}: {e}")
            return None
    
    def batch_predict(self, image_directory: str) -> Dict:
        """Predict setups for all images in directory"""
        try:
            image_dir = Path(image_directory)
            image_files = list(image_dir.glob("*.png"))
            
            if not image_files:
                return {"error": "No images found"}
            
            results = {}
            
            for image_file in image_files:
                # Extract symbol from filename
                symbol = image_file.name.split('_')[0]
                
                prediction = self.predict_setup(str(image_file), symbol)
                results[image_file.name] = prediction
            
            print(f"[CV PREDICTOR] Batch prediction completed: {len(results)} images")
            return {"predictions": results, "total": len(results)}
            
        except Exception as e:
            print(f"[CV PREDICTOR] Batch prediction failed: {e}")
            return {"error": str(e)}
    
    def get_prediction_stats(self) -> Dict:
        """Get statistics about predictions"""
        try:
            prediction_files = list(self.predictions_dir.glob("*.json"))
            
            if not prediction_files:
                return {"total_predictions": 0}
            
            # Analyze predictions
            setups = {}
            phases = {}
            symbols = {}
            
            for pred_file in prediction_files:
                try:
                    with open(pred_file, 'r') as f:
                        pred = json.load(f)
                    
                    setup = pred.get("setup", "unknown")
                    phase = pred.get("phase", "unknown")
                    symbol = pred_file.name.split('_')[0]
                    
                    setups[setup] = setups.get(setup, 0) + 1
                    phases[phase] = phases.get(phase, 0) + 1
                    symbols[symbol] = symbols.get(symbol, 0) + 1
                    
                except:
                    continue
            
            return {
                "total_predictions": len(prediction_files),
                "setup_distribution": setups,
                "phase_distribution": phases,
                "symbol_distribution": symbols,
                "predictions_dir": str(self.predictions_dir)
            }
            
        except Exception as e:
            return {"error": str(e)}


# Global predictor instance
cv_predictor = CVSetupPredictor()


def load_cv_prediction(symbol: str) -> Optional[Dict]:
    """
    Load latest CV prediction for symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Prediction data or None
    """
    return cv_predictor.load_cv_prediction(symbol)


def predict_setup_from_chart(image_path: str, symbol: str = None) -> Dict:
    """
    Predict setup from chart image
    
    Args:
        image_path: Path to chart image
        symbol: Trading symbol
        
    Returns:
        Prediction results
    """
    return cv_predictor.predict_setup(image_path, symbol)


def main():
    """Test CV setup prediction"""
    print("🔍 Computer Vision Setup Prediction")
    print("=" * 40)
    
    if not CLIP_AVAILABLE:
        print("❌ CLIP not available. Install required packages.")
        return
    
    # Check if model loaded
    if not cv_predictor.model:
        print("❌ Model not loaded")
        return
    
    # Check training data
    if not cv_predictor.training_embeddings:
        print("❌ No training embeddings available. Run train_cv_model.py first.")
        return
    
    print(f"✅ Model loaded: {cv_predictor.model_name}")
    print(f"✅ Training samples: {len(cv_predictor.training_embeddings)}")
    
    # Test prediction on available charts
    charts_dir = "charts"
    if Path(charts_dir).exists():
        results = cv_predictor.batch_predict(charts_dir)
        
        if "predictions" in results:
            print(f"\n🎯 Prediction Results:")
            for filename, pred in list(results["predictions"].items())[:3]:
                if "error" not in pred:
                    print(f"  {filename}: {pred['setup']} / {pred['phase']} ({pred['confidence']:.3f})")
    
    # Show statistics
    stats = cv_predictor.get_prediction_stats()
    print(f"\n📊 Prediction Statistics:")
    print(f"  Total predictions: {stats.get('total_predictions', 0)}")
    
    setup_dist = stats.get('setup_distribution', {})
    if setup_dist:
        print(f"  Setup types: {list(setup_dist.keys())}")


if __name__ == "__main__":
    main()