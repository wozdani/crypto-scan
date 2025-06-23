"""
CLIP Similarity Prediction for Chart Pattern Recognition
Uses trained CLIP embeddings to predict chart patterns and market phases
"""

import torch
import clip
from PIL import Image
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CLIPPredictor:
    """CLIP-based chart pattern predictor"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize CLIP predictor
        
        Args:
            model_name: CLIP model variant to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
        
        self.clip_dir = Path("data/clip")
        self.embeddings_file = self.clip_dir / "clip_embeddings.pt"
        
        # Load training embeddings if available
        self.training_embeddings = None
        self.training_labels = None
        self._load_training_embeddings()
        
        # Predefined market phase labels
        self.market_phases = [
            "trending-up",
            "trending-down", 
            "consolidation",
            "pullback-in-trend",
            "breakout-continuation",
            "fakeout",
            "accumulation",
            "distribution",
            "exhaustion",
            "range-bound"
        ]
        
        self.setup_patterns = [
            "pullback-in-trend",
            "breakout-continuation",
            "support-bounce",
            "resistance-rejection",
            "flag-pattern",
            "triangle-breakout",
            "volume-backed",
            "weak-bounce",
            "failed-breakout",
            "consolidation-range"
        ]
    
    def _load_training_embeddings(self):
        """Load pre-computed training embeddings"""
        try:
            if self.embeddings_file.exists():
                data = torch.load(self.embeddings_file, map_location=self.device)
                self.training_embeddings = {
                    "image": data["image_embeddings"].to(self.device),
                    "text": data["text_embeddings"].to(self.device)
                }
                self.training_labels = data["text_labels"]
                logger.info(f"Loaded {len(self.training_labels)} training embeddings")
            else:
                logger.warning("No training embeddings found")
        except Exception as e:
            logger.error(f"Error loading training embeddings: {e}")
    
    def predict_from_image(self, image_path: str, candidate_labels: Optional[List[str]] = None) -> Dict:
        """
        Predict chart pattern from image using CLIP
        
        Args:
            image_path: Path to chart image
            candidate_labels: List of candidate labels (uses defaults if None)
            
        Returns:
            Prediction results with probabilities
        """
        try:
            if not Path(image_path).exists():
                return {"error": "Image file not found"}
            
            # Use default labels if none provided
            if candidate_labels is None:
                candidate_labels = self.market_phases + self.setup_patterns
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text labels
            text_tokens = clip.tokenize(candidate_labels, truncate=True).to(self.device)
            
            # Generate predictions
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(image_input, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Create results
            predictions = []
            for label, prob in zip(candidate_labels, probs):
                predictions.append({
                    "label": label,
                    "probability": float(prob),
                    "confidence": float(prob)
                })
            
            # Sort by probability
            predictions.sort(key=lambda x: x["probability"], reverse=True)
            
            # Extract top prediction
            top_prediction = predictions[0]
            
            # Determine phase and setup
            phase = "unknown"
            setup = "unknown"
            
            for pred in predictions[:3]:  # Check top 3 predictions
                if pred["label"] in self.market_phases and phase == "unknown":
                    phase = pred["label"]
                elif pred["label"] in self.setup_patterns and setup == "unknown":
                    setup = pred["label"]
            
            results = {
                "success": True,
                "image_path": image_path,
                "top_prediction": top_prediction,
                "phase": phase,
                "setup": setup,
                "confidence": top_prediction["confidence"],
                "all_predictions": predictions[:10],  # Top 10 only
                "phase_description": self._get_phase_description(phase, setup)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"success": False, "error": str(e)}
    
    def predict_similarity_with_training(self, image_path: str, top_k: int = 5) -> Dict:
        """
        Find similar training examples using embedding similarity
        
        Args:
            image_path: Path to chart image
            top_k: Number of similar examples to return
            
        Returns:
            Similarity results with training examples
        """
        try:
            if self.training_embeddings is None:
                return {"error": "No training embeddings available"}
            
            if not Path(image_path).exists():
                return {"error": "Image file not found"}
            
            # Process input image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate image embedding
            with torch.no_grad():
                query_embedding = self.model.encode_image(image_input)
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
            
            # Calculate similarities with training images
            training_image_embeddings = self.training_embeddings["image"]
            similarities = torch.cosine_similarity(
                query_embedding, 
                training_image_embeddings, 
                dim=1
            )
            
            # Get top-k most similar
            top_similarities, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
            
            # Collect results
            similar_examples = []
            for sim, idx in zip(top_similarities, top_indices):
                similar_examples.append({
                    "similarity": float(sim),
                    "label": self.training_labels[idx],
                    "index": int(idx)
                })
            
            # Predict phase based on similar examples
            predicted_phase = self._aggregate_similar_predictions(similar_examples)
            
            results = {
                "success": True,
                "image_path": image_path,
                "similar_examples": similar_examples,
                "predicted_phase": predicted_phase,
                "avg_similarity": float(top_similarities.mean()),
                "confidence": float(top_similarities[0])  # Use highest similarity as confidence
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity prediction error: {e}")
            return {"success": False, "error": str(e)}
    
    def _aggregate_similar_predictions(self, similar_examples: List[Dict]) -> Dict:
        """Aggregate predictions from similar training examples"""
        try:
            # Extract phases and setups from similar examples
            phases = {}
            setups = {}
            
            for example in similar_examples:
                label = example["label"]
                similarity = example["similarity"]
                
                # Parse label (format: "phase | setup | additional_info")
                parts = [part.strip() for part in label.split("|")]
                
                if len(parts) >= 1:
                    phase = parts[0]
                    if phase in phases:
                        phases[phase] += similarity
                    else:
                        phases[phase] = similarity
                
                if len(parts) >= 2:
                    setup = parts[1]
                    if setup in setups:
                        setups[setup] += similarity
                    else:
                        setups[setup] = similarity
            
            # Find most likely phase and setup
            best_phase = max(phases.items(), key=lambda x: x[1]) if phases else ("unknown", 0)
            best_setup = max(setups.items(), key=lambda x: x[1]) if setups else ("unknown", 0)
            
            return {
                "phase": best_phase[0],
                "phase_confidence": best_phase[1],
                "setup": best_setup[0],
                "setup_confidence": best_setup[1]
            }
            
        except Exception as e:
            logger.error(f"Error aggregating predictions: {e}")
            return {"phase": "unknown", "setup": "unknown"}
    
    def _get_phase_description(self, phase: str, setup: str) -> str:
        """Generate description for predicted phase and setup"""
        descriptions = {
            "trending-up": "Strong upward trend with momentum",
            "trending-down": "Strong downward trend with momentum", 
            "consolidation": "Sideways movement within range",
            "pullback-in-trend": "Temporary retracement in established trend",
            "breakout-continuation": "Continuation after level breakout",
            "fakeout": "False breakout or breakdown",
            "accumulation": "Base building phase",
            "distribution": "Top formation phase",
            "exhaustion": "Trend showing signs of weakness",
            "range-bound": "Trading within defined boundaries"
        }
        
        setup_descriptions = {
            "volume-backed": "with volume confirmation",
            "weak-bounce": "with weak reaction",
            "failed-breakout": "with failed momentum",
            "support-bounce": "bouncing from support level",
            "resistance-rejection": "rejected at resistance level"
        }
        
        base_desc = descriptions.get(phase, "Market pattern detected")
        setup_desc = setup_descriptions.get(setup, "")
        
        return f"{base_desc} {setup_desc}".strip()


def predict_chart_pattern(image_path: str, method: str = "zero_shot") -> Dict:
    """
    Convenience function for chart pattern prediction
    
    Args:
        image_path: Path to chart image
        method: Prediction method ("zero_shot" or "similarity")
        
    Returns:
        Prediction results
    """
    predictor = CLIPPredictor()
    
    if method == "similarity" and predictor.training_embeddings is not None:
        return predictor.predict_similarity_with_training(image_path)
    else:
        return predictor.predict_from_image(image_path)


def main():
    """Demo prediction functionality"""
    print("🎯 CLIP Chart Pattern Prediction Demo")
    print("=" * 50)
    
    try:
        predictor = CLIPPredictor()
        
        # Check for training data
        if predictor.training_embeddings is not None:
            print(f"✅ Loaded {len(predictor.training_labels)} training embeddings")
        else:
            print("⚠️ No training embeddings found - using zero-shot prediction only")
        
        # Demo with example labels
        demo_labels = [
            "pullback-in-trend",
            "breakout-continuation", 
            "fakeout",
            "range-bound",
            "accumulation",
            "exhaustion",
            "trending-up",
            "consolidation"
        ]
        
        print(f"\n🔍 Available prediction labels:")
        for i, label in enumerate(demo_labels, 1):
            print(f"   {i}. {label}")
        
        # Example prediction (would need actual chart image)
        example_chart = "data/training/charts/example.png"
        if Path(example_chart).exists():
            results = predictor.predict_from_image(example_chart, demo_labels)
            
            if results.get("success"):
                print(f"\n📊 Prediction Results for {example_chart}:")
                print(f"   Phase: {results['phase']}")
                print(f"   Setup: {results['setup']}")
                print(f"   Confidence: {results['confidence']:.4f}")
                print(f"   Description: {results['phase_description']}")
            else:
                print(f"❌ Prediction failed: {results.get('error')}")
        else:
            print(f"\n📝 Example chart not found at {example_chart}")
            print("   Place a chart image there to test predictions")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main()