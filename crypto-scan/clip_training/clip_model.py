"""
CLIP Model Implementation for Real-time Chart Learning
Implementacja modelu CLIP do uczenia się na wykresach w czasie rzeczywistym
"""

import torch
import torch.nn as nn
import clip
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class CLIPChartModel:
    """CLIP model for chart pattern recognition with online learning"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize CLIP model
        
        Args:
            model_name: CLIP model variant
            device: Device to use (auto-detected if None)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.model = None
        self.preprocess = None
        self.optimizer = None
        self.loss_fn = None
        
        # Training state
        self.training_step = 0
        self.is_initialized = False
        
        # Model save path
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.model_path = self.models_dir / "clip_model_latest.pt"
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CLIP model and training components"""
        try:
            logger.info(f"Initializing CLIP model: {self.model_name}")
            
            # Load CLIP model
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            
            # Set model to training mode
            self.model.train()
            
            # Initialize optimizer for fine-tuning
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-5,  # Low learning rate for fine-tuning
                weight_decay=0.01
            )
            
            # Contrastive loss (CLIP's native loss)
            self.loss_fn = nn.CrossEntropyLoss()
            
            # Try to load existing model if available
            if self.model_path.exists():
                self._load_model()
            
            self.is_initialized = True
            logger.info(f"CLIP model initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            self.is_initialized = False
    
    def _load_model(self):
        """Load saved model state"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_step = checkpoint.get('training_step', 0)
            
            logger.info(f"Loaded CLIP model from {self.model_path} (step {self.training_step})")
            
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
    
    def save_model(self):
        """Save current model state"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_step': self.training_step,
                'model_name': self.model_name,
                'device': self.device
            }
            
            torch.save(checkpoint, self.model_path)
            logger.info(f"Saved CLIP model to {self.model_path} (step {self.training_step})")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def train_on_batch(self, images: List[torch.Tensor], texts: List[str]) -> float:
        """
        Train model on a batch of image-text pairs
        
        Args:
            images: List of preprocessed image tensors
            texts: List of text descriptions
            
        Returns:
            Training loss
        """
        if not self.is_initialized:
            return 0.0
        
        try:
            # Prepare batch
            image_batch = torch.stack(images).to(self.device)
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            
            # Forward pass
            logits_per_image, logits_per_text = self.model(image_batch, text_tokens)
            
            # Create labels for contrastive learning
            batch_size = image_batch.size(0)
            labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
            
            # Calculate contrastive loss
            loss_i2t = self.loss_fn(logits_per_image, labels)
            loss_t2i = self.loss_fn(logits_per_text, labels)
            loss = (loss_i2t + loss_t2i) / 2
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.training_step += 1
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Training batch failed: {e}")
            return 0.0
    
    def predict(self, image_path: str, candidate_texts: List[str]) -> Dict:
        """
        Predict best matching text for image
        
        Args:
            image_path: Path to image
            candidate_texts: List of candidate text descriptions
            
        Returns:
            Prediction results
        """
        if not self.is_initialized:
            return {"error": "Model not initialized"}
        
        try:
            from PIL import Image
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize texts
            text_tokens = clip.tokenize(candidate_texts, truncate=True).to(self.device)
            
            # Set model to eval mode for prediction
            self.model.eval()
            
            with torch.no_grad():
                # Get features
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)
                
                # Calculate similarities
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarities = (image_features @ text_features.T).cpu().numpy()[0]
                probabilities = torch.softmax(torch.tensor(similarities), dim=0).numpy()
            
            # Set back to training mode
            self.model.train()
            
            # Get best prediction
            best_idx = np.argmax(probabilities)
            
            return {
                "success": True,
                "predicted_text": candidate_texts[best_idx],
                "confidence": float(probabilities[best_idx]),
                "all_predictions": [
                    {"text": text, "confidence": float(prob)}
                    for text, prob in zip(candidate_texts, probabilities)
                ],
                "training_step": self.training_step
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "training_step": self.training_step,
            "is_initialized": self.is_initialized,
            "model_path": str(self.model_path),
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }


# Global model instance
_global_clip_model = None

def get_clip_model() -> CLIPChartModel:
    """Get global CLIP model instance"""
    global _global_clip_model
    if _global_clip_model is None:
        _global_clip_model = CLIPChartModel()
    return _global_clip_model

def main():
    """Test CLIP model functionality"""
    print("Testing CLIP Chart Model")
    print("=" * 40)
    
    model = CLIPChartModel()
    
    if model.is_initialized:
        print("✅ Model initialized successfully")
        
        info = model.get_model_info()
        print(f"   Model: {info['model_name']}")
        print(f"   Device: {info['device']}")
        print(f"   Training step: {info['training_step']}")
        print(f"   Parameters: {info['parameters']:,}")
        
    else:
        print("❌ Model initialization failed")

if __name__ == "__main__":
    main()