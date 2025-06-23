"""
CLIP Model Wrapper for Vision-AI System
Enhanced CLIP model with embedding generation capabilities
"""

import torch
import logging
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class CLIPWrapper:
    """Wrapper for CLIP model with multiple backend support"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """Initialize CLIP model"""
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CLIP model with fallback options"""
        try:
            # Try transformers first
            from transformers import CLIPModel, CLIPProcessor
            
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.initialized = True
            
            logger.info(f"CLIP model initialized with transformers: {self.model_name}")
            
        except Exception as e:
            logger.warning(f"Transformers CLIP failed: {e}, trying fallback")
            
            try:
                # Fallback to openai/clip
                import clip
                
                self.model, self.preprocess = clip.load(self.model_name, device=self.device)
                self.use_fallback = True
                self.initialized = True
                
                logger.info(f"CLIP model initialized with fallback: {self.model_name}")
                
            except Exception as e2:
                logger.error(f"All CLIP backends failed: {e2}")
                self.initialized = False
    
    def get_image_embedding(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Get CLIP image embedding for integration with embedding pipeline
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding tensor or None if failed
        """
        if not self.initialized:
            logger.warning("CLIP model not initialized")
            return None
        
        try:
            from PIL import Image
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            if hasattr(self, 'use_fallback'):
                # Use fallback CLIP
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                return image_features.cpu()
            else:
                # Use transformers CLIP
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                return image_features.cpu()
                
        except Exception as e:
            logger.error(f"Error getting image embedding: {e}")
            return None
    
    def get_text_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Get CLIP text embedding
        
        Args:
            text: Text to encode
            
        Returns:
            Text embedding tensor or None if failed
        """
        if not self.initialized:
            logger.warning("CLIP model not initialized")
            return None
        
        try:
            if hasattr(self, 'use_fallback'):
                # Use fallback CLIP
                import clip
                text_tokens = clip.tokenize([text]).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                return text_features.cpu()
            else:
                # Use transformers CLIP
                inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)
                return text_features.cpu()
                
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "initialized": self.initialized,
            "using_fallback": hasattr(self, 'use_fallback'),
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }


# Global instance
_global_clip_model = None

def get_clip_model() -> CLIPWrapper:
    """Get global CLIP model instance"""
    global _global_clip_model
    if _global_clip_model is None:
        _global_clip_model = CLIPWrapper()
    return _global_clip_model

def get_clip_image_embedding(image_path: str) -> Optional[torch.Tensor]:
    """Get CLIP image embedding for specified image"""
    model = get_clip_model()
    return model.get_image_embedding(image_path)