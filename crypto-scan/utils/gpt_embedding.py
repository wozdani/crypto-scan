"""
GPT Text Embedding Utility
Tworzy embeddingi tekstowe z komentarzy GPT dla integracji z Vision-AI
"""

import os
import logging
import numpy as np
from typing import Optional, List
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

class GPTEmbeddingGenerator:
    """Generator embeddingów tekstowych używający OpenAI"""
    
    def __init__(self):
        """Initialize GPT embedding generator"""
        self.client = None
        self.initialized = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment")
                return
            
            self.client = OpenAI(api_key=api_key)
            self.initialized = True
            logger.info("GPT embedding generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPT embedding generator: {e}")
            self.initialized = False
    
    def get_text_embedding(self, text: str, model: str = "text-embedding-3-small") -> Optional[np.ndarray]:
        """
        Generate embedding for text using OpenAI
        
        Args:
            text: Text to embed
            model: OpenAI embedding model to use
            
        Returns:
            Numpy array embedding or None if failed
        """
        if not self.initialized:
            logger.warning("GPT embedding generator not initialized")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            # Clean and prepare text
            cleaned_text = text.strip().replace('\n', ' ').replace('\r', ' ')
            
            # Get embedding from OpenAI
            response = self.client.embeddings.create(
                input=cleaned_text,
                model=model
            )
            
            # Extract embedding vector
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            logger.debug(f"Generated embedding with shape {embedding.shape} for text: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None
    
    def get_batch_embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            model: OpenAI embedding model to use
            
        Returns:
            List of embeddings (or None for failed ones)
        """
        if not self.initialized:
            return [None] * len(texts)
        
        embeddings = []
        
        try:
            # Clean texts
            cleaned_texts = [text.strip().replace('\n', ' ').replace('\r', ' ') for text in texts if text and text.strip()]
            
            if not cleaned_texts:
                return [None] * len(texts)
            
            # Get batch embeddings
            response = self.client.embeddings.create(
                input=cleaned_texts,
                model=model
            )
            
            # Extract embeddings
            for i, data in enumerate(response.data):
                embedding = np.array(data.embedding, dtype=np.float32)
                embeddings.append(embedding)
            
            logger.info(f"Generated {len(embeddings)} batch embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    def get_embedding_dimension(self, model: str = "text-embedding-3-small") -> int:
        """Get embedding dimension for model"""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return model_dimensions.get(model, 1536)


# Global instance
_global_gpt_embedder = None

def get_gpt_embedder() -> GPTEmbeddingGenerator:
    """Get global GPT embedding generator instance"""
    global _global_gpt_embedder
    if _global_gpt_embedder is None:
        _global_gpt_embedder = GPTEmbeddingGenerator()
    return _global_gpt_embedder

def get_gpt_text_embedding(text: str) -> Optional[np.ndarray]:
    """
    Convenience function to get GPT text embedding
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector or None if failed
    """
    embedder = get_gpt_embedder()
    return embedder.get_text_embedding(text)

def main():
    """Test GPT embedding functionality"""
    print("Testing GPT Text Embedding")
    print("=" * 40)
    
    embedder = GPTEmbeddingGenerator()
    
    if embedder.initialized:
        print("✅ GPT embedder initialized")
        
        # Test embedding generation
        test_texts = [
            "pullback-in-trend | trending-up | volume-backed",
            "breakout-continuation | bullish momentum",
            "trend-reversal | exhaustion pattern"
        ]
        
        print(f"\nTesting embedding generation:")
        
        for i, text in enumerate(test_texts, 1):
            embedding = embedder.get_text_embedding(text)
            
            if embedding is not None:
                print(f"   {i}. Text: {text}")
                print(f"      Embedding shape: {embedding.shape}")
                print(f"      Sample values: {embedding[:5]}")
            else:
                print(f"   {i}. Failed to generate embedding for: {text}")
        
        # Test batch embeddings
        print(f"\nTesting batch embeddings:")
        batch_embeddings = embedder.get_batch_embeddings(test_texts)
        
        successful = sum(1 for emb in batch_embeddings if emb is not None)
        print(f"   Generated {successful}/{len(test_texts)} batch embeddings")
        
        # Test embedding dimension
        dimension = embedder.get_embedding_dimension()
        print(f"   Embedding dimension: {dimension}")
        
    else:
        print("❌ GPT embedder initialization failed")
        print("   Check OPENAI_API_KEY environment variable")

if __name__ == "__main__":
    main()