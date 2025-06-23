"""
Score Embedding Utility
Zamienia TJDE scoring na znormalizowane wektory embeddingów
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback implementation
    class MinMaxScaler:
        def __init__(self, feature_range=(0, 1)):
            self.feature_range = feature_range
            self.fitted = False
            
        def fit(self, X):
            self.fitted = True
            return self
            
        def transform(self, X):
            import numpy as np
            X = np.array(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            # Simple min-max normalization
            return np.clip((X - 0) / (1 - 0), 0, 1)
            
        def fit_transform(self, X):
            return self.fit(X).transform(X)
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ScoreEmbeddingGenerator:
    """Generator embeddingów z TJDE scoring features"""
    
    def __init__(self):
        """Initialize score embedding generator"""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False
        
        # Define core scoring features in expected order
        self.core_features = [
            'trend_strength',
            'pullback_quality', 
            'support_reaction',
            'liquidity_pattern_score',
            'psych_score',
            'htf_supportive_score',
            'market_phase_modifier'
        ]
        
        # Extended features that might be present
        self.extended_features = [
            'volume_behavior_score',
            'momentum_score',
            'volatility_score',
            'clip_modifier',
            'final_score'
        ]
        
        self.all_features = self.core_features + self.extended_features
        
        logger.info("Score embedding generator initialized")
    
    def extract_score_vector(self, score_dict: Dict[str, Any]) -> np.ndarray:
        """
        Extract scoring features into normalized vector
        
        Args:
            score_dict: Dictionary containing TJDE scoring results
            
        Returns:
            Normalized feature vector
        """
        try:
            # Extract core features
            feature_values = []
            
            # Get from score_breakdown if available, otherwise from main dict
            score_breakdown = score_dict.get('score_breakdown', score_dict)
            used_features = score_dict.get('used_features', score_dict)
            
            # Extract core features
            for feature in self.core_features:
                value = 0.0
                
                # Try different sources
                if feature in score_breakdown:
                    value = float(score_breakdown[feature])
                elif feature in used_features:
                    value = float(used_features[feature])
                elif feature in score_dict:
                    value = float(score_dict[feature])
                
                # Ensure reasonable bounds
                value = max(-1.0, min(2.0, value))
                feature_values.append(value)
            
            # Extract extended features if available
            for feature in self.extended_features:
                value = 0.0
                
                if feature in score_dict:
                    value = float(score_dict[feature])
                elif feature == 'volume_behavior_score':
                    # Derive from volume_behavior if present
                    volume_behavior = score_dict.get('volume_behavior', '')
                    if 'buying_volume_increase' in volume_behavior:
                        value = 0.8
                    elif 'selling_pressure' in volume_behavior:
                        value = 0.2
                    else:
                        value = 0.5
                elif feature == 'momentum_score':
                    # Derive from trend and pullback
                    trend = feature_values[0] if len(feature_values) > 0 else 0.5
                    pullback = feature_values[1] if len(feature_values) > 1 else 0.5
                    value = (trend + pullback) / 2
                elif feature == 'volatility_score':
                    # Derive from liquidity pattern score
                    value = feature_values[3] if len(feature_values) > 3 else 0.5
                
                # Ensure reasonable bounds
                value = max(-1.0, min(2.0, value))
                feature_values.append(value)
            
            # Convert to numpy array
            vector = np.array(feature_values, dtype=np.float32)
            
            # Normalize the vector
            normalized_vector = self._normalize_vector(vector)
            
            logger.debug(f"Extracted score vector with {len(normalized_vector)} features")
            return normalized_vector
            
        except Exception as e:
            logger.error(f"Error extracting score vector: {e}")
            # Return zero vector with correct dimension
            return np.zeros(len(self.all_features), dtype=np.float32)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to [0, 1] range"""
        try:
            # Reshape for scaler
            vector_reshaped = vector.reshape(1, -1)
            
            if not self.fitted:
                # For first time, fit with reasonable defaults
                # Create sample data with expected ranges
                sample_data = np.array([
                    [0.0] * len(vector),  # Min values
                    [1.0] * len(vector),  # Max values  
                    vector.flatten()      # Current values
                ])
                
                self.scaler.fit(sample_data)
                self.fitted = True
            
            # Transform the vector
            normalized = self.scaler.transform(vector_reshaped)
            return normalized.flatten().astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error normalizing vector: {e}")
            # Return clipped vector as fallback
            return np.clip(vector, 0, 1).astype(np.float32)
    
    def embed_score_vector(self, score_dict: Dict[str, Any]) -> np.ndarray:
        """
        Main function to create embedding from score dictionary
        
        Args:
            score_dict: TJDE scoring results
            
        Returns:
            Normalized embedding vector
        """
        return self.extract_score_vector(score_dict)
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        return self.all_features.copy()
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return len(self.all_features)
    
    def save_scaler(self, path: str):
        """Save fitted scaler for consistency"""
        try:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Saved scaler to {path}")
        except Exception as e:
            logger.error(f"Error saving scaler: {e}")
    
    def load_scaler(self, path: str):
        """Load fitted scaler"""
        try:
            import pickle
            with open(path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.fitted = True
            logger.info(f"Loaded scaler from {path}")
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")


# Global instance
_global_score_embedder = None

def get_score_embedder() -> ScoreEmbeddingGenerator:
    """Get global score embedding generator instance"""
    global _global_score_embedder
    if _global_score_embedder is None:
        _global_score_embedder = ScoreEmbeddingGenerator()
    return _global_score_embedder

def embed_score_vector(score_dict: Dict[str, Any]) -> np.ndarray:
    """
    Convenience function to embed score dictionary
    
    Args:
        score_dict: TJDE scoring results
        
    Returns:
        Normalized embedding vector
    """
    embedder = get_score_embedder()
    return embedder.embed_score_vector(score_dict)

def main():
    """Test score embedding functionality"""
    print("Testing Score Embedding Generator")
    print("=" * 40)
    
    embedder = ScoreEmbeddingGenerator()
    
    # Test with sample TJDE results
    test_scores = [
        {
            "score_breakdown": {
                "trend_strength": 0.75,
                "pullback_quality": 0.68,
                "support_reaction": 0.72,
                "liquidity_pattern_score": 0.65,
                "psych_score": 0.80,
                "htf_supportive_score": 0.55,
                "market_phase_modifier": 0.05
            },
            "final_score": 0.685,
            "clip_modifier": 0.08,
            "volume_behavior": "buying_volume_increase"
        },
        {
            "trend_strength": 0.45,
            "pullback_quality": 0.40,
            "support_reaction": 0.35,
            "liquidity_pattern_score": 0.30,
            "psych_score": 0.25,
            "htf_supportive_score": 0.20,
            "market_phase_modifier": -0.05,
            "final_score": 0.281,
            "clip_modifier": -0.08
        }
    ]
    
    print(f"Feature names: {embedder.get_feature_names()}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    
    print(f"\nTesting score embedding generation:")
    
    for i, score_dict in enumerate(test_scores, 1):
        embedding = embedder.embed_score_vector(score_dict)
        
        print(f"\n   Test {i}:")
        print(f"   Input score: {score_dict.get('final_score', 'N/A')}")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        print(f"   Sample values: {embedding[:5]}")
    
    print(f"\n✅ Score embedding test completed")

if __name__ == "__main__":
    main()