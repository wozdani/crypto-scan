"""
Cluster Prediction for Vision-AI System
Przewiduje klastery dla nowych embeddingów i integruje z TJDE
"""

import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

logger = logging.getLogger(__name__)

class ClusterPredictor:
    """Predyktor klastrów dla embeddingów"""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize cluster predictor"""
        self.models_dir = Path(models_dir)
        
        # Model components
        self.kmeans_model = None
        self.cluster_centers = None
        self.scaler = None
        self.reducer = None
        self.symbol_clusters = None
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load trained models and components"""
        try:
            # Load cluster centers (fastest for prediction)
            centers_path = self.models_dir / "embedding_cluster_centers.npy"
            if centers_path.exists():
                self.cluster_centers = np.load(centers_path)
                logger.info(f"Loaded cluster centers: {self.cluster_centers.shape}")
            
            # Load KMeans model
            kmeans_path = self.models_dir / "embedding_kmeans.pkl"
            if kmeans_path.exists():
                with open(kmeans_path, 'rb') as f:
                    self.kmeans_model = pickle.load(f)
                logger.info("Loaded KMeans model")
            
            # Load scaler
            scaler_path = self.models_dir / "embedding_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Loaded scaler")
            
            # Load dimensionality reducer
            for reducer_name in ["embedding_pca_reducer.pkl", "embedding_umap_reducer.pkl"]:
                reducer_path = self.models_dir / reducer_name
                if reducer_path.exists():
                    with open(reducer_path, 'rb') as f:
                        self.reducer = pickle.load(f)
                    logger.info(f"Loaded reducer: {reducer_name}")
                    break
            
            # Load symbol clusters
            symbol_clusters_path = self.models_dir / "symbol_clusters.json"
            if symbol_clusters_path.exists():
                with open(symbol_clusters_path, 'r') as f:
                    self.symbol_clusters = json.load(f)
                logger.info(f"Loaded symbol clusters for {len(self.symbol_clusters)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def preprocess_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Preprocess embedding for prediction
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Processed embedding
        """
        try:
            # Ensure 2D shape
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Apply scaler if available
            if self.scaler:
                embedding = self.scaler.transform(embedding)
            
            # Apply dimensionality reduction if available
            if self.reducer:
                embedding = self.reducer.transform(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error preprocessing embedding: {e}")
            return embedding
    
    def predict_cluster(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Predict cluster for embedding
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            Prediction results dictionary
        """
        result = {
            "cluster": -1,
            "confidence": 0.0,
            "distance": float('inf'),
            "method": "none",
            "success": False
        }
        
        try:
            # Preprocess embedding
            processed_embedding = self.preprocess_embedding(embedding)
            
            # Method 1: Use cluster centers (fastest)
            if self.cluster_centers is not None:
                distances = np.linalg.norm(self.cluster_centers - processed_embedding, axis=1)
                cluster = int(np.argmin(distances))
                min_distance = float(distances[cluster])
                
                # Calculate confidence (inverse of normalized distance)
                max_distance = np.max(distances)
                confidence = 1.0 - (min_distance / max_distance) if max_distance > 0 else 1.0
                
                result.update({
                    "cluster": cluster,
                    "confidence": confidence,
                    "distance": min_distance,
                    "method": "cluster_centers",
                    "success": True
                })
                
                return result
            
            # Method 2: Use KMeans model
            elif self.kmeans_model is not None:
                cluster = self.kmeans_model.predict(processed_embedding)[0]
                
                # Calculate distance to cluster center
                center = self.kmeans_model.cluster_centers_[cluster]
                distance = np.linalg.norm(processed_embedding - center)
                
                result.update({
                    "cluster": int(cluster),
                    "confidence": 0.8,  # Default confidence
                    "distance": float(distance),
                    "method": "kmeans_model",
                    "success": True
                })
                
                return result
            
            else:
                logger.warning("No cluster models available for prediction")
                return result
                
        except Exception as e:
            logger.error(f"Error predicting cluster: {e}")
            result["error"] = str(e)
            return result
    
    def get_similar_symbols(self, cluster: int, limit: int = 5) -> List[str]:
        """
        Get symbols from the same cluster
        
        Args:
            cluster: Cluster ID
            limit: Maximum number of symbols to return
            
        Returns:
            List of similar symbols
        """
        if not self.symbol_clusters:
            return []
        
        try:
            similar_symbols = [
                symbol for symbol, sym_cluster in self.symbol_clusters.items()
                if sym_cluster == cluster
            ]
            
            return similar_symbols[:limit]
            
        except Exception as e:
            logger.error(f"Error getting similar symbols: {e}")
            return []
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about clusters"""
        stats = {
            "total_clusters": 0,
            "symbols_per_cluster": {},
            "cluster_sizes": {},
            "models_loaded": {
                "cluster_centers": self.cluster_centers is not None,
                "kmeans_model": self.kmeans_model is not None,
                "scaler": self.scaler is not None,
                "reducer": self.reducer is not None,
                "symbol_clusters": self.symbol_clusters is not None
            }
        }
        
        try:
            if self.cluster_centers is not None:
                stats["total_clusters"] = len(self.cluster_centers)
            
            if self.symbol_clusters:
                cluster_counts = {}
                for symbol, cluster in self.symbol_clusters.items():
                    cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
                
                stats["symbols_per_cluster"] = cluster_counts
                stats["cluster_sizes"] = {
                    "min": min(cluster_counts.values()) if cluster_counts else 0,
                    "max": max(cluster_counts.values()) if cluster_counts else 0,
                    "avg": sum(cluster_counts.values()) / len(cluster_counts) if cluster_counts else 0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cluster stats: {e}")
            return stats
    
    def predict_setup_quality(self, embedding: np.ndarray, symbol: str = "") -> Dict[str, Any]:
        """
        Predict setup quality based on cluster analysis
        
        Args:
            embedding: Input embedding
            symbol: Symbol name for logging
            
        Returns:
            Quality prediction results
        """
        result = {
            "cluster": -1,
            "quality_score": 0.5,  # Neutral
            "similar_symbols": [],
            "recommendation": "neutral",
            "confidence": 0.0
        }
        
        try:
            # Get cluster prediction
            cluster_result = self.predict_cluster(embedding)
            
            if not cluster_result["success"]:
                return result
            
            cluster = cluster_result["cluster"]
            confidence = cluster_result["confidence"]
            
            # Get similar symbols
            similar_symbols = self.get_similar_symbols(cluster, limit=5)
            
            # Simple quality scoring based on cluster characteristics
            # In practice, this could be enhanced with historical performance data
            quality_score = 0.5  # Base neutral score
            
            # Higher confidence -> higher quality
            quality_score += confidence * 0.3
            
            # More similar symbols -> higher confidence in classification
            if len(similar_symbols) > 2:
                quality_score += 0.1
            
            # Ensure score is in [0, 1] range
            quality_score = max(0.0, min(1.0, quality_score))
            
            # Determine recommendation
            if quality_score >= 0.7:
                recommendation = "consider"
            elif quality_score >= 0.6:
                recommendation = "neutral"
            else:
                recommendation = "avoid"
            
            result.update({
                "cluster": cluster,
                "quality_score": quality_score,
                "similar_symbols": similar_symbols,
                "recommendation": recommendation,
                "confidence": confidence
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting setup quality: {e}")
            return result


# Global instance
_global_cluster_predictor = None

def get_cluster_predictor() -> ClusterPredictor:
    """Get global cluster predictor instance"""
    global _global_cluster_predictor
    if _global_cluster_predictor is None:
        _global_cluster_predictor = ClusterPredictor()
    return _global_cluster_predictor

def predict_cluster(embedding: np.ndarray) -> Dict[str, Any]:
    """
    Convenience function to predict cluster
    
    Args:
        embedding: Input embedding vector
        
    Returns:
        Prediction results
    """
    predictor = get_cluster_predictor()
    return predictor.predict_cluster(embedding)

def predict_setup_quality(embedding: np.ndarray, symbol: str = "") -> Dict[str, Any]:
    """
    Convenience function to predict setup quality
    
    Args:
        embedding: Input embedding vector
        symbol: Symbol name
        
    Returns:
        Quality prediction results
    """
    predictor = get_cluster_predictor()
    return predictor.predict_setup_quality(embedding, symbol)

def main():
    """Test cluster prediction functionality"""
    print("Testing Cluster Prediction")
    print("=" * 40)
    
    predictor = ClusterPredictor()
    
    # Get cluster statistics
    stats = predictor.get_cluster_stats()
    
    print("Cluster Statistics:")
    print(f"   Models loaded:")
    for model_name, loaded in stats["models_loaded"].items():
        status = "✅" if loaded else "❌"
        print(f"     {status} {model_name}")
    
    print(f"   Total clusters: {stats['total_clusters']}")
    
    if stats["symbols_per_cluster"]:
        print(f"   Symbols per cluster: {stats['symbols_per_cluster']}")
        print(f"   Cluster sizes: {stats['cluster_sizes']}")
    
    # Test prediction if we have models
    if any(stats["models_loaded"].values()):
        print(f"\nTesting prediction with random embedding...")
        
        # Create a random embedding for testing
        test_embedding = np.random.rand(2060)  # Match expected embedding dimension
        
        # Predict cluster
        cluster_result = predict_cluster(test_embedding)
        print(f"   Cluster prediction: {cluster_result}")
        
        # Predict quality
        quality_result = predict_setup_quality(test_embedding, "TESTUSDT")
        print(f"   Quality prediction: {quality_result}")
        
    else:
        print(f"\n⚠️ No models loaded. Train models first using train_embedding_model.py")

if __name__ == "__main__":
    main()