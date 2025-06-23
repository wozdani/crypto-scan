"""
Embedding Model Training for Vision-AI System
Trenuje modele do klasyfikacji i grupowania podobnych setupów
"""

import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pickle

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available - clustering features disabled")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

logger = logging.getLogger(__name__)

class EmbeddingTrainer:
    """Trainer dla modeli embeddingowych"""
    
    def __init__(self, embeddings_dir: str = "data/embeddings"):
        """Initialize embedding trainer"""
        self.embeddings_dir = Path(embeddings_dir)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.embeddings = None
        self.labels = None
        self.metadata = None
        
        logger.info("Embedding trainer initialized")
    
    def load_embeddings(self) -> Tuple[np.ndarray, List[str], List[Dict]]:
        """
        Load embeddings from data/embeddings directory
        
        Returns:
            Tuple of (embeddings array, labels list, metadata list)
        """
        try:
            embeddings = []
            labels = []
            metadata = []
            
            # Find all embedding files
            embedding_files = list(self.embeddings_dir.glob("*.npy"))
            
            if not embedding_files:
                logger.warning("No embedding files found")
                return np.array([]), [], []
            
            for emb_file in embedding_files:
                try:
                    # Load embedding
                    embedding = np.load(emb_file)
                    embeddings.append(embedding.flatten())  # Ensure 1D
                    
                    # Extract label from filename
                    label = emb_file.stem
                    labels.append(label)
                    
                    # Load metadata if available
                    metadata_file = emb_file.parent / f"{emb_file.stem}_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            meta = json.load(f)
                        metadata.append(meta)
                    else:
                        metadata.append({"filename": label})
                        
                except Exception as e:
                    logger.error(f"Error loading {emb_file}: {e}")
                    continue
            
            embeddings_array = np.array(embeddings)
            
            logger.info(f"Loaded {len(embeddings_array)} embeddings with shape {embeddings_array.shape}")
            
            self.embeddings = embeddings_array
            self.labels = labels
            self.metadata = metadata
            
            return embeddings_array, labels, metadata
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return np.array([]), [], []
    
    def preprocess_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """
        Preprocess embeddings with normalization
        
        Args:
            embeddings: Raw embeddings array
            
        Returns:
            Tuple of (processed embeddings, scaler)
        """
        try:
            # Standardize features
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            logger.info(f"Preprocessed embeddings: {embeddings_scaled.shape}")
            return embeddings_scaled, scaler
            
        except Exception as e:
            logger.error(f"Error preprocessing embeddings: {e}")
            return embeddings, StandardScaler()
    
    def reduce_dimensionality(self, embeddings: np.ndarray, method: str = "pca", n_components: int = 64) -> Tuple[np.ndarray, Any]:
        """
        Reduce embedding dimensionality
        
        Args:
            embeddings: Input embeddings
            method: Reduction method (pca, umap, tsne)
            n_components: Target dimensions
            
        Returns:
            Tuple of (reduced embeddings, reducer model)
        """
        try:
            if method == "pca":
                reducer = PCA(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(embeddings)
                
            elif method == "umap" and UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
                reduced = reducer.fit_transform(embeddings)
                
            elif method == "tsne":
                reducer = TSNE(n_components=min(n_components, 3), random_state=42, perplexity=min(30, len(embeddings)//4))
                reduced = reducer.fit_transform(embeddings)
                
            else:
                logger.warning(f"Method {method} not available, using PCA")
                reducer = PCA(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(embeddings)
            
            logger.info(f"Reduced dimensions from {embeddings.shape[1]} to {reduced.shape[1]} using {method}")
            return reduced, reducer
            
        except Exception as e:
            logger.error(f"Error reducing dimensionality: {e}")
            return embeddings, None
    
    def train_kmeans(self, embeddings: np.ndarray, n_clusters: int = 10) -> KMeans:
        """
        Train KMeans clustering model
        
        Args:
            embeddings: Input embeddings
            n_clusters: Number of clusters
            
        Returns:
            Trained KMeans model
        """
        try:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            model.fit(embeddings)
            
            # Calculate clustering metrics
            if len(embeddings) > n_clusters:
                silhouette = silhouette_score(embeddings, model.labels_)
                calinski = calinski_harabasz_score(embeddings, model.labels_)
                
                logger.info(f"KMeans trained: {n_clusters} clusters")
                logger.info(f"  Silhouette score: {silhouette:.3f}")
                logger.info(f"  Calinski-Harabasz score: {calinski:.3f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training KMeans: {e}")
            return None
    
    def train_hdbscan(self, embeddings: np.ndarray, min_cluster_size: int = 5) -> Optional[Any]:
        """
        Train HDBSCAN clustering model
        
        Args:
            embeddings: Input embeddings
            min_cluster_size: Minimum cluster size
            
        Returns:
            Trained HDBSCAN model or None if not available
        """
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available")
            return None
        
        try:
            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
            model.fit(embeddings)
            
            n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
            n_noise = list(model.labels_).count(-1)
            
            logger.info(f"HDBSCAN trained: {n_clusters} clusters, {n_noise} noise points")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training HDBSCAN: {e}")
            return None
    
    def save_model(self, model: Any, model_name: str, additional_data: Optional[Dict] = None):
        """Save trained model and metadata"""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata = {
                "model_name": model_name,
                "model_type": type(model).__name__,
                "training_date": datetime.now().isoformat(),
                "n_samples": len(self.embeddings) if self.embeddings is not None else 0,
                "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0
            }
            
            if additional_data:
                metadata.update(additional_data)
            
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved model: {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    def train_complete_pipeline(self, n_clusters: int = 10, reduction_method: str = "pca", n_components: int = 64) -> Dict[str, Any]:
        if not SKLEARN_AVAILABLE:
            return {
                "success": False,
                "error": "sklearn not available - install scikit-learn for clustering",
                "models_trained": [],
                "metrics": {}
            }
        """
        Train complete embedding pipeline
        
        Args:
            n_clusters: Number of clusters for KMeans
            reduction_method: Dimensionality reduction method
            n_components: Target dimensions after reduction
            
        Returns:
            Training results dictionary
        """
        results = {
            "success": False,
            "models_trained": [],
            "metrics": {},
            "error": None
        }
        
        try:
            # Load embeddings
            embeddings, labels, metadata = self.load_embeddings()
            
            if len(embeddings) == 0:
                results["error"] = "No embeddings found"
                return results
            
            print(f"[EMBEDDING TRAIN] Loaded {len(embeddings)} embeddings")
            
            # Preprocess
            embeddings_scaled, scaler = self.preprocess_embeddings(embeddings)
            
            # Reduce dimensionality
            embeddings_reduced, reducer = self.reduce_dimensionality(
                embeddings_scaled, reduction_method, n_components
            )
            
            # Train KMeans
            kmeans_model = self.train_kmeans(embeddings_reduced, n_clusters)
            if kmeans_model:
                self.save_model(
                    kmeans_model, 
                    "embedding_kmeans",
                    {"n_clusters": n_clusters, "reduction_method": reduction_method}
                )
                results["models_trained"].append("kmeans")
                
                # Save cluster centers for quick prediction
                np.save(self.models_dir / "embedding_cluster_centers.npy", kmeans_model.cluster_centers_)
            
            # Train HDBSCAN if available
            hdbscan_model = self.train_hdbscan(embeddings_reduced)
            if hdbscan_model:
                self.save_model(hdbscan_model, "embedding_hdbscan")
                results["models_trained"].append("hdbscan")
            
            # Save preprocessing components
            self.save_model(scaler, "embedding_scaler")
            if reducer:
                self.save_model(reducer, f"embedding_{reduction_method}_reducer")
            
            # Create symbol to cluster mapping
            if kmeans_model:
                symbol_clusters = {}
                for i, label in enumerate(labels):
                    symbol = label.split('_')[0]  # Extract symbol from filename
                    cluster = int(kmeans_model.labels_[i])
                    symbol_clusters[symbol] = cluster
                
                cluster_file = self.models_dir / "symbol_clusters.json"
                with open(cluster_file, 'w') as f:
                    json.dump(symbol_clusters, f, indent=2)
                
                results["metrics"]["symbol_clusters"] = len(set(symbol_clusters.values()))
            
            results["success"] = True
            results["metrics"]["n_samples"] = len(embeddings)
            results["metrics"]["original_dim"] = embeddings.shape[1]
            results["metrics"]["reduced_dim"] = embeddings_reduced.shape[1]
            
            print(f"[EMBEDDING TRAIN] Training completed successfully")
            print(f"  Models trained: {', '.join(results['models_trained'])}")
            print(f"  Samples: {results['metrics']['n_samples']}")
            print(f"  Dimensions: {results['metrics']['original_dim']} → {results['metrics']['reduced_dim']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            results["error"] = str(e)
            return results


def train_kmeans(n_clusters: int = 10, reduction_method: str = "pca") -> Dict[str, Any]:
    """
    Convenience function to train KMeans on embeddings
    
    Args:
        n_clusters: Number of clusters
        reduction_method: Dimensionality reduction method
        
    Returns:
        Training results
    """
    trainer = EmbeddingTrainer()
    return trainer.train_complete_pipeline(n_clusters, reduction_method)

def main():
    """Main training function"""
    print("Training Embedding Models for Vision-AI")
    print("=" * 50)
    
    trainer = EmbeddingTrainer()
    
    # Check if embeddings exist
    embedding_files = list(trainer.embeddings_dir.glob("*.npy"))
    print(f"Found {len(embedding_files)} embedding files")
    
    if len(embedding_files) == 0:
        print("❌ No embeddings found. Generate embeddings first using generate_embeddings.py")
        return
    
    # Train models with different configurations
    configs = [
        {"n_clusters": 8, "reduction_method": "pca", "n_components": 64},
        {"n_clusters": 10, "reduction_method": "pca", "n_components": 32},
    ]
    
    if UMAP_AVAILABLE:
        configs.append({"n_clusters": 10, "reduction_method": "umap", "n_components": 32})
    
    for i, config in enumerate(configs, 1):
        print(f"\n🔄 Training configuration {i}/{len(configs)}")
        print(f"   Config: {config}")
        
        results = trainer.train_complete_pipeline(**config)
        
        if results["success"]:
            print(f"   ✅ Training successful")
            print(f"   Models: {', '.join(results['models_trained'])}")
        else:
            print(f"   ❌ Training failed: {results.get('error', 'Unknown error')}")
    
    print(f"\n✅ Embedding model training completed")

if __name__ == "__main__":
    main()