"""
Whale Style Detector - Klasyfikator stylu portfela oparty na embeddingach behavioral
Identyfikuje czy wallet przypomina styl whale'a, relay_bot, market_maker czy bridge_router
"""

import numpy as np
import pickle
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from wallet_behavior_encoder import encode_advanced_wallet_behavior

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhaleStyleDetector:
    """
    Advanced wallet style classifier using behavioral embeddings
    Supports multiple wallet types: whale, relay_bot, market_maker, bridge_router, normal
    """
    
    def __init__(self, model_path: str = "cache/whale_style_models/", model_type: str = "knn"):
        """
        Initialize WhaleStyleDetector
        
        Args:
            model_path: Directory path for model storage
            model_type: Type of classifier ('knn', 'rf', 'mlp')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.class_labels = ['normal', 'whale', 'relay_bot', 'market_maker', 'bridge_router']
        
        # Create model directory if not exists
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
        
        # Load existing models if available
        self._load_models()
        
        # Training data storage
        self.training_data = {
            'embeddings': [],
            'labels': [],
            'wallet_addresses': [],
            'metadata': []
        }
        
        logger.info(f"[WHALE DETECTOR] Initialized with model_type={model_type}")
    
    def _initialize_models(self):
        """Initialize ML models based on type"""
        if not SKLEARN_AVAILABLE:
            logger.warning("[WHALE DETECTOR] sklearn not available - limited functionality")
            return
            
        if self.model_type == 'knn':
            self.models['binary'] = KNeighborsClassifier(n_neighbors=5, weights='distance')
            self.models['multiclass'] = KNeighborsClassifier(n_neighbors=5, weights='distance')
        elif self.model_type == 'rf':
            self.models['binary'] = RandomForestClassifier(n_estimators=100, random_state=42)
            self.models['multiclass'] = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'mlp':
            self.models['binary'] = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
            self.models['multiclass'] = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        
        # Initialize scalers
        self.scalers['binary'] = StandardScaler()
        self.scalers['multiclass'] = StandardScaler()
    
    def _load_models(self):
        """Load existing trained models"""
        model_files = {
            'binary': f"{self.model_path}/whale_detector_binary_{self.model_type}.pkl",
            'multiclass': f"{self.model_path}/whale_detector_multiclass_{self.model_type}.pkl"
        }
        
        scaler_files = {
            'binary': f"{self.model_path}/scaler_binary_{self.model_type}.pkl",
            'multiclass': f"{self.model_path}/scaler_multiclass_{self.model_type}.pkl"
        }
        
        for model_name, file_path in model_files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"[WHALE DETECTOR] Loaded {model_name} model from {file_path}")
                except Exception as e:
                    logger.error(f"[WHALE DETECTOR] Failed to load {model_name} model: {e}")
        
        for scaler_name, file_path in scaler_files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        self.scalers[scaler_name] = pickle.load(f)
                    logger.info(f"[WHALE DETECTOR] Loaded {scaler_name} scaler from {file_path}")
                except Exception as e:
                    logger.error(f"[WHALE DETECTOR] Failed to load {scaler_name} scaler: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        model_files = {
            'binary': f"{self.model_path}/whale_detector_binary_{self.model_type}.pkl",
            'multiclass': f"{self.model_path}/whale_detector_multiclass_{self.model_type}.pkl"
        }
        
        scaler_files = {
            'binary': f"{self.model_path}/scaler_binary_{self.model_type}.pkl",
            'multiclass': f"{self.model_path}/scaler_multiclass_{self.model_type}.pkl"
        }
        
        for model_name, file_path in model_files.items():
            if model_name in self.models:
                try:
                    with open(file_path, 'wb') as f:
                        pickle.dump(self.models[model_name], f)
                    logger.info(f"[WHALE DETECTOR] Saved {model_name} model to {file_path}")
                except Exception as e:
                    logger.error(f"[WHALE DETECTOR] Failed to save {model_name} model: {e}")
        
        for scaler_name, file_path in scaler_files.items():
            if scaler_name in self.scalers:
                try:
                    with open(file_path, 'wb') as f:
                        pickle.dump(self.scalers[scaler_name], f)
                    logger.info(f"[WHALE DETECTOR] Saved {scaler_name} scaler to {file_path}")
                except Exception as e:
                    logger.error(f"[WHALE DETECTOR] Failed to save {scaler_name} scaler: {e}")
    
    def add_training_sample(self, embedding: np.ndarray, wallet_type: str, 
                           wallet_address: str = None, metadata: Dict = None):
        """
        Add training sample to the dataset
        
        Args:
            embedding: Behavioral embedding vector
            wallet_type: Type of wallet ('whale', 'normal', 'relay_bot', etc.)
            wallet_address: Wallet address (optional)
            metadata: Additional metadata (optional)
        """
        if wallet_type not in self.class_labels:
            logger.warning(f"[WHALE DETECTOR] Unknown wallet type: {wallet_type}, using 'normal'")
            wallet_type = 'normal'
        
        self.training_data['embeddings'].append(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
        self.training_data['labels'].append(wallet_type)
        self.training_data['wallet_addresses'].append(wallet_address or f"unknown_{len(self.training_data['embeddings'])}")
        self.training_data['metadata'].append(metadata or {})
        
        logger.info(f"[WHALE DETECTOR] Added training sample: {wallet_type} "
                   f"(total samples: {len(self.training_data['embeddings'])})")
    
    def train_models(self, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train both binary and multiclass models
        
        Args:
            test_size: Fraction of data to use for testing
            
        Returns:
            Training results and metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.error("[WHALE DETECTOR] sklearn not available for training")
            return {'error': 'sklearn_not_available'}
        
        if len(self.training_data['embeddings']) < 5:
            logger.error("[WHALE DETECTOR] Insufficient training data (minimum 5 samples required)")
            return {'error': 'insufficient_data'}
        
        embeddings = np.array(self.training_data['embeddings'])
        labels = np.array(self.training_data['labels'])
        
        # Create binary labels (whale vs non-whale)
        binary_labels = (labels == 'whale').astype(int)
        
        results = {'training_timestamp': datetime.now().isoformat()}
        
        # Train binary classifier
        if len(np.unique(binary_labels)) > 1:  # Check if we have both classes
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    embeddings, binary_labels, test_size=test_size, random_state=42, stratify=binary_labels
                )
                
                # Scale features
                self.scalers['binary'].fit(X_train)
                X_train_scaled = self.scalers['binary'].transform(X_train)
                X_test_scaled = self.scalers['binary'].transform(X_test)
                
                # Train model
                self.models['binary'].fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = self.models['binary'].score(X_train_scaled, y_train)
                test_score = self.models['binary'].score(X_test_scaled, y_test)
                cv_scores = cross_val_score(self.models['binary'], X_train_scaled, y_train, cv=3)
                
                results['binary'] = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'samples': len(embeddings)
                }
                
                logger.info(f"[WHALE DETECTOR] Binary model trained: "
                           f"train_acc={train_score:.3f}, test_acc={test_score:.3f}")
                
            except Exception as e:
                logger.error(f"[WHALE DETECTOR] Binary training failed: {e}")
                results['binary'] = {'error': str(e)}
        
        # Train multiclass classifier
        if len(np.unique(labels)) > 1:  # Check if we have multiple classes
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    embeddings, labels, test_size=test_size, random_state=42
                )
                
                # Scale features
                self.scalers['multiclass'].fit(X_train)
                X_train_scaled = self.scalers['multiclass'].transform(X_train)
                X_test_scaled = self.scalers['multiclass'].transform(X_test)
                
                # Train model
                self.models['multiclass'].fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = self.models['multiclass'].score(X_train_scaled, y_train)
                test_score = self.models['multiclass'].score(X_test_scaled, y_test)
                
                results['multiclass'] = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'samples': len(embeddings),
                    'classes': list(np.unique(labels))
                }
                
                logger.info(f"[WHALE DETECTOR] Multiclass model trained: "
                           f"train_acc={train_score:.3f}, test_acc={test_score:.3f}")
                
            except Exception as e:
                logger.error(f"[WHALE DETECTOR] Multiclass training failed: {e}")
                results['multiclass'] = {'error': str(e)}
        
        # Save models
        self._save_models()
        
        # Save training results
        self._save_training_results(results)
        
        return results
    
    def predict_whale_style(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Predict if wallet exhibits whale-style behavior
        
        Args:
            embedding: Behavioral embedding vector
            
        Returns:
            Prediction results with confidence scores
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn_not_available'}
        
        if 'binary' not in self.models:
            return {'error': 'model_not_trained'}
        
        try:
            # Prepare embedding
            embedding_array = np.array(embedding).reshape(1, -1)
            embedding_scaled = self.scalers['binary'].transform(embedding_array)
            
            # Binary prediction
            is_whale_prob = self.models['binary'].predict_proba(embedding_scaled)[0]
            is_whale = bool(self.models['binary'].predict(embedding_scaled)[0])
            whale_confidence = float(is_whale_prob[1])  # Probability of being whale
            
            result = {
                'is_whale': is_whale,
                'whale_confidence': whale_confidence,
                'normal_confidence': float(is_whale_prob[0]),
                'prediction_type': 'binary'
            }
            
            logger.info(f"[WHALE DETECTOR] Whale prediction: {is_whale} "
                       f"(confidence: {whale_confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"[WHALE DETECTOR] Prediction failed: {e}")
            return {'error': str(e)}
    
    def predict_wallet_type(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Predict detailed wallet type (whale, relay_bot, market_maker, etc.)
        
        Args:
            embedding: Behavioral embedding vector
            
        Returns:
            Multiclass prediction results
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn_not_available'}
        
        if 'multiclass' not in self.models:
            return {'error': 'multiclass_model_not_trained'}
        
        try:
            # Prepare embedding
            embedding_array = np.array(embedding).reshape(1, -1)
            embedding_scaled = self.scalers['multiclass'].transform(embedding_array)
            
            # Multiclass prediction
            predicted_class = self.models['multiclass'].predict(embedding_scaled)[0]
            class_probabilities = self.models['multiclass'].predict_proba(embedding_scaled)[0]
            
            # Get class names
            classes = self.models['multiclass'].classes_
            
            # Create probability dictionary
            class_confidences = {}
            for i, class_name in enumerate(classes):
                class_confidences[class_name] = float(class_probabilities[i])
            
            result = {
                'predicted_type': predicted_class,
                'confidence': float(max(class_probabilities)),
                'all_confidences': class_confidences,
                'prediction_type': 'multiclass'
            }
            
            logger.info(f"[WHALE DETECTOR] Type prediction: {predicted_class} "
                       f"(confidence: {max(class_probabilities):.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"[WHALE DETECTOR] Multiclass prediction failed: {e}")
            return {'error': str(e)}
    
    def analyze_wallet_comprehensive(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive wallet analysis combining binary and multiclass predictions
        
        Args:
            embedding: Behavioral embedding vector
            
        Returns:
            Complete analysis results
        """
        # Binary whale detection
        whale_result = self.predict_whale_style(embedding)
        
        # Multiclass type detection
        type_result = self.predict_wallet_type(embedding)
        
        # Combine results
        comprehensive_result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'whale_analysis': whale_result,
            'type_analysis': type_result,
            'embedding_dimension': len(embedding)
        }
        
        # Enhanced interpretation
        if not whale_result.get('error') and not type_result.get('error'):
            # Risk assessment
            whale_confidence = whale_result.get('whale_confidence', 0)
            predicted_type = type_result.get('predicted_type', 'normal')
            type_confidence = type_result.get('confidence', 0)
            
            # Combined risk score
            risk_score = 0.0
            if predicted_type == 'whale':
                risk_score = max(whale_confidence, type_confidence)
            elif predicted_type in ['relay_bot', 'market_maker']:
                risk_score = type_confidence * 0.7  # Medium risk
            elif predicted_type == 'bridge_router':
                risk_score = type_confidence * 0.5  # Lower risk
            
            comprehensive_result['risk_assessment'] = {
                'risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.4 else 'LOW',
                'is_suspicious': risk_score > 0.6,
                'interpretation': f"Wallet classified as {predicted_type} with {type_confidence:.1%} confidence"
            }
        
        return comprehensive_result
    
    def load_training_data_from_gnn_results(self, gnn_results_dir: str = "cache/") -> int:
        """
        Load training data from GNN analysis results
        
        Args:
            gnn_results_dir: Directory containing GNN analysis results
            
        Returns:
            Number of samples loaded
        """
        samples_loaded = 0
        
        # Look for behavior analysis files
        for filename in os.listdir(gnn_results_dir):
            if filename.startswith('behavior_analysis_') and filename.endswith('.json'):
                try:
                    filepath = os.path.join(gnn_results_dir, filename)
                    with open(filepath, 'r') as f:
                        analysis_data = json.load(f)
                    
                    # Extract embeddings and create labels based on anomaly scores
                    gnn_correlation = analysis_data.get('gnn_behavior_correlation', {})
                    
                    for wallet_addr, corr_data in gnn_correlation.items():
                        if corr_data.get('gnn_status') == 'success':
                            embedding = corr_data.get('behavioral_embedding', [])
                            max_anomaly = corr_data.get('max_anomaly_score', 0)
                            
                            if embedding and len(embedding) >= 6:  # Valid embedding
                                # Label based on anomaly score and transaction patterns
                                if max_anomaly > 0.8:
                                    label = 'whale'
                                elif max_anomaly > 0.5:
                                    label = 'relay_bot'  # Medium risk, possibly automated
                                else:
                                    label = 'normal'
                                
                                # Add to training data
                                self.add_training_sample(
                                    embedding=np.array(embedding),
                                    wallet_type=label,
                                    wallet_address=wallet_addr,
                                    metadata={
                                        'max_anomaly_score': max_anomaly,
                                        'transaction_count': corr_data.get('transaction_count', 0),
                                        'source_file': filename
                                    }
                                )
                                samples_loaded += 1
                                
                except Exception as e:
                    logger.error(f"[WHALE DETECTOR] Failed to load from {filename}: {e}")
        
        logger.info(f"[WHALE DETECTOR] Loaded {samples_loaded} training samples from GNN results")
        return samples_loaded
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to file"""
        results_file = f"{self.model_path}/training_results_{self.model_type}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"[WHALE DETECTOR] Training results saved to {results_file}")
        except Exception as e:
            logger.error(f"[WHALE DETECTOR] Failed to save training results: {e}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        stats = {
            'model_type': self.model_type,
            'class_labels': self.class_labels,
            'training_samples': len(self.training_data['embeddings']),
            'models_available': list(self.models.keys()),
            'scalers_available': list(self.scalers.keys()),
            'sklearn_available': SKLEARN_AVAILABLE
        }
        
        # Add class distribution
        if self.training_data['labels']:
            from collections import Counter
            class_distribution = Counter(self.training_data['labels'])
            stats['class_distribution'] = dict(class_distribution)
        
        return stats

def create_synthetic_training_data() -> List[Tuple[np.ndarray, str]]:
    """
    Create synthetic training data for demonstration purposes
    In production, this should be replaced with real labeled data
    """
    training_samples = []
    
    # Whale patterns (high volume, few unique addresses, large average values)
    for i in range(20):
        whale_embedding = np.array([
            np.random.uniform(1e7, 1e8),      # total_sent - very high
            np.random.uniform(5e6, 5e7),      # total_received - high  
            np.random.uniform(1e6, 1e7),      # avg_value - very high
            np.random.uniform(10, 50),        # tx_count - moderate
            np.random.uniform(5, 20),         # unique_to - low
            np.random.uniform(5, 20),         # unique_from - low
            np.random.uniform(1e12, 1e14),    # value_variance - very high
            np.random.uniform(5e5, 5e6),      # value_median - high
            np.random.uniform(0.5, 2.0),      # sent_ratio - varied
            np.random.uniform(0.3, 0.8),      # unique_density - low
            np.random.uniform(1e8, 1e10),     # interval_variance - high
            np.random.uniform(5.0, 15.0)      # whale_indicator - very high
        ])
        training_samples.append((whale_embedding, 'whale'))
    
    # Normal user patterns (lower volume, more distributed)
    for i in range(30):
        normal_embedding = np.array([
            np.random.uniform(1e3, 1e5),      # total_sent - low
            np.random.uniform(1e3, 1e5),      # total_received - low
            np.random.uniform(1e2, 1e4),      # avg_value - low
            np.random.uniform(5, 100),        # tx_count - varied
            np.random.uniform(3, 30),         # unique_to - moderate
            np.random.uniform(3, 30),         # unique_from - moderate
            np.random.uniform(1e6, 1e9),      # value_variance - moderate
            np.random.uniform(1e2, 1e3),      # value_median - low
            np.random.uniform(0.3, 3.0),      # sent_ratio - varied
            np.random.uniform(0.5, 2.0),      # unique_density - moderate
            np.random.uniform(1e6, 1e8),      # interval_variance - moderate
            np.random.uniform(0.5, 3.0)       # whale_indicator - low
        ])
        training_samples.append((normal_embedding, 'normal'))
    
    # Relay bot patterns (many transactions, consistent values)
    for i in range(15):
        relay_embedding = np.array([
            np.random.uniform(1e6, 1e7),      # total_sent - high
            np.random.uniform(1e6, 1e7),      # total_received - high
            np.random.uniform(1e4, 1e5),      # avg_value - moderate
            np.random.uniform(100, 1000),     # tx_count - very high
            np.random.uniform(20, 100),       # unique_to - high
            np.random.uniform(20, 100),       # unique_from - high
            np.random.uniform(1e8, 1e10),     # value_variance - moderate
            np.random.uniform(1e4, 1e5),      # value_median - moderate
            np.random.uniform(0.8, 1.2),      # sent_ratio - balanced
            np.random.uniform(1.0, 3.0),      # unique_density - high
            np.random.uniform(1e4, 1e6),      # interval_variance - low (regular)
            np.random.uniform(1.0, 3.0)       # whale_indicator - low
        ])
        training_samples.append((relay_embedding, 'relay_bot'))
    
    return training_samples

def test_whale_style_detector():
    """Test the WhaleStyleDetector functionality"""
    logger.info("[TEST] Starting WhaleStyleDetector tests")
    
    # Initialize detector
    detector = WhaleStyleDetector(model_type='rf')  # Use RandomForest for testing
    
    # Create synthetic training data
    training_samples = create_synthetic_training_data()
    
    # Add training samples
    for embedding, label in training_samples:
        detector.add_training_sample(embedding, label)
    
    # Train models
    logger.info("[TEST] Training models...")
    training_results = detector.train_models()
    
    if 'error' not in training_results:
        logger.info(f"[TEST] Training completed successfully")
        if 'binary' in training_results:
            logger.info(f"[TEST] Binary model accuracy: {training_results['binary']['test_accuracy']:.3f}")
        if 'multiclass' in training_results:
            logger.info(f"[TEST] Multiclass model accuracy: {training_results['multiclass']['test_accuracy']:.3f}")
    
    # Test predictions
    test_whale_embedding = np.array([
        5e7, 2e7, 5e6, 25, 10, 8, 5e13, 2e6, 1.5, 0.4, 5e9, 10.0
    ])
    
    # Whale prediction
    whale_result = detector.predict_whale_style(test_whale_embedding)
    logger.info(f"[TEST] Whale prediction: {whale_result}")
    
    # Type prediction
    type_result = detector.predict_wallet_type(test_whale_embedding)
    logger.info(f"[TEST] Type prediction: {type_result}")
    
    # Comprehensive analysis
    comprehensive_result = detector.analyze_wallet_comprehensive(test_whale_embedding)
    logger.info(f"[TEST] Comprehensive analysis: {comprehensive_result.get('risk_assessment', {})}")
    
    # Model stats
    stats = detector.get_model_stats()
    logger.info(f"[TEST] Model stats: training_samples={stats['training_samples']}, "
               f"classes={stats.get('class_distribution', {})}")
    
    logger.info("[TEST] WhaleStyleDetector tests completed")
    return True

if __name__ == "__main__":
    test_whale_style_detector()