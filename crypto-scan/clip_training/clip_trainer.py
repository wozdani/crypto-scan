"""
CLIP Trainer for Online Learning
Trenuje model CLIP na nowych danych z auto-labelingu w czasie rzeczywistym
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

from .clip_model import get_clip_model
from .dataset_loader import CLIPDatasetLoader

logger = logging.getLogger(__name__)

class CLIPOnlineTrainer:
    """Online trainer for CLIP model with real-time data"""
    
    def __init__(self):
        self.model = get_clip_model()
        self.loader = CLIPDatasetLoader()
        self.training_history = []
        
        # Training configuration
        self.batch_size = 8
        self.max_pairs_per_session = 50
        self.min_pairs_for_training = 2
        
        logger.info("CLIP Online Trainer initialized")
    
    def train_clip_on_new_data(self, num_epochs: int = 1, hours_back: int = 6) -> Dict:
        """
        Train CLIP model on new data from recent scans
        
        Args:
            num_epochs: Number of training epochs
            hours_back: How many hours back to look for new data
            
        Returns:
            Training results summary
        """
        start_time = time.time()
        
        logger.info(f"Starting CLIP training on new data (last {hours_back}h, {num_epochs} epochs)")
        
        if not self.model.is_initialized:
            return {
                "success": False,
                "error": "CLIP model not initialized",
                "training_time": 0
            }
        
        # Load recent training pairs
        recent_pairs = self.loader.load_recent_pairs(
            hours=hours_back, 
            limit=self.max_pairs_per_session
        )
        
        if len(recent_pairs) < self.min_pairs_for_training:
            logger.info(f"Insufficient new data for training: {len(recent_pairs)} pairs")
            return {
                "success": True,
                "message": f"Insufficient data ({len(recent_pairs)} pairs < {self.min_pairs_for_training})",
                "pairs_found": len(recent_pairs),
                "training_time": time.time() - start_time
            }
        
        logger.info(f"Found {len(recent_pairs)} recent training pairs")
        
        # Train model
        training_results = self._train_on_pairs(recent_pairs, num_epochs)
        
        # Save model after training
        self.model.save_model()
        
        training_time = time.time() - start_time
        
        # Record training session
        session_info = {
            "timestamp": datetime.now().isoformat(),
            "pairs_trained": len(recent_pairs),
            "epochs": num_epochs,
            "training_time": training_time,
            "avg_loss": training_results.get("avg_loss", 0),
            "training_step": self.model.training_step
        }
        
        self.training_history.append(session_info)
        
        logger.info(f"CLIP training completed in {training_time:.2f}s, avg loss: {training_results.get('avg_loss', 0):.4f}")
        
        return {
            "success": True,
            "pairs_trained": len(recent_pairs),
            "epochs": num_epochs,
            "training_time": training_time,
            "avg_loss": training_results.get("avg_loss", 0),
            "total_training_step": self.model.training_step,
            "model_saved": True
        }
    
    def _train_on_pairs(self, pairs: List[tuple], num_epochs: int) -> Dict:
        """Train model on provided pairs"""
        total_loss = 0
        num_batches = 0
        
        for epoch in range(num_epochs):
            # Process pairs in batches
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                
                # Prepare batch for training
                try:
                    images, texts = self.loader.prepare_batch_for_training(batch_pairs, self.model)
                    
                    if len(images) > 0:
                        # Train on batch
                        batch_loss = self.model.train_on_batch(images, texts)
                        total_loss += batch_loss
                        num_batches += 1
                        
                        logger.debug(f"Epoch {epoch+1}/{num_epochs}, Batch {num_batches}: loss={batch_loss:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training batch: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            "avg_loss": avg_loss,
            "num_batches": num_batches,
            "total_loss": total_loss
        }
    
    def validate_model(self, validation_pairs: Optional[List[tuple]] = None) -> Dict:
        """
        Validate model performance on test data
        
        Args:
            validation_pairs: Optional validation data (uses recent data if None)
            
        Returns:
            Validation results
        """
        if validation_pairs is None:
            # Use some recent data for validation
            all_pairs = self.loader.load_training_pairs(limit=20)
            validation_pairs = all_pairs[-10:] if len(all_pairs) >= 10 else all_pairs
        
        if not validation_pairs:
            return {"error": "No validation data available"}
        
        logger.info(f"Validating model on {len(validation_pairs)} pairs")
        
        # Candidate texts for prediction
        candidate_texts = [
            "bullish breakout continuation",
            "bearish trend reversal", 
            "sideways consolidation",
            "pullback in uptrend",
            "fakeout pattern",
            "volume-backed breakout",
            "exhaustion pattern",
            "accumulation phase"
        ]
        
        correct_predictions = 0
        total_predictions = 0
        
        for image_path, actual_text in validation_pairs:
            try:
                # Get model prediction
                prediction = self.model.predict(image_path, candidate_texts)
                
                if prediction.get("success"):
                    predicted_text = prediction["predicted_text"]
                    confidence = prediction["confidence"]
                    
                    # Simple validation: check if key words match
                    actual_words = set(actual_text.lower().split())
                    predicted_words = set(predicted_text.lower().split())
                    
                    # Count as correct if there's significant overlap
                    overlap = len(actual_words.intersection(predicted_words))
                    if overlap >= 1 or confidence > 0.7:
                        correct_predictions += 1
                    
                    total_predictions += 1
                
            except Exception as e:
                logger.error(f"Validation error for {image_path}: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "validation_pairs": len(validation_pairs)
        }
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        stats = self.loader.get_dataset_stats()
        model_info = self.model.get_model_info()
        
        return {
            "model_info": model_info,
            "dataset_stats": stats,
            "training_sessions": len(self.training_history),
            "last_training": self.training_history[-1] if self.training_history else None,
            "total_training_time": sum(session.get("training_time", 0) for session in self.training_history)
        }


# Global trainer instance
_global_trainer = None

def get_clip_trainer() -> CLIPOnlineTrainer:
    """Get global CLIP trainer instance"""
    global _global_trainer
    if _global_trainer is None:
        _global_trainer = CLIPOnlineTrainer()
    return _global_trainer

def train_clip_on_new_data(num_epochs: int = 1, hours_back: int = 6) -> Dict:
    """
    Convenience function for training CLIP on new data
    
    Args:
        num_epochs: Number of training epochs
        hours_back: Hours back to look for new data
        
    Returns:
        Training results
    """
    trainer = get_clip_trainer()
    return trainer.train_clip_on_new_data(num_epochs=num_epochs, hours_back=hours_back)

def main():
    """Test CLIP trainer functionality"""
    print("Testing CLIP Online Trainer")
    print("=" * 40)
    
    trainer = CLIPOnlineTrainer()
    
    # Get training statistics
    stats = trainer.get_training_stats()
    
    print("Training Statistics:")
    print(f"   Model initialized: {stats['model_info']['is_initialized']}")
    print(f"   Training step: {stats['model_info']['training_step']}")
    print(f"   Available pairs: {stats['dataset_stats'].get('valid_pairs', 0)}")
    print(f"   Recent pairs (24h): {stats['dataset_stats'].get('recent_pairs_24h', 0)}")
    
    # Test training if data available
    if stats['dataset_stats'].get('valid_pairs', 0) > 0:
        print(f"\nTesting training on new data...")
        
        result = trainer.train_clip_on_new_data(num_epochs=1, hours_back=24)
        
        if result["success"]:
            print(f"   ✅ Training completed")
            print(f"   Pairs trained: {result.get('pairs_trained', 0)}")
            print(f"   Training time: {result.get('training_time', 0):.2f}s")
            print(f"   Average loss: {result.get('avg_loss', 0):.4f}")
        else:
            print(f"   ❌ Training failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"\n⚠️ No training data available")

if __name__ == "__main__":
    main()