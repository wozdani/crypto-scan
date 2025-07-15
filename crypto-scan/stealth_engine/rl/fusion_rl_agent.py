#!/usr/bin/env python3
"""
Stage 6/7: RL Agent for Fusion Engine Weight Learning
Advanced neural network-based reinforcement learning agent that learns optimal weights
for CaliforniumWhale AI, DiamondWhale AI, and WhaleCLIP based on alert effectiveness
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLAgentV4(nn.Module):
    """
    Advanced RL Agent for learning optimal fusion weights
    Uses neural network to adapt weights based on detector scores and outcome feedback
    """
    
    def __init__(self, state_size: int = 3, action_size: int = 3, learning_rate: float = 0.005):
        """
        Initialize RLAgentV4 with neural network architecture
        
        Args:
            state_size: Input dimension (CaliforniumWhale, DiamondWhale, WhaleCLIP scores)
            action_size: Output dimension (weight adjustments for 3 detectors)
            learning_rate: Learning rate for Adam optimizer
        """
        super(RLAgentV4, self).__init__()
        
        # Neural network architecture
        self.model = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_size),
            nn.Softmax(dim=0)  # Ensures weights sum to 1.0
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initial weights (CaliforniumWhale, DiamondWhale, WhaleCLIP)
        self.weights = [0.4, 0.35, 0.25]
        
        # Training statistics
        self.training_stats = {
            "total_updates": 0,
            "positive_rewards": 0,
            "negative_rewards": 0,
            "avg_reward": 0.0,
            "weight_history": [],
            "last_update": None
        }
        
        # Model persistence
        self.model_path = "crypto-scan/cache/rl_fusion"
        self.weights_file = os.path.join(self.model_path, "fusion_rl_weights.json")
        self.model_file = os.path.join(self.model_path, "fusion_rl_model.pt")
        
        # Ensure directory exists
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load existing model if available
        self.load_model()
        
        logger.info(f"[RL FUSION] RLAgentV4 initialized with weights: {self.weights}")
    
    def get_weights(self, scores: List[float]) -> List[float]:
        """
        Get optimal weights based on current detector scores
        
        Args:
            scores: [californium_score, diamond_score, whaleclip_score]
            
        Returns:
            List of weights [californium_weight, diamond_weight, whaleclip_weight]
        """
        try:
            # Normalize input scores
            input_tensor = torch.tensor(scores, dtype=torch.float32)
            
            # Forward pass through neural network
            with torch.no_grad():
                output = self.model(input_tensor)
                self.weights = output.detach().numpy().tolist()
            
            # Ensure weights are valid
            total_weight = sum(self.weights)
            if total_weight > 0:
                self.weights = [w / total_weight for w in self.weights]
            else:
                # Fallback to default weights
                self.weights = [0.4, 0.35, 0.25]
            
            logger.info(f"[RL FUSION] Computed weights for scores {scores}: {[round(w, 3) for w in self.weights]}")
            
            return self.weights
            
        except Exception as e:
            logger.error(f"[RL FUSION] Error computing weights: {e}")
            return [0.4, 0.35, 0.25]  # Fallback to defaults
    
    def update(self, scores: List[float], reward: float, alert_outcome: str = "unknown"):
        """
        Update the neural network based on alert outcome
        
        Args:
            scores: [californium_score, diamond_score, whaleclip_score] that led to alert
            reward: Reward signal (+1.0 for successful alert, -1.0 for false positive)
            alert_outcome: Description of alert outcome for logging
        """
        try:
            logger.info(f"[RL FUSION] Training update: scores={scores}, reward={reward}, outcome={alert_outcome}")
            
            # Convert to tensor
            input_tensor = torch.tensor(scores, dtype=torch.float32)
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Calculate loss using policy gradient
            # Positive reward reinforces current weights, negative reward discourages them
            loss = -reward * torch.sum(torch.log(output + 1e-8))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            self.training_stats["total_updates"] += 1
            if reward > 0:
                self.training_stats["positive_rewards"] += 1
            else:
                self.training_stats["negative_rewards"] += 1
            
            # Update average reward
            total_rewards = self.training_stats["positive_rewards"] - self.training_stats["negative_rewards"]
            self.training_stats["avg_reward"] = total_rewards / self.training_stats["total_updates"]
            
            # Store weight snapshot
            current_weights = self.get_weights(scores)
            self.training_stats["weight_history"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "weights": current_weights.copy(),
                "reward": reward,
                "outcome": alert_outcome,
                "scores": scores.copy()
            })
            
            # Keep only last 100 weight snapshots
            if len(self.training_stats["weight_history"]) > 100:
                self.training_stats["weight_history"] = self.training_stats["weight_history"][-100:]
            
            self.training_stats["last_update"] = datetime.now(timezone.utc).isoformat()
            
            # Save model and statistics
            self.save_model()
            
            logger.info(f"[RL FUSION] Training complete. Loss: {loss.item():.4f}, New weights: {[round(w, 3) for w in current_weights]}")
            
        except Exception as e:
            logger.error(f"[RL FUSION] Error during training update: {e}")
    
    def batch_update(self, training_data: List[Dict[str, Any]]):
        """
        Perform batch training from multiple alert outcomes
        
        Args:
            training_data: List of dicts with 'scores', 'reward', 'outcome' keys
        """
        logger.info(f"[RL FUSION] Starting batch training with {len(training_data)} samples")
        
        total_loss = 0.0
        for i, data in enumerate(training_data):
            scores = data.get('scores', [0.5, 0.3, 0.2])
            reward = data.get('reward', 0.0)
            outcome = data.get('outcome', f'batch_sample_{i}')
            
            self.update(scores, reward, outcome)
            
        logger.info(f"[RL FUSION] Batch training completed. Updated model with {len(training_data)} samples")
    
    def get_weight_dict(self, scores: List[float]) -> Dict[str, float]:
        """
        Get weights as dictionary for fusion engine compatibility
        
        Args:
            scores: [californium_score, diamond_score, whaleclip_score]
            
        Returns:
            Dictionary with detector names as keys
        """
        weights_list = self.get_weights(scores)
        return {
            "californium": weights_list[0],
            "diamond": weights_list[1],
            "whaleclip": weights_list[2]
        }
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = self.training_stats.copy()
        
        # Add current model info
        stats["current_weights"] = self.weights.copy()
        stats["model_parameters"] = sum(p.numel() for p in self.model.parameters())
        stats["success_rate"] = (
            self.training_stats["positive_rewards"] / max(1, self.training_stats["total_updates"]) * 100
        )
        
        # Recent performance (last 10 updates)
        recent_history = self.training_stats["weight_history"][-10:]
        if recent_history:
            recent_rewards = [h["reward"] for h in recent_history]
            stats["recent_avg_reward"] = sum(recent_rewards) / len(recent_rewards)
            stats["recent_positive_rate"] = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards) * 100
        else:
            stats["recent_avg_reward"] = 0.0
            stats["recent_positive_rate"] = 0.0
        
        return stats
    
    def save_model(self):
        """Save model state and training statistics"""
        try:
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': self.training_stats,
                'current_weights': self.weights
            }, self.model_file)
            
            # Save weights and stats as JSON
            save_data = {
                "weights": self.weights,
                "training_stats": self.training_stats,
                "model_info": {
                    "state_size": 3,
                    "action_size": 3,
                    "architecture": "32->16->3 with ReLU and Softmax"
                },
                "last_saved": datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.weights_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"[RL FUSION] Model and statistics saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"[RL FUSION] Error saving model: {e}")
    
    def load_model(self):
        """Load existing model state if available"""
        try:
            if os.path.exists(self.model_file):
                checkpoint = torch.load(self.model_file, map_location='cpu')
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_stats = checkpoint.get('training_stats', self.training_stats)
                self.weights = checkpoint.get('current_weights', self.weights)
                
                logger.info(f"[RL FUSION] Loaded existing model with {self.training_stats['total_updates']} training updates")
                
            elif os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    data = json.load(f)
                    self.weights = data.get('weights', self.weights)
                    self.training_stats = data.get('training_stats', self.training_stats)
                    
                logger.info(f"[RL FUSION] Loaded weights from JSON: {self.weights}")
                
        except Exception as e:
            logger.warning(f"[RL FUSION] Error loading model, using defaults: {e}")
    
    def reset_training(self):
        """Reset training statistics and reinitialize model"""
        logger.info("[RL FUSION] Resetting training statistics and model weights")
        
        # Reinitialize model
        self.model.apply(self._init_weights)
        
        # Reset statistics
        self.training_stats = {
            "total_updates": 0,
            "positive_rewards": 0,
            "negative_rewards": 0,
            "avg_reward": 0.0,
            "weight_history": [],
            "last_update": None
        }
        
        # Reset to default weights
        self.weights = [0.4, 0.35, 0.25]
        
        # Save reset state
        self.save_model()
    
    def _init_weights(self, module):
        """Initialize neural network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

# Global instance for singleton pattern
_rl_fusion_agent = None

def get_rl_fusion_agent() -> RLAgentV4:
    """Get singleton RLAgentV4 instance"""
    global _rl_fusion_agent
    if _rl_fusion_agent is None:
        _rl_fusion_agent = RLAgentV4()
    return _rl_fusion_agent

def train_fusion_weights(scores: List[float], reward: float, outcome: str = "unknown"):
    """
    Convenience function to train fusion weights
    
    Args:
        scores: [californium_score, diamond_score, whaleclip_score]
        reward: Reward signal (+1.0 for success, -1.0 for failure)
        outcome: Description of outcome
    """
    agent = get_rl_fusion_agent()
    agent.update(scores, reward, outcome)

def get_learned_fusion_weights(scores: List[float]) -> Dict[str, float]:
    """
    Get learned fusion weights for given detector scores
    
    Args:
        scores: [californium_score, diamond_score, whaleclip_score]
        
    Returns:
        Dictionary with learned weights
    """
    agent = get_rl_fusion_agent()
    return agent.get_weight_dict(scores)

def get_fusion_training_stats() -> Dict[str, Any]:
    """Get fusion RL training statistics"""
    agent = get_rl_fusion_agent()
    return agent.get_training_statistics()

def test_rl_fusion_agent():
    """Test RLAgentV4 functionality"""
    print("🧪 Testing RL Fusion Agent V4...")
    
    # Initialize agent
    agent = RLAgentV4()
    
    # Test weight calculation
    test_scores = [0.8, 0.6, 0.4]
    weights = agent.get_weights(test_scores)
    print(f"📊 Initial weights for scores {test_scores}: {[round(w, 3) for w in weights]}")
    
    # Test training with positive outcome
    agent.update(test_scores, 1.0, "successful_alert")
    weights_after_positive = agent.get_weights(test_scores)
    print(f"📈 Weights after positive training: {[round(w, 3) for w in weights_after_positive]}")
    
    # Test training with negative outcome
    agent.update(test_scores, -1.0, "false_positive")
    weights_after_negative = agent.get_weights(test_scores)
    print(f"📉 Weights after negative training: {[round(w, 3) for w in weights_after_negative]}")
    
    # Test batch training
    batch_data = [
        {"scores": [0.9, 0.7, 0.5], "reward": 1.0, "outcome": "pump_detected"},
        {"scores": [0.3, 0.2, 0.8], "reward": -1.0, "outcome": "false_alarm"},
        {"scores": [0.7, 0.8, 0.6], "reward": 1.0, "outcome": "whale_activity"}
    ]
    agent.batch_update(batch_data)
    
    # Test statistics
    stats = agent.get_training_statistics()
    print(f"📊 Training Statistics:")
    print(f"   Total Updates: {stats['total_updates']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Average Reward: {stats['avg_reward']:.3f}")
    print(f"   Recent Performance: {stats['recent_positive_rate']:.1f}%")
    
    # Test weight dictionary format
    weight_dict = agent.get_weight_dict(test_scores)
    print(f"📋 Weight Dictionary: {weight_dict}")
    
    print("✅ RL Fusion Agent V4 test completed")

if __name__ == "__main__":
    test_rl_fusion_agent()