#!/usr/bin/env python3
"""
RLAgentV3 - Adaptacyjny Agent RL z Dynamicznymi Wagami Boosterów
Advanced Reinforcement Learning Agent with adaptive importance weighting for signal boosters
"""

import json
import os
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLAgentV3:
    """
    Adaptacyjny Agent RL, który uczy się dynamicznych wag importance dla każdego boostera:
    - GNN anomaly score
    - WhaleCLIP confidence
    - DEX inflow detection
    - Dodatkowe sygnały scoringowe
    
    System automatycznie dostosowuje wagi w oparciu o skuteczność alertów
    """
    
    def __init__(self, booster_names: Tuple[str, ...] = ("gnn", "whaleClip", "dexInflow"), 
                 learning_rate: float = 0.05, decay: float = 0.995, 
                 weight_path: str = None, min_weight: float = 0.1, 
                 max_weight: float = 5.0, normalize_weights: bool = True):
        """
        Initialize RLAgentV3 with adaptive booster weighting
        
        Args:
            booster_names: Tuple of signal booster names
            learning_rate: Learning rate for weight updates (0.0-1.0)
            decay: Decay factor for weight stability (0.0-1.0)
            weight_path: Path to save/load weights JSON file
            min_weight: Minimum allowed weight value
            max_weight: Maximum allowed weight value
            normalize_weights: Whether to normalize weights to sum=1
        """
        self.learning_rate = learning_rate
        self.decay = decay
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.normalize_weights = normalize_weights
        
        # Initialize equal weights for all boosters
        self.weights = {name: 1.0 for name in booster_names}
        self.weight_path = weight_path or "cache/rl_agent_v3_weights.json"
        
        # Statistics tracking
        self.update_count = 0
        self.total_reward = 0.0
        self.successful_alerts = 0
        self.failed_alerts = 0
        self.weight_history = []
        
        # Training statistics per booster
        self.booster_stats = {
            name: {
                "total_contribution": 0.0,
                "successful_contribution": 0.0,
                "update_count": 0,
                "avg_effectiveness": 0.0
            } for name in booster_names
        }
        
        # Load existing weights if available
        self.load_weights()
        
        logger.info(f"[RL V3] Initialized with {len(self.weights)} boosters: {list(self.weights.keys())}")
        logger.info(f"[RL V3] Learning rate: {self.learning_rate}, Decay: {self.decay}")
        logger.info(f"[RL V3] Weight bounds: [{self.min_weight}, {self.max_weight}]")
    
    def compute_final_score(self, inputs: Dict[str, float]) -> float:
        """
        Compute weighted final score from booster inputs
        
        Args:
            inputs: Dictionary of booster values {'gnn': 0.6, 'whaleClip': 0.9, 'dexInflow': 1.0}
            
        Returns:
            Weighted final score
        """
        total_score = 0.0
        total_weight = 0.0
        
        for key, value in inputs.items():
            if key in self.weights:
                weight = self.weights[key]
                contribution = value * weight
                total_score += contribution
                total_weight += weight
                
                # Track contribution for statistics
                if key in self.booster_stats:
                    self.booster_stats[key]["total_contribution"] += contribution
        
        # Normalize by total weight if enabled
        if self.normalize_weights and total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = total_score
        
        logger.debug(f"[RL V3 SCORE] Inputs: {inputs} → Final: {final_score:.4f}")
        return final_score
    
    def update_weights(self, inputs: Dict[str, float], reward: float, 
                      alert_metadata: Dict[str, Any] = None):
        """
        Update booster weights based on alert outcome
        
        Args:
            inputs: Dictionary of booster values used in decision
            reward: Alert outcome (+1.0 = successful pump, -1.0 = false alert, 0.0 = neutral)
            alert_metadata: Optional metadata about the alert
        """
        self.update_count += 1
        self.total_reward += reward
        
        if reward > 0:
            self.successful_alerts += 1
        elif reward < 0:
            self.failed_alerts += 1
        
        # Store current weights for history
        self.weight_history.append({
            "timestamp": datetime.now().isoformat(),
            "weights": self.weights.copy(),
            "reward": reward,
            "inputs": inputs.copy()
        })
        
        # Update weights for each booster
        weight_changes = {}
        
        for key, value in inputs.items():
            if key in self.weights:
                # Calculate weight update: δw = lr × reward × signal_strength
                delta = self.learning_rate * reward * value
                old_weight = self.weights[key]
                
                # Apply update and decay
                new_weight = old_weight + delta
                new_weight *= self.decay  # Apply decay for stability
                
                # Apply bounds
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                
                self.weights[key] = new_weight
                weight_changes[key] = new_weight - old_weight
                
                # Update booster statistics
                if key in self.booster_stats:
                    stats = self.booster_stats[key]
                    stats["update_count"] += 1
                    if reward > 0:
                        stats["successful_contribution"] += value
                    
                    # Calculate average effectiveness
                    if stats["total_contribution"] > 0:
                        stats["avg_effectiveness"] = stats["successful_contribution"] / stats["total_contribution"]
        
        # Normalize weights if enabled
        if self.normalize_weights:
            self._normalize_weights()
        
        # Log weight updates
        logger.info(f"[RL V3 UPDATE] Reward: {reward:+.2f}, Updates: {len(weight_changes)}")
        for key, change in weight_changes.items():
            logger.info(f"[RL V3 WEIGHT] {key}: {self.weights[key]:.4f} ({change:+.4f})")
    
    def _normalize_weights(self):
        """Normalize weights to sum to number of boosters"""
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            target_sum = len(self.weights)  # Keep average weight = 1.0
            for key in self.weights:
                self.weights[key] = (self.weights[key] / total_weight) * target_sum
    
    def get_booster_importance_ranking(self) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Get ranking of boosters by importance (weight + effectiveness)
        
        Returns:
            List of (booster_name, importance_score, detailed_stats) tuples
        """
        ranking = []
        
        for booster, weight in self.weights.items():
            stats = self.booster_stats.get(booster, {})
            effectiveness = stats.get("avg_effectiveness", 0.0)
            update_count = stats.get("update_count", 0)
            
            # Importance = weight × effectiveness × experience_factor
            experience_factor = min(1.0, update_count / 10.0)  # Max at 10 updates
            importance = weight * (1.0 + effectiveness) * (0.5 + 0.5 * experience_factor)
            
            detailed_stats = {
                "weight": weight,
                "effectiveness": effectiveness,
                "updates": update_count,
                "experience_factor": experience_factor
            }
            
            ranking.append((booster, importance, detailed_stats))
        
        # Sort by importance (descending)
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics
        
        Returns:
            Dictionary with training metrics
        """
        total_alerts = self.successful_alerts + self.failed_alerts
        success_rate = (self.successful_alerts / total_alerts * 100) if total_alerts > 0 else 0.0
        avg_reward = self.total_reward / self.update_count if self.update_count > 0 else 0.0
        
        # Get booster ranking
        ranking = self.get_booster_importance_ranking()
        
        stats = {
            "update_count": self.update_count,
            "total_reward": self.total_reward,
            "avg_reward": avg_reward,
            "successful_alerts": self.successful_alerts,
            "failed_alerts": self.failed_alerts,
            "success_rate": success_rate,
            "current_weights": self.weights.copy(),
            "booster_ranking": ranking,
            "booster_stats": self.booster_stats.copy(),
            "learning_rate": self.learning_rate,
            "decay": self.decay,
            "weight_history_length": len(self.weight_history)
        }
        
        return stats
    
    def predict_alert_quality(self, inputs: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict alert quality based on current weights and booster effectiveness
        
        Args:
            inputs: Booster input values
            
        Returns:
            Prediction with confidence and breakdown
        """
        final_score = self.compute_final_score(inputs)
        
        # Calculate confidence based on booster effectiveness
        total_confidence = 0.0
        booster_contributions = {}
        
        for key, value in inputs.items():
            if key in self.weights and key in self.booster_stats:
                weight = self.weights[key]
                effectiveness = self.booster_stats[key].get("avg_effectiveness", 0.5)
                contribution = value * weight * effectiveness
                
                booster_contributions[key] = {
                    "value": value,
                    "weight": weight,
                    "effectiveness": effectiveness,
                    "contribution": contribution
                }
                
                total_confidence += contribution
        
        # Normalize confidence to 0-1 range
        confidence = min(1.0, total_confidence / len(inputs))
        
        # Generate recommendation with more lenient thresholds
        if final_score >= 0.75 and confidence >= 0.6:
            recommendation = "STRONG_ALERT"
        elif final_score >= 0.5 and confidence >= 0.3:
            recommendation = "MODERATE_ALERT"
        elif final_score >= 0.3:
            recommendation = "WEAK_SIGNAL"
        else:
            recommendation = "NO_ALERT"
        
        return {
            "final_score": final_score,
            "confidence": confidence,
            "recommendation": recommendation,
            "booster_contributions": booster_contributions,
            "dominant_booster": max(booster_contributions.keys(), 
                                  key=lambda k: booster_contributions[k]["contribution"]) if booster_contributions else None
        }
    
    def save_weights(self, custom_path: str = None):
        """
        Save weights and statistics to JSON file
        
        Args:
            custom_path: Optional custom save path
        """
        save_path = custom_path or self.weight_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            "weights": self.weights,
            "metadata": {
                "update_count": self.update_count,
                "total_reward": self.total_reward,
                "successful_alerts": self.successful_alerts,
                "failed_alerts": self.failed_alerts,
                "learning_rate": self.learning_rate,
                "decay": self.decay,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
                "normalize_weights": self.normalize_weights,
                "last_updated": datetime.now().isoformat()
            },
            "booster_stats": self.booster_stats,
            "weight_history": self.weight_history[-50:]  # Keep last 50 updates
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"[RL V3 SAVE] Weights saved to {save_path}")
    
    def load_weights(self, custom_path: str = None):
        """
        Load weights and statistics from JSON file
        
        Args:
            custom_path: Optional custom load path
        """
        load_path = custom_path or self.weight_path
        
        if not os.path.exists(load_path):
            logger.info(f"[RL V3] No existing weights found at {load_path}")
            return
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Load weights
            if "weights" in data:
                self.weights.update(data["weights"])
            
            # Load metadata
            if "metadata" in data:
                metadata = data["metadata"]
                self.update_count = metadata.get("update_count", 0)
                self.total_reward = metadata.get("total_reward", 0.0)
                self.successful_alerts = metadata.get("successful_alerts", 0)
                self.failed_alerts = metadata.get("failed_alerts", 0)
            
            # Load booster statistics
            if "booster_stats" in data:
                self.booster_stats.update(data["booster_stats"])
            
            # Load weight history
            if "weight_history" in data:
                self.weight_history = data["weight_history"]
            
            logger.info(f"[RL V3 LOAD] Weights loaded from {load_path}")
            logger.info(f"[RL V3 LOAD] Updates: {self.update_count}, Reward: {self.total_reward:.2f}")
            
        except Exception as e:
            logger.error(f"[RL V3] Failed to load weights from {load_path}: {e}")
    
    def reset_weights(self, equal_weights: bool = True):
        """
        Reset all weights to initial values
        
        Args:
            equal_weights: If True, set all weights to 1.0, else random
        """
        if equal_weights:
            for key in self.weights:
                self.weights[key] = 1.0
        else:
            import random
            for key in self.weights:
                self.weights[key] = random.uniform(0.5, 1.5)
        
        # Reset statistics
        self.update_count = 0
        self.total_reward = 0.0
        self.successful_alerts = 0
        self.failed_alerts = 0
        self.weight_history = []
        
        for stats in self.booster_stats.values():
            stats.update({
                "total_contribution": 0.0,
                "successful_contribution": 0.0,
                "update_count": 0,
                "avg_effectiveness": 0.0
            })
        
        logger.info(f"[RL V3 RESET] Weights reset to equal values")
    
    def save_weights(self):
        """Save current weights and statistics to file"""
        try:
            os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
            
            data = {
                "weights": self.weights,
                "metadata": {
                    "update_count": self.update_count,
                    "total_reward": self.total_reward,
                    "successful_alerts": self.successful_alerts,
                    "failed_alerts": self.failed_alerts
                },
                "booster_stats": self.booster_stats,
                "weight_history": self.weight_history[-50:],  # Keep last 50 entries
                "last_updated": datetime.now().isoformat(),
                "config": {
                    "learning_rate": self.learning_rate,
                    "decay": self.decay,
                    "min_weight": self.min_weight,
                    "max_weight": self.max_weight,
                    "normalize_weights": self.normalize_weights
                }
            }
            
            with open(self.weight_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"[RL V3] Saved weights to {self.weight_path}")
            
        except Exception as e:
            logger.error(f"[RL V3] Failed to save weights: {e}")
    
    def save(self):
        """Alias for save_weights() for backward compatibility"""
        self.save_weights()

def test_rl_agent_v3():
    """Test RLAgentV3 functionality"""
    print("🧪 Testing RLAgentV3 - Adaptive Booster Weighting")
    
    # Initialize agent
    agent = RLAgentV3(
        booster_names=("gnn", "whaleClip", "dexInflow", "volumeSpike"),
        learning_rate=0.1,
        weight_path=None  # Don't save during test
    )
    
    print(f"✅ Initial weights: {agent.weights}")
    
    # Test scenarios
    scenarios = [
        # Successful pump with strong GNN + WhaleCLIP
        ({"gnn": 0.95, "whaleClip": 0.87, "dexInflow": 0.2, "volumeSpike": 0.1}, 1.0),
        # Failed alert with weak signals
        ({"gnn": 0.3, "whaleClip": 0.1, "dexInflow": 0.8, "volumeSpike": 0.9}, -1.0),
        # Successful pump with DEX + Volume
        ({"gnn": 0.4, "whaleClip": 0.2, "dexInflow": 1.0, "volumeSpike": 0.9}, 1.0),
        # Another successful GNN + CLIP
        ({"gnn": 0.88, "whaleClip": 0.92, "dexInflow": 0.1, "volumeSpike": 0.0}, 1.0),
    ]
    
    print("\n📊 Training scenarios:")
    for i, (inputs, reward) in enumerate(scenarios, 1):
        score = agent.compute_final_score(inputs)
        prediction = agent.predict_alert_quality(inputs)
        
        print(f"\n{i}. Inputs: {inputs}")
        print(f"   Score: {score:.3f}, Confidence: {prediction['confidence']:.3f}")
        print(f"   Recommendation: {prediction['recommendation']}")
        print(f"   Reward: {reward:+.1f}")
        
        agent.update_weights(inputs, reward)
    
    # Final statistics
    stats = agent.get_training_statistics()
    print(f"\n📈 Final Statistics:")
    print(f"   Updates: {stats['update_count']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Average reward: {stats['avg_reward']:.3f}")
    
    print(f"\n🏆 Booster Importance Ranking:")
    for i, (booster, importance, details) in enumerate(stats['booster_ranking'], 1):
        print(f"   {i}. {booster}: {importance:.3f} (weight: {details['weight']:.3f}, "
              f"effectiveness: {details['effectiveness']:.3f})")
    
    print(f"\n✅ Final weights: {stats['current_weights']}")
    return True

if __name__ == "__main__":
    test_rl_agent_v3()