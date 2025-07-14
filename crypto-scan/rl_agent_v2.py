#!/usr/bin/env python3
"""
RLAgentV2 - Rozszerzony agent RL z epsilon-greedy eksploracja, decay i learning rate
Enhanced Reinforcement Learning Agent with epsilon decay, exploration, and advanced Q-table management
"""

import random
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLAgentV2:
    """
    Enhanced Reinforcement Learning Agent with:
    - Epsilon-greedy exploration strategy
    - Learning rate and decay mechanisms
    - Persistent Q-table storage
    - Advanced training statistics
    """
    
    def __init__(self, learning_rate: float = 0.1, decay: float = 0.99, 
                 epsilon: float = 0.1, epsilon_min: float = 0.01, 
                 epsilon_decay: float = 0.995, q_path: str = None):
        """
        Initialize enhanced RL Agent V2
        
        Args:
            learning_rate: How fast Q-values are updated (0.0-1.0)
            decay: How fast non-selected actions decay (0.0-1.0)
            epsilon: Initial exploration rate (0.0-1.0)
            epsilon_min: Minimum epsilon value
            epsilon_decay: Epsilon decay rate per update
            q_path: Path to Q-table JSON file for persistence
        """
        self.lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_path = q_path or "cache/rl_agent_v2_qtable.json"
        
        # Q-table: state → [score_no_alert, score_alert]
        self.q_table = {}
        
        # Training statistics
        self.training_stats = {
            "total_updates": 0,
            "exploration_actions": 0,
            "exploitation_actions": 0,
            "states_explored": 0,
            "average_reward": 0.0,
            "last_trained": None
        }
        
        # Load existing Q-table if available
        self.load_q_table()
        
        logger.info(f"[RL AGENT V2] Initialized with lr={learning_rate}, decay={decay}, "
                   f"epsilon={epsilon}, min_epsilon={epsilon_min}")
    
    def get_action(self, state: tuple, use_exploration: bool = True) -> int:
        """
        Zwraca akcję: 0 = brak alertu, 1 = alert
        Uses epsilon-greedy exploration strategy
        
        Args:
            state: State tuple (gnn_score, whale_clip_conf, dex_inflow)
            use_exploration: Whether to use epsilon-greedy exploration
            
        Returns:
            Action: 0 (no alert) or 1 (send alert)
        """
        # Convert state to string for JSON compatibility
        state_key = self._state_to_key(state)
        
        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0]  # [no_alert_value, alert_value]
            self.training_stats["states_explored"] += 1
        
        # Epsilon-greedy exploration
        if use_exploration and random.random() < self.epsilon:
            # Exploration: random action
            action = random.choice([0, 1])
            self.training_stats["exploration_actions"] += 1
            logger.debug(f"[RL V2] Exploration action: {action} (epsilon: {self.epsilon:.3f})")
        else:
            # Exploitation: greedy action based on Q-values
            q_values = self.q_table[state_key]
            action = int(q_values[1] > q_values[0])  # 1 if alert value > no alert value
            self.training_stats["exploitation_actions"] += 1
            logger.debug(f"[RL V2] Greedy action: {action}, Q-values: {q_values}")
        
        return action
    
    def update(self, state: tuple, action: int, reward: float):
        """
        Aktualizuje Q-table z decay i learning rate
        Updates Q-table with advanced learning mechanisms
        
        Args:
            state: State tuple
            action: Action taken (0 or 1)
            reward: Reward received
        """
        state_key = self._state_to_key(state)
        
        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0]
        
        # Q-learning update with learning rate
        old_value = self.q_table[state_key][action]
        new_value = (1 - self.lr) * old_value + self.lr * reward
        self.q_table[state_key][action] = new_value
        
        # Apply decay to the other action (reduces its influence)
        other_action = 1 - action
        self.q_table[state_key][other_action] *= self.decay
        
        # Update epsilon (exploration decay)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update training statistics
        self.training_stats["total_updates"] += 1
        current_avg = self.training_stats["average_reward"]
        n = self.training_stats["total_updates"]
        self.training_stats["average_reward"] = ((n-1) * current_avg + reward) / n
        self.training_stats["last_trained"] = datetime.now().isoformat()
        
        logger.info(f"[RL V2 UPDATE] State: {state}, Action: {action}, "
                   f"Reward: {reward:.2f}, Q: {old_value:.3f} → {new_value:.3f}, "
                   f"Epsilon: {self.epsilon:.3f}")
    
    def predict_with_confidence(self, state: tuple) -> Dict[str, Any]:
        """
        Make prediction with confidence scoring
        
        Args:
            state: State tuple
            
        Returns:
            Dictionary with prediction details
        """
        action = self.get_action(state, use_exploration=False)  # No exploration for prediction
        state_key = self._state_to_key(state)
        
        q_values = self.q_table.get(state_key, [0.0, 0.0])
        
        # Calculate confidence based on Q-value difference
        confidence = abs(q_values[1] - q_values[0]) / (max(abs(q_values[1]), abs(q_values[0]), 1.0))
        
        # Calculate value advantage
        value_advantage = q_values[1] - q_values[0]
        
        prediction = {
            'state': state,
            'action': action,
            'should_alert': bool(action),
            'action_type': 'ALERT' if action == 1 else 'HOLD',
            'confidence': round(confidence, 3),
            'value_advantage': round(value_advantage, 3),
            'q_values': [round(q, 3) for q in q_values],
            'epsilon': round(self.epsilon, 3),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug(f"[RL V2 PREDICT] {state} → Action: {action}, "
                    f"Confidence: {confidence:.3f}, Advantage: {value_advantage:.3f}")
        
        return prediction
    
    def batch_update(self, experiences: List[Tuple[tuple, int, float]]):
        """
        Batch update from multiple experiences
        
        Args:
            experiences: List of (state, action, reward) tuples
        """
        logger.info(f"[RL V2] Starting batch update with {len(experiences)} experiences")
        
        for state, action, reward in experiences:
            self.update(state, action, reward)
        
        logger.info(f"[RL V2] Batch update complete. Total updates: {self.training_stats['total_updates']}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics
        
        Returns:
            Dictionary with training metrics
        """
        stats = self.training_stats.copy()
        stats.update({
            'q_table_size': len(self.q_table),
            'states_count': len(self.q_table),  # Added for compatibility
            'current_epsilon': round(self.epsilon, 3),
            'exploration_rate': round(
                self.training_stats['exploration_actions'] / 
                max(self.training_stats['total_updates'], 1), 3
            ),
            'top_q_states': self._get_top_q_states(5)
        })
        
        return stats
    
    def save(self, path: str = None):
        """
        Save Q-table and statistics to file
        
        Args:
            path: Optional custom path for saving
        """
        save_path = path or self.q_path
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Prepare data for saving
            save_data = {
                'q_table': self.q_table,
                'training_stats': self.training_stats,
                'agent_config': {
                    'learning_rate': self.lr,
                    'decay': self.decay,
                    'epsilon': self.epsilon,
                    'initial_epsilon': self.initial_epsilon,
                    'epsilon_min': self.epsilon_min,
                    'epsilon_decay': self.epsilon_decay
                },
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'version': '2.0'
                }
            }
            
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"[RL V2] Saved Q-table to {save_path}")
            logger.info(f"   • Q-table size: {len(self.q_table)} states")
            logger.info(f"   • Total updates: {self.training_stats['total_updates']}")
            
        except Exception as e:
            logger.error(f"[RL V2] Failed to save Q-table to {save_path}: {e}")
    
    def load_q_table(self, path: str = None):
        """
        Load Q-table and statistics from file
        
        Args:
            path: Optional custom path for loading
        """
        load_path = path or self.q_path
        
        try:
            if not os.path.exists(load_path):
                logger.info(f"[RL V2] No existing Q-table found at {load_path}")
                return
            
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Load Q-table
            self.q_table = data.get('q_table', {})
            
            # Load training statistics
            if 'training_stats' in data:
                self.training_stats.update(data['training_stats'])
            
            # Load agent configuration
            if 'agent_config' in data:
                config = data['agent_config']
                self.epsilon = config.get('epsilon', self.epsilon)
                # Note: Don't override lr, decay from constructor
            
            logger.info(f"[RL V2] Loaded Q-table from {load_path}")
            logger.info(f"   • Q-table size: {len(self.q_table)} states")
            logger.info(f"   • Total updates: {self.training_stats['total_updates']}")
            logger.info(f"   • Current epsilon: {self.epsilon:.3f}")
            
        except Exception as e:
            logger.error(f"[RL V2] Failed to load Q-table from {load_path}: {e}")
    
    def reset_exploration(self, new_epsilon: float = None):
        """
        Reset exploration rate for continued learning
        
        Args:
            new_epsilon: New epsilon value (uses initial if None)
        """
        old_epsilon = self.epsilon
        self.epsilon = new_epsilon or self.initial_epsilon
        
        logger.info(f"[RL V2] Reset exploration: {old_epsilon:.3f} → {self.epsilon:.3f}")
    
    def _state_to_key(self, state: tuple) -> str:
        """
        Convert state tuple to string key for JSON compatibility
        
        Args:
            state: State tuple
            
        Returns:
            String representation of state
        """
        return f"{state[0]},{state[1]},{state[2]}"
    
    def _key_to_state(self, key: str) -> tuple:
        """
        Convert string key back to state tuple
        
        Args:
            key: String key
            
        Returns:
            State tuple
        """
        parts = key.split(',')
        return (float(parts[0]), float(parts[1]), int(parts[2]))
    
    def _get_top_q_states(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N states by Q-value advantage
        
        Args:
            n: Number of top states to return
            
        Returns:
            List of top states with their Q-values
        """
        if not self.q_table:
            return []
        
        # Calculate advantages for all states
        state_advantages = []
        for state_key, q_values in self.q_table.items():
            advantage = q_values[1] - q_values[0]  # alert_value - no_alert_value
            state_advantages.append({
                'state': self._key_to_state(state_key),
                'state_key': state_key,
                'q_values': q_values,
                'advantage': advantage
            })
        
        # Sort by advantage and return top N
        state_advantages.sort(key=lambda x: x['advantage'], reverse=True)
        return state_advantages[:n]

def test_rl_agent_v2():
    """Test RLAgentV2 functionality"""
    print("🧠 RL AGENT V2 TEST")
    print("=" * 40)
    
    try:
        # Initialize agent
        agent = RLAgentV2(
            learning_rate=0.1,
            decay=0.99,
            epsilon=0.3,  # Higher exploration for testing
            q_path="cache/test_rl_v2.json"
        )
        
        # Test prediction
        test_state = (0.75, 0.85, 1)  # High GNN, high whale, DEX inflow
        prediction = agent.predict_with_confidence(test_state)
        print(f"✅ Prediction: {prediction['action']} (confidence: {prediction['confidence']})")
        
        # Test training
        experiences = [
            (test_state, 1, 1.5),  # Successful alert
            ((0.3, 0.2, 0), 0, 0.0),  # Correct no-alert
            ((0.9, 0.8, 1), 1, 2.0),  # Very successful alert
            ((0.4, 0.3, 0), 1, -1.0)  # False alert
        ]
        
        agent.batch_update(experiences)
        print(f"✅ Batch training: {len(experiences)} experiences processed")
        
        # Test statistics
        stats = agent.get_training_statistics()
        print(f"✅ Training stats: {stats['total_updates']} updates, "
              f"exploration rate: {stats['exploration_rate']}")
        
        # Test save/load
        agent.save()
        print(f"✅ Q-table saved: {len(agent.q_table)} states")
        
        # Test prediction after training
        new_prediction = agent.predict_with_confidence(test_state)
        print(f"✅ Updated prediction: {new_prediction['action']} "
              f"(confidence: {new_prediction['confidence']}, "
              f"advantage: {new_prediction['value_advantage']})")
        
        return True
        
    except Exception as e:
        print(f"❌ RL AGENT V2 TEST ERROR: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test
        success = test_rl_agent_v2()
        print(f"\n🎯 TEST RESULT: {'✅ PASS' if success else '❌ FAIL'}")
    else:
        # Example usage
        print("🧠 RL AGENT V2 EXAMPLE USAGE")
        print("=" * 40)
        
        # Initialize agent
        agent = RLAgentV2(q_path="cache/rl_agent_v2_qtable.json", epsilon=0.1)
        
        # Make prediction
        state = (0.67, 0.91, 1)  # gnn_score, whale_clip, dex_inflow
        action = agent.get_action(state)
        print(f"📊 State: {state} → Action: {action}")
        
        # Update with reward
        reward = 1.0  # Successful outcome
        agent.update(state, action, reward)
        print(f"🎯 Updated with reward: {reward}")
        
        # Save progress
        agent.save()
        print(f"💾 Progress saved")