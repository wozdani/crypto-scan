#!/usr/bin/env python3
"""
Reinforcement Learning Agent - Samouczący się agent do predykcji pumpów
Agent learns from GNN anomaly predictions and real market outcomes

Rewards system based on actual market behavior:
- Price +5% after 1h = +1 reward
- No significant movement = 0 reward  
- Price drop >3% = -1 reward
"""

import torch
import json
import os
import time
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLAgent:
    """
    Reinforcement Learning Agent for pump prediction based on GNN anomaly scores.
    Uses Q-learning with simple Q-table for state-action value estimation.
    """
    
    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.1, 
                 decay_rate: float = 0.99, q_table_file: str = "cache/q_table.json"):
        """
        Initialize RL Agent with Q-learning parameters.
        
        Args:
            learning_rate: Learning rate for Q-table updates
            epsilon: Exploration rate for epsilon-greedy strategy
            decay_rate: Decay rate for epsilon over time
            q_table_file: File path for persistent Q-table storage
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.q_table_file = q_table_file
        
        # Q-table: state → [score_no_alert, score_alert]
        self.q_table = {}
        
        # Experience tracking
        self.experiences = []
        self.prediction_history = {}
        
        # Load existing Q-table if available
        self.load_q_table()
        
        logger.info(f"[RL AGENT] Initialized with lr={learning_rate}, epsilon={epsilon}")
    
    def _discretize_state(self, anomaly_scores: List[float]) -> tuple:
        """
        Convert continuous anomaly scores to discrete state representation.
        
        Args:
            anomaly_scores: List of anomaly scores from GNN [0, 1]
            
        Returns:
            Discretized state tuple for Q-table lookup
        """
        # Discretize scores into bins for state representation
        def discretize_score(score: float, bins: int = 10) -> int:
            return min(int(score * bins), bins - 1)
        
        # Take top 3 anomaly scores for state (limit state space)
        top_scores = sorted(anomaly_scores, reverse=True)[:3]
        
        # Pad with zeros if less than 3 scores
        while len(top_scores) < 3:
            top_scores.append(0.0)
        
        # Discretize each score
        discrete_state = tuple(discretize_score(score) for score in top_scores)
        
        return discrete_state
    
    def get_action(self, state: tuple, use_epsilon_greedy: bool = True) -> int:
        """
        Zwraca 0 = brak alertu, 1 = alert
        
        Args:
            state: Discretized state tuple
            use_epsilon_greedy: Whether to use exploration
            
        Returns:
            Action: 0 (no alert) or 1 (send alert)
        """
        # Initialize state if not seen before
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]  # [no_alert_value, alert_value]
        
        # Epsilon-greedy exploration
        if use_epsilon_greedy and torch.rand(1).item() < self.epsilon:
            action = torch.randint(0, 2, (1,)).item()
            logger.debug(f"[RL AGENT] Exploration action: {action}")
        else:
            # Greedy action (exploit)
            q_values = torch.tensor(self.q_table[state])
            action = torch.argmax(q_values).item()
            logger.debug(f"[RL AGENT] Greedy action: {action}, Q-values: {q_values}")
        
        return action
    
    def update(self, state: tuple, action: int, reward: float):
        """
        Aktualizuje Q-table dla danego stanu i akcji
        
        Args:
            state: State tuple
            action: Action taken (0 or 1)
            reward: Reward received (+1, 0, -1)
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        
        # Q-learning update: Q(s,a) = Q(s,a) + lr * (reward - Q(s,a))
        old_value = self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * (reward - old_value)
        
        logger.info(f"[RL UPDATE] State: {state}, Action: {action}, "
                   f"Reward: {reward:.2f}, Q: {old_value:.3f} → {self.q_table[state][action]:.3f}")
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.decay_rate)
    
    def predict_alert(self, anomaly_scores: List[float], symbol: str = None) -> Dict[str, Any]:
        """
        Main prediction function combining anomaly scores with RL decision.
        
        Args:
            anomaly_scores: List of anomaly scores from GNN
            symbol: Token symbol for tracking
            
        Returns:
            Dict with prediction results
        """
        # Convert to discrete state
        state = self._discretize_state(anomaly_scores)
        
        # Get RL action
        action = self.get_action(state)
        
        # Calculate confidence based on Q-values
        q_values = self.q_table.get(state, [0.0, 0.0])
        confidence = abs(q_values[1] - q_values[0]) / (max(abs(q_values[1]), abs(q_values[0]), 1.0))
        
        prediction = {
            'symbol': symbol,
            'state': state,
            'action': action,
            'should_alert': bool(action),
            'confidence': round(confidence, 3),
            'q_values': q_values,
            'top_anomaly_scores': sorted(anomaly_scores, reverse=True)[:3],
            'timestamp': datetime.now().isoformat()
        }
        
        # Store prediction for later reward calculation
        if symbol:
            self.prediction_history[symbol] = prediction
        
        logger.info(f"[RL PREDICT] {symbol}: Action={action}, Confidence={confidence:.3f}, "
                   f"State={state}")
        
        return prediction
    
    def calculate_reward(self, price_before: float, price_after: float, 
                        time_elapsed: float) -> float:
        """
        Calculate reward based on price movement after prediction.
        
        Args:
            price_before: Price when prediction was made
            price_after: Price after time_elapsed
            time_elapsed: Time elapsed in hours
            
        Returns:
            Reward value: +1 (pump), 0 (no movement), -1 (dump)
        """
        if price_before <= 0:
            return 0.0
        
        price_change_pct = ((price_after - price_before) / price_before) * 100
        
        # Reward system based on price movement
        if price_change_pct >= 5.0:  # Pump detected
            reward = +1.0
        elif price_change_pct <= -3.0:  # Dump detected
            reward = -1.0
        elif abs(price_change_pct) < 1.0:  # No significant movement
            reward = 0.0
        else:  # Small movements
            reward = price_change_pct / 10.0  # Proportional reward
        
        logger.info(f"[RL REWARD] Price change: {price_change_pct:.2f}% → Reward: {reward:.2f}")
        
        return reward
    
    def update_from_market_outcome(self, symbol: str, current_price: float) -> bool:
        """
        Update agent based on real market outcome for a previous prediction.
        
        Args:
            symbol: Token symbol
            current_price: Current market price
            
        Returns:
            True if update was performed, False if no prediction found
        """
        if symbol not in self.prediction_history:
            return False
        
        prediction = self.prediction_history[symbol]
        
        # Calculate time elapsed since prediction
        pred_time = datetime.fromisoformat(prediction['timestamp'])
        time_elapsed = (datetime.now() - pred_time).total_seconds() / 3600  # Hours
        
        # Only update if enough time has passed (min 1 hour)
        if time_elapsed < 1.0:
            return False
        
        # We need the price at prediction time - for now use placeholder
        # In real implementation, this would be stored with the prediction
        price_before = prediction.get('price_at_prediction', current_price)
        
        # Calculate reward
        reward = self.calculate_reward(price_before, current_price, time_elapsed)
        
        # Update Q-table
        self.update(prediction['state'], prediction['action'], reward)
        
        # Add to experience replay
        experience = {
            'symbol': symbol,
            'state': prediction['state'],
            'action': prediction['action'],
            'reward': reward,
            'price_change_pct': ((current_price - price_before) / price_before) * 100 if price_before > 0 else 0,
            'time_elapsed': time_elapsed,
            'timestamp': datetime.now().isoformat()
        }
        self.experiences.append(experience)
        
        # Remove from pending predictions
        del self.prediction_history[symbol]
        
        logger.info(f"[RL OUTCOME] {symbol}: Updated with reward {reward:.2f} "
                   f"after {time_elapsed:.1f}h")
        
        return True
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get current agent statistics and performance metrics.
        
        Returns:
            Dict with agent statistics
        """
        stats = {
            'q_table_size': len(self.q_table),
            'total_experiences': len(self.experiences),
            'pending_predictions': len(self.prediction_history),
            'current_epsilon': round(self.epsilon, 4),
            'learning_rate': self.learning_rate,
        }
        
        # Calculate recent performance
        recent_experiences = [exp for exp in self.experiences 
                            if (datetime.now() - datetime.fromisoformat(exp['timestamp'])).days < 7]
        
        if recent_experiences:
            avg_reward = sum(exp['reward'] for exp in recent_experiences) / len(recent_experiences)
            positive_rewards = sum(1 for exp in recent_experiences if exp['reward'] > 0)
            stats.update({
                'recent_avg_reward': round(avg_reward, 3),
                'recent_success_rate': round(positive_rewards / len(recent_experiences), 3),
                'recent_experiences': len(recent_experiences)
            })
        
        return stats
    
    def save_q_table(self):
        """Save Q-table to persistent storage."""
        try:
            os.makedirs(os.path.dirname(self.q_table_file), exist_ok=True)
            
            # Convert tuple keys to strings for JSON serialization
            serializable_q_table = {
                str(state): values for state, values in self.q_table.items()
            }
            
            data = {
                'q_table': serializable_q_table,
                'epsilon': self.epsilon,
                'experiences_count': len(self.experiences),
                'last_update': datetime.now().isoformat()
            }
            
            with open(self.q_table_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"[RL SAVE] Q-table saved: {len(self.q_table)} states")
            
        except Exception as e:
            logger.error(f"[RL SAVE] Error saving Q-table: {e}")
    
    def load_q_table(self):
        """Load Q-table from persistent storage."""
        try:
            if os.path.exists(self.q_table_file):
                with open(self.q_table_file, 'r') as f:
                    data = json.load(f)
                
                # Convert string keys back to tuples
                self.q_table = {
                    eval(state): values for state, values in data.get('q_table', {}).items()
                }
                
                self.epsilon = data.get('epsilon', self.initial_epsilon)
                
                logger.info(f"[RL LOAD] Q-table loaded: {len(self.q_table)} states, epsilon={self.epsilon:.4f}")
            else:
                logger.info(f"[RL LOAD] No existing Q-table found, starting fresh")
                
        except Exception as e:
            logger.error(f"[RL LOAD] Error loading Q-table: {e}")
            self.q_table = {}

def test_rl_agent():
    """Test RL Agent functionality"""
    print("\n🧪 Testing RL Agent...")
    
    # Initialize agent
    agent = RLAgent(learning_rate=0.1, epsilon=0.2)
    
    # Test prediction
    test_anomaly_scores = [0.12, 0.77, 0.85, 0.45, 0.23]
    prediction = agent.predict_alert(test_anomaly_scores, symbol="TESTUSDT")
    
    print(f"✅ Prediction: {prediction}")
    
    # Simulate market outcome after 1 hour
    # Test positive outcome (pump)
    agent.update_from_market_outcome("TESTUSDT", 1.05)  # 5% price increase
    
    # Test agent stats
    stats = agent.get_agent_stats()
    print(f"✅ Agent stats: {stats}")
    
    # Test Q-table persistence
    agent.save_q_table()
    
    # Test another prediction with same scores (should be influenced by learning)
    prediction2 = agent.predict_alert(test_anomaly_scores, symbol="TESTUSDT2")
    print(f"✅ Second prediction: {prediction2}")
    
    # Test different anomaly pattern
    low_anomaly_scores = [0.05, 0.12, 0.08]
    prediction3 = agent.predict_alert(low_anomaly_scores, symbol="NORMALUSDT")
    print(f"✅ Low anomaly prediction: {prediction3}")
    
    print("🎉 RL Agent test completed!")

if __name__ == "__main__":
    test_rl_agent()