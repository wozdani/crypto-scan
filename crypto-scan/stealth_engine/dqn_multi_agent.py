"""
Advanced Reinforcement Learning DQN System for Multi-Agent Crypto Intelligence
Allows agents (OnChainAgent, SentimentAgent, RiskAgent, PredictorAgent) to self-learn
by adjusting weights and thresholds based on feedback from past decisions/alerts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Learning parameters
DQN_STATE_SIZE = 6  # [consensus_score, market_volatility, past_reward, buy_votes_ratio, detector_confidence, time_factor]
DQN_ACTION_SIZE = 3  # 0: lower threshold/weight, 1: keep, 2: raise
DQN_MEMORY_SIZE = 5000
DQN_BATCH_SIZE = 64
DQN_LEARNING_RATE = 0.001
DQN_GAMMA = 0.95
DQN_EPSILON_START = 1.0
DQN_EPSILON_MIN = 0.01
DQN_EPSILON_DECAY = 0.995

# File paths
DQN_MODEL_PATH = "crypto-scan/cache/dqn_model.pth"
DQN_MEMORY_PATH = "crypto-scan/cache/dqn_memory.json"
DQN_CONFIG_PATH = "crypto-scan/cache/dqn_config.json"

@dataclass
class DQNExperience:
    """Single DQN learning experience"""
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    timestamp: str
    symbol: str
    detector_name: str

class DQN(nn.Module):
    """Deep Q-Network neural network model"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    """
    Advanced DQN Agent for Multi-Agent Crypto Intelligence
    🎯 SINGLETON PATTERN: Prevents double initialization and state drift issues
    """
    
    _instance = None
    _initialized_round_id = None
    
    @classmethod
    def instance(cls, *args, **kwargs):
        """
        🎯 SINGLETON INSTANCE: Get or create single instance to prevent double initialization
        """
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
            print(f"[SINGLETON] MultiAgent DQNAgent: Created new instance")
        else:
            print(f"[SINGLETON] MultiAgent DQNAgent: Reusing existing instance")
        return cls._instance
    
    @classmethod
    def reset_singleton(cls):
        """Reset singleton for testing or restart scenarios"""
        cls._instance = None
        cls._initialized_round_id = None
        print(f"[SINGLETON] MultiAgent DQNAgent: Singleton reset")
    
    @classmethod
    def mark_round_initialized(cls, round_id: str):
        """Mark current round as initialized to prevent double loading"""
        cls._initialized_round_id = round_id
        print(f"[SINGLETON] MultiAgent DQNAgent: Round {round_id} marked as initialized")
    
    @classmethod
    def is_round_initialized(cls, round_id: str) -> bool:
        """Check if current round is already initialized"""
        return cls._initialized_round_id == round_id
    
    def __init__(self, 
                 state_size: int = DQN_STATE_SIZE,
                 action_size: int = DQN_ACTION_SIZE,
                 gamma: float = DQN_GAMMA,
                 epsilon: float = DQN_EPSILON_START,
                 epsilon_min: float = DQN_EPSILON_MIN,
                 epsilon_decay: float = DQN_EPSILON_DECAY,
                 lr: float = DQN_LEARNING_RATE,
                 memory_size: int = DQN_MEMORY_SIZE,
                 batch_size: int = DQN_BATCH_SIZE):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)  # Target network for stability
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Training metrics
        self.training_loss = []
        self.episode_rewards = []
        self.total_steps = 0
        self.update_target_frequency = 100  # Update target network every 100 steps
        
        # Load existing model if available
        self.load_model()
        
        print(f"[DQN AGENT] Initialized with state_size={state_size}, action_size={action_size}")
        print(f"[DQN AGENT] Memory size: {memory_size}, Batch size: {batch_size}")
        print(f"[DQN AGENT] Epsilon: {epsilon:.3f}, Learning rate: {lr}")
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state vector
            deterministic: If True, always choose best action (no exploration)
            
        Returns:
            Action index (0: lower, 1: keep, 2: raise)
        """
        if not deterministic and np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.action_size)
            print(f"[DQN ACT] Exploration action: {action} (epsilon={self.epsilon:.3f})")
            return action
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
            
        print(f"[DQN ACT] Exploitation action: {action} (Q-values: {q_values.squeeze().tolist()})")
        return action
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool, 
                 symbol: str = "", detector_name: str = ""):
        """Store experience in replay buffer"""
        experience = DQNExperience(
            state=state.tolist(),
            action=action,
            reward=reward,
            next_state=next_state.tolist(),
            done=done,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            detector_name=detector_name
        )
        self.memory.append(experience)
        
        print(f"[DQN MEMORY] Stored experience: {symbol} | Action: {action} | Reward: {reward:.3f} | Buffer: {len(self.memory)}")
    
    def replay(self) -> Optional[float]:
        """
        Train the DQN using random batch from replay buffer
        
        Returns:
            Training loss if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample random batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        experiences = [self.memory[i] for i in batch]
        
        # Prepare batch tensors
        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.LongTensor([exp.action for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.BoolTensor([exp.done for exp in experiences])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network periodically
        self.total_steps += 1
        if self.total_steps % self.update_target_frequency == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        loss_value = loss.item()
        self.training_loss.append(loss_value)
        
        print(f"[DQN REPLAY] Loss: {loss_value:.6f} | Epsilon: {self.epsilon:.3f} | Steps: {self.total_steps}")
        
        return loss_value
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"[DQN TARGET] Updated target network at step {self.total_steps}")
    
    def save_model(self):
        """Save DQN model and training state"""
        try:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'total_steps': self.total_steps,
                'training_loss': self.training_loss[-100:],  # Keep last 100 losses
                'episode_rewards': self.episode_rewards[-50:],  # Keep last 50 episodes
            }, DQN_MODEL_PATH)
            
            # Save configuration
            config = {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'epsilon': self.epsilon,
                'total_steps': self.total_steps,
                'memory_size': len(self.memory),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(DQN_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"[DQN SAVE] Model saved successfully (steps: {self.total_steps}, epsilon: {self.epsilon:.3f})")
            
        except Exception as e:
            print(f"[DQN SAVE] Error saving model: {e}")
    
    def load_model(self):
        """Load DQN model and training state"""
        try:
            if os.path.exists(DQN_MODEL_PATH):
                checkpoint = torch.load(DQN_MODEL_PATH, map_location='cpu')
                
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.total_steps = checkpoint.get('total_steps', 0)
                self.training_loss = checkpoint.get('training_loss', [])
                self.episode_rewards = checkpoint.get('episode_rewards', [])
                
                print(f"[DQN LOAD] Model loaded successfully (steps: {self.total_steps}, epsilon: {self.epsilon:.3f})")
                
        except Exception as e:
            print(f"[DQN LOAD] Error loading model: {e}, starting fresh")

class MultiAgentDQNEnvironment:
    """
    Custom environment for Multi-Agent Crypto Intelligence
    Manages detector thresholds and weights based on DQN actions
    """
    
    def __init__(self, 
                 initial_threshold: float = 0.7,
                 min_threshold: float = 0.4,
                 max_threshold: float = 0.9,
                 weight_adjustment: float = 0.05):
        
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.weight_adjustment = weight_adjustment
        
        # Detector weights management
        self.detector_weights = {
            'CaliforniumWhale': 0.25,
            'DiamondWhale': 0.30,
            'WhaleCLIP': 0.20,
            'StealthEngine': 0.25
        }
        
        # Performance tracking
        self.performance_history = []
        self.current_episode = 0
        
        print(f"[DQN ENV] Initialized with threshold={initial_threshold}, weight_adj={weight_adjustment}")
    
    def apply_action(self, action: int, detector_name: str, current_score: float) -> Dict:
        """
        Apply DQN action to adjust detector parameters
        
        Args:
            action: 0=lower, 1=keep, 2=raise
            detector_name: Name of detector to adjust
            current_score: Current consensus score
            
        Returns:
            Dict with adjustment details
        """
        old_weight = self.detector_weights.get(detector_name, 0.25)
        old_threshold = self.threshold
        
        adjustment_info = {
            'action': action,
            'detector': detector_name,
            'old_weight': old_weight,
            'old_threshold': old_threshold,
            'new_weight': old_weight,
            'new_threshold': old_threshold,
            'adjustment_reason': ''
        }
        
        if action == 0:  # Lower
            # Lower detector weight if underperforming
            if current_score < 0.5:
                new_weight = max(0.05, old_weight - self.weight_adjustment)
                self.detector_weights[detector_name] = new_weight
                adjustment_info['new_weight'] = new_weight
                adjustment_info['adjustment_reason'] = 'lowered_weight_due_to_low_score'
            
            # Lower threshold for more sensitivity
            self.threshold = max(self.min_threshold, self.threshold - 0.05)
            adjustment_info['new_threshold'] = self.threshold
            
        elif action == 2:  # Raise
            # Raise detector weight if performing well
            if current_score > 0.7:
                new_weight = min(0.50, old_weight + self.weight_adjustment)
                self.detector_weights[detector_name] = new_weight
                adjustment_info['new_weight'] = new_weight
                adjustment_info['adjustment_reason'] = 'raised_weight_due_to_high_score'
            
            # Raise threshold for more selectivity
            self.threshold = min(self.max_threshold, self.threshold + 0.05)
            adjustment_info['new_threshold'] = self.threshold
        
        # Action 1 (keep) - no changes
        
        print(f"[DQN ENV] Action {action} applied to {detector_name}: weight {old_weight:.3f}→{adjustment_info['new_weight']:.3f}, threshold {old_threshold:.3f}→{adjustment_info['new_threshold']:.3f}")
        
        return adjustment_info
    
    def calculate_reward(self, 
                        action: int, 
                        consensus_decision: str, 
                        actual_outcome: Optional[str] = None,
                        price_change_pct: Optional[float] = None) -> float:
        """
        Calculate reward based on action effectiveness
        
        Args:
            action: DQN action taken
            consensus_decision: BUY/HOLD/AVOID decision
            actual_outcome: Actual market outcome (if available)
            price_change_pct: Price change percentage after decision
            
        Returns:
            Reward value (-1 to +1)
        """
        reward = 0.0
        
        # Base reward from price performance
        if price_change_pct is not None:
            if consensus_decision == 'BUY':
                if price_change_pct > 5.0:  # Good pump
                    reward += 1.0
                elif price_change_pct > 0:  # Small gain
                    reward += 0.3
                else:  # Loss
                    reward -= 0.8
            elif consensus_decision == 'HOLD':
                if abs(price_change_pct) < 2.0:  # Stable, good hold
                    reward += 0.2
                else:
                    reward -= 0.1
            elif consensus_decision == 'AVOID':
                if price_change_pct < -2.0:  # Good avoidance
                    reward += 0.5
                else:
                    reward -= 0.2
        
        # Action-specific adjustments
        if action == 0:  # Lower thresholds/weights
            # Reward if it led to catching more opportunities
            if consensus_decision == 'BUY' and price_change_pct and price_change_pct > 3.0:
                reward += 0.2
        elif action == 2:  # Raise thresholds/weights
            # Reward if it led to avoiding bad decisions
            if consensus_decision == 'AVOID' and price_change_pct and price_change_pct < 0:
                reward += 0.2
        
        # Clip reward to [-1, 1]
        reward = np.clip(reward, -1.0, 1.0)
        
        print(f"[DQN REWARD] Action {action}, Decision {consensus_decision}, Price Δ{price_change_pct}% → Reward: {reward:.3f}")
        
        return reward
    
    def get_state_vector(self, 
                        consensus_score: float,
                        market_volatility: float,
                        past_reward: float,
                        buy_votes_ratio: float,
                        detector_confidence: float,
                        time_factor: float = 0.5) -> np.ndarray:
        """
        Create state vector for DQN
        
        Args:
            consensus_score: Current consensus score (0-1)
            market_volatility: Market volatility measure (0-1)  
            past_reward: Previous episode reward (-1 to 1)
            buy_votes_ratio: Ratio of BUY votes (0-1)
            detector_confidence: Average detector confidence (0-1)
            time_factor: Time-based factor (0-1)
            
        Returns:
            State vector for DQN
        """
        state = np.array([
            np.clip(consensus_score, 0, 1),
            np.clip(market_volatility, 0, 1),
            np.clip(past_reward, -1, 1),
            np.clip(buy_votes_ratio, 0, 1),
            np.clip(detector_confidence, 0, 1),
            np.clip(time_factor, 0, 1)
        ], dtype=np.float32)
        
        return state

class DQNIntegrationManager:
    """
    Integration manager for DQN with existing multi-agent consensus system
    """
    
    def __init__(self):
        self.dqn_agent = DQNAgent.instance()  # SINGLETON PATTERN: Use singleton instance
        self.environment = MultiAgentDQNEnvironment()
        self.active = True
        self.last_state = None
        self.last_action = None
        self.episode_count = 0
        
        print("[DQN INTEGRATION] Advanced Reinforcement Learning system initialized")
        print(f"[DQN INTEGRATION] Ready for multi-agent consensus integration")
    
    def process_consensus_decision(self,
                                 consensus_score: float,
                                 consensus_decision: str,
                                 detector_votes: Dict[str, str],
                                 detector_scores: Dict[str, float],
                                 market_context: Dict) -> Dict:
        """
        Process consensus decision through DQN system
        
        Args:
            consensus_score: Final consensus score
            consensus_decision: BUY/HOLD/AVOID
            detector_votes: Individual detector votes
            detector_scores: Individual detector scores  
            market_context: Market volatility, volume etc.
            
        Returns:
            DQN adjustment information
        """
        if not self.active:
            return {}
        
        # Calculate state components
        buy_votes = sum(1 for vote in detector_votes.values() if vote == 'BUY')
        buy_votes_ratio = buy_votes / max(len(detector_votes), 1)
        
        detector_confidence = float(np.mean(list(detector_scores.values()))) if detector_scores else 0.0
        market_volatility = market_context.get('volatility', 0.5)
        past_reward = getattr(self, 'last_reward', 0.0)
        
        # Create state vector
        current_state = self.environment.get_state_vector(
            consensus_score=consensus_score,
            market_volatility=market_volatility,
            past_reward=past_reward,
            buy_votes_ratio=buy_votes_ratio,
            detector_confidence=detector_confidence,
            time_factor=0.5  # Can be adjusted based on time of day, market conditions
        )
        
        # Get DQN action
        action = self.dqn_agent.act(current_state)
        
        # Apply action to environment (adjust weights/thresholds)
        primary_detector = max(detector_scores.keys(), key=lambda k: detector_scores[k]) if detector_scores else 'StealthEngine'
        adjustment_info = self.environment.apply_action(action, primary_detector, consensus_score)
        
        # Store for next iteration
        self.last_state = current_state
        self.last_action = action
        
        # Prepare return information
        result = {
            'dqn_action': action,
            'dqn_state': current_state.tolist(),
            'adjustment_info': adjustment_info,
            'new_weights': self.environment.detector_weights.copy(),
            'new_threshold': self.environment.threshold,
            'dqn_epsilon': self.dqn_agent.epsilon,
            'dqn_steps': self.dqn_agent.total_steps
        }
        
        print(f"[DQN PROCESS] Processed consensus for {primary_detector}: action={action}, new_threshold={self.environment.threshold:.3f}")
        
        return result
    
    def update_with_feedback(self, 
                           symbol: str,
                           price_change_pct: float,
                           consensus_decision: str,
                           market_context: Dict):
        """
        Update DQN with feedback from market outcomes
        
        Args:
            symbol: Token symbol
            price_change_pct: Price change since decision
            consensus_decision: Original consensus decision
            market_context: Updated market context
        """
        if not self.active or self.last_state is None or self.last_action is None:
            return
        
        # Calculate reward
        reward = self.environment.calculate_reward(
            action=self.last_action,
            consensus_decision=consensus_decision,
            price_change_pct=price_change_pct
        )
        
        # Create next state (updated market conditions)
        next_volatility = market_context.get('volatility', 0.5)
        next_state = self.environment.get_state_vector(
            consensus_score=market_context.get('score', 0.5),
            market_volatility=next_volatility,
            past_reward=reward,
            buy_votes_ratio=market_context.get('buy_ratio', 0.5),
            detector_confidence=market_context.get('confidence', 0.5)
        )
        
        # Store experience in DQN memory
        self.dqn_agent.remember(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=next_state,
            done=False,  # Continuous learning
            symbol=symbol,
            detector_name='MultiAgent'
        )
        
        # Train DQN
        loss = self.dqn_agent.replay()
        
        # Update tracking
        self.last_reward = reward
        self.episode_count += 1
        
        # Save model periodically
        if self.episode_count % 50 == 0:
            self.dqn_agent.save_model()
        
        print(f"[DQN FEEDBACK] Updated for {symbol}: reward={reward:.3f}, loss={loss:.6f if loss else 'N/A'}")
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current detector weights adjusted by DQN"""
        return self.environment.detector_weights.copy()
    
    def get_current_threshold(self) -> float:
        """Get current consensus threshold adjusted by DQN"""
        return self.environment.threshold
    
    def save_state(self):
        """Save DQN agent state"""
        self.dqn_agent.save_model()
        print("[DQN INTEGRATION] State saved successfully")

# Global DQN integration instance
dqn_integration = None

def initialize_dqn_system():
    """Initialize global DQN integration system"""
    global dqn_integration
    if dqn_integration is None:
        dqn_integration = DQNIntegrationManager()
    return dqn_integration

def get_dqn_integration():
    """Get global DQN integration instance"""
    global dqn_integration
    if dqn_integration is None:
        dqn_integration = initialize_dqn_system()
    return dqn_integration

if __name__ == "__main__":
    # Test DQN system
    print("🧠 Testing Advanced DQN Multi-Agent System...")
    
    dqn_manager = DQNIntegrationManager()
    
    # Simulate consensus decision
    test_result = dqn_manager.process_consensus_decision(
        consensus_score=0.75,
        consensus_decision='BUY',
        detector_votes={'CaliforniumWhale': 'BUY', 'DiamondWhale': 'HOLD', 'StealthEngine': 'BUY'},
        detector_scores={'CaliforniumWhale': 0.8, 'DiamondWhale': 0.6, 'StealthEngine': 0.9},
        market_context={'volatility': 0.3, 'volume_ratio': 1.2}
    )
    
    print("Test result:", test_result)
    
    # Simulate feedback
    dqn_manager.update_with_feedback(
        symbol='TESTUSDT',
        price_change_pct=8.5,  # Good pump
        consensus_decision='BUY',
        market_context={'volatility': 0.4, 'score': 0.8, 'buy_ratio': 0.7, 'confidence': 0.8}
    )
    
    print("✅ DQN system test completed successfully")