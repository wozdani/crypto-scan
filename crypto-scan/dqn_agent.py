#!/usr/bin/env python3
"""
🧠 DQN AGENT - Deep Q-Network dla dynamicznej adaptacji thresholds i detector weights
Integracja z istniejącym RLAgentV3 dla zaawansowanego uczenia się systemu
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 🧠 DQN Model - Neural Network dla Q-values
class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# 🎯 DQN Agent - Główny agent uczący się
class DQNAgent:
    def __init__(self, 
                 state_size: int = 5, 
                 action_size: int = 7,  # threshold actions + weight adjustments
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        
        # Main network i Target network (Double DQN)
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Load environment variables
        self.threshold_min = float(os.getenv('THRESHOLD_MIN', '0.5'))
        self.threshold_max = float(os.getenv('THRESHOLD_MAX', '0.9'))
        self.current_threshold = 0.7  # Starting threshold
        
        # Experience tracking
        self.experience_file = "crypto-scan/cache/dqn_experience.jsonl"
        self.load_experience()
        
        print("[DQN AGENT] Initialized with Double DQN architecture")
        print(f"[DQN CONFIG] State size: {state_size}, Action size: {action_size}")
        print(f"[DQN THRESHOLDS] Min: {self.threshold_min}, Max: {self.threshold_max}")
    
    def get_state(self, score: float, volatility: float, past_rewards: List[float], 
                 detector_performance: Dict[str, float], market_trend: float) -> np.ndarray:
        """Konstruuje state vector dla DQN"""
        recent_reward = np.mean(past_rewards[-5:]) if past_rewards else 0.0
        avg_detector_perf = np.mean(list(detector_performance.values())) if detector_performance else 0.5
        
        state = np.array([
            score,                    # Current stealth score
            volatility,               # Market volatility
            recent_reward,            # Recent reward average
            avg_detector_perf,        # Detector performance
            market_trend              # Market trend indicator
        ], dtype=np.float32)
        
        return state
    
    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience w replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Save to persistent storage
        experience = {
            "timestamp": datetime.now().isoformat(),
            "state": state.tolist(),
            "action": action,
            "reward": reward,
            "next_state": next_state.tolist(),
            "done": done
        }
        
        try:
            with open(self.experience_file, 'a') as f:
                f.write(json.dumps(experience) + '\n')
        except Exception as e:
            print(f"[DQN ERROR] Failed to save experience: {e}")
    
    def replay(self) -> float:
        """Experience replay training"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample minibatch
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values z target network (Double DQN)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"[DQN TARGET] Updated target network at step {self.step_count}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def apply_action(self, action: int) -> Dict[str, float]:
        """Aplikuje action do systemu - adjust threshold lub weights"""
        changes = {}
        
        if action == 0:  # Lower threshold significantly
            old_threshold = self.current_threshold
            self.current_threshold = max(self.threshold_min, self.current_threshold - 0.1)
            changes['threshold'] = self.current_threshold
            print(f"[DQN ACTION] Lower threshold: {old_threshold:.3f} → {self.current_threshold:.3f}")
            
        elif action == 1:  # Lower threshold slightly
            old_threshold = self.current_threshold
            self.current_threshold = max(self.threshold_min, self.current_threshold - 0.05)
            changes['threshold'] = self.current_threshold
            print(f"[DQN ACTION] Lower threshold slightly: {old_threshold:.3f} → {self.current_threshold:.3f}")
            
        elif action == 2:  # Keep threshold
            changes['threshold'] = self.current_threshold
            print(f"[DQN ACTION] Keep threshold: {self.current_threshold:.3f}")
            
        elif action == 3:  # Raise threshold slightly
            old_threshold = self.current_threshold
            self.current_threshold = min(self.threshold_max, self.current_threshold + 0.05)
            changes['threshold'] = self.current_threshold
            print(f"[DQN ACTION] Raise threshold slightly: {old_threshold:.3f} → {self.current_threshold:.3f}")
            
        elif action == 4:  # Raise threshold significantly
            old_threshold = self.current_threshold
            self.current_threshold = min(self.threshold_max, self.current_threshold + 0.1)
            changes['threshold'] = self.current_threshold
            print(f"[DQN ACTION] Raise threshold: {old_threshold:.3f} → {self.current_threshold:.3f}")
            
        elif action == 5:  # Boost whale_ping weight
            changes['weight_boost'] = {'whale_ping': 1.2}
            print("[DQN ACTION] Boost whale_ping weight by 20%")
            
        elif action == 6:  # Boost dex_inflow weight
            changes['weight_boost'] = {'dex_inflow': 1.2}
            print("[DQN ACTION] Boost dex_inflow weight by 20%")
        
        return changes
    
    def calculate_reward(self, alert_outcome: bool, score: float, threshold: float, 
                        time_to_pump: Optional[float] = None) -> float:
        """Oblicza reward na podstawie alert outcome"""
        base_reward = 0.0
        
        if alert_outcome:  # Alert był trafny (pump occurred)
            # Higher reward dla alerts bliskich threshold
            threshold_bonus = 1.0 - abs(score - threshold) / threshold
            base_reward = 10.0 + (5.0 * threshold_bonus)
            
            # Time bonus - wcześniejsze wykrycie = wyższy reward
            if time_to_pump and time_to_pump > 0:
                time_bonus = max(0, (120 - time_to_pump) / 120)  # 2h = max bonus
                base_reward += 5.0 * time_bonus
            
            print(f"[DQN REWARD] Positive: {base_reward:.2f} (threshold_bonus: {threshold_bonus:.2f})")
            
        else:  # Alert był błędny (no pump)
            # Penalty za false positive, większy penalty dla wysokich scores
            score_penalty = score / threshold
            base_reward = -5.0 * score_penalty
            print(f"[DQN REWARD] Negative: {base_reward:.2f} (score_penalty: {score_penalty:.2f})")
        
        return base_reward
    
    def load_experience(self):
        """Load persistent experience z decision_registry"""
        try:
            decision_file = "crypto-scan/logs/decision_registry.jsonl"
            if os.path.exists(decision_file):
                count = 0
                with open(decision_file, 'r') as f:
                    for line in f:
                        if count >= 1000:  # Limit loaded experiences
                            break
                        try:
                            data = json.loads(line.strip())
                            # Convert do DQN format jeśli possible
                            if 'score' in data and 'outcome' in data:
                                # Simplified state reconstruction
                                state = np.array([
                                    data.get('score', 0.5),
                                    data.get('volatility', 0.2),
                                    data.get('past_reward', 0.0),
                                    0.5,  # avg detector performance placeholder
                                    0.0   # market trend placeholder
                                ], dtype=np.float32)
                                
                                # Dummy action and next_state
                                action = 2  # keep threshold
                                reward = 10.0 if data['outcome'] == 1 else -5.0
                                next_state = state.copy()
                                
                                self.memory.append((state, action, reward, next_state, True))
                                count += 1
                        except Exception:
                            continue
                
                print(f"[DQN EXPERIENCE] Loaded {count} historical experiences")
                
        except Exception as e:
            print(f"[DQN ERROR] Failed to load experience: {e}")
    
    def save_model(self, filepath: str = "crypto-scan/models/dqn_model.pth"):
        """Save DQN model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'current_threshold': self.current_threshold
        }, filepath)
        print(f"[DQN MODEL] Saved to {filepath}")
    
    def load_model(self, filepath: str = "crypto-scan/models/dqn_model.pth"):
        """Load DQN model"""
        try:
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.current_threshold = checkpoint['current_threshold']
            print(f"[DQN MODEL] Loaded from {filepath}")
        except Exception as e:
            print(f"[DQN ERROR] Failed to load model: {e}")

# 🌍 Custom Crypto Environment dla symulacji (bez gym dependency)
class CryptoThresholdEnv:
    def __init__(self):
        # Action space: 7 actions (threshold adjustments + weight boosts)
        self.action_space_n = 7
        
        # Observation space bounds: [score, volatility, past_reward, detector_perf, trend]
        self.obs_low = np.array([0.0, 0.0, -10.0, 0.0, -1.0])
        self.obs_high = np.array([5.0, 1.0, 10.0, 1.0, 1.0])
        
        self.current_step = 0
        self.max_steps = 100
        self.reset()
    
    def step(self, action):
        # Simulate environment response
        reward = 0.0
        
        # Mock reward calculation based on action
        if action == 0 and self.state[0] > 0.8:  # Lower threshold when score high
            reward = 5.0
        elif action == 4 and self.state[0] < 0.4:  # Raise threshold when score low
            reward = 3.0
        elif action == 2:  # Keep threshold
            reward = 1.0
        else:
            reward = -1.0
        
        # Add noise
        reward += np.random.normal(0, 0.5)
        
        # Update state (mock)
        self.state[2] = reward  # Update past reward
        self.current_step += 1
        
        done = self.current_step >= self.max_steps
        
        return self.state.copy(), reward, done, {}
    
    def reset(self):
        self.state = np.array([
            np.random.uniform(0.3, 1.2),  # score
            np.random.uniform(0.1, 0.5),  # volatility
            0.0,                          # past_reward
            np.random.uniform(0.4, 0.8),  # detector_performance
            np.random.uniform(-0.2, 0.2)  # market_trend
        ], dtype=np.float32)
        
        self.current_step = 0
        return self.state.copy()
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, State: {self.state}")

# 🎯 DQN Integration Class dla głównego systemu
class DQNCryptoIntegration:
    def __init__(self):
        self.agent = DQNAgent()
        self.env = CryptoThresholdEnv()
        self.training_active = True
        
        # Load saved model if exists
        model_path = "crypto-scan/models/dqn_model.pth"
        if os.path.exists(model_path):
            self.agent.load_model(model_path)
        
        print("[DQN INTEGRATION] Initialized and ready for adaptive learning")
    
    def process_alert_outcome(self, symbol: str, score: float, alert_sent: bool, 
                            outcome: bool, detector_performance: Dict[str, float],
                            market_data: Dict) -> Dict[str, float]:
        """Main integration point - process alert i update DQN"""
        
        # Extract state features
        volatility = market_data.get('volatility', 0.2)
        market_trend = market_data.get('trend', 0.0)
        past_rewards = getattr(self, 'recent_rewards', [])
        
        state = self.agent.get_state(
            score=score,
            volatility=volatility,
            past_rewards=past_rewards,
            detector_performance=detector_performance,
            market_trend=market_trend
        )
        
        # Get action from agent
        action = self.agent.act(state)
        
        # Calculate reward
        reward = self.agent.calculate_reward(
            alert_outcome=outcome,
            score=score,
            threshold=self.agent.current_threshold
        )
        
        # Store reward
        if not hasattr(self, 'recent_rewards'):
            self.recent_rewards = []
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 20:
            self.recent_rewards.pop(0)
        
        # Apply action i get changes
        changes = self.agent.apply_action(action)
        
        # Simulate next state
        next_state = state.copy()
        next_state[2] = reward  # Update past reward
        
        # Remember experience
        self.agent.remember(state, action, reward, next_state, done=True)
        
        # Train agent
        if self.training_active and len(self.agent.memory) >= self.agent.batch_size:
            loss = self.agent.replay()
            if loss > 0:
                print(f"[DQN TRAINING] Loss: {loss:.4f}, Epsilon: {self.agent.epsilon:.3f}")
        
        # Save model periodically
        if self.agent.step_count % 500 == 0:
            self.agent.save_model()
        
        return changes

if __name__ == "__main__":
    # Standalone test
    print("🧠 DQN Agent Standalone Test")
    
    integration = DQNCryptoIntegration()
    
    # Symuluj kilka alertów
    for i in range(10):
        outcome = np.random.choice([True, False], p=[0.3, 0.7])  # 30% success rate
        score = np.random.uniform(0.5, 1.5)
        
        detector_perf = {
            'whale_ping': np.random.uniform(0.4, 0.9),
            'dex_inflow': np.random.uniform(0.3, 0.8)
        }
        
        market_data = {
            'volatility': np.random.uniform(0.1, 0.5),
            'trend': np.random.uniform(-0.3, 0.3)
        }
        
        changes = integration.process_alert_outcome(
            symbol=f"TEST{i}USDT",
            score=score,
            alert_sent=True,
            outcome=outcome,
            detector_performance=detector_perf,
            market_data=market_data
        )
        
        print(f"Alert {i}: Score={score:.3f}, Outcome={outcome}, Changes={changes}")
    
    print(f"Final threshold: {integration.agent.current_threshold:.3f}")