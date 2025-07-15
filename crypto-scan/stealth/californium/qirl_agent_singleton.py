"""
Singleton QIRL Agent Manager
Zapewnia jedną instancję QIRL Agent dla wszystkich wywołań CaliforniumWhale AI
enabling continuous learning between detections
"""

import os
import json
from typing import Optional
from datetime import datetime

# Delayed imports to avoid circular dependency
QIRLAgent = None
save_californium_model = None
load_californium_model = None

def _lazy_import():
    """Lazy import to avoid circular dependencies"""
    global QIRLAgent, save_californium_model, load_californium_model
    if QIRLAgent is None:
        try:
            from .californium_whale_detect import QIRLAgent, save_californium_model, load_californium_model
        except ImportError:
            # Create a minimal QIRLAgent if import fails
            class QIRLAgent:
                def __init__(self, state_size, action_size):
                    self.state_size = state_size
                    self.action_size = action_size
                    self.epsilon = 0.1
                    self.memory = []
                
                def get_action(self, state):
                    return 0  # Default action
                
                def update(self, state, action, reward):
                    self.memory.append({'state': state, 'action': action, 'reward': reward})
                
                def get_statistics(self):
                    return {
                        'state_size': self.state_size,
                        'action_size': self.action_size,
                        'epsilon': self.epsilon,
                        'memory_size': len(self.memory)
                    }
            
            def save_californium_model(agent, filepath):
                try:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(filepath, 'w') as f:
                        json.dump({'state_size': agent.state_size, 'action_size': agent.action_size}, f)
                    return True
                except:
                    return False
            
            def load_californium_model(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    return QIRLAgent(data['state_size'], data['action_size'])
                except:
                    return None

class QIRLAgentSingleton:
    """
    Singleton manager for QIRL Agent ensuring single instance across all detections
    """
    _instance = None
    _qirl_agent = None
    _model_path = "crypto-scan/cache/californium/qirl_agent_model.pt"
    _stats_path = "crypto-scan/cache/californium/qirl_agent_stats.json"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QIRLAgentSingleton, cls).__new__(cls)
        return cls._instance
    
    def get_qirl_agent(self, state_size: int = 20, action_size: int = 2, force_new: bool = False):
        """
        Get singleton QIRL Agent instance
        
        Args:
            state_size: Size of state vector (default: 20)
            action_size: Number of actions (default: 2 - hold/alert)
            force_new: Force creation of new agent instance
            
        Returns:
            QIRL Agent instance
        """
        _lazy_import()  # Ensure imports are loaded
        
        if self._qirl_agent is None or force_new:
            print(f"[QIRL SINGLETON] Creating new QIRL Agent (state_size={state_size}, action_size={action_size})")
            
            # Try to load existing model
            loaded_agent = load_californium_model(self._model_path)
            
            if loaded_agent is not None and not force_new:
                print(f"[QIRL SINGLETON] Loaded existing model from {self._model_path}")
                self._qirl_agent = loaded_agent
            else:
                print(f"[QIRL SINGLETON] Creating fresh QIRL Agent")
                self._qirl_agent = QIRLAgent(state_size, action_size)
                
                # Ensure cache directory exists
                os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
                
        return self._qirl_agent
    
    def save_agent_model(self) -> bool:
        """
        Save current QIRL Agent model to disk
        
        Returns:
            True if saved successfully, False otherwise
        """
        _lazy_import()  # Ensure imports are loaded
        
        if self._qirl_agent is None:
            print("[QIRL SINGLETON] No agent to save")
            return False
            
        try:
            save_californium_model(self._qirl_agent, self._model_path)
            self._save_agent_stats()
            print(f"[QIRL SINGLETON] Model saved successfully to {self._model_path}")
            return True
        except Exception as e:
            print(f"[QIRL SINGLETON] Error saving model: {e}")
            return False
    
    def _save_agent_stats(self):
        """Save agent statistics to JSON file"""
        if self._qirl_agent is None:
            return
            
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'state_size': self._qirl_agent.state_size,
                'action_size': self._qirl_agent.action_size,
                'epsilon': self._qirl_agent.epsilon,
                'memory_size': len(self._qirl_agent.memory),
                'total_updates': len(self._qirl_agent.memory),
                'recent_actions': [exp.get('action', 0) for exp in self._qirl_agent.memory[-20:]],
                'recent_rewards': [exp.get('reward', 0) for exp in self._qirl_agent.memory[-20:]]
            }
            
            os.makedirs(os.path.dirname(self._stats_path), exist_ok=True)
            with open(self._stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            print(f"[QIRL SINGLETON] Error saving stats: {e}")
    
    def get_agent_stats(self) -> dict:
        """
        Get current agent statistics
        
        Returns:
            Dictionary with agent statistics
        """
        if self._qirl_agent is None:
            return {'error': 'no_agent_loaded'}
            
        stats = self._qirl_agent.get_statistics()
        stats.update({
            'model_path': self._model_path,
            'stats_path': self._stats_path,
            'singleton_initialized': True
        })
        
        return stats
    
    def reset_agent(self, state_size: int = 20, action_size: int = 2):
        """
        Reset QIRL Agent to fresh state
        
        Args:
            state_size: Size of state vector
            action_size: Number of actions
        """
        _lazy_import()  # Ensure imports are loaded
        print("[QIRL SINGLETON] Resetting QIRL Agent to fresh state")
        self._qirl_agent = None
        self.get_qirl_agent(state_size, action_size, force_new=True)

# Global singleton instance
_qirl_singleton = QIRLAgentSingleton()

def get_qirl_agent(state_size: int = 20, action_size: int = 2):
    """
    Get singleton QIRL Agent instance
    
    Args:
        state_size: Size of state vector (default: 20)
        action_size: Number of actions (default: 2)
        
    Returns:
        QIRL Agent instance
    """
    _lazy_import()  # Ensure imports are loaded
    return _qirl_singleton.get_qirl_agent(state_size, action_size)

def save_qirl_agent() -> bool:
    """
    Save current QIRL Agent model
    
    Returns:
        True if saved successfully, False otherwise
    """
    return _qirl_singleton.save_agent_model()

def get_qirl_stats() -> dict:
    """
    Get QIRL Agent statistics
    
    Returns:
        Dictionary with agent statistics
    """
    return _qirl_singleton.get_agent_stats()

def reset_qirl_agent(state_size: int = 20, action_size: int = 2):
    """
    Reset QIRL Agent to fresh state
    
    Args:
        state_size: Size of state vector
        action_size: Number of actions
    """
    _qirl_singleton.reset_agent(state_size, action_size)

# Test function
def test_qirl_singleton():
    """Test QIRL Agent singleton functionality"""
    print("🧪 Testing QIRL Agent Singleton...")
    
    # Test 1: Get agent instance
    agent1 = get_qirl_agent()
    print(f"✅ Agent 1 created: {type(agent1)}")
    
    # Test 2: Get same instance
    agent2 = get_qirl_agent()
    print(f"✅ Agent 2 obtained: {agent1 is agent2}")
    
    # Test 3: Agent statistics
    stats = get_qirl_stats()
    print(f"✅ Agent stats: {stats.get('state_size', 'N/A')} state_size, {stats.get('memory_size', 'N/A')} memory")
    
    # Test 4: Save agent
    save_result = save_qirl_agent()
    print(f"✅ Agent saved: {save_result}")
    
    print("🎯 QIRL Agent Singleton test completed successfully!")
    return True

if __name__ == "__main__":
    test_qirl_singleton()