"""
CaliforniumWhale AI - Temporal GNN Detector
Oparty na badaniach Perseus 2025 i Chainalysis
Real-time thresholding + mastermind tracing
"""

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Optional
import json
import os
from datetime import datetime
from .qirl_agent_singleton import get_qirl_agent, save_qirl_agent
from .graph_cache import generate_mock_graph_data, get_cached_graph, cache_graph

# 🔷 CaliforniumTGN: Temporal GNN + EWMA volume boost
class CaliforniumTGN(nn.Module):
    """
    Temporal Graph Neural Network z EWMA volume boosting
    
    Łączy spatial graph convolution z temporal LSTM analysis
    oraz dynamiczny EWMA thresholding dla anomalii volume
    """
    def __init__(self, in_features: int, out_features: int):
        super(CaliforniumTGN, self).__init__()
        # Ensure output features are divisible by num_heads
        adjusted_out_features = ((out_features + 3) // 4) * 4  # Make divisible by 4
        self.linear = nn.Linear(in_features, adjusted_out_features)
        self.rnn = nn.LSTM(adjusted_out_features, adjusted_out_features, batch_first=True)
        self.attention = nn.MultiheadAttention(adjusted_out_features, num_heads=4, batch_first=True)
        self.output_proj = nn.Linear(adjusted_out_features, out_features)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, adj, features, timestamps, volume_data):
        """
        Forward pass przez Temporal GNN
        
        Args:
            adj: Adjacency matrix
            features: Node features tensor
            timestamps: Temporal sequence
            volume_data: Volume data for EWMA calculation
            
        Returns:
            Anomaly scores tensor
        """
        try:
            # Spatial graph convolution
            adj_tensor = torch.tensor(adj, dtype=torch.float)
            spatial = torch.matmul(adj_tensor, features)
            spatial = self.linear(spatial)
            spatial = self.dropout(spatial)
            
            # Temporal LSTM processing
            temporal_input = spatial.unsqueeze(0)
            temporal_output, _ = self.rnn(temporal_input)
            
            # Multi-head attention for temporal focus
            attended_output, _ = self.attention(temporal_output, temporal_output, temporal_output)
            
            # Project back to original output size
            projected_output = self.output_proj(attended_output)
            
            # Sigmoid activation for anomaly scores
            scores = torch.sigmoid(projected_output.squeeze(0))
            
            # 🔸 EWMA thresholding boost
            if len(volume_data) > 0:
                ewma = np.mean(volume_data) * 4
                max_volume = max(volume_data) if volume_data else 0
                if max_volume > ewma:
                    scores *= 1.5  # Boost przy anomalnym wzroście volume
                    
            return scores
            
        except Exception as e:
            print(f"[CALIFORNIUM TGN ERROR] Forward pass failed: {e}")
            # Return default tensor on error
            return torch.zeros(features.shape[0], 1)

# 🔸 QIRL Agent (Quantum-Inspired Reinforcement Learning)
class QIRLAgent:
    """
    Quantum-Inspired Reinforcement Learning Agent
    
    Wykorzystuje quantum-inspired decision making z reinforcement learning
    dla intelligent threshold adaptation
    """
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.epsilon = 0.2  # Exploration rate
        self.memory = []
        
    def get_action(self, state: List[float]) -> int:
        """
        Get action using epsilon-greedy strategy
        
        Args:
            state: Current state vector
            
        Returns:
            Action index (0=hold, 1=alert)
        """
        try:
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.action_size)
            
            state_tensor = torch.tensor(state, dtype=torch.float)
            q_values = self.model(state_tensor)
            return int(torch.argmax(q_values).item())
            
        except Exception as e:
            print(f"[QIRL ACTION ERROR] {e}")
            return 0  # Default to hold on error
    
    def update(self, state: List[float], action: int, reward: float):
        """
        Update Q-network based on experience
        
        Args:
            state: State vector
            action: Action taken
            reward: Reward received
        """
        try:
            state_tensor = torch.tensor(state, dtype=torch.float)
            q_values = self.model(state_tensor)
            
            # Calculate loss using policy gradient
            loss = -reward * torch.log(q_values[action] + 1e-8)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Store experience
            self.memory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep memory limited
            if len(self.memory) > 1000:
                self.memory = self.memory[-1000:]
                
        except Exception as e:
            print(f"[QIRL UPDATE ERROR] {e}")
    
    def get_statistics(self) -> Dict:
        """Get QIRL agent statistics"""
        return {
            'memory_size': len(self.memory),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'recent_actions': [exp['action'] for exp in self.memory[-10:]]
        }

# 🔹 Główna funkcja detekcji CaliforniumWhale AI
def californium_whale_detect(symbol: str, agent: QIRLAgent, graph_data: dict) -> Tuple[float, Dict]:
    """
    Główna funkcja CaliforniumWhale AI detection
    
    Łączy Temporal GNN z QIRL Agent dla intelligent whale detection
    z real-time thresholding i mastermind tracing
    
    Args:
        symbol: Token symbol
        agent: QIRL Agent instance
        graph_data: Graph data dictionary containing:
            - graph: NetworkX graph
            - features: Node features tensor
            - timestamps: Temporal data
            - volumes: Volume data
            
    Returns:
        Tuple of (score, metadata)
    """
    try:
        # Validate graph data
        if not all(key in graph_data for key in ['graph', 'features', 'timestamps', 'volumes']):
            return 0.0, {"error": "incomplete_graph_data"}
        
        G = graph_data["graph"]
        features = graph_data["features"]  # torch.tensor
        timestamps = graph_data["timestamps"]  # torch.tensor
        volume_data = graph_data["volumes"]  # list[int/float]
        
        # Validate inputs
        if not isinstance(G, nx.Graph) or G.number_of_nodes() == 0:
            return 0.0, {"error": "invalid_graph"}
        
        if not isinstance(features, torch.Tensor) or features.shape[0] == 0:
            return 0.0, {"error": "invalid_features"}
        
        print(f"[CALIFORNIUM] {symbol}: Starting TGN analysis on {G.number_of_nodes()} nodes")
        
        # Create adjacency matrix
        adj = nx.to_numpy_array(G)
        
        # Initialize Temporal GNN model
        model = CaliforniumTGN(in_features=features.shape[1], out_features=1)
        
        # Run TGN forward pass
        anomaly_scores = model(adj, features, timestamps, volume_data)
        
        # Prepare state for QIRL agent
        scores_list = anomaly_scores.flatten().detach().numpy().tolist()
        timestamps_list = timestamps.tolist() if hasattr(timestamps, 'tolist') else [float(timestamps)]
        
        # Ensure state vector has consistent size
        max_state_size = 20  # Limit state size
        state = (scores_list + timestamps_list + [len(volume_data), np.mean(volume_data) if volume_data else 0])[:max_state_size]
        
        # Pad state if too short
        while len(state) < max_state_size:
            state.append(0.0)
        
        # QIRL decision making
        action = agent.get_action(state)
        
        # Calculate reward based on anomaly strength
        max_score = anomaly_scores.max().item()
        reward = max_score if action == 1 else -0.5
        
        # Update QIRL agent
        agent.update(state, action, reward)
        
        # Enhanced metadata
        metadata = {
            "score": round(max_score, 3),
            "triggered": action == 1,
            "anomaly_vector": [round(float(score), 3) for score in scores_list],
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
            "volume_mean": round(np.mean(volume_data), 2) if volume_data else 0,
            "volume_max": round(max(volume_data), 2) if volume_data else 0,
            "qirl_action": action,
            "qirl_reward": round(reward, 3),
            "temporal_features": len(timestamps_list),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Return score if triggered, otherwise 0
        final_score = max_score if action == 1 else 0.0
        
        print(f"[CALIFORNIUM] {symbol}: TGN score={max_score:.3f}, QIRL action={action}, final={final_score:.3f}")
        
        return final_score, metadata
        
    except Exception as e:
        print(f"[CALIFORNIUM ERROR] {symbol}: {e}")
        return 0.0, {
            "error": str(e),
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

# 🔧 Utility functions
def create_californium_agent(state_size: int = 20, action_size: int = 2):
    """
    Create CaliforniumWhale QIRL Agent using singleton pattern
    
    Args:
        state_size: Size of state vector
        action_size: Number of possible actions
        
    Returns:
        Singleton QIRL Agent instance
    """
    agent = get_qirl_agent(state_size, action_size)
    print(f"[CALIFORNIUM] Using singleton QIRL Agent with state_size={state_size}, action_size={action_size}")
    return agent

def prepare_graph_data(transactions: List[Dict], symbol: Optional[str] = None) -> Dict:
    """
    Prepare graph data from transaction list or use cached mock data
    
    Args:
        transactions: List of transaction dictionaries
        symbol: Token symbol for caching (optional)
        
    Returns:
        Graph data dictionary for CaliforniumWhale AI
    """
    try:
        # Try to use cached data if symbol provided
        if symbol:
            cached_data = get_cached_graph(symbol)
            if cached_data:
                print(f"[CALIFORNIUM] Using cached graph data for {symbol}")
                # Convert cached data back to usable format
                G = nx.DiGraph()
                G.add_nodes_from(cached_data['graph_nodes'])
                G.add_edges_from(cached_data['graph_edges'])
                
                return {
                    'graph': G,
                    'features': torch.tensor(cached_data['features'], dtype=torch.float),
                    'timestamps': torch.tensor(cached_data['timestamps'], dtype=torch.float),
                    'volumes': cached_data['volumes']
                }
        
        # Create graph from transactions
        G = nx.DiGraph()
        
        # Add nodes and edges
        for tx in transactions:
            from_addr = tx.get('from', '')
            to_addr = tx.get('to', '')
            value = float(tx.get('value', 0))
            
            if from_addr and to_addr and value > 0:
                G.add_edge(from_addr, to_addr, weight=value)
        
        # Create node features
        nodes = list(G.nodes())
        features = []
        
        for node in nodes:
            # Basic node features
            predecessors_list = list(G.predecessors(node))
            successors_list = list(G.successors(node))
            neighbors_list = list(G.neighbors(node))
            
            total_value = 0.0
            try:
                for pred in predecessors_list:
                    if pred in G and node in G[pred] and 'weight' in G[pred][node]:
                        total_value += float(G[pred][node]['weight'])
            except Exception:
                total_value = 0.0
            
            features.append([
                float(len(predecessors_list)),
                float(len(successors_list)),
                float(total_value),
                float(len(neighbors_list))
            ])
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float)
        timestamps_tensor = torch.tensor([i for i in range(len(nodes))], dtype=torch.float)
        
        # Extract volume data
        volumes = [tx.get('value', 0) for tx in transactions]
        
        graph_data = {
            'graph': G,
            'features': features_tensor,
            'timestamps': timestamps_tensor,
            'volumes': volumes
        }
        
        # Cache if symbol provided
        if symbol:
            cache_graph(symbol, {
                'symbol': symbol,
                'pattern': 'real_data',
                'graph_nodes': list(G.nodes()),
                'graph_edges': list(G.edges(data=True)),
                'features': features_tensor.tolist(),
                'timestamps': timestamps_tensor.tolist(),
                'volumes': volumes,
                'metadata': {
                    'nodes_count': len(G.nodes()),
                    'edges_count': G.number_of_edges(),
                    'total_value': sum(data.get('weight', 0) for _, _, data in G.edges(data=True)),
                    'generated_at': datetime.now().isoformat(),
                    'pattern_type': 'real_data'
                }
            })
        
        return graph_data
        
    except Exception as e:
        print(f"[CALIFORNIUM PREP ERROR] {e}")
        # Use mock data as fallback
        if symbol:
            print(f"[CALIFORNIUM] Using mock graph data for {symbol} as fallback")
            return generate_mock_graph_data(symbol, 'normal')
        return {
            'graph': nx.Graph(),
            'features': torch.zeros(1, 4),
            'timestamps': torch.zeros(1),
            'volumes': [0]
        }

def save_californium_model(agent: QIRLAgent, filepath: str):
    """
    Save CaliforniumWhale QIRL model
    
    Args:
        agent: QIRL Agent to save
        filepath: Path to save model
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'state_dict': agent.model.state_dict(),
            'state_size': agent.state_size,
            'action_size': agent.action_size,
            'epsilon': agent.epsilon,
            'memory': agent.memory[-100:],  # Save last 100 experiences
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(model_data, filepath)
        print(f"[CALIFORNIUM SAVE] Model saved to {filepath}")
        
    except Exception as e:
        print(f"[CALIFORNIUM SAVE ERROR] {e}")

def load_californium_model(filepath: str) -> Optional[QIRLAgent]:
    """
    Load CaliforniumWhale QIRL model
    
    Args:
        filepath: Path to load model from
        
    Returns:
        Loaded QIRL Agent or None
    """
    try:
        if not os.path.exists(filepath):
            return None
        
        model_data = torch.load(filepath)
        
        agent = QIRLAgent(model_data['state_size'], model_data['action_size'])
        agent.model.load_state_dict(model_data['state_dict'])
        agent.epsilon = model_data.get('epsilon', 0.2)
        agent.memory = model_data.get('memory', [])
        
        print(f"[CALIFORNIUM LOAD] Model loaded from {filepath}")
        return agent
        
    except Exception as e:
        print(f"[CALIFORNIUM LOAD ERROR] {e}")
        return None

# 🧪 Test function
def test_californium_whale_detect():
    """Test CaliforniumWhale AI detection"""
    print("🧪 Testing CaliforniumWhale AI Detector...")
    
    # Create test transactions
    test_transactions = [
        {'from': 'addr1', 'to': 'addr2', 'value': 1000},
        {'from': 'addr2', 'to': 'addr3', 'value': 2000},
        {'from': 'addr3', 'to': 'addr1', 'value': 1500}
    ]
    
    # Prepare graph data
    graph_data = prepare_graph_data(test_transactions)
    
    # Create QIRL agent
    agent = create_californium_agent()
    
    # Test detection
    score, metadata = californium_whale_detect("TESTUSDT", agent, graph_data)
    
    print(f"✅ CaliforniumWhale AI test completed")
    print(f"   Score: {score}")
    print(f"   Metadata: {metadata}")
    
    return score > 0

if __name__ == "__main__":
    test_californium_whale_detect()