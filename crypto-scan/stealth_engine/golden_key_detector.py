"""
Złoty Klucz - Golden Key GNN Detector
=====================================================

Zaawansowany system wykrywania pump-ów w krypto oparty na Graph Neural Networks (GNN)
do analizy grafów transakcji on-chain. Wykrywa ukryte akumulacje whales przed wzrostem ceny.

Używa Reinforcement Learning (RL) do samouczenia się - agent uczy się na rewards
(+1 za trafiony pump, -1 za false alert).

Autor: System TJDE v3.0+ Enhanced
Data: 27 lipca 2025
"""

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SimpleGCN(nn.Module):
    """
    Prosty Graph Convolutional Network do detekcji anomalii w grafach transakcji
    """
    def __init__(self, in_features: int, out_features: int):
        super(SimpleGCN, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, adj_matrix: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass GCN
        
        Args:
            adj_matrix: Macierz sąsiedztwa grafu (tensor)
            features: Cechy węzłów (tensor)
            
        Returns:
            Anomaly scores dla każdego węzła
        """
        # Graph convolution: A * X
        output = torch.matmul(adj_matrix, features)
        output = self.dropout(output)
        
        # Linear transformation + sigmoid
        return torch.sigmoid(self.linear(output))

class RLAgent:
    """
    Reinforcement Learning Agent do samouczenia się na wykrywaniu pump-ów
    Używa Q-Learning do optymalizacji decyzji alert/no-alert
    """
    
    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.1):
        self.q_table = {}  # State -> [no_alert_value, alert_value]
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.action_history = []  # Historia akcji dla analiz
        
    def get_state_key(self, state: torch.Tensor) -> tuple:
        """Konwertuj tensor state na klucz dla Q-table"""
        return tuple(np.round(state.flatten().detach().numpy(), 3).tolist())
    
    def get_action(self, state: torch.Tensor) -> int:
        """
        Wybierz akcję na podstawie epsilon-greedy policy
        
        Args:
            state: Stan z GNN (anomaly scores)
            
        Returns:
            0 = no alert, 1 = send alert
        """
        state_key = self.get_state_key(state)
        
        # Inicjalizuj state jeśli nie istnieje
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0]  # [no alert, alert]
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = int(np.random.choice([0, 1]))  # Random exploration
        else:
            # Exploit: wybierz najlepszą akcję
            action = int(np.argmax(self.q_table[state_key]))
        
        self.action_history.append({
            'timestamp': datetime.now().isoformat(),
            'state': state_key,
            'action': action,
            'q_values': self.q_table[state_key].copy()
        })
        
        return action
    
    def update(self, state: torch.Tensor, action: int, reward: float):
        """
        Aktualizuj Q-table na podstawie otrzymanej nagrody
        
        Args:
            state: Stan w którym podjęto akcję
            action: Podjęta akcja (0 lub 1)
            reward: Otrzymana nagroda (+1 za trafiony pump, -1 za false alert)
        """
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0]
        
        # Q-Learning update: Q(s,a) = Q(s,a) + α * reward
        old_value = self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * reward
        
        logger.debug(f"[RL UPDATE] State: {state_key[:3]}..., Action: {action}, "
                    f"Reward: {reward}, Q: {old_value:.3f} → {self.q_table[state_key][action]:.3f}")
    
    def save_model(self, filepath: str):
        """Zapisz model RL do pliku"""
        model_data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'action_history': self.action_history[-100:]  # Ostatnie 100 akcji
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """Wczytaj model RL z pliku"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.q_table = model_data.get('q_table', {})
            self.learning_rate = model_data.get('learning_rate', 0.1)
            self.epsilon = model_data.get('epsilon', 0.1)
            self.action_history = model_data.get('action_history', [])
            
            logger.info(f"[RL LOAD] Loaded model with {len(self.q_table)} states")

class GoldenKeyDetector:
    """
    Główna klasa Złoty Klucz - GNN Detector
    
    Integruje Graph Neural Network z Reinforcement Learning
    do wykrywania pump-ów w cryptocurrency markets
    """
    
    def __init__(self, config_path: str = "crypto-scan/cache"):
        self.config_path = config_path
        self.model_path = os.path.join(config_path, "golden_key_model.json")
        
        # Inicjalizuj GNN model
        self.gcn_model = SimpleGCN(in_features=3, out_features=1)  # 3 features per node
        
        # Inicjalizuj RL Agent
        self.rl_agent = RLAgent(learning_rate=0.1, epsilon=0.1)
        self.rl_agent.load_model(self.model_path)
        
        # Historia wykryć dla feedback learning
        self.detection_history = []
        
        logger.info(f"[GOLDEN KEY] Initialized GNN detector with RL agent")
    
    def build_transaction_graph(self, transactions: List[Dict]) -> Tuple[nx.DiGraph, Dict]:
        """
        Buduj graf transakcji z danych blockchain
        
        Args:
            transactions: Lista transakcji z blockchain
            
        Returns:
            Tuple[Graf NetworkX, mapowanie adresów]
        """
        G = nx.DiGraph()
        address_mapping = {}
        address_features = {}
        
        # Analiza transakcji do budowy grafu
        for tx in transactions:
            from_addr = tx.get('from_address', '0x0')
            to_addr = tx.get('to_address', '0x0')
            value_usd = float(tx.get('value_usd', 0))
            
            # Dodaj węzły (adresy)
            if from_addr not in address_features:
                address_features[from_addr] = {
                    'total_sent': 0,
                    'total_received': 0,
                    'tx_count': 0,
                    'whale_score': 0
                }
            
            if to_addr not in address_features:
                address_features[to_addr] = {
                    'total_sent': 0,
                    'total_received': 0,
                    'tx_count': 0,
                    'whale_score': 0
                }
            
            # Aktualizuj statystyki adresów
            address_features[from_addr]['total_sent'] = address_features[from_addr]['total_sent'] + value_usd
            address_features[from_addr]['tx_count'] = address_features[from_addr]['tx_count'] + 1
            address_features[to_addr]['total_received'] = address_features[to_addr]['total_received'] + value_usd
            address_features[to_addr]['tx_count'] = address_features[to_addr]['tx_count'] + 1
            
            # Wykryj whales (>$50k transactions)
            if value_usd > 50000:
                address_features[from_addr]['whale_score'] = address_features[from_addr]['whale_score'] + 1
                address_features[to_addr]['whale_score'] = address_features[to_addr]['whale_score'] + 1
            
            # Dodaj krawędź (transakcję)
            if G.has_edge(from_addr, to_addr):
                G[from_addr][to_addr]['weight'] += value_usd
                G[from_addr][to_addr]['count'] += 1
            else:
                G.add_edge(from_addr, to_addr, weight=value_usd, count=1)
        
        # Dodaj węzły do grafu z features
        for addr, features in address_features.items():
            G.add_node(addr, **features)
        
        logger.debug(f"[GOLDEN KEY] Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G, address_features
    
    def extract_graph_features(self, G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ekstraktuj features z grafu dla GNN
        
        Args:
            G: Graf NetworkX
            
        Returns:
            Tuple[macierz sąsiedztwa, features węzłów]
        """
        if G.number_of_nodes() == 0:
            # Empty graph fallback
            adj_matrix = torch.zeros(1, 1)
            node_features = torch.zeros(1, 3)
            return adj_matrix, node_features
        
        # Konwertuj do macierzy sąsiedztwa
        adj_matrix = nx.to_numpy_array(G, weight='weight')
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
        
        # Normalizacja macierzy sąsiedztwa (dodaj self-loops)
        adj_tensor += torch.eye(adj_tensor.size(0))
        
        # Ekstraktuj features węzłów
        nodes = list(G.nodes())
        node_features = []
        
        for node in nodes:
            attrs = G.nodes[node]
            features = [
                np.log1p(attrs.get('total_sent', 0)),      # Log total sent
                np.log1p(attrs.get('total_received', 0)),  # Log total received  
                attrs.get('whale_score', 0)                # Whale activity score
            ]
            node_features.append(features)
        
        features_tensor = torch.tensor(node_features, dtype=torch.float)
        
        logger.debug(f"[GOLDEN KEY] Extracted features: adj={adj_tensor.shape}, features={features_tensor.shape}")
        
        return adj_tensor, features_tensor
    
    def detect_pump_patterns(self, symbol: str, transactions: List[Dict]) -> Dict:
        """
        Główna funkcja wykrywania wzorców pump-ów
        
        Args:
            symbol: Symbol tokena (np. 'BTCUSDT')
            transactions: Lista transakcji blockchain
            
        Returns:
            Dict z wynikami detekcji
        """
        try:
            # Krok 1: Buduj graf transakcji
            G, address_features = self.build_transaction_graph(transactions)
            
            if G.number_of_nodes() < 2:
                return {
                    'status': 'insufficient_data',
                    'pump_score': 0.0,
                    'alert_decision': 0,
                    'anomaly_scores': [],
                    'reasoning': f'Insufficient graph nodes for analysis (need ≥2 nodes, got {G.number_of_nodes()})'
                }
            
            # Krok 2: Ekstraktuj features dla GNN
            adj_matrix, node_features = self.extract_graph_features(G)
            
            # Krok 3: Uruchom GNN dla detekcji anomalii
            with torch.no_grad():
                anomaly_scores = self.gcn_model(adj_matrix, node_features)
            
            # Krok 4: Analiza anomalii
            max_anomaly = float(anomaly_scores.max())
            mean_anomaly = float(anomaly_scores.mean())
            
            # Wykryj wzorce charakterystyczne dla pump-ów
            pump_indicators = self._analyze_pump_indicators(G, anomaly_scores)
            
            # Krok 5: RL Decision Making
            state = torch.tensor([max_anomaly, mean_anomaly, pump_indicators['whale_concentration']])
            alert_decision = self.rl_agent.get_action(state)
            
            # Oblicz finalne pump_score
            pump_score = (max_anomaly * 0.4 + 
                         mean_anomaly * 0.3 + 
                         pump_indicators['whale_concentration'] * 0.3)
            
            result = {
                'status': 'success',
                'pump_score': float(pump_score),
                'alert_decision': alert_decision,
                'anomaly_scores': anomaly_scores.flatten().tolist(),
                'pump_indicators': pump_indicators,
                'graph_stats': {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'max_anomaly': max_anomaly,
                    'mean_anomaly': mean_anomaly
                },
                'reasoning': self._generate_reasoning(pump_score, alert_decision, pump_indicators)
            }
            
            # Zapisz w historii dla learning
            self.detection_history.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'result': result
            })
            
            logger.info(f"[GOLDEN KEY] {symbol}: pump_score={pump_score:.3f}, "
                       f"alert={alert_decision}, nodes={G.number_of_nodes()}")
            
            return result
            
        except Exception as e:
            logger.error(f"[GOLDEN KEY ERROR] {symbol}: {e}")
            return {
                'status': 'error',
                'pump_score': 0.0,
                'alert_decision': 0,
                'anomaly_scores': [],
                'reasoning': f'GNN analysis failed: {str(e)}'
            }
    
    def _analyze_pump_indicators(self, G: nx.DiGraph, anomaly_scores: torch.Tensor) -> Dict:
        """
        Analizuj wskaźniki charakterystyczne dla pump-ów
        """
        indicators = {
            'whale_concentration': 0.0,
            'accumulation_pattern': 0.0,
            'network_centrality': 0.0,
            'volume_spike': 0.0
        }
        
        if G.number_of_nodes() == 0:
            return indicators
        
        # Koncentracja whale activity
        whale_nodes = [node for node in G.nodes() 
                      if G.nodes[node].get('whale_score', 0) > 0]
        indicators['whale_concentration'] = len(whale_nodes) / G.number_of_nodes()
        
        # Wzorce akumulacji (high in-degree nodes)
        in_degree_values = []
        for node in G.nodes():
            degree_val = G.in_degree(node, weight='weight')
            in_degree_values.append(float(degree_val))
        
        if in_degree_values:
            max_in_degree = max(in_degree_values)
            mean_in_degree = float(np.mean(in_degree_values))
            indicators['accumulation_pattern'] = max_in_degree / (mean_in_degree + 1)
        
        # Centralność sieci
        if G.number_of_nodes() > 1:
            centrality = nx.degree_centrality(G)
            indicators['network_centrality'] = max(centrality.values())
        
        # Volume spike detection
        edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
        if edge_weights:
            indicators['volume_spike'] = max(edge_weights) / (np.mean(edge_weights) + 1)
        
        return indicators
    
    def _generate_reasoning(self, pump_score: float, alert_decision: int, indicators: Dict) -> str:
        """
        Generuj uzasadnienie decyzji
        """
        if alert_decision == 1:
            reasons = []
            if pump_score > 0.7:
                reasons.append(f"High pump score ({pump_score:.3f})")
            if indicators['whale_concentration'] > 0.3:
                reasons.append(f"Whale concentration ({indicators['whale_concentration']:.3f})")
            if indicators['accumulation_pattern'] > 2.0:
                reasons.append(f"Accumulation pattern detected ({indicators['accumulation_pattern']:.3f})")
            
            return f"PUMP ALERT: {', '.join(reasons)}"
        else:
            return f"No significant pump pattern (score: {pump_score:.3f})"
    
    def update_feedback(self, symbol: str, actual_pump: bool):
        """
        Aktualizuj RL agent na podstawie rzeczywistego wyniku
        
        Args:
            symbol: Symbol tokena
            actual_pump: Czy rzeczywiście nastąpił pump
        """
        # Znajdź ostatnią detekcję dla tego symbolu
        recent_detection = None
        for detection in reversed(self.detection_history):
            if detection['symbol'] == symbol:
                recent_detection = detection
                break
        
        if recent_detection:
            alert_decision = recent_detection['result']['alert_decision']
            
            # Oblicz reward
            if actual_pump and alert_decision == 1:
                reward = 1.0  # Correctly predicted pump
            elif not actual_pump and alert_decision == 0:
                reward = 0.5  # Correctly avoided false alert
            elif actual_pump and alert_decision == 0:
                reward = -0.5  # Missed pump
            else:
                reward = -1.0  # False alert
            
            # Odtwórz state i zaktualizuj RL
            pump_score = recent_detection['result']['pump_score']
            indicators = recent_detection['result']['pump_indicators']
            state = torch.tensor([
                pump_score,
                indicators['whale_concentration'],
                indicators['accumulation_pattern']
            ])
            
            self.rl_agent.update(state, alert_decision, reward)
            
            logger.info(f"[GOLDEN KEY FEEDBACK] {symbol}: alert={alert_decision}, "
                       f"actual_pump={actual_pump}, reward={reward}")
    
    def save_model(self):
        """Zapisz model RL"""
        self.rl_agent.save_model(self.model_path)
        logger.info(f"[GOLDEN KEY] Model saved to {self.model_path}")
    
    def get_statistics(self) -> Dict:
        """
        Zwróć statystyki działania detektora
        """
        recent_detections = self.detection_history[-100:]  # Ostatnie 100
        
        if not recent_detections:
            return {'total_detections': 0}
        
        alerts_count = sum(1 for d in recent_detections 
                          if d['result']['alert_decision'] == 1)
        avg_pump_score = np.mean([d['result']['pump_score'] 
                                 for d in recent_detections])
        
        return {
            'total_detections': len(recent_detections),
            'alerts_generated': alerts_count,
            'alert_rate': alerts_count / len(recent_detections),
            'avg_pump_score': avg_pump_score,
            'q_table_size': len(self.rl_agent.q_table),
            'learning_rate': self.rl_agent.learning_rate,
            'epsilon': self.rl_agent.epsilon
        }

# Globalna instancja detektora
_golden_key_detector = None

def get_golden_key_detector() -> GoldenKeyDetector:
    """Singleton access do Golden Key detector"""
    global _golden_key_detector
    if _golden_key_detector is None:
        _golden_key_detector = GoldenKeyDetector()
    return _golden_key_detector

def run_golden_key_analysis(symbol: str, transactions: List[Dict]) -> Dict:
    """
    Główna funkcja do uruchomienia analizy Złoty Klucz
    
    Args:
        symbol: Symbol tokena
        transactions: Lista transakcji blockchain
        
    Returns:
        Dict z wynikami analizy GNN
    """
    detector = get_golden_key_detector()
    return detector.detect_pump_patterns(symbol, transactions)