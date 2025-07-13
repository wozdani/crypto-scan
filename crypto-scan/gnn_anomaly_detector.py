#!/usr/bin/env python3
"""
GNN Anomaly Detector - Graph Neural Network dla wykrywania anomalii w transakcjach
Używa prostego Graph Convolution Layer do analizy wzorców transakcji blockchain

Model converts transaction graphs to GNN input and computes anomaly scores
for wallet addresses based on transaction patterns and network topology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGCN(nn.Module):
    """
    Simple Graph Convolution Network without torch_geometric dependency.
    Performs basic graph convolution using adjacency matrix multiplication.
    """
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super(SimpleGCN, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN layers.
        
        Args:
            adj: Adjacency matrix [N, N]
            features: Node features [N, in_features]
            
        Returns:
            Node embeddings [N, out_features]
        """
        # First GCN layer: adj @ features @ W1
        x = torch.matmul(adj, features)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer: adj @ x @ W2
        x = torch.matmul(adj, x)
        x = self.linear2(x)
        
        return x

class GNNAnomalyDetector:
    """
    GNN-based anomaly detector for blockchain transaction graphs.
    Analyzes wallet behavior patterns and network topology.
    """
    
    def __init__(self, hidden_dim: int = 16):
        self.hidden_dim = hidden_dim
        self.model = None
        self.node_mapping = None
        self.feature_scaler = None
        
    def _prepare_graph_data(self, graph: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Converts NetworkX graph to tensors for GNN processing.
        
        Args:
            graph: NetworkX directed graph with node attributes
            
        Returns:
            Tuple of (adjacency_matrix, node_features, node_list)
        """
        if graph.number_of_nodes() == 0:
            logger.warning("[GNN PREP] Empty graph provided")
            return torch.zeros((0, 0)), torch.zeros((0, 3)), []
        
        # Get node list and create mapping
        nodes = list(graph.nodes())
        self.node_mapping = {node: idx for idx, node in enumerate(nodes)}
        
        # Create adjacency matrix (add self-loops for stability)
        adj_matrix = nx.to_numpy_array(graph, nodelist=nodes, weight='weight')
        adj_matrix = adj_matrix + np.eye(len(nodes))  # Add self-loops
        
        # Normalize adjacency matrix (degree normalization)
        degree = np.sum(adj_matrix, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
        degree_matrix = np.diag(degree_inv_sqrt)
        adj_normalized = degree_matrix @ adj_matrix @ degree_matrix
        
        # Extract enhanced node features: [total_tx_volume, in_degree, out_degree, degree_ratio, value_per_tx]
        node_features = []
        for node in nodes:
            attrs = graph.nodes[node]
            in_deg = attrs.get('in_degree', 0)
            out_deg = attrs.get('out_degree', 0)
            total_val = attrs.get('total_value', 0.0)
            
            # Additional derived features for better anomaly detection
            total_degree = in_deg + out_deg
            degree_ratio = (out_deg - in_deg) / max(total_degree, 1)  # Balance indicator
            value_per_tx = total_val / max(total_degree, 1)  # Average transaction size
            
            features = [
                total_val,
                in_deg,
                out_deg,
                degree_ratio,  # -1 (only incoming) to +1 (only outgoing)
                value_per_tx   # Average value per transaction
            ]
            node_features.append(features)
        
        # Convert to tensors
        adj_tensor = torch.tensor(adj_normalized, dtype=torch.float32)
        features_tensor = torch.tensor(node_features, dtype=torch.float32)
        
        # Feature normalization
        if features_tensor.shape[0] > 0:
            # Log-scale for transaction volume to handle large values
            features_tensor[:, 0] = torch.log1p(features_tensor[:, 0])
            
            # Min-max normalization
            for i in range(features_tensor.shape[1]):
                col = features_tensor[:, i]
                if col.max() > col.min():
                    features_tensor[:, i] = (col - col.min()) / (col.max() - col.min())
        
        logger.info(f"[GNN PREP] Prepared graph: {len(nodes)} nodes, "
                   f"features shape: {features_tensor.shape}")
        
        return adj_tensor, features_tensor, nodes
    
    def _compute_anomaly_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores from node embeddings.
        
        Args:
            embeddings: Node embeddings from GNN
            
        Returns:
            Anomaly scores [0, 1] for each node
        """
        if embeddings.shape[0] <= 1:
            return torch.zeros(embeddings.shape[0])
        
        # Compute deviation from median embedding (more robust than mean)
        median_embedding = torch.median(embeddings, dim=0)[0]
        deviations = torch.norm(embeddings - median_embedding, dim=1)
        
        # Normalize deviations to [0, 1] using min-max scaling
        if deviations.max() > deviations.min():
            normalized_deviations = (deviations - deviations.min()) / (deviations.max() - deviations.min())
        else:
            normalized_deviations = torch.zeros_like(deviations)
        
        # Apply sigmoid with scaling to spread values across [0, 1]
        anomaly_scores = torch.sigmoid((normalized_deviations - 0.5) * 6)  # Scale factor for better distribution
        
        return anomaly_scores
    
    def detect_graph_anomalies(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Przekształca graf do GNN, liczy anomaly score dla każdego node'a.
        
        Args:
            graph: NetworkX directed graph with transaction data
            
        Returns:
            Dictionary {address: anomaly_score} where score is in [0, 1]
            Higher scores indicate more suspicious/anomalous behavior
        """
        logger.info(f"[GNN DETECT] Starting anomaly detection on graph with "
                   f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Prepare graph data
        adj_matrix, node_features, nodes = self._prepare_graph_data(graph)
        
        if len(nodes) == 0:
            return {}
        
        # Initialize model if not exists
        if self.model is None:
            input_dim = node_features.shape[1] if node_features.shape[0] > 0 else 5
            self.model = SimpleGCN(
                in_features=input_dim,
                hidden_features=self.hidden_dim,
                out_features=8  # Embedding dimension
            )
            logger.info(f"[GNN MODEL] Initialized GCN with {input_dim} input features")
        
        # Forward pass through GNN
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(adj_matrix, node_features)
            anomaly_scores = self._compute_anomaly_scores(embeddings)
        
        # Create result dictionary
        result = {}
        for i, node in enumerate(nodes):
            score = float(anomaly_scores[i])
            result[node] = round(score, 3)
        
        # Log statistics
        high_anomaly = sum(1 for score in result.values() if score > 0.7)
        medium_anomaly = sum(1 for score in result.values() if 0.4 <= score <= 0.7)
        
        logger.info(f"[GNN DETECT] Anomaly detection complete: "
                   f"{high_anomaly} high-risk, {medium_anomaly} medium-risk addresses")
        
        return result
    
    def get_top_anomalies(self, anomaly_scores: Dict[str, float], 
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top K most anomalous addresses.
        
        Args:
            anomaly_scores: Dictionary of {address: score}
            top_k: Number of top anomalies to return
            
        Returns:
            List of (address, score) tuples sorted by score descending
        """
        sorted_scores = sorted(anomaly_scores.items(), 
                              key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
    
    def classify_risk_level(self, score: float) -> str:
        """
        Classify anomaly score into risk levels.
        
        Args:
            score: Anomaly score [0, 1]
            
        Returns:
            Risk level string
        """
        if score >= 0.8:
            return "VERY_HIGH"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        else:
            return "NORMAL"

def detect_graph_anomalies(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Główna funkcja wykrywania anomalii GNN.
    
    Args:
        graph: Graf transakcji NetworkX
        
    Returns:
        Słownik {address: anomaly_score} gdzie score ∈ [0, 1]
    """
    detector = GNNAnomalyDetector()
    return detector.detect_graph_anomalies(graph)

def analyze_anomaly_patterns(anomaly_scores: Dict[str, float], 
                           graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze patterns in detected anomalies.
    
    Args:
        anomaly_scores: Results from anomaly detection
        graph: Original transaction graph
        
    Returns:
        Analysis results with patterns and statistics
    """
    detector = GNNAnomalyDetector()
    top_anomalies = detector.get_top_anomalies(anomaly_scores)
    
    # Analyze network properties of anomalous nodes
    anomaly_analysis = {
        'total_addresses': len(anomaly_scores),
        'high_risk_count': sum(1 for score in anomaly_scores.values() if score > 0.7),
        'medium_risk_count': sum(1 for score in anomaly_scores.values() if 0.4 <= score <= 0.7),
        'top_anomalies': top_anomalies,
        'risk_distribution': {}
    }
    
    # Risk level distribution
    for addr, score in anomaly_scores.items():
        risk_level = detector.classify_risk_level(score)
        anomaly_analysis['risk_distribution'][risk_level] = \
            anomaly_analysis['risk_distribution'].get(risk_level, 0) + 1
    
    # Network centrality of anomalous addresses
    if graph.number_of_nodes() > 0:
        centrality = nx.degree_centrality(graph)
        anomalous_centrality = []
        
        for addr, score in top_anomalies:
            if addr in centrality:
                anomalous_centrality.append({
                    'address': addr,
                    'anomaly_score': score,
                    'centrality': centrality[addr]
                })
        
        anomaly_analysis['centrality_analysis'] = anomalous_centrality
    
    return anomaly_analysis

def test_gnn_anomaly_detector():
    """Test GNN anomaly detection"""
    print("\n🧪 Testing GNN Anomaly Detector...")
    
    # Import graph builder for test
    from gnn_graph_builder import build_transaction_graph
    
    # Create test transactions with suspicious patterns
    test_transactions = [
        {"from": "0xNormal1", "to": "0xNormal2", "value": 1000},
        {"from": "0xNormal2", "to": "0xNormal3", "value": 1500},
        {"from": "0xSuspicious", "to": "0xNormal1", "value": 100000},  # Large unusual transfer
        {"from": "0xSuspicious", "to": "0xNormal2", "value": 95000},   # Another large transfer
        {"from": "0xSuspicious", "to": "0xNormal3", "value": 87000},   # Pattern of large outflows
        {"from": "0xExchange", "to": "0xNormal1", "value": 5000},
        {"from": "0xExchange", "to": "0xNormal2", "value": 4500},
        {"from": "0xExchange", "to": "0xSuspicious", "value": 200000}, # Large exchange transfer
    ]
    
    # Build graph
    graph = build_transaction_graph(test_transactions)
    
    # Detect anomalies
    anomaly_scores = detect_graph_anomalies(graph)
    print(f"✅ Anomaly scores: {anomaly_scores}")
    
    # Analyze patterns
    analysis = analyze_anomaly_patterns(anomaly_scores, graph)
    print(f"✅ Risk distribution: {analysis['risk_distribution']}")
    print(f"✅ Top anomalies: {analysis['top_anomalies']}")
    
    # Test individual classifier
    detector = GNNAnomalyDetector()
    for addr, score in anomaly_scores.items():
        risk_level = detector.classify_risk_level(score)
        print(f"  {addr}: {score:.3f} ({risk_level})")
    
    print("🎉 GNN Anomaly Detector test completed!")

if __name__ == "__main__":
    test_gnn_anomaly_detector()