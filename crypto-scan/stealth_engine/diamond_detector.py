"""
DiamondWhale AI - Temporal Graph + QIRL Detector
Advanced temporal analysis of blockchain transactions with Quantum-Inspired Reinforcement Learning

Wykrywa ukrytą akumulację whale'ów i wzorce prowadzące do pumpów przez analizę:
- Sekwencyjnych ruchów adresów w czasie
- Subgraph patterns i anomalie temporalne
- Subtelne wzorce niedające się ukryć przez layering/spoofing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network z pamięcią sekwencji
    Analizuje dynamiczny graf transakcji z uwzględnieniem czasu
    """
    
    def __init__(self, in_features: int, out_features: int, hidden_dim: int = 64):
        super(TemporalGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        
        # Spatial convolution layer
        self.spatial_conv = nn.Linear(in_features, hidden_dim)
        
        # Temporal LSTM layer for sequence modeling
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_features)
        
        # Attention mechanism for temporal focus
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        logger.info(f"[DIAMOND TGN] Initialized with {in_features}→{hidden_dim}→{out_features}")
    
    def forward(self, adj_matrix: torch.Tensor, features: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal graph network
        
        Args:
            adj_matrix: Adjacency matrix [N, N]
            features: Node features [N, in_features]
            timestamps: Transaction timestamps [N]
            
        Returns:
            Temporal anomaly scores [N, out_features]
        """
        try:
            # Spatial convolution: aggregate neighbor features
            spatial_features = torch.matmul(adj_matrix, features)  # [N, in_features]
            spatial_features = F.relu(self.spatial_conv(spatial_features))  # [N, hidden_dim]
            
            # Sort by timestamps for temporal sequence
            sorted_indices = torch.argsort(timestamps)
            sequential_features = spatial_features[sorted_indices].unsqueeze(0)  # [1, N, hidden_dim]
            
            # Temporal LSTM: capture sequence dependencies
            lstm_output, (hidden, cell) = self.temporal_lstm(sequential_features)
            
            # Self-attention for temporal focus
            attended_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
            
            # Restore original order
            restore_indices = torch.argsort(sorted_indices)
            temporal_features = attended_output.squeeze(0)[restore_indices]  # [N, hidden_dim]
            
            # Final output with sigmoid activation for anomaly scores
            anomaly_scores = torch.sigmoid(self.output_layer(temporal_features))
            
            return anomaly_scores
            
        except Exception as e:
            logger.error(f"[DIAMOND TGN] Forward pass error: {e}")
            # Return zero scores on error
            return torch.zeros(features.shape[0], self.out_features)


class QIRLAgent:
    """
    Quantum-Inspired Reinforcement Learning Agent
    Adaptacyjny agent do podejmowania decyzji o alertach na podstawie wzorców temporalnych
    """
    
    def __init__(self, state_size: int, action_size: int = 3, learning_rate: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size  # [ALERT, WATCH, IGNORE]
        self.learning_rate = learning_rate
        
        # Q-network with quantum-inspired architecture
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        
        # Quantum-inspired components
        self.quantum_weights = nn.Parameter(torch.randn(state_size, 16))
        self.phase_rotations = nn.Parameter(torch.randn(16))
        
        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + [self.quantum_weights, self.phase_rotations],
            lr=learning_rate
        )
        
        # Experience replay buffer
        self.memory = []
        self.memory_size = 1000
        
        # Statistics
        self.total_decisions = 0
        self.correct_predictions = 0
        self.action_counts = {0: 0, 1: 0, 2: 0}  # ALERT, WATCH, IGNORE
        
        logger.info(f"[DIAMOND QIRL] Initialized with state_size={state_size}, actions={action_size}")
    
    def quantum_encoding(self, state: torch.Tensor) -> torch.Tensor:
        """
        Quantum-inspired state encoding with phase rotations
        """
        # Quantum superposition encoding
        quantum_state = torch.matmul(state, self.quantum_weights)
        
        # Apply phase rotations
        phases = self.phase_rotations.unsqueeze(0).expand(quantum_state.shape[0], -1)
        quantum_encoded = quantum_state * torch.cos(phases) + torch.sin(phases)
        
        return quantum_encoded
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Epsilon-greedy action selection with quantum encoding
        
        Args:
            state: Current state features
            epsilon: Exploration rate
            
        Returns:
            Action index (0=ALERT, 1=WATCH, 2=IGNORE)
        """
        self.total_decisions += 1
        
        if np.random.random() < epsilon:
            # Exploration: random action
            action = np.random.randint(0, self.action_size)
        else:
            # Exploitation: best Q-value action
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            
            with torch.no_grad():
                # Quantum encoding
                quantum_features = self.quantum_encoding(state_tensor)
                
                # Standard Q-network
                q_values = self.q_network(state_tensor)
                
                # Combine quantum and classical features
                combined_features = torch.cat([state_tensor, quantum_features], dim=1)
                quantum_bias = torch.sum(combined_features, dim=1) * 0.1
                
                # Apply quantum bias to Q-values
                adjusted_q_values = q_values + quantum_bias.unsqueeze(1)
                action = torch.argmax(adjusted_q_values).item()
        
        self.action_counts[action] += 1
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray = None):
        """
        Update Q-network based on reward feedback
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received (+1.0 success, -1.0 failure, 0.0 neutral)
            next_state: New state (optional)
        """
        # Store experience in replay buffer
        experience = (state, action, reward, next_state)
        self.memory.append(experience)
        
        # Limit memory size
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        # Update statistics
        if reward > 0:
            self.correct_predictions += 1
        
        # Train on batch of experiences
        if len(self.memory) >= 32:
            self._train_batch()
        
        logger.debug(f"[DIAMOND QIRL] Updated: action={action}, reward={reward:.2f}")
    
    def _train_batch(self, batch_size: int = 32):
        """
        Train Q-network on batch of experiences
        """
        try:
            # Sample random batch
            batch_indices = np.random.choice(len(self.memory), min(batch_size, len(self.memory)), replace=False)
            batch = [self.memory[i] for i in batch_indices]
            
            states = torch.tensor([exp[0] for exp in batch], dtype=torch.float)
            actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long)
            rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float)
            
            # Current Q-values
            current_q_values = self.q_network(states)
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Target Q-values (simplified for immediate rewards)
            target_q = rewards
            
            # Loss calculation
            loss = F.mse_loss(current_q, target_q)
            
            # Quantum regularization
            quantum_reg = torch.sum(self.quantum_weights ** 2) * 0.001
            total_loss = loss + quantum_reg
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            logger.debug(f"[DIAMOND QIRL] Batch training: loss={total_loss:.4f}")
            
        except Exception as e:
            logger.error(f"[DIAMOND QIRL] Training error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent performance statistics
        """
        accuracy = (self.correct_predictions / self.total_decisions * 100) if self.total_decisions > 0 else 0.0
        
        return {
            "total_decisions": self.total_decisions,
            "accuracy": accuracy,
            "action_distribution": self.action_counts.copy(),
            "memory_size": len(self.memory),
            "exploration_ratio": self.action_counts.get(2, 0) / max(self.total_decisions, 1)
        }


class DiamondDetector:
    """
    Main DiamondWhale AI Detector
    Integruje Temporal GCN z QIRL Agent dla zaawansowanej detekcji whale patterns
    """
    
    def __init__(self, config_path: str = "cache/diamond_detector_config.json"):
        self.config_path = config_path
        self.model_path = "cache/diamond_detector_model.pth"
        self.agent_path = "cache/diamond_qirl_agent.pth"
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.tgn_model = None
        self.qirl_agent = None
        
        # Statistics tracking
        self.detections = 0
        self.alerts_sent = 0
        self.true_positives = 0
        self.false_positives = 0
        
        logger.info("[DIAMOND DETECTOR] Initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load detector configuration
        """
        default_config = {
            "tgn_features": 12,
            "tgn_output": 1,
            "tgn_hidden": 64,
            "qirl_state_size": 15,
            "qirl_learning_rate": 0.01,
            "alert_threshold": 0.7,
            "watch_threshold": 0.4,
            "min_transactions": 5,
            "time_window_hours": 24
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"[DIAMOND CONFIG] Loaded from {self.config_path}")
                return {**default_config, **config}
            except Exception as e:
                logger.warning(f"[DIAMOND CONFIG] Load error: {e}, using defaults")
        
        return default_config
    
    def _save_config(self):
        """
        Save current configuration
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"[DIAMOND CONFIG] Saved to {self.config_path}")
        except Exception as e:
            logger.error(f"[DIAMOND CONFIG] Save error: {e}")
    
    def analyze_transactions(self, transactions: List[Dict], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Main analysis function for transaction pattern detection
        
        Args:
            transactions: List of transaction dictionaries
            symbol: Token symbol for context
            
        Returns:
            Analysis result with anomaly scores and recommendations
        """
        try:
            logger.info(f"[DIAMOND ANALYSIS] Starting analysis for {symbol} ({len(transactions)} transactions)")
            
            if len(transactions) < self.config["min_transactions"]:
                logger.warning(f"[DIAMOND ANALYSIS] Insufficient transactions: {len(transactions)} < {self.config['min_transactions']}")
                return self._create_empty_result(symbol, "insufficient_data")
            
            # Build temporal graph
            graph_data = self._build_temporal_graph(transactions)
            if graph_data is None:
                return self._create_empty_result(symbol, "graph_build_failed")
            
            # Initialize models if needed
            if self.tgn_model is None:
                self._initialize_models(graph_data["features"].shape[1])
            
            # Run temporal GCN analysis
            anomaly_scores = self._run_tgn_analysis(graph_data)
            
            # Extract features for QIRL agent
            state_features = self._extract_state_features(graph_data, anomaly_scores)
            
            # Get QIRL agent decision
            action = self.qirl_agent.get_action(state_features)
            action_name = ["ALERT", "WATCH", "IGNORE"][action]
            
            # Calculate final diamond score
            diamond_score = float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else 0.0
            
            # Determine recommendation
            recommendation = self._make_recommendation(diamond_score, action)
            
            self.detections += 1
            if action == 0:  # ALERT
                self.alerts_sent += 1
            
            result = {
                "symbol": symbol,
                "diamond_score": diamond_score,
                "qirl_action": action,
                "qirl_action_name": action_name,
                "recommendation": recommendation,
                "anomaly_scores": anomaly_scores.tolist() if hasattr(anomaly_scores, 'tolist') else [],
                "graph_stats": {
                    "nodes": graph_data["features"].shape[0],
                    "edges": int(torch.sum(graph_data["adj_matrix"]).item()),
                    "time_span_hours": float((graph_data["timestamps"].max() - graph_data["timestamps"].min()).item() / 3600)
                },
                "temporal_patterns": self._analyze_temporal_patterns(graph_data),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[DIAMOND RESULT] {symbol}: score={diamond_score:.3f}, action={action_name}, recommendation={recommendation}")
            
            # 🚨 UNIFIED TELEGRAM ALERT SYSTEM - Stage 8/7
            # Wysyłaj alert dla DiamondWhale AI jeśli action == ALERT i score >= 0.70
            if action == 0 and diamond_score >= 0.70:  # ALERT action
                try:
                    from alerts.unified_telegram_alerts import send_stealth_alert
                    
                    # Przygotuj komentarz na podstawie recommendation
                    if recommendation == "STRONG_DIAMOND_ALERT":
                        comment = "Strong temporal whale pattern 💎📊"
                        action_suggestion = "IMMEDIATE WATCH & EVALUATE 🚀"
                    elif recommendation == "QIRL_SUGGESTED_ALERT":
                        comment = "QIRL-suggested temporal pattern 🧠"
                        action_suggestion = "Watch & Evaluate"
                    else:
                        comment = "Diamond temporal analysis pattern 💎"
                        action_suggestion = "Monitor closely 🔍"
                    
                    # Przygotuj dodatkowe dane
                    additional_data = {
                        "qirl_action": action_name,
                        "recommendation": recommendation,
                        "anomaly_scores": len(anomaly_scores),
                        "graph_nodes": result["graph_stats"]["nodes"],
                        "graph_edges": result["graph_stats"]["edges"],
                        "time_span_hours": f"{result['graph_stats']['time_span_hours']:.1f}h"
                    }
                    
                    # Wyślij unified alert
                    alert_sent = send_stealth_alert(
                        symbol, "DiamondWhale AI", diamond_score, 
                        comment, action_suggestion, additional_data
                    )
                    
                    if alert_sent:
                        logger.info(f"[UNIFIED ALERT] ✅ {symbol}: DiamondWhale alert sent (score: {diamond_score:.3f})")
                    else:
                        logger.info(f"[UNIFIED ALERT] ℹ️ {symbol}: Alert not sent (cooldown/config)")
                        
                except ImportError:
                    logger.warning(f"[UNIFIED ALERT] ⚠️ {symbol}: Unified alerts module not available")
                except Exception as alert_error:
                    logger.error(f"[UNIFIED ALERT] ❌ {symbol}: Alert error: {alert_error}")
            
            return result
            
        except Exception as e:
            logger.error(f"[DIAMOND ANALYSIS] Error analyzing {symbol}: {e}")
            return self._create_empty_result(symbol, f"analysis_error: {e}")
    
    def _build_temporal_graph(self, transactions: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build temporal graph from transactions
        """
        try:
            # Extract unique addresses
            addresses = set()
            for tx in transactions:
                addresses.add(tx.get('from_address', ''))
                addresses.add(tx.get('to_address', ''))
            
            address_list = list(filter(None, addresses))
            if len(address_list) < 2:
                return None
            
            # Create address mapping
            addr_to_idx = {addr: idx for idx, addr in enumerate(address_list)}
            n_nodes = len(address_list)
            
            # Build adjacency matrix
            adj_matrix = torch.zeros(n_nodes, n_nodes)
            
            # Extract features and timestamps
            node_features = []
            edge_timestamps = []
            
            for addr in address_list:
                # Node features: [total_value, tx_count, avg_value, first_seen, last_seen, ...]
                addr_txs = [tx for tx in transactions 
                           if tx.get('from_address') == addr or tx.get('to_address') == addr]
                
                total_value = sum(float(tx.get('value_usd', 0)) for tx in addr_txs)
                tx_count = len(addr_txs)
                avg_value = total_value / max(tx_count, 1)
                
                timestamps = [tx.get('timestamp', 0) for tx in addr_txs]
                first_seen = min(timestamps) if timestamps else 0
                last_seen = max(timestamps) if timestamps else 0
                
                # Additional features
                unique_counterparts = len(set(
                    tx.get('to_address' if tx.get('from_address') == addr else 'from_address', '')
                    for tx in addr_txs
                ))
                
                # Whale indicators
                is_whale = 1.0 if total_value > 100000 else 0.0  # $100k threshold
                activity_score = min(tx_count / 10.0, 1.0)  # Normalized activity
                
                # Temporal features
                time_span = last_seen - first_seen
                frequency = tx_count / max(time_span / 3600, 1)  # tx per hour
                
                features = [
                    total_value / 1000000,  # Normalized to millions
                    tx_count / 100,  # Normalized
                    avg_value / 10000,  # Normalized to 10k
                    first_seen / 1000000,  # Normalized timestamp
                    last_seen / 1000000,
                    unique_counterparts / 10,
                    is_whale,
                    activity_score,
                    frequency,
                    time_span / 86400,  # days
                    # Additional pattern features
                    len([tx for tx in addr_txs if float(tx.get('value_usd', 0)) > 1000]) / max(tx_count, 1),  # Large tx ratio
                    1.0 if any(tx.get('from_address') == addr for tx in addr_txs) else 0.0  # Is sender
                ]
                
                node_features.append(features)
            
            # Build edges and collect edge timestamps
            for tx in transactions:
                from_addr = tx.get('from_address', '')
                to_addr = tx.get('to_address', '')
                
                if from_addr in addr_to_idx and to_addr in addr_to_idx:
                    from_idx = addr_to_idx[from_addr]
                    to_idx = addr_to_idx[to_addr]
                    
                    # Add edge weight (transaction value)
                    weight = float(tx.get('value_usd', 0)) / 1000  # Normalized
                    adj_matrix[from_idx, to_idx] += weight
                    
                    edge_timestamps.append(tx.get('timestamp', 0))
            
            # Convert to tensors
            features_tensor = torch.tensor(node_features, dtype=torch.float)
            timestamps_tensor = torch.tensor(edge_timestamps + [0] * (n_nodes - len(edge_timestamps)), 
                                           dtype=torch.float)
            
            return {
                "adj_matrix": adj_matrix,
                "features": features_tensor,
                "timestamps": timestamps_tensor[:n_nodes],  # Ensure same length as nodes
                "address_mapping": addr_to_idx
            }
            
        except Exception as e:
            logger.error(f"[DIAMOND GRAPH] Build error: {e}")
            return None
    
    def _initialize_models(self, feature_dim: int):
        """
        Initialize TGN model and QIRL agent
        """
        try:
            # Initialize Temporal GCN
            self.tgn_model = TemporalGCN(
                in_features=feature_dim,
                out_features=self.config["tgn_output"],
                hidden_dim=self.config["tgn_hidden"]
            )
            
            # Initialize QIRL Agent
            self.qirl_agent = QIRLAgent(
                state_size=self.config["qirl_state_size"],
                learning_rate=self.config["qirl_learning_rate"]
            )
            
            # Load saved models if available
            self._load_models()
            
            logger.info("[DIAMOND MODELS] Initialized successfully")
            
        except Exception as e:
            logger.error(f"[DIAMOND MODELS] Initialization error: {e}")
    
    def _run_tgn_analysis(self, graph_data: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Run Temporal GCN analysis
        """
        try:
            with torch.no_grad():
                anomaly_scores = self.tgn_model(
                    graph_data["adj_matrix"],
                    graph_data["features"],
                    graph_data["timestamps"]
                )
                return anomaly_scores.numpy().flatten()
                
        except Exception as e:
            logger.error(f"[DIAMOND TGN] Analysis error: {e}")
            return np.array([0.0])
    
    def _extract_state_features(self, graph_data: Dict[str, torch.Tensor], anomaly_scores: np.ndarray) -> np.ndarray:
        """
        Extract state features for QIRL agent
        """
        try:
            features = graph_data["features"].numpy()
            
            # Aggregate features
            mean_anomaly = np.mean(anomaly_scores)
            max_anomaly = np.max(anomaly_scores)
            std_anomaly = np.std(anomaly_scores)
            
            # Graph statistics
            n_nodes = features.shape[0]
            total_value = np.sum(features[:, 0])  # First feature is total_value
            avg_activity = np.mean(features[:, 7])  # Activity score
            
            # Temporal features
            timestamps = graph_data["timestamps"].numpy()
            time_span = np.max(timestamps) - np.min(timestamps)
            
            # Whale indicators
            whale_count = np.sum(features[:, 6])  # Whale indicator
            large_tx_ratio = np.mean(features[:, 10])  # Large transaction ratio
            
            # Network topology
            adj_matrix = graph_data["adj_matrix"].numpy()
            edge_density = np.sum(adj_matrix > 0) / (n_nodes * n_nodes)
            
            state_features = np.array([
                mean_anomaly,
                max_anomaly,
                std_anomaly,
                total_value,
                avg_activity,
                time_span / 86400,  # Convert to days
                whale_count / max(n_nodes, 1),
                large_tx_ratio,
                edge_density,
                n_nodes / 100,  # Normalized node count
                np.mean(features[:, 1]),  # Average tx count
                np.mean(features[:, 2]),  # Average value
                np.mean(features[:, 5]),  # Average unique counterparts
                np.max(features[:, 8]),  # Max frequency
                np.mean(features[:, 9])   # Average time span
            ])
            
            return state_features
            
        except Exception as e:
            logger.error(f"[DIAMOND STATE] Feature extraction error: {e}")
            return np.zeros(self.config["qirl_state_size"])
    
    def _make_recommendation(self, diamond_score: float, qirl_action: int) -> str:
        """
        Make final recommendation based on diamond score and QIRL action
        """
        if qirl_action == 0:  # ALERT
            if diamond_score >= self.config["alert_threshold"]:
                return "STRONG_DIAMOND_ALERT"
            else:
                return "QIRL_SUGGESTED_ALERT"
        elif qirl_action == 1:  # WATCH
            return "DIAMOND_WATCH"
        else:  # IGNORE
            if diamond_score >= self.config["watch_threshold"]:
                return "DIAMOND_ONLY"
            else:
                return "NO_SIGNAL"
    
    def _analyze_temporal_patterns(self, graph_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the graph
        """
        try:
            timestamps = graph_data["timestamps"].numpy()
            features = graph_data["features"].numpy()
            
            # Time-based analysis
            time_range = np.max(timestamps) - np.min(timestamps)
            activity_peaks = []
            
            if time_range > 0:
                # Divide time into windows and analyze activity
                n_windows = min(10, int(time_range / 3600))  # Hourly windows
                if n_windows > 1:
                    for i in range(n_windows):
                        window_start = np.min(timestamps) + i * time_range / n_windows
                        window_end = window_start + time_range / n_windows
                        
                        window_activity = np.sum([
                            features[j, 1] for j in range(len(features))
                            if window_start <= timestamps[j] <= window_end
                        ])
                        
                        activity_peaks.append(window_activity)
            
            return {
                "time_span_hours": float(time_range / 3600),
                "activity_variance": float(np.var(activity_peaks)) if activity_peaks else 0.0,
                "peak_activity": float(np.max(activity_peaks)) if activity_peaks else 0.0,
                "consistent_activity": len(activity_peaks) > 0 and np.std(activity_peaks) < np.mean(activity_peaks) * 0.5
            }
            
        except Exception as e:
            logger.error(f"[DIAMOND TEMPORAL] Pattern analysis error: {e}")
            return {}
    
    def _create_empty_result(self, symbol: str, reason: str) -> Dict[str, Any]:
        """
        Create empty result for failed analysis
        """
        return {
            "symbol": symbol,
            "diamond_score": 0.0,
            "qirl_action": 2,  # IGNORE
            "qirl_action_name": "IGNORE",
            "recommendation": "NO_SIGNAL",
            "anomaly_scores": [],
            "graph_stats": {"nodes": 0, "edges": 0, "time_span_hours": 0.0},
            "temporal_patterns": {},
            "error_reason": reason,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_agent(self, symbol: str, action: int, outcome: float):
        """
        Update QIRL agent based on outcome feedback
        
        Args:
            symbol: Token symbol
            action: Action taken (0=ALERT, 1=WATCH, 2=IGNORE)
            outcome: Outcome score (+1.0 success, -1.0 failure, 0.0 neutral)
        """
        if self.qirl_agent is not None:
            # Use dummy state for update (in production, store actual state)
            dummy_state = np.zeros(self.config["qirl_state_size"])
            self.qirl_agent.update(dummy_state, action, outcome)
            
            # Update statistics
            if outcome > 0:
                self.true_positives += 1
            elif outcome < 0:
                self.false_positives += 1
            
            logger.info(f"[DIAMOND UPDATE] {symbol}: action={action}, outcome={outcome}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detector performance statistics
        """
        accuracy = (self.true_positives / max(self.alerts_sent, 1)) * 100
        qirl_stats = self.qirl_agent.get_statistics() if self.qirl_agent else {}
        
        return {
            "detections": self.detections,
            "alerts_sent": self.alerts_sent,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "accuracy_percent": accuracy,
            "qirl_agent": qirl_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    def _load_models(self):
        """
        Load saved models if available
        """
        try:
            if os.path.exists(self.model_path) and self.tgn_model is not None:
                self.tgn_model.load_state_dict(torch.load(self.model_path))
                logger.info(f"[DIAMOND MODELS] Loaded TGN from {self.model_path}")
        except Exception as e:
            logger.warning(f"[DIAMOND MODELS] TGN load error: {e}")
    
    def save_models(self):
        """
        Save current models
        """
        try:
            if self.tgn_model is not None:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save(self.tgn_model.state_dict(), self.model_path)
                logger.info(f"[DIAMOND MODELS] Saved TGN to {self.model_path}")
            
            self._save_config()
        except Exception as e:
            logger.error(f"[DIAMOND MODELS] Save error: {e}")


def run_diamond_detector(transactions: List[Dict], symbol: str = "UNKNOWN") -> Dict[str, Any]:
    """
    Main entry point for DiamondWhale AI detection
    
    Args:
        transactions: List of transaction dictionaries
        symbol: Token symbol for context
        
    Returns:
        Detection result with diamond score and recommendations
    """
    detector = DiamondDetector()
    result = detector.analyze_transactions(transactions, symbol)
    
    # Save models after each run
    detector.save_models()
    
    return result


def test_diamond_detector():
    """
    Test DiamondWhale AI Detector functionality
    """
    print("🧠 Testing DiamondWhale AI - Temporal Graph + QIRL Detector")
    
    # Create sample transaction data
    sample_transactions = [
        {
            "from_address": "0x1234567890abcdef",
            "to_address": "0xabcdef1234567890",
            "value_usd": 50000,
            "timestamp": 1640995200,  # 2022-01-01 00:00:00
        },
        {
            "from_address": "0xabcdef1234567890",
            "to_address": "0x9876543210fedcba",
            "value_usd": 75000,
            "timestamp": 1640995800,  # 10 minutes later
        },
        {
            "from_address": "0x9876543210fedcba",
            "to_address": "0x1111222233334444",
            "value_usd": 125000,
            "timestamp": 1640996400,  # 20 minutes later
        },
        {
            "from_address": "0x1111222233334444",
            "to_address": "0x5555666677778888",
            "value_usd": 200000,
            "timestamp": 1640997000,  # 30 minutes later
        },
        {
            "from_address": "0x5555666677778888",
            "to_address": "0x1234567890abcdef",
            "value_usd": 300000,
            "timestamp": 1640997600,  # 40 minutes later
        }
    ]
    
    # Run analysis
    result = run_diamond_detector(sample_transactions, "TESTUSDT")
    
    print(f"✅ Analysis complete:")
    print(f"   Diamond Score: {result['diamond_score']:.3f}")
    print(f"   QIRL Action: {result['qirl_action_name']}")
    print(f"   Recommendation: {result['recommendation']}")
    print(f"   Graph Stats: {result['graph_stats']}")
    print(f"   Temporal Patterns: {result['temporal_patterns']}")
    
    return result


if __name__ == "__main__":
    test_diamond_detector()