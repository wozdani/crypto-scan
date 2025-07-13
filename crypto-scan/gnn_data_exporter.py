#!/usr/bin/env python3
"""
GNN Data Exporter for ML Training
Eksportuje dane z grafu + anomaly_scores + wynik do pliku .jsonl dla treningu ML
"""

import json
import os
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GNNDataExporter:
    """
    Eksporter danych GNN do treningu modeli uczenia maszynowego
    """
    
    def __init__(self, training_dir: str = "training_data"):
        """
        Initialize GNN Data Exporter
        
        Args:
            training_dir: Katalog dla danych treningowych
        """
        self.training_dir = training_dir
        self.dataset_file = os.path.join(training_dir, "gnn_pump_dataset.jsonl")
        self.metadata_file = os.path.join(training_dir, "gnn_dataset_metadata.json")
        
        # Ensure training directory exists
        os.makedirs(training_dir, exist_ok=True)
        
        # Initialize metadata if not exists
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Inicjalizuj metadata pliku jeśli nie istnieje"""
        if not os.path.exists(self.metadata_file):
            metadata = {
                "created": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat(),
                "total_samples": 0,
                "pump_samples": 0,
                "no_pump_samples": 0,
                "tokens_tracked": [],
                "feature_description": {
                    "nodes": "List of blockchain addresses in transaction graph",
                    "edges": "List of [from_addr, to_addr, value_usd] transactions",
                    "anomaly_scores": "Dict with anomaly scores per address from GNN",
                    "graph_stats": "Basic graph statistics (nodes, edges, total_value)",
                    "market_data": "Token price and volume at analysis time",
                    "label": "1 if pump occurred within prediction window, 0 otherwise"
                }
            }
            self._save_metadata(metadata)
    
    def _save_metadata(self, metadata: Dict):
        """Zapisz metadata do pliku"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> Dict:
        """Załaduj metadata z pliku"""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except:
            logger.error(f"Failed to load metadata from {self.metadata_file}")
            return {}
    
    def save_training_sample(self, 
                           graph: nx.Graph, 
                           anomaly_scores: Dict[str, float], 
                           token: str, 
                           pump_occurred: bool,
                           market_data: Optional[Dict] = None,
                           graph_stats: Optional[Dict] = None,
                           analysis_metadata: Optional[Dict] = None) -> bool:
        """
        Eksportuje snapshot grafu i wynik do pliku jsonl
        
        Args:
            graph: NetworkX graf transakcji
            anomaly_scores: Wyniki anomaly detection z GNN
            token: Symbol tokena
            pump_occurred: Czy wystąpił pump (True/False)
            market_data: Opcjonalne dane rynkowe
            graph_stats: Statystyki grafu
            analysis_metadata: Dodatkowe metadane analizy
            
        Returns:
            True jeśli zapisano pomyślnie, False w przeciwnym razie
        """
        try:
            # Prepare graph data
            nodes_data = []
            for node in graph.nodes(data=True):
                node_id = node[0]
                node_attrs = node[1] if len(node) > 1 else {}
                nodes_data.append({
                    "address": node_id,
                    "attributes": node_attrs
                })
            
            edges_data = []
            for edge in graph.edges(data=True):
                from_addr, to_addr = edge[0], edge[1]
                edge_attrs = edge[2] if len(edge) > 2 else {}
                
                edges_data.append({
                    "from": from_addr,
                    "to": to_addr,
                    "value_usd": edge_attrs.get('value', 0.0),
                    "attributes": edge_attrs
                })
            
            # Prepare training sample
            sample = {
                "timestamp": datetime.utcnow().isoformat(),
                "token": token,
                "graph": {
                    "nodes": nodes_data,
                    "edges": edges_data,
                    "node_count": len(nodes_data),
                    "edge_count": len(edges_data)
                },
                "anomaly_scores": anomaly_scores,
                "graph_stats": graph_stats or {},
                "market_data": market_data or {},
                "analysis_metadata": analysis_metadata or {},
                "label": int(pump_occurred),  # 1 = był pump, 0 = nie było pump
                "prediction_window": "1h"  # Domyślne okno predykcji
            }
            
            # Append to dataset file
            with open(self.dataset_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            # Update metadata
            self._update_metadata(token, pump_occurred)
            
            logger.info(f"[GNN EXPORT] Saved training sample for {token} (pump: {pump_occurred})")
            return True
            
        except Exception as e:
            logger.error(f"[GNN EXPORT] Failed to save training sample for {token}: {e}")
            return False
    
    def _update_metadata(self, token: str, pump_occurred: bool):
        """Aktualizuj metadata po dodaniu nowego sampla"""
        try:
            metadata = self._load_metadata()
            
            metadata["last_updated"] = datetime.utcnow().isoformat()
            metadata["total_samples"] = metadata.get("total_samples", 0) + 1
            
            if pump_occurred:
                metadata["pump_samples"] = metadata.get("pump_samples", 0) + 1
            else:
                metadata["no_pump_samples"] = metadata.get("no_pump_samples", 0) + 1
            
            # Track unique tokens
            tokens_tracked = metadata.get("tokens_tracked", [])
            if token not in tokens_tracked:
                tokens_tracked.append(token)
                metadata["tokens_tracked"] = tokens_tracked
            
            self._save_metadata(metadata)
            
        except Exception as e:
            logger.error(f"[GNN EXPORT] Failed to update metadata: {e}")
    
    def get_dataset_stats(self) -> Dict:
        """Pobierz statystyki dataset'u"""
        metadata = self._load_metadata()
        
        return {
            "total_samples": metadata.get("total_samples", 0),
            "pump_samples": metadata.get("pump_samples", 0),
            "no_pump_samples": metadata.get("no_pump_samples", 0),
            "unique_tokens": len(metadata.get("tokens_tracked", [])),
            "last_updated": metadata.get("last_updated", "Never"),
            "pump_ratio": metadata.get("pump_samples", 0) / max(metadata.get("total_samples", 1), 1)
        }
    
    def load_training_data(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Załaduj dane treningowe z pliku
        
        Args:
            limit: Maksymalna liczba sampli do załadowania
            
        Returns:
            Lista sampli treningowych
        """
        samples = []
        
        try:
            if not os.path.exists(self.dataset_file):
                logger.warning(f"[GNN EXPORT] Dataset file not found: {self.dataset_file}")
                return samples
            
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.error(f"[GNN EXPORT] Failed to parse line {i+1}: {e}")
                        continue
            
            logger.info(f"[GNN EXPORT] Loaded {len(samples)} training samples")
            
        except Exception as e:
            logger.error(f"[GNN EXPORT] Failed to load training data: {e}")
        
        return samples
    
    def create_detection_placeholder(self, token: str, price_at_analysis: float) -> bool:
        """
        Placeholder dla detekcji pump'a na podstawie wzrostu ceny
        
        Args:
            token: Symbol tokena
            price_at_analysis: Cena w momencie analizy
            
        Returns:
            True jeśli wykryto pump (placeholder logic)
        """
        # PLACEHOLDER: Docelowo sprawdzać wzrost ceny w ciągu 1h
        # Na razie używamy losowej logiki dla celów demonstracyjnych
        
        # Mogłaby tutaj być logika sprawdzająca:
        # 1. Cenę po 1h od analizy
        # 2. Wzrost > X% (np. 5-10%)
        # 3. Volume spike potwierdzający pump
        
        # Placeholder: losowe True/False dla różnorodności danych
        import random
        return random.choice([True, False])
    
    def export_scheduler_data(self, address: str, analysis_result: Dict) -> bool:
        """
        Eksportuj dane z schedulera GNN dla konkretnego adresu
        
        Args:
            address: Adres blockchain do analizy
            analysis_result: Wynik analizy z stealth_engine_advanced
            
        Returns:
            True jeśli eksport się powiódł
        """
        try:
            # Extract relevant data from scheduler analysis
            if "graph" not in analysis_result or "anomaly_scores" not in analysis_result:
                logger.warning(f"[GNN EXPORT] Incomplete analysis result for {address}")
                return False
            
            graph = analysis_result["graph"]
            anomaly_scores = analysis_result["anomaly_scores"]
            
            # For scheduler data, we track address behavior rather than token pumps
            # Label could be: suspicious activity detected (1) or normal (0)
            suspicious_activity = any(score > 0.7 for score in anomaly_scores.values())
            
            sample = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "address_monitoring",
                "address": address,
                "graph": {
                    "nodes": list(graph.nodes()),
                    "edges": [(u, v, d.get('value', 0)) for u, v, d in graph.edges(data=True)],
                    "node_count": len(graph.nodes()),
                    "edge_count": len(graph.edges())
                },
                "anomaly_scores": anomaly_scores,
                "rl_decision": analysis_result.get("rl_decision", {}),
                "alert_sent": analysis_result.get("alert_sent", False),
                "label": int(suspicious_activity),
                "prediction_type": "address_anomaly"
            }
            
            # Save to scheduler-specific dataset
            scheduler_file = os.path.join(self.training_dir, "gnn_scheduler_dataset.jsonl")
            with open(scheduler_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            logger.info(f"[GNN EXPORT] Saved scheduler data for {address[:10]}... (suspicious: {suspicious_activity})")
            return True
            
        except Exception as e:
            logger.error(f"[GNN EXPORT] Failed to export scheduler data for {address}: {e}")
            return False

def test_gnn_exporter():
    """Test GNN Data Exporter functionality"""
    print("🧪 Testing GNN Data Exporter...")
    
    try:
        # Initialize exporter
        exporter = GNNDataExporter()
        
        # Create test graph
        import networkx as nx
        test_graph = nx.DiGraph()
        test_graph.add_edge("0x123", "0x456", value=1000.0)
        test_graph.add_edge("0x456", "0x789", value=2000.0)
        
        # Test anomaly scores
        test_anomaly_scores = {
            "0x123": 0.2,
            "0x456": 0.8,  # High anomaly
            "0x789": 0.1
        }
        
        # Test market data
        test_market_data = {
            "price_usd": 0.5,
            "volume_24h": 1000000,
            "price_change_24h": 15.5
        }
        
        # Save test sample
        success = exporter.save_training_sample(
            graph=test_graph,
            anomaly_scores=test_anomaly_scores,
            token="TESTUSDT",
            pump_occurred=True,
            market_data=test_market_data
        )
        
        if success:
            print("✅ Training sample saved successfully")
        else:
            print("❌ Failed to save training sample")
        
        # Get stats
        stats = exporter.get_dataset_stats()
        print(f"📊 Dataset stats: {stats}")
        
        # Test loading data
        samples = exporter.load_training_data(limit=1)
        print(f"📥 Loaded {len(samples)} samples")
        
        print("🎉 GNN Data Exporter test completed!")
        return True
        
    except Exception as e:
        print(f"❌ GNN Data Exporter test failed: {e}")
        return False

if __name__ == "__main__":
    test_gnn_exporter()