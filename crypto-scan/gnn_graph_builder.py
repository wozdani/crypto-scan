#!/usr/bin/env python3
"""
GNN Graph Builder - On-chain Transaction Graph Construction
Buduje graf skierowany z transakcji blockchain dla analizy GNN

Module builds directed graphs from blockchain transactions where:
- Nodes: wallet addresses
- Edges: transaction transfers with value weights
- Attributes: in_degree, out_degree, total_value per node
"""

import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_transaction_graph(transactions: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Buduje graf skierowany z transakcji (wallets jako nodes, transfery jako edges).
    
    Args:
        transactions: Lista słowników {'from': str, 'to': str, 'value': float/int}
                     Przykład: [
                         {"from": "0xWhale1", "to": "0xWhale2", "value": 100000},
                         {"from": "0xWhale2", "to": "0xExchange", "value": 50000}
                     ]
    
    Returns:
        nx.DiGraph: Graf skierowany z wagami krawędzi i atrybutami węzłów
                   - Nodes mają atrybuty: in_degree, out_degree, total_value
                   - Edges mają wagi odpowiadające wartości transakcji
    """
    if not transactions:
        logger.warning("Empty transactions list provided")
        return nx.DiGraph()
    
    # Initialize directed graph
    graph = nx.DiGraph()
    
    # Track node statistics for attributes
    node_stats = {}
    
    logger.info(f"[GNN GRAPH] Building graph from {len(transactions)} transactions...")
    
    # Process each transaction
    for i, tx in enumerate(transactions):
        try:
            # Validate transaction structure
            if not all(key in tx for key in ['from', 'to', 'value']):
                logger.warning(f"[GNN GRAPH] Invalid transaction {i}: missing required keys")
                continue
            
            from_addr = tx['from']
            to_addr = tx['to']
            value = float(tx['value'])
            
            # Skip zero-value or self-transactions
            if value <= 0 or from_addr == to_addr:
                continue
            
            # Initialize node statistics if first occurrence
            for addr in [from_addr, to_addr]:
                if addr not in node_stats:
                    node_stats[addr] = {
                        'in_degree': 0,
                        'out_degree': 0,
                        'total_value': 0.0
                    }
            
            # Add or update edge with cumulative value
            if graph.has_edge(from_addr, to_addr):
                # Accumulate value for existing edge
                current_weight = graph[from_addr][to_addr]['weight']
                graph[from_addr][to_addr]['weight'] = current_weight + value
            else:
                # Create new edge
                graph.add_edge(from_addr, to_addr, weight=value)
                
                # Update node degrees
                node_stats[from_addr]['out_degree'] += 1
                node_stats[to_addr]['in_degree'] += 1
            
            # Update total value for both nodes
            node_stats[from_addr]['total_value'] += value
            node_stats[to_addr]['total_value'] += value
            
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"[GNN GRAPH] Error processing transaction {i}: {e}")
            continue
    
    # Apply node attributes to graph
    for node, stats in node_stats.items():
        if node in graph:
            graph.nodes[node].update(stats)
    
    # Log graph statistics
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    total_value = sum(data['weight'] for _, _, data in graph.edges(data=True))
    
    logger.info(f"[GNN GRAPH] Built graph: {num_nodes} nodes, {num_edges} edges, total value: ${total_value:,.2f}")
    
    return graph

def get_graph_metrics(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Oblicza metryki grafu dla analizy GNN.
    
    Args:
        graph: Graf NetworkX
        
    Returns:
        Dict zawierający kluczowe metryki grafu
    """
    if graph.number_of_nodes() == 0:
        return {
            'nodes': 0,
            'edges': 0,
            'density': 0.0,
            'avg_degree': 0.0,
            'total_value': 0.0,
            'top_nodes': []
        }
    
    metrics = {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        'total_value': sum(data['weight'] for _, _, data in graph.edges(data=True)),
    }
    
    # Find top nodes by total_value
    node_values = [(node, data.get('total_value', 0)) 
                   for node, data in graph.nodes(data=True)]
    top_nodes = sorted(node_values, key=lambda x: x[1], reverse=True)[:5]
    metrics['top_nodes'] = top_nodes
    
    return metrics

def filter_graph_by_value(graph: nx.DiGraph, min_value: float) -> nx.DiGraph:
    """
    Filtruje graf usuwając krawędzie poniżej minimalnej wartości.
    
    Args:
        graph: Graf wejściowy
        min_value: Minimalna wartość krawędzi
        
    Returns:
        Przefiltrowany graf
    """
    filtered_graph = nx.DiGraph()
    
    # Copy nodes with attributes
    filtered_graph.add_nodes_from(graph.nodes(data=True))
    
    # Add only edges above threshold
    for u, v, data in graph.edges(data=True):
        if data.get('weight', 0) >= min_value:
            filtered_graph.add_edge(u, v, **data)
    
    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(filtered_graph))
    filtered_graph.remove_nodes_from(isolated_nodes)
    
    logger.info(f"[GNN FILTER] Filtered graph: {filtered_graph.number_of_nodes()} nodes, "
                f"{filtered_graph.number_of_edges()} edges (min_value: ${min_value:,.2f})")
    
    return filtered_graph

def get_whale_clusters(graph: nx.DiGraph, min_cluster_size: int = 3) -> List[List[str]]:
    """
    Znajduje klastry wielorybów w grafie używając connected components.
    
    Args:
        graph: Graf transakcji
        min_cluster_size: Minimalna wielkość klastra
        
    Returns:
        Lista klastrów (każdy klaster to lista adresów)
    """
    # Convert to undirected for clustering
    undirected = graph.to_undirected()
    
    # Find connected components
    components = list(nx.connected_components(undirected))
    
    # Filter by minimum size
    whale_clusters = [list(component) for component in components 
                     if len(component) >= min_cluster_size]
    
    logger.info(f"[GNN CLUSTERS] Found {len(whale_clusters)} whale clusters "
                f"(min_size: {min_cluster_size})")
    
    return whale_clusters

def export_graph_for_gnn(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Eksportuje graf w formacie przygotowanym dla modeli GNN.
    
    Args:
        graph: Graf NetworkX
        
    Returns:
        Dict z danymi grafu w formacie GNN
    """
    # Node mapping to integers
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Edge list as integer pairs
    edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    
    # Edge weights
    edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    
    # Node features (in_degree, out_degree, total_value)
    node_features = []
    for node in nodes:
        attrs = graph.nodes[node]
        features = [
            attrs.get('in_degree', 0),
            attrs.get('out_degree', 0),
            attrs.get('total_value', 0.0)
        ]
        node_features.append(features)
    
    return {
        'nodes': nodes,
        'node_features': node_features,
        'edge_list': edge_list,
        'edge_weights': edge_weights,
        'node_mapping': node_to_idx
    }

def test_graph_builder():
    """Test funkcji budowy grafu"""
    print("\n🧪 Testing GNN Graph Builder...")
    
    # Test data
    test_transactions = [
        {"from": "0xWhale1", "to": "0xWhale2", "value": 100000},
        {"from": "0xWhale2", "to": "0xExchange", "value": 50000},
        {"from": "0xWhale1", "to": "0xExchange", "value": 75000},
        {"from": "0xWhale3", "to": "0xWhale1", "value": 200000},
    ]
    
    # Build graph
    graph = build_transaction_graph(test_transactions)
    
    # Test metrics
    metrics = get_graph_metrics(graph)
    print(f"✅ Graph metrics: {metrics}")
    
    # Test filtering
    filtered = filter_graph_by_value(graph, 60000)
    print(f"✅ Filtered graph: {filtered.number_of_nodes()} nodes, {filtered.number_of_edges()} edges")
    
    # Test clustering
    clusters = get_whale_clusters(graph, min_cluster_size=2)
    print(f"✅ Whale clusters: {clusters}")
    
    # Test GNN export
    gnn_data = export_graph_for_gnn(graph)
    print(f"✅ GNN export: {len(gnn_data['nodes'])} nodes, {len(gnn_data['edge_list'])} edges")
    
    print("🎉 GNN Graph Builder test completed!")

if __name__ == "__main__":
    test_graph_builder()