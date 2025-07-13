#!/usr/bin/env python3
"""
Graph Visualizer - Renderowanie grafów transakcji z kolorowaniem wg anomaly score
Tworzy wizualizacje PNG grafów GNN dla analizy wzorców i anomalii
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import os
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_transaction_graph(graph: nx.DiGraph, anomaly_scores: Dict[str, float], 
                              token: str, output_dir: str = "graphs_output",
                              save_metadata: bool = True) -> str:
    """
    Renderuje graf transakcji z kolorami wg anomaly score i zapisuje do pliku PNG.
    
    Args:
        graph: Graf transakcji NetworkX
        anomaly_scores: Słownik z wynikami anomaly detection
        token: Symbol tokena lub identyfikator
        output_dir: Katalog docelowy dla plików
        save_metadata: Czy zapisać metadane grafu
        
    Returns:
        Ścieżka do zapisanego pliku PNG
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{token}_graph_{timestamp}.png"
        
        # Create figure with high DPI for quality
        plt.figure(figsize=(14, 10), dpi=150)
        
        # Generate layout with improved positioning
        try:
            if len(graph.nodes) <= 10:
                pos = nx.spring_layout(graph, k=3, iterations=50, seed=42)
            else:
                pos = nx.kamada_kawai_layout(graph)
        except ZeroDivisionError:
            # Fallback for graphs with very small edge weights or isolated nodes
            pos = nx.circular_layout(graph)
        
        # Prepare node colors based on anomaly scores (default = 0.1 for unknown)
        node_colors = []
        node_sizes = []
        
        for node in graph.nodes:
            anomaly_score = anomaly_scores.get(node, 0.1)
            node_colors.append(anomaly_score)
            
            # Size nodes based on anomaly score (larger = more suspicious)
            base_size = 800
            size_multiplier = 1 + (anomaly_score * 2)  # 1x to 3x size
            node_sizes.append(base_size * size_multiplier)
        
        # Prepare edge weights for visualization
        edge_weights = []
        edge_labels = {}
        
        for u, v, data in graph.edges(data=True):
            weight = data.get('value', 0)
            edge_weights.append(weight)
            
            # Add edge labels for significant transactions
            if weight > 1000:  # Show labels for transactions > $1000
                edge_labels[(u, v)] = f"${weight:.0f}"
        
        # Normalize edge weights for thickness
        if edge_weights and max(edge_weights) > 0:
            max_weight = max(edge_weights)
            edge_widths = [max(0.5, (w / max_weight) * 5) for w in edge_weights]
        else:
            edge_widths = [1.0] * len(graph.edges)
        
        # Draw nodes with color mapping
        nodes = nx.draw_networkx_nodes(
            graph, pos, 
            node_color=node_colors, 
            node_size=node_sizes,
            cmap=plt.cm.Reds, 
            vmin=0, vmax=1,
            alpha=0.8,
            edgecolors='black',
            linewidths=1
        )
        
        # Draw edges with varying thickness
        nx.draw_networkx_edges(
            graph, pos, 
            width=edge_widths,
            arrows=True, 
            arrowsize=20,
            alpha=0.6,
            edge_color='gray',
            arrowstyle='->'
        )
        
        # Draw node labels (shortened addresses)
        node_labels = {}
        for node in graph.nodes:
            if len(str(node)) > 10:
                # Shorten long addresses
                short_label = f"{str(node)[:6]}...{str(node)[-4:]}"
            else:
                short_label = str(node)
            node_labels[node] = short_label
        
        nx.draw_networkx_labels(
            graph, pos, 
            labels=node_labels,
            font_size=8, 
            font_weight='bold',
            font_color='white'
        )
        
        # Draw edge labels for significant transactions
        if edge_labels:
            nx.draw_networkx_edge_labels(
                graph, pos, 
                edge_labels=edge_labels,
                font_size=7,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7)
            )
        
        # Create color bar
        if nodes:
            cbar = plt.colorbar(nodes, label="Anomaly Score", shrink=0.8)
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cbar.set_ticklabels(['0.0\n(Normal)', '0.2', '0.4', '0.6', '0.8', '1.0\n(High Risk)'])
        
        # Create legend for risk levels
        legend_elements = [
            mpatches.Circle((0, 0), 0.1, facecolor=plt.cm.Reds(0.2), label='Low Risk (0.0-0.3)'),
            mpatches.Circle((0, 0), 0.1, facecolor=plt.cm.Reds(0.5), label='Medium Risk (0.3-0.6)'),
            mpatches.Circle((0, 0), 0.1, facecolor=plt.cm.Reds(0.8), label='High Risk (0.6-1.0)')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # Calculate statistics for title
        high_risk_nodes = len([s for s in node_colors if s >= 0.6])
        total_value = sum(data.get('value', 0) for _, _, data in graph.edges(data=True))
        
        # Set title with comprehensive information
        title = f"Transaction Graph Analysis: {token}\n"
        title += f"Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)} | "
        title += f"High Risk Nodes: {high_risk_nodes} | Total Value: ${total_value:,.0f}"
        
        plt.title(title, fontsize=12, fontweight='bold', pad=20)
        
        # Remove axes
        plt.axis('off')
        
        # Save with high quality
        plt.savefig(filename, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        # Save metadata if requested
        if save_metadata:
            metadata_file = filename.replace('.png', '_metadata.json')
            save_graph_metadata(graph, anomaly_scores, token, metadata_file)
        
        logger.info(f"[GRAPH VIZ] Saved graph visualization: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"[GRAPH VIZ] Failed to visualize graph for {token}: {e}")
        return ""

def save_graph_metadata(graph: nx.DiGraph, anomaly_scores: Dict[str, float], 
                       token: str, metadata_file: str):
    """
    Zapisuje metadane grafu do pliku JSON
    
    Args:
        graph: Graf NetworkX
        anomaly_scores: Wyniki anomaly detection
        token: Symbol tokena
        metadata_file: Ścieżka do pliku metadanych
    """
    import json
    
    try:
        # Calculate detailed statistics
        total_value = sum(data.get('value', 0) for _, _, data in graph.edges(data=True))
        avg_anomaly = np.mean(list(anomaly_scores.values())) if anomaly_scores else 0
        
        risk_distribution = {
            'low_risk': len([s for s in anomaly_scores.values() if s < 0.3]),
            'medium_risk': len([s for s in anomaly_scores.values() if 0.3 <= s < 0.6]),
            'high_risk': len([s for s in anomaly_scores.values() if s >= 0.6])
        }
        
        # Find most suspicious addresses
        top_suspicious = sorted(anomaly_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        metadata = {
            'token': token,
            'timestamp': datetime.now().isoformat(),
            'graph_stats': {
                'nodes': len(graph.nodes),
                'edges': len(graph.edges),
                'total_transaction_value': total_value,
                'avg_anomaly_score': float(avg_anomaly)
            },
            'risk_analysis': risk_distribution,
            'top_suspicious_addresses': [
                {'address': addr, 'anomaly_score': float(score)} 
                for addr, score in top_suspicious
            ],
            'anomaly_scores': {addr: float(score) for addr, score in anomaly_scores.items()}
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[GRAPH VIZ] Saved metadata: {metadata_file}")
        
    except Exception as e:
        logger.error(f"[GRAPH VIZ] Failed to save metadata: {e}")

def visualize_batch_graphs(graph_data_list: list, output_dir: str = "graphs_output") -> list:
    """
    Renderuje wiele grafów w batch'u
    
    Args:
        graph_data_list: Lista słowników z kluczami: graph, anomaly_scores, token
        output_dir: Katalog docelowy
        
    Returns:
        Lista ścieżek do zapisanych plików
    """
    saved_files = []
    
    for i, data in enumerate(graph_data_list):
        try:
            graph = data['graph']
            anomaly_scores = data['anomaly_scores']
            token = data.get('token', f'GRAPH_{i+1}')
            
            filename = visualize_transaction_graph(
                graph=graph,
                anomaly_scores=anomaly_scores,
                token=token,
                output_dir=output_dir
            )
            
            if filename:
                saved_files.append(filename)
                
        except Exception as e:
            logger.error(f"[BATCH VIZ] Failed to process graph {i+1}: {e}")
    
    logger.info(f"[BATCH VIZ] Processed {len(saved_files)}/{len(graph_data_list)} graphs")
    return saved_files

def create_anomaly_heatmap(anomaly_scores: Dict[str, float], token: str, 
                          output_dir: str = "graphs_output") -> str:
    """
    Tworzy heatmapę anomaly scores dla adresów
    
    Args:
        anomaly_scores: Słownik z wynikami anomaly detection
        token: Symbol tokena
        output_dir: Katalog docelowy
        
    Returns:
        Ścieżka do zapisanego pliku
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{token}_heatmap_{timestamp}.png"
        
        # Prepare data for heatmap
        addresses = list(anomaly_scores.keys())
        scores = list(anomaly_scores.values())
        
        if not addresses:
            logger.warning(f"[HEATMAP] No data for {token}")
            return ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(addresses) * 0.4)))
        
        # Create heatmap
        scores_array = np.array(scores).reshape(-1, 1)
        im = ax.imshow(scores_array, cmap='Reds', aspect='auto', vmin=0, vmax=1)
        
        # Set labels
        short_addresses = [f"{addr[:8]}...{addr[-6:]}" if len(addr) > 20 else addr 
                          for addr in addresses]
        
        ax.set_yticks(range(len(addresses)))
        ax.set_yticklabels(short_addresses)
        ax.set_xticks([])
        
        # Add score text on bars
        for i, score in enumerate(scores):
            color = 'white' if score > 0.5 else 'black'
            ax.text(0, i, f'{score:.3f}', ha='center', va='center', 
                   color=color, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Anomaly Score', rotation=270, labelpad=20)
        
        # Set title
        plt.title(f'Anomaly Score Heatmap: {token}\n'
                 f'{len(addresses)} addresses analyzed', 
                 fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
        
        logger.info(f"[HEATMAP] Saved heatmap: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"[HEATMAP] Failed to create heatmap for {token}: {e}")
        return ""

def test_graph_visualizer():
    """Test funkcjonalności wizualizatora grafów"""
    print("🧪 Testing Graph Visualizer...")
    
    try:
        # Create test graph
        test_graph = nx.DiGraph()
        test_graph.add_edge("0xABC123", "0xDEF456", value=5000.0)
        test_graph.add_edge("0xDEF456", "0xGHI789", value=15000.0)
        test_graph.add_edge("0xJKL012", "0xDEF456", value=2000.0)
        test_graph.add_edge("0xDEF456", "0xMNO345", value=8000.0)
        
        # Test anomaly scores
        test_anomaly_scores = {
            "0xABC123": 0.2,   # Low risk
            "0xDEF456": 0.9,   # High risk - central node
            "0xGHI789": 0.1,   # Normal
            "0xJKL012": 0.6,   # Medium risk
            "0xMNO345": 0.3    # Low-medium risk
        }
        
        # Test main visualization
        filename = visualize_transaction_graph(
            graph=test_graph,
            anomaly_scores=test_anomaly_scores,
            token="TESTGRAPH"
        )
        
        if filename and os.path.exists(filename):
            print("✅ Graph visualization created successfully")
            print(f"📁 Saved to: {filename}")
        else:
            print("❌ Graph visualization failed")
            return False
        
        # Test heatmap
        heatmap_file = create_anomaly_heatmap(
            anomaly_scores=test_anomaly_scores,
            token="TESTGRAPH"
        )
        
        if heatmap_file and os.path.exists(heatmap_file):
            print("✅ Anomaly heatmap created successfully")
            print(f"📁 Heatmap saved to: {heatmap_file}")
        else:
            print("❌ Heatmap creation failed")
            return False
        
        print("🎉 Graph Visualizer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Graph Visualizer test failed: {e}")
        return False

if __name__ == "__main__":
    test_graph_visualizer()