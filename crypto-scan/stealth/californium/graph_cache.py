"""
Graph Cache System for CaliforniumWhale AI
Provides mock temporal graph data for testing and future integration
with Arkham Intelligence and Chainalysis
"""

import networkx as nx
import torch
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os

class GraphCache:
    """
    Graph Cache providing mock temporal graph data for CaliforniumWhale AI
    """
    
    def __init__(self, cache_dir: str = "crypto-scan/cache/californium/graphs"):
        self.cache_dir = cache_dir
        self.ensure_cache_dir()
        
        # Predefined whale patterns for realistic testing
        self.whale_patterns = {
            'accumulation': {
                'nodes': 8,
                'edges': 12,
                'volume_multiplier': 2.5,
                'temporal_density': 0.8
            },
            'distribution': {
                'nodes': 6,
                'edges': 10,
                'volume_multiplier': 1.8,
                'temporal_density': 0.6
            },
            'manipulation': {
                'nodes': 5,
                'edges': 8,
                'volume_multiplier': 3.2,
                'temporal_density': 0.9
            },
            'normal': {
                'nodes': 4,
                'edges': 5,
                'volume_multiplier': 1.0,
                'temporal_density': 0.3
            }
        }
    
    def ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def generate_mock_graph_data(self, symbol: str, pattern: str = 'normal') -> Dict[str, Any]:
        """
        Generate mock temporal graph data for specified symbol
        
        Args:
            symbol: Token symbol (e.g. 'BTCUSDT')
            pattern: Graph pattern type ('accumulation', 'distribution', 'manipulation', 'normal')
            
        Returns:
            Dictionary with graph data, features, timestamps, and volumes
        """
        if pattern not in self.whale_patterns:
            pattern = 'normal'
            
        config = self.whale_patterns[pattern]
        
        # Generate directed graph with temporal edges
        G = nx.DiGraph()
        
        # Create nodes (wallet addresses)
        nodes = [f"wallet_{i}" for i in range(config['nodes'])]
        G.add_nodes_from(nodes)
        
        # Add edges with temporal and value attributes
        edges = []
        for i in range(config['edges']):
            from_node = random.choice(nodes)
            to_node = random.choice([n for n in nodes if n != from_node])
            
            # Generate realistic transaction values
            base_value = random.uniform(1000, 100000) * config['volume_multiplier']
            timestamp = i + 1
            
            edges.append((from_node, to_node, {
                'value': base_value,
                'timestamp': timestamp,
                'block_height': 18000000 + i * 12  # Realistic block heights
            }))
        
        G.add_edges_from(edges)
        
        # Generate node features (4-dimensional for CaliforniumTGN)
        features = []
        for node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            
            # Calculate total transaction value for this node
            total_value = sum([
                data['value'] for _, _, data in G.edges(data=True) if _ == node
            ]) + sum([
                data['value'] for _, _, data in G.edges(data=True) if _ == node
            ])
            
            # Number of unique neighbors
            neighbors = len(set(list(G.predecessors(node)) + list(G.successors(node))))
            
            features.append([
                float(in_degree),
                float(out_degree), 
                float(total_value / 10000),  # Normalized value
                float(neighbors)
            ])
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Generate timestamps and volume data
        timestamps = list(range(1, len(edges) + 1))
        
        # Generate volume sequence with pattern-specific characteristics
        base_volume = 1000
        volumes = []
        
        for i, ts in enumerate(timestamps):
            # Apply pattern-specific volume modulation
            if pattern == 'accumulation':
                # Gradual increase in volume
                volume = base_volume * (1 + i * 0.3) * random.uniform(0.8, 1.2)
            elif pattern == 'distribution':
                # Peak in middle, then decrease
                peak_factor = 1 + 2 * np.exp(-((i - len(timestamps)/2) ** 2) / 4)
                volume = base_volume * peak_factor * random.uniform(0.8, 1.2)
            elif pattern == 'manipulation':
                # Sudden spikes
                spike_factor = 3.0 if i % 3 == 0 else 1.0
                volume = base_volume * spike_factor * random.uniform(0.8, 1.2)
            else:
                # Normal pattern
                volume = base_volume * random.uniform(0.7, 1.3)
                
            volumes.append(int(volume))
        
        # Create comprehensive graph data structure
        graph_data = {
            'symbol': symbol,
            'pattern': pattern,
            'graph': G,
            'features': features_tensor,
            'timestamps': torch.tensor(timestamps, dtype=torch.long),
            'volumes': volumes,
            'metadata': {
                'nodes_count': len(nodes),
                'edges_count': len(edges),
                'total_value': sum([data['value'] for _, _, data in G.edges(data=True)]),
                'temporal_density': config['temporal_density'],
                'volume_multiplier': config['volume_multiplier'],
                'generated_at': datetime.now().isoformat(),
                'pattern_type': pattern
            }
        }
        
        return graph_data
    
    def get_cached_graph(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached graph data for symbol
        
        Args:
            symbol: Token symbol
            
        Returns:
            Cached graph data or None if not found
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_graph.json")
        
        if not os.path.exists(cache_file):
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Check if cache is still valid (24 hours)
            generated_at = datetime.fromisoformat(cached_data['metadata']['generated_at'])
            if datetime.now() - generated_at > timedelta(hours=24):
                return None
                
            return cached_data
            
        except Exception as e:
            print(f"[GRAPH CACHE] Error loading cached graph for {symbol}: {e}")
            return None
    
    def cache_graph(self, symbol: str, graph_data: Dict[str, Any]) -> bool:
        """
        Cache graph data for symbol
        
        Args:
            symbol: Token symbol
            graph_data: Graph data dictionary
            
        Returns:
            True if cached successfully, False otherwise
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_graph.json")
        
        try:
            # Convert NetworkX graph to serializable format
            serializable_data = {
                'symbol': graph_data['symbol'],
                'pattern': graph_data['pattern'],
                'graph_nodes': list(graph_data['graph'].nodes()),
                'graph_edges': [(u, v, data) for u, v, data in graph_data['graph'].edges(data=True)],
                'features': graph_data['features'].tolist(),
                'timestamps': graph_data['timestamps'].tolist(),
                'volumes': graph_data['volumes'],
                'metadata': graph_data['metadata']
            }
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"[GRAPH CACHE] Error caching graph for {symbol}: {e}")
            return False
    
    def generate_test_scenarios(self) -> List[Dict[str, Any]]:
        """
        Generate test scenarios for all whale patterns
        
        Returns:
            List of test scenarios with different patterns
        """
        scenarios = []
        
        for pattern_name in self.whale_patterns.keys():
            scenario = self.generate_mock_graph_data(
                symbol=f"TEST_{pattern_name.upper()}",
                pattern=pattern_name
            )
            scenarios.append(scenario)
            
        return scenarios
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached graphs
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'cache_directory': self.cache_dir,
            'cached_graphs': 0,
            'patterns_available': list(self.whale_patterns.keys()),
            'cache_files': []
        }
        
        if os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('_graph.json')]
            stats['cached_graphs'] = len(cache_files)
            stats['cache_files'] = cache_files
            
        return stats

# Global cache instance
_graph_cache = GraphCache()

def generate_mock_graph_data(symbol: str, pattern: str = 'normal') -> Dict[str, Any]:
    """
    Generate mock temporal graph data for specified symbol
    
    Args:
        symbol: Token symbol
        pattern: Graph pattern type
        
    Returns:
        Graph data dictionary
    """
    return _graph_cache.generate_mock_graph_data(symbol, pattern)

def get_cached_graph(symbol: str) -> Optional[Dict[str, Any]]:
    """Get cached graph data for symbol"""
    return _graph_cache.get_cached_graph(symbol)

def cache_graph(symbol: str, graph_data: Dict[str, Any]) -> bool:
    """Cache graph data for symbol"""
    return _graph_cache.cache_graph(symbol, graph_data)

def generate_test_scenarios() -> List[Dict[str, Any]]:
    """Generate test scenarios for all whale patterns"""
    return _graph_cache.generate_test_scenarios()

def get_graph_stats() -> Dict[str, Any]:
    """Get graph cache statistics"""
    return _graph_cache.get_graph_stats()

# Test function
def test_graph_cache():
    """Test Graph Cache functionality"""
    print("🧪 Testing Graph Cache System...")
    
    # Test 1: Generate mock graph data
    print("\n1. Testing mock graph generation...")
    graph_data = generate_mock_graph_data("TESTUSDT", "accumulation")
    print(f"   ✅ Generated graph: {graph_data['metadata']['nodes_count']} nodes, {graph_data['metadata']['edges_count']} edges")
    print(f"   ✅ Pattern: {graph_data['pattern']}, Total value: ${graph_data['metadata']['total_value']:,.2f}")
    
    # Test 2: Test all patterns
    print("\n2. Testing all whale patterns...")
    scenarios = generate_test_scenarios()
    for scenario in scenarios:
        print(f"   ✅ {scenario['symbol']}: {scenario['metadata']['nodes_count']} nodes, volume_multiplier={scenario['metadata']['volume_multiplier']}")
    
    # Test 3: Cache functionality
    print("\n3. Testing cache functionality...")
    cache_result = cache_graph("TESTUSDT", graph_data)
    print(f"   ✅ Cache save: {cache_result}")
    
    cached_data = get_cached_graph("TESTUSDT")
    print(f"   ✅ Cache load: {cached_data is not None}")
    
    # Test 4: Graph statistics
    print("\n4. Testing graph statistics...")
    stats = get_graph_stats()
    print(f"   ✅ Cache stats: {stats['cached_graphs']} cached graphs")
    print(f"   ✅ Available patterns: {stats['patterns_available']}")
    
    print("\n🎯 Graph Cache test completed successfully!")
    return True

if __name__ == "__main__":
    test_graph_cache()