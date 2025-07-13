"""
Wallet Behavior Encoder - Rozpoznawanie zachowań portfeli przez embedding historii transakcji
Koduje styl działania walleta jako wektor numeryczny umożliwiający klasyfikację i klasterowanie
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_wallet_behavior(transactions: List[Dict[str, Any]], wallet_address: str = None) -> np.ndarray:
    """
    Przyjmuje listę transakcji i zwraca embedding zachowania walleta.
    
    Args:
        transactions: Lista transakcji z kluczami: value, from, to, timestamp
        wallet_address: Adres analizowanego walleta (opcjonalny)
        
    Returns:
        np.ndarray: Wektor embedding [6 wymiarów] opisujący zachowanie walleta
    """
    if not transactions:
        logger.warning("Empty transaction list - returning zero embedding")
        return np.zeros(6)
    
    # Określ adres walleta jeśli nie podano
    if not wallet_address:
        all_addresses = set()
        for tx in transactions:
            all_addresses.add(tx.get('from', ''))
            all_addresses.add(tx.get('to', ''))
        # Użyj najczęściej występującego adresu
        address_counts = defaultdict(int)
        for tx in transactions:
            address_counts[tx.get('from', '')] += 1
            address_counts[tx.get('to', '')] += 1
        wallet_address = max(address_counts.items(), key=lambda x: x[1])[0]
    
    # Oblicz podstawowe metryki transakcji
    total_sent = sum(float(tx.get('value', 0)) for tx in transactions 
                    if tx.get('from') == wallet_address)
    total_received = sum(float(tx.get('value', 0)) for tx in transactions 
                        if tx.get('to') == wallet_address)
    
    all_values = [float(tx.get('value', 0)) for tx in transactions]
    avg_value = np.mean(all_values) if all_values else 0
    tx_count = len(transactions)
    
    unique_to = len(set(tx.get('to', '') for tx in transactions 
                       if tx.get('from') == wallet_address))
    unique_from = len(set(tx.get('from', '') for tx in transactions 
                         if tx.get('to') == wallet_address))
    
    embedding = np.array([
        total_sent,
        total_received,
        avg_value,
        tx_count,
        unique_to,
        unique_from
    ], dtype=np.float64)
    
    logger.info(f"[BEHAVIOR ENCODE] {wallet_address[:10]}... → "
               f"sent:{total_sent:.2f}, recv:{total_received:.2f}, "
               f"avg:{avg_value:.2f}, count:{tx_count}, to:{unique_to}, from:{unique_from}")
    
    return embedding

def encode_advanced_wallet_behavior(transactions: List[Dict[str, Any]], 
                                   wallet_address: str = None) -> np.ndarray:
    """
    Rozszerzona wersja embedding z dodatkowymi metrykami behawioralnymi.
    
    Returns:
        np.ndarray: Wektor embedding [12 wymiarów] z zaawansowanymi metrykami
    """
    basic_embedding = encode_wallet_behavior(transactions, wallet_address)
    
    if not transactions:
        return np.zeros(12)
    
    # Dodatkowe metryki zaawansowane
    values = [float(tx.get('value', 0)) for tx in transactions]
    
    # Wariancja wartości transakcji (volatility stylu)
    value_variance = np.var(values) if len(values) > 1 else 0
    
    # Mediana wartości transakcji
    value_median = np.median(values) if values else 0
    
    # Ratio wysyłanych do otrzymywanych transakcji
    sent_ratio = basic_embedding[0] / (basic_embedding[1] + 1e-8)  # Avoid division by zero
    
    # Średnia liczba unikalnych adresów na transakcję
    unique_density = (basic_embedding[4] + basic_embedding[5]) / (basic_embedding[3] + 1e-8)
    
    # Frequency analysis - czy transakcje są regularne
    if len(transactions) > 2:
        timestamps = sorted([tx.get('timestamp', 0) for tx in transactions if tx.get('timestamp')])
        if len(timestamps) > 1:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            interval_variance = np.var(intervals) if len(intervals) > 1 else 0
        else:
            interval_variance = 0
    else:
        interval_variance = 0
    
    # Whale-style indicator (duże pojedyncze transakcje)
    max_value = max(values) if values else 0
    whale_indicator = max_value / (np.mean(values) + 1e-8) if values else 0
    
    # Połącz wszystkie metryki
    advanced_embedding = np.concatenate([
        basic_embedding,  # 6 wymiarów podstawowych
        np.array([
            value_variance,      # 7. Wariancja wartości
            value_median,        # 8. Mediana wartości  
            sent_ratio,          # 9. Ratio sent/received
            unique_density,      # 10. Gęstość unikalnych adresów
            interval_variance,   # 11. Regularity transakcji
            whale_indicator      # 12. Whale-style indicator
        ])
    ])
    
    logger.info(f"[ADVANCED ENCODE] {wallet_address[:10] if wallet_address else 'Unknown'}... → "
               f"variance:{value_variance:.2f}, whale_ind:{whale_indicator:.2f}")
    
    return advanced_embedding

def cluster_wallet_behaviors(embeddings: List[np.ndarray], 
                           wallet_addresses: List[str],
                           n_clusters: int = 5) -> Dict[str, Any]:
    """
    Klasterowanie portfeli na podstawie ich embeddings zachowań.
    
    Args:
        embeddings: Lista embeddings dla każdego walleta
        wallet_addresses: Lista adresów walletów
        n_clusters: Liczba klastrów
        
    Returns:
        Dict z wynikami klasterowania i analizą
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("sklearn not available - install with: pip install scikit-learn")
        return {"error": "sklearn_not_available"}
    
    if len(embeddings) < n_clusters:
        logger.warning(f"Not enough wallets ({len(embeddings)}) for {n_clusters} clusters")
        n_clusters = max(1, len(embeddings))
    
    # Standaryzacja embeddings
    embeddings_array = np.array(embeddings)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array)
    
    # KMeans klasterowanie
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)
    
    # Analiza klastrów
    cluster_analysis = defaultdict(list)
    for i, (wallet, label) in enumerate(zip(wallet_addresses, cluster_labels)):
        cluster_analysis[int(label)].append({
            'wallet': wallet,
            'embedding': embeddings[i].tolist(),
            'cluster': int(label)
        })
    
    # Znajdź centroidy klastrów w oryginalnej przestrzeni
    cluster_centroids = {}
    for cluster_id in range(n_clusters):
        cluster_embeddings = [emb for emb, label in zip(embeddings, cluster_labels) if label == cluster_id]
        if cluster_embeddings:
            cluster_centroids[cluster_id] = np.mean(cluster_embeddings, axis=0).tolist()
    
    result = {
        'clusters': dict(cluster_analysis),
        'centroids': cluster_centroids,
        'n_clusters': n_clusters,
        'wallet_count': len(wallet_addresses),
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
    }
    
    logger.info(f"[CLUSTERING] Clustered {len(wallet_addresses)} wallets into {n_clusters} groups")
    return result

def identify_whale_wallets(embeddings: List[np.ndarray], 
                          wallet_addresses: List[str],
                          whale_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Identyfikacja whale wallets na podstawie embeddings.
    
    Args:
        embeddings: Lista embeddings
        wallet_addresses: Lista adresów
        whale_threshold: Próg klasyfikacji whale (percentyl)
        
    Returns:
        Lista walletów z whale score
    """
    if not embeddings:
        return []
    
    whale_wallets = []
    embeddings_array = np.array(embeddings)
    
    # Whale indicators based on embedding features
    for i, (wallet, embedding) in enumerate(zip(wallet_addresses, embeddings)):
        total_sent = embedding[0]
        total_received = embedding[1] 
        avg_value = embedding[2]
        tx_count = embedding[3]
        
        # Whale score calculation
        volume_score = (total_sent + total_received) / (np.max(embeddings_array[:, 0]) + np.max(embeddings_array[:, 1]) + 1e-8)
        avg_value_score = avg_value / (np.max(embeddings_array[:, 2]) + 1e-8)
        activity_score = tx_count / (np.max(embeddings_array[:, 3]) + 1e-8)
        
        # Combined whale score
        whale_score = (volume_score * 0.5 + avg_value_score * 0.3 + activity_score * 0.2)
        
        if whale_score >= whale_threshold:
            whale_wallets.append({
                'wallet': wallet,
                'whale_score': float(whale_score),
                'total_volume': float(total_sent + total_received),
                'avg_value': float(avg_value),
                'tx_count': int(tx_count),
                'embedding': embedding.tolist()
            })
    
    # Sort by whale score descending
    whale_wallets.sort(key=lambda x: x['whale_score'], reverse=True)
    
    logger.info(f"[WHALE DETECTION] Found {len(whale_wallets)} whale wallets (threshold: {whale_threshold})")
    return whale_wallets

def save_behavior_analysis(analysis_results: Dict[str, Any], 
                          output_file: str = "cache/wallet_behavior_analysis.json"):
    """
    Zapisuje wyniki analizy zachowań do pliku JSON.
    """
    try:
        analysis_results['timestamp'] = datetime.now().isoformat()
        analysis_results['analysis_type'] = 'wallet_behavior_embedding'
        
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"[SAVE] Behavior analysis saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"[SAVE ERROR] Failed to save analysis: {e}")
        return False

def load_behavior_analysis(input_file: str = "cache/wallet_behavior_analysis.json") -> Optional[Dict[str, Any]]:
    """
    Ładuje zapisaną analizę zachowań z pliku JSON.
    """
    try:
        with open(input_file, 'r') as f:
            analysis = json.load(f)
        
        logger.info(f"[LOAD] Behavior analysis loaded from {input_file}")
        return analysis
    except FileNotFoundError:
        logger.warning(f"[LOAD] Analysis file not found: {input_file}")
        return None
    except Exception as e:
        logger.error(f"[LOAD ERROR] Failed to load analysis: {e}")
        return None

def analyze_wallet_behavior_complete(transactions_by_wallet: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Kompletna analiza zachowań dla wielu walletów z klasterowaniem i whale detection.
    
    Args:
        transactions_by_wallet: Dict {wallet_address: [transactions]}
        
    Returns:
        Dict z kompletną analizą zachowań
    """
    if not transactions_by_wallet:
        logger.warning("No wallet transactions provided")
        return {"error": "no_data"}
    
    logger.info(f"[COMPLETE ANALYSIS] Starting analysis for {len(transactions_by_wallet)} wallets")
    
    # Generate embeddings for all wallets
    wallet_addresses = list(transactions_by_wallet.keys())
    embeddings = []
    
    for wallet_addr in wallet_addresses:
        transactions = transactions_by_wallet[wallet_addr]
        embedding = encode_advanced_wallet_behavior(transactions, wallet_addr)
        embeddings.append(embedding)
    
    # Clustering analysis
    cluster_results = cluster_wallet_behaviors(embeddings, wallet_addresses)
    
    # Whale detection
    whale_wallets = identify_whale_wallets(embeddings, wallet_addresses)
    
    # Complete analysis results
    complete_analysis = {
        'wallet_count': len(wallet_addresses),
        'embedding_dimension': len(embeddings[0]) if embeddings else 0,
        'clustering': cluster_results,
        'whale_wallets': whale_wallets,
        'embeddings': {addr: emb.tolist() for addr, emb in zip(wallet_addresses, embeddings)},
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    # Save results
    save_behavior_analysis(complete_analysis)
    
    logger.info(f"[COMPLETE ANALYSIS] Finished - {len(whale_wallets)} whales, "
               f"{cluster_results.get('n_clusters', 0)} clusters")
    
    return complete_analysis

def test_wallet_behavior_encoder():
    """Test funkcjonalności wallet behavior encoder"""
    logger.info("[TEST] Starting wallet behavior encoder tests")
    
    # Sample transaction data
    sample_transactions = [
        {'from': '0xwhale1', 'to': '0xuser1', 'value': 1000000, 'timestamp': 1642000000},
        {'from': '0xuser2', 'to': '0xwhale1', 'value': 500000, 'timestamp': 1642001000},
        {'from': '0xwhale1', 'to': '0xuser3', 'value': 750000, 'timestamp': 1642002000},
        {'from': '0xuser4', 'to': '0xwhale1', 'value': 250000, 'timestamp': 1642003000},
    ]
    
    # Test basic encoding
    embedding = encode_wallet_behavior(sample_transactions, '0xwhale1')
    logger.info(f"[TEST] Basic embedding shape: {embedding.shape}")
    logger.info(f"[TEST] Basic embedding values: {embedding}")
    
    # Test advanced encoding
    advanced_embedding = encode_advanced_wallet_behavior(sample_transactions, '0xwhale1')
    logger.info(f"[TEST] Advanced embedding shape: {advanced_embedding.shape}")
    logger.info(f"[TEST] Advanced embedding values: {advanced_embedding}")
    
    # Test complete analysis
    transactions_by_wallet = {
        '0xwhale1': sample_transactions,
        '0xuser1': [{'from': '0xuser1', 'to': '0xother', 'value': 100, 'timestamp': 1642000500}],
        '0xuser2': [{'from': '0xuser2', 'to': '0xother', 'value': 200, 'timestamp': 1642001500}]
    }
    
    complete_analysis = analyze_wallet_behavior_complete(transactions_by_wallet)
    logger.info(f"[TEST] Complete analysis keys: {list(complete_analysis.keys())}")
    
    logger.info("[TEST] Wallet behavior encoder tests completed successfully")
    return True

if __name__ == "__main__":
    test_wallet_behavior_encoder()