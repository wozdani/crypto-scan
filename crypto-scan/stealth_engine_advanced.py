#!/usr/bin/env python3
"""
Advanced Stealth Engine - GNN + Reinforcement Learning Pipeline
Łączy wszystkie komponenty: transaction fetch → graph building → GNN analysis → RL decisions → Telegram alerts
"""

import os
import json
import time
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from gnn_graph_builder import build_transaction_graph
from gnn_anomaly_detector import detect_graph_anomalies
from rl_agent import RLAgent
from alert_manager import process_alert_decision, save_alert_history
from gnn_data_exporter import GNNDataExporter
from graph_visualizer import visualize_transaction_graph, create_anomaly_heatmap
from wallet_behavior_encoder import (encode_advanced_wallet_behavior, 
                                   analyze_wallet_behavior_complete,
                                   identify_whale_wallets)
from whale_style_detector import WhaleStyleDetector

# Configuration
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY")
BSCSCAN_API_KEY = os.environ.get("BSCSCAN_API_KEY")
POLYGONSCAN_API_KEY = os.environ.get("POLYGONSCAN_API_KEY")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StealthEngineAdvanced:
    """
    Advanced Stealth Engine combining GNN anomaly detection with reinforcement learning
    """
    
    def __init__(self):
        """Initialize the advanced stealth engine"""
        self.rl_agent = RLAgent()
        self.gnn_exporter = GNNDataExporter()
        self.whale_detector = WhaleStyleDetector(model_type='rf')  # Use RandomForest for production
        self.api_keys = {
            'ethereum': ETHERSCAN_API_KEY,
            'bsc': BSCSCAN_API_KEY,
            'polygon': POLYGONSCAN_API_KEY
        }
        self.api_endpoints = {
            'ethereum': 'https://api.etherscan.io/api',
            'bsc': 'https://api.bscscan.com/api',
            'polygon': 'https://api.polygonscan.com/api'
        }
        logger.info("[STEALTH ENGINE] Advanced GNN + RL system initialized")
    
    def fetch_transactions_from_blockchain(self, address: str, chain: str = 'ethereum', limit: int = 100) -> List[Dict]:
        """
        Fetch real blockchain transactions for given address
        
        Args:
            address: Wallet address to analyze
            chain: Blockchain network (ethereum, bsc, polygon)
            limit: Maximum number of transactions to fetch
            
        Returns:
            List of transaction dictionaries
        """
        api_key = self.api_keys.get(chain)
        endpoint = self.api_endpoints.get(chain)
        
        if not api_key or not endpoint:
            logger.warning(f"[BLOCKCHAIN] Missing API key or endpoint for {chain}")
            return []
        
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",
            "apikey": api_key
        }
        
        try:
            logger.info(f"[BLOCKCHAIN] Fetching transactions for {address[:10]}... on {chain}")
            response = requests.get(endpoint, params=params, timeout=10)
            data = response.json()
            
            txs = []
            if data.get("status") == "1" and data.get("result"):
                for tx in data["result"][:limit]:
                    # Convert Wei to ETH/BNB/MATIC
                    value_wei = int(tx.get("value", 0))
                    value_native = value_wei / 1e18
                    
                    # Only include transactions with meaningful value
                    if value_native > 0.001:  # More than 0.001 native token
                        txs.append({
                            "from": tx["from"].lower(),
                            "to": tx["to"].lower(),
                            "value": value_native,
                            "hash": tx["hash"],
                            "timestamp": int(tx["timeStamp"]),
                            "gas_used": int(tx.get("gasUsed", 0)),
                            "gas_price": int(tx.get("gasPrice", 0))
                        })
                
                logger.info(f"[BLOCKCHAIN] Found {len(txs)} meaningful transactions for {address[:10]}...")
                return txs
            else:
                logger.warning(f"[BLOCKCHAIN] No transactions found for {address} on {chain}")
                return []
                
        except Exception as e:
            logger.error(f"[BLOCKCHAIN] Error fetching transactions: {e}")
            return []
    
    def analyze_address_with_gnn(self, address: str, chain: str = 'ethereum') -> Dict[str, Any]:
        """
        Complete GNN analysis pipeline for given address
        
        Args:
            address: Target wallet address
            chain: Blockchain network
            
        Returns:
            Complete analysis results
        """
        logger.info(f"[GNN ANALYSIS] Starting analysis for {address[:10]}... on {chain}")
        
        # Step 1: Fetch transactions
        transactions = self.fetch_transactions_from_blockchain(address, chain)
        if not transactions:
            logger.warning(f"[GNN ANALYSIS] No transactions found for {address}")
            return {
                'address': address,
                'chain': chain,
                'status': 'no_transactions',
                'anomaly_scores': {},
                'graph_stats': {},
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 2: Build transaction graph
        logger.info(f"[GNN ANALYSIS] Building transaction graph from {len(transactions)} transactions")
        graph = build_transaction_graph(transactions)
        
        if not graph or graph.number_of_nodes() == 0:
            logger.warning(f"[GNN ANALYSIS] Failed to build graph for {address}")
            return {
                'address': address,
                'chain': chain,
                'status': 'graph_failed',
                'anomaly_scores': {},
                'graph_stats': {},
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 3: GNN anomaly detection
        logger.info(f"[GNN ANALYSIS] Running anomaly detection on graph ({graph.number_of_nodes()} nodes)")
        anomaly_results = detect_graph_anomalies(graph)
        
        # Step 4: Compile results
        analysis_result = {
            'address': address,
            'chain': chain,
            'status': 'success',
            'transaction_count': len(transactions),
            'anomaly_scores': anomaly_results.get('anomaly_scores', {}),
            'graph_stats': anomaly_results.get('graph_stats', {}),
            'risk_analysis': anomaly_results.get('risk_analysis', {}),
            'pattern_analysis': anomaly_results.get('pattern_analysis', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"[GNN ANALYSIS] Analysis complete. Found {len(analysis_result['anomaly_scores'])} scored addresses")
        return analysis_result
    
    def run_stealth_engine_for_address(self, address: str, chain: str = 'ethereum', 
                                     symbol: str = None) -> Dict[str, Any]:
        """
        Complete stealth engine pipeline: GNN analysis → RL decision → Telegram alert
        
        Args:
            address: Target wallet address
            chain: Blockchain network
            symbol: Associated token symbol (optional)
            
        Returns:
            Complete pipeline results
        """
        logger.info(f"[STEALTH ENGINE] Starting full pipeline for {address[:10]}...")
        
        # Step 1: GNN Analysis
        gnn_results = self.analyze_address_with_gnn(address, chain)
        
        if gnn_results['status'] != 'success':
            logger.warning(f"[STEALTH ENGINE] GNN analysis failed: {gnn_results['status']}")
            return {
                'pipeline_status': 'gnn_failed',
                'gnn_results': gnn_results,
                'rl_prediction': None,
                'alert_result': None,
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 2: RL Agent Decision
        logger.info(f"[STEALTH ENGINE] Processing RL decision for anomaly scores")
        symbol_for_rl = symbol or f"ADDR_{address[:8].upper()}"
        
        # Extract anomaly scores as list
        anomaly_scores_list = list(gnn_results['anomaly_scores'].values())
        
        rl_prediction = self.rl_agent.predict_alert(
            anomaly_scores=anomaly_scores_list,
            symbol=symbol_for_rl
        )
        
        # Step 3: Alert Decision & Telegram
        if rl_prediction['should_alert']:
            logger.info(f"[STEALTH ENGINE] RL decision: ALERT - processing Telegram notification")
            
            # Prepare market data for alert
            market_data = {
                'address': address,
                'chain': chain,
                'transaction_count': gnn_results['transaction_count'],
                'graph_nodes': gnn_results['graph_stats'].get('nodes', 0),
                'graph_edges': gnn_results['graph_stats'].get('edges', 0),
                'analysis_timestamp': gnn_results['timestamp']
            }
            
            alert_result = process_alert_decision(
                symbol=symbol_for_rl,
                anomaly_scores=gnn_results['anomaly_scores'],
                rl_agent=self.rl_agent,
                market_data=market_data,
                token=TELEGRAM_BOT_TOKEN,
                chat_id=TELEGRAM_CHAT_ID
            )
        else:
            logger.info(f"[STEALTH ENGINE] RL decision: HOLD - no alert needed")
            alert_result = {
                'alert_sent': False,
                'reason': 'rl_decision_hold',
                'confidence': rl_prediction['confidence']
            }
        
        # Step 4: Export training data (GNN + RL + outcome)
        try:
            # Fetch transactions again for data export (could be optimized by storing in gnn_results)
            transactions = self.fetch_transactions_from_blockchain(address, chain)
            
            if transactions:
                # Determine if suspicious activity occurred (placeholder logic)
                # In production, this could check for actual pump detection within time window
                suspicious_activity = any(score > 0.7 for score in gnn_results['anomaly_scores'].values())
                
                # Build graph for export
                graph_for_export = build_transaction_graph(transactions)
                
                # Export data for ML training
                export_success = self.gnn_exporter.save_training_sample(
                    graph=graph_for_export,
                    anomaly_scores=gnn_results['anomaly_scores'],
                    token=symbol_for_rl,
                    pump_occurred=suspicious_activity,
                    market_data={
                        'address': address,
                        'chain': chain,
                        'alert_sent': alert_result.get('alert_sent', False),
                        'rl_confidence': rl_prediction['confidence'],
                        'rl_action': rl_prediction['action']
                    },
                    graph_stats=gnn_results['graph_stats'],
                    analysis_metadata={
                        'transaction_count': gnn_results['transaction_count'],
                        'risk_analysis': gnn_results.get('risk_analysis', {}),
                        'pattern_analysis': gnn_results.get('pattern_analysis', {})
                    }
                )
                
                if export_success:
                    logger.info(f"[GNN EXPORT] Training data exported for {symbol_for_rl} (suspicious: {suspicious_activity})")
                
                # Step 4.5: Generate graph visualization for debugging
                try:
                    # Generate graph visualization with anomaly scores
                    graph_filename = visualize_transaction_graph(
                        graph=graph_for_export,
                        anomaly_scores=gnn_results['anomaly_scores'],
                        token=symbol_for_rl,
                        output_dir="graphs_output"
                    )
                    
                    if graph_filename:
                        logger.info(f"[GRAPH VIZ] Saved graph visualization: {graph_filename}")
                    
                    # Generate anomaly heatmap for detailed analysis
                    heatmap_filename = create_anomaly_heatmap(
                        anomaly_scores=gnn_results['anomaly_scores'],
                        token=symbol_for_rl,
                        output_dir="graphs_output"
                    )
                    
                    if heatmap_filename:
                        logger.info(f"[GRAPH VIZ] Saved anomaly heatmap: {heatmap_filename}")
                        
                except Exception as viz_e:
                    logger.warning(f"[GRAPH VIZ] Failed to generate visualization: {viz_e}")
            
        except Exception as e:
            logger.error(f"[GNN EXPORT] Failed to export training data: {e}")
        
        # Step 5: Compile final results
        pipeline_result = {
            'pipeline_status': 'success',
            'address': address,
            'chain': chain,
            'symbol': symbol_for_rl,
            'gnn_results': gnn_results,
            'rl_prediction': rl_prediction,
            'alert_result': alert_result,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"[STEALTH ENGINE] Pipeline complete. Alert sent: {alert_result.get('alert_sent', False)}")
        return pipeline_result
    
    def run_batch_analysis(self, addresses: List[str], chain: str = 'ethereum') -> List[Dict[str, Any]]:
        """
        Run stealth engine analysis on multiple addresses
        
        Args:
            addresses: List of wallet addresses to analyze
            chain: Blockchain network
            
        Returns:
            List of analysis results
        """
        logger.info(f"[BATCH ANALYSIS] Starting batch analysis for {len(addresses)} addresses")
        
        results = []
        for i, address in enumerate(addresses):
            logger.info(f"[BATCH ANALYSIS] Processing address {i+1}/{len(addresses)}: {address[:10]}...")
            
            try:
                result = self.run_stealth_engine_for_address(address, chain)
                results.append(result)
                
                # Rate limiting
                time.sleep(0.2)  # 200ms between API calls
                
            except Exception as e:
                logger.error(f"[BATCH ANALYSIS] Error processing {address}: {e}")
                results.append({
                    'pipeline_status': 'error',
                    'address': address,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save batch results
        batch_filename = f"cache/batch_analysis_{int(time.time())}.json"
        try:
            with open(batch_filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"[BATCH ANALYSIS] Results saved to {batch_filename}")
        except Exception as e:
            logger.error(f"[BATCH ANALYSIS] Failed to save results: {e}")
        
        # Summary statistics
        successful = len([r for r in results if r.get('pipeline_status') == 'success'])
        alerts_sent = len([r for r in results if r.get('alert_result', {}).get('alert_sent')])
        
        logger.info(f"[BATCH ANALYSIS] Complete: {successful}/{len(addresses)} successful, {alerts_sent} alerts sent")
        return results
    
    def analyze_wallet_behaviors(self, addresses: List[str], chain: str = 'ethereum') -> Dict[str, Any]:
        """
        Complete wallet behavior analysis using transaction history embeddings
        
        Args:
            addresses: List of wallet addresses to analyze
            chain: Blockchain network
            
        Returns:
            Complete behavior analysis with clustering and whale detection
        """
        logger.info(f"[BEHAVIOR ANALYSIS] Starting behavioral analysis for {len(addresses)} wallets")
        
        # Step 1: Fetch transactions for all addresses
        transactions_by_wallet = {}
        for i, address in enumerate(addresses):
            logger.info(f"[BEHAVIOR ANALYSIS] Fetching transactions {i+1}/{len(addresses)}: {address[:10]}...")
            
            try:
                transactions = self.fetch_transactions_from_blockchain(address, chain)
                if transactions:
                    transactions_by_wallet[address] = transactions
                    logger.info(f"[BEHAVIOR ANALYSIS] {address[:10]}... → {len(transactions)} transactions")
                else:
                    logger.warning(f"[BEHAVIOR ANALYSIS] {address[:10]}... → no transactions")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"[BEHAVIOR ANALYSIS] Error fetching transactions for {address}: {e}")
        
        if not transactions_by_wallet:
            logger.warning("[BEHAVIOR ANALYSIS] No transaction data available for behavioral analysis")
            return {'error': 'no_transaction_data'}
        
        # Step 2: Generate behavioral embeddings for each wallet
        logger.info(f"[BEHAVIOR ANALYSIS] Generating embeddings for {len(transactions_by_wallet)} wallets")
        embeddings = []
        wallet_addresses = list(transactions_by_wallet.keys())
        
        for wallet_addr in wallet_addresses:
            transactions = transactions_by_wallet[wallet_addr]
            embedding = encode_advanced_wallet_behavior(transactions, wallet_addr)
            embeddings.append(embedding)
        
        # Step 3: Complete behavioral analysis with clustering and whale detection
        logger.info("[BEHAVIOR ANALYSIS] Running complete behavioral analysis")
        complete_analysis = analyze_wallet_behavior_complete(transactions_by_wallet)
        
        # Step 4: Enhanced analysis with GNN anomaly scores correlation and ML whale classification
        gnn_behavior_correlation = {}
        whale_style_predictions = {}
        
        for wallet_addr in wallet_addresses:
            try:
                # Get GNN anomaly analysis for this wallet
                gnn_result = self.analyze_address_with_gnn(wallet_addr, chain)
                
                # Find wallet embedding for ML whale style detection
                wallet_index = wallet_addresses.index(wallet_addr)
                wallet_embedding = embeddings[wallet_index]
                
                # ML-based whale style prediction
                try:
                    whale_style_result = self.whale_detector.analyze_wallet_comprehensive(wallet_embedding)
                    whale_style_predictions[wallet_addr] = whale_style_result
                    logger.info(f"[WHALE ML] {wallet_addr[:10]}... → ML whale style: {whale_style_result.get('whale_analysis', {}).get('is_whale', 'unknown')}")
                except Exception as ml_e:
                    logger.warning(f"[WHALE ML] Failed ML prediction for {wallet_addr}: {ml_e}")
                    whale_style_predictions[wallet_addr] = {'error': str(ml_e)}
                
                if gnn_result['status'] == 'success':
                    # Correlate behavioral embedding with anomaly scores
                    max_anomaly_score = max(gnn_result['anomaly_scores'].values()) if gnn_result['anomaly_scores'] else 0
                    
                    gnn_behavior_correlation[wallet_addr] = {
                        'behavioral_embedding': wallet_embedding.tolist(),
                        'max_anomaly_score': max_anomaly_score,
                        'anomaly_addresses': len(gnn_result['anomaly_scores']),
                        'transaction_count': gnn_result['transaction_count'],
                        'whale_style_prediction': whale_style_predictions[wallet_addr],
                        'gnn_status': 'success'
                    }
                else:
                    gnn_behavior_correlation[wallet_addr] = {
                        'behavioral_embedding': embeddings[wallet_addresses.index(wallet_addr)].tolist(),
                        'whale_style_prediction': whale_style_predictions[wallet_addr],
                        'gnn_status': 'failed',
                        'reason': gnn_result['status']
                    }
                    
            except Exception as e:
                logger.error(f"[BEHAVIOR ANALYSIS] Error correlating GNN for {wallet_addr}: {e}")
        
        # Step 5: Enhanced whale detection using behavioral, GNN, and ML whale classification
        enhanced_whale_wallets = []
        for whale_info in complete_analysis.get('whale_wallets', []):
            wallet_addr = whale_info['wallet']
            gnn_info = gnn_behavior_correlation.get(wallet_addr, {})
            whale_style_pred = whale_style_predictions.get(wallet_addr, {})
            
            # Enhanced whale score combining behavioral, anomaly, and ML features
            behavioral_score = whale_info['whale_score']
            anomaly_bonus = gnn_info.get('max_anomaly_score', 0) * 0.2  # 20% weight to anomaly
            
            # ML whale classification contribution
            ml_whale_confidence = 0
            ml_whale_type = 'unknown'
            if whale_style_pred and 'whale_analysis' in whale_style_pred:
                ml_whale_confidence = whale_style_pred['whale_analysis'].get('whale_confidence', 0)
                ml_whale_type = whale_style_pred.get('type_analysis', {}).get('predicted_type', 'unknown')
            
            ml_bonus = ml_whale_confidence * 0.15  # 15% weight to ML classification
            
            enhanced_score = min(1.0, behavioral_score + anomaly_bonus + ml_bonus)
            
            enhanced_whale_wallets.append({
                **whale_info,
                'enhanced_whale_score': enhanced_score,
                'anomaly_contribution': anomaly_bonus,
                'ml_whale_contribution': ml_bonus,
                'ml_whale_confidence': ml_whale_confidence,
                'ml_whale_type': ml_whale_type,
                'gnn_anomaly_score': gnn_info.get('max_anomaly_score', 0),
                'behavior_only_score': behavioral_score
            })
        
        # Sort by enhanced score
        enhanced_whale_wallets.sort(key=lambda x: x['enhanced_whale_score'], reverse=True)
        
        # Step 6: Final comprehensive analysis
        behavior_analysis_result = {
            'analysis_type': 'comprehensive_wallet_behavior',
            'wallet_count': len(wallet_addresses),
            'transaction_wallets': len(transactions_by_wallet),
            'behavioral_clustering': complete_analysis.get('clustering', {}),
            'whale_wallets_behavioral': complete_analysis.get('whale_wallets', []),
            'whale_wallets_enhanced': enhanced_whale_wallets,
            'whale_style_predictions': whale_style_predictions,
            'gnn_behavior_correlation': gnn_behavior_correlation,
            'embeddings_summary': {
                'dimension': len(embeddings[0]) if embeddings else 0,
                'wallet_count': len(embeddings)
            },
            'ml_whale_summary': {
                'predictions_made': len(whale_style_predictions),
                'successful_predictions': len([p for p in whale_style_predictions.values() if 'whale_analysis' in p]),
                'whale_predictions': len([p for p in whale_style_predictions.values() 
                                        if p.get('whale_analysis', {}).get('is_whale', False)])
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'chain': chain
        }
        
        # Save comprehensive analysis
        analysis_filename = f"cache/behavior_analysis_{int(time.time())}.json"
        try:
            with open(analysis_filename, 'w') as f:
                json.dump(behavior_analysis_result, f, indent=2, default=str)
            logger.info(f"[BEHAVIOR ANALYSIS] Results saved to {analysis_filename}")
        except Exception as e:
            logger.error(f"[BEHAVIOR ANALYSIS] Failed to save analysis: {e}")
        
        # Summary statistics
        whale_count_behavioral = len(complete_analysis.get('whale_wallets', []))
        whale_count_enhanced = len(enhanced_whale_wallets)
        cluster_count = complete_analysis.get('clustering', {}).get('n_clusters', 0)
        ml_predictions = behavior_analysis_result['ml_whale_summary']['predictions_made']
        ml_whales = behavior_analysis_result['ml_whale_summary']['whale_predictions']
        
        logger.info(f"[BEHAVIOR ANALYSIS] Complete: {whale_count_behavioral} behavioral whales, "
                   f"{whale_count_enhanced} enhanced whales, {cluster_count} clusters")
        logger.info(f"[ML WHALE ANALYSIS] ML predictions: {ml_predictions}, ML whales detected: {ml_whales}")
        
        return behavior_analysis_result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'rl_agent_stats': self.rl_agent.get_agent_stats(),
            'api_keys_configured': {
                chain: bool(key) for chain, key in self.api_keys.items()
            },
            'telegram_configured': bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            'timestamp': datetime.now().isoformat()
        }

def test_stealth_engine_advanced():
    """Test the advanced stealth engine with sample addresses"""
    logger.info("[TEST] Starting advanced stealth engine test")
    
    # Test addresses (known whale addresses)
    test_addresses = [
        "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Binance hot wallet
        "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",  # Binance cold wallet
        "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503"   # Another large wallet
    ]
    
    engine = StealthEngineAdvanced()
    
    # Test single address analysis
    logger.info("[TEST] Testing single address analysis")
    result = engine.run_stealth_engine_for_address(test_addresses[0], 'ethereum', 'TEST_WHALE')
    
    print(f"[TEST RESULT] Pipeline status: {result['pipeline_status']}")
    if result['gnn_results']['status'] == 'success':
        print(f"[TEST RESULT] Anomaly scores found: {len(result['gnn_results']['anomaly_scores'])}")
        print(f"[TEST RESULT] RL decision: {'ALERT' if result['rl_prediction']['should_alert'] else 'HOLD'}")
        print(f"[TEST RESULT] Alert sent: {result['alert_result'].get('alert_sent', False)}")
    
    # Test system stats
    stats = engine.get_system_stats()
    print(f"[TEST STATS] RL Agent Q-table size: {stats['rl_agent_stats']['q_table_size']}")
    print(f"[TEST STATS] Telegram configured: {stats['telegram_configured']}")
    
    return result

if __name__ == "__main__":
    test_stealth_engine_advanced()