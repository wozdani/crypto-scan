"""
Blockchain Transaction Fetcher for DiamondWhale AI
Pobiera rzeczywiste transakcje blockchain dla analizy temporal graph
"""

import requests
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BlockchainTransactionFetcher:
    """
    Fetches real blockchain transactions for DiamondWhale AI analysis
    """
    
    def __init__(self):
        self.api_keys = {
            'ethereum': os.getenv('ETHERSCAN_API_KEY', ''),
            'bsc': os.getenv('BSCSCAN_API_KEY', ''),
            'polygon': os.getenv('POLYGONSCAN_API_KEY', ''),
            'arbitrum': os.getenv('ARBISCAN_API_KEY', ''),
            'optimism': os.getenv('OPTIMISMSCAN_API_KEY', '')
        }
        
        self.api_endpoints = {
            'ethereum': 'https://api.etherscan.io/api',
            'bsc': 'https://api.bscscan.com/api',
            'polygon': 'https://api.polygonscan.com/api',
            'arbitrum': 'https://api.arbiscan.io/api',
            'optimism': 'https://api-optimistic.etherscan.io/api'
        }
        
        logger.info("[BLOCKCHAIN FETCHER] Initialized")
    
    def fetch_token_transactions(self, contract_address: str, chain: str = 'ethereum', limit: int = 100) -> List[Dict]:
        """
        Fetch ERC-20 token transfer transactions for given contract
        
        Args:
            contract_address: Token contract address
            chain: Blockchain network
            limit: Maximum number of transactions to fetch
            
        Returns:
            List of transaction dictionaries with from/to addresses, values, timestamps
        """
        api_key = self.api_keys.get(chain)
        endpoint = self.api_endpoints.get(chain)
        
        if not api_key or not endpoint:
            logger.warning(f"[BLOCKCHAIN FETCH] Missing API key or endpoint for {chain}")
            return []
        
        # Get recent token transfers
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": contract_address,
            "page": 1,
            "offset": limit,
            "sort": "desc",
            "apikey": api_key
        }
        
        try:
            logger.info(f"[BLOCKCHAIN FETCH] Fetching token transfers for {contract_address[:10]}... on {chain}")
            response = requests.get(endpoint, params=params, timeout=15)
            data = response.json()
            
            transactions = []
            if data.get("status") == "1" and data.get("result"):
                for tx in data["result"][:limit]:
                    # Convert token value (considering decimals)
                    try:
                        value_raw = int(tx.get("value", 0))
                        decimals = int(tx.get("tokenDecimal", 18))
                        value_tokens = value_raw / (10 ** decimals)
                        
                        # Only include meaningful transfers (> 0.001 tokens)
                        if value_tokens > 0.001:
                            transactions.append({
                                "from_address": tx["from"].lower(),
                                "to_address": tx["to"].lower(),
                                "value_tokens": value_tokens,
                                "value_usd": value_tokens * float(tx.get("tokenPrice", 1.0)),  # Approximate USD value
                                "timestamp": int(tx["timeStamp"]),
                                "hash": tx["hash"],
                                "token_symbol": tx.get("tokenSymbol", ""),
                                "token_name": tx.get("tokenName", ""),
                                "gas_used": int(tx.get("gasUsed", 0))
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[BLOCKCHAIN FETCH] Error parsing transaction: {e}")
                        continue
                
                logger.info(f"[BLOCKCHAIN FETCH] Found {len(transactions)} meaningful token transfers")
                return transactions
            else:
                logger.warning(f"[BLOCKCHAIN FETCH] No token transfers found for {contract_address}")
                return []
                
        except Exception as e:
            logger.error(f"[BLOCKCHAIN FETCH] Error fetching token transactions: {e}")
            return []
    
    def fetch_whale_transactions(self, contract_address: str, chain: str = 'ethereum', 
                               min_value_usd: float = 10000.0, hours_back: int = 24) -> List[Dict]:
        """
        Fetch large whale transactions for the token
        
        Args:
            contract_address: Token contract address
            chain: Blockchain network
            min_value_usd: Minimum USD value to consider as whale transaction
            hours_back: Hours to look back for transactions
            
        Returns:
            List of whale transaction dictionaries
        """
        all_transactions = self.fetch_token_transactions(contract_address, chain, limit=200)
        
        # Filter for whale transactions
        whale_transactions = []
        cutoff_time = datetime.now().timestamp() - (hours_back * 3600)
        
        for tx in all_transactions:
            if (tx["timestamp"] > cutoff_time and 
                tx["value_usd"] >= min_value_usd):
                whale_transactions.append(tx)
        
        logger.info(f"[WHALE FETCH] Found {len(whale_transactions)} whale transactions (>${min_value_usd:.0f})")
        return whale_transactions
    
    def get_transaction_graph_data(self, contract_address: str, chain: str = 'ethereum') -> Dict:
        """
        Get transaction data formatted for temporal graph analysis
        
        Args:
            contract_address: Token contract address
            chain: Blockchain network
            
        Returns:
            Dictionary with transaction graph data
        """
        transactions = self.fetch_token_transactions(contract_address, chain)
        
        if not transactions:
            return {"transactions": [], "nodes": [], "edges": []}
        
        # Extract unique addresses (nodes)
        addresses = set()
        for tx in transactions:
            addresses.add(tx["from_address"])
            addresses.add(tx["to_address"])
        
        address_list = list(addresses)
        
        # Create edges (transactions between addresses)
        edges = []
        for i, tx in enumerate(transactions):
            from_idx = address_list.index(tx["from_address"])
            to_idx = address_list.index(tx["to_address"])
            
            edges.append({
                "from_idx": from_idx,
                "to_idx": to_idx,
                "value_usd": tx["value_usd"],
                "timestamp": tx["timestamp"],
                "tx_hash": tx["hash"]
            })
        
        result = {
            "transactions": transactions,
            "nodes": address_list,
            "edges": edges,
            "graph_stats": {
                "total_nodes": len(address_list),
                "total_edges": len(edges),
                "total_value": sum(tx["value_usd"] for tx in transactions),
                "time_span_hours": (max(tx["timestamp"] for tx in transactions) - 
                                   min(tx["timestamp"] for tx in transactions)) / 3600 if transactions else 0
            }
        }
        
        logger.info(f"[GRAPH DATA] Generated graph: {len(address_list)} nodes, {len(edges)} edges")
        return result


# Global instance
_blockchain_fetcher = None

def get_blockchain_fetcher() -> BlockchainTransactionFetcher:
    """Get singleton blockchain fetcher instance"""
    global _blockchain_fetcher
    if _blockchain_fetcher is None:
        _blockchain_fetcher = BlockchainTransactionFetcher()
    return _blockchain_fetcher

def fetch_diamond_transactions(contract_address: str, chain: str = 'ethereum') -> List[Dict]:
    """
    Convenience function to fetch transactions for DiamondWhale AI
    
    Args:
        contract_address: Token contract address
        chain: Blockchain network
        
    Returns:
        List of transaction dictionaries
    """
    fetcher = get_blockchain_fetcher()
    return fetcher.fetch_token_transactions(contract_address, chain)

def fetch_diamond_whale_transactions(contract_address: str, chain: str = 'ethereum') -> List[Dict]:
    """
    Fetch whale transactions for DiamondWhale AI analysis
    
    Args:
        contract_address: Token contract address
        chain: Blockchain network
        
    Returns:
        List of whale transaction dictionaries
    """
    fetcher = get_blockchain_fetcher()
    return fetcher.fetch_whale_transactions(contract_address, chain)