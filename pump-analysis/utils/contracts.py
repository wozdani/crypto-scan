#!/usr/bin/env python3
"""
Contract Address Management
Handles token contract addresses and chain identification
Transferred from crypto-scan for pump-analysis independence
"""

import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Chain mapping for symbol-to-chain identification
CHAIN_MAPPINGS = {
    # Ethereum tokens
    "ETH": "ethereum",
    "USDT": "ethereum", 
    "USDC": "ethereum",
    "DAI": "ethereum",
    "WETH": "ethereum",
    "UNI": "ethereum",
    "LINK": "ethereum",
    "AAVE": "ethereum",
    "MKR": "ethereum",
    "COMP": "ethereum",
    "SNX": "ethereum",
    "SUSHI": "ethereum",
    "CRV": "ethereum",
    "YFI": "ethereum",
    "1INCH": "ethereum",
    "BAL": "ethereum",
    "ZRX": "ethereum",
    "LRC": "ethereum",
    "ENJ": "ethereum",
    "MANA": "ethereum",
    "SAND": "ethereum",
    "AXS": "ethereum",
    "CHZ": "ethereum",
    "BAT": "ethereum",
    "ZIL": "ethereum",
    "HOT": "ethereum",
    "ICX": "ethereum",
    "IOST": "ethereum",
    "QTUM": "ethereum",
    "OMG": "ethereum",
    "REP": "ethereum",
    "ZEN": "ethereum",
    "STORJ": "ethereum",
    "GNT": "ethereum",
    "REN": "ethereum",
    "KNC": "ethereum",
    "ANT": "ethereum",
    "MLN": "ethereum",
    "NMR": "ethereum",
    "DNT": "ethereum",
    "WBTC": "ethereum",
    
    # BSC tokens
    "BNB": "bsc",
    "CAKE": "bsc",
    "BUSD": "bsc",
    "XVS": "bsc",
    "BAKE": "bsc",
    "BURGER": "bsc",
    "AUTO": "bsc",
    "ALPHA": "bsc",
    "SXP": "bsc",
    "TWT": "bsc",
    "VAI": "bsc",
    "WIN": "bsc",
    "TRX": "bsc",
    "BTT": "bsc",
    
    # Polygon tokens
    "MATIC": "polygon",
    "QUICK": "polygon",
    "GHST": "polygon",
    "DFYN": "polygon",
    "POLYSWAP": "polygon",
    
    # Arbitrum tokens
    "ARB": "arbitrum",
    "GMX": "arbitrum",
    "MAGIC": "arbitrum",
    "DPX": "arbitrum",
    "RDNT": "arbitrum",
    
    # Optimism tokens
    "OP": "optimism",
    "VELO": "optimism",
    "SNX": "optimism",
    
    # Native blockchain tokens
    "BTC": "bitcoin",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "ETC": "ethereum-classic",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOT": "polkadot",
    "SOL": "solana",
    "AVAX": "avalanche",
    "LUNA": "terra",
    "ATOM": "cosmos",
    "NEAR": "near",
    "FTM": "fantom",
    "ALGO": "algorand",
    "XTZ": "tezos",
    "EGLD": "elrond",
    "FLOW": "flow",
    "ICP": "internet-computer",
    "THETA": "theta",
    "VET": "vechain",
    "FIL": "filecoin",
    "EOS": "eos",
    "XLM": "stellar",
    "XMR": "monero",
    "DASH": "dash",
    "ZEC": "zcash",
    "WAVES": "waves",
    "KSM": "kusama",
    "DCR": "decred",
    "ONT": "ontology",
    "ZIL": "zilliqa",
    "RVN": "ravencoin"
}

def get_contract_address(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get contract address for a trading symbol
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT', 'ETHUSDT')
    
    Returns:
        Dictionary with contract information or None
    """
    try:
        # Remove USDT suffix for contract lookup
        base_symbol = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
        
        # Check if we have cached contract data
        contract_cache_file = "data/cache/contract_cache.json"
        if os.path.exists(contract_cache_file):
            try:
                with open(contract_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                if base_symbol in cache_data:
                    return cache_data[base_symbol]
            except Exception as e:
                logger.warning(f"Error reading contract cache: {e}")
        
        # Fallback to chain mapping
        chain = get_chain_for_symbol(base_symbol)
        if chain:
            return {
                "symbol": base_symbol,
                "chain": chain,
                "address": None,  # Would need to be populated from actual contract data
                "source": "chain_mapping"
            }
        
        logger.warning(f"No contract information found for {symbol}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting contract address for {symbol}: {e}")
        return None

def get_chain_for_symbol(symbol: str) -> Optional[str]:
    """
    Get blockchain chain for a given symbol
    
    Args:
        symbol: Token symbol (e.g., 'ETH', 'BNB', 'MATIC')
    
    Returns:
        Chain name or None if not found
    """
    try:
        # Direct mapping lookup
        chain = CHAIN_MAPPINGS.get(symbol.upper())
        if chain:
            return chain
        
        # Default fallback for unknown tokens - assume Ethereum
        if symbol not in ['BTC', 'LTC', 'BCH', 'XRP', 'ADA', 'DOT', 'SOL', 'AVAX']:
            return "ethereum"
        
        return None
        
    except Exception as e:
        logger.error(f"Error determining chain for {symbol}: {e}")
        return None

def load_contract_cache() -> Dict[str, Any]:
    """
    Load contract cache from file
    
    Returns:
        Dictionary with cached contract data
    """
    try:
        cache_file = "data/cache/contract_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading contract cache: {e}")
    
    return {}

def save_contract_cache(cache_data: Dict[str, Any]) -> bool:
    """
    Save contract cache to file
    
    Args:
        cache_data: Dictionary with contract data to cache
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cache_dir = "data/cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, "contract_cache.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Contract cache saved with {len(cache_data)} entries")
        return True
        
    except Exception as e:
        logger.error(f"Error saving contract cache: {e}")
        return False

def update_contract_info(symbol: str, contract_info: Dict[str, Any]) -> bool:
    """
    Update contract information in cache
    
    Args:
        symbol: Token symbol
        contract_info: Contract information dictionary
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cache_data = load_contract_cache()
        cache_data[symbol.upper()] = contract_info
        return save_contract_cache(cache_data)
        
    except Exception as e:
        logger.error(f"Error updating contract info for {symbol}: {e}")
        return False

def get_explorer_url(chain: str, address: str, tx_hash: Optional[str] = None) -> Optional[str]:
    """
    Get explorer URL for a given chain and address/transaction
    
    Args:
        chain: Blockchain name
        address: Contract address or wallet address
        tx_hash: Transaction hash (optional)
    
    Returns:
        Explorer URL or None
    """
    try:
        explorers = {
            "ethereum": "https://etherscan.io",
            "bsc": "https://bscscan.com", 
            "polygon": "https://polygonscan.com",
            "arbitrum": "https://arbiscan.io",
            "optimism": "https://optimistic.etherscan.io"
        }
        
        base_url = explorers.get(chain.lower())
        if not base_url:
            return None
        
        if tx_hash:
            return f"{base_url}/tx/{tx_hash}"
        else:
            return f"{base_url}/address/{address}"
            
    except Exception as e:
        logger.error(f"Error generating explorer URL: {e}")
        return None

def validate_contract_address(address: str, chain: str) -> bool:
    """
    Validate if an address is a valid contract address format
    
    Args:
        address: Contract address to validate
        chain: Blockchain name
    
    Returns:
        True if valid format, False otherwise
    """
    try:
        if not address:
            return False
        
        # Ethereum-based chains use 42-character hex addresses
        if chain.lower() in ["ethereum", "bsc", "polygon", "arbitrum", "optimism"]:
            if len(address) == 42 and address.startswith('0x'):
                try:
                    int(address[2:], 16)  # Check if hex
                    return True
                except ValueError:
                    return False
        
        # Add validation for other chains as needed
        return False
        
    except Exception as e:
        logger.error(f"Error validating contract address: {e}")
        return False