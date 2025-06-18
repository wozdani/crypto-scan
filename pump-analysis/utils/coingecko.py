#!/usr/bin/env python3
"""
CoinGecko API Integration
Handles token contract data and metadata from CoinGecko
Transferred from crypto-scan for pump-analysis independence
"""

import os
import json
import requests
import logging
from typing import Optional, Dict, Any, List
import time

logger = logging.getLogger(__name__)

def get_token_contract_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get token contract data from CoinGecko
    
    Args:
        symbol: Token symbol (e.g., 'ETH', 'UNI')
    
    Returns:
        Dictionary with contract information or None
    """
    try:
        # Check cache first
        cache_data = load_coingecko_cache()
        if symbol.upper() in cache_data:
            return cache_data[symbol.upper()]
        
        # Search for token in CoinGecko
        token_data = search_coingecko_token(symbol)
        if token_data:
            # Cache the result
            cache_data[symbol.upper()] = token_data
            save_coingecko_cache(cache_data)
            return token_data
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting CoinGecko contract data for {symbol}: {e}")
        return None

def search_coingecko_token(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Search for token in CoinGecko API
    
    Args:
        symbol: Token symbol
    
    Returns:
        Token data dictionary or None
    """
    try:
        api_key = os.getenv('COINGECKO_API_KEY')
        
        if api_key:
            url = "https://pro-api.coingecko.com/api/v3/coins/list"
            headers = {'x-cg-pro-api-key': api_key}
        else:
            url = "https://api.coingecko.com/api/v3/coins/list"
            headers = {}
        
        params = {
            'include_platform': 'true'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            coins = response.json()
            
            # Search for matching symbol
            for coin in coins:
                if coin.get('symbol', '').upper() == symbol.upper():
                    # Get platform data (contract addresses)
                    platforms = coin.get('platforms', {})
                    
                    # Determine primary chain and address
                    primary_chain = None
                    primary_address = None
                    
                    # Priority order for chains
                    chain_priority = ['ethereum', 'binance-smart-chain', 'polygon-pos', 'arbitrum-one', 'optimistic-ethereum']
                    
                    for chain in chain_priority:
                        if chain in platforms and platforms[chain]:
                            primary_chain = chain
                            primary_address = platforms[chain]
                            break
                    
                    # If no priority chain found, use first available
                    if not primary_chain and platforms:
                        for chain, address in platforms.items():
                            if address:
                                primary_chain = chain
                                primary_address = address
                                break
                    
                    return {
                        'symbol': coin.get('symbol', '').upper(),
                        'name': coin.get('name', ''),
                        'id': coin.get('id', ''),
                        'chain': map_coingecko_chain(primary_chain) if primary_chain else None,
                        'address': primary_address,
                        'platforms': platforms,
                        'source': 'coingecko'
                    }
        
        return None
        
    except Exception as e:
        logger.error(f"Error searching CoinGecko for {symbol}: {e}")
        return None

def map_coingecko_chain(coingecko_chain: str) -> str:
    """
    Map CoinGecko chain names to our internal chain names
    
    Args:
        coingecko_chain: CoinGecko chain identifier
    
    Returns:
        Internal chain name
    """
    mapping = {
        'ethereum': 'ethereum',
        'binance-smart-chain': 'bsc',
        'polygon-pos': 'polygon',
        'arbitrum-one': 'arbitrum',
        'optimistic-ethereum': 'optimism',
        'avalanche': 'avalanche',
        'fantom': 'fantom',
        'harmony-shard-0': 'harmony',
        'xdai': 'gnosis',
        'moonbeam': 'moonbeam',
        'cronos': 'cronos'
    }
    
    return mapping.get(coingecko_chain, coingecko_chain)

def load_coingecko_cache() -> Dict[str, Any]:
    """
    Load CoinGecko cache from file
    
    Returns:
        Dictionary with cached data
    """
    try:
        cache_file = "data/cache/coingecko_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading CoinGecko cache: {e}")
    
    return {}

def save_coingecko_cache(cache_data: Dict[str, Any]) -> bool:
    """
    Save CoinGecko cache to file
    
    Args:
        cache_data: Data to cache
    
    Returns:
        True if successful
    """
    try:
        cache_dir = "data/cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, "coingecko_cache.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"CoinGecko cache saved with {len(cache_data)} entries")
        return True
        
    except Exception as e:
        logger.error(f"Error saving CoinGecko cache: {e}")
        return False

def get_coin_market_data(coin_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed market data for a coin from CoinGecko
    
    Args:
        coin_id: CoinGecko coin ID
    
    Returns:
        Market data dictionary or None
    """
    try:
        api_key = os.getenv('COINGECKO_API_KEY')
        
        if api_key:
            url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}"
            headers = {'x-cg-pro-api-key': api_key}
        else:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            headers = {}
        
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'false',
            'developer_data': 'false',
            'sparkline': 'false'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            market_data = data.get('market_data', {})
            
            return {
                'id': data.get('id'),
                'symbol': data.get('symbol', '').upper(),
                'name': data.get('name'),
                'current_price': market_data.get('current_price', {}).get('usd'),
                'market_cap': market_data.get('market_cap', {}).get('usd'),
                'total_volume': market_data.get('total_volume', {}).get('usd'),
                'price_change_24h': market_data.get('price_change_percentage_24h'),
                'circulating_supply': market_data.get('circulating_supply'),
                'total_supply': market_data.get('total_supply'),
                'platforms': data.get('platforms', {}),
                'categories': data.get('categories', []),
                'description': data.get('description', {}).get('en', '')[:200] if data.get('description', {}).get('en') else None
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting market data for {coin_id}: {e}")
        return None

def batch_get_prices(coin_ids: List[str]) -> Dict[str, float]:
    """
    Get prices for multiple coins in a single request
    
    Args:
        coin_ids: List of CoinGecko coin IDs
    
    Returns:
        Dictionary mapping coin_id to USD price
    """
    try:
        if not coin_ids:
            return {}
        
        # Limit to 100 coins per request
        coin_ids = coin_ids[:100]
        
        api_key = os.getenv('COINGECKO_API_KEY')
        
        if api_key:
            url = "https://pro-api.coingecko.com/api/v3/simple/price"
            headers = {'x-cg-pro-api-key': api_key}
        else:
            url = "https://api.coingecko.com/api/v3/simple/price"
            headers = {}
        
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd',
            'include_24hr_change': 'true'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            prices = {}
            for coin_id, price_data in data.items():
                if 'usd' in price_data:
                    prices[coin_id] = price_data['usd']
            
            return prices
        
        return {}
        
    except Exception as e:
        logger.error(f"Error getting batch prices: {e}")
        return {}

def verify_contract_address(symbol: str, address: str, chain: str) -> bool:
    """
    Verify if a contract address is correct for a given symbol
    
    Args:
        symbol: Token symbol
        address: Contract address to verify
        chain: Blockchain name
    
    Returns:
        True if verified, False otherwise
    """
    try:
        token_data = get_token_contract_data(symbol)
        if not token_data:
            return False
        
        # Check if address matches
        if token_data.get('address', '').lower() == address.lower():
            return True
        
        # Check platforms for matching chain and address
        platforms = token_data.get('platforms', {})
        coingecko_chain = None
        
        # Map our chain to CoinGecko chain
        chain_mapping = {
            'ethereum': 'ethereum',
            'bsc': 'binance-smart-chain',
            'polygon': 'polygon-pos',
            'arbitrum': 'arbitrum-one',
            'optimism': 'optimistic-ethereum'
        }
        
        coingecko_chain = chain_mapping.get(chain.lower())
        if coingecko_chain and coingecko_chain in platforms:
            return platforms[coingecko_chain].lower() == address.lower()
        
        return False
        
    except Exception as e:
        logger.error(f"Error verifying contract address: {e}")
        return False