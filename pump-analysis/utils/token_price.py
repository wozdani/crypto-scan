#!/usr/bin/env python3
"""
Token Price Utilities
Handles token price fetching from various sources
Transferred from crypto-scan for pump-analysis independence
"""

import os
import requests
import logging
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)

def get_token_price_usd(symbol: str) -> Optional[float]:
    """
    Get current USD price for a token symbol
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT', 'ETHUSDT')
    
    Returns:
        USD price as float or None if not found
    """
    try:
        # Remove USDT suffix for price lookup
        base_symbol = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
        
        # Try CoinGecko API first
        price = get_coingecko_price(base_symbol)
        if price:
            return price
        
        # Fallback to Bybit price if available
        price = get_bybit_price(symbol)
        if price:
            return price
        
        logger.warning(f"No price found for {symbol}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting token price for {symbol}: {e}")
        return None

def get_coingecko_price(symbol: str) -> Optional[float]:
    """
    Get price from CoinGecko API
    
    Args:
        symbol: Token symbol
    
    Returns:
        USD price or None
    """
    try:
        # Map common symbols to CoinGecko IDs
        symbol_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'ADA': 'cardano',
            'DOT': 'polkadot',
            'LINK': 'chainlink',
            'LTC': 'litecoin',
            'BCH': 'bitcoin-cash',
            'XRP': 'ripple',
            'MATIC': 'matic-network',
            'AVAX': 'avalanche-2',
            'SOL': 'solana',
            'UNI': 'uniswap',
            'AAVE': 'aave',
            'SUSHI': 'sushi',
            'CAKE': 'pancakeswap-token'
        }
        
        coin_id = symbol_mapping.get(symbol.upper(), symbol.lower())
        
        api_key = os.getenv('COINGECKO_API_KEY')
        if api_key:
            url = f"https://pro-api.coingecko.com/api/v3/simple/price"
            headers = {'x-cg-pro-api-key': api_key}
        else:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            headers = {}
        
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if coin_id in data and 'usd' in data[coin_id]:
                price = data[coin_id]['usd']
                logger.debug(f"CoinGecko price for {symbol}: ${price}")
                return float(price)
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting CoinGecko price for {symbol}: {e}")
        return None

def get_bybit_price(symbol: str) -> Optional[float]:
    """
    Get price from Bybit API
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
    
    Returns:
        USD price or None
    """
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        params = {
            'category': 'linear',
            'symbol': symbol
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                ticker = data['result']['list'][0]
                price = float(ticker.get('lastPrice', 0))
                if price > 0:
                    logger.debug(f"Bybit price for {symbol}: ${price}")
                    return price
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting Bybit price for {symbol}: {e}")
        return None

def get_historical_price(symbol: str, timestamp: int) -> Optional[float]:
    """
    Get historical price for a symbol at specific timestamp
    
    Args:
        symbol: Trading symbol
        timestamp: Unix timestamp
    
    Returns:
        USD price at that time or None
    """
    try:
        base_symbol = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
        
        # Map to CoinGecko ID
        symbol_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'ADA': 'cardano',
            'DOT': 'polkadot',
            'LINK': 'chainlink'
        }
        
        coin_id = symbol_mapping.get(base_symbol.upper(), base_symbol.lower())
        
        # Convert timestamp to date format
        from datetime import datetime
        date_str = datetime.fromtimestamp(timestamp).strftime('%d-%m-%Y')
        
        api_key = os.getenv('COINGECKO_API_KEY')
        if api_key:
            url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/history"
            headers = {'x-cg-pro-api-key': api_key}
        else:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/history"
            headers = {}
        
        params = {
            'date': date_str
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if 'market_data' in data and 'current_price' in data['market_data']:
                price = data['market_data']['current_price'].get('usd')
                if price:
                    logger.debug(f"Historical price for {symbol} on {date_str}: ${price}")
                    return float(price)
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting historical price for {symbol}: {e}")
        return None

def calculate_price_change(current_price: float, historical_price: float) -> Dict[str, float]:
    """
    Calculate price change metrics
    
    Args:
        current_price: Current price
        historical_price: Historical price for comparison
    
    Returns:
        Dictionary with change metrics
    """
    try:
        if not current_price or not historical_price or historical_price <= 0:
            return {"error": "invalid_prices"}
        
        absolute_change = current_price - historical_price
        percentage_change = (absolute_change / historical_price) * 100
        
        return {
            "current_price": current_price,
            "historical_price": historical_price,
            "absolute_change": absolute_change,
            "percentage_change": percentage_change,
            "multiplier": current_price / historical_price
        }
        
    except Exception as e:
        logger.error(f"Error calculating price change: {e}")
        return {"error": str(e)}