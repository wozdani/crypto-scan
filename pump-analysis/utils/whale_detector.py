#!/usr/bin/env python3
"""
Whale Transaction Detector
Enhanced detector for large transactions in 15-minute windows
Transferred from crypto-scan for pump-analysis independence
"""

import os
import requests
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Tuple, Dict, List, Optional

from .contracts import get_contract_address
from .token_price import get_token_price_usd

logger = logging.getLogger(__name__)

WHALE_MIN_USD = 50000  # Próg detekcji whale

def detect_whale_tx(symbol: str, price_usd: Optional[float] = None) -> Tuple[bool, int, float]:
    """
    Enhanced Whale Transaction Detector
    Wykrywa minimum 3 transakcje > 50,000 USD w ciągu ostatnich 15 minut
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        price_usd: Current token price in USD
    
    Returns:
        tuple: (whale_active, large_tx_count, total_usd)
    """
    logger.info(f"Detecting whale transactions for {symbol}")
    
    try:
        # Get contract information
        contract_info = get_contract_address(symbol)
        if not contract_info:
            logger.warning(f"No contract found for {symbol}")
            return False, 0, 0.0

        if not isinstance(contract_info, dict):
            logger.error(f"Invalid contract info for {symbol}: {type(contract_info)}")
            return False, 0, 0.0

        # Get price if not provided
        if not price_usd or price_usd == 0:
            price_usd = get_token_price_usd(symbol)
            if not price_usd:
                logger.warning(f"No USD price available for {symbol}")
                return False, 0, 0.0

        chain = contract_info.get("chain", "").lower()
        address = contract_info.get("address")

        if not chain or not address:
            logger.warning(f"Missing chain/address data for {symbol}")
            return False, 0, 0.0

        # Configure blockchain explorers
        explorer_configs = {
            "ethereum": {
                "url": "https://api.etherscan.io/api",
                "api_key": os.getenv("ETHERSCAN_API_KEY")
            },
            "bsc": {
                "url": "https://api.bscscan.com/api",
                "api_key": os.getenv("BSCSCAN_API_KEY")
            },
            "polygon": {
                "url": "https://api.polygonscan.com/api",
                "api_key": os.getenv("POLYGONSCAN_API_KEY")
            },
            "arbitrum": {
                "url": "https://api.arbiscan.io/api",
                "api_key": os.getenv("ARBISCAN_API_KEY")
            },
            "optimism": {
                "url": "https://api-optimistic.etherscan.io/api",
                "api_key": os.getenv("OPTIMISMSCAN_API_KEY")
            }
        }

        config = explorer_configs.get(chain)
        if not config or not config["api_key"]:
            logger.warning(f"Chain {chain} not supported or missing API key")
            return False, 0, 0.0

        # Calculate 15 minutes ago timestamp
        now_utc = datetime.now(timezone.utc)
        fifteen_minutes_ago = now_utc - timedelta(minutes=15)
        timestamp_threshold = int(fifteen_minutes_ago.timestamp())
        
        logger.info(f"Searching for transactions after {fifteen_minutes_ago.strftime('%H:%M:%S UTC')} for {symbol}")

        # API request parameters
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",
            "apikey": config["api_key"]
        }

        # Make API request with retry logic
        large_transactions = []
        large_tx_count = 0
        total_usd = 0.0

        for attempt in range(2):
            try:
                response = requests.get(config["url"], params=params, timeout=20)
                
                if response.status_code != 200:
                    logger.error(f"HTTP {response.status_code} for {symbol} on {chain}")
                    return False, 0, 0.0
                    
                data = response.json()

                if data.get("status") != "1":
                    logger.warning(f"API Error for {symbol}: {data.get('message', 'unknown')}")
                    return False, 0, 0.0

                txs = data.get("result", [])
                
                # Analyze transactions from last 15 minutes
                for tx in txs[:50]:  # Check more transactions for time filtering
                    try:
                        # Parse transaction timestamp
                        tx_timestamp = int(tx.get("timeStamp", 0))
                        
                        # Skip transactions older than 15 minutes
                        if tx_timestamp < timestamp_threshold:
                            continue
                        
                        # Calculate USD value
                        raw_value = int(tx["value"])
                        decimals = int(tx["tokenDecimal"])
                        token_amount = raw_value / (10 ** decimals)
                        usd_value = token_amount * price_usd

                        # Check if transaction is large enough (>50k USD)
                        if usd_value >= WHALE_MIN_USD:
                            large_transactions.append({
                                "hash": tx.get("hash", ""),
                                "timestamp": tx_timestamp,
                                "usd_value": usd_value,
                                "token_amount": token_amount
                            })
                            large_tx_count += 1
                            total_usd += usd_value
                            
                            tx_time = datetime.fromtimestamp(tx_timestamp, timezone.utc).strftime('%H:%M:%S')
                            logger.info(f"Large transaction for {symbol}: ${usd_value:,.0f} at {tx_time} UTC")

                    except (ValueError, KeyError, TypeError):
                        continue
                
                # Enhanced whale detection logic
                whale_active = False
                
                if large_tx_count >= 3 and total_usd > 150_000:
                    whale_active = True
                    logger.info(f"Whale activity detected for {symbol}: {large_tx_count} large TXs, total ${total_usd:,.0f}")
                elif large_tx_count >= 3:
                    whale_active = True
                    logger.info(f"Whale activity detected for {symbol}: {large_tx_count} large TXs (count threshold)")
                elif total_usd > 300_000:  # Single very large transaction
                    whale_active = True
                    logger.info(f"Whale activity detected for {symbol}: Single large volume ${total_usd:,.0f}")
                
                if large_tx_count > 0:
                    logger.info(f"Whale summary for {symbol}: {large_tx_count} TXs, ${total_usd:,.0f} total, active: {whale_active}")
                
                return whale_active, large_tx_count, total_usd
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {symbol} on {chain} (attempt {attempt + 1}/2)")
                if attempt == 0:
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"Definitive timeout for {symbol}")
                    return False, 0, 0.0
                    
            except Exception as e:
                logger.error(f"Error in whale detection for {symbol}: {e}")
                return False, 0, 0.0

        return False, 0, 0.0
        
    except Exception as e:
        logger.error(f"Unexpected error in whale detection for {symbol}: {e}")
        return False, 0, 0.0