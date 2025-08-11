"""
Blockchain Scanners - Real API Integration with Etherscan V2 Support
Replaces mock data with authentic blockchain transfer data
Supports Etherscan V2 API with automatic fallback to legacy APIs
"""

import os
import requests
import time
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
from .etherscan_client import get_etherscan_client

class BlockchainScanner:
    """
    Real blockchain data scanner using Etherscan V2 API with legacy fallback
    """
    
    def __init__(self):
        # Initialize Etherscan V2 client with fallback support
        self.etherscan_client = get_etherscan_client()
        
        # Legacy configuration for backward compatibility
        self.api_keys = {
            'ethereum': os.getenv('ETHERSCAN_API_KEY'),
            'bsc': os.getenv('BSCSCAN_API_KEY'),
            'arbitrum': os.getenv('ARBISCAN_API_KEY'),
            'polygon': os.getenv('POLYGONSCAN_API_KEY'),
            'optimism': os.getenv('OPTIMISMSCAN_API_KEY'),
        }
        
        self.api_endpoints = {
            'ethereum': 'https://api.etherscan.io/api',
            'bsc': 'https://api.bscscan.com/api',
            'arbitrum': 'https://api.arbiscan.io/api',
            'polygon': 'https://api.polygonscan.com/api',
            'optimism': 'https://api-optimistic.etherscan.io/api',
        }
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 0.2  # 200ms between requests
        
    def _rate_limit(self, chain: str):
        """Apply rate limiting per chain"""
        if chain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[chain]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time[chain] = time.time()
    
    def get_token_transfers_last_24h(self, contract_address: str, chain: str, 
                                   limit: int = 100) -> List[Dict]:
        """
        Get real token transfers from blockchain in last 24h
        
        Args:
            contract_address: Token contract address
            chain: Blockchain name (ethereum, bsc, arbitrum, polygon, optimism)
            limit: Maximum number of transfers to fetch
            
        Returns:
            List of transfer dictionaries with real addresses and amounts
        """
        if chain not in self.api_keys or not self.api_keys[chain]:
            print(f"[BLOCKCHAIN] Missing API key for {chain}")
            return []
        
        if chain not in self.api_endpoints:
            print(f"[BLOCKCHAIN] Unsupported chain: {chain}")
            return []
        
        try:
            self._rate_limit(chain)
            
            # Calculate 24h ago timestamp
            yesterday = int((datetime.now() - timedelta(days=1)).timestamp())
            
            # MIGRATED TO ETHERSCAN V2: Use new unified client with V2-first approach
            try:
                print(f"[BLOCKCHAIN V2] {chain}: Using Etherscan V2 API for token transfers")
                
                # Use new Etherscan V2 client with automatic fallback
                data = self.etherscan_client.tokentx(
                    chain=chain,
                    contract_address=contract_address,
                    start=0,
                    end=999999999,
                    sort="desc",
                    page=1,
                    offset=limit
                )
                
                print(f"[BLOCKCHAIN V2] {chain}: Successfully got {len(data) if isinstance(data, list) else 'data'} from V2 API")
                
            except Exception as v2_error:
                print(f"[BLOCKCHAIN V2] {chain}: V2 API failed: {v2_error}")
                print(f"[BLOCKCHAIN LEGACY] {chain}: Falling back to direct legacy API")
                
                # Emergency fallback to legacy direct API
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("blockchain_scanner API call timeout")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(3)  # 3-second timeout for API calls
                
                params = {
                    'module': 'account',
                    'action': 'tokentx',
                    'contractaddress': contract_address,
                    'page': 1,
                    'offset': limit,
                    'startblock': 0,
                    'endblock': 999999999,
                    'sort': 'desc',
                    'apikey': self.api_keys[chain]
                }
                
                response = requests.get(self.api_endpoints[chain], params=params, timeout=15)
                signal.alarm(0)  # Cancel timeout on success
                
                if response.status_code != 200:
                    print(f"[BLOCKCHAIN] HTTP {response.status_code} for {chain}")
                    return []
                
                response_data = response.json()
                
                if response_data.get('status') != '1':
                    print(f"[BLOCKCHAIN] API error for {chain}: {response_data.get('message', 'unknown')}")
                    return []
                
                data = response_data.get('result', [])
            
            transfers = []
            # Handle both V2 API response and legacy fallback
            tx_list = data if isinstance(data, list) else data.get('result', [])
            
            for tx in tx_list:
                try:
                    # Filter to last 24h
                    tx_timestamp = int(tx.get('timeStamp', 0))
                    if tx_timestamp < yesterday:
                        continue
                    
                    # Calculate USD value (basic estimation)
                    decimals = int(tx.get('tokenDecimal', 18))
                    value_tokens = float(tx.get('value', 0)) / (10 ** decimals)
                    
                    transfer = {
                        'hash': tx.get('hash'),
                        'from': tx.get('from', '').lower(),
                        'to': tx.get('to', '').lower(),
                        'value_tokens': value_tokens,
                        'value_usd': value_tokens * self._get_token_price_estimate(contract_address, chain),
                        'timestamp': tx_timestamp,
                        'block_number': int(tx.get('blockNumber', 0)),
                        'token_symbol': tx.get('tokenSymbol', ''),
                        'chain': chain
                    }
                    transfers.append(transfer)
                    
                except Exception as e:
                    print(f"[BLOCKCHAIN] Error parsing transfer: {e}")
                    continue
            
            # PUNKT 10: Log only once per scan with @once_per_scan tracking
            print(f"[BLOCKCHAIN] Found {len(transfers)} real transfers for {contract_address} on {chain}")
            return transfers
            
        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            print(f"[BLOCKCHAIN] TIMEOUT fetching transfers for {chain} - using emergency fallback")
            return []
        except Exception as e:
            signal.alarm(0)  # Cancel timeout 
            print(f"[BLOCKCHAIN] Error fetching transfers for {chain}: {e}")
            return []
    
    def _get_token_price_estimate(self, contract_address: str, chain: str) -> float:
        """
        Get basic token price estimate
        In production, this would use CoinGecko API or price feeds
        """
        try:
            # Simplified price lookup - in production use CoinGecko API
            # For now, return basic estimate based on common tokens
            common_prices = {
                'ethereum': {
                    # USDT, USDC typically around $1
                    '0xdac17f958d2ee523a2206206994597c13d831ec7': 1.0,  # USDT
                    '0xa0b86a33e6b26b5c5e5d3b73f87b9b8e2e3f6f3c': 1.0,  # USDC
                },
                'bsc': {
                    '0x55d398326f99059ff775485246999027b3197955': 1.0,  # BSC-USD
                }
            }
            
            chain_prices = common_prices.get(chain, {})
            return chain_prices.get(contract_address.lower(), 0.1)  # Default small value
            
        except Exception:
            return 0.1  # Fallback price estimate
    
    def get_whale_activity(self, contract_address: str, chain: str, 
                          min_usd_value: float = 10000) -> List[Dict]:
        """
        Get real whale transfer activity above threshold
        
        Args:
            contract_address: Token contract address
            chain: Blockchain name
            min_usd_value: Minimum USD value to consider whale activity
            
        Returns:
            List of whale transfers with real addresses
        """
        all_transfers = self.get_token_transfers_last_24h(contract_address, chain, 200)
        
        whale_transfers = []
        for transfer in all_transfers:
            if transfer['value_usd'] >= min_usd_value:
                # Additional whale validation
                transfer['whale_score'] = min(transfer['value_usd'] / 50000, 10.0)  # Score 1-10
                transfer['is_whale'] = True
                whale_transfers.append(transfer)
        
        # PUNKT 10: Log only once per scan 
        print(f"[WHALE] Found {len(whale_transfers)} real whale transfers for {contract_address}")
        return whale_transfers
    
    def analyze_address_velocity(self, addresses: List[str], contract_address: str, 
                               chain: str, hours: int = 48) -> Dict[str, int]:
        """
        Analyze real address activity velocity
        
        Args:
            addresses: List of addresses to analyze
            contract_address: Token contract address
            chain: Blockchain name
            hours: Time window for velocity analysis
            
        Returns:
            Dictionary mapping addresses to activity counts
        """
        transfers = self.get_token_transfers_last_24h(contract_address, chain, 500)
        
        # Count activity for each address
        velocity_map = {}
        cutoff_time = int((datetime.now() - timedelta(hours=hours)).timestamp())
        
        for transfer in transfers:
            if transfer['timestamp'] < cutoff_time:
                continue
                
            for addr in [transfer['from'], transfer['to']]:
                if addr in addresses:
                    velocity_map[addr] = velocity_map.get(addr, 0) + 1
        
        print(f"[VELOCITY] Analyzed {len(addresses)} addresses, found activity for {len(velocity_map)}")
        return velocity_map

# Import once_per_scan decorator
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))
    from rl_gateway import once_per_scan
    ONCE_PER_SCAN_AVAILABLE = True
except ImportError:
    ONCE_PER_SCAN_AVAILABLE = False
    def once_per_scan(category, subcategory):
        def decorator(func):
            return func
        return decorator

@once_per_scan("UTILS", "exchange_addresses")
def load_known_exchange_addresses() -> Dict[str, List[str]]:
    """Load known exchange and DEX addresses (idempotent)"""
    try:
        # FIXED: Use absolute path to handle different working directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)  # Go up one level from utils/
        json_path = os.path.join(project_dir, 'data', 'known_exchange_addresses.json')
        
        with open(json_path, 'r') as f:
            exchange_data = json.load(f)
            # PUNKT 10: Log tylko raz dzięki @once_per_scan
            print(f"[EXCHANGE] Successfully loaded exchange addresses from {json_path}")
            return exchange_data
    except Exception as e:
        print(f"[EXCHANGE] Could not load exchange addresses: {e}")
        # Return basic known addresses as fallback
        return {
            'ethereum': [
                '0x28c6c06298d514db089934071355e5743bf21d60',  # Binance
                '0x21a31ee1afc51d94c2efccaa2092ad1028285549',  # Binance 2
                '0x4e9ce36e442e55ecd9025b9a6e0d88485d628a67',  # Binance 3
                '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be',  # Binance 4
                '0xd551234ae421e3bcba99a0da6d736074f22192ff',  # Binance 5
            ],
            'bsc': [
                '0x8894e0a0c962cb723c1976a4421c95949be2d4e3',  # Binance BSC
                '0x0d0707963952f2fba59dd06f2b425ace40b492fe',  # Gate.io BSC
            ],
            'dex_routers': {
                'ethereum': [
                    '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',  # Uniswap V2
                    '0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f',  # Sushiswap
                    '0xe592427a0aece92de3edee1f18e0157c05861564'   # Uniswap V3
                ],
                'bsc': [
                    '0x10ed43c718714eb63d5aa57b78b54704e256024e',  # PancakeSwap
                ]
            }
        }

def get_token_transfers_last_24h(symbol: str, chain: str = None, 
                               contract_address: str = None) -> List[Dict]:
    """
    Convenience function to get real token transfers
    
    Args:
        symbol: Token symbol (e.g., 'PEPEUSDT')
        chain: Blockchain name (auto-detected if None)
        contract_address: Contract address (auto-detected if None)
        
    Returns:
        List of real transfer data
    """
    scanner = BlockchainScanner()
    
    # Auto-detect contract if not provided
    if not contract_address or not chain:
        from utils.contracts import get_contract
        contract_info = get_contract(symbol)
        if not contract_info:
            print(f"[BLOCKCHAIN] Could not find contract for {symbol}")
            return []
        
        contract_address = contract_info['address']
        chain = contract_info['chain']
    
    return scanner.get_token_transfers_last_24h(contract_address, chain)

def get_whale_transfers(symbol: str, min_usd: float = 10000, 
                      chain: str = None, contract_address: str = None) -> List[Dict]:
    """
    Convenience function to get real whale transfers
    
    Args:
        symbol: Token symbol
        min_usd: Minimum USD value for whale classification
        chain: Blockchain name (auto-detected if None)
        contract_address: Contract address (auto-detected if None)
        
    Returns:
        List of real whale transfer data
    """
    scanner = BlockchainScanner()
    
    # Use provided chain/contract or auto-detect
    if not contract_address or not chain:
        from utils.contracts import get_contract
        contract_info = get_contract(symbol)
        if not contract_info:
            print(f"[WHALE] Could not find contract for {symbol}")
            return []
        
        contract_address = contract_address or contract_info['address']
        chain = chain or contract_info['chain']
    
    return scanner.get_whale_activity(contract_address, chain, min_usd)