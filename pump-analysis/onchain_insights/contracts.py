import json
import os
from typing import Optional, Dict

# Mapowanie nazw chainów z CoinGecko na nasze aliasy
CHAIN_ALIASES = {
    "ethereum": "ethereum",
    "binance-smart-chain": "bsc",
    "polygon-pos": "polygon",
    "arbitrum-one": "arbitrum",
    "optimistic-ethereum": "optimism",
    "tron": "tron"
}

def normalize_token_name(symbol: str, cache: Dict) -> str:
    """Normalize token name for cache lookup"""
    symbol = symbol.upper().replace('USDT', '').replace('BUSD', '').replace('USD', '')
    
    # Try direct lookup first
    if symbol.lower() in cache:
        return symbol.lower()
    
    # Try variations
    variations = [symbol, symbol.lower(), symbol.upper()]
    for var in variations:
        if var in cache:
            return var
    
    return symbol.lower()

def load_coingecko_cache() -> Dict:
    """Load CoinGecko cache from multiple possible locations"""
    cache_paths = [
        os.path.join('..', 'crypto-scan', 'cache', 'coingecko_cache.json'),
        os.path.join('..', 'crypto-scan', 'coingecko_cache.json'),
        'cache/coingecko_cache.json',
        'coingecko_cache.json'
    ]
    
    for path in cache_paths:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache from {path}: {e}")
    
    return {}

def get_contract(symbol: str) -> Optional[Dict[str, str]]:
    """Get contract address and chain for a symbol"""
    cache = load_coingecko_cache()
    normalized = normalize_token_name(symbol, cache)
    token_data = cache.get(normalized)

    if token_data:
        platforms = token_data.get("platforms", {})
        for cg_chain, address in platforms.items():
            chain = CHAIN_ALIASES.get(cg_chain)
            if chain and address:
                return {"address": address, "chain": chain}

    return None