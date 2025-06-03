import requests
import time
import os
import json

# Global cache for CoinGecko data
COINGECKO_TOKEN_LIST = []
CACHE_FILE = "coingecko_cache.json"
CACHE_DURATION_HOURS = 6

def load_cache():
    """Load cached CoinGecko data if valid"""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is still valid (6 hours)
        cache_time = cache_data.get('timestamp', 0)
        current_time = time.time()
        if current_time - cache_time < CACHE_DURATION_HOURS * 3600:
            return cache_data.get('contracts', {})
    except Exception as e:
        print(f"Error loading cache: {e}")
    
    return None

def save_cache(contracts_data):
    """Save contracts data to cache with timestamp"""
    try:
        cache_data = {
            'timestamp': time.time(),
            'contracts': contracts_data
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")

def fetch_coingecko_token_list():
    """Fetch complete token list from CoinGecko - single request at startup"""
    global COINGECKO_TOKEN_LIST
    if COINGECKO_TOKEN_LIST:
        return COINGECKO_TOKEN_LIST
    
    try:
        headers = {}
        api_key = os.getenv("COINGECKO_API_KEY")
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
            
        response = requests.get("https://api.coingecko.com/api/v3/coins/list", headers=headers, timeout=15)
        response.raise_for_status()
        COINGECKO_TOKEN_LIST = response.json()
        print(f"Loaded {len(COINGECKO_TOKEN_LIST)} tokens from CoinGecko")
        return COINGECKO_TOKEN_LIST
    except Exception as e:
        print(f"Error fetching CoinGecko token list: {e}")
        return []

def normalize_symbol_for_search(symbol):
    """Normalize symbol for CoinGecko search"""
    # Remove USDT, numbers from beginning
    cleaned = symbol.upper().replace("USDT", "").replace("PERP", "")
    cleaned = ''.join([c for c in cleaned if not c.isdigit()])
    return cleaned.strip()

def get_multiple_token_contracts_from_coingecko(symbols):
    """Fetch contracts for multiple tokens with caching and rate limiting"""
    # Check cache first
    cached_contracts = load_cache()
    if cached_contracts:
        print("Using cached contract data")
        return {symbol: cached_contracts.get(symbol) for symbol in symbols}
    
    token_list = fetch_coingecko_token_list()
    if not token_list:
        return {}
    
    result = {}
    found_tokens = {}
    
    # Find tokens in the list
    for symbol in symbols:
        normalized = normalize_symbol_for_search(symbol)
        match = next((t for t in token_list if t["symbol"].upper() == normalized), None)
        
        if match:
            found_tokens[symbol] = match["id"]
        else:
            result[symbol] = None
    
    print(f"Found {len(found_tokens)}/{len(symbols)} tokens in CoinGecko")
    
    # Fetch details with rate limiting and retry logic
    headers = {}
    api_key = os.getenv("COINGECKO_API_KEY")
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    
    for i, (symbol, token_id) in enumerate(found_tokens.items()):
        try:
            # Rate limiting - 1.5s between requests
            if i > 0:
                time.sleep(1.5)
            
            url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
            response = requests.get(url, headers=headers, timeout=10)
            
            # Handle rate limit with retry
            if response.status_code == 429:
                print(f"Rate limit hit - waiting 5 seconds...")
                time.sleep(5)
                response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                platforms = data.get("platforms", {})
                
                # Prefer BSC, then Ethereum, then others
                preferred_chains = ["binance-smart-chain", "ethereum", "polygon-pos", "arbitrum-one", "optimistic-ethereum"]
                
                contract_found = False
                for chain in preferred_chains:
                    if chain in platforms and platforms[chain]:
                        result[symbol] = {
                            "address": platforms[chain],
                            "chain": chain.replace("binance-smart-chain", "bsc")
                                        .replace("polygon-pos", "polygon")
                                        .replace("arbitrum-one", "arbitrum")
                                        .replace("optimistic-ethereum", "optimism")
                        }
                        contract_found = True
                        break
                
                # If no preferred chains, take first available
                if not contract_found:
                    for chain, address in platforms.items():
                        if address:
                            result[symbol] = {
                                "address": address,
                                "chain": chain.replace("binance-smart-chain", "bsc")
                                            .replace("polygon-pos", "polygon")
                                            .replace("arbitrum-one", "arbitrum")
                                            .replace("optimistic-ethereum", "optimism")
                            }
                            break
                
                if symbol not in result:
                    result[symbol] = None
                    
            else:
                print(f"CoinGecko API error {response.status_code} for {symbol}")
                result[symbol] = None
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            result[symbol] = None
    
    # Save to cache
    save_cache(result)
    return result