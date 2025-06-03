import requests
import json
import os

CACHE_FILE = "token_contract_map.json"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"

def get_perpetual_symbols():
    """Get Bybit perpetual symbols"""
    try:
        from utils.data_fetchers import get_symbols_cached
        return get_symbols_cached()
    except Exception as e:
        print(f"Error getting Bybit symbols: {e}")
        return []

def build_contract_cache():
    """Build contract cache from CoinGecko with single API call"""
    print("📥 Pobieram dane z CoinGecko...")
    try:
        headers = {}
        api_key = os.getenv("COINGECKO_API_KEY")
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
            
        response = requests.get(COINGECKO_URL, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            bybit_symbols = set(get_perpetual_symbols())
            result = {}

            for item in data:
                symbol = item.get("symbol", "").upper()
                if f"{symbol}USDT" in bybit_symbols:
                    platforms = item.get("platforms", {})
                    
                    # Prefer BSC, then Ethereum, then others
                    preferred_chains = ["binance-smart-chain", "ethereum", "polygon-pos", "arbitrum-one", "optimistic-ethereum"]
                    
                    for chain in preferred_chains:
                        address = platforms.get(chain)
                        if address:
                            result[f"{symbol}USDT"] = {
                                "address": address,
                                "chain": chain.replace("binance-smart-chain", "bsc")
                                            .replace("polygon-pos", "polygon")
                                            .replace("arbitrum-one", "arbitrum")
                                            .replace("optimistic-ethereum", "optimism")
                            }
                            break
                    
                    # If no preferred chain found, use first available
                    if f"{symbol}USDT" not in result:
                        for chain, address in platforms.items():
                            if address:
                                result[f"{symbol}USDT"] = {
                                    "address": address,
                                    "chain": chain.replace("binance-smart-chain", "bsc")
                                                .replace("polygon-pos", "polygon")
                                                .replace("arbitrum-one", "arbitrum")
                                                .replace("optimistic-ethereum", "optimism")
                                }
                                break

            with open(CACHE_FILE, "w") as f:
                json.dump(result, f, indent=2)
            print(f"✅ Zapisano {len(result)} kontraktów w cache.")
        else:
            print(f"❌ Błąd CoinGecko: {response.status_code}")
    except Exception as e:
        print(f"❌ Wyjątek przy pobieraniu: {e}")

def get_contract(symbol):
    """Get contract for symbol from cache"""
    if not os.path.exists(CACHE_FILE):
        build_contract_cache()
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        return cache.get(symbol.upper(), None)
    except Exception as e:
        print(f"❌ Błąd przy odczycie cache: {e}")
        return None

def get_multiple_token_contracts_from_coingecko(symbols):
    """Get contracts for multiple symbols using cache"""
    if not os.path.exists(CACHE_FILE):
        build_contract_cache()
    
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        
        result = {}
        for symbol in symbols:
            result[symbol] = cache.get(symbol.upper())
        
        return result
    except Exception as e:
        print(f"❌ Błąd przy odczycie cache: {e}")
        return {symbol: None for symbol in symbols}