import os
import json
import requests
import time
from datetime import datetime, timedelta
from utils.normalize import normalize_token_name

CACHE_FILE = "cache/coingecko_cache.json"
CACHE_EXPIRATION_HOURS = 24

os.makedirs("cache", exist_ok=True)

CHAIN_ALIASES = {
    "ethereum": "ethereum",
    "binance-smart-chain": "bsc",
    "arbitrum-one": "arbitrum",
    "polygon-pos": "polygon",
    "optimistic-ethereum": "optimism",
    "tron": "tron"
}

def is_cache_valid():
    if not os.path.exists(CACHE_FILE):
        return False
    
    # Check if file has actual content (not just {})
    try:
        with open(CACHE_FILE, "r") as f:
            cache_content = json.load(f)
        if not cache_content or len(cache_content) == 0:
            print("🚫 Cache file is empty - marking as invalid")
            return False
    except (json.JSONDecodeError, Exception) as e:
        print(f"🚫 Cache file corrupted: {e}")
        return False
    
    file_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
    is_expired = datetime.now() - file_time >= timedelta(hours=CACHE_EXPIRATION_HOURS)
    
    if is_expired:
        print(f"🚫 Cache expired - age: {datetime.now() - file_time}")
        return False
    
    print(f"✅ Cache valid - contains {len(cache_content)} tokens")
    return True

def load_coingecko_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r") as f:
        return json.load(f)

def save_coingecko_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def build_coingecko_cache():
    print("🛠 Buduję pełny cache CoinGecko dla wszystkich tokenów...")
    try:
        # Jedno zapytanie o wszystkie tokeny z platforms (contract addresses)
        token_list = requests.get("https://api.coingecko.com/api/v3/coins/list?include_platform=true", timeout=15).json()
        
        print(f"🌍 Przetwarzam {len(token_list)} tokenów z CoinGecko...")
        
        cache_data = {}
        excluded_prefixes = ['wrapped', 'bridged', 'binance-peg', 'polygon-peg', 'avalanche-peg', 'fantom-peg']
        
        # Grupuj tokeny wg symbolu dla inteligentnego wyboru
        symbol_groups = {}
        for token in token_list:
            symbol = token.get('symbol', '').upper()
            if symbol:
                if symbol not in symbol_groups:
                    symbol_groups[symbol] = []
                symbol_groups[symbol].append(token)

        # Mapowanie specjalne dla głównych kryptowalut
        special_mappings = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'USDT': 'tether',
            'USDC': 'usd-coin',
            'BNB': 'binancecoin',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOGE': 'dogecoin',
            'MATIC': 'matic-network',
            'AVAX': 'avalanche-2',
            'TRX': 'tron',
            'DOT': 'polkadot',
            'UNI': 'uniswap',
            'LINK': 'chainlink',
            'LTC': 'litecoin'
        }
        
        # Przetwarzaj każdy symbol z inteligentnym wyborem najlepszego tokena
        for symbol, candidates in symbol_groups.items():
            best_candidate = None
            symbol_lower = symbol.lower()
            
            # 1. Sprawdź specjalne mapowania dla głównych kryptowalut
            if symbol in special_mappings:
                target_id = special_mappings[symbol]
                for candidate in candidates:
                    if candidate['id'] == target_id:
                        best_candidate = candidate
                        break
            
            # 2. Preferuj tokeny z dokładnym dopasowaniem ID do symbolu
            if not best_candidate:
                for candidate in candidates:
                    candidate_id = candidate['id'].lower()
                    if candidate_id == symbol_lower:
                        best_candidate = candidate
                        break
            
            # 2. Preferuj tokeny gdzie nazwa == symbol (np. MAGIC -> "Magic")
            if not best_candidate:
                for candidate in candidates:
                    name = candidate.get('name', '').lower()
                    candidate_id = candidate['id'].lower()
                    if (name == symbol_lower and 
                        not any(prefix in candidate_id for prefix in excluded_prefixes)):
                        best_candidate = candidate
                        break
            
            # 3. Preferuj główne tokeny z ID zawierającym symbol (np. FLOKI -> floki)
            if not best_candidate:
                for candidate in candidates:
                    candidate_id = candidate['id'].lower()
                    if (symbol_lower in candidate_id and 
                        candidate_id.startswith(symbol_lower) and
                        not any(prefix in candidate_id for prefix in excluded_prefixes)):
                        best_candidate = candidate
                        break
            
            # 4. Preferuj tokeny z nazwą zaczynającą się od symbolu
            if not best_candidate:
                for candidate in candidates:
                    name = candidate.get('name', '').lower()
                    candidate_id = candidate['id'].lower()
                    if (name.startswith(symbol_lower) and 
                        not any(prefix in candidate_id for prefix in excluded_prefixes)):
                        best_candidate = candidate
                        break
            
            # 5. Weź pierwszy nie-wrapped token
            if not best_candidate:
                for candidate in candidates:
                    if not any(prefix in candidate['id'] for prefix in excluded_prefixes):
                        best_candidate = candidate
                        break
            
            # 6. Fallback - weź pierwszy dostępny
            if not best_candidate and candidates:
                best_candidate = candidates[0]
            
            if best_candidate:
                platforms = best_candidate.get("platforms", {})
                cache_data[symbol] = {
                    "id": best_candidate['id'],
                    "name": best_candidate.get("name", "").lower(),
                    "platforms": platforms
                }
                
        print(f"🎯 Cache zbudowany: {len(cache_data)} unikalnych symboli z {len(symbol_groups)} grup")
        
        save_coingecko_cache(cache_data)
        print("✅ Zakończono budowę cache CoinGecko")

    except requests.exceptions.RequestException as e:
        print(f"❌ Błąd podczas budowy cache: {e}")

def get_contract_from_coingecko(symbol):
    print(f"[DEBUG] get_contract_from_coingecko({symbol}) - START")
    
    cache = load_coingecko_cache()
    print(f"[DEBUG] Cache loaded: {len(cache)} symbols")
    
    normalized_symbol = normalize_token_name(symbol, cache)
    print(f"[DEBUG] Normalized symbol: {symbol} → {normalized_symbol}")

    # First check if symbol is directly in cache (new format)
    if symbol in cache:
        entry = cache[symbol]
        platforms = entry.get("platforms", {})
        print(f"[DEBUG] {symbol} found directly in cache, platforms: {list(platforms.keys())}")
        
        # Priorytet dla różnych chainów
        chains = ["ethereum", "polygon-pos", "binance-smart-chain", "arbitrum-one", "optimistic-ethereum"]
        
        for chain in chains:
            if chain in platforms and platforms[chain]:
                contract_address = platforms[chain]
                print(f"[DEBUG] {symbol} contract found on {chain}: {contract_address}")
                return {
                    "address": contract_address,
                    "chain": chain
                }
        
        print(f"[DEBUG] {symbol} no valid contract addresses found in platforms")
    
    # Fallback to old format search
    print(f"[DEBUG] Searching cache using old format for {normalized_symbol}")
    for entry in cache:
        if not isinstance(entry, dict):
            continue
        entry_symbol = entry.get("symbol", "").lower()
   
        if entry_symbol == normalized_symbol.lower():
            print(f"[DEBUG] Found match in old format: {entry_symbol}")
            return {
                "address": entry.get("platforms", {}).get("ethereum", ""),
                "chain": "ethereum"
            }

    print(f"⚠️ Nie znaleziono kontraktu w cache dla {symbol}")
    return {"address": "", "chain": ""}