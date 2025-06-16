import os
import json
import requests
import time
from datetime import datetime, timedelta
from utils.data_fetchers import get_symbols_cached, get_basic_bybit_symbols_for_cache
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
    import time
    import requests

    print("🛠 Buduję pełny cache CoinGecko dla wszystkich tokenów...")
    try:
        # Jedno zapytanie o wszystkie tokeny z platforms (contract addresses)
        token_list = requests.get("https://api.coingecko.com/api/v3/coins/list?include_platform=true", timeout=15).json()
        
        print(f"🌍 Przetwarzam {len(token_list)} tokenów z CoinGecko...")
        
        cache_data = {}
        excluded_prefixes = ['wrapped', 'bridged', 'binance-peg', 'polygon-peg', 'avalanche-peg', 'fantom-peg']
        
        # Przetwarzaj wszystkie tokeny - dla każdego symbolu bierz pierwszy nie-wrapped
        for i, token in enumerate(token_list):
            symbol = token.get('symbol', '').upper()
            if not symbol:
                continue
                
            # Dla każdego symbolu zapisz tylko jeśli jeszcze nie ma lub jest lepszy
            if symbol not in cache_data:
                # Sprawdź czy to nie wrapped token
                if not any(prefix in token['id'] for prefix in excluded_prefixes):
                    platforms = token.get("platforms", {})
                    cache_data[symbol] = {
                        "id": token['id'],
                        "name": token.get("name", "").lower(),
                        "platforms": platforms
                    }
                    
            if i % 2000 == 0:
                print(f"📦 {i}/{len(token_list)} tokenów przetworzonych")
                
        print(f"🎯 Cache zbudowany: {len(cache_data)} unikalnych symboli")
        
        save_coingecko_cache(cache_data)
        print("✅ Zakończono budowę cache CoinGecko")

    except requests.exceptions.RequestException as e:
        print(f"❌ Błąd podczas budowy cache: {e}")
            
            if i == len(bybit_base) - 1:
                print("✅ Pętla zakończona prawidłowo – wszystkie symbole przetworzone")

        save_coingecko_cache(cache_data)
        print("✅ Zakończono budowę cache CoinGecko")

    except requests.exceptions.RequestException as e:
        print(f"❌ Błąd ogólny podczas budowy cache: {e}")

def get_contract_from_coingecko(symbol):
    cache = load_coingecko_cache()
    normalized_symbol = normalize_token_name(symbol, cache)

    for entry in cache:
        if not isinstance(entry, dict):
            continue
        entry_symbol = entry.get("symbol", "").lower()
   
        if entry_symbol == normalized_symbol.lower():
            return {
                "address": entry.get("platforms", {}).get("ethereum", ""),
                "chain": "ethereum"
            }

    print(f"⚠️ Nie znaleziono kontraktu w cache dla {symbol}")
    return {"address": "", "chain": ""}

