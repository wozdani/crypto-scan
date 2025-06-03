import requests
import json
import os
from datetime import datetime, timedelta

CACHE_FILE = "coingecko_cache.json"
CACHE_DURATION_MINUTES = 30

def is_cache_valid():
    if not os.path.exists(CACHE_FILE):
        return False
    modified_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
    return datetime.now() - modified_time < timedelta(minutes=CACHE_DURATION_MINUTES)

def load_cache():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Błąd podczas ładowania cache: {e}")
        return {}

def save_cache(data):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"⚠️ Błąd podczas zapisu cache: {e}")

def build_coingecko_cache():
    print("📡 Pobieram listę tokenów z CoinGecko (coins/list)...")
    url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
    try:
        headers = {}
        api_key = os.getenv("COINGECKO_API_KEY")
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
            
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        coins = response.json()
        id_map = {}
        for coin in coins:
            symbol = coin["symbol"].upper()
            id_map[symbol] = {
                "id": coin["id"],
                "platforms": coin.get("platforms", {})
            }
        save_cache(id_map)
        print(f"✅ Cache zapisany ({len(id_map)} tokenów).")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Błąd pobierania z CoinGecko: {e}")

def get_contract(symbol, chain="ethereum"):
    symbol = symbol.upper()
    chain = chain.lower()

    if not is_cache_valid():
        build_coingecko_cache()

    id_map = load_cache()
    coin = id_map.get(symbol)
    if not coin:
        print(f"⚠️ Brak danych dla {symbol}")
        return None

    address = coin["platforms"].get(chain)
    if not address:
        print(f"⚠️ Brak kontraktu dla {symbol} na {chain}")
        return None

    return {"address": address, "chain": chain}

# Legacy function names for compatibility - all redirect to cache
def get_multiple_token_contracts_from_coingecko(symbols):
    """Legacy function - now uses cache only"""
    result = {}
    for symbol in symbols:
        result[symbol] = get_contract(symbol)
    return result

def build_contract_cache():
    """Legacy function - redirects to build_coingecko_cache"""
    build_coingecko_cache()

def get_perpetual_symbols():
    """Mock function to avoid Bybit dependency during cache building"""
    return []