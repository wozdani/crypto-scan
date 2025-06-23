import os
import json
import requests
import time
from datetime import datetime, timedelta
from utils.data_fetchers import get_symbols_cached
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
    file_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
    return datetime.now() - file_time < timedelta(hours=CACHE_EXPIRATION_HOURS)

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
    from utils.data_fetchers import get_symbols_cached

    print("🛠 Buduję cache CoinGecko na podstawie tokenów z Bybit...")
    try:
        token_list = requests.get("https://api.coingecko.com/api/v3/coins/list", timeout=15).json()
        symbol_to_id = {t['symbol'].upper(): t['id'] for t in token_list if 'symbol' in t and 'id' in t}

        # 🔍 DEBUG: test czy TRX, WLD, XRP itd. są w mapie
        test_tokens = ["TRX", "XRP", "WLD", "ZETA"]
        for t in test_tokens:
            print(f"🔍 Test ID CoinGecko dla {t}: {'✅ jest' if t in symbol_to_id else '❌ brak'}")


        bybit_symbols = get_symbols_cached()
        bybit_base = [s.replace("USDT", "").upper() for s in bybit_symbols]

        print(f"🔢 Liczba symboli z Bybit: {len(bybit_base)}")
        print(f"🔚 Ostatnie 10 symboli: {bybit_base[-10:]}")


        cache_data = load_coingecko_cache()

        for i, symbol in enumerate(bybit_base):
            print(f"➡️ [{i}/{len(bybit_base)}] Analiza {symbol}")
            if symbol in cache_data and cache_data[symbol].get("platforms"):
                continue

            token_id = symbol_to_id.get(symbol)
            if not token_id:
                print(f"⛔ Nie znaleziono ID CoinGecko dla {symbol}")
                continue

            detail_url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
            retries = 0
            while retries < 3:
                try:
                    r = requests.get(detail_url, timeout=10)
                    if r.status_code == 429:
                        print(f"🕒 Rate limit (429) dla {symbol}, retry {retries+1}/3 – czekam 10s...")
                        time.sleep(10)
                        retries += 1
                        continue
                    elif r.status_code != 200:
                        print(f"❌ HTTP {r.status_code} dla {symbol}, pomijam.")
                        break

                    details = r.json()
                    platforms = details.get("platforms", {})
                    cache_data[symbol] = {
                        "id": token_id,
                        "name": details.get("name", "").lower(),
                        "platforms": platforms
                    }
                    print(f"✅ Dodano kontrakt dla {symbol}: {platforms}")
                    break  # sukces – wychodzimy z retry loop

                except Exception as e:
                    print(f"⚠️ Błąd przy {symbol}, retry {retries+1}/3: {e}")
                    retries += 1
                    time.sleep(3)

            if i % 10 == 0:
                print(f"📦 {i}/{len(bybit_base)} tokenów przetworzonych")
                save_coingecko_cache(cache_data)

            time.sleep(1.2)
            
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

