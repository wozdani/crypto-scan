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
    from utils.data_fetchers import get_symbols_cached

    print("🛠 Buduję cache CoinGecko na podstawie tokenów z Bybit...")
    try:
        # Jedno zapytanie o wszystkie tokeny z platforms (contract addresses)
        token_list = requests.get("https://api.coingecko.com/api/v3/coins/list?include_platform=true", timeout=15).json()
        
        # Grupuj tokeny wg symbolu dla dokładnej weryfikacji
        symbol_groups = {}
        for token in token_list:
            if 'symbol' in token and 'id' in token:
                symbol = token['symbol'].upper()
                if symbol not in symbol_groups:
                    symbol_groups[symbol] = []
                symbol_groups[symbol].append(token)

        # 🔍 DEBUG: test czy TRX, WLD, XRP itd. są w mapie
        test_tokens = ["TRX", "XRP", "WLD", "ZETA", "MAGIC"]
        for t in test_tokens:
            if t in symbol_groups:
                candidates = [token['id'] for token in symbol_groups[t]]
                print(f"🔍 {t} candidates: {candidates[:3]}...")  # Pierwszych 3
            else:
                print(f"🔍 {t}: ❌ brak")


        # Get basic Bybit symbols without chain requirements for cache building
        bybit_symbols = get_basic_bybit_symbols_for_cache()
        bybit_base = [s.replace("USDT", "").upper() for s in bybit_symbols]

        print(f"🔢 Liczba symboli z Bybit: {len(bybit_base)}")
        print(f"🔚 Ostatnie 10 symboli: {bybit_base[-10:]}")


        cache_data = load_coingecko_cache()

        for i, symbol in enumerate(bybit_base):
            print(f"➡️ [{i}/{len(bybit_base)}] Analiza {symbol}")
            if symbol in cache_data and cache_data[symbol].get("platforms"):
                continue

            # Inteligentny wybór tokena z weryfikacją symbolu - FIXED
            token_id = None
            best_candidate = None
            if symbol in symbol_groups:
                candidates = symbol_groups[symbol]
                print(f"🔍 {symbol}: {len(candidates)} kandydatów")
                
                # Znajdź najlepszy kandydat
                best_candidate = None
                
                # Lista priorytetowych nazw tokenów (główne tokeny)
                priority_names = [symbol.lower(), f"{symbol.lower()}-token", f"{symbol.lower()}-coin"]
                
                # Lista wykluczonych prefiksów (wrapped/bridged tokens)
                excluded_prefixes = ['wrapped', 'bridged', 'binance-peg', 'polygon-peg', 'avalanche-peg', 'fantom-peg']
                
                # 1. Preferuj tokeny z priorytetowymi nazwami (nie wrapped)
                for candidate in candidates:
                    if candidate['symbol'].upper() == symbol.upper():
                        candidate_id = candidate['id'].lower()
                        # Sprawdź czy to nie wrapped token
                        is_wrapped = any(prefix in candidate_id for prefix in excluded_prefixes)
                        
                        if any(priority in candidate_id for priority in priority_names) and not is_wrapped:
                            best_candidate = candidate
                            print(f"✅ Priorytetowy token dla {symbol}: {candidate['id']}")
                            break
                
                # 2. Jeśli nie znaleziono priorytetowego, weź najlepszy pasujący (nie wrapped)
                if not best_candidate:
                    for candidate in candidates:
                        if candidate['symbol'].upper() == symbol.upper():
                            candidate_id = candidate['id'].lower()
                            is_wrapped = any(prefix in candidate_id for prefix in excluded_prefixes)
                            
                            if not is_wrapped:
                                best_candidate = candidate
                                print(f"✅ Standardowy token dla {symbol}: {candidate['id']}")
                                break
                
                # 3. Ostateczny fallback - weź dowolny pasujący symbol
                if not best_candidate:
                    for candidate in candidates:
                        if candidate['symbol'].upper() == symbol.upper():
                            best_candidate = candidate
                            print(f"✅ Fallback token dla {symbol}: {candidate['id']}")
                            break
                
                if best_candidate:
                    token_id = best_candidate['id']
            
            if not token_id or not best_candidate:
                print(f"⛔ Nie znaleziono prawidłowego ID CoinGecko dla {symbol}")
                continue

            # Użyj danych z platforms jeśli są dostępne w głównym zapytaniu
            platforms = best_candidate.get("platforms", {})
            cache_data[symbol] = {
                "id": token_id,
                "name": best_candidate.get("name", "").lower(),
                "platforms": platforms
            }
            print(f"✅ Dodano token dla {symbol}: {best_candidate['id']}")

            if i % 50 == 0:
                print(f"📦 {i}/{len(bybit_base)} tokenów przetworzonych")
                save_coingecko_cache(cache_data)
            
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

