import json
from utils.normalize import normalize_token_name

# Mapowanie nazw chainów z CoinGecko na nasze aliasy
CHAIN_ALIASES = {
    "ethereum": "ethereum",
    "binance-smart-chain": "bsc",
    "polygon-pos": "polygon",
    "arbitrum-one": "arbitrum",
    "optimistic-ethereum": "optimism",
    "tron": "tron"
}

def load_coingecko_cache():
    try:
        with open("cache/coingecko_cache.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Błąd przy ładowaniu cache CoinGecko: {e}")
        return {}

def get_contract(symbol):
    """
    CRITICAL FIX: Add timeout protection to prevent hanging on cache operations
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("get_contract cache operation timeout")
    
    try:
        # EMERGENCY TIMEOUT: 2-second timeout for cache operations
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)
        
        cache = load_coingecko_cache()
        normalized = normalize_token_name(symbol, cache)
        token_data = cache.get(normalized)
        
        signal.alarm(0)  # Cancel timeout
    except TimeoutError:
        signal.alarm(0)
        print(f"[CONTRACT TIMEOUT] {symbol} - cache operation timed out, returning None")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"[CONTRACT ERROR] {symbol} - cache error: {e}")
        return None

    if token_data:
        platforms = token_data.get("platforms", {})
        signal.alarm(0)  # Cancel timeout for platform processing
        for cg_chain, address in platforms.items():
            chain = CHAIN_ALIASES.get(cg_chain)
            if chain and address:
                print(f"✅ Dodano kontrakt dla {symbol}: {address} na {chain}")
                return {"address": address, "chain": chain}

    if token_data is None:
        print(f"⛔ Token {symbol} nie istnieje w cache CoinGecko")
        return None

    print(f"⚠️ Brak kontraktu dla {symbol}")
    return None
