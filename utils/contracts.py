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
    cache = load_coingecko_cache()
    normalized = normalize_token_name(symbol, cache)
    token_data = cache.get(normalized)

    if token_data:
        platforms = token_data.get("platforms", {})
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
