import os
import requests
from utils.token_price import get_token_price_usd
import json
from datetime import datetime, timedelta

WHALE_MIN_USD = 50000  # minimalna wartość USD dla uznania jako whale transfer

def get_native_token_prices():
    """Get prices of native tokens from cache - no direct API calls"""
    try:
        from utils.coingecko import load_cache, is_cache_valid, build_coingecko_cache
        
        if not is_cache_valid():
            build_coingecko_cache()
        
        cache = load_cache()
        
        # Return placeholder values to avoid API calls - this would need enhancement
        # to include price data in the cache system
        return {
            "ethereum": 2500,  # Approximate values
            "bsc": 300,
            "polygon": 0.8,
            "arbitrum": 2500,
            "optimism": 2500
        }
    except Exception as e:
        print(f"❌ Błąd pobierania cen native tokenów: {e}")
        return {
            "ethereum": 0,
            "bsc": 0,
            "polygon": 0,
            "arbitrum": 0,
            "optimism": 0
        }

def detect_whale_transfers(symbol, token_map):
    if symbol not in token_map:
        print(f"⚠️ Brak mapowania dla {symbol}")
        return False, 0.0

    token_data = token_map[symbol]
    chain = token_data.get("chain", "").lower()
    address = token_data.get("address")

    if not chain or not address:
        print(f"⚠️ Brak danych chain/address dla {symbol}")
        return False, 0.0

    # Wybierz API explorer w zależności od sieci
    explorer_configs = {
        "ethereum": {
            "url": "https://api.etherscan.io/api",
            "api_key": os.getenv("ETHERSCAN_API_KEY")
        },
        "bsc": {
            "url": "https://api.bscscan.com/api",
            "api_key": os.getenv("BSCSCAN_API_KEY")
        },
        "polygon": {
            "url": "https://api.polygonscan.com/api",
            "api_key": os.getenv("POLYGONSCAN_API_KEY")
        },
        "arbitrum": {
            "url": "https://api.arbiscan.io/api",
            "api_key": os.getenv("ARBISCAN_API_KEY")
        },
        "optimism": {
            "url": "https://api-optimistic.etherscan.io/api",
            "api_key": os.getenv("OPTIMISMSCAN_API_KEY")
        }
    }

    config = explorer_configs.get(chain)
    if not config:
        print(f"⚠️ Chain {chain} nieobsługiwany")
        return False, 0.0

    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "desc",
        "apikey": config["api_key"]
    }

    try:
        response = requests.get(config["url"], params=params, timeout=10)
        data = response.json()

        if data.get("status") != "1":
            return False, 0.0

        txs = data.get("result", [])
        token_price = get_token_price_usd(symbol)
        
        if not token_price:
            return False, 0.0

        for tx in txs[:10]:  # Check last 10 transactions
            try:
                raw_value = int(tx["value"]) / (10 ** int(tx["tokenDecimal"]))
                usd_value = raw_value * token_price
                
                if usd_value >= WHALE_MIN_USD:
                    print(f"🐋 Whale TX wykryty dla {symbol}: {usd_value:.2f} USD")
                    return True, usd_value
            except (ValueError, KeyError) as e:
                continue

        return False, 0.0

    except Exception as e:
        print(f"❌ Błąd przy wykrywaniu whale TX dla {symbol}: {e}")
        return False, 0.0