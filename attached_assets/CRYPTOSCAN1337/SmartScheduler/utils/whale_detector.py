import os
import requests
from utils.token_price import get_token_price_usd
import json
from datetime import datetime, timedelta

WHALE_MIN_USD = 50000  # minimalna wartość USD dla uznania jako whale transfer

def get_native_token_prices():
    """Get prices of native tokens for different chains"""
    try:
        api_key = os.getenv("COINGECKO_API_KEY")
        
        if api_key:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids=ethereum,binancecoin,matic-network,arbitrum,optimism&vs_currencies=usd&x_cg_demo_api_key={api_key}"
        else:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids=ethereum,binancecoin,matic-network,arbitrum,optimism&vs_currencies=usd"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            "ethereum": data.get("ethereum", {}).get("usd", 0),
            "bsc": data.get("binancecoin", {}).get("usd", 0),
            "polygon": data.get("matic-network", {}).get("usd", 0),
            "arbitrum": data.get("arbitrum", {}).get("usd", 0),
            "optimism": data.get("optimism", {}).get("usd", 0)
        }
    except Exception as e:
        print(f"❌ Błąd pobierania cen native tokenów: {e}")
        return {}

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