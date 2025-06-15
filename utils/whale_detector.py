from utils.contracts import get_contract
from utils.token_price import get_token_price_usd
import os
import requests

WHALE_MIN_USD = 50000  # Próg detekcji whale

def detect_whale_tx(symbol, price_usd=None):
    token_info = get_contract(symbol)
    if not token_info:
        print(f"⚠️ Brak kontraktu dla {symbol}")
        return False, 0.0

    # Sprawdź czy token_info jest dictionary
    if not isinstance(token_info, dict):
        print(f"❌ [whale_detector] token_info nie jest dict dla {symbol}: {type(token_info)} → {token_info}")
        return False, 0.0

    if not price_usd or price_usd == 0:
        print(f"⚠️ Brak ceny USD dla {symbol} (price_usd={price_usd}) – pomijam whale tx.")
        return False, 0.0

    chain = token_info.get("chain", "").lower()
    address = token_info.get("address")

    if not chain or not address:
        print(f"⚠️ Brak danych chain/address dla {symbol}")
        return False, 0.0

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
        
        for tx in txs[:10]:
            try:
                raw_value = int(tx["value"])
                decimals = int(tx["tokenDecimal"])
                token_amount = raw_value / (10 ** decimals)
                usd_value = token_amount * price_usd

                print(f"🧪 {symbol}: token_amount={token_amount}, price_usd={price_usd}, usd_value={usd_value:.2f}")

                if usd_value >= WHALE_MIN_USD:
                    return True, usd_value
            except:
                continue

    except Exception as e:
        print(f"❌ Błąd w detekcji whale TX dla {symbol}: {e}")
        return False, 0.0

    return False, 0.0
