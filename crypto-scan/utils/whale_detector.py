from utils.contracts import get_contract
from utils.token_price import get_token_price_usd
import os
import requests
import time

WHALE_MIN_USD = 50000  # Próg detekcji whale

def detect_whale_tx(symbol, price_usd=None):
    """
    Enhanced Whale Transaction Detector
    Wykrywa minimum 3 transakcje > 50,000 USD w ciągu ostatnich 15 minut
    
    Returns:
        tuple: (whale_active, large_tx_count, total_usd)
    """

    
    token_info = get_contract(symbol)
    if not token_info:
        print(f"⚠️ Brak kontraktu dla {symbol}")
        return False, 0, 0.0

    # Sprawdź czy token_info jest dictionary
    if not isinstance(token_info, dict):
        print(f"❌ [whale_detector] token_info nie jest dict dla {symbol}: {type(token_info)} → {token_info}")
        return False, 0, 0.0

    if not price_usd or price_usd == 0:
        print(f"⚠️ Brak ceny USD dla {symbol} (price_usd={price_usd}) – pomijam whale tx.")
        return False, 0, 0.0

    chain_raw = token_info.get("chain", "")
    chain = chain_raw.lower() if isinstance(chain_raw, str) else ""
    address = token_info.get("address")

    if not chain or not address:
        print(f"⚠️ Brak danych chain/address dla {symbol}")
        return False, 0, 0.0

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
    if not config or not config["api_key"]:
        print(f"⚠️ Chain {chain} nieobsługiwany lub brak API key")
        return False, 0, 0.0

    # Oblicz timestamp 15 minut temu
    from datetime import datetime, timedelta, timezone
    import time
    
    now_utc = datetime.now(timezone.utc)
    fifteen_minutes_ago = now_utc - timedelta(minutes=15)
    timestamp_threshold = int(fifteen_minutes_ago.timestamp())
    
    print(f"[WHALE TX] Szukam transakcji po {fifteen_minutes_ago.strftime('%H:%M:%S UTC')} dla {symbol}")

    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "desc",
        "apikey": config["api_key"]
    }

    # Enhanced retry logic with timeout handling
    for attempt in range(2):
        try:
            response = requests.get(config["url"], params=params, timeout=20)
            
            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code} dla {symbol} na {chain}")
                return False, 0, 0.0
                
            data = response.json()

            if data.get("status") != "1":
                print(f"⚠️ API Error dla {symbol}: {data.get('message', 'unknown')}")
                return False, 0, 0.0

            txs = data.get("result", [])
            
            # Analyze transactions from last 15 minutes
            large_transactions = []
            large_tx_count = 0
            total_usd = 0.0
            
            for tx in txs[:50]:  # Check more transactions for time filtering
                try:
                    # Parse transaction timestamp
                    tx_timestamp = int(tx.get("timeStamp", 0))
                    
                    # Skip transactions older than 15 minutes
                    if tx_timestamp < timestamp_threshold:
                        continue
                    
                    # Calculate USD value
                    raw_value = int(tx["value"])
                    decimals = int(tx["tokenDecimal"])
                    token_amount = raw_value / (10 ** decimals)
                    usd_value = token_amount * price_usd

                    # Check if transaction is large enough (>50k USD)
                    if usd_value >= 50000:
                        large_transactions.append({
                            "hash": tx.get("hash", ""),
                            "timestamp": tx_timestamp,
                            "usd_value": usd_value,
                            "token_amount": token_amount
                        })
                        large_tx_count += 1
                        total_usd += usd_value
                        
                        tx_time = datetime.fromtimestamp(tx_timestamp, timezone.utc).strftime('%H:%M:%S')
                        print(f"[WHALE TX] {symbol}: ${usd_value:,.0f} at {tx_time} UTC")

                except (ValueError, KeyError, TypeError) as e:
                    continue
            
            # Enhanced whale detection logic
            whale_active = False
            
            if large_tx_count >= 3 and total_usd > 150_000:
                whale_active = True
                print(f"[WHALE ACTIVITY] {symbol}: {large_tx_count} large TXs, total ${total_usd:,.0f}")
            elif large_tx_count >= 3:
                whale_active = True
                print(f"[WHALE ACTIVITY] {symbol}: {large_tx_count} large TXs (count threshold)")
            elif total_usd > 300_000:  # Single very large transaction
                whale_active = True
                print(f"[WHALE ACTIVITY] {symbol}: Single large volume ${total_usd:,.0f}")
            
            if large_tx_count > 0:
                print(f"[WHALE SUMMARY] {symbol}: {large_tx_count} TXs, ${total_usd:,.0f} total, active: {whale_active}")
            
            return whale_active, large_tx_count, total_usd
            
        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout dla {symbol} na {chain} (próba {attempt + 1}/2)")
            if attempt == 0:
                time.sleep(2)
                continue
            else:
                print(f"❌ Timeout definitywny dla {symbol}")
                return False, 0, 0.0
                
        except Exception as e:
            print(f"❌ Błąd w detekcji whale TX dla {symbol}: {e}")
            return False, 0, 0.0

    return False, 0, 0.0
