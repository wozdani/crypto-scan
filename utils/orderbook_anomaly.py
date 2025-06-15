import requests
import time
import os
import hmac
import hashlib

ORDERBOOK_DEPTH = 3
BID_MULTIPLIER = 2.0      # jeśli suma bidów wzrosła 2x
ASK_DROP_THRESHOLD = 0.5  # jeśli suma asków spadła o 50%

_orderbook_cache = {}

def get_bybit_headers(params=None):
    """Generate authenticated headers for Bybit API"""
    api_key = os.getenv("BYBIT_API_KEY")
    secret_key = os.getenv("BYBIT_SECRET_KEY")
    
    if not api_key or not secret_key:
        return {}
    
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    
    if params:
        param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        raw_str = timestamp + api_key + recv_window + param_str
    else:
        raw_str = timestamp + api_key + recv_window
    
    signature = hmac.new(
        secret_key.encode('utf-8'),
        raw_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-SIGN": signature,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }

def fetch_orderbook_snapshot(symbol):
    try:
        # Try v5 API first
        url = f"https://api.bybit.com/v5/market/orderbook"
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": ORDERBOOK_DEPTH
        }
        
        headers = get_bybit_headers(params)
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict): return False, 0.0
            if data.get("retCode") == 0:
                result = data.get("result", {})
                if not isinstance(result, dict): return False, 0.0
                bids = [float(entry[1]) for entry in result.get("b", [])][:ORDERBOOK_DEPTH]
                asks = [float(entry[1]) for entry in result.get("a", [])][:ORDERBOOK_DEPTH]
                return sum(bids), sum(asks)
        
        # Fallback to v2 API
        url = f"https://api.bybit.com/v2/public/orderBook/L2"
        params = {"symbol": symbol}
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict): return False, 0.0
            if data.get("ret_code") == 0:
                result = data.get("result", [])
                if not isinstance(result, list): return False, 0.0
                
                bids = [float(entry["size"]) for entry in result if entry["side"] == "Buy"][:ORDERBOOK_DEPTH]
                asks = [float(entry["size"]) for entry in result if entry["side"] == "Sell"][:ORDERBOOK_DEPTH]
                return sum(bids), sum(asks)
        
        return None, None
        
    except Exception as e:
        print(f"❌ Błąd pobierania orderbook dla {symbol}: {e}")
        return None, None

def detect_orderbook_anomaly(symbol):
    current_bid_sum, current_ask_sum = fetch_orderbook_snapshot(symbol)
    if current_bid_sum is None:
        return False, 0.0

    prev = _orderbook_cache.get(symbol)
    _orderbook_cache[symbol] = (current_bid_sum, current_ask_sum)

    if prev is None:
        return False, 0.0  # brak danych do porównania

    prev_bid_sum, prev_ask_sum = prev

    if prev_bid_sum == 0 or prev_ask_sum == 0:
        return False, 0.0

    bid_ratio = current_bid_sum / prev_bid_sum
    ask_ratio = current_ask_sum / prev_ask_sum

    if bid_ratio >= BID_MULTIPLIER or ask_ratio <= ASK_DROP_THRESHOLD:
        print(f"📊 Anomalia orderbook dla {symbol} (bid×{bid_ratio:.2f}, ask×{ask_ratio:.2f})")
        return True, bid_ratio

    return False, 0.0