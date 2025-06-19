"""
Bybit Orderbook Data Fetcher
Pobiera dane orderbook z Bybit API dla trend mode pipeline
"""

import requests
import time
import hmac
import hashlib
import os
from urllib.parse import urlencode

def get_bybit_headers(params_str=""):
    """Generate authenticated headers for Bybit API"""
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_SECRET_KEY")
    
    if not api_key or not api_secret:
        return {}
    
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    
    param_string = f"timestamp={timestamp}&api_key={api_key}&recv_window={recv_window}"
    if params_str:
        param_string += f"&{params_str}"
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        param_string.encode('utf-8'),
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

def get_orderbook_from_bybit(symbol, limit=25):
    """
    Pobiera orderbook z Bybit API dla określonego symbolu
    
    Args:
        symbol: Symbol trading pair (np. 'BTCUSDT')
        limit: Liczba poziomów bid/ask (domyślnie 25)
        
    Returns:
        dict: Orderbook data z kluczami 'bids' i 'asks' lub None w przypadku błędu
    """
    try:
        # Parametry zapytania
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit
        }
        
        params_str = urlencode(params)
        headers = get_bybit_headers(params_str)
        
        # Wywołanie API
        url = f"https://api.bybit.com/v5/market/orderbook?{params_str}"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0 and "result" in data:
                orderbook_data = data["result"]
                
                # Formatuj dane do standardowego formatu
                return {
                    "bids": orderbook_data.get("b", []),  # Bybit używa 'b' dla bids
                    "asks": orderbook_data.get("a", []),  # Bybit używa 'a' dla asks
                    "timestamp": orderbook_data.get("ts", int(time.time() * 1000)),
                    "symbol": symbol
                }
            else:
                print(f"⚠️ Bybit API error for {symbol}: {data.get('retMsg', 'Unknown error')}")
                return None
        else:
            print(f"⚠️ HTTP error {response.status_code} for {symbol} orderbook")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching orderbook for {symbol}: {e}")
        return None

def get_orderbook_with_fallback(symbol):
    """
    Pobiera orderbook z fallback mechanism dla development environment
    
    Args:
        symbol: Trading symbol
        
    Returns:
        dict: Orderbook data lub mock data w development
    """
    # Spróbuj pobrać rzeczywiste dane
    orderbook = get_orderbook_from_bybit(symbol)
    
    if orderbook:
        return orderbook
    
    # Fallback dla development environment
    print(f"🔄 Using mock orderbook for {symbol} (development mode)")
    
    # Mock orderbook data pokazujący pozytywny sentiment
    base_price = 50000.0  # Przykładowa cena
    
    return {
        "bids": [
            [str(base_price * 0.9999), "1.5"],  # Bid blisko ceny
            [str(base_price * 0.9998), "2.1"], 
            [str(base_price * 0.9997), "1.8"],
            [str(base_price * 0.9996), "2.3"],
            [str(base_price * 0.9995), "1.9"],
            [str(base_price * 0.9999), "0.8"],  # Duplicate level (reloading)
            [str(base_price * 0.9994), "2.5"],
            [str(base_price * 0.9993), "1.7"],
            [str(base_price * 0.9992), "2.0"],
            [str(base_price * 0.9991), "1.6"]
        ],
        "asks": [
            [str(base_price * 1.0001), "0.9"],  # Ask levels mniejsze (cofają się)
            [str(base_price * 1.0002), "1.1"],
            [str(base_price * 1.0003), "0.8"],
            [str(base_price * 1.0004), "1.0"],
            [str(base_price * 1.0005), "0.7"],
            [str(base_price * 1.0006), "0.9"],
            [str(base_price * 1.0007), "0.8"],
            [str(base_price * 1.0008), "0.6"],
            [str(base_price * 1.0009), "0.7"],
            [str(base_price * 1.0010), "0.5"]
        ],
        "timestamp": int(time.time() * 1000),
        "symbol": symbol,
        "mock": True
    }

def validate_orderbook_data(orderbook):
    """
    Walidacja danych orderbook
    
    Args:
        orderbook: Dict z danymi orderbook
        
    Returns:
        bool: True jeśli dane są prawidłowe
    """
    if not orderbook or not isinstance(orderbook, dict):
        return False
    
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    # Sprawdź podstawową strukturę
    if not bids or not asks:
        return False
    
    # Sprawdź format danych
    try:
        for bid in bids[:3]:  # Sprawdź pierwsze 3
            float(bid[0])  # price
            float(bid[1])  # volume
            
        for ask in asks[:3]:
            float(ask[0])  # price  
            float(ask[1])  # volume
            
        return True
    except (IndexError, ValueError, TypeError):
        return False