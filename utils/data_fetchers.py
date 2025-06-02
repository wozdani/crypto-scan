# -*- coding: utf-8 -*-
import requests
import time
import os
import hmac
import hashlib
from dotenv import load_dotenv

load_dotenv()

BYBIT_BASE_URL = "https://api.bybit.com"
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")

def get_bybit_headers(params=None):
    """Generate authenticated headers for Bybit API"""
    if not BYBIT_API_KEY or not BYBIT_SECRET_KEY:
        return {}
    
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    
    if params:
        param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        raw_str = timestamp + BYBIT_API_KEY + recv_window + param_str
    else:
        raw_str = timestamp + BYBIT_API_KEY + recv_window
    
    signature = hmac.new(
        BYBIT_SECRET_KEY.encode('utf-8'),
        raw_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": signature,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }

def fetch_klines(symbol, interval="15", limit=2):
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    try:
        headers = get_bybit_headers(params)
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data["retCode"] == 0:
            return data["result"]["list"]
        else:
            print(f"❌ Blad danych Bybit dla {symbol}: {data}")
            return None
    except Exception as e:
        print(f"❌ Wyjątek dla {symbol}: {e}")
        return None

def get_last_candles(symbol):
    candles = fetch_klines(symbol, interval="15", limit=2)
    if not candles:
        return None, None
    return candles[-2], candles[-1]  # poprzednia i obecna swieca

def get_all_data(symbol):
    prev_candle, last_candle = get_last_candles(symbol)
    if not last_candle:
        return None

    try:
        # Kandle: [timestamp, open, high, low, close, volume, turnover]
        open_price = float(last_candle[1])
        close_price = float(last_candle[4])
        high = float(last_candle[2])
        low = float(last_candle[3])
        volume = float(last_candle[5])
        price_change = close_price - open_price
        candle_body = abs(close_price - open_price)
        candle_range = high - low
        if candle_range == 0:  # zabezpieczenie przed zerem
            candle_range = 0.0001
        body_ratio = candle_body / candle_range

        return {
            "open": open_price,
            "close": close_price,
            "high": high,
            "low": low,
            "volume": volume,
            "price_change": price_change,
            "body_ratio": round(body_ratio, 4),
            "prev_candle": prev_candle,
            "last_candle": last_candle,
        }
    except Exception as e:
        print(f"❌ Blad przy parsowaniu Swiecy {symbol}: {e}")
        return None

# === COMPATIBILITY FUNCTIONS FOR EXISTING CODE ===

import requests

def get_symbols_cached():
    """Pobiera tylko prawdziwe kontrakty perpetual z Bybit (linear USDT-PERP) z obsługiwanych chainów"""
    import os
    from utils.contracts import get_or_fetch_token_contract
    
    # Definiuj obsługiwane chainy z kluczami API
    SUPPORTED_CHAINS = {
        "ethereum": os.getenv("ETHERSCAN_API_KEY"),
        "bsc": os.getenv("BSCSCAN_API_KEY"),
        "arbitrum": os.getenv("ARBISCAN_API_KEY"),
        "polygon": os.getenv("POLYGONSCAN_API_KEY"),
        "optimism": os.getenv("OPTIMISMSCAN_API_KEY"),
        "tron": os.getenv("TRONGRID_API_KEY")
    }
    
    # Tylko chainy z dostępnymi kluczami
    VALID_CHAINS = {chain for chain, key in SUPPORTED_CHAINS.items() if key}
    
    try:
        url = "https://api.bybit.com/v5/market/instruments-info"
        params = {
            "category": "linear"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data["retCode"] == 0:
            valid_symbols = []
            
            for item in data["result"]["list"]:
                symbol = item["symbol"]
                contract_type = item.get("contractType", "")
                status = item.get("status", "")
                
                # Filtruj tylko LinearPerpetual kontrakty w statusie Trading
                if (status == "Trading" and 
                    contract_type == "LinearPerpetual" and 
                    symbol.endswith("USDT")):
                    
                    # Filtruj dziwne nazwy tokenów
                    if not is_valid_perpetual_symbol(symbol):
                        continue
                    
                    # Sprawdź czy token ma obsługiwany chain
                    token_info = get_or_fetch_token_contract(symbol)
                    if token_info:
                        chain = token_info["chain"].lower()
                        if chain not in VALID_CHAINS:
                            continue  # Pomiń tokeny z nieobsługiwanych chainów
                        
                    valid_symbols.append(symbol)
            
            print(f"✅ Pobrano {len(valid_symbols)} kontraktów USDT-PERP z obsługiwanych chainów")
            if VALID_CHAINS:
                print(f"📡 Obsługiwane chainy: {', '.join(VALID_CHAINS)}")
            else:
                print("⚠️ Brak kluczy API dla chainów - wszystkie tokeny będą pomijane")
            return valid_symbols
        else:
            print("❌ Błąd pobierania symboli Bybit:", data)
            return []
            
    except Exception as e:
        print(f"❌ Wyjątek przy pobieraniu symboli: {e}")
        return []

def is_valid_perpetual_symbol(symbol):
    """Sprawdza czy symbol to prawdziwy kontrakt perpetual"""
    if not symbol.endswith("USDT"):
        return False
    
    # Usuń USDT z końca dla analizy
    base_symbol = symbol[:-4]
    
    # Pozwól na niektóre popularne tokeny z cyframi jako wyjątki
    allowed_with_numbers = ["1INCH", "1000SATS"]
    if base_symbol in allowed_with_numbers:
        return True
    
    # Odrzuć tokeny z długimi prefixami numerycznymi
    if base_symbol.startswith(("1000000", "100000", "10000", "1000", "100")):
        return False
    
    # Odrzuć tokeny z dziwnym formatem (za krótkie lub za długie)
    if len(base_symbol) < 3 or len(base_symbol) > 10:
        return False
    
    # Odrzuć tokeny ze znakami numerycznymi (poza wyjątkami)
    if any(char.isdigit() for char in base_symbol):
        return False
    
    # Sprawdź czy składa się tylko z liter
    if not base_symbol.isalpha():
        return False
    
    return True

def fetch_top_symbols():
    return get_symbols_cached()

def get_fallback_symbols():
    return get_symbols_cached()

def get_market_data(symbol):
    """Get comprehensive market data for a symbol"""
    return get_all_data(symbol)

def get_price_data(symbol):
    """Get price and volume data"""
    data = get_all_data(symbol)
    if data:
        return {
            "current": data["close"],
            "change_15m": data["price_change"]
        }
    return None

def get_historical_data(symbol, days=30):
    """Get historical price data"""
    try:
        # Get more candles for historical data
        candles = fetch_klines(symbol, interval="15", limit=min(days * 96, 1000))  # 96 candles per day
        if candles:
            prices = [float(candle[4]) for candle in candles]  # close prices
            volumes = [float(candle[5]) for candle in candles]  # volumes
            timestamps = [int(candle[0]) for candle in candles]  # timestamps
            return {
                "prices": prices,
                "volumes": volumes,
                "timestamps": timestamps
            }
    except Exception as e:
        print(f"❌ Error getting historical data for {symbol}: {e}")
    
    return {"prices": [], "volumes": [], "timestamps": []}

def get_blockchain_data(symbol):
    """Get blockchain-specific data for supported networks"""
    # This would require contract addresses and blockchain API calls
    # For now, return empty structure
    return {
        "network": "unknown",
        "contract_address": None,
        "holders": 0,
        "transactions_24h": 0
    }

def get_ethereum_data(symbol):
    """Get Ethereum blockchain data"""
    return get_blockchain_data(symbol)

def get_bsc_data(symbol):
    """Get Binance Smart Chain data"""
    return get_blockchain_data(symbol)

def get_polygon_data(symbol):
    """Get Polygon network data"""
    return get_blockchain_data(symbol)