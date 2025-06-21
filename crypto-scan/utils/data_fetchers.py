# -*- coding: utf-8 -*-
import requests
import time
import os
import hmac
import hashlib
import json
from dotenv import load_dotenv
from utils.contracts import get_contract
from datetime import datetime, timedelta

from pathlib import Path
dotenv_path = Path(__file__).resolve().parents[1] / ".env"
print(f"🧪 Szukam .env w: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

BYBIT_BASE_URL = "https://api.bybit.com"
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")
BYBIT_SYMBOLS_PATH = "utils/data/cache/bybit_symbols.json"
BYBIT_ENDPOINT = "https://api.bybit.com/v5/market/tickers"
VALID_CHAINS = ["ethereum", "bsc", "arbitrum", "polygon", "optimism", "tron"]

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


def get_basic_bybit_symbols_for_cache():
    """Get basic Bybit USDT symbols without chain requirements for cache building"""
    try:
        # Try to load from existing cache first
        if os.path.exists(BYBIT_SYMBOLS_PATH):
            with open(BYBIT_SYMBOLS_PATH, "r") as f:
                symbols = json.load(f)
            if symbols:
                print(f"📦 Loaded {len(symbols)} symbols from Bybit cache")
                return symbols
    except:
        pass
    
    # If no cache or cache is empty, rebuild
    print("🔄 Building fresh Bybit symbols list...")
    build_bybit_symbol_cache_all_categories()
    
    # Load the fresh cache
    try:
        with open(BYBIT_SYMBOLS_PATH, "r") as f:
            symbols = json.load(f)
        print(f"✅ Built and loaded {len(symbols)} Bybit symbols")
        return symbols
    except:
        print("❌ Failed to build Bybit symbols cache")
        return []

def get_symbols_cached(require_chain=True):
    # Sprawdź czy cache istnieje i czy nie jest wygasły
    if is_bybit_cache_expired():
        print("⚠️ Cache Bybit wygasł lub nie istnieje – buduję cache...")
        from utils.data_fetchers import build_bybit_symbol_cache_all_categories
        build_bybit_symbol_cache_all_categories()
    
    # Załaduj symbole z cache'u
    try:
        with open(BYBIT_SYMBOLS_PATH, "r") as f:
            symbols = json.load(f)
    except FileNotFoundError:
        print("❌ Błąd: Nie można załadować cache'u symboli po rebuild")
        symbols = []

    valid_symbols = []
    for symbol in symbols:
        if not symbol.endswith("USDT"):
            continue

        if require_chain:
            token_info = get_contract(symbol)
            if not token_info or "chain" not in token_info:
                continue
            chain = token_info["chain"].lower() if isinstance(token_info.get("chain"), str) else ""
            if chain not in VALID_CHAINS:
                continue

        if is_valid_symbol(symbol):
            valid_symbols.append(symbol)

    # Essential fallback when API/cache fails
    if not valid_symbols:
        print("🔄 No symbols from cache/API - using essential trading pairs...")
        essential_symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "AVAXUSDT",
            "MATICUSDT", "LINKUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "FILUSDT",
            "NEARUSDT", "AAVEUSDT", "CRVUSDT", "SUSHIUSDT", "1INCHUSDT", "CKBUSDT",
            "MANAUSDT", "SANDUSDT", "AXSUSDT", "CHZUSDT", "ENJUSDT", "GALAUSDT"
        ]
        print(f"📋 Using {len(essential_symbols)} essential trading pairs")
        return essential_symbols

    print(f"🔢 Liczba symboli z Bybit: {len(valid_symbols)}")
    print(f"🔚 Ostatnie 10 symboli: {valid_symbols[-10:]}")
    return valid_symbols

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
    print(f"🧪 [get_market_data] Start dla {symbol}")
    
    # Najpierw spróbuj pobrać dane z tickers endpoint dla prawidłowego wolumenu
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        # Endpoint tickers nie wymaga autoryzacji
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            ticker_data = response.json()
            if ticker_data.get("result", {}).get("list"):
                ticker = ticker_data["result"]["list"][0]
                
                # Pobierz prawidłowy wolumen USDT z turnover24h
                volume_usdt = float(ticker.get("turnover24h", 0))
                price = float(ticker.get("lastPrice", 0))
                
                if price > 0:
                    ticker_market_data = {
                        "price": price,
                        "volume": volume_usdt,  # Prawidłowy wolumen USDT
                        "best_bid": float(ticker.get("bid1Price", 0)),
                        "best_ask": float(ticker.get("ask1Price", 0)),
                        "close": price,
                        "high24h": float(ticker.get("highPrice24h", 0)),
                        "low24h": float(ticker.get("lowPrice24h", 0)),
                        "volume24h": float(ticker.get("volume24h", 0)),
                        "price_change_pct": float(ticker.get("price24hPcnt", 0)) * 100
                    }
                    print(f"✅ [get_market_data] Ticker data dla {symbol}: cena=${price}, volume=${volume_usdt:,.0f}")
                    return True, ticker_market_data, price, True
        else:
            print(f"❌ Ticker API status: {response.status_code} dla {symbol}")
    except Exception as e:
        print(f"❌ Błąd pobierania ticker data dla {symbol}: {e}")
    
    # Fallback do starej metody tylko jeśli ticker API nie działa
    print(f"🔄 Fallback do get_all_data() dla {symbol}")
    data = get_all_data(symbol)
    
    # FIX: Sprawdź czy data nie jest None
    if data is None:
        print(f"❌ [get_market_data] get_all_data zwróciło None dla {symbol}")
        return False, {}, 0.0, False
    
    # Sprawdź czy data to dict przed wywołaniem .get()
    if not isinstance(data, dict):
        print(f"❌ [get_market_data] data nie jest dict dla {symbol}: {type(data)}")
        return False, {}, 0.0, False
    
    for key in ["prev_candle", "last_candle"]:
        if isinstance(data.get(key), dict):
            data[key] = list(data[key].values())

    # FIX: Rozpakuj tuple jeśli taka została zwrócona
    if isinstance(data, tuple):
        print(f"⚠️ [get_market_data] Zwrócono tuple zamiast dict → {data}")
        if len(data) > 1:
            second_element = data[1]
            if isinstance(second_element, dict):
                data = second_element
            else:
                data = {}
        else:
            data = {}

    if not isinstance(data, dict):
        return False, {}, 0.0, False

    # ✅ Zamień listy świec na dict (dla kompatybilności)
    for key in ["prev_candle", "last_candle"]:
        if key in data and isinstance(data[key], list) and len(data[key]) >= 6:
            c = data[key]
            data[key] = {
                "timestamp": int(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "quote_volume": float(c[6]) if len(c) > 6 else None
            }

    # Sprawdź czy zawiera dane zamknięcia
    price = data.get("close")
    if price is not None:
        print(f"✅ [get_market_data] Zwracam dane dla {symbol}: cena = {price}")
        return True, data, price, True


    # Fallback v2 - try v5 tickers without auth
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        print(f"🧪 Fallback v5 tickers URL: {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            ticker_data = response.json()
            if ticker_data.get("result", {}).get("list"):
                ticker = ticker_data["result"]["list"][0]
                volume_usdt = float(ticker.get("turnover24h", 0))
                price = float(ticker.get("lastPrice", 0))
                
                if price > 0:
                    print(f"✅ Fallback v5: {symbol} price=${price}, volume=${volume_usdt:,.0f}")
                    fallback_data = {
                        "price": price,
                        "volume": volume_usdt,  # Correct USDT volume from turnover24h
                        "best_bid": float(ticker.get("bid1Price", 0)),
                        "best_ask": float(ticker.get("ask1Price", 0)),
                        "close": price,
                        "high24h": float(ticker.get("highPrice24h", 0)),
                        "low24h": float(ticker.get("lowPrice24h", 0)),
                        "price_change": float(ticker.get("price24hPcnt", 0)) * 100
                    }
                    return True, fallback_data, price, True
        else:
            print(f"❌ Fallback v5 status: {response.status_code}")
    except Exception as e:
        print(f"❌ Błąd fallback v5 dla {symbol}: {e}")

    # Fallback v2 (legacy)
    try:
        url = f"https://api.bybit.com/v2/public/tickers?symbol={symbol}"
        print(f"🧪 Fallback v2 URL: {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            ticker_data = response.json()
            if ticker_data and "result" in ticker_data:
                price_str = ticker_data["result"].get("last_price") or ticker_data["result"].get("lastPrice")
                # Try to get volume from v2 API if available
                volume_str = ticker_data["result"].get("turnover_24h")
                
                if price_str:
                    price = float(price_str)
                    volume = float(volume_str) if volume_str else None
                    
                    if price > 0:
                        print(f"✅ Fallback v2: {symbol} price=${price}, volume=${volume or 0:,.0f}")
                        fallback_data = {
                            "price": price,
                            "volume": volume,  # May be None if not available
                            "open": None,
                            "high": None,
                            "low": None,
                            "close": price,
                            "price_change": None,
                            "body_ratio": None,
                            "last_candle": None,
                            "prev_candle": None
                        }
                        return True, fallback_data, price, True
    except Exception as e:
        print(f"❌ Błąd fallbacku tickera dla {symbol}: {e}")

    # Fallback prywatny
    price = get_price_from_bybit_private(symbol)
    if price:
        fallback_data = {
            "price": price,
            "volume": None,
            "open": None,
            "high": None,
            "low": None,
            "close": price,
            "price_change": None,
            "body_ratio": None,
            "last_candle": None,
            "prev_candle": None
        }
        return True, fallback_data, price, True

    print(f"❌ Nie udało się pobrać danych dla {symbol}")
    return False, {}, 0.0, False

def get_price_from_bybit_private(symbol):
    from utils.data_fetchers import get_bybit_headers

    for category in ["linear", "inverse"]:
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {
                "category": category,
                "symbol": symbol
            }
            headers = get_bybit_headers(params)
            print(f"🧪 [PRIVATE API] Pobieram cenę dla {symbol} z kategorią {category}")
            print("📤 REQUEST:", url)
            print("📤 PARAMS:", params)
            print("📤 HEADERS:", {k: headers[k] for k in headers if k.startswith("X-BAPI")})

            response = requests.get(url, headers=headers, params=params, timeout=10)
            print("📥 STATUS CODE:", response.status_code)
            json_data = response.json()
            print("📥 JSON (surowy):", json.dumps(json_data)[:1000])

            if json_data.get("retCode") == 0:
                results = json_data["result"].get("list", [])
                if not results:
                    print("⚠️ Brak wyników w liście result – fallback na pełną listę")
                    fallback_params = {"category": category}
                    fallback_response = requests.get(url, headers=headers, params=fallback_params, timeout=10)
                    fallback_data = fallback_response.json()
                    results = fallback_data["result"].get("list", [])

                for item in results:
                    if item.get("symbol") == symbol:
                        price = float(item.get("lastPrice", 0))
                        if price > 0:
                            print(f"✅ Cena z {category} dla {symbol}: {price}")
                            return price

        except Exception as e:
            print(f"❌ Wyjątek przy pobieraniu ceny z Bybit ({category}) dla {symbol}: {e}")

    print(f"❌ Brak ceny dla {symbol} z obu kategorii")
    return None

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

def build_bybit_symbol_cache():
    from utils.data_fetchers import get_bybit_headers

    symbols = set()
    cursor = ""

    while True:
        try:
            params = {
                "category": "linear",
                "limit": 1000
            }
            if cursor:
                params["cursor"] = cursor

            param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            url = f"{BYBIT_ENDPOINT}?{param_str}"

            headers = get_bybit_headers(params)
            response = requests.get(url, headers=headers)
            data = response.json()

            print("📦 API Response (skrót):", json.dumps(data, indent=2)[:800])

            if "result" not in data or "list" not in data["result"]:
                print("❌ Brak danych w odpowiedzi Bybit.")
                break

            page_symbols = data["result"]["list"]
            for item in page_symbols:
                symbol = item.get("symbol")
                status = item.get("status")
                if symbol and symbol.endswith("USDT") and status == "Trading":
                    symbols.add(symbol)

            cursor = data["result"].get("nextPageCursor")
            print(f"➡️ nextPageCursor: {cursor}")
            if not cursor:
                break

            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Błąd podczas pobierania symboli z Bybit: {e}")
            break

    os.makedirs(os.path.dirname(BYBIT_SYMBOLS_PATH), exist_ok=True)
    with open(BYBIT_SYMBOLS_PATH, "w") as f:
        json.dump(sorted(list(symbols)), f, indent=2)

    print(f"✅ Zapisano {len(symbols)} symboli do {BYBIT_SYMBOLS_PATH}")

def is_bybit_cache_expired(hours=24):
    """Check if Bybit cache is older than specified hours"""
    if not os.path.exists(BYBIT_SYMBOLS_PATH):
        print(f"🕒 Cache file nie istnieje: {BYBIT_SYMBOLS_PATH}")
        return True
    
    file_time = datetime.fromtimestamp(os.path.getmtime(BYBIT_SYMBOLS_PATH))
    age = datetime.now() - file_time
    is_expired = age > timedelta(hours=hours)
    
    print(f"🕒 Cache age: {age}, expired: {is_expired}")
    return is_expired

def build_bybit_symbol_cache_all_categories():
    from utils.data_fetchers import get_bybit_headers
    import time

    all_symbols = set()
    categories = ["linear", "inverse", "spot"]

    for category in categories:
        print(f"📦 Pobieram symbole dla kategorii: {category}")
        cursor = ""

        while True:
            try:
                params = {
                    "category": category,
                    "limit": 1000
                }
                if cursor:
                    params["cursor"] = cursor

                param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
                url = f"{BYBIT_ENDPOINT}?{param_str}"

                headers = get_bybit_headers(params)
                response = requests.get(url, headers=headers)
                
                print(f"🌐 URL: {url}")
                print(f"🔐 Headers: {headers}")
                print(f"📥 Response status code: {response.status_code}")
                print(f"📥 Response text (first 300 chars): {response.text[:300]}")
                try:
                    data = response.json()
                except Exception as e:
                    print(f"❌ Nie udało się sparsować JSON: {e}")
                    print(f"🔴 Odpowiedź serwera: {response.text[:300]}")
                    break


                if "result" not in data or "list" not in data["result"]:
                    print(f"❌ Brak danych w odpowiedzi dla kategorii {category}")
                    break

                page_symbols = data["result"]["list"]
                usdt_symbols = [item["symbol"] for item in page_symbols if item["symbol"].endswith("USDT")]
                all_symbols.update(usdt_symbols)

                cursor = data["result"].get("nextPageCursor")
                print(f"➡️ nextPageCursor: {cursor} ({len(usdt_symbols)} symboli)")
                if not cursor:
                    break

                time.sleep(0.5)

            except Exception as e:
                print(f"❌ Błąd dla kategorii {category}: {e}")
                break

    os.makedirs(os.path.dirname(BYBIT_SYMBOLS_PATH), exist_ok=True)
    with open(BYBIT_SYMBOLS_PATH, "w") as f:
        json.dump(sorted(list(all_symbols)), f, indent=2)

    print(f"✅ Zapisano {len(all_symbols)} unikalnych symboli do {BYBIT_SYMBOLS_PATH}")

