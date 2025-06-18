# -*- coding: utf-8 -*-
import requests
import time
import os
import hmac
import hashlib
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

BYBIT_BASE_URL = "https://api.bybit.com"
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")
BYBIT_SYMBOLS_PATH = "utils/data/cache/bybit_symbols.json"
BYBIT_ENDPOINT = "https://api.bybit.com/v5/market/tickers"

def get_bybit_headers(params=None):
    """Generate authenticated headers for Bybit API - exact crypto-scan logic"""
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

def fetch_klines(symbol, interval="15", limit=1000):
    """Fetch kline data from Bybit using crypto-scan logic"""
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
            logger.error(f"Bybit API error for {symbol}: {data}")
            return None
    except Exception as e:
        logger.error(f"Exception for {symbol}: {e}")
        return None

def get_symbols_cached(require_chain=False):
    """Get symbols from cache - crypto-scan compatible"""
    try:
        # Check if cache exists and is not expired
        if is_bybit_cache_expired():
            logger.info("Bybit cache expired or missing - building cache...")
            build_bybit_symbol_cache_all_categories()
        
        # Load symbols from cache
        if os.path.exists(BYBIT_SYMBOLS_PATH):
            with open(BYBIT_SYMBOLS_PATH, "r") as f:
                symbols = json.load(f)
            
            valid_symbols = []
            for symbol in symbols:
                if symbol.endswith("USDT"):
                    valid_symbols.append(symbol)
            
            logger.info(f"Loaded {len(valid_symbols)} USDT symbols from cache")
            return valid_symbols  # Return all symbols - no limit
        
    except Exception as e:
        logger.error(f"Error loading symbols cache: {e}")
    
    return []

def build_bybit_symbol_cache_all_categories():
    """Build symbol cache using all categories - crypto-scan logic"""
    all_symbols = set()
    categories = ["linear", "spot"]  # Focus on linear (futures) and spot
    
    for category in categories:
        logger.info(f"Fetching symbols from category: {category}")
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
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    logger.warning(f"HTTP error {response.status_code} for category {category}")
                    break
                
                try:
                    data = response.json()
                except Exception as e:
                    logger.error(f"JSON parse error: {e}")
                    break

                if "result" not in data or "list" not in data["result"]:
                    logger.warning(f"No data in response for category {category}")
                    break

                page_symbols = data["result"]["list"]
                usdt_symbols = [item["symbol"] for item in page_symbols if item["symbol"].endswith("USDT")]
                all_symbols.update(usdt_symbols)

                cursor = data["result"].get("nextPageCursor")
                logger.debug(f"Processed {len(usdt_symbols)} USDT symbols, cursor: {bool(cursor)}")
                if not cursor:
                    break

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error for category {category}: {e}")
                break

    # Save to cache
    os.makedirs(os.path.dirname(BYBIT_SYMBOLS_PATH), exist_ok=True)
    with open(BYBIT_SYMBOLS_PATH, "w") as f:
        json.dump(sorted(list(all_symbols)), f, indent=2)

    logger.info(f"Saved {len(all_symbols)} unique symbols to cache")

def is_bybit_cache_expired(hours=24):
    """Check if Bybit cache is older than specified hours"""
    if not os.path.exists(BYBIT_SYMBOLS_PATH):
        return True
    
    file_time = datetime.fromtimestamp(os.path.getmtime(BYBIT_SYMBOLS_PATH))
    age = datetime.now() - file_time
    is_expired = age > timedelta(hours=hours)
    
    if is_expired:
        logger.info(f"Cache is {age.total_seconds()/3600:.1f} hours old - expired")
    
    return is_expired

def get_historical_data(symbol, days=7, interval="15"):
    """Get historical kline data for pump analysis"""
    try:
        # Calculate start time for the specified days
        start_time = int((time.time() - (days * 24 * 3600)) * 1000)
        
        # Fetch klines with start time
        url = f"{BYBIT_BASE_URL}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,  # Maximum limit
            "start": start_time
        }
        
        headers = get_bybit_headers(params)
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        if data["retCode"] == 0:
            return data["result"]["list"]
        else:
            logger.error(f"Historical data error for {symbol}: {data}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return []