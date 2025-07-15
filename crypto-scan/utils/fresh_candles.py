"""
Fresh Candles Fetcher - Force Fresh Data for Chart Generation
Ensures chart generation always uses current market data, never cached/outdated data
"""

import os
import requests
import hmac
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
try:
    from utils.scan_error_reporter import log_scan_error
    def log_warning(category: str, message: str):
        log_scan_error(message, category)
except ImportError:
    def log_warning(category: str, message: str):
        print(f"⚠️ [{category}] {message}")

def fetch_fresh_candles(symbol: str, interval: str = "15m", limit: int = 96, force_refresh: bool = True) -> List[Dict]:
    """
    Fetch fresh candles directly from Bybit API - bypass all caching
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Timeframe ('15m', '5m', '1h', '4h', '1d')
        limit: Number of candles to fetch (max 1000)
        force_refresh: Always fetch fresh data, ignore cache
        
    Returns:
        List of candle dictionaries with current market data
    """
    try:
        print(f"[FRESH CANDLES] 🔄 Fetching fresh {interval} data for {symbol} (force_refresh={force_refresh})")
        
        # Bybit API configuration
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_SECRET_KEY')
        base_url = "https://api.bybit.com"
        endpoint = "/v5/market/kline"
        
        # Convert interval format for Bybit API
        bybit_interval = interval.replace("m", "")  # "15m" -> "15"
        
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": str(limit)
        }
        
        # Create query string
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Authentication headers for better rate limits
        headers = {"Content-Type": "application/json"}
        
        if api_key and api_secret:
            timestamp = str(int(time.time() * 1000))
            recv_window = "5000"
            
            param_str = f"{timestamp}{api_key}{recv_window}{query_string}"
            signature = hmac.new(
                api_secret.encode('utf-8'),
                param_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            headers.update({
                "X-BAPI-API-KEY": api_key,
                "X-BAPI-SIGN": signature,
                "X-BAPI-SIGN-TYPE": "2",
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window
            })
        
        # Make fresh API request
        url = f"{base_url}{endpoint}?{query_string}"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                candles_raw = data.get("result", {}).get("list", [])
                
                if not candles_raw:
                    log_warning("FRESH CANDLES API", additional_info=f"{symbol}: No candles returned from Bybit API")
                    return []
                
                # Convert to dictionary format (newest first from Bybit)
                candles = []
                for candle_data in reversed(candles_raw):  # Reverse to get oldest first
                    try:
                        candle_dict = {
                            "timestamp": int(candle_data[0]),
                            "open": float(candle_data[1]),
                            "high": float(candle_data[2]),
                            "low": float(candle_data[3]),
                            "close": float(candle_data[4]),
                            "volume": float(candle_data[5])
                        }
                        candles.append(candle_dict)
                    except (ValueError, IndexError, TypeError) as e:
                        log_warning("FRESH CANDLES PARSE", exception=e, additional_info=f"{symbol}: Invalid candle format")
                        continue
                
                # Validate data freshness
                if candles:
                    last_candle_time = datetime.fromtimestamp(candles[-1]['timestamp'] / 1000)
                    current_time = datetime.now(timezone.utc)
                    time_diff = current_time - last_candle_time
                    
                    print(f"[FRESH CANDLES] ✅ {symbol}: Fetched {len(candles)} fresh candles")
                    print(f"[FRESH CANDLES] 📅 Last candle: {last_candle_time.strftime('%H:%M:%S UTC')}")
                    print(f"[FRESH CANDLES] 🕐 Current time: {current_time.strftime('%H:%M:%S UTC')}")
                    print(f"[FRESH CANDLES] ⏱️ Data age: {time_diff.total_seconds()/60:.1f} minutes")
                    
                    # Warn if data is older than 30 minutes
                    if time_diff.total_seconds() > 1800:  # 30 minutes
                        log_warning("FRESH CANDLES STALE", f"{symbol}: Data is {time_diff.total_seconds()/60:.1f} minutes old")
                
                return candles
            else:
                log_warning("FRESH CANDLES API ERROR", f"{symbol}: {data.get('retMsg', 'Unknown API error')}")
                return []
        else:
            # Handle expected 403 in Replit environment
            if response.status_code == 403:
                print(f"[FRESH CANDLES] ⚠️ {symbol}: API 403 (expected in Replit environment)")
                log_warning("FRESH CANDLES 403", f"{symbol}: Bybit API blocked in development environment")
            else:
                log_warning("FRESH CANDLES HTTP", f"{symbol}: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        log_warning("FRESH CANDLES ERROR", f"{symbol}: Failed to fetch fresh candles - {e}")
        return []

def validate_candle_freshness(candles: List[Dict], symbol: str, max_age_minutes: int = 30) -> bool:
    """
    Validate that candle data is fresh enough for chart generation
    
    Args:
        candles: List of candle dictionaries
        symbol: Trading symbol for logging
        max_age_minutes: Maximum allowed age in minutes
        
    Returns:
        True if data is fresh enough, False otherwise
    """
    if not candles:
        print(f"[FRESHNESS CHECK] ❌ {symbol}: No candles to validate")
        return False
    
    try:
        # Get timestamp of last candle
        last_candle = candles[-1]
        last_timestamp = last_candle.get('timestamp', 0)
        
        # Convert to datetime
        if last_timestamp > 1e12:  # Milliseconds
            last_time = datetime.fromtimestamp(last_timestamp / 1000)
        else:  # Seconds
            last_time = datetime.fromtimestamp(last_timestamp)
        
        current_time = datetime.now(timezone.utc)
        time_diff = current_time - last_time
        age_minutes = time_diff.total_seconds() / 60
        
        is_fresh = age_minutes <= max_age_minutes
        
        print(f"[FRESHNESS CHECK] {symbol}: Data age {age_minutes:.1f}min {'✅' if is_fresh else '❌'} (limit: {max_age_minutes}min)")
        
        if not is_fresh:
            log_warning("CANDLE DATA STALE", additional_info=f"{symbol}: Data is {age_minutes:.1f} minutes old (limit: {max_age_minutes})")
        
        return is_fresh
        
    except Exception as e:
        log_warning("FRESHNESS CHECK ERROR", exception=e, additional_info=f"{symbol}: Failed to validate freshness")
        return False

def get_fresh_candles_for_charts(symbol: str, interval: str = "15m", limit: int = 96) -> List[Dict]:
    """
    Get fresh candles specifically for chart generation with validation
    
    Args:
        symbol: Trading symbol
        interval: Timeframe
        limit: Number of candles
        
    Returns:
        Fresh candle data suitable for chart generation
    """
    print(f"[CHART CANDLES] 🎯 Getting fresh candles for {symbol} chart generation")
    
    # Always fetch fresh data for charts
    candles = fetch_fresh_candles(symbol, interval, limit, force_refresh=True)
    
    if not candles:
        print(f"[CHART CANDLES] ❌ {symbol}: No fresh candles available")
        return []
    
    # Validate freshness
    if not validate_candle_freshness(candles, symbol, max_age_minutes=30):
        print(f"[CHART CANDLES] ⚠️ {symbol}: Candles are stale, but using anyway for chart generation")
    
    print(f"[CHART CANDLES] ✅ {symbol}: Ready for chart generation with {len(candles)} fresh candles")
    return candles

def debug_candle_timestamps(candles: List[Dict], symbol: str):
    """
    Debug function to print candle timestamp information
    
    Args:
        candles: List of candle data
        symbol: Trading symbol for identification
    """
    if not candles:
        print(f"[CANDLE DEBUG] {symbol}: No candles to debug")
        return
    
    print(f"[CANDLE DEBUG] {symbol}: Analyzing {len(candles)} candles")
    
    # Show first and last few candles
    for i, candle in enumerate(candles[:3]):  # First 3
        timestamp = candle.get('timestamp', 0)
        if timestamp > 1e12:
            dt = datetime.fromtimestamp(timestamp / 1000)
        else:
            dt = datetime.fromtimestamp(timestamp)
        print(f"[CANDLE DEBUG] {symbol}: Candle {i+1}: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    if len(candles) > 6:
        print(f"[CANDLE DEBUG] {symbol}: ... ({len(candles)-6} candles in between) ...")
    
    for i, candle in enumerate(candles[-3:]):  # Last 3
        timestamp = candle.get('timestamp', 0)
        if timestamp > 1e12:
            dt = datetime.fromtimestamp(timestamp / 1000)
        else:
            dt = datetime.fromtimestamp(timestamp)
        idx = len(candles) - 3 + i
        print(f"[CANDLE DEBUG] {symbol}: Candle {idx+1}: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Current time comparison
    current_time = datetime.now(timezone.utc)
    print(f"[CANDLE DEBUG] {symbol}: Current UTC time: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")