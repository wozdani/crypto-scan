"""
Safe Candles Fetcher for Trend-Mode Analysis
Provides robust candle data fetching with validation and error handling
"""

from utils.data_fetchers import get_all_data
import requests
import os
import json

def get_candles(symbol, interval="15m", limit=96):
    """
    Get candles from Bybit API - ported from pump-analysis
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Timeframe (default '15m')
        limit: Number of candles to fetch (default 96)
        
    Returns:
        list: Candles in format [[timestamp, open, high, low, close, volume], ...]
    """
    import os
    import hashlib
    import hmac
    import time
    
    try:
        # Prepare authenticated request
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_SECRET_KEY')
        
        base_url = "https://api.bybit.com"
        endpoint = "/v5/market/kline"
        
        # Parameters - fix interval format for Bybit API
        bybit_interval = interval.replace("m", "")  # "15m" -> "15"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": str(limit)
        }
        
        # Create query string
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Authentication headers
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        if api_key and api_secret:
            # Authenticated request
            param_str = f"{timestamp}{api_key}{recv_window}{query_string}"
            signature = hmac.new(
                api_secret.encode('utf-8'),
                param_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            headers = {
                "X-BAPI-API-KEY": api_key,
                "X-BAPI-SIGN": signature,
                "X-BAPI-SIGN-TYPE": "2",
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "Content-Type": "application/json"
            }
        else:
            # Public request (no auth)
            headers = {"Content-Type": "application/json"}
        
        # Make request
        url = f"{base_url}{endpoint}?{query_string}"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                candles_raw = data.get("result", {}).get("list", [])
                
                if not candles_raw:
                    print(f"[TREND DEBUG] {symbol}: No candles returned from API (retCode: {data.get('retCode')}, retMsg: {data.get('retMsg', 'N/A')})")
                    return []
                
                # Convert to standard format and reverse (oldest first)
                candles = []
                for candle_data in reversed(candles_raw):  # Bybit returns newest first
                    try:
                        timestamp = int(candle_data[0])
                        open_price = float(candle_data[1])
                        high_price = float(candle_data[2])
                        low_price = float(candle_data[3])
                        close_price = float(candle_data[4])
                        volume = float(candle_data[5])
                        
                        candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
                    except (ValueError, IndexError, TypeError) as e:
                        print(f"[TREND ERROR] {symbol}: Invalid candle format – {e}")
                        continue
                
                print(f"[TREND DEBUG] {symbol}: Successfully fetched {len(candles)} candles from Bybit API")
                return candles
                
            else:
                print(f"[TREND ERROR] {symbol}: API error – {data.get('retMsg', 'Unknown')}")
                return []
        else:
            print(f"[TREND ERROR] {symbol}: HTTP {response.status_code} – {response.text[:100]}")
            # In Replit environment, Bybit API returns 403 - this is expected
            if response.status_code == 403:
                print(f"[TREND DEBUG] {symbol}: Bybit API 403 (expected in Replit) - skipping to fallback")
            return []
            
    except Exception as e:
        print(f"[TREND ERROR] {symbol}: Exception in get_candles – {e}")
        return []

def safe_get_candles(symbol, interval="15m", limit=96):
    """
    Safely fetch candles with comprehensive validation and error handling
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Timeframe (default '15m')
        limit: Number of candles to fetch (default 96)
        
    Returns:
        list: Valid candles list or empty list if insufficient data
    """
    try:
        # Prioritize async scan results cache first
        async_file = f"data/async_results/{symbol}.json"
        if os.path.exists(async_file):
            try:
                import json
                with open(async_file, 'r') as f:
                    async_data = json.load(f)
                    cached_candles = async_data.get("candles", [])
                    if cached_candles and len(cached_candles) >= 10:
                        print(f"[TREND DEBUG] {symbol}: Using async cached candles ({len(cached_candles)})")
                        return cached_candles
            except Exception as e:
                print(f"[TREND DEBUG] {symbol}: Async cache read error: {e}")
        
        # Try scan results cache
        scan_file = f"data/scan_results/latest_scan.json"
        if os.path.exists(scan_file):
            try:
                import json
                with open(scan_file, 'r') as f:
                    scan_data = json.load(f)
                    for result in scan_data.get("results", []):
                        if result.get("symbol") == symbol:
                            scan_candles = result.get("candles", result.get("market_data", {}).get("candles", []))
                            if scan_candles and len(scan_candles) >= 10:
                                print(f"[TREND DEBUG] {symbol}: Using scan cached candles ({len(scan_candles)})")
                                return scan_candles
            except Exception as e:
                print(f"[TREND DEBUG] {symbol}: Scan cache read error: {e}")
        
        # Fallback to existing data fetcher
        print(f"[TREND DEBUG] {symbol}: Trying fallback data source...")
        from utils.data_fetchers import get_all_data
        market_data = get_all_data(symbol)
        
        if market_data and isinstance(market_data, dict):
            fallback_candles = market_data.get('candles', [])
            if fallback_candles and len(fallback_candles) >= 10:
                print(f"[TREND DEBUG] {symbol}: Fallback successful with {len(fallback_candles)} candles")
                return fallback_candles
            else:
                print(f"[TREND DEBUG] {symbol}: Fallback insufficient ({len(fallback_candles) if fallback_candles else 0}/10)")
        else:
            print(f"[TREND DEBUG] {symbol}: Fallback data unavailable or invalid format")
        
        # Last resort: Use pump-analysis get_candles (will fail in Replit but try anyway)
        print(f"[TREND DEBUG] {symbol}: Trying direct API as last resort...")
        candles = get_candles(symbol, interval, limit)
        
        if candles and len(candles) >= 10:
            print(f"[TREND DEBUG] {symbol}: Direct API successful with {len(candles)} candles")
            return candles
        
        print(f"[TREND DEBUG] {symbol}: No valid candle source available")
        return []
        
    except Exception as e:
        print(f"[TREND ERROR] {symbol}: Exception in safe_get_candles – {e}")
        return []

def validate_candles_quality(candles, symbol=None, min_candles=10):
    """
    Validate candle data quality for trend analysis
    
    Args:
        candles: List of candle data
        symbol: Symbol name for logging
        min_candles: Minimum number of candles required
        
    Returns:
        bool: True if candles are suitable for analysis
    """
    if not candles or len(candles) < min_candles:
        if symbol:
            print(f"[CANDLE QUALITY] {symbol}: Insufficient candles ({len(candles) if candles else 0}/{min_candles})")
        return False
    
    try:
        # Check for valid price data
        valid_candles = 0
        for candle in candles[-10:]:  # Check last 10 candles
            if (len(candle) >= 5 and 
                all(isinstance(x, (int, float)) for x in candle[1:5]) and  # OHLC must be numbers
                candle[2] >= candle[3] and  # High >= Low
                candle[4] > 0):  # Close > 0
                valid_candles += 1
        
        quality_ratio = valid_candles / 10
        
        if quality_ratio >= 0.8:  # At least 80% valid candles
            if symbol:
                print(f"[TREND DEBUG] {symbol}: Candle quality check passed ({valid_candles}/10 valid)")
            return True
        else:
            if symbol:
                print(f"[TREND DEBUG] {symbol}: Candle quality check failed ({valid_candles}/10 valid)")
            return False
            
    except Exception as e:
        if symbol:
            print(f"[TREND ERROR] {symbol}: Error validating candle quality – {e}")
        return False

def safe_trend_analysis_check(symbol, market_data=None):
    """
    Complete safety check for trend analysis readiness
    
    Args:
        symbol: Trading pair symbol
        market_data: Optional pre-fetched market data
        
    Returns:
        tuple: (candles_list, is_ready_for_analysis)
    """
    try:
        # Try to use existing market data first
        if market_data and isinstance(market_data, dict):
            existing_candles = market_data.get('candles', [])
            if validate_candles_quality(existing_candles, symbol):
                print(f"[TREND DEBUG] {symbol}: Using existing market data candles")
                return existing_candles, True
        
        # Fetch fresh candles
        candles = safe_get_candles(symbol, "15m", 96)
        
        if validate_candles_quality(candles, symbol):
            return candles, True
        else:
            print(f"[TREND DEBUG] {symbol}: Skipping trend analysis - insufficient quality data")
            return [], False
            
    except Exception as e:
        print(f"[TREND ERROR] {symbol}: Safety check failed – {e}")
        return [], False