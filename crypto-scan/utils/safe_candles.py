"""
Safe Candles Fetcher for Trend-Mode Analysis
Provides robust candle data fetching with validation and error handling
"""

from utils.data_fetchers import get_all_data
import requests

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
        # Try direct Bybit API call for candles
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                candles_raw = data.get("result", {}).get("list", [])
                
                if candles_raw and len(candles_raw) >= 10:
                    # Convert to standard format: [timestamp, open, high, low, close, volume]
                    candles = []
                    for candle in candles_raw:
                        try:
                            timestamp = int(candle[0])
                            open_price = float(candle[1])
                            high_price = float(candle[2])
                            low_price = float(candle[3])
                            close_price = float(candle[4])
                            volume = float(candle[5])
                            
                            candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
                        except (ValueError, IndexError) as e:
                            print(f"[TREND ERROR] {symbol}: Invalid candle data format – {e}")
                            continue
                    
                    if len(candles) >= 10:
                        print(f"[TREND DEBUG] {symbol}: Successfully fetched {len(candles)} valid candles")
                        return candles
                    else:
                        print(f"[TREND DEBUG] {symbol}: Insufficient valid candles after parsing ({len(candles)}/10)")
                        return []
                else:
                    print(f"[TREND DEBUG] {symbol}: Insufficient raw candles ({len(candles_raw) if candles_raw else 0}/10)")
                    return []
            else:
                print(f"[TREND ERROR] {symbol}: Bybit API error – {data.get('retMsg', 'Unknown error')}")
        else:
            print(f"[TREND ERROR] {symbol}: HTTP {response.status_code} fetching candles")
        
        # Fallback to existing data fetcher
        print(f"[TREND DEBUG] {symbol}: Trying fallback data source...")
        market_data = get_all_data(symbol)
        
        if market_data and isinstance(market_data, dict):
            fallback_candles = market_data.get('candles', [])
            if fallback_candles and len(fallback_candles) >= 10:
                print(f"[TREND DEBUG] {symbol}: Fallback successful with {len(fallback_candles)} candles")
                return fallback_candles
            else:
                print(f"[TREND DEBUG] {symbol}: Fallback insufficient ({len(fallback_candles) if fallback_candles else 0}/10)")
        
        print(f"[TREND DEBUG] {symbol}: No valid candle source available")
        return []
        
    except requests.exceptions.RequestException as e:
        print(f"[TREND ERROR] {symbol}: Network error fetching candles – {e}")
        return []
    except Exception as e:
        print(f"[TREND ERROR] {symbol}: Unexpected error fetching candles – {e}")
        return []

def validate_candles_quality(candles, symbol=None):
    """
    Validate candle data quality for trend analysis
    
    Args:
        candles: List of candle data
        symbol: Symbol name for logging
        
    Returns:
        bool: True if candles are suitable for analysis
    """
    if not candles or len(candles) < 10:
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