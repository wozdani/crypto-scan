"""
Robust API Client with Retry Logic and Comprehensive Fallback
Fixes issues with MAVIAUSDT and other symbols failing to fetch data
"""
import requests
import time
import json
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

class RobustAPIClient:
    """API client with retry logic and fallback mechanisms"""
    
    def __init__(self):
        self.max_retries = 3
        self.timeout = 5
        self.retry_delay = 1.0
        
    def api_call_with_retry(self, url: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """
        Make API call with retry logic and detailed error handling
        
        Args:
            url: API endpoint URL
            params: Request parameters
            headers: Request headers
            
        Returns:
            API response JSON or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    headers=headers or {}, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                print(f"[API RETRY {attempt+1}/{self.max_retries}] Timeout for {url}")
            except requests.exceptions.HTTPError as e:
                print(f"[API RETRY {attempt+1}/{self.max_retries}] HTTP {e.response.status_code} for {url}")
                if e.response.status_code == 403:
                    # Don't retry 403 errors
                    break
            except requests.exceptions.ConnectionError:
                print(f"[API RETRY {attempt+1}/{self.max_retries}] Connection error for {url}")
            except Exception as e:
                print(f"[API RETRY {attempt+1}/{self.max_retries}] Unexpected error for {url}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        print(f"[API FAIL] All retries failed for {url}")
        return None

    def get_bybit_ticker_v5(self, symbol: str) -> Optional[Dict]:
        """Get ticker data from Bybit v5 API with retry logic"""
        url = f"https://api.bybit.com/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        
        data = self.api_call_with_retry(url, params)
        if data and data.get("result", {}).get("list"):
            return data["result"]["list"][0]
        return None

    def get_bybit_kline_v5(self, symbol: str, interval: str = "15", limit: int = 50) -> Optional[list]:
        """Get kline/candle data from Bybit v5 API with retry logic"""
        url = f"https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        data = self.api_call_with_retry(url, params)
        if data and data.get("result", {}).get("list"):
            return data["result"]["list"]
        return None

    def get_bybit_orderbook_v5(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """Get orderbook data from Bybit v5 API with retry logic"""
        url = f"https://api.bybit.com/v5/market/orderbook"
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit
        }
        
        data = self.api_call_with_retry(url, params)
        if data and data.get("result"):
            return data["result"]
        return None

    def validate_symbol_availability(self, symbol: str) -> bool:
        """Check if symbol is available and active on Bybit"""
        ticker = self.get_bybit_ticker_v5(symbol)
        if not ticker:
            # If API fails, assume symbol is available for essential symbols
            essential_symbols = [
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT",
                "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT", "LTCUSDT", "UNIUSDT"
            ]
            return symbol in essential_symbols
            
        # Check if symbol has valid price and volume
        try:
            price = float(ticker.get("lastPrice", 0))
            volume = float(ticker.get("turnover24h", 0))
            return price > 0 and volume > 0
        except (ValueError, TypeError):
            return False

def get_robust_market_data(symbol: str) -> Tuple[bool, Dict[str, Any], float, bool]:
    """
    Enhanced market data fetcher with robust error handling
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Tuple of (success, data_dict, price, is_valid)
    """
    client = RobustAPIClient()
    
    print(f"[ROBUST API] Fetching data for {symbol}...")
    
    # Step 1: Get ticker data with retry (skip validation during API outages)
    ticker = client.get_bybit_ticker_v5(symbol)
    if not ticker:
        print(f"[API FAIL] {symbol} ticker data unavailable - API might be down")
        return False, {}, 0.0, False
    
    try:
        # Extract essential data
        price = float(ticker.get("lastPrice", 0))
        volume_usdt = float(ticker.get("turnover24h", 0))
        
        if price <= 0:
            print(f"[DATA INVALID] {symbol} has invalid price: {price}")
            return False, {}, 0.0, False
            
        if volume_usdt < 50000:  # Minimum liquidity threshold
            print(f"[LIQUIDITY LOW] {symbol} volume too low: ${volume_usdt:,.0f}")
            return False, {}, 0.0, False
        
        # Build comprehensive market data
        market_data = {
            "symbol": symbol,
            "price": price,
            "volume": volume_usdt,
            "best_bid": float(ticker.get("bid1Price", 0)),
            "best_ask": float(ticker.get("ask1Price", 0)),
            "close": price,
            "high24h": float(ticker.get("highPrice24h", 0)),
            "low24h": float(ticker.get("lowPrice24h", 0)),
            "volume24h": float(ticker.get("volume24h", 0)),
            "price_change_pct": float(ticker.get("price24hPcnt", 0)) * 100,
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 3: Enhance with candle data (optional)
        candles = client.get_bybit_kline_v5(symbol, "15", 10)
        if candles and len(candles) >= 2:
            # Parse latest candle: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
            latest = candles[0]  # Most recent candle
            previous = candles[1] if len(candles) > 1 else candles[0]
            
            market_data.update({
                "candles": candles,
                "last_candle": {
                    "timestamp": int(latest[0]),
                    "open": float(latest[1]),
                    "high": float(latest[2]),
                    "low": float(latest[3]),
                    "close": float(latest[4]),
                    "volume": float(latest[5])
                },
                "prev_candle": {
                    "timestamp": int(previous[0]),
                    "open": float(previous[1]),
                    "high": float(previous[2]),
                    "low": float(previous[3]),
                    "close": float(previous[4]),
                    "volume": float(previous[5])
                }
            })
            
            # Calculate additional metrics
            last_close = float(latest[4])
            last_open = float(latest[1])
            last_high = float(latest[2])
            last_low = float(latest[3])
            
            price_change = last_close - last_open
            candle_range = last_high - last_low
            body_ratio = abs(last_close - last_open) / max(candle_range, 0.0001)
            
            market_data.update({
                "open": last_open,
                "high": last_high,
                "low": last_low,
                "price_change": price_change,
                "body_ratio": round(body_ratio, 4)
            })
        
        # Step 4: Add orderbook data (optional)
        orderbook = client.get_bybit_orderbook_v5(symbol, 25)
        if orderbook:
            market_data["orderbook"] = orderbook
        
        print(f"[ROBUST SUCCESS] {symbol}: ${price:,.4f}, Volume: ${volume_usdt:,.0f}")
        return True, market_data, price, True
        
    except Exception as e:
        print(f"[DATA PARSE ERROR] {symbol}: {e}")
        return False, {}, 0.0, False

def get_data_fallback(symbol: str) -> Dict[str, Any]:
    """
    Fallback data provider using essential market information
    Only returns authentic data or explicit None values
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Basic market data dict with None for unavailable fields
    """
    client = RobustAPIClient()
    
    # Try basic ticker only
    ticker = client.get_bybit_ticker_v5(symbol)
    if ticker:
        price = float(ticker.get("lastPrice", 0))
        if price > 0:
            return {
                "price": price,
                "volume": float(ticker.get("turnover24h", 0)),
                "close": price,
                "open": None,  # Explicitly None when unavailable
                "high": None,
                "low": None,
                "price_change": None,
                "body_ratio": None,
                "last_candle": None,
                "prev_candle": None,
                "candles": None,
                "orderbook": None,
                "best_bid": float(ticker.get("bid1Price", 0)),
                "best_ask": float(ticker.get("ask1Price", 0)),
                "timestamp": datetime.now().isoformat()
            }
    
    # Return None if no authentic data available
    return None

# Global instance for easy access
robust_api = RobustAPIClient()

def check_symbol_health(symbol: str) -> Dict[str, Any]:
    """
    Comprehensive symbol health check
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Health status dictionary
    """
    client = RobustAPIClient()
    
    health = {
        "symbol": symbol,
        "available": False,
        "price_valid": False,
        "volume_adequate": False,
        "spread_reasonable": False,
        "candles_available": False,
        "overall_health": "unhealthy"
    }
    
    # Check ticker
    ticker = client.get_bybit_ticker_v5(symbol)
    if ticker:
        health["available"] = True
        
        price = float(ticker.get("lastPrice", 0))
        volume = float(ticker.get("turnover24h", 0))
        bid = float(ticker.get("bid1Price", 0))
        ask = float(ticker.get("ask1Price", 0))
        
        health["price_valid"] = price > 0
        health["volume_adequate"] = volume >= 50000
        
        if bid > 0 and ask > 0:
            spread = (ask - bid) / ask
            health["spread_reasonable"] = spread <= 0.02
        
        # Check candles
        candles = client.get_bybit_kline_v5(symbol, "15", 5)
        health["candles_available"] = bool(candles and len(candles) >= 2)
        
        # Overall health assessment
        if all([
            health["available"],
            health["price_valid"], 
            health["volume_adequate"],
            health["spread_reasonable"],
            health["candles_available"]
        ]):
            health["overall_health"] = "healthy"
        elif health["available"] and health["price_valid"]:
            health["overall_health"] = "limited"
        else:
            health["overall_health"] = "unhealthy"
    
    return health