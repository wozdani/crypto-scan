"""
API Fallback System - Provides market data during API outages
Uses cached data and alternative endpoints when primary APIs fail
"""
import json
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

class APIFallbackSystem:
    """Intelligent fallback system for API outages"""
    
    def __init__(self):
        self.cache_dir = "data/cache"
        self.price_cache_file = f"{self.cache_dir}/price_cache.json"
        self.last_successful_scan_file = f"{self.cache_dir}/last_successful_scan.json"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def cache_successful_data(self, symbol: str, data: Dict[str, Any]):
        """Cache successful API responses for fallback use"""
        try:
            cache_data = self.load_price_cache()
            cache_data[symbol] = {
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "cached_at": time.time()
            }
            
            with open(self.price_cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            print(f"[CACHE] Failed to cache data for {symbol}: {e}")
    
    def load_price_cache(self) -> Dict:
        """Load cached price data"""
        try:
            if os.path.exists(self.price_cache_file):
                with open(self.price_cache_file, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def get_cached_data(self, symbol: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Get cached data if available and recent enough"""
        cache = self.load_price_cache()
        if symbol not in cache:
            return None
            
        cached_entry = cache[symbol]
        cached_at = cached_entry.get("cached_at", 0)
        age_hours = (time.time() - cached_at) / 3600
        
        if age_hours <= max_age_hours:
            return cached_entry.get("data")
        return None
    
    def try_alternative_endpoint(self, symbol: str) -> Optional[Dict]:
        """Try alternative data sources when primary API fails"""
        # Try CoinGecko as backup (public endpoint)
        try:
            import requests
            
            # Convert USDT symbols to CoinGecko format
            if symbol.endswith("USDT"):
                base_symbol = symbol[:-4].lower()
                
                # Map common symbols to CoinGecko IDs
                symbol_map = {
                    "btc": "bitcoin",
                    "eth": "ethereum", 
                    "bnb": "binancecoin",
                    "ada": "cardano",
                    "sol": "solana",
                    "xrp": "ripple",
                    "doge": "dogecoin",
                    "avax": "avalanche-2",
                    "link": "chainlink",
                    "matic": "matic-network",
                    "ltc": "litecoin",
                    "uni": "uniswap"
                }
                
                coin_id = symbol_map.get(base_symbol, base_symbol)
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true"
                
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if coin_id in data:
                        coin_data = data[coin_id]
                        price = coin_data.get("usd", 0)
                        volume = coin_data.get("usd_24h_vol", 0)
                        change_24h = coin_data.get("usd_24h_change", 0)
                        
                        if price > 0:
                            fallback_data = {
                                "symbol": symbol,
                                "price": price,
                                "close": price,
                                "volume": volume * 1000 if volume else 0,  # Estimate turnover
                                "price_change_pct": change_24h,
                                "source": "coingecko",
                                "timestamp": datetime.now().isoformat()
                            }
                            print(f"[FALLBACK] CoinGecko data for {symbol}: ${price}")
                            return fallback_data
        except Exception as e:
            print(f"[FALLBACK] CoinGecko failed for {symbol}: {e}")
        
        return None
    
    def get_fallback_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data using fallback methods"""
        # Try cached data first
        cached_data = self.get_cached_data(symbol, max_age_hours=6)
        if cached_data:
            print(f"[FALLBACK] Using cached data for {symbol}")
            return cached_data
        
        # Try alternative endpoints
        alt_data = self.try_alternative_endpoint(symbol)
        if alt_data:
            return alt_data
        
        return None
    
    def save_scan_state(self, symbols_scanned: List[str], successful_count: int):
        """Save state of current scan for recovery"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "symbols_scanned": symbols_scanned,
            "successful_count": successful_count,
            "scan_time": time.time()
        }
        
        try:
            with open(self.last_successful_scan_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[FALLBACK] Failed to save scan state: {e}")
    
    def get_emergency_symbols(self) -> List[str]:
        """Get essential symbols for emergency scanning when all APIs fail"""
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT",
            "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT", "LTCUSDT", "UNIUSDT"
        ]
    
    def detect_api_outage(self, failed_requests: int, total_requests: int) -> bool:
        """Detect if we're experiencing an API outage"""
        if total_requests < 5:
            return False
        
        failure_rate = failed_requests / total_requests
        return failure_rate > 0.8  # 80% failure rate indicates outage

# Global fallback system
api_fallback = APIFallbackSystem()

def enhanced_get_market_data(symbol: str) -> tuple:
    """Enhanced market data with intelligent fallback"""
    try:
        # Try primary API first
        from utils.data_fetchers import get_market_data_legacy
        success, data, price, is_valid = get_market_data_legacy(symbol)
        
        if success and price > 0:
            # Cache successful data
            api_fallback.cache_successful_data(symbol, data)
            return success, data, price, is_valid
        
        # Primary API failed - try fallback
        fallback_data = api_fallback.get_fallback_market_data(symbol)
        if fallback_data and fallback_data.get("price", 0) > 0:
            fallback_price = fallback_data["price"]
            return True, fallback_data, fallback_price, True
        
        return False, {}, 0.0, False
        
    except Exception as e:
        print(f"[ENHANCED] Critical error for {symbol}: {e}")
        return False, {}, 0.0, False