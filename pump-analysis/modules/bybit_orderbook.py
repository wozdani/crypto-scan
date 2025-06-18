"""
Bybit Orderbook Fetcher Module
Fetches orderbook data from Bybit API for heatmap analysis
"""

import logging
import os
import time
from typing import Dict, Optional, List
import requests
from datetime import datetime
import hashlib
import hmac
import urllib.parse

logger = logging.getLogger(__name__)

class BybitOrderbookFetcher:
    """Fetches orderbook data from Bybit API"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.getenv('BYBIT_API_KEY', '')
        self.api_secret = api_secret or os.getenv('BYBIT_SECRET_KEY', '')
        self.base_url = "https://api.bybit.com"
        self.session = requests.Session()
        
        # Set headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'pump-analysis-orderbook/1.0'
        })
    
    def _generate_signature(self, params: str, timestamp: str) -> str:
        """Generate signature for authenticated requests"""
        if not self.api_secret:
            return ""
        
        param_str = timestamp + self.api_key + params
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, params: Dict = None, auth_required: bool = False) -> Optional[Dict]:
        """Make request to Bybit API"""
        
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if auth_required and self.api_key and self.api_secret:
                # Add authentication for private endpoints
                timestamp = str(int(time.time() * 1000))
                query_string = urllib.parse.urlencode(sorted(params.items()))
                signature = self._generate_signature(query_string, timestamp)
                
                headers = {
                    'X-BAPI-API-KEY': self.api_key,
                    'X-BAPI-SIGN': signature,
                    'X-BAPI-SIGN-TYPE': '2',
                    'X-BAPI-TIMESTAMP': timestamp
                }
                
                response = self.session.get(url, params=params, headers=headers)
            else:
                # Public endpoint
                response = self.session.get(url, params=params)
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('retCode') == 0:
                return data.get('result', {})
            else:
                logger.error(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {endpoint}: {e}")
            return None
    
    def get_orderbook(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """
        Fetch orderbook data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            limit: Number of orderbook levels (max 50)
            
        Returns:
            Orderbook data or None if failed
        """
        
        endpoint = "/v5/market/orderbook"
        params = {
            'category': 'linear',  # Futures
            'symbol': symbol,
            'limit': min(limit, 50)  # Bybit limit
        }
        
        logger.debug(f"Fetching orderbook for {symbol}")
        
        result = self._make_request(endpoint, params)
        
        if result:
            logger.debug(f"Orderbook fetched for {symbol}: {len(result.get('b', []))} bids, {len(result.get('a', []))} asks")
            return result
        else:
            logger.warning(f"Failed to fetch orderbook for {symbol}")
            return None
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Fetch ticker data for current price
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker data or None if failed
        """
        
        endpoint = "/v5/market/tickers"
        params = {
            'category': 'linear',
            'symbol': symbol
        }
        
        result = self._make_request(endpoint, params)
        
        if result and 'list' in result and result['list']:
            return result['list'][0]
        else:
            logger.warning(f"Failed to fetch ticker for {symbol}")
            return None
    
    def get_symbols(self, limit: int = 1000) -> List[str]:
        """
        Get all available trading symbols
        
        Args:
            limit: Maximum number of symbols to return
            
        Returns:
            List of symbol names
        """
        
        endpoint = "/v5/market/tickers"
        params = {
            'category': 'linear',
            'limit': limit
        }
        
        result = self._make_request(endpoint, params)
        
        if result and 'list' in result:
            symbols = [item['symbol'] for item in result['list']]
            logger.info(f"Fetched {len(symbols)} symbols from Bybit")
            return symbols
        else:
            logger.warning("Failed to fetch symbols from Bybit")
            return []
    
    def test_connection(self) -> bool:
        """Test connection to Bybit API"""
        
        endpoint = "/v5/market/time"
        result = self._make_request(endpoint)
        
        if result:
            server_time = result.get('timeSecond', 0)
            local_time = int(time.time())
            time_diff = abs(server_time - local_time)
            
            logger.info(f"Bybit connection test successful. Time diff: {time_diff}s")
            return time_diff < 30  # Allow 30s time difference
        else:
            logger.error("Bybit connection test failed")
            return False

class OrderbookDataCollector:
    """Collects and manages orderbook data for multiple symbols"""
    
    def __init__(self, fetcher: BybitOrderbookFetcher):
        self.fetcher = fetcher
        self.collection_interval = 30  # seconds
        self.active_symbols: List[str] = []
        self.last_collection_time = 0
    
    def add_symbol(self, symbol: str):
        """Add symbol to collection list"""
        if symbol not in self.active_symbols:
            self.active_symbols.append(symbol)
            logger.info(f"Added {symbol} to orderbook collection")
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from collection list"""
        if symbol in self.active_symbols:
            self.active_symbols.remove(symbol)
            logger.info(f"Removed {symbol} from orderbook collection")
    
    def collect_orderbooks(self) -> Dict[str, Dict]:
        """Collect orderbook data for all active symbols"""
        
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_collection_time < self.collection_interval:
            return {}
        
        orderbooks = {}
        
        for symbol in self.active_symbols:
            try:
                orderbook = self.fetcher.get_orderbook(symbol)
                if orderbook:
                    orderbooks[symbol] = orderbook
                
                # Small delay between requests
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error collecting orderbook for {symbol}: {e}")
                continue
        
        self.last_collection_time = current_time
        logger.debug(f"Collected orderbooks for {len(orderbooks)}/{len(self.active_symbols)} symbols")
        
        return orderbooks
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        
        ticker = self.fetcher.get_ticker(symbol)
        if ticker:
            try:
                return float(ticker.get('lastPrice', 0))
            except (ValueError, TypeError):
                return None
        
        return None

# Integration helper functions
def create_bybit_fetcher() -> BybitOrderbookFetcher:
    """Create Bybit fetcher with environment credentials"""
    return BybitOrderbookFetcher()

def test_bybit_connection() -> bool:
    """Test Bybit API connection"""
    fetcher = create_bybit_fetcher()
    return fetcher.test_connection()