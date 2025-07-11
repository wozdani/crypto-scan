"""
Enhanced Data Validation System with Fallback and Partial Validity
Resolves ticker/orderbook None issues while preserving candle data
"""

import requests
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from utils.enhanced_error_logging import log_api_error, log_data_validation_error, log_processing_error

class DataValidationResult:
    """Structured result for data validation with partial validity support"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.is_valid = False
        self.is_partial = False
        self.ticker_data: Optional[Dict] = None
        self.candles_15m: List[Dict] = []
        self.candles_5m: List[Dict] = []
        self.orderbook_data: Optional[Dict] = None
        self.price_usd = 0.0
        self.validation_issues = []
        self.fallback_used = []
        self.api_calls_made = []
    
    def add_issue(self, component: str, reason: str):
        """Add validation issue"""
        self.validation_issues.append(f"{component}: {reason}")
    
    def add_fallback(self, method: str):
        """Record fallback method used"""
        self.fallback_used.append(method)
    
    def add_api_call(self, endpoint: str, status_code: int, success: bool):
        """Record API call attempt"""
        self.api_calls_made.append({
            "endpoint": endpoint,
            "status_code": status_code,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
    
    def calculate_validity(self):
        """Calculate overall validity status - Enhanced for partial validity"""
        has_candles = len(self.candles_15m) > 0 or len(self.candles_5m) > 0
        has_ticker = self.ticker_data is not None
        has_orderbook = self.orderbook_data is not None
        has_price = self.price_usd > 0
        
        # Complete validity: All components available
        if has_ticker and has_candles and has_orderbook and has_price:
            self.is_valid = True
            self.is_partial = False
        # Enhanced partial validity: Accept tokens with valid candles even without ticker
        elif has_candles:
            # If we have candles but no price, try to extract it
            if not has_price:
                self.price_usd = self._extract_price_from_candles()
                has_price = self.price_usd > 0
            
            # Accept any token with candles and valid price
            if has_price:
                self.is_valid = True
                self.is_partial = True
            else:
                # Accept candles even without price for processing
                self.is_valid = True
                self.is_partial = True
                print(f"[VALIDATION] {self.symbol}: Accepting with candles only (no price)")
        # Ticker-only fallback
        elif has_ticker and has_price:
            self.is_valid = True
            self.is_partial = True
        else:
            self.is_valid = False
            self.is_partial = False
    
    def _extract_price_from_candles(self) -> float:
        """Extract price from candle data when ticker is unavailable"""
        # Try 15m candles first
        if self.candles_15m:
            try:
                latest_candle = self.candles_15m[0]
                price = latest_candle.get("close", 0)
                if price > 0:
                    print(f"[PRICE EXTRACT] {self.symbol}: ${float(price)} from 15m candles")
                    return float(price)
            except (ValueError, TypeError, IndexError):
                pass
        
        # Try 5m candles as fallback
        if self.candles_5m:
            try:
                latest_candle = self.candles_5m[0]
                price = latest_candle.get("close", 0)
                if price > 0:
                    print(f"[PRICE EXTRACT] {self.symbol}: ${float(price)} from 5m candles")
                    return float(price)
            except (ValueError, TypeError, IndexError):
                pass
        
        return 0.0
    
    def get_summary(self) -> str:
        """Get validation summary"""
        status = "VALID" if self.is_valid else "INVALID"
        if self.is_partial:
            status += " (PARTIAL)"
        
        components = []
        if self.ticker_data: components.append("ticker")
        if self.candles_15m: components.append("15m_candles")
        if self.candles_5m: components.append("5m_candles")
        if self.orderbook_data: components.append("orderbook")
        
        return f"{status} - Components: {', '.join(components) if components else 'none'}"

class EnhancedDataValidator:
    """Enhanced data validator with comprehensive fallback mechanisms"""
    
    def __init__(self):
        self.ticker_cache = {}  # 5-minute TTL cache
        self.orderbook_cache = {}  # 1-minute TTL cache
        self.failed_symbols = set()  # Session-level failed symbols
        
    def validate_market_data(self, symbol: str, retry_count: int = 2) -> DataValidationResult:
        """
        Comprehensive market data validation with fallback mechanisms
        
        Args:
            symbol: Trading symbol to validate
            retry_count: Number of retry attempts for failed requests
            
        Returns:
            DataValidationResult with complete validation status
        """
        result = DataValidationResult(symbol)
        
        print(f"[DATA VALIDATION START] {symbol}")
        
        # STEP 1: Validate ticker data with fallback
        result.ticker_data = self._get_ticker_with_fallback(symbol, result, retry_count)
        
        # STEP 2: Validate candle data (both 15m and 5m)
        result.candles_15m = self._get_candles_with_fallback(symbol, "15", result, retry_count)
        result.candles_5m = self._get_candles_with_fallback(symbol, "5", result, retry_count)
        
        # STEP 3: Validate orderbook data with fallback
        result.orderbook_data = self._get_orderbook_with_fallback(symbol, result, retry_count)
        
        # STEP 4: Extract price from available data
        result.price_usd = self._extract_price(result)
        
        # STEP 5: Calculate overall validity
        result.calculate_validity()
        
        # STEP 6: Enhanced debugging output
        self._debug_validation_result(result)
        
        return result
    
    def _get_ticker_with_fallback(self, symbol: str, result: DataValidationResult, retry_count: int) -> Optional[Dict]:
        """Get ticker data with comprehensive fallback mechanisms"""
        
        # Check cache first (5-minute TTL)
        cache_key = f"ticker_{symbol}"
        if cache_key in self.ticker_cache:
            cache_entry = self.ticker_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < 300:  # 5 minutes
                print(f"[TICKER CACHE] {symbol}: Using cached ticker data")
                result.add_fallback("ticker_cache")
                return cache_entry["data"]
        
        # FALLBACK 1: Standard tickers endpoint
        ticker_data = self._fetch_ticker_standard(symbol, result, retry_count)
        if ticker_data:
            self._cache_ticker(symbol, ticker_data)
            return ticker_data
        
        # FALLBACK 2: Individual symbol ticker endpoint
        ticker_data = self._fetch_ticker_individual(symbol, result, retry_count)
        if ticker_data:
            self._cache_ticker(symbol, ticker_data)
            return ticker_data
        
        # FALLBACK 3: Extract from candle data if available
        if result.candles_15m:
            ticker_data = self._extract_ticker_from_candles(symbol, result.candles_15m, result)
            if ticker_data:
                result.add_fallback("ticker_from_candles")
                return ticker_data
        
        result.add_issue("ticker", "All fallback methods failed")
        return None
    
    def _fetch_ticker_standard(self, symbol: str, result: DataValidationResult, retry_count: int) -> Optional[Dict]:
        """Fetch ticker from standard tickers endpoint"""
        url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
        
        for attempt in range(retry_count + 1):
            try:
                response = requests.get(url, timeout=10)
                result.add_api_call("tickers", response.status_code, response.status_code == 200)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("result", {}).get("list"):
                        ticker = data["result"]["list"][0]
                        if ticker.get("lastPrice") and float(ticker["lastPrice"]) > 0:
                            print(f"[TICKER OK] {symbol}: Standard endpoint - Price ${ticker['lastPrice']}")
                            return ticker
                
                print(f"[TICKER ERROR] {symbol}: Standard endpoint failed - HTTP {response.status_code}")
                if attempt < retry_count:
                    time.sleep(0.5)
                    
            except Exception as e:
                log_api_error(symbol, "tickers_standard", "request_failed", None, e)
                if attempt < retry_count:
                    time.sleep(0.5)
        
        return None
    
    def _fetch_ticker_individual(self, symbol: str, result: DataValidationResult, retry_count: int) -> Optional[Dict]:
        """Fetch ticker from individual symbol endpoint"""
        url = f"https://api.bybit.com/v5/market/ticker?category=spot&symbol={symbol}"
        
        for attempt in range(retry_count + 1):
            try:
                response = requests.get(url, timeout=10)
                result.add_api_call("ticker_individual", response.status_code, response.status_code == 200)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("result") and data["result"].get("lastPrice"):
                        ticker = data["result"]
                        if float(ticker["lastPrice"]) > 0:
                            print(f"[TICKER OK] {symbol}: Individual endpoint - Price ${ticker['lastPrice']}")
                            result.add_fallback("ticker_individual")
                            return ticker
                
                print(f"[TICKER ERROR] {symbol}: Individual endpoint failed - HTTP {response.status_code}")
                
            except Exception as e:
                log_api_error(symbol, "ticker_individual", "request_failed", None, e)
                if attempt < retry_count:
                    time.sleep(0.5)
        
        return None
    
    def _extract_ticker_from_candles(self, symbol: str, candles: List[Dict], result: DataValidationResult) -> Optional[Dict]:
        """Extract ticker-like data from candle data"""
        if not candles:
            return None
        
        try:
            latest_candle = candles[0]  # Most recent candle
            price = latest_candle.get("close", 0)
            
            if price > 0:
                synthetic_ticker = {
                    "lastPrice": str(price),
                    "volume24h": str(sum(c.get("volume", 0) for c in candles[:24])),  # Approximate 24h volume
                    "highPrice24h": str(max(c.get("high", 0) for c in candles[:24])),
                    "lowPrice24h": str(min(c.get("low", 0) for c in candles[:24])),
                    "price24hPcnt": "0",  # Cannot calculate without older data
                    "synthetic": True
                }
                print(f"[TICKER SYNTHETIC] {symbol}: Extracted from candles - Price ${price}")
                return synthetic_ticker
                
        except Exception as e:
            log_processing_error(symbol, "ticker_extraction", {"error": str(e)}, e)
        
        return None
    
    def _get_candles_with_fallback(self, symbol: str, interval: str, result: DataValidationResult, retry_count: int) -> List[Dict]:
        """Get candle data with fallback mechanisms"""
        
        # FALLBACK 1: Standard kline endpoint
        candles = self._fetch_candles_standard(symbol, interval, result, retry_count)
        if candles:
            return candles
        
        # FALLBACK 2: Linear category endpoint
        candles = self._fetch_candles_linear(symbol, interval, result, retry_count)
        if candles:
            result.add_fallback(f"candles_{interval}m_linear")
            return candles
        
        result.add_issue(f"candles_{interval}m", "All candle endpoints failed")
        return []
    
    def _fetch_candles_standard(self, symbol: str, interval: str, result: DataValidationResult, retry_count: int) -> List[Dict]:
        """Fetch candles from standard endpoint"""
        url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={interval}&limit=100"
        
        for attempt in range(retry_count + 1):
            try:
                response = requests.get(url, timeout=10)
                result.add_api_call(f"kline_{interval}m", response.status_code, response.status_code == 200)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("result", {}).get("list"):
                        candles_raw = data["result"]["list"]
                        candles = []
                        
                        for candle in candles_raw:
                            try:
                                candles.append({
                                    "timestamp": int(candle[0]),
                                    "open": float(candle[1]),
                                    "high": float(candle[2]),
                                    "low": float(candle[3]),
                                    "close": float(candle[4]),
                                    "volume": float(candle[5])
                                })
                            except (ValueError, IndexError) as e:
                                continue
                        
                        if candles:
                            print(f"[CANDLES OK] {symbol}: Got {len(candles)} {interval}m candles")
                            return candles
                
                print(f"[CANDLES ERROR] {symbol}: {interval}m endpoint failed - HTTP {response.status_code}")
                
            except Exception as e:
                log_api_error(symbol, f"kline_{interval}m", "request_failed", None, e)
                if attempt < retry_count:
                    time.sleep(0.5)
        
        return []
    
    def _fetch_candles_linear(self, symbol: str, interval: str, result: DataValidationResult, retry_count: int) -> List[Dict]:
        """Fetch candles from linear category endpoint"""
        url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit=100"
        
        try:
            response = requests.get(url, timeout=10)
            result.add_api_call(f"kline_{interval}m_linear", response.status_code, response.status_code == 200)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("result", {}).get("list"):
                    candles_raw = data["result"]["list"]
                    candles = []
                    
                    for candle in candles_raw:
                        try:
                            candles.append({
                                "timestamp": int(candle[0]),
                                "open": float(candle[1]),
                                "high": float(candle[2]),
                                "low": float(candle[3]),
                                "close": float(candle[4]),
                                "volume": float(candle[5])
                            })
                        except (ValueError, IndexError):
                            continue
                    
                    if candles:
                        print(f"[CANDLES LINEAR OK] {symbol}: Got {len(candles)} {interval}m candles from linear")
                        return candles
                        
        except Exception as e:
            log_api_error(symbol, f"kline_{interval}m_linear", "request_failed", None, e)
        
        return []
    
    def _get_orderbook_with_fallback(self, symbol: str, result: DataValidationResult, retry_count: int) -> Optional[Dict]:
        """Get orderbook data with fallback mechanisms"""
        
        # Check cache first (1-minute TTL)
        cache_key = f"orderbook_{symbol}"
        if cache_key in self.orderbook_cache:
            cache_entry = self.orderbook_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < 60:  # 1 minute
                print(f"[ORDERBOOK CACHE] {symbol}: Using cached orderbook data")
                result.add_fallback("orderbook_cache")
                return cache_entry["data"]
        
        # FALLBACK 1: Standard orderbook endpoint
        orderbook = self._fetch_orderbook_standard(symbol, result, retry_count)
        if orderbook:
            self._cache_orderbook(symbol, orderbook)
            return orderbook
        
        # FALLBACK 2: Reduced limit orderbook
        orderbook = self._fetch_orderbook_reduced(symbol, result, retry_count)
        if orderbook:
            self._cache_orderbook(symbol, orderbook)
            result.add_fallback("orderbook_reduced")
            return orderbook
        
        # FALLBACK 3: Synthetic orderbook from ticker
        if result.ticker_data:
            orderbook = self._create_synthetic_orderbook(symbol, result.ticker_data, result)
            if orderbook:
                result.add_fallback("orderbook_synthetic")
                return orderbook
        
        result.add_issue("orderbook", "All orderbook methods failed")
        return None
    
    def _fetch_orderbook_standard(self, symbol: str, result: DataValidationResult, retry_count: int) -> Optional[Dict]:
        """Fetch orderbook from standard endpoint - ENHANCED DEPTH"""
        url = f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit=200"
        
        for attempt in range(retry_count + 1):
            try:
                response = requests.get(url, timeout=10)
                result.add_api_call("orderbook", response.status_code, response.status_code == 200)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("result"):
                        ob_result = data["result"]
                        bids = ob_result.get("b", [])
                        asks = ob_result.get("a", [])
                        
                        if bids and asks:
                            orderbook = {
                                "bids": [[float(bid[0]), float(bid[1])] for bid in bids],
                                "asks": [[float(ask[0]), float(ask[1])] for ask in asks]
                            }
                            print(f"[ORDERBOOK OK] {symbol}: Got {len(bids)} bids, {len(asks)} asks")
                            return orderbook
                
                print(f"[ORDERBOOK ERROR] {symbol}: Standard endpoint failed - HTTP {response.status_code}")
                if attempt < retry_count:
                    time.sleep(0.5)
                    
            except Exception as e:
                log_api_error(symbol, "orderbook", "request_failed", None, e)
                if attempt < retry_count:
                    time.sleep(0.5)
        
        return None
    
    def _fetch_orderbook_reduced(self, symbol: str, result: DataValidationResult, retry_count: int) -> Optional[Dict]:
        """Fetch orderbook with reduced limit"""
        url = f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit=50"
        
        try:
            response = requests.get(url, timeout=10)
            result.add_api_call("orderbook_reduced", response.status_code, response.status_code == 200)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("result"):
                    ob_result = data["result"]
                    bids = ob_result.get("b", [])
                    asks = ob_result.get("a", [])
                    
                    if bids and asks:
                        orderbook = {
                            "bids": [[float(bid[0]), float(bid[1])] for bid in bids],
                            "asks": [[float(ask[0]), float(ask[1])] for ask in asks]
                        }
                        print(f"[ORDERBOOK REDUCED OK] {symbol}: Got {len(bids)} bids, {len(asks)} asks")
                        return orderbook
                        
        except Exception as e:
            log_api_error(symbol, "orderbook_reduced", "request_failed", None, e)
        
        return None
    
    def _create_synthetic_orderbook(self, symbol: str, ticker_data: Dict, result: DataValidationResult) -> Optional[Dict]:
        """Create synthetic orderbook from ticker data"""
        try:
            last_price = float(ticker_data.get("lastPrice", 0))
            if last_price <= 0:
                return None
            
            # Create synthetic spread (0.1%)
            spread = last_price * 0.001
            bid_price = last_price - spread
            ask_price = last_price + spread
            
            # Create MULTI-LEVEL synthetic orderbook for better Stealth Engine analysis
            bids_levels = []
            asks_levels = []
            for i in range(10):  # Create 10 levels each side
                bid_level_price = bid_price - (spread * i * 0.5)
                ask_level_price = ask_price + (spread * i * 0.5) 
                level_size = 100.0 / (1 + i * 0.1)  # Decreasing size
                bids_levels.append([bid_level_price, level_size])
                asks_levels.append([ask_level_price, level_size])
            
            synthetic_orderbook = {
                "bids": bids_levels,
                "asks": asks_levels,
                "synthetic": True
            }
            
            print(f"[ORDERBOOK SYNTHETIC] {symbol}: Created from ticker - Bid ${bid_price:.4f}, Ask ${ask_price:.4f}")
            return synthetic_orderbook
            
        except Exception as e:
            log_processing_error(symbol, "synthetic_orderbook", {"error": str(e)}, e)
        
        return None
    
    def _extract_price(self, result: DataValidationResult) -> float:
        """Extract price from available data sources"""
        # Priority 1: Ticker data
        if result.ticker_data and result.ticker_data.get("lastPrice"):
            try:
                price = float(result.ticker_data["lastPrice"])
                if price > 0:
                    return price
            except (ValueError, TypeError):
                pass
        
        # Priority 2: Latest candle close price
        if result.candles_15m:
            try:
                price = result.candles_15m[0].get("close", 0)
                if price > 0:
                    return float(price)
            except (ValueError, TypeError, IndexError):
                pass
        
        if result.candles_5m:
            try:
                price = result.candles_5m[0].get("close", 0)
                if price > 0:
                    return float(price)
            except (ValueError, TypeError, IndexError):
                pass
        
        # Priority 3: Orderbook mid price
        if result.orderbook_data:
            try:
                bids = result.orderbook_data.get("bids", [])
                asks = result.orderbook_data.get("asks", [])
                if bids and asks:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    mid_price = (best_bid + best_ask) / 2
                    if mid_price > 0:
                        return mid_price
            except (ValueError, TypeError, IndexError):
                pass
        
        return 0.0
    
    def _cache_ticker(self, symbol: str, ticker_data: Dict):
        """Cache ticker data with TTL"""
        self.ticker_cache[f"ticker_{symbol}"] = {
            "data": ticker_data,
            "timestamp": time.time()
        }
    
    def _cache_orderbook(self, symbol: str, orderbook_data: Dict):
        """Cache orderbook data with TTL"""
        self.orderbook_cache[f"orderbook_{symbol}"] = {
            "data": orderbook_data,
            "timestamp": time.time()
        }
    
    def _debug_validation_result(self, result: DataValidationResult):
        """Enhanced debugging output for validation results"""
        components = []
        if result.ticker_data: components.append("ticker")
        if result.candles_15m: components.append(f"15m({len(result.candles_15m)})")
        if result.candles_5m: components.append(f"5m({len(result.candles_5m)})")
        if result.orderbook_data: components.append("orderbook")
        
        print(f"[DATA VALIDATION] {result.symbol} → {', '.join(components) if components else 'no_data'}")
        
        if result.validation_issues:
            for issue in result.validation_issues:
                print(f"[DATA ISSUE] {result.symbol} → {issue}")
        
        if result.fallback_used:
            print(f"[DATA FALLBACK] {result.symbol} → Used: {', '.join(result.fallback_used)}")
        
        if result.is_valid:
            status = "PARTIAL" if result.is_partial else "COMPLETE"
            print(f"[DATA VALIDATION SUCCESS] {result.symbol} → {status} validity, Price: ${result.price_usd}")
        else:
            print(f"[DATA VALIDATION FAILED] {result.symbol} → No valid data sources")

# Global validator instance
_data_validator = None

def get_data_validator() -> EnhancedDataValidator:
    """Get global data validator instance"""
    global _data_validator
    if _data_validator is None:
        _data_validator = EnhancedDataValidator()
    return _data_validator

def validate_market_data_enhanced(symbol: str) -> DataValidationResult:
    """
    Enhanced market data validation with comprehensive fallback mechanisms
    
    Args:
        symbol: Trading symbol to validate
        
    Returns:
        DataValidationResult with complete validation status and partial validity support
    """
    validator = get_data_validator()
    return validator.validate_market_data(symbol)

def main():
    """Test enhanced data validation system"""
    print("Testing Enhanced Data Validation System...")
    
    # Test with problematic symbols from user's report
    test_symbols = ["SATSUSDT", "SCRUSDT", "SEIUSDT", "SHIBUSDT", "SENDUSDT"]
    
    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")
        result = validate_market_data_enhanced(symbol)
        print(f"Result: {result.get_summary()}")
        if result.validation_issues:
            for issue in result.validation_issues:
                print(f"Issue: {issue}")

if __name__ == "__main__":
    main()