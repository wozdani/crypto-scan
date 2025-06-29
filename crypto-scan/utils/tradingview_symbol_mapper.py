"""
TradingView Symbol Mapper
Maps crypto symbols to valid TradingView exchanges with proper validation
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Add global error logging
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crypto_scan_service import log_warning
except ImportError:
    def log_warning(label, exception=None, additional_info=None):
        print(f"⚠️ [{label}] {exception} - {additional_info}" if exception else f"⚠️ [{label}] {additional_info}")

class TradingViewSymbolMapper:
    """Maps symbols to valid TradingView exchanges"""
    
    def __init__(self):
        self.cache_file = "data/tradingview_symbol_cache.json"
        self.cache_duration = 24 * 3600  # 24 hours
        self.cache = self._load_cache()
        
        # Exchange priority order for TradingView
        self.exchange_priority = [
            "BINANCE",     # Primary - highest liquidity
            "COINBASE",    # US market
            "BYBIT",       # Sometimes available
            "MEXC",        # Good coverage
            "GATEIO",      # Alternative
            "KUCOIN",      # Backup
            "OKX"          # Last resort
        ]
        
        # Common symbol mappings
        self.symbol_fixes = {
            "1000SATSUSDT": "1000SATS/USDT",
            "1000PEPEUSDT": "1000PEPE/USDT", 
            "1000FLOKIUSDT": "1000FLOKI/USDT",
            "1000BONKUSDT": "1000BONK/USDT",
            # Add more as needed
        }
    
    def _load_cache(self) -> Dict:
        """Load symbol mapping cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                
                # Check cache age
                cache_time = cache.get('timestamp', 0)
                if time.time() - cache_time < self.cache_duration:
                    return cache.get('mappings', {})
                    
        except Exception as e:
            log_warning("SYMBOL CACHE", e, "Failed to load cache")
        
        return {}
    
    def _save_cache(self):
        """Save symbol mapping cache"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache_data = {
                'timestamp': time.time(),
                'mappings': self.cache
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            log_warning("SYMBOL CACHE", e, "Failed to save cache")
    
    def verify_tradingview_symbol(self, symbol: str, exchange: str) -> bool:
        """Verify if symbol exists on TradingView"""
        try:
            # Quick check - try to access TradingView symbol page
            test_url = f"https://www.tradingview.com/symbols/{exchange}-{symbol}/"
            
            response = requests.head(test_url, timeout=5, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            # Status 200 = symbol exists, 404 = doesn't exist
            return response.status_code == 200
            
        except Exception as e:
            log_warning("SYMBOL VERIFY", e, f"Failed to verify {exchange}:{symbol}")
            return False
    
    def map_symbol_to_tradingview(self, symbol: str) -> Optional[str]:
        """Map symbol to best available TradingView exchange"""
        
        # Check cache first
        cache_key = symbol.upper()
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if cached_result['timestamp'] > time.time() - self.cache_duration:
                return cached_result.get('tv_symbol')
        
        # Clean symbol
        clean_symbol = symbol.upper().replace("USDT", "/USDT")
        
        # Apply symbol fixes
        if symbol in self.symbol_fixes:
            clean_symbol = self.symbol_fixes[symbol]
        
        # Try each exchange in priority order
        for exchange in self.exchange_priority:
            test_symbol = f"{exchange}:{clean_symbol}"
            
            print(f"[TV SYMBOL] Testing {test_symbol}...")
            
            if self.verify_tradingview_symbol(clean_symbol.replace("/", ""), exchange):
                print(f"[TV SYMBOL] ✅ Found: {test_symbol}")
                
                # Cache the result
                self.cache[cache_key] = {
                    'tv_symbol': test_symbol,
                    'timestamp': time.time(),
                    'exchange': exchange
                }
                self._save_cache()
                
                return test_symbol
            else:
                print(f"[TV SYMBOL] ❌ Not found: {test_symbol}")
        
        # No valid mapping found
        print(f"[TV SYMBOL] ⚠️ No valid TradingView mapping for {symbol}")
        
        # Cache negative result
        self.cache[cache_key] = {
            'tv_symbol': None,
            'timestamp': time.time(),
            'exchange': None
        }
        self._save_cache()
        
        return None
    
    def get_tradingview_url(self, symbol: str, timeframe: str = "15") -> Optional[str]:
        """Get complete TradingView chart URL"""
        
        tv_symbol = self.map_symbol_to_tradingview(symbol)
        if not tv_symbol:
            return None
        
        # Create TradingView chart URL
        url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}&interval={timeframe}"
        return url
    
    def batch_validate_symbols(self, symbols: List[str]) -> Dict[str, Optional[str]]:
        """Validate multiple symbols at once"""
        results = {}
        
        print(f"[TV BATCH] Validating {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                print(f"[TV BATCH] Progress: {i}/{len(symbols)}")
            
            results[symbol] = self.map_symbol_to_tradingview(symbol)
            
            # Rate limiting
            time.sleep(0.1)
        
        valid_count = sum(1 for v in results.values() if v is not None)
        print(f"[TV BATCH] Results: {valid_count}/{len(symbols)} symbols valid")
        
        return results

# Global mapper instance
_symbol_mapper = None

def get_symbol_mapper() -> TradingViewSymbolMapper:
    """Get global symbol mapper instance"""
    global _symbol_mapper
    if _symbol_mapper is None:
        _symbol_mapper = TradingViewSymbolMapper()
    return _symbol_mapper

def map_to_tradingview(symbol: str) -> Optional[str]:
    """Convenience function to map symbol"""
    mapper = get_symbol_mapper()
    return mapper.map_symbol_to_tradingview(symbol)

def get_tradingview_chart_url(symbol: str, timeframe: str = "15") -> Optional[str]:
    """Convenience function to get chart URL"""
    mapper = get_symbol_mapper()
    return mapper.get_tradingview_url(symbol, timeframe)

def test_symbol_mapping():
    """Test symbol mapping functionality"""
    test_symbols = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", 
        "HUSDT", "AVLUSDT", "SQDUSDT"  # These should fail
    ]
    
    mapper = get_symbol_mapper()
    results = mapper.batch_validate_symbols(test_symbols)
    
    print("\n[TV TEST] Symbol Mapping Results:")
    for symbol, tv_symbol in results.items():
        status = "✅ VALID" if tv_symbol else "❌ INVALID"
        print(f"  {symbol} → {tv_symbol or 'NOT FOUND'} {status}")

if __name__ == "__main__":
    test_symbol_mapping()