"""
BINANCE Symbol Filter for TradingView Compatibility
Filters tokens to only process those available on BINANCE exchange
"""

import requests
import json
import os
from typing import List, Set, Dict, Optional
from debug_config import log_debug

class BinanceSymbolFilter:
    """Filter system for BINANCE-compatible symbols only"""
    
    def __init__(self):
        self.cache_file = "data/binance_symbols_cache.json"
        self.binance_symbols: Set[str] = set()
        self.cache_loaded = False
        
    def load_binance_symbols(self) -> bool:
        """Load available BINANCE symbols from API or cache"""
        try:
            # Try to load from cache first
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Check cache age (refresh every 24 hours)
                import time
                cache_age = time.time() - cache_data.get('timestamp', 0)
                if cache_age < 86400:  # 24 hours
                    self.binance_symbols = set(cache_data.get('symbols', []))
                    self.cache_loaded = True
                    log_debug(f"[BINANCE FILTER] Loaded {len(self.binance_symbols)} symbols from cache")
                    return True
            
            # Fetch fresh data from Binance API
            log_debug("[BINANCE FILTER] Fetching fresh symbol list from Binance API...")
            response = requests.get(
                'https://api.binance.com/api/v3/exchangeInfo',
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                symbols = []
                
                for symbol_info in data.get('symbols', []):
                    symbol = symbol_info.get('symbol', '')
                    status = symbol_info.get('status', '')
                    
                    # Only include active USDT pairs
                    if status == 'TRADING' and symbol.endswith('USDT'):
                        symbols.append(symbol)
                
                self.binance_symbols = set(symbols)
                
                # Save to cache
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                with open(self.cache_file, 'w') as f:
                    json.dump({
                        'symbols': list(self.binance_symbols),
                        'timestamp': time.time(),
                        'count': len(self.binance_symbols)
                    }, f, indent=2)
                
                self.cache_loaded = True
                log_debug(f"[BINANCE FILTER] Loaded {len(self.binance_symbols)} USDT symbols from Binance API")
                return True
                
        except Exception as e:
            log_debug(f"[BINANCE FILTER ERROR] Failed to load symbols: {e}")
            
        # Fallback to common symbols if API fails
        self._load_fallback_symbols()
        return False
    
    def _load_fallback_symbols(self):
        """Load fallback list of common BINANCE symbols"""
        common_symbols = {
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
            'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'SHIBUSDT',
            'LTCUSDT', 'LINKUSDT', 'BCHUSDT', 'XLMUSDT', 'UNIUSDT',
            'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT', 'TRXUSDT',
            'ETCUSDT', 'XLMUSDT', 'ATOMUSDT', 'HBARUSDT', 'NEARUSDT',
            'FTMUSDT', 'MANAUSDT', 'SANDUSDT', 'AXSUSDT', 'THETAUSDT',
            'GRTUSDT', 'ENJUSDT', 'CHZUSDT', 'MKRUSDT', 'COMPUSDT'
        }
        self.binance_symbols = common_symbols
        self.cache_loaded = True
        log_debug(f"[BINANCE FILTER] Using fallback symbols: {len(common_symbols)} tokens")
    
    def is_binance_symbol(self, symbol: str) -> bool:
        """Check if symbol is available on BINANCE"""
        if not self.cache_loaded:
            self.load_binance_symbols()
            
        return symbol.upper() in self.binance_symbols
    
    def filter_symbols(self, symbols: List[str]) -> List[str]:
        """Filter symbol list to only BINANCE-compatible tokens"""
        if not self.cache_loaded:
            self.load_binance_symbols()
            
        filtered = []
        skipped = []
        
        for symbol in symbols:
            if self.is_binance_symbol(symbol):
                filtered.append(symbol)
            else:
                skipped.append(symbol)
        
        log_debug(f"[BINANCE FILTER] Filtered {len(symbols)} → {len(filtered)} symbols")
        if skipped:
            log_debug(f"[BINANCE FILTER] Skipped non-BINANCE: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
            
        return filtered
    
    def get_filter_stats(self) -> Dict[str, int]:
        """Get filtering statistics"""
        return {
            'total_binance_symbols': len(self.binance_symbols),
            'cache_loaded': self.cache_loaded,
            'cache_exists': os.path.exists(self.cache_file)
        }

# Global instance
_binance_filter = None

def get_binance_filter() -> BinanceSymbolFilter:
    """Get global BINANCE filter instance"""
    global _binance_filter
    if _binance_filter is None:
        _binance_filter = BinanceSymbolFilter()
    return _binance_filter

def filter_binance_symbols(symbols: List[str]) -> List[str]:
    """Convenience function to filter symbols for BINANCE compatibility"""
    return get_binance_filter().filter_symbols(symbols)

def is_binance_compatible(symbol: str) -> bool:
    """Check if single symbol is BINANCE compatible"""
    return get_binance_filter().is_binance_symbol(symbol)

def main():
    """Test BINANCE symbol filtering"""
    filter_obj = get_binance_filter()
    
    # Test symbols
    test_symbols = [
        'BTCUSDT', 'ETHUSDT', 'BANANAS31USDT', 
        'ZEUSUSDT', 'PROMUSDT', 'ADAUSDT'
    ]
    
    print("🔍 Testing BINANCE Symbol Filter:")
    for symbol in test_symbols:
        compatible = filter_obj.is_binance_symbol(symbol)
        print(f"  {symbol}: {'✅ BINANCE' if compatible else '❌ NOT BINANCE'}")
    
    # Test filtering
    filtered = filter_obj.filter_symbols(test_symbols)
    print(f"\n📊 Filter Results:")
    print(f"  Original: {len(test_symbols)} symbols")
    print(f"  Filtered: {len(filtered)} symbols")
    print(f"  BINANCE-compatible: {filtered}")

if __name__ == "__main__":
    main()