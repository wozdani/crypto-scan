"""
TradingView Symbol Resolver - Automatic Exchange Prefix Detection
Resolves BYBIT: vs BINANCE: prefixes with caching for optimal performance
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

# Import centralized logging
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crypto_scan_service import log_warning
except ImportError:
    def log_warning(label, exception=None, additional_info=None):
        msg = f"[{label}]"
        if exception:
            msg += f" {str(exception)}"
        if additional_info:
            msg += f" - {additional_info}"
        print(f"⚠️ {msg}")

# Cache file for resolved symbols
SYMBOL_CACHE_FILE = Path("crypto-scan/data/cache/tradingview_symbol_fallbacks.json")

class TradingViewSymbolResolver:
    """
    Intelligent TradingView symbol resolver with caching
    Automatically detects which exchange prefix works for each symbol
    """
    
    def __init__(self):
        self.cache = self._load_cache()
        self.session_cache = {}  # In-memory cache for current session
        
    def _load_cache(self) -> Dict[str, Optional[str]]:
        """Load symbol resolution cache from disk"""
        try:
            if SYMBOL_CACHE_FILE.exists():
                with open(SYMBOL_CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                print(f"[SYMBOL RESOLVER] Loaded cache with {len(cache)} entries")
                return cache
            else:
                print("[SYMBOL RESOLVER] No cache found - creating new cache")
                return {}
        except Exception as e:
            log_warning("SYMBOL CACHE LOAD ERROR", e, "Failed to load TradingView symbol cache")
            return {}
    
    def _save_cache(self):
        """Save symbol resolution cache to disk"""
        try:
            # Ensure cache directory exists
            SYMBOL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata to cache
            cache_with_meta = {
                "last_updated": datetime.now().isoformat(),
                "total_symbols": len(self.cache),
                "symbols": self.cache
            }
            
            with open(SYMBOL_CACHE_FILE, 'w') as f:
                json.dump(cache_with_meta, f, indent=2)
            
            print(f"[SYMBOL RESOLVER] Cache saved with {len(self.cache)} symbols")
            
        except Exception as e:
            log_warning("SYMBOL CACHE SAVE ERROR", e, "Failed to save TradingView symbol cache")
    
    def resolve_symbol(self, symbol: str) -> Optional[str]:
        """
        Resolve TradingView symbol with automatic fallback
        
        Args:
            symbol: Base symbol (e.g., 'BTCUSDT', 'BANANAS31USDT')
            
        Returns:
            Resolved symbol with prefix (e.g., 'BINANCE:BTCUSDT') or None if not available
        """
        # Check session cache first (fastest)
        if symbol in self.session_cache:
            result = self.session_cache[symbol]
            print(f"[SYMBOL RESOLVER] 🚀 Session cache hit: {symbol} → {result}")
            return result
        
        # Check disk cache
        cache_symbols = self.cache.get("symbols", self.cache)
        if symbol in cache_symbols:
            result = cache_symbols[symbol]
            self.session_cache[symbol] = result  # Store in session cache
            print(f"[SYMBOL RESOLVER] 💾 Disk cache hit: {symbol} → {result}")
            return result
        
        # No cache - need to resolve
        print(f"[SYMBOL RESOLVER] 🔍 Resolving new symbol: {symbol}")
        resolved = self._resolve_symbol_live(symbol)
        
        # Cache the result
        if "symbols" in self.cache:
            self.cache["symbols"][symbol] = resolved
        else:
            self.cache[symbol] = resolved
            
        self.session_cache[symbol] = resolved
        self._save_cache()
        
        print(f"[SYMBOL RESOLVER] ✅ Resolved: {symbol} → {resolved}")
        return resolved
    
    def _resolve_symbol_live(self, symbol: str) -> Optional[str]:
        """
        Live resolution of TradingView symbol by testing different exchanges
        
        Args:
            symbol: Base symbol to test
            
        Returns:
            Working symbol with prefix or None
        """
        # Test exchange prefixes in order of preference
        exchange_prefixes = ["BYBIT", "BINANCE", "COINBASE", "KRAKEN"]
        
        for prefix in exchange_prefixes:
            test_symbol = f"{prefix}:{symbol.upper()}"
            
            try:
                if self._test_symbol_validity(test_symbol):
                    print(f"[SYMBOL RESOLVER] ✅ {test_symbol} is valid")
                    return test_symbol
                else:
                    print(f"[SYMBOL RESOLVER] ❌ {test_symbol} is invalid")
                    
            except Exception as e:
                log_warning("SYMBOL TEST ERROR", e, f"Error testing {test_symbol}")
                continue
        
        # No working symbol found
        print(f"[SYMBOL RESOLVER] ❌ No valid exchange found for {symbol}")
        return None
    
    def _test_symbol_validity(self, tv_symbol: str) -> bool:
        """
        Test if a TradingView symbol is valid using lightweight check
        
        Args:
            tv_symbol: Full TradingView symbol (e.g., 'BINANCE:BTCUSDT')
            
        Returns:
            True if symbol is valid
        """
        try:
            # 🎯 LIGHTWEIGHT CHECK: Use TradingView symbol API
            import requests
            
            # TradingView symbol info endpoint
            api_url = f"https://symbol-search.tradingview.com/symbol_search/?text={tv_symbol}&type=stock"
            
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Check if symbol exists in response
                if data and isinstance(data, list) and len(data) > 0:
                    for item in data:
                        if item.get('symbol') == tv_symbol:
                            return True
                
            return False
            
        except Exception as e:
            # Fallback: assume valid if can't test
            log_warning("SYMBOL VALIDITY TEST ERROR", e, f"Cannot test {tv_symbol} - assuming invalid")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring"""
        cache_symbols = self.cache.get("symbols", self.cache)
        
        valid_symbols = sum(1 for v in cache_symbols.values() if v is not None)
        invalid_symbols = sum(1 for v in cache_symbols.values() if v is None)
        
        return {
            "total_symbols": len(cache_symbols),
            "valid_symbols": valid_symbols,
            "invalid_symbols": invalid_symbols,
            "session_cache_size": len(self.session_cache),
            "cache_file_exists": SYMBOL_CACHE_FILE.exists()
        }
    
    def clear_cache(self):
        """Clear both disk and session cache"""
        self.cache = {}
        self.session_cache = {}
        try:
            if SYMBOL_CACHE_FILE.exists():
                SYMBOL_CACHE_FILE.unlink()
                print("[SYMBOL RESOLVER] Cache cleared")
        except Exception as e:
            log_warning("CACHE CLEAR ERROR", e, "Failed to clear symbol cache")

# Global resolver instance
_symbol_resolver = None

def get_symbol_resolver() -> TradingViewSymbolResolver:
    """Get global symbol resolver instance"""
    global _symbol_resolver
    if _symbol_resolver is None:
        _symbol_resolver = TradingViewSymbolResolver()
    return _symbol_resolver

def resolve_tradingview_symbol(symbol: str) -> Optional[str]:
    """
    Convenience function to resolve TradingView symbol
    
    Args:
        symbol: Base symbol (e.g., 'BTCUSDT')
        
    Returns:
        Resolved symbol with prefix or None
    """
    resolver = get_symbol_resolver()
    return resolver.resolve_symbol(symbol)

def get_resolver_stats() -> Dict:
    """Get resolver cache statistics"""
    resolver = get_symbol_resolver()
    return resolver.get_cache_stats()

def main():
    """Test symbol resolver"""
    test_symbols = ["BTCUSDT", "ETHUSDT", "BANANAS31USDT", "XYZABCUSDT"]
    
    resolver = get_symbol_resolver()
    
    print("Testing TradingView Symbol Resolver:")
    for symbol in test_symbols:
        resolved = resolver.resolve_symbol(symbol)
        print(f"  {symbol} → {resolved}")
    
    print(f"\nCache stats: {resolver.get_cache_stats()}")

if __name__ == "__main__":
    main()