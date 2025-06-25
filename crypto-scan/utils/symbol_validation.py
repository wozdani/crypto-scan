"""
Symbol Validation and Caching System
Prevents unnecessary API calls for non-existent or delisted tokens
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

class SymbolValidator:
    """Validates symbols and caches results to prevent repeated failed API calls"""
    
    def __init__(self, cache_file: str = "data/symbol_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.session_validated = set()  # In-memory cache for current session
        
    def _load_cache(self) -> Dict:
        """Load symbol validation cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    print(f"[SYMBOL CACHE] Loaded {len(cache.get('invalid', []))} invalid symbols")
                    return cache
        except Exception as e:
            print(f"[SYMBOL CACHE ERROR] Failed to load cache: {e}")
        
        return {
            "invalid": [],
            "valid": [],
            "last_updated": datetime.now().isoformat(),
            "stats": {"api_calls_saved": 0, "validation_hits": 0}
        }
    
    def _save_cache(self):
        """Save symbol validation cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            self.cache["last_updated"] = datetime.now().isoformat()
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[SYMBOL CACHE ERROR] Failed to save cache: {e}")
    
    def is_symbol_valid(self, symbol: str) -> Optional[bool]:
        """
        Check if symbol is valid (exists on exchange)
        
        Returns:
            True: Symbol is valid
            False: Symbol is invalid (cached)
            None: Unknown - needs validation
        """
        # Check session cache first (fastest)
        if symbol in self.session_validated:
            return True
            
        # Check invalid cache (24h TTL)
        if symbol in self.cache.get("invalid", []):
            self.cache["stats"]["validation_hits"] += 1
            print(f"[SYMBOL SKIP] {symbol}: Cached as invalid - skipping API calls")
            return False
        
        # Check valid cache (less critical, 1h TTL)
        if symbol in self.cache.get("valid", []):
            self.session_validated.add(symbol)
            return True
            
        return None  # Needs validation
    
    def mark_symbol_invalid(self, symbol: str, reason: str = "API error"):
        """Mark symbol as invalid and cache for 24h"""
        if symbol not in self.cache.get("invalid", []):
            self.cache.setdefault("invalid", []).append(symbol)
            self.cache["stats"]["api_calls_saved"] += 3  # Saves 15m, 5m, orderbook calls
            print(f"[SYMBOL INVALID] {symbol}: Marked as invalid ({reason}) - will skip for 24h")
            self._save_cache()
    
    def mark_symbol_valid(self, symbol: str):
        """Mark symbol as valid for current session"""
        self.session_validated.add(symbol)
        if symbol not in self.cache.get("valid", []):
            self.cache.setdefault("valid", []).append(symbol)
            # Remove from invalid if present
            if symbol in self.cache.get("invalid", []):
                self.cache["invalid"].remove(symbol)
                print(f"[SYMBOL RECOVERED] {symbol}: Moved from invalid to valid cache")
            self._save_cache()
    
    def cleanup_old_cache(self):
        """Remove old cached entries (24h for invalid, 1h for valid)"""
        current_time = datetime.now()
        try:
            last_updated = datetime.fromisoformat(self.cache.get("last_updated", current_time.isoformat()))
            
            old_invalid_count = 0
            old_valid_count = 0
            
            # Clean invalid cache after 24h
            if current_time - last_updated > timedelta(hours=24):
                old_invalid_count = len(self.cache.get("invalid", []))
                self.cache["invalid"] = []
                print(f"[SYMBOL CLEANUP] Cleared {old_invalid_count} invalid symbols after 24h")
            
            # Clean valid cache after 1h
            if current_time - last_updated > timedelta(hours=1):
                old_valid_count = len(self.cache.get("valid", []))
                self.cache["valid"] = []
                print(f"[SYMBOL CLEANUP] Cleared {old_valid_count} valid symbols after 1h")
                
            if old_invalid_count > 0 or old_valid_count > 0:
                self._save_cache()
                
        except Exception as e:
            print(f"[SYMBOL CLEANUP ERROR] {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "invalid_symbols": len(self.cache.get("invalid", [])),
            "valid_symbols": len(self.cache.get("valid", [])),
            "session_validated": len(self.session_validated),
            "api_calls_saved": self.cache.get("stats", {}).get("api_calls_saved", 0),
            "validation_hits": self.cache.get("stats", {}).get("validation_hits", 0),
            "last_updated": self.cache.get("last_updated", "Never")
        }

# Global validator instance
_symbol_validator = None

def get_symbol_validator() -> SymbolValidator:
    """Get global symbol validator instance"""
    global _symbol_validator
    if _symbol_validator is None:
        _symbol_validator = SymbolValidator()
    return _symbol_validator

def symbol_exists_on_bybit(symbol: str) -> bool:
    """
    Check if symbol exists on Bybit exchange
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        
    Returns:
        bool: True if symbol exists and is tradeable
    """
    validator = get_symbol_validator()
    
    # Check cache first
    cached_result = validator.is_symbol_valid(symbol)
    if cached_result is not None:
        return cached_result
    
    # Validate with minimal API call (ticker endpoint is fastest)
    try:
        import requests
        
        # Use ticker endpoint for fast validation
        url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("result", {})
            tickers = result.get("list", [])
            
            if tickers and len(tickers) > 0:
                ticker = tickers[0]
                # Validate ticker has essential data
                if ticker.get("symbol") == symbol and ticker.get("lastPrice"):
                    validator.mark_symbol_valid(symbol)
                    print(f"[SYMBOL VALID] {symbol}: Confirmed on Bybit")
                    return True
            
            # Symbol not found or no data
            validator.mark_symbol_invalid(symbol, "No ticker data")
            return False
            
        else:
            print(f"[SYMBOL VALIDATION ERROR] {symbol}: HTTP {response.status_code}")
            # Don't cache on HTTP errors, might be temporary
            return True  # Assume valid on API errors
            
    except Exception as e:
        print(f"[SYMBOL VALIDATION ERROR] {symbol}: {e}")
        return True  # Assume valid on errors
    
    return False

def validate_symbol_batch(symbols: List[str]) -> Dict[str, bool]:
    """
    Validate multiple symbols efficiently
    
    Args:
        symbols: List of symbols to validate
        
    Returns:
        Dict mapping symbol to validity status
    """
    validator = get_symbol_validator()
    results = {}
    
    # Check cache for all symbols first
    uncached_symbols = []
    for symbol in symbols:
        cached_result = validator.is_symbol_valid(symbol)
        if cached_result is not None:
            results[symbol] = cached_result
        else:
            uncached_symbols.append(symbol)
    
    # Validate uncached symbols in batch
    if uncached_symbols:
        try:
            import requests
            
            # Use batch ticker endpoint
            url = "https://api.bybit.com/v5/market/tickers?category=spot"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                all_tickers = data.get("result", {}).get("list", [])
                available_symbols = {ticker.get("symbol") for ticker in all_tickers if ticker.get("symbol")}
                
                for symbol in uncached_symbols:
                    if symbol in available_symbols:
                        validator.mark_symbol_valid(symbol)
                        results[symbol] = True
                    else:
                        validator.mark_symbol_invalid(symbol, "Not in ticker list")
                        results[symbol] = False
                        
        except Exception as e:
            print(f"[SYMBOL BATCH VALIDATION ERROR] {e}")
            # Assume all uncached symbols are valid on error
            for symbol in uncached_symbols:
                results[symbol] = True
    
    return results

def get_invalid_symbols() -> List[str]:
    """Get list of currently cached invalid symbols"""
    validator = get_symbol_validator()
    return validator.cache.get("invalid", [])

def clear_symbol_cache():
    """Clear all symbol validation cache"""
    validator = get_symbol_validator()
    validator.cache = {
        "invalid": [],
        "valid": [],
        "last_updated": datetime.now().isoformat(),
        "stats": {"api_calls_saved": 0, "validation_hits": 0}
    }
    validator.session_validated.clear()
    validator._save_cache()
    print("[SYMBOL CACHE] All validation cache cleared")

def main():
    """Test symbol validation system"""
    print("Testing Symbol Validation System...")
    
    # Test known valid symbols
    valid_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    for symbol in valid_symbols:
        result = symbol_exists_on_bybit(symbol)
        print(f"{symbol}: {'✓' if result else '✗'}")
    
    # Test known invalid symbols
    invalid_symbols = ["USDTBUSDT", "USTCUSDT", "INVALIDUSDT"]
    for symbol in invalid_symbols:
        result = symbol_exists_on_bybit(symbol)
        print(f"{symbol}: {'✓' if result else '✗'}")
    
    # Show cache stats
    validator = get_symbol_validator()
    stats = validator.get_cache_stats()
    print(f"\nCache Stats: {stats}")

if __name__ == "__main__":
    main()