"""
Symbol Validator - Check if symbols are active and tradeable on Bybit
Prevents scanning inactive or delisted symbols like MAVIAUSDT
"""
import json
import os
from typing import List, Dict, Set
from utils.robust_api import robust_api, check_symbol_health

class SymbolValidator:
    """Validates symbols before scanning to avoid API failures"""
    
    def __init__(self):
        self.cache_file = "data/cache/symbol_health_cache.json"
        self.health_cache = self.load_health_cache()
        
    def load_health_cache(self) -> Dict:
        """Load symbol health cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[VALIDATOR] Failed to load health cache: {e}")
        return {}
    
    def save_health_cache(self):
        """Save symbol health cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self.health_cache, f, indent=2)
        except Exception as e:
            print(f"[VALIDATOR] Failed to save health cache: {e}")
    
    def validate_symbol(self, symbol: str, force_refresh: bool = False) -> bool:
        """
        Validate if symbol is healthy for scanning
        
        Args:
            symbol: Trading symbol
            force_refresh: Force fresh validation
            
        Returns:
            True if symbol is healthy for scanning
        """
        # Check cache first
        if not force_refresh and symbol in self.health_cache:
            cached_health = self.health_cache[symbol]
            if cached_health.get("overall_health") in ["healthy", "limited"]:
                return True
            elif cached_health.get("overall_health") == "unhealthy":
                return False
        
        # Perform fresh health check
        health = check_symbol_health(symbol)
        self.health_cache[symbol] = health
        
        # Save cache periodically
        if len(self.health_cache) % 10 == 0:
            self.save_health_cache()
        
        return health.get("overall_health") in ["healthy", "limited"]
    
    def filter_healthy_symbols(self, symbols: List[str]) -> List[str]:
        """
        Filter symbols to only include healthy ones
        
        Args:
            symbols: List of symbols to filter
            
        Returns:
            List of healthy symbols
        """
        healthy_symbols = []
        unhealthy_symbols = []
        api_error_count = 0
        
        print(f"[VALIDATOR] Checking health of {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            if i % 20 == 0:  # Progress update every 20 symbols
                print(f"[VALIDATOR] Progress: {i}/{len(symbols)}")
            
            # Check for widespread API failures
            if i > 10 and api_error_count > 8:  # More than 80% failures in first 10
                print(f"[VALIDATOR] Detected widespread API failures ({api_error_count}/10) - using essential symbols")
                return self.get_essential_symbols()
            
            if self.validate_symbol(symbol):
                healthy_symbols.append(symbol)
            else:
                unhealthy_symbols.append(symbol)
                if i < 10:  # Track early failures
                    api_error_count += 1
        
        # Save final cache
        self.save_health_cache()
        
        print(f"[VALIDATOR] Healthy: {len(healthy_symbols)}, Unhealthy: {len(unhealthy_symbols)}")
        
        # If no healthy symbols found, use essential list
        if not healthy_symbols:
            print(f"[VALIDATOR] No healthy symbols found - using essential symbols as fallback")
            return self.get_essential_symbols()
        
        if unhealthy_symbols:
            print(f"[VALIDATOR] Skipping unhealthy symbols: {unhealthy_symbols[:10]}")
        
        return healthy_symbols
    
    def get_health_summary(self) -> Dict:
        """Get summary of symbol health status"""
        healthy = sum(1 for h in self.health_cache.values() if h.get("overall_health") == "healthy")
        limited = sum(1 for h in self.health_cache.values() if h.get("overall_health") == "limited")
        unhealthy = sum(1 for h in self.health_cache.values() if h.get("overall_health") == "unhealthy")
        
        return {
            "total_cached": len(self.health_cache),
            "healthy": healthy,
            "limited": limited,
            "unhealthy": unhealthy,
            "healthy_rate": (healthy + limited) / max(len(self.health_cache), 1) * 100
        }
    
    def get_essential_symbols(self) -> List[str]:
        """Get essential trading symbols when API validation fails"""
        essential_symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "AVAXUSDT",
            "MATICUSDT", "LINKUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "FILUSDT",
            "NEARUSDT", "AAVEUSDT", "CRVUSDT", "SUSHIUSDT", "1INCHUSDT", "CKBUSDT",
            "MANAUSDT", "SANDUSDT", "AXSUSDT", "CHZUSDT", "ENJUSDT", "GALAUSDT",
            "BNBUSDT", "XRPUSDT", "DOGEUSDT", "PEPEUSDT", "SHIBUSDT", "FLOKIUSDT",
            "ARBUSDT", "OPUSDT", "APTUSDT", "SUIUSDT", "INJUSDT", "TIAUSDT",
            "WIFUSDT", "BONKUSDT", "JUPUSDT", "WUSDT", "RAYUSDT", "BOMEUSDT"
        ]
        print(f"[VALIDATOR] Using {len(essential_symbols)} essential symbols")
        return essential_symbols

    def refresh_all_symbols(self, symbols: List[str]):
        """Refresh health status for all symbols"""
        print(f"[VALIDATOR] Refreshing health for {len(symbols)} symbols...")
        for symbol in symbols:
            self.validate_symbol(symbol, force_refresh=True)
        self.save_health_cache()
        print(f"[VALIDATOR] Health refresh complete")

# Global validator instance
symbol_validator = SymbolValidator()

def validate_symbols_before_scan(symbols: List[str]) -> List[str]:
    """
    Validate symbols before scanning to avoid API failures
    
    Args:
        symbols: List of symbols to validate
        
    Returns:
        List of healthy symbols ready for scanning
    """
    return symbol_validator.filter_healthy_symbols(symbols)

def check_symbol_is_healthy(symbol: str) -> bool:
    """
    Quick health check for single symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        True if symbol is healthy
    """
    return symbol_validator.validate_symbol(symbol)

def get_validator_stats() -> Dict:
    """Get symbol validator statistics"""
    return symbol_validator.get_health_summary()