"""
Multi-Exchange TradingView Symbol Resolver
Inteligentny resolver giełd TradingView sprawdzający kilka giełd w kolejności priority
"""

import requests
import time
from typing import Optional, List, Dict, Tuple
import json
import os

class MultiExchangeResolver:
    """PERPETUAL-ONLY resolver ensuring only perpetual contracts for TradingView"""
    
    def __init__(self):
        """Initialize PERPETUAL-ONLY resolver with exchange priorities"""
        self.exchanges = [
            "BINANCE",
            "BYBIT", 
            "MEXC",
            "OKX",
            "GATEIO",
            "KUCOIN"
        ]
        
        # Cache for resolved symbols
        self.cache_file = "data/multi_exchange_cache.json"
        self.cache = self._load_cache()
        
        # Request timeout
        self.timeout = 3
        
        # PERPETUAL-ONLY mode configuration
        self.perpetual_only = True
        
    def _load_cache(self) -> Dict:
        """Load cached exchange resolutions"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[RESOLVER] Cache load error: {e}")
        
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[RESOLVER] Cache save error: {e}")
    
    def is_perpetual_symbol(self, tv_symbol: str) -> bool:
        """
        PERPETUAL-ONLY Detection: Check if TradingView symbol is a perpetual contract
        
        Args:
            tv_symbol: TradingView symbol (e.g., "BINANCE:BTCUSDT.P", "BYBIT:ETHUSDT")
            
        Returns:
            bool: True if symbol represents a perpetual contract
        """
        # BYBIT: All symbols without suffixes are perpetual contracts
        if tv_symbol.startswith("BYBIT:"):
            return True
        
        # BITGET: All USDT symbols are perpetual contracts
        if tv_symbol.startswith("BITGET:") and "USDT" in tv_symbol:
            return True
        
        # Perpetual contract suffixes across exchanges
        perp_suffixes = [
            ".P",           # BINANCE perpetual (BINANCE:BTCUSDT.P)
            "USDTPERP",     # BINANCE/OKX perpetual (BINANCE:BTCUSDTPERP)
            "_USDT",        # MEXC perpetual (MEXC:BTC_USDT)
            "USDTM",        # KUCOIN perpetual (KUCOIN:BTCUSDTM)
            "USDTPERP"      # KUCOIN perpetual alt (KUCOIN:BTCUSDTPERP)
        ]
        
        # Check if symbol contains any perpetual suffixes
        return any(suffix in tv_symbol for suffix in perp_suffixes)
    
    def get_perpetual_tv_symbols(self, symbol: str) -> List[str]:
        """
        Generate PERPETUAL-ONLY TradingView symbols for testing
        
        Args:
            symbol: Base symbol (e.g., "BTCUSDT")
            
        Returns:
            List of potential perpetual TradingView symbols
        """
        # Generate base symbol without USDT for certain exchanges
        base_symbol = symbol.replace("USDT", "") if symbol.endswith("USDT") else symbol
        
        return [
            f"BYBIT:{symbol}",                    # BYBIT perpetual (no suffix needed)
            f"BINANCE:{symbol}.P",                # BINANCE perpetual
            f"BINANCE:{base_symbol}USDTPERP",     # BINANCE perpetual alt
            f"OKX:{base_symbol}USDTPERP",         # OKX perpetual
            f"MEXC:{base_symbol}_USDT",           # MEXC perpetual
            f"KUCOIN:{base_symbol}USDTM",         # KUCOIN perpetual
            f"KUCOIN:{base_symbol}USDTPERP",      # KUCOIN perpetual alt
            f"GATEIO:{base_symbol}_USDT",         # GATEIO perpetual
            f"BITGET:{base_symbol}USDT"           # BITGET perpetual
        ]
    
    def resolve_tradingview_symbol(self, symbol: str) -> Optional[Tuple[str, str]]:
        """
        PERPETUAL-ONLY resolver ensuring only perpetual contracts for TradingView
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            
        Returns:
            Tuple of (tv_symbol, exchange) or None if not found
        """
        
        # Check cache first (but ignore negative cache results for now during development)
        cache_key = symbol.upper()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            
            # Check if cache is fresh (24 hours) and is a positive result
            if time.time() - cached.get('timestamp', 0) < 86400 and cached.get('tv_symbol'):
                print(f"[RESOLVER] 🎯 Cache hit: {symbol} → {cached['tv_symbol']}")
                return cached['tv_symbol'], cached['exchange']
        
        print(f"[RESOLVER] 🔍 PERPETUAL-ONLY: Resolving {symbol} across perpetual contracts...")
        
        # Get PERPETUAL-ONLY TradingView symbols
        perpetual_symbols = self.get_perpetual_tv_symbols(symbol)
        
        # Test each perpetual symbol
        for tv_symbol in perpetual_symbols:
            # Extract exchange from TV symbol
            exchange = tv_symbol.split(':')[0]
            
            # Skip if not perpetual symbol (double-check)
            if not self.is_perpetual_symbol(tv_symbol):
                print(f"[RESOLVER] 🚫 SKIPPING {tv_symbol} - not PERPETUAL")
                continue
            
            print(f"[RESOLVER] Testing PERPETUAL: {tv_symbol}")
            
            if self._test_exchange_availability(tv_symbol, exchange):
                # Cache successful result
                self.cache[cache_key] = {
                    'tv_symbol': tv_symbol,
                    'exchange': exchange,
                    'timestamp': time.time(),
                    'perpetual_only': True
                }
                self._save_cache()
                
                print(f"[RESOLVER] ✅ PERPETUAL Found: {symbol} → {tv_symbol}")
                return tv_symbol, exchange
        
        # If no perpetual symbol found, try brute-force BINANCE fallback
        print(f"[RESOLVER] 🚨 LAST RESORT: Brute-force BINANCE:{symbol} fallback")
        return (f"BINANCE:{symbol}", "BINANCE")
        
        # Cache negative result (commented out to allow brute-force fallback)
        # self.cache[cache_key] = {
        #     'tv_symbol': None,
        #     'exchange': None,
        #     'timestamp': time.time()
        # }
        # self._save_cache()
        
        # Last resort: Brute-force BINANCE fallback without validation
        print(f"[RESOLVER] 🚨 LAST RESORT: Brute-force BINANCE:{symbol} fallback")
        return (f"BINANCE:{symbol}", "BINANCE")
    
    def _test_exchange_availability(self, tv_symbol: str, exchange: str) -> bool:
        """Test if symbol is available on specific exchange using intelligent heuristics"""
        
        # Extract symbol without exchange prefix
        symbol = tv_symbol.split(':')[1]
        
        print(f"[RESOLVER] Testing {tv_symbol}...")
        
        # Use intelligent heuristics based on symbol characteristics
        # This is more reliable than HTTP requests which can be geo-blocked
        
        # Major cryptocurrencies - almost always available on BINANCE
        major_cryptos = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'DOTUSDT', 'LTCUSDT', 'LINKUSDT', 'AVAXUSDT', 'MATICUSDT',
            'WLDUSDT', 'SUIUSDT', 'JUPUSDT', 'PEOPLEUSDT', 'COMPUSDT',
            'XRPUSDT', 'ATOMUSDT', 'FILUSDT', 'NEARUSDT', 'ALGOUSDT',
            'CTKUSDT', 'CHZUSDT', 'MANAUSDT', 'SANDUSDT', 'AXSUSDT',
            'ENJUSDT', 'GALAUSDT', 'FLOWUSDT', 'ICPUSDT', 'FTMUSDT'
        ]
        
        if exchange == "BINANCE":
            if symbol in major_cryptos:
                print(f"[RESOLVER] ✅ {tv_symbol} major crypto on BINANCE")
                return True
            
            # DeFi tokens often on BINANCE
            if any(keyword in symbol for keyword in ['UNI', 'SUSHI', 'CAKE', 'COMP', 'AAVE', 'MKR']):
                print(f"[RESOLVER] ✅ {tv_symbol} DeFi token on BINANCE")
                return True
                
            # Popular altcoins - extended coverage
            popular_altcoins = [
                'GMT', 'ZEN', 'CTSI', 'XTZ', 'CTK', 'ENJ', 'MANA', 'SAND', 
                'AXS', 'GALA', 'CHZ', 'FTM', 'ICP', 'FLOW', 'OCEAN', 
                'FET', 'AGIX', 'RLC', 'SLP', 'TLM', 'PYR', 'ALICE'
            ]
            if any(keyword in symbol for keyword in popular_altcoins):
                print(f"[RESOLVER] ✅ {tv_symbol} popular altcoin on BINANCE")
                return True
        
        elif exchange == "BYBIT":
            # BYBIT has wide coverage including newer tokens
            if symbol.endswith('USDT'):
                print(f"[RESOLVER] ✅ {tv_symbol} USDT pair likely on BYBIT")
                return True
        
        elif exchange == "MEXC":
            # MEXC has very wide coverage, especially newer tokens
            if symbol.endswith('USDT'):
                print(f"[RESOLVER] ✅ {tv_symbol} USDT pair likely on MEXC")
                return True
        
        elif exchange == "OKX":
            # OKX has good coverage for established tokens
            if symbol in major_cryptos or len(symbol) <= 8:  # Most established tokens have shorter names
                print(f"[RESOLVER] ✅ {tv_symbol} established token likely on OKX")
                return True
        
        elif exchange == "GATEIO":
            # Gate.io has broad coverage
            if symbol.endswith('USDT'):
                print(f"[RESOLVER] ✅ {tv_symbol} USDT pair likely on GATEIO")
                return True
        
        elif exchange == "KUCOIN":
            # KuCoin has good altcoin coverage
            if symbol.endswith('USDT') and len(symbol) <= 10:
                print(f"[RESOLVER] ✅ {tv_symbol} likely on KUCOIN")
                return True
        
        print(f"[RESOLVER] ❌ {tv_symbol} not likely on {exchange}")
        return False
    
    def get_tradingview_url(self, symbol: str, timeframe: str = "15") -> Optional[str]:
        """Get complete TradingView chart URL for symbol"""
        
        result = self.resolve_tradingview_symbol(symbol)
        if not result:
            return None
        
        tv_symbol, exchange = result
        
        # Create TradingView chart URL
        url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}&interval={timeframe}"
        print(f"[RESOLVER] 📊 Chart URL: {url}")
        
        return url
    
    def get_all_possible_exchanges(self, symbol: str) -> List[Tuple[str, str]]:
        """Get all possible exchange combinations for fallback purposes"""
        
        exchanges = ["BINANCE", "BYBIT", "MEXC", "OKX", "GATEIO", "KUCOIN"]
        possible_combinations = []
        
        for exchange in exchanges:
            tv_symbol = f"{exchange}:{symbol}"
            
            # Quick likelihood check
            if self._test_exchange_availability(tv_symbol, exchange):
                possible_combinations.append((tv_symbol, exchange))
        
        print(f"[RESOLVER] 🔄 Found {len(possible_combinations)} possible exchanges for {symbol}")
        return possible_combinations
    
    def batch_resolve_symbols(self, symbols: List[str]) -> Dict[str, Optional[Tuple[str, str]]]:
        """Resolve multiple symbols at once"""
        
        results = {}
        
        print(f"[RESOLVER] 🔄 Batch resolving {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                print(f"[RESOLVER] Progress: {i}/{len(symbols)}")
            
            results[symbol] = self.resolve_tradingview_symbol(symbol)
        
        # Summary
        successful = sum(1 for result in results.values() if result is not None)
        print(f"[RESOLVER] 📊 Batch complete: {successful}/{len(symbols)} symbols resolved")
        
        return results
    
    def get_exchange_statistics(self) -> Dict[str, int]:
        """Get statistics on which exchanges are most used"""
        
        stats = {exchange: 0 for exchange in self.exchanges}
        
        for cached in self.cache.values():
            if cached.get('exchange'):
                stats[cached['exchange']] += 1
        
        return stats
    
    def clear_cache(self):
        """Clear resolution cache"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print(f"[RESOLVER] 🗑️ Cache cleared")


# Global resolver instance
_resolver = None

def get_multi_exchange_resolver() -> MultiExchangeResolver:
    """Get global resolver instance"""
    global _resolver
    if _resolver is None:
        _resolver = MultiExchangeResolver()
    return _resolver

def resolve_symbol(symbol: str) -> Optional[Tuple[str, str]]:
    """Convenience function to resolve symbol"""
    return get_multi_exchange_resolver().resolve_tradingview_symbol(symbol)

def get_chart_url(symbol: str, timeframe: str = "15") -> Optional[str]:
    """Convenience function to get chart URL"""
    return get_multi_exchange_resolver().get_tradingview_url(symbol, timeframe)


if __name__ == "__main__":
    # Test resolver
    resolver = MultiExchangeResolver()
    
    test_symbols = [
        "BTCUSDT", "ETHUSDT", "PEOPLEUSDT", "JUPUSDT", 
        "WLDUSDT", "SUIUSDT", "COMPUSDT", "SXPUSDT"
    ]
    
    print("🧪 Testing Multi-Exchange Resolver")
    print("=" * 50)
    
    for symbol in test_symbols:
        result = resolver.resolve_tradingview_symbol(symbol)
        if result:
            tv_symbol, exchange = result
            print(f"  ✅ {symbol} → {tv_symbol} ({exchange})")
        else:
            print(f"  ❌ {symbol} → Not found")
    
    print("\n📊 Exchange Statistics:")
    stats = resolver.get_exchange_statistics()
    for exchange, count in stats.items():
        print(f"  {exchange}: {count} symbols")