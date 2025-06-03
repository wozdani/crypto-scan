from utils.coingecko import load_cache, is_cache_valid, build_coingecko_cache

def get_token_price_usd(symbol):
    """Get token price from cache - no direct API calls"""
    try:
        if not is_cache_valid():
            build_coingecko_cache()
        
        cache = load_cache()
        symbol_upper = symbol.upper()
        
        # Remove USDT suffix to get base symbol
        base_symbol = symbol_upper.replace("USDT", "")
        
        coin_data = cache.get(base_symbol)
        if not coin_data:
            print(f"⚠️ Brak danych cenowych dla {symbol} w cache")
            return None
            
        # Note: Price data would need to be added to cache in future enhancement
        # For now, return None to avoid API calls
        print(f"⚠️ Cena dla {symbol} dostępna tylko z cache (wymaga rozszerzenia)")
        return None
        
    except Exception as e:
        print(f"❌ Błąd pobierania ceny dla {symbol}: {e}")
        return None