"""
Token Symbol Normalizer for CoinGecko Cache Mapping
Fixes issues with 10000* and 1000000* tokens not matching CoinGecko data
"""

import re
from typing import Dict, Set


def normalize_token_symbol(symbol: str) -> str:
    """
    Normalize token symbol for CoinGecko cache lookup
    
    Args:
        symbol: Trading symbol (e.g., "10000SATSUSDT", "1000000SHIBASWAPUSDT")
        
    Returns:
        Normalized symbol for cache lookup (e.g., "SATS", "SHIBASWAP")
    """
    if not symbol:
        return symbol
    
    # Remove common prefixes that CoinGecko doesn't use
    normalized = symbol
    
    # Remove numeric prefixes (10000, 1000000, etc.)
    normalized = re.sub(r'^(\d+)', '', normalized)
    
    # Remove trading pair suffixes
    normalized = normalized.replace("USDT", "").replace("BUSD", "").replace("BTC", "").replace("ETH", "")
    
    # Clean up any remaining artifacts
    normalized = normalized.strip()
    
    return normalized


def get_coingecko_mapping_candidates(symbol: str) -> list:
    """
    Generate multiple candidate mappings for a token symbol
    
    Args:
        symbol: Original trading symbol
        
    Returns:
        List of possible CoinGecko identifiers to check
    """
    candidates = []
    
    # Original symbol
    candidates.append(symbol)
    
    # Normalized version
    normalized = normalize_token_symbol(symbol)
    if normalized != symbol:
        candidates.append(normalized)
    
    # Common variations
    base_symbol = symbol.replace("USDT", "").replace("BUSD", "")
    candidates.append(base_symbol)
    
    # Lowercase versions
    candidates.extend([c.lower() for c in candidates])
    
    # Remove duplicates while preserving order
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen and candidate:
            unique_candidates.append(candidate)
            seen.add(candidate)
    
    return unique_candidates


def find_in_coingecko_cache(symbol: str, coingecko_cache: Dict) -> tuple:
    """
    Enhanced CoinGecko cache lookup with normalization
    
    Args:
        symbol: Trading symbol to look up
        coingecko_cache: CoinGecko cache dictionary
        
    Returns:
        Tuple of (found_symbol, cache_entry) or (None, None) if not found
    """
    if not coingecko_cache:
        return None, None
    
    candidates = get_coingecko_mapping_candidates(symbol)
    
    for candidate in candidates:
        # Direct lookup
        if candidate in coingecko_cache:
            print(f"[CACHE CHECK] {symbol} → {candidate} → FOUND (direct)")
            return candidate, coingecko_cache[candidate]
        
        # Case-insensitive search through cache keys
        for cache_key in coingecko_cache.keys():
            if cache_key.lower() == candidate.lower():
                print(f"[CACHE CHECK] {symbol} → {candidate} → FOUND ({cache_key})")
                return cache_key, coingecko_cache[cache_key]
        
        # Partial match search (for complex token names)
        for cache_key in coingecko_cache.keys():
            if candidate.lower() in cache_key.lower() or cache_key.lower() in candidate.lower():
                if len(candidate) > 2 and len(cache_key) > 2:  # Avoid false matches with short symbols
                    print(f"[CACHE CHECK] {symbol} → {candidate} → FOUND (partial: {cache_key})")
                    return cache_key, coingecko_cache[cache_key]
    
    print(f"[CACHE CHECK] {symbol} normalized → {normalize_token_symbol(symbol)} → NOT FOUND")
    return None, None


def test_token_normalizer():
    """Test the token normalizer with common problematic tokens"""
    test_cases = [
        "10000SATSUSDT",
        "10000WENUSDT", 
        "10000LADYSUSDT",
        "10000COQUSDT",
        "10000ELONUSDT",
        "1000000SHIBASWAPUSDT",
        "1000XECUSDT",
        "BTCUSDT",
        "ETHUSDT"
    ]
    
    print("Testing Token Normalizer:")
    for symbol in test_cases:
        normalized = normalize_token_symbol(symbol)
        candidates = get_coingecko_mapping_candidates(symbol)
        print(f"  {symbol} → {normalized} (candidates: {candidates[:3]})")
    
    return True


if __name__ == "__main__":
    test_token_normalizer()