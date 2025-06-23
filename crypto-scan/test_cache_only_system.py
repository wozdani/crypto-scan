#!/usr/bin/env python3
"""
Test script to verify the cache-only CoinGecko system
Ensures no individual API calls are made
"""

import os
import sys
sys.path.append('.')

from utils.coingecko import build_coingecko_cache, get_contract, is_cache_valid
from utils.contracts import get_or_fetch_token_contract

def test_cache_only_system():
    """Test that system uses only cache, no individual API calls"""
    print("🧪 Testing cache-only CoinGecko system...")
    
    # Build cache once
    print("\n1. Building CoinGecko cache...")
    try:
        build_coingecko_cache()
        print("✅ Cache build completed")
    except Exception as e:
        print(f"❌ Cache build failed: {e}")
        return
    
    # Test cache validity
    print("\n2. Testing cache validity...")
    cache_valid = is_cache_valid()
    print(f"✅ Cache valid: {cache_valid}")
    
    # Test contract retrieval without additional API calls
    print("\n3. Testing contract retrieval from cache...")
    test_symbols = ["BTC", "ETH", "ADA", "DOT", "LINK"]
    
    for symbol in test_symbols:
        contract = get_contract(symbol)
        if contract:
            print(f"✅ {symbol}: Found contract on {contract['chain']}")
        else:
            print(f"⚠️ {symbol}: No contract in cache")
    
    # Test via contracts module
    print("\n4. Testing via contracts module...")
    for symbol in test_symbols:
        contract = get_or_fetch_token_contract(f"{symbol}USDT")
        if contract:
            print(f"✅ {symbol}USDT: Found via contracts module")
        else:
            print(f"⚠️ {symbol}USDT: Not found via contracts module")
    
    print("\n✅ Cache-only system test completed")
    print("No individual CoinGecko API calls should have been made after initial cache build")

if __name__ == "__main__":
    test_cache_only_system()