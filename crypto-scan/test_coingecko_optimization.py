#!/usr/bin/env python3
"""
Test script for optimized CoinGecko integration
Tests the single API call approach with cache
"""

import os
import sys
sys.path.append('.')

from utils.coingecko import build_contract_cache, get_contract, get_multiple_token_contracts_from_coingecko

def test_coingecko_optimization():
    """Test the optimized CoinGecko cache system"""
    print("🧪 Testing CoinGecko optimization...")
    
    # Test symbols (common Bybit perpetuals)
    test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
    
    print("\n1. Testing cache building...")
    try:
        build_contract_cache()
        print("✅ Cache build completed")
    except Exception as e:
        print(f"❌ Cache build failed: {e}")
        return
    
    print("\n2. Testing individual contract retrieval...")
    for symbol in test_symbols:
        contract = get_contract(symbol)
        if contract:
            print(f"✅ {symbol}: {contract['chain']} - {contract['address'][:10]}...")
        else:
            print(f"⚠️ {symbol}: No contract found")
    
    print("\n3. Testing batch contract retrieval...")
    batch_results = get_multiple_token_contracts_from_coingecko(test_symbols)
    found_count = sum(1 for v in batch_results.values() if v is not None)
    print(f"✅ Batch processing: {found_count}/{len(test_symbols)} contracts found")
    
    print("\n4. Checking cache file...")
    if os.path.exists("token_contract_map.json"):
        with open("token_contract_map.json", "r") as f:
            import json
            cache_data = json.load(f)
            print(f"✅ Cache file contains {len(cache_data)} contracts")
    else:
        print("❌ Cache file not found")

if __name__ == "__main__":
    test_coingecko_optimization()