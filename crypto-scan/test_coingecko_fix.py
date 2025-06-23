#!/usr/bin/env python3

import requests

def test_coingecko_token_selection():
    """Test the improved CoinGecko token selection logic"""
    print("🧪 Testing CoinGecko token selection logic...")
    
    try:
        # Get token list from CoinGecko
        response = requests.get("https://api.coingecko.com/api/v3/coins/list", timeout=15)
        token_list = response.json()
        
        # Group tokens by symbol
        symbol_groups = {}
        for token in token_list:
            if 'symbol' in token and 'id' in token:
                symbol = token['symbol'].upper()
                if symbol not in symbol_groups:
                    symbol_groups[symbol] = []
                symbol_groups[symbol].append(token)
        
        # Test cases
        test_symbols = ['MAGIC', 'TRX', 'XRP', 'BTC', 'ETH']
        
        for symbol in test_symbols:
            print(f"\n🔍 Testing {symbol}:")
            
            if symbol not in symbol_groups:
                print(f"  ❌ No candidates found")
                continue
                
            candidates = symbol_groups[symbol]
            print(f"  📊 {len(candidates)} candidates found:")
            
            # Show first few candidates
            for i, candidate in enumerate(candidates[:3]):
                print(f"    {i+1}. {candidate['id']} -> {candidate['symbol']}")
            
            # Apply selection logic
            best_candidate = None
            priority_names = [symbol.lower(), f"{symbol.lower()}-token", f"{symbol.lower()}-coin"]
            
            # 1. Priority selection
            for candidate in candidates:
                if candidate['symbol'].upper() == symbol.upper():
                    candidate_id = candidate['id'].lower()
                    if any(priority in candidate_id for priority in priority_names):
                        best_candidate = candidate
                        print(f"  ✅ Priority match: {candidate['id']}")
                        break
            
            # 2. Fallback to first exact symbol match
            if not best_candidate:
                for candidate in candidates:
                    if candidate['symbol'].upper() == symbol.upper():
                        best_candidate = candidate
                        print(f"  ✅ Standard match: {candidate['id']}")
                        break
            
            if not best_candidate:
                print(f"  ❌ No valid match found")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_coingecko_token_selection()