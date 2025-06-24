#!/usr/bin/env python3
"""
Production Test for Async Scanner
Tests the complete async scanning pipeline with realistic scenarios
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scan_all_tokens_async import scan_symbols_async

async def test_production_async():
    """Test async scanner with production-like conditions"""
    
    print("🔥 PRODUCTION ASYNC SCANNER TEST")
    print("=" * 50)
    
    # Test with small sample for quick validation
    test_symbols = [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT',
        'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT'
    ]
    
    print(f"Testing with {len(test_symbols)} symbols...")
    print("Expected: Each token should get realistic PPWCS and TJDE scores")
    print("Not expected: All tokens with 0.0 scores or fallback values")
    print()
    
    try:
        # Run async scan
        results = await scan_symbols_async(test_symbols, max_concurrent=5)
        
        print(f"\n📊 TEST RESULTS ANALYSIS:")
        print("=" * 50)
        
        if not results:
            print("❌ NO RESULTS - All tokens failed or skipped")
            return False
        
        # Analyze results
        total_tokens = len(test_symbols)
        successful_tokens = len(results)
        
        print(f"Success Rate: {successful_tokens}/{total_tokens} ({successful_tokens/total_tokens*100:.1f}%)")
        
        # Check for realistic scores
        realistic_ppwcs = sum(1 for r in results if r.get('ppwcs_score', 0) > 10)
        realistic_tjde = sum(1 for r in results if r.get('tjde_score', 0) > 0.1)
        
        print(f"Realistic PPWCS: {realistic_ppwcs}/{successful_tokens}")
        print(f"Realistic TJDE: {realistic_tjde}/{successful_tokens}")
        
        # Check for suspicious fallback patterns
        fallback_ppwcs = sum(1 for r in results if r.get('ppwcs_score') in [0.0, 40.0, 25.0])
        fallback_tjde = sum(1 for r in results if r.get('tjde_score') in [0.0, 0.4])
        
        print(f"Suspicious fallbacks - PPWCS: {fallback_ppwcs}, TJDE: {fallback_tjde}")
        
        # Show sample results
        print(f"\n📋 SAMPLE RESULTS:")
        for i, result in enumerate(results[:5], 1):
            symbol = result.get('symbol', 'UNKNOWN')
            ppwcs = result.get('ppwcs_score', 0)
            tjde_score = result.get('tjde_score', 0)
            tjde_decision = result.get('tjde_decision', 'avoid')
            
            print(f"{i}. {symbol:10} PPWCS: {ppwcs:5.1f} TJDE: {tjde_score:.3f} ({tjde_decision})")
        
        # Validation
        success_criteria = [
            successful_tokens >= total_tokens * 0.5,  # At least 50% success rate
            realistic_ppwcs >= successful_tokens * 0.7,  # 70% realistic PPWCS
            realistic_tjde >= successful_tokens * 0.7,   # 70% realistic TJDE
            fallback_ppwcs <= successful_tokens * 0.3,   # ≤30% fallbacks
        ]
        
        if all(success_criteria):
            print(f"\n✅ PRODUCTION TEST PASSED")
            print("Async scanner is ready for production deployment")
            return True
        else:
            print(f"\n⚠️  PRODUCTION TEST NEEDS IMPROVEMENT")
            print("Some scoring functions may still be using fallbacks")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False

def main():
    """Main test function"""
    result = asyncio.run(test_production_async())
    
    print(f"\n" + "=" * 50)
    if result:
        print("🎯 ASYNC SCANNER READY FOR PRODUCTION")
        print("All scoring functions working correctly")
    else:
        print("🔧 ASYNC SCANNER NEEDS FIXES")
        print("Check import errors and scoring function availability")
    
    return result

if __name__ == "__main__":
    main()