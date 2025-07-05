#!/usr/bin/env python3
"""
Debug Basic Engine - Test with real production data format
"""

import json
from trader_ai_engine_basic import simulate_trader_decision_basic

def test_with_production_data():
    """Test basic engine with actual production data format"""
    
    # Load actual production data
    with open('data/async_results/1INCHUSDT_async.json', 'r') as f:
        prod_data = json.load(f)
    
    print("TESTING BASIC ENGINE WITH PRODUCTION DATA")
    print("=" * 50)
    
    # Extract data in same format as production
    symbol = prod_data['symbol']
    current_price = prod_data['price_usd']
    volume_24h = prod_data['volume_24h']
    candles_15m = prod_data['candles_15m']
    candles_5m = prod_data['candles_5m']
    
    print(f"Symbol: {symbol}")
    print(f"Price: {current_price}")
    print(f"Volume 24h: ${volume_24h:,.0f}")
    print(f"15M Candles: {len(candles_15m)}")
    print(f"5M Candles: {len(candles_5m)}")
    print(f"First 15M candle: {candles_15m[0] if candles_15m else 'None'}")
    print(f"Last 15M candle: {candles_15m[-1] if candles_15m else 'None'}")
    print()
    
    # Test basic engine with this data
    try:
        result = simulate_trader_decision_basic(
            symbol=symbol,
            current_price=current_price,
            candles_15m=candles_15m,
            candles_5m=candles_5m,
            orderbook_data={},  # No orderbook in saved data
            volume_24h=volume_24h,
            price_change_24h=None  # No price change in saved data
        )
        
        print("BASIC ENGINE RESULT:")
        print(f"  Score: {result['final_score']}")
        print(f"  Decision: {result['decision']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Components: {result.get('components', {})}")
        print(f"  Reason: {result.get('reason', 'N/A')}")
        
        # Analyze components
        if 'components' in result:
            print("\nCOMPONENT ANALYSIS:")
            for component, score in result['components'].items():
                print(f"  {component}: {score:.4f}")
        
        # Compare with current production score
        print(f"\nCOMPARISON:")
        print(f"  Production score: {prod_data['tjde_score']}")
        print(f"  Basic engine score: {result['final_score']}")
        print(f"  Difference: {result['final_score'] - prod_data['tjde_score']:.4f}")
        
    except Exception as e:
        print(f"ERROR testing basic engine: {e}")
        import traceback
        traceback.print_exc()

def test_candle_format_handling():
    """Test how basic engine handles different candle formats"""
    
    print("\nTESTING CANDLE FORMAT HANDLING")
    print("=" * 50)
    
    # Test different candle formats
    dict_candle = {"timestamp": 1751753700000, "open": 0.1804, "high": 0.1804, "low": 0.1804, "close": 0.1804, "volume": 100.0}
    list_candle = [1751753700000, 0.1804, 0.1804, 0.1804, 0.1804, 100.0]
    
    print(f"Dict candle: {dict_candle}")
    print(f"List candle: {list_candle}")
    
    # Test extraction
    from trader_ai_engine_basic import _compute_basic_trend_strength
    
    try:
        # Test with dict format
        dict_candles = [dict_candle] * 20
        dict_result = _compute_basic_trend_strength(dict_candles, 0.1804)
        print(f"Dict format result: {dict_result}")
        
        # Test with list format  
        list_candles = [list_candle] * 20
        list_result = _compute_basic_trend_strength(list_candles, 0.1804)
        print(f"List format result: {list_result}")
        
    except Exception as e:
        print(f"Error in candle format test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_production_data()
    test_candle_format_handling()