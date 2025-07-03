#!/usr/bin/env python3
"""
Debug TJDE v2 Scoring Issues - Complete Component Analysis
Diagnoses why all tokens receive identical scores (e.g., 0.384)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from unified_tjde_engine_v2 import (
    analyze_symbol_with_unified_tjde_v2,
    detect_market_phase_v2,
    load_scoring_profile,
    unified_tjde_decision_engine_with_signals
)
from utils.feature_extractor import extract_features, validate_features
from utils.market_phase_modifier import apply_market_phase_modifier
from utils.final_decision_classifier import classify_final_decision

def create_mock_token_data(symbol: str, price_variance: float = 1.0):
    """Create mock token data with variance for testing"""
    base_price = 1.0 + (hash(symbol) % 100) / 100.0 * price_variance
    
    # Generate varied candle data based on symbol hash
    seed = hash(symbol) % 1000
    candles_15m = []
    candles_5m = []
    
    for i in range(96):  # 15M candles
        price_offset = (seed + i) % 50 / 1000.0
        open_price = base_price + price_offset
        close_price = base_price + price_offset + ((i % 3 - 1) * 0.01)
        high_price = max(open_price, close_price) + 0.005
        low_price = min(open_price, close_price) - 0.005
        volume = 1000000 + (seed + i) % 500000
        
        candles_15m.append([
            int(1672531200000 + i * 900000),  # timestamp
            str(open_price),
            str(high_price), 
            str(low_price),
            str(close_price),
            str(volume),
            str(volume * base_price)  # turnover
        ])
    
    for i in range(288):  # 5M candles  
        price_offset = (seed + i) % 30 / 2000.0
        open_price = base_price + price_offset
        close_price = base_price + price_offset + ((i % 5 - 2) * 0.005)
        high_price = max(open_price, close_price) + 0.002
        low_price = min(open_price, close_price) - 0.002
        volume = 300000 + (seed + i) % 200000
        
        candles_5m.append([
            int(1672531200000 + i * 300000),  # timestamp
            str(open_price),
            str(high_price),
            str(low_price), 
            str(close_price),
            str(volume),
            str(volume * base_price)  # turnover
        ])
    
    return {
        'symbol': symbol,
        'price_usd': base_price,
        'volume_24h': 1000000 + seed % 5000000,
        'price_change_24h': ((seed % 21) - 10) / 10.0,  # -10% to +10%
        'candles_15m': candles_15m,
        'candles_5m': candles_5m,
        'ticker_data': {
            'symbol': symbol,
            'lastPrice': str(base_price),
            'volume24h': str(1000000 + seed % 5000000),
            'price24hPcnt': str(((seed % 21) - 10) / 1000.0)  # Convert to proper percentage
        },
        'orderbook': {
            'bids': [[str(base_price - 0.01), '1000'], [str(base_price - 0.02), '2000']],
            'asks': [[str(base_price + 0.01), '1000'], [str(base_price + 0.02), '2000']]
        }
    }

def debug_tjde_component(symbol: str, verbose: bool = True):
    """Debug TJDE scoring for a single symbol"""
    print(f"\n{'='*60}")
    print(f"🔍 DEBUGGING TJDE v2 SCORING: {symbol}")
    print(f"{'='*60}")
    
    # 1. Create test data
    market_data = create_mock_token_data(symbol)
    candles_15m = market_data['candles_15m']
    candles_5m = market_data['candles_5m']
    
    if verbose:
        print(f"📊 Market Data Created:")
        print(f"   Price: ${market_data['price_usd']:.4f}")
        print(f"   Volume 24h: ${market_data['volume_24h']:,}")
        print(f"   Price Change 24h: {market_data['price_change_24h']:+.2%}")
        print(f"   15M Candles: {len(candles_15m)}")
        print(f"   5M Candles: {len(candles_5m)}")
    
    # 2. Test Market Phase Detection
    try:
        market_phase = detect_market_phase_v2(symbol, market_data, candles_15m, candles_5m)
        print(f"🎯 Market Phase: {market_phase}")
    except Exception as e:
        print(f"❌ Market Phase Detection Failed: {e}")
        market_phase = "trend-following"
    
    # 3. Test Scoring Profile Loading
    try:
        profile = load_scoring_profile(market_phase)
        print(f"📋 Profile Loaded: {profile is not None}")
        if profile and verbose:
            print(f"   Profile components: {list(profile.keys())}")
            total_weight = sum(profile.values()) if profile else 0
            print(f"   Total weight: {total_weight:.3f}")
    except Exception as e:
        print(f"❌ Profile Loading Failed: {e}")
        profile = None
    
    # 4. Test Feature Extraction
    try:
        features = extract_features(market_data)
        if features:
            validation_result = validate_features(features)
            is_valid = validation_result.get('valid', False) if isinstance(validation_result, dict) else validation_result
            print(f"🧠 Features Extracted: {is_valid}")
            if verbose and features:
                print(f"   Features:")
                for key, value in features.items():
                    print(f"     {key}: {value:.4f}")
        else:
            print(f"🧠 Features Extracted: False")
            features = {}
    except Exception as e:
        print(f"❌ Feature Extraction Failed: {e}")
        features = {}
    
    # 5. Test Market Phase Modifier
    try:
        base_score = 0.65  # Test score
        # Create mock market context
        market_context = {
            "htf_phase": "uptrend",
            "volume_trend": "rising", 
            "fear_greed": 60,
            "market_sentiment": "bullish",
            "volatility_regime": "normal"
        }
        modified_score, modifier = apply_market_phase_modifier(base_score, market_context)
        print(f"⚡ Market Phase Modifier: {modifier:+.4f}")
        print(f"   Base Score: {base_score:.4f} → Modified: {modified_score:.4f}")
    except Exception as e:
        print(f"❌ Market Phase Modifier Failed: {e}")
        modified_score = 0.65
        modifier = 0.0
    
    # 6. Test Final Decision Classification
    try:
        clip_confidence = 0.7  # Mock CLIP confidence
        decision, classification_info = classify_final_decision(
            modified_score, 
            clip_confidence, 
            market_phase, 
            features.get('support_reaction', 0.5)
        )
        print(f"🎯 Final Decision: {decision.upper()}")
        if verbose:
            print(f"   LONG threshold: {classification_info['threshold_long']:.3f}")
            print(f"   WAIT threshold: {classification_info['threshold_wait']:.3f}")
            print(f"   CLIP adjustment: {classification_info.get('clip_adjustment_long', 0):+.3f}")
    except Exception as e:
        print(f"❌ Final Decision Classification Failed: {e}")
        decision = "unknown"
    
    # 7. Test Complete TJDE Analysis
    try:
        print(f"\n🚀 Running Complete TJDE v2 Analysis...")
        tjde_result = analyze_symbol_with_unified_tjde_v2(
            symbol, market_data, candles_15m, candles_5m
        )
        
        print(f"✅ TJDE v2 Complete Analysis:")
        score_value = tjde_result.get('final_score', 'N/A')
        if isinstance(score_value, (int, float)):
            print(f"   Final Score: {score_value:.4f}")
        else:
            print(f"   Final Score: {score_value}")
        print(f"   Decision: {tjde_result.get('decision', 'N/A')}")
        print(f"   Market Phase: {tjde_result.get('market_phase', 'N/A')}")
        
        return tjde_result
        
    except Exception as e:
        print(f"❌ Complete TJDE Analysis Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_multiple_tokens():
    """Compare TJDE scoring across multiple tokens to identify uniformity issues"""
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'SOLUSDT']
    results = {}
    
    print(f"\n{'='*80}")
    print(f"🔬 COMPARING TJDE SCORING ACROSS MULTIPLE TOKENS")
    print(f"{'='*80}")
    
    for symbol in test_symbols:
        result = debug_tjde_component(symbol, verbose=False)
        if result:
            results[symbol] = result
    
    # Analysis of results
    print(f"\n{'='*80}")
    print(f"📊 SCORING COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    scores = [r.get('final_score', 0) for r in results.values()]
    decisions = [r.get('decision', 'unknown') for r in results.values()]
    phases = [r.get('market_phase', 'unknown') for r in results.values()]
    
    print(f"Scores: {[f'{s:.4f}' for s in scores]}")
    print(f"Decisions: {decisions}")
    print(f"Phases: {phases}")
    
    # Check for uniformity
    unique_scores = len(set(f'{s:.4f}' for s in scores))
    unique_decisions = len(set(decisions))
    unique_phases = len(set(phases))
    
    print(f"\n🎯 UNIFORMITY CHECK:")
    print(f"   Unique Scores: {unique_scores}/{len(scores)} ({'❌ IDENTICAL' if unique_scores == 1 else '✅ VARIED'})")
    print(f"   Unique Decisions: {unique_decisions}/{len(decisions)} ({'❌ IDENTICAL' if unique_decisions == 1 else '✅ VARIED'})")
    print(f"   Unique Phases: {unique_phases}/{len(phases)} ({'❌ IDENTICAL' if unique_phases == 1 else '✅ VARIED'})")
    
    if unique_scores == 1:
        print(f"\n🚨 CRITICAL ISSUE: All tokens have identical score {scores[0]:.4f}")
        print(f"   This suggests a fundamental scoring bug!")
    
    return results

if __name__ == "__main__":
    print("🔍 TJDE v2 Scoring Debug Tool")
    print("="*50)
    
    # Test single token in detail
    debug_tjde_component('BTCUSDT', verbose=True)
    
    # Compare multiple tokens
    compare_multiple_tokens()
    
    print(f"\n✅ Debug analysis complete!")