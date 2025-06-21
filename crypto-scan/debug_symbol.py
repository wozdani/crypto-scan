#!/usr/bin/env python3
"""
Debug Symbol Tool - Detailed analysis for specific symbol

Usage: python debug_symbol.py SYMBOL
Example: python debug_symbol.py BTCUSDT

Provides step-by-step analysis breakdown for troubleshooting and optimization.
"""

import sys
import os
import json
from datetime import datetime, timezone
from trader_ai_engine import analyze_symbol_with_trader_ai
from utils.safe_candles import get_candles

def debug_single_symbol(symbol: str, enable_detailed_logs: bool = True):
    """
    Comprehensive debug analysis for a single symbol
    
    Args:
        symbol: Trading symbol to analyze
        enable_detailed_logs: Whether to save detailed step-by-step logs
    """
    print(f"🔍 DEBUG ANALYSIS FOR {symbol}")
    print("=" * 50)
    
    try:
        # Get recent candle data
        print("📊 Fetching candle data...")
        candles = get_candles(symbol, interval="15", limit=96)
        
        if not candles or len(candles) < 20:
            print(f"❌ Insufficient candle data for {symbol}: {len(candles) if candles else 0} candles")
            return
        
        print(f"✅ Fetched {len(candles)} candles")
        print(f"📈 Price range: {candles[0][4]} → {candles[-1][4]} (latest)")
        
        # Run comprehensive analysis
        print("\n🧠 Running Trader AI Engine analysis...")
        result = analyze_symbol_with_trader_ai(
            symbol, 
            candles, 
            market_data=None, 
            enable_description=True
        )
        
        # Display comprehensive results
        print("\n" + "=" * 50)
        print("📋 ANALYSIS RESULTS")
        print("=" * 50)
        
        print(f"🎯 Decision: {result['decision'].upper()}")
        print(f"📊 Final Score: {result['final_score']:.3f}")
        print(f"🏆 Quality Grade: {result['quality_grade']}")
        print(f"🔥 Confidence: {result['confidence']:.3f}")
        print(f"🌍 Market Context: {result['market_context']}")
        
        if 'score_breakdown' in result:
            print("\n📈 SCORE BREAKDOWN:")
            for component, score in result['score_breakdown'].items():
                component_name = component.replace('_', ' ').title()
                print(f"  {component_name}: {score:.3f}")
        
        if 'scoring_details' in result:
            weights = result['scoring_details'].get('weights_used', {})
            context_adj = result['scoring_details'].get('context_adjustment', 'unknown')
            print(f"\n⚙️  Context Adjustment: {context_adj}")
            print("⚖️  Weights Used:")
            for component, weight in weights.items():
                component_name = component.replace('_', ' ').title()
                print(f"  {component_name}: {weight:.3f}")
        
        print(f"\n💡 REASONS:")
        for i, reason in enumerate(result.get('reasons', [])[:8], 1):
            print(f"  {i}. {reason.replace('_', ' ').title()}")
        
        if result.get('red_flags'):
            print(f"\n⚠️  RED FLAGS:")
            for flag in result['red_flags']:
                print(f"  ⚠️  {flag.replace('_', ' ').title()}")
        
        if result.get('description'):
            print(f"\n📝 NATURAL DESCRIPTION:")
            print(f"  {result['description']}")
        
        # Alert assessment
        print("\n" + "=" * 50)
        print("🚨 ALERT ASSESSMENT")
        print("=" * 50)
        
        score = result['final_score']
        confidence = result['confidence']
        
        if score >= 0.75 and confidence >= 0.3:
            print("🚀 HIGH-QUALITY SETUP - Would trigger Telegram alert!")
            print(f"   Score: {score:.3f} ≥ 0.75 ✅")
            print(f"   Confidence: {confidence:.3f} ≥ 0.3 ✅")
        else:
            print("ℹ️  Setup does not meet alert criteria:")
            if score < 0.75:
                print(f"   Score: {score:.3f} < 0.75 ❌")
            else:
                print(f"   Score: {score:.3f} ≥ 0.75 ✅")
            
            if confidence < 0.3:
                print(f"   Confidence: {confidence:.3f} < 0.3 ❌")
            else:
                print(f"   Confidence: {confidence:.3f} ≥ 0.3 ✅")
        
        # Save detailed debug log
        if enable_detailed_logs:
            debug_log_entry = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_type": "detailed_debug",
                "candles_count": len(candles),
                "price_range": {
                    "start": float(candles[0][4]),
                    "end": float(candles[-1][4]),
                    "change_pct": ((float(candles[-1][4]) - float(candles[0][4])) / float(candles[0][4])) * 100
                },
                "complete_result": result,
                "alert_eligible": score >= 0.75 and confidence >= 0.3
            }
            
            os.makedirs("logs", exist_ok=True)
            debug_file = f"logs/debug_symbol_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(debug_log_entry, f, indent=2)
            
            print(f"\n💾 Detailed debug log saved: {debug_file}")
        
        print("\n✅ Debug analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during debug analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for debug symbol tool"""
    if len(sys.argv) != 2:
        print("Usage: python debug_symbol.py SYMBOL")
        print("Example: python debug_symbol.py BTCUSDT")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    debug_single_symbol(symbol)


if __name__ == "__main__":
    main()