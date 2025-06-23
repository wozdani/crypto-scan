#!/usr/bin/env python3
"""
Test ZEREBRO TraderWeightedDecisionEngine
Testuje nowy weighted scoring system dla ZEREBROUSDT

Usage: python test_zerebro_weighted.py
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trader_ai_engine import analyze_symbol_with_trader_ai
from utils.bybit_candles import get_candles


def test_zerebro_weighted_scoring(symbol="ZEREBROUSDT"):
    """
    Test TraderWeightedDecisionEngine na ZEREBROUSDT
    
    Args:
        symbol: Trading symbol to test
        
    Returns:
        dict: Analysis result with weighted scoring
    """
    print(f"🧠 Testing TraderWeightedDecisionEngine for {symbol}")
    print("=" * 70)
    
    try:
        # Step 1: Fetch real market data
        print("📊 Fetching 15m candles...")
        candles = get_candles(symbol, interval="15", limit=96)
        
        if not candles or len(candles) < 20:
            print(f"❌ Insufficient candle data: {len(candles) if candles else 0} candles")
            print("⚠️  Check API keys and network connectivity")
            return None
        
        print(f"✅ Retrieved {len(candles)} candles")
        print(f"📈 Price range: {candles[0][4]} → {candles[-1][4]}")
        
        # Step 2: Optional orderbook data
        print("\n🧾 Attempting to fetch orderbook data...")
        market_data = None
        try:
            from utils.bybit_orderbook import get_orderbook_snapshot
            orderbook_data = get_orderbook_snapshot(symbol)
            if orderbook_data:
                market_data = {'orderbook': orderbook_data}
                print("✅ Orderbook data retrieved")
            else:
                print("⚠️  No orderbook data available")
        except Exception as ob_error:
            print(f"⚠️  Orderbook fetch failed: {ob_error}")
        
        # Step 3: Run TraderWeightedDecisionEngine analysis
        print("\n🧠 Running TraderWeightedDecisionEngine analysis...")
        result = analyze_symbol_with_trader_ai(
            symbol, 
            candles, 
            market_data, 
            enable_description=True
        )
        
        if not result:
            print("❌ Analysis failed")
            return None
        
        # Step 4: Display comprehensive results
        print("\n" + "=" * 70)
        print("📊 TRADERWEIGTEDDECISIONENGINE RESULTS")
        print("=" * 70)
        
        decision_info = result.get("decision_info", {})
        
        print(f"🎯 DECISION: {decision_info.get('decision', 'unknown').upper()}")
        print(f"📈 FINAL SCORE: {decision_info.get('final_score', 0.0):.4f}")
        print(f"🎖️  CONFIDENCE: {decision_info.get('confidence', 0.0):.4f}")
        print(f"⭐ QUALITY GRADE: {decision_info.get('quality_grade', 'unknown').upper()}")
        
        # Score breakdown display
        score_breakdown = decision_info.get("score_breakdown", {})
        if score_breakdown:
            print(f"\n📊 WEIGHTED SCORE BREAKDOWN:")
            for feature, score in score_breakdown.items():
                print(f"  {feature:25} = {score:+.4f}")
        
        # Weights used
        weights_used = decision_info.get("weights_used", {})
        if weights_used:
            print(f"\n⚖️  WEIGHTS APPLIED:")
            for weight_name, weight_value in weights_used.items():
                print(f"  {weight_name:25} = {weight_value:.2f}")
        
        # Reasons
        reasons = decision_info.get("reasons", [])
        if reasons:
            print(f"\n🔍 DECISION REASONS:")
            for i, reason in enumerate(reasons[:5], 1):
                print(f"  {i}. {reason}")
        
        # Market context
        market_context = result.get("market_context", "unknown")
        candle_behavior = result.get("candle_behavior", {})
        orderbook_info = result.get("orderbook_info", {})
        
        print(f"\n📈 MARKET CONTEXT: {market_context.upper()}")
        print(f"🕯️  CANDLE PATTERN: {candle_behavior.get('pattern', 'unknown')}")
        print(f"📊 MOMENTUM: {candle_behavior.get('momentum', 'unknown')}")
        print(f"🧾 ORDERBOOK: Bids={orderbook_info.get('bids_layered', False)}, Spoofing={orderbook_info.get('spoofing_suspected', False)}")
        
        # Alert assessment
        final_score = decision_info.get('final_score', 0.0)
        confidence = decision_info.get('confidence', 0.0)
        decision = decision_info.get('decision', 'avoid')
        
        print(f"\n🚨 ALERT ASSESSMENT:")
        if final_score >= 0.75 and confidence >= 0.3:
            print(f"  ✅ WOULD TRIGGER ALERT (Score: {final_score:.3f}, Confidence: {confidence:.3f})")
        elif decision == "join_trend":
            print(f"  ⚠️  JOIN_TREND but below alert threshold")
        else:
            print(f"  ❌ NO ALERT (Score: {final_score:.3f}, Decision: {decision})")
        
        # Step 5: Save detailed results
        save_weighted_test_results(symbol, result, candles)
        
        return result
        
    except Exception as e:
        print(f"❌ Error during weighted analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_weighted_test_results(symbol: str, result: dict, candles: list):
    """Save test results to JSON Lines format"""
    try:
        os.makedirs("logs", exist_ok=True)
        
        # Prepare comprehensive log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "TraderWeightedDecisionEngine",
            "symbol": symbol,
            "candle_count": len(candles),
            "price_range": {
                "start": candles[0][4] if candles else None,
                "end": candles[-1][4] if candles else None
            },
            "analysis_result": result
        }
        
        # Save to JSON Lines
        with open("logs/test_zerebro_weighted_log.jsonl", "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(log_entry, ensure_ascii=False)}\n")
        
        # Save readable version
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        readable_file = f"logs/test_zerebro_weighted_readable_{timestamp_str}.txt"
        
        with open(readable_file, "w", encoding="utf-8") as f:
            f.write(f"TraderWeightedDecisionEngine Test Results\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Candles: {len(candles)}\n")
            f.write("=" * 60 + "\n\n")
            
            decision_info = result.get("decision_info", {})
            f.write(f"Decision: {decision_info.get('decision', 'unknown')}\n")
            f.write(f"Final Score: {decision_info.get('final_score', 0.0):.4f}\n")
            f.write(f"Confidence: {decision_info.get('confidence', 0.0):.4f}\n")
            f.write(f"Quality Grade: {decision_info.get('quality_grade', 'unknown')}\n\n")
            
            # Score breakdown
            score_breakdown = decision_info.get("score_breakdown", {})
            if score_breakdown:
                f.write("Weighted Score Breakdown:\n")
                for feature, score in score_breakdown.items():
                    f.write(f"  {feature}: {score:+.4f}\n")
                f.write("\n")
            
            # Reasons
            reasons = decision_info.get("reasons", [])
            if reasons:
                f.write("Decision Reasons:\n")
                for i, reason in enumerate(reasons, 1):
                    f.write(f"  {i}. {reason}\n")
                f.write("\n")
            
            f.write(f"Full Analysis:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
        
        print(f"✅ Test results saved to:")
        print(f"  📄 logs/test_zerebro_weighted_log.jsonl")
        print(f"  📖 {readable_file}")
        
    except Exception as e:
        print(f"❌ Error saving test results: {e}")


if __name__ == "__main__":
    symbol = "ZEREBROUSDT"
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    
    print(f"🚀 Starting TraderWeightedDecisionEngine test for {symbol}")
    
    result = test_zerebro_weighted_scoring(symbol)
    
    if result:
        print(f"\n✅ TraderWeightedDecisionEngine test completed successfully!")
        print(f"📊 Check logs/test_zerebro_weighted_log.jsonl for detailed results")
    else:
        print(f"\n❌ TraderWeightedDecisionEngine test failed")
        sys.exit(1)