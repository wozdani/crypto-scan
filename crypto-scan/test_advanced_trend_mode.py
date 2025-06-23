#!/usr/bin/env python3
"""
Test Advanced Trend Mode Full Decision
Testuje rozbudowany system trend-mode z profesjonalną logiką tradera

Usage: python test_advanced_trend_mode.py [SYMBOL1] [SYMBOL2]
Default: ZEREBROUSDT LPTUSDT
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trader_ai_engine import simulate_trader_decision_advanced
from utils.safe_candles import get_candles


def test_advanced_trend_mode(symbols=None):
    """
    Test Advanced Trend Mode na wybranych symbolach
    
    Args:
        symbols: Lista symboli do testowania
        
    Returns:
        dict: Wyniki testów dla wszystkich symboli
    """
    if symbols is None:
        symbols = ["ZEREBROUSDT", "LPTUSDT"]
    
    print("🧠 Testing Advanced Trend Mode Full Decision System")
    print("=" * 80)
    print(f"📊 Testing symbols: {', '.join(symbols)}")
    print("=" * 80)
    
    results = {}
    
    for symbol in symbols:
        print(f"\n🔍 Testing {symbol}...")
        print("-" * 60)
        
        try:
            # Get market data
            print(f"📊 Fetching 15m candles for {symbol}...")
            candles = get_candles(symbol, interval="15", limit=50)
            
            if not candles or len(candles) < 20:
                print(f"❌ Insufficient candle data for {symbol}: {len(candles) if candles else 0} candles")
                results[symbol] = {"error": "insufficient_data"}
                continue
            
            print(f"✅ Retrieved {len(candles)} candles")
            
            # Optional orderbook data
            market_data = None
            try:
                from utils.bybit_orderbook import get_orderbook_snapshot
                orderbook_data = get_orderbook_snapshot(symbol)
                if orderbook_data:
                    market_data = {'orderbook': orderbook_data}
                    print("✅ Orderbook data retrieved")
                else:
                    print("⚠️  No orderbook data available")
            except Exception:
                print("⚠️  Orderbook fetch failed")
            
            # Run advanced analysis
            print(f"\n🧠 Running Advanced Trend Mode analysis for {symbol}...")
            result = simulate_trader_decision_advanced(
                symbol=symbol,
                candles=candles,
                market_data=market_data,
                current_price=candles[-1][4]
            )
            
            if not result:
                print(f"❌ Analysis failed for {symbol}")
                results[symbol] = {"error": "analysis_failed"}
                continue
            
            # Display results
            display_advanced_results(symbol, result)
            
            # Store results
            results[symbol] = result
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            results[symbol] = {"error": str(e)}
    
    # Save comprehensive results
    save_advanced_test_results(results)
    
    return results


def display_advanced_results(symbol: str, result: dict):
    """Display comprehensive results for a symbol"""
    try:
        print(f"\n" + "=" * 60)
        print(f"📊 ADVANCED TREND MODE RESULTS FOR {symbol}")
        print("=" * 60)
        
        # Main decision
        decision = result.get("decision", "unknown")
        final_score = result.get("final_score", 0.0)
        confidence = result.get("confidence", 0.0)
        quality_grade = result.get("quality_grade", "unknown")
        
        print(f"🎯 DECISION: {decision.upper()}")
        print(f"📈 FINAL SCORE: {final_score:.4f}")
        print(f"🎖️  CONFIDENCE: {confidence:.4f}")
        print(f"⭐ QUALITY GRADE: {quality_grade.upper()}")
        
        # Advanced components
        phase_analysis = result.get("phase_analysis", {})
        liquidity_analysis = result.get("liquidity_analysis", {})
        psychology_analysis = result.get("psychology_analysis", {})
        htf_analysis = result.get("htf_analysis", {})
        
        print(f"\n📊 ADVANCED COMPONENTS:")
        print(f"  🏗️  Market Phase: {phase_analysis.get('market_phase', 'unknown')}")
        print(f"  💧 Liquidity Score: {liquidity_analysis.get('liquidity_pattern_score', 0.0):.3f}")
        print(f"  🧠 Psychology Score: {psychology_analysis.get('psych_score', 0.0):.3f}")
        print(f"  ⏰ HTF Supportive: {htf_analysis.get('htf_supportive_score', 0.0):.3f}")
        
        # Phase details
        if phase_analysis:
            phase_details = phase_analysis.get("structure_details", {})
            print(f"\n🏗️  MARKET PHASE DETAILS:")
            print(f"  Phase: {phase_analysis.get('market_phase', 'unknown')}")
            print(f"  Confidence: {phase_analysis.get('confidence', 0.0):.3f}")
            if phase_details:
                print(f"  Current Price: {phase_details.get('current_price', 'N/A')}")
                print(f"  EMA21: {phase_details.get('current_ema21', 'N/A')}")
        
        # Liquidity details
        if liquidity_analysis:
            print(f"\n💧 LIQUIDITY ANALYSIS:")
            print(f"  Bid Stacking: {liquidity_analysis.get('bid_stacking', False)}")
            print(f"  Absorption: {liquidity_analysis.get('absorption_detected', False)}")
            print(f"  Data Quality: {liquidity_analysis.get('data_quality', 'unknown')}")
        
        # Psychology flags
        psych_flags = psychology_analysis.get("psychological_flags", [])
        if psych_flags:
            print(f"\n🧠 PSYCHOLOGICAL FLAGS:")
            for flag in psych_flags:
                print(f"  ⚠️  {flag}")
        else:
            print(f"\n🧠 PSYCHOLOGY: Clean move detected")
        
        # HTF confirmation
        if htf_analysis.get("data_available", False):
            print(f"\n⏰ HTF CONFIRMATION ({htf_analysis.get('htf_timeframe', 'unknown')}m):")
            print(f"  Trend Match: {htf_analysis.get('htf_trend_match', False)}")
            htf_details = htf_analysis.get('htf_details', {})
            if htf_details:
                ema_analysis = htf_details.get('ema_analysis', {})
                print(f"  EMA Slope: {ema_analysis.get('slope', 'unknown')}")
                green_analysis = htf_details.get('green_ratio_analysis', {})
                print(f"  Green Ratio: {green_analysis.get('green_ratio', 0.0):.3f}")
        
        # Weighted features breakdown
        advanced_features = result.get("advanced_features", {})
        if advanced_features:
            print(f"\n📊 WEIGHTED FEATURES BREAKDOWN:")
            for feature, value in advanced_features.items():
                print(f"  {feature:25} = {value:+.4f}")
        
        # Decision reasons
        reasons = result.get("reasons", [])
        if reasons:
            print(f"\n🔍 DECISION REASONS:")
            for i, reason in enumerate(reasons[:5], 1):
                print(f"  {i}. {reason}")
        
        # Alert assessment
        print(f"\n🚨 ALERT ASSESSMENT:")
        if final_score >= 0.75 and confidence >= 0.3:
            print(f"  ✅ WOULD TRIGGER ALERT (Score: {final_score:.3f}, Confidence: {confidence:.3f})")
        elif decision == "join_trend":
            print(f"  ⚠️  JOIN_TREND but below alert threshold")
        else:
            print(f"  ❌ NO ALERT (Score: {final_score:.3f}, Decision: {decision})")
        
    except Exception as e:
        print(f"❌ Error displaying results for {symbol}: {e}")


def save_advanced_test_results(results: dict):
    """Save comprehensive test results"""
    try:
        os.makedirs("logs", exist_ok=True)
        
        # Create comprehensive test report
        timestamp = datetime.now()
        test_report = {
            "timestamp": timestamp.isoformat(),
            "test_type": "AdvancedTrendModeFullDecision",
            "engine_version": "AdvancedTraderWeightedDecisionEngine",
            "symbols_tested": list(results.keys()),
            "results": results,
            "summary": _generate_test_summary(results)
        }
        
        # Save to JSON Lines for machine processing
        with open("logs/test_advanced_trend_mode_log.jsonl", "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(test_report, ensure_ascii=False)}\n")
        
        # Save human-readable report
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        readable_file = f"logs/test_trend_mode_full_decision_{timestamp_str}.txt"
        
        with open(readable_file, "w", encoding="utf-8") as f:
            f.write("Advanced Trend Mode Full Decision Test Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {timestamp.isoformat()}\n")
            f.write(f"Engine: AdvancedTraderWeightedDecisionEngine\n")
            f.write(f"Symbols Tested: {', '.join(results.keys())}\n\n")
            
            # Summary
            summary = test_report["summary"]
            f.write("SUMMARY:\n")
            f.write(f"  Total Symbols: {summary['total_symbols']}\n")
            f.write(f"  Successful Tests: {summary['successful_tests']}\n")
            f.write(f"  Join Trend Decisions: {summary['join_trend_count']}\n")
            f.write(f"  Consider Entry: {summary['consider_entry_count']}\n")
            f.write(f"  Avoid Decisions: {summary['avoid_count']}\n")
            f.write(f"  Average Score: {summary['average_score']:.3f}\n\n")
            
            # Detailed results per symbol
            for symbol, result in results.items():
                f.write(f"\n{symbol}:\n")
                f.write("-" * 40 + "\n")
                
                if "error" in result:
                    f.write(f"  Error: {result['error']}\n")
                    continue
                
                f.write(f"  Decision: {result.get('decision', 'unknown')}\n")
                f.write(f"  Final Score: {result.get('final_score', 0.0):.4f}\n")
                f.write(f"  Confidence: {result.get('confidence', 0.0):.4f}\n")
                f.write(f"  Quality Grade: {result.get('quality_grade', 'unknown')}\n")
                
                # Advanced components
                phase_analysis = result.get("phase_analysis", {})
                f.write(f"  Market Phase: {phase_analysis.get('market_phase', 'unknown')}\n")
                
                psychology_analysis = result.get("psychology_analysis", {})
                psych_flags = psychology_analysis.get("psychological_flags", [])
                f.write(f"  Psychology Flags: {len(psych_flags)} ({', '.join(psych_flags[:3])})\n")
                
                htf_analysis = result.get("htf_analysis", {})
                f.write(f"  HTF Match: {htf_analysis.get('htf_trend_match', False)}\n")
                
                # Top reasons
                reasons = result.get("reasons", [])[:3]
                if reasons:
                    f.write(f"  Top Reasons: {'; '.join(reasons)}\n")
        
        print(f"\n✅ Test results saved to:")
        print(f"  📄 logs/test_advanced_trend_mode_log.jsonl")
        print(f"  📖 {readable_file}")
        
    except Exception as e:
        print(f"❌ Error saving test results: {e}")


def _generate_test_summary(results: dict) -> dict:
    """Generate summary statistics from test results"""
    try:
        total_symbols = len(results)
        successful_tests = len([r for r in results.values() if "error" not in r])
        
        decisions = [r.get("decision", "error") for r in results.values() if "error" not in r]
        join_trend_count = decisions.count("join_trend")
        consider_entry_count = decisions.count("consider_entry")
        avoid_count = decisions.count("avoid")
        
        scores = [r.get("final_score", 0.0) for r in results.values() if "error" not in r]
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "total_symbols": total_symbols,
            "successful_tests": successful_tests,
            "join_trend_count": join_trend_count,
            "consider_entry_count": consider_entry_count,
            "avoid_count": avoid_count,
            "average_score": average_score,
            "error_count": total_symbols - successful_tests
        }
        
    except Exception:
        return {"error": "summary_generation_failed"}


if __name__ == "__main__":
    # Parse command line arguments for symbols
    symbols = sys.argv[1:] if len(sys.argv) > 1 else None
    if symbols:
        symbols = [s.upper() for s in symbols]
    
    print(f"🚀 Starting Advanced Trend Mode Full Decision test")
    
    results = test_advanced_trend_mode(symbols)
    
    successful_tests = len([r for r in results.values() if "error" not in r])
    total_tests = len(results)
    
    print(f"\n" + "=" * 80)
    print(f"✅ Advanced Trend Mode test completed!")
    print(f"📊 Results: {successful_tests}/{total_tests} successful tests")
    print(f"📄 Check logs/test_trend_mode_full_decision_*.txt for detailed results")
    print("=" * 80)