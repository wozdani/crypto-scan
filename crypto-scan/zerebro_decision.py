#!/usr/bin/env python3
"""
ZEREBRO Decision Analysis
Ręczny test analizy Trader AI Engine dla ZEREBROUSDT

Usage: python zerebro_decision.py
"""

import os
import json
from datetime import datetime, timezone
from trader_ai_engine import analyze_symbol_with_trader_ai
from utils.safe_candles import get_candles

def test_zerebro_analysis():
    """
    Przeprowadź pełną analizę ZEREBROUSDT przez Trader AI Engine
    """
    symbol = "ZEREBROUSDT"
    
    print(f"🔍 Testing Trader AI Engine for {symbol}")
    print("=" * 60)
    
    try:
        # Step 1: Pobranie świeczek 15m (fallback do przykładowych danych)
        print("📊 Fetching 15m candles...")
        try:
            candles = get_candles(symbol, interval="15", limit=96)
        except:
            candles = None
        
        if not candles or len(candles) < 20:
            print(f"⚠️  API unavailable, using sample ZEREBRO data for testing...")
            # Przykładowe dane świecowe symulujące silny setup pullback dla ZEREBROUSDT
            candles = generate_sample_zerebro_candles()
            print(f"✅ Using {len(candles)} sample candles"
        
        print(f"✅ Retrieved {len(candles)} candles")
        print(f"📈 Price range: {candles[0][4]} → {candles[-1][4]}")
        
        # Step 2: Pobranie danych orderbook (opcjonalne - może być None)
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
        
        # Step 3: Przeprowadzenie pełnej analizy przez Trader AI Engine
        print("\n🧠 Running comprehensive Trader AI analysis...")
        result = analyze_symbol_with_trader_ai(
            symbol, 
            candles, 
            market_data, 
            enable_description=True
        )
        
        # Step 4: Wyświetlenie wyników na konsoli
        print("\n" + "=" * 60)
        print("📋 TRADER AI ANALYSIS RESULTS")
        print("=" * 60)
        
        print(f"🎯 DECISION: {result['decision'].upper()}")
        print(f"📊 Final Score: {result['final_score']:.3f}")
        print(f"🔥 Confidence: {result['confidence']:.3f}")
        print(f"🏆 Quality Grade: {result['quality_grade']}")
        print(f"🌍 Market Context: {result['market_context']}")
        
        # Score breakdown
        if 'score_breakdown' in result:
            print(f"\n📈 SCORE BREAKDOWN:")
            for component, score in result['score_breakdown'].items():
                component_name = component.replace('_', ' ').title()
                print(f"  {component_name}: {score:.3f}")
        
        # Scoring details
        if 'scoring_details' in result:
            context_adj = result['scoring_details'].get('context_adjustment', 'unknown')
            print(f"\n⚙️  Context Adjustment: {context_adj}")
        
        # Reasons
        print(f"\n💡 REASONS:")
        for i, reason in enumerate(result.get('reasons', [])[:8], 1):
            print(f"  {i}. {reason.replace('_', ' ').title()}")
        
        # Red flags
        if result.get('red_flags'):
            print(f"\n⚠️  RED FLAGS:")
            for flag in result['red_flags']:
                print(f"  ⚠️  {flag.replace('_', ' ').title()}")
        
        # Natural description
        if result.get('description'):
            print(f"\n📝 TRADER DESCRIPTION:")
            print(f"  {result['description']}")
        
        # Alert assessment
        print(f"\n" + "=" * 60)
        print("🚨 ALERT ASSESSMENT")
        print("=" * 60)
        
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
        
        # Step 5: Zapis do pliku test log
        print(f"\n💾 Saving test results to log...")
        save_test_results(symbol, result, candles)
        
        print(f"\n✅ Test completed successfully for {symbol}")
        return result
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_test_results(symbol: str, result: dict, candles: list):
    """
    Zapisz wyniki testu do pliku JSON Lines
    """
    try:
        os.makedirs("logs", exist_ok=True)
        
        # Przygotuj dane do zapisu
        test_entry = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_type": "manual_zerebro_test",
            "candles_count": len(candles),
            "price_info": {
                "start_price": float(candles[0][4]),
                "end_price": float(candles[-1][4]),
                "price_change_pct": ((float(candles[-1][4]) - float(candles[0][4])) / float(candles[0][4])) * 100
            },
            "final_score": result.get('final_score', 0.0),
            "confidence": result.get('confidence', 0.0),
            "decision": result.get('decision', 'unknown'),
            "quality_grade": result.get('quality_grade', 'unknown'),
            "market_context": result.get('market_context', 'unknown'),
            "reasons": result.get('reasons', []),
            "red_flags": result.get('red_flags', []),
            "score_breakdown": result.get('score_breakdown', {}),
            "scoring_details": result.get('scoring_details', {}),
            "description": result.get('description', ''),
            "alert_eligible": result.get('final_score', 0) >= 0.75 and result.get('confidence', 0) >= 0.3
        }
        
        # Zapisz do pliku JSONL
        log_file = "logs/test_zerebro_decision_log.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(test_entry) + "\n")
        
        print(f"✅ Results saved to: {log_file}")
        
        # Także zapisz czytelną wersję
        readable_file = f"logs/test_zerebro_readable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(readable_file, "w", encoding="utf-8") as f:
            f.write(f"ZEREBRO TRADER AI TEST RESULTS\n")
            f.write(f"Timestamp: {test_entry['timestamp']}\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Candles: {test_entry['candles_count']}\n")
            f.write(f"Price: {test_entry['price_info']['start_price']} → {test_entry['price_info']['end_price']} ({test_entry['price_info']['price_change_pct']:+.2f}%)\n\n")
            f.write(f"DECISION: {test_entry['decision'].upper()}\n")
            f.write(f"Score: {test_entry['final_score']:.3f}\n")
            f.write(f"Confidence: {test_entry['confidence']:.3f}\n")
            f.write(f"Grade: {test_entry['quality_grade']}\n")
            f.write(f"Context: {test_entry['market_context']}\n\n")
            f.write(f"Score Breakdown:\n")
            for comp, score in test_entry['score_breakdown'].items():
                f.write(f"  {comp}: {score:.3f}\n")
            f.write(f"\nReasons:\n")
            for reason in test_entry['reasons']:
                f.write(f"  - {reason}\n")
            if test_entry['red_flags']:
                f.write(f"\nRed Flags:\n")
                for flag in test_entry['red_flags']:
                    f.write(f"  ⚠️ {flag}\n")
            f.write(f"\nDescription:\n{test_entry['description']}\n")
            f.write(f"\nAlert Eligible: {'YES' if test_entry['alert_eligible'] else 'NO'}\n")
        
        print(f"✅ Readable version: {readable_file}")
        
    except Exception as e:
        print(f"❌ Failed to save test results: {e}")


def main():
    """
    Główna funkcja testowa
    """
    print("🚀 ZEREBRO Trader AI Engine Test")
    print("Testing market analysis and decision making capabilities\n")
    
    result = test_zerebro_analysis()
    
    if result:
        print(f"\n🎯 FINAL VERDICT:")
        decision = result.get('decision', 'unknown')
        score = result.get('final_score', 0)
        grade = result.get('quality_grade', 'unknown')
        
        if decision == 'join_trend':
            print(f"✅ ZEREBROUSDT: JOIN TREND (Score: {score:.3f}, Grade: {grade})")
        elif decision == 'wait':
            print(f"⏳ ZEREBROUSDT: WAIT (Score: {score:.3f}, Grade: {grade})")
        else:
            print(f"❌ ZEREBROUSDT: AVOID (Score: {score:.3f}, Grade: {grade})")
    else:
        print("❌ Test failed - unable to analyze ZEREBROUSDT")


if __name__ == "__main__":
    main()