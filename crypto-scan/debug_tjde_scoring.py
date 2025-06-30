#!/usr/bin/env python3
"""
TJDE Scoring Diagnostic Tool
Diagnoses TJDE scoring calibration issues and provides verbose component analysis
"""

import asyncio
import json
import sys
from typing import Dict, List
from scan_token_async import scan_token_async
from trader_ai_engine import simulate_trader_decision_advanced

class TJDEScoringDiagnostic:
    """TJDE scoring diagnostic and calibration tool"""
    
    def __init__(self):
        self.diagnostic_results = []
        
    async def diagnose_token_scoring(self, symbol: str) -> Dict:
        """Diagnose TJDE scoring for a single token with verbose output"""
        print(f"\n🔍 [TJDE DIAGNOSTIC] Starting comprehensive analysis for {symbol}")
        print("=" * 70)
        
        try:
            # Get full token scan data
            token_data = await scan_token_async(symbol)
            if not token_data:
                print(f"❌ Failed to get token data for {symbol}")
                return {"error": "No token data"}
                
            print(f"✅ Token data retrieved for {symbol}")
            print(f"   - 15M candles: {len(token_data.get('candles_15m', []))}")
            print(f"   - 5M candles: {len(token_data.get('candles_5m', []))}")
            print(f"   - Price: ${token_data.get('price_usd', 0):.6f}")
            print(f"   - Volume: ${token_data.get('volume_24h', 0):,.0f}")
            
            # Extract signals from token data
            signals = {
                'market_phase': token_data.get('market_phase', 'trend-following'),
                'trend_strength': token_data.get('trend_strength', 0.5),
                'pullback_quality': token_data.get('pullback_quality', 0.3),
                'support_reaction': token_data.get('support_reaction', 0.3),
                'liquidity_pattern_score': token_data.get('liquidity_pattern_score', 0.2),
                'psych_score': token_data.get('psych_score', 0.5),
                'htf_supportive_score': token_data.get('htf_supportive_score', 0.3),
                'market_phase_modifier': token_data.get('market_phase_modifier', 0.0),
                'volume_behavior': token_data.get('volume_behavior', 'neutral'),
                'price_action_pattern': token_data.get('price_action_pattern', 'continuation'),
                'htf_trend_match': token_data.get('htf_trend_match', True),
            }
            
            print(f"\n📊 [SIGNAL EXTRACTION] Base signals extracted:")
            for key, value in signals.items():
                if isinstance(value, float):
                    print(f"   - {key}: {value:.3f}")
                else:
                    print(f"   - {key}: {value}")
            
            # Run verbose simulate_trader_decision_advanced
            print(f"\n🧠 [VERBOSE ANALYSIS] Running simulate_trader_decision_advanced(verbose=True)")
            decision_result = simulate_trader_decision_advanced(
                symbol=symbol,
                market_data=token_data,
                signals=signals,
                debug_info={"verbose": True}
            )
            
            print(f"\n📋 [DECISION RESULT] Final result for {symbol}:")
            print(f"   - TJDE Score: {decision_result.get('score', 0):.3f}")
            print(f"   - Decision: {decision_result.get('decision', 'unknown')}")
            print(f"   - Grade: {decision_result.get('grade', 'unknown')}")
            print(f"   - CLIP Confidence: {decision_result.get('clip_confidence', 0):.3f}")
            
            # Analyze component contributions
            self._analyze_component_contributions(decision_result, signals)
            
            # Check for calibration issues
            issues = self._identify_calibration_issues(decision_result, signals)
            if issues:
                print(f"\n⚠️ [CALIBRATION ISSUES] Found {len(issues)} issues:")
                for i, issue in enumerate(issues, 1):
                    print(f"   {i}. {issue}")
            
            diagnostic_data = {
                "symbol": symbol,
                "tjde_score": decision_result.get('score', 0),
                "decision": decision_result.get('decision', 'unknown'),
                "grade": decision_result.get('grade', 'unknown'),
                "signals": signals,
                "issues": issues,
                "clip_confidence": decision_result.get('clip_confidence', 0),
                "market_phase": signals.get('market_phase', 'unknown')
            }
            
            self.diagnostic_results.append(diagnostic_data)
            return diagnostic_data
            
        except Exception as e:
            print(f"❌ [DIAGNOSTIC ERROR] Failed to diagnose {symbol}: {e}")
            return {"error": str(e)}
    
    def _analyze_component_contributions(self, decision_result: Dict, signals: Dict):
        """Analyze individual component contributions to final score"""
        print(f"\n🔬 [COMPONENT ANALYSIS] Breaking down score components:")
        
        # Load weights to calculate individual contributions
        try:
            from utils.scoring import load_tjde_weights, apply_phase_adjustments
            
            market_phase = signals.get('market_phase', 'trend-following')
            base_weights = load_tjde_weights()
            weights = apply_phase_adjustments(base_weights, market_phase)
            
            print(f"   Phase-adjusted weights for '{market_phase}':")
            for key, weight in weights.items():
                print(f"     - {key}: {weight:.3f}")
            
            # Calculate individual contributions
            components = {
                'trend_strength': signals.get('trend_strength', 0.5),
                'pullback_quality': signals.get('pullback_quality', 0.3),
                'support_reaction': signals.get('support_reaction', 0.3),
                'clip_confidence_score': decision_result.get('clip_confidence', 0),
                'liquidity_pattern_score': signals.get('liquidity_pattern_score', 0.2),
                'psych_score': signals.get('psych_score', 0.5),
                'htf_supportive_score': signals.get('htf_supportive_score', 0.3),
                'market_phase_modifier': signals.get('market_phase_modifier', 0.0)
            }
            
            print(f"\n   Individual component contributions:")
            total_contribution = 0.0
            for component, value in components.items():
                weight = weights.get(component, 0.0)
                contribution = float(value) * float(weight)
                total_contribution += contribution
                print(f"     - {component}: {value:.3f} × {weight:.3f} = {contribution:.3f}")
            
            print(f"\n   Total calculated contribution: {total_contribution:.3f}")
            actual_score = decision_result.get('score', 0)
            print(f"   Actual TJDE score: {actual_score:.3f}")
            
            if abs(total_contribution - actual_score) > 0.01:
                print(f"   ⚠️ SCORE MISMATCH: {abs(total_contribution - actual_score):.3f} difference!")
            
        except Exception as e:
            print(f"   ❌ Component analysis failed: {e}")
    
    def _identify_calibration_issues(self, decision_result: Dict, signals: Dict) -> List[str]:
        """Identify potential calibration issues"""
        issues = []
        
        score = decision_result.get('score', 0)
        decision = decision_result.get('decision', 'unknown')
        
        # Issue 1: Very low maximum scores
        if score < 0.35:
            issues.append(f"Maximum score too low ({score:.3f}) - weights or booster issue")
        
        # Issue 2: Unknown decisions
        if decision == 'unknown':
            issues.append("Unknown decision - threshold calibration issue")
        
        # Issue 3: Component values too low
        trend_strength = signals.get('trend_strength', 0)
        if trend_strength < 0.2:
            issues.append(f"Trend strength too low ({trend_strength:.3f}) - baseline issue")
        
        pullback_quality = signals.get('pullback_quality', 0)
        if pullback_quality < 0.2:
            issues.append(f"Pullback quality too low ({pullback_quality:.3f}) - baseline issue")
        
        # Issue 4: CLIP confidence not contributing
        clip_confidence = decision_result.get('clip_confidence', 0)
        if clip_confidence > 0.5 and score < 0.6:
            issues.append(f"High CLIP confidence ({clip_confidence:.3f}) not boosting score enough")
        
        # Issue 5: Good setup but avoid decision
        if (signals.get('trend_strength', 0) > 0.6 and 
            signals.get('support_reaction', 0) > 0.5 and 
            decision == 'avoid'):
            issues.append("Strong technical setup but avoid decision - threshold too high")
        
        return issues
    
    async def run_comprehensive_diagnostic(self, symbols: List[str]):
        """Run comprehensive diagnostic on multiple symbols"""
        print(f"\n🚀 [COMPREHENSIVE DIAGNOSTIC] Analyzing {len(symbols)} symbols")
        print("=" * 80)
        
        for symbol in symbols:
            await self.diagnose_token_scoring(symbol)
            print("\n" + "=" * 80)
        
        # Summary analysis
        if self.diagnostic_results:
            self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate summary report of all diagnostics"""
        print(f"\n📊 [SUMMARY REPORT] Analysis of {len(self.diagnostic_results)} tokens")
        print("=" * 80)
        
        # Score distribution
        scores = [r.get('tjde_score', 0) for r in self.diagnostic_results if 'tjde_score' in r]
        if scores:
            print(f"   TJDE Score Statistics:")
            print(f"     - Maximum: {max(scores):.3f}")
            print(f"     - Minimum: {min(scores):.3f}")
            print(f"     - Average: {sum(scores)/len(scores):.3f}")
        
        # Decision distribution
        decisions = [r.get('decision', 'unknown') for r in self.diagnostic_results]
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        print(f"\n   Decision Distribution:")
        for decision, count in decision_counts.items():
            print(f"     - {decision}: {count}")
        
        # Common issues
        all_issues = []
        for result in self.diagnostic_results:
            all_issues.extend(result.get('issues', []))
        
        if all_issues:
            issue_counts = {}
            for issue in all_issues:
                issue_key = issue.split(' - ')[1] if ' - ' in issue else issue
                issue_counts[issue_key] = issue_counts.get(issue_key, 0) + 1
            
            print(f"\n   Most Common Issues:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"     - {issue}: {count} tokens")


async def main():
    """Main diagnostic function"""
    # Test symbols provided by user
    test_symbols = ["VELODROMEUSDT", "LOOKSUSDT", "DARKUSDT"]
    
    # Also test some symbols from recent scans
    additional_symbols = ["MAGICUSDT", "AXSUSDT", "FTTUSDT"]
    
    all_symbols = test_symbols + additional_symbols
    
    diagnostic = TJDEScoringDiagnostic()
    await diagnostic.run_comprehensive_diagnostic(all_symbols)
    
    # Save diagnostic results
    with open("tjde_diagnostic_results.json", "w") as f:
        json.dump(diagnostic.diagnostic_results, f, indent=2)
    
    print(f"\n💾 Diagnostic results saved to tjde_diagnostic_results.json")


if __name__ == "__main__":
    asyncio.run(main())