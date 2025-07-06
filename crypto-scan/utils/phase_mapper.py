#!/usr/bin/env python3
"""
Phase Mapper - Maps GPT analysis results to market phase modifiers
Fixes the issue where market_phase_modifier always returns 0.000
"""

import logging
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

class PhaseMapper:
    """Maps GPT setup labels to market phases with proper modifiers"""
    
    def __init__(self):
        # GPT Setup Labels → Market Phases mapping
        self.phase_mapping = {
            # High-confidence bullish setups
            "breakout_pattern": "breakout",
            "momentum_follow": "bullish-momentum", 
            "trend_continuation": "trend-following",
            "retest_confirmation": "retest-confirmation",
            
            # Moderate confidence setups
            "pullback_in_trend": "pullback-in-trend",
            "range_trading": "range",
            
            # Lower confidence or bearish setups  
            "reversal_pattern": "reversal",
            "exhaustion_pattern": "exhaustion-pullback",
            
            # Invalid/unclear setups
            "setup_analysis": "basic_screening",
            "no_clear_pattern": "basic_screening", 
            "unknown": "basic_screening"
        }
        
        # Market Phase → Modifier values
        self.phase_modifiers = {
            # Bullish phases
            "breakout": +0.15,
            "bullish-momentum": +0.12,
            "trend-following": +0.10,
            "retest-confirmation": +0.08,
            "pullback-in-trend": +0.05,
            
            # Neutral phases
            "range": +0.00,
            "consolidation": +0.00,
            "basic_screening": +0.00,
            
            # Bearish phases
            "reversal": -0.05,
            "exhaustion-pullback": -0.08,
            "bear_trend": -0.10,
            
            # Fallback
            "unknown": +0.00,
            "insufficient_data": +0.00,
            "error": +0.00
        }
    
    def extract_recognized_phase(self, 
                                gpt_analysis: Optional[Dict] = None,
                                ai_label: Optional[Dict] = None,
                                clip_info: Optional[Dict] = None,
                                fallback_phase: str = "basic_screening") -> str:
        """
        Extract recognized phase from various sources with priority order:
        1. GPT analysis setup_label
        2. AI label pattern
        3. CLIP pattern
        4. Fallback phase
        """
        
        # Priority 1: GPT Analysis
        if gpt_analysis and isinstance(gpt_analysis, dict):
            gpt_setup = gpt_analysis.get('setup_label', '') or gpt_analysis.get('setup', '')
            if gpt_setup and gpt_setup != 'unknown':
                print(f"[PHASE MAPPER] Using GPT setup: {gpt_setup}")
                return gpt_setup
        
        # Priority 2: AI Label
        if ai_label and isinstance(ai_label, dict):
            ai_pattern = ai_label.get('label', '') or ai_label.get('pattern', '')
            if ai_pattern and ai_pattern != 'unknown':
                print(f"[PHASE MAPPER] Using AI pattern: {ai_pattern}")
                return ai_pattern
        
        # Priority 3: CLIP Info
        if clip_info and isinstance(clip_info, dict):
            clip_pattern = clip_info.get('pattern', '') or clip_info.get('label', '')
            if clip_pattern and clip_pattern != 'unknown':
                print(f"[PHASE MAPPER] Using CLIP pattern: {clip_pattern}")
                return clip_pattern
        
        # Fallback
        print(f"[PHASE MAPPER] Using fallback phase: {fallback_phase}")
        return fallback_phase
    
    def map_to_market_phase(self, recognized_phase: str) -> str:
        """
        Map recognized phase (GPT/AI/CLIP) to market phase
        """
        market_phase = self.phase_mapping.get(recognized_phase, recognized_phase)
        print(f"[PHASE MAPPER] {recognized_phase} → {market_phase}")
        return market_phase
    
    def get_phase_modifier(self, market_phase: str, 
                          trend_strength: float = 0.0,
                          volatility_ratio: float = 1.0,
                          volume_range: float = 1.0) -> float:
        """
        Get phase modifier with contextual adjustments
        """
        base_modifier = self.phase_modifiers.get(market_phase, 0.0)
        
        # Context adjustments
        context_bonus = 0.0
        
        # Strong trend enhancement
        if trend_strength > 0.8 and market_phase in ["breakout", "bullish-momentum", "trend-following"]:
            context_bonus += 0.03
            print(f"[PHASE MAPPER] Strong trend bonus: +0.03")
        
        # High volatility penalty for range/consolidation
        if volatility_ratio > 2.0 and market_phase in ["range", "consolidation"]:
            context_bonus -= 0.02
            print(f"[PHASE MAPPER] High volatility penalty: -0.02")
        
        final_modifier = base_modifier + context_bonus
        
        print(f"[PHASE MAPPER] Phase: {market_phase} → Base: {base_modifier:+.3f}, Context: {context_bonus:+.3f}, Final: {final_modifier:+.3f}")
        
        return final_modifier
    
    def process_complete_phase_detection(self,
                                       symbol: str,
                                       gpt_analysis: Optional[Dict] = None,
                                       ai_label: Optional[Dict] = None, 
                                       clip_info: Optional[Dict] = None,
                                       trend_strength: float = 0.0,
                                       volatility_ratio: float = 1.0,
                                       volume_range: float = 1.0) -> Dict:
        """
        Complete phase detection and modifier calculation
        """
        
        print(f"[PHASE MAPPER] Processing complete phase detection for {symbol}")
        
        # Step 1: Extract recognized phase
        recognized_phase = self.extract_recognized_phase(
            gpt_analysis=gpt_analysis,
            ai_label=ai_label,
            clip_info=clip_info
        )
        
        # Step 2: Map to market phase
        market_phase = self.map_to_market_phase(recognized_phase)
        
        # Step 3: Calculate modifier
        modifier = self.get_phase_modifier(
            market_phase=market_phase,
            trend_strength=trend_strength,
            volatility_ratio=volatility_ratio,
            volume_range=volume_range
        )
        
        result = {
            "symbol": symbol,
            "recognized_phase": recognized_phase,
            "market_phase": market_phase,
            "phase_modifier": modifier,
            "trend_strength": trend_strength,
            "volatility_ratio": volatility_ratio,
            "volume_range": volume_range
        }
        
        print(f"[PHASE MAPPER] Complete result for {symbol}: {result}")
        
        return result

# Global instance
phase_mapper = PhaseMapper()

def enhanced_market_phase_modifier(symbol: str,
                                 gpt_analysis: Optional[Dict] = None,
                                 ai_label: Optional[Dict] = None,
                                 clip_info: Optional[Dict] = None,
                                 trend_strength: float = 0.0,
                                 volatility_ratio: float = 1.0,
                                 volume_range: float = 1.0) -> float:
    """
    Enhanced market phase modifier that fixes the 0.000 issue
    
    Args:
        symbol: Trading symbol
        gpt_analysis: GPT analysis results with setup_label
        ai_label: AI pattern recognition results
        clip_info: CLIP visual pattern results
        trend_strength: Trend strength metric
        volatility_ratio: Volatility ratio
        volume_range: Volume range metric
        
    Returns:
        float: Phase modifier (-0.20 to +0.20)
    """
    
    try:
        result = phase_mapper.process_complete_phase_detection(
            symbol=symbol,
            gpt_analysis=gpt_analysis,
            ai_label=ai_label,
            clip_info=clip_info,
            trend_strength=trend_strength,
            volatility_ratio=volatility_ratio,
            volume_range=volume_range
        )
        
        modifier = result["phase_modifier"]
        
        print(f"[ENHANCED PHASE MODIFIER] {symbol}: {modifier:+.3f} (phase: {result['market_phase']})")
        
        return modifier
        
    except Exception as e:
        logger.error(f"Enhanced phase modifier failed for {symbol}: {e}")
        print(f"[ENHANCED PHASE MODIFIER ERROR] {symbol}: {e}")
        return 0.0

if __name__ == "__main__":
    print("🔧 Testing Phase Mapper")
    print("=" * 40)
    
    # Test cases
    test_cases = [
        {
            "symbol": "BTCUSDT",
            "gpt_analysis": {"setup_label": "breakout_pattern"},
            "expected_phase": "breakout",
            "expected_modifier": 0.15
        },
        {
            "symbol": "ETHUSDT", 
            "gpt_analysis": {"setup_label": "momentum_follow"},
            "expected_phase": "bullish-momentum",
            "expected_modifier": 0.12
        },
        {
            "symbol": "ADAUSDT",
            "gpt_analysis": {"setup_label": "setup_analysis"},
            "expected_phase": "basic_screening", 
            "expected_modifier": 0.0
        }
    ]
    
    mapper = PhaseMapper()
    
    for test in test_cases:
        print(f"\n🧪 Testing {test['symbol']}")
        result = mapper.process_complete_phase_detection(
            symbol=test['symbol'],
            gpt_analysis=test['gpt_analysis']
        )
        
        phase_match = result['market_phase'] == test['expected_phase']
        modifier_match = abs(result['phase_modifier'] - test['expected_modifier']) < 0.01
        
        print(f"  ✅ Phase: {phase_match} ({result['market_phase']})")
        print(f"  ✅ Modifier: {modifier_match} ({result['phase_modifier']:+.3f})")
    
    print(f"\n🎯 Phase Mapper ready for integration")