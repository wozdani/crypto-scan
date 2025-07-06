#!/usr/bin/env python3
"""
TJDE Scoring Layer Order Fixes
Addresses three critical optimization issues:

1. Proper scoring layer order with fallback logic
2. Market phase modifier fix for proper phase detection
3. Chart generation threshold enforcement (>= 0.6)
4. BONUS: GPT setup scoring booster integration
"""

import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

def fix_scoring_layer_order():
    """
    Fix 1: Implement proper scoring layer order:
    1. AI-EYE → 2. HTF Overlay → 3. Trap Detector → 4. Future Mapping
    → if none >= 0.4, then 5. Legacy Scoring fallback
    """
    return """
    # PROPER SCORING LAYER ORDER IMPLEMENTATION
    
    # === PHASE 1: ADVANCED MODULES (1-4) ===
    advanced_score = 0.0
    advanced_modules_active = 0
    
    # Module 1: AI-EYE
    if ai_label and ai_label.get("confidence", 0) >= 0.5:
        ai_score = score_from_ai_label(ai_label, market_phase)
        advanced_score += ai_score
        if ai_score != 0.0:
            advanced_modules_active += 1
    
    # Module 2: HTF Overlay  
    if htf_candles and len(htf_candles) >= 30:
        htf_score = score_from_htf_overlay(htf_candles, ai_label)
        advanced_score += htf_score
        if htf_score != 0.0:
            advanced_modules_active += 1
    
    # Module 3: Trap Detector
    if candles and len(candles) >= 20:
        trap_score = score_from_trap_detector(candles, ai_label)
        advanced_score += trap_score
        if trap_score != 0.0:
            advanced_modules_active += 1
    
    # Module 4: Future Mapping
    if candles and len(candles) >= 20:
        future_score = score_from_future_mapping(candles, ema_data)
        advanced_score += future_score
        if future_score != 0.0:
            advanced_modules_active += 1
    
    # === FALLBACK LOGIC ===
    if advanced_score < 0.4 and advanced_modules_active == 0:
        # Phase 2: Legacy Scoring Fallback
        print(f"[FALLBACK] {symbol}: Advanced modules insufficient, activating legacy scoring")
        legacy_score = legacy_volume_score + legacy_orderbook_score + legacy_cluster_score + legacy_psychology_score
        final_score = max(advanced_score, legacy_score)
    else:
        # Advanced modules sufficient
        final_score = advanced_score
        print(f"[ADVANCED] {symbol}: Using advanced modules score: {final_score:.4f}")
    """

def fix_market_phase_modifier():
    """
    Fix 2: Market phase modifier always returns 0.000
    Issue: simulate_trader_decision_advanced() doesn't pass recognized phase
    """
    return {
        "phase_mapping": {
            # GPT Labels → Market Phase Modifiers
            "breakout_pattern": "breakout",
            "momentum_follow": "bullish-momentum", 
            "trend_continuation": "trend-following",
            "reversal_pattern": "reversal",
            "range_trading": "range",
            "pullback_in_trend": "pullback-in-trend",
            "retest_confirmation": "retest-confirmation"
        },
        "modifiers": {
            "breakout": +0.15,
            "bullish-momentum": +0.12,
            "trend-following": +0.10,
            "retest-confirmation": +0.08,
            "pullback-in-trend": +0.05,
            "range": +0.00,
            "reversal": -0.05,
            "basic_screening": +0.00
        },
        "implementation": """
        # Extract recognized phase from GPT analysis or AI label
        gpt_setup = gpt_analysis.get('setup_label', '') if gpt_analysis else ''
        ai_pattern = ai_label.get('label', '') if ai_label else ''
        clip_pattern = clip_info.get('pattern', '') if clip_info else ''
        
        # Priority: GPT > AI > CLIP > fallback
        recognized_phase = gpt_setup or ai_pattern or clip_pattern or 'basic_screening'
        
        # Map to market phase modifier
        phase_mapping = {
            'breakout_pattern': 'breakout',
            'momentum_follow': 'bullish-momentum',
            'trend_continuation': 'trend-following'
        }
        
        market_phase = phase_mapping.get(recognized_phase, recognized_phase)
        modifier = market_phase_modifier(market_phase, trend_strength, volatility_ratio, volume_range)
        """
    }

def fix_chart_generation_threshold():
    """
    Fix 3: Ensure chart generation only for score >= 0.6 OR decision == 'enter'
    """
    return """
    def should_generate_chart(symbol: str, tjde_score: float, tjde_decision: str) -> Dict[str, Any]:
        # Strong signals always generate charts
        if tjde_decision in ['enter', 'long', 'short']:
            return {
                "generate": True,
                "reason": f"Strong signal ({tjde_decision}, score: {tjde_score:.3f})"
            }
        
        # High score threshold check
        if tjde_score >= 0.6:
            return {
                "generate": True, 
                "reason": f"High score threshold (score: {tjde_score:.3f})"
            }
        
        # Skip low-value signals
        return {
            "generate": False,
            "reason": f"Below threshold (score: {tjde_score:.3f}, decision: {tjde_decision})"
        }
    """

def implement_gpt_setup_booster():
    """
    BONUS: GPT setup scoring booster based on setup quality
    """
    return {
        "setup_bonuses": {
            # High-quality setups
            "momentum_follow": +0.10,
            "breakout_pattern": +0.08,
            "retest_confirmation": +0.07,
            "trend_continuation": +0.06,
            
            # Moderate-quality setups  
            "pullback_in_trend": +0.04,
            "range_trading": +0.02,
            
            # Neutral/negative setups
            "reversal_pattern": +0.00,
            "setup_analysis": -0.05,
            "no_clear_pattern": -0.03,
            "unknown": -0.02
        },
        "implementation": """
        # Extract GPT setup from chart analysis
        gpt_setup = gpt_analysis.get('setup_label', '') if gpt_analysis else ''
        
        # Apply setup bonus
        setup_bonuses = {
            'momentum_follow': +0.10,
            'breakout_pattern': +0.08,
            'retest_confirmation': +0.07,
            'trend_continuation': +0.06,
            'pullback_in_trend': +0.04,
            'range_trading': +0.02,
            'reversal_pattern': +0.00,
            'setup_analysis': -0.05,
            'no_clear_pattern': -0.03,
            'unknown': -0.02
        }
        
        setup_bonus = setup_bonuses.get(gpt_setup, 0.0)
        
        if setup_bonus != 0.0:
            final_score += setup_bonus
            print(f"[SETUP BONUS] {symbol}: +{setup_bonus:.3f} for {gpt_setup}")
        """
    }

def create_unified_fix():
    """
    Create comprehensive fix integrating all three issues
    """
    fixes = {
        "scoring_layer_order": fix_scoring_layer_order(),
        "market_phase_modifier": fix_market_phase_modifier(),
        "chart_generation_threshold": fix_chart_generation_threshold(),
        "gpt_setup_booster": implement_gpt_setup_booster()
    }
    
    return fixes

def apply_fixes_to_unified_engine():
    """
    Apply all fixes to unified_scoring_engine.py
    """
    fixes_summary = {
        "fix_1_scoring_order": {
            "description": "Proper layer order: AI-EYE → HTF → Trap → Future → (fallback: Legacy)",
            "target_files": ["unified_scoring_engine.py"],
            "status": "ready_to_implement"
        },
        "fix_2_phase_modifier": {
            "description": "Fix market_phase_modifier always returning 0.000",
            "target_files": ["scan_token_async.py", "utils/market_phase.py"],
            "status": "ready_to_implement"
        },
        "fix_3_chart_threshold": {
            "description": "Enforce chart generation threshold >= 0.6 OR decision == 'enter'",
            "target_files": ["chart_generator.py", "two_stage_tjde_system.py"],
            "status": "ready_to_implement"
        },
        "bonus_setup_booster": {
            "description": "GPT setup quality bonuses (+0.10 for momentum_follow, +0.08 for breakout_pattern)",
            "target_files": ["unified_scoring_engine.py"],
            "status": "ready_to_implement"
        }
    }
    
    return fixes_summary

if __name__ == "__main__":
    print("🔧 TJDE SCORING LAYER ORDER FIXES")
    print("=" * 50)
    
    fixes = create_unified_fix()
    summary = apply_fixes_to_unified_engine()
    
    print("\n✅ FIXES READY FOR IMPLEMENTATION:")
    for fix_name, fix_info in summary.items():
        print(f"  • {fix_name}: {fix_info['description']}")
    
    print(f"\n🎯 Total fixes: {len(summary)}")
    print("📝 All fixes documented and ready for deployment")