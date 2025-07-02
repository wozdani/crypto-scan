#!/usr/bin/env python3
"""
Unified TJDE Engine v2 - Enhanced Decision System
Główny silnik decyzyjny TJDE v2 łączący pre-pump i trend-mode w jedną spójną logikę.

Analizuje wykresy, etykiety VisionAI i GPT, wolumeny, fazę rynku i inne czynniki,
by podjąć decyzję typu: enter, avoid, scalp_entry, wait
"""

import json
import os
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

class UnifiedTJDEEngineV2:
    """
    Enhanced Unified TJDE Engine v2
    Single decision system for all market phases with advanced analysis
    """
    
    def __init__(self):
        self.scoring_profiles = self._load_scoring_profiles()
        print("[TJDE v2] Enhanced TJDE Engine v2 initialized")
    
    def _load_scoring_profiles(self) -> Dict:
        """Load phase-specific scoring profiles"""
        profiles = {}
        
        # Load existing profiles from v1
        profile_files = {
            "pre-pump": "data/weights/tjde_pre_pump_profile.json",
            "trend-following": "data/weights/tjde_trend_following_profile.json", 
            "consolidation": "data/weights/tjde_consolidation_profile.json",
            "breakout": "data/weights/tjde_breakout_profile.json"
        }
        
        for phase, filepath in profile_files.items():
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        profiles[phase] = json.load(f)
                    print(f"[TJDE v2] Loaded {phase} profile from {filepath}")
                else:
                    # Create enhanced default profile for v2
                    profiles[phase] = self._create_default_profile_v2(phase)
                    print(f"[TJDE v2] Created default v2 profile for {phase}")
            except Exception as e:
                print(f"[TJDE v2] Error loading {phase} profile: {e}")
                profiles[phase] = self._create_default_profile_v2(phase)
        
        return profiles
    
    def _create_default_profile_v2(self, phase: str) -> Dict:
        """Create enhanced default profile for TJDE v2"""
        base_profile = {
            "volume_structure": 0.20,
            "clip_confidence": 0.15,
            "gpt_label_match": 0.18,
            "liquidity_behavior": 0.12,
            "heatmap_window": 0.10,
            "pre_breakout_structure": 0.15,
            "market_phase_modifier": 0.10
        }
        
        # Phase-specific adjustments
        if phase == "pre-pump":
            base_profile.update({
                "pre_breakout_structure": 0.25,  # Higher weight for pre-pump
                "volume_structure": 0.15,
                "heatmap_window": 0.15
            })
        elif phase == "breakout":
            base_profile.update({
                "volume_structure": 0.30,  # Volume crucial for breakouts
                "clip_confidence": 0.20,
                "pre_breakout_structure": 0.10
            })
        elif phase == "trend-following":
            base_profile.update({
                "gpt_label_match": 0.25,  # GPT important for trend analysis
                "clip_confidence": 0.20
            })
        
        return base_profile

def unified_tjde_decision_engine_with_signals(token_data: Dict, market_phase: str, clip_result: Dict, 
                               gpt_label: str, volume_info: Dict, liquidity_info: Dict, signals: Dict = None) -> Tuple[str, float, Dict]:
    """
    Enhanced TJDE decision engine that uses pre-calculated signals for accurate momentum_follow scoring
    """
    try:
        engine = UnifiedTJDEEngineV2()
        
        # Get profile
        if market_phase == "pre-pump":
            profile = engine.scoring_profiles.get("pre-pump", engine.scoring_profiles["trend-following"])
        else:
            profile = engine.scoring_profiles.get(market_phase, engine.scoring_profiles["trend-following"])
        
        # Use signals if available, otherwise fallback to calculated components
        if signals:
            score_components = {
                "trend_strength": signals.get("trend_strength", 0.5),
                "pullback_quality": signals.get("pullback_quality", 0.5), 
                "support_reaction": signals.get("support_reaction_strength", 0.5),
                "volume_behavior_score": signals.get("volume_behavior_score", 0.5),
                "psych_score": signals.get("psych_score", 0.5),
                "htf_supportive_score": signals.get("htf_supportive_score", 0.5),
                "liquidity_pattern_score": signals.get("liquidity_behavior", 0.5),
                "clip_confidence": clip_result.get("confidence", 0.0) if clip_result else 0.0,
                "gpt_label_match": match_gpt_label(gpt_label, market_phase),
                "market_phase_modifier": get_phase_modifier(market_phase, token_data)
            }
            print(f"[TJDE v2 SIGNALS] {token_data.get('symbol', 'UNKNOWN')}: Using pre-calculated signals for enhanced scoring")
        else:
            # Fallback to original calculation method
            score_components = {
                "volume_structure": evaluate_volume_structure(volume_info),
                "clip_confidence": clip_result.get("confidence", 0.0) if clip_result else 0.0,
                "gpt_label_match": match_gpt_label(gpt_label, market_phase),
                "liquidity_behavior": analyze_liquidity_v2(liquidity_info),
                "heatmap_window": analyze_heatmap_exhaustion(token_data),
                "pre_breakout_structure": analyze_price_structure_v2(token_data),
                "market_phase_modifier": get_phase_modifier(market_phase, token_data)
            }
        
        # Calculate final score using profile weights
        final_score = 0.0
        for component, value in score_components.items():
            weight = profile.get(component, 0.1)
            final_score += value * weight
        
        # Normalize score
        final_score = max(0.0, min(1.0, final_score))
        
        # === GPT+CLIP PATTERN ALIGNMENT BOOSTER v2 ===
        original_score = final_score
        clip_confidence = clip_result.get("confidence", 0.0) if clip_result else 0.0
        
        # Trusted patterns that get score boost
        trusted_patterns = ["momentum_follow", "breakout-continuation", "trend-following", "trend_continuation"]
        
        # Apply pattern boost for trusted patterns  
        if gpt_label in trusted_patterns:
            final_score += 0.15
            print(f"[GPT PATTERN BOOST v2] {token_data.get('symbol', 'UNKNOWN')}: '{gpt_label}' → Score {original_score:.3f} + 0.15 = {final_score:.3f}")
            
            # Additional boost for high CLIP confidence
            if clip_confidence > 0.6:
                final_score += 0.05
                print(f"[CLIP CONFIDENCE BOOST v2] {token_data.get('symbol', 'UNKNOWN')}: CLIP {clip_confidence:.2f} → Score {final_score-0.05:.3f} + 0.05 = {final_score:.3f}")
        
        # Ensure boosted score is still bounded [0.0, 1.0]
        final_score = max(0.0, min(1.0, final_score))
        
        # Generate decision
        decision = make_advanced_decision(final_score, market_phase, score_components, clip_result, gpt_label)
        
        print(f"[TJDE v2] {token_data.get('symbol', 'UNKNOWN')}: phase={market_phase}, score={final_score:.3f}, decision={decision}")
        
        return decision, final_score, score_components
        
    except Exception as e:
        print(f"[UNIFIED TJDE v2 ERROR] {e}")
        return "avoid", 0.0, {"error": str(e)}

def unified_tjde_decision_engine(token_data: Dict, market_phase: str, clip_result: Dict, 
                               gpt_label: str, volume_info: Dict, liquidity_info: Dict) -> Tuple[str, float, Dict]:
    """
    Główny silnik decyzyjny TJDE v2.
    Łączy pre-pump i trend-mode w jedną spójną logikę decyzyjną.
    
    Args:
        token_data: Complete token data including candles, price, volume
        market_phase: Detected market phase (pre-pump, trend-following, consolidation, breakout)
        clip_result: Vision AI CLIP analysis result with confidence
        gpt_label: GPT-4o chart analysis label
        volume_info: Volume structure and behavior analysis
        liquidity_info: Liquidity behavior and orderbook analysis
        
    Returns:
        tuple: (decision, final_score, score_components)
               decision: "enter", "avoid", "scalp_entry", "wait"
               final_score: Float score 0.0-1.0
               score_components: Dict of individual component scores
    """
    try:
        engine = UnifiedTJDEEngineV2()
        
        # 1. Dobierz profil scoringu
        if market_phase == "pre-pump":
            profile = engine.scoring_profiles.get("pre-pump", engine.scoring_profiles["trend-following"])
        else:
            profile = engine.scoring_profiles.get(market_phase, engine.scoring_profiles["trend-following"])
        
        # 2. Oblicz komponenty scoringu
        score_components = {
            "volume_structure": evaluate_volume_structure(volume_info),
            "clip_confidence": clip_result.get("confidence", 0.0) if clip_result else 0.0,
            "gpt_label_match": match_gpt_label(gpt_label, market_phase),
            "liquidity_behavior": analyze_liquidity_v2(liquidity_info),
            "heatmap_window": analyze_heatmap_exhaustion(token_data),
            "pre_breakout_structure": analyze_price_structure_v2(token_data),
            "market_phase_modifier": get_phase_modifier(market_phase, token_data)
        }
        
        # 3. Oblicz końcowy score używając profilu
        final_score = 0.0
        for component, value in score_components.items():
            weight = profile.get(component, 0.1)
            final_score += value * weight
        
        # Normalizuj score do zakresu 0.0-1.0
        final_score = max(0.0, min(1.0, final_score))
        
        # === GPT+CLIP PATTERN ALIGNMENT BOOSTER v2 ===
        original_score = final_score
        clip_confidence = clip_result.get("confidence", 0.0) if clip_result else 0.0
        
        # Trusted patterns that get score boost
        trusted_patterns = ["momentum_follow", "breakout-continuation", "trend-following", "trend_continuation"]
        
        # Apply pattern boost for trusted patterns  
        if gpt_label in trusted_patterns:
            final_score += 0.15
            print(f"[GPT PATTERN BOOST v2] {token_data.get('symbol', 'UNKNOWN')}: '{gpt_label}' → Score {original_score:.3f} + 0.15 = {final_score:.3f}")
            
            # Additional boost for high CLIP confidence
            if clip_confidence > 0.6:
                final_score += 0.05
                print(f"[CLIP CONFIDENCE BOOST v2] {token_data.get('symbol', 'UNKNOWN')}: CLIP {clip_confidence:.2f} → Score {final_score-0.05:.3f} + 0.05 = {final_score:.3f}")
        
        # Ensure boosted score is still bounded [0.0, 1.0]
        final_score = max(0.0, min(1.0, final_score))
        
        # 4. Zaawansowana logika decyzyjna TJDE v2
        decision = make_advanced_decision(final_score, market_phase, score_components, clip_result, gpt_label)
        
        print(f"[TJDE v2] {token_data.get('symbol', 'UNKNOWN')}: phase={market_phase}, score={final_score:.3f}, decision={decision}")
        
        return decision, final_score, score_components
        
    except Exception as e:
        print(f"[UNIFIED TJDE v2 ERROR] {e}")
        return "avoid", 0.0, {"error": str(e)}

def evaluate_volume_structure(volume_info: Dict) -> float:
    """Evaluate volume structure for decision making"""
    try:
        if not volume_info:
            return 0.3
        
        # Analiza struktury wolumenu
        volume_trend = volume_info.get("trend", "neutral")
        volume_spike = volume_info.get("spike_detected", False)
        volume_accumulation = volume_info.get("accumulation_score", 0.0)
        
        score = 0.3  # baseline
        
        if volume_spike:
            score += 0.2
        if volume_trend == "increasing":
            score += 0.15
        elif volume_trend == "decreasing":
            score -= 0.1
        
        score += volume_accumulation * 0.3
        
        return max(0.0, min(1.0, score))
        
    except Exception:
        return 0.3

def match_gpt_label(gpt_label: str, market_phase: str) -> float:
    """Match GPT label with market phase for scoring"""
    try:
        if not gpt_label or gpt_label == "unknown":
            return 0.2
        
        # Phase-specific label matching
        if market_phase == "pre-pump":
            pre_pump_labels = ["accumulation", "squeeze_pattern", "consolidation_squeeze", 
                             "pre_breakout", "support_bounce", "trend_pullback"]
            if any(label in gpt_label.lower() for label in pre_pump_labels):
                return 0.8
        
        elif market_phase == "breakout":
            breakout_labels = ["breakout_pattern", "breakout_continuation", "impulse",
                             "trend_continuation", "momentum_breakout"]
            if any(label in gpt_label.lower() for label in breakout_labels):
                return 0.9
        
        elif market_phase == "trend-following":
            trend_labels = ["trend_continuation", "pullback_in_trend", "trend_pullback_reacted"]
            if any(label in gpt_label.lower() for label in trend_labels):
                return 0.85
        
        # Generic positive patterns
        positive_labels = ["bullish", "support", "bounce", "continuation"]
        if any(label in gpt_label.lower() for label in positive_labels):
            return 0.6
        
        # Negative patterns
        negative_labels = ["bearish", "resistance", "rejection", "reversal"]
        if any(label in gpt_label.lower() for label in negative_labels):
            return 0.1
        
        return 0.4  # neutral
        
    except Exception:
        return 0.2

def analyze_liquidity_v2(liquidity_info: Dict) -> float:
    """Enhanced liquidity behavior analysis"""
    try:
        if not liquidity_info:
            return 0.4
        
        bid_ask_spread = liquidity_info.get("bid_ask_spread", 0.01)
        orderbook_depth = liquidity_info.get("depth_score", 0.5)
        liquidity_trend = liquidity_info.get("trend", "stable")
        
        score = 0.4  # baseline
        
        # Tight spreads are positive
        if bid_ask_spread < 0.005:
            score += 0.2
        elif bid_ask_spread > 0.02:
            score -= 0.1
        
        # Good orderbook depth
        score += orderbook_depth * 0.3
        
        # Liquidity trends
        if liquidity_trend == "improving":
            score += 0.15
        elif liquidity_trend == "deteriorating":
            score -= 0.15
        
        return max(0.0, min(1.0, score))
        
    except Exception:
        return 0.4

def analyze_heatmap_exhaustion(token_data: Dict) -> float:
    """Analyze heatmap exhaustion patterns"""
    try:
        candles = token_data.get("candles_15m", [])
        if len(candles) < 20:
            return 0.3
        
        # Look for exhaustion patterns in recent candles
        recent_candles = candles[-10:]
        
        # Calculate range exhaustion
        ranges = []
        for candle in recent_candles:
            if isinstance(candle, dict):
                high = candle.get("high", 0)
                low = candle.get("low", 0)
            else:
                high = candle[2] if len(candle) > 2 else 0
                low = candle[3] if len(candle) > 3 else 0
            
            if high > 0 and low > 0:
                ranges.append((high - low) / low)
        
        if ranges:
            avg_range = sum(ranges) / len(ranges)
            recent_range = ranges[-1] if ranges else 0
            
            # Exhaustion when recent range is significantly smaller
            if recent_range < avg_range * 0.5:
                return 0.8  # High exhaustion
            elif recent_range < avg_range * 0.7:
                return 0.6
        
        return 0.4
        
    except Exception:
        return 0.3

def analyze_price_structure_v2(token_data: Dict) -> float:
    """Enhanced price structure analysis for pre-breakout detection"""
    try:
        candles = token_data.get("candles_15m", [])
        if len(candles) < 15:
            return 0.3
        
        # Extract closes
        closes = []
        for candle in candles[-15:]:
            if isinstance(candle, dict):
                close = candle.get("close", 0)
            else:
                close = candle[4] if len(candle) > 4 else 0
            
            if close > 0:
                closes.append(close)
        
        if len(closes) < 10:
            return 0.3
        
        # Analyze structure
        recent_closes = closes[-5:]
        earlier_closes = closes[-10:-5]
        
        recent_avg = sum(recent_closes) / len(recent_closes)
        earlier_avg = sum(earlier_closes) / len(earlier_closes)
        
        # Look for consolidation followed by strength
        price_change = (recent_avg - earlier_avg) / earlier_avg
        
        # Volatility analysis
        recent_volatility = calculate_volatility(recent_closes)
        earlier_volatility = calculate_volatility(earlier_closes)
        
        score = 0.4
        
        # Positive price momentum
        if price_change > 0.02:
            score += 0.2
        elif price_change > 0.01:
            score += 0.1
        
        # Decreasing volatility (consolidation)
        if recent_volatility < earlier_volatility * 0.8:
            score += 0.15
        
        return max(0.0, min(1.0, score))
        
    except Exception:
        return 0.3

def calculate_volatility(prices: List[float]) -> float:
    """Calculate price volatility"""
    if len(prices) < 2:
        return 0.0
    
    avg = sum(prices) / len(prices)
    variance = sum((p - avg) ** 2 for p in prices) / len(prices)
    return variance ** 0.5

def get_phase_modifier(market_phase: str, token_data: Dict) -> float:
    """Get phase-specific modifier based on market conditions"""
    try:
        modifiers = {
            "pre-pump": 0.1,      # Bonus for pre-pump detection
            "breakout": 0.15,     # High bonus for breakouts
            "trend-following": 0.05,
            "consolidation": 0.0,
            "unknown": -0.05
        }
        
        base_modifier = modifiers.get(market_phase, 0.0)
        
        # Additional context-based modifiers
        price = token_data.get("price", 0)
        volume_24h = token_data.get("volume_24h", 0)
        
        # Volume confirmation
        if volume_24h > 1000000:  # High volume
            base_modifier += 0.02
        
        return base_modifier
        
    except Exception:
        return 0.0

def make_advanced_decision(final_score: float, market_phase: str, score_components: Dict, 
                          clip_result: Dict, gpt_label: str) -> str:
    """
    Zaawansowana logika decyzyjna TJDE v2
    
    Returns: "enter", "avoid", "scalp_entry", "wait"
    """
    try:
        # Pre-pump specific logic
        if market_phase == "pre-pump":
            if final_score >= 0.75:
                return "enter"  # Strong pre-pump signal
            elif final_score >= 0.65 and score_components.get("clip_confidence", 0) > 0.7:
                return "enter"  # Visual confirmation boost
            elif final_score >= 0.55:
                return "wait"   # Monitor for development
            else:
                return "avoid"
        
        # Breakout phase logic
        elif market_phase == "breakout":
            if final_score >= 0.7:
                return "enter"  # Strong breakout confirmation
            elif final_score >= 0.6 and score_components.get("volume_structure", 0) > 0.7:
                return "scalp_entry"  # Quick scalp on volume
            elif final_score >= 0.5:
                return "wait"
            else:
                return "avoid"
        
        # Trend-following logic
        elif market_phase == "trend-following":
            if final_score >= 0.7:
                return "enter"
            elif final_score >= 0.6:
                return "scalp_entry"  # Trend scalp
            elif final_score >= 0.45:
                return "wait"
            else:
                return "avoid"
        
        # Consolidation logic
        elif market_phase == "consolidation":
            if final_score >= 0.75:
                return "scalp_entry"  # Range trading
            elif final_score >= 0.6:
                return "wait"  # Wait for breakout
            else:
                return "avoid"
        
        # Default logic
        else:
            if final_score >= 0.7:
                return "enter"
            elif final_score >= 0.5:
                return "wait"
            else:
                return "avoid"
                
    except Exception:
        return "avoid"

def analyze_volume_trend(candles: List) -> str:
    """Analyze volume trend from candles"""
    try:
        if len(candles) < 10:
            return "neutral"
        
        recent_volumes = []
        for candle in candles[-10:]:
            if isinstance(candle, dict):
                volume = candle.get("volume", 0)
            else:
                volume = candle[5] if len(candle) > 5 else 0
            
            if volume > 0:
                recent_volumes.append(volume)
        
        if len(recent_volumes) < 5:
            return "neutral"
        
        # Compare recent vs earlier volumes
        recent_avg = sum(recent_volumes[-5:]) / 5
        earlier_avg = sum(recent_volumes[-10:-5]) / 5
        
        change = (recent_avg - earlier_avg) / earlier_avg
        
        if change > 0.2:
            return "increasing"
        elif change < -0.2:
            return "decreasing"
        else:
            return "neutral"
            
    except Exception:
        return "neutral"

def detect_volume_spike(candles: List) -> bool:
    """Detect volume spike in recent candles"""
    try:
        if len(candles) < 20:
            return False
        
        volumes = []
        for candle in candles[-20:]:
            if isinstance(candle, dict):
                volume = candle.get("volume", 0)
            else:
                volume = candle[5] if len(candle) > 5 else 0
            
            if volume > 0:
                volumes.append(volume)
        
        if len(volumes) < 15:
            return False
        
        # Check if recent volume is significantly higher than average
        recent_volume = volumes[-1]
        avg_volume = sum(volumes[:-3]) / len(volumes[:-3])
        
        # Spike if recent volume is 2x+ average
        return recent_volume > avg_volume * 2.0
        
    except Exception:
        return False

def analyze_symbol_with_unified_tjde_v2(symbol: str, market_data: Dict, candles_15m: List, 
                                       candles_5m: List, signals: Dict = None) -> Dict:
    """
    Enhanced TJDE v2 analysis with advanced decision engine
    
    Args:
        symbol: Trading symbol
        market_data: Market data dictionary
        candles_15m: 15-minute candles
        candles_5m: 5-minute candles
        signals: Pre-calculated signals (optional)
        
    Returns:
        dict: Complete TJDE v2 analysis result with enhanced decisions
    """
    try:
        # Import market phase detection from v1
        from unified_tjde_engine import UnifiedTJDEEngine
        engine_v1 = UnifiedTJDEEngine()
        
        # Detect market phase using v1 engine
        market_phase = engine_v1.detect_market_phase(symbol, market_data, candles_15m, candles_5m)
        
        # Prepare enhanced token data
        token_data = {
            "symbol": symbol,
            "price": market_data.get("price", 0),
            "volume_24h": market_data.get("volume_24h", 0),
            "candles_15m": candles_15m,
            "candles_5m": candles_5m,
            **market_data
        }
        
        # Prepare CLIP result
        clip_result = {
            "confidence": signals.get("clip_confidence", 0.0) if signals else 0.0,
            "pattern": signals.get("clip_pattern", "unknown") if signals else "unknown"
        }
        
        # Get GPT label
        gpt_label = signals.get("gpt_label", "unknown") if signals else "unknown"
        
        # Prepare volume info with enhanced analysis
        volume_info = {
            "trend": analyze_volume_trend(candles_15m),
            "spike_detected": detect_volume_spike(candles_15m),
            "accumulation_score": signals.get("volume_behavior_score", 0.5) if signals else 0.5
        }
        
        # Prepare liquidity info
        liquidity_info = {
            "bid_ask_spread": 0.01,
            "depth_score": signals.get("liquidity_behavior", 0.5) if signals else 0.5,
            "trend": "stable"
        }
        
        # Run unified decision engine v2 with signals
        decision, final_score, score_components = unified_tjde_decision_engine_with_signals(
            token_data=token_data,
            market_phase=market_phase,
            clip_result=clip_result,
            gpt_label=gpt_label,
            volume_info=volume_info,
            liquidity_info=liquidity_info,
            signals=signals
        )
        
        # Calculate quality grade using v1 engine
        quality_grade = engine_v1._calculate_quality_grade(final_score, score_components)
        
        return {
            "final_score": final_score,
            "decision": decision,
            "market_phase": market_phase,
            "quality_grade": quality_grade,
            "score_breakdown": score_components,
            "components": score_components,
            "analysis_type": "unified_tjde_v2",
            "decision_options": ["enter", "avoid", "scalp_entry", "wait"],
            "engine_version": "v2.0"
        }
        
    except Exception as e:
        return {
            "error": f"Unified TJDE v2 analysis failed: {e}",
            "final_score": 0.0,
            "decision": "avoid",
            "market_phase": "unknown",
            "quality_grade": "error",
            "engine_version": "v2.0"
        }

# Market phase analysis function from v1 for standalone usage
def detect_market_phase_v2(symbol: str, market_data: Dict, candles_15m: List, candles_5m: List) -> str:
    """
    Enhanced market phase detection for TJDE v2
    
    Returns: "pre-pump", "trend-following", "consolidation", "breakout"
    """
    try:
        from unified_tjde_engine import UnifiedTJDEEngine
        engine = UnifiedTJDEEngine()
        return engine.detect_market_phase(symbol, market_data, candles_15m, candles_5m)
    except Exception:
        return "unknown"

if __name__ == "__main__":
    # Test TJDE v2 functionality
    print("Testing Unified TJDE Engine v2...")
    
    # Test data
    test_data = {
        "symbol": "BTCUSDT",
        "price": 50000.0,
        "volume_24h": 1000000000,
        "candles_15m": [[1640995200000, 49000, 51000, 48500, 50000, 1000000]],
        "candles_5m": [[1640995200000, 49000, 51000, 48500, 50000, 1000000]]
    }
    
    result = analyze_symbol_with_unified_tjde_v2(
        symbol="BTCUSDT",
        market_data=test_data,
        candles_15m=test_data["candles_15m"],
        candles_5m=test_data["candles_5m"]
    )
    
    print(f"✅ TJDE v2 Test Result:")
    print(f"   Score: {result.get('final_score', 0):.3f}")
    print(f"   Decision: {result.get('decision', 'unknown')}")
    print(f"   Market Phase: {result.get('market_phase', 'unknown')}")
    print(f"   Engine: {result.get('engine_version', 'unknown')}")