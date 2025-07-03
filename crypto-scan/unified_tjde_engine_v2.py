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

def load_scoring_profile(market_phase: str) -> dict:
    """
    ETAP 3 - Ładowanie profilu scoringowego dla wykrytej fazy rynku
    
    Args:
        market_phase: Detected market phase (trend-following, consolidation, breakout, pre-pump)
        
    Returns:
        dict: Scoring profile with component weights or None if not found
    """
    profile_map = {
        "trend-following": "tjde_trend_following_profile.json",
        "consolidation": "tjde_consolidation_profile.json", 
        "breakout": "tjde_breakout_profile.json",
        "pre-pump": "tjde_pre_pump_profile.json"
    }
    
    profile_file = profile_map.get(market_phase)
    if not profile_file:
        print(f"[PROFILE LOAD] No profile defined for phase: {market_phase}")
        return None
    
    profile_path = os.path.join("data", "weights", profile_file)
    try:
        with open(profile_path, "r") as f:
            profile = json.load(f)
            print(f"[PROFILE LOAD] Loaded {market_phase} profile from {profile_file}")
            return profile
    except Exception as e:
        print(f"[PROFILE ERROR] Failed to load profile {profile_file}: {e}")
        return None


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
        
        # Use signals if available, otherwise SKIP token (no fallback)
        if signals:
            # Extract authentic signals - NO FALLBACKS
            trend_strength = signals.get("trend_strength")
            pullback_quality = signals.get("pullback_quality")
            support_reaction = signals.get("support_reaction_strength")
            volume_behavior_score = signals.get("volume_behavior_score")
            psych_score = signals.get("psych_score")
            
            # Block tokens without complete authentic data (check None only, 0.0 is valid)
            if any(x is None for x in [trend_strength, pullback_quality, volume_behavior_score, psych_score, support_reaction]):
                print(f"[TJDE v2 INCOMPLETE] {token_data.get('symbol', 'UNKNOWN')}: Missing authentic signals - blocking token")
                print(f"  trend_strength={trend_strength}, pullback_quality={pullback_quality}, volume_behavior_score={volume_behavior_score}")
                print(f"  psych_score={psych_score}, support_reaction={support_reaction}")
                return "skip", 0.0, {"error": "incomplete_signals"}
            
            # Convert to float with safe defaults for None values
            trend_strength = float(trend_strength) if trend_strength is not None else 0.0
            pullback_quality = float(pullback_quality) if pullback_quality is not None else 0.0
            volume_behavior_score = float(volume_behavior_score) if volume_behavior_score is not None else 0.0
            psych_score = float(psych_score) if psych_score is not None else 0.0
            support_reaction = float(support_reaction) if support_reaction is not None else 0.0
            
            # Apply dynamic scoring formula: final_score = (0.35 * trend_strength + 0.25 * pullback_quality + 0.2 * volume_behavior_score + 0.1 * psych_score + 0.1 * support_reaction)
            weighted_score = (
                0.35 * trend_strength +
                0.25 * pullback_quality +
                0.20 * volume_behavior_score +
                0.10 * psych_score +
                0.10 * support_reaction
            )
            
            score_components = {
                "trend_strength": trend_strength,
                "pullback_quality": pullback_quality, 
                "support_reaction": support_reaction,
                "volume_behavior_score": volume_behavior_score,
                "psych_score": psych_score,
                "htf_supportive_score": signals.get("htf_supportive_score", 0.0),
                "liquidity_pattern_score": signals.get("liquidity_behavior", 0.0),
                "clip_confidence": clip_result.get("confidence", 0.0) if clip_result else 0.0,
                "gpt_label_match": match_gpt_label(gpt_label, market_phase),
                "market_phase_modifier": get_phase_modifier(market_phase, token_data),
                "weighted_base_score": weighted_score
            }
            print(f"[TJDE v2 DYNAMIC] {token_data.get('symbol', 'UNKNOWN')}: Using dynamic scoring formula")
            print(f"[WEIGHTED FORMULA] trend:{trend_strength:.3f} pullback:{pullback_quality:.3f} volume:{volume_behavior_score:.3f} psych:{psych_score:.3f} support:{support_reaction:.3f} → {weighted_score:.3f}")
        else:
            # GENERATE BASIC SIGNALS - TJDE v2 should not block tokens without signals
            print(f"[TJDE v2 BASIC SIGNALS] {token_data.get('symbol', 'UNKNOWN')}: No pre-calculated signals - generating basic signals")
            
            # Generate basic signals from available data
            candles_15m = token_data.get("candles_15m", [])
            candles_5m = token_data.get("candles_5m", [])
            
            if candles_15m and len(candles_15m) >= 10:
                # Basic trend strength from price movement
                recent_prices = [c[4] if isinstance(c, list) else c.get("close", 0) for c in candles_15m[-10:]]
                if recent_prices and all(p > 0 for p in recent_prices):
                    price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    trend_strength = min(1.0, max(0.0, 0.5 + price_change * 10))  # Scale to 0-1
                else:
                    trend_strength = 0.5
                
                # Basic pullback quality from volatility
                if len(recent_prices) > 1:
                    price_range = max(recent_prices) - min(recent_prices)
                    avg_price = sum(recent_prices) / len(recent_prices)
                    volatility = price_range / avg_price if avg_price > 0 else 0
                    pullback_quality = min(1.0, max(0.0, 0.5 + volatility * 5))
                else:
                    pullback_quality = 0.5
                
                # Basic volume behavior
                recent_volumes = [c[5] if isinstance(c, list) else c.get("volume", 0) for c in candles_15m[-5:]]
                if recent_volumes and all(v >= 0 for v in recent_volumes):
                    avg_volume = sum(recent_volumes) / len(recent_volumes)
                    volume_behavior_score = 0.6 if avg_volume > 0 else 0.4
                else:
                    volume_behavior_score = 0.5
                
                # Basic values for other components
                psych_score = 0.5
                support_reaction = 0.5
                
            else:
                # Minimal fallback values
                trend_strength = 0.4
                pullback_quality = 0.4  
                volume_behavior_score = 0.4
                psych_score = 0.4
                support_reaction = 0.4
            
            # Apply same dynamic scoring formula
            weighted_score = (
                0.35 * trend_strength +
                0.25 * pullback_quality +
                0.20 * volume_behavior_score +
                0.10 * psych_score +
                0.10 * support_reaction
            )
            
            score_components = {
                "trend_strength": trend_strength,
                "pullback_quality": pullback_quality, 
                "support_reaction": support_reaction,
                "volume_behavior_score": volume_behavior_score,
                "psych_score": psych_score,
                "htf_supportive_score": 0.5,
                "liquidity_pattern_score": 0.5,
                "clip_confidence": clip_result.get("confidence", 0.0) if clip_result else 0.0,
                "gpt_label_match": match_gpt_label(gpt_label, market_phase),
                "market_phase_modifier": get_phase_modifier(market_phase, token_data),
                "weighted_base_score": weighted_score
            }
            print(f"[TJDE v2 BASIC] {token_data.get('symbol', 'UNKNOWN')}: Generated basic signals")
            print(f"[BASIC FORMULA] trend:{trend_strength:.3f} pullback:{pullback_quality:.3f} volume:{volume_behavior_score:.3f} psych:{psych_score:.3f} support:{support_reaction:.3f} → {weighted_score:.3f}")
        
        # Use dynamic weighted score as base instead of profile weights
        if "weighted_base_score" in score_components:
            final_score = score_components["weighted_base_score"]
            print(f"[DYNAMIC SCORING] {token_data.get('symbol', 'UNKNOWN')}: Using weighted formula base score: {final_score:.3f}")
        else:
            # Calculate final score using profile weights (legacy fallback)
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
        # ETAP 1 - SANITY CHECK DANYCH RYNKOWYCH
        print(f"[TJDE v2 STAGE 1] {symbol}: Starting market data validation")
        
        # Wymagane dane
        ticker_data = market_data.get("ticker_data")
        orderbook = market_data.get("orderbook")
        volume_24h = market_data.get("volume_24h", 0.0)
        
        # Sanity check – brak świec = brak scoringu
        if not candles_15m or not candles_5m or not orderbook or not ticker_data:
            missing_data = []
            if not candles_15m: missing_data.append("candles_15m")
            if not candles_5m: missing_data.append("candles_5m") 
            if not orderbook: missing_data.append("orderbook")
            if not ticker_data: missing_data.append("ticker_data")
            
            print(f"[TJDE BLOCK] Missing market data for {symbol}: {', '.join(missing_data)} – skipping scoring.")
            return {"final_score": 0.0, "decision": "skip", "error": f"Missing data: {', '.join(missing_data)}"}

        if len(candles_15m) < 30 or len(candles_5m) < 30:
            print(f"[TJDE BLOCK] Insufficient candle history for {symbol}: 15M={len(candles_15m)}, 5M={len(candles_5m)} (need 30+ each)")
            return {"final_score": 0.0, "decision": "skip", "error": f"Insufficient candles: 15M={len(candles_15m)}, 5M={len(candles_5m)}"}

        if volume_24h == 0.0:
            print(f"[TJDE BLOCK] No volume data for {symbol}: volume_24h={volume_24h}")
            return {"final_score": 0.0, "decision": "skip", "error": "Zero volume data"}
            
        print(f"[TJDE v2 STAGE 1] {symbol}: ✅ Market data validation passed - 15M={len(candles_15m)}, 5M={len(candles_5m)}, Volume=${volume_24h:,.0f}")
        
        # ETAP 2 - DETEKCJA FAZY RYNKU
        market_phase = detect_market_phase_v2(symbol, market_data, candles_15m, candles_5m)
        
        # Blokowanie tokenów z nieznaną fazą (zgodnie ze specyfikacją)
        if market_phase == "unknown":
            print(f"[TJDE BLOCK] Market phase undetectable for {symbol} – skipping scoring")
            return {"final_score": 0.0, "decision": "skip", "error": "Market phase undetectable", "market_phase": "unknown"}
        
        # ETAP 3 - ŁADOWANIE PROFILU SCORINGOWEGO
        scoring_profile = load_scoring_profile(market_phase)
        if scoring_profile is None:
            print(f"[TJDE BLOCK] Failed to load scoring profile for phase '{market_phase}' – skipping {symbol}")
            return {"final_score": 0.0, "decision": "skip", "error": f"No scoring profile for phase: {market_phase}", "market_phase": market_phase}
        
        print(f"[TJDE v2 STAGE 3] {symbol}: ✅ Loaded {market_phase} scoring profile")
        
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
        
        # Calculate quality grade directly
        if final_score >= 0.8:
            quality_grade = "excellent"
        elif final_score >= 0.65:
            quality_grade = "good"
        elif final_score >= 0.5:
            quality_grade = "moderate"
        elif final_score >= 0.3:
            quality_grade = "low"
        else:
            quality_grade = "poor"
        
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
    ETAP 2 - Enhanced Market Phase Detection for TJDE v2
    
    Obsługiwane fazy:
    - "trend-following" – silny kierunkowy trend
    - "consolidation" – boczny ruch z zawężającym się zakresem  
    - "breakout" – świeży impuls wybicia po kompresji
    - "pre-pump" – akumulacja przed potencjalnym wybiciem
    - "unknown" – brak jednoznacznego rozpoznania (NO FALLBACK!)
    
    Returns: Detected market phase or "unknown"
    """
    try:
        print(f"[TJDE v2 STAGE 2] {symbol}: Starting market phase detection")
        
        if not candles_15m or len(candles_15m) < 10:
            print(f"[PHASE DETECTION] {symbol}: Insufficient 15M candles ({len(candles_15m) if candles_15m else 0}) - returning unknown")
            return "unknown"
        
        # Zbieranie ostatnich 10 świec z candles_15m do detekcji struktury
        recent_candles = candles_15m[-10:]
        print(f"[PHASE DETECTION] {symbol}: Analyzing {len(recent_candles)} recent candles")
        
        # Extract data from candles (support both dict and list formats)
        prices = []
        highs = []
        lows = []
        volumes = []
        
        for candle in recent_candles:
            if isinstance(candle, dict):
                prices.append(candle.get("close", 0))
                highs.append(candle.get("high", 0))
                lows.append(candle.get("low", 0))
                volumes.append(candle.get("volume", 0))
            elif isinstance(candle, list) and len(candle) >= 6:
                prices.append(candle[4])  # close
                highs.append(candle[2])   # high
                lows.append(candle[3])    # low
                volumes.append(candle[5]) # volume
            else:
                print(f"[PHASE DETECTION ERROR] {symbol}: Invalid candle format")
                return "unknown"
        
        # Validacja danych
        if not prices or not highs or not lows or not volumes:
            print(f"[PHASE DETECTION ERROR] {symbol}: Empty price/volume data")
            return "unknown"
            
        if any(p <= 0 for p in prices):
            print(f"[PHASE DETECTION ERROR] {symbol}: Invalid price data (zero/negative)")
            return "unknown"
        
        # Obliczanie kluczowych metryk
        price_range = max(highs) - min(lows)
        price_slope = (prices[-1] - prices[0]) / len(prices) if len(prices) > 0 else 0
        volatility_ratio = price_range / prices[0] if prices[0] != 0 else 0
        volume_range = max(volumes) - min(volumes) if volumes else 0
        max_volume = max(volumes) if volumes else 1
        
        print(f"[PHASE METRICS] {symbol}: price_slope={price_slope:.6f}, volatility_ratio={volatility_ratio:.6f}")
        print(f"[PHASE METRICS] {symbol}: price_range={price_range:.2f}, volume_range={volume_range:.0f}, max_volume={max_volume:.0f}")
        
        # Detekcja fazy zgodnie ze specyfikacją
        if price_slope > 0.002 and volatility_ratio > 0.01:
            phase = "trend-following"
            print(f"[PHASE DETECTED] {symbol}: {phase} (strong directional trend)")
            return phase
            
        elif price_slope < 0.001 and volatility_ratio < 0.008:
            phase = "consolidation"
            print(f"[PHASE DETECTED] {symbol}: {phase} (sideways movement)")
            return phase
            
        elif volatility_ratio > 0.015 and volume_range > 0.4 * max_volume:
            phase = "breakout"
            print(f"[PHASE DETECTED] {symbol}: {phase} (fresh breakout impulse)")
            return phase
            
        elif max_volume > 0 and volume_range < 0.1 * max_volume:
            phase = "pre-pump"
            print(f"[PHASE DETECTED] {symbol}: {phase} (accumulation phase)")
            return phase
            
        else:
            # Brak jednoznacznego rozpoznania - NIE MA FALLBACK!
            print(f"[PHASE UNDETECTED] {symbol}: No clear phase pattern - returning unknown")
            print(f"[PHASE DEBUG] {symbol}: Criteria not met - price_slope={price_slope:.6f}, volatility={volatility_ratio:.6f}, vol_ratio={volume_range/max_volume if max_volume > 0 else 0:.3f}")
            return "unknown"
            
    except Exception as e:
        print(f"[PHASE DETECTION ERROR] {symbol}: {e}")
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