#!/usr/bin/env python3
"""
Unified TJDE Engine - Single Decision Engine for All Market Phases
Replaces PPWCS with intelligent trader-level decision making across all phases:
- pre-pump (early entry detection)
- trend-following (momentum continuation)
- consolidation (range trading)
- breakout (breakout confirmation)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from trader_ai_engine import simulate_trader_decision_advanced

class UnifiedTJDEEngine:
    """
    Unified TJDE Engine for all market phases
    Replaces PPWCS with intelligent context-aware scoring
    """
    
    def __init__(self):
        self.scoring_profiles = {}
        self.load_scoring_profiles()
        
    def load_scoring_profiles(self):
        """Load phase-specific scoring profiles"""
        profiles_dir = "data/weights"
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Define phase-specific scoring profiles
        default_profiles = {
            "pre-pump": {
                "volume_structure": 0.20,
                "clip_confidence": 0.10,
                "gpt_label_match": 0.10,
                "liquidity_behavior": 0.15,
                "heatmap_window": 0.10,
                "pre_breakout_structure": 0.25,
                "market_phase_modifier": 0.10
            },
            "trend-following": {
                "trend_strength": 0.235,
                "clip_confidence": 0.12,
                "gpt_label_match": 0.08,
                "liquidity_behavior": 0.12,
                "momentum_persistence": 0.18,
                "volume_confirmation": 0.15,
                "market_phase_modifier": 0.10
            },
            "consolidation": {
                "range_quality": 0.25,
                "clip_confidence": 0.15,
                "support_resistance": 0.20,
                "volume_dryup": 0.15,
                "compression_factor": 0.15,
                "market_phase_modifier": 0.10
            },
            "breakout": {
                "breakout_strength": 0.30,
                "clip_confidence": 0.15,
                "volume_expansion": 0.20,
                "follow_through": 0.15,
                "false_breakout_filter": 0.10,
                "market_phase_modifier": 0.10
            }
        }
        
        # Load or create profiles
        for phase, weights in default_profiles.items():
            profile_path = f"{profiles_dir}/tjde_{phase.replace('-', '_')}_profile.json"
            
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r') as f:
                        self.scoring_profiles[phase] = json.load(f)
                    print(f"[TJDE PROFILES] Loaded {phase} profile from {profile_path}")
                except Exception as e:
                    print(f"[TJDE PROFILES] Error loading {phase}: {e}, using defaults")
                    self.scoring_profiles[phase] = weights
            else:
                # Create default profile
                self.scoring_profiles[phase] = weights
                try:
                    with open(profile_path, 'w') as f:
                        json.dump(weights, f, indent=2)
                    print(f"[TJDE PROFILES] Created default {phase} profile at {profile_path}")
                except Exception as e:
                    print(f"[TJDE PROFILES] Error saving {phase}: {e}")
    
    def detect_market_phase(self, symbol: str, market_data: Dict, candles_15m: List, candles_5m: List) -> str:
        """
        Detect current market phase for phase-specific scoring
        
        Returns:
            str: "pre-pump", "trend-following", "consolidation", "breakout"
        """
        try:
            print(f"[PHASE DEBUG] {symbol}: Starting phase detection with {len(candles_15m) if candles_15m else 0} candles")
            
            if not candles_15m or len(candles_15m) < 10:
                print(f"[PHASE DEBUG] {symbol}: Insufficient candles, returning 'unknown'")
                return "unknown"
            
            # Calculate key metrics for phase detection
            recent_candles = candles_15m[-10:]  # Last 10 candles for analysis
            print(f"[PHASE DEBUG] {symbol}: Using {len(recent_candles)} recent candles")
            
            # Price metrics
            prices = [float(c[4]) for c in recent_candles if len(c) > 4]  # Close prices
            if not prices:
                print(f"[PHASE DEBUG] {symbol}: No valid prices found, returning 'unknown'")
                return "unknown"
                
            current_price = prices[-1]
            price_range = max(prices) - min(prices)
            avg_price = sum(prices) / len(prices)
            print(f"[PHASE DEBUG] {symbol}: current_price={current_price:.4f}, range={price_range:.4f}, avg={avg_price:.4f}")
            
            # Volume metrics
            volumes = [float(c[5]) for c in recent_candles if len(c) > 5]
            if not volumes:
                volumes = [1000] * len(prices)  # Fallback
                print(f"[PHASE DEBUG] {symbol}: Using volume fallback")
                
            avg_volume = sum(volumes) / len(volumes)
            recent_volume = volumes[-3:] if len(volumes) >= 3 else volumes
            volume_trend = sum(recent_volume) / len(recent_volume) / avg_volume if avg_volume > 0 else 1.0
            print(f"[PHASE DEBUG] {symbol}: avg_volume={avg_volume:.0f}, volume_trend={volume_trend:.3f}")
            
            # Volatility and compression
            volatility = price_range / avg_price if avg_price > 0 else 0.01
            print(f"[PHASE DEBUG] {symbol}: volatility={volatility:.4f}")
            
            # Phase detection logic
            
            # 1. PRE-PUMP: Low volatility + building volume + price near support
            if (volatility < 0.03 and  # Low volatility (< 3%)
                volume_trend > 1.1 and  # Volume increasing
                current_price < avg_price * 1.02):  # Price near average
                return "pre-pump"
            
            # 2. BREAKOUT: High volume + price breaking recent range
            if (volume_trend > 1.5 and  # High volume spike
                (current_price > max(prices[:-1]) * 1.01 or  # Upward breakout
                 current_price < min(prices[:-1]) * 0.99)):   # Downward breakout
                return "breakout"
            
            # 3. TREND-FOLLOWING: Consistent direction + sustained volume
            price_momentum = (current_price - prices[0]) / prices[0] if prices[0] > 0 else 0
            if (abs(price_momentum) > 0.02 and  # Strong momentum (> 2%)
                volume_trend > 0.8):  # Sustained volume
                return "trend-following"
            
            # 4. CONSOLIDATION: Low volatility + declining volume
            if (volatility < 0.02 and  # Very low volatility
                volume_trend < 0.9):   # Declining volume
                return "consolidation"
            
            # Default fallback
            return "trend-following"
            
        except Exception as e:
            print(f"[PHASE DETECTION ERROR] {symbol}: {e}")
            print(f"[PHASE FALLBACK] Using default: trend-following (error={e})")
            # Add debug information about input data
            try:
                candles_count = len(candles_15m) if candles_15m else 0
                print(f"[PHASE FALLBACK DEBUG] {symbol}: Input candles_15m length = {candles_count}")
                if candles_15m and len(candles_15m) > 0:
                    print(f"[PHASE FALLBACK DEBUG] {symbol}: First candle structure = {type(candles_15m[0])}")
            except Exception as debug_e:
                print(f"[PHASE FALLBACK DEBUG ERROR] {symbol}: {debug_e}")
            return "trend-following"
    
    def calculate_unified_score(self, symbol: str, market_data: Dict, signals: Dict, 
                              market_phase: str, debug_info: Dict = None) -> Dict:
        """
        Calculate unified TJDE score using phase-specific weights
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            signals: Signal components dictionary
            market_phase: Detected market phase
            debug_info: Optional debug information
            
        Returns:
            dict: Unified scoring result with decision
        """
        try:
            # Get phase-specific weights
            weights = self.scoring_profiles.get(market_phase, self.scoring_profiles.get("trend-following", {}))
            
            # Calculate component scores based on phase
            components = {}
            
            if market_phase == "pre-pump":
                components = self._calculate_prepump_components(symbol, market_data, signals)
            elif market_phase == "trend-following":
                components = self._calculate_trend_components(symbol, market_data, signals)
            elif market_phase == "consolidation":
                components = self._calculate_consolidation_components(symbol, market_data, signals)
            elif market_phase == "breakout":
                components = self._calculate_breakout_components(symbol, market_data, signals)
            else:
                # Fallback to trend-following
                components = self._calculate_trend_components(symbol, market_data, signals)
            
            # Apply weights and calculate final score
            weighted_score = 0.0
            score_breakdown = {}
            
            for component, value in components.items():
                weight = weights.get(component, 0.0)
                weighted_contribution = value * weight
                weighted_score += weighted_contribution
                score_breakdown[component] = {
                    "value": value,
                    "weight": weight,
                    "contribution": weighted_contribution
                }
            
            # Ensure score is bounded [0.0, 1.0]
            final_score = max(0.0, min(1.0, weighted_score))
            
            # === GPT+CLIP PATTERN ALIGNMENT BOOSTER ===
            original_score = final_score
            gpt_label = signals.get("gpt_label", "unknown")
            clip_confidence = signals.get("clip_confidence", 0.0)
            
            # Trusted patterns that get score boost
            trusted_patterns = ["momentum_follow", "breakout-continuation", "trend-following", "trend_continuation"]
            
            # Apply pattern boost for trusted patterns
            if gpt_label in trusted_patterns:
                final_score += 0.15
                print(f"[GPT PATTERN BOOST] {symbol}: '{gpt_label}' → Score {original_score:.3f} + 0.15 = {final_score:.3f}")
                
                # Additional boost for high CLIP confidence
                if clip_confidence > 0.6:
                    final_score += 0.05
                    print(f"[CLIP CONFIDENCE BOOST] {symbol}: CLIP {clip_confidence:.2f} → Score {final_score-0.05:.3f} + 0.05 = {final_score:.3f}")
            
            # Ensure boosted score is still bounded [0.0, 1.0]
            final_score = max(0.0, min(1.0, final_score))
            
            # Generate decision based on phase and boosted score
            decision = self._generate_phase_decision(market_phase, final_score, components)
            
            # Generate quality grade
            quality_grade = self._calculate_quality_grade(final_score, components)
            
            result = {
                "symbol": symbol,
                "market_phase": market_phase,
                "final_score": final_score,
                "decision": decision,
                "quality_grade": quality_grade,
                "score_breakdown": score_breakdown,
                "components": components,
                "weights_used": weights,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"[UNIFIED TJDE] {symbol}: {market_phase} → {decision} (score: {final_score:.3f})")
            
            return result
            
        except Exception as e:
            print(f"[UNIFIED TJDE ERROR] {symbol}: {e}")
            return {
                "symbol": symbol,
                "market_phase": "error",
                "final_score": 0.0,
                "decision": "avoid",
                "quality_grade": "error",
                "error": str(e)
            }
    
    def _calculate_prepump_components(self, symbol: str, market_data: Dict, signals: Dict) -> Dict:
        """Calculate pre-pump specific components"""
        components = {}
        
        # Volume structure analysis
        candles_15m = market_data.get("candles_15m", [])
        if candles_15m and len(candles_15m) >= 5:
            recent_volumes = [float(c[5]) for c in candles_15m[-5:] if len(c) > 5]
            if recent_volumes:
                volume_trend = recent_volumes[-1] / (sum(recent_volumes[:-1]) / len(recent_volumes[:-1])) if len(recent_volumes) > 1 else 1.0
                components["volume_structure"] = min(1.0, max(0.0, (volume_trend - 0.8) * 2.5))  # 0.8-1.2 maps to 0-1
            else:
                components["volume_structure"] = 0.3
        else:
            components["volume_structure"] = 0.3
        
        # CLIP confidence (from signals)
        clip_confidence = signals.get("clip_confidence", 0.0)
        components["clip_confidence"] = clip_confidence
        
        # GPT label match for pre-pump patterns
        gpt_label = signals.get("gpt_label", "unknown")
        prepump_keywords = ["accumulation", "compression", "pre-breakout", "consolidation", "building"]
        label_match = 1.0 if any(keyword in gpt_label.lower() for keyword in prepump_keywords) else 0.3
        components["gpt_label_match"] = label_match
        
        # Liquidity behavior
        orderbook = market_data.get("orderbook")
        if orderbook:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            if bids and asks:
                bid_strength = sum([float(bid[1]) for bid in bids[:5]])  # Top 5 bid sizes
                ask_strength = sum([float(ask[1]) for ask in asks[:5]])  # Top 5 ask sizes
                liquidity_ratio = bid_strength / (bid_strength + ask_strength) if (bid_strength + ask_strength) > 0 else 0.5
                components["liquidity_behavior"] = liquidity_ratio
            else:
                components["liquidity_behavior"] = 0.5
        else:
            components["liquidity_behavior"] = 0.5
        
        # Heatmap window (price compression)
        if candles_15m and len(candles_15m) >= 10:
            prices = [float(c[4]) for c in candles_15m[-10:] if len(c) > 4]
            if prices:
                price_range = max(prices) - min(prices)
                avg_price = sum(prices) / len(prices)
                compression = 1.0 - (price_range / avg_price) if avg_price > 0 else 0.0
                components["heatmap_window"] = min(1.0, max(0.0, compression * 10))  # Amplify compression signal
            else:
                components["heatmap_window"] = 0.3
        else:
            components["heatmap_window"] = 0.3
        
        # Pre-breakout structure
        trend_strength = signals.get("trend_strength", 0.0)
        pullback_quality = signals.get("pullback_quality", 0.0)
        structure_score = (trend_strength * 0.6 + pullback_quality * 0.4)
        components["pre_breakout_structure"] = structure_score
        
        # Market phase modifier (always 1.0 for correct phase)
        components["market_phase_modifier"] = 1.0
        
        return components
    
    def _calculate_trend_components(self, symbol: str, market_data: Dict, signals: Dict) -> Dict:
        """Calculate trend-following specific components"""
        components = {}
        
        # Use existing trend analysis
        components["trend_strength"] = signals.get("trend_strength", 0.0)
        components["pullback_quality"] = signals.get("pullback_quality", 0.0)
        components["support_reaction"] = signals.get("support_reaction_strength", 0.0)
        components["clip_confidence"] = signals.get("clip_confidence", 0.0)
        components["liquidity_pattern_score"] = signals.get("liquidity_behavior", 0.5)
        components["psych_score"] = signals.get("psych_score", 0.0)
        components["htf_supportive_score"] = signals.get("htf_supportive_score", 0.5)
        
        # === GPT+CLIP PATTERN ALIGNMENT BOOSTER ===
        gpt_label = signals.get("gpt_label", "unknown")
        clip_confidence = signals.get("clip_confidence", 0.0)
        
        # Enhanced GPT pattern matching for trend patterns
        trend_keywords = ["trend", "momentum", "continuation", "following", "impulse"]
        trusted_patterns = ["momentum_follow", "breakout-continuation", "trend-following", "trend_continuation"]
        
        # Check for trusted patterns that deserve boost
        is_trusted_pattern = gpt_label in trusted_patterns
        has_high_clip_confidence = clip_confidence > 0.6
        
        if is_trusted_pattern:
            label_match = 1.0  # High match for trusted patterns
            print(f"[GPT PATTERN BOOST] {symbol}: '{gpt_label}' is trusted pattern")
        elif any(keyword in gpt_label.lower() for keyword in trend_keywords):
            label_match = 0.8  # Good match for trend keywords
        else:
            label_match = 0.3  # Default low match
            
        components["gpt_label_match"] = label_match
        
        components["market_phase_modifier"] = 1.0
        
        print(f"[UNIFIED TJDE] {symbol}: Component scores:")
        print(f"  trend_strength: {components['trend_strength']:.3f}")
        print(f"  pullback_quality: {components['pullback_quality']:.3f}")
        print(f"  support_reaction: {components['support_reaction']:.3f}")
        print(f"  psych_score: {components['psych_score']:.3f}")
        print(f"  clip_confidence: {components['clip_confidence']:.3f}")
        print(f"  liquidity_pattern_score: {components['liquidity_pattern_score']:.3f}")
        
        return components
    
    def _calculate_consolidation_components(self, symbol: str, market_data: Dict, signals: Dict) -> Dict:
        """Calculate consolidation specific components"""
        components = {}
        
        # Range quality analysis
        candles_15m = market_data.get("candles_15m", [])
        if candles_15m and len(candles_15m) >= 20:
            prices = [float(c[4]) for c in candles_15m[-20:] if len(c) > 4]
            if prices:
                price_range = max(prices) - min(prices)
                avg_price = sum(prices) / len(prices)
                range_consistency = 1.0 - (price_range / avg_price) if avg_price > 0 else 0.0
                components["range_quality"] = min(1.0, max(0.0, range_consistency * 5))
            else:
                components["range_quality"] = 0.3
        else:
            components["range_quality"] = 0.3
        
        components["clip_confidence"] = signals.get("clip_confidence", 0.0)
        components["support_resistance"] = signals.get("support_reaction_strength", 0.0)
        components["volume_dryup"] = 1.0 - min(1.0, signals.get("volume_behavior_score", 0.5))
        components["compression_factor"] = components["range_quality"]  # Similar to range quality
        components["market_phase_modifier"] = 1.0
        
        return components
    
    def _calculate_breakout_components(self, symbol: str, market_data: Dict, signals: Dict) -> Dict:
        """Calculate breakout specific components"""
        components = {}
        
        # Breakout strength
        trend_strength = signals.get("trend_strength", 0.0)
        volume_score = signals.get("volume_behavior_score", 0.0)
        breakout_strength = (trend_strength * 0.7 + volume_score * 0.3)
        components["breakout_strength"] = breakout_strength
        
        components["clip_confidence"] = signals.get("clip_confidence", 0.0)
        components["volume_expansion"] = signals.get("volume_behavior_score", 0.0)
        components["follow_through"] = signals.get("bounce_confirmation_strength", 0.0)
        
        # False breakout filter (anti-fakeout)
        candles_15m = market_data.get("candles_15m", [])
        if candles_15m and len(candles_15m) >= 3:
            recent_candles = candles_15m[-3:]
            fakeout_risk = 0.0
            
            for candle in recent_candles:
                if len(candle) >= 5:
                    open_price = float(candle[1])
                    high_price = float(candle[2])
                    low_price = float(candle[3])
                    close_price = float(candle[4])
                    
                    total_range = high_price - low_price
                    upper_wick = high_price - max(open_price, close_price)
                    
                    if total_range > 0:
                        wick_ratio = upper_wick / total_range
                        if wick_ratio > 0.5 and close_price < open_price:
                            fakeout_risk += 0.3
            
            components["false_breakout_filter"] = max(0.0, 1.0 - fakeout_risk)
        else:
            components["false_breakout_filter"] = 0.7
        
        components["market_phase_modifier"] = 1.0
        
        return components
    
    def _generate_phase_decision(self, market_phase: str, score: float, components: Dict) -> str:
        """Generate decision based on phase and score"""
        
        # Phase-specific thresholds
        if market_phase == "pre-pump":
            if score >= 0.65:
                return "early_entry"  # Pre-pump specific decision
            elif score >= 0.45:
                return "monitor"
            else:
                return "avoid"
        
        elif market_phase == "breakout":
            if score >= 0.70:
                return "enter"
            elif score >= 0.50:
                return "confirm_breakout"
            else:
                return "avoid"
        
        elif market_phase == "trend-following":
            if score >= 0.70:
                return "join_trend"
            elif score >= 0.45:
                return "consider_entry"
            else:
                return "avoid"
        
        elif market_phase == "consolidation":
            if score >= 0.60:
                return "range_trade"
            elif score >= 0.40:
                return "wait_breakout"
            else:
                return "avoid"
        
        else:
            # Fallback decision logic
            if score >= 0.70:
                return "enter"
            elif score >= 0.45:
                return "consider"
            else:
                return "avoid"
    
    def _calculate_quality_grade(self, score: float, components: Dict) -> str:
        """Calculate quality grade based on score and component consistency"""
        
        # Base grade from score
        if score >= 0.80:
            base_grade = "excellent"
        elif score >= 0.65:
            base_grade = "strong"
        elif score >= 0.50:
            base_grade = "good"
        elif score >= 0.35:
            base_grade = "fair"
        else:
            base_grade = "weak"
        
        # Check component consistency
        component_values = [v for v in components.values() if isinstance(v, (int, float))]
        if component_values:
            consistency = 1.0 - (max(component_values) - min(component_values))
            if consistency < 0.3:  # Very inconsistent components
                if base_grade in ["excellent", "strong"]:
                    base_grade = "good"
                elif base_grade == "good":
                    base_grade = "fair"
        
        return base_grade

def analyze_symbol_with_unified_tjde(symbol: str, market_data: Dict, candles_15m: List, 
                                   candles_5m: List, signals: Dict = None) -> Dict:
    """
    Main function to analyze symbol with Unified TJDE Engine
    
    Args:
        symbol: Trading symbol
        market_data: Market data dictionary
        candles_15m: 15-minute candles
        candles_5m: 5-minute candles
        signals: Pre-calculated signals (optional)
        
    Returns:
        dict: Complete TJDE analysis result
    """
    try:
        engine = UnifiedTJDEEngine()
        
        # Detect market phase
        market_phase = engine.detect_market_phase(symbol, market_data, candles_15m, candles_5m)
        
        # Prepare signals if not provided
        if signals is None:
            signals = {
                "trend_strength": 0.5,
                "clip_confidence": 0.0,
                "gpt_label": "unknown",
                "liquidity_behavior": 0.5,
                "volume_behavior_score": 0.5,
                "pullback_quality": 0.5,
                "support_reaction_strength": 0.5,
                "bounce_confirmation_strength": 0.5,
                "momentum_persistence": 0.5,
                "volume_confirmation": 0.5
            }
        
        # Calculate unified score
        result = engine.calculate_unified_score(symbol, market_data, signals, market_phase)
        
        # Add candle information
        result["candles_15m_count"] = len(candles_15m) if candles_15m else 0
        result["candles_5m_count"] = len(candles_5m) if candles_5m else 0
        
        return result
        
    except Exception as e:
        print(f"[UNIFIED TJDE ERROR] {symbol}: {e}")
        return {
            "symbol": symbol,
            "market_phase": "error",
            "final_score": 0.0,
            "decision": "avoid",
            "quality_grade": "error",
            "error": str(e)
        }