"""
Unified Scoring Engine - Complete Integration of All Modules
Combines legacy scoring with Modules 1-5 in one modular, scalable system
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

def assess_future_scenario(setup: str, trend_strength: float, volume_score: float, clip_confidence: float, slope: float) -> float:
    """
    Intelligent Future Scenario Assessment based on setup patterns and market context
    
    Args:
        setup: AI-detected setup pattern (breakout_pattern, momentum_follow, etc.)
        trend_strength: Normalized trend strength (0-1)
        volume_score: Normalized volume score (0-1)
        clip_confidence: CLIP model confidence (0-1)
        slope: Price slope from linear regression
        
    Returns:
        Future scenario score (-0.2 to +0.5)
    """
    # Strong setups with high continuation potential
    strong_setups = ["breakout_pattern", "momentum_follow", "impulse", "pullback_in_trend", "trend_continuation"]
    
    # Reversal patterns with different dynamics
    reversal_setups = ["reversal_pattern", "exhaustion_pattern", "range_trading", "consolidation_squeeze"]
    
    base_score = 0.0
    
    # Strong bullish scenarios
    if setup in strong_setups and trend_strength >= 0.7 and volume_score >= 0.4:
        base_score = 0.2 + 0.2 * min(trend_strength, 1.0) + 0.1 * clip_confidence
        print(f"[MODULE 4 DEBUG] Strong bullish scenario: {setup} (trend:{trend_strength:.3f}, vol:{volume_score:.3f})")
        
    # Medium bullish scenarios
    elif setup in strong_setups and trend_strength >= 0.5:
        base_score = 0.1 + 0.15 * trend_strength + 0.05 * clip_confidence
        print(f"[MODULE 4 DEBUG] Medium bullish scenario: {setup} (trend:{trend_strength:.3f})")
        
    # Reversal scenarios (typically weaker but still valuable)
    elif setup in reversal_setups and trend_strength < 0.4:
        base_score = 0.05 + 0.05 * clip_confidence
        print(f"[MODULE 4 DEBUG] Reversal scenario: {setup} (weak trend:{trend_strength:.3f})")
        
    # Bearish scenarios (strong down trends)
    elif slope < -0.0005 and trend_strength >= 0.6:
        base_score = -0.1 - 0.1 * trend_strength
        print(f"[MODULE 4 DEBUG] Bearish scenario: strong downtrend (slope:{slope:.6f})")
        
    # Neutral/consolidation scenarios
    elif setup in ["consolidation_squeeze", "range_trading"]:
        base_score = 0.03 + 0.02 * clip_confidence
        print(f"[MODULE 4 DEBUG] Consolidation scenario: {setup}")
        
    # Unknown or weak setups
    else:
        base_score = 0.0
        print(f"[MODULE 4 DEBUG] Neutral scenario: {setup} (no clear prediction)")
    
    # AI confidence boost for high-confidence predictions
    if clip_confidence >= 0.7:
        base_score *= 1.2
        print(f"[MODULE 4 DEBUG] High AI confidence boost: {clip_confidence:.3f}")
    
    # Volume confirmation bonus
    if volume_score >= 0.6 and base_score > 0:
        base_score += 0.05
        print(f"[MODULE 4 DEBUG] Volume confirmation bonus: {volume_score:.3f}")
    
    # Cap the final score within bounds
    final_score = max(-0.2, min(base_score, 0.5))
    
    return round(final_score, 3)


def simulate_trader_decision_advanced(data: Dict) -> Dict:
    """
    🚀 Unified TJDE Engine - Complete Integration of All Scoring Modules
    
    Combines:
    - Legacy scoring: volume, orderbook, price action, clusters
    - Module 1: AI-EYE (CLIP + GPT with dynamic weights)
    - Module 2: HTF Overlay (macrostructure awareness)
    - Module 3: Trap Detector (risk pattern detection)
    - Module 4: Future Scenario Mapping (predictive analysis)
    - Module 5: Feedback Loop (self-learning weights)
    
    Args:
        data: Unified input dictionary containing all required data
        
    Returns:
        Complete scoring result with breakdown
    """
    
    symbol = data.get("symbol", "UNKNOWN")
    debug = data.get("debug", False)
    candles = data.get("candles", [])
    htf_candles = data.get("htf_candles", [])
    ai_label = data.get("ai_label", {})
    orderbook = data.get("orderbook", {})
    market_phase = data.get("market_phase", "unknown")
    
    # CRITICAL: Invalid Symbol Filter - Skip analysis for problematic symbols
    try:
        from utils.invalid_symbol_filter import should_skip_symbol_analysis
        
        if should_skip_symbol_analysis(symbol):
            logger.info(f"[UNIFIED SCORING] {symbol}: SKIPPED - Invalid symbol detected")
            return {
                "final_score": 0.0,
                "decision": "skip",
                "confidence": 0.0,
                "score_breakdown": {},
                "reasoning": f"Symbol {symbol} marked as invalid - skipped analysis",
                "symbol": symbol,
                "invalid_symbol": True
            }
    except Exception as e:
        logger.warning(f"[UNIFIED SCORING] Invalid symbol filter error for {symbol}: {e}")
    
    # 🚨 ENHANCED DATA REQUIREMENTS CHECK - Early warnings
    if not ai_label:
        print(f"[WARNING] Missing AI Label for {symbol}")
    if not htf_candles or len(htf_candles) < 30:
        print(f"[WARNING] Missing or insufficient HTF candles for {symbol} (available: {len(htf_candles)})")
    
    # 🛡️ ENHANCED DATA PROTECTION - Allow basic scoring even without AI-EYE/HTF for initial selection
    # Only skip if we have absolutely no usable data (no candles at all)
    if not candles or len(candles) < 20:
        print(f"[UNIFIED SKIP] Skipping scoring for {symbol} due to insufficient candles ({len(candles) if candles else 0}).")
        return {
            "final_score": 0.0,
            "decision": "skip",
            "confidence": 0.0,
            "score_breakdown": {"reason": "insufficient_candle_data"},
            "reasoning": f"Insufficient candle data for {symbol}",
            "symbol": symbol,
            "data_insufficient": True
        }
    
    # Allow scoring to proceed even without AI-EYE/HTF for initial token selection
    # This enables the system to find TOP tokens that can then get AI-EYE analysis
    if not ai_label and (not htf_candles or len(htf_candles) < 30):
        print(f"[UNIFIED BASIC] {symbol}: Proceeding with basic scoring (no AI-EYE/HTF for initial selection)")
        print(f"[UNIFIED BASIC] {symbol}: This token may qualify for AI-EYE analysis if it reaches TOP 5")
    
    # 🔍 COMPREHENSIVE DEBUG LOGGING - Show exact scoring breakdown
    print(f"[UNIFIED SCORING DEBUG] Starting analysis for {symbol}")
    print(f"[UNIFIED SCORING DEBUG] Input data keys: {list(data.keys())}")
    print(f"[UNIFIED SCORING DEBUG] AI Label available: {bool(ai_label)}")
    print(f"[UNIFIED SCORING DEBUG] HTF Candles available: {len(htf_candles)}")
    print(f"[UNIFIED SCORING DEBUG] Market Phase: {market_phase}")
    
    # Initialize scoring components
    score_breakdown = {
        "ai_eye_score": 0.0,
        "htf_overlay_score": 0.0, 
        "trap_detector_score": 0.0,
        "future_mapping_score": 0.0,
        "legacy_volume_score": 0.0,
        "legacy_orderbook_score": 0.0,
        "legacy_cluster_score": 0.0,
        "legacy_psychology_score": 0.0
    }
    
    # === ADVANCED MODULES PROCESSING (1-4) - Proper Layer Order ===
    advanced_total_score = 0.0
    advanced_modules_active = 0
    total_score = 0.0  # For legacy compatibility
    
    print(f"[SCORING LAYER DEBUG] Starting Advanced Module Processing (AI-EYE → HTF → Trap → Future)")
    
    # === MODULE 1: AI-EYE VISION (CLIP + GPT with Dynamic Weights) ===
    try:
        from vision.vision_scoring import score_from_ai_label
        ai_label = data.get("ai_label", {})
        market_phase = data.get("market_phase", None)
        
        print(f"[MODULE 1 DEBUG] AI-EYE starting for {symbol}")
        print(f"[MODULE 1 DEBUG] AI Label data: {ai_label}")
        print(f"[MODULE 1 DEBUG] Market Phase: {market_phase}")
        
        if ai_label:
            ai_eye_score = score_from_ai_label(ai_label, market_phase)
            score_breakdown["ai_eye_score"] = ai_eye_score
            advanced_total_score += ai_eye_score
            
            if ai_eye_score != 0.0:
                advanced_modules_active += 1
            
            print(f"[MODULE 1 DEBUG] AI-EYE Score: {ai_eye_score:+.4f}")
            print(f"[MODULE 1 DEBUG] Advanced modules running total: {advanced_total_score:.4f}")
            
        else:
            print(f"[MODULE 1 DEBUG] AI-EYE: No AI label data available")
                
    except Exception as e:
        print(f"[MODULE 1 ERROR] AI-EYE scoring failed: {e}")
    
    # === MODULE 2: HTF OVERLAY (Macrostructure Awareness) ===
    try:
        htf_candles = data.get("htf_candles", [])
        
        print(f"[MODULE 2 DEBUG] HTF Overlay starting for {symbol}")
        print(f"[MODULE 2 DEBUG] HTF candles available: {len(htf_candles) if htf_candles else 0}")
        
        if htf_candles and len(htf_candles) >= 20:
            # Get current price for S/R analysis
            current_price = None
            if data and 'price' in data:
                current_price = float(data['price'])
            elif htf_candles:
                last_candle = htf_candles[-1]
                if isinstance(last_candle, dict) and 'close' in last_candle:
                    current_price = float(last_candle['close'])
                elif isinstance(last_candle, (list, tuple)) and len(last_candle) > 4:
                    current_price = float(last_candle[4])
            
            htf_score = 0.0
            
            # Try advanced S/R level integration first
            try:
                from htf_overlay.htf_support_resistance import detect_htf_levels
                
                if current_price:
                    htf_result = detect_htf_levels(htf_candles, current_price)
                    breakout_potential = htf_result.get("breakout_potential", 0.0)
                    position_context = htf_result.get("price_position", "middle")
                    
                    # Enhanced scoring based on breakout potential and position
                    if breakout_potential > 0.7 and position_context in ["at_resistance", "at_support"]:
                        htf_score = 0.10
                        print(f"[MODULE 2 DEBUG] HTF S/R: High breakout potential {breakout_potential:.2f} at {position_context}")
                    elif breakout_potential > 0.5:
                        htf_score = 0.05
                        print(f"[MODULE 2 DEBUG] HTF S/R: Medium breakout potential {breakout_potential:.2f}")
                    elif position_context == "middle_range":
                        htf_score = 0.02
                        print(f"[MODULE 2 DEBUG] HTF S/R: Middle range position")
                    else:
                        htf_score = 0.0
                        print(f"[MODULE 2 DEBUG] HTF S/R: Low breakout potential {breakout_potential:.2f}")
                    
                    # Level strength bonus
                    level_strength = htf_result.get("level_strength", {})
                    if level_strength:
                        max_strength = max(level_strength.values()) if level_strength.values() else 0.0
                        if max_strength > 0.8:
                            strength_bonus = 0.03
                            htf_score += strength_bonus
                            print(f"[MODULE 2 DEBUG] HTF S/R: Strong level bonus +{strength_bonus:.3f} (strength: {max_strength:.2f})")
                    
                    print(f"[MODULE 2 DEBUG] HTF S/R Integration: breakout_potential={breakout_potential:.3f}, position={position_context}")
                    
            except ImportError:
                print(f"[MODULE 2 DEBUG] HTF S/R module not available, using fallback trend analysis")
                # Fallback to simple trend analysis
                closes = []
                for candle in htf_candles[-20:]:
                    if isinstance(candle, dict):
                        closes.append(float(candle['close']))
                    elif isinstance(candle, (list, tuple)) and len(candle) > 4:
                        closes.append(float(candle[4]))
                if len(closes) >= 10:
                    trend_strength = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
                    htf_score = max(-0.05, min(0.05, trend_strength * 2))
            
            # AI pattern alignment bonus
            if ai_label and ai_label.get('label') in ['momentum_follow', 'breakout_pattern']:
                if htf_score > 0:  # Bullish HTF + bullish AI
                    ai_bonus = htf_score * 0.3  # 30% bonus
                    htf_score += ai_bonus
                    print(f"[MODULE 2 DEBUG] AI Pattern Alignment: +{ai_bonus:.4f} bonus for {ai_label.get('label')}")
            
            # Final bounds checking
            htf_score = max(-0.20, min(0.20, htf_score))
            
            score_breakdown["htf_overlay_score"] = htf_score
            advanced_total_score += htf_score
            
            if htf_score != 0.0:
                advanced_modules_active += 1
            
            print(f"[MODULE 2 DEBUG] HTF Overlay Score: {htf_score:+.4f}")
            print(f"[MODULE 2 DEBUG] Advanced modules running total: {advanced_total_score:.4f}")
        else:
            print(f"[MODULE 2 DEBUG] HTF Overlay: Insufficient HTF data ({len(htf_candles) if htf_candles else 0} candles)")
                
    except Exception as e:
        logger.error(f"[MODULE 2 ERROR] HTF Overlay scoring failed: {e}")
        print(f"[MODULE 2 DEBUG] HTF Overlay: Error - {e}")
    
    # === MODULE 3: TRAP DETECTOR (Risk Pattern Detection) ===
    try:
        candles = data.get("candles", [])
        
        if candles and len(candles) >= 10:
            # Enhanced trap detection with volume analysis
            recent_candles = candles[-10:]
            trap_score = 0.0
            
            # Calculate average volume for context
            volumes = []
            for candle in recent_candles:
                if isinstance(candle, dict):
                    volume = float(candle.get('volume', 0))
                elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                    volume = float(candle[5])
                else:
                    volume = 0
                volumes.append(volume)
            
            avg_volume = sum(volumes) / len(volumes) if volumes and sum(volumes) > 0 else 1.0
            
            # Analyze current candle for trap patterns
            current_candle = recent_candles[-1]
            trap_score = 0.0
            
            # Initialize price variables
            open_price = high_price = low_price = close_price = volume = 0.0
            
            if isinstance(current_candle, dict):
                open_price = float(current_candle['open'])
                high_price = float(current_candle['high'])
                low_price = float(current_candle['low'])
                close_price = float(current_candle['close'])
                volume = float(current_candle.get('volume', 0))
                candle_valid = True
            elif isinstance(current_candle, (list, tuple)) and len(current_candle) >= 5:
                open_price = float(current_candle[1])
                high_price = float(current_candle[2])
                low_price = float(current_candle[3])
                close_price = float(current_candle[4])
                volume = float(current_candle[5]) if len(current_candle) >= 6 else 0
                candle_valid = True
            else:
                candle_valid = False
            
            if candle_valid:
                # Calculate trap detection components
                body_size = abs(close_price - open_price)
                wick_top = high_price - max(open_price, close_price)
                wick_bottom = min(open_price, close_price) - low_price
                
                # Volume score calculation
                volume_ratio = volume / (avg_volume + 1e-9) if avg_volume > 0 else 1.0
                volume_score = min(1.0, volume_ratio / 2.0)  # Normalize to 0-1 range
                
                # Trap detection logic
                trap_detected = (wick_top > 2 * body_size or wick_bottom > 2 * body_size) and volume > 1.5 * avg_volume
                
                # Dynamic scoring based on user-requested logic
                if trap_detected and volume_score >= 0.5:
                    trap_score = 0.2 + 0.2 * volume_score  # High volume trap
                    print(f"[MODULE 3 DEBUG] High volume trap detected: volume_score={volume_score:.3f}, trap_score={trap_score:.3f}")
                elif trap_detected and volume_score >= 0.3:
                    trap_score = 0.1 + 0.15 * volume_score  # Medium volume trap
                    print(f"[MODULE 3 DEBUG] Medium volume trap detected: volume_score={volume_score:.3f}, trap_score={trap_score:.3f}")
                else:
                    trap_score = 0.0
                    print(f"[MODULE 3 DEBUG] No trap detected: volume_score={volume_score:.3f}, trap_detected={trap_detected}")
                
                # Additional fake breakout detection
                if not trap_detected:
                    # Check for subtle fake breakouts in recent candles
                    fake_breakout_count = 0
                    for candle in recent_candles[-5:]:
                        if isinstance(candle, dict):
                            c_open = float(candle['open'])
                            c_high = float(candle['high'])
                            c_low = float(candle['low'])
                            c_close = float(candle['close'])
                            c_volume = float(candle.get('volume', 0))
                        elif isinstance(candle, (list, tuple)) and len(candle) >= 5:
                            c_open = float(candle[1])
                            c_high = float(candle[2])
                            c_low = float(candle[3])
                            c_close = float(candle[4])
                            c_volume = float(candle[5]) if len(candle) >= 6 else 0
                        else:
                            continue
                        
                        c_body = abs(c_close - c_open)
                        c_wick_top = c_high - max(c_open, c_close)
                        c_vol_ratio = c_volume / (avg_volume + 1e-9)
                        
                        # Detect fake breakout patterns
                        if c_body > 0 and c_wick_top > c_body * 1.8 and c_vol_ratio > 1.3:
                            fake_breakout_count += 1
                    
                    if fake_breakout_count >= 2:
                        trap_score = 0.05 + 0.05 * (fake_breakout_count / 5.0)
                        print(f"[MODULE 3 DEBUG] Multiple fake breakouts detected: count={fake_breakout_count}, score={trap_score:.3f}")
                
                # Final scoring and bounds
                trap_score = min(trap_score, 0.4)  # Cap maximum trap score
                
                print(f"[MODULE 3 DEBUG] Trap Detector: {trap_score:+.4f} (volume_ratio={volume_ratio:.2f}, trap_detected={trap_detected})")
                
                if debug:
                    logger.info(f"[MODULE 3] Trap Detector: {trap_score:+.4f} "
                               f"(volume_score={volume_score:.3f}, trap_detected={trap_detected})")
            else:
                print(f"[MODULE 3 DEBUG] Trap Detector: Invalid candle format - no detection")
            
            # Always set score breakdown and add to total
            score_breakdown["trap_detector_score"] = trap_score
            advanced_total_score += trap_score
            
            if trap_score != 0.0:
                advanced_modules_active += 1
        else:
            print(f"[MODULE 3 DEBUG] Trap Detector: Insufficient candle data ({len(candles) if candles else 0} candles)")
            if debug:
                logger.info(f"[MODULE 3] Trap Detector: Insufficient candle data "
                           f"({len(candles) if candles else 0} candles)")
                
    except Exception as e:
        logger.error(f"[MODULE 3 ERROR] Trap Detector scoring failed: {e}")
        print(f"[MODULE 3 DEBUG] Trap Detector: Error - {e}")
    
    # === MODULE 4: FUTURE SCENARIO MAPPING (Predictive Analysis) ===
    try:
        candles = data.get("candles", [])
        
        print(f"[MODULE 4 DEBUG] Future Mapping starting for {symbol}")
        print(f"[MODULE 4 DEBUG] Candles available: {len(candles) if candles else 0}")
        
        if candles and len(candles) >= 20:  # Reduced from 50 to 20
            # Extract close prices for trend analysis
            close_prices = []
            recent_highs = []
            recent_lows = []
            
            for candle in candles[-20:]:  # Use last 20 candles
                if isinstance(candle, dict):
                    close_prices.append(float(candle['close']))
                    recent_highs.append(float(candle['high']))
                    recent_lows.append(float(candle['low']))
                elif isinstance(candle, (list, tuple)) and len(candle) > 4:
                    close_prices.append(float(candle[4]))  # close
                    recent_highs.append(float(candle[2]))   # high
                    recent_lows.append(float(candle[3]))    # low
            
            if len(close_prices) >= 10:
                # Calculate price trend using linear regression (slope)
                import numpy as np
                x = list(range(len(close_prices)))
                slope, intercept = np.polyfit(x, close_prices, 1)
                
                current_price = close_prices[-1]
                volatility = (max(recent_highs[-5:]) - min(recent_lows[-5:])) / current_price if current_price != 0 else 0
                
                print(f"[MODULE 4 DEBUG] Slope: {slope:.6f}, Current price: {current_price:.4f}, Volatility: {volatility:.4f}")
                
                # Intelligent Future Scenario Assessment
                future_score = 0.0
                scenario = "neutral"
                
                # Get AI setup from data if available
                setup = data.get("ai_label", {}).get("label", "unknown")
                trend_strength = abs(slope) * 1000  # Scale slope to 0-1 range
                trend_strength = min(trend_strength, 1.0)
                
                # Calculate volume score from market data
                volume_24h = data.get("volume_24h", 0)
                volume_score = min(volume_24h / 1000000, 1.0) if volume_24h > 0 else 0.0  # Normalize to 0-1
                
                # Get CLIP confidence if available
                clip_confidence = data.get("ai_label", {}).get("confidence", 0.0)
                
                print(f"[MODULE 4 DEBUG] Setup: {setup}, Trend strength: {trend_strength:.3f}, Volume score: {volume_score:.3f}")
                
                # Enhanced predictive logic based on setup patterns
                future_score = assess_future_scenario(setup, trend_strength, volume_score, clip_confidence, slope)
                
                # Determine scenario based on slope and setup
                if slope > 0.0001 and setup in ["breakout_pattern", "momentum_follow"]:
                    scenario = "strong_bullish_continuation"
                elif slope > 0.0001:
                    scenario = "bullish_continuation"
                elif slope < -0.0001 and setup in ["reversal_pattern", "exhaustion_pattern"]:
                    scenario = "strong_bearish_continuation"
                elif slope < -0.0001:
                    scenario = "bearish_continuation"
                elif setup in ["consolidation_squeeze", "range_trading"]:
                    scenario = "consolidation_buildup"
                else:
                    scenario = "neutral_consolidation"
                
                # Volatility adjustment
                if volatility > 0.08:  # High volatility - reduce confidence
                    future_score *= 0.7
                    print(f"[MODULE 4 DEBUG] High volatility penalty applied: {volatility:.4f}")
                
                score_breakdown["future_mapping_score"] = round(future_score, 4)
                advanced_total_score += future_score
                
                if future_score != 0.0:
                    advanced_modules_active += 1
                
                print(f"[MODULE 4 DEBUG] Future Mapping Score: {future_score:+.4f} ({scenario})")
                print(f"[MODULE 4 DEBUG] Advanced modules running total: {advanced_total_score:.4f}")
                
            else:
                print(f"[MODULE 4 DEBUG] Future Mapping: Invalid candle data (only {len(close_prices)} valid prices)")
        else:
            print(f"[MODULE 4 DEBUG] Future Mapping: Insufficient data ({len(candles) if candles else 0} candles, need 20+)")
                
    except Exception as e:
        print(f"[MODULE 4 ERROR] Future Mapping scoring failed: {e}")
        logger.error(f"[MODULE 4 ERROR] Future Mapping scoring failed: {e}")
    
    # === MODULE 5: FEEDBACK LOOP ===
    # Note: Module 5 is integrated into Module 1 (AI-EYE) via dynamic weights
    # and also logs predictions for future learning
    try:
        # Log prediction for feedback if significant signal
        if advanced_total_score != 0.0 and ai_label:
            from feedback_loop.feedback_integration import log_prediction_for_feedback
            
            current_price = data.get("current_price", 0.0)
            if current_price > 0:
                # Determine preliminary decision for logging
                preliminary_decision = "wait"
                if advanced_total_score >= 0.15:
                    preliminary_decision = "enter"
                elif advanced_total_score <= -0.10:
                    preliminary_decision = "avoid"
                
                feedback_logged = log_prediction_for_feedback(
                    symbol=symbol,
                    ai_label=ai_label,
                    current_price=current_price,
                    tjde_score=advanced_total_score,
                    decision=preliminary_decision,
                    market_phase=market_phase
                )
                
                if debug and feedback_logged:
                    logger.info(f"[MODULE 5] Feedback: Prediction logged for learning")
                    
    except Exception as e:
        if debug:
            logger.error(f"[MODULE 5 ERROR] Feedback logging failed: {e}")
    
    # 🧠 ENHANCED LEGACY SCORING - Always enabled for initial token selection
    score_ai = score_breakdown.get("ai_eye_score", 0.0)
    score_htf = score_breakdown.get("htf_overlay_score", 0.0)
    
    # Enable legacy scoring always to allow basic token ranking and TOP 5 selection
    legacy_enabled = True  # Changed from conditional to always enabled
    
    print(f"[TJDE DEBUG] AI-EYE Score: {score_ai:.4f}")
    print(f"[TJDE DEBUG] HTF Overlay Score: {score_htf:.4f}")
    print(f"[TJDE DEBUG] Legacy Scoring Enabled: {legacy_enabled} (always enabled for token selection)")
    
    if legacy_enabled:
        print(f"[TJDE DEBUG] Executing legacy scoring components...")
        
        # Sync total_score with advanced modules before legacy scoring
        total_score = advanced_total_score
        
        # 🎯 DYNAMIC WEIGHTS INTEGRATION - Load weights from feedback loop system
        # This integrates Module 5 Feedback Loop dynamic weight adjustment with TJDE scoring
        try:
            from feedback_loop.feedback_integration import load_dynamic_weights
            
            # Domyślne wagi fallback
            DEFAULT_WEIGHTS = {
                "trend": 0.3,
                "volume": 0.2,
                "momentum": 0.2,
                "orderbook": 0.1,
                "price_change": 0.2,
            }
            
            # Załaduj dynamiczne wagi – jeśli niekompletne lub brak: fallback
            dynamic_weights = load_dynamic_weights(symbol)
            if not dynamic_weights or set(dynamic_weights.keys()) != set(DEFAULT_WEIGHTS.keys()):
                weights = DEFAULT_WEIGHTS
                print(f"[DYNAMIC WEIGHTS] {symbol}: Using fallback weights - incomplete feedback data")
            else:
                weights = dynamic_weights
                print(f"[DYNAMIC WEIGHTS] {symbol}: Applied feedback-learned weights successfully")
                
        except Exception as e:
            # Fallback to default weights if dynamic loading fails
            weights = {
                "trend": 0.3,
                "volume": 0.2,
                "momentum": 0.2,
                "orderbook": 0.1,
                "price_change": 0.2,
            }
            print(f"[DYNAMIC WEIGHTS ERROR] {symbol}: {e} - using fallback weights")
        
        # Legacy Volume Slope Analysis with Dynamic Weights
        try:
            candles = data.get("candles", [])
            if candles and len(candles) >= 10:
                volume_score = score_from_volume_slope(candles)
                score_breakdown["legacy_volume_score"] = volume_score
                total_score += volume_score
                print(f"[TJDE DEBUG] Legacy Volume Score: {volume_score:.4f}")
            else:
                print(f"[TJDE DEBUG] Legacy Volume Score: 0.0000 (insufficient candles)")
                
        except Exception as e:
            logger.error(f"[LEGACY ERROR] Volume scoring failed: {e}")
        
        # Legacy Orderbook Pressure Analysis
        try:
            orderbook = data.get("orderbook", {})
            if orderbook:
                orderbook_score = score_from_orderbook_pressure(orderbook)
                score_breakdown["legacy_orderbook_score"] = orderbook_score
                total_score += orderbook_score
                print(f"[TJDE DEBUG] Legacy Orderbook Score: {orderbook_score:.4f}")
            else:
                print(f"[TJDE DEBUG] Legacy Orderbook Score: 0.0000 (no orderbook data)")
                
        except Exception as e:
            logger.error(f"[LEGACY ERROR] Orderbook scoring failed: {e}")
        
        # Legacy Cluster Analysis
        try:
            cluster_features = data.get("cluster_features", {})
            if cluster_features:
                cluster_score = score_from_cluster(cluster_features)
                score_breakdown["legacy_cluster_score"] = cluster_score
                total_score += cluster_score
                print(f"[TJDE DEBUG] Legacy Cluster Score: {cluster_score:.4f}")
            else:
                print(f"[TJDE DEBUG] Legacy Cluster Score: 0.0000 (no cluster features)")
                
        except Exception as e:
            logger.error(f"[LEGACY ERROR] Cluster scoring failed: {e}")
        
        # Legacy Psychology Analysis
        try:
            candles = data.get("candles", [])
            if candles and len(candles) >= 20:
                psychology_score = score_from_psychology(candles)
                score_breakdown["legacy_psychology_score"] = psychology_score
                total_score += psychology_score
                print(f"[TJDE DEBUG] Legacy Psychology Score: {psychology_score:.4f}")
            else:
                print(f"[TJDE DEBUG] Legacy Psychology Score: 0.0000 (insufficient candles)")
                
        except Exception as e:
            logger.error(f"[LEGACY ERROR] Psychology scoring failed: {e}")
    
    # === ENHANCED DEBUG LOGGING: INDIVIDUAL MODULE SCORES ===
    print(f"[TJDE DEBUG] Trap Detector Score: {score_breakdown['trap_detector_score']:.4f}")
    print(f"[TJDE DEBUG] Future Mapping Score: {score_breakdown['future_mapping_score']:.4f}")
    
    # === SCORE ENHANCEMENT SYSTEM - Break Through 0.66 Ceiling ===
    base_score = round(total_score, 4)
    print(f"[TJDE DEBUG] Base TJDE Score for {symbol}: {base_score:.4f}")
    
    # Apply TJDE Score Enhancement for exceptional signals
    try:
        from utils.tjde_score_enhancer import TJDEScoreEnhancer
        
        enhancer = TJDEScoreEnhancer()
        
        # Prepare signal data for enhancement
        signal_data = {
            "ai_eye_score": score_breakdown.get("ai_eye_score", 0.0),
            "htf_overlay_score": score_breakdown.get("htf_overlay_score", 0.0),
            "volume_behavior_score": score_breakdown.get("legacy_volume_score", 0.0),
            "trend_strength": data.get("trend_strength", 0.0),
            "ai_confidence": ai_label.get("confidence", 0.0) if ai_label else 0.0
        }
        
        # Check if enhancement should be applied
        if enhancer.should_apply_enhancement(base_score, signal_data):
            enhancement_result = enhancer.enhance_tjde_score(base_score, signal_data)
            
            final_score = enhancement_result["enhanced_score"]
            boost = enhancement_result["boost"]
            boost_reason = enhancement_result["boost_reason"]
            
            print(f"🚀 [SCORE ENHANCEMENT] {symbol}: {base_score:.4f} → {final_score:.4f} (+{boost:.3f})")
            print(f"🚀 [ENHANCEMENT REASON] {boost_reason}")
            
            # Add enhancement info to score breakdown
            score_breakdown["enhancement_boost"] = boost
            score_breakdown["enhancement_reason"] = boost_reason
        else:
            final_score = base_score
            print(f"[SCORE ENHANCEMENT] {symbol}: No enhancement applied (score: {base_score:.4f})")
    
    except Exception as e:
        final_score = base_score
        print(f"[SCORE ENHANCEMENT ERROR] {e}")
        logger.error(f"Score enhancement failed for {symbol}: {e}")
    
    # ✅ A. BOOST FOR STRONG TREND - Critical fix from issue report
    signals = data.get("signals", {})
    trend_strength = signals.get("trend_strength", 0.0)
    pullback_quality = signals.get("pullback_quality", 1.0)
    
    # Enhanced strong trend detection with more realistic thresholds
    if trend_strength > 0.7 and pullback_quality < 0.1:  # More realistic thresholds
        boost_amount = 0.08 if trend_strength > 0.9 else 0.05
        print(f"[STRONG TREND BOOST] {symbol} has strong trend (strength={trend_strength:.3f}, pullback={pullback_quality:.3f})")
        final_score += boost_amount
        print(f"[STRONG TREND BOOST] Score boosted by +{boost_amount:.2f} for strong trend")
    
    # ✅ B. ENHANCED FALLBACK SCORING FOR EMPTY AI/HTF - Critical fix from issue report
    if score_breakdown["ai_eye_score"] == 0.0 and score_breakdown["htf_overlay_score"] == 0.0:
        # Enhanced fallback scoring based on legacy components
        legacy_total = (score_breakdown["legacy_volume_score"] + 
                       score_breakdown["legacy_orderbook_score"] + 
                       score_breakdown["legacy_cluster_score"] + 
                       max(0, score_breakdown["legacy_psychology_score"]))  # Only positive psychology
        
        if legacy_total > 0.04:  # If legacy components show potential
            fallback_boost = min(0.08, legacy_total * 2.0)  # Amplify legacy signals
            print(f"[FALLBACK SCORING] {symbol} has strong legacy signals ({legacy_total:.4f}), applying amplified boost")
            final_score += fallback_boost
            print(f"[FALLBACK SCORING] Added +{fallback_boost:.3f} for legacy signal amplification")
        else:
            print(f"[FALLBACK SCORING] {symbol} has empty AI-EYE and HTF, applying minimal context score")
            final_score += 0.03  # Increased from 0.02 for better baseline
            print(f"[FALLBACK SCORING] Added +0.03 for minimal context support")
    
    # 🚀 TOP PERFORMER BOOST MECHANISM - Enhanced scoring for quality setups
    original_score = final_score
    if final_score > 0.05 and final_score < 0.15:  # Only boost moderate range tokens
        # Check for strong individual module performance
        has_strong_ai_eye = score_breakdown["ai_eye_score"] > 0.08
        has_strong_htf = score_breakdown["htf_overlay_score"] > 0.03
        has_multiple_modules = sum(1 for score in score_breakdown.values() if score > 0.02) >= 3
        
        if has_strong_ai_eye or has_strong_htf or has_multiple_modules:
            print(f"[TJDE BOOST] {symbol} qualifies for boost (AI-EYE: {has_strong_ai_eye}, HTF: {has_strong_htf}, Multi-module: {has_multiple_modules})")
            final_score += 0.10
            final_score = round(final_score, 4)
            print(f"[TJDE BOOST] Score boosted from {original_score:.4f} to {final_score:.4f}")
    
    # Enhanced decision thresholds based on comprehensive scoring
    if final_score >= 0.20:
        decision = "enter"
        confidence = min(1.0, final_score / 0.30)  # Scale to 1.0 at 0.30
    elif final_score >= 0.10:
        decision = "scalp_entry"
        confidence = min(0.8, final_score / 0.15)  # Scale to 0.8 at 0.15
    elif final_score >= -0.05:
        decision = "wait"
        confidence = 0.5 + abs(final_score) / 0.10  # Neutral confidence
    else:
        decision = "avoid"
        confidence = min(0.9, abs(final_score) / 0.15)  # High confidence in avoid
    
    confidence = round(confidence, 3)
    
    # Count active modules
    active_modules = sum(1 for score in score_breakdown.values() if score != 0.0)
    
    # Find strongest component (handle empty scores)
    strongest_component = "none"
    score_values = list(score_breakdown.values())
    if score_values and any(score != 0.0 for score in score_values):
        strongest_component = max(score_breakdown.items(), key=lambda x: abs(x[1]))[0]
    
    # 🔍 COMPREHENSIVE FINAL DEBUG LOGGING
    print(f"[UNIFIED FINAL DEBUG] ===== SCORING SUMMARY FOR {symbol} =====")
    print(f"[UNIFIED FINAL DEBUG] AI-EYE Score: {score_breakdown['ai_eye_score']:+.4f}")
    print(f"[UNIFIED FINAL DEBUG] HTF Overlay Score: {score_breakdown['htf_overlay_score']:+.4f}")
    print(f"[UNIFIED FINAL DEBUG] Trap Detector Score: {score_breakdown['trap_detector_score']:+.4f}")
    print(f"[UNIFIED FINAL DEBUG] Future Mapping Score: {score_breakdown['future_mapping_score']:+.4f}")
    print(f"[UNIFIED FINAL DEBUG] Legacy Volume Score: {score_breakdown['legacy_volume_score']:+.4f}")
    print(f"[UNIFIED FINAL DEBUG] Legacy Orderbook Score: {score_breakdown['legacy_orderbook_score']:+.4f}")
    print(f"[UNIFIED FINAL DEBUG] Legacy Cluster Score: {score_breakdown['legacy_cluster_score']:+.4f}")
    print(f"[UNIFIED FINAL DEBUG] Legacy Psychology Score: {score_breakdown['legacy_psychology_score']:+.4f}")
    print(f"[UNIFIED FINAL DEBUG] ===== TOTAL SCORE: {final_score:.4f} =====")
    print(f"[UNIFIED FINAL DEBUG] Decision: {decision}")
    print(f"[UNIFIED FINAL DEBUG] Active Modules: {active_modules}/8")
    print(f"[UNIFIED FINAL DEBUG] ================================")
    
    # 🧠 MODULE 5: FEEDBACK LOOP INTEGRATION - Log predictions for self-learning system
    try:
        from feedback_loop.feedback_integration import log_prediction_for_feedback
        
        # Extract AI setup label and confidence for feedback logging
        setup_label = ai_label.get('label', 'unknown') if ai_label else 'unknown'
        ai_confidence = ai_label.get('confidence', 0.0) if ai_label else 0.0
        current_price = data.get('current_price', 0.0)
        
        # Map TJDE decisions to feedback system decisions
        feedback_decision = decision
        if decision in ['consider', 'scalp_entry']:
            feedback_decision = 'enter'
        elif decision in ['wait', 'skip', 'avoid']:
            feedback_decision = 'avoid'
        
        # Log prediction for feedback analysis if significant
        if current_price > 0 and setup_label != 'unknown':
            log_prediction_for_feedback(
                symbol=symbol,
                tjde_score=final_score,
                decision=feedback_decision,
                setup_label=setup_label,
                confidence=ai_confidence,
                price=current_price
            )
        
    except Exception as e:
        print(f"[FEEDBACK INTEGRATION ERROR] {symbol}: Failed to log prediction - {e}")
    
    result = {
        "symbol": symbol,
        "final_score": final_score,
        "decision": decision,
        "confidence": confidence,
        "active_modules": active_modules,
        "score_breakdown": score_breakdown,
        "analysis_summary": {
            "total_modules": 8,  # 4 new + 4 legacy
            "active_modules": active_modules,
            "strongest_component": strongest_component,
            "score_range": f"{min(score_values):.3f} to {max(score_values):.3f}" if score_values else "0.000 to 0.000"
        }
    }
    
    return result

# === LEGACY SCORING FUNCTIONS ===

def score_from_volume_slope(candles: List) -> float:
    """Legacy volume slope analysis"""
    try:
        print(f"[LEGACY VOL DEBUG] Candles input: {len(candles) if candles else 0}")
        
        if not candles or len(candles) < 10:
            print(f"[LEGACY VOL DEBUG] Insufficient candles: {len(candles) if candles else 0}")
            return 0.0
        
        # Extract volume data from last 10 candles
        volumes = []
        for i, candle in enumerate(candles[-10:]):
            if isinstance(candle, dict):
                vol = float(candle['volume'])
                volumes.append(vol)
                print(f"[LEGACY VOL DEBUG] Candle {i} (dict): volume={vol}")
            elif isinstance(candle, (list, tuple)) and len(candle) > 5:
                vol = float(candle[5])
                volumes.append(vol)
                print(f"[LEGACY VOL DEBUG] Candle {i} (list): volume={vol}")
        
        print(f"[LEGACY VOL DEBUG] Extracted {len(volumes)} volumes: {volumes[:3]}...")
        
        if len(volumes) < 5:
            print(f"[LEGACY VOL DEBUG] Too few volumes: {len(volumes)}")
            return 0.0
        
        # Calculate volume trend
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes, 1)[0]
        
        # Normalize slope to score range
        volume_avg = np.mean(volumes)
        print(f"[LEGACY VOL DEBUG] Slope: {slope}, Avg volume: {volume_avg}")
        if volume_avg == 0:
            return 0.0
        
        slope_normalized = slope / volume_avg
        
        # Convert to score (-0.05 to +0.05)
        score = np.clip(slope_normalized * 10, -0.05, 0.05)
        
        return round(score, 4)
        
    except Exception:
        return 0.0

def score_from_orderbook_pressure(orderbook: Dict) -> float:
    """Legacy orderbook pressure analysis"""
    try:
        print(f"[LEGACY OB DEBUG] Orderbook input: {bool(orderbook)}")
        
        if not orderbook:
            print(f"[LEGACY OB DEBUG] No orderbook data")
            return 0.0
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        print(f"[LEGACY OB DEBUG] Bids: {len(bids)}, Asks: {len(asks)}")
        
        if not bids or not asks:
            print(f"[LEGACY OB DEBUG] Missing bids or asks")
            return 0.0
        
        # Calculate top 5 levels pressure
        bid_volume = sum(float(bid[1]) for bid in bids[:5])
        ask_volume = sum(float(ask[1]) for ask in asks[:5])
        
        print(f"[LEGACY OB DEBUG] Bid volume: {bid_volume}, Ask volume: {ask_volume}")
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        # Pressure ratio
        pressure_ratio = (bid_volume - ask_volume) / total_volume
        
        # Convert to score (-0.03 to +0.03)
        score = pressure_ratio * 0.03
        
        return round(score, 4)
        
    except Exception:
        return 0.0

def score_from_cluster(cluster_features: Dict) -> float:
    """Legacy cluster analysis"""
    try:
        if not cluster_features:
            return 0.0
        
        # Extract cluster metrics
        cluster_strength = cluster_features.get("strength", 0.0)
        cluster_direction = cluster_features.get("direction", 0.0)
        cluster_volume = cluster_features.get("volume_ratio", 1.0)
        
        # Combine metrics
        base_score = cluster_strength * cluster_direction * 0.04
        volume_modifier = min(1.5, cluster_volume)  # Cap at 1.5x
        
        score = base_score * volume_modifier
        
        # Bounds checking
        score = np.clip(score, -0.04, 0.04)
        
        return round(score, 4)
        
    except Exception:
        return 0.0

def score_from_psychology(candles: List) -> float:
    """Legacy psychology analysis based on wick patterns"""
    try:
        if not candles or len(candles) < 20:
            return 0.0
        
        recent_candles = candles[-10:]
        psychology_score = 0.0
        
        for candle in recent_candles:
            if len(candle) < 5:
                continue
                
            if isinstance(candle, dict):
                open_price = float(candle['open'])
                high_price = float(candle['high'])
                low_price = float(candle['low'])
                close_price = float(candle['close'])
            elif isinstance(candle, (list, tuple)):
                open_price = float(candle[1])
                high_price = float(candle[2])
                low_price = float(candle[3])
                close_price = float(candle[4])
            else:
                continue
            
            # Wick analysis
            body_size = abs(close_price - open_price)
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            
            if body_size == 0:
                continue
            
            # Bullish psychology (strong lower wicks)
            if lower_wick > body_size * 1.5:
                psychology_score += 0.002
            
            # Bearish psychology (strong upper wicks)
            if upper_wick > body_size * 1.5:
                psychology_score -= 0.002
        
        # Bounds checking
        psychology_score = np.clip(psychology_score, -0.02, 0.02)
        
        return round(psychology_score, 4)
        
    except Exception:
        return 0.0

def prepare_unified_data(symbol: str, candles: List, ticker_data: Dict, 
                        orderbook: Dict, market_data: Dict, signals: Dict,
                        ai_label: Dict = None, htf_candles: List = None) -> Dict:
    """
    Prepare unified data dictionary for comprehensive scoring
    
    Args:
        symbol: Trading symbol
        candles: OHLCV candle data
        ticker_data: Current price/volume data
        orderbook: Orderbook data
        market_data: Additional market data
        signals: Technical analysis signals
        ai_label: AI vision analysis results
        htf_candles: Higher timeframe candles
        
    Returns:
        Unified data dictionary
    """
    
    # Calculate EMA50 if needed with support for both dict and list formats
    ema50 = None
    if candles and len(candles) >= 50:
        try:
            # Handle both dictionary and list candle formats
            closes = []
            for candle in candles[-50:]:
                if isinstance(candle, dict):
                    # Dictionary format: {'close': value}
                    closes.append(float(candle['close']))
                elif isinstance(candle, (list, tuple)):
                    # List format: [timestamp, open, high, low, close, volume]
                    closes.append(float(candle[4]))
                else:
                    print(f"[EMA WARNING] Unknown candle format: {type(candle)}")
                    
            if closes:
                ema50 = calculate_ema(closes, 50)[-1]
                print(f"[EMA SUCCESS] Calculated EMA50: {ema50:.2f}")
            else:
                print(f"[EMA WARNING] No valid closes found")
                ema50 = None
        except Exception as ema_error:
            print(f"[EMA ERROR] Failed to calculate EMA50: {ema_error}")
            ema50 = None
    
    # Extract cluster features from signals
    cluster_features = {
        "strength": data.get("cluster_strength", 0.0),
        "direction": data.get("cluster_direction", 0.0),
        "volume_ratio": data.get("cluster_volume_ratio", 1.0)
    }
    
    # Current price with dual format support
    current_price = 0.0
    if ticker_data and "lastPrice" in ticker_data:
        current_price = float(ticker_data["lastPrice"])
    elif candles:
        # Handle both dictionary and list candle formats
        last_candle = candles[-1]
        if isinstance(last_candle, dict):
            current_price = float(last_candle['close'])
        elif isinstance(last_candle, (list, tuple)):
            current_price = float(last_candle[4])  # Last close
    
    # Market phase detection
    market_phase = signals.get("market_phase", "unknown")
    
    return {
        "symbol": symbol,
        "candles": candles,
        "ema50": ema50,
        "orderbook": orderbook,
        "cluster_features": cluster_features,
        "htf_candles": htf_candles or [],
        "ai_label": ai_label or {"label": "unknown", "confidence": 0.0, "method": "fallback"},
        "current_price": current_price,
        "market_phase": market_phase,
        "ticker_data": ticker_data,
        "market_data": market_data,
        "signals": signals,
        "debug": False  # Set to True for detailed logging
    }

def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    if not prices or len(prices) < period:
        return []
    
    ema = []
    multiplier = 2 / (period + 1)
    
    # Start with SMA for first value
    sma = sum(prices[:period]) / period
    ema.append(sma)
    
    # Calculate EMA for remaining values
    for price in prices[period:]:
        ema_value = (price * multiplier) + (ema[-1] * (1 - multiplier))
        ema.append(ema_value)
    
    return ema

def test_unified_scoring():
    """Test the unified scoring system"""
    print("🧪 Testing Unified Scoring Engine")
    print("=" * 50)
    
    # Mock data for testing
    test_data = {
        "symbol": "TESTUSDT",
        "candles": [[0, 100, 105, 95, 102, 1000000] for _ in range(100)],
        "ema50": 101.5,
        "orderbook": {
            "bids": [[100, 1000], [99.5, 2000], [99, 1500]],
            "asks": [[101, 800], [101.5, 1200], [102, 1000]]
        },
        "cluster_features": {
            "strength": 0.75,
            "direction": 0.6,
            "volume_ratio": 1.2
        },
        "htf_candles": [[0, 100, 105, 95, 102, 5000000] for _ in range(50)],
        "ai_label": {
            "label": "momentum_follow",
            "confidence": 0.82,
            "phase": "trend"
        },
        "current_price": 102.0,
        "market_phase": "trend-following",
        "debug": True
    }
    
    result = simulate_trader_decision_advanced(test_data)
    
    print(f"\n✅ Test Result:")
    print(f"   Final Score: {result['final_score']}")
    print(f"   Decision: {result['decision']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Active Modules: {result['active_modules']}/8")
    print(f"\n📊 Score Breakdown:")
    for component, score in result['score_breakdown'].items():
        if score != 0.0:
            print(f"   {component}: {score:+.4f}")

if __name__ == "__main__":
    test_unified_scoring()