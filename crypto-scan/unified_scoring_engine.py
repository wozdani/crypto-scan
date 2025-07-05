"""
Unified Scoring Engine - Complete Integration of All Modules
Combines legacy scoring with Modules 1-5 in one modular, scalable system
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

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
    ai_label = data.get("ai_label", {})
    market_phase = data.get("market_phase", "unknown")
    
    if debug:
        logger.info(f"[UNIFIED SCORING] Starting comprehensive analysis for {symbol}")
    
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
    
    total_score = 0.0
    
    # === MODULE 1: AI-EYE VISION (CLIP + GPT with Dynamic Weights) ===
    try:
        from vision.vision_scoring import score_from_ai_label
        ai_label = data.get("ai_label", {})
        market_phase = data.get("market_phase", None)
        
        if ai_label:
            ai_eye_score = score_from_ai_label(ai_label, market_phase)
            score_breakdown["ai_eye_score"] = ai_eye_score
            total_score += ai_eye_score
            
            if debug:
                logger.info(f"[MODULE 1] AI-EYE: {ai_eye_score:+.4f} "
                           f"(label: {ai_label.get('label', 'unknown')}, "
                           f"conf: {ai_label.get('confidence', 0.0):.2f})")
        else:
            if debug:
                logger.info(f"[MODULE 1] AI-EYE: No AI label data available")
                
    except Exception as e:
        logger.error(f"[MODULE 1 ERROR] AI-EYE scoring failed: {e}")
    
    # === MODULE 2: HTF OVERLAY (Macrostructure Awareness) ===
    try:
        htf_candles = data.get("htf_candles", [])
        
        if htf_candles and len(htf_candles) >= 20:
            # Simplified HTF scoring - calculate basic trend
            closes = [float(candle[4]) for candle in htf_candles[-20:] if len(candle) > 4]
            if len(closes) >= 10:
                trend_strength = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
                
                # Simple HTF trend scoring (-0.05 to +0.05)
                htf_score = np.clip(trend_strength * 2, -0.05, 0.05)
                
                # AI pattern alignment bonus
                if ai_label and ai_label.get('label') in ['momentum_follow', 'breakout_pattern']:
                    if htf_score > 0:  # Bullish HTF + bullish AI
                        htf_score *= 1.2
                
                score_breakdown["htf_overlay_score"] = htf_score
                total_score += htf_score
                
                if debug:
                    logger.info(f"[MODULE 2] HTF Overlay: {htf_score:+.4f} "
                               f"(trend: {trend_strength:+.3f})")
            else:
                if debug:
                    logger.info(f"[MODULE 2] HTF Overlay: Invalid candle data")
        else:
            if debug:
                logger.info(f"[MODULE 2] HTF Overlay: Insufficient HTF data "
                           f"({len(htf_candles) if htf_candles else 0} candles)")
                
    except Exception as e:
        logger.error(f"[MODULE 2 ERROR] HTF Overlay scoring failed: {e}")
    
    # === MODULE 3: TRAP DETECTOR (Risk Pattern Detection) ===
    try:
        candles = data.get("candles", [])
        
        if candles and len(candles) >= 20:
            # Simplified trap detection - look for fake breakouts
            recent_candles = candles[-10:]
            trap_score = 0.0
            
            for candle in recent_candles:
                if len(candle) >= 5:
                    open_price = float(candle[1])
                    high_price = float(candle[2])
                    low_price = float(candle[3])
                    close_price = float(candle[4])
                    
                    # Check for long upper wicks (potential trap)
                    body_size = abs(close_price - open_price)
                    upper_wick = high_price - max(open_price, close_price)
                    
                    if body_size > 0 and upper_wick > body_size * 2:
                        trap_score -= 0.002  # Penalty for fake breakout pattern
            
            # Cap the score
            trap_score = max(trap_score, -0.05)
            
            score_breakdown["trap_detector_score"] = trap_score
            total_score += trap_score
            
            if debug:
                logger.info(f"[MODULE 3] Trap Detector: {trap_score:+.4f} "
                           f"(analyzed {len(recent_candles)} candles)")
        else:
            if debug:
                logger.info(f"[MODULE 3] Trap Detector: Insufficient candle data "
                           f"({len(candles) if candles else 0} candles)")
                
    except Exception as e:
        logger.error(f"[MODULE 3 ERROR] Trap Detector scoring failed: {e}")
    
    # === MODULE 4: FUTURE SCENARIO MAPPING (Predictive Analysis) ===
    try:
        candles = data.get("candles", [])
        ema50 = data.get("ema50", None)
        
        if candles and len(candles) >= 50 and ema50 is not None:
            # Simplified future scenario analysis
            current_price = float(candles[-1][4])  # Last close
            price_vs_ema = (current_price - ema50) / ema50 if ema50 != 0 else 0
            
            # Recent volatility assessment
            recent_highs = [float(candle[2]) for candle in candles[-5:]]
            recent_lows = [float(candle[3]) for candle in candles[-5:]]
            volatility = (max(recent_highs) - min(recent_lows)) / current_price if current_price != 0 else 0
            
            # Future scenario scoring
            future_score = 0.0
            
            # Bullish scenario: above EMA with controlled volatility
            if price_vs_ema > 0.02 and volatility < 0.05:
                future_score += 0.03
            
            # Bearish scenario: far below EMA
            if price_vs_ema < -0.05:
                future_score -= 0.02
            
            # AI pattern alignment
            if ai_label and ai_label.get('label') == 'momentum_follow':
                if price_vs_ema > 0:
                    future_score += 0.01  # Momentum continuation bonus
            
            score_breakdown["future_mapping_score"] = future_score
            total_score += future_score
            
            if debug:
                logger.info(f"[MODULE 4] Future Mapping: {future_score:+.4f} "
                           f"(price/EMA: {price_vs_ema:+.3f}, vol: {volatility:.3f})")
        else:
            if debug:
                logger.info(f"[MODULE 4] Future Mapping: Missing data "
                           f"(candles: {len(candles) if candles else 0}, EMA50: {ema50})")
                
    except Exception as e:
        logger.error(f"[MODULE 4 ERROR] Future Mapping scoring failed: {e}")
    
    # === MODULE 5: FEEDBACK LOOP ===
    # Note: Module 5 is integrated into Module 1 (AI-EYE) via dynamic weights
    # and also logs predictions for future learning
    try:
        # Log prediction for feedback if significant signal
        if total_score != 0.0 and ai_label:
            from feedback_loop.feedback_integration import log_prediction_for_feedback
            
            current_price = data.get("current_price", 0.0)
            if current_price > 0:
                # Determine preliminary decision for logging
                preliminary_decision = "wait"
                if total_score >= 0.15:
                    preliminary_decision = "enter"
                elif total_score <= -0.10:
                    preliminary_decision = "avoid"
                
                feedback_logged = log_prediction_for_feedback(
                    symbol=symbol,
                    ai_label=ai_label,
                    current_price=current_price,
                    tjde_score=total_score,
                    decision=preliminary_decision,
                    market_phase=market_phase
                )
                
                if debug and feedback_logged:
                    logger.info(f"[MODULE 5] Feedback: Prediction logged for learning")
                    
    except Exception as e:
        if debug:
            logger.error(f"[MODULE 5 ERROR] Feedback logging failed: {e}")
    
    # === LEGACY SCORING COMPONENTS ===
    
    # Legacy Volume Slope Analysis
    try:
        candles = data.get("candles", [])
        if candles and len(candles) >= 10:
            volume_score = score_from_volume_slope(candles)
            score_breakdown["legacy_volume_score"] = volume_score
            total_score += volume_score
            
            if debug:
                logger.info(f"[LEGACY] Volume Slope: {volume_score:+.4f}")
                
    except Exception as e:
        logger.error(f"[LEGACY ERROR] Volume scoring failed: {e}")
    
    # Legacy Orderbook Pressure Analysis
    try:
        orderbook = data.get("orderbook", {})
        if orderbook:
            orderbook_score = score_from_orderbook_pressure(orderbook)
            score_breakdown["legacy_orderbook_score"] = orderbook_score
            total_score += orderbook_score
            
            if debug:
                logger.info(f"[LEGACY] Orderbook: {orderbook_score:+.4f}")
                
    except Exception as e:
        logger.error(f"[LEGACY ERROR] Orderbook scoring failed: {e}")
    
    # Legacy Cluster Analysis
    try:
        cluster_features = data.get("cluster_features", {})
        if cluster_features:
            cluster_score = score_from_cluster(cluster_features)
            score_breakdown["legacy_cluster_score"] = cluster_score
            total_score += cluster_score
            
            if debug:
                logger.info(f"[LEGACY] Cluster: {cluster_score:+.4f}")
                
    except Exception as e:
        logger.error(f"[LEGACY ERROR] Cluster scoring failed: {e}")
    
    # Legacy Psychology Analysis
    try:
        candles = data.get("candles", [])
        if candles and len(candles) >= 20:
            psychology_score = score_from_psychology(candles)
            score_breakdown["legacy_psychology_score"] = psychology_score
            total_score += psychology_score
            
            if debug:
                logger.info(f"[LEGACY] Psychology: {psychology_score:+.4f}")
                
    except Exception as e:
        logger.error(f"[LEGACY ERROR] Psychology scoring failed: {e}")
    
    # === FINAL DECISION LOGIC ===
    final_score = round(total_score, 4)
    
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
    
    if debug:
        logger.info(f"[UNIFIED RESULT] {symbol}: {final_score:.4f} → {decision} "
                   f"(confidence: {confidence:.3f}, modules: {active_modules}/8)")
        logger.info(f"[BREAKDOWN] {score_breakdown}")
    
    return result

# === LEGACY SCORING FUNCTIONS ===

def score_from_volume_slope(candles: List) -> float:
    """Legacy volume slope analysis"""
    try:
        if not candles or len(candles) < 10:
            return 0.0
        
        # Extract volume data from last 10 candles
        volumes = [float(candle[5]) for candle in candles[-10:] if len(candle) > 5]
        
        if len(volumes) < 5:
            return 0.0
        
        # Calculate volume trend
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes, 1)[0]
        
        # Normalize slope to score range
        volume_avg = np.mean(volumes)
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
        if not orderbook:
            return 0.0
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return 0.0
        
        # Calculate top 5 levels pressure
        bid_volume = sum(float(bid[1]) for bid in bids[:5])
        ask_volume = sum(float(ask[1]) for ask in asks[:5])
        
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
                
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            
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
    
    # Calculate EMA50 if needed
    ema50 = None
    if candles and len(candles) >= 50:
        closes = [float(candle[4]) for candle in candles[-50:]]
        ema50 = calculate_ema(closes, 50)[-1] if closes else None
    
    # Extract cluster features from signals
    cluster_features = {
        "strength": signals.get("cluster_strength", 0.0),
        "direction": signals.get("cluster_direction", 0.0),
        "volume_ratio": signals.get("cluster_volume_ratio", 1.0)
    }
    
    # Current price
    current_price = 0.0
    if ticker_data and "lastPrice" in ticker_data:
        current_price = float(ticker_data["lastPrice"])
    elif candles:
        current_price = float(candles[-1][4])  # Last close
    
    # Market phase detection
    market_phase = signals.get("market_phase", "unknown")
    
    return {
        "symbol": symbol,
        "candles": candles,
        "ema50": ema50,
        "orderbook": orderbook,
        "cluster_features": cluster_features,
        "htf_candles": htf_candles or [],
        "ai_label": ai_label or {},
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