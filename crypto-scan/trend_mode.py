#!/usr/bin/env python3
"""
Trend Mode TJDE - Advanced Trader Weighted Decision Engine

Kompletna implementacja TJDE (AdvancedTraderWeightedDecisionEngine)
zastępująca dotychczasowy system trend-mode.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import advanced analysis modules
from trader_ai_engine import (
    analyze_market_structure, 
    analyze_candle_behavior, 
    interpret_orderbook
)
from utils.market_phase import detect_market_phase
from utils.liquidity_analysis import analyze_liquidity_behavior
from utils.psychology import detect_psychological_traps
from utils.htf_overlay import get_htf_confirmation

# Import utilities
from utils.safe_candles import get_candles
# Import trend alert cache functions
try:
    from utils.trend_alert_cache import TrendAlertCache
    trend_cache = TrendAlertCache()
except ImportError:
    print("Warning: TrendAlertCache not available")
    trend_cache = None
from utils.telegram_bot import send_alert


def compute_trader_score(ctx: Dict, adaptive_engine=None, market_context: Dict = None) -> Dict:
    """
    🧠 TJDE Adaptive Scoring - Dynamiczny system z uczeniem się i modyfikatorami kontekstowymi
    
    Nowa wersja wykorzystuje AdaptiveWeightEngine + ContextualModifiers
    
    Args:
        ctx: Trend context z wszystkimi analizami
        adaptive_engine: AdaptiveWeightEngine instance (opcjonalne)
        market_context: Kontekst rynkowy dla modyfikacji (opcjonalne)
        
    Returns:
        dict: Zaktualizowany context z final_score i grade
    """
    try:
        # Import adaptive modules
        from utils.adaptive_weights import get_adaptive_engine
        from utils.context_modifiers import apply_contextual_modifiers, get_market_context
        
        # Get adaptive engine instance
        if adaptive_engine is None:
            adaptive_engine = get_adaptive_engine()
        
        # Get market context if not provided
        if market_context is None:
            market_context = get_market_context()
            # Add market phase from context
            market_context["market_phase"] = ctx.get("market_phase", "unknown")
        
        # Extract raw features
        features = {
            "trend_strength": ctx.get("trend_strength", 0.0),
            "pullback_quality": ctx.get("pullback_quality", 0.0),
            "support_reaction": ctx.get("support_reaction", 0.0),
            "liquidity_pattern_score": ctx.get("liquidity_pattern_score", 0.0),
            "psych_score": 1.0 - ctx.get("psych_score", 1.0),  # Invert psychology score
            "htf_supportive_score": ctx.get("htf_supportive_score", 0.0),
            "market_phase_modifier": ctx.get("market_phase_modifier", 0.0),
        }
        
        # Apply contextual modifiers
        modified_features = apply_contextual_modifiers(features, market_context)
        
        # Get adaptive weights
        weights = adaptive_engine.compute_weights()
        
        # Calculate weighted score
        score = 0.0
        score_breakdown = {}
        
        for key in modified_features:
            feature_value = modified_features[key]
            weight = weights.get(key, 0.0)
            component_score = feature_value * weight
            score += component_score
            score_breakdown[key] = round(component_score, 4)
        
        # Update context
        ctx.update({
            "final_score": round(score, 3),
            "grade": classify_grade(score),
            "score_breakdown": score_breakdown,
            "adaptive_weights": weights,
            "original_features": features,
            "modified_features": modified_features,
            "market_context": market_context,
            "adaptive_stats": adaptive_engine.get_performance_stats()
        })
        
        print(f"[TJDE ADAPTIVE] Score: {score:.3f}, Context: {market_context.get('session', 'unknown')}, Adaptations: {len(adaptive_engine.memory)}")
        
        return ctx
        
    except Exception as e:
        print(f"❌ [TJDE ADAPTIVE ERROR]: {e}")
        # Fallback to static scoring
        return _compute_static_score(ctx)


def _compute_static_score(ctx: Dict) -> Dict:
    """Fallback static scoring when adaptive system fails"""
    weights = {
        "trend_strength": 0.25,
        "pullback_quality": 0.20,
        "support_reaction": 0.15,
        "liquidity_pattern_score": 0.10,
        "psych_score": 0.10,
        "htf_supportive_score": 0.10,
        "market_phase_modifier": 0.10
    }
    
    score = 0.0
    score_breakdown = {}
    
    for key, weight in weights.items():
        val = ctx.get(key, 0.0)
        if key == "psych_score":
            val = 1.0 - val
        component_score = val * weight
        score += component_score
        score_breakdown[key] = round(component_score, 4)
    
    ctx.update({
        "final_score": round(score, 3),
        "grade": classify_grade(score),
        "score_breakdown": score_breakdown,
        "weights_used": weights
    })
    
    return ctx


def classify_grade(score: float) -> str:
    """Klasyfikuje grade na podstawie score"""
    if score >= 0.8:
        return "excellent"
    elif score >= 0.7:
        return "strong"
    elif score >= 0.6:
        return "good"
    elif score >= 0.45:
        return "neutral-watch"
    elif score >= 0.3:
        return "weak"
    else:
        return "very_poor"


def simulate_trader_decision_advanced(ctx: Dict) -> Dict:
    """
    🎯 TJDE Decision Engine - Zastępuje stary decision engine
    
    Inteligentny system decyzyjny oparty na weighted scoring
    
    Args:
        ctx: Trend context z compute_trader_score()
        
    Returns:
        dict: Context z decision i reasons
    """
    try:
        ctx = compute_trader_score(ctx)
        
        score = ctx["final_score"]
        grade = ctx["grade"]
        
        # Enhanced decision logic
        if score >= 0.7:
            ctx["decision"] = "join"
            ctx["confidence"] = min(0.95, score + 0.15)
        elif score >= 0.45:
            ctx["decision"] = "consider"
            ctx["confidence"] = score * 0.8
        else:
            ctx["decision"] = "avoid"
            ctx["confidence"] = max(0.1, 1.0 - score)
        
        # Build decision reasons
        reasons = []
        
        if score >= 0.7:
            reasons.append(f"Professional trader setup - TJDE score {score:.3f}")
        elif score >= 0.45:
            reasons.append(f"Moderate opportunity - TJDE score {score:.3f}")
        else:
            reasons.append(f"Weak setup - TJDE score {score:.3f}")
        
        # Feature-specific reasons from score breakdown
        score_breakdown = ctx.get("score_breakdown", {})
        top_features = sorted(score_breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for feature_name, contribution in top_features:
            if contribution > 0.08:  # Significant contribution
                clean_name = feature_name.replace('_', ' ').title()
                reasons.append(f"Strong {clean_name} ({contribution:.3f})")
        
        # Risk assessment
        psych_raw_score = ctx.get("psych_score", 0.7)
        if psych_raw_score < 0.5:  # High manipulation risk
            reasons.append("⚠️ Market manipulation detected")
        
        htf_score = score_breakdown.get("htf_supportive_score", 0.0)
        if htf_score < 0.03:
            reasons.append("⚠️ Higher timeframe not supportive")
        
        ctx["decision_reasons"] = reasons
        
        return ctx
        
    except Exception as e:
        print(f"❌ [TJDE DECISION ERROR]: {e}")
        ctx["decision"] = "avoid"
        ctx["confidence"] = 0.0
        ctx["decision_reasons"] = ["Decision engine error"]
        return ctx


def analyze_trend_opportunity(symbol: str, candles: List[List] = None, enable_vision: bool = True) -> Dict:
    """
    🧠 TJDE Main Analysis - Pełna integracja AdvancedTraderWeightedDecisionEngine
    
    Zastępuje dotychczasowy 9-etapowy pipeline nowym systemem TJDE:
    1. Zbieranie danych z modułów analitycznych
    2. Compute trader score z dynamicznymi wagami
    3. Advanced decision engine
    4. Alert i logging integration
    
    Args:
        symbol: Trading symbol to analyze
        candles: Optional pre-fetched candles
        
    Returns:
        dict: TJDE analysis result
    """
    try:
        print(f"[TJDE] {symbol}: Starting AdvancedTraderWeightedDecisionEngine analysis...")
        
        # Get candles if not provided
        if candles is None:
            candles = get_candles(symbol, interval="15", limit=96)
        
        if not candles or len(candles) < 20:
            return {
                "symbol": symbol,
                "decision": "avoid",
                "final_score": 0.0,
                "confidence": 0.0,
                "decision_reasons": ["Insufficient candle data"],
                "grade": "data_error"
            }
        
        print(f"[TJDE] {symbol}: Retrieved {len(candles)} candles")
        
        # === KROK 1: ZBIERANIE DANYCH Z MODUŁÓW ANALITYCZNYCH ===
        
        trend_context = {}
        
        # Basic market structure analysis
        print(f"[TJDE] {symbol}: Analyzing market structure...")
        market_structure = analyze_market_structure(candles, symbol)
        trend_context["trend_strength"] = _extract_trend_strength(market_structure)
        
        # Candle behavior analysis
        print(f"[TJDE] {symbol}: Analyzing candle behavior...")
        candle_behavior = analyze_candle_behavior(candles, symbol)
        trend_context["pullback_quality"] = _extract_pullback_quality(candle_behavior)
        
        # Orderbook interpretation
        print(f"[TJDE] {symbol}: Interpreting orderbook...")
        orderbook_info = interpret_orderbook(symbol)
        trend_context["support_reaction"] = _extract_support_reaction(orderbook_info)
        
        # Market phase detection
        print(f"[TJDE] {symbol}: Detecting market phase...")
        phase_analysis = detect_market_phase(candles, symbol)
        trend_context.update(_extract_phase_data(phase_analysis))
        
        # Liquidity behavior analysis
        print(f"[TJDE] {symbol}: Analyzing liquidity behavior...")
        liquidity_analysis = analyze_liquidity_behavior(symbol, None, candles)
        trend_context["liquidity_pattern_score"] = liquidity_analysis.get("liquidity_pattern_score", 0.5)
        
        # Psychological traps detection
        print(f"[TJDE] {symbol}: Detecting psychological traps...")
        psychology_analysis = detect_psychological_traps(candles, symbol)
        trend_context["psych_score"] = psychology_analysis.get("psych_score", 0.7)
        
        # HTF confirmation
        print(f"[TJDE] {symbol}: Getting HTF confirmation...")
        htf_analysis = get_htf_confirmation(symbol, "15")
        trend_context["htf_supportive_score"] = htf_analysis.get("htf_supportive_score", 0.5)
        
        # Add metadata
        trend_context["symbol"] = symbol
        trend_context["candle_count"] = len(candles)
        trend_context["analysis_timestamp"] = datetime.now().isoformat()
        
        # === KROK 2: NEW ADAPTIVE DECISION ENGINE ===
        
        print(f"[TJDE] {symbol}: Extracting features for adaptive decision engine...")
        
        # Extract all features for new decision system
        from utils.feature_extractor import extract_all_features_for_token
        features = extract_all_features_for_token(symbol, candles)
        
        # Phase 1: Perception Synchronization - CLIP + TJDE + GPT Integration
        from trader_ai_engine import simulate_trader_decision_advanced
        from perception_sync import simulate_trader_decision_perception_sync
        
        try:
            adaptive_result = simulate_trader_decision_perception_sync(symbol, market_data, features)
            print(f"[TREND MODE] {symbol}: Using Phase 1 Perception Synchronization")
        except Exception as e:
            print(f"[TREND MODE] {symbol}: Fallback to standard TJDE due to: {e}")
            adaptive_result = simulate_trader_decision_advanced(symbol, market_data, features)
        
        # Map result to trend_context format
        trend_context.update({
            "final_score": adaptive_result["final_score"],
            "grade": adaptive_result["quality_grade"],
            "decision": adaptive_result["decision"],
            "confidence": adaptive_result.get("confidence", 0.0),
            "context_modifiers": adaptive_result.get("context_modifiers", []),
            "adaptive_weights": adaptive_result.get("weights", {}),
            "market_phase": adaptive_result.get("market_phase", "unknown"),
            "used_features": adaptive_result.get("used_features", {})
        })
        
        print(f"[TJDE] {symbol}: Adaptive decision: {adaptive_result['decision'].upper()}, Score: {adaptive_result['final_score']:.3f}")
        
        # === KROK 3: FINAL DECISION VALIDATION ===
        
        print(f"[TJDE] {symbol}: Validating final decision...")
        
        # Decision already made by adaptive engine, just validate
        decision = trend_context.get("decision", "avoid")
        final_score = trend_context.get("final_score", 0.0)
        grade = trend_context.get("grade", "weak")
        
        # === KROK 4: ALERT AND LOGGING ===
        
        print(f"[TJDE] {symbol}: {decision.upper()} | Score: {final_score:.3f} | Grade: {grade}")
        
        # Send alert ONLY for join_trend decisions with high score
        if decision == "join_trend" and final_score >= 0.7:
            print(f"[TJDE ALERT] {symbol}: Triggering trend alert...")
            send_tjde_alert(symbol, trend_context)
        elif decision == "join_trend":
            print(f"[TJDE] {symbol}: JOIN decision but score too low ({final_score:.3f} < 0.7)")
        else:
            print(f"[TJDE] {symbol}: No alert - decision: {decision}, score: {final_score:.3f}")
        
        decision = trend_context.get("decision", "avoid")
        final_score = trend_context.get("final_score", 0.0)
        grade = trend_context.get("grade", "unknown")
        
        print(f"[TJDE] {symbol}: {decision.upper()} | Score: {final_score:.3f} | Grade: {grade}")
        
        # Alert tylko przy "join"
        if decision == "join":
            print(f"[TJDE ALERT] {symbol}: Triggering trend alert...")
            send_tjde_alert(symbol, trend_context)
        
        # Zapis do loga
        log_trend_decision(symbol, trend_context)
        
        return trend_context
        
    except Exception as e:
        print(f"❌ [TJDE ERROR] {symbol}: {e}")
        return {
            "symbol": symbol,
            "decision": "avoid",
            "final_score": 0.0,
            "confidence": 0.0,
            "decision_reasons": [f"TJDE analysis error: {e}"],
            "grade": "error"
        }


def _extract_trend_strength(market_structure: str) -> float:
    """Extract trend strength from market structure analysis"""
    if market_structure == "impulse":
        return 0.85
    elif market_structure == "breakout":
        return 0.8
    elif market_structure == "pullback":
        return 0.7
    elif market_structure == "range":
        return 0.3
    else:
        return 0.4


def _extract_pullback_quality(candle_behavior: Dict) -> float:
    """Extract pullback quality from candle behavior"""
    base_quality = 0.5
    
    if candle_behavior.get("shows_buy_pressure", False):
        base_quality += 0.2
    
    momentum = candle_behavior.get("momentum", "neutral")
    if momentum == "building":
        base_quality += 0.15
    elif momentum == "negative":
        base_quality -= 0.15
    
    pattern = candle_behavior.get("pattern", "neutral")
    if pattern in ["momentum_building", "absorption_bounce"]:
        base_quality += 0.15
    
    return min(1.0, max(0.0, base_quality))


def _extract_support_reaction(orderbook_info: Dict) -> float:
    """Extract support reaction from orderbook analysis"""
    base_support = 0.5
    
    if orderbook_info.get("bids_layered", False):
        base_support += 0.2
    
    bid_strength = orderbook_info.get("bid_strength", "neutral")
    if bid_strength == "strong_support":
        base_support += 0.25
    elif bid_strength == "decent_support":
        base_support += 0.1
    elif bid_strength == "weak":
        base_support -= 0.15
    
    if orderbook_info.get("spoofing_suspected", False):
        base_support -= 0.2
    
    return min(1.0, max(0.0, base_support))


def _extract_phase_data(phase_analysis: Dict) -> Dict:
    """Extract relevant data from market phase analysis"""
    phase_score = phase_analysis.get("phase_score", 0.5)
    phase_name = phase_analysis.get("market_phase", "undefined")
    
    # Market phase modifier based on phase type
    modifier = 1.0
    if phase_name == "breakout-continuation":
        modifier = 1.2  # 20% boost
    elif phase_name == "pre-breakout":
        modifier = 1.15  # 15% boost
    elif phase_name == "retest-confirmation":
        modifier = 1.1   # 10% boost
    elif phase_name == "exhaustion-pullback":
        modifier = 0.8   # 20% penalty
    
    market_phase_modifier = (phase_score * modifier - phase_score)
    
    return {
        "market_phase": phase_name,
        "market_phase_modifier": market_phase_modifier,
        "phase_confidence": phase_analysis.get("confidence", 0.0)
    }


def send_tjde_alert(symbol: str, ctx: Dict):
    """
    Wysyła alert Telegram dla TJDE decision = "join_trend"
    Nowa wersja z context_modifiers i .jsonl logging
    """
    try:
        from utils.trend_alert_cache import is_trend_cooldown_active, set_trend_cooldown
        
        # Check cooldown (60 minutes)
        if is_trend_cooldown_active(symbol):
            print(f"[TJDE ALERT] {symbol}: Cooldown active, skipping alert")
            return
        
        # Prepare enhanced alert data with context_modifiers
        alert_data = {
            "symbol": symbol,
            "decision": ctx.get("decision", "unknown"),
            "final_score": ctx.get("final_score", 0.0),
            "grade": ctx.get("grade", "unknown"),
            "confidence": ctx.get("confidence", 0.0),
            "context_modifiers": ctx.get("context_modifiers", []),
            "market_phase": ctx.get("market_phase", "unknown"),
            "adaptive_weights": ctx.get("adaptive_weights", {}),
            "timestamp": datetime.now().isoformat(),
            "alert_type": "TJDE_adaptive"
        }
        
        # Log to .jsonl file for analysis
        _log_tjde_alert_to_jsonl(symbol, alert_data)
        
        # Also save to alerts history for feedback loop
        _save_alert_for_feedback_analysis(symbol, alert_data)
        
        # Log to feedback_loop_v2 system
        from utils.feedback_integration import log_tjde_alert_for_feedback
        log_tjde_alert_for_feedback(symbol, ctx, ctx.get("used_features", {}))
        
        # Enhanced Telegram message with context_modifiers
        enhanced_message = f"""🚨 TJDE ADAPTIVE ALERT: {symbol}
📊 Score: {alert_data['final_score']:.3f} | Grade: {alert_data['grade']}
🎯 Decision: {alert_data['decision'].upper()}
📈 Phase: {alert_data['market_phase']}
🧠 Context: {', '.join(alert_data['context_modifiers']) if alert_data['context_modifiers'] else 'None'}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        
        # Send enhanced alert
        from utils.telegram_bot import send_alert
        success = send_alert(
            symbol=symbol,
            signals=alert_data,
            score=ctx.get("final_score", 0.0),
            alert_level=3  # High priority for TJDE
        )
        
        if success:
            set_trend_cooldown(symbol)
            print(f"[TJDE ALERT] {symbol}: Enhanced alert sent with context modifiers")
        else:
            print(f"[TJDE ALERT] {symbol}: Alert failed to send")
            
    except Exception as e:
        print(f"❌ [TJDE ALERT ERROR] {symbol}: {e}")


def _log_tjde_alert_to_jsonl(symbol: str, alert_data: Dict):
    """Log TJDE alert to .jsonl file for analysis"""
    try:
        import os
        import json
        
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/tjde_alerts.jsonl", "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(alert_data, ensure_ascii=False)}\n")
            
    except Exception as e:
        print(f"❌ [TJDE LOG ERROR] {symbol}: {e}")


def _save_alert_for_feedback_analysis(symbol: str, alert_data: Dict):
    """Save alert to history file for feedback loop analysis"""
    try:
        import os
        import json
        
        os.makedirs("data/alerts", exist_ok=True)
        alerts_file = "data/alerts/alerts_history.json"
        
        # Load existing alerts
        if os.path.exists(alerts_file):
            with open(alerts_file, "r", encoding="utf-8") as f:
                alerts_history = json.load(f)
        else:
            alerts_history = {"alerts": []}
        
        # Add new alert
        alerts_history["alerts"].append(alert_data)
        
        # Keep only last 1000 alerts to manage file size
        if len(alerts_history["alerts"]) > 1000:
            alerts_history["alerts"] = alerts_history["alerts"][-1000:]
        
        alerts_history["last_updated"] = datetime.now().isoformat()
        
        # Save updated history
        with open(alerts_file, "w", encoding="utf-8") as f:
            json.dump(alerts_history, f, indent=2, ensure_ascii=False)
        
        print(f"[FEEDBACK] {symbol}: Alert saved for feedback analysis")
        
    except Exception as e:
        print(f"❌ [FEEDBACK ERROR] {symbol}: {e}")


def log_trend_decision(symbol: str, ctx: Dict, output_path: str = "logs/advanced_trader_log.txt"):
    """
    Zapis decyzji TJDE do loga
    Zastępuje poprzednie systemy logowania
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "engine": "TJDE_AdvancedTraderWeightedDecisionEngine",
            "decision": ctx.get("decision", "unknown"),
            "final_score": ctx.get("final_score", 0.0),
            "confidence": ctx.get("confidence", 0.0),
            "grade": ctx.get("grade", "unknown"),
            "score_breakdown": ctx.get("score_breakdown", {}),
            "weights_used": ctx.get("weights_used", {}),
            "decision_reasons": ctx.get("decision_reasons", []),
            "market_phase": ctx.get("market_phase", "unknown"),
            "liquidity_pattern_score": ctx.get("liquidity_pattern_score", 0.0),
            "psych_score": ctx.get("psych_score", 0.0),
            "htf_supportive_score": ctx.get("htf_supportive_score", 0.0),
            "phase_confidence": ctx.get("phase_confidence", 0.0)
        }
        
        os.makedirs("logs", exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(log_entry, ensure_ascii=False)}\n")
        
    except Exception as e:
        print(f"❌ [TJDE LOG ERROR] {symbol}: {e}")


if __name__ == "__main__":
    # Test TJDE system
    symbol = "ZEREBROUSDT"
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    
    print(f"🧠 Testing TJDE AdvancedTraderWeightedDecisionEngine for {symbol}")
    result = analyze_trend_opportunity(symbol)
    
    decision = result.get("decision", "unknown")
    final_score = result.get("final_score", 0.0)
    grade = result.get("grade", "unknown")
    confidence = result.get("confidence", 0.0)
    
    print(f"\n📊 TJDE RESULT: {decision.upper()} | Score: {final_score:.3f} | Grade: {grade} | Confidence: {confidence:.3f}")
    
    # Score breakdown
    score_breakdown = result.get("score_breakdown", {})
    if score_breakdown:
        print(f"\n📊 SCORE BREAKDOWN:")
        for component, value in score_breakdown.items():
            print(f"  {component:25} = {value:+.4f}")
    
    # Decision reasons
    reasons = result.get("decision_reasons", [])
    if reasons:
        print(f"\n🔍 DECISION REASONS:")
        for i, reason in enumerate(reasons[:5], 1):
            print(f"  {i}. {reason}")
    
    # Market phase info
    market_phase = result.get("market_phase", "unknown")
    liquidity_score = result.get("liquidity_pattern_score", 0.0)
    psych_score = result.get("psych_score", 0.0)
    
    print(f"\n🏗️  MARKET ANALYSIS:")
    print(f"  Phase: {market_phase}")
    print(f"  Liquidity Score: {liquidity_score:.3f}")
    print(f"  Psychology Score: {psych_score:.3f}")
    
    print(f"\n✅ TJDE Analysis complete for {symbol}")
    print(f"📄 Check logs/advanced_trader_log.txt for detailed logs")