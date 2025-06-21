"""
Perception Evaluator - Behavioral Market Analysis Module

Zastępuje klasyczny system scoringowy analizą behawioralną, która ocenia setup
jako całość z perspektywy doświadczonego tradera. Brak punktów, brak progów -
tylko logiczna ocena warunków rynkowych.
"""

from typing import Dict, List, Tuple, Optional
import time


def is_setup_convincing(context: Dict) -> bool:
    """
    Elastyczna funkcja oceny setupu - zmniejszone wymagania dla więcej alertów
    
    Args:
        context: Market context z danymi o trendzie, flow, orderbook, etc.
        
    Returns:
        bool: True jeśli setup jest przekonujący dla wejścia
    """
    try:
        # === PODSTAWOWY WARUNEK ===
        # Tylko trend musi być potwierdzony - reszta jest elastyczna
        if not context.get("trend_confirmed", False):
            return False
        
        # === ELASTYCZNE WARUNKI WEJŚCIA ===
        
        # Warunki pullback - akceptujemy weak i medium
        pullback_ok = False
        if context.get("pullback_detected", False):
            pullback_strength = context.get("pullback_strength", "strong")
            pullback_ok = pullback_strength in ["weak", "medium"]
        
        # Flow - zmniejszone wymaganie lub aktywne detektory
        flow_ok = (context.get("flow_consistent", False) or 
                   context.get("heatmap_vacuum", False) or
                   context.get("orderbook_freeze", False) or
                   context.get("vwap_pinning", False))
        
        # Orderbook - akceptujemy więcej stanów
        orderbook_behavior = context.get("orderbook_behavior", "neutral")
        orderbook_ok = orderbook_behavior in ["bullish_control", "slightly_bullish", "accumulation", "squeeze"]
        
        # Support - zwiększona tolerancja do 2.5%
        support_ok = context.get("near_support", False)
        
        # === SCENARIUSZE WEJŚCIA (elastyczne kombinacje) ===
        
        # Scenariusz 1: Pullback Recovery (zrelaksowany)
        if pullback_ok and flow_ok and orderbook_ok:
            return True
        
        # Scenariusz 2: Strong Flow + Support
        if flow_ok and support_ok and orderbook_ok:
            return True
        
        # Scenariusz 3: Multiple Detectors Active (bez pullback requirement)
        active_detectors = sum([
            context.get("heatmap_vacuum", False),
            context.get("orderbook_freeze", False),
            context.get("vwap_pinning", False),
            context.get("human_flow", False),
            context.get("micro_echo", False)
        ])
        
        if active_detectors >= 2 and orderbook_ok:
            return True
        
        # Scenariusz 4: Support Bounce (zrelaksowany)
        if support_ok and (flow_ok or orderbook_behavior == "bullish_control"):
            return True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Error in perception evaluator: {e}")
        return False


def _evaluate_pullback_recovery_scenario(context: Dict) -> bool:
    """
    Scenariusz: Powrót z korekty na poziomie wsparcia
    
    Trader logic: "Trend jest zdrowy, była mała korekta, teraz odbijamy od wsparcia
    z malejącym volume sprzedaży i rosnącą presją kupujących"
    """
    pullback_detected = context.get("pullback_detected", False)
    pullback_strength = context.get("pullback_strength", "unknown")
    price_near_support = context.get("price_near_support", False)
    buyer_pressure = context.get("buyer_pressure", "neutral")
    heatmap_vacuum = context.get("heatmap_vacuum", False)
    
    # Warunki dla pullback recovery:
    # - Słaba/umiarkowana korekta (nie silna)
    # - Cena blisko wsparcia
    # - Rosnąca presja kupujących
    # - Opcjonalnie: vacuum powyżej (mało resistance)
    
    if (pullback_detected and 
        pullback_strength in ["weak", "moderate"] and
        price_near_support and
        buyer_pressure in ["growing", "strong"]):
        
        # Bonus points jeśli jest vacuum
        if heatmap_vacuum:
            return True
        
        # Lub jeśli orderbook pokazuje kontrolę
        if context.get("orderbook_behavior") == "bullish_control":
            return True
    
    return False


def _evaluate_momentum_continuation_scenario(context: Dict) -> bool:
    """
    Scenariusz: Kontynuacja momentum bez większych korekt
    
    Trader logic: "Trend jest silny, flow jest spójny, kupujący kontrolują sytuację,
    nie ma korekt - to jest zdrowa kontynuacja trendu"
    """
    pullback_detected = context.get("pullback_detected", False)
    momentum_strong = context.get("momentum_strong", False)
    buyer_pressure = context.get("buyer_pressure", "neutral")
    volume_increasing = context.get("volume_increasing", False)
    
    # Warunki dla momentum continuation:
    # - Brak znaczących pullbacków
    # - Silne momentum
    # - Silna presja kupujących
    # - Rosnący volume
    
    if (not pullback_detected and
        momentum_strong and
        buyer_pressure == "strong" and
        volume_increasing):
        return True
    
    return False


def _evaluate_breakout_preparation_scenario(context: Dict) -> bool:
    """
    Scenariusz: Przygotowanie do breakoutu
    
    Trader logic: "Cena konsoliduje blisko resistance, volume spada (cisza przed burzą),
    orderbook pokazuje accumulation, heatmap vacuum - przygotowuje się breakout"
    """
    consolidation_detected = context.get("consolidation_detected", False)
    near_resistance = context.get("near_resistance", False)
    volume_declining = context.get("volume_declining", False)
    heatmap_vacuum = context.get("heatmap_vacuum", False)
    orderbook_behavior = context.get("orderbook_behavior", "neutral")
    
    # Warunki dla breakout preparation:
    # - Konsolidacja blisko resistance
    # - Malejący volume (accumulation phase)
    # - Vacuum powyżej
    # - Accumulation w orderbook
    
    if (consolidation_detected and
        near_resistance and
        volume_declining and
        heatmap_vacuum and
        orderbook_behavior == "accumulation"):
        return True
    
    return False


def _evaluate_support_bounce_scenario(context: Dict) -> bool:
    """
    Scenariusz: Odbicie od kluczowego wsparcia
    
    Trader logic: "Cena testuje ważne wsparcie, pojawia się obrona poziomu,
    volume rośnie na odbicia, orderbook pokazuje absorbcję sprzedaży"
    """
    price_near_support = context.get("price_near_support", False)
    support_strength = context.get("support_strength", "weak")
    volume_on_bounce = context.get("volume_on_bounce", False)
    seller_exhaustion = context.get("seller_exhaustion", False)
    orderbook_behavior = context.get("orderbook_behavior", "neutral")
    
    # Warunki dla support bounce:
    # - Cena przy silnym wsparciu
    # - Volume rośnie na odbicia
    # - Wyczerpanie sprzedających
    # - Orderbook absorbs selling pressure
    
    if (price_near_support and
        support_strength in ["strong", "major"] and
        volume_on_bounce and
        seller_exhaustion):
        return True
    
    # Alternatywnie: silne wsparcie + bullish control
    if (price_near_support and
        support_strength == "strong" and
        orderbook_behavior == "bullish_control"):
        return True
    
    return False


def build_market_context_from_trend_mode(trend_mode_data: Dict) -> Dict:
    """
    Konwertuje dane z trend_mode na market_context dla perception evaluator
    
    Args:
        trend_mode_data: Dane z trend_mode_pipeline
        
    Returns:
        dict: Market context dla is_setup_convincing()
    """
    context = {}
    
    try:
        # Mapowanie danych z trend_mode na context
        
        # Trend confirmation
        context["trend_confirmed"] = trend_mode_data.get("uptrend_active", False)
        
        # Flow analysis - zmniejszone wymaganie z 70% na 50%
        flow_score = trend_mode_data.get("flow_consistency_score", 0)
        context["flow_consistent"] = flow_score >= 50
        
        # Orderbook behavior
        bid_pressure = trend_mode_data.get("bid_pressure", 0)
        ask_pressure = trend_mode_data.get("ask_pressure", 0)
        
        if bid_pressure > ask_pressure * 1.5:
            context["orderbook_behavior"] = "bullish_control"
        elif bid_pressure > ask_pressure * 1.2:
            context["orderbook_behavior"] = "slightly_bullish"  # Nowa kategoria
        elif bid_pressure > ask_pressure * 1.0:
            context["orderbook_behavior"] = "accumulation"
        elif abs(bid_pressure - ask_pressure) < ask_pressure * 0.1:
            context["orderbook_behavior"] = "squeeze"
        else:
            context["orderbook_behavior"] = "neutral"
        
        # Buyer pressure
        volume_trend = trend_mode_data.get("volume_trend", "neutral")
        price_momentum = trend_mode_data.get("price_momentum", "neutral")
        
        if volume_trend == "increasing" and price_momentum == "strong":
            context["buyer_pressure"] = "strong"
        elif volume_trend == "increasing" or price_momentum == "moderate":
            context["buyer_pressure"] = "growing"
        else:
            context["buyer_pressure"] = "neutral"
        
        # Pullback analysis
        context["pullback_detected"] = trend_mode_data.get("pullback_detected", False)
        pullback_magnitude = trend_mode_data.get("pullback_magnitude", 0)
        
        if pullback_magnitude < 2:
            context["pullback_strength"] = "weak"
        elif pullback_magnitude < 5:
            context["pullback_strength"] = "moderate"
        else:
            context["pullback_strength"] = "strong"
        
        # Support/Resistance analysis - zwiększona tolerancja do 2.5%
        near_support_raw = trend_mode_data.get("near_support", False)
        support_distance = trend_mode_data.get("support_distance_pct", 100)
        context["near_support"] = near_support_raw or support_distance <= 2.5
        context["near_resistance"] = trend_mode_data.get("near_resistance", False)
        context["support_strength"] = trend_mode_data.get("support_strength", "weak")
        
        # Dodaj informacje o detektorach
        context["heatmap_vacuum"] = trend_mode_data.get("heatmap_vacuum", False)
        context["orderbook_freeze"] = trend_mode_data.get("orderbook_freeze", False)
        context["vwap_pinning"] = trend_mode_data.get("vwap_pinning", False)
        context["human_flow"] = trend_mode_data.get("human_flow", False)
        context["micro_echo"] = trend_mode_data.get("micro_echo", False)
        
        # Market structure
        context["heatmap_vacuum"] = trend_mode_data.get("heatmap_vacuum", False)
        context["vwap_pinning"] = trend_mode_data.get("vwap_pinning", False)
        
        # Volume patterns
        context["volume_increasing"] = volume_trend == "increasing"
        context["volume_declining"] = volume_trend == "declining"
        context["volume_on_bounce"] = trend_mode_data.get("volume_on_bounce", False)
        
        # Market conditions
        context["consolidation_detected"] = trend_mode_data.get("consolidation", False)
        context["momentum_strong"] = price_momentum == "strong"
        context["seller_exhaustion"] = trend_mode_data.get("seller_exhaustion", False)
        
    except Exception as e:
        print(f"⚠️ Error building market context: {e}")
    
    return context


def evaluate_trend_mode_with_perception(trend_mode_data: Dict) -> Tuple[bool, str, Dict]:
    """
    Główna funkcja integracyjna - ocenia trend_mode przez perception evaluator
    
    Args:
        trend_mode_data: Dane z trend_mode_pipeline
        
    Returns:
        tuple: (setup_convincing, reasoning, context_used)
    """
    try:
        # 1. Zbuduj market context
        context = build_market_context_from_trend_mode(trend_mode_data)
        
        # 2. Oceń setup
        setup_convincing = is_setup_convincing(context)
        
        # 3. Przygotuj reasoning
        if setup_convincing:
            reasoning = _generate_positive_reasoning(context)
        else:
            reasoning = _generate_negative_reasoning(context)
        
        return setup_convincing, reasoning, context
        
    except Exception as e:
        return False, f"Error in perception evaluation: {e}", {}


def _generate_positive_reasoning(context: Dict) -> str:
    """Generuje uzasadnienie dla pozytywnej oceny"""
    reasons = []
    
    if context.get("trend_confirmed"):
        reasons.append("trend potwierdzony")
    
    if context.get("flow_consistent"):
        reasons.append("spójny flow")
    
    if context.get("orderbook_behavior") == "bullish_control":
        reasons.append("bullish control w orderbook")
    
    if context.get("pullback_detected") and context.get("pullback_strength") == "weak":
        reasons.append("słaba korekta + recovery")
    
    if context.get("price_near_support"):
        reasons.append("wsparcie blisko")
    
    if context.get("heatmap_vacuum"):
        reasons.append("vacuum powyżej")
    
    if context.get("buyer_pressure") in ["growing", "strong"]:
        reasons.append("rosnąca presja kupujących")
    
    return "Setup convincing: " + " + ".join(reasons)


def _generate_negative_reasoning(context: Dict) -> str:
    """Generuje uzasadnienie dla negatywnej oceny"""
    issues = []
    
    if not context.get("trend_confirmed"):
        issues.append("brak potwierdzenia trendu")
    
    if not context.get("flow_consistent"):
        issues.append("niespójny flow")
    
    if context.get("orderbook_behavior") not in ["bullish_control", "accumulation", "squeeze"]:
        issues.append("neutralny/bearish orderbook")
    
    if context.get("pullback_strength") == "strong":
        issues.append("silna korekta")
    
    if context.get("buyer_pressure") == "neutral":
        issues.append("brak presji kupujących")
    
    return "Setup not convincing: " + " + ".join(issues) if issues else "Setup not convincing: warunki niespełnione"