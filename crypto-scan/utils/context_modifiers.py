#!/usr/bin/env python3
"""
ContextualModifiers - Kontekstowe modyfikacje cech dla TJDE

Dostosowuje cechy scoringu w zależności od kontekstu rynkowego:
- Faza rynku (retest, breakout, range)
- Globalny trend BTC
- Sesja tradingowa (Asia, EU, US)
- Volatilność rynku
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any


def apply_contextual_modifiers(features: Dict[str, float], market_context: Dict[str, Any]) -> Dict[str, float]:
    """
    Zastosuj modyfikacje kontekstowe do cech TJDE
    
    Args:
        features: Oryginalny dict cech
        market_context: Kontekst rynkowy
        
    Returns:
        Dict z zmodyfikowanymi cechami
    """
    modified = features.copy()
    
    try:
        # === MODYFIKACJE FAZY RYNKU ===
        market_phase = market_context.get("market_phase", "")
        
        if market_phase == "retest-confirmation":
            # W fazie retest zwiększ wagę support reaction
            modified["support_reaction"] *= 1.25
            modified["trend_strength"] *= 1.1
            print(f"[CONTEXT] Retest phase: support_reaction +25%, trend_strength +10%")
            
        elif market_phase == "breakout-continuation":
            # W breakout zwiększ wagę trend strength
            modified["trend_strength"] *= 1.3
            modified["liquidity_pattern_score"] *= 1.15
            print(f"[CONTEXT] Breakout phase: trend_strength +30%, liquidity +15%")
            
        elif market_phase == "pre-breakout":
            # Przed breakoutem zwiększ wagę accumulation patterns
            modified["pullback_quality"] *= 1.2
            modified["psych_score"] *= 0.8  # Mniej ważne manipulacje
            print(f"[CONTEXT] Pre-breakout: pullback_quality +20%, psych_score -20%")
            
        elif market_phase == "exhaustion-pullback":
            # W exhaustion zwiększ wagę psychology
            modified["psych_score"] *= 1.4
            modified["trend_strength"] *= 0.7
            print(f"[CONTEXT] Exhaustion phase: psych_score +40%, trend_strength -30%")
        
        # === MODYFIKACJE GLOBALNEGO TRENDU BTC ===
        btc_trend = market_context.get("btc_global_trend", "")
        
        if btc_trend == "strong_up":
            # Silny trend BTC w górę wspiera wszystkie longi
            modified["trend_strength"] *= 1.2
            modified["htf_supportive_score"] *= 1.1
            modified["psych_score"] *= 0.9  # Mniej ważne bear traps
            print(f"[CONTEXT] Strong BTC uptrend: trend +20%, HTF +10%, psych -10%")
            
        elif btc_trend == "strong_down":
            # Silny trend BTC w dół - ostrożność
            modified["trend_strength"] *= 0.7
            modified["psych_score"] *= 1.3  # Więcej manipulacji w bear market
            modified["support_reaction"] *= 1.1  # Ważniejsze poziomy support
            print(f"[CONTEXT] Strong BTC downtrend: trend -30%, psych +30%, support +10%")
            
        elif btc_trend == "sideways":
            # Rynek boczny - altcoiny mogą być niezależne
            modified["liquidity_pattern_score"] *= 1.15
            modified["htf_supportive_score"] *= 0.9
            print(f"[CONTEXT] BTC sideways: liquidity +15%, HTF -10%")
        
        # === MODYFIKACJE SESJI TRADINGOWEJ ===
        session = market_context.get("session", "")
        
        if session == "asia":
            # Sesja azjatycka - mniejsza płynność
            modified["liquidity_pattern_score"] *= 0.8
            modified["support_reaction"] *= 0.9
            print(f"[CONTEXT] Asian session: liquidity -20%, support -10%")
            
        elif session == "london":
            # Sesja londyńska - wysoka płynność
            modified["liquidity_pattern_score"] *= 1.2
            modified["trend_strength"] *= 1.1
            print(f"[CONTEXT] London session: liquidity +20%, trend +10%")
            
        elif session == "ny":
            # Sesja amerykańska - największa płynność i volatilność
            modified["trend_strength"] *= 1.15
            modified["liquidity_pattern_score"] *= 1.25
            modified["psych_score"] *= 1.1  # Więcej manipulacji w NY
            print(f"[CONTEXT] NY session: trend +15%, liquidity +25%, psych +10%")
        
        # === MODYFIKACJE VOLATILNOŚCI ===
        volatility = market_context.get("volatility_level", "")
        
        if volatility == "high":
            # Wysoka volatilność - większa ostrożność
            modified["psych_score"] *= 1.2
            modified["support_reaction"] *= 1.1
            print(f"[CONTEXT] High volatility: psych +20%, support +10%")
            
        elif volatility == "low":
            # Niska volatilność - trudniejsze breakouty
            modified["trend_strength"] *= 0.9
            modified["pullback_quality"] *= 1.1
            print(f"[CONTEXT] Low volatility: trend -10%, pullback +10%")
        
        # === MODYFIKACJE KORELACJI RYNKU ===
        market_correlation = market_context.get("market_correlation", "")
        
        if market_correlation == "high":
            # Wysoka korelacja - ważniejszy HTF
            modified["htf_supportive_score"] *= 1.3
            print(f"[CONTEXT] High correlation: HTF +30%")
            
        elif market_correlation == "low":
            # Niska korelacja - altcoin może być niezależny
            modified["htf_supportive_score"] *= 0.8
            modified["liquidity_pattern_score"] *= 1.1
            print(f"[CONTEXT] Low correlation: HTF -20%, liquidity +10%")
        
        # === OGRANICZENIA WARTOŚCI ===
        # Upewnij się, że wartości pozostają w rozsądnych granicach
        for key in modified:
            modified[key] = max(0.0, min(2.0, modified[key]))  # Zakres 0.0-2.0
        
        return modified
        
    except Exception as e:
        print(f"❌ [CONTEXT ERROR] Failed to apply modifiers: {e}")
        return features  # Zwróć oryginalne cechy w przypadku błędu


def get_market_context(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Zbierz kontekst rynkowy dla modyfikacji
    
    Args:
        symbol: Symbol do analizy
        
    Returns:
        Dict z kontekstem rynkowym
    """
    try:
        context = {}
        
        # === WYKRYWANIE SESJI ===
        utc_hour = datetime.now(timezone.utc).hour
        
        if 0 <= utc_hour < 8:
            context["session"] = "asia"
        elif 8 <= utc_hour < 16:
            context["session"] = "london"
        else:
            context["session"] = "ny"
        
        # === DOMYŚLNE WARTOŚCI (mogą być rozszerzone) ===
        context.update({
            "market_phase": "unknown",  # Będzie ustawiane przez market_phase.py
            "btc_global_trend": "unknown",  # Może być pobierane z BTC analizy
            "volatility_level": "medium",
            "market_correlation": "medium"
        })
        
        return context
        
    except Exception as e:
        print(f"❌ [CONTEXT ERROR] Failed to get market context: {e}")
        return {
            "session": "unknown",
            "market_phase": "unknown",
            "btc_global_trend": "unknown",
            "volatility_level": "medium",
            "market_correlation": "medium"
        }


def log_context_modifications(original_features: Dict[str, float], 
                            modified_features: Dict[str, float], 
                            market_context: Dict[str, Any],
                            symbol: str = "UNKNOWN"):
    """
    Zapisz modyfikacje kontekstowe do loga
    """
    try:
        os.makedirs("logs", exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "market_context": market_context,
            "modifications": {},
            "total_change": 0.0
        }
        
        total_change = 0.0
        for key in original_features:
            original = original_features[key]
            modified = modified_features[key]
            change = modified - original
            change_pct = (change / original * 100) if original != 0 else 0.0
            
            log_entry["modifications"][key] = {
                "original": round(original, 4),
                "modified": round(modified, 4),
                "change": round(change, 4),
                "change_pct": round(change_pct, 2)
            }
            
            total_change += abs(change)
        
        log_entry["total_change"] = round(total_change, 4)
        
        with open("logs/context_modifications.jsonl", "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(log_entry, ensure_ascii=False)}\n")
        
    except Exception as e:
        print(f"❌ [CONTEXT LOG ERROR] {e}")


if __name__ == "__main__":
    # Test context modifiers
    print("🧪 Testing ContextualModifiers...")
    
    # Sample features
    test_features = {
        "trend_strength": 0.7,
        "pullback_quality": 0.6,
        "support_reaction": 0.5,
        "liquidity_pattern_score": 0.4,
        "psych_score": 0.3,
        "htf_supportive_score": 0.8,
        "market_phase_modifier": 0.1
    }
    
    # Test different contexts
    test_contexts = [
        {
            "market_phase": "retest-confirmation",
            "btc_global_trend": "strong_up",
            "session": "london",
            "volatility_level": "high"
        },
        {
            "market_phase": "breakout-continuation", 
            "btc_global_trend": "sideways",
            "session": "asia",
            "volatility_level": "low"
        },
        {
            "market_phase": "exhaustion-pullback",
            "btc_global_trend": "strong_down",
            "session": "ny",
            "volatility_level": "high"
        }
    ]
    
    for i, context in enumerate(test_contexts, 1):
        print(f"\n📊 TEST CONTEXT {i}: {context}")
        
        modified = apply_contextual_modifiers(test_features, context)
        
        print(f"🔄 MODIFICATIONS:")
        for key in test_features:
            original = test_features[key]
            new_val = modified[key]
            change_pct = ((new_val - original) / original * 100) if original != 0 else 0
            print(f"  {key:25} = {original:.3f} → {new_val:.3f} ({change_pct:+.1f}%)")
    
    # Test market context detection
    current_context = get_market_context()
    print(f"\n🌍 CURRENT MARKET CONTEXT: {current_context}")
    
    print(f"\n✅ ContextualModifiers test complete")