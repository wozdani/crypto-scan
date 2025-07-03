#!/usr/bin/env python3
"""
Market Phase Modifier - Etap 6 TJDE v2
Dynamiczny wzmacniacz lub tłumik końcowego scoringu w zależności od globalnej fazy rynku.
Dodaje trzeci wymiar oceny: mikro (cechy), mezo (profil), makro (kontekst rynku).
"""

import time
from typing import Dict, Optional


def compute_market_phase_modifier(market_context: Dict) -> float:
    """
    Oblicza modyfikator fazy rynku na podstawie kontekstu makro
    
    Args:
        market_context: Słownik z kontekstem rynku zawierający:
            - htf_phase: Faza na wyższych ramach czasowych
            - volume_trend: Trend wolumenu (rising/falling/neutral)
            - fear_greed: Indeks strachu/chciwości (0-100)
            - market_sentiment: Ogólny sentyment (bullish/bearish/neutral)
            - volatility_regime: Reżim zmienności (low/normal/high/extreme)
    
    Returns:
        float: Modyfikator w zakresie -0.20 do +0.10
    """
    try:
        modifier = 0.0
        
        # Pobierz kontekst rynku
        htf_phase = market_context.get("htf_phase", "unknown")
        volume_trend = market_context.get("volume_trend", "neutral")
        fear_greed = market_context.get("fear_greed", 50)
        market_sentiment = market_context.get("market_sentiment", "neutral")
        volatility_regime = market_context.get("volatility_regime", "normal")
        
        print(f"[MARKET MODIFIER] HTF Phase: {htf_phase}, Volume: {volume_trend}, Fear/Greed: {fear_greed}")
        
        # === POZYTYWNE MODYFIKATORY ===
        
        # Silny trend na HTF z rosnącym wolumenem
        if htf_phase == "uptrend" and volume_trend == "rising":
            modifier += 0.07
            print(f"[MODIFIER +] HTF uptrend + rising volume: +0.07")
        
        # Faza akumulacji - idealna do wejść
        if htf_phase == "accumulation":
            modifier += 0.05
            print(f"[MODIFIER +] Accumulation phase: +0.05")
        
        # Extreme greed (potencjalne kontynuacje trendów)
        if fear_greed >= 80:
            modifier += 0.03
            print(f"[MODIFIER +] Extreme greed ({fear_greed}): +0.03")
        
        # Bullish sentiment z normalną zmiennością
        if market_sentiment == "bullish" and volatility_regime == "normal":
            modifier += 0.04
            print(f"[MODIFIER +] Bullish sentiment + normal vol: +0.04")
        
        # === NEGATYWNE MODYFIKATORY ===
        
        # Downtrend z spadającym wolumenem
        if htf_phase == "downtrend" and volume_trend == "falling":
            modifier -= 0.10
            print(f"[MODIFIER -] HTF downtrend + falling volume: -0.10")
        
        # Panika rynkowa
        if htf_phase == "panic" or fear_greed < 20:
            modifier -= 0.20
            print(f"[MODIFIER -] Panic phase or extreme fear ({fear_greed}): -0.20")
        
        # Wysoka zmienność bez kierunku
        if volatility_regime == "extreme" and htf_phase == "sideways":
            modifier -= 0.15
            print(f"[MODIFIER -] Extreme volatility + sideways: -0.15")
        
        # Bearish sentiment
        if market_sentiment == "bearish":
            modifier -= 0.08
            print(f"[MODIFIER -] Bearish sentiment: -0.08")
        
        # Stagnacja wolumenu w trendzie
        if htf_phase in ["uptrend", "downtrend"] and volume_trend == "stagnant":
            modifier -= 0.05
            print(f"[MODIFIER -] Trend without volume support: -0.05")
        
        # === OGRANICZENIA ===
        
        # Ogranicz modyfikator do bezpiecznego zakresu
        modifier = max(-0.20, min(0.10, modifier))
        
        print(f"[MARKET MODIFIER] Final modifier: {modifier}")
        return round(modifier, 3)
        
    except Exception as e:
        print(f"[MARKET MODIFIER ERROR] {e}")
        return 0.0


def analyze_market_context(market_data: Dict, candles_15m: list, candles_5m: list) -> Dict:
    """
    Analizuje kontekst rynku na podstawie dostępnych danych
    
    Args:
        market_data: Dane rynkowe tokena
        candles_15m: Świece 15-minutowe
        candles_5m: Świece 5-minutowe
        
    Returns:
        Dict: Kontekst rynku do obliczenia modyfikatora
    """
    try:
        context = {
            "htf_phase": "unknown",
            "volume_trend": "neutral", 
            "fear_greed": 50,
            "market_sentiment": "neutral",
            "volatility_regime": "normal"
        }
        
        if not candles_15m or len(candles_15m) < 20:
            return context
        
        # Analiza trendu HTF na podstawie EMA
        recent_candles = candles_15m[-20:]
        prices = [float(candle[4]) for candle in recent_candles]  # close prices
        
        # Prosty trend analysis
        if len(prices) >= 10:
            early_avg = sum(prices[:10]) / 10
            late_avg = sum(prices[-10:]) / 10
            
            change_pct = (late_avg - early_avg) / early_avg * 100
            
            if change_pct > 3:
                context["htf_phase"] = "uptrend"
                context["market_sentiment"] = "bullish"
            elif change_pct < -3:
                context["htf_phase"] = "downtrend" 
                context["market_sentiment"] = "bearish"
            elif abs(change_pct) < 1:
                context["htf_phase"] = "sideways"
            else:
                context["htf_phase"] = "accumulation" if change_pct > 0 else "distribution"
        
        # Analiza wolumenu
        if len(recent_candles) >= 10:
            volumes = [float(candle[5]) for candle in recent_candles]
            early_vol_avg = sum(volumes[:10]) / 10
            late_vol_avg = sum(volumes[-10:]) / 10
            
            vol_change = (late_vol_avg - early_vol_avg) / early_vol_avg * 100
            
            if vol_change > 20:
                context["volume_trend"] = "rising"
            elif vol_change < -20:
                context["volume_trend"] = "falling"
            elif abs(vol_change) < 5:
                context["volume_trend"] = "stagnant"
        
        # Analiza zmienności
        if len(prices) >= 5:
            price_changes = []
            for i in range(1, len(prices)):
                change = abs((prices[i] - prices[i-1]) / prices[i-1] * 100)
                price_changes.append(change)
            
            avg_volatility = sum(price_changes) / len(price_changes)
            
            if avg_volatility > 5:
                context["volatility_regime"] = "extreme"
            elif avg_volatility > 2:
                context["volatility_regime"] = "high"
            elif avg_volatility < 0.5:
                context["volatility_regime"] = "low"
        
        # Symulacja Fear & Greed Index na podstawie price action
        if context["htf_phase"] == "uptrend" and context["volume_trend"] == "rising":
            context["fear_greed"] = 75  # Greed
        elif context["htf_phase"] == "downtrend" and context["volume_trend"] == "rising":
            context["fear_greed"] = 25  # Fear
        elif context["volatility_regime"] == "extreme":
            context["fear_greed"] = 15  # Extreme Fear
        
        print(f"[MARKET CONTEXT] Analyzed: {context}")
        return context
        
    except Exception as e:
        print(f"[MARKET CONTEXT ERROR] {e}")
        return {
            "htf_phase": "unknown",
            "volume_trend": "neutral",
            "fear_greed": 50,
            "market_sentiment": "neutral", 
            "volatility_regime": "normal"
        }


def apply_market_phase_modifier(base_score: float, market_context: Dict) -> tuple:
    """
    Aplikuje modyfikator fazy rynku do bazowego score
    
    Args:
        base_score: Bazowy score TJDE
        market_context: Kontekst rynku
        
    Returns:
        tuple: (adjusted_score, modifier)
    """
    try:
        modifier = compute_market_phase_modifier(market_context)
        adjusted_score = base_score + modifier
        
        # Upewnij się, że score pozostaje w zakresie 0.0-1.0
        adjusted_score = max(0.0, min(1.0, adjusted_score))
        
        print(f"[TJDE SCORE] Base: {base_score:.3f} + Modifier: {modifier:+.3f} → Final: {adjusted_score:.3f}")
        
        return round(adjusted_score, 3), modifier
        
    except Exception as e:
        print(f"[MODIFIER APPLICATION ERROR] {e}")
        return base_score, 0.0


def test_market_phase_modifier():
    """Test różnych scenariuszy modyfikatora"""
    print("🧪 Testing Market Phase Modifier")
    
    # Test 1: Bullish scenario
    bullish_context = {
        "htf_phase": "uptrend",
        "volume_trend": "rising", 
        "fear_greed": 75,
        "market_sentiment": "bullish",
        "volatility_regime": "normal"
    }
    
    modifier1 = compute_market_phase_modifier(bullish_context)
    print(f"✅ Bullish scenario modifier: {modifier1}")
    
    # Test 2: Bearish scenario  
    bearish_context = {
        "htf_phase": "downtrend",
        "volume_trend": "falling",
        "fear_greed": 15,
        "market_sentiment": "bearish", 
        "volatility_regime": "high"
    }
    
    modifier2 = compute_market_phase_modifier(bearish_context)
    print(f"✅ Bearish scenario modifier: {modifier2}")
    
    # Test 3: Neutral scenario
    neutral_context = {
        "htf_phase": "sideways",
        "volume_trend": "neutral",
        "fear_greed": 50,
        "market_sentiment": "neutral",
        "volatility_regime": "normal"
    }
    
    modifier3 = compute_market_phase_modifier(neutral_context)
    print(f"✅ Neutral scenario modifier: {modifier3}")


if __name__ == "__main__":
    test_market_phase_modifier()