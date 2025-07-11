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
        
        # CRITICAL FIX: Generate market context from candle data when not available
        htf_phase = market_context.get("htf_phase", "unknown")
        volume_trend = market_context.get("volume_trend", "neutral")
        fear_greed = market_context.get("fear_greed", 50)
        market_sentiment = market_context.get("market_sentiment", "neutral")
        volatility_regime = market_context.get("volatility_regime", "normal")
        
        # If htf_phase is unknown, try to derive it from candle data
        if htf_phase == "unknown":
            candles = market_context.get("candles", [])
            htf_candles = market_context.get("htf_candles", [])
            
            # Try HTF candles first, then regular candles
            analysis_candles = htf_candles if htf_candles and len(htf_candles) >= 10 else candles
            
            if analysis_candles and len(analysis_candles) >= 20:
                print(f"[MARKET MODIFIER] Deriving HTF phase from {len(analysis_candles)} candles")
                
                closes = []
                volumes = []
                highs = []
                lows = []
                
                for candle in analysis_candles[-20:]:
                    if isinstance(candle, dict):
                        closes.append(float(candle.get('close', 0)))
                        volumes.append(float(candle.get('volume', 0)))
                        highs.append(float(candle.get('high', 0)))
                        lows.append(float(candle.get('low', 0)))
                    elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                        closes.append(float(candle[4]))
                        volumes.append(float(candle[5]))
                        highs.append(float(candle[2]))
                        lows.append(float(candle[3]))
                
                if len(closes) >= 15:
                    # Calculate trend
                    recent_close = closes[-1]
                    old_close = closes[-15]
                    trend_change = (recent_close - old_close) / old_close if old_close != 0 else 0
                    
                    # Calculate volatility
                    price_ranges = [(highs[i] - lows[i]) / closes[i] for i in range(len(closes)) if closes[i] != 0]
                    avg_volatility = sum(price_ranges) / len(price_ranges) if price_ranges else 0
                    
                    # Calculate volume trend
                    if len(volumes) >= 10 and sum(volumes) > 0:
                        recent_vol = sum(volumes[-5:]) / 5
                        historical_vol = sum(volumes[-15:-5]) / 10
                        vol_change = (recent_vol - historical_vol) / historical_vol if historical_vol != 0 else 0
                        
                        if vol_change > 0.3:
                            volume_trend = "rising"
                        elif vol_change < -0.3:
                            volume_trend = "falling"
                        else:
                            volume_trend = "neutral"
                    
                    # Determine HTF phase
                    if trend_change > 0.05 and avg_volatility < 0.04:
                        htf_phase = "uptrend"
                    elif trend_change > 0.02:
                        htf_phase = "bullish_momentum"
                    elif trend_change < -0.05 and avg_volatility < 0.04:
                        htf_phase = "downtrend"
                    elif trend_change < -0.02:
                        htf_phase = "bearish_momentum"
                    elif abs(trend_change) < 0.01 and volume_trend == "rising":
                        htf_phase = "accumulation"
                    elif abs(trend_change) < 0.01:
                        htf_phase = "consolidation"
                    elif avg_volatility > 0.06:
                        htf_phase = "high_volatility"
                    else:
                        htf_phase = "range"
                    
                    # Determine volatility regime
                    if avg_volatility > 0.08:
                        volatility_regime = "extreme"
                    elif avg_volatility > 0.05:
                        volatility_regime = "high"
                    elif avg_volatility < 0.02:
                        volatility_regime = "low"
                    else:
                        volatility_regime = "normal"
                    
                    print(f"[MARKET MODIFIER] Derived: HTF={htf_phase}, Vol={volume_trend}, Volatility={volatility_regime}")
                    print(f"[MARKET MODIFIER] Trend change: {trend_change:.4f}, Avg volatility: {avg_volatility:.4f}")
        
        print(f"[MARKET MODIFIER] Final context: HTF={htf_phase}, Volume={volume_trend}, Fear/Greed={fear_greed}, Volatility={volatility_regime}")
        
        # === POZYTYWNE MODYFIKATORY ===
        
        # Silny trend na HTF z rosnącym wolumenem
        if htf_phase == "uptrend" and volume_trend == "rising":
            modifier += 0.07
            print(f"[MODIFIER +] HTF uptrend + rising volume: +0.07")
        
        # Faza akumulacji - idealna do wejść
        if htf_phase == "accumulation":
            modifier += 0.05
            print(f"[MODIFIER +] Accumulation phase: +0.05")
        
        # Konsolidacja z niską zmiennością - potencjał na breakout
        if htf_phase == "consolidation" and volatility_regime == "low":
            modifier += 0.04
            print(f"[MODIFIER +] Consolidation + low volatility (breakout potential): +0.04")
        
        # Range z rosnącym wolumenem - przygotowanie do wybicia
        if htf_phase == "range" and volume_trend == "rising":
            modifier += 0.03
            print(f"[MODIFIER +] Range + rising volume (breakout prep): +0.03")
        
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


def market_phase_modifier(market_phase: str, trend_strength: float = 0.0) -> float:
    """
    Simple market phase modifier function called from scan_token_async.py
    
    Args:
        market_phase: Market phase string (basic_screening, pullback, etc.)
        trend_strength: Trend strength for fallback enhancement
        
    Returns:
        float: Phase modifier
    """
    try:
        print(f"[PHASE MODIFIER] Input: market_phase={market_phase}, trend_strength={trend_strength:.3f}")
        
        # CRITICAL FIX: Add basic_screening → pullback fallback mapping  
        if market_phase == "basic_screening":
            # Use trend_strength to enhance basic_screening with smart fallback
            if trend_strength >= 0.7:
                fallback_modifier = 0.2  # Strong trend gets pullback-like boost
                print(f"[PHASE FALLBACK] basic_screening + strong trend ({trend_strength:.3f}) → pullback-like modifier: +{fallback_modifier}")
                return fallback_modifier
            elif trend_strength >= 0.5:
                fallback_modifier = 0.1  # Medium trend gets moderate boost
                print(f"[PHASE FALLBACK] basic_screening + medium trend ({trend_strength:.3f}) → moderate modifier: +{fallback_modifier}")
                return fallback_modifier
            else:
                fallback_modifier = 0.05  # Low trend gets small boost to prevent zero
                print(f"[PHASE FALLBACK] basic_screening + weak trend ({trend_strength:.3f}) → small modifier: +{fallback_modifier}")
                return fallback_modifier
        
        # Regular phase modifiers
        phase_modifiers = {
            "pre-breakout": 0.15,
            "pullback": 0.12,
            "neutral-pullback": 0.08,  # New fallback phase  
            "breakout-continuation": 0.10,
            "retest-confirmation": 0.08,
            "exhaustion-pullback": -0.05,
            "range-accumulation": 0.03,
            "uptrend": 0.12,
            "downtrend": -0.10,
            "consolidation": 0.02,
            "accumulation": 0.08,
            "distribution": -0.08,
            "sideways": 0.0,
            "unknown": 0.0
        }
        
        modifier = phase_modifiers.get(market_phase, 0.0)
        print(f"[PHASE MODIFIER] {market_phase} → modifier: {modifier}")
        return modifier
        
    except Exception as e:
        print(f"[PHASE MODIFIER ERROR] {e}")
        # Emergency fallback for basic_screening
        if market_phase == "basic_screening":
            emergency_modifier = max(0.05, trend_strength * 0.3)  # Minimum 0.05, up to 0.3
            print(f"[PHASE EMERGENCY] basic_screening emergency fallback: +{emergency_modifier}")
            return emergency_modifier
        return 0.0


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