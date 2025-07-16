#!/usr/bin/env python3
"""
🔔 Alert Router - Inteligentna kolejka alertów z priority scoring
🎯 Cel: Automatyczne zarządzanie kolejką alertów na podstawie priority_score i momentu rynkowego

📦 Funkcjonalności:
1. compute_priority_score() - obliczanie priorytetu tokena
2. Dynamiczne bonusy na podstawie tagów i DEX inflow
3. Integracja z Stage 7 Trigger Alert System
4. Wsparcie dla trusted addresses i stealth signals
"""

import time
import os
from typing import Dict, List, Tuple, Optional


def compute_priority_score(score: float, tags: List[str], inflow_usd: float = 0.0, 
                          trust_score: float = 0.0, trigger_detected: bool = False) -> float:
    """
    🎯 Oblicz priority_score tokena na podstawie scoring, tagów, DEX inflow i trust score
    
    Args:
        score: Bazowy score tokena (TJDE, Stealth, etc.)
        tags: Lista tagów tokena ["trusted", "priority", "stealth_ready", etc.]
        inflow_usd: Wartość DEX inflow w USD 
        trust_score: Trust score wykrytych adresów (0.0-1.0)
        trigger_detected: Czy wykryto trigger alert (Stage 7)
        
    Returns:
        float: Priority score (wyższy = wyższy priorytet)
    """
    
    # Bazowy priority score = score tokena
    priority_score = score
    
    # 🏷️ Tag bonuses
    tag_bonus = 0.0
    
    if "trusted" in tags:
        tag_bonus += 0.8       # Zaufane adresy
    if "priority" in tags:
        tag_bonus += 1.0       # Priorytetowe tokeny
    if "stealth_ready" in tags:
        tag_bonus += 0.5       # Gotowe sygnały stealth
    if "whale_detected" in tags:
        tag_bonus += 0.6       # Wykryto wieloryby
    if "smart_money" in tags:
        tag_bonus += 1.2       # Smart money detected
    if "high_volume" in tags:
        tag_bonus += 0.3       # Wysoki wolumen
    if "breakout" in tags:
        tag_bonus += 0.4       # Wybicie
    
    # 💰 DEX Inflow bonus - większy inflow zwiększa priorytet
    inflow_bonus = 0.0
    if inflow_usd > 0:
        # Skalowanie: $100k inflow = +1.0 bonus (max)
        inflow_bonus = min(inflow_usd / 100_000, 1.0)
    
    # 🧠 Trust Score bonus - wysokie trust score = wyższy priorytet
    trust_bonus = 0.0
    if trust_score > 0:
        # Trust score 80%+ = +0.8 bonus, 90%+ = +1.0 bonus
        if trust_score >= 0.9:
            trust_bonus = 1.0
        elif trust_score >= 0.8:
            trust_bonus = 0.8
        elif trust_score >= 0.7:
            trust_bonus = 0.5
        elif trust_score >= 0.6:
            trust_bonus = 0.3
    
    # 🚨 Stage 7 Trigger Alert bonus - natychmiastowy priorytet
    trigger_bonus = 0.0
    if trigger_detected:
        trigger_bonus = 2.0  # Bardzo wysoki bonus dla trigger alerts
    
    # 📊 Oblicz końcowy priority score
    final_priority = priority_score + tag_bonus + inflow_bonus + trust_bonus + trigger_bonus
    
    print(f"[PRIORITY ROUTER] Priority calculation:")
    print(f"  Base score: {score:.3f}")
    print(f"  Tag bonus: +{tag_bonus:.3f} (tags: {tags})")
    print(f"  Inflow bonus: +{inflow_bonus:.3f} (${inflow_usd:,.0f})")
    print(f"  Trust bonus: +{trust_bonus:.3f} (trust: {trust_score:.1%})")
    print(f"  Trigger bonus: +{trigger_bonus:.3f} (detected: {trigger_detected})")
    print(f"  Final priority: {final_priority:.3f}")
    
    return round(final_priority, 3)


def generate_alert_tags(symbol: str, market_data: Dict, stealth_signals: List[Dict] = None,
                       trust_score: float = 0.0, trigger_detected: bool = False) -> List[str]:
    """
    🏷️ Generuj tagi dla tokena na podstawie danych rynkowych i sygnałów
    
    Args:
        symbol: Symbol tokena
        market_data: Dane rynkowe tokena
        stealth_signals: Lista aktywnych sygnałów stealth
        trust_score: Trust score wykrytych adresów
        trigger_detected: Czy wykryto trigger alert
        
    Returns:
        List[str]: Lista tagów dla tokena
    """
    
    tags = []
    
    # 🧠 Trust-based tags
    if trust_score >= 0.9:
        tags.append("super_trusted")
    elif trust_score >= 0.8:
        tags.append("trusted")
    elif trust_score >= 0.7:
        tags.append("reliable")
    
    # 🚨 Trigger alert tag
    if trigger_detected:
        tags.append("smart_money")
        tags.append("priority")
    
    # 📊 Volume-based tags
    volume_24h = market_data.get("volume_24h", 0)
    if volume_24h > 50_000_000:  # $50M+
        tags.append("high_volume")
    elif volume_24h > 10_000_000:  # $10M+
        tags.append("medium_volume")
    
    # 🐋 Whale detection tags
    orderbook = market_data.get("orderbook", {})
    if orderbook:
        bids = orderbook.get("bids", [])
        if bids and len(bids) > 0:
            try:
                # Sprawdź duże zlecenia (>$100k)
                max_order_usd = 0
                for bid in bids[:5]:  # Top 5 bids
                    if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                        price = float(bid[0])
                        size = float(bid[1])
                        order_usd = price * size
                        max_order_usd = max(max_order_usd, order_usd)
                
                if max_order_usd > 500_000:  # $500k+
                    tags.append("whale_detected")
                    tags.append("priority")
                elif max_order_usd > 100_000:  # $100k+
                    tags.append("large_orders")
            except (ValueError, TypeError, IndexError):
                pass
    
    # 🔍 Stealth signals tags
    if stealth_signals:
        active_signals = [s for s in stealth_signals if s.get("active", False)]
        if len(active_signals) >= 3:
            tags.append("stealth_ready")
        if len(active_signals) >= 5:
            tags.append("multi_signal")
        
        # Sprawdź konkretne sygnały
        signal_names = [s.get("signal_name", "") for s in active_signals]
        if "whale_ping" in signal_names:
            tags.append("whale_ping")
        if "dex_inflow" in signal_names:
            tags.append("dex_inflow")
        if "volume_spike" in signal_names:
            tags.append("volume_spike")
        if "orderbook_imbalance" in signal_names:
            tags.append("orderbook_anomaly")
    
    # 💹 DEX inflow tags
    dex_inflow = market_data.get("dex_inflow", 0)
    if dex_inflow > 200_000:  # $200k+
        tags.append("high_inflow")
    elif dex_inflow > 50_000:  # $50k+
        tags.append("medium_inflow")
    
    # 📈 Price action tags
    price_change_24h = market_data.get("price_change_24h", 0)
    if price_change_24h > 15:  # +15%+
        tags.append("pumping")
    elif price_change_24h > 5:  # +5%+
        tags.append("rising")
    elif price_change_24h < -10:  # -10%+
        tags.append("falling")
    
    print(f"[ALERT TAGS] {symbol}: Generated {len(tags)} tags: {tags}")
    
    return tags


def extract_active_functions(stealth_signals: List[Dict]) -> List[str]:
    """
    🔍 Wyciągnij nazwy aktywnych funkcji z sygnałów stealth
    
    Args:
        stealth_signals: Lista sygnałów stealth
        
    Returns:
        List[str]: Lista nazw aktywnych funkcji
    """
    active_functions = []
    
    if not stealth_signals:
        return active_functions
    
    for signal in stealth_signals:
        if signal.get("active", False):
            signal_name = signal.get("signal_name", "")
            if signal_name:
                active_functions.append(signal_name)
    
    return active_functions


def should_fast_track_alert(priority_score: float, tags: List[str], 
                           trust_score: float = 0.0) -> Tuple[bool, str]:
    """
    🚀 Sprawdź czy token powinien otrzymać fast-track alert (natychmiastowy)
    
    Args:
        priority_score: Priority score tokena
        tags: Lista tagów tokena
        trust_score: Trust score wykrytych adresów
        
    Returns:
        Tuple[bool, str]: (czy_fast_track, powód)
    """
    
    # 🚨 Stage 7 Trigger Alert - natychmiastowy fast-track
    if "smart_money" in tags:
        return True, "Smart money detected (Stage 7 trigger)"
    
    # 🧠 Super trusted addresses - fast-track
    if "super_trusted" in tags and trust_score >= 0.9:
        return True, f"Super trusted address ({trust_score:.1%} success rate)"
    
    # 📊 Bardzo wysoki priority score
    if priority_score >= 5.0:
        return True, f"Exceptional priority score ({priority_score:.1f})"
    
    # 🐋 Wielkie wieloryby + wysokie zaufanie
    if "whale_detected" in tags and "trusted" in tags:
        return True, "Trusted whale combination"
    
    # 🔥 Multi-signal + high inflow
    if "multi_signal" in tags and "high_inflow" in tags:
        return True, "Multiple stealth signals + high DEX inflow"
    
    # 📈 Pumping + whale activity
    if "pumping" in tags and ("whale_ping" in tags or "whale_detected" in tags):
        return True, "Price momentum + whale activity"
    
    return False, "Standard priority queue"


def calculate_alert_delay(priority_score: float, tags: List[str], 
                         base_delay: int = 30) -> int:
    """
    ⏰ Oblicz opóźnienie alertu na podstawie priority score
    
    Args:
        priority_score: Priority score tokena
        tags: Lista tagów tokena
        base_delay: Bazowe opóźnienie w sekundach
        
    Returns:
        int: Opóźnienie w sekundach
    """
    
    # Fast-track alerts = 0 sekund opóźnienia
    is_fast_track, reason = should_fast_track_alert(priority_score, tags)
    if is_fast_track:
        print(f"[ALERT DELAY] Fast-track: 0s delay ({reason})")
        return 0
    
    # Oblicz opóźnienie na podstawie priority score
    # Wysoki priority = krótsze opóźnienie
    if priority_score >= 4.0:
        delay = 5   # 5 sekund
    elif priority_score >= 3.0:
        delay = 15  # 15 sekund
    elif priority_score >= 2.0:
        delay = 30  # 30 sekund (base)
    elif priority_score >= 1.0:
        delay = 60  # 1 minuta
    else:
        delay = 120 # 2 minuty
    
    print(f"[ALERT DELAY] Priority {priority_score:.1f} → {delay}s delay")
    
    return delay


def create_priority_alert_data(symbol: str, score: float, priority_score: float,
                              tags: List[str], market_data: Dict,
                              trust_score: float = 0.0, 
                              trigger_detected: bool = False,
                              active_functions: List[str] = None,
                              gpt_feedback: str = "",
                              ai_confidence: float = 0.0) -> Dict:
    """
    📦 Utwórz kompletne dane alertu z priority scoring
    
    Args:
        symbol: Symbol tokena
        score: Bazowy score
        priority_score: Priority score
        tags: Lista tagów
        market_data: Dane rynkowe
        trust_score: Trust score adresów
        trigger_detected: Czy wykryto trigger
        
    Returns:
        Dict: Kompletne dane alertu
    """
    
    # Oblicz opóźnienie alertu
    delay = calculate_alert_delay(priority_score, tags)
    
    # Sprawdź fast-track
    is_fast_track, fast_track_reason = should_fast_track_alert(priority_score, tags, trust_score)
    
    alert_data = {
        "symbol": symbol,
        "score": score,
        "priority_score": priority_score,
        "tags": tags,
        "timestamp": time.time(),
        "delay_seconds": delay,
        "is_fast_track": is_fast_track,
        "fast_track_reason": fast_track_reason,
        "trust_score": trust_score,
        "trigger_detected": trigger_detected,
        
        # 🔐 CRITICAL CONSENSUS DECISION DATA - Required for Telegram Alert Manager
        "consensus_decision": market_data.get("consensus_decision", "HOLD"),
        "consensus_enabled": market_data.get("consensus_enabled", False),
        "consensus_confidence": market_data.get("consensus_confidence", 0.0),
        "consensus_detectors": market_data.get("consensus_detectors", []),
        
        # Active functions and AI feedback for enhanced Telegram alerts
        "active_functions": active_functions or [],
        "gpt_feedback": gpt_feedback,
        "ai_confidence": ai_confidence,
        
        # Market data summary
        "price_usd": market_data.get("price_usd", 0),
        "volume_24h": market_data.get("volume_24h", 0),
        "price_change_24h": market_data.get("price_change_24h", 0),
        "dex_inflow": market_data.get("dex_inflow", 0),
        
        # Alert scheduling
        "ready_at": time.time() + delay,
        "processed": False
    }
    
    print(f"[PRIORITY ALERT] {symbol} alert created:")
    print(f"  Priority: {priority_score:.1f} | Delay: {delay}s | Fast-track: {is_fast_track}")
    
    return alert_data


# Global convenience functions for easy integration

def route_alert_with_priority(symbol: str, score: float, market_data: Dict,
                             stealth_signals: List[Dict] = None,
                             trust_score: float = 0.0, 
                             trigger_detected: bool = False,
                             active_functions: List[str] = None,
                             gpt_feedback: str = "",
                             ai_confidence: float = 0.0) -> Dict:
    """
    🎯 Kompletna funkcja routingu alertu z priority scoring
    
    Args:
        symbol: Symbol tokena
        score: Bazowy score tokena
        market_data: Dane rynkowe
        stealth_signals: Lista sygnałów stealth
        trust_score: Trust score wykrytych adresów
        trigger_detected: Czy wykryto trigger alert
        
    Returns:
        Dict: Kompletne dane alertu z priority
    """
    
    # 🔧 TYPE SAFETY FIX: Ensure market_data is dict to prevent 'str' object has no attribute 'get' error
    if not isinstance(market_data, dict):
        print(f"[QUEUE PRIORITY ALERT] Error for {symbol}: market_data is {type(market_data)}, expected dict")
        # Create minimal market_data dict from available data
        market_data = {
            "volume_24h": 0,
            "price_usd": 0,
            "price_change_24h": 0,
            "dex_inflow": 0,
            "orderbook": {}
        }
    
    # 🔧 TYPE SAFETY FIX: Ensure stealth_signals is list 
    if stealth_signals is not None and not isinstance(stealth_signals, list):
        print(f"[QUEUE PRIORITY ALERT] Warning for {symbol}: stealth_signals is {type(stealth_signals)}, expected list")
        stealth_signals = []
    
    # 🎯 CRITICAL BUY FILTER: Only send alerts for BUY decisions
    consensus_decision = market_data.get("consensus_decision", "HOLD")
    if consensus_decision != "BUY":
        print(f"[ALERT FILTER] {symbol} → Consensus decision '{consensus_decision}' != BUY - blocking alert")
        return None
    
    try:
        # 1. Generuj tagi
        tags = generate_alert_tags(symbol, market_data, stealth_signals, trust_score, trigger_detected)
        
        # 1.5. Ekstrakcja active_functions z stealth_signals jeśli nie podano
        if active_functions is None and stealth_signals:
            active_functions = extract_active_functions(stealth_signals)
        
        # 2. Oblicz priority score
        dex_inflow = market_data.get("dex_inflow", 0)
        priority_score = compute_priority_score(score, tags, dex_inflow, trust_score, trigger_detected)
        
        # 3. Utwórz dane alertu z active_functions i GPT feedback
        alert_data = create_priority_alert_data(
            symbol, score, priority_score, tags, market_data, trust_score, trigger_detected,
            active_functions, gpt_feedback, ai_confidence
        )
        
        return alert_data
        
    except Exception as e:
        print(f"[QUEUE PRIORITY ALERT] Error for {symbol}: {e}")
        print(f"[QUEUE PRIORITY ALERT] market_data type: {type(market_data)}")
        print(f"[QUEUE PRIORITY ALERT] stealth_signals type: {type(stealth_signals)}")
        
        # Return minimal alert data to prevent complete failure
        return {
            "symbol": symbol,
            "score": score,
            "priority_score": score,
            "tags": ["error_recovery"],
            "timestamp": time.time(),
            "delay_seconds": 60,
            "is_fast_track": False,
            "fast_track_reason": "Error recovery mode",
            "trust_score": trust_score,
            "trigger_detected": trigger_detected,
            "price_usd": 0,
            "volume_24h": 0,
            "price_change_24h": 0,
            "dex_inflow": 0,
            "ready_at": time.time() + 60,
            "processed": False,
            "error": str(e)
        }


def get_priority_stats() -> Dict:
    """
    📊 Pobierz statystyki priority alert router
    
    Returns:
        Dict: Statystyki systemu
    """
    
    return {
        "router_version": "1.0",
        "features": [
            "Priority scoring",
            "Dynamic tag generation", 
            "Fast-track alerts",
            "Trust score integration",
            "Stage 7 trigger support",
            "DEX inflow bonuses"
        ],
        "tag_types": [
            "trusted", "priority", "stealth_ready", "whale_detected",
            "smart_money", "high_volume", "high_inflow", "pumping"
        ],
        "fast_track_triggers": [
            "Smart money detection",
            "Super trusted addresses (≥90%)",
            "Exceptional priority score (≥5.0)",
            "Trusted whale combination",
            "Multi-signal + high inflow"
        ]
    }


if __name__ == "__main__":
    # Test podstawowej funkcjonalności
    print("🔔 ALERT ROUTER - Priority Scoring Test")
    print("=" * 50)
    
    # Test compute_priority_score
    test_score = 2.5
    test_tags = ["trusted", "whale_detected", "stealth_ready"]
    test_inflow = 150000
    test_trust = 0.85
    test_trigger = True
    
    priority = compute_priority_score(test_score, test_tags, test_inflow, test_trust, test_trigger)
    print(f"\n✅ Test priority score: {priority}")
    
    # Test tag generation
    test_market_data = {
        "volume_24h": 25000000,
        "price_change_24h": 8.5,
        "dex_inflow": 120000,
        "orderbook": {
            "bids": [["100.0", "8000.0"], ["99.9", "5000.0"]]
        }
    }
    
    tags = generate_alert_tags("TESTUSDT", test_market_data, trust_score=0.85, trigger_detected=True)
    print(f"\n✅ Test tags: {tags}")
    
    print("\n🎉 Alert Router ready for priority-based alert management!")