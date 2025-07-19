#!/usr/bin/env python3
"""
Adaptive Thresholds System for Stealth Engine
Implements dynamic whale_ping thresholds, enhanced spoofing weight, and contextual phase estimation
Based on real-world analysis of HIGHUSDT and similar tokens
"""

import json
import os
from typing import Dict, Tuple, Optional
from datetime import datetime

def compute_adaptive_whale_threshold(volume_24h: float, max_order_usd: float = 0) -> float:
    """
    Oblicz adaptacyjny próg whale_ping na podstawie wolumenu 24h
    
    Args:
        volume_24h: Wolumen 24h w USD
        max_order_usd: Opcjonalnie maksymalny order w USD dla dodatkowego kontekstu
        
    Returns:
        Adaptacyjny próg whale_ping w USD
        
    Problem HIGHUSDT:
    - Volume 24h: $1.23M
    - Max order: $60  
    - Static threshold: $12,318.74
    - Result: Whale detection skipped
    
    Solution:
    - Dynamiczny próg: max(500, volume_24h * 0.0025) 
    - HIGHUSDT: max(500, 1230000 * 0.0025) = $3,075
    - Max order $60 vs $3,075 → Still skip (correctly)
    - Token z $20M volume → $50k threshold (lepsze wykrywanie)
    """
    # 🔧 BELUSDT/ETCUSDT FIX: Much lower whale detection thresholds
    base_threshold = 300  # Minimum $300 dla bardzo małych tokenów (reduced from $500)
    volume_factor = 0.0008  # 0.08% wolumenu jako próg whale (reduced from 0.25%)
    max_threshold = 15_000  # Maksymalny próg $15k dla wielkich tokenów (reduced from $50k)
    
    # Oblicz próg bazujący na wolumenie
    volume_based_threshold = volume_24h * volume_factor
    
    # Zastosuj minimum i maksimum
    adaptive_threshold = max(base_threshold, min(volume_based_threshold, max_threshold))
    
    return adaptive_threshold

def get_enhanced_spoofing_weight(is_only_signal: bool = False, base_weight: float = 0.1) -> float:
    """
    Oblicz ulepszoną wagę spoofing_layers na podstawie kontekstu
    
    Args:
        is_only_signal: Czy spoofing jest jedynym aktywnym sygnałem
        base_weight: Podstawowa waga spoofing (domyślnie 0.1)
        
    Returns:
        Ulepszona waga spoofing
        
    Problem HIGHUSDT:
    - Spoofing strength: 0.449
    - Base weight: 0.1
    - Contribution: 0.449 * 0.1 = 0.045
    - Final score: 0.086 → No alert
    
    Solution:
    - Jeśli jedyny sygnał: zwiększ wagę do 0.2
    - Jeśli z innymi sygnałami: zwiększ do 0.15
    - HIGHUSDT nowy: 0.449 * 0.2 = 0.090 → lepszy score
    """
    if is_only_signal:
        # Spoofing jako jedyny sygnał - podwyższ wagę znacząco
        enhanced_weight = max(base_weight * 2.0, 0.2)
    else:
        # Spoofing z innymi sygnałami - umiarkowane podwyższenie
        enhanced_weight = max(base_weight * 1.5, 0.15)
    
    return enhanced_weight

def estimate_contextual_phase(volume_24h: float, spoofing_active: bool = False, 
                             orderbook_depth: int = 0, price_change_24h: float = 0) -> str:
    """
    Oszacuj fazę rynku na podstawie kontekstu gdy tjde_phase=unknown
    
    Args:
        volume_24h: Wolumen 24h w USD
        spoofing_active: Czy wykryto spoofing
        orderbook_depth: Głębokość orderbooku (liczba poziomów)
        price_change_24h: Zmiana ceny 24h w %
        
    Returns:
        Oszacowana faza rynku
        
    Problem HIGHUSDT:
    - phase=unknown → brak obniżenia progu
    - Volume: $1.23M (wysokie)
    - Spoofing: aktywne  
    - Brak detekcji pre-pump phase
    
    Solution:
    - Wysokie volume + spoofing → "pre_pump_weak"
    - Dodatkowe kryteria dla lepszej klasyfikacji
    """
    # Kryteria klasyfikacji fazy
    high_volume = volume_24h >= 1_000_000  # $1M+ 
    medium_volume = volume_24h >= 500_000  # $500k+
    
    positive_momentum = price_change_24h > 2.0  # +2%+
    negative_momentum = price_change_24h < -5.0  # -5%-
    
    deep_orderbook = orderbook_depth >= 20
    shallow_orderbook = orderbook_depth <= 5
    
    # Logika klasyfikacji
    if spoofing_active and high_volume:
        if positive_momentum:
            return "pre_pump"  # Silny sygnał pre-pump
        else:
            return "pre_pump_weak"  # Słabszy ale aktywny
    
    elif spoofing_active and medium_volume:
        return "accumulation"  # Spoofing + średni volume = akumulacja
    
    elif high_volume and positive_momentum:
        return "momentum"  # Wysoki volume + pozytywna cena
    
    elif high_volume and deep_orderbook:
        return "accumulation"  # Wysoki volume + głęboki orderbook
    
    elif medium_volume and not negative_momentum:
        return "basic_screening"  # Neutralne warunki
    
    else:
        return "unknown"  # Faktycznie nieznane warunki

def calculate_stealth_alert_weak_threshold() -> float:
    """
    Oblicz próg dla stealth_alert_weak - niższy próg dla słabych ale wartościowych sygnałów
    
    Returns:
        Próg stealth_alert_weak (domyślnie 0.08)
        
    Problem HIGHUSDT:
    - Score: 0.086 (powyżej 0.08)
    - Strong spoofing + high volume
    - No alert due to standard threshold
    
    Solution:
    - stealth_alert_weak dla score > 0.08 przy spoofing + high volume
    """
    return 0.08

def should_trigger_weak_alert(stealth_score: float, volume_24h: float, 
                             spoofing_active: bool = False) -> bool:
    """
    Sprawdź czy należy wywołać słaby alert stealth
    
    Args:
        stealth_score: Stealth score tokena
        volume_24h: Wolumen 24h w USD  
        spoofing_active: Czy wykryto spoofing
        
    Returns:
        True jeśli należy wywołać słaby alert
        
    HIGHUSDT case:
    - Score: 0.086
    - Volume: $1.23M
    - Spoofing: True
    → Should trigger weak alert: True
    """
    weak_threshold = calculate_stealth_alert_weak_threshold()
    high_volume = volume_24h >= 1_000_000
    
    # Warunki słabego alertu
    score_above_weak = stealth_score >= weak_threshold
    valuable_context = spoofing_active and high_volume
    
    return score_above_weak and valuable_context

def get_adaptive_system_stats() -> Dict:
    """
    Pobierz statystyki systemu adaptacyjnych progów
    
    Returns:
        Dict ze statystykami systemu
    """
    return {
        "adaptive_whale_threshold": {
            "base_threshold": 500,
            "volume_factor": 0.0025,
            "max_threshold": 50_000,
            "description": "Dynamic whale detection based on 24h volume"
        },
        "enhanced_spoofing": {
            "base_weight": 0.1,
            "enhanced_single": 0.2,
            "enhanced_multi": 0.15,
            "description": "Enhanced spoofing weight based on signal context"
        },
        "contextual_phase": {
            "phases": ["pre_pump", "pre_pump_weak", "accumulation", "momentum", "basic_screening", "unknown"],
            "criteria": "Volume, spoofing, price change, orderbook depth",
            "description": "Contextual phase estimation when TJDE phase unknown"
        },
        "weak_alert": {
            "threshold": 0.08,
            "requirements": "Spoofing + high volume + score > 0.08",
            "description": "Weak stealth alerts for valuable low-score signals"
        }
    }

def test_adaptive_thresholds():
    """Test wszystkich adaptacyjnych funkcji z przykładem HIGHUSDT"""
    print("🧪 TEST ADAPTIVE THRESHOLDS - HIGHUSDT CASE")
    print("=" * 50)
    
    # Dane HIGHUSDT z logów
    volume_24h = 1_230_000  # $1.23M
    max_order_usd = 60      # $60
    spoofing_strength = 0.449
    original_score = 0.086
    
    # Test 1: Adaptive whale threshold
    adaptive_threshold = compute_adaptive_whale_threshold(volume_24h, max_order_usd)
    print(f"✅ Adaptive whale threshold: ${adaptive_threshold:,.0f}")
    print(f"   Original: $12,318.74 vs New: ${adaptive_threshold:,.0f}")
    print(f"   Max order: ${max_order_usd} → Still skip: {max_order_usd < adaptive_threshold}")
    
    # Test 2: Enhanced spoofing weight  
    original_weight = 0.1
    enhanced_weight = get_enhanced_spoofing_weight(is_only_signal=True)
    original_contribution = spoofing_strength * original_weight
    enhanced_contribution = spoofing_strength * enhanced_weight
    
    print(f"\n✅ Enhanced spoofing weight:")
    print(f"   Original: {original_weight} → New: {enhanced_weight}")
    print(f"   Contribution: {original_contribution:.3f} → {enhanced_contribution:.3f}")
    
    # Test 3: Contextual phase estimation
    estimated_phase = estimate_contextual_phase(volume_24h, spoofing_active=True)
    print(f"\n✅ Contextual phase: {estimated_phase}")
    
    # Test 4: Weak alert system
    should_weak_alert = should_trigger_weak_alert(original_score, volume_24h, spoofing_active=True)
    print(f"\n✅ Weak alert trigger: {should_weak_alert}")
    print(f"   Score: {original_score} > 0.08 + spoofing + high volume")
    
    print(f"\n📊 HIGHUSDT IMPROVEMENT SUMMARY:")
    print(f"   Whale threshold: ${adaptive_threshold:,.0f} (more appropriate for ${volume_24h:,.0f})")
    print(f"   Spoofing contribution: +{(enhanced_contribution - original_contribution):.3f}")
    print(f"   Phase estimation: {estimated_phase}")
    print(f"   Weak alert eligible: {should_weak_alert}")

if __name__ == "__main__":
    test_adaptive_thresholds()