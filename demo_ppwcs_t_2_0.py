#!/usr/bin/env python3
"""
Demo PPWCS-T 2.0 - Demonstracja nowej logiki trend mode
Pokazuje działanie systemu bez wymagania zewnętrznych API
"""

import sys
import numpy as np
from datetime import datetime

# Add utils to path
sys.path.insert(0, 'utils')

from utils.trend_mode import compute_ppwcs_t_trend_boost, compute_trend_score

def create_demo_trend_data():
    """Tworzy przykładowe dane idealnie pasujące do PPWCS-T 2.0"""
    data = []
    base_price = 1.0000
    
    for i in range(30):
        timestamp = int(datetime.now().timestamp() * 1000) - (29-i) * 300000
        
        # Stabilny wzrost z małą zmiennością (idealne dla PPWCS-T)
        price_increase = 0.0005 * i  # 0.05% na świecę
        
        # Małe losowe wahania (VWAP pinning)
        noise = np.random.uniform(-0.002, 0.002)
        
        current_price = base_price * (1 + price_increase + noise)
        
        open_price = current_price * 0.9995
        high_price = current_price * 1.0015  # Małe wicki
        low_price = current_price * 0.9985
        close_price = current_price
        
        # Wolumen rośnie stopniowo (volume slope up)
        volume = 10000 + i * 300 + np.random.uniform(0, 1000)
        
        data.append([timestamp, open_price, high_price, low_price, close_price, volume])
    
    return data

def demo_ppwcs_t_scoring():
    """Demonstracja scoringu PPWCS-T 2.0"""
    print("🎯 PPWCS-T 2.0 - Demo Scoring System")
    print("=" * 50)
    
    # Stwórz dane testowe
    data = create_demo_trend_data()
    
    print("📊 Analiza danych:")
    print(f"   Liczba świec: {len(data)}")
    print(f"   Zakres cenowy: ${data[0][4]:.6f} → ${data[-1][4]:.6f}")
    print(f"   Wzrost ceny: {((data[-1][4] / data[0][4]) - 1) * 100:.2f}%")
    print(f"   Zakres wolumenu: {data[0][5]:.0f} → {data[-1][5]:.0f}")
    print()
    
    # Test PPWCS-T 2.0
    print("🧠 PPWCS-T 2.0 Analysis:")
    result = compute_ppwcs_t_trend_boost(data, "DEMOTOKEN")
    
    print(f"   Trend Boost: {result['trend_boost']}/20 punktów")
    print(f"   Trend Mode Active: {result['trend_mode_active']}")
    print("   Aktywne detektory:")
    for detail in result['boost_details']:
        print(f"     ✅ {detail}")
    print()
    
    # Test pełnej integracji
    print("🚀 Full Trend Mode Integration:")
    full_result = compute_trend_score(data, "DEMOTOKEN", enable_extensions=False)
    
    print(f"   Total Score: {full_result['trend_score']}/77 punktów")
    print(f"   Trend Active: {full_result['trend_mode_active']}")
    print(f"   PPWCS-T Contribution: {full_result.get('ppwcs_t_boost', {}).get('trend_boost', 0)} points")
    print("   Wszystkie sygnały:")
    for i, signal in enumerate(full_result['trend_summary'], 1):
        print(f"     {i}. {signal}")
    
    return result, full_result

def demo_different_scenarios():
    """Demo różnych scenariuszy PPWCS-T 2.0"""
    print("\n📋 PPWCS-T 2.0 - Różne Scenariusze")
    print("=" * 50)
    
    scenarios = [
        ("Stabilny trend wzrostowy", create_demo_trend_data()),
        ("RSI w zakresie 50-60", create_rsi_accumulation_data()),
        ("VWAP pinning", create_vwap_pinning_data()),
        ("Volume slope bez pumpy", create_volume_slope_data())
    ]
    
    for name, data in scenarios:
        print(f"\n🔍 Scenariusz: {name}")
        result = compute_ppwcs_t_trend_boost(data, "TEST")
        
        print(f"   Boost: {result['trend_boost']} points")
        print(f"   Active: {result['trend_mode_active']}")
        if result['boost_details']:
            print(f"   Detektory: {', '.join(result['boost_details'])}")
        else:
            print("   Detektory: brak aktywnych")

def create_rsi_accumulation_data():
    """Dane z RSI w zakresie 50-60"""
    data = []
    base_price = 1.0
    
    for i in range(25):
        timestamp = int(datetime.now().timestamp() * 1000) - (24-i) * 300000
        
        # Małe wahania generujące RSI ~55
        if i % 4 == 0:
            change = 0.008   # +0.8%
        elif i % 4 == 1:
            change = -0.003  # -0.3%
        elif i % 4 == 2:
            change = 0.006   # +0.6%
        else:
            change = -0.002  # -0.2%
        
        current_price = base_price * (1 + change)
        
        data.append([
            timestamp,
            current_price * 0.999,
            current_price * 1.002,
            current_price * 0.998,
            current_price,
            10000 + np.random.uniform(0, 2000)
        ])
        
        base_price = current_price
    
    return data

def create_vwap_pinning_data():
    """Dane z VWAP pinning"""
    data = []
    base_price = 1.0
    
    for i in range(20):
        timestamp = int(datetime.now().timestamp() * 1000) - (19-i) * 300000
        
        # Cena bardzo blisko VWAP (±0.5%)
        deviation = np.random.uniform(-0.005, 0.005)
        current_price = base_price * (1 + deviation)
        
        data.append([
            timestamp,
            current_price * 0.9998,
            current_price * 1.0008,
            current_price * 0.9992,
            current_price,
            12000 + np.random.uniform(0, 1500)
        ])
    
    return data

def create_volume_slope_data():
    """Dane z rosnącym wolumenem bez pumpy"""
    data = []
    base_price = 1.0
    
    for i in range(15):
        timestamp = int(datetime.now().timestamp() * 1000) - (14-i) * 300000
        
        # Stabilne ceny
        noise = np.random.uniform(-0.001, 0.001)
        current_price = base_price * (1 + noise)
        
        # Stopniowo rosnący wolumen
        volume = 8000 + i * 400
        
        data.append([
            timestamp,
            current_price * 0.9995,
            current_price * 1.0012,
            current_price * 0.9988,
            current_price,
            volume
        ])
    
    return data

def main():
    """Główna funkcja demo"""
    print("🎯 PPWCS-T 2.0 - Live Demo")
    print("Nowa logika trend mode dla stabilnego wzrostu bez breakoutów")
    print("=" * 70)
    
    # Demo głównego systemu
    demo_result, full_result = demo_ppwcs_t_scoring()
    
    # Demo różnych scenariuszy
    demo_different_scenarios()
    
    # Podsumowanie
    print("\n📈 Podsumowanie PPWCS-T 2.0:")
    print("=" * 40)
    print("✅ RSI trendowa akumulacja (50-60): +5 points")
    print("✅ VWAP pinning detection: +5 points") 
    print("✅ Volume slope up (bez pumpy): +5 points")
    print("✅ Liquidity box + higher lows: +5 points")
    print("✅ Breakout exclusion filter: aktywny")
    print("✅ Próg aktywacji: ≥10 points + brak breakout")
    print("✅ Maksymalny boost: 20 points")
    print("✅ Integracja z legacy system: kompletna")
    
    print(f"\n🎯 Demo Result: {full_result['trend_score']} total points")
    print(f"🚀 System Status: {'ACTIVE' if full_result['trend_mode_active'] else 'INACTIVE'}")
    
    if full_result['trend_mode_active']:
        print("\n💡 PPWCS-T 2.0 successfully detected stable growth trend!")
    
    print("\n✨ Implementation complete - ready for production use")

if __name__ == "__main__":
    main()