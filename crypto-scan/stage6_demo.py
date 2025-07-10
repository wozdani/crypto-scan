#!/usr/bin/env python3
"""
🛠 Stage 6 DEMO: Address Trust Manager - Feedback Loop dla adresów DEX/Whale
🎯 Cel: Automatycznie nadawać większe znaczenie tym adresom (DEX i whale), które w przeszłości poprzedzały wzrosty

📦 Demonstracja:
1. System rejestruje predykcje adresów podczas whale_ping i dex_inflow
2. Symulacja historii skuteczności adresów
3. Obliczanie trust boost na podstawie track record
4. Integracja z Stealth Engine
"""

import sys
import os
import time
import tempfile

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def demo_address_trust_basic():
    """Demo 1: Podstawowe funkcjonalności Address Trust Manager"""
    print("🧠 DEMO 1: Podstawowe funkcjonalności Trust Manager")
    print("=" * 50)
    
    from stealth_engine.address_trust_manager import AddressTrustManager
    
    # Utwórz tymczasowy manager
    temp_file = "/tmp/demo_trust.json"
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    manager = AddressTrustManager(cache_file=temp_file)
    
    # Symuluj różne adresy z różną skutecznością
    addresses = {
        "0xSMARTMONEY": {"hits": 8, "misses": 2, "description": "Smart Money Whale"},
        "0xMEDIUMWHALE": {"hits": 6, "misses": 4, "description": "Medium Whale"},
        "0xNOOBTRADER": {"hits": 2, "misses": 8, "description": "Noob Trader"},
        "0xNEWADDRESS": {"hits": 1, "misses": 0, "description": "New Address"}
    }
    
    print("\n📊 Symulacja historii adresów:")
    for address, data in addresses.items():
        # Dodaj historię
        for _ in range(data["hits"]):
            manager.update_address_performance(address, True)
        for _ in range(data["misses"]):
            manager.update_address_performance(address, False)
        
        # Pokaż rezultaty
        stats = manager.get_trust_statistics(address)
        boost = manager.get_address_boost(address)
        
        print(f"  {data['description']}: {stats['hits']}/{stats['total_predictions']} = {stats['trust_score']:.1%} → boost: +{boost:.3f}")
    
    print("\n🎯 Trust Boost Levels:")
    print("  • 0.100 = Bardzo wysokie zaufanie (≥80% skuteczność)")  
    print("  • 0.050 = Wysokie zaufanie (≥70% skuteczność)")
    print("  • 0.030 = Średnie zaufanie (≥60% skuteczność)")
    print("  • 0.020 = Niskie zaufanie (≥50% skuteczność)")
    print("  • 0.000 = Brak zaufania (<50% lub <3 predykcji)")
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print("\n✅ Demo 1 zakończone pomyślnie!")

def demo_stealth_integration():
    """Demo 2: Integracja z funkcjami Stealth Engine"""
    print("\n🔗 DEMO 2: Integracja z Stealth Engine")
    print("=" * 50)
    
    from stealth_engine.stealth_signals import StealthSignalDetector
    from stealth_engine.address_trust_manager import get_trust_manager, update_address_performance
    
    detector = StealthSignalDetector()
    
    # Symuluj dane tokena z wysoką aktywnością whale/DEX
    token_data = {
        "symbol": "DEMOTOKEN",
        "orderbook": {
            "bids": [["50.0", "50000.0"], ["49.9", "30000.0"], ["49.8", "20000.0"]],  # Duże zlecenia
            "asks": [["50.1", "40000.0"], ["50.2", "25000.0"], ["50.3", "15000.0"]]
        },
        "volume_24h": 5000000,  # Wysoki wolumen
        "candles_15m": [{"volume": 200000}, {"volume": 180000}, {"volume": 220000}, {"volume": 250000}],
        "dex_inflow": 100000,  # Duży DEX inflow
        "dex_inflow_history": [20000, 25000, 30000, 35000, 40000, 45000, 50000, 60000]
    }
    
    print(f"\n📈 Analizując token: {token_data['symbol']}")
    print(f"  Największe zlecenie bid: ${float(token_data['orderbook']['bids'][0][1]) * float(token_data['orderbook']['bids'][0][0]):,.0f}")
    print(f"  DEX inflow: ${token_data['dex_inflow']:,}")
    
    # Test 1: Pierwsze skanowanie (bez trust history)
    print("\n🔍 Pierwsze skanowanie (bez trust history):")
    whale_signal = detector.check_whale_ping(token_data)
    dex_signal = detector.check_dex_inflow(token_data)
    
    print(f"  Whale ping: active={whale_signal.active}, strength={whale_signal.strength:.3f}")
    print(f"  DEX inflow: active={dex_signal.active}, strength={dex_signal.strength:.3f}")
    
    # Symuluj że te adresy miały trafne predykcje w przeszłości
    trust_manager = get_trust_manager()
    
    # Adres whale z dobrą historią (5 trafnych, 1 nietrafna)
    whale_addr = f"whale_{token_data['symbol'].lower()}_2500000"[:42]
    for _ in range(5):
        update_address_performance(whale_addr, True)  # Trafne predykcje
    update_address_performance(whale_addr, False)     # 1 nietrafna
    
    # Adres DEX z bardzo dobrą historią (7 trafnych, 1 nietrafna)  
    dex_addr = f"dex_{token_data['symbol'].lower()}_100000"[:42]
    for _ in range(7):
        update_address_performance(dex_addr, True)    # Trafne predykcje
    update_address_performance(dex_addr, False)       # 1 nietrafna
    
    whale_stats = trust_manager.get_trust_statistics(whale_addr)
    dex_stats = trust_manager.get_trust_statistics(dex_addr)
    
    print(f"\n📊 Historia adresów:")
    print(f"  Whale addr: {whale_stats['hits']}/{whale_stats['total_predictions']} = {whale_stats['trust_score']:.1%} → boost: +{whale_stats['boost_value']:.3f}")
    print(f"  DEX addr: {dex_stats['hits']}/{dex_stats['total_predictions']} = {dex_stats['trust_score']:.1%} → boost: +{dex_stats['boost_value']:.3f}")
    
    # Test 2: Drugie skanowanie (z trust boost)
    print("\n🚀 Drugie skanowanie (z trust boost):")
    whale_signal_boosted = detector.check_whale_ping(token_data)
    dex_signal_boosted = detector.check_dex_inflow(token_data)
    
    print(f"  Whale ping: active={whale_signal_boosted.active}, strength={whale_signal_boosted.strength:.3f} (+{whale_signal_boosted.strength - whale_signal.strength:.3f})")
    print(f"  DEX inflow: active={dex_signal_boosted.active}, strength={dex_signal_boosted.strength:.3f} (+{dex_signal_boosted.strength - dex_signal.strength:.3f})")
    
    print("\n✨ Efekt:")
    print("  System automatycznie zwiększa siłę sygnałów od zaufanych adresów!")
    print("  Adresy z historią trafnych predykcji otrzymują wyższe scoring.")
    
    print("\n✅ Demo 2 zakończone pomyślnie!")

def demo_smart_money_detection():
    """Demo 3: Smart Money Detection - automatyczna identyfikacja"""
    print("\n🎯 DEMO 3: Smart Money Detection")
    print("=" * 50)
    
    from stealth_engine.address_trust_manager import AddressTrustManager
    
    # Utwórz manager
    temp_file = "/tmp/smart_money_demo.json"
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    manager = AddressTrustManager(cache_file=temp_file)
    
    # Symuluj populację różnych typów adresów
    address_profiles = [
        # Smart Money - konsystentnie trafne predykcje
        {"addr": "0xSMARTFUND001", "profile": [True]*9 + [False]*1, "type": "Institutional Fund"},
        {"addr": "0xWHALEEXPERT", "profile": [True]*8 + [False]*2, "type": "Expert Whale"},
        
        # Average Traders - mieszane wyniki
        {"addr": "0xAVERAGETRADER1", "profile": [True]*6 + [False]*4, "type": "Average Trader"},
        {"addr": "0xAVERAGETRADER2", "profile": [True]*5 + [False]*5, "type": "Average Trader"},
        
        # Poor Performers - głównie nietrafne
        {"addr": "0xREKTTRADER", "profile": [True]*2 + [False]*8, "type": "Rekt Trader"},
        {"addr": "0xFOMOBUYER", "profile": [True]*1 + [False]*9, "type": "FOMO Buyer"},
        
        # New addresses - za mało danych
        {"addr": "0xNEWBIE001", "profile": [True]*2, "type": "New Address"},
        {"addr": "0xNEWBIE002", "profile": [True]*1 + [False]*1, "type": "New Address"}
    ]
    
    print("\n📊 Symulacja różnych typów adresów:")
    
    smart_money = []
    average_performers = []
    poor_performers = []
    insufficient_data = []
    
    for profile in address_profiles:
        # Symuluj historię
        for success in profile["profile"]:
            manager.update_address_performance(profile["addr"], success)
        
        # Analizuj rezultaty
        stats = manager.get_trust_statistics(profile["addr"])
        boost = manager.get_address_boost(profile["addr"])
        
        print(f"  {profile['type']}: {stats['hits']}/{stats['total_predictions']} = {stats['trust_score']:.1%} → boost: +{boost:.3f}")
        
        # Kategoryzuj na podstawie trust score i boost
        if boost >= 0.05:
            smart_money.append(profile)
        elif boost >= 0.02:
            average_performers.append(profile)
        elif stats['total_predictions'] >= 3:
            poor_performers.append(profile)
        else:
            insufficient_data.append(profile)
    
    print(f"\n🧠 AUTOMATYCZNA KLASYFIKACJA:")
    print(f"  🟢 Smart Money ({len(smart_money)}): Boost ≥0.05 (≥70% skuteczność)")
    for profile in smart_money:
        print(f"    - {profile['addr'][:12]}... ({profile['type']})")
    
    print(f"  🟡 Average Performers ({len(average_performers)}): Boost 0.02-0.049 (50-69% skuteczność)")
    for profile in average_performers:
        print(f"    - {profile['addr'][:12]}... ({profile['type']})")
    
    print(f"  🔴 Poor Performers ({len(poor_performers)}): Boost 0.00 (<50% skuteczność)")
    for profile in poor_performers:
        print(f"    - {profile['addr'][:12]}... ({profile['type']})")
    
    print(f"  ⚪ Insufficient Data ({len(insufficient_data)}): <3 predykcji")
    for profile in insufficient_data:
        print(f"    - {profile['addr'][:12]}... ({profile['type']})")
    
    print(f"\n💡 INTELIGENCJA SYSTEMU:")
    print(f"  System automatycznie identyfikuje 'smart money' na podstawie track record.")
    print(f"  Tylko adresy z ≥70% skutecznością otrzymują znaczący boost.")
    print(f"  Redukuje szum i koncentruje się na realnych sygnałach.")
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print("\n✅ Demo 3 zakończone pomyślnie!")

def demo_production_workflow():
    """Demo 4: Workflow produkcyjny"""
    print("\n⚙️ DEMO 4: Workflow Produkcyjny")
    print("=" * 50)
    
    print("🔄 LIFECYCLE ADRESU W SYSTEMIE:")
    print()
    print("1. 📝 REJESTRACJA PREDYKCJI")
    print("   └─ Gdy whale_ping/dex_inflow wykrywa adres")
    print("   └─ record_address_prediction(token, address)")
    print()
    print("2. ⏱️ OCZEKIWANIE NA OCENĘ") 
    print("   └─ 6 godzin na sprawdzenie czy cena wzrosła ≥2%")
    print("   └─ evaluate_pending_predictions(price_fetcher)")
    print()
    print("3. 📊 AKTUALIZACJA SKUTECZNOŚCI")
    print("   └─ update_address_performance(address, success)")
    print("   └─ hits/misses → trust_score → boost_value")
    print()
    print("4. 🚀 ZASTOSOWANIE BOOST")
    print("   └─ get_address_boost(address) → +0.02 do +0.10")
    print("   └─ strength = min(1.0, base_strength + trust_boost)")
    print()
    print("5. 🧹 OCZYSZCZANIE")
    print("   └─ cleanup_old_data() usuwa nieaktywne adresy")
    print("   └─ trust decay po 30 dniach nieaktywności")
    
    print(f"\n📈 BENEFITY SYSTEMU:")
    print(f"  ✅ Automatyczna identyfikacja smart money")
    print(f"  ✅ Redukcja false positive przez trust filtering")  
    print(f"  ✅ Uczenie się na podstawie historical performance")
    print(f"  ✅ Self-improving accuracy przez feedback loop")
    print(f"  ✅ Dynamiczne dostrajanie bez manual intervention")
    
    print("\n✅ Demo 4 zakończone pomyślnie!")

def main():
    """Uruchom wszystkie demo Stage 6"""
    print("🛠 STAGE 6 DEMONSTRATION: Address Trust Manager")
    print("🎯 Feedback Loop dla adresów DEX/Whale - dynamiczna nauka predykcyjna")
    print("=" * 70)
    
    try:
        demo_address_trust_basic()
        demo_stealth_integration()
        demo_smart_money_detection()
        demo_production_workflow()
        
        print("\n" + "=" * 70)
        print("🎉 STAGE 6 COMPLETE - Address Trust Manager OPERATIONAL")
        print("✅ System samodzielnie identyfikuje 'smart money' adresy")
        print("✅ Trust scoring automatycznie zwiększa skuteczność detektorów")
        print("✅ Feedback loop redukuje szum i koncentruje na realnych sygnałach")
        print("✅ Self-learning intelligence poprawia accuracy w czasie")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()