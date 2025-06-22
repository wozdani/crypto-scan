#!/usr/bin/env python3
"""
Manual Feedback Tool - Narzędzie do ręcznego oznaczania skuteczności alertów

Pozwala na łatwe dodawanie etykiet 'good' lub 'bad' do alertów w celu treningu systemu
"""

import json
import os
from datetime import datetime
from utils.feedback_loop_trainer import add_manual_feedback, update_weights_based_on_feedback


def add_feedback_interactive():
    """Interaktywne dodawanie feedback"""
    print("🎯 Manual Feedback Tool")
    print("=" * 30)
    
    while True:
        print("\nDodaj feedback dla alertu:")
        
        symbol = input("Symbol (np. BTCUSDT): ").strip().upper()
        if not symbol:
            break
            
        decision = input("Decyzja systemu (join_trend/consider_entry/avoid): ").strip().lower()
        
        try:
            score = float(input("Final score (0.0-1.0): ").strip())
        except ValueError:
            print("❌ Nieprawidłowy score")
            continue
            
        label = input("Ocena alertu (good/bad): ").strip().lower()
        if label not in ['good', 'bad']:
            print("❌ Ocena musi być 'good' lub 'bad'")
            continue
            
        reason = input("Powód oceny (opcjonalnie): ").strip()
        
        # Add feedback
        add_manual_feedback(symbol, decision, score, label, reason)
        print(f"✅ Feedback dodany: {symbol} -> {label}")
        
        continue_input = input("\nDodać kolejny feedback? (y/n): ").strip().lower()
        if continue_input != 'y':
            break
    
    # Ask if user wants to retrain
    retrain = input("\nUruchomić trening na podstawie nowego feedback? (y/n): ").strip().lower()
    if retrain == 'y':
        print("🧠 Uruchamianie treningu...")
        success = update_weights_based_on_feedback()
        if success:
            print("✅ Trening zakończony pomyślnie")
        else:
            print("❌ Trening nie powiódł się")


def create_sample_feedback():
    """Tworzy przykładowe dane feedback do testów"""
    print("📝 Tworzenie przykładowych danych feedback...")
    
    # Sample good alerts
    good_alerts = [
        {"symbol": "BTCUSDT", "decision": "join_trend", "score": 0.85, "reason": "Strong breakout confirmed"},
        {"symbol": "ETHUSDT", "decision": "join_trend", "score": 0.78, "reason": "Good volume + trend alignment"},
        {"symbol": "ADAUSDT", "decision": "consider_entry", "score": 0.65, "reason": "Moderate setup worked well"}
    ]
    
    # Sample bad alerts
    bad_alerts = [
        {"symbol": "DOGEUSDT", "decision": "join_trend", "score": 0.72, "reason": "False breakout, price reversed"},
        {"symbol": "XRPUSDT", "decision": "consider_entry", "score": 0.55, "reason": "Range-bound, no follow-through"},
        {"symbol": "SOLUSDT", "decision": "join_trend", "score": 0.81, "reason": "Market manipulation detected"}
    ]
    
    # Add good feedback
    for alert in good_alerts:
        add_manual_feedback(alert["symbol"], alert["decision"], alert["score"], "good", alert["reason"])
    
    # Add bad feedback  
    for alert in bad_alerts:
        add_manual_feedback(alert["symbol"], alert["decision"], alert["score"], "bad", alert["reason"])
    
    print(f"✅ Dodano {len(good_alerts)} good i {len(bad_alerts)} bad feedback entries")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "add":
            add_feedback_interactive()
        elif command == "sample":
            create_sample_feedback()
            print("\nUruchomić trening na przykładowych danych? (y/n): ", end="")
            if input().strip().lower() == 'y':
                update_weights_based_on_feedback()
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: add, sample")
    else:
        print("Manual Feedback Tool Commands:")
        print("  add    - Interactively add feedback")
        print("  sample - Create sample feedback data")
        print("\nExample: python utils/feedback_manual_tool.py add")