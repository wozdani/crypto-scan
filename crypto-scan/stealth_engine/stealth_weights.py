"""
Stealth Weights Manager
Zarządzanie wagami sygnałów dla Stealth Engine

Pozwala na dynamiczną aktualizację wag na podstawie feedback loop
i skuteczności poszczególnych sygnałów w predykcji rynku
"""

import json
import os
from typing import Dict, Optional

# Ścieżka do pliku wag
WEIGHTS_PATH = "crypto-scan/cache/stealth_weights.json"

# Domyślne wagi dla wszystkich sygnałów Stealth Engine
DEFAULT_WEIGHTS = {
    # Sygnały orderbook
    "orderbook_imbalance": 0.15,
    "large_bid_walls": 0.12,
    "bid_ask_spread_tightening": 0.16,
    
    # Sygnały volume
    "volume_spike": 0.18,
    "volume_accumulation": 0.14,
    "volume_slope": 0.17,
    
    # Sygnały DEX i whale
    "dex_inflow": 0.20,
    "whale_ping": 0.22,
    
    # Sygnały mikrostruktury
    "spoofing_layers": 0.10,
    "ghost_orders": 0.11,
    "liquidity_absorption": 0.13,
    "event_tag": 0.10,
    
    # Negatywne sygnały
    "spoofing_detected": -0.25,
    "ask_wall_removal": 0.14
}

def load_weights():
    """
    Załaduj wagi sygnałów z pliku JSON
    
    Returns:
        Słownik z wagami sygnałów
    """
    if not os.path.exists(WEIGHTS_PATH):
        # Jeśli plik nie istnieje, utwórz go z domyślnymi wagami
        save_weights(DEFAULT_WEIGHTS)
        return DEFAULT_WEIGHTS.copy()
    
    try:
        with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
            weights = json.load(f)
        
        # Sprawdź czy wszystkie domyślne sygnały są obecne
        updated = False
        for signal_name, default_weight in DEFAULT_WEIGHTS.items():
            if signal_name not in weights:
                weights[signal_name] = default_weight
                updated = True
                print(f"[STEALTH WEIGHTS] Added missing signal: {signal_name} = {default_weight}")
        
        # Zapisz zaktualizowane wagi jeśli dodano nowe sygnały
        if updated:
            save_weights(weights)
        
        print(f"[STEALTH WEIGHTS] Loaded {len(weights)} weights from {WEIGHTS_PATH}")
        return weights
        
    except Exception as e:
        print(f"[STEALTH WEIGHTS ERROR] Failed to load weights: {e}")
        print(f"[STEALTH WEIGHTS] Using default weights")
        return DEFAULT_WEIGHTS.copy()

def save_weights(weights):
    """
    Zapisz wagi do pliku JSON
    
    Args:
        weights: Słownik z wagami do zapisania
        
    Returns:
        True jeśli sukces, False w przeciwnym razie
    """
    try:
        # Upewnij się że katalog istnieje
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        
        with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2, ensure_ascii=False)
        
        print(f"[STEALTH WEIGHTS] Saved {len(weights)} weights to {WEIGHTS_PATH}")
        return True
        
    except Exception as e:
        print(f"[STEALTH WEIGHTS ERROR] Failed to save weights: {e}")
        return False

def update_weight(signal_name, delta):
    """
    Aktualizuj wagę konkretnego sygnału (używane przez feedback loop)
    
    Args:
        signal_name: Nazwa sygnału
        delta: Zmiana wagi (może być ujemna)
        
    Returns:
        True jeśli sukces, False w przeciwnym razie
    """
    try:
        weights = load_weights()
        current = weights.get(signal_name, 1.0)
        new_weight = current + delta
        
        # Nie schodzimy poniżej 0.1 dla pozytywnych sygnałów
        if signal_name == "spoofing_detected":
            # Negatywny sygnał - może być od -0.5 do 0.0
            new_weight = max(-0.5, min(0.0, new_weight))
        else:
            # Pozytywne sygnały - minimum 0.1, maksimum 0.5
            new_weight = max(0.1, min(0.5, new_weight))
        
        # Zaokrąglij do 3 miejsc po przecinku
        weights[signal_name] = round(new_weight, 3)
        
        if save_weights(weights):
            print(f"[STEALTH WEIGHTS] Updated {signal_name}: {current:.3f} → {new_weight:.3f} (Δ{delta:+.3f})")
            return True
        
        return False
        
    except Exception as e:
        print(f"[STEALTH WEIGHTS ERROR] Failed to update {signal_name}: {e}")
        return False

def get_weight_stats():
    """
    Pobierz statystyki wag sygnałów
    
    Returns:
        Słownik ze statystykami
    """
    weights = load_weights()
    
    positive_weights = {k: v for k, v in weights.items() if v > 0}
    negative_weights = {k: v for k, v in weights.items() if v < 0}
    zero_weights = {k: v for k, v in weights.items() if v == 0}
    
    return {
        'total_signals': len(weights),
        'positive_weights': len(positive_weights),
        'negative_weights': len(negative_weights),
        'zero_weights': len(zero_weights),
        'max_weight': max(weights.values()) if weights else 0,
        'min_weight': min(weights.values()) if weights else 0,
        'avg_positive': sum(positive_weights.values()) / len(positive_weights) if positive_weights else 0,
        'weights_breakdown': weights
    }

def reset_to_defaults():
    """
    Resetuj wszystkie wagi do wartości domyślnych
    
    Returns:
        True jeśli sukces
    """
    print("[STEALTH WEIGHTS] Resetting all weights to default values")
    return save_weights(DEFAULT_WEIGHTS.copy())

def batch_update_weights(weight_updates):
    """
    Zaktualizuj wiele wag jednocześnie (feedback loop)
    
    Args:
        weight_updates: Słownik {signal_name: delta}
        
    Returns:
        True jeśli wszystkie aktualizacje się powiodły
    """
    success_count = 0
    
    for signal_name, delta in weight_updates.items():
        if update_weight(signal_name, delta):
            success_count += 1
    
    success_rate = success_count / len(weight_updates) if weight_updates else 0
    print(f"[STEALTH WEIGHTS] Batch update: {success_count}/{len(weight_updates)} successful ({success_rate*100:.1f}%)")
    
    return success_rate == 1.0


# Klasa dla zaawansowanego zarządzania wagami (backward compatibility)
class StealthWeightManager:
    """
    Manager wag sygnałów dla Stealth Engine
    Obsługuje ładowanie, zapisywanie i aktualizację wag na podstawie feedback
    """
    
    def __init__(self, weights_path: str = WEIGHTS_PATH):
        """
        Inicjalizacja managera wag
        
        Args:
            weights_path: Ścieżka do pliku z wagami
        """
        self.weights_path = weights_path
        
        # Zapewnij katalog
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        
        # Załaduj lub utwórz wagi
        if not os.path.exists(self.weights_path):
            save_weights(DEFAULT_WEIGHTS)
    
    def load_weights(self):
        """Załaduj wagi z pliku"""
        return load_weights()
    
    def save_weights(self, weights):
        """Zapisz wagi do pliku"""
        return save_weights(weights)
    
    def update_weight(self, signal_name, delta):
        """Aktualizuj wagę sygnału"""
        return update_weight(signal_name, delta)
    
    def get_weight_stats(self):
        """Pobierz statystyki wag"""
        return get_weight_stats()
    
    def reset_to_defaults(self):
        """Resetuj wagi do wartości domyślnych"""
        return reset_to_defaults()
    
    def batch_update_weights(self, weight_updates):
        """Zaktualizuj wiele wag jednocześnie"""
        return batch_update_weights(weight_updates)