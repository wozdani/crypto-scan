"""
Stealth Weight Manager
Loader i saver wag sygnałów dla Stealth Engine

Zarządza wagami sygnałów w pliku JSON i zapewnia
domyślne wartości przy pierwszym uruchomieniu
"""

import json
import os
from typing import Dict, Optional
import time


class StealthWeightManager:
    """
    Zarządca wag sygnałów Stealth Engine
    Ładuje, zapisuje i zarządza wagami w pliku JSON
    """
    
    def __init__(self, config_path: str = "crypto-scan/cache"):
        """
        Inicjalizacja menagera wag
        
        Args:
            config_path: Ścieżka do katalogu z konfiguracją
        """
        self.config_path = config_path
        self.weights_file = os.path.join(config_path, "stealth_weights.json")
        self.backup_file = os.path.join(config_path, "stealth_weights_backup.json")
        
        # Domyślne wagi sygnałów
        self.default_weights = {
            # ORDERBOOK SIGNALS
            'orderbook_imbalance': 0.15,
            'large_bid_walls': 0.12,
            'ask_wall_removal': 0.18,
            'spoofing_detected': -0.25,  # Negatywny sygnał
            
            # VOLUME SIGNALS
            'volume_spike': 0.20,
            'volume_accumulation': 0.14,
            'unusual_volume_profile': 0.16,
            
            # DEX SIGNALS
            'dex_inflow_spike': 0.22,
            'whale_accumulation': 0.19,
            
            # MICROSTRUCTURE SIGNALS
            'bid_ask_spread_tightening': 0.10,
            'order_flow_pressure': 0.13,
            'liquidity_absorption': 0.17
        }
        
        self.ensure_directory()
    
    def ensure_directory(self):
        """Upewnij się że katalog konfiguracji istnieje"""
        os.makedirs(self.config_path, exist_ok=True)
    
    def load_weights(self) -> Dict[str, float]:
        """
        Załaduj wagi sygnałów z pliku JSON
        
        Returns:
            Słownik wag sygnałów
        """
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Sprawdź strukturę pliku
                if isinstance(data, dict) and 'weights' in data:
                    weights = data['weights']
                elif isinstance(data, dict):
                    weights = data  # Backward compatibility
                else:
                    weights = {}
                
                # Sprawdź czy wszystkie sygnały mają wagi
                updated = False
                for signal_name, default_weight in self.default_weights.items():
                    if signal_name not in weights:
                        weights[signal_name] = default_weight
                        updated = True
                        print(f"[STEALTH WEIGHTS] Added missing weight: {signal_name} = {default_weight}")
                
                # Zapisz jeśli dodano nowe wagi
                if updated:
                    self.save_weights(weights)
                
                print(f"[STEALTH WEIGHTS] Loaded {len(weights)} weights from {self.weights_file}")
                return weights
            else:
                # Pierwszego uruchomienia - utwórz plik z domyślnymi wagami
                print(f"[STEALTH WEIGHTS] Creating default weights file: {self.weights_file}")
                self.save_weights(self.default_weights)
                return self.default_weights.copy()
                
        except Exception as e:
            print(f"[STEALTH WEIGHTS ERROR] Failed to load weights: {e}")
            print(f"[STEALTH WEIGHTS] Using default weights")
            return self.default_weights.copy()
    
    def save_weights(self, weights: Dict[str, float]) -> bool:
        """
        Zapisz wagi sygnałów do pliku JSON
        
        Args:
            weights: Słownik wag do zapisu
            
        Returns:
            True jeśli zapis powiódł się
        """
        try:
            # Utwórz backup istniejącego pliku
            if os.path.exists(self.weights_file):
                import shutil
                shutil.copy2(self.weights_file, self.backup_file)
            
            # Przygotuj dane do zapisu
            weights_data = {
                'weights': weights,
                'last_updated': time.time(),
                'last_updated_human': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                'total_signals': len(weights),
                'positive_weights': sum(1 for w in weights.values() if w > 0),
                'negative_weights': sum(1 for w in weights.values() if w < 0),
                'zero_weights': sum(1 for w in weights.values() if w == 0)
            }
            
            # Zapisz do pliku
            with open(self.weights_file, 'w', encoding='utf-8') as f:
                json.dump(weights_data, f, indent=2, ensure_ascii=False)
            
            print(f"[STEALTH WEIGHTS] Saved {len(weights)} weights to {self.weights_file}")
            return True
            
        except Exception as e:
            print(f"[STEALTH WEIGHTS ERROR] Failed to save weights: {e}")
            return False
    
    def get_weight(self, signal_name: str) -> float:
        """
        Pobierz wagę dla konkretnego sygnału
        
        Args:
            signal_name: Nazwa sygnału
            
        Returns:
            Waga sygnału lub 0.0 jeśli nie istnieje
        """
        weights = self.load_weights()
        return weights.get(signal_name, 0.0)
    
    def update_weight(self, signal_name: str, new_weight: float) -> bool:
        """
        Aktualizuj wagę pojedynczego sygnału
        
        Args:
            signal_name: Nazwa sygnału
            new_weight: Nowa waga
            
        Returns:
            True jeśli aktualizacja powiodła się
        """
        try:
            weights = self.load_weights()
            old_weight = weights.get(signal_name, 0.0)
            weights[signal_name] = new_weight
            
            success = self.save_weights(weights)
            
            if success:
                print(f"[STEALTH WEIGHTS] Updated {signal_name}: {old_weight:.3f} → {new_weight:.3f}")
            
            return success
            
        except Exception as e:
            print(f"[STEALTH WEIGHTS ERROR] Failed to update {signal_name}: {e}")
            return False
    
    def update_weights_batch(self, weight_updates: Dict[str, float]) -> bool:
        """
        Aktualizuj wiele wag jednocześnie
        
        Args:
            weight_updates: Słownik aktualizacji {signal_name: new_weight}
            
        Returns:
            True jeśli aktualizacja powiodła się
        """
        try:
            weights = self.load_weights()
            
            # Zastosuj aktualizacje
            updated_signals = []
            for signal_name, new_weight in weight_updates.items():
                old_weight = weights.get(signal_name, 0.0)
                weights[signal_name] = new_weight
                updated_signals.append(f"{signal_name}: {old_weight:.3f}→{new_weight:.3f}")
            
            success = self.save_weights(weights)
            
            if success:
                print(f"[STEALTH WEIGHTS] Batch update: {len(weight_updates)} weights")
                for update_info in updated_signals:
                    print(f"[STEALTH WEIGHTS]   {update_info}")
            
            return success
            
        except Exception as e:
            print(f"[STEALTH WEIGHTS ERROR] Failed batch update: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """
        Zresetuj wszystkie wagi do wartości domyślnych
        
        Returns:
            True jeśli reset się powiódł
        """
        try:
            success = self.save_weights(self.default_weights.copy())
            
            if success:
                print(f"[STEALTH WEIGHTS] Reset to defaults: {len(self.default_weights)} weights")
            
            return success
            
        except Exception as e:
            print(f"[STEALTH WEIGHTS ERROR] Failed to reset: {e}")
            return False
    
    def get_weights_summary(self) -> Dict:
        """
        Pobierz podsumowanie wag
        
        Returns:
            Słownik ze statystykami wag
        """
        try:
            weights = self.load_weights()
            
            weight_values = list(weights.values())
            positive_weights = {k: v for k, v in weights.items() if v > 0}
            negative_weights = {k: v for k, v in weights.items() if v < 0}
            zero_weights = {k: v for k, v in weights.items() if v == 0}
            
            summary = {
                'total_signals': len(weights),
                'positive_weights': len(positive_weights),
                'negative_weights': len(negative_weights),
                'zero_weights': len(zero_weights),
                'max_weight': max(weight_values) if weight_values else 0,
                'min_weight': min(weight_values) if weight_values else 0,
                'avg_weight': sum(weight_values) / len(weight_values) if weight_values else 0,
                'weights_file': self.weights_file,
                'backup_exists': os.path.exists(self.backup_file)
            }
            
            return summary
            
        except Exception as e:
            print(f"[STEALTH WEIGHTS ERROR] Failed to get summary: {e}")
            return {}
    
    def validate_weights(self) -> Dict:
        """
        Zwaliduj integralność pliku wag
        
        Returns:
            Słownik z wynikami walidacji
        """
        validation = {
            'file_exists': os.path.exists(self.weights_file),
            'file_readable': False,
            'valid_json': False,
            'has_weights': False,
            'missing_signals': [],
            'extra_signals': [],
            'invalid_weights': []
        }
        
        try:
            # Sprawdź czy plik istnieje i jest czytelny
            if validation['file_exists']:
                with open(self.weights_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                validation['file_readable'] = True
                validation['valid_json'] = True
                
                # Sprawdź strukturę
                if 'weights' in data and isinstance(data['weights'], dict):
                    weights = data['weights']
                    validation['has_weights'] = True
                    
                    # Sprawdź brakujące sygnały
                    for signal_name in self.default_weights:
                        if signal_name not in weights:
                            validation['missing_signals'].append(signal_name)
                    
                    # Sprawdź dodatkowe sygnały
                    for signal_name in weights:
                        if signal_name not in self.default_weights:
                            validation['extra_signals'].append(signal_name)
                    
                    # Sprawdź niepoprawne wagi
                    for signal_name, weight in weights.items():
                        if not isinstance(weight, (int, float)):
                            validation['invalid_weights'].append(f"{signal_name}: {type(weight)}")
                        elif abs(weight) > 1.0:  # Wagi >1.0 mogą być problematyczne
                            validation['invalid_weights'].append(f"{signal_name}: {weight} (>1.0)")
        
        except json.JSONDecodeError:
            validation['valid_json'] = False
        except Exception as e:
            validation['error'] = str(e)
        
        return validation