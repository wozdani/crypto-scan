#!/usr/bin/env python3
"""
Multi-Address Repeat Detection System - Stage 5
Wykrywa skoordynowaną aktywność grup adresów na tym samym tokenie

Logika:
- Jeśli 3+ różnych adresów pojawia się w transakcjach na ten sam token więcej niż X razy w 7 dni
- Traktuje to jako "koordynowaną aktywność" lub wzór akumulacji
- Może być oznaka działania większej grupy (fundusz, market maker, whale consortium)

Struktura danych:
token_to_active_addresses[token] = {"addr1": [ts1, ts2], "addr2": [ts3], ...}
"""

import os
import json
import time
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class MultiAddressDetector:
    """
    Detektor grupowej aktywności adresów
    Wykrywa skoordynowane wzorce akumulacji przez różne adresy
    """
    
    def __init__(self, cache_file: str = "cache/multi_address_groups.json"):
        self.cache_file = cache_file
        self.lock = threading.Lock()
        
        # Struktura: {token: {address: [timestamp1, timestamp2, ...]}}
        self.token_to_active_addresses = defaultdict(lambda: defaultdict(list))
        
        # Konfiguracja
        self.default_time_window_hours = 72  # 3 dni
        self.min_unique_addresses = 3  # Minimum 3 różnych adresów
        self.min_total_events = 3      # Minimum 3 zdarzeń w oknie czasowym (po 1 na adres)
        self.max_history_days = 7      # Maksymalnie 7 dni historii
        
        self._load_group_data()
    
    def _load_group_data(self):
        """Wczytaj dane grup z cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Konwertuj z JSON do defaultdict
                for token, addresses in data.items():
                    for address, timestamps in addresses.items():
                        self.token_to_active_addresses[token][address] = timestamps
                        
                print(f"[MULTI-ADDRESS] Loaded group data for {len(data)} tokens")
            else:
                print(f"[MULTI-ADDRESS] No existing group data found - starting fresh")
                
        except Exception as e:
            print(f"[MULTI-ADDRESS ERROR] Failed to load group data: {e}")
            self.token_to_active_addresses = defaultdict(lambda: defaultdict(list))
    
    def _save_group_data(self):
        """Zapisz dane grup do cache"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            # Konwertuj defaultdict do regular dict dla JSON
            data = {}
            for token, addresses in self.token_to_active_addresses.items():
                data[token] = dict(addresses)
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"[MULTI-ADDRESS ERROR] Failed to save group data: {e}")
    
    def record_address_activity(self, token: str, address: str, timestamp: Optional[float] = None):
        """
        Zarejestruj aktywność adresu dla tokena
        
        Args:
            token: Symbol tokena
            address: Adres kryptowalutowy
            timestamp: Timestamp aktywności (domyślnie: teraz)
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            # Dodaj nowy timestamp
            self.token_to_active_addresses[token][address].append(timestamp)
            
            # Oczyść stare wpisy (starsze niż max_history_days)
            cutoff_time = timestamp - (self.max_history_days * 86400)
            
            for addr in list(self.token_to_active_addresses[token].keys()):
                # Filtruj stare timestampy
                recent_timestamps = [ts for ts in self.token_to_active_addresses[token][addr] 
                                   if ts > cutoff_time]
                
                if recent_timestamps:
                    self.token_to_active_addresses[token][addr] = recent_timestamps
                else:
                    # Usuń adres jeśli nie ma aktywności
                    del self.token_to_active_addresses[token][addr]
            
            # Usuń token jeśli nie ma aktywnych adresów
            if not self.token_to_active_addresses[token]:
                del self.token_to_active_addresses[token]
    
    def detect_address_group_activity(self, token: str, 
                                    time_window_hours: int = None,
                                    min_unique_addresses: int = None,
                                    min_total_events: int = None) -> Tuple[bool, int, int, float]:
        """
        Wykryj grupową aktywność adresów dla tokena
        
        Args:
            token: Symbol tokena
            time_window_hours: Okno czasowe w godzinach (domyślnie: 72h)
            min_unique_addresses: Min liczba unikalnych adresów (domyślnie: 3)
            min_total_events: Min liczba zdarzeń (domyślnie: 5)
            
        Returns:
            Tuple[active_group, unique_addresses, total_events, group_intensity]
        """
        if time_window_hours is None:
            time_window_hours = self.default_time_window_hours
        if min_unique_addresses is None:
            min_unique_addresses = self.min_unique_addresses
        if min_total_events is None:
            min_total_events = self.min_total_events
        
        now = time.time()
        cutoff_time = now - (time_window_hours * 3600)
        
        total_events = 0
        unique_addresses = 0
        
        with self.lock:
            if token not in self.token_to_active_addresses:
                return False, 0, 0, 0.0
            
            for address, timestamps in self.token_to_active_addresses[token].items():
                # Zlicz niedawne aktywności
                recent_events = [ts for ts in timestamps if ts > cutoff_time]
                
                if recent_events:
                    total_events += len(recent_events)
                    unique_addresses += 1
        
        # Sprawdź czy spełnia kryteria grupowej aktywności
        active_group = (unique_addresses >= min_unique_addresses and 
                       total_events >= min_total_events)
        
        # Oblicz intensywność grupy (0.0 - 1.0)
        # Im więcej adresów i zdarzeń, tym wyższa intensywność
        group_intensity = 0.0
        if active_group:
            # Bazowa intensywność na podstawie liczby adresów
            address_factor = min(unique_addresses / 10.0, 1.0)  # Max przy 10+ adresach
            
            # Bonus za częstotliwość zdarzeń
            events_per_address = total_events / unique_addresses if unique_addresses > 0 else 0
            frequency_factor = min(events_per_address / 5.0, 1.0)  # Max przy 5+ zdarzeniach/adres
            
            # Kombinowana intensywność
            group_intensity = (address_factor * 0.6 + frequency_factor * 0.4)
            group_intensity = min(group_intensity, 1.0)
        
        return active_group, unique_addresses, total_events, group_intensity
    
    def get_group_statistics(self, token: str = None) -> Dict:
        """
        Pobierz statystyki grup adresów
        
        Args:
            token: Konkretny token lub None dla wszystkich
            
        Returns:
            Dict ze statystykami
        """
        with self.lock:
            if token:
                # Statystyki dla konkretnego tokena
                if token not in self.token_to_active_addresses:
                    return {
                        'token': token,
                        'unique_addresses': 0,
                        'total_events': 0,
                        'group_active': False,
                        'group_intensity': 0.0
                    }
                
                active_group, unique_addresses, total_events, intensity = self.detect_address_group_activity(token)
                
                return {
                    'token': token,
                    'unique_addresses': unique_addresses,
                    'total_events': total_events,
                    'group_active': active_group,
                    'group_intensity': intensity,
                    'addresses': list(self.token_to_active_addresses[token].keys())
                }
            else:
                # Statystyki globalne
                total_tokens = len(self.token_to_active_addresses)
                total_addresses = sum(len(addresses) for addresses in self.token_to_active_addresses.values())
                active_groups = 0
                
                group_tokens = []
                for token_symbol in self.token_to_active_addresses.keys():
                    active_group, unique_addr, total_events, intensity = self.detect_address_group_activity(token_symbol)
                    if active_group:
                        active_groups += 1
                        group_tokens.append({
                            'token': token_symbol,
                            'addresses': unique_addr,
                            'events': total_events,
                            'intensity': intensity
                        })
                
                # Sortuj według intensywności
                group_tokens.sort(key=lambda x: x['intensity'], reverse=True)
                
                return {
                    'total_tokens': total_tokens,
                    'total_addresses': total_addresses,
                    'active_groups': active_groups,
                    'top_groups': group_tokens[:10]  # Top 10 najaktywniejszych grup
                }
    
    def cleanup_old_data(self):
        """Oczyść stare dane (starsze niż max_history_days)"""
        cutoff_time = time.time() - (self.max_history_days * 86400)
        removed_addresses = 0
        removed_tokens = 0
        
        with self.lock:
            tokens_to_remove = []
            
            for token in list(self.token_to_active_addresses.keys()):
                addresses_to_remove = []
                
                for address in list(self.token_to_active_addresses[token].keys()):
                    # Filtruj stare timestampy
                    recent_timestamps = [ts for ts in self.token_to_active_addresses[token][address] 
                                       if ts > cutoff_time]
                    
                    if recent_timestamps:
                        self.token_to_active_addresses[token][address] = recent_timestamps
                    else:
                        addresses_to_remove.append(address)
                
                # Usuń nieaktywne adresy
                for address in addresses_to_remove:
                    del self.token_to_active_addresses[token][address]
                    removed_addresses += 1
                
                # Usuń token jeśli nie ma aktywnych adresów
                if not self.token_to_active_addresses[token]:
                    tokens_to_remove.append(token)
            
            # Usuń nieaktywne tokeny
            for token in tokens_to_remove:
                del self.token_to_active_addresses[token]
                removed_tokens += 1
        
        # Zapisz po oczyszczeniu
        self._save_group_data()
        
        print(f"[MULTI-ADDRESS CLEANUP] Removed {removed_addresses} addresses and {removed_tokens} tokens")
        return removed_addresses, removed_tokens
    
    def save_data(self):
        """Zapisz dane do cache"""
        self._save_group_data()

# Globalny instancja detektora
_global_detector = None

def get_multi_address_detector() -> MultiAddressDetector:
    """Pobierz globalną instancję detektora"""
    global _global_detector
    if _global_detector is None:
        _global_detector = MultiAddressDetector()
    return _global_detector

def record_address_activity(token: str, address: str, timestamp: Optional[float] = None):
    """Convenience function - zarejestruj aktywność adresu"""
    detector = get_multi_address_detector()
    detector.record_address_activity(token, address, timestamp)

def detect_group_activity(token: str) -> Tuple[bool, int, int, float]:
    """Convenience function - wykryj grupową aktywność"""
    detector = get_multi_address_detector()
    return detector.detect_address_group_activity(token)

def get_group_statistics(token: str = None) -> Dict:
    """Convenience function - pobierz statystyki grup"""
    detector = get_multi_address_detector()
    return detector.get_group_statistics(token)

def cleanup_group_data():
    """Convenience function - oczyść stare dane"""
    detector = get_multi_address_detector()
    return detector.cleanup_old_data()