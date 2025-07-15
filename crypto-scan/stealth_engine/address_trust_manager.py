#!/usr/bin/env python3
"""
Stage 6: Address Trust Manager - Feedback Loop dla adresów DEX/Whale
Dynamiczna nauka adresów predykcyjnych

System automatycznie nadaje większe znaczenie tym adresom (DEX i whale), 
które w przeszłości poprzedzały wzrosty. System sam "uczy się", które portfele przewidują ruch.

Logika:
- Jeśli token wzrośnie w ciągu X godzin po aktywności danego adresu, zwiększamy jego "trust score"
- Jeśli token spadnie lub zostanie bez zmian, zmniejszamy jego trust
- Finalnie adresy o wysokim trust wpływają na scoring tokena (+0.05 lub więcej w dex_inflow/whale_ping)
"""

import os
import json
import time
import threading
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class AddressTrustManager:
    """
    Manager do śledzenia skuteczności predykcyjnej adresów
    Implementuje feedback loop dla smart money detection
    """
    
    def __init__(self, cache_file: str = "cache/address_trust_scores.json"):
        self.cache_file = cache_file
        self.lock = threading.Lock()
        
        # Struktura: {address: {"hits": int, "misses": int, "score": float, "total_predictions": int, "last_updated": timestamp}}
        self.address_performance = defaultdict(lambda: {
            "hits": 0,
            "misses": 0, 
            "score": 0.0,
            "total_predictions": 0,
            "last_updated": time.time(),
            "history": []  # Lista ostatnich predykcji
        })
        
        # Konfiguracja
        self.min_predictions_required = 3  # Minimum prób przed przyznaniem trust score
        self.max_history_entries = 20      # Maksymalna liczba historycznych predykcji
        self.trust_decay_days = 30         # Po ilu dniach trust score zaczyna maleć
        self.evaluation_hours = 6          # Okno czasowe dla oceny skuteczności (6h)
        
        self._load_trust_data()
    
    def _load_trust_data(self):
        """Wczytaj dane trust score z cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Konwertuj z JSON do defaultdict
                for address, perf_data in data.items():
                    self.address_performance[address] = perf_data
                        
                print(f"[TRUST MANAGER] Loaded trust data for {len(data)} addresses")
            else:
                print(f"[TRUST MANAGER] No existing trust data found - starting fresh")
                
        except Exception as e:
            print(f"[TRUST MANAGER ERROR] Failed to load trust data: {e}")
            self.address_performance = defaultdict(lambda: {
                "hits": 0, "misses": 0, "score": 0.0, "total_predictions": 0,
                "last_updated": time.time(), "history": []
            })
    
    def _save_trust_data(self):
        """Zapisz dane trust score do cache"""
        try:
            # Sprawdź czy cache_file ma prawidłową ścieżkę
            if not self.cache_file or self.cache_file.strip() == "":
                print(f"[TRUST MANAGER ERROR] Invalid cache file path: '{self.cache_file}'")
                return
                
            # Utwórz katalog jeśli nie istnieje
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir:  # Tylko jeśli nie jest pustym stringiem
                os.makedirs(cache_dir, exist_ok=True)
            
            # Konwertuj defaultdict do regular dict dla JSON
            data = {}
            for address, perf_data in self.address_performance.items():
                data[address] = dict(perf_data)
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"[TRUST MANAGER ERROR] Failed to save trust data: {e}")
    
    def record_address_prediction(self, token: str, address: str, prediction_timestamp: Optional[float] = None):
        """
        Zarejestruj predykcję adresu dla tokena
        
        Args:
            token: Symbol tokena
            address: Adres kryptowalutowy
            prediction_timestamp: Timestamp predykcji (domyślnie: teraz)
        """
        if prediction_timestamp is None:
            prediction_timestamp = time.time()
        
        with self.lock:
            # Dodaj predykcję do historii
            prediction_entry = {
                "token": token,
                "timestamp": prediction_timestamp,
                "evaluation_due": prediction_timestamp + (self.evaluation_hours * 3600),
                "evaluated": False,
                "result": None  # True/False po ocenie
            }
            
            self.address_performance[address]["history"].append(prediction_entry)
            self.address_performance[address]["last_updated"] = prediction_timestamp
            
            # Ogranicz historię do maksymalnej liczby wpisów
            if len(self.address_performance[address]["history"]) > self.max_history_entries:
                self.address_performance[address]["history"] = \
                    self.address_performance[address]["history"][-self.max_history_entries:]
            
            print(f"[TRUST MANAGER] Recorded prediction: {address} → {token} at {datetime.fromtimestamp(prediction_timestamp)}")
    
    def update_address_performance(self, address: str, success: bool, token: str = None):
        """
        Aktualizuj skuteczność adresu na podstawie wyniku predykcji
        
        Args:
            address: Adres kryptowalutowy
            success: Czy predykcja była trafna
            token: Symbol tokena (opcjonalnie)
        """
        with self.lock:
            perf = self.address_performance[address]
            
            if success:
                perf["hits"] += 1
            else:
                perf["misses"] += 1
            
            perf["total_predictions"] += 1
            perf["last_updated"] = time.time()
            
            # Oblicz trust score = skuteczność z minimum 3 prób
            total = perf["hits"] + perf["misses"]
            if total >= self.min_predictions_required:
                perf["score"] = perf["hits"] / total
            else:
                perf["score"] = 0.0  # Brak wystarczających danych
            
            # Zaktualizuj historię jeśli token podany
            if token:
                for entry in perf["history"]:
                    if entry["token"] == token and not entry["evaluated"]:
                        entry["evaluated"] = True
                        entry["result"] = success
                        break
            
            print(f"[TRUST MANAGER] Updated {address}: {perf['hits']}/{total} = {perf['score']:.3f} trust score")
            
            # Zapisz po każdej aktualizacji
            self._save_trust_data()
    
    def get_address_boost(self, address: str) -> float:
        """
        Pobierz boost score dla adresu na podstawie jego trust score
        
        Args:
            address: Adres kryptowalutowy
            
        Returns:
            float: Boost score (0.0 - 0.1)
        """
        with self.lock:
            perf = self.address_performance[address]
            trust_score = perf["score"]
            total_predictions = perf["total_predictions"]
            
            # Sprawdź czy adres ma wystarczającą historię
            if total_predictions < self.min_predictions_required:
                return 0.0
            
            # Sprawdź decay na podstawie ostatniej aktualizacji
            days_since_update = (time.time() - perf["last_updated"]) / 86400
            if days_since_update > self.trust_decay_days:
                decay_factor = max(0.1, 1.0 - (days_since_update - self.trust_decay_days) / 30)
                trust_score *= decay_factor
            
            # Konwertuj trust score na boost
            if trust_score >= 0.8:
                return 0.10  # Bardzo wysokie zaufanie
            elif trust_score >= 0.7:
                return 0.05  # Wysokie zaufanie
            elif trust_score >= 0.6:
                return 0.03  # Średnie zaufanie
            elif trust_score >= 0.5:
                return 0.02  # Niskie zaufanie
            else:
                return 0.0   # Brak zaufania
    
    def get_trust_statistics(self, address: str = None) -> Dict:
        """
        🔧 TRIGGER TIMEOUT FIX: Pobierz statystyki trust score z NO-LOCK CACHE optimization
        
        Args:
            address: Konkretny adres lub None dla wszystkich
            
        Returns:
            Dict ze statystykami
        """
        print(f"[TRUST STATS] Called get_trust_statistics for {address[:12] if address else 'ALL'}...")
        
        # 🔧 TRIGGER TIMEOUT FIX: Use lockless cache read for performance
        try:
            if address:
                # Fast cache lookup without lock for individual addresses
                if address not in self.address_performance:
                    print(f"[TRUST STATS FAST] Address {address[:12]}... not found in cache - returning default")
                    return {
                        'address': address,
                        'hits': 0,
                        'misses': 0,
                        'total_predictions': 0,
                        'trust_score': 0.0,
                        'boost_value': 0.0,
                        'last_updated': None,
                        'cache_source': 'not_found'
                    }
                
                # LOCKLESS READ: Direct cache access for performance  
                perf = self.address_performance[address]
                boost_value = self.get_address_boost(address)
                
                print(f"[TRUST STATS FAST] Address {address[:12]}... found - trust: {perf['score']:.3f}, predictions: {perf['total_predictions']}")
                return {
                    'address': address,
                    'hits': perf['hits'],
                    'misses': perf['misses'],
                    'total_predictions': perf['total_predictions'],
                    'trust_score': perf['score'],
                    'boost_value': boost_value,
                    'last_updated': datetime.fromtimestamp(perf['last_updated']).isoformat(),
                    'recent_history': perf['history'][-5:],  # Ostatnie 5 predykcji
                    'cache_source': 'direct_access'
                }
            else:
                # Global stats - still need minimal lock but much faster
                try:
                    lock_acquired = self.lock.acquire(timeout=0.05)  # Very short timeout for global stats
                    if not lock_acquired:
                        print(f"[TRUST STATS EMERGENCY] Global stats lock timeout - using emergency fallback")
                        return {
                            'total_addresses': 0,
                            'trusted_addresses': 0,
                            'trust_ratio': 0.0,
                            'top_trusted': [],
                            'fallback_reason': 'global_lock_timeout'
                        }
                    
                    # Fast global calculation
                    total_addresses = len(self.address_performance)
                    trusted_addresses = sum(1 for perf in self.address_performance.values() 
                                          if perf['score'] >= 0.5 and perf['total_predictions'] >= self.min_predictions_required)
                    
                    print(f"[TRUST STATS FAST] Global stats - total: {total_addresses}, trusted: {trusted_addresses}")
                    return {
                        'total_addresses': total_addresses,
                        'trusted_addresses': trusted_addresses,
                        'trust_ratio': trusted_addresses / total_addresses if total_addresses > 0 else 0.0,
                        'top_trusted': [],  # Skip heavy calculation for performance
                        'cache_source': 'fast_global'
                    }
                finally:
                    if lock_acquired:
                        self.lock.release()
                        print(f"[TRUST STATS] Released global lock")
                
        except Exception as e:
            print(f"[TRUST STATS ERROR] Cache access failed for {address[:12] if address else 'ALL'}: {e}")
            # Super fast emergency fallback
            if address:
                return {
                    'address': address,
                    'hits': 0, 'misses': 0, 'total_predictions': 0,
                    'trust_score': 0.0, 'boost_value': 0.0, 'last_updated': None,
                    'fallback_reason': 'cache_error'
                }
            else:
                return {
                    'total_addresses': 0, 'trusted_addresses': 0, 'trust_ratio': 0.0,
                    'top_trusted': [], 'fallback_reason': 'cache_error'
                }
    
    def evaluate_pending_predictions(self, price_fetcher_callback=None):
        """
        Oceń oczekujące predykcje na podstawie zmian cen
        
        Args:
            price_fetcher_callback: Funkcja do pobierania cen (token, timestamp) -> price
        """
        current_time = time.time()
        evaluated_count = 0
        
        with self.lock:
            for address, perf in self.address_performance.items():
                for entry in perf["history"]:
                    # Sprawdź czy predykcja jest gotowa do oceny
                    if (not entry["evaluated"] and 
                        current_time >= entry["evaluation_due"]):
                        
                        if price_fetcher_callback:
                            try:
                                # Pobierz cenę z momentu predykcji i po evaluation_hours
                                price_before = price_fetcher_callback(entry["token"], entry["timestamp"])
                                price_after = price_fetcher_callback(entry["token"], entry["evaluation_due"])
                                
                                if price_before and price_after:
                                    price_change = (price_after - price_before) / price_before
                                    success = price_change >= 0.02  # Wzrost o minimum 2%
                                    
                                    self.update_address_performance(address, success, entry["token"])
                                    evaluated_count += 1
                                    
                                    print(f"[TRUST EVALUATION] {address} → {entry['token']}: {price_change:.2%} change = {'SUCCESS' if success else 'MISS'}")
                                
                            except Exception as e:
                                print(f"[TRUST EVALUATION ERROR] Failed to evaluate {address} → {entry['token']}: {e}")
                        else:
                            # Bez price fetcher - oznacz jako ocenione ale nie aktualizuj score
                            entry["evaluated"] = True
                            evaluated_count += 1
        
        if evaluated_count > 0:
            print(f"[TRUST EVALUATION] Evaluated {evaluated_count} pending predictions")
            self._save_trust_data()
        
        return evaluated_count
    
    def cleanup_old_data(self):
        """Oczyść stare dane trust score"""
        cutoff_time = time.time() - (self.trust_decay_days * 2 * 86400)  # 2x decay period
        removed_addresses = 0
        
        with self.lock:
            addresses_to_remove = []
            
            for address, perf in self.address_performance.items():
                # Usuń adresy które nie były aktualizowane przez długi czas
                if perf["last_updated"] < cutoff_time:
                    addresses_to_remove.append(address)
                else:
                    # Oczyść starą historię
                    perf["history"] = [entry for entry in perf["history"] 
                                     if entry["timestamp"] > cutoff_time]
            
            # Usuń nieaktywne adresy
            for address in addresses_to_remove:
                del self.address_performance[address]
                removed_addresses += 1
        
        # Zapisz po oczyszczeniu
        self._save_trust_data()
        
        print(f"[TRUST CLEANUP] Removed {removed_addresses} old addresses")
        return removed_addresses

# Globalny instancja managera
_global_trust_manager = None

def get_trust_manager() -> AddressTrustManager:
    """Pobierz globalną instancję trust managera"""
    global _global_trust_manager
    if _global_trust_manager is None:
        _global_trust_manager = AddressTrustManager()
    return _global_trust_manager

def record_address_prediction(token: str, address: str, timestamp: Optional[float] = None):
    """Convenience function - zarejestruj predykcję adresu"""
    manager = get_trust_manager()
    manager.record_address_prediction(token, address, timestamp)

def update_address_performance(address: str, success: bool, token: str = None):
    """Convenience function - aktualizuj skuteczność adresu"""
    manager = get_trust_manager()
    manager.update_address_performance(address, success, token)

def get_address_boost(address: str) -> float:
    """Convenience function - pobierz boost dla adresu"""
    manager = get_trust_manager()
    return manager.get_address_boost(address)

def get_trust_statistics(address: str = None) -> Dict:
    """Convenience function - pobierz statystyki trust"""
    manager = get_trust_manager()
    return manager.get_trust_statistics(address)

def evaluate_pending_predictions(price_fetcher_callback=None):
    """Convenience function - oceń oczekujące predykcje"""
    manager = get_trust_manager()
    return manager.evaluate_pending_predictions(price_fetcher_callback)

def cleanup_trust_data():
    """Convenience function - oczyść stare dane"""
    manager = get_trust_manager()
    return manager.cleanup_old_data()