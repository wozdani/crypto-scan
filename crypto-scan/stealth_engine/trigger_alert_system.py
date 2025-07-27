#!/usr/bin/env python3
"""
🚨 Stage 7: Trigger Alert Boost System - Enhanced Smart Money Detection
🎯 Natychmiastowe wyzwalanie alertów przy wykryciu "smart money" z obniżonymi progami dla lepszego detection

📦 Funkcjonalności:
1. Detekcja trigger addresses (trust score ≥0.3 + min 1 predykcja) - ENHANCED SENSITIVITY
2. Automatyczne podniesienie scoring do poziomu alertu (min 3.0)
3. Priorytetowe dodanie do kolejki alertów
4. Bypass standardowych filtrów dla zaufanych adresów
5. Instant alert generation bez czekania na inne sygnały
"""

import time
import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class TriggerAlertSystem:
    """
    System natychmiastowych alertów dla zaufanych adresów
    """
    
    def __init__(self, 
                 trust_threshold: float = 0.0,
                 min_predictions: int = 0,
                 trigger_score: float = 3.0,
                 cache_file: str = "cache/trigger_alerts.json"):
        """
        Inicjalizacja systemu trigger alerts
        
        Args:
            trust_threshold: Minimalny trust score dla trigger (default: 0.0 = 0% - ALL ADDRESSES)
            min_predictions: Minimalna liczba predykcji dla trigger (default: 0 - ALL ADDRESSES)
            trigger_score: Score ustawiony przy trigger alert (default: 3.0)
            cache_file: Plik cache dla trigger alerts
        """
        self.trust_threshold = trust_threshold
        self.min_predictions = min_predictions
        self.trigger_score = trigger_score
        self.cache_file = cache_file
        
        # Statystyki
        self.trigger_count = 0
        self.triggered_tokens = []
        self.trigger_history = []
        
        # Load existing data
        self._load_trigger_cache()
    
    def _load_trigger_cache(self):
        """Wczytaj cache trigger alerts"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.trigger_count = data.get('trigger_count', 0)
                    self.triggered_tokens = data.get('triggered_tokens', [])
                    self.trigger_history = data.get('trigger_history', [])
        except Exception as e:
            print(f"[TRIGGER ALERT] Error loading cache: {e}")
    
    def _save_trigger_cache(self):
        """Zapisz cache trigger alerts"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            data = {
                'trigger_count': self.trigger_count,
                'triggered_tokens': self.triggered_tokens,
                'trigger_history': self.trigger_history,
                'last_updated': time.time()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[TRIGGER ALERT] Error saving cache: {e}")
    
    def check_trigger_addresses(self, detected_addresses: List[str], 
                               address_trust_manager) -> Tuple[bool, List[Dict]]:
        """
        Sprawdź czy wykryte adresy to trigger addresses (smart money)
        
        Args:
            detected_addresses: Lista wykrytych adresów
            address_trust_manager: Instance AddressTrustManager
            
        Returns:
            Tuple[bool, List[Dict]]: (czy_trigger, lista_trigger_addresses)
        """
        print(f"[SMART MONEY CHECK] Starting check for {len(detected_addresses)} addresses...")
        print(f"[SMART MONEY CHECK] Thresholds: trust≥{self.trust_threshold:.1%}, predictions≥{self.min_predictions}")
        trigger_addresses = []
        
        for i, addr in enumerate(detected_addresses):
            try:
                print(f"[TRIGGER DEBUG] Processing address {i+1}/{len(detected_addresses)}: {addr[:12]}...")
                
                # EMERGENCY TIMEOUT PROTECTION: Prevent hanging in get_trust_statistics
                try:
                    print(f"[TRIGGER DEBUG] Calling get_trust_statistics for {addr[:12]} with emergency timeout...")
                    
                    # Import signal for timeout
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("get_trust_statistics emergency timeout")
                    
                    # Set 1-second emergency timeout
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(1)
                    
                    try:
                        stats = address_trust_manager.get_trust_statistics(addr)
                        signal.alarm(0)  # Cancel timeout
                        print(f"[TRIGGER DEBUG] Got stats for {addr[:12]}: trust={stats.get('trust_score', 0):.3f}")
                        
                        # Sprawdź czy spełnia kryteria trigger
                        if (stats['trust_score'] >= self.trust_threshold and 
                            stats['total_predictions'] >= self.min_predictions):
                            
                            trigger_info = {
                                'address': addr,
                                'trust_score': stats['trust_score'],
                                'total_predictions': stats['total_predictions'],
                                'hits': stats['hits'],
                                'boost_value': stats['boost_value'],
                                'trigger_timestamp': time.time()
                            }
                            
                            trigger_addresses.append(trigger_info)
                            
                            print(f"[TRIGGER ALERT] 🚨 Smart money detected: {addr[:12]}... "
                                  f"(trust={stats['trust_score']:.1%}, {stats['hits']}/{stats['total_predictions']})")
                        else:
                            print(f"[TRIGGER DEBUG] Address {addr[:12]}... doesn't meet criteria: trust={stats['trust_score']:.3f} < {self.trust_threshold} or predictions={stats['total_predictions']} < {self.min_predictions}")
                    
                    except TimeoutError:
                        signal.alarm(0)  # Cancel timeout
                        print(f"[TRIGGER EMERGENCY] TIMEOUT for {addr[:12]}... - using emergency fallback (no trigger)")
                        continue
                        
                except Exception as stats_e:
                    print(f"[TRIGGER ERROR] get_trust_statistics failed for {addr[:12]}...: {stats_e}")
                    continue
                
            except Exception as addr_e:
                print(f"[TRIGGER ERROR] Failed processing address {addr[:12]}...: {addr_e}")
                continue
        
        print(f"[TRIGGER DEBUG] check_trigger_addresses completed: {len(trigger_addresses)} trigger addresses found")
        return len(trigger_addresses) > 0, trigger_addresses
    
    def apply_trigger_boost(self, symbol: str, base_score: float, 
                           trigger_addresses: List[Dict],
                           detection_source: str = "unknown") -> Tuple[float, bool]:
        """
        Zastosuj trigger boost do scoring tokena
        
        Args:
            symbol: Symbol tokena
            base_score: Bazowy score
            trigger_addresses: Lista trigger addresses
            detection_source: Źródło detekcji (whale_ping, dex_inflow)
            
        Returns:
            Tuple[float, bool]: (boosted_score, priority_alert_flag)
        """
        if not trigger_addresses:
            return base_score, False
        
        # Oblicz najwyższy trust score z trigger addresses
        max_trust = max(addr['trust_score'] for addr in trigger_addresses)
        
        # Ustaw score na poziom alertu lub wyżej
        boosted_score = max(base_score, self.trigger_score)
        
        # Dodatkowy boost za bardzo wysokie zaufanie (>90%)
        if max_trust >= 0.9:
            boosted_score += 0.5  # Extra boost za exceptional trust
        
        # Zapisz trigger event
        trigger_event = {
            'symbol': symbol,
            'timestamp': time.time(),
            'base_score': base_score,
            'boosted_score': boosted_score,
            'max_trust': max_trust,
            'trigger_addresses': trigger_addresses,
            'detection_source': detection_source
        }
        
        self.trigger_history.append(trigger_event)
        self.trigger_count += 1
        
        if symbol not in self.triggered_tokens:
            self.triggered_tokens.append(symbol)
        
        # Zachowaj tylko ostatnie 100 wydarzeń
        if len(self.trigger_history) > 100:
            self.trigger_history = self.trigger_history[-100:]
        
        # Zapisz do cache
        self._save_trigger_cache()
        
        print(f"[TRIGGER BOOST] 🚀 {symbol}: {base_score:.3f} → {boosted_score:.3f} "
              f"(trust={max_trust:.1%}, source={detection_source})")
        
        return boosted_score, True
    
    def should_bypass_filters(self, trigger_addresses: List[Dict]) -> bool:
        """
        Sprawdź czy trigger addresses powinny pominąć standardowe filtry
        
        Args:
            trigger_addresses: Lista trigger addresses
            
        Returns:
            bool: Czy pominąć filtry
        """
        if not trigger_addresses:
            return False
        
        # Bypass filters for highly trusted addresses (≥90%)
        max_trust = max(addr['trust_score'] for addr in trigger_addresses)
        
        return max_trust >= 0.9
    
    def create_priority_alert(self, symbol: str, trigger_addresses: List[Dict],
                             score: float, detection_source: str) -> Dict:
        """
        Utwórz priorytetowy alert dla trigger event
        
        Args:
            symbol: Symbol tokena
            trigger_addresses: Lista trigger addresses
            score: Score po boost
            detection_source: Źródło detekcji
            
        Returns:
            Dict: Alert do kolejki priorytetowej
        """
        max_trust_addr = max(trigger_addresses, key=lambda x: x['trust_score'])
        
        alert = {
            'symbol': symbol,
            'type': 'TRIGGER_ALERT',
            'priority': 'HIGH',
            'score': score,
            'timestamp': time.time(),
            'detection_source': detection_source,
            'trigger_reason': f"Smart money activity (trust={max_trust_addr['trust_score']:.1%})",
            'trusted_address': max_trust_addr['address'][:12] + "...",
            'trust_score': max_trust_addr['trust_score'],
            'address_history': f"{max_trust_addr['hits']}/{max_trust_addr['total_predictions']}",
            'bypass_filters': self.should_bypass_filters(trigger_addresses)
        }
        
        return alert
    
    def get_trigger_statistics(self) -> Dict:
        """
        Pobierz statystyki trigger alert system
        
        Returns:
            Dict: Statystyki systemu
        """
        recent_triggers = [event for event in self.trigger_history 
                          if time.time() - event['timestamp'] < 86400]  # Ostatnie 24h
        
        return {
            'total_triggers': self.trigger_count,
            'triggered_tokens': len(self.triggered_tokens),
            'recent_triggers_24h': len(recent_triggers),
            'trust_threshold': self.trust_threshold,
            'trigger_score': self.trigger_score,
            'recent_events': self.trigger_history[-10:],  # Ostatnie 10 wydarzeń
            'top_triggered_symbols': list(set([event['symbol'] for event in recent_triggers]))
        }
    
    def cleanup_old_triggers(self, days: int = 7):
        """
        Oczyść stare trigger events
        
        Args:
            days: Liczba dni do zachowania
        """
        cutoff_time = time.time() - (days * 86400)
        
        old_count = len(self.trigger_history)
        self.trigger_history = [event for event in self.trigger_history 
                               if event['timestamp'] > cutoff_time]
        
        removed = old_count - len(self.trigger_history)
        if removed > 0:
            print(f"[TRIGGER CLEANUP] Removed {removed} old trigger events")
            self._save_trigger_cache()

# Global instance
_global_trigger_system = None

def get_trigger_system() -> TriggerAlertSystem:
    """Pobierz globalną instancję trigger alert system"""
    global _global_trigger_system
    if _global_trigger_system is None:
        _global_trigger_system = TriggerAlertSystem()
    return _global_trigger_system

def check_smart_money_trigger(detected_addresses: List[str], 
                             address_trust_manager) -> Tuple[bool, List[Dict]]:
    """
    Convenience function: sprawdź trigger addresses
    
    Args:
        detected_addresses: Lista wykrytych adresów
        address_trust_manager: Instance AddressTrustManager
        
    Returns:
        Tuple[bool, List[Dict]]: (czy_trigger, trigger_addresses)
    """
    system = get_trigger_system()
    return system.check_trigger_addresses(detected_addresses, address_trust_manager)

def apply_smart_money_boost(symbol: str, base_score: float,
                           trigger_addresses: List[Dict],
                           detection_source: str = "unknown") -> Tuple[float, bool]:
    """
    Convenience function: zastosuj trigger boost
    
    Args:
        symbol: Symbol tokena
        base_score: Bazowy score
        trigger_addresses: Lista trigger addresses  
        detection_source: Źródło detekcji
        
    Returns:
        Tuple[float, bool]: (boosted_score, priority_alert)
    """
    system = get_trigger_system()
    return system.apply_trigger_boost(symbol, base_score, trigger_addresses, detection_source)

def create_smart_money_alert(symbol: str, trigger_addresses: List[Dict],
                            score: float, detection_source: str) -> Dict:
    """
    Convenience function: utwórz priority alert
    
    Args:
        symbol: Symbol tokena
        trigger_addresses: Lista trigger addresses
        score: Score po boost
        detection_source: Źródło detekcji
        
    Returns:
        Dict: Priority alert
    """
    system = get_trigger_system()
    return system.create_priority_alert(symbol, trigger_addresses, score, detection_source)

def get_trigger_stats() -> Dict:
    """Convenience function: pobierz statystyki trigger system"""
    system = get_trigger_system()
    return system.get_trigger_statistics()