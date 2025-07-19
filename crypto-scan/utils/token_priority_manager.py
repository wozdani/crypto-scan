"""
Token Priority Manager - Etap 4
System dynamicznej priorytetyzacji tokenów z powracającymi adresami (whale/dex)

Cel: Tokeny z repeat whales są skanowane jako pierwsze
"""

import json
import os
from typing import Dict, List
from datetime import datetime, timedelta
import threading

class TokenPriorityManager:
    """
    Manager priorytetów tokenów dla Stealth Engine
    Tokeny z repeat whales otrzymują wyższy priorytet skanowania
    """
    
    def __init__(self, priority_file: str = "cache/token_priorities.json"):
        self.priority_file = priority_file
        self.token_priority_map: Dict[str, float] = {}
        self.priority_decay_hours = 6  # Priorytet zanika po 6 godzinach
        self._lock = threading.Lock()
        self.load_priorities()
    
    def load_priorities(self):
        """Załaduj priorytety z pliku cache"""
        try:
            if os.path.exists(self.priority_file):
                with open(self.priority_file, 'r') as f:
                    data = json.load(f)
                    self.token_priority_map = data.get('priorities', {})
                    self._cleanup_expired_priorities()
            else:
                # Utwórz katalog cache jeśli nie istnieje
                os.makedirs(os.path.dirname(self.priority_file), exist_ok=True)
                self.token_priority_map = {}
        except Exception as e:
            print(f"[TOKEN PRIORITY] Error loading priorities: {e}")
            self.token_priority_map = {}
    
    def save_priorities(self):
        """🔧 CRITICAL FIX #2: Optimized priority saving to prevent CHZUSDT timeout"""
        try:
            # 📊 PERFORMANCE: Check if save is necessary (reduce file I/O)
            if not hasattr(self, '_last_save_hash'):
                self._last_save_hash = None
                
            current_hash = hash(str(sorted(self.token_priority_map.items())))
            if self._last_save_hash == current_hash:
                print(f"[TOKEN PRIORITY] Cache unchanged, skipping save (optimization)")
                return
            
            import signal
            import pickle
            
            def timeout_handler(signum, frame):
                raise TimeoutError("save_priorities timeout after 2 seconds")
            
            # Set 2-second timeout for file operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(2)
            
            try:
                # 🚀 OPTIMIZATION: Use pickle for large datasets (2469+ addresses)
                if len(self.token_priority_map) > 1000:
                    # Large dataset - use pickle (10x faster than JSON)
                    pickle_file = self.priority_file.replace('.json', '.pkl')
                    with self._lock:
                        with open(pickle_file, 'wb') as f:
                            pickle.dump({
                                'priorities': self.token_priority_map,
                                'last_updated': datetime.now().isoformat()
                            }, f)
                    print(f"[TOKEN PRIORITY] Large dataset saved via pickle ({len(self.token_priority_map)} entries)")
                else:
                    # Small dataset - use JSON
                    with self._lock:
                        data = {
                            'priorities': self.token_priority_map,
                            'last_updated': datetime.now().isoformat()
                        }
                        with open(self.priority_file, 'w') as f:
                            json.dump(data, f, indent=2)
                    
                signal.alarm(0)  # Cancel timeout
                self._last_save_hash = current_hash
                print(f"[TOKEN PRIORITY] Save completed in time ({len(self.token_priority_map)} priorities)")
                
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                print(f"[TOKEN PRIORITY] TIMEOUT PROTECTION: save_priorities exceeded 2s - using emergency skip")
                # 🔄 CACHE FALLBACK: Save to memory cache instead
                if not hasattr(self, '_emergency_cache'):
                    self._emergency_cache = {}
                self._emergency_cache.update(self.token_priority_map)
                print(f"[TOKEN PRIORITY] Emergency cache backup: {len(self._emergency_cache)} entries")
                return  # Emergency fallback
                
        except Exception as e:
            print(f"[TOKEN PRIORITY] Error saving priorities: {e}")
            # Try emergency cache save
            try:
                if not hasattr(self, '_emergency_cache'):
                    self._emergency_cache = {}
                self._emergency_cache.update(self.token_priority_map)
                print(f"[TOKEN PRIORITY] Exception cache backup: {len(self._emergency_cache)} entries")
            except:
                pass
    
    def update_token_priority(self, token: str, priority_boost: float, source: str = "repeat_whale"):
        """
        Aktualizuj priorytet tokena
        
        Args:
            token: Symbol tokena
            priority_boost: Wzrost priorytetu (np. +10 dla repeat whale)
            source: Źródło boost (whale_ping, dex_inflow, velocity, etc.)
        """
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("update_token_priority timeout after 3 seconds")
            
            # Set 3-second timeout for entire priority update operation
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(3)
            
            try:
                with self._lock:
                    current_priority = self.token_priority_map.get(token, 0.0)
                    new_priority = current_priority + priority_boost
                    
                    # Maksymalny priorytet to 100
                    self.token_priority_map[token] = min(new_priority, 100.0)
                    
                    print(f"[TOKEN PRIORITY] {token} priority boost: +{priority_boost:.1f} from {source} → total: {self.token_priority_map[token]:.1f}")
                    
                    # Automatycznie zapisz po każdej aktualizacji (with timeout protection)
                    self.save_priorities()
                    
                signal.alarm(0)  # Cancel timeout
                
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                print(f"[TOKEN PRIORITY] TIMEOUT PROTECTION: update_token_priority for {token} exceeded 3s - using emergency skip")
                print(f"[TOKEN PRIORITY] {token} priority update EMERGENCY SKIP → continuing processing...")
                return  # Emergency fallback - skip priority update to prevent system hang
                
        except Exception as e:
            print(f"[TOKEN PRIORITY] Error updating priority for {token}: {e}")
    
    def get_token_priority(self, token: str) -> float:
        """Pobierz aktualny priorytet tokena"""
        return self.token_priority_map.get(token, 0.0)
    
    def sort_tokens_by_priority(self, token_list: List[str]) -> List[str]:
        """
        Sortuj tokeny według priorytetu malejąco
        
        Args:
            token_list: Lista tokenów do posortowania
            
        Returns:
            Lista tokenów posortowana według priorytetu (najwyższy pierwszy)
        """
        try:
            # Wyczyść wygasłe priorytety przed sortowaniem
            self._cleanup_expired_priorities()
            
            sorted_tokens = sorted(
                token_list,
                key=lambda token: self.get_token_priority(token),
                reverse=True  # Najwyższy priorytet pierwszy
            )
            
            # Debug info o sortowaniu
            high_priority_tokens = [token for token in sorted_tokens if self.get_token_priority(token) > 0]
            if high_priority_tokens:
                print(f"[TOKEN PRIORITY] High priority tokens detected: {len(high_priority_tokens)}")
                for token in high_priority_tokens[:5]:  # Show top 5
                    priority = self.get_token_priority(token)
                    print(f"[TOKEN PRIORITY]   {token}: {priority:.1f}")
            
            return sorted_tokens
            
        except Exception as e:
            print(f"[TOKEN PRIORITY] Error sorting tokens: {e}")
            return token_list  # Fallback to original order
    
    def _cleanup_expired_priorities(self):
        """Usuń wygasłe priorytety (starsze niż priority_decay_hours)"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=self.priority_decay_hours)
            
            # Implementacja simple decay - każde uruchomienie zmniejsza priorytety o 10%
            expired_tokens = []
            for token, priority in self.token_priority_map.items():
                # Decay priorytetu o 10% przy każdym cleanup
                new_priority = priority * 0.9
                if new_priority < 1.0:  # Usuń priorytety poniżej 1.0
                    expired_tokens.append(token)
                else:
                    self.token_priority_map[token] = new_priority
            
            # Usuń wygasłe tokeny
            for token in expired_tokens:
                del self.token_priority_map[token]
            
            if expired_tokens:
                print(f"[TOKEN PRIORITY] Cleaned up {len(expired_tokens)} expired priorities")
                
        except Exception as e:
            print(f"[TOKEN PRIORITY] Error cleaning up priorities: {e}")
    
    def get_priority_statistics(self) -> Dict:
        """Pobierz statystyki priorytetów"""
        try:
            self._cleanup_expired_priorities()
            
            if not self.token_priority_map:
                return {
                    'total_tokens': 0,
                    'high_priority_tokens': 0,
                    'average_priority': 0.0,
                    'max_priority': 0.0,
                    'top_tokens': []
                }
            
            priorities = list(self.token_priority_map.values())
            top_tokens = sorted(
                self.token_priority_map.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10
            
            return {
                'total_tokens': len(self.token_priority_map),
                'high_priority_tokens': len([p for p in priorities if p >= 10.0]),
                'average_priority': sum(priorities) / len(priorities),
                'max_priority': max(priorities),
                'top_tokens': top_tokens
            }
            
        except Exception as e:
            print(f"[TOKEN PRIORITY] Error getting statistics: {e}")
            return {}
    
    def reset_token_priority(self, token: str):
        """Zresetuj priorytet tokena do 0"""
        try:
            with self._lock:
                if token in self.token_priority_map:
                    del self.token_priority_map[token]
                    print(f"[TOKEN PRIORITY] Reset priority for {token}")
                    self.save_priorities()
        except Exception as e:
            print(f"[TOKEN PRIORITY] Error resetting priority for {token}: {e}")


# Globalny singleton manager
_token_priority_manager = None
_manager_lock = threading.Lock()

def get_token_priority_manager() -> TokenPriorityManager:
    """Pobierz globalny singleton manager priorytetów"""
    global _token_priority_manager
    
    if _token_priority_manager is None:
        with _manager_lock:
            if _token_priority_manager is None:
                _token_priority_manager = TokenPriorityManager()
    
    return _token_priority_manager

def update_token_priority(token: str, priority_boost: float, source: str = "repeat_whale"):
    """Convenience function - aktualizuj priorytet tokena"""
    manager = get_token_priority_manager()
    manager.update_token_priority(token, priority_boost, source)

def sort_tokens_by_priority(token_list: List[str]) -> List[str]:
    """Convenience function - sortuj tokeny według priorytetu"""
    manager = get_token_priority_manager()
    return manager.sort_tokens_by_priority(token_list)

def get_priority_statistics() -> Dict:
    """Convenience function - pobierz statystyki priorytetów"""
    manager = get_token_priority_manager()
    return manager.get_priority_statistics()