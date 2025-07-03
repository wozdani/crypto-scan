#!/usr/bin/env python3
"""
Score Unification Module - Ensures consistent TJDE scoring per symbol
Prevents multiple different scores for the same token within single scan cycle
"""

import threading
from typing import Dict, Optional
from datetime import datetime, timedelta

class ScoreUnificationManager:
    """
    🔒 Zarządza jednolitością scoringu per token per cycle
    
    Zapewnia że jeden token otrzyma tylko jeden finalny score w danym cyklu,
    niezależnie od tego ile razy zostanie analizowany przez różne detektory.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._token_scores = {}  # {symbol: {score, decision, timestamp, locked}}
        self._cycle_id = None
        self._cycle_start = None
    
    def start_new_cycle(self) -> str:
        """Rozpocznij nowy cykl skanowania z unique cycle_id"""
        with self._lock:
            self._cycle_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._cycle_start = datetime.now()
            self._token_scores.clear()
            print(f"[SCORE UNIFY] Started new cycle: {self._cycle_id}")
            return self._cycle_id
    
    def register_token_score(self, symbol: str, tjde_score: float, tjde_decision: str, 
                           market_phase: str = "unknown", setup_type: str = "unknown") -> bool:
        """
        Rejestruj score dla tokena - pierwszy wygrywa
        
        Returns:
            bool: True jeśli score został zaakceptowany, False jeśli token już ma score
        """
        with self._lock:
            if symbol in self._token_scores:
                existing = self._token_scores[symbol]
                print(f"[SCORE UNIFY] ⚠️ {symbol}: Score już zarejestrowany - {existing['score']:.3f} ({existing['decision']}) - blokuję duplikat {tjde_score:.3f} ({tjde_decision})")
                return False
            
            # Zarejestruj pierwszy score dla tego tokena w tym cyklu
            self._token_scores[symbol] = {
                'score': tjde_score,
                'decision': tjde_decision,
                'market_phase': market_phase,
                'setup_type': setup_type,
                'timestamp': datetime.now(),
                'locked': True,
                'cycle_id': self._cycle_id
            }
            
            print(f"[SCORE UNIFY] ✅ {symbol}: Registered unified score {tjde_score:.3f} ({tjde_decision}) - setup: {setup_type}")
            return True
    
    def get_unified_score(self, symbol: str) -> Optional[Dict]:
        """Pobierz unified score dla tokena jeśli istnieje"""
        with self._lock:
            return self._token_scores.get(symbol)
    
    def is_token_processed(self, symbol: str) -> bool:
        """Sprawdź czy token już ma unified score w tym cyklu"""
        with self._lock:
            return symbol in self._token_scores
    
    def get_cycle_stats(self) -> Dict:
        """Zwróć statystyki obecnego cyklu"""
        with self._lock:
            if not self._cycle_start:
                return {"error": "No active cycle"}
            
            duration = (datetime.now() - self._cycle_start).total_seconds()
            
            return {
                "cycle_id": self._cycle_id,
                "processed_tokens": len(self._token_scores),
                "duration_seconds": duration,
                "tokens_per_second": len(self._token_scores) / max(duration, 1),
                "token_list": list(self._token_scores.keys())
            }
    
    def cleanup_old_cycles(self, max_age_hours: int = 24):
        """Wyczyść stare dane starsze niż max_age_hours"""
        with self._lock:
            if not self._cycle_start:
                return
            
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            if self._cycle_start < cutoff:
                print(f"[SCORE UNIFY] Cleaning up old cycle {self._cycle_id}")
                self._token_scores.clear()
                self._cycle_id = None
                self._cycle_start = None

# Global instance dla całego systemu
_global_score_manager = ScoreUnificationManager()

def start_unified_scan_cycle() -> str:
    """Rozpocznij nowy cycle z unified scoring"""
    return _global_score_manager.start_new_cycle()

def register_unified_score(symbol: str, tjde_score: float, tjde_decision: str, 
                          market_phase: str = "unknown", setup_type: str = "unknown") -> bool:
    """Zarejestruj unified score dla tokena"""
    return _global_score_manager.register_token_score(
        symbol, tjde_score, tjde_decision, market_phase, setup_type
    )

def get_unified_token_score(symbol: str) -> Optional[Dict]:
    """Pobierz unified score dla tokena"""
    return _global_score_manager.get_unified_score(symbol)

def is_token_score_locked(symbol: str) -> bool:
    """Sprawdź czy token ma już unified score"""
    return _global_score_manager.is_token_processed(symbol)

def get_unification_stats() -> Dict:
    """Pobierz statystyki unifikacji"""
    return _global_score_manager.get_cycle_stats()

def cleanup_unified_scores():
    """Wyczyść stare unified scores"""
    _global_score_manager.cleanup_old_cycles()