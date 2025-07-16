#!/usr/bin/env python3
"""
🎯 STAGE 15: ALERT PRIORITIZATION - Dynamic Token Queue Management
================================================================

Priority Scheduler - Dynamiczne kolejkowanie tokenów wg wagi sygnału stealth
Umożliwia skanowanie tokenów w kolejności od najwyższego early_score

Funkcje:
1. get_token_priority_list() - Sortuje tokeny wg early_score  
2. calculate_early_score() - Oblicza combined score z DEX inflow, whale_ping, identity boost
3. update_stealth_scores() - Aktualizuje cache/stealth_last_scores.json
4. get_priority_scanning_queue() - Zwraca kolejkę skanowania z priorytetem
5. AlertQueueManager - Zaawansowany manager kolejki alertów

Cel: Skrócenie opóźnienia między sygnałem a decyzją przez inteligentne szeregowanie
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class AlertQueueManager:
    """
    Zaawansowany manager kolejki alertów z dynamicznym priorytetem
    """
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.stealth_scores_file = self.cache_dir / "stealth_last_scores.json"
        self.priority_queue_file = self.cache_dir / "priority_queue.json"
        self.alert_history_file = self.cache_dir / "alert_history.json"
        
        # Inicjalizuj puste pliki jeśli nie istnieją
        self._initialize_cache_files()
    
    def _initialize_cache_files(self):
        """Inicjalizuj puste pliki cache"""
        for cache_file in [self.stealth_scores_file, self.priority_queue_file, self.alert_history_file]:
            if not cache_file.exists():
                cache_file.write_text("{}")
    
    def calculate_early_score(self, token: str, stealth_data: Dict) -> float:
        """
        Oblicz early_score dla tokena na podstawie dostępnych danych
        
        Args:
            token: Symbol tokena
            stealth_data: Dane stealth z cache
            
        Returns:
            float: Early score (0.0-5.0+)
        """
        try:
            # Podstawowe komponenty score
            base_score = stealth_data.get("score", 0.0)
            dex_inflow_strength = stealth_data.get("dex_inflow", 0.0)
            whale_ping_strength = stealth_data.get("whale_ping", 0.0)
            identity_boost = stealth_data.get("identity_boost", 0.0)
            trust_boost = stealth_data.get("trust_boost", 0.0)
            
            # Oblicz early_score z wagami
            early_score = (
                base_score * 1.0 +           # Base stealth score
                dex_inflow_strength * 0.6 +  # DEX inflow waga
                whale_ping_strength * 0.4 +  # Whale ping waga
                identity_boost * 2.0 +       # Identity boost multiplier
                trust_boost * 1.5            # Trust boost multiplier
            )
            
            # Bonus za kombinację sygnałów
            signal_count = sum(1 for val in [dex_inflow_strength, whale_ping_strength] if val > 0)
            if signal_count >= 2:
                early_score += 0.3  # Multi-signal bonus
            
            # Bonus za high-trust addresses
            if identity_boost > 0.1:
                early_score += 0.2  # High identity bonus
            
            return round(early_score, 3)
            
        except Exception as e:
            print(f"[PRIORITY SCHEDULER] Error calculating early_score for {token}: {e}")
            return 0.0
    
    def get_token_priority_list(self, all_tokens: List[str], stealth_signals: Dict = None) -> List[Tuple[str, float]]:
        """
        Zwraca listę tokenów posortowaną wg early_score (malejąco)
        
        Args:
            all_tokens: Lista wszystkich tokenów do posortowania
            stealth_signals: Opcjonalne dane stealth (domyślnie z cache)
            
        Returns:
            List[Tuple[str, float]]: Lista (token, early_score) posortowana malejąco
        """
        try:
            # Pobierz dane stealth z cache jeśli nie podano
            if stealth_signals is None:
                stealth_signals = self.load_stealth_scores()
            
            scores = []
            for token in all_tokens:
                stealth_data = stealth_signals.get(token, {})
                early_score = self.calculate_early_score(token, stealth_data)
                scores.append((token, early_score))
            
            # Sortuj malejąco według early_score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"[PRIORITY SCHEDULER] Sorted {len(scores)} tokens by early_score")
            return scores
            
        except Exception as e:
            print(f"[PRIORITY SCHEDULER] Error creating priority list: {e}")
            return [(token, 0.0) for token in all_tokens]
    
    def load_stealth_scores(self) -> Dict:
        """Załaduj ostatnie stealth scores z cache"""
        try:
            if self.stealth_scores_file.exists():
                with open(self.stealth_scores_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"[PRIORITY SCHEDULER] Error loading stealth scores: {e}")
            return {}
    
    def update_stealth_scores(self, token: str, stealth_result: Dict):
        """
        Aktualizuj stealth scores w cache po zakończeniu skanu
        
        Args:
            token: Symbol tokena
            stealth_result: Wynik stealth analysis
        """
        try:
            scores = self.load_stealth_scores()
            
            # Aktualizuj dane dla tokena
            scores[token] = {
                "score": stealth_result.get("score", 0.0),
                "dex_inflow": stealth_result.get("dex_inflow", 0.0),
                "whale_ping": stealth_result.get("whale_ping", 0.0),
                "identity_boost": stealth_result.get("identity_boost", 0.0),
                "trust_boost": stealth_result.get("trust_boost", 0.0),
                "timestamp": datetime.now().isoformat(),
                "early_score": self.calculate_early_score(token, stealth_result),
                
                # === CONSENSUS DECISION ENGINE DATA ===
                "consensus_decision": stealth_result.get("consensus_decision"),
                "consensus_score": stealth_result.get("consensus_score"),
                "consensus_confidence": stealth_result.get("consensus_confidence"),
                "consensus_detectors": stealth_result.get("consensus_detectors", []),
                "consensus_votes": stealth_result.get("consensus_votes", [])
            }
            
            # Zapisz zaktualizowane scores
            with open(self.stealth_scores_file, 'w') as f:
                json.dump(scores, f, indent=2)
            
            print(f"[PRIORITY SCHEDULER] Updated stealth scores for {token}: early_score={scores[token]['early_score']:.3f}")
            
        except Exception as e:
            print(f"[PRIORITY SCHEDULER] Error updating stealth scores for {token}: {e}")
    
    def get_priority_scanning_queue(self, all_tokens: List[str], top_n: int = None) -> List[str]:
        """
        Zwraca kolejkę skanowania z priorytetem
        
        Args:
            all_tokens: Lista wszystkich tokenów
            top_n: Ograniczenie do top N tokenów (None = wszystkie)
            
        Returns:
            List[str]: Lista tokenów w kolejności priorytetu
        """
        try:
            priority_list = self.get_token_priority_list(all_tokens)
            
            if top_n:
                priority_list = priority_list[:top_n]
            
            # Zapisz aktualną kolejkę priorytetu
            queue_data = {
                "timestamp": datetime.now().isoformat(),
                "total_tokens": len(all_tokens),
                "priority_queue": priority_list[:50],  # Top 50 dla monitorowania
                "queue_size": len(priority_list)
            }
            
            with open(self.priority_queue_file, 'w') as f:
                json.dump(queue_data, f, indent=2)
            
            # Zwróć tylko symbole tokenów
            return [token for token, _ in priority_list]
            
        except Exception as e:
            print(f"[PRIORITY SCHEDULER] Error creating priority queue: {e}")
            return all_tokens  # Fallback do oryginalnej kolejności
    
    def get_top_priority_tokens(self, limit: int = 10) -> List[Dict]:
        """
        Zwraca top priority tokens z dodatkowymi informacjami
        
        Args:
            limit: Maksymalna liczba tokenów do zwrócenia
            
        Returns:
            List[Dict]: Lista tokenów z metadata
        """
        try:
            stealth_scores = self.load_stealth_scores()
            
            # Konwertuj na listę z metadata
            tokens_with_metadata = []
            for token, data in stealth_scores.items():
                tokens_with_metadata.append({
                    "token": token,
                    "early_score": data.get("early_score", 0.0),
                    "base_score": data.get("score", 0.0),
                    "dex_inflow": data.get("dex_inflow", 0.0),
                    "whale_ping": data.get("whale_ping", 0.0),
                    "identity_boost": data.get("identity_boost", 0.0),
                    "trust_boost": data.get("trust_boost", 0.0),
                    "timestamp": data.get("timestamp", "")
                })
            
            # Sortuj według early_score
            tokens_with_metadata.sort(key=lambda x: x["early_score"], reverse=True)
            
            return tokens_with_metadata[:limit]
            
        except Exception as e:
            print(f"[PRIORITY SCHEDULER] Error getting top priority tokens: {e}")
            return []
    
    def log_alert_sent(self, token: str, alert_type: str, priority_score: float):
        """
        Loguj wysłany alert do historii
        
        Args:
            token: Symbol tokena
            alert_type: Typ alertu (stealth, whale_ping, dex_inflow)
            priority_score: Score priorytetu
        """
        try:
            # Załaduj historię alertów
            if self.alert_history_file.exists():
                with open(self.alert_history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = {"alerts": []}
            
            # Dodaj nowy alert
            alert_entry = {
                "timestamp": datetime.now().isoformat(),
                "token": token,
                "alert_type": alert_type,
                "priority_score": priority_score,
                "queue_position": self._get_current_queue_position(token)
            }
            
            history["alerts"].append(alert_entry)
            
            # Zachowaj ostatnie 1000 alertów
            if len(history["alerts"]) > 1000:
                history["alerts"] = history["alerts"][-1000:]
            
            # Zapisz historię
            with open(self.alert_history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"[PRIORITY SCHEDULER] Logged alert: {token} ({alert_type}, score: {priority_score:.3f})")
            
        except Exception as e:
            print(f"[PRIORITY SCHEDULER] Error logging alert for {token}: {e}")
    
    def _get_current_queue_position(self, token: str) -> int:
        """Pobierz aktualną pozycję tokena w kolejce"""
        try:
            if self.priority_queue_file.exists():
                with open(self.priority_queue_file, 'r') as f:
                    queue_data = json.load(f)
                    priority_queue = queue_data.get("priority_queue", [])
                    
                    for i, (queue_token, _) in enumerate(priority_queue):
                        if queue_token == token:
                            return i + 1
            return 0
        except:
            return 0
    
    def get_queue_statistics(self) -> Dict:
        """Zwraca statystyki kolejki prioritetów"""
        try:
            stealth_scores = self.load_stealth_scores()
            
            if not stealth_scores:
                return {"total_tokens": 0, "high_priority": 0, "avg_early_score": 0.0}
            
            early_scores = [data.get("early_score", 0.0) for data in stealth_scores.values()]
            high_priority_count = sum(1 for score in early_scores if score > 1.0)
            
            return {
                "total_tokens": len(stealth_scores),
                "high_priority": high_priority_count,
                "avg_early_score": sum(early_scores) / len(early_scores) if early_scores else 0.0,
                "max_early_score": max(early_scores) if early_scores else 0.0,
                "min_early_score": min(early_scores) if early_scores else 0.0,
                "last_update": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"[PRIORITY SCHEDULER] Error getting queue statistics: {e}")
            return {"error": str(e)}

# Convenience functions dla easy integration
_queue_manager = None

def get_queue_manager() -> AlertQueueManager:
    """Pobierz globalny instance AlertQueueManager"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = AlertQueueManager()
    return _queue_manager

def get_token_priority_list(all_tokens: List[str], stealth_signals: Dict = None) -> List[Tuple[str, float]]:
    """Convenience function - zwraca listę tokenów posortowaną wg early_score"""
    return get_queue_manager().get_token_priority_list(all_tokens, stealth_signals)

def update_stealth_scores(token: str, stealth_result: Dict):
    """Convenience function - aktualizuj stealth scores"""
    return get_queue_manager().update_stealth_scores(token, stealth_result)

def get_priority_scanning_queue(all_tokens: List[str], top_n: int = None) -> List[str]:
    """Convenience function - zwraca kolejkę skanowania z priorytetem"""
    return get_queue_manager().get_priority_scanning_queue(all_tokens, top_n)

def get_top_priority_tokens(limit: int = 10) -> List[Dict]:
    """Convenience function - zwraca top priority tokens"""
    return get_queue_manager().get_top_priority_tokens(limit)

def log_alert_sent(token: str, alert_type: str, priority_score: float):
    """Convenience function - loguj wysłany alert"""
    return get_queue_manager().log_alert_sent(token, alert_type, priority_score)

def get_queue_statistics() -> Dict:
    """Convenience function - zwraca statystyki kolejki"""
    return get_queue_manager().get_queue_statistics()