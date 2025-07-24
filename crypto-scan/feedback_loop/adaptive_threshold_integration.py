"""
Adaptive Threshold Integration - Etap 3
Integracja samouczącego się progu z systemem selekcji tokenów

Łączy wszystkie komponenty: logowanie, analizę i wykorzystanie dynamicznego progu
w głównym systemie selekcji tokenów Dynamic Token Selector.
"""
import os
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .basic_score_logger import BasicScoreLogger, log_basic_score_result, evaluate_pending_basic_score_results
from .threshold_analyzer import ThresholdAnalyzer, load_learned_threshold, update_learned_threshold, should_update_threshold


class AdaptiveThresholdManager:
    """
    Menedżer adaptacyjnego progu łączący wszystkie komponenty systemu samouczenia się
    """
    
    def __init__(self):
        self.logger = BasicScoreLogger()
        self.analyzer = ThresholdAnalyzer()
        self.evaluation_schedule_file = "feedback_loop/last_evaluation.json"
        self.analysis_schedule_file = "feedback_loop/last_analysis.json"
        
    def log_token_result(self, symbol: str, basic_score: float, final_score: float,
                        decision: str, price_at_scan: float,
                        consensus_decision: str = None,
                        consensus_score: float = None,
                        consensus_enabled: bool = False) -> str:
        """
        Loguje wynik tokena do systemu uczenia się progu
        
        Args:
            symbol: Symbol tokena
            basic_score: Wynik z basic engine
            final_score: Finalny wynik TJDE
            decision: Decyzja systemu
            price_at_scan: Cena w momencie skanu
            consensus_decision: Decyzja konsensusu (BUY/HOLD/AVOID)
            consensus_score: Wynik konsensusu
            consensus_enabled: Czy konsensus był włączony
            
        Returns:
            ID wpisu
        """
        return self.logger.log_basic_score_result(
            symbol, basic_score, final_score, decision, price_at_scan,
            consensus_decision, consensus_score, consensus_enabled
        )
    
    async def run_scheduled_evaluation(self) -> int:
        """
        Uruchamia ewaluację pending wyników jeśli nadszedł czas
        
        Returns:
            Liczba ocenionych wpisów
        """
        try:
            # Sprawdź czy należy uruchomić ewaluację (co 2 godziny)
            if not self._should_run_evaluation():
                return 0
            
            print("[ADAPTIVE THRESHOLD] Running scheduled evaluation...")
            evaluated_count = await self.logger.evaluate_pending_results()
            
            # Zapisz czas ostatniej ewaluacji
            self._save_last_evaluation_time()
            
            # Sprawdź czy należy uruchomić analizę progu
            if evaluated_count > 0 and should_update_threshold():
                print("[ADAPTIVE THRESHOLD] Running threshold analysis...")
                new_threshold = self.analyzer.run_threshold_analysis()
                
                if new_threshold is not None:
                    print(f"[ADAPTIVE THRESHOLD] Updated threshold to: {new_threshold:.3f}")
                    self._save_last_analysis_time()
            
            return evaluated_count
            
        except Exception as e:
            print(f"[ADAPTIVE THRESHOLD ERROR] Scheduled evaluation: {e}")
            return 0
    
    def get_adaptive_threshold(self, max_score: float, sentry_cutoff: float = 0.25) -> float:
        """
        Oblicza adaptacyjny próg dla selekcji tokenów
        
        Args:
            max_score: Najwyższy score w bieżącym skanie
            sentry_cutoff: Minimalne bezpieczne cutoff
            
        Returns:
            Obliczony adaptacyjny próg
        """
        try:
            # Załaduj nauczony próg
            learned_threshold = load_learned_threshold()
            
            # Oblicz próg bazowany na max_score (70%)
            market_threshold = max_score * 0.7
            
            if learned_threshold is not None:
                # Użyj maksimum z: learned, market, sentry
                adaptive_threshold = max(learned_threshold, market_threshold, sentry_cutoff)
                
                print(f"[ADAPTIVE THRESHOLD] Learned: {learned_threshold:.3f}, "
                      f"Market (70%): {market_threshold:.3f}, "
                      f"Sentry: {sentry_cutoff:.3f} → Final: {adaptive_threshold:.3f}")
                
                return adaptive_threshold
            else:
                # Fallback do market threshold
                fallback_threshold = max(market_threshold, sentry_cutoff)
                
                print(f"[ADAPTIVE THRESHOLD] No learned threshold, using market fallback: {fallback_threshold:.3f}")
                
                return fallback_threshold
                
        except Exception as e:
            print(f"[ADAPTIVE THRESHOLD ERROR] Getting threshold: {e}")
            # Ultimate fallback
            return max(max_score * 0.7, sentry_cutoff)
    
    def _should_run_evaluation(self) -> bool:
        """
        Sprawdza czy należy uruchomić ewaluację (co 2 godziny)
        
        Returns:
            True jeśli należy uruchomić ewaluację
        """
        try:
            if not os.path.exists(self.evaluation_schedule_file):
                return True  # Pierwsza ewaluacja
            
            import json
            with open(self.evaluation_schedule_file, 'r') as f:
                data = json.load(f)
                last_evaluation = datetime.fromisoformat(data['timestamp'])
                
                # Ewaluacja co 2 godziny
                return datetime.now() - last_evaluation > timedelta(hours=2)
                
        except Exception:
            return True  # W razie błędu, uruchom ewaluację
    
    def _save_last_evaluation_time(self):
        """Zapisuje czas ostatniej ewaluacji"""
        try:
            import json
            os.makedirs(os.path.dirname(self.evaluation_schedule_file), exist_ok=True)
            
            data = {
                "timestamp": datetime.now().isoformat(),
                "type": "evaluation"
            }
            
            with open(self.evaluation_schedule_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"[ADAPTIVE THRESHOLD ERROR] Saving evaluation time: {e}")
    
    def _save_last_analysis_time(self):
        """Zapisuje czas ostatniej analizy"""
        try:
            import json
            os.makedirs(os.path.dirname(self.analysis_schedule_file), exist_ok=True)
            
            data = {
                "timestamp": datetime.now().isoformat(),
                "type": "analysis"
            }
            
            with open(self.analysis_schedule_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"[ADAPTIVE THRESHOLD ERROR] Saving analysis time: {e}")
    
    def get_system_status(self) -> Dict:
        """
        Pobiera status całego systemu adaptacyjnego progu
        
        Returns:
            Dictionary ze statusem systemu
        """
        try:
            # Statystyki loggera
            logger_stats = {
                "total_entries": self.logger.get_total_entries_count(),
                "pending_evaluation": self.logger.get_pending_evaluation_count()
            }
            
            # Statystyki analyzera
            analyzer_stats = self.analyzer.get_threshold_statistics()
            
            # Status harmonogramu
            needs_evaluation = self._should_run_evaluation()
            needs_analysis = should_update_threshold()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "logger": logger_stats,
                "analyzer": analyzer_stats,
                "schedule": {
                    "needs_evaluation": needs_evaluation,
                    "needs_analysis": needs_analysis
                },
                "status": "operational"
            }
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }


# Global instance dla łatwego dostępu
_adaptive_threshold_manager = None

def get_adaptive_threshold_manager() -> AdaptiveThresholdManager:
    """
    Pobiera globalną instancję AdaptiveThresholdManager
    
    Returns:
        Instancja managera
    """
    global _adaptive_threshold_manager
    if _adaptive_threshold_manager is None:
        _adaptive_threshold_manager = AdaptiveThresholdManager()
    return _adaptive_threshold_manager


# Convenience functions dla łatwego używania

def log_token_for_adaptive_learning(symbol: str, basic_score: float, final_score: float,
                                   decision: str, price_at_scan: float,
                                   consensus_decision: str = None,
                                   consensus_score: float = None,
                                   consensus_enabled: bool = False) -> str:
    """
    Loguje token do systemu adaptacyjnego uczenia się progu
    
    Args:
        symbol: Symbol tokena
        basic_score: Wynik z basic engine
        final_score: Finalny wynik TJDE
        decision: Decyzja systemu
        price_at_scan: Cena w momencie skanu
        consensus_decision: Decyzja konsensusu (BUY/HOLD/AVOID)
        consensus_score: Wynik konsensusu
        consensus_enabled: Czy konsensus był włączony
        
    Returns:
        ID wpisu
    """
    manager = get_adaptive_threshold_manager()
    return manager.log_token_result(symbol, basic_score, final_score, decision, price_at_scan,
                                  consensus_decision, consensus_score, consensus_enabled)


async def run_adaptive_threshold_maintenance():
    """
    Uruchamia zaplanowaną konserwację systemu adaptacyjnego progu
    
    Returns:
        Liczba ocenionych wpisów
    """
    manager = get_adaptive_threshold_manager()
    return await manager.run_scheduled_evaluation()


def get_dynamic_selection_threshold(max_score: float, sentry_cutoff: float = 0.25) -> float:
    """
    Pobiera adaptacyjny próg dla selekcji tokenów
    
    Args:
        max_score: Najwyższy score w bieżącym skanie
        sentry_cutoff: Minimalne bezpieczne cutoff
        
    Returns:
        Adaptacyjny próg
    """
    manager = get_adaptive_threshold_manager()
    return manager.get_adaptive_threshold(max_score, sentry_cutoff)


def get_adaptive_system_status() -> Dict:
    """
    Pobiera status systemu adaptacyjnego uczenia się progu
    
    Returns:
        Dictionary ze statusem
    """
    manager = get_adaptive_threshold_manager()
    return manager.get_system_status()