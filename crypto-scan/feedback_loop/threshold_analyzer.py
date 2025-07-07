"""
Threshold Analyzer - Etap 2
Analiza skuteczności progu base_score i uczenie się optymalnych wartości

Analizuje historyczne wyniki tokenów i określa minimalny próg learned_threshold,
który daje wysoką trafność przewidywań (>55% success rate).
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


class ThresholdAnalyzer:
    """
    Analyzer skuteczności progów base_score dla samouczenia się systemu
    """
    
    def __init__(self):
        self.log_file = "feedback_loop/basic_score_results.jsonl"
        self.threshold_file = "feedback_loop/learned_threshold.json"
        self.analysis_file = "feedback_loop/threshold_analysis.json"
        self.min_sample_size = 50  # Minimum wpisów do analizy
        self.target_success_rate = 0.55  # Docelowa trafność (55%)
        
    def load_evaluated_results(self) -> List[Dict]:
        """
        Ładuje wszystkie ocenione wyniki z pliku logów
        
        Returns:
            Lista ocenionych wpisów
        """
        try:
            if not os.path.exists(self.log_file):
                return []
            
            evaluated_results = []
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        # Tylko ocenione wpisy z kompletymi danymi
                        if (entry.get('evaluated', False) and 
                            entry.get('success') is not None and
                            entry.get('basic_score') is not None):
                            evaluated_results.append(entry)
            
            print(f"[THRESHOLD ANALYZER] Loaded {len(evaluated_results)} evaluated results")
            return evaluated_results
            
        except Exception as e:
            print(f"[THRESHOLD ANALYZER ERROR] Loading results: {e}")
            return []
    
    def analyze_score_ranges(self, results: List[Dict]) -> Dict:
        """
        Analizuje skuteczność w różnych przedziałach base_score
        
        Args:
            results: Lista ocenionych wyników
            
        Returns:
            Dictionary z analizą przedziałów
        """
        if len(results) < self.min_sample_size:
            print(f"[THRESHOLD ANALYZER] Insufficient data: {len(results)} < {self.min_sample_size}")
            return {}
        
        # Definiuj przedziały analizy
        bins = [
            (0.0, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 0.4),
            (0.4, 0.45), (0.45, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)
        ]
        
        range_analysis = {}
        
        for bin_start, bin_end in bins:
            # Filtruj wyniki w tym przedziale
            range_results = [r for r in results 
                           if bin_start <= r['basic_score'] < bin_end]
            
            if len(range_results) >= 5:  # Minimum 5 próbek na przedział
                successes = sum(1 for r in range_results if r['success'])
                success_rate = successes / len(range_results)
                
                # Oblicz średni zysk/stratę
                returns = [r.get('result_pct_6h', 0) for r in range_results]
                avg_return = np.mean(returns)
                
                range_analysis[f"{bin_start:.2f}-{bin_end:.2f}"] = {
                    "min_score": bin_start,
                    "max_score": bin_end,
                    "sample_size": len(range_results),
                    "successes": successes,
                    "success_rate": round(success_rate, 3),
                    "avg_return_pct": round(avg_return, 2),
                    "qualifies": success_rate >= self.target_success_rate
                }
        
        print(f"[THRESHOLD ANALYZER] Analyzed {len(range_analysis)} score ranges")
        return range_analysis
    
    def find_optimal_threshold(self, range_analysis: Dict) -> Optional[float]:
        """
        Znajduje optymalny próg na podstawie analizy przedziałów
        
        Args:
            range_analysis: Wyniki analizy przedziałów
            
        Returns:
            Optymalny próg lub None jeśli nie znaleziono
        """
        if not range_analysis:
            return None
        
        # Znajdź wszystkie przedziały spełniające kryterium success_rate
        qualifying_ranges = [
            (data["min_score"], data) 
            for data in range_analysis.values() 
            if data["qualifies"]
        ]
        
        if not qualifying_ranges:
            print("[THRESHOLD ANALYZER] No ranges meet success rate criteria")
            return None
        
        # Sortuj według min_score (od najniższego)
        qualifying_ranges.sort(key=lambda x: x[0])
        
        # Wybierz najniższy próg spełniający kryteria
        optimal_threshold = qualifying_ranges[0][0]
        best_range = qualifying_ranges[0][1]
        
        print(f"[THRESHOLD ANALYZER] Optimal threshold: {optimal_threshold:.3f}")
        print(f"  Success rate: {best_range['success_rate']:.1%}")
        print(f"  Sample size: {best_range['sample_size']}")
        print(f"  Avg return: {best_range['avg_return_pct']:+.2f}%")
        
        return optimal_threshold
    
    def save_learned_threshold(self, threshold: float, analysis: Dict, 
                             total_samples: int) -> bool:
        """
        Zapisuje nauczony próg do pliku
        
        Args:
            threshold: Nauczony próg
            analysis: Pełna analiza przedziałów
            total_samples: Całkowita liczba próbek
            
        Returns:
            True jeśli zapisano pomyślnie
        """
        try:
            # Przygotuj dane do zapisu
            learned_data = {
                "learned_threshold": threshold,
                "sample_size": total_samples,
                "timestamp": datetime.now().isoformat(),
                "target_success_rate": self.target_success_rate,
                "analysis_summary": {
                    "total_ranges": len(analysis),
                    "qualifying_ranges": len([d for d in analysis.values() if d["qualifies"]]),
                    "best_success_rate": max([d["success_rate"] for d in analysis.values()], default=0)
                }
            }
            
            # Zapisz learned threshold
            os.makedirs(os.path.dirname(self.threshold_file), exist_ok=True)
            with open(self.threshold_file, 'w') as f:
                json.dump(learned_data, f, indent=2)
            
            # Zapisz pełną analizę
            full_analysis = {
                "timestamp": datetime.now().isoformat(),
                "learned_threshold": threshold,
                "total_samples": total_samples,
                "range_analysis": analysis,
                "statistics": {
                    "total_ranges_analyzed": len(analysis),
                    "qualifying_ranges": len([d for d in analysis.values() if d["qualifies"]]),
                    "min_sample_per_range": min([d["sample_size"] for d in analysis.values()], default=0),
                    "max_sample_per_range": max([d["sample_size"] for d in analysis.values()], default=0)
                }
            }
            
            with open(self.analysis_file, 'w') as f:
                json.dump(full_analysis, f, indent=2)
            
            print(f"[THRESHOLD ANALYZER] Saved learned threshold: {threshold:.3f}")
            return True
            
        except Exception as e:
            print(f"[THRESHOLD ANALYZER ERROR] Saving threshold: {e}")
            return False
    
    def run_threshold_analysis(self) -> Optional[float]:
        """
        Uruchamia pełną analizę progu i zapisuje wyniki
        
        Returns:
            Nauczony próg lub None jeśli analiza się nie powiodła
        """
        print("[THRESHOLD ANALYZER] Starting threshold analysis...")
        
        # Załaduj ocenione wyniki
        results = self.load_evaluated_results()
        
        if len(results) < self.min_sample_size:
            print(f"[THRESHOLD ANALYZER] Insufficient data for analysis: {len(results)} < {self.min_sample_size}")
            return None
        
        # Analizuj przedziały skuteczności
        range_analysis = self.analyze_score_ranges(results)
        
        if not range_analysis:
            print("[THRESHOLD ANALYZER] Range analysis failed")
            return None
        
        # Znajdź optymalny próg
        optimal_threshold = self.find_optimal_threshold(range_analysis)
        
        if optimal_threshold is not None:
            # Zapisz wyniki
            success = self.save_learned_threshold(optimal_threshold, range_analysis, len(results))
            
            if success:
                return optimal_threshold
        
        return None
    
    def get_threshold_statistics(self) -> Dict:
        """
        Pobiera statystyki analizy progów
        
        Returns:
            Dictionary ze statystykami
        """
        try:
            stats = {
                "has_learned_threshold": False,
                "learned_threshold": None,
                "last_analysis": None,
                "sample_size": 0,
                "needs_update": False
            }
            
            # Sprawdź learned threshold
            if os.path.exists(self.threshold_file):
                with open(self.threshold_file, 'r') as f:
                    data = json.load(f)
                    stats.update({
                        "has_learned_threshold": True,
                        "learned_threshold": data.get("learned_threshold"),
                        "last_analysis": data.get("timestamp"),
                        "sample_size": data.get("sample_size", 0)
                    })
            
            # Sprawdź czy potrzeba aktualizacji
            current_results = self.load_evaluated_results()
            if len(current_results) >= stats["sample_size"] + 50:  # 50 nowych próbek
                stats["needs_update"] = True
            
            return stats
            
        except Exception as e:
            print(f"[THRESHOLD ANALYZER ERROR] Getting statistics: {e}")
            return {"error": str(e)}


def load_learned_threshold() -> Optional[float]:
    """
    Ładuje aktualny nauczony próg z pliku
    
    Returns:
        Nauczony próg lub None jeśli nie istnieje
    """
    try:
        threshold_file = "feedback_loop/learned_threshold.json"
        
        if os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                data = json.load(f)
                threshold = data.get("learned_threshold")
                
                if threshold is not None:
                    print(f"[LEARNED THRESHOLD] Loaded: {threshold:.3f}")
                    return threshold
        
        print("[LEARNED THRESHOLD] No learned threshold available")
        return None
        
    except Exception as e:
        print(f"[LEARNED THRESHOLD ERROR] {e}")
        return None


def update_learned_threshold() -> Optional[float]:
    """
    Aktualizuje nauczony próg na podstawie najnowszych danych
    
    Returns:
        Nowy nauczony próg lub None jeśli aktualizacja się nie powiodła
    """
    analyzer = ThresholdAnalyzer()
    return analyzer.run_threshold_analysis()


def should_update_threshold() -> bool:
    """
    Sprawdza czy pora na aktualizację progu (co 50+ nowych wpisów)
    
    Returns:
        True jeśli należy uruchomić analizę
    """
    analyzer = ThresholdAnalyzer()
    stats = analyzer.get_threshold_statistics()
    return stats.get("needs_update", False)