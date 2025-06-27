#!/usr/bin/env python3
"""
AdaptiveTraderScore - Inteligentna warstwa scoringu dla TJDE

Uczenie się wag na podstawie historii decyzji i wyników.
System adaptuje się do rynku jak profesjonalny trader.
"""

import os
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

# Import error logging system
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crypto_scan_service import log_warning


class AdaptiveWeightEngine:
    """
    Silnik adaptacyjnych wag dla TJDE
    
    Zapamiętuje historię decyzji (cechy + wynik) i oblicza dynamiczne wagi
    dla compute_trader_score() na podstawie rzeczywistej skuteczności.
    """
    
    def __init__(self, memory_size: int = 200, persistence_file: str = "data/adaptive_weights.pkl"):
        self.memory_size = memory_size
        self.persistence_file = persistence_file
        self.memory: List[Tuple[Dict, bool]] = []
        self.performance_history = []
        self.last_updated = None
        
        # Load persisted data
        self._load_from_disk()
        
        # AdaptiveWeightEngine initialized successfully
    
    def add_example(self, features: Dict[str, float], outcome: bool, symbol: str = "UNKNOWN"):
        """
        Dodaj przykład do pamięci adaptacyjnej
        
        Args:
            features: Dict cech używanych w scoringu
            outcome: True jeśli decyzja była poprawna, False jeśli błędna
            symbol: Symbol dla logowania
        """
        try:
            example = (features.copy(), outcome)
            self.memory.append(example)
            
            # Maintain memory size limit
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)
            
            # Track performance
            self.performance_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "outcome": outcome,
                "features": features.copy()
            })
            
            # Keep only recent performance history
            if len(self.performance_history) > self.memory_size:
                self.performance_history.pop(0)
            
            self.last_updated = datetime.now(timezone.utc)
            
            # Auto-save periodically
            if len(self.memory) % 10 == 0:
                self._save_to_disk()
                
        except Exception as e:
            log_warning("ADAPTIVE WEIGHTS ADD EXAMPLE", e, f"Failed to add example for {symbol}")
    
    def compute_weights(self) -> Dict[str, float]:
        """
        Oblicz dynamiczne wagi na podstawie historii
        
        Returns:
            Dict z wagami dla każdej cechy
        """
        if len(self.memory) < 5:
            return self._get_default_weights()
        
        try:
            importance = defaultdict(float)
            count = defaultdict(int)
            
            # Analyze each historical example
            for features, outcome in self.memory:
                outcome_multiplier = 1.0 if outcome else -0.5
                
                for feature_name, feature_value in features.items():
                    # Weight by feature value and outcome
                    contribution = feature_value * outcome_multiplier
                    importance[feature_name] += contribution
                    count[feature_name] += 1
            
            # Calculate normalized weights
            weights = {}
            for feature_name in importance:
                if count[feature_name] > 0:
                    avg_importance = importance[feature_name] / count[feature_name]
                    weights[feature_name] = max(0.05, avg_importance)  # Minimum weight
            
            # Normalize to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                for feature_name in weights:
                    weights[feature_name] /= total_weight
            else:
                return self._get_default_weights()
            
            return weights
            
        except Exception as e:
            log_warning("ADAPTIVE WEIGHTS COMPUTE", e, f"Failed to compute weights from {len(self.memory) if hasattr(self, 'memory') else 0} examples")
            return self._get_default_weights()
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Wagi domyślne gdy brak danych historycznych"""
        return {
            "trend_strength": 0.25,
            "pullback_quality": 0.20,
            "support_reaction": 0.15,
            "liquidity_pattern_score": 0.10,
            "psych_score": 0.10,
            "htf_supportive_score": 0.10,
            "market_phase_modifier": 0.10
        }
    
    def get_performance_stats(self) -> Dict:
        """Statystyki wydajności adaptacyjnego scoringu"""
        if not self.performance_history:
            return {"total_examples": 0, "success_rate": 0.0, "last_updated": None}
        
        total_examples = len(self.performance_history)
        successful = sum(1 for entry in self.performance_history if entry["outcome"])
        success_rate = successful / total_examples if total_examples > 0 else 0.0
        
        return {
            "total_examples": total_examples,
            "successful_examples": successful,
            "success_rate": round(success_rate, 3),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "memory_utilization": f"{len(self.memory)}/{self.memory_size}",
            "memory_size": len(self.memory),
            "max_memory": self.memory_size
        }
    
    def _save_to_disk(self):
        """Zapisz stan do pliku"""
        try:
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            
            data = {
                "memory": self.memory,
                "performance_history": self.performance_history,
                "last_updated": self.last_updated.isoformat() if self.last_updated else None,
                "version": "1.0"
            }
            
            with open(self.persistence_file, 'wb') as f:
                pickle.dump(data, f)
                
            # State saved successfully
            
        except Exception as e:
            log_warning("ADAPTIVE WEIGHTS SAVE", e, f"Failed to save state to {self.persistence_file}")
    
    def _load_from_disk(self):
        """Wczytaj stan z pliku"""
        try:
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.memory = data.get("memory", [])
                self.performance_history = data.get("performance_history", [])
                
                last_updated_str = data.get("last_updated")
                if last_updated_str:
                    self.last_updated = datetime.fromisoformat(last_updated_str)
                
                # Memory loaded successfully
                
        except Exception as e:
            log_warning("ADAPTIVE WEIGHTS LOAD", e, f"Could not load memory from {self.persistence_file}")
            self.memory = []
            self.performance_history = []
    
    def reset_memory(self):
        """Resetuj pamięć adaptacyjną (dla testów)"""
        self.memory = []
        self.performance_history = []
        self.last_updated = None
        # Memory reset completed
    
    def export_weights_analysis(self, output_file: str = "logs/adaptive_weights_analysis.json"):
        """Eksportuj analizę wag do pliku"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            current_weights = self.compute_weights()
            default_weights = self._get_default_weights()
            stats = self.get_performance_stats()
            
            analysis = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "current_weights": current_weights,
                "default_weights": default_weights,
                "weight_differences": {
                    k: round(current_weights.get(k, 0) - default_weights.get(k, 0), 4)
                    for k in default_weights.keys()
                },
                "performance_stats": stats,
                "memory_size": len(self.memory),
                "recent_examples": self.performance_history[-10:] if self.performance_history else []
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            return analysis
            
        except Exception as e:
            log_warning("ADAPTIVE WEIGHTS EXPORT", e, f"Failed to export analysis to {output_file}")
            return None


# Global singleton instance
adaptive_engine = AdaptiveWeightEngine()


def get_adaptive_engine() -> AdaptiveWeightEngine:
    """Pobierz globalną instancję AdaptiveWeightEngine"""
    return adaptive_engine


if __name__ == "__main__":
    # Test adaptive weight engine
    engine = AdaptiveWeightEngine()
    
    # Add some test examples
    test_features = {
        "trend_strength": 0.8,
        "pullback_quality": 0.7,
        "support_reaction": 0.6,
        "liquidity_pattern_score": 0.5,
        "psych_score": 0.3,
        "htf_supportive_score": 0.8,
        "market_phase_modifier": 0.1
    }
    
    # Simulate some outcomes
    engine.add_example(test_features, True, "TEST1")
    engine.add_example({**test_features, "trend_strength": 0.9}, True, "TEST2")
    engine.add_example({**test_features, "psych_score": 0.8}, False, "TEST3")
    
    # Compute adaptive weights
    weights = engine.compute_weights()
    stats = engine.get_performance_stats()
    
    # Export analysis
    engine.export_weights_analysis()
    
    # Test completed successfully