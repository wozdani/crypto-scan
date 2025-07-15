#!/usr/bin/env python3
"""
🎯 FEEDBACK LOOP V4 z DYNAMICZNYM SELF-LEARNING DECAY - Core System
====================================================================

Weights Decay - Samouczący się decay system dla dynamicznych wag scoringowych
komponentów detektorów. System automatycznie osłabia komponenty, które przestają
działać i wzmacnia te z rosnącą skutecznością.

Funkcjonalności:
✅ Dynamiczne osłabianie wag komponentów z spadającą skutecznością
✅ Wzmacnianie komponentów z rosnącą skutecznością  
✅ Per-component, per-detector decay analysis
✅ Automatic operation bez sztywnego harmonogramu
✅ Intelligent trend analysis z historical performance
✅ Comprehensive logging z decay reasoning

Obsługiwane komponenty:
- DEX inflow (dex_score)
- Whale ping (whale_score)  
- Trust detection (trust_score)
- Identity signals (id_score)
- DiamondWhale (diamond_score)
- WhaleCLIP (clip_score)
- GraphGNN (gnn_score)
- CaliforniumWhale (californium_score)
- MultiAgentConsensus (vote accuracy per agent)
"""

import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicWeightsDecay:
    """
    Samouczący się decay system dla component weights z intelligent trend analysis
    """
    
    def __init__(self, feedback_dir: str = "feedback_loop"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
        self.component_memory_file = self.feedback_dir / "component_score_memory.jsonl"
        
        # Decay configuration
        self.analysis_window_short = 10   # Last 10 alerts for recent performance
        self.analysis_window_long = 50    # Last 50 alerts for baseline performance
        self.min_history_required = 15   # Minimum history needed for decay analysis
        
        # Decay thresholds
        self.decline_threshold = 0.4      # Below this rate = decline detected
        self.improvement_threshold = 0.6  # Above this rate = improvement detected
        
        # Decay factors
        self.decay_decline = 0.95         # Osłab wagę o 5%
        self.decay_improvement = 1.05     # Wzmocnij wagę o 5%
        self.decay_neutral = 1.00         # Bez zmian
        
        # Weight bounds
        self.min_weight = 0.1
        self.max_weight = 2.0
        
        logger.info(f"[DECAY SYSTEM] Initialized with analysis windows: {self.analysis_window_short}/{self.analysis_window_long}")
    
    def compute_decay_factor(self, component_history: List[Dict[str, Any]]) -> Tuple[float, str]:
        """
        Zwraca współczynnik decay dla danego komponentu scoringowego
        
        Args:
            component_history: Lista historii alertów dla komponentu
            
        Returns:
            Tuple[float, str]: (decay_factor, reasoning)
        """
        try:
            if len(component_history) < self.min_history_required:
                return self.decay_neutral, f"insufficient_history_{len(component_history)}"
            
            # Pobierz ostatnie okresy
            last_short = component_history[-self.analysis_window_short:]
            last_long = component_history[-self.analysis_window_long:] if len(component_history) >= self.analysis_window_long else component_history
            
            # Oblicz success rates
            rate_short = np.mean([entry.get("was_successful", False) for entry in last_short])
            rate_long = np.mean([entry.get("was_successful", False) for entry in last_long])
            
            # Analiza trendu
            trend_direction = rate_short - rate_long
            
            # Decay decision logic
            if rate_short < self.decline_threshold and trend_direction < -0.1:
                # Znaczący spadek skuteczności
                decay_factor = self.decay_decline
                reasoning = f"declining_effectiveness_{rate_short:.3f}_vs_{rate_long:.3f}"
                
            elif rate_short > self.improvement_threshold and trend_direction > 0.1:
                # Znacząca poprawa skuteczności
                decay_factor = self.decay_improvement
                reasoning = f"improving_effectiveness_{rate_short:.3f}_vs_{rate_long:.3f}"
                
            elif abs(trend_direction) < 0.05 and rate_short > 0.5:
                # Stabilna dobra skuteczność
                decay_factor = self.decay_neutral
                reasoning = f"stable_good_performance_{rate_short:.3f}"
                
            elif rate_short < 0.3:
                # Bardzo niska skuteczność - silny decay
                decay_factor = 0.90
                reasoning = f"very_low_effectiveness_{rate_short:.3f}"
                
            else:
                # Neutralne zachowanie
                decay_factor = self.decay_neutral
                reasoning = f"neutral_performance_{rate_short:.3f}"
            
            logger.debug(f"[DECAY ANALYSIS] Component decay: factor={decay_factor:.3f}, reason={reasoning}")
            return decay_factor, reasoning
            
        except Exception as e:
            logger.error(f"[DECAY SYSTEM] Error computing decay factor: {e}")
            return self.decay_neutral, f"error_{str(e)[:20]}"
    
    def load_component_history(self, days_back: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ładuje historię komponentów z JSONL file z time filtering
        
        Args:
            days_back: Liczba dni wstecz do analizy
            
        Returns:
            Dict[str, List[Dict]]: Component history grouped by component name
        """
        try:
            if not self.component_memory_file.exists():
                logger.warning(f"[DECAY SYSTEM] No component memory file found")
                return {}
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            component_history = defaultdict(list)
            
            with open(self.component_memory_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                        
                        if entry_time < cutoff_date:
                            continue
                        
                        # Group by individual components
                        scores = entry.get("scores", {})
                        detector = entry.get("detector", "Unknown")
                        was_successful = entry.get("was_successful", False)
                        
                        for component, score in scores.items():
                            if isinstance(score, (int, float)) and score > 0:
                                component_key = f"{detector}_{component}"
                                
                                component_history[component_key].append({
                                    "timestamp": entry["timestamp"],
                                    "detector": detector,
                                    "component": component,
                                    "score": score,
                                    "was_successful": was_successful,
                                    "symbol": entry.get("symbol", ""),
                                    "price_change_2h": entry.get("price_change_2h", 0.0)
                                })
                    
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.debug(f"[DECAY SYSTEM] Skipping invalid entry: {e}")
                        continue
            
            # Sort by timestamp for proper temporal analysis
            for component_key in component_history:
                component_history[component_key].sort(key=lambda x: x["timestamp"])
            
            logger.info(f"[DECAY SYSTEM] Loaded {len(component_history)} component histories")
            return dict(component_history)
            
        except Exception as e:
            logger.error(f"[DECAY SYSTEM] Error loading component history: {e}")
            return {}
    
    def compute_all_decay_factors(self, days_back: int = 30) -> Dict[str, Tuple[float, str]]:
        """
        Oblicza decay factors dla wszystkich komponentów
        
        Args:
            days_back: Liczba dni wstecz do analizy
            
        Returns:
            Dict[str, Tuple[float, str]]: Component decay factors and reasoning
        """
        try:
            component_history = self.load_component_history(days_back)
            decay_factors = {}
            
            for component_key, history in component_history.items():
                if len(history) >= self.min_history_required:
                    decay_factor, reasoning = self.compute_decay_factor(history)
                    decay_factors[component_key] = (decay_factor, reasoning)
                    
                    logger.info(f"[DECAY] {component_key}: factor={decay_factor:.3f} ({reasoning})")
                else:
                    logger.debug(f"[DECAY] {component_key}: insufficient history ({len(history)} entries)")
            
            return decay_factors
            
        except Exception as e:
            logger.error(f"[DECAY SYSTEM] Error computing decay factors: {e}")
            return {}
    
    def apply_decay_to_weights(self, current_weights: Dict[str, float], 
                             decay_factors: Dict[str, Tuple[float, str]]) -> Dict[str, float]:
        """
        Aplikuje decay factors do current weights z proper mapping
        
        Args:
            current_weights: Aktualne wagi komponentów
            decay_factors: Decay factors per component
            
        Returns:
            Dict[str, float]: Updated weights po decay application
        """
        try:
            updated_weights = current_weights.copy()
            decay_log = []
            
            for component in current_weights:
                # Find matching decay factor (może być per-detector specific)
                matching_decay = None
                matching_key = ""
                
                # Try exact match first
                for decay_key, (factor, reason) in decay_factors.items():
                    if decay_key.endswith(f"_{component}"):
                        matching_decay = factor
                        matching_key = decay_key
                        break
                
                # If no exact match, try component name only
                if matching_decay is None:
                    for decay_key, (factor, reason) in decay_factors.items():
                        if component in decay_key:
                            matching_decay = factor
                            matching_key = decay_key
                            break
                
                if matching_decay is not None:
                    old_weight = current_weights[component]
                    new_weight = max(self.min_weight, min(self.max_weight, old_weight * matching_decay))
                    
                    updated_weights[component] = new_weight
                    
                    if abs(new_weight - old_weight) > 0.01:  # Log significant changes
                        change_pct = ((new_weight - old_weight) / old_weight) * 100
                        decay_log.append(f"{component}: {old_weight:.3f}→{new_weight:.3f} ({change_pct:+.1f}%)")
            
            # Log decay applications
            if decay_log:
                logger.info(f"[DECAY APPLIED] {', '.join(decay_log)}")
            
            return updated_weights
            
        except Exception as e:
            logger.error(f"[DECAY SYSTEM] Error applying decay to weights: {e}")
            return current_weights
    
    def get_decay_statistics(self) -> Dict[str, Any]:
        """
        Generuje statystyki decay system performance
        
        Returns:
            Dict[str, Any]: Decay system statistics
        """
        try:
            component_history = self.load_component_history(30)
            decay_factors = self.compute_all_decay_factors(30)
            
            stats = {
                "components_analyzed": len(component_history),
                "components_with_decay": len(decay_factors),
                "decay_distribution": {
                    "improving": len([f for f, r in decay_factors.values() if f > 1.0]),
                    "declining": len([f for f, r in decay_factors.values() if f < 1.0]),
                    "stable": len([f for f, r in decay_factors.values() if f == 1.0])
                },
                "average_decay_factor": np.mean([f for f, r in decay_factors.values()]) if decay_factors else 1.0,
                "analysis_period_days": 30,
                "min_history_required": self.min_history_required,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"[DECAY SYSTEM] Error generating statistics: {e}")
            return {"error": str(e)}


# Global instance and convenience functions
_decay_system = None

def get_decay_system() -> DynamicWeightsDecay:
    """Get global decay system instance"""
    global _decay_system
    if _decay_system is None:
        _decay_system = DynamicWeightsDecay()
    return _decay_system

def compute_component_decay_factors(days_back: int = 30) -> Dict[str, Tuple[float, str]]:
    """Compute decay factors for all components"""
    return get_decay_system().compute_all_decay_factors(days_back)

def apply_decay_to_component_weights(current_weights: Dict[str, float]) -> Dict[str, float]:
    """Apply decay factors to current component weights"""
    decay_system = get_decay_system()
    decay_factors = decay_system.compute_all_decay_factors()
    return decay_system.apply_decay_to_weights(current_weights, decay_factors)

def get_decay_system_stats() -> Dict[str, Any]:
    """Get decay system performance statistics"""
    return get_decay_system().get_decay_statistics()


if __name__ == "__main__":
    # Test Dynamic Weights Decay System
    print("=== DYNAMIC WEIGHTS DECAY SYSTEM TEST ===")
    
    decay_system = DynamicWeightsDecay()
    
    print("Testing component history loading...")
    history = decay_system.load_component_history(30)
    print(f"Loaded {len(history)} component histories")
    
    print("Testing decay factor computation...")
    decay_factors = decay_system.compute_all_decay_factors(30)
    print(f"Computed {len(decay_factors)} decay factors")
    
    if decay_factors:
        print("Sample decay factors:")
        for component, (factor, reason) in list(decay_factors.items())[:5]:
            print(f"  {component}: {factor:.3f} ({reason})")
    
    print("Testing weight decay application...")
    test_weights = {
        "dex": 1.0, "whale": 1.2, "trust": 0.8, "id": 1.1,
        "diamond": 1.0, "clip": 0.9, "gnn": 1.3, "californium": 1.0
    }
    
    updated_weights = decay_system.apply_decay_to_weights(test_weights, decay_factors)
    print(f"Updated weights: {updated_weights}")
    
    print("Testing decay statistics...")
    stats = decay_system.get_decay_statistics()
    print(f"Decay stats: {stats}")
    
    print("=== TEST COMPLETE ===")