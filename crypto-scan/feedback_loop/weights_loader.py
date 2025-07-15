#!/usr/bin/env python3
"""
🎯 COMPONENT WEIGHTS LOADER - Dynamic Weight Loading System
============================================================

Weights Loader - Agreguje historyczne skuteczności z component feedback
i dostarcza dynamic weights dla wszystkich detektorów w systemie.

Funkcjonalności:
✅ Ładowanie dynamic component weights z feedback training
✅ Agregacja skuteczności komponentów z JSONL history  
✅ Fallback na safe defaults w przypadku braku danych
✅ Integration z wszystkimi detektorami (Classic, AI, Consensus)
✅ Caching i performance optimization

Obsługiwane komponenty:
- dex, whale, trust, id (Classic Stealth)
- diamond (DiamondWhale AI)  
- californium (CaliforniumWhale AI)
- clip (WhaleCLIP)
- gnn (GraphGNN/DiamondGraph)
- consensus (MultiAgentConsensus) 
- rl_agent (RLAgentV3)
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentWeightsLoader:
    """
    Loader dla dynamic component weights z advanced caching i fallback logic
    """
    
    def __init__(self, feedback_dir: str = "feedback_loop", cache_timeout_minutes: int = 15):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
        self.component_weights_file = self.feedback_dir / "component_dynamic_weights.json"
        self.component_memory_file = self.feedback_dir / "component_score_memory.jsonl"
        
        # Caching configuration
        self.cache_timeout_minutes = cache_timeout_minutes
        self._weights_cache = None
        self._cache_timestamp = None
        
        # Safe default weights
        self.default_weights = {
            # Classic Stealth Components
            "dex": 1.0,
            "whale": 1.0,
            "trust": 1.0, 
            "id": 1.0,
            
            # AI Detector Components
            "diamond": 1.0,        # DiamondWhale AI
            "californium": 1.0,    # CaliforniumWhale AI
            "clip": 1.0,           # WhaleCLIP
            "gnn": 1.0,            # GraphGNN/DiamondGraph
            
            # Advanced Components
            "consensus": 1.0,      # MultiAgentConsensus
            "rl_agent": 1.0        # RLAgentV3
        }
        
        logger.info(f"[WEIGHTS LOADER] Initialized with feedback_dir={feedback_dir}, cache_timeout={cache_timeout_minutes}min")
    
    def get_dynamic_component_weights(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        Pobierz dynamic component weights z caching
        
        Args:
            force_refresh: Wymuś odświeżenie cache
            
        Returns:
            Dict[str, float]: Component weights z learned adjustments
        """
        try:
            # Check cache validity
            if not force_refresh and self._is_cache_valid():
                logger.debug(f"[WEIGHTS LOADER] Using cached weights ({len(self._weights_cache)} components)")
                return self._weights_cache.copy()
            
            # Load weights from file
            weights = self._load_weights_from_file()
            
            # Update cache
            self._weights_cache = weights.copy()
            self._cache_timestamp = datetime.now(timezone.utc)
            
            logger.info(f"[WEIGHTS LOADER] Loaded {len(weights)} dynamic weights")
            return weights
            
        except Exception as e:
            logger.error(f"[WEIGHTS LOADER] Error loading weights: {e}")
            return self.default_weights.copy()
    
    def _is_cache_valid(self) -> bool:
        """Check if current cache is still valid"""
        if self._weights_cache is None or self._cache_timestamp is None:
            return False
        
        elapsed = datetime.now(timezone.utc) - self._cache_timestamp
        return elapsed.total_seconds() < (self.cache_timeout_minutes * 60)
    
    def _load_weights_from_file(self) -> Dict[str, float]:
        """Load weights from JSON file with fallback logic"""
        try:
            if self.component_weights_file.exists():
                with open(self.component_weights_file, 'r') as f:
                    raw_weights = json.load(f)
                
                # Extract only numeric weights (filter metadata)
                weights = {}
                for key, value in raw_weights.items():
                    if key not in ["last_updated", "total_updates"] and isinstance(value, (int, float)):
                        weights[key] = float(value)
                
                # Ensure all expected components are present
                for component in self.default_weights:
                    if component not in weights:
                        weights[component] = self.default_weights[component]
                        logger.warning(f"[WEIGHTS LOADER] Missing component {component}, using default: {self.default_weights[component]}")
                
                logger.info(f"[WEIGHTS LOADER] Loaded weights from file: {len(weights)} components")
                return weights
            else:
                logger.warning(f"[WEIGHTS LOADER] Weights file not found, using defaults")
                return self.default_weights.copy()
                
        except Exception as e:
            logger.error(f"[WEIGHTS LOADER] Error reading weights file: {e}")
            return self.default_weights.copy()
    
    def get_component_effectiveness_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Agreguje skuteczność komponentów z JSONL history za ostatnie dni
        
        Args:
            days_back: Liczba dni wstecz do analizy
            
        Returns:
            Dict[str, Any]: Podsumowanie skuteczności komponentów
        """
        try:
            if not self.component_memory_file.exists():
                return {"error": "No component history found", "components": {}}
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            # Component statistics tracking
            component_stats = defaultdict(lambda: {
                "total_uses": 0,
                "successful_uses": 0, 
                "avg_score": 0.0,
                "score_sum": 0.0,
                "recent_trend": []
            })
            
            entries_processed = 0
            
            with open(self.component_memory_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                        
                        # Skip entries older than cutoff
                        if entry_time < cutoff_date:
                            continue
                        
                        entries_processed += 1
                        was_successful = entry.get("was_successful", False)
                        scores = entry.get("scores", {})
                        
                        # Update component statistics
                        for component, score in scores.items():
                            if isinstance(score, (int, float)) and score > 0:
                                stats = component_stats[component]
                                stats["total_uses"] += 1
                                stats["score_sum"] += score
                                
                                if was_successful:
                                    stats["successful_uses"] += 1
                                
                                # Track recent trend (last 20 entries per component)
                                stats["recent_trend"].append({
                                    "timestamp": entry["timestamp"],
                                    "score": score,
                                    "success": was_successful
                                })
                                
                                # Keep only last 20 for trend analysis
                                if len(stats["recent_trend"]) > 20:
                                    stats["recent_trend"] = stats["recent_trend"][-20:]
                    
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            
            # Calculate final statistics
            effectiveness_summary = {}
            for component, stats in component_stats.items():
                if stats["total_uses"] > 0:
                    success_rate = (stats["successful_uses"] / stats["total_uses"]) * 100
                    avg_score = stats["score_sum"] / stats["total_uses"]
                    
                    # Calculate recent trend (last 10 vs previous 10)
                    trend = "stable"
                    if len(stats["recent_trend"]) >= 10:
                        recent_successes = sum(1 for entry in stats["recent_trend"][-10:] if entry["success"])
                        previous_successes = sum(1 for entry in stats["recent_trend"][-20:-10] if entry["success"])
                        
                        if recent_successes > previous_successes + 1:
                            trend = "improving"
                        elif recent_successes < previous_successes - 1:
                            trend = "declining"
                    
                    effectiveness_summary[component] = {
                        "success_rate": round(success_rate, 1),
                        "total_uses": stats["total_uses"],
                        "successful_uses": stats["successful_uses"],
                        "avg_score": round(avg_score, 3),
                        "trend": trend
                    }
            
            result = {
                "analysis_period_days": days_back,
                "entries_processed": entries_processed,
                "components": effectiveness_summary,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"[WEIGHTS LOADER] Generated effectiveness summary: {entries_processed} entries, {len(effectiveness_summary)} components")
            return result
            
        except Exception as e:
            logger.error(f"[WEIGHTS LOADER] Error generating effectiveness summary: {e}")
            return {"error": str(e), "components": {}}
    
    def get_component_weights_for_detector(self, detector_name: str) -> Dict[str, float]:
        """
        Pobierz component weights specyficzne dla danego detektora
        
        Args:
            detector_name: Nazwa detektora (ClassicStealth, DiamondWhale, etc.)
            
        Returns:
            Dict[str, float]: Relevant component weights for detector
        """
        try:
            all_weights = self.get_dynamic_component_weights()
            
            # Map detector to relevant components
            detector_components = {
                "ClassicStealth": ["dex", "whale", "trust", "id"],
                "DiamondWhale": ["diamond", "whale", "trust"],
                "CaliforniumWhale": ["californium", "trust", "whale"],
                "WhaleCLIP": ["clip", "whale", "trust"],
                "GraphGNN": ["gnn", "whale", "trust"],
                "MultiAgentConsensus": ["consensus", "diamond", "clip", "gnn"],
                "RLAgentV3": ["rl_agent", "dex", "whale", "clip"]
            }
            
            relevant_components = detector_components.get(detector_name, list(all_weights.keys()))
            
            # Filter weights for relevant components
            detector_weights = {comp: all_weights.get(comp, 1.0) for comp in relevant_components}
            
            logger.info(f"[WEIGHTS LOADER] Loaded {len(detector_weights)} weights for {detector_name}")
            return detector_weights
            
        except Exception as e:
            logger.error(f"[WEIGHTS LOADER] Error getting detector weights for {detector_name}: {e}")
            return {comp: 1.0 for comp in ["dex", "whale", "trust", "id"]}  # Safe fallback
    
    def invalidate_cache(self):
        """Invalidate current cache to force reload"""
        self._weights_cache = None
        self._cache_timestamp = None
        logger.info(f"[WEIGHTS LOADER] Cache invalidated")


# Global instance for easy access
_weights_loader = None

def get_weights_loader() -> ComponentWeightsLoader:
    """Get global weights loader instance"""
    global _weights_loader
    if _weights_loader is None:
        _weights_loader = ComponentWeightsLoader()
    return _weights_loader

def get_dynamic_component_weights(force_refresh: bool = False) -> Dict[str, float]:
    """
    Agreguje historyczne skuteczności z JSONL i zwraca dict wag
    
    Args:
        force_refresh: Wymuś odświeżenie cache
        
    Returns:
        Dict[str, float]: Dynamic component weights learned from feedback
        
    Example:
        {
            "dex": 0.75,
            "whale": 1.20, 
            "id": 0.90,
            "diamond": 1.15,
            "clip": 1.10,
            "gnn": 1.05
        }
    """
    return get_weights_loader().get_dynamic_component_weights(force_refresh)

def get_component_weights_for_detector(detector_name: str) -> Dict[str, float]:
    """Get component weights specific to detector"""
    return get_weights_loader().get_component_weights_for_detector(detector_name)

def get_component_effectiveness_summary(days_back: int = 7) -> Dict[str, Any]:
    """Get component effectiveness summary from recent history"""
    return get_weights_loader().get_component_effectiveness_summary(days_back)

def invalidate_weights_cache():
    """Invalidate weights cache to force reload"""
    get_weights_loader().invalidate_cache()


if __name__ == "__main__":
    # Test Weights Loader
    print("=== COMPONENT WEIGHTS LOADER V4 TEST ===")
    
    loader = ComponentWeightsLoader()
    
    print("Testing dynamic component weights loading...")
    weights = loader.get_dynamic_component_weights()
    print(f"Loaded weights: {weights}")
    
    print("\nTesting detector-specific weights...")
    classic_weights = loader.get_component_weights_for_detector("ClassicStealth")
    diamond_weights = loader.get_component_weights_for_detector("DiamondWhale")
    print(f"ClassicStealth weights: {classic_weights}")
    print(f"DiamondWhale weights: {diamond_weights}")
    
    print("\nTesting effectiveness summary...")
    summary = loader.get_component_effectiveness_summary(days_back=7)
    print(f"Effectiveness summary: {summary}")
    
    print("\nTesting cache functionality...")
    weights_cached = loader.get_dynamic_component_weights()  # Should use cache
    print(f"Cached weights identical: {weights == weights_cached}")
    
    print("=== TEST COMPLETE ===")