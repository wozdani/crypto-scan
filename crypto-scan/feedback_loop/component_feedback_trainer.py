#!/usr/bin/env python3
"""
🎯 COMPONENT-AWARE FEEDBACK LOOP V4 - Dynamic Component Learning System
========================================================================

Component Feedback Trainer - Śledzi skuteczność poszczególnych komponentów
scoringowych (DEX, Whale, Trust, ID, DiamondWhale, Californium, WhaleCLIP, GNN)
i dostosowuje dynamicznie ich wpływ na końcowy score alertu.

Obsługiwane detektory:
✅ ClassicStealth (whale, dex, trust, id)  
✅ DiamondWhale AI (diamond)
✅ CaliforniumWhale AI (californium/mastermind)
✅ WhaleCLIP (clip)  
✅ GraphGNN/DiamondGraph (gnn)
✅ MultiAgentConsensus (consensus_weight)
✅ RLAgentV3 (feedback-adjusted)

Cel: System uczy się które składniki wzmacniają/osłabiają alerty,
eliminując fałszywe boosty i optymalizując scoring do realnych warunków rynkowych.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentFeedbackTrainer:
    """
    Trainer dla komponentów wszystkich detektorów z dynamic weight adjustment
    """
    
    def __init__(self, cache_dir: str = "cache", feedback_dir: str = "feedback_loop"):
        self.cache_dir = Path(cache_dir)
        self.feedback_dir = Path(feedback_dir)
        
        # Ensure directories exist
        self.cache_dir.mkdir(exist_ok=True)
        self.feedback_dir.mkdir(exist_ok=True)
        
        self.component_memory_file = self.feedback_dir / "component_score_memory.jsonl"
        self.component_weights_file = self.feedback_dir / "component_dynamic_weights.json"
        
        # Learning parameters
        self.learning_rate = 0.05  # Pozytywny learning rate
        self.decay_factor = 0.03   # Negatywny decay rate
        self.min_weight = 0.1      # Minimum weight limit
        self.max_weight = 2.0      # Maximum weight limit
        
        # Initialize component weights if not exist
        self._initialize_component_weights()
        
        logger.info(f"[COMPONENT TRAINER] Initialized with cache_dir={cache_dir}, feedback_dir={feedback_dir}")
    
    def _initialize_component_weights(self):
        """Initialize default component weights if file doesn't exist"""
        if not self.component_weights_file.exists():
            default_weights = {
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
                "rl_agent": 1.0,       # RLAgentV3
                
                # Meta information
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_updates": 0
            }
            
            with open(self.component_weights_file, 'w') as f:
                json.dump(default_weights, f, indent=2)
            
            logger.info(f"[COMPONENT TRAINER] Initialized default weights: {len(default_weights)-2} components")
    
    def update_component_scores_from_result(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Aktualizuje component weights na podstawie wyniku alertu
        
        Args:
            result: Dictionary z wynikiem alertu zawierający:
                - symbol: Symbol tokena
                - detector: Nazwa detektora  
                - scores: Dict komponentów i ich scores
                - was_successful: Bool czy alert był skuteczny
                - price_change_2h: Zmiana ceny w % po 2h (opcjonalnie)
                - metadata: Dodatkowe dane (opcjonalnie)
                
        Returns:
            Dict[str, float]: Zaktualizowane component weights
        """
        try:
            # Validate input
            required_fields = ["symbol", "detector", "scores", "was_successful"]
            for field in required_fields:
                if field not in result:
                    logger.error(f"[COMPONENT TRAINER] Missing required field: {field}")
                    return self.get_dynamic_component_weights()
            
            symbol = result["symbol"]
            detector = result["detector"]
            scores = result["scores"]
            was_successful = result["was_successful"]
            price_change = result.get("price_change_2h", 0.0)
            
            logger.info(f"[COMPONENT TRAINER] Processing feedback for {symbol} ({detector}): success={was_successful}")
            
            # Load current weights
            current_weights = self.get_dynamic_component_weights()
            
            # Update weights based on success/failure
            updated_weights = current_weights.copy()
            
            for component, score in scores.items():
                if component in ["last_updated", "total_updates"]:
                    continue  # Skip metadata fields
                
                if score > 0:  # Only update components that contributed
                    if was_successful:
                        # Positive feedback - increase weight
                        weight_increase = self.learning_rate * score
                        updated_weights[component] = min(
                            updated_weights.get(component, 1.0) + weight_increase,
                            self.max_weight
                        )
                        logger.info(f"[COMPONENT TRAINER] {component}: +{weight_increase:.3f} → {updated_weights[component]:.3f}")
                    else:
                        # Negative feedback - decrease weight  
                        weight_decrease = self.decay_factor * score
                        updated_weights[component] = max(
                            updated_weights.get(component, 1.0) - weight_decrease,
                            self.min_weight
                        )
                        logger.info(f"[COMPONENT TRAINER] {component}: -{weight_decrease:.3f} → {updated_weights[component]:.3f}")
            
            # Update metadata
            updated_weights["last_updated"] = datetime.now(timezone.utc).isoformat()
            updated_weights["total_updates"] = updated_weights.get("total_updates", 0) + 1
            
            # Save updated weights
            with open(self.component_weights_file, 'w') as f:
                json.dump(updated_weights, f, indent=2)
            
            # Log to component memory
            self._log_component_feedback(result)
            
            logger.info(f"[COMPONENT TRAINER] Updated weights successfully: {updated_weights['total_updates']} total updates")
            return updated_weights
            
        except Exception as e:
            logger.error(f"[COMPONENT TRAINER] Error updating component scores: {e}")
            return self.get_dynamic_component_weights()
    
    def _log_component_feedback(self, result: Dict[str, Any]):
        """Log component feedback to JSONL memory file"""
        try:
            feedback_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": result["symbol"],
                "detector": result["detector"], 
                "scores": result["scores"],
                "was_successful": result["was_successful"],
                "price_change_2h": result.get("price_change_2h", 0.0),
                "metadata": result.get("metadata", {})
            }
            
            # Append to JSONL file
            with open(self.component_memory_file, 'a') as f:
                f.write(json.dumps(feedback_entry) + '\n')
            
            logger.info(f"[COMPONENT TRAINER] Logged feedback entry for {result['symbol']}")
            
        except Exception as e:
            logger.error(f"[COMPONENT TRAINER] Error logging feedback: {e}")
    
    def get_dynamic_component_weights(self) -> Dict[str, float]:
        """
        Pobierz aktualne dynamic component weights
        
        Returns:
            Dict[str, float]: Component weights learned from feedback
        """
        try:
            if self.component_weights_file.exists():
                with open(self.component_weights_file, 'r') as f:
                    weights = json.load(f)
                
                # Filter out metadata
                component_weights = {k: v for k, v in weights.items() 
                                   if k not in ["last_updated", "total_updates"] and isinstance(v, (int, float))}
                
                logger.info(f"[COMPONENT TRAINER] Loaded {len(component_weights)} dynamic weights")
                return component_weights
            else:
                logger.warning(f"[COMPONENT TRAINER] Weights file not found, using defaults")
                self._initialize_component_weights()
                return self.get_dynamic_component_weights()
                
        except Exception as e:
            logger.error(f"[COMPONENT TRAINER] Error loading weights: {e}")
            # Return safe defaults
            return {
                "dex": 1.0, "whale": 1.0, "trust": 1.0, "id": 1.0,
                "diamond": 1.0, "californium": 1.0, "clip": 1.0, "gnn": 1.0,
                "consensus": 1.0, "rl_agent": 1.0
            }
    
    def get_component_effectiveness_stats(self) -> Dict[str, Any]:
        """
        Analiza skuteczności komponentów z component_score_memory.jsonl
        
        Returns:
            Dict[str, Any]: Statystyki skuteczności komponentów
        """
        try:
            if not self.component_memory_file.exists():
                return {"error": "No component feedback history found"}
            
            stats = {}
            total_entries = 0
            component_success_counts = {}
            component_total_counts = {}
            
            with open(self.component_memory_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        total_entries += 1
                        was_successful = entry.get("was_successful", False)
                        scores = entry.get("scores", {})
                        
                        for component, score in scores.items():
                            if isinstance(score, (int, float)) and score > 0:
                                if component not in component_total_counts:
                                    component_total_counts[component] = 0
                                    component_success_counts[component] = 0
                                
                                component_total_counts[component] += 1
                                if was_successful:
                                    component_success_counts[component] += 1
                    
                    except json.JSONDecodeError:
                        continue
            
            # Calculate success rates
            effectiveness = {}
            for component in component_total_counts:
                success_rate = (component_success_counts[component] / component_total_counts[component]) * 100
                effectiveness[component] = {
                    "success_rate": round(success_rate, 1),
                    "total_uses": component_total_counts[component],
                    "successful_uses": component_success_counts[component]
                }
            
            current_weights = self.get_dynamic_component_weights()
            
            stats = {
                "total_feedback_entries": total_entries,
                "component_effectiveness": effectiveness,
                "current_weights": current_weights,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"[COMPONENT TRAINER] Generated effectiveness stats: {total_entries} entries analyzed")
            return stats
            
        except Exception as e:
            logger.error(f"[COMPONENT TRAINER] Error generating stats: {e}")
            return {"error": str(e)}
    
    def log_component_feedback_from_alert(self, symbol: str, detector: str, 
                                        component_scores: Dict[str, float],
                                        alert_success: bool, price_change_2h: float = 0.0,
                                        metadata: Dict[str, Any] = None) -> bool:
        """
        Convenience function to log component feedback from alert result
        
        Args:
            symbol: Token symbol
            detector: Detector name
            component_scores: Dict of component scores  
            alert_success: Whether alert was successful
            price_change_2h: Price change after 2 hours
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            result = {
                "symbol": symbol,
                "detector": detector,
                "scores": component_scores,
                "was_successful": alert_success,
                "price_change_2h": price_change_2h,
                "metadata": metadata or {}
            }
            
            self.update_component_scores_from_result(result)
            logger.info(f"[COMPONENT TRAINER] Logged feedback for {symbol} ({detector}): success={alert_success}")
            return True
            
        except Exception as e:
            logger.error(f"[COMPONENT TRAINER] Error logging alert feedback: {e}")
            return False


# Global instance for easy access
_component_trainer = None

def get_component_trainer() -> ComponentFeedbackTrainer:
    """Get global component trainer instance"""
    global _component_trainer
    if _component_trainer is None:
        _component_trainer = ComponentFeedbackTrainer()
    return _component_trainer

def get_dynamic_component_weights() -> Dict[str, float]:
    """Get current dynamic component weights"""
    return get_component_trainer().get_dynamic_component_weights()

def update_component_feedback(symbol: str, detector: str, component_scores: Dict[str, float],
                            alert_success: bool, price_change_2h: float = 0.0) -> bool:
    """Update component feedback from alert result"""
    return get_component_trainer().log_component_feedback_from_alert(
        symbol, detector, component_scores, alert_success, price_change_2h
    )

def get_component_stats() -> Dict[str, Any]:
    """Get component effectiveness statistics"""
    return get_component_trainer().get_component_effectiveness_stats()


if __name__ == "__main__":
    # Test Component Feedback Trainer
    print("=== COMPONENT FEEDBACK TRAINER V4 TEST ===")
    
    trainer = ComponentFeedbackTrainer()
    
    # Test successful alert
    test_result_success = {
        "symbol": "TESTUSDT",
        "detector": "CaliforniumWhale",
        "scores": {
            "dex": 0.0,
            "whale": 0.0,
            "id": 0.0,
            "diamond": 0.82,
            "clip": 0.65,
            "gnn": 0.71
        },
        "was_successful": True,
        "price_change_2h": 15.2
    }
    
    # Test failed alert
    test_result_fail = {
        "symbol": "FAILUSDT", 
        "detector": "ClassicStealth",
        "scores": {
            "dex": 0.45,
            "whale": 0.30,
            "trust": 0.20,
            "id": 0.15
        },
        "was_successful": False,
        "price_change_2h": -2.8
    }
    
    print("Testing successful alert feedback...")
    trainer.update_component_scores_from_result(test_result_success)
    
    print("Testing failed alert feedback...")
    trainer.update_component_scores_from_result(test_result_fail)
    
    print("Getting component effectiveness stats...")
    stats = trainer.get_component_effectiveness_stats()
    
    print(f"Total feedback entries: {stats.get('total_feedback_entries', 0)}")
    print(f"Component weights: {stats.get('current_weights', {})}")
    
    print("=== TEST COMPLETE ===")