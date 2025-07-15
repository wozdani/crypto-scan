#!/usr/bin/env python3
"""
🎯 COMPONENT SCORE UPDATER V4 - Dynamic Weights with Self-Learning Decay
========================================================================

Component Score Updater - Główny moduł aktualizujący dynamic component weights
z integracją Self-Learning Decay system. Automatycznie dostosowuje wagi na
podstawie rzeczywistej skuteczności i trendów performance.

Kluczowe funkcjonalności:
✅ Dynamic weight updates z decay integration
✅ Automatic decay factor computation i application
✅ Comprehensive logging z detailed reasoning
✅ Per-component, per-detector decay analysis
✅ Intelligent trend detection i weight adjustment
✅ Production-ready error handling i fallbacks

Workflow:
1. Load current component weights
2. Compute decay factors based on historical performance
3. Apply decay factors to weights  
4. Update weights with new feedback
5. Save updated weights with metadata
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

# Import decay system and feedback components
try:
    from feedback_loop.weights_decay import get_decay_system, apply_decay_to_component_weights
    from feedback_loop.component_feedback_trainer import get_component_trainer
    from feedback_loop.weights_loader import get_weights_loader
    DECAY_AVAILABLE = True
except ImportError:
    DECAY_AVAILABLE = False
    logging.warning("[COMPONENT UPDATER] Decay system not available, using basic updates")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentScoreUpdater:
    """
    Enhanced component score updater z Self-Learning Decay integration
    """
    
    def __init__(self, feedback_dir: str = "feedback_loop"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
        self.enable_decay = DECAY_AVAILABLE
        self.component_weights_file = self.feedback_dir / "component_dynamic_weights.json"
        
        # Update configuration
        self.decay_update_frequency = 5  # Apply decay every 5 feedback updates
        self.min_feedback_for_decay = 10  # Minimum feedback entries before enabling decay
        
        self.update_counter = 0
        
        logger.info(f"[COMPONENT UPDATER] Initialized with decay={'enabled' if self.enable_decay else 'disabled'}")
    
    def load_current_weights(self) -> Dict[str, float]:
        """
        Ładuje current component weights z fallback na defaults
        
        Returns:
            Dict[str, float]: Current component weights
        """
        try:
            if self.enable_decay:
                weights_loader = get_weights_loader()
                return weights_loader.get_dynamic_component_weights()
            else:
                # Fallback loading
                if self.component_weights_file.exists():
                    with open(self.component_weights_file, 'r') as f:
                        raw_weights = json.load(f)
                    
                    # Filter numeric weights
                    weights = {k: v for k, v in raw_weights.items() 
                             if k not in ["last_updated", "total_updates"] and isinstance(v, (int, float))}
                    return weights
                else:
                    return self._get_default_weights()
                    
        except Exception as e:
            logger.error(f"[COMPONENT UPDATER] Error loading weights: {e}")
            return self._get_default_weights()
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default component weights"""
        return {
            "dex": 1.0, "whale": 1.0, "trust": 1.0, "id": 1.0,
            "diamond": 1.0, "californium": 1.0, "clip": 1.0, "gnn": 1.0,
            "consensus": 1.0, "rl_agent": 1.0
        }
    
    def update_dynamic_weights_with_decay(self, new_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Główna funkcja aktualizująca dynamic weights z Self-Learning Decay
        
        Args:
            new_feedback: Nowy feedback do przetworzenia (opcjonalny)
            
        Returns:
            Dict[str, float]: Updated component weights
        """
        try:
            logger.info(f"[COMPONENT UPDATER] Starting weight update with decay (counter: {self.update_counter})")
            
            # Load current weights
            current_weights = self.load_current_weights()
            
            # Process new feedback if provided
            if new_feedback and self.enable_decay:
                trainer = get_component_trainer()
                trainer.update_component_scores_from_result(new_feedback)
                logger.info(f"[COMPONENT UPDATER] Processed feedback for {new_feedback.get('symbol', 'Unknown')}")
            
            # Apply decay system (co kilka aktualizacji)
            if self.enable_decay and self.should_apply_decay():
                logger.info(f"[COMPONENT UPDATER] Applying Self-Learning Decay...")
                
                decay_system = get_decay_system()
                decay_factors = decay_system.compute_all_decay_factors()
                
                if decay_factors:
                    decayed_weights = decay_system.apply_decay_to_weights(current_weights, decay_factors)
                    
                    # Log significant changes
                    significant_changes = []
                    for component in current_weights:
                        old_weight = current_weights[component]
                        new_weight = decayed_weights[component]
                        
                        if abs(new_weight - old_weight) > 0.05:
                            change_pct = ((new_weight - old_weight) / old_weight) * 100
                            significant_changes.append(f"{component}: {old_weight:.3f}→{new_weight:.3f} ({change_pct:+.1f}%)")
                    
                    if significant_changes:
                        logger.info(f"[DECAY APPLIED] Significant changes: {', '.join(significant_changes)}")
                    
                    current_weights = decayed_weights
                else:
                    logger.info(f"[DECAY SKIP] No decay factors computed")
            
            # Save updated weights
            self.save_updated_weights(current_weights)
            
            self.update_counter += 1
            logger.info(f"[COMPONENT UPDATER] Weight update complete: {len(current_weights)} components")
            
            return current_weights
            
        except Exception as e:
            logger.error(f"[COMPONENT UPDATER] Error updating weights with decay: {e}")
            return self.load_current_weights()  # Return current weights as fallback
    
    def should_apply_decay(self) -> bool:
        """
        Określa czy decay powinien być zastosowany w tym update cycle
        
        Returns:
            bool: True jeśli decay powinien być zastosowany
        """
        try:
            # Apply decay every N updates
            if self.update_counter % self.decay_update_frequency == 0:
                return True
            
            # Check if enough feedback accumulated
            if self.enable_decay:
                decay_system = get_decay_system()
                component_history = decay_system.load_component_history()
                
                # If any component has enough history, enable decay
                for history in component_history.values():
                    if len(history) >= self.min_feedback_for_decay:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"[COMPONENT UPDATER] Error checking decay condition: {e}")
            return False
    
    def save_updated_weights(self, weights: Dict[str, float]) -> bool:
        """
        Zapisuje updated weights do pliku z metadata
        
        Args:
            weights: Updated component weights
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare weights with metadata
            weights_with_metadata = weights.copy()
            weights_with_metadata.update({
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_updates": weights_with_metadata.get("total_updates", 0) + 1,
                "decay_enabled": self.enable_decay,
                "update_counter": self.update_counter
            })
            
            # Save to file
            with open(self.component_weights_file, 'w') as f:
                json.dump(weights_with_metadata, f, indent=2)
            
            logger.info(f"[COMPONENT UPDATER] Saved updated weights: {self.component_weights_file}")
            return True
            
        except Exception as e:
            logger.error(f"[COMPONENT UPDATER] Error saving weights: {e}")
            return False
    
    def force_decay_update(self) -> Dict[str, float]:
        """
        Wymusza immediate decay update bez względu na frequency
        
        Returns:
            Dict[str, float]: Updated weights po decay
        """
        try:
            logger.info(f"[COMPONENT UPDATER] Forcing immediate decay update...")
            
            current_weights = self.load_current_weights()
            
            if self.enable_decay:
                updated_weights = apply_decay_to_component_weights(current_weights)
                self.save_updated_weights(updated_weights)
                
                logger.info(f"[COMPONENT UPDATER] Forced decay update complete")
                return updated_weights
            else:
                logger.warning(f"[COMPONENT UPDATER] Decay system not available for forced update")
                return current_weights
                
        except Exception as e:
            logger.error(f"[COMPONENT UPDATER] Error in forced decay update: {e}")
            return current_weights
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """
        Pobierz statystyki update system
        
        Returns:
            Dict[str, Any]: Update system statistics
        """
        try:
            current_weights = self.load_current_weights()
            
            stats = {
                "update_counter": self.update_counter,
                "decay_enabled": self.enable_decay,
                "component_count": len([k for k in current_weights.keys() 
                                      if k not in ["last_updated", "total_updates", "decay_enabled", "update_counter"]]),
                "last_update": current_weights.get("last_updated", "never"),
                "total_updates": current_weights.get("total_updates", 0),
                "weights_file_exists": self.component_weights_file.exists()
            }
            
            if self.enable_decay:
                from feedback_loop.weights_decay import get_decay_system_stats
                decay_stats = get_decay_system_stats()
                stats["decay_statistics"] = decay_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"[COMPONENT UPDATER] Error getting statistics: {e}")
            return {"error": str(e)}


# Global instance and convenience functions
_component_updater = None

def get_component_updater() -> ComponentScoreUpdater:
    """Get global component updater instance"""
    global _component_updater
    if _component_updater is None:
        _component_updater = ComponentScoreUpdater()
    return _component_updater

def update_dynamic_weights_with_decay(new_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Główna funkcja aktualizująca component weights z Self-Learning Decay
    
    Args:
        new_feedback: Nowy feedback do przetworzenia
        
    Returns:
        Dict[str, float]: Updated component weights
    """
    return get_component_updater().update_dynamic_weights_with_decay(new_feedback)

def force_component_decay_update() -> Dict[str, float]:
    """Force immediate decay update"""
    return get_component_updater().force_decay_update()

def get_component_update_stats() -> Dict[str, Any]:
    """Get component update system statistics"""
    return get_component_updater().get_update_statistics()


if __name__ == "__main__":
    # Test Component Score Updater with Decay
    print("=== COMPONENT SCORE UPDATER V4 WITH DECAY TEST ===")
    
    updater = ComponentScoreUpdater()
    
    print("Testing weight loading...")
    current_weights = updater.load_current_weights()
    print(f"Current weights: {current_weights}")
    
    print("Testing feedback processing with decay...")
    test_feedback = {
        "symbol": "TESTUSDT",
        "detector": "CaliforniumWhale",
        "scores": {"diamond": 0.85, "clip": 0.72, "whale": 0.58},
        "was_successful": True,
        "price_change_2h": 12.3
    }
    
    updated_weights = updater.update_dynamic_weights_with_decay(test_feedback)
    print(f"Updated weights: {updated_weights}")
    
    print("Testing forced decay update...")
    forced_weights = updater.force_decay_update()
    print(f"Forced decay weights: {forced_weights}")
    
    print("Testing update statistics...")
    stats = updater.get_update_statistics()
    print(f"Update stats: {stats}")
    
    print("=== TEST COMPLETE ===")