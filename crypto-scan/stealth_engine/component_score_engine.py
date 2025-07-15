#!/usr/bin/env python3
"""
🎯 COMPONENT SCORE ENGINE V4 - Dynamic Component-Aware Scoring System
======================================================================

Component Score Engine - Aplikuje dynamic component weights do wszystkich
detektorów w systemie stealth_engine z comprehensive component tracking.

Integracja z detektorami:
✅ ClassicStealth - whale, dex, trust, id components  
✅ DiamondWhale AI - diamond component z temporal graph analysis
✅ CaliforniumWhale AI - californium component z mastermind detection
✅ WhaleCLIP - clip component z behavioral analysis
✅ GraphGNN - gnn component z graph neural network
✅ MultiAgentConsensus - consensus component z multi-detector fusion
✅ RLAgentV3 - rl_agent component z reinforcement learning

Kluczowe funkcjonalności:
- Dynamic weight application z learned feedback
- Component breakdown tracking dla każdego detektora
- Intelligent fallback na default weights
- Performance logging i monitoring
- Integration z unified alert system
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

# Import feedback system components
try:
    from feedback_loop.weights_loader import get_dynamic_component_weights, get_component_weights_for_detector
    from feedback_loop.component_feedback_trainer import get_component_trainer
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    logging.warning("[COMPONENT ENGINE] Feedback system not available, using fallback weights")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentScoreEngine:
    """
    Engine dla dynamic component scoring z learned weights integration
    """
    
    def __init__(self, enable_feedback: bool = True):
        self.enable_feedback = enable_feedback and FEEDBACK_AVAILABLE
        
        # Default component weights (fallback)
        self.default_weights = {
            "dex": 1.0, "whale": 1.0, "trust": 1.0, "id": 1.0,
            "diamond": 1.0, "californium": 1.0, "clip": 1.0, "gnn": 1.0,
            "consensus": 1.0, "rl_agent": 1.0
        }
        
        logger.info(f"[COMPONENT ENGINE] Initialized with feedback={'enabled' if self.enable_feedback else 'disabled'}")
    
    def apply_dynamic_weights_to_components(self, component_scores: Dict[str, float], 
                                          detector_name: str = "Unknown") -> Tuple[Dict[str, float], float]:
        """
        Aplikuje dynamic component weights do component scores
        
        Args:
            component_scores: Dict komponentów i ich raw scores
            detector_name: Nazwa detektora dla logowania
            
        Returns:
            Tuple[Dict[str, float], float]: (weighted_scores, final_total_score)
        """
        try:
            # Get dynamic weights from feedback system
            if self.enable_feedback:
                weights = get_dynamic_component_weights()
            else:
                weights = self.default_weights.copy()
            
            # Apply weights to each component
            weighted_scores = {}
            total_score = 0.0
            
            for component, score in component_scores.items():
                if isinstance(score, (int, float)) and score > 0:
                    weight = weights.get(component, 1.0)
                    weighted_score = score * weight
                    weighted_scores[component] = weighted_score
                    total_score += weighted_score
                    
                    logger.debug(f"[COMPONENT ENGINE] {detector_name}.{component}: {score:.3f} × {weight:.3f} = {weighted_score:.3f}")
                else:
                    weighted_scores[component] = 0.0
            
            # Log component weighting summary
            active_components = [comp for comp, score in weighted_scores.items() if score > 0]
            logger.info(f"[COMPONENT ENGINE] {detector_name}: Applied weights to {len(active_components)} components → total={total_score:.3f}")
            
            return weighted_scores, total_score
            
        except Exception as e:
            logger.error(f"[COMPONENT ENGINE] Error applying weights for {detector_name}: {e}")
            # Fallback: return original scores without weighting
            total = sum(score for score in component_scores.values() if isinstance(score, (int, float)))
            return component_scores.copy(), total
    
    def calculate_component_breakdown_classic_stealth(self, stealth_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Ekstraktuje component breakdown z Classic Stealth Engine result
        
        Args:
            stealth_result: Wynik z compute_stealth_score()
            
        Returns:
            Dict[str, float]: Component scores (dex, whale, trust, id)
        """
        try:
            # Extract components from stealth result
            components = {
                "dex": stealth_result.get("dex_inflow", 0.0),
                "whale": stealth_result.get("whale_ping", 0.0), 
                "trust": stealth_result.get("trust_boost", 0.0),
                "id": stealth_result.get("identity_boost", 0.0)
            }
            
            # Ensure numeric values
            for key in components:
                if not isinstance(components[key], (int, float)):
                    components[key] = 0.0
            
            logger.info(f"[COMPONENT ENGINE] Classic Stealth breakdown: {components}")
            return components
            
        except Exception as e:
            logger.error(f"[COMPONENT ENGINE] Error extracting classic stealth breakdown: {e}")
            return {"dex": 0.0, "whale": 0.0, "trust": 0.0, "id": 0.0}
    
    def calculate_component_breakdown_ai_detectors(self, detection_result: Dict[str, Any], 
                                                  detector_type: str) -> Dict[str, float]:
        """
        Ekstraktuje component breakdown z AI detektorów (Diamond, Californium, WhaleCLIP, GNN)
        
        Args:
            detection_result: Wynik z AI detektora
            detector_type: Typ detektora ("diamond", "californium", "clip", "gnn")
            
        Returns:
            Dict[str, float]: Component scores specyficzne dla detektora
        """
        try:
            components = {}
            
            if detector_type == "diamond":
                # DiamondWhale AI - temporal graph analysis
                components = {
                    "diamond": detection_result.get("diamond_score", 0.0),
                    "whale": detection_result.get("whale_correlation", 0.0),  # Temporal whale patterns
                    "trust": detection_result.get("trust_factor", 0.0)       # Address trust from temporal analysis
                }
            
            elif detector_type == "californium":
                # CaliforniumWhale AI - mastermind detection
                components = {
                    "californium": detection_result.get("californium_score", 0.0),
                    "trust": detection_result.get("mastermind_trust", 0.0),   # Mastermind activity trust
                    "whale": detection_result.get("coordinated_whale", 0.0)  # Coordinated whale activity
                }
            
            elif detector_type == "clip":
                # WhaleCLIP - behavioral analysis
                components = {
                    "clip": detection_result.get("clip_score", 0.0),
                    "whale": detection_result.get("behavioral_whale", 0.0),   # Behavioral whale indicators
                    "trust": detection_result.get("pattern_trust", 0.0)      # Pattern reliability
                }
            
            elif detector_type == "gnn":
                # GraphGNN - graph neural network
                components = {
                    "gnn": detection_result.get("gnn_score", 0.0),
                    "whale": detection_result.get("graph_whale", 0.0),       # Graph-detected whale activity
                    "trust": detection_result.get("network_trust", 0.0)     # Network trust signals
                }
            
            else:
                # Generic AI detector
                score = detection_result.get("score", 0.0)
                components = {detector_type: score}
            
            # Ensure numeric values
            for key in components:
                if not isinstance(components[key], (int, float)):
                    components[key] = 0.0
            
            logger.info(f"[COMPONENT ENGINE] {detector_type.upper()} AI breakdown: {components}")
            return components
            
        except Exception as e:
            logger.error(f"[COMPONENT ENGINE] Error extracting {detector_type} breakdown: {e}")
            return {detector_type: 0.0}
    
    def apply_component_weights_to_final_score(self, all_component_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Aplikuje component weights do wszystkich detektorów i oblicza final unified score
        
        Args:
            all_component_scores: Dict[detector_name, Dict[component, score]]
            
        Returns:
            Dict[str, Any]: Unified result z weighted scores i component breakdown
        """
        try:
            weighted_detector_results = {}
            total_weighted_score = 0.0
            all_components_aggregated = {}
            
            # Process each detector separately
            for detector_name, component_scores in all_component_scores.items():
                weighted_scores, detector_total = self.apply_dynamic_weights_to_components(
                    component_scores, detector_name
                )
                
                weighted_detector_results[detector_name] = {
                    "weighted_scores": weighted_scores,
                    "total_score": detector_total
                }
                
                total_weighted_score += detector_total
                
                # Aggregate components across detectors
                for component, score in weighted_scores.items():
                    if component not in all_components_aggregated:
                        all_components_aggregated[component] = 0.0
                    all_components_aggregated[component] += score
            
            # Generate final result
            result = {
                "final_weighted_score": round(total_weighted_score, 3),
                "detector_breakdown": weighted_detector_results,
                "component_totals": {comp: round(score, 3) for comp, score in all_components_aggregated.items()},
                "active_detectors": len(weighted_detector_results),
                "active_components": len([comp for comp, score in all_components_aggregated.items() if score > 0]),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"[COMPONENT ENGINE] Final weighted score: {total_weighted_score:.3f} from {len(weighted_detector_results)} detectors")
            return result
            
        except Exception as e:
            logger.error(f"[COMPONENT ENGINE] Error calculating final weighted score: {e}")
            return {
                "final_weighted_score": 0.0,
                "detector_breakdown": {},
                "component_totals": {},
                "error": str(e)
            }
    
    def log_component_feedback_effectiveness(self) -> Dict[str, Any]:
        """
        Loguje aktualną skuteczność komponentów z feedback system
        
        Returns:
            Dict[str, Any]: Component effectiveness percentages
        """
        try:
            if not self.enable_feedback:
                return {"feedback_disabled": True}
            
            # Get current weights and effectiveness
            weights = get_dynamic_component_weights()
            
            # Format effectiveness log
            effectiveness_log = []
            for component, weight in weights.items():
                effectiveness_pct = int((weight - 0.5) * 100) if weight >= 0.5 else int((weight - 1.0) * 100)
                effectiveness_log.append(f"{component}={effectiveness_pct:+d}%")
            
            effectiveness_summary = ", ".join(effectiveness_log)
            logger.info(f"[COMPONENT FEEDBACK] {effectiveness_summary}")
            
            # Format booster weights
            booster_log = []
            for component, weight in weights.items():
                if abs(weight - 1.0) > 0.1:  # Only log significantly different weights
                    booster_log.append(f"{component}={weight:.1f}")
            
            if booster_log:
                booster_summary = ", ".join(booster_log)
                logger.info(f"[BOOSTER] Dynamic weights applied: {booster_summary}")
            
            return {
                "component_weights": weights,
                "effectiveness_summary": effectiveness_summary,
                "significant_adjustments": len(booster_log)
            }
            
        except Exception as e:
            logger.error(f"[COMPONENT ENGINE] Error logging effectiveness: {e}")
            return {"error": str(e)}


# Global instance for easy access
_component_engine = None

def get_component_engine() -> ComponentScoreEngine:
    """Get global component score engine instance"""
    global _component_engine
    if _component_engine is None:
        _component_engine = ComponentScoreEngine()
    return _component_engine

def apply_dynamic_component_weights(component_scores: Dict[str, float], detector_name: str = "Unknown") -> Tuple[Dict[str, float], float]:
    """
    Przed końcowym score - aplikuje dynamic component weights
    
    Args:
        component_scores: Component scores dict
        detector_name: Nazwa detektora
        
    Returns:
        Tuple[Dict[str, float], float]: (weighted_scores, final_score)
        
    Example:
        weights = get_dynamic_component_weights()
        final_score = sum(component_score[k] * weights.get(k, 1.0) for k in component_score)
    """
    return get_component_engine().apply_dynamic_weights_to_components(component_scores, detector_name)

def extract_classic_stealth_components(stealth_result: Dict[str, Any]) -> Dict[str, float]:
    """Extract component breakdown from Classic Stealth result"""
    return get_component_engine().calculate_component_breakdown_classic_stealth(stealth_result)

def extract_ai_detector_components(detection_result: Dict[str, Any], detector_type: str) -> Dict[str, float]:
    """Extract component breakdown from AI detector result"""
    return get_component_engine().calculate_component_breakdown_ai_detectors(detection_result, detector_type)

def calculate_unified_weighted_score(all_component_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Calculate unified score with dynamic component weights applied"""
    return get_component_engine().apply_component_weights_to_final_score(all_component_scores)

def log_component_effectiveness():
    """Log current component effectiveness percentages"""
    return get_component_engine().log_component_feedback_effectiveness()


if __name__ == "__main__":
    # Test Component Score Engine
    print("=== COMPONENT SCORE ENGINE V4 TEST ===")
    
    engine = ComponentScoreEngine()
    
    # Test Classic Stealth components
    print("Testing Classic Stealth component extraction...")
    stealth_result = {
        "score": 2.45,
        "dex_inflow": 0.8,
        "whale_ping": 1.2,
        "trust_boost": 0.3,
        "identity_boost": 0.15
    }
    classic_components = engine.calculate_component_breakdown_classic_stealth(stealth_result)
    print(f"Classic components: {classic_components}")
    
    # Test AI detector components
    print("\nTesting AI detector component extraction...")
    diamond_result = {"diamond_score": 0.82, "whale_correlation": 0.65, "trust_factor": 0.4}
    diamond_components = engine.calculate_component_breakdown_ai_detectors(diamond_result, "diamond")
    print(f"Diamond components: {diamond_components}")
    
    # Test dynamic weight application
    print("\nTesting dynamic weight application...")
    weighted_scores, total = engine.apply_dynamic_weights_to_components(classic_components, "ClassicStealth")
    print(f"Weighted scores: {weighted_scores}")
    print(f"Total weighted score: {total}")
    
    # Test unified scoring
    print("\nTesting unified weighted scoring...")
    all_scores = {
        "ClassicStealth": classic_components,
        "DiamondWhale": diamond_components
    }
    unified_result = engine.apply_component_weights_to_final_score(all_scores)
    print(f"Unified result: {unified_result}")
    
    # Test effectiveness logging
    print("\nTesting effectiveness logging...")
    effectiveness = engine.log_component_feedback_effectiveness()
    print(f"Effectiveness: {effectiveness}")
    
    print("=== TEST COMPLETE ===")