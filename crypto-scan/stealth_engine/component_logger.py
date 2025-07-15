#!/usr/bin/env python3
"""
🎯 COMPONENT LOGGER V4 - Enhanced Component Feedback Logging System
===================================================================

Component Logger - Dodatkowe logi w stealth_engine dla component feedback
tracking z comprehensive logging formatów i performance monitoring.

Log formats zgodnie z user specification:
[COMPONENT FEEDBACK] whale=83%, dex=61%, id=74%, diamond=91%, clip=79%, gnn=87%
[BOOSTER] Dynamic weights applied: whale=1.2, dex=0.7, gnn=1.1

Funkcjonalności:
✅ Component effectiveness percentage logging
✅ Dynamic weight application logging  
✅ Performance monitoring i statistics
✅ Integration z unified alert system
✅ Detector-specific component tracking
✅ Historical trend analysis logging
"""

import logging
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Setup enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComponentLogger:
    """
    Enhanced logger dla component feedback i effectiveness tracking
    """
    
    def __init__(self, log_dir: str = "logs", enable_detailed_logging: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.enable_detailed_logging = enable_detailed_logging
        
        # Component log files
        self.component_feedback_log = self.log_dir / "component_feedback.log"
        self.booster_weights_log = self.log_dir / "booster_weights.log" 
        self.effectiveness_log = self.log_dir / "component_effectiveness.log"
        
        # Performance tracking
        self.performance_stats = {
            "total_logs": 0,
            "feedback_logs": 0,
            "booster_logs": 0,
            "effectiveness_logs": 0
        }
        
        logger.info(f"[COMPONENT LOGGER] Initialized with log_dir={log_dir}, detailed={'enabled' if enable_detailed_logging else 'disabled'}")
    
    def log_component_feedback_percentages(self, component_effectiveness: Dict[str, float], 
                                         symbol: str = "", detector: str = "") -> None:
        """
        Loguje component effectiveness percentages w formacie user specification
        
        Args:
            component_effectiveness: Dict[component, effectiveness_percentage]
            symbol: Token symbol (opcjonalnie)
            detector: Detector name (opcjonalnie)
        
        Format: [COMPONENT FEEDBACK] whale=83%, dex=61%, id=74%, diamond=91%, clip=79%, gnn=87%
        """
        try:
            # Format component percentages
            percentage_strings = []
            for component, effectiveness in component_effectiveness.items():
                if isinstance(effectiveness, (int, float)):
                    pct = int(round(effectiveness))
                    percentage_strings.append(f"{component}={pct}%")
            
            if percentage_strings:
                feedback_summary = ", ".join(percentage_strings)
                
                # Create log message
                prefix = f"[COMPONENT FEEDBACK]"
                if symbol and detector:
                    prefix += f" {symbol} ({detector})"
                elif symbol:
                    prefix += f" {symbol}"
                elif detector:
                    prefix += f" ({detector})"
                
                log_message = f"{prefix} {feedback_summary}"
                
                # Log to console and file
                logger.info(log_message)
                
                if self.enable_detailed_logging:
                    self._write_to_file(self.component_feedback_log, log_message)
                
                self.performance_stats["feedback_logs"] += 1
                self.performance_stats["total_logs"] += 1
                
        except Exception as e:
            logger.error(f"[COMPONENT LOGGER] Error logging feedback percentages: {e}")
    
    def log_dynamic_weights_applied(self, applied_weights: Dict[str, float], 
                                  symbol: str = "", detector: str = "") -> None:
        """
        Loguje applied dynamic weights w formacie user specification
        
        Args:
            applied_weights: Dict[component, weight] tylko dla znaczących zmian
            symbol: Token symbol (opcjonalnie)
            detector: Detector name (opcjonalnie)
        
        Format: [BOOSTER] Dynamic weights applied: whale=1.2, dex=0.7, gnn=1.1
        """
        try:
            # Filter significant weight changes (> 10% difference from 1.0)
            significant_weights = {}
            for component, weight in applied_weights.items():
                if isinstance(weight, (int, float)) and abs(weight - 1.0) > 0.1:
                    significant_weights[component] = weight
            
            if significant_weights:
                # Format weight strings
                weight_strings = []
                for component, weight in significant_weights.items():
                    weight_strings.append(f"{component}={weight:.1f}")
                
                weights_summary = ", ".join(weight_strings)
                
                # Create log message
                prefix = f"[BOOSTER]"
                if symbol and detector:
                    prefix += f" {symbol} ({detector})"
                elif symbol:
                    prefix += f" {symbol}"
                elif detector:
                    prefix += f" ({detector})"
                
                log_message = f"{prefix} Dynamic weights applied: {weights_summary}"
                
                # Log to console and file
                logger.info(log_message)
                
                if self.enable_detailed_logging:
                    self._write_to_file(self.booster_weights_log, log_message)
                
                self.performance_stats["booster_logs"] += 1
                self.performance_stats["total_logs"] += 1
            else:
                logger.debug(f"[COMPONENT LOGGER] No significant weight changes to log for {symbol}")
                
        except Exception as e:
            logger.error(f"[COMPONENT LOGGER] Error logging dynamic weights: {e}")
    
    def log_component_effectiveness_summary(self, detector_name: str, 
                                          component_breakdown: Dict[str, Any],
                                          symbol: str = "") -> None:
        """
        Loguje comprehensive component effectiveness summary dla detektora
        
        Args:
            detector_name: Nazwa detektora
            component_breakdown: Breakdown komponentów z effectiveness data
            symbol: Token symbol (opcjonalnie)
        """
        try:
            # Extract effectiveness data
            effectiveness_data = {}
            weighted_scores = component_breakdown.get("weighted_scores", {})
            raw_scores = component_breakdown.get("raw_scores", {})
            
            for component in weighted_scores:
                raw_score = raw_scores.get(component, 0.0)
                weighted_score = weighted_scores.get(component, 0.0)
                
                if raw_score > 0:
                    # Calculate effectiveness as weighted/raw ratio
                    effectiveness = (weighted_score / raw_score) * 100 if raw_score > 0 else 100
                    effectiveness_data[component] = effectiveness
            
            if effectiveness_data:
                # Log component feedback percentages
                self.log_component_feedback_percentages(effectiveness_data, symbol, detector_name)
                
                # Log applied weights if significant
                weights = {comp: weighted_scores.get(comp, 0.0) / raw_scores.get(comp, 1.0) 
                          for comp in effectiveness_data if raw_scores.get(comp, 0.0) > 0}
                self.log_dynamic_weights_applied(weights, symbol, detector_name)
                
                # Detailed effectiveness log
                if self.enable_detailed_logging:
                    detailed_log = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "detector": detector_name,
                        "component_breakdown": component_breakdown,
                        "effectiveness_percentages": effectiveness_data
                    }
                    
                    self._write_to_file(self.effectiveness_log, json.dumps(detailed_log))
                    self.performance_stats["effectiveness_logs"] += 1
                
        except Exception as e:
            logger.error(f"[COMPONENT LOGGER] Error logging effectiveness summary: {e}")
    
    def log_unified_component_results(self, unified_result: Dict[str, Any], symbol: str = "") -> None:
        """
        Loguje unified component results z multiple detektorów
        
        Args:
            unified_result: Result z calculate_unified_weighted_score()
            symbol: Token symbol
        """
        try:
            detector_breakdown = unified_result.get("detector_breakdown", {})
            component_totals = unified_result.get("component_totals", {})
            final_score = unified_result.get("final_weighted_score", 0.0)
            
            # Log overall component effectiveness
            if component_totals:
                # Convert component totals to percentages (relative to final score)
                if final_score > 0:
                    component_percentages = {
                        comp: (score / final_score) * 100 
                        for comp, score in component_totals.items() if score > 0
                    }
                    self.log_component_feedback_percentages(component_percentages, symbol, "Unified")
            
            # Log per-detector results
            for detector_name, detector_data in detector_breakdown.items():
                weighted_scores = detector_data.get("weighted_scores", {})
                total_score = detector_data.get("total_score", 0.0)
                
                if total_score > 0 and weighted_scores:
                    # Calculate detector-specific effectiveness
                    detector_effectiveness = {
                        comp: (score / total_score) * 100 
                        for comp, score in weighted_scores.items() if score > 0
                    }
                    
                    if detector_effectiveness:
                        self.log_component_feedback_percentages(detector_effectiveness, symbol, detector_name)
            
            # Log final unified summary
            logger.info(f"[UNIFIED SCORING] {symbol}: Final weighted score={final_score:.3f} from {len(detector_breakdown)} detectors")
            
        except Exception as e:
            logger.error(f"[COMPONENT LOGGER] Error logging unified results: {e}")
    
    def _write_to_file(self, file_path: Path, message: str) -> None:
        """Write message to log file with timestamp"""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            formatted_message = f"[{timestamp}] {message}\n"
            
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(formatted_message)
                
        except Exception as e:
            logger.error(f"[COMPONENT LOGGER] Error writing to file {file_path}: {e}")
    
    def get_logging_statistics(self) -> Dict[str, Any]:
        """Get component logging statistics"""
        try:
            # File sizes
            file_stats = {}
            for log_file in [self.component_feedback_log, self.booster_weights_log, self.effectiveness_log]:
                if log_file.exists():
                    file_stats[log_file.name] = {
                        "size_bytes": log_file.stat().st_size,
                        "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                    }
                else:
                    file_stats[log_file.name] = {"size_bytes": 0, "modified": None}
            
            return {
                "performance_stats": self.performance_stats,
                "file_stats": file_stats,
                "detailed_logging": self.enable_detailed_logging,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"[COMPONENT LOGGER] Error getting statistics: {e}")
            return {"error": str(e)}


# Global instance for easy access
_component_logger = None

def get_component_logger() -> ComponentLogger:
    """Get global component logger instance"""
    global _component_logger
    if _component_logger is None:
        _component_logger = ComponentLogger()
    return _component_logger

def log_component_feedback(component_effectiveness: Dict[str, float], symbol: str = "", detector: str = ""):
    """Log component feedback percentages - zgodnie z user specification"""
    get_component_logger().log_component_feedback_percentages(component_effectiveness, symbol, detector)

def log_dynamic_weights(applied_weights: Dict[str, float], symbol: str = "", detector: str = ""):
    """Log dynamic weights applied - zgodnie z user specification"""
    get_component_logger().log_dynamic_weights_applied(applied_weights, symbol, detector)

def log_component_effectiveness_summary(detector_name: str, component_breakdown: Dict[str, Any], symbol: str = ""):
    """Log comprehensive component effectiveness summary"""
    get_component_logger().log_component_effectiveness_summary(detector_name, component_breakdown, symbol)

def log_unified_component_results(unified_result: Dict[str, Any], symbol: str = ""):
    """Log unified component results from multiple detectors"""
    get_component_logger().log_unified_component_results(unified_result, symbol)

def get_component_logging_stats() -> Dict[str, Any]:
    """Get component logging statistics"""
    return get_component_logger().get_logging_statistics()


if __name__ == "__main__":
    # Test Component Logger
    print("=== COMPONENT LOGGER V4 TEST ===")
    
    logger_instance = ComponentLogger()
    
    # Test component feedback logging
    print("Testing component feedback logging...")
    effectiveness = {"whale": 83, "dex": 61, "id": 74, "diamond": 91, "clip": 79, "gnn": 87}
    logger_instance.log_component_feedback_percentages(effectiveness, "TESTUSDT", "CaliforniumWhale")
    
    # Test dynamic weights logging
    print("Testing dynamic weights logging...")
    weights = {"whale": 1.2, "dex": 0.7, "gnn": 1.1, "diamond": 0.95}  # Only significant changes logged
    logger_instance.log_dynamic_weights_applied(weights, "TESTUSDT", "ClassicStealth")
    
    # Test effectiveness summary
    print("Testing effectiveness summary...")
    breakdown = {
        "weighted_scores": {"dex": 0.64, "whale": 1.44, "trust": 0.27},
        "raw_scores": {"dex": 0.8, "whale": 1.2, "trust": 0.3},
        "total_score": 2.35
    }
    logger_instance.log_component_effectiveness_summary("ClassicStealth", breakdown, "TESTUSDT")
    
    # Test unified results logging
    print("Testing unified results logging...")
    unified_result = {
        "final_weighted_score": 3.85,
        "detector_breakdown": {
            "ClassicStealth": {"weighted_scores": {"dex": 0.8, "whale": 1.2}, "total_score": 2.0},
            "DiamondWhale": {"weighted_scores": {"diamond": 1.5, "trust": 0.35}, "total_score": 1.85}
        },
        "component_totals": {"dex": 0.8, "whale": 1.2, "diamond": 1.5, "trust": 0.35}
    }
    logger_instance.log_unified_component_results(unified_result, "TESTUSDT")
    
    # Test statistics
    print("Testing logging statistics...")
    stats = logger_instance.get_logging_statistics()
    print(f"Logging stats: {stats}")
    
    print("=== TEST COMPLETE ===")