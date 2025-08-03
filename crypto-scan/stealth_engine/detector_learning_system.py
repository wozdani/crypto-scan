#!/usr/bin/env python3
"""
Detector Self-Learning System - Inteligentny System Samouczenia się Detektorów
Adaptive score adjustment based on past decisions and explore mode feedback

Funkcje:
- Indywidualne uczenie się każdego detektora (CaliforniumWhale, DiamondWhale, WhaleCLIP, StealthEngine)
- Score adaptation na podstawie skuteczności przeszłych decyzji
- Explore mode learning i feedback integration
- Performance tracking i accuracy monitoring
- Dynamic threshold adjustment per detector
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Konfiguracja
LEARNING_DATA_DIR = "crypto-scan/cache/detector_learning/"
DECISION_HISTORY_FILE = "crypto-scan/cache/detector_decisions.json"
EXPLORE_FEEDBACK_FILE = "crypto-scan/cache/explore_mode_feedback.json"
PERFORMANCE_LOG_FILE = "crypto-scan/cache/detector_performance.json"

# Learning parameters
LEARNING_RATE = 0.05  # Jak szybko detektory adaptują swoje score
MIN_DECISIONS_FOR_LEARNING = 5  # Minimum decyzji przed rozpoczęciem uczenia
EXPLORE_MODE_WEIGHT = 0.3  # Waga explore mode w learning
DECISION_HISTORY_DAYS = 14  # Dni historii do analizy

@dataclass
class DetectorDecision:
    """Pojedyncza decyzja detektora do uczenia"""
    detector_name: str
    symbol: str
    timestamp: str
    original_score: float
    adjusted_score: float
    decision: str  # BUY/HOLD/AVOID
    was_correct: Optional[bool] = None
    profit_loss_pct: Optional[float] = None
    explore_mode: bool = False
    market_context: Optional[Dict] = None

@dataclass
class DetectorPerformance:
    """Performance metrics detektora"""
    detector_name: str
    total_decisions: int
    correct_decisions: int
    accuracy_rate: float
    avg_profit_loss: float
    explore_mode_accuracy: float
    last_updated: str
    score_adjustments: List[float]
    confidence_evolution: List[float]

class DetectorLearningSystem:
    """
    Główny system uczenia się detektorów
    """
    
    def __init__(self):
        self.ensure_directories()
        self.decision_history = self.load_decision_history()
        self.explore_feedback = self.load_explore_feedback()
        self.performance_data = self.load_performance_data()
        self.logger = logging.getLogger(__name__)
        
        # Bootstrap learning system with existing multi-agent data if no learning data exists
        if not self.decision_history and not self.performance_data:
            self.bootstrap_from_existing_data()
        
        # Default detector configurations
        self.detector_configs = {
            "CaliforniumWhale": {
                "base_threshold": 0.7,
                "learning_sensitivity": 0.8,
                "explore_mode_boost": 0.2
            },
            "DiamondWhale": {
                "base_threshold": 0.6,
                "learning_sensitivity": 0.7,
                "explore_mode_boost": 0.15
            },
            "WhaleCLIP": {
                "base_threshold": 0.8,
                "learning_sensitivity": 0.9,
                "explore_mode_boost": 0.25
            },
            "StealthEngine": {
                "base_threshold": 2.0,
                "learning_sensitivity": 0.6,
                "explore_mode_boost": 0.3
            }
        }
        
        print("[DETECTOR LEARNING] Intelligent self-learning system initialized for all detectors")
        print(f"[DETECTOR LEARNING] Current data: {len(self.decision_history)} decisions, {len(self.performance_data)} detector profiles")
    
    def bootstrap_from_existing_data(self):
        """Bootstrap learning system z istniejących multi-agent decisions i agent learning history"""
        print("[DETECTOR LEARNING] Bootstrapping system with existing decision data...")
        
        # Load multi-agent decisions
        multi_agent_file = "crypto-scan/cache/multi_agent_decisions.json"
        if os.path.exists(multi_agent_file):
            try:
                with open(multi_agent_file, 'r') as f:
                    multi_decisions = json.load(f)
                    
                bootstrapped_count = 0
                for decision in multi_decisions:
                    detector_name = decision.get("detector_name")
                    # Convert multi-agent decisions to detector decisions
                    if detector_name in self.detector_configs:
                        # Create synthetic detector decision from multi-agent data
                        decision_record = DetectorDecision(
                            detector_name=detector_name,
                            symbol="BOOTSTRAP",  # Placeholder since symbol not in multi-agent decisions
                            timestamp=decision.get("timestamp", datetime.now().isoformat()),
                            original_score=decision.get("score", 0.0),
                            adjusted_score=decision.get("score", 0.0),
                            decision=decision.get("final_decision", "HOLD"),
                            was_correct=None,  # We don't have outcomes yet
                            explore_mode=False,
                            market_context={"bootstrap": True}
                        )
                        self.decision_history.append(decision_record)
                        bootstrapped_count += 1
                        
                        if bootstrapped_count >= 20:  # Limit to avoid memory issues
                            break
                            
                print(f"[DETECTOR LEARNING] Bootstrapped {bootstrapped_count} decisions from multi-agent data")
                
                # Create initial performance data for all detectors
                for detector_name in self.detector_configs.keys():
                    detector_decisions = [d for d in self.decision_history if d.detector_name == detector_name]
                    if detector_decisions:
                        # Create initial performance record
                        self.performance_data[detector_name] = DetectorPerformance(
                            detector_name=detector_name,
                            total_decisions=len(detector_decisions),
                            correct_decisions=0,  # No outcomes yet
                            accuracy_rate=0.5,  # Start with neutral accuracy
                            avg_profit_loss=0.0,
                            explore_mode_accuracy=0.5,
                            last_updated=datetime.now().isoformat(),
                            score_adjustments=[],
                            confidence_evolution=[0.5]
                        )
                
                print(f"[DETECTOR LEARNING] Created initial performance profiles for {len(self.performance_data)} detectors")
                
                # Save bootstrapped data
                self.save_decision_history()
                self.save_performance_data()
                
            except Exception as e:
                print(f"[DETECTOR LEARNING] Error bootstrapping from multi-agent data: {e}")
    
    def ensure_directories(self):
        """Stwórz niezbędne katalogi"""
        os.makedirs(LEARNING_DATA_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(DECISION_HISTORY_FILE), exist_ok=True)
    
    def load_decision_history(self) -> List[DetectorDecision]:
        """Załaduj historię decyzji detektorów"""
        if os.path.exists(DECISION_HISTORY_FILE):
            try:
                with open(DECISION_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    return [DetectorDecision(**item) for item in data]
            except Exception as e:
                print(f"[DETECTOR LEARNING] Error loading decision history: {e}")
        return []
    
    def load_explore_feedback(self) -> List[Dict]:
        """Załaduj feedback z explore mode"""
        if os.path.exists(EXPLORE_FEEDBACK_FILE):
            try:
                with open(EXPLORE_FEEDBACK_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DETECTOR LEARNING] Error loading explore feedback: {e}")
        return []
    
    def load_performance_data(self) -> Dict[str, DetectorPerformance]:
        """Załaduj dane performance detektorów"""
        if os.path.exists(PERFORMANCE_LOG_FILE):
            try:
                with open(PERFORMANCE_LOG_FILE, 'r') as f:
                    data = json.load(f)
                    return {
                        name: DetectorPerformance(**perf_data) 
                        for name, perf_data in data.items()
                    }
            except Exception as e:
                print(f"[DETECTOR LEARNING] Error loading performance data: {e}")
        return {}
    
    def save_decision_history(self):
        """Zapisz historię decyzji"""
        try:
            data = [
                {
                    'detector_name': d.detector_name,
                    'symbol': d.symbol,
                    'timestamp': d.timestamp,
                    'original_score': d.original_score,
                    'adjusted_score': d.adjusted_score,
                    'decision': d.decision,
                    'was_correct': d.was_correct,
                    'profit_loss_pct': d.profit_loss_pct,
                    'explore_mode': d.explore_mode,
                    'market_context': d.market_context
                }
                for d in self.decision_history
            ]
            with open(DECISION_HISTORY_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[DETECTOR LEARNING] Error saving decision history: {e}")
    
    def save_performance_data(self):
        """Zapisz dane performance"""
        try:
            data = {
                name: {
                    'detector_name': perf.detector_name,
                    'total_decisions': perf.total_decisions,
                    'correct_decisions': perf.correct_decisions,
                    'accuracy_rate': perf.accuracy_rate,
                    'avg_profit_loss': perf.avg_profit_loss,
                    'explore_mode_accuracy': perf.explore_mode_accuracy,
                    'last_updated': perf.last_updated,
                    'score_adjustments': perf.score_adjustments,
                    'confidence_evolution': perf.confidence_evolution
                }
                for name, perf in self.performance_data.items()
            }
            with open(PERFORMANCE_LOG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[DETECTOR LEARNING] Error saving performance data: {e}")
    
    def record_detector_decision(
        self, 
        detector_name: str, 
        symbol: str, 
        original_score: float, 
        adjusted_score: float, 
        decision: str,
        explore_mode: bool = False,
        market_context: Optional[Dict] = None
    ):
        """
        Zapisz decyzję detektora dla późniejszego uczenia
        
        Args:
            detector_name: Nazwa detektora (CaliforniumWhale, DiamondWhale, WhaleCLIP, StealthEngine)
            symbol: Symbol tokena
            original_score: Oryginalny score przed adaptacją
            adjusted_score: Score po adaptacji (jeśli była)
            decision: Decyzja (BUY/HOLD/AVOID)
            explore_mode: Czy decyzja z explore mode
            market_context: Kontekst rynkowy (volume, price_change, etc.)
        """
        decision_record = DetectorDecision(
            detector_name=detector_name,
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            original_score=original_score,
            adjusted_score=adjusted_score,
            decision=decision,
            explore_mode=explore_mode,
            market_context=market_context or {}
        )
        
        self.decision_history.append(decision_record)
        
        # Ogranicz historię do ostatnich 1000 decyzji per detektor
        detector_decisions = [d for d in self.decision_history if d.detector_name == detector_name]
        if len(detector_decisions) > 1000:
            # Usuń najstarsze decyzje tego detektora
            cutoff_timestamp = sorted(detector_decisions, key=lambda x: x.timestamp)[-1000].timestamp
            self.decision_history = [
                d for d in self.decision_history 
                if d.detector_name != detector_name or d.timestamp >= cutoff_timestamp
            ]
        
        print(f"[DETECTOR LEARNING] Recorded decision: {detector_name} → {symbol} → {decision} (score: {original_score:.3f}→{adjusted_score:.3f}, explore: {explore_mode})")
        
        # Zapisz automatycznie co 10 decyzji
        if len(self.decision_history) % 10 == 0:
            self.save_decision_history()
    
    def update_decision_outcome(
        self, 
        detector_name: str, 
        symbol: str, 
        timestamp: str, 
        was_correct: bool, 
        profit_loss_pct: Optional[float] = None
    ):
        """
        Zaktualizuj wynik decyzji (po sprawdzeniu czy była trafna)
        
        Args:
            detector_name: Nazwa detektora
            symbol: Symbol tokena
            timestamp: Timestamp decyzji do aktualizacji
            was_correct: Czy decyzja była trafna
            profit_loss_pct: Procent zysku/straty (jeśli dostępny)
        """
        for decision in self.decision_history:
            if (decision.detector_name == detector_name and 
                decision.symbol == symbol and 
                decision.timestamp == timestamp):
                
                decision.was_correct = was_correct
                decision.profit_loss_pct = profit_loss_pct
                
                print(f"[DETECTOR LEARNING] Updated outcome: {detector_name} → {symbol} → {'✓' if was_correct else '✗'} ({profit_loss_pct:.2f}% P/L)")
                break
        
        # Przelicz performance metrics
        self.update_detector_performance(detector_name)
    
    def adapt_detector_score(
        self, 
        detector_name: str, 
        original_score: float, 
        symbol: str, 
        market_context: Optional[Dict] = None
    ) -> Tuple[float, str]:
        """
        Adaptuj score detektora na podstawie historical performance
        
        Args:
            detector_name: Nazwa detektora
            original_score: Oryginalny score
            symbol: Symbol tokena
            market_context: Kontekst rynkowy
            
        Returns:
            Tuple[float, str]: (adapted_score, adaptation_reason)
        """
        if detector_name not in self.detector_configs:
            return original_score, "detector_not_configured"
        
        config = self.detector_configs[detector_name]
        performance = self.performance_data.get(detector_name)
        
        if not performance or performance.total_decisions < MIN_DECISIONS_FOR_LEARNING:
            return original_score, "insufficient_learning_data"
        
        # Oblicz adaptation factor na podstawie accuracy
        accuracy_factor = performance.accuracy_rate
        explore_factor = performance.explore_mode_accuracy
        
        # Compute adaptive multiplier
        if accuracy_factor > 0.7:
            # High accuracy detector - boost confident signals
            if original_score > config["base_threshold"] * 0.8:
                adaptation_multiplier = 1.0 + (accuracy_factor - 0.7) * config["learning_sensitivity"]
            else:
                adaptation_multiplier = 1.0
        elif accuracy_factor < 0.4:
            # Low accuracy detector - reduce confidence
            adaptation_multiplier = 0.7 + (accuracy_factor / 0.4) * 0.3
        else:
            # Medium accuracy - slight adjustments
            adaptation_multiplier = 0.85 + (accuracy_factor - 0.4) / 0.3 * 0.3
        
        # Apply explore mode boost if applicable
        if explore_factor > accuracy_factor and explore_factor > 0.6:
            adaptation_multiplier += config["explore_mode_boost"]
        
        # Apply recent performance trend
        if len(performance.score_adjustments) > 3:
            # Filter out None values and convert to float
            valid_adjustments = [adj for adj in performance.score_adjustments[-3:] if adj is not None]
            if valid_adjustments:
                recent_trend = np.mean(valid_adjustments)
                if recent_trend > 0.1:
                    adaptation_multiplier *= 1.1
                elif recent_trend < -0.1:
                    adaptation_multiplier *= 0.9
        
        adapted_score = original_score * adaptation_multiplier
        
        # Bounds checking
        if detector_name == "StealthEngine":
            adapted_score = max(0.0, min(10.0, adapted_score))
        else:
            adapted_score = max(0.0, min(1.0, adapted_score))
        
        adaptation_reason = f"acc={accuracy_factor:.2f}_exp={explore_factor:.2f}_mult={adaptation_multiplier:.2f}"
        
        print(f"[DETECTOR ADAPTATION] {detector_name}: {original_score:.3f} → {adapted_score:.3f} (reason: {adaptation_reason})")
        
        return adapted_score, adaptation_reason
    
    def update_detector_performance(self, detector_name: str):
        """Przelicz performance metrics dla detektora"""
        detector_decisions = [
            d for d in self.decision_history 
            if d.detector_name == detector_name and d.was_correct is not None
        ]
        
        if not detector_decisions:
            return
        
        # Calculate metrics
        total_decisions = len(detector_decisions)
        correct_decisions = sum(1 for d in detector_decisions if d.was_correct)
        accuracy_rate = correct_decisions / total_decisions
        
        # Calculate profit/loss
        pnl_decisions = [d for d in detector_decisions if d.profit_loss_pct is not None]
        if pnl_decisions:
            pnl_values = [d.profit_loss_pct for d in pnl_decisions]
            avg_profit_loss = np.mean(pnl_values)
        else:
            avg_profit_loss = 0.0
        
        # Explore mode accuracy
        explore_decisions = [d for d in detector_decisions if d.explore_mode and d.was_correct is not None]
        explore_mode_accuracy = (
            sum(1 for d in explore_decisions if d.was_correct) / len(explore_decisions)
            if explore_decisions else 0.0
        )
        
        # Score adjustments tracking
        score_adjustments = [
            d.adjusted_score - d.original_score 
            for d in detector_decisions 
            if d.adjusted_score != d.original_score
        ]
        
        # Confidence evolution (accuracy over time)
        confidence_evolution = []
        window_size = 20
        for i in range(window_size, len(detector_decisions), window_size):
            window_decisions = detector_decisions[i-window_size:i]
            window_accuracy = sum(1 for d in window_decisions if d.was_correct) / len(window_decisions)
            confidence_evolution.append(window_accuracy)
        
        # Update performance record
        self.performance_data[detector_name] = DetectorPerformance(
            detector_name=detector_name,
            total_decisions=total_decisions,
            correct_decisions=correct_decisions,
            accuracy_rate=accuracy_rate,
            avg_profit_loss=avg_profit_loss,
            explore_mode_accuracy=explore_mode_accuracy,
            last_updated=datetime.now().isoformat(),
            score_adjustments=score_adjustments[-50:],  # Keep last 50
            confidence_evolution=confidence_evolution[-10:]  # Keep last 10 windows
        )
        
        print(f"[DETECTOR PERFORMANCE] {detector_name}: Accuracy={accuracy_rate:.1%}, Explore={explore_mode_accuracy:.1%}, P/L={avg_profit_loss:.2f}%, Decisions={total_decisions}")
        
        # Save every 5 updates
        if total_decisions % 5 == 0:
            self.save_performance_data()
    
    def get_detector_learning_stats(self, detector_name: str) -> Dict[str, Any]:
        """Pobierz statystyki uczenia się detektora"""
        performance = self.performance_data.get(detector_name)
        if not performance:
            return {"status": "no_data", "total_decisions": 0}
        
        recent_decisions = [
            d for d in self.decision_history 
            if d.detector_name == detector_name and 
            datetime.fromisoformat(d.timestamp) > datetime.now() - timedelta(days=7)
        ]
        
        return {
            "status": "learning_active",
            "total_decisions": performance.total_decisions,
            "accuracy_rate": performance.accuracy_rate,
            "explore_mode_accuracy": performance.explore_mode_accuracy,
            "avg_profit_loss": performance.avg_profit_loss,
            "recent_decisions_7d": len(recent_decisions),
            "learning_trend": "improving" if len(performance.confidence_evolution) > 1 and 
                            performance.confidence_evolution[-1] > performance.confidence_evolution[-2] else "stable",
            "last_updated": performance.last_updated
        }
    
    def get_all_detectors_summary(self) -> Dict[str, Dict]:
        """Podsumowanie uczenia się wszystkich detektorów"""
        summary = {}
        for detector_name in self.detector_configs.keys():
            summary[detector_name] = self.get_detector_learning_stats(detector_name)
        return summary

# Global instance
_learning_system = None

def get_detector_learning_system() -> DetectorLearningSystem:
    """Singleton access do learning system"""
    global _learning_system
    if _learning_system is None:
        _learning_system = DetectorLearningSystem()
    return _learning_system

# Convenience functions
def adapt_detector_score(detector_name: str, original_score: float, symbol: str, market_context: Optional[Dict] = None) -> Tuple[float, str]:
    """Convenience function dla score adaptation"""
    return get_detector_learning_system().adapt_detector_score(detector_name, original_score, symbol, market_context)

def record_detector_decision(detector_name: str, symbol: str, original_score: float, adjusted_score: float, decision: str, explore_mode: bool = False, market_context: Optional[Dict] = None):
    """Convenience function dla recording decisions"""
    return get_detector_learning_system().record_detector_decision(detector_name, symbol, original_score, adjusted_score, decision, explore_mode, market_context)

def get_learning_summary() -> Dict[str, Dict]:
    """Convenience function dla learning summary"""
    return get_detector_learning_system().get_all_detectors_summary()