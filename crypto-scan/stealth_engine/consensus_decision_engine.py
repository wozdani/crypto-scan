"""
Consensus Decision Engine - Multi-Agent Decision Layer for Stealth Engine
Etap 1: Interface and Core Structure Implementation

Warstwa decyzyjna konsensusu łącząca różne detektory (WhaleCLIP, CaliforniumWhale, 
DiamondWhale, StealthSignal) w unified decision system z sophisticated consensus strategies.
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio


class ConsensusStrategy(Enum):
    """Strategie konsensusu dla multi-agent decision making"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    UNANIMOUS_AGREEMENT = "unanimous_agreement"
    DOMINANT_DETECTOR = "dominant_detector"


class AlertDecision(Enum):
    """Możliwe decyzje alertów"""
    ALERT = "ALERT"
    NO_ALERT = "NO_ALERT"
    WATCH = "WATCH"
    ESCALATE = "ESCALATE"


@dataclass
class DetectorScore:
    """Struktura score detektora"""
    name: str
    score: float
    confidence: float
    signal_type: str
    metadata: Dict[str, Any] = None


@dataclass
class ConsensusResult:
    """Wynik decyzji konsensusu"""
    decision: AlertDecision
    final_score: float
    confidence: float
    strategy_used: ConsensusStrategy
    contributing_detectors: List[str]
    reasoning: str
    consensus_strength: float
    votes: List[AlertDecision] = None
    alert_sent: bool = False
    timestamp: str = None


class ConsensusDecisionEngine:
    """
    Multi-Agent Consensus Decision Engine dla Stealth Detection System
    
    Główna klasa odpowiedzialna za agregację scores z różnych detektorów
    i podejmowanie finalnych decyzji o alertach w oparciu o consensus strategies.
    """
    
    def __init__(self, telegram_token: str = None, chat_id: str = None):
        """
        Inicjalizacja Consensus Decision Engine
        
        Args:
            telegram_token: Telegram bot token dla alertów
            chat_id: Chat ID dla alertów Telegram
        """
        self.telegram_token = telegram_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        # Default detector weights (można dostosować dynamicznie)
        self.detector_weights = {
            'WhaleCLIP': 0.25,
            'CaliforniumWhale': 0.30,
            'DiamondWhale': 0.25,
            'StealthSignal': 0.20
        }
        
        # Consensus thresholds
        self.consensus_thresholds = {
            'alert_threshold': 0.7,
            'watch_threshold': 0.5,
            'minimum_detectors': 2,
            'unanimous_threshold': 0.8
        }
        
        # Statistics tracking
        self.decision_history = []
        self.detector_performance = {}
        
        # Multi-agent decision system
        self.enable_multi_agent = False  # Flag to enable/disable 5-agent system
        self.multi_agent_override_threshold = 0.5  # Override if agents vote YES even with low score
        
        print("[CONSENSUS ENGINE] Initialized multi-agent decision layer")
    
    def run(self, token: str, scores: Union[Dict[str, float], Dict[str, Dict[str, float]]], 
            strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_AVERAGE,
            metadata: Dict[str, Any] = None, 
            use_simple_consensus: bool = False,
            use_dynamic_boosting: bool = False,
            market_data: Dict[str, Any] = None) -> ConsensusResult:
        """
        Główna funkcja uruchamiająca consensus decision process
        
        Args:
            token: Symbol tokena (np. 'RSRUSDT')
            scores: Dict z nazwami detektorów i ich scores
                   Simple format: {"WhaleCLIP": 0.84, "DiamondWhale": 0.78}
                   Extended format: {"WhaleCLIP": {"score": 0.84, "confidence": 0.66, "weight": 0.28}}
            strategy: Strategia konsensusu do użycia
            metadata: Dodatkowe metadane z detektorów
            use_simple_consensus: True dla prostej logiki (Etap 2), False dla advanced strategies
            use_dynamic_boosting: True dla Etap 3 confidence-based weighted scoring
            
        Returns:
            ConsensusResult: Kompletny wynik decyzji konsensusu
        """
        is_extended_format = self._is_extended_score_format(scores)
        booster_mode = "DYNAMIC_BOOST" if use_dynamic_boosting else "SIMPLE_CONSENSUS" if use_simple_consensus else strategy.value
        
        print(f"[CONSENSUS ENGINE] Processing {token} with {len(scores)} detectors")
        print(f"[CONSENSUS ENGINE] Mode: {booster_mode}")
        print(f"[CONSENSUS ENGINE] Extended format: {is_extended_format}")
        print(f"[CONSENSUS ENGINE] Input scores: {scores}")
        
        # Walidacja input data
        if not scores:
            return self._create_no_alert_result("No detector scores provided", strategy)
        
        # Multi-Agent 5-Agent Decision System Integration
        if self.enable_multi_agent and market_data:
            multi_agent_override = self._evaluate_with_multi_agents_sync(token, scores, market_data, metadata)
            if multi_agent_override:
                print(f"[CONSENSUS ENGINE] ⚡ MULTI-AGENT OVERRIDE detected for {token}")
                return multi_agent_override
        
        # ETAP 3: Dynamic Boosting Logic Implementation
        if use_dynamic_boosting:
            return self._dynamic_boosting_logic(token, scores)
        
        # ETAP 2: Simple Consensus Logic Implementation
        if use_simple_consensus:
            return self._simple_consensus_logic(token, scores)
        
        # Original advanced strategy logic
        if len(scores) < self.consensus_thresholds['minimum_detectors']:
            return self._create_no_alert_result(
                f"Insufficient detectors: {len(scores)} < {self.consensus_thresholds['minimum_detectors']}", 
                strategy
            )
        
        # Konwersja scores na DetectorScore objects
        detector_scores = self._convert_to_detector_scores(scores, metadata or {})
        
        # Wybór i wykonanie strategii konsensusu
        result = self._execute_consensus_strategy(token, detector_scores, strategy)
        
        # Zapisanie do historii
        self._record_decision(token, result, detector_scores)
        
        # Wysłanie alertu jeśli wymagane
        if result.decision == AlertDecision.ALERT:
            result.alert_sent = self._send_consensus_alert(token, result, detector_scores)
        
        print(f"[CONSENSUS ENGINE] Final decision: {result.decision.value} (score: {result.final_score:.3f})")
        print(f"[CONSENSUS ENGINE] Reasoning: {result.reasoning}")
        
        return result
    
    async def _evaluate_with_multi_agents(
        self, 
        token: str, 
        scores: Dict[str, Any], 
        market_data: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Optional[ConsensusResult]:
        """
        Evaluate each detector with 5-agent system
        
        Args:
            token: Token symbol
            scores: Detector scores
            market_data: Market data for context
            metadata: Additional metadata
            
        Returns:
            ConsensusResult if override triggered, None otherwise
        """
        from .multi_agent_decision import evaluate_detector_with_agents
        
        print(f"[MULTI-AGENT] Evaluating {len(scores)} detectors with 5-agent system")
        
        # Normalize scores to get detector data
        normalized_scores = self._normalize_to_extended_format(scores)
        
        # Prepare signal data from metadata
        signal_data = metadata.get('signal_data', {}) if metadata else {}
        
        # Track override decisions
        override_detectors = []
        agent_logs = []
        
        # Evaluate each detector with 5 agents
        for detector_name, detector_data in normalized_scores.items():
            score = detector_data.get('score', 0.0)
            
            # Skip detectors with zero score
            if score <= 0:
                continue
            
            # Get detector-specific signal data
            detector_signals = signal_data.get(detector_name, {})
            
            # Run 5-agent evaluation
            decision, confidence, log = await evaluate_detector_with_agents(
                detector_name=detector_name,
                score=score,
                signal_data=detector_signals,
                market_data=market_data,
                threshold=self.multi_agent_override_threshold
            )
            
            agent_logs.append(log)
            
            # Check for override
            if decision == "YES" and score < self.multi_agent_override_threshold:
                print(f"[MULTI-AGENT] ⚡ OVERRIDE: {detector_name} agents voted YES despite low score {score:.3f}")
                override_detectors.append({
                    'detector': detector_name,
                    'score': score,
                    'agent_confidence': confidence,
                    'decision': decision
                })
        
        # If any detector triggered override, create override result
        if override_detectors:
            # Calculate aggregate confidence
            avg_confidence = sum(d['agent_confidence'] for d in override_detectors) / len(override_detectors)
            
            # Create override reasoning
            override_reasoning = f"Multi-Agent Override: {len(override_detectors)} detector(s) triggered agent override. "
            override_reasoning += f"Detectors: {', '.join(d['detector'] for d in override_detectors)}. "
            override_reasoning += f"Despite low scores, agent analysis indicates strong signal."
            
            # Create override result
            result = ConsensusResult(
                decision=AlertDecision.ALERT,
                final_score=max(d['score'] for d in override_detectors),
                confidence=avg_confidence,
                strategy_used=ConsensusStrategy.DOMINANT_DETECTOR,
                contributing_detectors=[d['detector'] for d in override_detectors],
                reasoning=override_reasoning,
                consensus_strength=avg_confidence,
                votes=[AlertDecision.ALERT] * len(override_detectors),
                alert_sent=False,
                timestamp=datetime.now().isoformat()
            )
            
            # Log the override
            print("\n[MULTI-AGENT] Override Summary:")
            for log in agent_logs:
                if "OVERRIDE ALERT" in log:
                    print(log)
            
            return result
        
        # No override triggered
        return None
    
    def _evaluate_with_multi_agents_sync(
        self, 
        token: str, 
        scores: Dict[str, Any], 
        market_data: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Optional[ConsensusResult]:
        """
        Synchronous version of multi-agent evaluation
        
        Args:
            token: Token symbol
            scores: Detector scores
            market_data: Market data for context
            metadata: Additional metadata
            
        Returns:
            ConsensusResult if override triggered, None otherwise
        """
        # Try to run async function in sync context
        try:
            # Check if event loop is already running
            try:
                loop = asyncio.get_running_loop()
                # Already in async context - create new thread
                import concurrent.futures
                import threading
                
                result = None
                exception = None
                
                def run_in_thread():
                    nonlocal result, exception
                    try:
                        # Create new event loop in thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(
                            self._evaluate_with_multi_agents(token, scores, market_data, metadata)
                        )
                        new_loop.close()
                    except Exception as e:
                        exception = e
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join(timeout=30)  # 30 second timeout
                
                if exception:
                    raise exception
                    
                return result
                
            except RuntimeError:
                # No event loop running - safe to use asyncio.run()
                return asyncio.run(
                    self._evaluate_with_multi_agents(token, scores, market_data, metadata)
                )
                
        except Exception as e:
            print(f"[MULTI-AGENT ERROR] Failed to run multi-agent evaluation: {e}")
            return None
    
    def _is_extended_score_format(self, scores: Dict) -> bool:
        """
        Sprawdza czy scores są w extended format (confidence + weight)
        
        Args:
            scores: Input scores dictionary
            
        Returns:
            True jeśli extended format, False jeśli simple format
        """
        if not scores:
            return False
        
        first_value = next(iter(scores.values()))
        return isinstance(first_value, dict) and 'score' in first_value
    
    def _dynamic_boosting_logic(self, token: str, scores: Union[Dict[str, float], Dict[str, Dict]], use_adaptive_weights: bool = True) -> ConsensusResult:
        """
        ETAP 3: Dynamic Boosting Decision Logic + ETAP 5: Fallback Logic + ETAP 6: Adaptive Weights
        Implementuje confidence-based weighted scoring z booster strategies i fallback alerts:
        - Aktywne detektory: score >= 0.75 AND confidence >= 0.60
        - Global score: suma (score × weight) dla aktywnych detektorów
        - Booster: confidence > 0.85 AND score > 0.90 → score × 1.1
        - ETAP 5: Fallback alert: jeśli 1 detektor > 0.92 score i > 0.85 confidence = instant alert
        - ETAP 6: Adaptive weights: dynamiczne dostosowanie wag na podstawie historycznej skuteczności
        
        Args:
            token: Symbol tokena
            scores: Dict w extended format z confidence i weight
            use_adaptive_weights: Czy stosować adaptive weights (default: True)
            
        Returns:
            ConsensusResult z dynamic boosting decision lub fallback alert
        """
        print(f"[DYNAMIC BOOST] Processing {token} using Etap 3 + Etap 5 + Etap 6 logic")
        
        # Konwersja do unified format jeśli potrzebne
        unified_scores = self._normalize_to_extended_format(scores)
        
        # ETAP 6: Aplikuj adaptive weights jeśli włączone
        if use_adaptive_weights:
            print(f"[ADAPTIVE WEIGHTS] Applying adaptive weights to {len(unified_scores)} detectors")
            unified_scores = self.apply_adaptive_weights(unified_scores, days=7)
        
        # ETAP 5: Check for fallback trigger first (very strong single detector)
        fallback_triggered, fallback_detector = self._should_trigger_fallback_alert(unified_scores)
        
        if fallback_triggered:
            print(f"[FALLBACK ALERT] {fallback_detector} exceeded threshold - instant alert!")
            return self._create_fallback_alert_result(token, unified_scores, fallback_detector)
        
        # Thresholds dla Etap 3 - ZMIENIONE: bardziej liberalne podejście
        score_threshold = 0.0  # Nieużywane teraz - każdy detektor z score > 0 jest aktywny
        confidence_threshold = 0.0  # Nieużywane teraz - nie wymagamy minimum confidence
        global_score_threshold = 0.7  # Obniżone z 0.8 na 0.7
        min_detectors = 1  # Zmienione z 2 na 1 - nawet 1 silny detektor może wywołać alert
        
        # Booster thresholds
        booster_confidence_threshold = 0.85
        booster_score_threshold = 0.90
        booster_multiplier = 1.1
        
        print(f"[DYNAMIC BOOST] Unified scores: {unified_scores}")
        
        # Znajdź aktywne detektory (tylko te które mają score > 0 i nie są None)
        active_detectors = {}
        boosted_detectors = []
        
        for detector_name, detector_data in unified_scores.items():
            # 🔧 CRITICAL FIX #1: Prevent dict * float error from CHZUSDT logs
            score_raw = detector_data.get('score', 0.0)
            confidence_raw = detector_data.get('confidence', 0.0)
            weight_raw = detector_data.get('weight', 0.25)
            
            # Extract nested values if dict format
            if isinstance(score_raw, dict):
                score = score_raw.get('score', score_raw.get('value', 0.0))
            else:
                score = score_raw
                
            if isinstance(confidence_raw, dict):
                confidence = confidence_raw.get('confidence', confidence_raw.get('value', 0.0))
            else:
                confidence = confidence_raw
                
            if isinstance(weight_raw, dict):
                weight = weight_raw.get('weight', weight_raw.get('value', 0.25))
            else:
                weight = weight_raw
            
            # Robust type conversion with validation
            try:
                score = float(score) if score is not None else 0.0
                confidence = float(confidence) if confidence is not None else 0.0
                weight = float(weight) if weight is not None else 0.25
            except (ValueError, TypeError) as e:
                print(f"[CONSENSUS FIX] {detector_name}: Type conversion error {e}, using defaults")
                score, confidence, weight = 0.0, 0.0, 0.25
            
            # Bounds checking
            score = max(0.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))
            weight = max(0.0, min(2.0, weight))
            
            print(f"[TYPE SAFETY] {detector_name}: score={score:.3f} (type: {type(score)}), confidence={confidence:.3f}, weight={weight:.3f}")
            
            # 🛠️ NOWA LOGIKA: Detektor jest aktywny jeśli ma score > 0 (niezależnie od confidence)
            if score > 0.0:
                # Zastosuj booster jeśli spełnia warunki
                if confidence > booster_confidence_threshold and score > booster_score_threshold:
                    original_score = score
                    score = score * booster_multiplier
                    boosted_detectors.append(detector_name)
                    print(f"[BOOSTER] {detector_name}: {original_score:.3f} → {score:.3f} (×{booster_multiplier})")
                
                # 🔧 BELUSDT FIX: Additional safety for weighted_contribution calculation
                try:
                    weighted_contribution = float(score) * float(weight)
                except (ValueError, TypeError) as calc_error:
                    print(f"[CONSENSUS CALC FIX] {detector_name}: Calculation error {calc_error}, using 0.0")
                    weighted_contribution = 0.0
                
                active_detectors[detector_name] = {
                    'score': score,
                    'confidence': confidence,
                    'weight': weight,
                    'weighted_contribution': weighted_contribution
                }
        
        print(f"[DYNAMIC BOOST] Active detectors: {len(active_detectors)}")
        print(f"[DYNAMIC BOOST] Boosted detectors: {boosted_detectors}")
        
        # Oblicz weighted global_score - NOWA LOGIKA: normalizuj tylko na podstawie aktywnych detektorów
        if active_detectors:
            total_weighted_contribution = sum(d['weighted_contribution'] for d in active_detectors.values())
            total_weight = sum(d['weight'] for d in active_detectors.values())
            # Normalizuj wynik tylko względem wag aktywnych detektorów
            global_score = total_weighted_contribution / total_weight if total_weight > 0 else 0.0
        else:
            global_score = 0.0
            total_weight = 0.0
        
        print(f"[DYNAMIC BOOST] Normalized global score: {global_score:.3f} (active weight: {total_weight:.3f})")
        
        # SPECJALNY PRZYPADEK: Jeśli dokładnie 2 detektory są aktywne i oba silne, to alert
        if len(active_detectors) == 2:
            # Sprawdź czy oba detektory mają wysokie score
            detector_scores = [d['score'] for d in active_detectors.values()]
            if all(score >= 0.8 for score in detector_scores):
                print(f"[CONSENSUS SPECIAL] 2 active detectors with high scores: {detector_scores}")
                # Wymuś alert dla 2 silnych detektorów
                global_score = max(global_score, 0.75)  # Boost score jeśli za niski
        
        # Logika decyzyjna Etap 3 - ZMIENIONA:
        # min. 1 detektor aktywny AND global_score > 0.7
        if len(active_detectors) >= min_detectors and global_score > global_score_threshold:
            decision = AlertDecision.ALERT
            reason = f"Dynamic consensus from {len(active_detectors)} detectors, weighted_score={global_score:.2f}"
            if boosted_detectors:
                reason += f", boosted: {', '.join(boosted_detectors)}"
            consensus_strength = min(1.0, len(active_detectors) / len(unified_scores))
            
            # Wysłanie enhanced Telegram alert
            alert_sent = self._send_dynamic_telegram_alert(token, global_score, active_detectors, boosted_detectors)
            
            print(f"[DYNAMIC BOOST] ALERT triggered: {len(active_detectors)} detectors >= {min_detectors}, weighted_score {global_score:.3f} > {global_score_threshold}")
            
        else:
            decision = AlertDecision.NO_ALERT
            alert_sent = False
            consensus_strength = 0.0
            
            if len(active_detectors) < min_detectors:
                reason = f"Insufficient consensus ({len(active_detectors)} active detectors, need {min_detectors}), weighted_score={global_score:.2f}"
            else:
                reason = f"Weighted score too low ({global_score:.2f} ≤ {global_score_threshold}), detectors={len(active_detectors)}"
            
            print(f"[DYNAMIC BOOST] NO ALERT: {reason}")
        
        # Oblicz confidence na podstawie weighted average confidence - ZMIENIONE: nie penalizuj za nieaktywne
        if active_detectors:
            weighted_confidence = sum(d['confidence'] * d['weight'] for d in active_detectors.values()) / total_weight if total_weight > 0 else 0.0
            # Nie mnóż przez (len(active_detectors) / len(unified_scores)) - używaj tylko aktywnych
            confidence = min(1.0, weighted_confidence)
        else:
            confidence = 0.0
        
        contributing_detectors = list(active_detectors.keys())
        
        # Zwróć rezultat w ConsensusResult format
        result = ConsensusResult(
            decision=decision,
            final_score=global_score,
            confidence=confidence,
            strategy_used=ConsensusStrategy.WEIGHTED_AVERAGE,  # Używamy jako proxy dla dynamic boosting
            contributing_detectors=contributing_detectors,
            reasoning=reason,
            consensus_strength=consensus_strength,
            alert_sent=alert_sent,
            timestamp=datetime.now().isoformat()
        )
        
        # Zapisz do historii decyzji
        self._record_dynamic_decision(token, result, unified_scores, active_detectors, boosted_detectors)
        
        return result
    
    def _normalize_to_extended_format(self, scores: Union[Dict[str, float], Dict[str, Dict]]) -> Dict[str, Dict]:
        """
        Normalizuje scores do extended format z confidence i weight
        
        Args:
            scores: Simple lub extended format
            
        Returns:
            Extended format z default values jeśli potrzebne
        """
        if self._is_extended_score_format(scores):
            return scores
        
        # Konwersja z simple format - dodaj default confidence i weight
        default_weights = {
            "CaliforniumWhale": 0.30,
            "WhaleCLIP": 0.25,
            "DiamondWhale": 0.25,
            "StealthSignal": 0.20
        }
        
        normalized = {}
        for detector_name, score in scores.items():
            normalized[detector_name] = {
                'score': score,
                'confidence': 0.70,  # Default confidence
                'weight': default_weights.get(detector_name, 0.20)  # Default weight
            }
        
        print(f"[NORMALIZE] Converted simple format to extended format with defaults")
        return normalized
    
    def _should_trigger_fallback_alert(self, active_detectors: Dict[str, Dict], 
                                      fallback_threshold: float = 0.92, 
                                      min_confidence: float = 0.85) -> Tuple[bool, str]:
        """
        ETAP 5: Sprawdza czy któryś detektor przekroczył próg fallback alert
        
        Args:
            active_detectors: Dict z detector data {"detector": {"score": 0.8, "confidence": 0.7}}
            fallback_threshold: Minimalny score dla fallback alert (default: 0.92)
            min_confidence: Minimalna confidence dla fallback alert (default: 0.85)
            
        Returns:
            Tuple[bool, str]: (czy_trigger, nazwa_detektora)
        """
        for detector_name, data in active_detectors.items():
            score = data.get('score', 0.0)
            confidence = data.get('confidence', 0.0)
            
            # Type safety - ensure float values before comparison
            try:
                score = float(score) if score is not None else 0.0
                confidence = float(confidence) if confidence is not None else 0.0
            except (ValueError, TypeError):
                score, confidence = 0.0, 0.0
            
            if score >= fallback_threshold and confidence >= min_confidence:
                print(f"[FALLBACK CHECK] {detector_name}: score={score:.3f} >= {fallback_threshold}, conf={confidence:.3f} >= {min_confidence} → TRIGGER")
                return True, detector_name
        
        print(f"[FALLBACK CHECK] No detector meets fallback criteria (score >= {fallback_threshold}, conf >= {min_confidence})")
        return False, None
    
    def _create_fallback_alert_result(self, token: str, all_scores: Dict[str, Dict], 
                                     trigger_detector: str) -> ConsensusResult:
        """
        ETAP 5: Tworzy ConsensusResult dla fallback alert z pojedynczego silnego detektora
        
        Args:
            token: Symbol tokena
            all_scores: Wszystkie scores detektorów
            trigger_detector: Nazwa detektora który triggered fallback
            
        Returns:
            ConsensusResult z fallback alert decision
        """
        trigger_data = all_scores[trigger_detector]
        trigger_score = trigger_data['score']
        trigger_confidence = trigger_data['confidence']
        
        # Wyślij fallback alert z enhanced reason
        reason = f"Fallback Trigger: {trigger_detector} exceeded threshold"
        alert_sent = self.send_telegram_alert(
            token=token,
            global_score=trigger_score,
            active_detectors={trigger_detector: trigger_data},
            boosted_detectors=[]
        )
        
        # Create result
        result = ConsensusResult(
            decision=AlertDecision.ALERT,
            final_score=trigger_score,
            confidence=trigger_confidence,
            strategy_used=ConsensusStrategy.DOMINANT_DETECTOR,
            contributing_detectors=[trigger_detector],
            reasoning=reason,
            consensus_strength=1.0,  # Single strong detector = high strength
            alert_sent=alert_sent,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"[FALLBACK RESULT] Created fallback alert: score={trigger_score:.3f}, detector={trigger_detector}")
        
        # Record fallback decision
        self._record_fallback_decision(token, result, all_scores, trigger_detector)
        
        return result
    
    def _record_fallback_decision(self, token: str, result: ConsensusResult, 
                                 all_scores: Dict[str, Dict], trigger_detector: str):
        """
        ETAP 5: Zapisuje fallback decision do historii
        
        Args:
            token: Symbol tokena
            result: ConsensusResult z fallback decision
            all_scores: Wszystkie scores detektorów
            trigger_detector: Nazwa detektora który triggered fallback
        """
        decision_record = {
            'timestamp': result.timestamp,
            'token': token,
            'decision': result.decision.value,
            'strategy': result.strategy_used.value,
            'final_score': result.final_score,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'consensus_strength': result.consensus_strength,
            'alert_sent': result.alert_sent,
            'fallback_trigger': True,
            'trigger_detector': trigger_detector,
            'trigger_score': all_scores[trigger_detector]['score'],
            'trigger_confidence': all_scores[trigger_detector]['confidence'],
            'all_detector_scores': {k: v['score'] for k, v in all_scores.items()},
            'all_detector_confidences': {k: v['confidence'] for k, v in all_scores.items()}
        }
        
        self.decision_history.append(decision_record)
        print(f"[FALLBACK RECORD] Recorded fallback decision for {token}: {trigger_detector} → {result.decision.value}")
    
    def update_detector_weights(self, feedback_history: Dict[str, Dict], decay: float = 0.95) -> Dict[str, float]:
        """
        ETAP 6: Dynamiczne modyfikowanie wag detektorów na podstawie feedback history
        
        Args:
            feedback_history: Dict z historią skuteczności detektorów
                {"detector": {"correct": int, "total": int, "avg_conf": float, "prev_weight": float}}
            decay: Czynnik wygaszania dla smooth weight transitions (default: 0.95)
            
        Returns:
            Dict z zaktualizowanymi wagami detektorów
        """
        updated_weights = {}
        
        for detector, stats in feedback_history.items():
            if stats["total"] == 0:
                # Brak danych - używaj domyślnej wagi
                updated_weights[detector] = stats.get("prev_weight", 0.25)
                continue
                
            # Oblicz success rate (0.0 - 1.0)
            success_rate = stats["correct"] / max(1, stats["total"])
            avg_confidence = stats.get("avg_conf", 0.70)
            prev_weight = stats.get("prev_weight", 0.25)
            
            # Adaptive weight formula: exponential moving average z confidence boost
            # new_weight = decay * prev_weight + (1 - decay) * (success_rate * avg_confidence)
            performance_factor = success_rate * avg_confidence
            new_weight = decay * prev_weight + (1 - decay) * performance_factor
            
            # Bounds protection: wagi między 0.10 a 0.50
            new_weight = max(0.10, min(0.50, new_weight))
            updated_weights[detector] = round(new_weight, 4)
            
            print(f"[ADAPTIVE WEIGHTS] {detector}: success={success_rate:.2f}, conf={avg_confidence:.2f}, "
                  f"prev={prev_weight:.3f} → new={new_weight:.3f}")
        
        return updated_weights
    
    def get_feedback_history_from_decisions(self, days: int = 7) -> Dict[str, Dict]:
        """
        ETAP 6: Pobiera feedback history z decision_history dla adaptive weights
        
        Args:
            days: Liczba dni wstecz do analizy (default: 7)
            
        Returns:
            Feedback history dict dla update_detector_weights()
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        feedback_history = {}
        
        # Domyślne wagi detektorów
        default_weights = {
            "CaliforniumWhale": 0.30,
            "DiamondWhale": 0.25,
            "WhaleCLIP": 0.25,
            "StealthSignal": 0.20
        }
        
        # Inicjalizuj feedback history dla wszystkich detektorów
        for detector, default_weight in default_weights.items():
            feedback_history[detector] = {
                "correct": 0,
                "total": 0,
                "avg_conf": 0.70,
                "prev_weight": default_weight,
                "confidence_sum": 0.0
            }
        
        # Analizuj decision history
        for decision in self.decision_history:
            try:
                decision_time = datetime.fromisoformat(decision["timestamp"])
                if decision_time < cutoff_date:
                    continue
                    
                # Sprawdź każdy contributing detector
                contributing_detectors = decision.get("contributing_detectors", [])
                decision_success = decision.get("alert_sent", False)  # Proxy dla success
                
                for detector in contributing_detectors:
                    if detector in feedback_history:
                        feedback_history[detector]["total"] += 1
                        if decision_success:
                            feedback_history[detector]["correct"] += 1
                        
                        # Dodaj confidence do sumy
                        conf = decision.get("confidence", 0.70)
                        feedback_history[detector]["confidence_sum"] += conf
                        
            except (KeyError, ValueError, TypeError) as e:
                print(f"[FEEDBACK HISTORY] Skipping malformed decision: {e}")
                continue
        
        # Oblicz średnie confidence dla każdego detektora
        for detector in feedback_history:
            if feedback_history[detector]["total"] > 0:
                avg_conf = feedback_history[detector]["confidence_sum"] / feedback_history[detector]["total"]
                feedback_history[detector]["avg_conf"] = round(avg_conf, 3)
            
            # Usuń tymczasową sumę
            del feedback_history[detector]["confidence_sum"]
        
        print(f"[FEEDBACK HISTORY] Analyzed {len(self.decision_history)} decisions from last {days} days")
        return feedback_history
    
    def apply_adaptive_weights(self, scores: Dict[str, Dict], days: int = 7) -> Dict[str, Dict]:
        """
        ETAP 6: Aplikuje adaptive weights do detector scores
        
        Args:
            scores: Extended format scores {"detector": {"score": float, "confidence": float, "weight": float}}
            days: Liczba dni dla feedback analysis (default: 7)
            
        Returns:
            Scores z zaktualizowanymi adaptive weights
        """
        # Pobierz feedback history
        feedback_history = self.get_feedback_history_from_decisions(days)
        
        # Zaktualizuj wagi na podstawie performance
        updated_weights = self.update_detector_weights(feedback_history)
        
        # Aplikuj nowe wagi do scores
        updated_scores = scores.copy()
        for detector in updated_scores:
            if detector in updated_weights:
                old_weight = updated_scores[detector]["weight"]
                new_weight = updated_weights[detector]
                updated_scores[detector]["weight"] = new_weight
                
                print(f"[ADAPTIVE WEIGHTS] {detector}: weight {old_weight:.3f} → {new_weight:.3f}")
        
        return updated_scores
    
    def _send_dynamic_telegram_alert(self, token: str, global_score: float, 
                                    active_detectors: Dict, boosted_detectors: List[str]) -> bool:
        """
        ETAP 4: Enhanced Dynamic Telegram Alert
        Forwards to new universal send_telegram_alert function
        
        Args:
            token: Symbol tokena
            global_score: Weighted global score
            active_detectors: Dict aktywnych detektorów z metadata
            boosted_detectors: Lista detektorów z applied booster
            
        Returns:
            True jeśli alert wysłany pomyślnie
        """
        return self.send_telegram_alert(token, global_score, active_detectors, boosted_detectors)
    
    def send_telegram_alert(self, token: str, global_score: float, active_detectors: Dict, 
                           boosted_detectors: List[str] = None) -> bool:
        """
        ETAP 4: Universal Telegram Alert Function
        Uniwersalna funkcja alertów z pełną transparentnością konsensus decision
        
        Args:
            token: Symbol tokena
            global_score: Weighted global score
            active_detectors: Dict z detector data {"detector": {"score": 0.8, "confidence": 0.7, "weight": 0.3}}
            boosted_detectors: Opcjonalna lista boosted detectors
            
        Returns:
            True jeśli alert wysłany pomyślnie
        """
        if boosted_detectors is None:
            boosted_detectors = []
            
        # ETAP 4: Enhanced Message Formatting z pełną informacją
        lines = [f"🚨 [Consensus Alert] {token}"]
        lines.append(f"🎯 Global Score: {global_score:.3f}")
        lines.append(f"📊 Active Detectors: {len(active_detectors)}")
        
        if boosted_detectors:
            lines.append(f"⚡ Boosted: {', '.join(boosted_detectors)}")
        
        lines.append("")
        lines.append("🔍 Detector Breakdown:")
        
        # Sort detectors by contribution (score × weight) with type safety
        def safe_sort_key(item):
            data = item[1]
            try:
                score_val = float(data.get('score', 0.0))
                weight_val = float(data.get('weight', 0.0))
                return score_val * weight_val
            except (ValueError, TypeError):
                return 0.0
        
        sorted_detectors = sorted(
            active_detectors.items(),
            key=safe_sort_key,
            reverse=True
        )
        
        # Type safety for sum calculations
        total_weight = 0.0
        total_contribution = 0.0
        for data in active_detectors.values():
            try:
                weight_val = float(data.get('weight', 0.0))
                score_val = float(data.get('score', 0.0))
                total_weight += weight_val
                total_contribution += score_val * weight_val
            except (ValueError, TypeError):
                continue  # Skip invalid entries
        
        for detector_name, data in sorted_detectors:
            is_boosted = detector_name in boosted_detectors
            
            # Type safety for calculation operations
            score_val = data.get('score', 0.0)
            weight_val = data.get('weight', 0.0)
            try:
                score_val = float(score_val) if score_val is not None else 0.0
                weight_val = float(weight_val) if weight_val is not None else 0.0
            except (ValueError, TypeError):
                score_val, weight_val = 0.0, 0.0
                
            contribution = score_val * weight_val
            contribution_pct = (contribution / total_contribution * 100) if total_contribution > 0 else 0
            
            # Enhanced detector icons based on performance - type safety
            confidence_val = data.get('confidence', 0.0)
            try:
                confidence_val = float(confidence_val) if confidence_val is not None else 0.0
            except (ValueError, TypeError):
                confidence_val = 0.0
                
            if is_boosted:
                icon = "🔥⚡"
            elif confidence_val >= 0.80:
                icon = "💎"
            elif confidence_val >= 0.70:
                icon = "⭐"
            elif confidence_val >= 0.60:
                icon = "💫"
            else:
                icon = "⚡"
            
            lines.append(f"  {icon} {detector_name}:")
            lines.append(f"    Score: {data['score']:.3f} | Conf: {data['confidence']:.2f} | Weight: {data['weight']:.2f}")
            lines.append(f"    Contribution: {contribution:.3f} ({contribution_pct:.1f}%)")
        
        lines.append("")
        lines.append(f"📈 Total Weight: {total_weight:.3f}")
        lines.append(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # ETAP 4: Enhanced confidence assessment
        avg_confidence = sum(data['confidence'] for data in active_detectors.values()) / len(active_detectors)
        confidence_level = self._assess_confidence_level(avg_confidence, len(boosted_detectors))
        lines.append(f"🎖️ Consensus Confidence: {confidence_level}")
        
        message = "\n".join(lines)
        
        print("[CONSENSUS TELEGRAM] Enhanced alert message prepared:")
        for line in lines:
            print(line)
        
        # ETAP 4: Rzeczywiste wysłanie na Telegram
        if not self.telegram_token or not self.chat_id:
            print("[CONSENSUS TELEGRAM] No Telegram credentials configured")
            return False
        
        try:
            import requests
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            params = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                print(f"[CONSENSUS TELEGRAM] Alert sent successfully for {token}")
                return True
            else:
                print(f"[CONSENSUS TELEGRAM] Alert failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"[CONSENSUS TELEGRAM] Alert error: {e}")
            return False
    
    def _assess_confidence_level(self, avg_confidence: float, booster_count: int) -> str:
        """
        ETAP 4: Assess overall consensus confidence level
        
        Args:
            avg_confidence: Average confidence across active detectors
            booster_count: Number of boosted detectors
            
        Returns:
            Confidence level string
        """
        # ETAP 4: Enhanced confidence assessment z booster weighting
        if booster_count >= 2 and avg_confidence >= 0.80:
            return "VERY HIGH ⭐⭐⭐"
        elif booster_count >= 1 and avg_confidence >= 0.75:
            return "HIGH ⭐⭐"
        elif booster_count >= 1 or avg_confidence >= 0.70:
            return "GOOD ⭐"
        elif avg_confidence >= 0.60:
            return "MODERATE 💫"
        else:
            return "LOW ⚡"
    
    def _record_dynamic_decision(self, token: str, result: ConsensusResult, 
                                all_scores: Dict, active_detectors: Dict, 
                                boosted_detectors: List[str]):
        """
        Zapisuje dynamic boosting decision do historii
        
        Args:
            token: Symbol tokena
            result: ConsensusResult
            all_scores: Wszystkie scores detektorów (extended format)
            active_detectors: Aktywne detektory z metadata
            boosted_detectors: Lista boosted detektorów
        """
        decision_record = {
            'token': token,
            'timestamp': result.timestamp,
            'decision': result.decision.value,
            'final_score': result.final_score,
            'confidence': result.confidence,
            'strategy': 'DYNAMIC_BOOSTING',
            'consensus_strength': result.consensus_strength,
            'all_detector_scores': all_scores,
            'active_detector_data': active_detectors,
            'active_count': len(active_detectors),
            'total_count': len(all_scores),
            'boosted_detectors': boosted_detectors,
            'boosted_count': len(boosted_detectors),
            'total_weight': sum(d['weight'] for d in active_detectors.values()) if active_detectors else 0.0,
            'reasoning': result.reasoning,
            'alert_sent': result.alert_sent
        }
        
        self.decision_history.append(decision_record)
        
        # Ogranicz historię
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
        
        print(f"[DYNAMIC BOOST] Decision recorded: {result.decision.value} for {token}")
    
    def _simple_consensus_logic(self, token: str, scores: Dict[str, float]) -> ConsensusResult:
        """
        ETAP 2: Simple Consensus Decision Logic
        Implementuje prostą logikę konsensusu zgodnie z specyfikacją:
        - min. 2 detektory > 0.75
        - global_score > 0.8 (średnia z aktywnych sygnałów)
        
        Args:
            token: Symbol tokena
            scores: Dict z scores detektorów
            
        Returns:
            ConsensusResult z simple consensus decision
        """
        print(f"[SIMPLE CONSENSUS] Processing {token} using Etap 2 logic")
        
        # 🛠️ NOWA LOGIKA: Bardziej liberalne podejście
        threshold = 0.0  # ZMIENIONE: Każdy detektor ze score > 0 jest aktywny
        global_score_threshold = 0.7  # Utrzymane na 0.7
        min_detectors = 1  # Utrzymane - nawet 1 detektor może wywołać alert
        
        # Znajdź aktywne detektory (score > 0) with type safety
        active_detectors = {}
        for k, v in scores.items():
            try:
                # Handle both dict and float format
                if isinstance(v, dict):
                    score_val = float(v.get('score', 0.0))
                else:
                    score_val = float(v) if v is not None else 0.0
                
                # ZMIENIONE: Detektor jest aktywny jeśli score > 0
                if score_val > 0.0:
                    active_detectors[k] = score_val
            except (ValueError, TypeError):
                continue  # Skip invalid entries
        
        print(f"[SIMPLE CONSENSUS] Active detectors (> 0): {len(active_detectors)}")
        print(f"[SIMPLE CONSENSUS] Active scores: {active_detectors}")
        
        # Oblicz global_score jako średnią z aktywnych sygnałów
        if active_detectors:
            global_score = sum(active_detectors.values()) / len(active_detectors)
        else:
            global_score = 0.0
        
        # 🛠️ SPECJALNY PRZYPADEK: Jeśli dokładnie 2 detektory aktywne z wysokimi score
        if len(active_detectors) == 2 and all(score >= 0.8 for score in active_detectors.values()):
            print(f"[SIMPLE CONSENSUS SPECIAL] 2 strong detectors detected: {list(active_detectors.values())}")
            # Boost score dla 2 silnych detektorów
            global_score = max(global_score, 0.75)
        
        print(f"[SIMPLE CONSENSUS] Global score (avg of active): {global_score:.3f}")
        
        # 🎯 STRICT CONSENSUS LOGIC - NO OVERRIDES PER USER REQUIREMENT
        # Only proper consensus threshold decisions allowed - no strong signal exceptions
        
        if len(active_detectors) >= min_detectors and global_score > global_score_threshold:
            decision = AlertDecision.ALERT
            reason = f"Simple consensus from {len(active_detectors)} detectors, score={global_score:.2f}"
            
            consensus_strength = min(1.0, len(active_detectors) / len(scores))
            
            # Wysłanie Telegram alert (Etap 4 feature preview)
            alert_sent = self._send_simple_telegram_alert(token, global_score, active_detectors)
            
            print(f"[SIMPLE CONSENSUS] ALERT triggered: {len(active_detectors)} detectors >= {min_detectors}, global_score {global_score:.3f} (threshold: {global_score_threshold})")
            
        else:
            decision = AlertDecision.NO_ALERT
            alert_sent = False
            consensus_strength = 0.0
            
            if len(active_detectors) < min_detectors:
                reason = f"Insufficient consensus ({len(active_detectors)} signals ≥ {threshold}, need {min_detectors}), score={global_score:.2f}"
            else:
                reason = f"Global score too low ({global_score:.2f} ≤ {global_score_threshold}), detectors={len(active_detectors)}"
            
            print(f"[SIMPLE CONSENSUS] NO ALERT: {reason}")
        
        # Oblicz confidence na podstawie aktywnych detektorów - ZMIENIONE: nie penalizuj za nieaktywne
        if active_detectors:
            # Nie dziel przez len(scores) - używaj tylko aktywnych detektorów
            confidence = min(1.0, global_score)
        else:
            confidence = 0.0
        
        contributing_detectors = list(active_detectors.keys())
        
        # Generuj głosy detektorów dla display with type safety
        detector_votes = []
        for detector, score in scores.items():
            try:
                # Handle both dict and float format
                if isinstance(score, dict):
                    score_val = float(score.get('score', 0.0))
                else:
                    score_val = float(score) if score is not None else 0.0
                
                if score_val >= threshold:
                    detector_votes.append(AlertDecision.ALERT)
                else:
                    detector_votes.append(AlertDecision.NO_ALERT)
            except (ValueError, TypeError):
                detector_votes.append(AlertDecision.NO_ALERT)  # Default to NO_ALERT for invalid entries
        
        print(f"[SIMPLE CONSENSUS] Generated {len(detector_votes)} votes: {[v.value for v in detector_votes]}")
        
        # Zwróć rezultat w ConsensusResult format
        result = ConsensusResult(
            decision=decision,
            final_score=global_score,
            confidence=confidence,
            strategy_used=ConsensusStrategy.WEIGHTED_AVERAGE,  # Używamy jako proxy dla simple consensus
            contributing_detectors=contributing_detectors,
            reasoning=reason,
            consensus_strength=consensus_strength,
            votes=detector_votes,
            alert_sent=alert_sent,
            timestamp=datetime.now().isoformat()
        )
        
        # Zapisz do historii decyzji
        self._record_simple_decision(token, result, scores, active_detectors)
        
        return result
    
    def _send_simple_telegram_alert(self, token: str, score: float, detectors: Dict[str, float]) -> bool:
        """
        ETAP 2: Simple Telegram Alert (preview dla Etap 4)
        Wysyła prostą wiadomość alertu zgodnie z specyfikacją
        
        Args:
            token: Symbol tokena
            score: Global consensus score
            detectors: Dict aktywnych detektorów i ich scores
            
        Returns:
            True jeśli alert wysłany pomyślnie
        """
        try:
            # Format message zgodnie z Etap 2 specyfikacją
            message = f"🚨 [Consensus Alert] {token} | Score: {score:.2f}\n"
            message += f"Detectors: {', '.join(detectors.keys())}\n"
            message += f"Active signals: {len(detectors)}\n"
            
            # Dodaj breakdown aktywnych detektorów
            for detector, detector_score in detectors.items():
                emoji = "🔥" if detector_score >= 0.9 else "⚡" if detector_score >= 0.8 else "💫"
                message += f"  {emoji} {detector}: {detector_score:.3f}\n"
            
            message += f"Timestamp: {datetime.now().strftime('%H:%M:%S')}"
            
            print(f"[SIMPLE TELEGRAM] Alert message prepared:")
            print(message)
            
            # TODO: Implementacja rzeczywistego wysłania (Etap 4)
            # For now, log the alert
            if self.telegram_token and self.chat_id:
                print(f"[SIMPLE TELEGRAM] Would send to chat {self.chat_id}")
                # W Etap 4: requests.get(url, params=params)
            else:
                print(f"[SIMPLE TELEGRAM] No Telegram credentials configured")
            
            return True
            
        except Exception as e:
            print(f"[SIMPLE TELEGRAM ERROR] Failed to send alert: {e}")
            return False
    
    def _record_simple_decision(self, token: str, result: ConsensusResult, 
                               all_scores: Dict[str, float], active_detectors: Dict[str, float]):
        """
        Zapisuje simple consensus decision do historii
        
        Args:
            token: Symbol tokena
            result: ConsensusResult
            all_scores: Wszystkie scores detektorów
            active_detectors: Tylko aktywne detektory (>= 0.75)
        """
        decision_record = {
            'token': token,
            'timestamp': result.timestamp,
            'decision': result.decision.value,
            'final_score': result.final_score,
            'confidence': result.confidence,
            'strategy': 'SIMPLE_CONSENSUS',
            'consensus_strength': result.consensus_strength,
            'all_detector_scores': all_scores,
            'active_detector_scores': active_detectors,
            'active_count': len(active_detectors),
            'total_count': len(all_scores),
            'reasoning': result.reasoning,
            'alert_sent': result.alert_sent
        }
        
        self.decision_history.append(decision_record)
        
        # Ogranicz historię
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
        
        print(f"[SIMPLE CONSENSUS] Decision recorded: {result.decision.value} for {token}")
    
    def _convert_to_detector_scores(self, scores: Dict[str, float], 
                                   metadata: Dict[str, Any]) -> List[DetectorScore]:
        """Konwertuje raw scores na DetectorScore objects"""
        detector_scores = []
        
        for detector_name, score in scores.items():
            # Oblicz confidence na podstawie score i historical performance
            confidence = self._calculate_detector_confidence(detector_name, score)
            
            # Pobierz signal type z metadata
            signal_type = metadata.get(f"{detector_name}_signal_type", "unknown")
            
            detector_score = DetectorScore(
                name=detector_name,
                score=score,
                confidence=confidence,
                signal_type=signal_type,
                metadata=metadata.get(detector_name, {})
            )
            
            detector_scores.append(detector_score)
        
        return detector_scores
    
    def _execute_consensus_strategy(self, token: str, detector_scores: List[DetectorScore], 
                                   strategy: ConsensusStrategy) -> ConsensusResult:
        """Wykonuje wybraną strategię konsensusu"""
        
        if strategy == ConsensusStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_consensus(token, detector_scores)
        elif strategy == ConsensusStrategy.MAJORITY_VOTE:
            return self._majority_vote_consensus(token, detector_scores)
        elif strategy == ConsensusStrategy.ADAPTIVE_THRESHOLD:
            return self._adaptive_threshold_consensus(token, detector_scores)
        elif strategy == ConsensusStrategy.UNANIMOUS_AGREEMENT:
            return self._unanimous_agreement_consensus(token, detector_scores)
        elif strategy == ConsensusStrategy.DOMINANT_DETECTOR:
            return self._dominant_detector_consensus(token, detector_scores)
        else:
            # Fallback to weighted average
            return self._weighted_average_consensus(token, detector_scores)
    
    def _weighted_average_consensus(self, token: str, 
                                   detector_scores: List[DetectorScore]) -> ConsensusResult:
        """Strategia ważonej średniej z confidence weighting - ZMIENIONA: tylko aktywne detektory"""
        
        total_weighted_score = 0.0
        total_weight = 0.0
        contributing_detectors = []
        active_count = 0
        
        for detector_score in detector_scores:
            # 🔧 CRITICAL FIX #1b: Prevent dict multiplication in weighted average
            # Safely extract score and confidence values
            score_value = detector_score.score
            confidence_value = detector_score.confidence
            
            if isinstance(score_value, dict):
                score_value = score_value.get('score', score_value.get('value', 0.0))
            if isinstance(confidence_value, dict):
                confidence_value = confidence_value.get('confidence', confidence_value.get('value', 0.0))
            
            # Ensure numeric types
            score_value = float(score_value) if score_value is not None else 0.0
            confidence_value = float(confidence_value) if confidence_value is not None else 0.0
            
            # 🛠️ NOWA LOGIKA: Pomiń detektory ze score = 0
            if score_value <= 0.0:
                continue
            
            active_count += 1
            
            # Pobierz wagę detektora
            base_weight = self.detector_weights.get(detector_score.name, 0.25)
            
            # Dostosuj wagę przez confidence
            adjusted_weight = base_weight * confidence_value if confidence_value > 0 else base_weight * 0.5
            
            total_weighted_score += score_value * adjusted_weight
            total_weight += adjusted_weight
            contributing_detectors.append(detector_score.name)
        
        # Oblicz final score - normalizuj tylko na podstawie aktywnych detektorów
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Określ decyzję
        decision = self._determine_decision_from_score(final_score)
        
        # Oblicz consensus confidence with type safety - ZMIENIONE: tylko aktywne detektory
        confidence_values = []
        for ds in detector_scores:
            # Pomiń detektory ze score = 0
            score_val = ds.score
            if isinstance(score_val, dict):
                score_val = score_val.get('score', score_val.get('value', 0.0))
            score_val = float(score_val) if score_val is not None else 0.0
            
            if score_val <= 0.0:
                continue
                
            conf_val = ds.confidence
            if isinstance(conf_val, dict):
                conf_val = conf_val.get('confidence', conf_val.get('value', 0.0))
            conf_val = float(conf_val) if conf_val is not None else 0.0
            confidence_values.append(conf_val)
        
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        # Consensus strength bazuje tylko na aktywnych detektorach
        consensus_strength = min(1.0, active_count / max(2, len(contributing_detectors)))
        
        reasoning = f"Weighted average consensus: {final_score:.3f} from {len(contributing_detectors)} detectors"
        
        return ConsensusResult(
            decision=decision,
            final_score=final_score,
            confidence=avg_confidence * consensus_strength,
            strategy_used=ConsensusStrategy.WEIGHTED_AVERAGE,
            contributing_detectors=contributing_detectors,
            reasoning=reasoning,
            consensus_strength=consensus_strength,
            timestamp=datetime.now().isoformat()
        )
    
    def _majority_vote_consensus(self, token: str, 
                                detector_scores: List[DetectorScore]) -> ConsensusResult:
        """Strategia większości głosów"""
        
        alert_votes = 0
        no_alert_votes = 0
        watch_votes = 0
        contributing_detectors = []
        
        for detector_score in detector_scores:
            # Type-safe score comparison
            score_val = detector_score.score
            if isinstance(score_val, dict):
                score_val = score_val.get('score', score_val.get('value', 0.0))
            score_val = float(score_val) if score_val is not None else 0.0
            
            if score_val >= self.consensus_thresholds['alert_threshold']:
                alert_votes += 1
            elif score_val >= self.consensus_thresholds['watch_threshold']:
                watch_votes += 1
            else:
                no_alert_votes += 1
            
            contributing_detectors.append(detector_score.name)
        
        # Type-safe score calculation helper function
        def safe_score_average(detector_scores):
            score_values = []
            for ds in detector_scores:
                score_val = ds.score
                if isinstance(score_val, dict):
                    score_val = score_val.get('score', score_val.get('value', 0.0))
                score_val = float(score_val) if score_val is not None else 0.0
                score_values.append(score_val)
            return sum(score_values) / len(score_values) if score_values else 0.0

        # Określ zwycięską decyzję
        if alert_votes > (len(detector_scores) / 2):
            decision = AlertDecision.ALERT
            final_score = safe_score_average(detector_scores)
        elif (alert_votes + watch_votes) > no_alert_votes:
            decision = AlertDecision.WATCH
            final_score = safe_score_average(detector_scores)
        else:
            decision = AlertDecision.NO_ALERT
            final_score = safe_score_average(detector_scores)
        
        consensus_strength = max(alert_votes, watch_votes, no_alert_votes) / len(detector_scores)
        
        # Type-safe confidence calculation
        confidence_values = []
        for ds in detector_scores:
            conf_val = ds.confidence
            if isinstance(conf_val, dict):
                conf_val = conf_val.get('confidence', conf_val.get('value', 0.0))
            conf_val = float(conf_val) if conf_val is not None else 0.0
            confidence_values.append(conf_val)
        
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        
        reasoning = f"Majority vote: {alert_votes} alert, {watch_votes} watch, {no_alert_votes} no-alert votes"
        
        return ConsensusResult(
            decision=decision,
            final_score=final_score,
            confidence=avg_confidence,
            strategy_used=ConsensusStrategy.MAJORITY_VOTE,
            contributing_detectors=contributing_detectors,
            reasoning=reasoning,
            consensus_strength=consensus_strength,
            timestamp=datetime.now().isoformat()
        )
    
    def _adaptive_threshold_consensus(self, token: str, 
                                     detector_scores: List[DetectorScore]) -> ConsensusResult:
        """Strategia adaptacyjnego progu bazująca na historical performance"""
        
        # Dostosuj threshold na podstawie historical accuracy detektorów
        adjusted_threshold = self._calculate_adaptive_threshold(detector_scores)
        
        # Użyj weighted average z adaptive threshold
        result = self._weighted_average_consensus(token, detector_scores)
        
        # Dostosuj decyzję na podstawie adaptive threshold
        if result.final_score >= adjusted_threshold:
            result.decision = AlertDecision.ALERT
        elif result.final_score >= (adjusted_threshold * 0.7):
            result.decision = AlertDecision.WATCH
        else:
            result.decision = AlertDecision.NO_ALERT
        
        result.strategy_used = ConsensusStrategy.ADAPTIVE_THRESHOLD
        result.reasoning = f"Adaptive threshold ({adjusted_threshold:.3f}): {result.reasoning}"
        
        return result
    
    def _unanimous_agreement_consensus(self, token: str, 
                                      detector_scores: List[DetectorScore]) -> ConsensusResult:
        """Strategia wymagająca jednomyślności detektorów"""
        
        unanimous_threshold = self.consensus_thresholds['unanimous_threshold']
        
        # Sprawdź czy wszystkie detektory są zgodne (with type safety)
        high_scores = []
        low_scores = []
        for ds in detector_scores:
            score_val = ds.score
            if isinstance(score_val, dict):
                score_val = score_val.get('score', score_val.get('value', 0.0))
            score_val = float(score_val) if score_val is not None else 0.0
            
            if score_val >= unanimous_threshold:
                high_scores.append(ds)
            if score_val < 0.3:
                low_scores.append(ds)
        
        # Type-safe score calculation
        def safe_score_average(detector_scores):
            score_values = []
            for ds in detector_scores:
                score_val = ds.score
                if isinstance(score_val, dict):
                    score_val = score_val.get('score', score_val.get('value', 0.0))
                score_val = float(score_val) if score_val is not None else 0.0
                score_values.append(score_val)
            return sum(score_values) / len(score_values) if score_values else 0.0

        if len(high_scores) == len(detector_scores):
            # Wszystkie detektory wskazują alert
            decision = AlertDecision.ALERT
            final_score = safe_score_average(detector_scores)
            consensus_strength = 1.0
            reasoning = f"Unanimous agreement: all {len(detector_scores)} detectors above {unanimous_threshold}"
        
        elif len(low_scores) == len(detector_scores):
            # Wszystkie detektory wskazują no alert
            decision = AlertDecision.NO_ALERT
            final_score = safe_score_average(detector_scores)
            consensus_strength = 1.0
            reasoning = f"Unanimous agreement: all {len(detector_scores)} detectors below 0.3"
        
        else:
            # Brak jednomyślności - fallback do weighted average
            result = self._weighted_average_consensus(token, detector_scores)
            result.strategy_used = ConsensusStrategy.UNANIMOUS_AGREEMENT
            result.reasoning = f"No unanimous agreement, fallback to weighted average: {result.reasoning}"
            result.consensus_strength *= 0.5  # Reduce confidence due to disagreement
            return result
        
        # Type-safe confidence calculation
        confidence_values = []
        for ds in detector_scores:
            conf_val = ds.confidence
            if isinstance(conf_val, dict):
                conf_val = conf_val.get('confidence', conf_val.get('value', 0.0))
            conf_val = float(conf_val) if conf_val is not None else 0.0
            confidence_values.append(conf_val)
        
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        contributing_detectors = [ds.name for ds in detector_scores]
        
        return ConsensusResult(
            decision=decision,
            final_score=final_score,
            confidence=avg_confidence,
            strategy_used=ConsensusStrategy.UNANIMOUS_AGREEMENT,
            contributing_detectors=contributing_detectors,
            reasoning=reasoning,
            consensus_strength=consensus_strength,
            timestamp=datetime.now().isoformat()
        )
    
    def _dominant_detector_consensus(self, token: str, 
                                    detector_scores: List[DetectorScore]) -> ConsensusResult:
        """Strategia dominant detector - najwyższy score decyduje"""
        
        # Type-safe score calculations
        def safe_score_confidence_product(ds):
            score_val = ds.score
            conf_val = ds.confidence
            
            if isinstance(score_val, dict):
                score_val = score_val.get('score', score_val.get('value', 0.0))
            if isinstance(conf_val, dict):
                conf_val = conf_val.get('confidence', conf_val.get('value', 0.0))
                
            score_val = float(score_val) if score_val is not None else 0.0
            conf_val = float(conf_val) if conf_val is not None else 0.0
            
            return score_val * conf_val
        
        def safe_score_value(ds):
            score_val = ds.score
            if isinstance(score_val, dict):
                score_val = score_val.get('score', score_val.get('value', 0.0))
            return float(score_val) if score_val is not None else 0.0

        # Znajdź detektor z najwyższym score
        dominant_detector = max(detector_scores, key=safe_score_confidence_product)
        
        # Sprawdź czy dominant detector ma wystarczającą przewagę
        other_scores = [safe_score_value(ds) for ds in detector_scores if ds.name != dominant_detector.name]
        avg_others = sum(other_scores) / len(other_scores) if other_scores else 0.0
        
        dominance_factor = safe_score_value(dominant_detector) - avg_others
        
        dominant_score = safe_score_value(dominant_detector)
        dominant_confidence = dominant_detector.confidence
        if isinstance(dominant_confidence, dict):
            dominant_confidence = dominant_confidence.get('confidence', dominant_confidence.get('value', 0.0))
        dominant_confidence = float(dominant_confidence) if dominant_confidence is not None else 0.0
        
        if dominance_factor >= 0.2 and dominant_score >= self.consensus_thresholds['alert_threshold']:
            decision = AlertDecision.ALERT
            consensus_strength = min(1.0, dominance_factor / 0.5)
        elif dominant_score >= self.consensus_thresholds['watch_threshold']:
            decision = AlertDecision.WATCH
            consensus_strength = min(1.0, dominance_factor / 0.3)
        else:
            decision = AlertDecision.NO_ALERT
            consensus_strength = 0.5
        
        reasoning = f"Dominant detector {dominant_detector.name}: {dominant_score:.3f} (dominance: {dominance_factor:.3f})"
        contributing_detectors = [ds.name for ds in detector_scores]
        
        return ConsensusResult(
            decision=decision,
            final_score=dominant_score,
            confidence=dominant_confidence,
            strategy_used=ConsensusStrategy.DOMINANT_DETECTOR,
            contributing_detectors=contributing_detectors,
            reasoning=reasoning,
            consensus_strength=consensus_strength,
            timestamp=datetime.now().isoformat()
        )
    
    def _determine_decision_from_score(self, score: float) -> AlertDecision:
        """Określa decyzję na podstawie final score"""
        if score >= self.consensus_thresholds['alert_threshold']:
            return AlertDecision.ALERT
        elif score >= self.consensus_thresholds['watch_threshold']:
            return AlertDecision.WATCH
        else:
            return AlertDecision.NO_ALERT
    
    def _calculate_detector_confidence(self, detector_name: str, score: float) -> float:
        """Oblicza confidence detektora na podstawie historical performance"""
        
        # Bazowa confidence na podstawie score
        base_confidence = min(1.0, score * 1.2)
        
        # Dostosowanie na podstawie historical performance
        if detector_name in self.detector_performance:
            historical_accuracy = self.detector_performance[detector_name].get('accuracy', 0.5)
            confidence_modifier = (historical_accuracy - 0.5) * 0.4  # Range: -0.2 to +0.2
            base_confidence = max(0.1, min(1.0, base_confidence + confidence_modifier))
        
        return base_confidence
    
    def _calculate_adaptive_threshold(self, detector_scores: List[DetectorScore]) -> float:
        """Oblicza adaptacyjny threshold na podstawie detector performance"""
        
        base_threshold = self.consensus_thresholds['alert_threshold']
        
        # Dostosuj threshold na podstawie average confidence detektorów with type safety
        confidence_values = []
        for ds in detector_scores:
            conf_val = ds.confidence
            if isinstance(conf_val, dict):
                conf_val = conf_val.get('confidence', conf_val.get('value', 0.0))
            conf_val = float(conf_val) if conf_val is not None else 0.0
            confidence_values.append(conf_val)
        
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        
        # Wyższa confidence = niższy threshold, niższa confidence = wyższy threshold
        confidence_adjustment = (0.8 - avg_confidence) * 0.2  # Range: -0.1 to +0.1
        
        adaptive_threshold = max(0.5, min(0.9, base_threshold + confidence_adjustment))
        
        return adaptive_threshold
    
    def _create_no_alert_result(self, reason: str, strategy: ConsensusStrategy) -> ConsensusResult:
        """Tworzy wynik NO_ALERT z podanym powodem"""
        return ConsensusResult(
            decision=AlertDecision.NO_ALERT,
            final_score=0.0,
            confidence=0.0,
            strategy_used=strategy,
            contributing_detectors=[],
            reasoning=reason,
            consensus_strength=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def _record_decision(self, token: str, result: ConsensusResult, 
                        detector_scores: List[DetectorScore]):
        """Zapisuje decyzję do historii dla future learning"""
        
        decision_record = {
            'token': token,
            'timestamp': result.timestamp,
            'decision': result.decision.value,
            'final_score': result.final_score,
            'confidence': result.confidence,
            'strategy': result.strategy_used.value,
            'consensus_strength': result.consensus_strength,
            'detector_scores': {ds.name: (ds.score.get('score', ds.score.get('value', 0.0)) if isinstance(ds.score, dict) else float(ds.score) if ds.score is not None else 0.0) for ds in detector_scores},
            'reasoning': result.reasoning
        }
        
        self.decision_history.append(decision_record)
        
        # Ogranicz historię do ostatnich 1000 decyzji
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def _send_consensus_alert(self, token: str, result: ConsensusResult, 
                             detector_scores: List[DetectorScore]) -> bool:
        """Wysyła consensus alert na Telegram"""
        
        try:
            # Format alert message
            message = self._format_consensus_alert_message(token, result, detector_scores)
            
            # TODO: Implement actual Telegram sending logic
            # For now, just log the alert
            print(f"[CONSENSUS ALERT] {token}: {message}")
            
            return True
            
        except Exception as e:
            print(f"[CONSENSUS ALERT ERROR] Failed to send alert for {token}: {e}")
            return False
    
    def _format_consensus_alert_message(self, token: str, result: ConsensusResult, 
                                       detector_scores: List[DetectorScore]) -> str:
        """Formatuje wiadomość alertu konsensusu"""
        
        # Header
        message = f"🤖 CONSENSUS ALERT: {token}\n\n"
        
        # Decision details
        message += f"📊 Final Decision: {result.decision.value}\n"
        message += f"🎯 Consensus Score: {result.final_score:.3f}\n"
        message += f"📈 Confidence: {result.confidence:.3f}\n"
        message += f"🔗 Strategy: {result.strategy_used.value}\n"
        message += f"💪 Consensus Strength: {result.consensus_strength:.3f}\n\n"
        
        # Detector breakdown
        message += "🔍 Detector Breakdown:\n"
        for detector_score in detector_scores:
            emoji = "🔥" if detector_score.score >= 0.7 else "⚡" if detector_score.score >= 0.5 else "💧"
            message += f"  {emoji} {detector_score.name}: {detector_score.score:.3f}\n"
        
        # Reasoning
        message += f"\n💡 Reasoning: {result.reasoning}\n"
        
        # Timestamp
        message += f"⏰ Time: {result.timestamp}"
        
        return message
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Pobiera statystyki consensus decision engine"""
        
        if not self.decision_history:
            return {
                'total_decisions': 0,
                'alert_rate': 0.0,
                'average_confidence': 0.0,
                'strategy_usage': {},
                'detector_contribution': {}
            }
        
        total_decisions = len(self.decision_history)
        alert_decisions = sum(1 for d in self.decision_history if d['decision'] == 'ALERT')
        
        # Strategy usage statistics
        strategy_usage = {}
        for decision in self.decision_history:
            strategy = decision['strategy']
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        # Average confidence
        avg_confidence = sum(d['confidence'] for d in self.decision_history) / total_decisions
        
        # Detector contribution analysis
        detector_contribution = {}
        for decision in self.decision_history:
            # Handle both old and new decision formats
            detector_scores = decision.get('detector_scores', {})
            if not detector_scores and 'all_detector_scores' in decision:
                detector_scores = decision['all_detector_scores']
            elif not detector_scores and 'active_detector_scores' in decision:
                detector_scores = decision['active_detector_scores']
            elif not detector_scores and 'all_scores' in decision:
                detector_scores = decision['all_scores']
                
            for detector, score in detector_scores.items():
                if detector not in detector_contribution:
                    detector_contribution[detector] = {'count': 0, 'avg_score': 0.0}
                detector_contribution[detector]['count'] += 1
                # Handle both float scores and dict scores
                numeric_score = score if isinstance(score, (int, float)) else score.get('score', 0.0) if isinstance(score, dict) else 0.0
                detector_contribution[detector]['avg_score'] += numeric_score
        
        # Calculate average scores
        for detector in detector_contribution:
            count = detector_contribution[detector]['count']
            detector_contribution[detector]['avg_score'] /= count
        
        return {
            'total_decisions': total_decisions,
            'alert_rate': alert_decisions / total_decisions,
            'average_confidence': avg_confidence,
            'strategy_usage': strategy_usage,
            'detector_contribution': detector_contribution,
            'recent_decisions': self.decision_history[-10:] if self.decision_history else []
        }


def create_consensus_engine() -> ConsensusDecisionEngine:
    """Factory function dla tworzenia Consensus Decision Engine"""
    return ConsensusDecisionEngine()


def test_consensus_engine():
    """Test funkcja dla Consensus Decision Engine"""
    
    print("🧪 Testing Consensus Decision Engine...")
    
    # Utwórz engine
    engine = create_consensus_engine()
    
    # ETAP 2: Test Simple Consensus Logic
    print("\n🎯 ETAP 2: Testing Simple Consensus Logic")
    print("=" * 60)
    
    # Test case 1: Simple Consensus - Alert scenario
    print("\n🔬 Test 1: Simple Consensus - Alert Trigger")
    alert_scores = {
        "WhaleCLIP": 0.84,
        "DiamondWhale": 0.78,
        "CaliforniumWhale": 0.92,
        "StealthSignal": 0.81
    }
    
    result = engine.run("ALERTTEST", alert_scores, use_simple_consensus=True)
    print(f"Result: {result.decision.value} (score: {result.final_score:.3f})")
    print(f"Active detectors: {result.contributing_detectors}")
    print(f"Reasoning: {result.reasoning}")
    
    # Test case 2: Simple Consensus - No Alert (insufficient detectors)
    print("\n🔬 Test 2: Simple Consensus - Insufficient Detectors")
    insufficient_scores = {
        "WhaleCLIP": 0.74,  # Below 0.75
        "DiamondWhale": 0.82,  # Only one above 0.75
        "CaliforniumWhale": 0.69,  # Below 0.75
        "StealthSignal": 0.71   # Below 0.75
    }
    
    result = engine.run("NOALERT1", insufficient_scores, use_simple_consensus=True)
    print(f"Result: {result.decision.value} (score: {result.final_score:.3f})")
    print(f"Reasoning: {result.reasoning}")
    
    # Test case 3: Simple Consensus - No Alert (low global score)
    print("\n🔬 Test 3: Simple Consensus - Low Global Score")
    low_global_scores = {
        "WhaleCLIP": 0.76,  # Above threshold
        "DiamondWhale": 0.78,  # Above threshold  
        "CaliforniumWhale": 0.77,  # Above threshold
        "StealthSignal": 0.75   # Exactly at threshold
    }
    
    result = engine.run("NOALERT2", low_global_scores, use_simple_consensus=True)
    print(f"Result: {result.decision.value} (score: {result.final_score:.3f})")
    print(f"Global score: {result.final_score:.3f} (need > 0.8)")
    print(f"Reasoning: {result.reasoning}")
    
    # Test case 4: Simple Consensus - Borderline case
    print("\n🔬 Test 4: Simple Consensus - Borderline Trigger")
    borderline_scores = {
        "WhaleCLIP": 0.81,
        "DiamondWhale": 0.80,
        "CaliforniumWhale": 0.82,
        "StealthSignal": 0.74   # Below threshold
    }
    
    result = engine.run("BORDERLINE", borderline_scores, use_simple_consensus=True)
    print(f"Result: {result.decision.value} (score: {result.final_score:.3f})")
    print(f"Active detectors: {len(result.contributing_detectors)}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\n" + "=" * 60)
    print("🎯 ADVANCED STRATEGIES: Testing Original Consensus Methods")
    
    # Test case 5: Weighted Average Consensus
    print("\n🔬 Test 5: Weighted Average Consensus")
    scores = {
        "WhaleCLIP": 0.84,
        "DiamondWhale": 0.78,
        "CaliforniumWhale": 0.92,
        "StealthSignal": 0.81
    }
    
    result = engine.run("TESTUSDT", scores, ConsensusStrategy.WEIGHTED_AVERAGE)
    print(f"Result: {result.decision.value} (score: {result.final_score:.3f})")
    
    # Test case 6: Majority Vote
    print("\n🔬 Test 6: Majority Vote Consensus")
    result = engine.run("TESTUSDT", scores, ConsensusStrategy.MAJORITY_VOTE)
    print(f"Result: {result.decision.value} (reasoning: {result.reasoning})")
    
    # Test case 7: Dominant Detector
    print("\n🔬 Test 7: Dominant Detector Consensus")
    dominant_scores = {
        "WhaleCLIP": 0.45,
        "DiamondWhale": 0.35,
        "CaliforniumWhale": 0.95,  # Dominant
        "StealthSignal": 0.40
    }
    
    result = engine.run("TESTUSDT", dominant_scores, ConsensusStrategy.DOMINANT_DETECTOR)
    print(f"Result: {result.decision.value} (reasoning: {result.reasoning})")
    
    # Statystyki
    print("\n📊 Final Consensus Statistics:")
    stats = engine.get_consensus_statistics()
    print(f"Total decisions: {stats['total_decisions']}")
    print(f"Alert rate: {stats['alert_rate']:.2%}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    
    print("\n✅ ETAP 2 Simple Consensus + Advanced Strategies tests completed!")
    return True


def test_etap2_simple_consensus():
    """Dedykowany test dla Etap 2 Simple Consensus Logic"""
    
    print("🎯 ETAP 2 SIMPLE CONSENSUS - DEDICATED TESTING")
    print("=" * 80)
    
    engine = create_consensus_engine()
    
    test_scenarios = [
        {
            'name': 'Perfect Consensus (4 detectors > 0.75, high avg)',
            'scores': {"WhaleCLIP": 0.88, "DiamondWhale": 0.85, "CaliforniumWhale": 0.91, "StealthSignal": 0.83},
            'expected': 'ALERT',
            'reason': 'All 4 detectors > 0.75, global_score = 0.87 > 0.8'
        },
        {
            'name': 'Minimum Alert Case (2 detectors > 0.75, avg = 0.81)',
            'scores': {"WhaleCLIP": 0.80, "DiamondWhale": 0.82, "CaliforniumWhale": 0.74, "StealthSignal": 0.70},
            'expected': 'ALERT',
            'reason': '2 detectors above threshold, avg > 0.8'
        },
        {
            'name': 'Insufficient Detectors (1 detector > 0.75)',
            'scores': {"WhaleCLIP": 0.85, "DiamondWhale": 0.74, "CaliforniumWhale": 0.70, "StealthSignal": 0.68},
            'expected': 'NO_ALERT',
            'reason': 'Only 1 detector above 0.75, need minimum 2'
        },
        {
            'name': 'Low Global Score (3 detectors > 0.75, avg = 0.77)',
            'scores': {"WhaleCLIP": 0.76, "DiamondWhale": 0.78, "CaliforniumWhale": 0.77, "StealthSignal": 0.73},
            'expected': 'NO_ALERT',
            'reason': '3 detectors active but global_score 0.77 < 0.8'
        },
        {
            'name': 'Exact Threshold Test (score = 0.8)',
            'scores': {"WhaleCLIP": 0.80, "DiamondWhale": 0.80, "CaliforniumWhale": 0.75, "StealthSignal": 0.73},
            'expected': 'NO_ALERT',
            'reason': 'Global score exactly 0.8, need > 0.8'
        }
    ]
    
    success_count = 0
    total_tests = len(test_scenarios)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔬 Test {i}: {scenario['name']}")
        print(f"   Scores: {scenario['scores']}")
        print(f"   Expected: {scenario['expected']}")
        print(f"   Scenario: {scenario['reason']}")
        
        result = engine.run(f"TEST{i}", scenario['scores'], use_simple_consensus=True)
        
        success = result.decision.value == scenario['expected']
        status = "✅ PASS" if success else "❌ FAIL"
        
        print(f"   Result: {result.decision.value} (score: {result.final_score:.3f}) {status}")
        print(f"   Active detectors: {len(result.contributing_detectors)}")
        print(f"   Reasoning: {result.reasoning}")
        
        if success:
            success_count += 1
    
    print(f"\n📊 ETAP 2 TEST SUMMARY:")
    print(f"Tests passed: {success_count}/{total_tests} ({(success_count/total_tests)*100:.1f}%)")
    
    if success_count == total_tests:
        print("🎉 ETAP 2 SIMPLE CONSENSUS LOGIC: COMPLETE ✅")
        return True
    else:
        print("❌ Some tests failed - review implementation")
        return False


def test_etap4_enhanced_telegram_alerts():
    """
    ETAP 4: Enhanced Telegram Alert Testing
    Test comprehensive alert message formatting z pełną informacją o detektorach
    """
    print("🚀 ETAP 4 ENHANCED TELEGRAM ALERTS - DEDICATED TESTING")
    print("=" * 80)
    
    engine = ConsensusDecisionEngine()
    
    test_scenarios = [
        {
            'name': 'High Confidence Multi-Booster Alert',
            'token': 'BTCUSDT',
            'global_score': 0.925,
            'active_detectors': {
                "CaliforniumWhale": {"score": 0.94, "confidence": 0.91, "weight": 0.34},
                "DiamondWhale": {"score": 0.89, "confidence": 0.87, "weight": 0.32},
                "WhaleCLIP": {"score": 0.82, "confidence": 0.78, "weight": 0.25},
                "StealthSignal": {"score": 0.76, "confidence": 0.68, "weight": 0.18}
            },
            'boosted_detectors': ['CaliforniumWhale', 'DiamondWhale'],
            'expected_confidence': 'VERY HIGH ⭐⭐⭐'
        },
        {
            'name': 'Medium Confidence Single Booster',
            'token': 'ETHUSDT',
            'global_score': 0.834,
            'active_detectors': {
                "WhaleCLIP": {"score": 0.91, "confidence": 0.85, "weight": 0.45},
                "DiamondWhale": {"score": 0.78, "confidence": 0.78, "weight": 0.35},
                "StealthSignal": {"score": 0.71, "confidence": 0.74, "weight": 0.20}
            },
            'boosted_detectors': ['WhaleCLIP'],
            'expected_confidence': 'HIGH ⭐⭐'
        },
        {
            'name': 'No Boosters Alert',
            'token': 'ADAUSDT',
            'global_score': 0.802,
            'active_detectors': {
                "CaliforniumWhale": {"score": 0.83, "confidence": 0.76, "weight": 0.40},
                "WhaleCLIP": {"score": 0.79, "confidence": 0.71, "weight": 0.35},
                "StealthSignal": {"score": 0.74, "confidence": 0.65, "weight": 0.25}
            },
            'boosted_detectors': [],
            'expected_confidence': 'GOOD ⭐'
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_scenarios)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔬 Test {i}: {scenario['name']}")
        print(f"   Token: {scenario['token']}")
        print(f"   Global Score: {scenario['global_score']}")
        print(f"   Active Detectors: {len(scenario['active_detectors'])}")
        print(f"   Boosted: {len(scenario['boosted_detectors'])}")
        print(f"   Expected Confidence: {scenario['expected_confidence']}")
        
        try:
            # Test enhanced alert message formatting
            success = engine.send_telegram_alert(
                token=scenario['token'],
                global_score=scenario['global_score'],
                active_detectors=scenario['active_detectors'],
                boosted_detectors=scenario['boosted_detectors']
            )
            
            # Verify confidence assessment
            avg_confidence = sum(d['confidence'] for d in scenario['active_detectors'].values()) / len(scenario['active_detectors'])
            confidence_level = engine._assess_confidence_level(avg_confidence, len(scenario['boosted_detectors']))
            
            if confidence_level == scenario['expected_confidence']:
                print(f"   Result: ✅ PASS - Confidence assessment correct")
                passed_tests += 1
            else:
                print(f"   Result: ❌ FAIL - Expected '{scenario['expected_confidence']}', got '{confidence_level}'")
        
        except Exception as e:
            print(f"   Result: ❌ FAIL - Exception: {e}")
    
    print(f"\n📊 ETAP 4 TEST SUMMARY:")
    print(f"Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ETAP 4 ENHANCED TELEGRAM ALERTS: COMPLETE ✅")
        return True
    else:
        print("❌ Some tests failed - review implementation")
        return False

def test_etap3_dynamic_boosting():
    """Dedykowany test dla Etap 3 Dynamic Boosting Logic"""
    
    print("🚀 ETAP 3 DYNAMIC BOOSTING - DEDICATED TESTING")
    print("=" * 80)
    
    engine = create_consensus_engine()
    
    # Test scenarios z extended format (confidence + weight)
    test_scenarios = [
        {
            'name': 'Perfect Boosted Consensus (2 boosted detectors)',
            'scores': {
                "WhaleCLIP": {"score": 0.92, "confidence": 0.88, "weight": 0.28},
                "DiamondWhale": {"score": 0.91, "confidence": 0.86, "weight": 0.33},
                "CaliforniumWhale": {"score": 0.78, "confidence": 0.65, "weight": 0.25},
                "StealthSignal": {"score": 0.74, "confidence": 0.59, "weight": 0.14}
            },
            'expected': 'ALERT',
            'reason': 'Multiple boosted detectors, high weighted score'
        },
        {
            'name': 'Minimum Dynamic Alert (2 active, weighted score > 0.8)',
            'scores': {
                "WhaleCLIP": {"score": 0.87, "confidence": 0.70, "weight": 0.50},
                "DiamondWhale": {"score": 0.82, "confidence": 0.72, "weight": 0.45},
                "CaliforniumWhale": {"score": 0.74, "confidence": 0.55, "weight": 0.25},
                "StealthSignal": {"score": 0.70, "confidence": 0.50, "weight": 0.14}
            },
            'expected': 'ALERT',
            'reason': '2 active detectors, weighted score > 0.8'
        },
        {
            'name': 'Low Confidence Block (high scores, low confidence)',
            'scores': {
                "WhaleCLIP": {"score": 0.90, "confidence": 0.55, "weight": 0.30},
                "DiamondWhale": {"score": 0.88, "confidence": 0.58, "weight": 0.30},
                "CaliforniumWhale": {"score": 0.85, "confidence": 0.45, "weight": 0.25},
                "StealthSignal": {"score": 0.82, "confidence": 0.50, "weight": 0.15}
            },
            'expected': 'NO_ALERT',
            'reason': 'All detectors below confidence threshold (0.60)'
        },
        {
            'name': 'Low Weighted Score (active detectors, low total weight)',
            'scores': {
                "WhaleCLIP": {"score": 0.76, "confidence": 0.65, "weight": 0.15},
                "DiamondWhale": {"score": 0.78, "confidence": 0.68, "weight": 0.20},
                "CaliforniumWhale": {"score": 0.77, "confidence": 0.62, "weight": 0.10},
                "StealthSignal": {"score": 0.75, "confidence": 0.60, "weight": 0.05}
            },
            'expected': 'NO_ALERT',
            'reason': 'Low weighted score despite 4 active detectors'
        },
        {
            'name': 'Single Booster Scenario (one very strong detector)',
            'scores': {
                "WhaleCLIP": {"score": 0.95, "confidence": 0.90, "weight": 0.60},
                "DiamondWhale": {"score": 0.80, "confidence": 0.70, "weight": 0.35},
                "CaliforniumWhale": {"score": 0.70, "confidence": 0.50, "weight": 0.20},
                "StealthSignal": {"score": 0.65, "confidence": 0.45, "weight": 0.10}
            },
            'expected': 'ALERT',
            'reason': 'One boosted detector + one active, high weighted contribution'
        }
    ]
    
    success_count = 0
    total_tests = len(test_scenarios)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔬 Test {i}: {scenario['name']}")
        print(f"   Scenario: {scenario['reason']}")
        print(f"   Expected: {scenario['expected']}")
        
        # Sprawdź booster candidates
        booster_candidates = []
        for detector, data in scenario['scores'].items():
            if data['confidence'] > 0.85 and data['score'] > 0.90:
                booster_candidates.append(detector)
        
        if booster_candidates:
            print(f"   Expected boosters: {', '.join(booster_candidates)}")
        
        result = engine.run(f"DTEST{i}", scenario['scores'], use_dynamic_boosting=True)
        
        success = result.decision.value == scenario['expected']
        status = "✅ PASS" if success else "❌ FAIL"
        
        print(f"   Result: {result.decision.value} (score: {result.final_score:.3f}) {status}")
        print(f"   Active detectors: {len(result.contributing_detectors)}")
        print(f"   Reasoning: {result.reasoning}")
        
        if success:
            success_count += 1
    
    print(f"\n📊 ETAP 3 TEST SUMMARY:")
    print(f"Tests passed: {success_count}/{total_tests} ({(success_count/total_tests)*100:.1f}%)")
    
    if success_count == total_tests:
        print("🎉 ETAP 3 DYNAMIC BOOSTING LOGIC: COMPLETE ✅")
        return True
    else:
        print("❌ Some tests failed - review implementation")
        return False


def test_etap3_format_conversion():
    """Test konwersji formatów między simple a extended"""
    
    print("\n🔄 TESTING FORMAT CONVERSION")
    print("=" * 50)
    
    engine = create_consensus_engine()
    
    # Test 1: Simple format conversion
    print("\n🔬 Test 1: Simple to Extended Format Conversion")
    simple_scores = {
        "WhaleCLIP": 0.84,
        "DiamondWhale": 0.78,
        "CaliforniumWhale": 0.92,
        "StealthSignal": 0.81
    }
    
    result = engine.run("CONVERT1", simple_scores, use_dynamic_boosting=True)
    print(f"Simple format processed successfully: {result.decision.value}")
    
    # Test 2: Extended format direct use
    print("\n🔬 Test 2: Extended Format Direct Processing")
    extended_scores = {
        "WhaleCLIP": {"score": 0.84, "confidence": 0.75, "weight": 0.28},
        "DiamondWhale": {"score": 0.78, "confidence": 0.80, "weight": 0.33},
        "CaliforniumWhale": {"score": 0.92, "confidence": 0.88, "weight": 0.25},
        "StealthSignal": {"score": 0.81, "confidence": 0.70, "weight": 0.14}
    }
    
    result = engine.run("CONVERT2", extended_scores, use_dynamic_boosting=True)
    print(f"Extended format processed successfully: {result.decision.value}")
    
    print("✅ FORMAT CONVERSION TESTS COMPLETE")
    return True


def test_etap5_fallback_logic():
    """
    ETAP 5: Test fallback logic dla bardzo silnych detektorów
    Sprawdza czy pojedynczy silny detektor może triggerować instant alert
    """
    print("\n[TEST ETAP 5] Fallback Logic Testing...")
    engine = create_consensus_engine()
    
    # Test 1: Fallback trigger - DiamondWhale przekracza próg
    fallback_scores = {
        "DiamondWhale": {"score": 0.94, "confidence": 0.89, "weight": 0.25},
        "WhaleCLIP": {"score": 0.45, "confidence": 0.55, "weight": 0.25},
        "CaliforniumWhale": {"score": 0.30, "confidence": 0.40, "weight": 0.30}
    }
    
    result = engine.run(
        token="TESTUSDT",
        scores=fallback_scores,
        use_dynamic_boosting=True
    )
    
    assert result.decision == AlertDecision.ALERT
    assert result.strategy_used == ConsensusStrategy.DOMINANT_DETECTOR
    assert "DiamondWhale" in result.contributing_detectors
    assert "Fallback Trigger" in result.reasoning
    assert result.consensus_strength == 1.0
    print("✅ Test 1: Fallback trigger - PASSED")
    
    # Test 2: No fallback trigger - high score but low confidence
    no_fallback_scores = {
        "DiamondWhale": {"score": 0.95, "confidence": 0.75, "weight": 0.25},  # < 0.85 confidence
        "WhaleCLIP": {"score": 0.50, "confidence": 0.60, "weight": 0.25},
        "CaliforniumWhale": {"score": 0.40, "confidence": 0.50, "weight": 0.30}
    }
    
    result = engine.run(
        token="TESTUSDT2", 
        scores=no_fallback_scores,
        use_dynamic_boosting=True
    )
    
    # Should fall through to normal dynamic boosting logic (insufficient consensus)
    assert result.decision == AlertDecision.NO_ALERT
    assert "Fallback Trigger" not in result.reasoning
    print("✅ Test 2: No fallback trigger (low confidence) - PASSED")
    
    # Test 3: CaliforniumWhale fallback trigger
    fallback_californium = {
        "CaliforniumWhale": {"score": 0.96, "confidence": 0.91, "weight": 0.30},
        "DiamondWhale": {"score": 0.40, "confidence": 0.50, "weight": 0.25},
        "WhaleCLIP": {"score": 0.35, "confidence": 0.45, "weight": 0.25}
    }
    
    result = engine.run(
        token="TESTUSDT4",
        scores=fallback_californium,
        use_dynamic_boosting=True
    )
    
    assert result.decision == AlertDecision.ALERT
    assert result.strategy_used == ConsensusStrategy.DOMINANT_DETECTOR
    assert "CaliforniumWhale" in result.contributing_detectors
    assert "Fallback Trigger" in result.reasoning
    print("✅ Test 3: CaliforniumWhale fallback trigger - PASSED")
    
    print("✅ ETAP 5 Fallback Logic: All tests PASSED!")

if __name__ == "__main__":
    test_consensus_engine()
    test_etap5_fallback_logic()