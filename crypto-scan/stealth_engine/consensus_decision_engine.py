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
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


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
        
        print("[CONSENSUS ENGINE] Initialized multi-agent decision layer")
    
    def run(self, token: str, scores: Dict[str, float], 
            strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_AVERAGE,
            metadata: Dict[str, Any] = None, 
            use_simple_consensus: bool = False) -> ConsensusResult:
        """
        Główna funkcja uruchamiająca consensus decision process
        
        Args:
            token: Symbol tokena (np. 'RSRUSDT')
            scores: Dict z nazwami detektorów i ich scores
                   {"WhaleCLIP": 0.84, "DiamondWhale": 0.78, "CaliforniumWhale": 0.92, "StealthSignal": 0.81}
            strategy: Strategia konsensusu do użycia
            metadata: Dodatkowe metadane z detektorów
            use_simple_consensus: True dla prostej logiki (Etap 2), False dla advanced strategies
            
        Returns:
            ConsensusResult: Kompletny wynik decyzji konsensusu
        """
        print(f"[CONSENSUS ENGINE] Processing {token} with {len(scores)} detectors")
        print(f"[CONSENSUS ENGINE] Strategy: {strategy.value if not use_simple_consensus else 'SIMPLE_CONSENSUS'}")
        print(f"[CONSENSUS ENGINE] Input scores: {scores}")
        
        # Walidacja input data
        if not scores:
            return self._create_no_alert_result("No detector scores provided", strategy)
        
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
        
        # Threshold dla aktywnych detektorów (zgodnie z specyfikacją)
        threshold = 0.75
        global_score_threshold = 0.8
        min_detectors = 2
        
        # Znajdź aktywne detektory (score >= 0.75)
        active_detectors = {k: v for k, v in scores.items() if v >= threshold}
        
        print(f"[SIMPLE CONSENSUS] Active detectors (>= {threshold}): {len(active_detectors)}")
        print(f"[SIMPLE CONSENSUS] Active scores: {active_detectors}")
        
        # Oblicz global_score jako średnią z aktywnych sygnałów
        if active_detectors:
            global_score = sum(active_detectors.values()) / len(active_detectors)
        else:
            global_score = 0.0
        
        print(f"[SIMPLE CONSENSUS] Global score (avg of active): {global_score:.3f}")
        
        # Logika decyzyjna zgodnie z Etap 2:
        # min. 2 detektory > 0.75 AND global_score > 0.8
        if len(active_detectors) >= min_detectors and global_score > global_score_threshold:
            decision = AlertDecision.ALERT
            reason = f"Simple consensus from {len(active_detectors)} detectors, score={global_score:.2f}"
            consensus_strength = min(1.0, len(active_detectors) / len(scores))
            
            # Wysłanie Telegram alert (Etap 4 feature preview)
            alert_sent = self._send_simple_telegram_alert(token, global_score, active_detectors)
            
            print(f"[SIMPLE CONSENSUS] ALERT triggered: {len(active_detectors)} detectors >= {min_detectors}, global_score {global_score:.3f} > {global_score_threshold}")
            
        else:
            decision = AlertDecision.NO_ALERT
            alert_sent = False
            consensus_strength = 0.0
            
            if len(active_detectors) < min_detectors:
                reason = f"Insufficient consensus ({len(active_detectors)} signals ≥ {threshold}, need {min_detectors}), score={global_score:.2f}"
            else:
                reason = f"Global score too low ({global_score:.2f} ≤ {global_score_threshold}), detectors={len(active_detectors)}"
            
            print(f"[SIMPLE CONSENSUS] NO ALERT: {reason}")
        
        # Oblicz confidence na podstawie aktywnych detektorów
        if active_detectors:
            confidence = min(1.0, global_score * (len(active_detectors) / len(scores)))
        else:
            confidence = 0.0
        
        contributing_detectors = list(active_detectors.keys())
        
        # Zwróć rezultat w ConsensusResult format
        result = ConsensusResult(
            decision=decision,
            final_score=global_score,
            confidence=confidence,
            strategy_used=ConsensusStrategy.WEIGHTED_AVERAGE,  # Używamy jako proxy dla simple consensus
            contributing_detectors=contributing_detectors,
            reasoning=reason,
            consensus_strength=consensus_strength,
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
        """Strategia ważonej średniej z confidence weighting"""
        
        total_weighted_score = 0.0
        total_weight = 0.0
        contributing_detectors = []
        
        for detector_score in detector_scores:
            # Pobierz wagę detektora
            base_weight = self.detector_weights.get(detector_score.name, 0.25)
            
            # Dostosuj wagę przez confidence
            adjusted_weight = base_weight * detector_score.confidence
            
            total_weighted_score += detector_score.score * adjusted_weight
            total_weight += adjusted_weight
            contributing_detectors.append(detector_score.name)
        
        # Oblicz final score
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Określ decyzję
        decision = self._determine_decision_from_score(final_score)
        
        # Oblicz consensus confidence
        avg_confidence = sum(ds.confidence for ds in detector_scores) / len(detector_scores)
        consensus_strength = min(1.0, total_weight / sum(self.detector_weights.values()))
        
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
            if detector_score.score >= self.consensus_thresholds['alert_threshold']:
                alert_votes += 1
            elif detector_score.score >= self.consensus_thresholds['watch_threshold']:
                watch_votes += 1
            else:
                no_alert_votes += 1
            
            contributing_detectors.append(detector_score.name)
        
        # Określ zwycięską decyzję
        if alert_votes > (len(detector_scores) / 2):
            decision = AlertDecision.ALERT
            final_score = sum(ds.score for ds in detector_scores) / len(detector_scores)
        elif (alert_votes + watch_votes) > no_alert_votes:
            decision = AlertDecision.WATCH
            final_score = sum(ds.score for ds in detector_scores) / len(detector_scores)
        else:
            decision = AlertDecision.NO_ALERT
            final_score = sum(ds.score for ds in detector_scores) / len(detector_scores)
        
        consensus_strength = max(alert_votes, watch_votes, no_alert_votes) / len(detector_scores)
        avg_confidence = sum(ds.confidence for ds in detector_scores) / len(detector_scores)
        
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
        
        # Sprawdź czy wszystkie detektory są zgodne
        high_scores = [ds for ds in detector_scores if ds.score >= unanimous_threshold]
        low_scores = [ds for ds in detector_scores if ds.score < 0.3]
        
        if len(high_scores) == len(detector_scores):
            # Wszystkie detektory wskazują alert
            decision = AlertDecision.ALERT
            final_score = sum(ds.score for ds in detector_scores) / len(detector_scores)
            consensus_strength = 1.0
            reasoning = f"Unanimous agreement: all {len(detector_scores)} detectors above {unanimous_threshold}"
        
        elif len(low_scores) == len(detector_scores):
            # Wszystkie detektory wskazują no alert
            decision = AlertDecision.NO_ALERT
            final_score = sum(ds.score for ds in detector_scores) / len(detector_scores)
            consensus_strength = 1.0
            reasoning = f"Unanimous agreement: all {len(detector_scores)} detectors below 0.3"
        
        else:
            # Brak jednomyślności - fallback do weighted average
            result = self._weighted_average_consensus(token, detector_scores)
            result.strategy_used = ConsensusStrategy.UNANIMOUS_AGREEMENT
            result.reasoning = f"No unanimous agreement, fallback to weighted average: {result.reasoning}"
            result.consensus_strength *= 0.5  # Reduce confidence due to disagreement
            return result
        
        avg_confidence = sum(ds.confidence for ds in detector_scores) / len(detector_scores)
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
        
        # Znajdź detektor z najwyższym score
        dominant_detector = max(detector_scores, key=lambda ds: ds.score * ds.confidence)
        
        # Sprawdź czy dominant detector ma wystarczającą przewagę
        other_scores = [ds.score for ds in detector_scores if ds.name != dominant_detector.name]
        avg_others = sum(other_scores) / len(other_scores) if other_scores else 0.0
        
        dominance_factor = dominant_detector.score - avg_others
        
        if dominance_factor >= 0.2 and dominant_detector.score >= self.consensus_thresholds['alert_threshold']:
            decision = AlertDecision.ALERT
            consensus_strength = min(1.0, dominance_factor / 0.5)
        elif dominant_detector.score >= self.consensus_thresholds['watch_threshold']:
            decision = AlertDecision.WATCH
            consensus_strength = min(1.0, dominance_factor / 0.3)
        else:
            decision = AlertDecision.NO_ALERT
            consensus_strength = 0.5
        
        reasoning = f"Dominant detector {dominant_detector.name}: {dominant_detector.score:.3f} (dominance: {dominance_factor:.3f})"
        contributing_detectors = [ds.name for ds in detector_scores]
        
        return ConsensusResult(
            decision=decision,
            final_score=dominant_detector.score,
            confidence=dominant_detector.confidence,
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
        
        # Dostosuj threshold na podstawie average confidence detektorów
        avg_confidence = sum(ds.confidence for ds in detector_scores) / len(detector_scores)
        
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
            'detector_scores': {ds.name: ds.score for ds in detector_scores},
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
            for detector, score in decision['detector_scores'].items():
                if detector not in detector_contribution:
                    detector_contribution[detector] = {'count': 0, 'avg_score': 0.0}
                detector_contribution[detector]['count'] += 1
                detector_contribution[detector]['avg_score'] += score
        
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


if __name__ == "__main__":
    test_consensus_engine()