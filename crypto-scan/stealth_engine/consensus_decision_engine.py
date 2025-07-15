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
    
    def run(self, token: str, scores: Union[Dict[str, float], Dict[str, Dict[str, float]]], 
            strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_AVERAGE,
            metadata: Dict[str, Any] = None, 
            use_simple_consensus: bool = False,
            use_dynamic_boosting: bool = False) -> ConsensusResult:
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
    
    def _dynamic_boosting_logic(self, token: str, scores: Union[Dict[str, float], Dict[str, Dict]]) -> ConsensusResult:
        """
        ETAP 3: Dynamic Boosting Decision Logic
        Implementuje confidence-based weighted scoring z booster strategies:
        - Aktywne detektory: score >= 0.75 AND confidence >= 0.60
        - Global score: suma (score × weight) dla aktywnych detektorów
        - Booster: confidence > 0.85 AND score > 0.90 → score × 1.1
        
        Args:
            token: Symbol tokena
            scores: Dict w extended format z confidence i weight
            
        Returns:
            ConsensusResult z dynamic boosting decision
        """
        print(f"[DYNAMIC BOOST] Processing {token} using Etap 3 logic")
        
        # Konwersja do unified format jeśli potrzebne
        unified_scores = self._normalize_to_extended_format(scores)
        
        # Thresholds dla Etap 3
        score_threshold = 0.75
        confidence_threshold = 0.60
        global_score_threshold = 0.8
        min_detectors = 2
        
        # Booster thresholds
        booster_confidence_threshold = 0.85
        booster_score_threshold = 0.90
        booster_multiplier = 1.1
        
        print(f"[DYNAMIC BOOST] Unified scores: {unified_scores}")
        
        # Znajdź aktywne detektory (score >= 0.75 AND confidence >= 0.60)
        active_detectors = {}
        boosted_detectors = []
        
        for detector_name, detector_data in unified_scores.items():
            score = detector_data['score']
            confidence = detector_data['confidence']
            weight = detector_data['weight']
            
            # Sprawdź warunki aktywności
            if score >= score_threshold and confidence >= confidence_threshold:
                # Zastosuj booster jeśli spełnia warunki
                if confidence > booster_confidence_threshold and score > booster_score_threshold:
                    original_score = score
                    score = score * booster_multiplier
                    boosted_detectors.append(detector_name)
                    print(f"[BOOSTER] {detector_name}: {original_score:.3f} → {score:.3f} (×{booster_multiplier})")
                
                active_detectors[detector_name] = {
                    'score': score,
                    'confidence': confidence,
                    'weight': weight,
                    'weighted_contribution': score * weight
                }
        
        print(f"[DYNAMIC BOOST] Active detectors: {len(active_detectors)}")
        print(f"[DYNAMIC BOOST] Boosted detectors: {boosted_detectors}")
        
        # Oblicz weighted global_score
        if active_detectors:
            global_score = sum(d['weighted_contribution'] for d in active_detectors.values())
            total_weight = sum(d['weight'] for d in active_detectors.values())
        else:
            global_score = 0.0
            total_weight = 0.0
        
        print(f"[DYNAMIC BOOST] Weighted global score: {global_score:.3f} (total weight: {total_weight:.3f})")
        
        # Logika decyzyjna Etap 3:
        # min. 2 detektory aktywne AND global_score > 0.8
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
        
        # Oblicz confidence na podstawie weighted average confidence
        if active_detectors:
            weighted_confidence = sum(d['confidence'] * d['weight'] for d in active_detectors.values()) / total_weight
            confidence = min(1.0, weighted_confidence * (len(active_detectors) / len(unified_scores)))
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
    
    def _send_dynamic_telegram_alert(self, token: str, global_score: float, 
                                    active_detectors: Dict, boosted_detectors: List[str]) -> bool:
        """
        ETAP 3: Dynamic Boosting Telegram Alert
        Wysyła enhanced alert z confidence, weight i booster info
        
        Args:
            token: Symbol tokena
            global_score: Weighted global score
            active_detectors: Dict aktywnych detektorów z metadata
            boosted_detectors: Lista detektorów z applied booster
            
        Returns:
            True jeśli alert wysłany pomyślnie
        """
        try:
            # Format enhanced message zgodnie z Etap 3 specyfikacją
            message = f"🚀 [Dynamic Consensus] {token} | Score: {global_score:.3f}\n"
            message += f"Active detectors: {len(active_detectors)}\n"
            
            if boosted_detectors:
                message += f"⚡ Boosted: {', '.join(boosted_detectors)}\n"
            
            message += "\n📊 Detector Breakdown:\n"
            
            # Dodaj detailed breakdown detektorów
            for detector, data in active_detectors.items():
                confidence_emoji = "🔥" if data['confidence'] >= 0.85 else "⚡" if data['confidence'] >= 0.70 else "💫"
                booster_emoji = "⚡" if detector in boosted_detectors else ""
                
                message += f"  {confidence_emoji}{booster_emoji} {detector}:\n"
                message += f"    Score: {data['score']:.3f} | Conf: {data['confidence']:.2f} | Weight: {data['weight']:.2f}\n"
                message += f"    Contribution: {data['weighted_contribution']:.3f}\n"
            
            total_weight = sum(d['weight'] for d in active_detectors.values())
            message += f"\n📈 Total Weight: {total_weight:.3f}"
            message += f"\nTimestamp: {datetime.now().strftime('%H:%M:%S')}"
            
            print(f"[DYNAMIC TELEGRAM] Enhanced alert message prepared:")
            print(message)
            
            # TODO: Implementacja rzeczywistego wysłania (Etap 4)
            if self.telegram_token and self.chat_id:
                print(f"[DYNAMIC TELEGRAM] Would send to chat {self.chat_id}")
                # W Etap 4: requests.get(url, params=params)
            else:
                print(f"[DYNAMIC TELEGRAM] No Telegram credentials configured")
            
            return True
            
        except Exception as e:
            print(f"[DYNAMIC TELEGRAM ERROR] Failed to send alert: {e}")
            return False
    
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


if __name__ == "__main__":
    test_consensus_engine()