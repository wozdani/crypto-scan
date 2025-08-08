#!/usr/bin/env python3
"""
ETAP 7: Decision Consensus Engine - Finalne centrum decyzyjne dla wszystkich detektorów
Integruje CaliforniumWhale AI, WhaleCLIP, Stealth Engine, DiamondWhale AI w unified decision system
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class DetectorResult:
    """Struktura wyniku pojedynczego detektora"""
    vote: str  # "BUY", "HOLD", "AVOID"
    score: float  # 0.0 - 1.0
    weight: float  # dynamiczna waga detektora
    confidence: float = 0.0  # poziom pewności
    metadata: Dict[str, Any] = None

@dataclass
class ConsensusResult:
    """Wynik finalnej decyzji konsensusu"""
    decision: str  # "BUY", "HOLD", "AVOID"
    final_score: float
    confidence: float
    contributing_detectors: List[str]
    weighted_scores: Dict[str, float]
    reasoning: str
    timestamp: str
    threshold_met: bool
    votes: List[str] = None  # Lista głosów detektorów: ["BUY", "HOLD", "AVOID", ...]

class DecisionConsensusEngine:
    """
    ETAP 7: Finalne centrum decyzyjne dla wszystkich AI detektorów
    Tworzy skonsolidowaną decyzję na podstawie ważonych głosów detektorów
    """
    
    def __init__(self, cache_dir: str = "crypto-scan/cache"):
        """
        Inicjalizacja Decision Consensus Engine
        Multi-agent system jest teraz głównym mechanizmem konsensusu
        
        Args:
            cache_dir: Katalog dla cache'owania wag i historii
        """
        self.cache_dir = cache_dir
        self.weights_file = os.path.join(cache_dir, "consensus_detector_weights.json")
        self.decisions_file = os.path.join(cache_dir, "consensus_decisions.json")
        
        # Default detector weights
        self.default_weights = {
            "CaliforniumWhale": 0.33,
            "WhaleCLIP": 0.26,
            "StealthEngine": 0.25,
            "DiamondWhale": 0.16
        }
        
        self.decision_history = []
        self._load_weights()
        self._load_decision_history()
        
        print("[CONSENSUS ENGINE] Finalne centrum decyzyjne initialized")
    
    def simulate_decision_consensus(self, detector_outputs: Dict[str, Dict[str, Any]], 
                                  threshold: float = 0.7, 
                                  token: str = "UNKNOWN",
                                  market_data: Dict[str, Any] = None) -> ConsensusResult:
        """
        ETAP 7: Główna funkcja konsensusu - finalna decyzja na podstawie ważonych głosów
        
        Args:
            detector_outputs: Dict z wynikami detektorów
                {
                    "CaliforniumWhale": {"vote": "BUY", "score": 0.91, "weight": 0.33},
                    "WhaleCLIP": {"vote": "BUY", "score": 0.72, "weight": 0.26},
                    "StealthEngine": {"vote": "HOLD", "score": 0.58, "weight": 0.41}
                }
            threshold: Minimalny próg dla akceptacji decyzji (default: 0.7)
            token: Symbol tokena dla logowania
            
        Returns:
            ConsensusResult z finalną decyzją i metadanami
        """
        # Always use multi-agent system as primary consensus mechanism
        try:
            if market_data:
                print(f"[MULTI-AGENT PRIMARY] Processing {token} with 5-agent system as primary consensus")
                
                # Use multi-agent evaluation directly
                multi_agent_decision = self._run_multi_agent_consensus(
                    detector_outputs, 
                    market_data, 
                    token,
                    threshold
                )
                
                if multi_agent_decision:
                    print(f"[MULTI-AGENT PRIMARY] Decision: {multi_agent_decision.decision}, Score: {multi_agent_decision.final_score:.3f}")
                    return multi_agent_decision
                else:
                    print(f"[MULTI-AGENT ERROR] Failed to get multi-agent decision, using fallback")
            else:
                print(f"[MULTI-AGENT WARNING] No market_data provided, using fallback weighted voting")
                
        except ImportError:
            print(f"[MULTI-AGENT INFO] Multi-agent module not available, using fallback weighted voting")
        except Exception as e:
            print(f"[MULTI-AGENT ERROR] System failed: {e}. Using fallback weighted voting.")
            # Log detailed error for debugging
            import traceback
            print(f"[MULTI-AGENT ERROR DETAILS] {traceback.format_exc()}")
                
        # Fallback to simple weighted voting only if multi-agent fails
        print(f"[FALLBACK VOTING] Processing {token} with weighted voting (multi-agent unavailable)")
        
        # Inicjalizacja ważonych scores dla każdej decyzji
        weighted_scores = {"BUY": 0.0, "HOLD": 0.0, "AVOID": 0.0}
        contributing_detectors = []
        total_weight = 0.0
        
        # Aplikuj dynamiczne wagi jeśli nie podano w input
        updated_outputs = self._apply_dynamic_weights(detector_outputs)
        
        # Oblicz ważone głosy dla każdego detektora - NOWA LOGIKA: tylko aktywne detektory
        active_count = 0
        for detector_name, result in updated_outputs.items():
            vote = result.get("vote", "HOLD")
            score = result.get("score", 0.0)
            weight = result.get("weight", 0.0)
            
            # 🛠️ NOWA LOGIKA: Pomiń detektory ze score = 0
            if score <= 0.0:
                print(f"[CONSENSUS SKIP] {detector_name}: Inactive detector (score=0.0), skipping")
                continue
            
            active_count += 1
            
            # Walidacja vote
            if vote not in weighted_scores:
                print(f"[CONSENSUS WARNING] Invalid vote '{vote}' from {detector_name}, defaulting to HOLD")
                vote = "HOLD"
            
            # Oblicz weighted contribution
            weighted_contribution = score * weight
            weighted_scores[vote] += weighted_contribution
            total_weight += weight
            contributing_detectors.append(detector_name)
            
            print(f"[CONSENSUS] {detector_name}: {vote} (score: {score:.3f}, weight: {weight:.3f}) → +{weighted_contribution:.3f}")
        
        # Debug weighted scores
        print(f"[CONSENSUS DEBUG] Weighted vote scores: {weighted_scores}")
        print(f"[CONSENSUS DEBUG] Total weight: {total_weight:.3f}")
        print(f"[CONSENSUS DEBUG] Active detectors: {active_count}")
        
        # 🛠️ SPECJALNY PRZYPADEK: Jeśli tylko 2 detektory aktywne i oba głosują BUY
        if active_count == 2 and weighted_scores["BUY"] > 0:
            # Sprawdź czy oba detektory głosują na BUY
            buy_votes = sum(1 for name, result in updated_outputs.items() 
                          if result.get("score", 0.0) > 0 and result.get("vote") == "BUY")
            if buy_votes == 2:
                print(f"[CONSENSUS SPECIAL] 2 active detectors, both voting BUY - applying special logic")
                # Boost BUY score dla 2-detector consensus
                if total_weight > 0:
                    buy_boost = 1.2  # 20% boost
                    weighted_scores["BUY"] *= buy_boost
        
        # Finalna decyzja - najwyższy ważony score
        if total_weight > 0:
            # Normalizuj scores przez total weight (tylko aktywnych detektorów)
            normalized_scores = {k: v / total_weight for k, v in weighted_scores.items()}
            decision = max(normalized_scores.items(), key=lambda x: x[1])[0]
            final_score = normalized_scores[decision]
        else:
            decision = "HOLD"
            final_score = 0.0
            normalized_scores = weighted_scores.copy()
        
        # Sprawdź czy przekracza threshold - ZMIENIONE: bardziej liberalne dla małej liczby detektorów
        adjusted_threshold = threshold
        if active_count <= 2:
            # Obniż próg dla małej liczby aktywnych detektorów
            adjusted_threshold = threshold * 0.8  # 20% niższy próg
            print(f"[CONSENSUS] Adjusting threshold for {active_count} detectors: {threshold} → {adjusted_threshold:.3f}")
        
        threshold_met = final_score >= adjusted_threshold
        
        # Jeśli nie przekracza threshold, domyślnie HOLD
        if not threshold_met:
            print(f"[CONSENSUS] Score {final_score:.3f} below threshold {adjusted_threshold:.3f}, defaulting to HOLD")
            if decision != "HOLD":
                decision = "HOLD"
        
        # Oblicz confidence na podstawie różnicy między top 2 decisions
        sorted_scores = sorted(normalized_scores.values(), reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        
        # Generuj reasoning
        reasoning = self._generate_reasoning(decision, final_score, threshold_met, 
                                           normalized_scores, contributing_detectors)
        
        # Zbierz głosy detektorów w formacie "DetectorName: VOTE"
        detector_votes = []
        for detector_name, result_data in updated_outputs.items():
            if result_data.get("score", 0.0) > 0:
                vote = result_data.get("vote", "HOLD")
                # Format głosu jako "DetectorName: VOTE" dla zgodności z systemem wyświetlania
                formatted_vote = f"{detector_name}: {vote}"
                detector_votes.append(formatted_vote)
        
        # Utwórz wynik
        result = ConsensusResult(
            decision=decision,
            final_score=final_score,
            confidence=confidence,
            contributing_detectors=contributing_detectors,
            weighted_scores=normalized_scores,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
            threshold_met=threshold_met,
            votes=detector_votes  # Dodaj listę głosów
        )
        
        # Zapisz decyzję do historii
        self._record_decision(token, result, updated_outputs)
        
        print(f"[FINAL DECISION] {token}: {decision} (score: {final_score:.3f}, confidence: {confidence:.3f})")
        
        return result
    
    def _apply_dynamic_weights(self, detector_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Aplikuje dynamiczne wagi do detector outputs jeśli nie są podane
        
        Args:
            detector_outputs: Raw detector outputs
            
        Returns:
            Updated detector outputs z dynamic weights
        """
        updated_outputs = {}
        
        for detector_name, result in detector_outputs.items():
            updated_result = result.copy()
            
            # Aplikuj dynamiczną wagę jeśli nie podano
            if "weight" not in updated_result or updated_result["weight"] == 0:
                dynamic_weight = self._get_detector_weight(detector_name)
                updated_result["weight"] = dynamic_weight
                print(f"[DYNAMIC WEIGHT] {detector_name}: Applied weight {dynamic_weight:.3f}")
            
            updated_outputs[detector_name] = updated_result
        
        return updated_outputs
    
    def _get_detector_weight(self, detector_name: str) -> float:
        """
        Pobiera aktualną wagę detektora (z cache lub default)
        
        Args:
            detector_name: Nazwa detektora
            
        Returns:
            Aktualna waga detektora
        """
        # Sprawdź czy mamy dynamiczną wagę w cache
        if hasattr(self, 'detector_weights') and detector_name in self.detector_weights:
            return self.detector_weights[detector_name]
        
        # Fallback do default weights
        return self.default_weights.get(detector_name, 0.2)
    
    def _generate_reasoning(self, decision: str, final_score: float, threshold_met: bool,
                          weighted_scores: Dict[str, float], detectors: List[str]) -> str:
        """
        Generuje human-readable reasoning dla decyzji
        
        Args:
            decision: Finalna decyzja
            final_score: Final score
            threshold_met: Czy threshold został osiągnięty
            weighted_scores: Ważone scores dla wszystkich opcji
            detectors: Lista contributing detectors
            
        Returns:
            String z reasoning
        """
        reasoning_parts = []
        
        # Główna decyzja
        if threshold_met:
            reasoning_parts.append(f"Strong consensus for {decision} (score: {final_score:.3f})")
        else:
            reasoning_parts.append(f"Weak signal below threshold, defaulting to {decision}")
        
        # Contributing detectors
        reasoning_parts.append(f"Based on {len(detectors)} detectors: {', '.join(detectors)}")
        
        # Score breakdown
        score_breakdown = ", ".join([f"{k}: {v:.3f}" for k, v in weighted_scores.items() if v > 0])
        reasoning_parts.append(f"Weighted scores: {score_breakdown}")
        
        return " | ".join(reasoning_parts)
    
    def _record_decision(self, token: str, result: ConsensusResult, detector_outputs: Dict[str, Dict]):
        """
        Zapisuje decyzję do historii dla feedback loop
        
        Args:
            token: Symbol tokena
            result: ConsensusResult
            detector_outputs: Raw detector outputs
        """
        decision_record = {
            "token": token,
            "timestamp": result.timestamp,
            "decision": result.decision,
            "final_score": result.final_score,
            "confidence": result.confidence,
            "threshold_met": result.threshold_met,
            "contributing_detectors": result.contributing_detectors,
            "weighted_scores": result.weighted_scores,
            "reasoning": result.reasoning,
            "detector_outputs": detector_outputs
        }
        
        self.decision_history.append(decision_record)
        
        # Zapisz do pliku
        self._save_decision_history()
        
        print(f"[CONSENSUS RECORD] Decision recorded for {token}: {result.decision}")
    
    def update_detector_weights_from_feedback(self, days: int = 7) -> Dict[str, float]:
        """
        Aktualizuje wagi detektorów na podstawie feedback z ostatnich dni
        Podobnie jak w Consensus Decision Engine (ETAP 6)
        
        Args:
            days: Liczba dni do analizy
            
        Returns:
            Zaktualizowane wagi detektorów
        """
        print(f"[WEIGHT UPDATE] Analyzing feedback from last {days} days")
        
        # Analizuj skuteczność detektorów
        detector_performance = self._analyze_detector_performance(days)
        
        # Oblicz nowe wagi
        updated_weights = {}
        for detector_name in self.default_weights.keys():
            if detector_name in detector_performance:
                perf = detector_performance[detector_name]
                success_rate = perf['correct'] / max(1, perf['total'])
                avg_confidence = perf['avg_confidence']
                prev_weight = perf['prev_weight']
                
                # Formula z ETAP 6: decay * prev + (1-decay) * (success * confidence)
                decay = 0.95
                new_weight = decay * prev_weight + (1 - decay) * (success_rate * avg_confidence)
                
                # Bounds [0.10, 0.50]
                new_weight = max(0.10, min(0.50, new_weight))
                updated_weights[detector_name] = new_weight
                
                print(f"[WEIGHT UPDATE] {detector_name}: {prev_weight:.3f} → {new_weight:.3f} (success: {success_rate:.2%})")
            else:
                # Preserve default weight if no data
                updated_weights[detector_name] = self.default_weights[detector_name]
        
        # Zapisz nowe wagi
        self.detector_weights = updated_weights
        self._save_weights()
        
        return updated_weights
    
    def _analyze_detector_performance(self, days: int) -> Dict[str, Dict]:
        """
        Analizuje performance detektorów z decision history
        
        Args:
            days: Liczba dni do analizy
            
        Returns:
            Dict z performance metrics dla każdego detektora
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_decisions = [
            d for d in self.decision_history 
            if datetime.fromisoformat(d['timestamp']) > cutoff_date
        ]
        
        print(f"[PERFORMANCE] Analyzing {len(recent_decisions)} recent decisions")
        
        detector_stats = {}
        for decision in recent_decisions:
            for detector in decision['contributing_detectors']:
                if detector not in detector_stats:
                    detector_stats[detector] = {
                        'correct': 0,
                        'total': 0,
                        'confidence_sum': 0.0,
                        'prev_weight': self._get_detector_weight(detector)
                    }
                
                # Assume decision was "correct" if threshold was met
                # W production można to zastąpić rzeczywistym feedback
                if decision['threshold_met']:
                    detector_stats[detector]['correct'] += 1
                
                detector_stats[detector]['total'] += 1
                detector_stats[detector]['confidence_sum'] += decision['confidence']
        
        # Oblicz average confidence
        for detector, stats in detector_stats.items():
            if stats['total'] > 0:
                stats['avg_confidence'] = stats['confidence_sum'] / stats['total']
            else:
                stats['avg_confidence'] = 0.7  # default
        
        return detector_stats
    
    async def _batch_evaluate_all_detectors(
        self,
        active_detectors: Dict[str, Dict[str, Any]],
        market_data: Dict[str, Any], 
        threshold: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        OPTIMIZED: Batch evaluate all detectors with single OpenAI call
        Eliminuje problem 429 rate limiting przez zmniejszenie zapytań z N*5 do 1
        
        Args:
            active_detectors: Dict of detector data {detector_name: {score, weight, etc}}
            market_data: Market data for context
            threshold: Decision threshold
            
        Returns:
            Dict with evaluation results for each detector
        """
        from .multi_agent_decision import MultiAgentDecisionSystem, AgentRole
        from datetime import datetime
        
        print(f"[BATCH EVALUATION] Starting batch evaluation for {len(active_detectors)} detectors")
        
        # Initialize multi-agent system
        multi_agent = MultiAgentDecisionSystem()
        
        # Prepare contexts for ALL detectors and ALL agents (5 per detector)
        all_contexts = []
        detector_mapping = {}  # Track which contexts belong to which detector
        
        context_idx = 0
        for detector_name, detector_data in active_detectors.items():
            score = detector_data.get("score", 0.0)
            
            shared_context = {
                'detector_name': detector_name,
                'score': score,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat(),
                'market_data': market_data,
                'signal_data': {}  # Could be enhanced later
            }
            
            # Add contexts for all 5 agents for this detector
            for role in [AgentRole.ANALYZER, AgentRole.REASONER, 
                        AgentRole.VOTER, AgentRole.DEBATER, AgentRole.DECIDER]:
                all_contexts.append((role, shared_context))
                # Track which detector this context belongs to
                if detector_name not in detector_mapping:
                    detector_mapping[detector_name] = []
                detector_mapping[detector_name].append(context_idx)
                context_idx += 1
        
        print(f"[BATCH EVALUATION] Prepared {len(all_contexts)} agent contexts for batch processing")
        
        try:
            # SINGLE OpenAI API call for ALL agents of ALL detectors
            all_responses = await multi_agent.batch_llm_reasoning(all_contexts)
            
            print(f"[BATCH EVALUATION] Expected {len(all_contexts)} responses, got {len(all_responses)}")
            
            # Validate response count matches expected
            if len(all_responses) != len(all_contexts):
                print(f"[BATCH EVALUATION WARNING] Response count mismatch - expected {len(all_contexts)}, got {len(all_responses)}")
            
            # Group responses back by detector
            results = {}
            for detector_name, context_indices in detector_mapping.items():
                # Extract the 5 agent responses for this detector with bounds checking
                detector_responses = []
                for i in context_indices:
                    if i < len(all_responses):
                        detector_responses.append(all_responses[i])
                    else:
                        print(f"[BATCH EVALUATION ERROR] Missing response for index {i} (detector: {detector_name})")
                        # Create fallback response
                        from multi_agent.multi_agent_decision import AgentResponse
                        fallback_response = AgentResponse(
                            decision="NO",
                            confidence=0.1,
                            reasoning="Fallback due to missing batch response"
                        )
                        detector_responses.append(fallback_response)
                
                if len(detector_responses) == 0:
                    print(f"[BATCH EVALUATION ERROR] No responses available for detector {detector_name}")
                    continue
                
                # Process responses for this detector (same logic as original)
                yes_votes = sum(1 for r in detector_responses if r.decision == "YES")
                no_votes = sum(1 for r in detector_responses if r.decision == "NO") 
                total_votes = len(detector_responses)
                
                # Decision based on majority vote
                decision = "YES" if yes_votes > no_votes else "NO"
                confidence = sum(r.confidence for r in detector_responses) / total_votes
                
                # Generate log summary
                log_summary = f"[MULTI-AGENT BATCH] {detector_name}: {yes_votes} YES / {no_votes} NO votes"
                for response in detector_responses:
                    log_summary += f"\n  {response.role.value}: {response.decision} ({response.confidence:.3f}) - {response.reasoning[:100]}..."
                
                results[detector_name] = {
                    "decision": decision,
                    "confidence": confidence,
                    "log": log_summary
                }
                
                print(f"[BATCH RESULT] {detector_name}: {decision} (confidence: {confidence:.3f}, votes: {yes_votes}/{total_votes})")
            
            print(f"[BATCH EVALUATION] Successfully processed {len(results)} detectors with single OpenAI call")
            return results
            
        except Exception as e:
            print(f"[BATCH EVALUATION ERROR] Batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            # Return empty results on failure
            return {}
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """
        Pobiera statystyki consensus decision engine
        
        Returns:
            Dict ze statystykami
        """
        total_decisions = len(self.decision_history)
        if total_decisions == 0:
            return {
                "total_decisions": 0,
                "decision_breakdown": {},
                "average_confidence": 0.0,
                "threshold_met_rate": 0.0,
                "detector_participation": {}
            }
        
        # Decision breakdown
        decision_breakdown = {}
        threshold_met_count = 0
        confidence_sum = 0.0
        
        for decision in self.decision_history:
            decision_type = decision['decision']
            decision_breakdown[decision_type] = decision_breakdown.get(decision_type, 0) + 1
            
            if decision['threshold_met']:
                threshold_met_count += 1
            
            confidence_sum += decision['confidence']
        
        # Detector participation
        detector_participation = {}
        for decision in self.decision_history:
            for detector in decision['contributing_detectors']:
                detector_participation[detector] = detector_participation.get(detector, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "decision_breakdown": decision_breakdown,
            "average_confidence": confidence_sum / total_decisions,
            "threshold_met_rate": threshold_met_count / total_decisions,
            "detector_participation": detector_participation,
            "current_weights": getattr(self, 'detector_weights', self.default_weights)
        }
    
    def _load_weights(self):
        """Ładuje wagi detektorów z cache"""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    self.detector_weights = json.load(f)
                print(f"[WEIGHTS] Loaded {len(self.detector_weights)} detector weights")
            else:
                self.detector_weights = self.default_weights.copy()
                print("[WEIGHTS] Using default detector weights")
        except Exception as e:
            print(f"[WEIGHTS ERROR] Failed to load weights: {e}")
            self.detector_weights = self.default_weights.copy()
    
    def _save_weights(self):
        """Zapisuje wagi detektorów do cache"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.weights_file, 'w') as f:
                json.dump(self.detector_weights, f, indent=2)
            print(f"[WEIGHTS] Saved detector weights to {self.weights_file}")
        except Exception as e:
            print(f"[WEIGHTS ERROR] Failed to save weights: {e}")
    
    def _load_decision_history(self):
        """Ładuje historię decyzji z cache"""
        try:
            if os.path.exists(self.decisions_file):
                with open(self.decisions_file, 'r') as f:
                    self.decision_history = json.load(f)
                print(f"[HISTORY] Loaded {len(self.decision_history)} historical decisions")
            else:
                self.decision_history = []
                print("[HISTORY] No decision history found, starting fresh")
        except Exception as e:
            print(f"[HISTORY ERROR] Failed to load decision history: {e}")
            self.decision_history = []
    
    def _save_decision_history(self):
        """Zapisuje historię decyzji do cache"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Keep only last 1000 decisions to prevent file bloat
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-1000:]
            
            with open(self.decisions_file, 'w') as f:
                json.dump(self.decision_history, f, indent=2)
            print(f"[HISTORY] Saved {len(self.decision_history)} decisions to {self.decisions_file}")
        except Exception as e:
            print(f"[HISTORY ERROR] Failed to save decision history: {e}")
    
    def _run_multi_agent_consensus(
        self, 
        detector_outputs: Dict[str, Dict[str, Any]], 
        market_data: Dict[str, Any], 
        token: str,
        threshold: float
    ) -> Optional[ConsensusResult]:
        """
        Run multi-agent consensus as primary decision mechanism
        Each detector gets evaluated by 5 agents (Analyzer, Reasoner, Voter, Debater, Decider)
        
        Returns:
            ConsensusResult with multi-agent decision
        """
        try:
            from .multi_agent_decision import evaluate_detector_with_agents
            import asyncio
            
            print(f"\n{'#'*80}")
            print(f"[MULTI-AGENT CONSENSUS] Starting evaluation for token: {token}")
            print(f"[MULTI-AGENT CONSENSUS] Evaluating {len(detector_outputs)} detectors with 5-agent system")
            print(f"[MULTI-AGENT CONSENSUS] Market data available: {'YES' if market_data else 'NO'}")
            if market_data:
                volume = market_data.get('volume_24h', 0)
                price_change = market_data.get('price_change_24h', 0)
                print(f"[MULTI-AGENT DEBUG] Market data contents: Volume=${volume:,.0f}, Price change={price_change:.2f}%")
            print(f"{'#'*80}\n")
            
            # Collect all agent evaluations
            agent_decisions = {}
            agent_confidences = {}
            all_logs = []
            
            # OPTIMIZED: Batch multi-agent evaluation for ALL detectors at once
            print(f"[MULTI-AGENT BATCH] Starting optimized batch evaluation for {len(detector_outputs)} detectors...")
            
            # Filter only active detectors (score > 0)
            active_detectors = {name: data for name, data in detector_outputs.items() if data.get("score", 0) > 0}
            
            if not active_detectors:
                print(f"[MULTI-AGENT SKIP] No active detectors with score > 0")
                return None
                
            print(f"[MULTI-AGENT BATCH] Processing {len(active_detectors)} active detectors: {list(active_detectors.keys())}")
            
            # Run BATCH evaluation for all detectors at once
            try:
                # Check if event loop is already running  
                try:
                    loop = asyncio.get_running_loop()
                    # If loop is running, create task in existing loop
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._batch_evaluate_all_detectors(
                                active_detectors, 
                                market_data or {}, 
                                float(threshold)
                            )
                        )
                        batch_results = future.result(timeout=60)  # Increased timeout for batch
                except RuntimeError:
                    # No loop running, safe to create new one
                    batch_results = asyncio.run(
                        self._batch_evaluate_all_detectors(
                            active_detectors, 
                            market_data or {}, 
                            float(threshold)
                        )
                    )
                
                # Parse batch results
                for detector_name, result in batch_results.items():
                    agent_decisions[detector_name] = result["decision"]
                    agent_confidences[detector_name] = result["confidence"]
                    all_logs.append(result["log"])
                    print(f"[MULTI-AGENT BATCH RESULT] {detector_name}: Decision={result['decision']}, Confidence={result['confidence']:.3f}")
                
            except Exception as e:
                print(f"[MULTI-AGENT BATCH ERROR] Failed batch evaluation: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to individual evaluations if batch fails
                return None
            
            # Aggregate multi-agent decisions into final consensus
            if agent_decisions:
                print(f"\n[CONSENSUS AGGREGATION] Combining detector decisions...")
                
                # Count YES/NO votes
                yes_votes = sum(1 for d in agent_decisions.values() if d == "YES")
                no_votes = sum(1 for d in agent_decisions.values() if d == "NO")
                total_votes = len(agent_decisions)
                
                print(f"[CONSENSUS VOTES] Detector votes: {yes_votes} YES / {no_votes} NO (total: {total_votes})")
                
                # Calculate average confidence
                avg_confidence = sum(agent_confidences.values()) / len(agent_confidences)
                
                # Determine final decision
                if yes_votes > no_votes:
                    final_decision = "BUY"
                    decision_strength = yes_votes / total_votes
                    print(f"[CONSENSUS DECISION] Majority YES → BUY decision!")
                elif no_votes > yes_votes:
                    final_decision = "AVOID"
                    decision_strength = no_votes / total_votes
                    print(f"[CONSENSUS DECISION] Majority NO → AVOID decision")
                else:
                    final_decision = "HOLD"
                    decision_strength = 0.5
                    print(f"[CONSENSUS DECISION] Tie vote → HOLD decision")
                
                print(f"[CONSENSUS STRENGTH] Decision strength: {decision_strength:.3f}")
                print(f"[CONSENSUS CONFIDENCE] Average confidence: {avg_confidence:.3f}")
                
                # Create reasoning
                reasoning = f"5-Agent Multi-Agent Consensus: {yes_votes} YES, {no_votes} NO votes. "
                reasoning += f"Average confidence: {avg_confidence:.3f}. "
                reasoning += f"Detectors evaluated: {', '.join(agent_decisions.keys())}. "
                reasoning += f"Primary decision mechanism using 5 agents per detector."
                
                # Create proper votes list with detector names and their agent decisions
                votes_list = []
                for detector_name, agent_decision in agent_decisions.items():
                    # Format: "DetectorName: DECISION" 
                    vote_text = f"{detector_name}: {'BUY' if agent_decision == 'YES' else 'AVOID'}"
                    votes_list.append(vote_text)
                
                # Create ConsensusResult
                result = ConsensusResult(
                    decision=final_decision,
                    final_score=decision_strength,
                    confidence=avg_confidence,
                    contributing_detectors=list(agent_decisions.keys()),
                    weighted_scores=detector_outputs,
                    reasoning=reasoning,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    threshold_met=decision_strength >= threshold,
                    votes=votes_list  # Now contains actual detector votes like ["StealthEngine: BUY", "DiamondWhale: AVOID"]
                )
                
                # Save decision
                self._save_decision_history()
                
                # Log decision
                print(f"\n{'#'*80}")
                print(f"[FINAL CONSENSUS DECISION] {token}")
                print(f"  📊 Decision: {final_decision}")
                print(f"  💪 Strength: {decision_strength:.3f}")
                print(f"  🎯 Confidence: {avg_confidence:.3f}")
                print(f"  🗳️ Votes: {yes_votes} YES / {no_votes} NO from {total_votes} detectors")
                print(f"  🤖 Each detector evaluated by 5 AI agents")
                if final_decision == "BUY":
                    print(f"  ✅ ALERT WILL BE SENT - Consensus reached for BUY!")
                else:
                    print(f"  ❌ NO ALERT - Consensus decision: {final_decision}")
                print(f"{'#'*80}\n")
                
                return result
            else:
                print(f"[MULTI-AGENT WARNING] No valid agent decisions generated")
                return None
                
        except Exception as e:
            print(f"[MULTI-AGENT ERROR] Failed to run multi-agent consensus: {e}")
            return None

def create_decision_consensus_engine() -> DecisionConsensusEngine:
    """Factory function dla Decision Consensus Engine
    Multi-agent system jest teraz głównym mechanizmem konsensusu
    """
    return DecisionConsensusEngine()

def test_decision_consensus():
    """Test funkcja dla Decision Consensus Engine"""
    print("🧪 Testing Decision Consensus Engine - ETAP 7...")
    
    engine = create_decision_consensus_engine()
    
    # Test case 1: Strong BUY consensus
    print("\n🔬 Test 1: Strong BUY Consensus")
    detector_outputs_1 = {
        "CaliforniumWhale": {"vote": "BUY", "score": 0.91, "weight": 0.33},
        "WhaleCLIP": {"vote": "BUY", "score": 0.82, "weight": 0.26},
        "StealthEngine": {"vote": "BUY", "score": 0.78, "weight": 0.25},
        "DiamondWhale": {"vote": "HOLD", "score": 0.65, "weight": 0.16}
    }
    
    result1 = engine.simulate_decision_consensus(detector_outputs_1, threshold=0.7, token="ETHUSDT")
    print(f"Result: {result1.decision} (score: {result1.final_score:.3f}, confidence: {result1.confidence:.3f})")
    
    # Test case 2: Mixed votes - HOLD default
    print("\n🔬 Test 2: Mixed Votes - Below Threshold")
    detector_outputs_2 = {
        "CaliforniumWhale": {"vote": "BUY", "score": 0.65},
        "WhaleCLIP": {"vote": "AVOID", "score": 0.68},
        "StealthEngine": {"vote": "HOLD", "score": 0.45}
    }
    
    result2 = engine.simulate_decision_consensus(detector_outputs_2, threshold=0.7, token="BTCUSDT")
    print(f"Result: {result2.decision} (score: {result2.final_score:.3f}, confidence: {result2.confidence:.3f})")
    
    # Test case 3: Strong AVOID consensus
    print("\n🔬 Test 3: Strong AVOID Consensus")
    detector_outputs_3 = {
        "CaliforniumWhale": {"vote": "AVOID", "score": 0.88, "weight": 0.35},
        "WhaleCLIP": {"vote": "AVOID", "score": 0.92, "weight": 0.30},
        "DiamondWhale": {"vote": "AVOID", "score": 0.85, "weight": 0.20}
    }
    
    result3 = engine.simulate_decision_consensus(detector_outputs_3, threshold=0.7, token="ADAUSDT")
    print(f"Result: {result3.decision} (score: {result3.final_score:.3f}, confidence: {result3.confidence:.3f})")
    
    # Test weight update
    print("\n🔬 Test 4: Weight Update from Feedback")
    updated_weights = engine.update_detector_weights_from_feedback(days=7)
    print(f"Updated weights: {updated_weights}")
    
    # Test statistics
    print("\n🔬 Test 5: Consensus Statistics")
    stats = engine.get_consensus_statistics()
    print(f"Total decisions: {stats['total_decisions']}")
    print(f"Decision breakdown: {stats['decision_breakdown']}")
    print(f"Threshold met rate: {stats['threshold_met_rate']:.1%}")
    
    print("\n✅ ETAP 7 Decision Consensus Engine testing complete!")

if __name__ == "__main__":
    test_decision_consensus()