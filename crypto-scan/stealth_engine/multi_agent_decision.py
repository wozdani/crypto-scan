"""
Multi-Agent Decision System dla każdego detektora
Koncepcja: 5 agentów (Analyzer, Reasoner, Voter, Debater, Decider) dla każdego detektora
debatuje i głosuje nad decyzją alertu używając LLM reasoning
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random  # Tymczasowo dla symulacji LLM
import os


class AgentRole(Enum):
    """Role agentów w systemie decyzyjnym"""
    ANALYZER = "Analyzer"
    REASONER = "Reasoner"
    VOTER = "Voter"
    DEBATER = "Debater"
    DECIDER = "Decider"


@dataclass
class AgentResponse:
    """Odpowiedź pojedynczego agenta"""
    role: AgentRole
    decision: str  # YES/NO
    reasoning: str
    confidence: float


class MultiAgentDecisionSystem:
    """
    System 5-agentowy dla każdego detektora
    Każdy agent ma swoją rolę w procesie decyzyjnym
    """
    
    def __init__(self):
        self.decision_log_file = "cache/multi_agent_decisions.json"
        self.debate_history = []
        
    async def llm_reasoning(self, role: AgentRole, context: Dict[str, Any]) -> AgentResponse:
        """
        Symulacja LLM reasoning - w produkcji zastąp prawdziwym API (np. Grok/OpenAI)
        
        Args:
            role: Rola agenta
            context: Kontekst decyzji (detector data, market data, etc.)
            
        Returns:
            AgentResponse z decyzją i uzasadnieniem
        """
        detector_name = context.get('detector_name', 'unknown')
        score = context.get('score', 0.0)
        threshold = context.get('threshold', 0.7)
        market_data = context.get('market_data', {})
        signal_data = context.get('signal_data', {})
        
        # Tymczasowa logika symulująca LLM reasoning
        # W produkcji zastąp prawdziwym API call
        
        if role == AgentRole.ANALYZER:
            # Analyzer sprawdza wiarygodność danych
            if score > 0.8:
                reasoning = f"Score {score:.3f} is highly reliable. Signal data shows strong patterns: {len(signal_data)} active signals."
                decision = "YES"
                confidence = 0.9
            elif score > threshold:
                reasoning = f"Score {score:.3f} above threshold but needs context verification. Data quality: moderate."
                decision = "YES"
                confidence = 0.7
            else:
                reasoning = f"Score {score:.3f} below threshold. Data reliability questionable."
                decision = "NO"
                confidence = 0.8
                
        elif role == AgentRole.REASONER:
            # Reasoner ocenia kontekst rynkowy
            volume_24h = market_data.get('volume_24h', 0)
            price_change = market_data.get('price_change_24h', 0)
            
            if volume_24h > 1000000 and price_change > 0:
                reasoning = f"Market context bullish: volume ${volume_24h:,.0f}, price change {price_change:.2f}%"
                decision = "YES"
                confidence = 0.85
            elif volume_24h < 500000:
                reasoning = f"Low volume ${volume_24h:,.0f} suggests caution despite score"
                decision = "NO"
                confidence = 0.75
            else:
                reasoning = f"Mixed market signals: volume moderate, price action unclear"
                decision = "YES" if score > threshold else "NO"
                confidence = 0.6
                
        elif role == AgentRole.VOTER:
            # Voter głosuje na podstawie overall assessment
            active_signals = len([s for s in signal_data.values() if s])
            if active_signals >= 3 and score > threshold:
                reasoning = f"Strong vote YES: {active_signals} active signals, score {score:.3f}"
                decision = "YES"
                confidence = 0.95
            elif active_signals >= 2:
                reasoning = f"Moderate support: {active_signals} signals active"
                decision = "YES"
                confidence = 0.7
            else:
                reasoning = f"Weak signal presence: only {active_signals} active"
                decision = "NO"
                confidence = 0.8
                
        elif role == AgentRole.DEBATER:
            # Debater kwestionuje poprzednie decyzje
            if len(self.debate_history) > 0:
                prev_decisions = [h['decision'] for h in self.debate_history[-3:]]
                if prev_decisions.count("YES") > prev_decisions.count("NO"):
                    reasoning = "Previous agents lean YES, but historical false positives suggest caution"
                    decision = "NO"
                    confidence = 0.65
                else:
                    reasoning = "Previous skepticism may be overly conservative given current signals"
                    decision = "YES"
                    confidence = 0.7
            else:
                reasoning = "Initial assessment: signals warrant investigation"
                decision = "YES" if score > threshold * 0.9 else "NO"
                confidence = 0.6
                
        elif role == AgentRole.DECIDER:
            # Decider podejmuje finalną decyzję
            yes_votes = sum(1 for h in self.debate_history if h['decision'] == "YES")
            total_votes = len(self.debate_history)
            
            if yes_votes >= 3:
                reasoning = f"Final decision: YES ({yes_votes}/{total_votes} agents agree)"
                decision = "YES"
                confidence = 0.9
            elif yes_votes >= 2 and score > threshold * 0.95:
                reasoning = f"Borderline case resolved: YES (score {score:.3f} tips balance)"
                decision = "YES"
                confidence = 0.75
            else:
                reasoning = f"Insufficient consensus: NO ({yes_votes}/{total_votes} YES votes)"
                decision = "NO"
                confidence = 0.85
        
        # Dodaj losowy element dla większej różnorodności (symulacja LLM variability)
        confidence *= (0.9 + random.random() * 0.2)
        
        return AgentResponse(
            role=role,
            decision=decision,
            reasoning=reasoning,
            confidence=min(confidence, 1.0)
        )
    
    async def agent_task(self, role: AgentRole, context: Dict[str, Any]) -> AgentResponse:
        """
        Zadanie pojedynczego agenta - async dla parallel processing
        
        Args:
            role: Rola agenta
            context: Kontekst decyzji
            
        Returns:
            AgentResponse
        """
        # Dodaj small delay dla symulacji LLM latency
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        response = await self.llm_reasoning(role, context)
        
        # Zapisz do historii debaty
        self.debate_history.append({
            'role': role.value,
            'decision': response.decision,
            'reasoning': response.reasoning,
            'confidence': response.confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    async def multi_agent_decision(
        self, 
        detector_name: str, 
        score: float, 
        signal_data: Dict[str, Any],
        market_data: Dict[str, Any],
        threshold: float = 0.7
    ) -> Tuple[str, float, str]:
        """
        Główna funkcja uruchamiająca multi-agent decision process
        
        Args:
            detector_name: Nazwa detektora (np. 'whale_ping')
            score: Score z detektora
            signal_data: Dane sygnałów
            market_data: Dane rynkowe
            threshold: Próg decyzyjny
            
        Returns:
            Tuple (decision, confidence, detailed_log)
        """
        print(f"\n[MULTI-AGENT] Starting 5-agent decision for {detector_name} (score: {score:.3f})")
        
        # Reset historii debaty dla nowej decyzji
        self.debate_history = []
        
        # Przygotuj kontekst dla agentów
        context = {
            'detector_name': detector_name,
            'score': score,
            'threshold': threshold,
            'signal_data': signal_data,
            'market_data': market_data
        }
        
        # Uruchom agentów równolegle
        agents = [
            AgentRole.ANALYZER,
            AgentRole.REASONER,
            AgentRole.VOTER,
            AgentRole.DEBATER,
            AgentRole.DECIDER
        ]
        
        # Pierwsze 4 agentów mogą działać równolegle
        parallel_tasks = [
            self.agent_task(role, context) 
            for role in agents[:4]
        ]
        
        # Czekaj na wyniki pierwszych 4 agentów
        parallel_results = await asyncio.gather(*parallel_tasks)
        
        # Decider potrzebuje wyników pozostałych, więc uruchom go osobno
        decider_response = await self.agent_task(AgentRole.DECIDER, context)
        
        # Zbierz wszystkie odpowiedzi
        all_responses = parallel_results + [decider_response]
        
        # Policz głosy
        yes_votes = sum(1 for r in all_responses if r.decision == "YES")
        total_votes = len(all_responses)
        
        # Oblicz średnią confidence
        avg_confidence = sum(r.confidence for r in all_responses) / total_votes
        
        # Finalna decyzja - majority vote
        final_decision = "YES" if yes_votes >= 3 else "NO"
        
        # Jeśli decyzja YES mimo niskiego score - to jest override!
        is_override = final_decision == "YES" and score < threshold
        
        # Stwórz detailed log
        detailed_log = self._create_detailed_log(
            detector_name, score, threshold, all_responses, 
            final_decision, avg_confidence, is_override
        )
        
        # Zapisz decyzję do pliku
        self._save_decision_log(
            detector_name, score, final_decision, 
            avg_confidence, all_responses, is_override
        )
        
        print(f"[MULTI-AGENT] Decision: {final_decision} (confidence: {avg_confidence:.3f}, votes: {yes_votes}/{total_votes})")
        if is_override:
            print(f"[MULTI-AGENT] ⚡ OVERRIDE ALERT! Agents voted YES despite low score {score:.3f}")
        
        return final_decision, avg_confidence, detailed_log
    
    def _create_detailed_log(
        self, 
        detector_name: str, 
        score: float, 
        threshold: float,
        responses: List[AgentResponse], 
        final_decision: str,
        avg_confidence: float,
        is_override: bool
    ) -> str:
        """Tworzy szczegółowy log z procesu decyzyjnego"""
        
        log = f"\n{'='*80}\n"
        log += f"MULTI-AGENT DECISION LOG - {datetime.now().isoformat()}\n"
        log += f"Detector: {detector_name} | Score: {score:.3f} | Threshold: {threshold}\n"
        log += f"{'='*80}\n\n"
        
        for response in responses:
            log += f"{response.role.value}:\n"
            log += f"  Decision: {response.decision} (confidence: {response.confidence:.3f})\n"
            log += f"  Reasoning: {response.reasoning}\n\n"
        
        log += f"{'-'*80}\n"
        log += f"FINAL DECISION: {final_decision}\n"
        log += f"Average Confidence: {avg_confidence:.3f}\n"
        log += f"Votes: {sum(1 for r in responses if r.decision == 'YES')}/{len(responses)} YES\n"
        
        if is_override:
            log += f"\n⚡ OVERRIDE ALERT - Agents overruled low score!\n"
        
        log += f"{'='*80}\n"
        
        return log
    
    def _save_decision_log(
        self, 
        detector_name: str, 
        score: float,
        decision: str, 
        confidence: float,
        responses: List[AgentResponse],
        is_override: bool
    ):
        """Zapisuje log decyzji do pliku dla feedback loop"""
        
        decision_entry = {
            'timestamp': datetime.now().isoformat(),
            'detector_name': detector_name,
            'score': score,
            'final_decision': decision,
            'confidence': confidence,
            'is_override': is_override,
            'agent_responses': [
                {
                    'role': r.role.value,
                    'decision': r.decision,
                    'confidence': r.confidence,
                    'reasoning': r.reasoning
                }
                for r in responses
            ]
        }
        
        # Załaduj istniejące logi
        try:
            if os.path.exists(self.decision_log_file):
                with open(self.decision_log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
        except:
            logs = []
        
        # Dodaj nowy wpis
        logs.append(decision_entry)
        
        # Zachowaj tylko ostatnie 1000 wpisów
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Zapisz
        os.makedirs(os.path.dirname(self.decision_log_file), exist_ok=True)
        with open(self.decision_log_file, 'w') as f:
            json.dump(logs, f, indent=2)


# Globalna instancja systemu
multi_agent_system = MultiAgentDecisionSystem()


async def evaluate_detector_with_agents(
    detector_name: str,
    score: float,
    signal_data: Dict[str, Any],
    market_data: Dict[str, Any],
    threshold: float = 0.7
) -> Tuple[str, float, str]:
    """
    Wrapper function dla łatwej integracji z consensus engine
    
    Returns:
        Tuple (decision, confidence, log)
    """
    return await multi_agent_system.multi_agent_decision(
        detector_name, score, signal_data, market_data, threshold
    )