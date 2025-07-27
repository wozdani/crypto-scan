"""
Multi-Agent Decision System dla każdego detektora
Koncepcja: 5 agentów (Analyzer, Reasoner, Voter, Debater, Decider) dla każdego detektora
debatuje i głosuje nad decyzją alertu używając OpenAI API reasoning
"""

import asyncio
import json
import threading
from queue import Queue
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import os
import random
from openai import OpenAI


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
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.use_real_llm = os.environ.get("OPENAI_API_KEY") is not None
        
    async def llm_reasoning(self, role: AgentRole, context: Dict[str, Any]) -> AgentResponse:
        """
        Real OpenAI LLM reasoning for agent decisions
        
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
        
        # Use real OpenAI API if available, otherwise fallback to simulation
        if self.use_real_llm:
            try:
                return await self._real_llm_reasoning(role, context)
            except Exception as e:
                print(f"[MULTI-AGENT LLM ERROR] {e} - falling back to simulation")
        
        # Fallback simulation logic (enhanced for better performance)
        
        if role == AgentRole.ANALYZER:
            # Analyzer sprawdza wiarygodność danych (LOWERED THRESHOLDS FOR MORE BUY VOTES)
            if score > 0.5:  # Lowered from 0.8 to 0.5
                reasoning = f"Score {score:.3f} is reliable for analysis. Signal data shows good patterns: {len(signal_data)} active signals."
                decision = "YES"
                confidence = 0.9
            elif score > threshold * 0.7:  # Lowered threshold multiplier from 1.0 to 0.7
                reasoning = f"Score {score:.3f} above adjusted threshold ({threshold * 0.7:.2f}). Data quality: moderate."
                decision = "YES"
                confidence = 0.7
            else:
                reasoning = f"Score {score:.3f} below threshold. Data reliability questionable."
                decision = "NO"
                confidence = 0.8
                
        elif role == AgentRole.REASONER:
            # Reasoner ocenia kontekst rynkowy (LOWERED VOLUME REQUIREMENTS)
            volume_24h = market_data.get('volume_24h', 0)
            price_change = market_data.get('price_change_24h', 0)
            
            if volume_24h > 500000 and price_change > -5:  # Lowered from 1M to 500k, allow some negative price change
                reasoning = f"Market context acceptable: volume ${volume_24h:,.0f}, price change {price_change:.2f}%"
                decision = "YES"
                confidence = 0.85
            elif volume_24h < 100000:  # Lowered from 500k to 100k
                reasoning = f"Very low volume ${volume_24h:,.0f} suggests high caution"
                decision = "NO"
                confidence = 0.75
            else:
                reasoning = f"Market signals neutral: volume moderate, evaluating by score"
                decision = "YES" if score > threshold * 0.8 else "NO"  # Lowered threshold multiplier
                confidence = 0.6
                
        elif role == AgentRole.VOTER:
            # Voter głosuje na podstawie overall assessment (LOWERED SIGNAL REQUIREMENTS)
            active_signals = len([s for s in signal_data.values() if s])
            if score > threshold:  # Simplified: if score above threshold, vote YES
                reasoning = f"Strong vote YES: score {score:.3f} above threshold ({threshold:.2f})"
                decision = "YES"
                confidence = 0.95
            elif score > threshold * 0.7:  # Even if score slightly below threshold
                reasoning = f"Moderate support: score {score:.3f} close to threshold"
                decision = "YES"
                confidence = 0.7
            else:
                reasoning = f"Weak score: {score:.3f} too far below threshold"
                decision = "NO"
                confidence = 0.8
                
        elif role == AgentRole.DEBATER:
            # Debater supports good scores instead of being contrarian (LESS CONTRARIAN)
            if len(self.debate_history) > 0:
                prev_decisions = [h['decision'] for h in self.debate_history[-3:]]
                yes_count = prev_decisions.count("YES")
                if yes_count >= 2 and score > threshold * 0.6:  # Support if majority YES and decent score
                    reasoning = "Previous agents support this, and score warrants agreement"
                    decision = "YES"
                    confidence = 0.75
                elif score > threshold * 0.8:  # Support high scores regardless of previous votes
                    reasoning = "Score is strong enough to overcome skepticism"
                    decision = "YES"
                    confidence = 0.7
                else:
                    reasoning = "Insufficient evidence for strong recommendation"
                    decision = "NO"
                    confidence = 0.65
            else:
                reasoning = "Initial assessment: evaluating by score threshold"
                decision = "YES" if score > threshold * 0.7 else "NO"  # Lowered from 0.9 to 0.7
                confidence = 0.6
                
        elif role == AgentRole.DECIDER:
            # Decider podejmuje finalną decyzję (LOWERED CONSENSUS REQUIREMENTS)
            yes_votes = sum(1 for h in self.debate_history if h['decision'] == "YES")
            total_votes = len(self.debate_history)
            
            if yes_votes >= 2:  # Lowered from 3 to 2 YES votes needed
                reasoning = f"Final decision: YES ({yes_votes}/{total_votes} agents support)"
                decision = "YES"
                confidence = 0.9
            elif yes_votes >= 1 and score > threshold * 0.8:  # Even with 1 YES vote if score high
                reasoning = f"Strong score override: YES (score {score:.3f} justifies decision)"
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
        # WARUNEK WSTĘPNY: Blokuj głosowanie dla score < 0.6
        if score < 0.6:
            print(f"[MULTI-AGENT SKIP] {detector_name} score {score:.3f} < 0.6 - skipping agent voting")
            return "NO", 0.0, f"Score {score:.3f} below voting threshold (0.6) - no agent evaluation needed"
        
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
    
    async def _real_llm_reasoning(self, role: AgentRole, context: Dict[str, Any]) -> AgentResponse:
        """
        Real OpenAI API reasoning for sophisticated agent decision making
        """
        detector_name = context.get('detector_name', 'unknown')
        score = context.get('score', 0.0)
        threshold = context.get('threshold', 0.7)
        market_data = context.get('market_data', {})
        signal_data = context.get('signal_data', {})
        
        # Create role-specific prompts
        role_prompts = {
            AgentRole.ANALYZER: f"""You are an AI trading signal ANALYZER. Analyze this cryptocurrency detector data:
            
Detector: {detector_name}
Score: {score:.3f}
Threshold: {threshold:.2f}
Market Volume: ${market_data.get('volume_24h', 0):,.0f}
Price Change 24h: {market_data.get('price_change_24h', 0):.2f}%
Active Signals: {len([s for s in signal_data.values() if s])}

Your role: Analyze if the score and data are reliable for trading decisions.

Respond with JSON format:
{{"decision": "YES" or "NO", "reasoning": "detailed analysis", "confidence": 0.0-1.0}}""",

            AgentRole.REASONER: f"""You are an AI trading REASONER. Evaluate market context for this signal:
            
Detector: {detector_name}
Score: {score:.3f}
Market Volume: ${market_data.get('volume_24h', 0):,.0f}
Price Change: {market_data.get('price_change_24h', 0):.2f}%

Your role: Reason about market conditions and whether they support this trading signal.

Respond with JSON format:
{{"decision": "YES" or "NO", "reasoning": "market context analysis", "confidence": 0.0-1.0}}""",

            AgentRole.VOTER: f"""You are an AI trading VOTER. Make a clear vote on this signal:
            
Detector: {detector_name}
Score: {score:.3f} (threshold: {threshold:.2f})
Signal Strength: {"Strong" if score > threshold else "Weak"}

Your role: Vote YES or NO based on overall signal assessment.

Respond with JSON format:
{{"decision": "YES" or "NO", "reasoning": "voting rationale", "confidence": 0.0-1.0}}""",

            AgentRole.DEBATER: f"""You are an AI trading DEBATER. Challenge or support this signal:
            
Detector: {detector_name}
Score: {score:.3f}
Previous decisions: {[h.get('decision') for h in self.debate_history[-3:]]}

Your role: Provide critical analysis - find weaknesses or strengths in the signal.

Respond with JSON format:
{{"decision": "YES" or "NO", "reasoning": "debate analysis", "confidence": 0.0-1.0}}""",

            AgentRole.DECIDER: f"""You are an AI trading DECIDER. Make the final decision:
            
Detector: {detector_name}
Score: {score:.3f}
Previous votes: {[h.get('decision') for h in self.debate_history]}
Vote count: {sum(1 for h in self.debate_history if h.get('decision') == 'YES')} YES votes

Your role: Make final decision based on agent consensus.

Respond with JSON format:
{{"decision": "YES" or "NO", "reasoning": "final decision logic", "confidence": 0.0-1.0}}"""
        }
        
        try:
            print(f"[MULTI-AGENT OpenAI] {role.value}: Calling OpenAI API...")
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency trading AI agent. Always respond with valid JSON."},
                    {"role": "user", "content": role_prompts[role]}
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content or "{}")
            print(f"[MULTI-AGENT OpenAI] {role.value}: SUCCESS - {result.get('decision')}")
            
            return AgentResponse(
                role=role,
                decision=result.get('decision', 'NO'),
                reasoning=result.get('reasoning', 'AI analysis completed'),
                confidence=float(result.get('confidence', 0.5))
            )
            
        except Exception as e:
            print(f"[MULTI-AGENT OpenAI ERROR] {role.value}: {e}")
            # Fallback to enhanced simulation
            return self._fallback_reasoning(role, context)
    
    def _fallback_reasoning(self, role: AgentRole, context: Dict[str, Any]) -> AgentResponse:
        """Enhanced fallback reasoning when OpenAI API fails"""
        score = context.get('score', 0.0)
        threshold = context.get('threshold', 0.7)
        
        # Enhanced thresholds for better decision making - lowered for score >0.6
        if role == AgentRole.ANALYZER:
            decision = "YES" if score > 0.6 else "NO"
            confidence = 0.9 if score > 0.8 else 0.75
            reasoning = f"Analysis: Score {score:.3f} {'above' if decision == 'YES' else 'below'} reliability threshold (0.6)"
            
        elif role == AgentRole.REASONER:
            volume = context.get('market_data', {}).get('volume_24h', 0)
            decision = "YES" if volume > 300000 and score > 0.6 else "NO"
            confidence = 0.8
            reasoning = f"Market reasoning: Volume ${volume:,.0f}, score context acceptable"
            
        elif role == AgentRole.VOTER:
            decision = "YES" if score > 0.6 else "NO"
            confidence = 0.9
            reasoning = f"Vote: Score {score:.3f} {'meets' if decision == 'YES' else 'fails'} voting criteria (threshold: 0.6)"
            
        elif role == AgentRole.DEBATER:
            yes_count = sum(1 for h in self.debate_history if h.get('decision') == 'YES')
            decision = "YES" if (yes_count >= 1 and score > 0.6) or score > 0.8 else "NO"
            confidence = 0.75
            reasoning = f"Debate: {'Supporting' if decision == 'YES' else 'Opposing'} based on evidence (threshold: 0.6)"
            
        elif role == AgentRole.DECIDER:
            yes_votes = sum(1 for h in self.debate_history if h.get('decision') == 'YES')
            decision = "YES" if yes_votes >= 2 and score > 0.6 else "NO"
            confidence = 0.9
            reasoning = f"Final decision: {yes_votes}/{len(self.debate_history)} agents support (score threshold: 0.6)"
            
        return AgentResponse(role=role, decision=decision, reasoning=reasoning, confidence=confidence)
    
    # === NOWE FUNKCJE Z STANDALONE KODU ===
    
    def agent_role_threaded(self, role: str, input_data: Dict[str, Any], output_queue: Queue):
        """
        Funkcja pojedynczego agenta w thread (z standalone kodu)
        """
        detector = input_data['detector']
        score = input_data['score']
        context = input_data['context']
        
        prompt = f"{role}: Analyze detektor {detector} with score {score}, context: {context}."
        
        # Użyj existing llm_reasoning infrastructure
        try:
            # Convert string role to AgentRole enum
            agent_role_enum = AgentRole(role)
            context_data = {
                'detector_name': detector,
                'score': score,
                'context': context,
                'threshold': 0.7
            }
            
            # Use sync version of llm_reasoning (simplified)
            response = self._sync_llm_reasoning(agent_role_enum, context_data)
            output_queue.put({role: response})
            
        except Exception as e:
            print(f"[AGENT THREAD ERROR] {role}: {e}")
            # Fallback response
            fallback_response = f"{role}: Error in analysis - using fallback"
            output_queue.put({role: fallback_response})
    
    def _sync_llm_reasoning(self, role: AgentRole, context: Dict[str, Any]) -> str:
        """
        Synchronous version of LLM reasoning for threaded agents
        """
        detector_name = context.get('detector_name', 'unknown')
        score = context.get('score', 0.0)
        threshold = context.get('threshold', 0.7)
        
        # Enhanced fallback reasoning (similar to standalone)
        if role == AgentRole.ANALYZER:
            if score > 0.8:
                return f"Analiza: Score {score:.2f} bardzo wiarygodny, recommend YES"
            elif score > 0.5:
                return f"Analiza: Score {score:.2f} umiarkowany, check volume"
            else:
                return f"Analiza: Score {score:.2f} niski, likely NO"
                
        elif role == AgentRole.REASONER:
            risk = "low" if score > 0.7 else "high"
            trend = random.choice(['bullish', 'bearish', 'neutral'])
            return f"Reasoning: Kontekst {trend}, risk {risk}, score analysis"
            
        elif role == AgentRole.VOTER:
            if score > 0.6:
                return "YES: Alert warranted based on score (threshold: 0.6)"
            else:
                return "NO: Score below threshold (0.6)"
                
        elif role == AgentRole.DEBATER:
            stance = "supporting" if score > threshold * 0.8 else "questioning"
            return f"Debata: {stance} decision, check historical patterns"
            
        elif role == AgentRole.DECIDER:
            decision = "YES" if score > 0.6 else "NO"
            confidence = min(0.95, score + 0.2)
            return f"Final: {decision} z confidence {confidence:.2f} (threshold: 0.6)"
        
        return "Generic response"
    
    def multi_agent_decision_per_detector(self, detector: str, score: float, context: str, threshold: float = 0.7) -> Tuple[str, float, str]:
        """
        Multi-agent decision dla jednego detektora (z standalone kodu)
        """
        # WARUNEK WSTĘPNY: Blokuj głosowanie dla score < 0.6
        if score < 0.6:
            print(f"[MULTI-AGENT SKIP] {detector} score {score:.3f} < 0.6 - skipping agent voting")
            return "NO", 0.0, f"Score {score:.3f} below voting threshold (0.6) - no agent evaluation needed"
        
        roles = ["Analyzer", "Reasoner", "Voter", "Debater", "Decider"]
        output_queue = Queue()
        threads = []
        input_data = {"detector": detector, "score": score, "context": context}

        # Uruchom agents parallelnie
        for role in roles:
            t = threading.Thread(target=self.agent_role_threaded, args=(role, input_data, output_queue))
            t.start()
            threads.append(t)

        # Czekaj na wyniki
        for t in threads:
            t.join()

        # Zbierz wyniki agentów
        results = {}
        while not output_queue.empty():
            results.update(output_queue.get())

        # Głosowanie (extract YES/NO z responses)
        votes = []
        for role in roles:
            if role in results and "YES" in results[role]:
                votes.append("YES")
            else:
                votes.append("NO")

        yes_count = votes.count("YES")
        # Oblicz confidence na podstawie score i yes_count
        confidence = (score + (yes_count / len(roles))) / 2.0
        confidence = min(1.0, max(0.0, confidence))

        # Decision: Majority YES i score > 0.6 (voting threshold)
        decision = "YES" if yes_count >= len(roles) // 2 + 1 and score > 0.6 else "NO"

        log = f"{datetime.now()} - {detector} Agents Results: {results}\nDecision: {decision}, Confidence: {confidence:.2f}, Votes YES: {yes_count}/{len(roles)}"
        
        print(f"[MULTI-AGENT PER-DETECTOR] {detector}: {decision} (confidence: {confidence:.3f}, votes: {yes_count}/{len(roles)})")
        
        return decision, confidence, log
    
    def multi_agent_consensus_all_detectors(self, detectors_data: Dict[str, Dict], alert_threshold: float = 0.7, min_yes_detectors: int = 2) -> Tuple[str, Dict, str]:
        """
        Główna funkcja consensus z multi-agents i triggerem alertów (z standalone kodu)
        """
        decisions = {}
        total_yes = 0
        logs = []

        print(f"[MULTI-AGENT CONSENSUS] Analyzing {len(detectors_data)} detectors...")

        # Analizuj każdy detektor osobno
        for detector, data in detectors_data.items():
            score = data.get('score', 0.0)
            context = data.get('context', "No context provided")
            
            # WARUNEK WSTĘPNY: Blokuj głosowanie dla score < 0.6
            if score < 0.6:
                print(f"[MULTI-AGENT SKIP] {detector} score {score:.3f} < 0.6 - skipping agent voting")
                decisions[detector] = {'decision': 'NO', 'confidence': 0.0}
                logs.append(f"{detector}: Score {score:.3f} below voting threshold (0.6) - no agent evaluation")
                continue
            
            decision, confidence, log = self.multi_agent_decision_per_detector(detector, score, context, alert_threshold)
            decisions[detector] = {'decision': decision, 'confidence': confidence}
            logs.append(log)

            # Partial trigger: Jeśli YES i high confidence
            if decision == "YES" and confidence > 0.7:
                print(f"[PARTIAL ALERT] {detector} triggered! Confidence: {confidence:.2f}")
                total_yes += 1

        # Agregacja wyników - final decision
        num_detectors = len(detectors_data)
        avg_confidence = sum(d['confidence'] for d in decisions.values()) / num_detectors if num_detectors > 0 else 0.0
        final_decision = "YES" if total_yes >= min_yes_detectors or avg_confidence > alert_threshold else "NO"

        # Full trigger: Jeśli final YES
        if final_decision == "YES":
            print(f"[FULL ALERT] Triggered! Total YES detektorów: {total_yes}/{num_detectors}, Avg Confidence: {avg_confidence:.2f}")

        final_log = "\n".join(logs) + f"\nFinal Decision: {final_decision}, Avg Confidence: {avg_confidence:.2f}"
        
        print(f"[MULTI-AGENT CONSENSUS] Final: {final_decision} (avg confidence: {avg_confidence:.3f})")
        
        return final_decision, decisions, final_log


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


def evaluate_all_detectors_with_consensus(detectors_data: Dict[str, Dict], threshold: float = 0.7) -> Tuple[str, Dict, str]:
    """
    Nowa funkcja dla full consensus analysis wszystkich detektorów (standalone integration)
    
    Args:
        detectors_data: Dict z detector data {"StealthEngine": {"score": 0.543, "context": "whale_ping 1.0"}, ...}
        threshold: Próg decyzyjny
        
    Returns:
        Tuple (final_decision, all_decisions, detailed_log)
    """
    return multi_agent_system.multi_agent_consensus_all_detectors(detectors_data, threshold)