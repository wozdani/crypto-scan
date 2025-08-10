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
import time
from openai import OpenAI

# Import DQN system for advanced reinforcement learning
try:
    from .dqn_multi_agent import get_dqn_integration, initialize_dqn_system
    DQN_AVAILABLE = True
    initialize_dqn_system = initialize_dqn_system  # Declare for LSP
    print("[MULTI-AGENT] ✅ DQN Advanced Reinforcement Learning system available")
except ImportError as e:
    DQN_AVAILABLE = False
    initialize_dqn_system = None  # Fallback for LSP
    print(f"[MULTI-AGENT] ⚠️ DQN system not available: {e}")


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
    decision: str  # BUY/HOLD/AVOID
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
        self.timeout = 30  # Default timeout for API calls
        # Initialize OpenAI client with proper error handling
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and api_key.strip():
                self.openai_client = OpenAI(api_key=api_key)
                self.use_real_llm = True
                print(f"[MULTI-AGENT] ✅ OpenAI client initialized successfully")
            else:
                self.openai_client = None
                self.use_real_llm = False
                print(f"[MULTI-AGENT] ⚠️ No valid OpenAI API key - using enhanced simulation")
        except Exception as e:
            self.openai_client = None
            self.use_real_llm = False
            print(f"[MULTI-AGENT] ⚠️ OpenAI initialization failed: {e} - using enhanced simulation")
        
    async def batch_llm_reasoning(self, all_contexts: List[Tuple[AgentRole, Dict[str, Any]]]) -> List[AgentResponse]:
        """
        OPTIMIZED: Pojedyncze zapytanie OpenAI dla wszystkich 5 agentów naraz
        Eliminuje problem 429 rate limiting przez redukcję z 5 zapytań do 1 zapytania
        """
        if not self.use_real_llm or not self.openai_client:
            # Fallback dla wszystkich agentów
            return [self._fallback_reasoning(role, context) for role, context in all_contexts]
        
        # Retry mechanism for OpenAI API calls
        max_retries = 3
        response = None
        
        for attempt in range(max_retries):
            try:
                # Przygotuj batch prompt dla wszystkich 5 agentów
                batch_prompt = self._create_batch_prompt(all_contexts)
                
                # Progressive timeout increase with each retry
                timeout_seconds = 30 + (attempt * 20)  # 30s, 50s, 70s
                print(f"[MULTI-AGENT BATCH] Attempt {attempt + 1}/{max_retries} with timeout {timeout_seconds}s")
                
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": "You are a cryptocurrency trading analysis panel of 5 expert agents. Each agent has a specialized role."},
                        {"role": "user", "content": batch_prompt}
                    ],
                    response_format={"type": "json_object"},
                    timeout=timeout_seconds,
                    temperature=1.0,
                    max_completion_tokens=2000
                )
                
                # If we get here, request succeeded
                break
                
            except Exception as retry_error:
                print(f"[MULTI-AGENT BATCH] Attempt {attempt + 1} failed: {retry_error}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"[MULTI-AGENT BATCH] Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed, fall through to error handling
                    raise retry_error
        
        # Check if we have a valid response after retries
        if response is None:
            print(f"[MULTI-AGENT BATCH ERROR] No valid response after {max_retries} retries")
            return [self._fallback_reasoning(role, context) for role, context in all_contexts]
        
        try:
            
            # Enhanced error handling for OpenAI response
            if not response.choices or len(response.choices) == 0:
                print(f"[MULTI-AGENT BATCH ERROR] No choices in OpenAI response")
                return [self._fallback_reasoning(role, context) for role, context in all_contexts]
            
            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                print(f"[MULTI-AGENT BATCH ERROR] Empty response from OpenAI API")
                print(f"[MULTI-AGENT BATCH ERROR] Response object: {response}")
                print(f"[MULTI-AGENT BATCH ERROR] Falling back to intelligent simulation")
                return [self._fallback_reasoning(role, context) for role, context in all_contexts]
                
            try:
                batch_result = json.loads(response_content)
                if not isinstance(batch_result, dict):
                    print(f"[MULTI-AGENT BATCH ERROR] Response is not a valid JSON object")
                    return [self._fallback_reasoning(role, context) for role, context in all_contexts]
                
                print(f"[MULTI-AGENT BATCH SUCCESS] Successfully processed batch response after retries")
                return self._parse_batch_response(batch_result, all_contexts)
                
            except json.JSONDecodeError as e:
                print(f"[MULTI-AGENT BATCH ERROR] Invalid JSON in response: {e}")
                print(f"[MULTI-AGENT BATCH ERROR] Raw response: {response_content[:300]}...")
                return [self._fallback_reasoning(role, context) for role, context in all_contexts]
            
        except Exception as e:
            error_type = type(e).__name__
            if "timeout" in str(e).lower() or "readtimeout" in str(e).lower():
                print(f"[MULTI-AGENT BATCH ERROR] OpenAI API timeout after {max_retries} retries: {e}")
            else:
                print(f"[MULTI-AGENT BATCH ERROR] OpenAI batch call failed: {e}")
                import traceback
                print(f"[MULTI-AGENT BATCH ERROR] Traceback: {traceback.format_exc()}")
            
            print(f"[MULTI-AGENT BATCH ERROR] Using intelligent fallback for all agents")
            # Enhanced fallback dla wszystkich agentów
            return [self._fallback_reasoning(role, context) for role, context in all_contexts]

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
                return await self._real_llm_reasoning_with_retry(role, context)
            except Exception as e:
                print(f"[MULTI-AGENT LLM ERROR] {e} - falling back to simulation")
        
        # Fallback to enhanced simulation when OpenAI unavailable
        return self._fallback_reasoning(role, context)
    
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
        # WARUNEK WSTĘPNY: Allow voting for any score >= 0.1 to enable proper evaluation
        if score < 0.1:
            print(f"[MULTI-AGENT SKIP] {detector_name} score {score:.3f} < 0.1 - skipping agent voting")
            return "AVOID", 0.0, f"Score {score:.3f} below voting threshold (0.1) - no agent evaluation needed"
        
        print(f"\n{'='*80}")
        print(f"[MULTI-AGENT VOTING] Starting 5-agent decision for {detector_name}")
        print(f"[MULTI-AGENT VOTING] Score: {score:.3f}, Threshold: {threshold:.3f}")
        
        # 🎯 CANONICAL PRICE INTEGRATION - Use consistent price for all agents
        volume_24h = market_data.get('volume_24h', 0) if market_data else 0
        price_change_24h = market_data.get('price_change_24h', 0) if market_data else 0
        canonical_price = market_data.get('canonical_price', 0) if market_data else 0
        price_source = market_data.get('canonical_price_source', 'unknown') if market_data else 'unknown'
        
        print(f"[MULTI-AGENT VOTING] Market data: Volume ${volume_24h:,.0f}, Price change {price_change_24h:.2f}%")
        print(f"[MULTI-AGENT VOTING] Canonical price: ${canonical_price:.6f} (source: {price_source})")
        print(f"[MULTI-AGENT DEBUG] Raw market_data type: {type(market_data)}, keys: {list(market_data.keys()) if market_data else 'None'}")
        print(f"{'='*80}\n")
        
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
        
        # BATCH EVALUATION: Wszystkich 5 agentów w jednym zapytaniu
        all_contexts = [
            (AgentRole.ANALYZER, context),
            (AgentRole.REASONER, context), 
            (AgentRole.VOTER, context),
            (AgentRole.DEBATER, context),
            (AgentRole.DECIDER, context)
        ]
        
        print(f"[MULTI-AGENT BATCH] Running batch evaluation for {detector_name} with all 5 agents in 1 API call...")
        
        # Uruchom batch evaluation dla wszystkich 5 agentów naraz
        all_responses = await self.batch_llm_reasoning(all_contexts)
        
        # Wyświetl głosy każdego agenta
        print(f"\n[MULTI-AGENT VOTES] Individual agent decisions:")
        for response in all_responses:
            vote_symbol = "🟢" if response.decision == "BUY" else "🟡" if response.decision == "HOLD" else "🔴"
            print(f"  {vote_symbol} {response.role.value}: {response.decision} (confidence: {response.confidence:.3f})")
            print(f"     Reasoning: {response.reasoning[:100]}...")
            
            # Zapisz do historii debaty
            self.debate_history.append({
                'role': response.role.value,
                'decision': response.decision,
                'reasoning': response.reasoning,
                'confidence': response.confidence,
                'timestamp': datetime.now().isoformat()
            })
        
        # Zlicz głosy agentów
        buy_votes = sum(1 for response in all_responses if response.decision == "BUY")
        hold_votes = sum(1 for response in all_responses if response.decision == "HOLD")
        avoid_votes = sum(1 for response in all_responses if response.decision == "AVOID")
        total_votes = len(all_responses)
        
        # Oblicz średnią confidence
        avg_confidence = sum(r.confidence for r in all_responses) / total_votes
        
        # Logika decyzyjna: BUY wymaga przewagi nad HOLD+AVOID
        if buy_votes > hold_votes + avoid_votes and buy_votes >= 3:
            final_decision = "BUY"
        elif hold_votes > buy_votes + avoid_votes:
            final_decision = "HOLD"
        else:
            final_decision = "AVOID"
        
        # Jeśli decyzja BUY mimo niskiego score - to jest override!
        is_override = final_decision == "BUY" and score < threshold
        
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
        
        # Podsumowanie głosowania
        print(f"\n[MULTI-AGENT SUMMARY]")
        print(f"  📊 Final Decision: {final_decision}")
        # Calculate vote distribution
        vote_difference = buy_votes - (hold_votes + avoid_votes)
        print(f"  🗳️ Votes: {buy_votes} BUY / {hold_votes} HOLD / {avoid_votes} AVOID")
        print(f"  💪 Average Confidence: {avg_confidence:.3f}")
        print(f"  🎯 Detector Score: {score:.3f} (threshold: {threshold:.3f})")
        
        if is_override:
            print(f"  ⚡ OVERRIDE ALERT! Agents voted BUY despite low score {score:.3f}")
        
        if final_decision == "BUY":
            print(f"  ✅ ALERT WILL BE TRIGGERED - Majority agents agree to BUY!")
        elif final_decision == "HOLD":
            print(f"  🟡 HOLD DECISION - Agents recommend waiting for more data")
        else:
            print(f"  ❌ AVOID DECISION - Agents recommend avoiding this signal")
            
        print(f"{'='*80}\n")
        
        # Ensure we return proper tuple with string types
        try:
            return str(final_decision), float(avg_confidence), str(detailed_log)
        except Exception as e:
            print(f"[MULTI-AGENT RETURN ERROR] {e}")
            return "AVOID", 0.0, f"Return error: {str(e)}"
    
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
        buy_count = sum(1 for r in responses if r.decision == 'BUY')
        hold_count = sum(1 for r in responses if r.decision == 'HOLD')
        avoid_count = sum(1 for r in responses if r.decision == 'AVOID')
        log += f"Votes: {buy_count} BUY / {hold_count} HOLD / {avoid_count} AVOID\n"
        
        if is_override:
            log += f"\n⚡ OVERRIDE ALERT - Agents overruled low score!\n"
        
        log += f"{'='*80}\n"
        
        return log
    
    async def batch_llm_reasoning(self, all_contexts: List[Tuple[AgentRole, Dict[str, Any]]]) -> List[AgentResponse]:
        """
        Wykonuje batch OpenAI call dla wszystkich 5 agentów w jednym zapytaniu
        """
        if not all_contexts:
            return []
            
        # Stwórz batch prompt dla wszystkich 5 agentów
        batch_prompt = self._create_batch_prompt(all_contexts)
        
        try:
            print(f"[BATCH API] Calling OpenAI with GPT-5 for {len(all_contexts)} agents...")
            
            # Single OpenAI API call for all 5 agents
            # GPT-5 only supports default temperature=1, not 0.1
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-5",
                messages=[{"role": "user", "content": batch_prompt}],
                response_format={"type": "json_object"},
                max_completion_tokens=2000,
                timeout=self.timeout
            )
            
            # Parse JSON response
            batch_result = json.loads(response.choices[0].message.content)
            
            # Convert batch result to individual AgentResponse objects
            responses = self._parse_batch_response(batch_result, all_contexts)
            
            print(f"[BATCH SUCCESS] Got {len(responses)} agent responses from single API call")
            return responses
            
        except asyncio.TimeoutError:
            print(f"[BATCH TIMEOUT] Batch API call timed out after {self.timeout}s")
            return [self._fallback_reasoning(role, context) for role, context in all_contexts]
        except json.JSONDecodeError as e:
            print(f"[BATCH JSON ERROR] Failed to parse batch response: {e}")
            return [self._fallback_reasoning(role, context) for role, context in all_contexts]
        except Exception as e:
            print(f"[BATCH ERROR] Batch evaluation failed: {e}")
            return [self._fallback_reasoning(role, context) for role, context in all_contexts]
    
    def _create_batch_prompt(self, all_contexts: List[Tuple[AgentRole, Dict[str, Any]]]) -> str:
        """
        Tworzy batch prompt dla wszystkich 5 agentów w jednym zapytaniu
        """
        if not all_contexts:
            return ""
            
        # Wyciągnij podstawowe dane z pierwszego kontekstu (wszystkie mają te same dane)
        _, context = all_contexts[0]
        detector_name = context.get('detector_name', 'unknown')
        score = context.get('score', 0.0)
        threshold = context.get('threshold', 0.7)
        market_data = context.get('market_data', {})
        signal_data = context.get('signal_data', {})
        
        volume_24h = market_data.get('volume_24h', 0) if market_data else 0
        price_change_24h = market_data.get('price_change_24h', 0) if market_data else 0
        
        prompt = f"""Analyze cryptocurrency detector signal: {detector_name}

DETECTOR DATA:
- Score: {score:.3f}
- Threshold: {threshold:.3f}
- Signal Data: {signal_data}

MARKET CONTEXT:
- Volume 24h: ${volume_24h:,.0f}
- Price Change 24h: {price_change_24h:.2f}%
- Market Data: {market_data}

You are a panel of 5 expert cryptocurrency trading agents. Each agent must evaluate this detector signal and provide a decision.

Return JSON with exactly this structure:
{{
    "analyzer": {{
        "decision": "BUY" or "HOLD" or "AVOID",
        "confidence": 0.0-1.0,
        "reasoning": "detailed analysis reasoning"
    }},
    "reasoner": {{
        "decision": "BUY" or "HOLD" or "AVOID", 
        "confidence": 0.0-1.0,
        "reasoning": "market reasoning and context"
    }},
    "voter": {{
        "decision": "BUY" or "HOLD" or "AVOID",
        "confidence": 0.0-1.0,
        "reasoning": "voting decision rationale"
    }},
    "debater": {{
        "decision": "BUY" or "HOLD" or "AVOID",
        "confidence": 0.0-1.0,
        "reasoning": "debate perspective and counterarguments"
    }},
    "decider": {{
        "decision": "BUY" or "HOLD" or "AVOID",
        "confidence": 0.0-1.0,
        "reasoning": "final decision based on all evidence"
    }}
}}

AGENT ROLES:
- Analyzer: Technical analysis of detector score vs threshold
- Reasoner: Market context and volume analysis
- Voter: Binary voting decision with confidence
- Debater: Challenge assumptions and provide counterarguments
- Decider: Final consensus based on other agents

Each agent should decide BUY (strong signal), HOLD (wait for more data), or AVOID (weak signal) based on the detector signal strength and market conditions."""

        return prompt

    def _parse_batch_response(self, batch_result: Dict, all_contexts: List[Tuple[AgentRole, Dict[str, Any]]]) -> List[AgentResponse]:
        """
        Parsuje batch response z OpenAI na listę AgentResponse
        """
        responses = []
        
        agent_keys = ['analyzer', 'reasoner', 'voter', 'debater', 'decider']
        agent_roles = [AgentRole.ANALYZER, AgentRole.REASONER, AgentRole.VOTER, AgentRole.DEBATER, AgentRole.DECIDER]
        
        for i, (role, agent_key) in enumerate(zip(agent_roles, agent_keys)):
            try:
                agent_data = batch_result.get(agent_key, {})
                decision = agent_data.get('decision', 'AVOID')
                confidence = float(agent_data.get('confidence', 0.5))
                reasoning = agent_data.get('reasoning', f'Default {role.value} reasoning')
                
                response = AgentResponse(
                    role=role,
                    decision=decision,
                    confidence=confidence,
                    reasoning=reasoning
                )
                responses.append(response)
                
            except Exception as e:
                print(f"[BATCH PARSE ERROR] Failed to parse {agent_key}: {e}")
                # Fallback response
                _, context = all_contexts[i] if i < len(all_contexts) else (role, {})
                responses.append(self._fallback_reasoning(role, context))
        
        return responses
    
    def _save_decision_log(
        self, 
        detector_name: str, 
        score: float,
        decision: str, 
        confidence: float,
        responses: List[AgentResponse],
        is_override: bool
    ) -> None:
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
        
        # USUNIĘTO LIMIT 1000 - system zachowuje wszystkie wpisy
        # Automatyczne czyszczenie explore mode odbywa się w osobnej funkcji
        
        # Zapisz
        os.makedirs(os.path.dirname(self.decision_log_file), exist_ok=True)
        with open(self.decision_log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    async def _real_llm_reasoning_with_retry(self, role: AgentRole, context: Dict[str, Any], retries: int = 3) -> AgentResponse:
        """
        Real OpenAI API reasoning with retry logic and fallback handling
        """
        for attempt in range(retries):
            try:
                return await self._real_llm_reasoning(role, context)
            except Exception as e:
                error_str = str(e)
                print(f"[MULTI-AGENT OpenAI ERROR] {role.value}: {error_str}")
                
                # Handle rate limiting (429) with exponential backoff
                if "insufficient_quota" in error_str or "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[MULTI-AGENT RETRY] {role.value}: Rate limit hit, waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"[MULTI-AGENT FALLBACK] {role.value}: All retries exhausted, using fallback reasoning")
                        return self._fallback_reasoning(role, context)
                
                # Handle invalid API key (401) or other critical errors
                elif "invalid_api_key" in error_str or "401" in error_str:
                    print(f"[MULTI-AGENT FALLBACK] {role.value}: Invalid API key, using fallback reasoning")
                    return self._fallback_reasoning(role, context)
                
                # Handle other errors with retry
                elif attempt < retries - 1:
                    wait_time = 1 + attempt
                    print(f"[MULTI-AGENT RETRY] {role.value}: Error occurred, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"[MULTI-AGENT FALLBACK] {role.value}: All retries failed, using fallback reasoning")
                    return self._fallback_reasoning(role, context)
        
        # Should not reach here, but safety fallback
        return self._fallback_reasoning(role, context)
    
    def _fallback_reasoning(self, role: AgentRole, context: Dict[str, Any]) -> AgentResponse:
        """Enhanced fallback reasoning when OpenAI API fails or returns empty response"""
        score = context.get('score', 0.0)
        threshold = context.get('threshold', 0.7)
        
        # Enhanced thresholds for better decision making using BUY/HOLD/AVOID
        if role == AgentRole.ANALYZER:
            market_data = context.get('market_data', {})
            canonical_price = market_data.get('canonical_price', 0)
            price_source = market_data.get('canonical_price_source', 'unknown')
            
            if score >= 0.75:
                decision = "BUY"
                confidence = 0.9
                reasoning = f"Analysis: Score {score:.3f} above strong threshold (0.75) - reliable signal. Price: ${canonical_price:.6f} ({price_source})"
            elif score >= 0.55:
                decision = "HOLD"
                confidence = 0.7
                reasoning = f"Analysis: Score {score:.3f} moderate - need more data. Price: ${canonical_price:.6f} ({price_source})"
            else:
                decision = "AVOID"
                confidence = 0.75
                reasoning = f"Analysis: Score {score:.3f} below reliability threshold (0.55). Price: ${canonical_price:.6f} ({price_source})"
            
        elif role == AgentRole.REASONER:
            market_data = context.get('market_data', {})
            volume = market_data.get('volume_24h', 0)
            canonical_price = market_data.get('canonical_price', 0)
            price_source = market_data.get('canonical_price_source', 'unknown')
            
            if volume > 1000000 and score >= 0.7:
                decision = "BUY"
                confidence = 0.8
                reasoning = f"Market reasoning: High volume ${volume:,.0f}, price ${canonical_price:.6f} ({price_source}), strong score context"
            elif volume > 300000 and score >= 0.6:
                decision = "HOLD"
                confidence = 0.7
                reasoning = f"Market reasoning: Volume ${volume:,.0f}, price ${canonical_price:.6f} ({price_source}), moderate score context"
            else:
                decision = "AVOID"
                confidence = 0.8
                reasoning = f"Market reasoning: Volume ${volume:,.0f}, price ${canonical_price:.6f} ({price_source}), insufficient for trading"
            
        elif role == AgentRole.VOTER:
            market_data = context.get('market_data', {})
            canonical_price = market_data.get('canonical_price', 0)
            price_source = market_data.get('canonical_price_source', 'unknown')
            
            if score >= 0.7:
                decision = "BUY"
                confidence = 0.9
                reasoning = f"Vote: Score {score:.3f} meets strong voting criteria (≥0.7). Price: ${canonical_price:.6f} ({price_source})"
            elif score >= 0.55:
                decision = "HOLD"
                confidence = 0.7
                reasoning = f"Vote: Score {score:.3f} meets moderate criteria - watchlist. Price: ${canonical_price:.6f} ({price_source})"
            else:
                decision = "AVOID"
                confidence = 0.85
                reasoning = f"Vote: Score {score:.3f} fails voting criteria (<0.55). Price: ${canonical_price:.6f} ({price_source})"
            
        elif role == AgentRole.DEBATER:
            buy_count = sum(1 for h in self.debate_history if h.get('decision') == 'BUY')
            hold_count = sum(1 for h in self.debate_history if h.get('decision') == 'HOLD')
            if (buy_count >= 1 and score > 0.65) or score > 0.8:
                decision = "BUY"
                confidence = 0.75
                reasoning = f"Debate: Supporting based on evidence (score: {score:.3f}, buy_votes: {buy_count})"
            elif buy_count == 0 and hold_count >= 1 and score > 0.55:
                decision = "HOLD"
                confidence = 0.7
                reasoning = f"Debate: Neutral position - need more evidence"
            else:
                decision = "AVOID"
                confidence = 0.75
                reasoning = f"Debate: Opposing based on evidence (score: {score:.3f})"
            
        elif role == AgentRole.DECIDER:
            buy_votes = sum(1 for h in self.debate_history if h.get('decision') == 'BUY')
            hold_votes = sum(1 for h in self.debate_history if h.get('decision') == 'HOLD')
            avoid_votes = sum(1 for h in self.debate_history if h.get('decision') == 'AVOID')
            
            if buy_votes >= 2 and score > 0.6:
                decision = "BUY"
                confidence = 0.9
                reasoning = f"Final decision: {buy_votes} BUY votes support signal (score: {score:.3f})"
            elif buy_votes + hold_votes >= 2 and score > 0.5:
                decision = "HOLD"
                confidence = 0.7
                reasoning = f"Final decision: Mixed signals - {buy_votes}B/{hold_votes}H/{avoid_votes}A"
            else:
                decision = "AVOID"
                confidence = 0.8
                reasoning = f"Final decision: Insufficient consensus - {buy_votes}B/{hold_votes}H/{avoid_votes}A"
            
        else:
            # Default fallback for any unknown role
            if score >= 0.7:
                decision = "BUY"
                confidence = 0.7
                reasoning = f"Default fallback: Score {score:.3f} above BUY threshold (0.7)"
            elif score >= 0.55:
                decision = "HOLD"
                confidence = 0.6
                reasoning = f"Default fallback: Score {score:.3f} moderate - HOLD"
            else:
                decision = "AVOID"
                confidence = 0.7
                reasoning = f"Default fallback: Score {score:.3f} below trading threshold"
            
        return AgentResponse(role=role, decision=decision, reasoning=reasoning, confidence=confidence)
    
# Function removed - now using consolidated _fallback_reasoning instead

    async def _real_llm_reasoning(self, role: AgentRole, context: Dict[str, Any]) -> AgentResponse:
        """
        Real OpenAI API reasoning for sophisticated agent decision making
        """
        detector_name = context.get('detector_name', 'unknown')
        score = context.get('score', 0.0)
        threshold = context.get('threshold', 0.7)
        market_data = context.get('market_data', {})
        signal_data = context.get('signal_data', {})
        
        print(f"[OPENAI API] Calling GPT-4o for {role.value} agent...")
        
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
{{"decision": "BUY" or "HOLD" or "AVOID", "reasoning": "detailed analysis", "confidence": 0.0-1.0}}""",

            AgentRole.REASONER: f"""You are an AI trading REASONER. Evaluate market context for this signal:
            
Detector: {detector_name}
Score: {score:.3f}
Market Volume: ${market_data.get('volume_24h', 0):,.0f}
Price Change: {market_data.get('price_change_24h', 0):.2f}%

Your role: Reason about market conditions and whether they support this trading signal.

Respond with JSON format:
{{"decision": "BUY" or "HOLD" or "AVOID", "reasoning": "market context analysis", "confidence": 0.0-1.0}}""",

            AgentRole.VOTER: f"""You are an AI trading VOTER. Make a clear vote on this signal:
            
Detector: {detector_name}
Score: {score:.3f} (threshold: {threshold:.2f})
Signal Strength: {"Strong" if score > threshold else "Weak"}

Your role: Vote BUY (strong signal), HOLD (wait for more data), or AVOID (weak signal).

Respond with JSON format:
{{"decision": "BUY" or "HOLD" or "AVOID", "reasoning": "voting rationale", "confidence": 0.0-1.0}}""",

            AgentRole.DEBATER: f"""You are an AI trading DEBATER. Challenge or support this signal:
            
Detector: {detector_name}
Score: {score:.3f}
Previous decisions: {[h.get('decision') for h in self.debate_history[-3:]]}

Your role: Provide critical analysis - find weaknesses or strengths in the signal.

Respond with JSON format:
{{"decision": "BUY" or "HOLD" or "AVOID", "reasoning": "debate analysis", "confidence": 0.0-1.0}}""",

            AgentRole.DECIDER: f"""You are an AI trading DECIDER. Make the final decision:
            
Detector: {detector_name}
Score: {score:.3f}
Previous votes: {[h.get('decision') for h in self.debate_history]}
BUY votes: {sum(1 for h in self.debate_history if h.get('decision') == 'BUY')}
HOLD votes: {sum(1 for h in self.debate_history if h.get('decision') == 'HOLD')}
AVOID votes: {sum(1 for h in self.debate_history if h.get('decision') == 'AVOID')}

Your role: Make final decision based on agent consensus.

Respond with JSON format:
{{"decision": "BUY" or "HOLD" or "AVOID", "reasoning": "final decision logic", "confidence": 0.0-1.0}}"""
        }
        
        try:
            # Check if OpenAI client is available
            if not self.openai_client:
                print(f"[MULTI-AGENT OpenAI] {role.value}: No OpenAI client - using fallback")
                return self._fallback_reasoning(role, context)
                
            print(f"[MULTI-AGENT OpenAI] {role.value}: Calling OpenAI API...")
            
            # Upgraded to GPT-5 for enhanced crypto trading decision capabilities
            # using latest OpenAI model for superior pattern recognition
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency trading AI agent. Always respond with valid JSON."},
                    {"role": "user", "content": role_prompts[role]}
                ],
                response_format={"type": "json_object"},
                temperature=1.0,
                max_completion_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content or "{}")
            print(f"[MULTI-AGENT OpenAI] {role.value}: SUCCESS - {result.get('decision')}")
            
            return AgentResponse(
                role=role,
                decision=result.get('decision', 'AVOID'),
                reasoning=result.get('reasoning', 'AI analysis completed'),
                confidence=float(result.get('confidence', 0.5))
            )
            
        except Exception as e:
            print(f"[MULTI-AGENT OpenAI ERROR] {role.value}: {e}")
            # Fallback to enhanced simulation
            return self._fallback_reasoning(role, context)
    
    # Removed duplicate _fallback_reasoning - using the correct BUY/HOLD/AVOID version above
    
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

        # Count measured vs unmeasured detectors
        total_detectors = len(detectors_data)
        measured_detectors = 0
        unmeasured_detectors = 0
        
        # Analizuj każdy detektor osobno
        for detector, data in detectors_data.items():
            score = data.get('score', 0.0)
            context = data.get('context', "No context provided")
            status = data.get('status', 'MEASURED')  # Check if detector is UNMEASURED
            
            # Skip UNMEASURED detectors neutrally (no penalty for missing microstructure data)
            if status == 'UNMEASURED':
                print(f"[MULTI-AGENT UNMEASURED] {detector} status=UNMEASURED - neutral impact (no orderbook data)")
                decisions[detector] = {'decision': 'UNMEASURED', 'confidence': 0.0, 'status': 'UNMEASURED'}
                logs.append(f"{detector}: UNMEASURED (synthetic orderbook) - neutral impact")
                unmeasured_detectors += 1
                continue
                
            measured_detectors += 1
            
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

        # Agregacja wyników with coverage ratio adjustment
        coverage_ratio = measured_detectors / total_detectors if total_detectors > 0 else 1.0
        avg_confidence_raw = sum(d['confidence'] for d in decisions.values() if d.get('status') != 'UNMEASURED') / measured_detectors if measured_detectors > 0 else 0.0
        avg_confidence = min(avg_confidence_raw, coverage_ratio)  # Cap confidence by coverage ratio
        final_decision = "YES" if total_yes >= min_yes_detectors or avg_confidence > alert_threshold else "NO"
        
        print(f"[COVERAGE RATIO] Total: {total_detectors}, Measured: {measured_detectors}, Unmeasured: {unmeasured_detectors}, Coverage: {coverage_ratio:.2f}")
        print(f"[CONFIDENCE CAP] Raw confidence: {avg_confidence_raw:.3f}, Coverage-capped: {avg_confidence:.3f}")

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
    # Fallback when full consensus not available
    try:
        return multi_agent_system.multi_agent_consensus_all_detectors(detectors_data, threshold)
    except AttributeError:
        print(f"[MULTI-AGENT WARNING] Full consensus method not available, using simplified approach")
        return ("HOLD", {}, "Simplified multi-agent analysis completed")


class DQNEnhancedMultiAgentSystem(MultiAgentDecisionSystem):
    """
    Enhanced Multi-Agent System with DQN Reinforcement Learning
    Allows agents to learn and adapt based on market feedback
    """
    
    def __init__(self):
        super().__init__()
        self.dqn_integration = None
        self.learning_enabled = True
        self.feedback_history = []
        
        # Initialize DQN system if available
        if DQN_AVAILABLE:
            try:
                if 'initialize_dqn_system' in globals():
                    self.dqn_integration = initialize_dqn_system()
                    print("[DQN MULTI-AGENT] ✅ Enhanced Multi-Agent System with DQN Reinforcement Learning initialized")
                else:
                    print("[DQN MULTI-AGENT] ⚠️ DQN initialize function not available")
                    self.dqn_integration = None
            except Exception as e:
                print(f"[DQN MULTI-AGENT] ⚠️ DQN initialization failed: {e}")
                self.dqn_integration = None
    
    async def enhanced_consensus_with_dqn(self,
                                         symbol: str,
                                         detector_scores: Dict[str, float],
                                         detector_votes: Dict[str, str],
                                         market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced consensus decision with DQN learning integration
        
        Args:
            symbol: Token symbol
            detector_scores: Detector scores
            detector_votes: Detector votes (BUY/HOLD/AVOID)
            market_context: Market data for learning
            
        Returns:
            Enhanced consensus result with DQN adjustments
        """
        # Calculate consensus score
        import numpy as np
        consensus_score = np.mean(list(detector_scores.values())) if detector_scores else 0.0
        
        # Determine majority decision
        vote_counts = {'BUY': 0, 'HOLD': 0, 'AVOID': 0}
        for vote in detector_votes.values():
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        consensus_decision = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        
        result = {
            'symbol': symbol,
            'consensus_score': consensus_score,
            'consensus_decision': consensus_decision,
            'detector_scores': detector_scores,
            'detector_votes': detector_votes,
            'vote_counts': vote_counts,
            'market_context': market_context,
            'dqn_enhanced': False,
            'dqn_adjustments': {}
        }
        
        # Apply DQN enhancements if available
        if self.dqn_integration and self.learning_enabled:
            try:
                dqn_result = self.dqn_integration.process_consensus_decision(
                    consensus_score=float(consensus_score),
                    consensus_decision=consensus_decision,
                    detector_votes=detector_votes,
                    detector_scores=detector_scores,
                    market_context=market_context
                )
                
                result['dqn_enhanced'] = True
                result['dqn_adjustments'] = dqn_result
                result['adjusted_weights'] = dqn_result.get('new_weights', {})
                result['adjusted_threshold'] = dqn_result.get('new_threshold', 0.7)
                
                print(f"[DQN ENHANCED CONSENSUS] {symbol}: DQN action={dqn_result.get('dqn_action')}, threshold={dqn_result.get('new_threshold', 0.7):.3f}")
                
            except Exception as e:
                print(f"[DQN ENHANCED CONSENSUS ERROR] {symbol}: {e}")
        
        return result
    
    def update_dqn_with_market_feedback(self,
                                       symbol: str,
                                       original_decision: str,
                                       price_change_pct: float,
                                       time_elapsed_hours: float,
                                       market_context: Dict[str, Any]):
        """
        Update DQN system with actual market feedback
        
        Args:
            symbol: Token symbol
            original_decision: Original consensus decision
            price_change_pct: Price change percentage
            time_elapsed_hours: Hours since decision
            market_context: Updated market context
        """
        if not self.dqn_integration or not self.learning_enabled:
            return
        
        try:
            # Update DQN with feedback
            self.dqn_integration.update_with_feedback(
                symbol=symbol,
                price_change_pct=price_change_pct,
                consensus_decision=original_decision,
                market_context=market_context
            )
            
            # Store feedback for analysis
            feedback_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'original_decision': original_decision,
                'price_change_pct': price_change_pct,
                'time_elapsed_hours': time_elapsed_hours,
                'outcome': 'positive' if price_change_pct > 2.0 else 'negative' if price_change_pct < -2.0 else 'neutral'
            }
            
            self.feedback_history.append(feedback_record)
            
            # Keep only last 100 feedback records
            if len(self.feedback_history) > 100:
                self.feedback_history = self.feedback_history[-100:]
            
            print(f"[DQN FEEDBACK] {symbol}: {original_decision} → {price_change_pct:+.2f}% ({feedback_record['outcome']})")
            
        except Exception as e:
            print(f"[DQN FEEDBACK ERROR] {symbol}: {e}")
    
    def get_dqn_adjusted_weights(self) -> Dict[str, float]:
        """Get current DQN-adjusted detector weights"""
        if self.dqn_integration:
            return self.dqn_integration.get_current_weights()
        return {}
    
    def get_dqn_adjusted_threshold(self) -> float:
        """Get current DQN-adjusted consensus threshold"""
        if self.dqn_integration:
            return self.dqn_integration.get_current_threshold()
        return 0.7
    
    def save_dqn_state(self):
        """Save DQN learning state"""
        if self.dqn_integration:
            self.dqn_integration.save_state()


# Global enhanced system instance
enhanced_multi_agent_system = DQNEnhancedMultiAgentSystem()


def evaluate_with_dqn_enhancement(symbol: str,
                                 detector_scores: Dict[str, float],
                                 detector_votes: Dict[str, str],
                                 market_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main integration function for DQN-enhanced multi-agent consensus
    
    Args:
        symbol: Token symbol
        detector_scores: Individual detector scores
        detector_votes: Individual detector votes
        market_context: Market data context
        
    Returns:
        Enhanced consensus result with DQN learning
    """
    import asyncio
    
    # Run enhanced consensus
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            enhanced_multi_agent_system.enhanced_consensus_with_dqn(
                symbol, detector_scores, detector_votes, market_context or {}
            )
        )
    finally:
        loop.close()
    
    return result


def update_dqn_feedback(symbol: str,
                       original_decision: str,
                       price_change_pct: float,
                       time_elapsed_hours: float = 2.0,
                       market_context: Dict[str, Any] = None) -> None:
    """
    Update DQN system with market feedback
    
    Args:
        symbol: Token symbol
        original_decision: Original consensus decision
        price_change_pct: Price change percentage since decision
        time_elapsed_hours: Hours elapsed since decision
        market_context: Updated market context
    """
    if market_context is None:
        market_context = {}
    
    if hasattr(enhanced_multi_agent_system, 'update_dqn_with_market_feedback'):
        enhanced_multi_agent_system.update_dqn_with_market_feedback(
            symbol, original_decision, price_change_pct, time_elapsed_hours, market_context
        )