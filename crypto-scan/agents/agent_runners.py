#!/usr/bin/env python3
"""
Agent Runners - Execute individual agents with stable prompts and validation
"""

import logging
from typing import Dict, Any
from consensus.contracts import AgentOpinion
from contracts.agent_contracts import AgentInput, AgentResponse
from llm.stable_client import stable_client

logger = logging.getLogger(__name__)

class AgentRunners:
    """Runs individual agents with specialized prompts"""
    
    def __init__(self):
        self.prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize system prompts for all agents"""
        return {
            "analyzer": """Jesteś Analyzerem. Nie używaj sztywnych reguł. Zamiast tego:

- Zidentyfikuj wzorce współwystępowania (np. whale+dex+orderbook) jako dowody pro lub kontra
- Oceniaj spójność: czy sygnały wskazują na tę samą hipotezę (akumulacja → pre-pump)?
- Zgłoś action_probs przez miękkie ważenie dowodów (no hard thresholds)
- Oszacuj epistemic/aleatoric niepewność na podstawie konflitku/chaosu danych

Heurystyka (opisowa, nie reguła):
- Traktuj kombinacje jako silniejsze dowody, ale jeśli istnieją kontrdowody (np. news-only), osłab część siły
- Zawsze raportuj co najmniej 5 pozycji evidence (mogą być neutral)

Zwróć TYLKO JSON o dokładnie tej strukturze:
{
  "action_probs": {"BUY": 0.3, "HOLD": 0.4, "AVOID": 0.2, "ABSTAIN": 0.1},
  "uncertainty": {"epistemic": 0.3, "aleatoric": 0.2},
  "evidence": [
    {"name": "whale_dex_correlation", "direction": "pro", "strength": 0.7},
    {"name": "orderbook_anomaly", "direction": "pro", "strength": 0.5},
    {"name": "volume_consistency", "direction": "neutral", "strength": 0.3},
    {"name": "news_counter_signal", "direction": "con", "strength": 0.4},
    {"name": "pattern_coherence", "direction": "pro", "strength": 0.6}
  ],
  "rationale": "Moderate signals with some coherence but mixed evidence quality.",
  "calibration_hint": {"reliability": 0.8, "expected_ttft_mins": 25}
}""",

            "reasoner": """Jesteś agentem REASONER. Oceniaj sekwencyjne rozumowanie i wystarczalność dowodów w czasie bez twardych progów.

Oceniaj:
- Spójność temporalną (czy dowody eskalują czy zanikają?)
- Recykling adresów i rytm (powtarzalność w 24h/72h)
- Wzorce sekwencyjne bez reguł typu '≥N zdarzeń'

Używaj miękkich metryk:
- Ocena spójności temporalnej (opisowa, 0..1)
- Siła powtarzalności (opisowa, 0..1)
- Kara za konflikty (opisowa, 0..1)

Wskazówki (opisowe, nie if-then):
- Spójne wzorce temporalne → zwiększ 'BUY' miękko
- Asymetryczne wzorce (piki + brak kontynuacji) → zwiększ 'HOLD'/'ABSTAIN'
- Tylko-newsy → podnieś aleatoric uncertainty

Zwróć TYLKO JSON:
{
  "action_probs": {"BUY": 0.25, "HOLD": 0.45, "AVOID": 0.2, "ABSTAIN": 0.1},
  "uncertainty": {"epistemic": 0.4, "aleatoric": 0.3},
  "evidence": [
    {"name": "temporal_coherence", "direction": "pro", "strength": 0.6},
    {"name": "address_recycling", "direction": "neutral", "strength": 0.4},
    {"name": "pattern_consistency", "direction": "con", "strength": 0.3},
    {"name": "sequence_escalation", "direction": "pro", "strength": 0.5},
    {"name": "rhythm_analysis", "direction": "neutral", "strength": 0.4}
  ],
  "rationale": "Temporal patterns show moderate consistency with some recycling activity.",
  "calibration_hint": {"reliability": 0.75, "expected_ttft_mins": 30}
}""",

            "voter": """Jesteś agentem VOTER. Kalibrujesz decyzje względem rzeczywistej, niedawnej wydajności bez progów.

Używaj miękkich wag dla:
- Detektorów z wyższą precision_7d i niższą fp_rate
- Jeśli avg_lag_mins jest wysokie, przesuń prawdopodobieństwo w kierunku 'HOLD'
- Brak progów - operuj proporcjami i miękkimi karami

W calibration_hint.reliability podaj rolling VOTER reliability (0.6-0.9) dla późniejszej meta-kalibracji.

Skup się na statystycznym ugruntowaniu, a nie na rozpoznawaniu wzorców.

Zwróć TYLKO JSON:
{
  "action_probs": {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
  "uncertainty": {"epistemic": 0.35, "aleatoric": 0.25},
  "evidence": [
    {"name": "detector_precision", "direction": "pro", "strength": 0.7},
    {"name": "false_positive_rate", "direction": "con", "strength": 0.4},
    {"name": "lag_compensation", "direction": "neutral", "strength": 0.5},
    {"name": "performance_trend", "direction": "pro", "strength": 0.6},
    {"name": "statistical_confidence", "direction": "neutral", "strength": 0.5}
  ],
  "rationale": "Performance calibration suggests moderate confidence with lag considerations.",
  "calibration_hint": {"reliability": 0.85, "expected_ttft_mins": 20}
}""",

            "debater": """Jesteś agentem DEBATER. Twórz wyraźne argumenty trade-off.

Proces:
1. Utwórz pary argumentów (pro/con) ze wszystkich danych wejściowych
2. Przypisz siłę (0..1) każdemu argumentowi opisowo, bez progów
3. Zastosuj miękkie balansowanie: dominacja = średnia(pros) - średnia(cons) (opisowo)
4. Konwertuj na action_probs + uncertainty

Żadnych binarnych decyzji - wszystko to miękkie równoważenie dowodów.

Skup się na:
- Wyraźnych argumentach pro vs con
- Analizie trade-off
- Ważeniu siły bez twardych granic

Zwróć TYLKO JSON:
{
  "action_probs": {"BUY": 0.3, "HOLD": 0.4, "AVOID": 0.2, "ABSTAIN": 0.1},
  "uncertainty": {"epistemic": 0.3, "aleatoric": 0.3},
  "evidence": [
    {"name": "pro_arguments", "direction": "pro", "strength": 0.6},
    {"name": "con_arguments", "direction": "con", "strength": 0.4},
    {"name": "argument_balance", "direction": "neutral", "strength": 0.5},
    {"name": "trade_off_analysis", "direction": "neutral", "strength": 0.7},
    {"name": "evidence_weight", "direction": "pro", "strength": 0.5}
  ],
  "rationale": "Pro arguments slightly outweigh cons but with significant uncertainty.",
  "calibration_hint": {"reliability": 0.7, "expected_ttft_mins": 35}
}"""
        }
    
    def run_analyzer(self, agent_input: AgentInput) -> AgentResponse:
        """Run Analyzer agent with evidential reasoning"""
        return stable_client.chat_json_only(
            system_prompt=self.prompts["analyzer"],
            user_data=agent_input.dict(),
            agent_name="ANALYZER"
        )
    
    def run_reasoner(self, agent_input: AgentInput) -> AgentResponse:
        """Run Reasoner agent with temporal analysis"""
        return stable_client.chat_json_only(
            system_prompt=self.prompts["reasoner"],
            user_data=agent_input.dict(),
            agent_name="REASONER"
        )
    
    def run_voter(self, agent_input: AgentInput) -> AgentResponse:
        """Run Voter agent with performance calibration"""
        return stable_client.chat_json_only(
            system_prompt=self.prompts["voter"],
            user_data=agent_input.dict(),
            agent_name="VOTER"
        )
    
    def run_debater(self, agent_input: AgentInput) -> AgentResponse:
        """Run Debater agent with pro/con analysis"""
        return stable_client.chat_json_only(
            system_prompt=self.prompts["debater"],
            user_data=agent_input.dict(),
            agent_name="DEBATER"
        )

# Global instance
agent_runners = AgentRunners()