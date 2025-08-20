#!/usr/bin/env python3
"""
Agent Runners - Execute individual agents with stable prompts and validation
"""

import logging
from typing import Dict, Any
from consensus.contracts import AgentOpinion
from consensus.prompts import ANALYZER_SYSTEM, REASONER_SYSTEM, VOTER_SYSTEM, DEBATER_SYSTEM
from consensus.agents import run_analyzer, run_reasoner, run_voter, run_debater
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
            "analyzer": ANALYZER_SYSTEM,
            "reasoner": REASONER_SYSTEM,
            "voter": VOTER_SYSTEM,
            "debater": DEBATER_SYSTEM
        }
    
    def run_analyzer(self, agent_input: AgentInput) -> AgentResponse:
        """Run Analyzer agent with evidential reasoning - using new simplified runner"""
        try:
            opinion = run_analyzer(agent_input.dict())
            # Convert AgentOpinion to AgentResponse for compatibility
            return AgentResponse(
                action_probs=opinion.action_probs,
                uncertainty=opinion.uncertainty,
                evidence=opinion.evidence,
                rationale=opinion.rationale,
                calibration_hint=opinion.calibration_hint
            )
        except Exception as e:
            # Fallback to stable client
            return stable_client.chat_json_only(
                system_prompt=self.prompts["analyzer"],
                user_data=agent_input.dict(),
                agent_name="ANALYZER"
            )
    
    def run_reasoner(self, agent_input: AgentInput) -> AgentResponse:
        """Run Reasoner agent with temporal analysis - using new simplified runner"""
        try:
            opinion = run_reasoner(agent_input.dict())
            return AgentResponse(
                action_probs=opinion.action_probs,
                uncertainty=opinion.uncertainty,
                evidence=opinion.evidence,
                rationale=opinion.rationale,
                calibration_hint=opinion.calibration_hint
            )
        except Exception as e:
            return stable_client.chat_json_only(
                system_prompt=self.prompts["reasoner"],
                user_data=agent_input.dict(),
                agent_name="REASONER"
            )
    
    def run_voter(self, agent_input: AgentInput) -> AgentResponse:
        """Run Voter agent with performance calibration - using new simplified runner"""
        try:
            opinion = run_voter(agent_input.dict())
            return AgentResponse(
                action_probs=opinion.action_probs,
                uncertainty=opinion.uncertainty,
                evidence=opinion.evidence,
                rationale=opinion.rationale,
                calibration_hint=opinion.calibration_hint
            )
        except Exception as e:
            return stable_client.chat_json_only(
                system_prompt=self.prompts["voter"],
                user_data=agent_input.dict(),
                agent_name="VOTER"
            )
    
    def run_debater(self, agent_input: AgentInput) -> AgentResponse:
        """Run Debater agent with pro/con analysis - using new simplified runner"""
        try:
            opinion = run_debater(agent_input.dict())
            return AgentResponse(
                action_probs=opinion.action_probs,
                uncertainty=opinion.uncertainty,
                evidence=opinion.evidence,
                rationale=opinion.rationale,
                calibration_hint=opinion.calibration_hint
            )
        except Exception as e:
            return stable_client.chat_json_only(
                system_prompt=self.prompts["debater"],
                user_data=agent_input.dict(),
                agent_name="DEBATER"
            )

# Global instance
agent_runners = AgentRunners()