# consensus/agents.py
import os
from .prompts import ANALYZER_SYSTEM, REASONER_SYSTEM, VOTER_SYSTEM, DEBATER_SYSTEM
from .contracts import AgentOpinion
from llm.llm_client import chat_json

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # fallback to gpt-4o if not set

def run_analyzer(payload) -> AgentOpinion:
    out = chat_json(DEFAULT_MODEL, ANALYZER_SYSTEM, payload)
    return AgentOpinion(**out)

def run_reasoner(payload) -> AgentOpinion:
    out = chat_json(DEFAULT_MODEL, REASONER_SYSTEM, payload)
    return AgentOpinion(**out)

def run_voter(payload) -> AgentOpinion:
    out = chat_json(DEFAULT_MODEL, VOTER_SYSTEM, payload)
    return AgentOpinion(**out)

def run_debater(payload) -> AgentOpinion:
    out = chat_json(DEFAULT_MODEL, DEBATER_SYSTEM, payload)
    return AgentOpinion(**out)