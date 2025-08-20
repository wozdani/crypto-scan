# consensus/agents.py
import os
from .prompts import ANALYZER_SYSTEM, REASONER_SYSTEM, VOTER_SYSTEM, DEBATER_SYSTEM
from .contracts import AgentOpinion, CalibrationHint
from .reliability import get as get_reliability
from llm.llm_client import chat_json

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # fallback to gpt-4o if not set

def run_analyzer(payload) -> AgentOpinion:
    out = chat_json(DEFAULT_MODEL, ANALYZER_SYSTEM, payload)
    # Override calibration_hint with current reliability from EMA
    if 'calibration_hint' not in out:
        out['calibration_hint'] = {}
    out['calibration_hint']['reliability'] = get_reliability("Analyzer")
    return AgentOpinion(**out)

def run_reasoner(payload) -> AgentOpinion:
    out = chat_json(DEFAULT_MODEL, REASONER_SYSTEM, payload)
    if 'calibration_hint' not in out:
        out['calibration_hint'] = {}
    out['calibration_hint']['reliability'] = get_reliability("Reasoner")
    return AgentOpinion(**out)

def run_voter(payload) -> AgentOpinion:
    out = chat_json(DEFAULT_MODEL, VOTER_SYSTEM, payload)
    if 'calibration_hint' not in out:
        out['calibration_hint'] = {}
    out['calibration_hint']['reliability'] = get_reliability("Voter")
    return AgentOpinion(**out)

def run_debater(payload) -> AgentOpinion:
    out = chat_json(DEFAULT_MODEL, DEBATER_SYSTEM, payload)
    if 'calibration_hint' not in out:
        out['calibration_hint'] = {}
    out['calibration_hint']['reliability'] = get_reliability("Debater")
    return AgentOpinion(**out)