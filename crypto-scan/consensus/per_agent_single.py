# consensus/per_agent_single.py
"""
Per-agent micro-calls as ultimate fallback to guarantee 'agents' section
"""
import json
from typing import Dict, Any
from llm.single_client import chat_json_schema_single, repair_to_schema

# Individual agent schemas
AGENT_OP_SCHEMA = {
    "type": "object",
    "properties": {
        "action_probs": {
            "type": "object",
            "properties": {
                "BUY": {"type": "number"},
                "HOLD": {"type": "number"}, 
                "AVOID": {"type": "number"},
                "ABSTAIN": {"type": "number"}
            },
            "required": ["BUY", "HOLD", "AVOID", "ABSTAIN"],
            "additionalProperties": False
        },
        "uncertainty": {
            "type": "object",
            "properties": {
                "epistemic": {"type": "number"},
                "aleatoric": {"type": "number"}
            },
            "required": ["epistemic", "aleatoric"],
            "additionalProperties": False
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "direction": {"type": "string", "enum": ["pro", "con", "neutral"]},
                    "strength": {"type": "number"}
                },
                "required": ["name", "direction", "strength"],
                "additionalProperties": False
            },
            "minItems": 3,
            "maxItems": 3
        },
        "rationale": {"type": "string"},
        "calibration_hint": {
            "type": "object",
            "properties": {
                "reliability": {"type": "number"},
                "expected_ttft_mins": {"type": "number"}
            },
            "required": ["reliability", "expected_ttft_mins"],
            "additionalProperties": False
        }
    },
    "required": ["action_probs", "uncertainty", "evidence", "rationale", "calibration_hint"],
    "additionalProperties": False
}

# Ultra-short per-agent prompts
AGENT_SYSTEMS = {
    "Analyzer": """Zwróć WYŁĄCZNIE JSON dla Analyzer:
{"action_probs":{"BUY":0.25,"HOLD":0.45,"AVOID":0.20,"ABSTAIN":0.10},
 "uncertainty":{"epistemic":0.3,"aleatoric":0.2},
 "evidence":[{"name":"whale_ping","direction":"pro","strength":0.6},
             {"name":"volume_spike","direction":"neutral","strength":0.4},
             {"name":"liquidity","direction":"con","strength":0.3}],
 "rationale":"Analiza fundamentalna detektorów",
 "calibration_hint":{"reliability":0.65,"expected_ttft_mins":20}}""",
 
    "Reasoner": """Zwróć WYŁĄCZNIE JSON dla Reasoner:
{"action_probs":{"BUY":0.20,"HOLD":0.50,"AVOID":0.20,"ABSTAIN":0.10},
 "uncertainty":{"epistemic":0.25,"aleatoric":0.18},
 "evidence":[{"name":"temporal_coherence","direction":"pro","strength":0.7},
             {"name":"address_recycling","direction":"con","strength":0.4},
             {"name":"pattern_consistency","direction":"neutral","strength":0.5}],
 "rationale":"Analiza temporalna i koherencja",
 "calibration_hint":{"reliability":0.7,"expected_ttft_mins":18}}""",
 
    "Voter": """Zwróć WYŁĄCZNIE JSON dla Voter:
{"action_probs":{"BUY":0.15,"HOLD":0.55,"AVOID":0.25,"ABSTAIN":0.05},
 "uncertainty":{"epistemic":0.28,"aleatoric":0.22},
 "evidence":[{"name":"performance_calibration","direction":"pro","strength":0.55},
             {"name":"statistical_weight","direction":"con","strength":0.35},
             {"name":"confidence_interval","direction":"neutral","strength":0.45}],
 "rationale":"Kalibracja wydajności i wagi statystyczne",
 "calibration_hint":{"reliability":0.75,"expected_ttft_mins":25}}""",
 
    "Debater": """Zwróć WYŁĄCZNIE JSON dla Debater:
{"action_probs":{"BUY":0.10,"HOLD":0.40,"AVOID":0.40,"ABSTAIN":0.10},
 "uncertainty":{"epistemic":0.32,"aleatoric":0.25},
 "evidence":[{"name":"counter_evidence","direction":"con","strength":0.6},
             {"name":"alternative_explanation","direction":"pro","strength":0.4},
             {"name":"risk_assessment","direction":"con","strength":0.5}],
 "rationale":"Analiza kontra-argumentów i ryzyka",
 "calibration_hint":{"reliability":0.6,"expected_ttft_mins":22}}""",
 
    "Decider": """Zwróć WYŁĄCZNIE JSON dla Decider (ostateczna decyzja detektora):
{"action_probs":{"BUY":0.05,"HOLD":0.85,"AVOID":0.05,"ABSTAIN":0.05},
 "uncertainty":{"epistemic":0.15,"aleatoric":0.12},
 "evidence":[{"name":"consensus_synthesis","direction":"neutral","strength":0.7},
             {"name":"risk_reward_ratio","direction":"con","strength":0.4},
             {"name":"market_context","direction":"neutral","strength":0.5}],
 "rationale":"Synteza wszystkich opinii i finalna decyzja",
 "calibration_hint":{"reliability":0.8,"expected_ttft_mins":30}}"""
}

REPAIRER_SYSTEM = """Napraw podany JSON tak aby był zgodny ze schematem. Zwróć tylko poprawiony JSON."""

def call_single_agent(model: str, role_name: str, compact_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call single agent with compact payload and schema repair fallback
    """
    try:
        result = chat_json_schema_single(
            model=model,
            system_prompt=AGENT_SYSTEMS[role_name],
            user_payload=compact_payload,
            schema_name=f"{role_name}Opinion",
            schema=AGENT_OP_SCHEMA,
            temperature=0.2,
            max_tokens=240
        )
        return result
        
    except Exception as e:
        print(f"[AGENT MICRO] {role_name} failed: {e}. Attempting repair...")
        
        # Try repair fallback
        try:
            repaired = repair_to_schema(
                model=model,
                repair_system=REPAIRER_SYSTEM,
                broken_payload=json.dumps({"__agent__": role_name, **compact_payload}, ensure_ascii=False),
                schema_name=f"{role_name}Opinion",
                schema=AGENT_OP_SCHEMA
            )
            return repaired
            
        except Exception as repair_error:
            print(f"[AGENT REPAIR] {role_name} repair failed: {repair_error}. Using fallback...")
            
            # Ultimate fallback - return minimal valid structure
            base_probs = {"BUY": 0.25, "HOLD": 0.50, "AVOID": 0.15, "ABSTAIN": 0.10}
            role_adjustments = {
                "Analyzer": {"BUY": 0.30, "HOLD": 0.45, "AVOID": 0.15, "ABSTAIN": 0.10},
                "Reasoner": {"BUY": 0.20, "HOLD": 0.55, "AVOID": 0.15, "ABSTAIN": 0.10},
                "Voter": {"BUY": 0.15, "HOLD": 0.60, "AVOID": 0.20, "ABSTAIN": 0.05},
                "Debater": {"BUY": 0.10, "HOLD": 0.45, "AVOID": 0.35, "ABSTAIN": 0.10},
                "Decider": {"BUY": 0.05, "HOLD": 0.80, "AVOID": 0.10, "ABSTAIN": 0.05}
            }
            
            return {
                "action_probs": role_adjustments.get(role_name, base_probs),
                "uncertainty": {"epistemic": 0.5, "aleatoric": 0.3},
                "evidence": [
                    {"name": "fallback_signal", "direction": "neutral", "strength": 0.5},
                    {"name": "error_recovery", "direction": "neutral", "strength": 0.3},
                    {"name": "minimal_data", "direction": "neutral", "strength": 0.2}
                ],
                "rationale": f"{role_name} fallback due to processing error",
                "calibration_hint": {"reliability": 0.3, "expected_ttft_mins": 30}
            }

def single_per_agent(model: str, token_id: str, compact_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process single token using 5 separate micro-calls - guarantees 'agents' section
    """
    print(f"[PER-AGENT MICRO] {token_id}: Starting 5 separate agent calls...")
    
    agents = {}
    for role in ["Analyzer", "Reasoner", "Voter", "Debater", "Decider"]:
        agent_result = call_single_agent(model, role, compact_payload)
        agents[role] = agent_result
        print(f"[PER-AGENT MICRO] {token_id}/{role}: ✅ Completed")
    
    result = {
        "token_id": token_id,
        "agents": agents
    }
    
    print(f"[PER-AGENT MICRO] {token_id}: ✅ All 5 agents completed, guaranteed 'agents' section")
    return result