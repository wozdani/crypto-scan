"""
Pydantic contracts for consensus system with strict validation
"""
from typing import Dict, List, Any
from pydantic import BaseModel, Field, validator

class Evidence(BaseModel):
    name: str
    direction: str = Field(..., regex="^(pro|con|neutral)$")
    strength: float = Field(..., ge=0.0, le=1.0)

class Uncertainty(BaseModel):
    epistemic: float = Field(..., ge=0.0, le=1.0)
    aleatoric: float = Field(..., ge=0.0, le=1.0)

class ActionProbs(BaseModel):
    BUY: float = Field(..., ge=0.0, le=1.0)
    HOLD: float = Field(..., ge=0.0, le=1.0)
    AVOID: float = Field(..., ge=0.0, le=1.0)
    ABSTAIN: float = Field(..., ge=0.0, le=1.0)
    
    @validator('*', pre=False, always=True)
    def check_sum_equals_one(cls, v, values):
        if len(values) == 4:  # All fields validated
            total = sum(values.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"action_probs must sum to 1.0, got {total:.6f}")
        return v

class CalibrationHint(BaseModel):
    reliability: float = Field(..., ge=0.0, le=1.0)
    expected_ttft_mins: int = Field(..., ge=0)

class AgentOpinion(BaseModel):
    """
    Strict Pydantic model for individual agent opinions
    """
    action_probs: ActionProbs
    uncertainty: Uncertainty
    evidence: List[Evidence] = Field(..., min_items=3, max_items=5)
    rationale: str = Field(..., max_length=200)
    calibration_hint: CalibrationHint
    
    @validator('evidence')
    def validate_evidence_count(cls, v):
        if len(v) < 3:
            raise ValueError(f"Agent must provide at least 3 evidence items, got {len(v)}")
        return v

class TokenConsensus(BaseModel):
    """
    Complete consensus result for a token with all agent opinions
    """
    token_id: str
    agent_opinions: List[AgentOpinion] = Field(..., min_items=4, max_items=4)
    evidence_count: int = Field(..., ge=12)  # 4 agents × 3 evidence minimum
    source: str
    
    @validator('agent_opinions')
    def validate_four_agents(cls, v):
        if len(v) != 4:
            raise ValueError(f"Must have exactly 4 agent opinions, got {len(v)}")
        return v
    
    @validator('evidence_count')
    def validate_evidence_matches(cls, v, values):
        if 'agent_opinions' in values:
            actual_count = sum(len(op.evidence) for op in values['agent_opinions'])
            if v != actual_count:
                raise ValueError(f"evidence_count {v} doesn't match actual evidence {actual_count}")
        return v