# consensus/contracts.py
from pydantic import BaseModel, Field, validator
from typing import Literal, Dict, List

Action = Literal["BUY","HOLD","AVOID","ABSTAIN"]

class Evidence(BaseModel):
    name: str
    direction: Literal["pro","con","neutral"]
    strength: float = Field(..., ge=0, le=1)

class Uncertainty(BaseModel):
    epistemic: float = Field(..., ge=0, le=1)
    aleatoric: float = Field(..., ge=0, le=1)

class CalibrationHint(BaseModel):
    reliability: float = Field(0.6, ge=0, le=1)
    expected_ttft_mins: float = Field(30, ge=0, le=1440)

class AgentOpinion(BaseModel):
    action_probs: Dict[Action, float]
    uncertainty: Uncertainty
    evidence: List[Evidence]
    rationale: str
    calibration_hint: CalibrationHint = Field(default_factory=CalibrationHint)

    @validator("action_probs")
    def probs_sum_to_1(cls, v):
        s = sum(v.values())
        if not (0.999 <= s <= 1.001):
            # miękka renormalizacja zamiast błędu twardego
            if s > 0:
                for k in v: v[k] = v[k] / s
            else:
                # fallback równomierny
                for k in ["BUY","HOLD","AVOID","ABSTAIN"]: v[k] = 0.25
        return v

class FinalDecision(BaseModel):
    final_probs: Dict[Action, float]
    uncertainty_global: Uncertainty
    top_evidence: List[str] = []
    rationale: str = ""