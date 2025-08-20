#!/usr/bin/env python3
"""
Pydantic Contracts for Multi-Agent System
Validates JSON responses and ensures stable data flow
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Literal
from enum import Enum

# Import updated contracts
from consensus.contracts import Action, Evidence, Uncertainty, CalibrationHint, AgentOpinion, FinalDecision

class EvidenceDirection(str, Enum):
    PRO = "pro"
    CON = "con" 
    NEUTRAL = "neutral"

class ActionProbs(BaseModel):
    BUY: float = Field(..., ge=0.0, le=1.0)
    HOLD: float = Field(..., ge=0.0, le=1.0)
    AVOID: float = Field(..., ge=0.0, le=1.0)
    ABSTAIN: float = Field(..., ge=0.0, le=1.0)

    @validator('*', pre=True)
    def validate_probabilities(cls, v, values):
        """Soft normalization instead of hard validation"""
        if len(values) == 3:  # Last field being validated
            total = sum(values.values()) + v
            if not (0.95 <= total <= 1.05):  # Allow small floating point errors
                # Soft renormalization instead of raising error
                all_values = dict(values)
                all_values[cls.__fields__[list(cls.__fields__.keys())[len(values)]].name] = v
                
                if total > 0:
                    # Normalize proportionally
                    scale_factor = 1.0 / total
                    return v * scale_factor
                else:
                    # Uniform fallback
                    return 0.25
        return v

# Use AgentOpinion as primary contract (replaces AgentResponse)
class AgentResponse(AgentOpinion):
    """Legacy compatibility wrapper for AgentOpinion"""
    pass

class ConsensusResult(BaseModel):
    final_probs: ActionProbs
    dominant_action: Literal["BUY", "HOLD", "AVOID", "ABSTAIN"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    entropy: float = Field(..., ge=0.0, le=2.0)  # Max entropy for 4 actions = log(4)
    top_evidence: List[str] = Field(..., min_items=1, max_items=5)
    uncertainty_global: Uncertainty
    rationale: str = Field(..., min_length=10, max_length=300)

class TokenMetaInput(BaseModel):
    symbol: str
    price: float = Field(..., gt=0.0)
    volume_24h: float = Field(..., ge=0.0)
    spread_bps: float = Field(..., ge=0.0)
    liquidity_tier: str
    is_perp: bool
    exchange: str
    funding_apr: Optional[float] = 0.0
    oi_change: Optional[float] = 0.0
    news_flag: Optional[bool] = False

class TrustProfileInput(BaseModel):
    trusted_addresses_share: float = Field(..., ge=0.0, le=1.0)
    recurring_wallets_7d: int = Field(..., ge=0)
    smart_money_score: float = Field(..., ge=0.0, le=1.0)

class TokenHistoryInput(BaseModel):
    events_72h: List[str] = Field(default_factory=list)
    repeats_24h: int = Field(..., ge=0)
    cooldown_active: bool = False
    last_alert_outcome: str = "unknown"

class DetectorPerfInput(BaseModel):
    precision_7d: float = Field(..., ge=0.0, le=1.0)
    tp_rate: float = Field(..., ge=0.0, le=1.0)
    fp_rate: float = Field(..., ge=0.0, le=1.0)
    avg_lag_mins: float = Field(..., ge=0.0, le=300.0)

class AgentInput(BaseModel):
    detector_breakdown: Dict[str, float]
    meta: TokenMetaInput
    trust: TrustProfileInput
    history: TokenHistoryInput
    perf: DetectorPerfInput

class ReliabilityUpdate(BaseModel):
    agent_name: str
    old_reliability: float = Field(..., ge=0.1, le=1.0)
    new_reliability: float = Field(..., ge=0.1, le=1.0)
    success_score: float = Field(..., ge=0.0, le=1.0)
    alpha: float = Field(default=0.1, ge=0.01, le=0.5)
    timestamp: str

class TelemetryRecord(BaseModel):
    symbol: str
    timestamp: str
    final_probs: ActionProbs
    entropy: float
    confidence: float
    dominant_action: str
    agent_count: int
    processing_time_ms: int
    api_calls_count: int
    cost_estimate_usd: float = Field(..., ge=0.0)

# Validation helpers
def validate_agent_response_json(data: dict) -> AgentResponse:
    """Validate and parse agent response JSON with error handling"""
    try:
        return AgentResponse(**data)
    except Exception as e:
        # Return fallback response for invalid JSON
        return AgentResponse(
            action_probs=ActionProbs(BUY=0.2, HOLD=0.5, AVOID=0.2, ABSTAIN=0.1),
            uncertainty=Uncertainty(epistemic=0.8, aleatoric=0.5),
            evidence=[
                Evidence(name="validation_error", direction="neutral", strength=0.0),
                Evidence(name="fallback_response", direction="neutral", strength=0.5),
                Evidence(name="json_parse_failure", direction="con", strength=0.7)
            ],
            rationale=f"JSON validation failed: {str(e)[:100]}",
            calibration_hint=CalibrationHint(reliability=0.3, expected_ttft_mins=30)
        )

def normalize_action_probs(probs: Dict[str, float]) -> ActionProbs:
    """Normalize probabilities to sum to 1.0"""
    total = sum(probs.values())
    if total == 0:
        return ActionProbs(BUY=0.25, HOLD=0.25, AVOID=0.25, ABSTAIN=0.25)
    
    normalized = {k: v/total for k, v in probs.items()}
    return ActionProbs(
        BUY=normalized.get("BUY", 0.0),
        HOLD=normalized.get("HOLD", 0.0), 
        AVOID=normalized.get("AVOID", 0.0),
        ABSTAIN=normalized.get("ABSTAIN", 0.0)
    )