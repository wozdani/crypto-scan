"""
Consensus Aggregator - Decision Aggregation with Silent Coordination
Implementuje filtrowanie głosów, liczenie margin i eskalację
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
from .consensus_contract import ConsensusContract


@dataclass
class VoteRecord:
    """Record of single agent vote"""
    agent: str
    detector: str
    vote: str  # BUY/SELL/NO_OP/ABSTAIN
    confidence: float
    is_active: bool
    weight: float = 1.0
    

@dataclass 
class AggregationResult:
    """Result of consensus aggregation"""
    decision: str  # BUY/SELL/NO_OP/ESCALATE
    buy_count: float  # Weighted count
    sell_count: float  # Weighted count
    no_op_count: float  # Weighted count
    abstain_count: int
    margin: float  # BUY - SELL
    valid_votes: int  # After filtering
    total_votes: int  # Before filtering
    avg_confidence: float
    filtered_reasons: Dict[str, int]  # Why votes were filtered
    escalated: bool
    escalation_reason: Optional[str]
    consensus_met: bool
    timestamp: str
    

class ConsensusAggregator:
    """
    Agreguje głosy agentów z filtrowaniem i margin counting
    """
    
    def __init__(self):
        self.contract = ConsensusContract
        self.agent_weights = {}  # Loaded from learning system
        self.detector_weights = {}  # Loaded from learning system
        
    def aggregate_votes(self, 
                       votes: List[VoteRecord],
                       enable_escalation: bool = True) -> AggregationResult:
        """
        Main aggregation logic with filtering and margin calculation
        
        Steps:
        1. Filter out ABSTAIN and inactive detector votes
        2. Filter out low confidence votes
        3. Calculate weighted counts
        4. Check margin and consensus conditions
        5. Escalate if needed
        """
        
        # Track filtering
        filtered_reasons = {
            "abstain": 0,
            "inactive_detector": 0,
            "low_confidence": 0,
            "no_weight": 0
        }
        
        # Step 1: Filter ABSTAIN and inactive
        active_votes = []
        for vote in votes:
            if vote.vote == "ABSTAIN":
                filtered_reasons["abstain"] += 1
                continue
            if not vote.is_active:
                filtered_reasons["inactive_detector"] += 1
                continue
            active_votes.append(vote)
        
        # Step 2: Filter low confidence
        valid_votes = []
        for vote in active_votes:
            if vote.confidence < self.contract.CONFIDENCE_FLOOR:
                filtered_reasons["low_confidence"] += 1
                continue
            valid_votes.append(vote)
        
        # Step 3: Calculate weighted counts
        buy_count = 0.0
        sell_count = 0.0
        no_op_count = 0.0
        confidence_sum = 0.0
        
        for vote in valid_votes:
            # Get weight (default 1.0 if not learned)
            weight = self._get_vote_weight(vote)
            
            if vote.vote == "BUY":
                buy_count += weight
            elif vote.vote == "SELL":
                sell_count += weight
            elif vote.vote == "NO_OP":
                no_op_count += weight
                
            confidence_sum += vote.confidence
        
        # Calculate metrics
        margin = buy_count - sell_count
        avg_confidence = confidence_sum / max(len(valid_votes), 1)
        
        # Step 4: Check consensus conditions
        consensus_met = False
        decision = "NO_OP"
        escalated = False
        escalation_reason = None
        
        if len(valid_votes) < self.contract.MIN_VALID_VOTES:
            decision = "NO_OP"
            escalation_reason = f"min_votes_not_met ({len(valid_votes)} < {self.contract.MIN_VALID_VOTES})"
            
        elif margin >= self.contract.REQUIRED_MARGIN:
            # BUY wins with sufficient margin
            consensus_met = True
            decision = "BUY"
            
        elif margin <= -self.contract.REQUIRED_MARGIN:
            # SELL wins with sufficient margin
            consensus_met = True
            decision = "SELL"
            
        elif enable_escalation and self.contract.should_escalate(int(buy_count), int(sell_count)):
            # Gap too small, escalate to Debater
            escalated = True
            decision = "ESCALATE"
            escalation_reason = f"margin_too_small ({margin:.1f} < {self.contract.REQUIRED_MARGIN})"
        
        else:
            # Default to NO_OP
            decision = "NO_OP"
            escalation_reason = f"margin_too_small ({margin:.1f} < {self.contract.REQUIRED_MARGIN})"
        
        return AggregationResult(
            decision=decision,
            buy_count=buy_count,
            sell_count=sell_count,
            no_op_count=no_op_count,
            abstain_count=filtered_reasons["abstain"],
            margin=margin,
            valid_votes=len(valid_votes),
            total_votes=len(votes),
            avg_confidence=avg_confidence,
            filtered_reasons=filtered_reasons,
            escalated=escalated,
            escalation_reason=escalation_reason,
            consensus_met=consensus_met,
            timestamp=datetime.now().isoformat()
        )
    
    def _get_vote_weight(self, vote: VoteRecord) -> float:
        """
        Get weight for vote based on agent and detector performance
        """
        # Check if we have learned weights
        agent_key = f"{vote.detector}_{vote.agent}"
        
        if agent_key in self.agent_weights:
            return self.agent_weights[agent_key]
        
        if vote.detector in self.detector_weights:
            return self.detector_weights[vote.detector]
            
        # Default weight
        return self.contract.DEFAULT_AGENT_WEIGHT
    
    def load_weights(self, weights_file: str = "cache/consensus_weights.json"):
        """
        Load learned weights from file
        """
        try:
            with open(weights_file, 'r') as f:
                data = json.load(f)
                self.agent_weights = data.get("agent_weights", {})
                self.detector_weights = data.get("detector_weights", {})
                print(f"[AGGREGATOR] Loaded {len(self.agent_weights)} agent weights, {len(self.detector_weights)} detector weights")
        except:
            print("[AGGREGATOR] No weights file found, using defaults")
    
    def format_telemetry(self, result: AggregationResult, votes: List[VoteRecord]) -> Dict[str, Any]:
        """
        Format aggregation result for telemetry/logging
        """
        return {
            "decision": result.decision,
            "consensus_met": result.consensus_met,
            "margin": result.margin,
            "counts": {
                "buy": result.buy_count,
                "sell": result.sell_count,
                "no_op": result.no_op_count,
                "abstain": result.abstain_count
            },
            "votes": {
                "valid": result.valid_votes,
                "total": result.total_votes,
                "filtered": result.filtered_reasons
            },
            "confidence": {
                "average": result.avg_confidence,
                "floor": self.contract.CONFIDENCE_FLOOR
            },
            "escalation": {
                "escalated": result.escalated,
                "reason": result.escalation_reason
            },
            "contract_params": {
                "required_margin": self.contract.REQUIRED_MARGIN,
                "min_valid_votes": self.contract.MIN_VALID_VOTES,
                "confidence_floor": self.contract.CONFIDENCE_FLOOR
            },
            "raw_votes": [
                {
                    "agent": v.agent,
                    "detector": v.detector,
                    "vote": v.vote,
                    "confidence": v.confidence,
                    "active": v.is_active,
                    "weight": v.weight
                }
                for v in votes
            ],
            "timestamp": result.timestamp
        }