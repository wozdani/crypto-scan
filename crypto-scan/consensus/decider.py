"""
New Decider Interface - Aggregate list of AgentOpinion objects using Bradley-Terry consensus
"""
from typing import List, Dict, Any, Tuple
import math
from typing import Union
try:
    from .contracts import AgentOpinion, FinalDecision, ActionProbs, Uncertainty
except ImportError:
    # Handle case where contracts module isn't available
    AgentOpinion = dict
    FinalDecision = dict
    ActionProbs = dict
    Uncertainty = dict

def decide_and_log(token_id: str, opinions: List[Union[AgentOpinion, Dict[str, Any]]]) -> Tuple[Dict[str, Any], str, float]:
    """
    Aggregate list of AgentOpinion objects and return consensus decision
    
    Args:
        token_id: Token identifier for logging
        opinions: List of 4 AgentOpinion objects from per-agent batch processing
        
    Returns:
        Tuple of (final_decision_dict, top_action, confidence)
        
    Raises:
        RuntimeError: If no agent opinions provided (mapping error)
    """
    if not opinions:
        raise RuntimeError(f"[DECIDER] No agent opinions for {token_id} (mapping error)")
    
    if len(opinions) != 4:
        print(f"[DECIDER WARNING] {token_id}: Expected 4 agents, got {len(opinions)}")
    
    # Aggregate using Bradley-Terry softmax with reliability weighting
    final_decision = aggregate(opinions)
    
    # Extract top action and confidence
    action_probs = final_decision.get("final_probs", {})
    if not action_probs:
        raise RuntimeError(f"[DECIDER] No final_probs in aggregated result for {token_id}")
    
    top_action, confidence = max(action_probs.items(), key=lambda kv: kv[1])
    
    # Count total evidence (handle both formats)
    evidence_count = 0
    for op in opinions:
        if hasattr(op, 'evidence'):
            evidence_count += len(op.evidence)
        else:
            evidence_count += len(op.get("evidence", []))
    
    print(f"[PROBABILISTIC] {token_id} → {top_action} | probs={action_probs} | conf={confidence:.3f} | evidence_count={evidence_count}")
    
    return final_decision, top_action, confidence

def aggregate(opinions: List[Union[AgentOpinion, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Bradley-Terry consensus aggregation with reliability weighting
    """
    if not opinions:
        raise ValueError("Cannot aggregate empty opinions list")
    
    # Initialize aggregated probabilities
    aggregated_probs = {"BUY": 0.0, "HOLD": 0.0, "AVOID": 0.0, "ABSTAIN": 0.0}
    total_weight = 0.0
    all_evidence = []
    all_rationales = []
    
    # Weight each agent's opinion by reliability and epistemic uncertainty
    for opinion in opinions:
        # Handle both Pydantic objects and dict formats
        if hasattr(opinion, 'calibration_hint'):
            # Pydantic AgentOpinion object
            reliability = opinion.calibration_hint.reliability
            epistemic = opinion.uncertainty.epistemic
            action_probs = opinion.action_probs.__dict__
            evidence_list = [e.name for e in opinion.evidence]
            rationale = opinion.rationale
        else:
            # Dict format from batch processing
            reliability = opinion.get("calibration_hint", {}).get("reliability", 0.7)
            epistemic = opinion.get("uncertainty", {}).get("epistemic", 0.5)
            action_probs = opinion.get("action_probs", {})
            evidence_list = [e.get("name", str(e)) if isinstance(e, dict) else str(e) for e in opinion.get("evidence", [])]
            rationale = opinion.get("rationale", "")
        
        # Calculate agent weight: reliability * (1 - epistemic_uncertainty)
        agent_weight = reliability * (1.0 - epistemic)
        
        # Aggregate action probabilities with weighting
        for action, prob in action_probs.items():
            if action in aggregated_probs:
                aggregated_probs[action] += prob * agent_weight
        
        total_weight += agent_weight
        
        # Collect evidence and rationales
        all_evidence.extend(evidence_list)
        all_rationales.append(rationale)
        
        print(f"[AGGREGATE] Agent weight={agent_weight:.3f} (reliability={reliability:.3f}, epistemic={epistemic:.3f})")
    
    # Normalize probabilities
    if total_weight > 0:
        for action in aggregated_probs:
            aggregated_probs[action] /= total_weight
    else:
        # Fallback to uniform distribution if no valid weights
        uniform_prob = 1.0 / len(aggregated_probs)
        aggregated_probs = {action: uniform_prob for action in aggregated_probs}
        print(f"[AGGREGATE WARNING] Zero total weight, using uniform distribution")
    
    # Calculate consensus entropy
    entropy = 0.0
    for prob in aggregated_probs.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)
    
    # Calculate weighted uncertainty (handle both formats)
    epistemic_values = []
    aleatoric_values = []
    
    for op in opinions:
        if hasattr(op, 'uncertainty'):
            epistemic_values.append(op.uncertainty.epistemic)
            aleatoric_values.append(op.uncertainty.aleatoric)
        else:
            uncertainty = op.get("uncertainty", {})
            epistemic_values.append(uncertainty.get("epistemic", 0.5))
            aleatoric_values.append(uncertainty.get("aleatoric", 0.3))
    
    avg_epistemic = sum(epistemic_values) / len(epistemic_values) if epistemic_values else 0.5
    avg_aleatoric = sum(aleatoric_values) / len(aleatoric_values) if aleatoric_values else 0.3
    
    final_decision = {
        "final_probs": aggregated_probs,
        "consensus_entropy": entropy,
        "uncertainty": {
            "epistemic": avg_epistemic,
            "aleatoric": avg_aleatoric
        },
        "evidence": all_evidence[:10],  # Top 10 evidence items
        "rationale": " | ".join(all_rationales[:3]),  # Top 3 rationales
        "agent_count": len(opinions),
        "total_evidence": len(all_evidence)
    }
    
    print(f"[AGGREGATE] Final probs: {aggregated_probs}, entropy: {entropy:.3f}, evidence: {len(all_evidence)}")
    
    return final_decision

def validate_consensus_decision(decision: Dict[str, Any], token_id: str) -> bool:
    """
    Validate consensus decision structure and probabilities
    """
    required_keys = ["final_probs", "consensus_entropy", "uncertainty", "evidence", "agent_count"]
    
    for key in required_keys:
        if key not in decision:
            print(f"[VALIDATION ERROR] {token_id}: Missing key '{key}' in consensus decision")
            return False
    
    # Validate probability sum
    probs = decision["final_probs"]
    prob_sum = sum(probs.values())
    if not (0.99 <= prob_sum <= 1.01):
        print(f"[VALIDATION ERROR] {token_id}: Invalid probability sum {prob_sum:.6f}")
        return False
    
    # Validate agent count
    if decision["agent_count"] != 4:
        print(f"[VALIDATION WARNING] {token_id}: Expected 4 agents, got {decision['agent_count']}")
    
    # Validate evidence count
    if len(decision["evidence"]) < 3:
        print(f"[VALIDATION WARNING] {token_id}: Low evidence count {len(decision['evidence'])}")
    
    return True