"""
Fixed consensus decider with proper ABSTAIN handling and full vector logging
"""
from typing import Dict, Any, List, Tuple
import numpy as np

def decide_and_log(token_id: str, consensus_result: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
    """
    Decide final action with proper ABSTAIN handling and full vector logging
    NO ABSTAIN→HOLD mapping! Returns actual top action.
    """
    # Extract probabilities
    final_probs = consensus_result.get("action_probs", {
        "BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1
    })
    
    # Ensure all required actions are present
    required_actions = ["BUY", "HOLD", "AVOID", "ABSTAIN"]
    for action in required_actions:
        if action not in final_probs:
            final_probs[action] = 0.0
    
    # Normalize to ensure sum = 1.0
    total_prob = sum(final_probs.values())
    if total_prob > 0:
        final_probs = {k: v/total_prob for k, v in final_probs.items()}
    
    # Find top action WITHOUT remapping ABSTAIN
    top_action = max(final_probs.items(), key=lambda kv: kv[1])[0]
    confidence = max(final_probs.values())
    
    # Calculate entropy for diagnostics
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in final_probs.values())
    
    # Log full vector - NO ABSTAIN REMAPPING
    print(f"[PROBABILISTIC] {token_id} final={top_action} probs={final_probs} conf={confidence:.3f} entropy={entropy:.3f}")
    
    # Additional diagnostics
    uncertainty = consensus_result.get("uncertainty", {"epistemic": 0.5, "aleatoric": 0.3})
    evidence = consensus_result.get("evidence", [])
    print(f"[PROBABILISTIC] {token_id} uncertainty=epi:{uncertainty.get('epistemic', 0):.3f}/ale:{uncertainty.get('aleatoric', 0):.3f} evidence_count={len(evidence)}")
    
    return top_action, confidence, final_probs

def aggregate_bradley_terry(opinions: List[Dict[str, Any]], reliability_weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Bradley-Terry aggregation with proper uncertainty quantification
    """
    if not opinions:
        return {
            "action_probs": {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
            "uncertainty": {"epistemic": 0.8, "aleatoric": 0.6},
            "evidence": [],
            "rationale": "No agent opinions provided"
        }
    
    # Default reliability weights
    if reliability_weights is None:
        reliability_weights = {
            "Analyzer": 1.0,
            "Reasoner": 0.9,
            "Voter": 0.8,
            "Debater": 0.7
        }
    
    # Aggregate action probabilities with weighted Bradley-Terry
    action_sums = {"BUY": 0.0, "HOLD": 0.0, "AVOID": 0.0, "ABSTAIN": 0.0}
    total_weight = 0.0
    
    for opinion in opinions:
        agent_name = opinion.get("agent", "Unknown")
        weight = reliability_weights.get(agent_name, 0.5)
        action_probs = opinion.get("action_probs", {})
        
        for action in action_sums.keys():
            action_sums[action] += action_probs.get(action, 0.0) * weight
        total_weight += weight
    
    # Normalize
    if total_weight > 0:
        final_probs = {k: v/total_weight for k, v in action_sums.items()}
    else:
        final_probs = {"BUY": 0.25, "HOLD": 0.25, "AVOID": 0.25, "ABSTAIN": 0.25}
    
    # Aggregate uncertainty (epistemic = disagreement, aleatoric = individual uncertainty)
    epistemic_uncertainty = calculate_epistemic_uncertainty(opinions)
    aleatoric_uncertainty = calculate_aleatoric_uncertainty(opinions)
    
    # Combine evidence
    all_evidence = []
    for opinion in opinions:
        evidence = opinion.get("evidence", [])
        all_evidence.extend(evidence)
    
    # Select top evidence (remove duplicates, sort by strength)
    unique_evidence = {}
    for ev in all_evidence:
        name = ev.get("name", "unknown")
        if name not in unique_evidence or ev.get("strength", 0) > unique_evidence[name].get("strength", 0):
            unique_evidence[name] = ev
    
    top_evidence = sorted(unique_evidence.values(), key=lambda x: x.get("strength", 0), reverse=True)[:5]
    
    return {
        "action_probs": final_probs,
        "uncertainty": {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty
        },
        "evidence": top_evidence,
        "rationale": f"Bradley-Terry aggregation of {len(opinions)} agent opinions"
    }

def calculate_epistemic_uncertainty(opinions: List[Dict[str, Any]]) -> float:
    """Calculate epistemic uncertainty (disagreement between agents)"""
    if len(opinions) < 2:
        return 0.5
    
    # Calculate variance in BUY probabilities across agents
    buy_probs = []
    for opinion in opinions:
        action_probs = opinion.get("action_probs", {})
        buy_probs.append(action_probs.get("BUY", 0.0))
    
    if not buy_probs:
        return 0.5
    
    mean_buy = sum(buy_probs) / len(buy_probs)
    variance = sum((p - mean_buy) ** 2 for p in buy_probs) / len(buy_probs)
    
    # Normalize variance to [0, 1] range
    max_variance = 0.25  # Maximum possible variance when probs are 0,0,1,1
    normalized_variance = min(variance / max_variance, 1.0)
    
    return normalized_variance

def calculate_aleatoric_uncertainty(opinions: List[Dict[str, Any]]) -> float:
    """Calculate aleatoric uncertainty (inherent uncertainty in individual agent opinions)"""
    uncertainties = []
    
    for opinion in opinions:
        # If agent provides uncertainty, use it
        if "uncertainty" in opinion:
            agent_uncertainty = opinion["uncertainty"]
            if isinstance(agent_uncertainty, dict):
                aleatoric = agent_uncertainty.get("aleatoric", 0.5)
            else:
                aleatoric = float(agent_uncertainty)
            uncertainties.append(aleatoric)
        else:
            # Calculate from action_probs entropy
            action_probs = opinion.get("action_probs", {})
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in action_probs.values())
            max_entropy = 2.0  # log2(4) for 4 actions
            normalized_entropy = entropy / max_entropy
            uncertainties.append(normalized_entropy)
    
    return sum(uncertainties) / len(uncertainties) if uncertainties else 0.5