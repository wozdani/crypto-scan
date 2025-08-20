# consensus/coerce.py
"""
Local coercion and shape fixing for consensus responses
"""
from typing import Dict, Any

def coerce_single_shape(response: Dict[str, Any], expected_token_id: str) -> Dict[str, Any]:
    """
    Coerce single response to expected shape with proper token_id and agents structure
    """
    # Ensure token_id is present and correct
    if "token_id" not in response:
        response["token_id"] = expected_token_id
    elif response["token_id"] != expected_token_id:
        response["token_id"] = expected_token_id
    
    # Ensure agents section exists
    if "agents" not in response:
        print(f"[COERCE] {expected_token_id}: Missing 'agents' section - cannot coerce")
        raise ValueError(f"Missing required 'agents' section for {expected_token_id}")
    
    agents = response["agents"]
    required_agents = ["Analyzer", "Reasoner", "Voter", "Debater"]
    
    # Validate all agents are present
    for agent_name in required_agents:
        if agent_name not in agents:
            print(f"[COERCE] {expected_token_id}: Missing agent '{agent_name}'")
            raise ValueError(f"Missing required agent '{agent_name}' for {expected_token_id}")
        
        agent_data = agents[agent_name]
        
        # Validate action_probs sum to 1.0
        if "action_probs" in agent_data:
            probs = agent_data["action_probs"]
            total = sum(probs.values()) if probs else 0
            
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                print(f"[COERCE] {expected_token_id}/{agent_name}: action_probs sum={total:.3f}, normalizing to 1.0")
                # Normalize probabilities
                if total > 0:
                    for key in probs:
                        probs[key] = probs[key] / total
                else:
                    # Default probabilities if all zero
                    probs.update({"BUY": 0.25, "HOLD": 0.50, "AVOID": 0.15, "ABSTAIN": 0.10})
        
        # Validate evidence has exactly 3 items
        if "evidence" in agent_data:
            evidence = agent_data["evidence"]
            if len(evidence) != 3:
                print(f"[COERCE] {expected_token_id}/{agent_name}: evidence count={len(evidence)}, expected=3")
                # Truncate or pad to exactly 3 items
                if len(evidence) > 3:
                    agent_data["evidence"] = evidence[:3]
                elif len(evidence) < 3:
                    # Pad with neutral evidence
                    while len(evidence) < 3:
                        evidence.append({
                            "name": f"padding_{len(evidence)}",
                            "direction": "neutral",
                            "strength": 0.3
                        })
    
    print(f"[COERCE] {expected_token_id}: Shape validation passed")
    return response

def ensure_batch_shape(batch_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure batch response has proper structure with items and agents
    """
    if "items" not in batch_response:
        print("[COERCE BATCH] Missing 'items' section")
        raise ValueError("Missing required 'items' section in batch response")
    
    items = batch_response["items"]
    
    for token_id, token_data in items.items():
        if "agents" not in token_data:
            print(f"[COERCE BATCH] {token_id}: Missing 'agents' section")
            raise ValueError(f"Missing required 'agents' section for {token_id}")
        
        # Apply single-token coercion to each token
        try:
            coerced_token = coerce_single_shape(
                {"token_id": token_id, "agents": token_data["agents"]}, 
                token_id
            )
            items[token_id]["agents"] = coerced_token["agents"]
        except Exception as e:
            print(f"[COERCE BATCH] {token_id}: Coercion failed: {e}")
            raise e
    
    print(f"[COERCE BATCH] Validated {len(items)} tokens")
    return batch_response