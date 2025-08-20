"""
Batch consensus validation system - prevents degeneracy and ensures proper token differentiation
"""
from typing import Dict, Any, List
import math

def validate_batch_keys(expected: List[str], got: List[str]) -> None:
    """Validate that batch response has exactly the expected token keys"""
    exp_set, got_set = set(expected), set(got)
    if exp_set != got_set:
        missing = exp_set - got_set
        extra = got_set - exp_set
        raise ValueError(f"[BATCH VALIDATION] Key mismatch - Missing: {sorted(missing)}, Extra: {sorted(extra)}")

def _action_vector(action_probs: Dict[str, float]) -> List[float]:
    """Convert action_probs to vector for comparison"""
    return [
        action_probs.get("BUY", 0.0),
        action_probs.get("HOLD", 0.0), 
        action_probs.get("AVOID", 0.0),
        action_probs.get("ABSTAIN", 0.0)
    ]

def _l2_distance(vec1: List[float], vec2: List[float]) -> float:
    """Calculate L2 distance between two vectors"""
    return sum((x - y) ** 2 for x, y in zip(vec1, vec2)) ** 0.5

def detect_degenerate_distributions(items: Dict[str, Any], similarity_threshold: float = 0.02, ratio_threshold: float = 0.8) -> bool:
    """
    Detect if ≥ratio_threshold of tokens have nearly identical action_probs distributions
    Returns True if degeneracy detected (needs fallback to per-token calls)
    """
    if not items:
        return True
    
    # Extract action probability vectors
    vectors = []
    for token_id, token_data in items.items():
        action_probs = token_data.get("action_probs", {})
        
        # Validate structure
        required_actions = ["BUY", "HOLD", "AVOID", "ABSTAIN"]
        if not all(action in action_probs for action in required_actions):
            print(f"[DEGENERACY] {token_id}: Missing required actions in action_probs")
            return True
            
        vectors.append(_action_vector(action_probs))
    
    if len(vectors) < 2:
        return False
    
    # Compare all pairs to reference (first vector)
    reference_vector = vectors[0]
    similar_count = 0
    
    for i, vector in enumerate(vectors):
        distance = _l2_distance(reference_vector, vector)
        if distance < similarity_threshold:
            similar_count += 1
            
    similarity_ratio = similar_count / len(vectors)
    is_degenerate = similarity_ratio >= ratio_threshold
    
    if is_degenerate:
        print(f"[DEGENERACY DETECTED] {similar_count}/{len(vectors)} tokens have similar distributions (ratio={similarity_ratio:.2f} ≥ {ratio_threshold})")
        print(f"[DEGENERACY] Reference vector: {reference_vector}")
        
    return is_degenerate

def calculate_batch_entropy(items: Dict[str, Any]) -> float:
    """Calculate average entropy across all tokens - low entropy indicates poor differentiation"""
    total_entropy = 0.0
    valid_count = 0
    
    for token_id, token_data in items.items():
        action_probs = token_data.get("action_probs", {})
        
        # Calculate entropy: H = -Σ p log p
        entropy = 0.0
        for prob in action_probs.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        total_entropy += entropy
        valid_count += 1
        
    avg_entropy = total_entropy / max(valid_count, 1)
    return avg_entropy

def calculate_action_variance(items: Dict[str, Any], action: str = "BUY") -> float:
    """Calculate variance of specific action probabilities across tokens"""
    probabilities = []
    
    for token_data in items.values():
        action_probs = token_data.get("action_probs", {})
        prob = action_probs.get(action, 0.0)
        probabilities.append(prob)
    
    if len(probabilities) < 2:
        return 0.0
        
    mean_prob = sum(probabilities) / len(probabilities)
    variance = sum((p - mean_prob) ** 2 for p in probabilities) / len(probabilities)
    
    return variance

def validate_batch_quality(items: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive batch quality validation with diagnostics"""
    diagnostics = {
        "is_valid": True,
        "issues": [],
        "metrics": {}
    }
    
    if not items:
        diagnostics["is_valid"] = False
        diagnostics["issues"].append("Empty batch results")
        return diagnostics
    
    # Check for degeneracy
    if detect_degenerate_distributions(items):
        diagnostics["is_valid"] = False
        diagnostics["issues"].append("Degenerate distributions detected")
    
    # Calculate quality metrics
    avg_entropy = calculate_batch_entropy(items)
    buy_variance = calculate_action_variance(items, "BUY")
    hold_variance = calculate_action_variance(items, "HOLD")
    
    diagnostics["metrics"] = {
        "avg_entropy": avg_entropy,
        "buy_variance": buy_variance,
        "hold_variance": hold_variance,
        "token_count": len(items)
    }
    
    # Quality thresholds
    if avg_entropy < 0.85:
        diagnostics["issues"].append(f"Low entropy ({avg_entropy:.2f}) - insufficient differentiation")
    
    if buy_variance < 1e-4:
        diagnostics["issues"].append(f"Low BUY variance ({buy_variance:.6f}) - potential degeneracy")
    
    # Check HOLD dominance
    hold_dominant_count = 0
    abstain_count = 0
    
    for token_data in items.values():
        action_probs = token_data.get("action_probs", {})
        top_action = max(action_probs.items(), key=lambda x: x[1])[0] if action_probs else "UNKNOWN"
        
        if top_action == "HOLD":
            hold_dominant_count += 1
        elif top_action == "ABSTAIN":
            abstain_count += 1
    
    hold_ratio = hold_dominant_count / len(items)
    if hold_ratio >= 0.9:
        diagnostics["issues"].append(f"HOLD dominance ({hold_ratio:.1%}) - potential collapse")
    
    print(f"[BATCH QUALITY] Entropy: {avg_entropy:.2f}, BUY variance: {buy_variance:.6f}, HOLD dominant: {hold_ratio:.1%}, ABSTAIN: {abstain_count}")
    
    return diagnostics