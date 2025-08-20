"""
Quality telemetry system to detect degenerate mapping in per-agent consensus
"""
from typing import Dict, List, Union, Any
import numpy as np

def quality_snapshot(per_agent_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analyze quality of per-agent mapping to detect degenerate patterns
    
    Args:
        per_agent_map: Dict mapping token_id to list of AgentOpinion objects
        
    Returns:
        Dict with quality metrics
    """
    
    def vec(op: Dict[str, Any]) -> np.ndarray:
        """Extract action_probs vector from opinion"""
        # Dict format from batch processing
        ap = op.get("action_probs", {})
        
        return np.array([
            ap.get("BUY", 0.0), 
            ap.get("HOLD", 0.0), 
            ap.get("AVOID", 0.0), 
            ap.get("ABSTAIN", 0.0)
        ], dtype=float)
    
    # Collect all opinion vectors
    all_vecs = []
    token_count = 0
    agent_count = 0
    
    for tk, ops in per_agent_map.items():
        if not ops:
            continue
        token_count += 1
        
        for op in ops:
            try:
                vec_data = vec(op)
                all_vecs.append(vec_data)
                agent_count += 1
            except Exception as e:
                print(f"[QUALITY ERROR] Failed to extract vector from opinion: {e}")
                continue
    
    if not all_vecs:
        print("[QUALITY] No opinions collected - cannot assess quality")
        return {
            "buy_variance": 0.0,
            "hold_dominance": 1.0,
            "tokens_processed": 0,
            "agents_total": 0,
            "degenerate_detected": True,
            "quality_status": "NO_DATA"
        }
    
    # Stack all vectors for analysis
    M = np.stack(all_vecs, axis=0)
    
    # Calculate quality metrics
    buy_var = float(np.var(M[:, 0]))  # Variance in BUY probabilities
    hold_var = float(np.var(M[:, 1]))  # Variance in HOLD probabilities
    avoid_var = float(np.var(M[:, 2]))  # Variance in AVOID probabilities
    abstain_var = float(np.var(M[:, 3]))  # Variance in ABSTAIN probabilities
    
    # HOLD dominance ratio (how often HOLD is the top action)
    hold_dom = float((M[:, 1] == M.max(axis=1)).mean())
    
    # Total variance across all actions
    total_variance = buy_var + hold_var + avoid_var + abstain_var
    
    # Detect degenerate mapping patterns
    degenerate_detected = False
    issues = []
    
    # Check for low variance (degenerate distributions)
    if buy_var < 1e-5:
        degenerate_detected = True
        issues.append("buy_wall")
    
    if total_variance < 1e-4:
        degenerate_detected = True
        issues.append("total_variance_collapse")
    
    if hold_dom > 0.95:
        degenerate_detected = True
        issues.append("hold_dominance")
    
    # Check for identical distributions
    unique_distributions = len(np.unique(M, axis=0))
    if unique_distributions == 1 and len(all_vecs) > 1:
        degenerate_detected = True
        issues.append("identical_distributions")
    
    # Determine quality status
    if degenerate_detected:
        quality_status = "DEGENERATE"
    elif total_variance < 0.01:
        quality_status = "LOW_VARIANCE"
    elif hold_dom > 0.8:
        quality_status = "HOLD_BIASED"
    else:
        quality_status = "HEALTHY"
    
    quality_metrics = {
        "buy_variance": buy_var,
        "hold_variance": hold_var,
        "avoid_variance": avoid_var,
        "abstain_variance": abstain_var,
        "total_variance": total_variance,
        "hold_dominance": hold_dom,
        "unique_distributions": unique_distributions,
        "tokens_processed": token_count,
        "agents_total": agent_count,
        "degenerate_detected": degenerate_detected,
        "quality_status": quality_status,
        "issues": issues
    }
    
    # Log quality assessment
    print(f"[QUALITY] BUY variance={buy_var:.6f} | HOLD_top_ratio={hold_dom:.2f}")
    print(f"[QUALITY] Status: {quality_status} | Issues: {issues}")
    print(f"[QUALITY] Tokens: {token_count}, Agents: {agent_count}, Unique: {unique_distributions}")
    
    if degenerate_detected:
        print(f"[QUALITY ALERT] Degenerate mapping detected: {issues}")
        print(f"[QUALITY ALERT] Total variance: {total_variance:.6f} (threshold: 0.0001)")
        print(f"[QUALITY ALERT] HOLD dominance: {hold_dom:.3f} (threshold: 0.95)")
    
    return quality_metrics

def suggest_recovery_action(quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest recovery actions based on quality metrics
    """
    if not quality_metrics.get("degenerate_detected", False):
        return {"action": "continue", "reason": "Quality metrics are healthy"}
    
    issues = quality_metrics.get("issues", [])
    recovery_actions = []
    
    if "buy_wall" in issues:
        recovery_actions.append({
            "type": "micro_fallback",
            "reason": "BUY variance below threshold",
            "action": "shorter_prompt_with_token_specific_data"
        })
    
    if "total_variance_collapse" in issues:
        recovery_actions.append({
            "type": "prompt_diversification",
            "reason": "All action probabilities identical",
            "action": "inject_more_evidence_variation"
        })
    
    if "hold_dominance" in issues:
        recovery_actions.append({
            "type": "bias_correction",
            "reason": "Excessive HOLD bias",
            "action": "emphasize_non_hold_evidence"
        })
    
    if "identical_distributions" in issues:
        recovery_actions.append({
            "type": "evidence_injection",
            "reason": "All agents producing identical results",
            "action": "add_token_specific_context"
        })
    
    return {
        "action": "recover",
        "recovery_actions": recovery_actions,
        "severity": "HIGH" if len(issues) > 2 else "MEDIUM"
    }

def detailed_distribution_analysis(per_agent_map: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Detailed analysis of probability distributions for debugging
    """
    print("[QUALITY DETAILED] Starting detailed distribution analysis...")
    
    for token_id, opinions in per_agent_map.items():
        if not opinions:
            continue
            
        print(f"[QUALITY DETAILED] Token: {token_id}")
        
        for i, opinion in enumerate(opinions):
            probs = opinion.get("action_probs", {})
            agent_name = opinion.get("agent", f'Agent_{i}')
            
            prob_str = " | ".join([f"{k}:{v:.3f}" for k, v in probs.items()])
            print(f"[QUALITY DETAILED]   {agent_name}: {prob_str}")
        
        # Check for identical opinions within token
        if len(opinions) > 1:
            first_probs = None
            identical = True
            
            for opinion in opinions:
                ap = opinion.get("action_probs", {})
                current_probs = tuple(ap.values())
                
                if first_probs is None:
                    first_probs = current_probs
                elif first_probs != current_probs:
                    identical = False
                    break
            
            if identical:
                print(f"[QUALITY DETAILED]   ⚠️  All agents have identical distributions for {token_id}")