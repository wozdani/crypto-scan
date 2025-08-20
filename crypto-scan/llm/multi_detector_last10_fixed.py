"""
FIXED: TRUE BATCH Multi-Agent Consensus - SINGLE API call for ALL tokens
"""
import json
import time
from typing import List, Dict, Any

def run_last10_all_detectors(items: List[Dict[str, Any]], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    TRUE BATCH Processing - SINGLE API call for ALL tokens with multi-agent consensus
    """
    if not items:
        return {"results": []}
    
    print(f"[LAST10 TRUE BATCH] Processing {len(items)} items with SINGLE API call")
    
    # Group items by token
    token_groups = {}
    for item in items:
        symbol = item["s"]
        if symbol not in token_groups:
            token_groups[symbol] = []
        token_groups[symbol].append(item)
    
    print(f"[LAST10 BATCH] Grouped {len(items)} items into {len(token_groups)} tokens")
    
    # Build comprehensive payload for ALL tokens
    batch_tokens_data = {}
    for symbol, token_items in token_groups.items():
        detector_breakdown = {}
        total_trust = 0.0
        total_liquidity = 0.0
        price = 0.0
        volume_24h = 0.0
        
        for item in token_items:
            detector = item["det"]
            features = item.get("f", {})
            
            if detector == "SE":
                detector_breakdown["stealth_engine"] = features.get("se", 0.0)
                detector_breakdown["whale_ping"] = features.get("wp", 0.0)
                detector_breakdown["dex_inflow"] = features.get("dx", 0.0)
            elif detector == "CAL":
                detector_breakdown["californium_whale"] = features.get("cal", 0.0)
            
            total_trust += features.get("tr", 0.0)
            total_liquidity += features.get("lq", 0.0)
            if features.get("price", 0.0) > 0:
                price = features.get("price", 0.0)
            if features.get("vol", 0.0) > 0:
                volume_24h = features.get("vol", 0.0)
        
        num_detectors = len(token_items)
        avg_trust = total_trust / max(num_detectors, 1)
        
        batch_tokens_data[symbol] = {
            "symbol": symbol,
            "detector_breakdown": detector_breakdown,
            "meta": {
                "price": price,
                "volume_24h": volume_24h,
                "price_change_24h": 0.02,
                "spread": 0.001,
                "market_cap": volume_24h * 50
            },
            "trust": {
                "whale_addresses": [],
                "trust_score": avg_trust,
                "dex_inflow": detector_breakdown.get("dex_inflow", 0.0)
            },
            "history": {
                "recent_pumps": [],
                "volume_pattern": "normal",
                "price_trend": "neutral"
            },
            "perf": {
                "detector_precision": {"stealth_engine": 0.7, "californium_whale": 0.65},
                "false_positive_rate": {"stealth_engine": 0.3, "californium_whale": 0.35},
                "avg_lag_mins": 20
            }
        }
    
    # CRITICAL: Make SINGLE API call for ALL tokens
    start_time = time.time()
    batch_consensus_results = _run_single_batch_consensus(batch_tokens_data)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"[LAST10 BATCH] ✅ SINGLE API call completed in {processing_time:.1f}ms for {len(token_groups)} tokens")
    
    # Convert batch results to individual token results
    results = []
    for symbol, token_items in token_groups.items():
        consensus_result = batch_consensus_results.get(symbol, {
            "final_probs": {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
            "dominant_action": "HOLD",
            "confidence": 0.4,
            "top_evidence": ["batch_processing"],
            "rationale": "Batch consensus result"
        })
        
        final_probs = consensus_result.get("final_probs", {})
        dominant_action = consensus_result.get("dominant_action", "HOLD")
        confidence = consensus_result.get("confidence", 0.0)
        
        # Convert to traditional format
        if dominant_action == "BUY" and confidence > 0.6:
            decision = "BUY"
        elif dominant_action == "AVOID" and confidence > 0.6:
            decision = "AVOID"
        else:
            decision = "HOLD"
        
        # Create results for each detector
        for item in token_items:
            detector = item["det"]
            
            # Simulate agent votes for display
            buy_prob = final_probs.get("BUY", 0.0)
            hold_prob = final_probs.get("HOLD", 0.0)
            avoid_prob = final_probs.get("AVOID", 0.0)
            
            buy_votes = max(1, int(buy_prob * 5))
            hold_votes = max(1, int(hold_prob * 5))
            avoid_votes = max(1, int(avoid_prob * 5))
            
            # Use NEW decider for proper ABSTAIN handling
            from consensus.decider_fixed import decide_and_log
            final_action, final_confidence, full_probs = decide_and_log(symbol, consensus_result)
            
            print(f"[PROBABILISTIC] {symbol}: Final consensus: {final_action} (confidence: {final_confidence:.3f})")
            print(f"[PROBABILISTIC] {symbol}: Full action_probs: {full_probs}")
            
            results.append({
                "s": symbol,
                "det": detector,
                "d": decision,
                "c": confidence,
                "cl": {"ok": 1 if decision == "BUY" else 0, "warn": 0},
                "dbg": {
                    "a": [f"consensus_{dominant_action}"],
                    "p": [f"confidence_{confidence:.2f}"],
                    "n": []
                }
            })
    
    print(f"[LAST10 BATCH] ✅ Completed with {len(results)} total results")
    return {"results": results}

def _run_single_batch_consensus(batch_tokens_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    CRITICAL: Make SINGLE API call for ALL tokens using NEW batch consensus system
    """
    if not batch_tokens_data:
        return {}
    
    print(f"[BATCH CONSENSUS] Making SINGLE API call for {len(batch_tokens_data)} tokens")
    
    from consensus.batch_runner import run_batch_consensus
    
    # Convert to new batch format
    tokens_payload = []
    for symbol, token_data in batch_tokens_data.items():
        tokens_payload.append({
            "token_id": symbol,
            "detector_breakdown": token_data["detector_breakdown"],
            "meta": token_data["meta"],
            "trust": token_data["trust"],
            "history": token_data["history"],
            "perf": token_data["perf"]
        })
    
    # Use NEW batch consensus system with degeneracy prevention
    try:
        batch_results = run_batch_consensus(tokens_payload)
        
        print(f"[BATCH CONSENSUS] ✅ NEW batch system completed successfully")
        
        # Convert to compatible format
        compatible_results = {}
        for symbol, result in batch_results.items():
            action_probs = result.get("action_probs", {})
            top_action = max(action_probs.items(), key=lambda x: x[1])[0] if action_probs else "HOLD"
            confidence = max(action_probs.values()) if action_probs else 0.4
            
            compatible_results[symbol] = {
                "final_probs": action_probs,
                "dominant_action": top_action,
                "confidence": confidence,
                "top_evidence": [ev.get("name", "unknown") for ev in result.get("evidence", [])[:3]],
                "rationale": result.get("rationale", "Batch consensus result"),
                "uncertainty_global": result.get("uncertainty", {"epistemic": 0.5, "aleatoric": 0.3})
            }
        
        return compatible_results
        
    except Exception as e:
        print(f"[BATCH CONSENSUS ERROR] NEW batch system failed: {e}")
        return _generate_fallback_results(batch_tokens_data)

def _generate_fallback_results(batch_tokens_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate fallback consensus results for all tokens"""
    fallback_results = {}
    for symbol in batch_tokens_data.keys():
        fallback_results[symbol] = {
            "final_probs": {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
            "dominant_action": "HOLD",
            "confidence": 0.4,
            "top_evidence": ["batch_fallback"],
            "rationale": "Fallback consensus due to API issues"
        }
    return fallback_results