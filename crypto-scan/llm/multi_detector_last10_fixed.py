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
    
    # CRITICAL: Make SINGLE API call for ALL tokens with NEW async system
    start_time = time.time()
    batch_consensus_results = _run_single_batch_consensus_async(batch_tokens_data)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"[LAST10 BATCH V3] ✅ SINGLE API call completed in {processing_time:.1f}ms for {len(token_groups)} tokens")
    
    # Convert batch results to individual token results
    results = []
    for symbol, token_items in token_groups.items():
        # NO FALLBACK DICTIONARY - only process tokens that passed consensus
        if symbol not in batch_consensus_results:
            print(f"[NO FALLBACK] {symbol}: Token rejected during consensus - skipping all detectors")
            continue
            
        consensus_result = batch_consensus_results[symbol]
        
        # Extract consensus data without fallback values
        final_probs = consensus_result.get("final_probs")
        dominant_action = consensus_result.get("dominant_action")
        confidence = consensus_result.get("confidence")
        
        # Validate all required fields are present
        if not final_probs or not dominant_action or confidence is None:
            print(f"[NO FALLBACK] {symbol}: Missing consensus data - skipping token")
            continue
            
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
            
            # Use NEW decider interface for proper AgentOpinion handling
            from consensus.decider import decide_and_log, validate_consensus_decision
            
            # Extract agent opinions from consensus result
            agent_opinions = consensus_result.get("agent_opinions", [])
            if agent_opinions and len(agent_opinions) == 4:
                try:
                    final_decision, final_action, final_confidence = decide_and_log(symbol, agent_opinions)
                    
                    # Validate decision structure
                    if validate_consensus_decision(final_decision, symbol):
                        full_probs = final_decision.get("final_probs", {})
                        print(f"[PROBABILISTIC V3] {symbol}: {final_action} (conf: {final_confidence:.3f})")
                        print(f"[PROBABILISTIC V3] {symbol}: Evidence: {final_decision.get('total_evidence', 0)}")
                    else:
                        print(f"[PROBABILISTIC V3 ERROR] {symbol}: Invalid decision structure - REJECTING TOKEN")
                        continue  # Skip this token entirely - NO FALLBACK
                        
                except Exception as e:
                    print(f"[PROBABILISTIC V3 ERROR] {symbol}: Decider failed: {e} - REJECTING TOKEN")
                    continue  # Skip this token entirely - NO FALLBACK
            else:
                print(f"[PROBABILISTIC V3 ERROR] {symbol}: Invalid agent opinions count: {len(agent_opinions)} - REJECTING TOKEN")
                continue  # Skip this token entirely - NO FALLBACK
            
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

def _run_single_batch_consensus_async(batch_tokens_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    CRITICAL: Make SINGLE API call for ALL tokens using NEW async batch consensus system V3
    """
    if not batch_tokens_data:
        return {}
    
    print(f"[BATCH CONSENSUS V3] Making SINGLE API call for {len(batch_tokens_data)} tokens")
    
    import asyncio
    import concurrent.futures
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
    
    # Use NEW async batch consensus system with per-agent validation
    try:
        # Use ThreadPoolExecutor to isolate async call from existing event loop
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, run_batch_consensus(tokens_payload))
            batch_results = future.result(timeout=120)  # 2 minute timeout
            
        print(f"[BATCH SUCCESS V3] ✅ Processed {len(batch_results)} tokens with per-agent consensus")
        
        # Transform NEW per-agent batch results to expected format
        final_results = {}
        for token_id, agent_results in batch_results.items():
            agent_opinions = agent_results.get("agent_opinions", [])
            evidence_count = agent_results.get("evidence_count", 0)
            source = agent_results.get("source", "unknown")
            
            print(f"[PER-AGENT MAP] {token_id}: {len(agent_opinions)} agents, evidence={evidence_count}, source={source}")
            
            # Extract consensus probabilities from agent opinions
            if agent_opinions and evidence_count >= 12:  # Require proper per-agent evidence
                # Aggregate agent probabilities using Bradley-Terry style aggregation
                total_probs = {"BUY": 0.0, "HOLD": 0.0, "AVOID": 0.0, "ABSTAIN": 0.0}
                
                for opinion in agent_opinions:
                    action_probs = opinion.get("action_probs", {})
                    for action, prob in action_probs.items():
                        total_probs[action] += prob
                
                # Average the probabilities
                num_agents = len(agent_opinions)
                avg_probs = {action: prob / num_agents for action, prob in total_probs.items()}
                
                # Determine dominant action and confidence
                dominant_action = max(avg_probs, key=avg_probs.get)
                confidence = avg_probs[dominant_action]
                
                # Extract evidence from all agents
                all_evidence = []
                for opinion in agent_opinions:
                    agent_name = opinion.get("agent", "unknown")
                    evidence = opinion.get("evidence", [])
                    for e in evidence:
                        evidence_name = e.get("name", str(e)) if isinstance(e, dict) else str(e)
                        all_evidence.append(f"{agent_name}:{evidence_name}")
                
                final_results[token_id] = {
                    "final_probs": avg_probs,
                    "dominant_action": dominant_action,
                    "confidence": confidence,
                    "top_evidence": all_evidence[:8],  # Top 8 evidence items from all agents
                    "rationale": f"Per-agent consensus: {num_agents} agents, {evidence_count} evidence, source={source}"
                }
                
                print(f"[CONSENSUS V3] {token_id}: {dominant_action} ({confidence:.3f}) evidence_count={evidence_count}")
            else:
                # REJECT tokens without proper per-agent structure - NO FALLBACK
                print(f"[CONSENSUS V3 REJECT] {token_id}: Insufficient evidence ({evidence_count}) or agents ({len(agent_opinions)}) - REJECTED")
                continue  # Skip this token entirely
        
        return final_results
        
    except Exception as e:
        print(f"[BATCH CONSENSUS V3 ERROR] Failed batch processing: {e}")
        # NO FALLBACK TO FIXED DISTRIBUTIONS - return empty results
        print(f"[BATCH CONSENSUS V3] NO FALLBACK - returning empty results to force proper per-agent processing")
        return {}

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