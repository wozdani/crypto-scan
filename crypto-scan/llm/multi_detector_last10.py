"""
LLM processing for last 10 tokens multi-detector analysis with 5-Agent Consensus
"""
import json
import openai
import os
from typing import List, Dict, Any

# Import Multi-Agent Consensus System
try:
    from stealth_engine.multi_agent_decision import MultiAgentDecisionSystem
    MULTI_AGENT_AVAILABLE = True
    print("[LAST10 LLM] ✅ Multi-Agent Decision System available")
except ImportError as e:
    MULTI_AGENT_AVAILABLE = False
    print(f"[LAST10 LLM] ⚠️ Multi-Agent system not available: {e}")

def build_prompt(items: List[Dict[str, Any]]) -> str:
    """
    Build minimal prompt with policy and items
    
    Args:
        items: List of compact items from last 10 tokens
    
    Returns:
        Prompt string
    """
    items_json = json.dumps(items, separators=(',', ':'))
    
    prompt = f"""Analyze crypto detector signals. Return STRICT JSON only.

POLICY:
- AVOID if tr<0.2 OR lq<50000
- BUY if (SE/WP/MMT/WCL/DIA/CAL signal OR DX strong OR OBA confirms) AND tr≥0.2 AND lq≥50000
- HOLD otherwise
- Confidence c: 0-1 based on feature strength

DETECTORS:
- SE (StealthEngine): se=score, wp=whale_ping, dx=dex_inflow, vs=volume_spike, oba=orderbook_anomaly
- CAL (CaliforniumWhale): cal=score, ai=confidence, sig=strength
- DIA (DiamondWhale): dia=score, tmp=temporal, grph=graph
- WP (whale_ping): wp=strength, rw=repeated_whale, sd=smart_money
- DX (dex_inflow): dx5=5m_inflow, dx15=15m_inflow, rx=reliability
- OBA (orderbook_anomaly): bw=bid_wall, oba=anomaly, imb=imbalance, spr=spread
- WCL (WhaleCLIP): vis=vision, pat=pattern, conf=confidence
- MMT (mastermind_tracing): mmt=score, trc=trace, net=network

ITEMS={items_json}

Return ONLY this JSON format:
{{"results":[{{"s":"SYMBOL","det":"SE|WP|DX|OBA|WCL|DIA|CAL|MMT","d":"BUY|HOLD|AVOID","c":0.0,"cl":{{"ok":0,"warn":0}},"dbg":{{"a":[],"p":[],"n":[]}}}}]}}"""
    
    return prompt

def run_last10_all_detectors(items: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Run Multi-Agent Consensus for all detector items from last 10 tokens
    
    Args:
        items: List of items to process
        model: OpenAI model to use
    
    Returns:
        Parsed response dictionary with detailed agent voting
    """
    if not items:
        return {"results": []}
    
    # Use Multi-Agent Consensus if available, fallback to simple LLM
    if MULTI_AGENT_AVAILABLE:
        return _run_multi_agent_consensus(items)
    else:
        return _run_simple_llm_fallback(items, model)

def _run_multi_agent_consensus(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run Probabilistic Multi-Agent Consensus with soft reasoning (no hard thresholds)
    """
    print(f"[LAST10 PROBABILISTIC] Processing {len(items)} items with probabilistic consensus")
    
    from stealth_engine.probabilistic_agents import (
        ProbabilisticMultiAgentSystem, TokenMeta, TrustProfile, 
        TokenHistory, DetectorPerfStats
    )
    
    prob_system = ProbabilisticMultiAgentSystem()
    results = []
    
    # Group items by token for easier processing
    token_groups = {}
    for item in items:
        symbol = item["s"]
        if symbol not in token_groups:
            token_groups[symbol] = []
        token_groups[symbol].append(item)
    
    print(f"[LAST10 MULTI-AGENT] Grouped {len(items)} items into {len(token_groups)} tokens")
    
    for symbol, token_items in token_groups.items():
        print(f"[LAST10 PROBABILISTIC] Processing {symbol} with {len(token_items)} detectors")
        
        # Prepare detector breakdown for probabilistic analysis
        detector_breakdown = {}
        total_trust = 0.0
        total_liquidity = 0.0
        price = 0.0
        volume_24h = 0.0
        
        for item in token_items:
            detector = item["det"]
            features = item.get("f", {})  # Compact features are in "f" key
            
            # Extract scores based on detector type from compact features
            if detector == "SE":  # StealthEngine
                detector_breakdown["stealth_engine"] = features.get("se", 0.0)
                detector_breakdown["whale_ping"] = features.get("wp", 0.0)
                detector_breakdown["dex_inflow"] = features.get("dx", 0.0)
                detector_breakdown["volume_spike"] = features.get("vs", 0.0)
                detector_breakdown["orderbook_anomaly"] = features.get("oba", 0.0)
            elif detector == "CAL":  # CaliforniumWhale  
                detector_breakdown["californium_whale"] = features.get("cal", 0.0)
                detector_breakdown["ai_confidence"] = features.get("ai", 0.0)
                detector_breakdown["signal_strength"] = features.get("sig", 0.0)
            elif detector == "DIA":  # DiamondWhale
                detector_breakdown["diamond_whale"] = features.get("dia", 0.0)
                detector_breakdown["temporal_score"] = features.get("tmp", 0.0)
                detector_breakdown["graph_score"] = features.get("grph", 0.0)
            
            # Accumulate meta data
            total_trust += features.get("tr", 0.0)
            total_liquidity += features.get("lq", 0.0)
            if features.get("price", 0.0) > 0:
                price = features.get("price", 0.0)
            if features.get("vol", 0.0) > 0:
                volume_24h = features.get("vol", 0.0)
        
        # Create data structures for probabilistic analysis
        num_detectors = len(token_items)
        avg_trust = total_trust / max(num_detectors, 1)
        avg_liquidity = total_liquidity / max(num_detectors, 1)
        
        meta = TokenMeta(
            price=price,
            volume_24h=volume_24h,
            spread_bps=10.0,  # Default
            liquidity_tier="medium",
            is_perp=True,
            exchange="bybit"
        )
        meta.symbol = symbol
        
        trust = TrustProfile(
            trusted_addresses_share=avg_trust,
            recurring_wallets_7d=0,  # Not available in compact features
            smart_money_score=avg_trust
        )
        
        history = TokenHistory(
            events_72h=[],
            repeats_24h=0,
            cooldown_active=False,
            last_alert_outcome="unknown"
        )
        
        perf = DetectorPerfStats(
            precision_7d=0.7,  # Default values - should be loaded from actual performance data
            tp_rate=0.6,
            fp_rate=0.3,
            avg_lag_mins=15.0
        )
        
        # Run probabilistic consensus with robust error handling
        try:
            consensus_result = prob_system.probabilistic_consensus(
                detector_breakdown, meta, trust, history, perf
            )
            
            # Validate consensus result
            if not isinstance(consensus_result, dict):
                print(f"[PROBABILISTIC WARNING] {symbol}: Invalid consensus result type, using fallback")
                consensus_result = {
                    "final_probs": {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
                    "dominant_action": "HOLD",
                    "confidence": 0.5,
                    "top_evidence": ["probabilistic_analysis"],
                    "uncertainty_global": {"epistemic": 0.5, "aleatoric": 0.3},
                    "rationale": "Fallback consensus due to parsing issues"
                }
        except Exception as consensus_error:
            print(f"[PROBABILISTIC ERROR] {symbol}: Consensus failed: {consensus_error}")
            consensus_result = {
                "final_probs": {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
                "dominant_action": "HOLD", 
                "confidence": 0.4,
                "top_evidence": ["error_recovery"],
                "uncertainty_global": {"epistemic": 0.8, "aleatoric": 0.5},
                "rationale": f"Error recovery: {str(consensus_error)[:100]}"
            }
            
            # Extract probabilistic results
            final_probs = consensus_result.get("final_probs", {})
            dominant_action = consensus_result.get("dominant_action", "HOLD")
            confidence = consensus_result.get("confidence", 0.0)
            top_evidence = consensus_result.get("top_evidence", [])
            uncertainty = consensus_result.get("uncertainty_global", {})
            rationale = consensus_result.get("rationale", "No rationale provided")
            
            # Convert probabilistic decision to traditional format for compatibility
            if dominant_action == "BUY" and confidence > 0.6:
                decision = "BUY"
            elif dominant_action == "AVOID" and confidence > 0.6:
                decision = "AVOID"
            else:
                decision = "HOLD"
            
            # Create detailed results for each detector with probabilistic breakdown
            for item in token_items:
                detector = item["det"]
                features = item.get("f", {})
                
                # Extract detector-specific probability (simplified)
                det_confidence = confidence * 0.8  # Scale down individual detector confidence
                
                # Simulate agent votes based on probabilities (for display compatibility)
                buy_prob = final_probs.get("BUY", 0.0)
                hold_prob = final_probs.get("HOLD", 0.0)
                avoid_prob = final_probs.get("AVOID", 0.0)
                
                # Convert probabilities to vote counts (5 agents)
                buy_votes = int(buy_prob * 5)
                hold_votes = int(hold_prob * 5)
                avoid_votes = int(avoid_prob * 5)
                
                # Ensure total is 5
                total_votes = buy_votes + hold_votes + avoid_votes
                if total_votes < 5:
                    hold_votes += (5 - total_votes)
                elif total_votes > 5:
                    # Reduce the largest category
                    if hold_votes >= buy_votes and hold_votes >= avoid_votes:
                        hold_votes -= (total_votes - 5)
                    elif buy_votes >= avoid_votes:
                        buy_votes -= (total_votes - 5)
                    else:
                        avoid_votes -= (total_votes - 5)
                
                # Display format: [TOP10] TOKEN - X BUY / Y HOLD / Z AVOID - DetectorName
                vote_summary = f"{buy_votes} BUY / {hold_votes} HOLD / {avoid_votes} AVOID"
                print(f"[TOP10] {symbol} - {vote_summary} - {detector}")
                
                # Add probabilistic details to output
                epistemic = uncertainty.get("epistemic", 0.0)
                aleatoric = uncertainty.get("aleatoric", 0.0)
                
                print(f"[PROBABILISTIC] {symbol}-{detector}: "
                      f"Action={dominant_action}({confidence:.2f}), "
                      f"Uncertainty=epi:{epistemic:.2f}/ale:{aleatoric:.2f}, "
                      f"Evidence={top_evidence[:2]}")
                
                results.append({
                    "symbol": symbol,
                    "detector": detector,
                    "decision": decision,
                    "confidence": confidence,
                    "buy_votes": buy_votes,
                    "hold_votes": hold_votes,
                    "avoid_votes": avoid_votes,
                    "features": item.get("f", {}),
                    "probabilistic_data": {
                        "final_probs": final_probs,
                        "dominant_action": dominant_action,
                        "uncertainty": uncertainty,
                        "top_evidence": top_evidence,
                        "rationale": rationale
                    }
                })
        
        except Exception as e:
            print(f"[LAST10 PROBABILISTIC ERROR] Failed consensus for {symbol}: {e}")
            # Fallback to basic results
            for item in token_items:
                results.append({
                    "symbol": symbol,
                    "detector": item["det"],
                    "decision": "HOLD",
                    "confidence": 0.5,
                    "buy_votes": 0,
                    "hold_votes": 5,
                    "avoid_votes": 0,
                    "features": item.get("f", {}),
                    "probabilistic_data": {
                        "final_probs": {"BUY": 0.2, "HOLD": 0.6, "AVOID": 0.1, "ABSTAIN": 0.1},
                        "dominant_action": "HOLD",
                        "uncertainty": {"epistemic": 0.8, "aleatoric": 0.5},
                        "top_evidence": ["error"],
                        "rationale": f"Analysis failed: {str(e)}"
                    }
                })
    
    print(f"[LAST10 PROBABILISTIC] Completed consensus analysis with {len(results)} results")
    return {"results": results}
    
    print(f"[LAST10 MULTI-AGENT] Completed consensus analysis with {len(results)} results")
    return {"results": results}

def _run_simple_llm_fallback(items: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """
    Fallback to simple LLM processing if Multi-Agent not available
    """
    print(f"[LAST10 FALLBACK] Using simple LLM processing for {len(items)} items")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = build_prompt(items)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a crypto signal analyzer. Return only valid JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        content = response.choices[0].message.content
        if content:
            content = content.strip()
        
        # Try to parse JSON
        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            print(f"[LAST10 FALLBACK] JSON parse error: {content[:200]}...")
            return {"results": []}
            
    except Exception as e:
        print(f"[LAST10 FALLBACK] OpenAI error: {e}")
        return {"results": []}