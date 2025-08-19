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
    Run Multi-Agent Consensus with 5 agents for detailed voting
    """
    print(f"[LAST10 MULTI-AGENT] Processing {len(items)} items with 5-agent consensus")
    
    multi_agent = MultiAgentDecisionSystem()
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
        print(f"[LAST10 MULTI-AGENT] Processing {symbol} with {len(token_items)} detectors")
        
        # Prepare detectors data for multi-agent consensus
        detectors_data = {}
        for item in token_items:
            detector = item["det"]
            features = item.get("f", {})  # Compact features are in "f" key
            
            # Extract score based on detector type from compact features
            if detector == "SE":  # StealthEngine
                score = features.get("se", 0.0)
            elif detector == "CAL":  # CaliforniumWhale  
                score = features.get("cal", 0.0)
            elif detector == "DIA":  # DiamondWhale
                score = features.get("dia", 0.0)
            else:
                score = 0.0
            
            # Build context from compact features
            context_parts = []
            for key, value in features.items():
                if key not in ["tr", "lq"] and value != 0.0:
                    context_parts.append(f"{key}:{value:.3f}")
            
            trust = features.get("tr", 0.0)
            liquidity = features.get("lq", 0.0)
            context = f"trust:{trust:.1f}, liq:${liquidity/1000:.0f}k, " + ", ".join(context_parts)
            
            detectors_data[detector] = {
                "score": score,
                "context": context,
                "trust": trust,
                "liquidity": liquidity
            }
        
        # Run multi-agent consensus for this token
        try:
            decision, consensus_data, log = multi_agent.multi_agent_consensus_all_detectors(
                detectors_data, 
                alert_threshold=0.7, 
                min_yes_detectors=2
            )
            
            # Extract detailed agent votes from consensus_data  
            agent_votes = {}
            for detector, det_data in consensus_data.items():
                if isinstance(det_data, dict) and "agents" in det_data:
                    agent_votes[detector] = det_data["agents"]
            
            # Create detailed results for each detector with agent breakdown
            for item in token_items:
                detector = item["det"]
                det_consensus = consensus_data.get(detector, {})
                
                # Extract final decision for this detector
                det_decision = det_consensus.get("decision", "HOLD")
                det_confidence = det_consensus.get("confidence", 0.0)
                
                # Extract agent voting breakdown
                agents_data = det_consensus.get("agents", {})
                buy_votes = 0
                hold_votes = 0 
                avoid_votes = 0
                agent_details = []
                
                for agent_name, agent_result in agents_data.items():
                    if isinstance(agent_result, dict):
                        agent_decision = agent_result.get("decision", "HOLD")
                        if agent_decision == "BUY":
                            buy_votes += 1
                        elif agent_decision == "AVOID":
                            avoid_votes += 1
                        else:
                            hold_votes += 1
                        agent_details.append(f"{agent_name}:{agent_decision}")
                
                # Add result with detailed agent voting
                results.append({
                    "s": symbol,
                    "det": detector,
                    "d": det_decision,
                    "c": det_confidence,
                    "cl": {"ok": 1 if det_decision == "BUY" else 0, "warn": 0},
                    "dbg": {
                        "a": agent_details,
                        "p": [f"agents_vote: {buy_votes}BUY/{hold_votes}HOLD/{avoid_votes}AVOID"],
                        "n": []
                    },
                    # Extra fields for TOP10 display
                    "agent_votes": {
                        "buy": buy_votes,
                        "hold": hold_votes, 
                        "avoid": avoid_votes
                    },
                    "agents_detail": agents_data
                })
                
                print(f"[TOP10] {symbol} - {buy_votes} BUY / {hold_votes} HOLD / {avoid_votes} AVOID - {detector}")
            
        except Exception as e:
            print(f"[LAST10 MULTI-AGENT ERROR] Failed consensus for {symbol}: {e}")
            # Fallback to basic results
            for item in token_items:
                results.append({
                    "s": symbol,
                    "det": item["det"],
                    "d": "HOLD",
                    "c": 0.5,
                    "cl": {"ok": 0, "warn": 1},
                    "dbg": {"a": [], "p": [], "n": [f"consensus_error: {str(e)}"]}
                })
    
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
        
        content = response.choices[0].message.content.strip()
        
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