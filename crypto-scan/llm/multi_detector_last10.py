"""
LLM processing for last 10 tokens multi-detector analysis
"""
import json
import openai
import os
from typing import List, Dict, Any

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
    Run single LLM call for all detector items from last 10 tokens
    
    Args:
        items: List of items to process
        model: OpenAI model to use
    
    Returns:
        Parsed response dictionary
    """
    if not items:
        return {"results": []}
    
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
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000  # Keep low for efficiency
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        if response_text is None:
            return {"results": []}
        result = json.loads(response_text)
        
        # Validate structure
        if "results" not in result:
            result = {"results": []}
        
        return result
        
    except Exception as e:
        print(f"[LLM ERROR] multi_detector_last10 failed: {e}")
        # Return empty results on error
        return {"results": []}