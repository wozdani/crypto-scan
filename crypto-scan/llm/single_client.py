# llm/single_client.py
"""
Specialized client for single-token operations with schema repair
"""
import json
import os
from typing import Dict, Any
from openai import OpenAI
from .usage_logger import log_usage

def _extract_json(text: str) -> str:
    """Extract largest JSON block from text"""
    if not text:
        return "{}"
    import re
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text

def chat_json_schema_single(model: str, system_prompt: str, user_payload: Dict[str, Any], 
                          schema_name: str, schema: Dict[str, Any], 
                          temperature: float = 0.2, max_tokens: int = 420) -> Dict[str, Any]:
    """
    Single-token optimized call with strict JSON schema
    """
    client = OpenAI(timeout=12.0)  # Shorter timeout for single calls
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True
                }
            },
            max_tokens=max_tokens
        )
        
        # Log usage for cost tracking
        try:
            log_usage(resp, "SingleSchema", user_payload.get("token_id", "unknown"), model)
        except Exception as e:
            print(f"[SINGLE USAGE] Failed to log usage: {e}")
        
        raw = resp.choices[0].message.content
        if not raw:
            raise ValueError("Empty response from OpenAI API")
        
        return json.loads(_extract_json(raw))
        
    except Exception as e:
        print(f"[SINGLE SCHEMA ERROR] {e}")
        raise e

def repair_to_schema(model: str, repair_system: str, broken_payload: str,
                    schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Repair broken JSON to match schema
    """
    client = OpenAI(timeout=10.0)
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": repair_system},
                {"role": "user", "content": f"Fix this JSON to match schema: {broken_payload}"}
            ],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True
                }
            },
            max_tokens=300
        )
        
        raw = resp.choices[0].message.content
        return json.loads(_extract_json(raw))
        
    except Exception as e:
        print(f"[REPAIR ERROR] {e}")
        # Return minimal valid structure as last resort
        return {
            "action_probs": {"BUY": 0.25, "HOLD": 0.50, "AVOID": 0.15, "ABSTAIN": 0.10},
            "uncertainty": {"epistemic": 0.5, "aleatoric": 0.3},
            "evidence": [
                {"name": "data_unavailable", "direction": "neutral", "strength": 0.5},
                {"name": "repair_fallback", "direction": "neutral", "strength": 0.3},
                {"name": "minimal_signal", "direction": "neutral", "strength": 0.2}
            ],
            "rationale": "Repair fallback due to parsing error",
            "calibration_hint": {"reliability": 0.3, "expected_ttft_mins": 30}
        }