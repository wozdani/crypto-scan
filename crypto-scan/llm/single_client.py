# llm/single_client.py
"""
Specialized client for single-token operations with schema repair
"""
import json
import os
import re
from typing import Dict, Any
from openai import OpenAI
from .usage_logger import log_usage

def _extract_json(text: str) -> str:
    """Extract and repair JSON from text"""
    if not text:
        return "{}"
    
    # First try to find complete JSON block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return text
    
    json_text = m.group(0)
    
    # Common JSON repair patterns for GPT-4o
    repairs = [
        # Fix trailing commas before closing braces/brackets
        (r',(\s*[}\]])', r'\1'),
        # Fix missing commas between object properties
        (r'(\"\s*:\s*(?:\d+\.?\d*|\"[^\"]*\"|true|false|null|\{[^}]*\}|\[[^\]]*\]))\s*(\"\s*:)', r'\1,\2'),
        # Fix missing quotes around property names
        (r'([{\s,])([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
        # Fix double quotes inside string values
        (r':\s*\"([^\"]*\"[^\"]*[^\"]*)\"\s*([,}])', r': "\1"\2'),
    ]
    
    for pattern, replacement in repairs:
        json_text = re.sub(pattern, replacement, json_text)
    
    return json_text

def chat_json_schema_single(model: str, system_prompt: str, user_payload: Dict[str, Any], 
                          schema_name: str, schema: Dict[str, Any], 
                          temperature: float = 0.05, max_tokens: int = 160) -> Dict[str, Any]:
    """
    Single-token optimized call with strict JSON schema
    """
    client = OpenAI(timeout=15.0)  # Slightly longer for complex JSON
    
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
        
        # Extract and repair JSON
        json_text = _extract_json(raw)
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"[SINGLE JSON ERROR] Parse failed: {e}")
            print(f"[SINGLE JSON ERROR] Error at position {e.pos}: {e.msg}")
            
            # Show context around error position
            if hasattr(e, 'pos') and e.pos:
                start = max(0, e.pos - 50)
                end = min(len(json_text), e.pos + 50)
                context = json_text[start:end]
                print(f"[SINGLE JSON ERROR] Context around error: ...{context}...")
            
            # Advanced repair strategies
            repaired_attempts = []
            
            # Strategy 1: Fix common comma delimiter issues
            fixed_commas = json_text
            # Fix missing commas between properties
            fixed_commas = re.sub(r'(\"\s*:\s*(?:\d+\.?\d*|\"[^\"]*\"|true|false|null))\s+(\"[^\"]*\"\s*:)', r'\1,\2', fixed_commas)
            # Fix trailing commas
            fixed_commas = re.sub(r',(\s*[}\]])', r'\1', fixed_commas)
            repaired_attempts.append(fixed_commas)
            
            # Strategy 2: Handle truncation patterns around char 1170-1240
            if "char 11" in str(e) or "char 12" in str(e) or (e.pos and 1160 <= e.pos <= 1250):
                # Find the last complete agent and properly close structure
                agent_sections = []
                current_pos = 0
                
                # Find all complete agent sections
                while True:
                    agent_start = json_text.find('"Analyzer":', current_pos)
                    if agent_start == -1:
                        agent_start = json_text.find('"Reasoner":', current_pos)
                    if agent_start == -1:
                        agent_start = json_text.find('"Voter":', current_pos)
                    if agent_start == -1:
                        agent_start = json_text.find('"Debater":', current_pos)
                    if agent_start == -1:
                        break
                    
                    # Find the end of this agent section
                    brace_count = 0
                    agent_end = agent_start
                    for i in range(agent_start, len(json_text)):
                        if json_text[i] == '{':
                            brace_count += 1
                        elif json_text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                agent_end = i + 1
                                break
                    
                    if brace_count == 0:  # Complete agent found
                        agent_sections.append((agent_start, agent_end))
                        current_pos = agent_end
                    else:
                        break
                
                if agent_sections:
                    # Take everything up to the last complete agent
                    last_agent_end = agent_sections[-1][1]
                    base_json = json_text[:last_agent_end]
                    # Complete the structure
                    repaired_attempts.append(base_json + '}}')
                
                # Alternative: find last complete calibration_hint
                last_hint = json_text.rfind('"expected_ttft_mins":')
                if last_hint > 0:
                    # Find the closing of this hint
                    end_search = json_text.find('}', last_hint)
                    if end_search > 0:
                        truncated = json_text[:end_search + 1]
                        # Add proper closing for agent and main structure
                        repaired_attempts.append(truncated + '}}}')
            
            # Strategy 3: Extract everything up to last complete structure
            last_brace = json_text.rfind('}')
            if last_brace > 0:
                repaired_attempts.append(json_text[:last_brace + 1])
            
            # Strategy 4: Remove any content after the agents section ends
            agents_end = json_text.find('}}')
            if agents_end > 0:
                repaired_attempts.append(json_text[:agents_end + 2])
            
            # Try each repair attempt
            for i, attempt in enumerate(repaired_attempts):
                try:
                    result = json.loads(attempt)
                    print(f"[SINGLE JSON REPAIR] ✅ Strategy {i+1} successful")
                    return result
                except json.JSONDecodeError as repair_error:
                    print(f"[SINGLE JSON REPAIR] Strategy {i+1} failed: {repair_error}")
                    continue
            
            # If all repairs fail, raise the original error
            print(f"[SINGLE JSON REPAIR] ❌ All repair strategies failed")
            raise e
        
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
            max_tokens=160
        )
        
        raw = resp.choices[0].message.content
        if not raw:
            raise ValueError("Empty response from repair API")
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