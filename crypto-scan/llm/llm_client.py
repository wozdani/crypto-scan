# llm/llm_client.py
import json, re, time, os
from typing import Dict, Any, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .usage_logger import log_usage

class LLMJsonError(Exception): pass

def _extract_json(text: str) -> str:
    # wytnij największy blok { ... } (na wypadek prefix/suffix)
    if not text:
        return "{}"
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text

@retry(stop=stop_after_attempt(int(os.getenv("OPENAI_MAX_RETRIES", "3"))), 
       wait=wait_exponential(multiplier=0.8, min=0.5, max=6),
       retry=retry_if_exception_type(LLMJsonError))
def chat_json(model: str, system_prompt: str, user_payload: Dict[str, Any],
              temperature: float = 0.2, response_format: Optional[Dict[str,Any]] = None,
              agent_name: str = "unknown", token: str = "unknown") -> Dict[str,Any]:
    # Użyj oficjalnego klienta; tu tylko interfejs
    from openai import OpenAI
    client = OpenAI()
    
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]
    
    # Handle response_format properly with strict JSON schema for GPT-4o
    rf = {"type": "json_object"} if response_format is None else response_format
    
    # Add strict schema enforcement if available
    if response_format and "json_schema" in response_format:
        rf = response_format
    
    timeout = int(os.getenv("OPENAI_TIMEOUT", "20"))
    max_tokens = 300 if "batch" in agent_name.lower() else 160  # Further reduced to prevent truncation
    
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=temperature,
        response_format=rf,
        timeout=timeout,
        max_tokens=max_tokens
    )
    
    # Log usage for cost tracking
    try:
        log_usage(resp, agent_name, token, model)
    except Exception as e:
        print(f"[LLM USAGE] Failed to log usage: {e}")
    
    raw = resp.choices[0].message.content
    if not raw:
        raise LLMJsonError("Empty response from OpenAI API")
    
    try:
        return json.loads(_extract_json(raw))
    except Exception:
        # szybka naprawa typowych artefaktów
        fixed = raw.replace("\n", " ").replace("\t", " ")
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            return json.loads(_extract_json(fixed))
        except Exception as e:
            raise LLMJsonError(f"Bad LLM JSON: {raw[:400]}...") from e