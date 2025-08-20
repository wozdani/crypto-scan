# llm/llm_client.py
import json, re, time
from typing import Dict, Any, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMJsonError(Exception): pass

def _extract_json(text: str) -> str:
    # wytnij największy blok { ... } (na wypadek prefix/suffix)
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=0.5, max=6),
       retry=retry_if_exception_type(LLMJsonError))
def chat_json(model: str, system_prompt: str, user_payload: Dict[str, Any],
              temperature: float = 0.2, response_format: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    # Użyj oficjalnego klienta; tu tylko interfejs
    from openai import OpenAI
    client = OpenAI()
    msgs = [
        {"role":"system","content": system_prompt},
        {"role":"user","content": json.dumps(user_payload, ensure_ascii=False)}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=temperature,
        response_format={"type": "json_object"} if response_format is None else response_format,
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(_extract_json(raw))
    except Exception:
        # szybka naprawa typowych artefaktów
        fixed = raw.replace("\n", " ").replace("\t"," ")
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            return json.loads(_extract_json(fixed))
        except Exception as e:
            raise LLMJsonError(f"Bad LLM JSON: {raw[:400]}...") from e