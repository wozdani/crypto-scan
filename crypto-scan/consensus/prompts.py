"""
Optimized prompts for batch consensus with timeout prevention
"""

BATCH_AGENT_SYSTEM_V2 = """Jesteś zespołem 4 agentów (Analyzer, Reasoner, Voter, Debater).

Dla KAŻDEGO tokena zwróć WYŁĄCZNIE:
- action_probs (BUY,HOLD,AVOID,ABSTAIN; suma=1.0)
- uncertainty {epistemic,aleatoric}
- evidence: DOKŁADNIE 3 pozycje (pro/con/neutral), zwięzłe nazwy
- rationale: maksymalnie 20 słów
- calibration_hint {reliability,expected_ttft_mins}

ZASADY:
- Nie używaj sztywnych progów
- Nie kopiuj rozkładów między tokenami (chyba że dane identyczne → wyższy epistemic)
- Każdy token ma unikalne action_probs
- evidence: krótkie nazwy jak "whale_ping", "volume_spike", "orderbook_thin"

Output JSON:
{
  "items": {
    "TOKEN_ID_1": {
      "action_probs": {"BUY":0.31,"HOLD":0.28,"AVOID":0.14,"ABSTAIN":0.27},
      "uncertainty": {"epistemic":0.33,"aleatoric":0.22},
      "evidence": [
        {"name":"whale_ping","direction":"pro","strength":0.62},
        {"name":"volume_spike","direction":"neutral","strength":0.41},
        {"name":"orderbook_thin","direction":"con","strength":0.38}
      ],
      "rationale": "Multi-agent consensus based on detector synergy",
      "calibration_hint": {"reliability":0.63,"expected_ttft_mins":22}
    }
  }
}"""

EMERGENCY_SINGLE_PROMPT = """Szybka analiza 1 tokena - 4 agenci consensus.

Zwróć JSON:
{
  "items": {
    "TOKEN_ID": {
      "action_probs": {"BUY":X,"HOLD":Y,"AVOID":Z,"ABSTAIN":W},
      "uncertainty": {"epistemic":A,"aleatoric":B},
      "evidence": [{"name":"signal","direction":"pro/con","strength":N}],
      "rationale": "Krótkie uzasadnienie",
      "calibration_hint": {"reliability":R,"expected_ttft_mins":T}
    }
  }
}

Maksymalnie 15 słów rationale. Fokus na najsilniejszych sygnałach."""

def get_prompt_for_context(token_count: int, is_emergency: bool = False) -> str:
    """
    Get appropriate prompt based on context
    """
    if is_emergency or token_count == 1:
        return EMERGENCY_SINGLE_PROMPT
    else:
        return BATCH_AGENT_SYSTEM_V2

def estimate_prompt_tokens(system_prompt: str, payload: dict) -> int:
    """
    Rough estimate of prompt tokens for timeout prediction
    """
    import json
    system_tokens = len(system_prompt.split()) * 1.3  # Rough conversion
    payload_tokens = len(json.dumps(payload, ensure_ascii=False)) / 4  # ~4 chars per token
    return int(system_tokens + payload_tokens)