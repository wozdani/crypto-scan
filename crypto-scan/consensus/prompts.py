"""
Optimized prompts for batch consensus with timeout prevention
"""

# Ultra-short SINGLE prompt with few-shot example to prevent missing 'agents' section
SINGLE_AGENT_SYSTEM_V3_1 = """Jesteś zespołem 4 agentów: Analyzer, Reasoner, Voter, Debater.
Zwróć WYŁĄCZNIE JSON dla JEDNEGO tokena w postaci:
{"token_id":"<ID>","agents":{
 "Analyzer":{"action_probs":{"BUY":0.33,"HOLD":0.33,"AVOID":0.17,"ABSTAIN":0.17},"uncertainty":{"epistemic":0.3,"aleatoric":0.2},"evidence":[{"name":"whale_ping","direction":"pro","strength":0.6},{"name":"dex_inflow","direction":"pro","strength":0.5},{"name":"spread","direction":"con","strength":0.3}],"rationale":"zwięzły powód","calibration_hint":{"reliability":0.6,"expected_ttft_mins":20}},
 "Reasoner":{"action_probs":{"BUY":0.25,"HOLD":0.40,"AVOID":0.20,"ABSTAIN":0.15},"uncertainty":{"epistemic":0.25,"aleatoric":0.18},"evidence":[{"name":"volume_spike","direction":"pro","strength":0.7},{"name":"orderbook_thin","direction":"con","strength":0.4},{"name":"trust_score","direction":"neutral","strength":0.5}],"rationale":"analiza czasowa","calibration_hint":{"reliability":0.65,"expected_ttft_mins":18}},
 "Voter":{"action_probs":{"BUY":0.20,"HOLD":0.45,"AVOID":0.25,"ABSTAIN":0.10},"uncertainty":{"epistemic":0.28,"aleatoric":0.22},"evidence":[{"name":"liquidity","direction":"pro","strength":0.55},{"name":"momentum","direction":"con","strength":0.35},{"name":"risk_level","direction":"con","strength":0.45}],"rationale":"ocena ryzyka","calibration_hint":{"reliability":0.7,"expected_ttft_mins":25}},
 "Debater":{"action_probs":{"BUY":0.15,"HOLD":0.35,"AVOID":0.35,"ABSTAIN":0.15},"uncertainty":{"epistemic":0.32,"aleatoric":0.25},"evidence":[{"name":"market_sentiment","direction":"con","strength":0.6},{"name":"technical_analysis","direction":"pro","strength":0.4},{"name":"fundamentals","direction":"neutral","strength":0.5}],"rationale":"kontra-argumenty","calibration_hint":{"reliability":0.6,"expected_ttft_mins":22}}
}}
Wymagania: (1) brak sztywnych progów; (2) DOKŁADNIE 3 evidence per agent; (3) suma action_probs=1; (4) nic poza JSON."""

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

BATCH_AGENT_SYSTEM_V3 = """
Jesteś zespołem 4 agentów: Analyzer, Reasoner, Voter, Debater.
Dla KAŻDEGO tokena zwróć wyniki PER-AGENT w formacie:
{
  "items": {
    "<TOKEN_ID>": {
      "agents": {
        "Analyzer": {
          "action_probs": {"BUY": 0.0, "HOLD": 0.0, "AVOID": 0.0, "ABSTAIN": 0.0},
          "uncertainty": {"epistemic": 0.0, "aleatoric": 0.0},
          "evidence": [{"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0},
                       {"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0},
                       {"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0}],
          "rationale": "Krótkie uzasadnienie (max 20 słów)",
          "calibration_hint": {"reliability": 0.0, "expected_ttft_mins": 0}
        },
        "Reasoner": {
          "action_probs": {"BUY": 0.0, "HOLD": 0.0, "AVOID": 0.0, "ABSTAIN": 0.0},
          "uncertainty": {"epistemic": 0.0, "aleatoric": 0.0},
          "evidence": [{"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0},
                       {"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0},
                       {"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0}],
          "rationale": "Krótkie uzasadnienie (max 20 słów)",
          "calibration_hint": {"reliability": 0.0, "expected_ttft_mins": 0}
        },
        "Voter": {
          "action_probs": {"BUY": 0.0, "HOLD": 0.0, "AVOID": 0.0, "ABSTAIN": 0.0},
          "uncertainty": {"epistemic": 0.0, "aleatoric": 0.0},
          "evidence": [{"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0},
                       {"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0},
                       {"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0}],
          "rationale": "Krótkie uzasadnienie (max 20 słów)",
          "calibration_hint": {"reliability": 0.0, "expected_ttft_mins": 0}
        },
        "Debater": {
          "action_probs": {"BUY": 0.0, "HOLD": 0.0, "AVOID": 0.0, "ABSTAIN": 0.0},
          "uncertainty": {"epistemic": 0.0, "aleatoric": 0.0},
          "evidence": [{"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0},
                       {"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0},
                       {"name": "signal_name", "direction": "pro|con|neutral", "strength": 0.0}],
          "rationale": "Krótkie uzasadnienie (max 20 słów)",
          "calibration_hint": {"reliability": 0.0, "expected_ttft_mins": 0}
        }
      }
    }
  }
}

Wymagania:
- Nie używaj sztywnych progów/reguł; zwracaj rozkłady probabilistyczne (suma=1).
- Dla KAŻDEGO agenta podaj DOKŁADNIE 3 evidence (różne sygnały).
- Nie kopiuj identycznych rozkładów między różnymi tokenami/agentami.
- Odpowiedź musi być JEDNYM obiektem JSON dokładnie w tej strukturze.
- KAŻDY agent musi mieć różne action_probs bazujące na swoich zadaniach.

Nie używaj tej samej wartości action_probs dla wielu tokenów bez konkretnego uzasadnienia w rationale (token-specyficznego).

Jeśli dane są słabe, ABSTAIN jest prawidłową akcją; nie zamieniaj jej na HOLD.
"""

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
        return BATCH_AGENT_SYSTEM_V3  # Per-agent output for proper mapping

def estimate_prompt_tokens(system_prompt: str, payload: dict) -> int:
    """
    Rough estimate of prompt tokens for timeout prediction
    """
    import json
    system_tokens = len(system_prompt.split()) * 1.3  # Rough conversion
    payload_tokens = len(json.dumps(payload, ensure_ascii=False)) / 4  # ~4 chars per token
    return int(system_tokens + payload_tokens)