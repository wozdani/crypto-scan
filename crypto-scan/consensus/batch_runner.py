"""
Adaptive batch consensus runner with degeneracy prevention and timeout handling
"""
import json
import time
from typing import Dict, Any, List, Optional
from .validators import validate_batch_keys, detect_degenerate_distributions, validate_batch_quality
from llm.llm_client import chat_json

# Adaptive batch configuration
MAX_BATCH_SIZE = 6           # Reduced from 10 to prevent timeouts
TIMEOUT_MS = 65000          # 65s timeout per chunk
MODEL = "gpt-4o"            # Use gpt-4o for consistency
FALLBACK_TEMPERATURE = 0.35  # Higher temp for per-token fallback

BATCH_AGENT_SYSTEM = """Jesteś zespołem czterech agentów (Analyzer, Reasoner, Voter, Debater) i masz ocenić *wiele tokenów naraz*.

CRITICAL REQUIREMENTS:
- Dla KAŻDEGO tokenu zwróć RÓŻNY rozkład action_probs (BUY/HOLD/AVOID/ABSTAIN; suma=1.0)
- NIE kopiuj identycznych rozkładów między różnymi tokenami
- Każdy token ma własne, unikalne evidence i rationale
- uncertainty (epistemic, aleatoric) musi być token-specyficzne
- Minimum 3 evidence items per token z name/direction/strength
- Jeśli dane są słabe → preferuj ABSTAIN zamiast domyślnego HOLD

STRUCTURE: Zwróć JEDYNY obiekt JSON:
{
  "items": {
    "TOKEN_ID_1": {
      "action_probs": {"BUY":0.31,"HOLD":0.28,"AVOID":0.14,"ABSTAIN":0.27},
      "uncertainty": {"epistemic":0.33,"aleatoric":0.22},
      "evidence":[{"name":"whale_ping","direction":"pro","strength":0.62}, {"name":"volume_spike","direction":"neutral","strength":0.41}, {"name":"orderbook_thin","direction":"con","strength":0.38}],
      "rationale":"Token-specific multi-agent reasoning explaining WHY this specific combination of detectors and market conditions leads to this action distribution",
      "calibration_hint":{"reliability":0.63,"expected_ttft_mins":22}
    },
    "TOKEN_ID_2": { ... },
    ...
  }
}

DIFFERENTIATION: Analyze each token's unique detector breakdown, market conditions, trust profile, and history. Generate distinct probability distributions that reflect genuine differences in signal strength and market context."""

def run_batch_consensus(tokens_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run adaptive batch consensus with degeneracy detection and fallback
    """
    if not tokens_payload:
        return {}
    
    print(f"[BATCH CONSENSUS] Processing {len(tokens_payload)} tokens with adaptive chunking")
    
    # Adaptive chunking to prevent timeouts
    chunks = [tokens_payload[i:i+MAX_BATCH_SIZE] for i in range(0, len(tokens_payload), MAX_BATCH_SIZE)]
    results: Dict[str, Dict[str, Any]] = {}
    
    for chunk_idx, chunk in enumerate(chunks):
        print(f"[BATCH CONSENSUS] Processing chunk {chunk_idx+1}/{len(chunks)} with {len(chunk)} tokens")
        
        chunk_results = _process_chunk_with_fallback(chunk, chunk_idx)
        results.update(chunk_results)
    
    # Final validation
    quality_report = validate_batch_quality(results)
    if not quality_report["is_valid"]:
        print(f"[BATCH QUALITY WARNING] Issues detected: {quality_report['issues']}")
    
    print(f"[BATCH CONSENSUS] ✅ Completed processing {len(results)} tokens")
    return results

def _process_chunk_with_fallback(chunk: List[Dict[str, Any]], chunk_idx: int) -> Dict[str, Any]:
    """Process chunk with automatic fallback on degeneracy or timeout"""
    
    payload = {
        "tokens": chunk,
        "chunk_info": {
            "chunk_id": chunk_idx,
            "token_count": len(chunk),
            "timestamp": time.time()
        }
    }
    
    start_time = time.time()
    
    try:
        # Primary batch attempt
        response = chat_json(
            model=MODEL,
            system_prompt=BATCH_AGENT_SYSTEM,
            user_payload=payload,
            agent_name="BatchConsensus",
            token=f"CHUNK_{chunk_idx}_{len(chunk)}",
            temperature=0.2
        )
        
        processing_time = (time.time() - start_time) * 1000
        print(f"[BATCH CONSENSUS] Chunk {chunk_idx} completed in {processing_time:.1f}ms")
        
        # Validate response structure
        items = response.get("items", {})
        expected_tokens = [token["token_id"] for token in chunk]
        
        validate_batch_keys(expected_tokens, list(items.keys()))
        
        # Check for degeneracy
        if detect_degenerate_distributions(items):
            print(f"[BATCH CONSENSUS] Chunk {chunk_idx}: Degeneracy detected → fallback to per-token calls")
            return _fallback_per_token_processing(chunk, chunk_idx)
        
        print(f"[BATCH CONSENSUS] Chunk {chunk_idx}: ✅ Quality validation passed")
        return items
        
    except Exception as e:
        print(f"[BATCH CONSENSUS ERROR] Chunk {chunk_idx} failed: {e}")
        print(f"[BATCH CONSENSUS] Falling back to per-token processing for chunk {chunk_idx}")
        return _fallback_per_token_processing(chunk, chunk_idx)

def _fallback_per_token_processing(chunk: List[Dict[str, Any]], chunk_idx: int) -> Dict[str, Any]:
    """Fallback to individual token processing when batch fails"""
    results = {}
    
    print(f"[FALLBACK] Processing {len(chunk)} tokens individually")
    
    for token_idx, token in enumerate(chunk):
        try:
            single_payload = {
                "tokens": [token],
                "processing_mode": "individual_fallback",
                "original_chunk": chunk_idx
            }
            
            response = chat_json(
                model=MODEL,
                system_prompt=BATCH_AGENT_SYSTEM,
                user_payload=single_payload,
                agent_name="FallbackConsensus",
                token=f"FALLBACK_{chunk_idx}_{token_idx}",
                temperature=FALLBACK_TEMPERATURE  # Slightly higher temp for diversity
            )
            
            items = response.get("items", {})
            token_id = token["token_id"]
            
            if token_id in items:
                results[token_id] = items[token_id]
                print(f"[FALLBACK] ✅ {token_id} processed successfully")
            else:
                print(f"[FALLBACK] ⚠️ {token_id} missing from response, using fallback result")
                results[token_id] = _generate_fallback_result(token)
                
        except Exception as e:
            print(f"[FALLBACK ERROR] {token['token_id']}: {e}")
            results[token["token_id"]] = _generate_fallback_result(token)
    
    return results

def _generate_fallback_result(token: Dict[str, Any]) -> Dict[str, Any]:
    """Generate minimal fallback result for failed token processing"""
    return {
        "action_probs": {"BUY": 0.15, "HOLD": 0.45, "AVOID": 0.25, "ABSTAIN": 0.15},
        "uncertainty": {"epistemic": 0.8, "aleatoric": 0.6},
        "evidence": [
            {"name": "processing_error", "direction": "con", "strength": 0.8},
            {"name": "fallback_mode", "direction": "neutral", "strength": 0.5},
            {"name": "insufficient_data", "direction": "con", "strength": 0.7}
        ],
        "rationale": f"Fallback result for {token['token_id']} due to processing failure",
        "calibration_hint": {"reliability": 0.2, "expected_ttft_mins": 999}
    }

def _build_evidence_pack(token: Dict[str, Any]) -> List[str]:
    """Build token-specific evidence pack to help LLM differentiate"""
    evidence_pack = []
    
    detector_breakdown = token.get("detector_breakdown", {})
    meta = token.get("meta", {})
    trust = token.get("trust", {})
    
    # Add specific detector signals
    if detector_breakdown.get("whale_ping", 0) > 0.1:
        evidence_pack.append(f"whale_ping: {detector_breakdown['whale_ping']:.3f}")
    
    if detector_breakdown.get("dex_inflow", 0) > 0.1:
        evidence_pack.append(f"dex_inflow: {detector_breakdown['dex_inflow']:.3f}")
    
    # Add market context
    if meta.get("volume_24h", 0) > 0:
        evidence_pack.append(f"volume_24h: ${meta['volume_24h']:,.0f}")
    
    if trust.get("trust_score", 0) > 0:
        evidence_pack.append(f"trust_score: {trust['trust_score']:.3f}")
    
    return evidence_pack[:5]  # Limit to 5 most relevant items