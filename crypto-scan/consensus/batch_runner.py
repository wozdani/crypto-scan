"""
Load-aware batch consensus runner with timeout prevention and retry splitting
"""
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from .validators import validate_batch_keys, detect_degenerate_distributions, validate_batch_quality
from .batching import make_balanced_chunks, analyze_chunk_distribution, order_tokens_for_first_chunk
from .context_packer import pack_token_context, compress_for_emergency
from .prompts import get_prompt_for_context, estimate_prompt_tokens
from llm.llm_client import chat_json

# Load-aware configuration
MAX_BATCH_SIZE = 5           # Further reduced to 5 for timeout prevention
TIMEOUT_MS = 20000          # 20s HTTP timeout per chunk
MODEL = "gpt-4o"            # Use gpt-4o for consistency
FALLBACK_TEMPERATURE = 0.3   # Slightly higher temp for diversity
MICRO_CONCURRENCY = 3       # Parallel micro-fallbacks

class LLMTimeout(Exception):
    """Custom exception for LLM timeout situations"""
    pass

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
    Run load-aware batch consensus with timeout prevention and retry splitting
    """
    if not tokens_payload:
        return {}
    
    print(f"[BATCH CONSENSUS] Processing {len(tokens_payload)} tokens with load-aware chunking")
    
    # Pack contexts to reduce payload size
    packed_tokens = [pack_token_context(token) for token in tokens_payload]
    
    # Priority ordering - most promising tokens first
    prioritized_tokens = order_tokens_for_first_chunk(packed_tokens)
    
    # Load-aware balanced chunking
    chunks = make_balanced_chunks(prioritized_tokens, max_chunk=MAX_BATCH_SIZE)
    
    # Analyze chunk distribution
    distribution_analysis = analyze_chunk_distribution(chunks)
    print(f"[BATCH LOAD] {distribution_analysis['total_chunks']} chunks, balanced: {distribution_analysis['balanced']}")
    
    # Process chunks with timeout handling
    results: Dict[str, Dict[str, Any]] = {}
    
    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            print(f"[BATCH ASYNC] Running in existing event loop, using create_task")
            # We're in an event loop, use create_task instead of asyncio.run
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _process_all_chunks_async(chunks))
                results = future.result(timeout=120)  # 2 minute total timeout
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            print(f"[BATCH ASYNC] No event loop detected, using asyncio.run")
            results = asyncio.run(_process_all_chunks_async(chunks))
    except Exception as e:
        print(f"[BATCH CONSENSUS ERROR] Async processing failed: {e}")
        # Fallback to synchronous processing (which works perfectly)
        results = _process_chunks_sync(chunks)
    
    # Final validation
    quality_report = validate_batch_quality(results)
    if not quality_report["is_valid"]:
        print(f"[BATCH QUALITY WARNING] Issues detected: {quality_report['issues']}")
    
    print(f"[BATCH CONSENSUS] ✅ Completed processing {len(results)} tokens")
    return results

async def _process_all_chunks_async(chunks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Process all chunks asynchronously with timeout handling"""
    results = {}
    
    for chunk_idx, chunk in enumerate(chunks):
        print(f"[BATCH ASYNC] Processing chunk {chunk_idx+1}/{len(chunks)} with {len(chunk)} tokens")
        
        try:
            chunk_results = await _process_chunk_async(chunk, chunk_idx)
            results.update(chunk_results)
        except Exception as e:
            print(f"[BATCH ASYNC ERROR] Chunk {chunk_idx} failed: {e}")
            # Emergency micro-fallback
            fallback_results = await _micro_fallback_async(chunk)
            results.update(fallback_results)
    
    return results

def _process_chunks_sync(chunks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Synchronous fallback chunk processing"""
    results = {}
    
    for chunk_idx, chunk in enumerate(chunks):
        print(f"[BATCH SYNC] Processing chunk {chunk_idx+1}/{len(chunks)} with {len(chunk)} tokens")
        chunk_results = _process_chunk_with_fallback(chunk, chunk_idx)
        results.update(chunk_results)
    
    return results

async def _process_chunk_async(chunk: List[Dict[str, Any]], chunk_idx: int) -> Dict[str, Any]:
    """Process chunk asynchronously with timeout protection"""
    
    payload = {
        "tokens": chunk,
        "chunk_info": {
            "chunk_id": chunk_idx,
            "token_count": len(chunk),
            "timestamp": time.time()
        }
    }
    
    # Estimate tokens and select appropriate prompt
    prompt = get_prompt_for_context(len(chunk))
    estimated_tokens = estimate_prompt_tokens(prompt, payload)
    
    print(f"[BATCH CHUNK {chunk_idx}] Estimated tokens: {estimated_tokens}, timeout: {TIMEOUT_MS/1000}s")
    
    try:
        # Use asyncio.wait_for for timeout control
        response = await asyncio.wait_for(
            asyncio.to_thread(
                chat_json,
                model=MODEL,
                system_prompt=prompt,
                user_payload=payload,
                agent_name="BatchConsensus",
                token=f"CHUNK_{chunk_idx}_{len(chunk)}",
                temperature=0.2
            ),
            timeout=TIMEOUT_MS / 1000.0
        )
        
        # Validate response
        items = response.get("items", {})
        expected_tokens = [token["token_id"] for token in chunk]
        
        validate_batch_keys(expected_tokens, list(items.keys()))
        
        # Check for degeneracy
        if detect_degenerate_distributions(items):
            print(f"[BATCH CHUNK {chunk_idx}] Degeneracy detected → micro-fallback")
            return await _micro_fallback_async(chunk)
        
        print(f"[BATCH CHUNK {chunk_idx}] ✅ Successful processing")
        return items
        
    except asyncio.TimeoutError:
        print(f"[BATCH TIMEOUT] Chunk {chunk_idx} timeout → splitting and retrying")
        return await _retry_with_split(chunk, chunk_idx)
    except Exception as e:
        print(f"[BATCH ERROR] Chunk {chunk_idx}: {e} → micro-fallback")
        return await _micro_fallback_async(chunk)

async def _retry_with_split(chunk: List[Dict[str, Any]], chunk_idx: int) -> Dict[str, Any]:
    """Retry failed chunk by splitting into smaller pieces"""
    if len(chunk) <= 2:
        # Can't split further, use emergency single processing
        return await _emergency_single_processing(chunk)
    
    # Split chunk in half
    mid = len(chunk) // 2
    chunk_a = chunk[:mid]
    chunk_b = chunk[mid:]
    
    print(f"[BATCH SPLIT] Chunk {chunk_idx}: {len(chunk)} → {len(chunk_a)} + {len(chunk_b)}")
    
    # Process both halves
    results_a = await _process_chunk_async(chunk_a, f"{chunk_idx}a")
    results_b = await _process_chunk_async(chunk_b, f"{chunk_idx}b")
    
    # Combine results
    combined_results = {}
    combined_results.update(results_a)
    combined_results.update(results_b)
    
    return combined_results

async def _micro_fallback_async(chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Async micro-fallback: process tokens individually with concurrency control"""
    semaphore = asyncio.Semaphore(MICRO_CONCURRENCY)
    
    async def _process_single(token: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        async with semaphore:
            try:
                # Compress context for emergency processing
                compressed_token = compress_for_emergency(token)
                single_payload = {"tokens": [compressed_token]}
                
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        chat_json,
                        model=MODEL,
                        system_prompt=get_prompt_for_context(1, is_emergency=True),
                        user_payload=single_payload,
                        agent_name="MicroFallback",
                        token=f"SINGLE_{token['token_id']}",
                        temperature=FALLBACK_TEMPERATURE
                    ),
                    timeout=10.0  # Shorter timeout for single tokens
                )
                
                return response.get("items", {})
                
            except Exception as e:
                print(f"[MICRO FALLBACK ERROR] {token['token_id']}: {e}")
                return {token["token_id"]: _generate_emergency_fallback(token)}
    
    # Process all tokens concurrently
    tasks = [_process_single(token) for token in chunk]
    individual_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results
    combined_results = {}
    for result in individual_results:
        if isinstance(result, dict):
            combined_results.update(result)
        else:
            print(f"[MICRO FALLBACK] Task failed: {result}")
    
    return combined_results

async def _emergency_single_processing(chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Emergency single-token processing for minimal chunks"""
    results = {}
    
    for token in chunk:
        try:
            compressed_token = compress_for_emergency(token)
            emergency_result = _generate_emergency_fallback(compressed_token)
            results[token["token_id"]] = emergency_result
        except Exception as e:
            print(f"[EMERGENCY ERROR] {token['token_id']}: {e}")
            results[token["token_id"]] = _generate_emergency_fallback(token)
    
    return results

def _generate_emergency_fallback(token: Dict[str, Any]) -> Dict[str, Any]:
    """Generate emergency fallback result for failed token processing"""
    return {
        "action_probs": {"BUY": 0.15, "HOLD": 0.45, "AVOID": 0.25, "ABSTAIN": 0.15},
        "uncertainty": {"epistemic": 0.9, "aleatoric": 0.7},
        "evidence": [
            {"name": "processing_error", "direction": "con", "strength": 0.8},
            {"name": "timeout_fallback", "direction": "neutral", "strength": 0.6},
            {"name": "insufficient_context", "direction": "con", "strength": 0.7}
        ],
        "rationale": f"Emergency fallback for {token.get('token_id', 'unknown')} due to processing failure",
        "calibration_hint": {"reliability": 0.1, "expected_ttft_mins": 999}
    }

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