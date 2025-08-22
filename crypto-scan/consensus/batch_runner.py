"""
Load-aware batch consensus runner with timeout prevention and retry splitting
"""
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from .validators import validate_batch_keys, detect_degenerate_distributions, validate_batch_quality, ensure_agents_complete, ensure_agents_or_per_agent
from .batching import make_balanced_chunks, analyze_chunk_distribution, order_tokens_for_first_chunk
from .context_packer import pack_token_context, compress_for_emergency
from .prompts import get_prompt_for_context, estimate_prompt_tokens, BATCH_AGENT_SYSTEM_V3, SINGLE_AGENT_SYSTEM_V3_1
from .schemas import SINGLE_SCHEMA, BATCH_SCHEMA
from .coerce import coerce_single_shape, ensure_batch_shape
from .per_agent_single import single_per_agent
from llm.llm_client import chat_json
from llm.single_client import chat_json_schema_single

# Load-aware configuration (Hotfix v3.1)
MAX_BATCH_SIZE = 2           # Minimal batch size for maximum reliability
TIMEOUT_MS = 10000          # 10s HTTP timeout per chunk  
MAX_CHUNK_RETRIES = 0       # No retries, immediate micro-fallback
MICRO_CONCURRENCY = 3       # Allow some concurrency but limited

# Model configuration with environment variable control
import os
MODEL = os.getenv("OPENAI_CONSENSUS_MODEL", "gpt-4o")  # Default to gpt-4o
MODEL_BATCH = os.getenv("OPENAI_CONSENSUS_MODEL_BATCH", MODEL)  # Separate batch model option
MODEL_SINGLE = os.getenv("OPENAI_CONSENSUS_MODEL_SINGLE", MODEL_BATCH)  # Single call fallback model

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

async def run_batch_consensus(tokens_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    NEW: Per-agent batch consensus with zero fallbacks to fixed distributions
    """
    if not tokens_payload:
        return {}
    
    print(f"[BATCH CONSENSUS V3] Processing {len(tokens_payload)} tokens with per-agent parsing")
    
    # Pack contexts to reduce payload size
    packed_tokens = [pack_token_context(token) for token in tokens_payload]
    
    # Priority ordering - most promising tokens first
    prioritized_tokens = order_tokens_for_first_chunk(packed_tokens)
    
    # Load-aware balanced chunking (max 5 for timeout prevention)
    chunks = make_balanced_chunks(prioritized_tokens, max_chunk=MAX_BATCH_SIZE)
    
    # Analyze chunk distribution
    distribution_analysis = analyze_chunk_distribution(chunks)
    print(f"[BATCH LOAD V3] {distribution_analysis['total_chunks']} chunks, balanced: {distribution_analysis['balanced']}")
    
    # Process chunks with per-agent parsing
    results: Dict[str, Dict[str, Any]] = {}
    
    for chunk_idx, chunk in enumerate(chunks):
        token_ids = [token["token_id"] for token in chunk]
        payload = {"tokens": chunk}
        
        try:
            print(f"[BATCH V3] Processing chunk {chunk_idx+1}/{len(chunks)} with {len(chunk)} tokens")
            batch_output = await _call_with_retry(
                payload, 
                tag=f"CHUNK_{chunk_idx}_{len(chunk)}", 
                system_prompt=get_prompt_for_context(len(chunk))  # Uses BATCH_AGENT_SYSTEM_V3
            )
            
            # Parse per-agent format with strict validation
            per_agent_results = parse_batch_per_agent(batch_output, expected_token_ids=token_ids)
            results.update(per_agent_results)
            
            print(f"[BATCH V3] ✅ Chunk {chunk_idx} completed with {len(per_agent_results)} validated tokens")
            
        except Exception as e:
            print(f"[BATCH V3.1 ERROR] Chunk {chunk_idx} failed: {e}")
            print(f"[BATCH V3.1] Switching to micro-fallback with per-agent guarantee...")
            
            # HOTFIX v3.1: Use new micro-fallback with per-agent guarantee
            try:
                sub_payloads = [{"tokens": [token]} for token in chunk]
                micro_results = await micro_fallback_per_agent(sub_payloads)
                
                # Convert micro results to expected format for aggregation
                for token_id, micro_data in micro_results.items():
                    # Find the original token data for agents section
                    for token in chunk:
                        if token["token_id"] == token_id:
                            # Create fake single result for parsing
                            single_result = await call_single(token, f"MICRO_{token_id}")
                            if single_result and "agents" in single_result:
                                parsed_result = parse_batch_per_agent(
                                    {"items": {token_id: {"agents": single_result["agents"]}}},
                                    expected_token_ids=[token_id]
                                )
                                results.update(parsed_result)
                                print(f"[MICRO V3.1] ✅ {token_id} processed with per-agent guarantee")
                            break
                
            except Exception as micro_error:
                print(f"[MICRO V3.1 ERROR] Micro-fallback failed: {micro_error}")
                # No hard fallbacks - let tokens be rejected
    
    print(f"[BATCH CONSENSUS V3] ✅ Completed {len(results)} tokens with per-agent validation")
    
    # Quality telemetry check for degenerate mapping
    try:
        from .quality import quality_snapshot, suggest_recovery_action
        
        # Prepare per-agent map for quality analysis
        per_agent_map = {}
        for token_id, result in results.items():
            agent_opinions = result.get("agent_opinions", [])
            if agent_opinions:
                per_agent_map[token_id] = agent_opinions
        
        # Run quality assessment
        quality_metrics = quality_snapshot(per_agent_map)
        
        # Check for degenerate patterns
        if quality_metrics.get("degenerate_detected", False):
            recovery_suggestion = suggest_recovery_action(quality_metrics)
            print(f"[QUALITY ALERT] Degenerate mapping detected - recovery needed")
            print(f"[QUALITY ALERT] Severity: {recovery_suggestion.get('severity', 'UNKNOWN')}")
            
            # Log detailed issues for debugging
            issues = quality_metrics.get("issues", [])
            for issue in issues:
                print(f"[QUALITY ISSUE] {issue}")
    except Exception as e:
        print(f"[QUALITY ERROR] Quality check failed: {e}")
    
    return results

def parse_batch_per_agent(batch_output: Dict[str, Any], expected_token_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Parse batch output with per-agent validation using Pydantic contracts
    """
    items = batch_output.get("items", {})
    
    # Validate expected tokens are present
    validate_batch_keys(expected_token_ids, list(items.keys()))
    
    # Ensure complete 4-agent structure with proper evidence
    ensure_agents_complete(items)
    
    # Map to AgentOpinion format
    opinions_by_token = {}
    for token_id, token_payload in items.items():
        agents_data = token_payload.get("agents", {})
        agent_opinions = []
        
        for agent_name in ["Analyzer", "Reasoner", "Voter", "Debater"]:
            if agent_name in agents_data:
                agent_data = agents_data[agent_name]
                # Convert to AgentOpinion format for strict validation
                opinion = {
                    "agent": agent_name,
                    "action_probs": agent_data.get("action_probs", {}),
                    "uncertainty": agent_data.get("uncertainty", {}),
                    "evidence": agent_data.get("evidence", []),
                    "rationale": agent_data.get("rationale", ""),
                    "calibration_hint": agent_data.get("calibration_hint", {})
                }
                agent_opinions.append(opinion)
                print(f"[PARSE AGENT] {token_id}/{agent_name}: evidence_count={len(opinion['evidence'])}")
        
        total_evidence = sum(len(op["evidence"]) for op in agent_opinions)
        opinions_by_token[token_id] = {
            "agent_opinions": agent_opinions,
            "evidence_count": total_evidence,
            "source": "batch_per_agent_validated"
        }
        print(f"[PARSE BATCH] {token_id}: Validated {len(agent_opinions)} agents, evidence_total={total_evidence}")
    
    return opinions_by_token

def _map_batch_results_to_agent_format(batch_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map batch results to per-agent format for Decider (legacy compatibility)
    """
    mapped_results = {}
    
    for token_id, token_result in batch_results.items():
        if "agents" in token_result:
            # Per-agent format from BATCH_AGENT_SYSTEM_V3
            agents_data = token_result["agents"]
            agent_opinions = []
            
            for agent_name in ["Analyzer", "Reasoner", "Voter", "Debater"]:
                if agent_name in agents_data:
                    agent_data = agents_data[agent_name]
                    agent_opinions.append({
                        "agent": agent_name,
                        "action_probs": agent_data.get("action_probs", {}),
                        "uncertainty": agent_data.get("uncertainty", {}),
                        "evidence": agent_data.get("evidence", []),
                        "rationale": agent_data.get("rationale", ""),
                        "calibration_hint": agent_data.get("calibration_hint", {})
                    })
                    print(f"[AGENT MAP] {token_id}/{agent_name}: evidence_count={len(agent_data.get('evidence', []))}")
            
            mapped_results[token_id] = {
                "agent_opinions": agent_opinions,
                "evidence_count": sum(len(op.get("evidence", [])) for op in agent_opinions),
                "source": "batch_per_agent"
            }
            print(f"[AGENT MAP] {token_id}: Mapped {len(agent_opinions)} agents, total_evidence={mapped_results[token_id]['evidence_count']}")
            
        else:
            # Legacy single result format - wrap as single opinion
            mapped_results[token_id] = {
                "agent_opinions": [token_result],
                "evidence_count": len(token_result.get("evidence", [])),
                "source": "batch_legacy"  
            }
            print(f"[AGENT MAP] {token_id}: Legacy format, evidence_count={mapped_results[token_id]['evidence_count']}")
    
    return mapped_results

# ensure_agents_or_per_agent moved to validators.py

async def call_single(item: Dict[str, Any], tag: str) -> Dict[str, Any]:
    """
    HOTFIX v3.1: Single token processing with per-agent micro fallback
    """
    token_id = item["token_id"]
    compact = item  # Assume pack_token_context already reduced payload
    
    print(f"[SINGLE V3.1] {token_id}: Attempting single call with strict schema...")
    
    # 1) Try SINGLE with json_schema(strict) + few-shot example
    try:
        result = chat_json_schema_single(
            model=MODEL_SINGLE,
            system_prompt=SINGLE_AGENT_SYSTEM_V3_1,
            user_payload={"token_id": token_id, **compact},
            schema_name="SingleAgentsResponse",
            schema=SINGLE_SCHEMA,
            temperature=0.2,
            max_tokens=160  # Further reduced from 180 to prevent truncation at char 409
        )
        
        # Coerce and validate
        result = coerce_single_shape(result, token_id)
        ensure_agents_complete({token_id: {"agents": result["agents"]}})
        
        print(f"[SINGLE V3.1] {token_id}: ✅ Single call succeeded")
        return result
        
    except Exception as e:
        print(f"[SINGLE V3.1] {token_id}: Single call failed: {e}")
        print(f"[SINGLE V3.1] {token_id}: Switching to per-agent micro calls...")
    
    # 2) Per-agent micro (4 separate calls) - guarantees 'agents' section
    try:
        result = single_per_agent(MODEL_SINGLE, token_id, compact)
        ensure_agents_complete({token_id: {"agents": result["agents"]}})
        
        print(f"[SINGLE V3.1] {token_id}: ✅ Per-agent micro succeeded")
        return result
        
    except Exception as micro_error:
        print(f"[SINGLE V3.1] {token_id}: Per-agent micro failed: {micro_error}")
        raise micro_error

async def micro_fallback_per_agent(sub_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process tokens individually with limited concurrency to prevent timeouts
    """
    print(f"[MICRO FALLBACK V3.1] Processing {len(sub_payloads)} tokens with concurrency={MICRO_CONCURRENCY}")
    
    sem = asyncio.Semaphore(MICRO_CONCURRENCY)
    
    async def _process_one(payload: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            token = payload["tokens"][0]
            return await call_single(token, tag=f"SINGLE_{token['token_id']}")
    
    results = await asyncio.gather(*[_process_one(p) for p in sub_payloads], return_exceptions=True)
    
    # Convert results to expected format
    final_results = {}
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            token_id = sub_payloads[i]["tokens"][0]["token_id"]
            print(f"[MICRO FALLBACK V3.1] {token_id}: Failed with exception: {result}")
            continue
        
        if isinstance(result, dict) and "token_id" in result:
            token_id = result["token_id"]
            final_results[token_id] = {
                "agent_opinions": [],  # Will be populated by parse_batch_per_agent
                "source": "micro_fallback_v3_1"
            }
    
    print(f"[MICRO FALLBACK V3.1] ✅ Completed {len(final_results)}/{len(sub_payloads)} tokens")
    return final_results

async def _call_with_retry(payload: Dict[str, Any], tag: str, system_prompt: str, max_retries: int = MAX_CHUNK_RETRIES) -> Dict[str, Any]:
    """
    Call LLM with retry logic and proper JSON response format
    """
    for attempt in range(max_retries):
        try:
            # Determine model based on operation type (batch vs single)
            operation_model = MODEL_BATCH if len(payload.get("tokens", [])) > 1 else MODEL_SINGLE
            
            response = await asyncio.to_thread(
                chat_json,
                model=operation_model,
                system_prompt=system_prompt,
                user_payload=payload,
                agent_name="BatchConsensusV3",
                token=tag,
                temperature=0.25,  # Optimized for GPT-4o
                response_format={"type": "json_object"}  # Force pure JSON
            )
            
            # Validate response structure
            if not isinstance(response, dict) or "items" not in response:
                raise ValueError(f"Invalid response structure: missing 'items' key")
            
            return response
            
        except Exception as e:
            print(f"[CALL RETRY] {tag} attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise e
            # Increased backoff for hotfix v3.1
            await asyncio.sleep(2.0 * (attempt + 1))  # Longer exponential backoff
    
    raise Exception(f"All {max_retries} attempts failed for {tag}")

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
    """Process chunk asynchronously with aggressive timeout protection"""
    
    # Force micro-fallback for ALL chunks due to persistent JSON truncation
    print(f"[BATCH CHUNK {chunk_idx}] FORCED micro-fallback (JSON truncation prevention)")
    return await _micro_fallback_async(chunk)
    
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
    
    print(f"[BATCH CHUNK {chunk_idx}] Small chunk: {estimated_tokens} tokens, timeout: {TIMEOUT_MS/1000}s")
    
    try:
        # Use asyncio.wait_for for timeout control with proper response format
        response = await asyncio.wait_for(
            asyncio.to_thread(
                chat_json,
                model=MODEL,
                system_prompt=prompt,
                user_payload=payload,
                agent_name="BatchConsensus",
                token=f"CHUNK_{chunk_idx}_{len(chunk)}",
                temperature=0.1,  # Lower temperature for reliability
                response_format={"type": "json_object"}  # Force pure JSON response
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
    results_a = await _process_chunk_async(chunk_a, chunk_idx * 10 + 1)  # Convert to int
    results_b = await _process_chunk_async(chunk_b, chunk_idx * 10 + 2)  # Convert to int
    
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