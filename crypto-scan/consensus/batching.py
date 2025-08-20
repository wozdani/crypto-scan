"""
Load-aware chunking system with token weight balancing and priority ordering
"""
from typing import List, Dict, Any
import json

def estimate_cost_hint(packed: Dict[str, Any]) -> float:
    """
    Estimate processing 'weight' of a token for load balancing
    Heavy tokens: complex detectors + news flags + long history
    """
    detector_breakdown = packed.get("detector_breakdown", {})
    meta = packed.get("meta", {})
    history = packed.get("history", {})
    
    # Heavy detectors (AI-powered ones are more expensive)
    heavy_detector_weight = 0.0
    heavy_detector_weight += 0.8 * detector_breakdown.get("californium_whale", 0.0)
    heavy_detector_weight += 0.7 * detector_breakdown.get("diamond_whale", 0.0)
    heavy_detector_weight += 0.5 * detector_breakdown.get("stealth_engine", 0.0)
    heavy_detector_weight += 0.3 * detector_breakdown.get("whale_ping", 0.0)
    
    # News and market complexity
    market_complexity = 0.0
    if meta.get("news_flag", False):
        market_complexity += 0.4
    if meta.get("volume_24h", 0) > 10000000:  # High volume = more complex
        market_complexity += 0.2
    if len(meta.get("whale_addresses", [])) > 3:
        market_complexity += 0.3
    
    # History complexity
    events = history.get("events", [])
    history_weight = min(0.1 * len(events), 0.5)  # Cap at 0.5
    
    # Context size (larger JSON = more tokens to process)
    context_size = len(json.dumps(packed, ensure_ascii=False))
    size_weight = min(context_size / 2000.0, 0.4)  # Normalize and cap
    
    total_weight = heavy_detector_weight + market_complexity + history_weight + size_weight
    return round(total_weight, 3)

def priority_score(packed: Dict[str, Any]) -> float:
    """
    Calculate priority score - higher priority tokens should be processed first
    """
    detector_breakdown = packed.get("detector_breakdown", {})
    
    # Detector synergy (multiple strong signals)
    whale_ping = detector_breakdown.get("whale_ping", 0.0)
    dex_inflow = detector_breakdown.get("dex_inflow", 0.0)
    orderbook_anomaly = detector_breakdown.get("orderbook_anomaly", 0.0)
    
    synergy_score = (whale_ping * dex_inflow) + (0.3 * orderbook_anomaly)
    
    # Elite detector strength
    californium = detector_breakdown.get("californium_whale", 0.0)
    diamond = detector_breakdown.get("diamond_whale", 0.0)
    stealth = detector_breakdown.get("stealth_engine", 0.0)
    
    elite_score = max(californium, diamond, stealth)
    
    # Combined priority (synergy is more important than individual strength)
    priority = (0.7 * synergy_score) + (0.3 * elite_score)
    return round(priority, 3)

def make_balanced_chunks(packed_items: List[Dict[str, Any]], max_chunk: int = 5) -> List[List[Dict[str, Any]]]:
    """
    Create balanced chunks with load-aware distribution
    Heavy tokens are distributed evenly across chunks to prevent timeout in any single chunk
    """
    if not packed_items:
        return []
    
    # Calculate weights and sort by heaviness (descending)
    items_with_weights = []
    for item in packed_items:
        weight = estimate_cost_hint(item)
        priority = priority_score(item)
        items_with_weights.append((item, weight, priority))
    
    # Sort by weight (heaviest first) for balanced distribution
    items_with_weights.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate number of chunks needed
    num_chunks = max(1, (len(packed_items) + max_chunk - 1) // max_chunk)
    
    # Initialize buckets with tracking
    buckets = [[] for _ in range(num_chunks)]
    bucket_weights = [0.0] * num_chunks
    
    # Distribute items using round-robin with weight balancing
    for i, (item, weight, priority) in enumerate(items_with_weights):
        # Find bucket with lowest current weight
        min_weight_idx = bucket_weights.index(min(bucket_weights))
        
        # Add to bucket if not exceeding max_chunk size
        if len(buckets[min_weight_idx]) < max_chunk:
            target_bucket = min_weight_idx
        else:
            # Find next available bucket
            target_bucket = next(
                (i for i, bucket in enumerate(buckets) if len(bucket) < max_chunk),
                len(buckets) - 1  # Fallback to last bucket
            )
        
        buckets[target_bucket].append(item)
        bucket_weights[target_bucket] += weight
    
    # Remove empty buckets
    balanced_chunks = [bucket for bucket in buckets if bucket]
    
    # Log chunk distribution for diagnostics
    print(f"[BATCH CHUNKING] Created {len(balanced_chunks)} balanced chunks:")
    for i, chunk in enumerate(balanced_chunks):
        chunk_weight = sum(estimate_cost_hint(item) for item in chunk)
        tokens = [item["token_id"] for item in chunk]
        print(f"[BATCH CHUNKING] Chunk {i}: {len(chunk)} tokens, weight: {chunk_weight:.2f}, tokens: {tokens}")
    
    return balanced_chunks

def order_tokens_for_first_chunk(packed_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Order tokens by priority for first chunk processing
    Most promising tokens should be processed first before any timeouts occur
    """
    items_with_priority = [(item, priority_score(item)) for item in packed_items]
    items_with_priority.sort(key=lambda x: x[1], reverse=True)  # Highest priority first
    
    ordered_items = [item for item, _ in items_with_priority]
    
    priorities = [priority for _, priority in items_with_priority]
    print(f"[BATCH PRIORITY] Ordered {len(ordered_items)} tokens by priority: {priorities[:5]}...")
    
    return ordered_items

def analyze_chunk_distribution(chunks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analyze chunk distribution for load balancing diagnostics
    """
    if not chunks:
        return {"total_chunks": 0, "analysis": "No chunks to analyze"}
    
    chunk_stats = []
    total_weight = 0.0
    
    for i, chunk in enumerate(chunks):
        chunk_weight = sum(estimate_cost_hint(item) for item in chunk)
        chunk_priority = sum(priority_score(item) for item in chunk) / len(chunk) if chunk else 0
        
        chunk_stats.append({
            "chunk_id": i,
            "token_count": len(chunk),
            "total_weight": round(chunk_weight, 2),
            "avg_priority": round(chunk_priority, 3),
            "tokens": [item["token_id"] for item in chunk]
        })
        
        total_weight += chunk_weight
    
    avg_weight = total_weight / len(chunks) if chunks else 0
    weight_variance = sum((stats["total_weight"] - avg_weight) ** 2 for stats in chunk_stats) / len(chunks)
    
    return {
        "total_chunks": len(chunks),
        "total_weight": round(total_weight, 2),
        "avg_weight_per_chunk": round(avg_weight, 2),
        "weight_variance": round(weight_variance, 3),
        "chunk_stats": chunk_stats,
        "balanced": weight_variance < 0.5  # Low variance indicates good balancing
    }