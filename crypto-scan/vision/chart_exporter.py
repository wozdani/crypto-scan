"""
Vision-AI Chart Export Controls
Determines when charts should be generated for training data quality optimization
"""

def should_generate_chart(symbol: str, decision: str, final_score: float, clip_confidence: float = 0.0) -> bool:
    """
    Określa, czy należy wygenerować wykres dla danego tokena w kontekście Vision-AI Training.
    
    Args:
        symbol: Trading symbol
        decision: TJDE decision (long/short/wait/avoid/skip)
        final_score: Final TJDE score (0.0-1.0)
        clip_confidence: Optional CLIP confidence for additional filtering
        
    Returns:
        bool: True if chart should be generated for Vision-AI training
    """
    
    # Skip invalid symbols (detected by invalid symbol filter)
    if hasattr(symbol, 'invalid_symbol') or 'invalid' in symbol.lower():
        return False
    
    # Always generate for strong trading signals
    if decision in ["long", "short", "enter"]:
        return True
    
    # Generate for scalp entries with decent scores
    if decision == "scalp_entry" and final_score >= 0.15:
        return True
        
    # Generate for wait decisions with high potential (quality setups)
    if decision == "wait" and final_score >= 0.6:
        return True
    
    # Generate for moderate wait signals with high CLIP confidence (visual patterns)
    if decision == "wait" and final_score >= 0.4 and clip_confidence >= 0.7:
        return True
    
    # Optional: Generate some avoid/wait cases for false-positive learning
    if decision in ["wait", "avoid"] and 0.5 <= final_score < 0.6:
        # Only generate 20% of these cases to avoid overwhelming dataset
        import random
        if random.random() < 0.2:
            return True
    
    # Skip all other cases (avoid, skip, invalid, low scores)
    return False

def get_chart_generation_reason(symbol: str, decision: str, final_score: float, clip_confidence: float = 0.0) -> str:
    """
    Returns human-readable reason why chart was/wasn't generated for debugging
    
    Args:
        symbol: Trading symbol
        decision: TJDE decision
        final_score: Final TJDE score
        clip_confidence: CLIP confidence
        
    Returns:
        str: Reason for chart generation decision
    """
    
    if not should_generate_chart(symbol, decision, final_score, clip_confidence):
        if decision in ["avoid", "skip"]:
            return f"SKIP: {decision} decision with score {final_score:.3f}"
        elif final_score < 0.4:
            return f"SKIP: Low score {final_score:.3f}"
        elif decision == "wait" and final_score < 0.6 and clip_confidence < 0.7:
            return f"SKIP: Weak wait signal (score: {final_score:.3f}, clip: {clip_confidence:.3f})"
        else:
            return f"SKIP: Unqualified signal"
    
    # Chart will be generated - provide reason
    if decision in ["long", "short", "enter"]:
        return f"GENERATE: Strong signal ({decision}, score: {final_score:.3f})"
    elif decision == "scalp_entry":
        return f"GENERATE: Scalp entry (score: {final_score:.3f})"
    elif decision == "wait" and final_score >= 0.6:
        return f"GENERATE: High potential wait (score: {final_score:.3f})"
    elif clip_confidence >= 0.7:
        return f"GENERATE: High CLIP confidence (clip: {clip_confidence:.3f})"
    else:
        return f"GENERATE: False-positive learning case"

def filter_tokens_for_chart_generation(scan_results: list) -> tuple:
    """
    Filters scan results to separate tokens that should have charts generated
    
    Args:
        scan_results: List of scan result dictionaries
        
    Returns:
        tuple: (tokens_for_charts, skipped_tokens, generation_stats)
    """
    
    tokens_for_charts = []
    skipped_tokens = []
    stats = {
        "total_tokens": len(scan_results),
        "charts_generated": 0,
        "charts_skipped": 0,
        "skip_reasons": {}
    }
    
    for result in scan_results:
        symbol = result.get("symbol", "UNKNOWN")
        decision = result.get("decision", "unknown")
        final_score = result.get("final_score", 0.0)
        clip_confidence = result.get("clip_confidence", 0.0)
        
        if should_generate_chart(symbol, decision, final_score, clip_confidence):
            tokens_for_charts.append(result)
            stats["charts_generated"] += 1
        else:
            reason = get_chart_generation_reason(symbol, decision, final_score, clip_confidence)
            skipped_tokens.append({
                "symbol": symbol,
                "decision": decision,
                "final_score": final_score,
                "skip_reason": reason
            })
            stats["charts_skipped"] += 1
            
            # Track skip reasons for analytics
            reason_key = reason.split(":")[0]  # Get SKIP category
            stats["skip_reasons"][reason_key] = stats["skip_reasons"].get(reason_key, 0) + 1
    
    return tokens_for_charts, skipped_tokens, stats

def log_chart_generation_summary(stats: dict):
    """
    Logs summary of chart generation filtering decisions
    
    Args:
        stats: Statistics dictionary from filter_tokens_for_chart_generation
    """
    
    total = stats["total_tokens"]
    generated = stats["charts_generated"]
    skipped = stats["charts_skipped"]
    
    print(f"\n📊 CHART GENERATION FILTER SUMMARY:")
    print(f"   Total tokens processed: {total}")
    print(f"   Charts to generate: {generated} ({generated/total*100:.1f}%)")
    print(f"   Charts skipped: {skipped} ({skipped/total*100:.1f}%)")
    
    if stats["skip_reasons"]:
        print(f"   Skip reasons breakdown:")
        for reason, count in stats["skip_reasons"].items():
            print(f"     - {reason}: {count} tokens")
    
    print(f"✅ Chart generation optimization: {skipped/total*100:.1f}% reduction in unnecessary charts")

def should_save_to_training_data(symbol: str, decision: str, final_score: float) -> bool:
    """
    Additional filter for saving to training_data/charts directory
    Even stricter criteria for final training dataset
    
    Args:
        symbol: Trading symbol
        decision: TJDE decision
        final_score: Final TJDE score
        
    Returns:
        bool: True if should be saved to training dataset
    """
    
    # Only highest quality signals for training data
    if decision in ["long", "short", "enter"] and final_score >= 0.7:
        return True
    
    if decision == "scalp_entry" and final_score >= 0.2:
        return True
        
    if decision == "wait" and final_score >= 0.65:
        return True
    
    return False