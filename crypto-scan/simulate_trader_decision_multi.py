#!/usr/bin/env python3
"""
Strategiczny Moduł Decyzyjny - Multi-Signal Trading Decision Engine
Analizuje kombinację GNN + WhaleCLIP + DEX inflow dla inteligentnych decyzji alertów

Symuluje intuicję tradera zamiast sztywnych kalkulacji matematycznych.
"""

import json
import os
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def simulate_trader_decision_multi(gnn_score: float,
                                   whale_clip_conf: float,
                                   dex_inflow_flag: bool,
                                   debug: bool = False) -> Tuple[str, float]:
    """
    Analizuje kombinację sygnałów i zwraca (etykietę, końcowy score).
    
    Myśli jak trader - interpretuje siłę sygnału na podstawie kombinacji detektorów
    zamiast sztywnych wzorów matematycznych.
    
    Args:
        gnn_score: GNN anomaly score [0.0-1.0]
        whale_clip_conf: WhaleCLIP confidence [0.0-1.0] 
        dex_inflow_flag: DEX inflow detection [True/False]
        debug: Enable debug logging
        
    Returns:
        Tuple of (decision_label, final_score)
        
    Decision Categories:
        - STRONG_SIGNAL: Bardzo silny sygnał (GNN + WhaleCLIP)
        - STEALTH_STYLE_ACCUMULATION: Stealth akumulacja (WhaleCLIP + DEX)
        - GRAPH_WEAK_BUT_CONFIRMED: Słaby graf ale potwierdzony DEX
        - STYLE_MATCH_WITH_LOW_GRAPH: Styl pasuje ale słaby graf
        - GRAPH_ONLY: Tylko sygnał grafowy
        - DEX_ONLY: Tylko DEX inflow
        - AVOID: Unikaj - słabe sygnały
    """
    
    if debug:
        print(f"[TJDE MULTI DEBUG] GNN: {gnn_score:.3f}, WhaleCLIP: {whale_clip_conf:.3f}, DEX Inflow: {dex_inflow_flag}")
        logger.info(f"[TRADER DECISION] Analyzing signals - GNN: {gnn_score:.3f}, "
                   f"WhaleCLIP: {whale_clip_conf:.3f}, DEX: {dex_inflow_flag}")

    # TIER 1: Najsilniejsze kombinacje - natychmiastowy alert
    if gnn_score > 0.75 and whale_clip_conf > 0.85:
        decision = "STRONG_SIGNAL"
        final_score = 1.0
        if debug:
            print(f"[TRADER LOGIC] TIER 1: Bardzo silny sygnał - GNN i WhaleCLIP potwierdzają")
    
    # TIER 2: Stealth akumulacja - WhaleCLIP + DEX bez silnego GNN
    elif whale_clip_conf > 0.9 and dex_inflow_flag:
        decision = "STEALTH_STYLE_ACCUMULATION" 
        final_score = 0.85
        if debug:
            print(f"[TRADER LOGIC] TIER 2: Stealth akumulacja - silny WhaleCLIP + DEX inflow")
    
    # TIER 3: Słaby graf ale potwierdzony DEX inflow
    elif gnn_score > 0.65 and dex_inflow_flag:
        decision = "GRAPH_WEAK_BUT_CONFIRMED"
        final_score = 0.78
        if debug:
            print(f"[TRADER LOGIC] TIER 3: Słaby graf ale potwierdzony DEX inflow")
    
    # TIER 4: Styl pasuje ale graf słaby - ostrożny alert
    elif whale_clip_conf > 0.88 and gnn_score > 0.55:
        decision = "STYLE_MATCH_WITH_LOW_GRAPH"
        final_score = 0.72
        if debug:
            print(f"[TRADER LOGIC] TIER 4: Styl pasuje ale graf słaby - ostrożny sygnał")
    
    # TIER 5: Tylko sygnał grafowy - GNN dominuje
    elif gnn_score > 0.6:
        decision = "GRAPH_ONLY"
        final_score = 0.65
        if debug:
            print(f"[TRADER LOGIC] TIER 5: Tylko sygnał grafowy - GNN dominuje")
    
    # TIER 6: Tylko DEX inflow - słaby ale zauważalny
    elif dex_inflow_flag:
        decision = "DEX_ONLY"
        final_score = 0.6
        if debug:
            print(f"[TRADER LOGIC] TIER 6: Tylko DEX inflow - słaby ale zauważalny")
    
    # TIER 7: Unikaj - zbyt słabe sygnały
    else:
        decision = "AVOID"
        final_score = 0.0
        if debug:
            print(f"[TRADER LOGIC] TIER 7: Unikaj - zbyt słabe sygnały")
    
    if debug:
        print(f"[TJDE MULTI RESULT] Decision: {decision}, Final Score: {final_score:.3f}")
    
    return decision, final_score

def save_decision_training_data(gnn_score: float, 
                               whale_clip_conf: float, 
                               dex_inflow_flag: bool,
                               decision: str, 
                               final_score: float,
                               symbol: Optional[str] = None,
                               timestamp: Optional[datetime] = None) -> bool:
    """
    Zapisuje dane treningowe dla przyszłego ML modelu decyzyjnego.
    
    Args:
        gnn_score: GNN anomaly score
        whale_clip_conf: WhaleCLIP confidence  
        dex_inflow_flag: DEX inflow detection
        decision: Decision label
        final_score: Final decision score
        symbol: Token symbol (optional)
        timestamp: Decision timestamp (optional)
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        training_file = "cache/decision_training_data.jsonl"
        os.makedirs(os.path.dirname(training_file), exist_ok=True)
        
        if timestamp is None:
            timestamp = datetime.now()
        
        training_record = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "inputs": {
                "gnn_score": round(gnn_score, 4),
                "whale_clip_confidence": round(whale_clip_conf, 4), 
                "dex_inflow_flag": dex_inflow_flag
            },
            "outputs": {
                "decision": decision,
                "final_score": round(final_score, 4)
            },
            "decision_tier": _get_decision_tier(decision)
        }
        
        # Append to JSONL file
        with open(training_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_record) + "\n")
        
        logger.info(f"[TRAINING DATA] Saved decision: {decision} (score: {final_score:.3f}) for {symbol or 'unknown'}")
        return True
        
    except Exception as e:
        logger.error(f"[TRAINING DATA] Failed to save: {e}")
        return False

def _get_decision_tier(decision: str) -> int:
    """Mapuje decision label na tier numeryczny"""
    tier_mapping = {
        "STRONG_SIGNAL": 1,
        "STEALTH_STYLE_ACCUMULATION": 2,
        "GRAPH_WEAK_BUT_CONFIRMED": 3,
        "STYLE_MATCH_WITH_LOW_GRAPH": 4,
        "GRAPH_ONLY": 5,
        "DEX_ONLY": 6,
        "AVOID": 7
    }
    return tier_mapping.get(decision, 7)

def analyze_decision_patterns(training_file: str = "cache/decision_training_data.jsonl") -> Dict[str, Any]:
    """
    Analizuje wzorce decyzyjne z danych treningowych.
    
    Args:
        training_file: Path to training data file
        
    Returns:
        Dictionary with analysis results
    """
    try:
        if not os.path.exists(training_file):
            return {"error": "No training data found"}
        
        decisions = []
        with open(training_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    decisions.append(json.loads(line))
        
        if not decisions:
            return {"error": "No decisions in training data"}
        
        # Analiza statystyk
        decision_counts = {}
        tier_counts = {}
        total_decisions = len(decisions)
        
        for record in decisions:
            decision = record["outputs"]["decision"]
            tier = record["decision_tier"]
            
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            tier_counts[f"Tier_{tier}"] = tier_counts.get(f"Tier_{tier}", 0) + 1
        
        # Średnie scores per decision
        decision_scores = {}
        for record in decisions:
            decision = record["outputs"]["decision"]
            score = record["outputs"]["final_score"]
            
            if decision not in decision_scores:
                decision_scores[decision] = []
            decision_scores[decision].append(score)
        
        avg_scores = {k: round(sum(v)/len(v), 3) for k, v in decision_scores.items()}
        
        analysis = {
            "total_decisions": total_decisions,
            "decision_distribution": decision_counts,
            "tier_distribution": tier_counts,
            "average_scores_by_decision": avg_scores,
            "most_common_decision": max(decision_counts, key=decision_counts.get),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[DECISION ANALYSIS] Analyzed {total_decisions} decisions")
        return analysis
        
    except Exception as e:
        logger.error(f"[DECISION ANALYSIS] Failed: {e}")
        return {"error": str(e)}

def get_decision_statistics() -> Dict[str, Any]:
    """
    Pobiera podstawowe statystyki decyzji.
    
    Returns:
        Dictionary with decision statistics
    """
    try:
        analysis = analyze_decision_patterns()
        
        if "error" in analysis:
            return {
                "total_decisions": 0,
                "unique_decision_types": 0,
                "most_common": "No data",
                "status": "No training data available"
            }
        
        return {
            "total_decisions": analysis["total_decisions"],
            "unique_decision_types": len(analysis["decision_distribution"]),
            "most_common": analysis["most_common_decision"],
            "decision_types": list(analysis["decision_distribution"].keys()),
            "status": "Active"
        }
        
    except Exception as e:
        return {
            "total_decisions": 0,
            "unique_decision_types": 0,
            "most_common": "Error",
            "status": f"Error: {e}"
        }

def test_trader_decision_multi():
    """Test funkcjonalności trader decision multi"""
    print("=" * 60)
    print("TRADER DECISION MULTI - TEST SCENARIOS")
    print("=" * 60)
    
    test_scenarios = [
        # (gnn_score, whale_clip_conf, dex_inflow, expected_tier)
        (0.80, 0.90, True, "STRONG_SIGNAL"),
        (0.60, 0.95, True, "STEALTH_STYLE_ACCUMULATION"),
        (0.70, 0.70, True, "GRAPH_WEAK_BUT_CONFIRMED"),
        (0.60, 0.90, False, "STYLE_MATCH_WITH_LOW_GRAPH"),
        (0.65, 0.50, False, "GRAPH_ONLY"),
        (0.30, 0.60, True, "DEX_ONLY"),
        (0.40, 0.50, False, "AVOID")
    ]
    
    for i, (gnn, whale_clip, dex, expected) in enumerate(test_scenarios, 1):
        print(f"\n[TEST {i}] GNN: {gnn}, WhaleCLIP: {whale_clip}, DEX: {dex}")
        decision, score = simulate_trader_decision_multi(gnn, whale_clip, dex, debug=True)
        
        # Save to training data
        save_decision_training_data(gnn, whale_clip, dex, decision, score, 
                                   symbol=f"TEST{i}USDT")
        
        status = "✅ PASS" if decision == expected else f"❌ FAIL (expected: {expected})"
        print(f"[RESULT] {decision} (score: {score:.3f}) - {status}")
    
    # Analyze patterns
    print(f"\n{'='*60}")
    print("DECISION PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    analysis = analyze_decision_patterns()
    if "error" not in analysis:
        print(f"Total Decisions: {analysis['total_decisions']}")
        print(f"Most Common: {analysis['most_common_decision']}")
        print(f"Decision Distribution: {analysis['decision_distribution']}")
        print(f"Average Scores: {analysis['average_scores_by_decision']}")
    
    print(f"\n✅ Trader Decision Multi - Test Complete")

if __name__ == "__main__":
    test_trader_decision_multi()