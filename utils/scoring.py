"""
PPWCS v3.0: Hard Signal Detection & Checklist Scoring System
Separated scoring layers for maximum precision
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

def compute_ppwcs(signals: dict, previous_score: int = 0) -> tuple[int, int, int]:
    """
    PPWCS v3.0: Hard Signal Detection Only (0-65 points max)
    Pre-Pump 2.0 compliant scoring for core hard detectors only
    
    Args:
        signals: Dictionary containing all detected signals
        previous_score: Previous score for trailing logic (unused in v3.0)
        
    Returns:
        tuple: (total_score, structure_score, quality_score)
    """
    if not isinstance(signals, dict):
        print(f"⚠️ signals nie jest dict: {signals}")
        return 0, 0, 0

    try:
        print(f"[PPWCS v3.0] === HARD SIGNALS ONLY ===")
        
        ppwcs_score = 0
        
        # Hard Detectors - volume_spike removed, stealth_inflow added as compensation
        hard_detectors = {
            "whale_activity": 10,     # Whale transactions detected
            "dex_inflow": 10,         # DEX inflow anomaly
            "compressed": 10,         # Stage -1 compression
            "stage1g_active": 10,     # Stage 1G breakout active
            "stealth_inflow": 5       # Stealth accumulation (compensation for volume_spike removal)
        }
        
        for detector, points in hard_detectors.items():
            if signals.get(detector) is True:
                ppwcs_score += points
                print(f"[PPWCS v3.0] ✅ {detector}: +{points}")
            else:
                print(f"[PPWCS v3.0] ❌ {detector}: not active")
        
        # Event Tags (positive +10, negative -15)
        event_tag = signals.get("event_tag")
        if event_tag and isinstance(event_tag, str):
            tag_lower = event_tag.lower()
            if tag_lower in ["listing", "partnership"]:
                ppwcs_score += 10
                print(f"[PPWCS v3.0] ✅ Positive event tag ({tag_lower}): +10")
            elif tag_lower in ["exploit", "unlock", "rug", "delisting"]:
                penalty = -15
                ppwcs_score += penalty
                print(f"[PPWCS v3.0] ❌ Risk tag ({tag_lower}): {penalty}")
                # Ensure score doesn't go below 0
                if ppwcs_score < 0:
                    ppwcs_score = 0
        
        final_score = max(0, ppwcs_score)
        print(f"[PPWCS v3.0] Final hard signals score: {final_score}/65")
        
        return final_score, final_score, 0
        
    except Exception as e:
        print(f"❌ Error computing PPWCS v3.0: {e}")
        return 0, 0, 0

def compute_checklist_score_simplified(signals: dict) -> tuple[int, list[str]]:
    """
    Simplified checklist scoring for soft signals (+5 each)
    
    Args:
        signals: Dictionary containing all detected signals
        
    Returns:
        tuple: (checklist_score, fulfilled_conditions_list)
    """
    try:
        print(f"[CHECKLIST] === SOFT SIGNALS EVALUATION ===")
        
        fulfilled_conditions = []
        checklist_score = 0
        
        # Soft Signal Checklist (+5 points each) - volume_spike removed per user request
        soft_signals = {
            # Technical indicators
            "RSI_flatline": "RSI flatline (45-55)",
            "vwap_pinning": "VWAP pinning",
            "fake_reject": "Fake reject pattern",
            
            # Smart money behavior (secondary)
            "spoofing": "Spoofing detected",
            "stealth_inflow": "Stealth inflow",
            "orderbook_anomaly": "Orderbook anomaly",
            
            # Microstructure patterns
            "fractal_momentum_echo": "Fractal momentum echo",
            "substructure_squeeze": "Substructure squeeze",
            "liquidity_box": "Liquidity box pattern",
            
            # Context signals
            "time_clustering": "Time clustering",
            "sector_clustering": "Sector clustering", 
            "whale_sequence": "Whale execution pattern",
            "gas_pressure": "Gas pressure/blockspace friction",
            "execution_intent": "Execution intent confirmed",
            "dex_divergence": "DEX pool divergence",
            "heatmap_trap": "Heatmap liquidity trap",
            
            # Quality filters
            "pure_accumulation": "Pure accumulation (no social hype)",
            "no_social_spike": "No social media hype",
        }
        
        # Evaluate each soft signal
        for signal_key, description in soft_signals.items():
            if signals.get(signal_key) is True:
                fulfilled_conditions.append(signal_key)
                checklist_score += 5
                print(f"[CHECKLIST] ✅ {description}: +5")
            else:
                print(f"[CHECKLIST] ❌ {description}: not detected")
        
        # Special logic for "no social spike" (inverted)
        if signals.get("social_spike") is False or signals.get("social_spike") is None:
            if "no_social_spike" not in fulfilled_conditions:
                fulfilled_conditions.append("no_social_spike")
                checklist_score += 5
                print(f"[CHECKLIST] ✅ No social media hype: +5")
        
        print(f"[CHECKLIST] Total checklist score: {checklist_score}/85")
        print(f"[CHECKLIST] Fulfilled conditions: {len(fulfilled_conditions)}/17")
        
        return checklist_score, fulfilled_conditions
        
    except Exception as e:
        print(f"❌ Error computing checklist score: {e}")
        return 0, []

def compute_combined_scores(signals: dict) -> dict:
    """
    Compute both PPWCS and checklist scores with combined analysis
    
    Args:
        signals: Dictionary containing all detected signals
        
    Returns:
        dict: Combined scoring results
    """
    try:
        # Compute PPWCS (hard signals)
        ppwcs_score, ppwcs_structure, ppwcs_quality = compute_ppwcs(signals)
        
        # Compute checklist (soft signals)  
        checklist_score, checklist_summary = compute_checklist_score_simplified(signals)
        
        # Combined analysis
        total_combined = ppwcs_score + checklist_score
        hard_signal_count = sum([1 for k in ["whale_activity", "dex_inflow", "compressed", "stage1g_active", "stealth_inflow"] 
                                if signals.get(k) is True])
        soft_signal_count = len(checklist_summary)
        
        print(f"[COMBINED ANALYSIS] PPWCS: {ppwcs_score}/65, Checklist: {checklist_score}/85")
        print(f"[COMBINED ANALYSIS] Total: {total_combined}/150, Hard: {hard_signal_count}/5, Soft: {soft_signal_count}/17")
        
        return {
            "ppwcs": ppwcs_score,
            "checklist_score": checklist_score, 
            "checklist_summary": checklist_summary,
            "total_combined": total_combined,
            "hard_signal_count": hard_signal_count,
            "soft_signal_count": soft_signal_count,
            "ppwcs_structure": ppwcs_structure,
            "ppwcs_quality": ppwcs_quality
        }
        
    except Exception as e:
        print(f"❌ Error in combined scoring: {e}")
        return {
            "ppwcs": 0,
            "checklist_score": 0,
            "checklist_summary": [],
            "total_combined": 0,
            "hard_signal_count": 0,
            "soft_signal_count": 0,
            "ppwcs_structure": 0,
            "ppwcs_quality": 0
        }

# Legacy function aliases for compatibility
def score_stage_minus2_1(signals):
    """Legacy compatibility function"""
    return 0

def score_stage_1g(signals):
    """Legacy compatibility function"""
    return 0

def get_previous_score(symbol):
    """Get the previous PPWCS score for trailing logic"""
    try:
        scores_file = os.path.join("data", "ppwcs_scores.json")
        if os.path.exists(scores_file):
            with open(scores_file, "r") as f:
                scores = json.load(f)
                return scores.get(symbol, 0)
        return 0
    except Exception:
        return 0

def save_score(symbol, score):
    """Save current PPWCS score for future trailing logic"""
    try:
        scores_file = os.path.join("data", "ppwcs_scores.json")
        scores = {}
        
        if os.path.exists(scores_file):
            with open(scores_file, "r") as f:
                scores = json.load(f)
        
        scores[symbol] = score
        
        os.makedirs(os.path.dirname(scores_file), exist_ok=True)
        
        with open(scores_file, "w") as f:
            json.dump(scores, f, indent=2)
            
    except Exception as e:
        print(f"Error saving score for {symbol}: {e}")

def should_alert(symbol, score):
    """Determine if an alert should be sent based on score"""
    return score >= 70  # Simplified threshold for v3.0

def log_ppwcs_score(symbol, score, signals):
    """Log PPWCS score for analysis"""
    try:
        log_file = os.path.join("data", "ppwcs_log.json")
        log_entry = {
            "symbol": symbol,
            "score": score,
            "timestamp": json.dumps(None, default=str),
            "signals": len([k for k, v in signals.items() if v is True])
        }
        
        logs = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
            
    except Exception as e:
        print(f"Error logging PPWCS score: {e}")

def get_top_performers(hours=24, limit=10):
    """Get top performing symbols by PPWCS score"""
    try:
        scores_file = os.path.join("data", "ppwcs_scores.json")
        if not os.path.exists(scores_file):
            return []
            
        with open(scores_file, "r") as f:
            scores = json.load(f)
        
        # Sort by score and return top performers
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"symbol": symbol, "score": score} for symbol, score in sorted_scores[:limit]]
        
    except Exception as e:
        print(f"Error getting top performers: {e}")
        return []

def get_symbol_stats(symbol):
    """Get statistics for a specific symbol"""
    try:
        scores_file = os.path.join("data", "ppwcs_scores.json")
        if not os.path.exists(scores_file):
            return {"symbol": symbol, "score": 0, "alerts": 0}
            
        with open(scores_file, "r") as f:
            scores = json.load(f)
        
        score = scores.get(symbol, 0)
        return {
            "symbol": symbol,
            "score": score,
            "alerts": 1 if score >= 70 else 0
        }
        
    except Exception as e:
        print(f"Error getting symbol stats: {e}")
        return {"symbol": symbol, "score": 0, "alerts": 0}

def compute_checklist_score(signals: dict):
    """
    Oblicza strukturę pre-pump na podstawie checklisty (maksymalnie 100 pkt)
    Każdy warunek = +5 punktów
    
    Args:
        signals: Dictionary containing all detected signals
        
    Returns:
        tuple: (checklist_score, checklist_summary)
    """
    score = 0
    summary = []

    # I. STRUKTURA TECHNICZNA
    if signals.get("rsi_flatline"):
        score += 5
        summary.append("rsi_flatline")

    if signals.get("fake_reject"):
        score += 5
        summary.append("fake_reject")

    if signals.get("compressed"):
        score += 5
        summary.append("compressed")

    # II. ZACHOWANIE SMART MONEY
    if signals.get("whale_activity"):
        score += 5
        summary.append("whale_activity")

    if signals.get("dex_inflow"):
        score += 5
        summary.append("dex_inflow")

    if signals.get("stealth_inflow"):
        score += 5
        summary.append("stealth_inflow")

    if signals.get("spoofing") or signals.get("heatmap_exhaustion"):
        score += 5
        summary.append("spoofing_or_exhaustion")

    # III. KONTEKST MIKROSTRUKTURALNY
    if signals.get("vwap_pinning"):
        score += 5
        summary.append("vwap_pinning")

    if signals.get("fractal_echo_squeeze"):
        score += 5
        summary.append("fractal_echo_squeeze")

    if signals.get("time_clustering"):
        score += 5
        summary.append("time_clustering")

    if signals.get("event_tag") in ["listing", "presale", "airdrop", "partnership"]:
        score += 5
        summary.append("positive_event_tag")

    # IV. FILTRY ANTY-FAKE
    if signals.get("pure_accumulation"):
        score += 5
        summary.append("pure_accumulation")

    if signals.get("rsi") and signals["rsi"] < 65:
        score += 5
        summary.append("rsi_below_65")

    if signals.get("risk_tag") not in ["exploit", "unlock", "rug", "delisting"]:
        score += 5
        summary.append("no_risk_tags")

    return score, summary

def get_recent_alerts(hours=24, limit=100):
    """Get recent alerts for dashboard"""
    try:
        alerts_file = os.path.join("data", "recent_alerts.json")
        if not os.path.exists(alerts_file):
            return []
            
        with open(alerts_file, "r") as f:
            alerts = json.load(f)
        
        # Return recent alerts (simplified for v3.0)
        return alerts[-limit:] if alerts else []
        
    except Exception as e:
        print(f"Error getting recent alerts: {e}")
        return []


def get_alert_level(ppwcs: int, checklist_score: int) -> int:
    """
    Zwraca poziom alertu na podstawie PPWCS (0-65) i checklist_score (0-85).
    
    Poziomy alertów po aktualizacji scoring system (volume_spike removed):
    - 0: brak alertu (słabe sygnały)
    - 1: obserwacja / watchlist (podstawowe sygnały)
    - 2: pre-pump aktywny (silne sygnały strukturalne)
    - 3: silny alert / impuls możliwy (maksymalna siła)
    
    Args:
        ppwcs: PPWCS score (0-65 range after stealth_inflow compensation)
        checklist_score: Checklist score (0-85 v3.0 range)
        
    Returns:
        int: Alert level (0-3)
    """
    print(f"[ALERT LEVEL] Evaluating: PPWCS {ppwcs}/65, Checklist {checklist_score}/85")
    
    # Level 0: Weak signals - no alert
    if ppwcs < 25 or checklist_score < 20:
        print(f"[ALERT LEVEL] Level 0: Too weak (PPWCS<25 or checklist<20)")
        return 0
    
    # Start with level 1 as base
    level = 1
    print(f"[ALERT LEVEL] Level 1: Watchlist (basic signals)")
    
    # Level 2: Pre-pump active - strong structural signals
    if ppwcs >= 40 and checklist_score >= 35:
        level = 2
        print(f"[ALERT LEVEL] Level 2: Pre-pump active (strong structure)")
    
    # Level 3: Strong alert - maximum signal strength
    if ppwcs >= 50 and checklist_score >= 50:
        level = 3
        print(f"[ALERT LEVEL] Level 3: Strong alert (maximum strength)")
    
    print(f"[ALERT LEVEL] Final level: {level}")
    return level