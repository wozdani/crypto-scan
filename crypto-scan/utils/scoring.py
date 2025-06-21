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
    PPWCS Pre-Pump 2.0: CORE Hard Detectors Only (0-97 points max)
    Only core structural signals affect PPWCS scoring
    
    CORE Detectors (Hard):
    - whale_activity: +10 (whale transaction patterns)
    - dex_inflow: +10 (DEX accumulation anomaly)
    - stealth_inflow: +5 (stealth accumulation pattern)
    - compressed: +10 (Stage -1 compression)
    - stage1g_active: +10 (Stage 1G breakout signal)
    - event_tag: +10 (listing/partnership events)
    
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
        print(f"[PPWCS Pre-Pump 2.0] === CORE HARD DETECTORS ONLY ===")
        
        ppwcs_score = 0
        active_core_detectors = []
        
        # CORE Hard Detectors - Only these affect PPWCS scoring
        core_detectors = {
            "whale_activity": 10,     # Whale transactions detected
            "dex_inflow": 10,         # DEX inflow anomaly 
            "stealth_inflow": 5,      # Stealth accumulation pattern
            "compressed": 10,         # Stage -1 compression
            "stage1g_active": 10,     # Stage 1G breakout active
            "event_tag": 10,          # Listing/partnership events
            "liquidity_behavior": 7,  # Liquidity Behavior Detector
            "shadow_sync_v2": 25     # Shadow Sync Detector v2 – Stealth Protocol (Premium)
        }
        
        for detector, points in core_detectors.items():
            if signals.get(detector) is True:
                ppwcs_score += points
                active_core_detectors.append(detector)
                print(f"[CORE] ✅ {detector}: +{points}")
            else:
                print(f"[CORE] ❌ {detector}: not active")
        
        # Event Tags - Core scoring component
        event_tag = signals.get("event_tag")
        if event_tag and isinstance(event_tag, str):
            tag_lower = event_tag.lower()
            if tag_lower in ["listing", "partnership"]:
                ppwcs_score += 10
                active_core_detectors.append("event_tag")
                print(f"[CORE] ✅ Positive event tag ({tag_lower}): +10")
        
        # Risk Tags - Core penalty component
        risk_tag = signals.get("risk_tag")
        if risk_tag and isinstance(risk_tag, str):
            tag_lower = risk_tag.lower()
            if tag_lower in ["exploit", "unlock", "rug", "delisting"]:
                penalty = -15
                ppwcs_score += penalty
                print(f"[CORE] ❌ Risk tag ({tag_lower}): {penalty}")
                # Ensure score doesn't go below 0
                if ppwcs_score < 0:
                    ppwcs_score = 0
        
        final_score = max(0, ppwcs_score)
        print(f"[PPWCS Pre-Pump 2.0] Final CORE score: {final_score}/97 (Active: {len(active_core_detectors)}/7)")
        
        # Log soft detectors for context (not scored)
        soft_detectors = ["rsi_flatline", "gas_pressure", "dominant_accumulation", "spoofing", 
                         "heatmap_exhaustion", "vwap_pinning", "liquidity_box", "cluster_slope_up"]
        active_soft = [d for d in soft_detectors if signals.get(d) is True]
        if active_soft:
            print(f"[SOFT CONTEXT] Active quality signals: {active_soft} (checklist only)")
        
        return final_score, final_score, 0
        
    except Exception as e:
        print(f"❌ Error computing PPWCS Pre-Pump 2.0: {e}")
        return 0, 0, 0

def compute_checklist_score_simplified(signals: dict) -> tuple[int, list[str]]:
    """
    Pre-Pump 2.0: Soft Detectors Checklist Scoring (+5 each)
    Only soft/quality signals - hard detectors excluded from checklist
    
    Args:
        signals: Dictionary containing all detected signals
        
    Returns:
        tuple: (checklist_score, fulfilled_conditions_list)
    """
    try:
        print(f"[CHECKLIST Pre-Pump 2.0] === SOFT DETECTORS ONLY ===")
        
        fulfilled_conditions = []
        checklist_score = 0
        
        # Soft Detectors - Pre-Pump 2.0 specification (miękkie detektory)
        soft_detectors = {
            "rsi_flatline": "RSI flatline (45-55)",
            "gas_pressure": "Gas pressure/blockspace friction", 
            "dominant_accumulation": "Dominant accumulation pattern",
            "spoofing": "Spoofing/orderbook manipulation",
            "heatmap_exhaustion": "Heatmap exhaustion pattern",
            "vwap_pinning": "VWAP pinning behavior",
            "liquidity_box": "Liquidity box consolidation",
            "cluster_slope_up": "Volume cluster slope up"
        }
        
        # Check each soft detector
        for detector_key, description in soft_detectors.items():
            if signals.get(detector_key) is True:
                fulfilled_conditions.append(detector_key)
                checklist_score += 5
                print(f"[SOFT] ✅ {description}: +5")
            else:
                print(f"[SOFT] ❌ {description}: not detected")
        
        # Contextual detectors (do kontekstu alertów i feedbacku)
        contextual_detectors = {
            "fake_reject": "Fake reject recovery pattern",
            "fractal_momentum_echo": "Fractal momentum echo",
            "substructure_squeeze": "Substructure squeeze",
            "time_clustering": "Time clustering (sector sync)",
            "sector_clustering": "Sector clustering pattern",
            "whale_sequence": "Whale execution sequence",
            "execution_intent": "Execution intent confirmed",
            "dex_divergence": "DEX pool divergence", 
            "heatmap_trap": "Heatmap liquidity trap",
            "pure_accumulation": "Pure accumulation (no social)",
            "orderbook_anomaly": "Orderbook anomaly"
        }
        
        for detector_key, description in contextual_detectors.items():
            if signals.get(detector_key) is True:
                fulfilled_conditions.append(detector_key)
                checklist_score += 5
                print(f"[CONTEXT] ✅ {description}: +5")
            else:
                print(f"[CONTEXT] ❌ {description}: not detected")
        
        # Anti-fake filters
        if signals.get("rsi") and isinstance(signals["rsi"], (int, float)) and signals["rsi"] < 65:
            fulfilled_conditions.append("rsi_below_65")
            checklist_score += 5
            print(f"[FILTER] ✅ RSI below 65 ({signals['rsi']}): +5")
        
        if signals.get("risk_tag") not in ["exploit", "rug", "delisting", "unlock"]:
            fulfilled_conditions.append("no_risk_tags")
            checklist_score += 5
            print(f"[FILTER] ✅ No risk tags: +5")
        
        # Special logic for "no social spike" (inverted)
        if signals.get("social_spike") is False or signals.get("social_spike") is None:
            if "no_social_spike" not in fulfilled_conditions:
                fulfilled_conditions.append("no_social_spike")
                checklist_score += 5
                print(f"[FILTER] ✅ No social media hype: +5")
        
        print(f"[CHECKLIST Pre-Pump 2.0] Total soft score: {checklist_score}/110")
        print(f"[CHECKLIST Pre-Pump 2.0] Fulfilled conditions: {len(fulfilled_conditions)}/22")
        
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
        
        print(f"[COMBINED Pre-Pump 2.0] PPWCS: {ppwcs_score}/65, Checklist: {checklist_score}/110")
        print(f"[COMBINED Pre-Pump 2.0] Total: {total_combined}/175, Hard: {hard_signal_count}/6, Soft: {soft_signal_count}/22")
        
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
            try:
                with open(scores_file, "r") as f:
                    scores = json.load(f)
                    return scores.get(symbol, 0)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted JSON in {scores_file}, returning 0 for {symbol}")
                return 0
        return 0
    except Exception:
        return 0

def save_score(symbol, score):
    """Save current PPWCS score for future trailing logic"""
    try:
        scores_file = os.path.join("data", "ppwcs_scores.json")
        scores = {}
        
        if os.path.exists(scores_file):
            try:
                with open(scores_file, "r") as f:
                    scores = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Corrupted JSON in {scores_file}, starting fresh. Error: {e}")
                # Backup corrupted file
                backup_file = f"{scores_file}.backup"
                if os.path.exists(scores_file):
                    import shutil
                    shutil.copy2(scores_file, backup_file)
                    print(f"Corrupted file backed up to {backup_file}")
                scores = {}
        
        scores[symbol] = score
        
        os.makedirs(os.path.dirname(scores_file), exist_ok=True)
        
        # Write with atomic operation to prevent corruption
        temp_file = f"{scores_file}.tmp"
        with open(temp_file, "w") as f:
            json.dump(scores, f, indent=2)
        
        # Atomic rename to prevent corruption during write
        os.rename(temp_file, scores_file)
            
    except Exception as e:
        print(f"Error saving score for {symbol}: {e}")

def should_alert(symbol, score):
    """Determine if an alert should be sent based on score"""
    return score >= 70  # Simplified threshold for v3.0

def log_ppwcs_score(symbol, score, signals):
    """Log PPWCS score for analysis"""
    try:
        from datetime import datetime, timezone
        
        log_file = os.path.join("data", "ppwcs_log.json")
        log_entry = {
            "symbol": symbol,
            "score": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": len([k for k, v in signals.items() if v is True])
        }
        
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted JSON in {log_file}, starting fresh")
                logs = []
        
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
        
        # Handle different data formats - ensure we extract scores properly
        performers = []
        for symbol, data in scores.items():
            if isinstance(data, dict):
                # New format with metadata
                score = data.get('ppwcs_score', 0) if 'ppwcs_score' in data else data.get('score', 0)
            else:
                # Legacy format - direct score
                score = data if isinstance(data, (int, float)) else 0
            
            performers.append({"symbol": symbol, "score": score})
        
        # Sort by score and return top performers
        sorted_performers = sorted(performers, key=lambda x: x['score'], reverse=True)
        return sorted_performers[:limit]
        
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
    Weighted Quality Score - Pre-Pump 2.0 Quality Analysis
    Based on quality detectors with assigned weights for precise signal assessment
    
    Args:
        signals: Dictionary containing all detected signals
        
    Returns:
        tuple: (checklist_score, checklist_summary)
    """
    score = 0
    summary = []

    # Quality detectors with weighted scoring
    quality_detectors = {
        "rsi_flatline": 7,           # High weight - strong technical signal
        "gas_pressure": 5,           # Medium - blockchain pressure indicator
        "dominant_accumulation": 5,  # Medium - smart money accumulation
        "spoofing": 3,              # Low - orderbook manipulation
        "heatmap_exhaustion": 3,    # Low - supply exhaustion signal
        "vwap_pinning": 5,          # Medium - price anchoring behavior
        "liquidity_box": 5,         # Medium - consolidation pattern
        "cluster_slope_up": 3,      # Low - volume cluster analysis
        "pure_accumulation": 5      # Medium - stealth accumulation
    }
    
    # Calculate weighted quality score
    for detector, weight in quality_detectors.items():
        if signals.get(detector) is True:
            score += weight
            summary.append(detector)
            print(f"[QUALITY] ✅ {detector}: +{weight}")
        else:
            print(f"[QUALITY] ❌ {detector}: not detected")
    
    print(f"[CHECKLIST DEBUG] Quality score: {score}")
    print(f"[CHECKLIST DEBUG] Active quality signals: {len(summary)}/9")
    
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
    # FIXED: Use AND logic - both must be weak to reject, not OR
    if ppwcs < 25 and checklist_score < 20:
        print(f"[ALERT LEVEL] Level 0: Too weak (PPWCS<25 AND checklist<20)")
        return 0
    
    # Alternative rejection: Very weak PPWCS regardless of checklist
    if ppwcs < 15:
        print(f"[ALERT LEVEL] Level 0: Too weak (PPWCS<15 regardless of checklist)")
        return 0
    
    # Start with level 1 as base
    level = 1
    print(f"[ALERT LEVEL] Level 1: Watchlist (basic signals)")
    
    # Level 2: Pre-pump active - strong PPWCS OR strong structure
    if ppwcs >= 40 or (ppwcs >= 35 and checklist_score >= 35):
        level = 2
        print(f"[ALERT LEVEL] Level 2: Pre-pump active (strong PPWCS or structure)")
    
    # Level 3: Strong alert - high PPWCS OR maximum combined strength
    if ppwcs >= 50 or (ppwcs >= 45 and checklist_score >= 40):
        level = 3
        print(f"[ALERT LEVEL] Level 3: Strong alert (high PPWCS or combined strength)")
    
    print(f"[ALERT LEVEL] Final level: {level}")
    return level