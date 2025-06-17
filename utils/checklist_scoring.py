"""
Pre-Pump v3.0 Checklist Scoring System
Real market conditions checklist with 20 criteria at 5 points each
"""

from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

def compute_checklist_score(signals: Dict[str, Any]) -> Tuple[int, List[str], Dict[str, int]]:
    """
    Pre-Pump v3.0 Checklist Scoring System
    
    Args:
        signals: Dictionary containing all detected signals
        
    Returns:
        tuple: (checklist_score, fulfilled_conditions, category_scores)
    """
    
    try:
        fulfilled_conditions = []
        category_scores = {
            "technical_structure": 0,
            "smart_money": 0, 
            "microstructure": 0,
            "anti_fake_filters": 0
        }
        
        print(f"[CHECKLIST v3.0] Starting checklist evaluation...")
        
        # I. TECHNICAL STRUCTURE (4 conditions)
        print(f"[CHECKLIST] === TECHNICAL STRUCTURE ===")
        
        # 1. RSI flatline (45-55) min. 2 candles
        rsi_value = signals.get('rsi_value', 50)
        if isinstance(rsi_value, (int, float)) and 45 <= rsi_value <= 55:
            fulfilled_conditions.append("rsi_flatline")
            category_scores["technical_structure"] += 5
            print(f"[CHECKLIST] ✅ RSI flatline: {rsi_value}")
        else:
            print(f"[CHECKLIST] ❌ RSI flatline: {rsi_value} (not in 45-55 range)")
            
        # 2. Volume spike (Z-score > 2.5 or 3x previous candles)
        if signals.get('volume_spike') is True:
            fulfilled_conditions.append("volume_spike")
            category_scores["technical_structure"] += 5
            print(f"[CHECKLIST] ✅ Volume spike detected")
        else:
            print(f"[CHECKLIST] ❌ Volume spike: not detected")
            
        # 3. Fake Reject (long lower wick + high close)
        if signals.get('fake_reject') is True:
            fulfilled_conditions.append("fake_reject")
            category_scores["technical_structure"] += 5
            print(f"[CHECKLIST] ✅ Fake reject pattern detected")
        else:
            print(f"[CHECKLIST] ❌ Fake reject: not detected")
            
        # 4. Stage -1 compression active
        if signals.get('compressed') is True:
            fulfilled_conditions.append("compressed")
            category_scores["technical_structure"] += 5
            print(f"[CHECKLIST] ✅ Stage -1 compression active")
        else:
            print(f"[CHECKLIST] ❌ Compression: not active")
        
        # II. SMART MONEY BEHAVIOR (4 conditions)
        print(f"[CHECKLIST] === SMART MONEY BEHAVIOR ===")
        
        # 5. Whale TX > dynamic threshold
        if signals.get('whale_activity') is True:
            fulfilled_conditions.append("whale_tx")
            category_scores["smart_money"] += 5
            print(f"[CHECKLIST] ✅ Whale transactions detected")
        else:
            print(f"[CHECKLIST] ❌ Whale TX: not detected")
            
        # 6. DEX inflow > threshold (dynamic)
        if signals.get('dex_inflow') is True:
            fulfilled_conditions.append("dex_inflow")
            category_scores["smart_money"] += 5
            print(f"[CHECKLIST] ✅ DEX inflow detected")
        else:
            print(f"[CHECKLIST] ❌ DEX inflow: not detected")
            
        # 7. Stealth inflow (whale + volume, but no direct inflow)
        if signals.get('stealth_inflow') is True:
            fulfilled_conditions.append("stealth_inflow")
            category_scores["smart_money"] += 5
            print(f"[CHECKLIST] ✅ Stealth inflow detected")
        else:
            print(f"[CHECKLIST] ❌ Stealth inflow: not detected")
            
        # 8. Spoofing or heatmap exhaustion active
        spoofing_active = signals.get('spoofing') is True or signals.get('heatmap_exhaustion') is True
        if spoofing_active:
            fulfilled_conditions.append("spoofing_or_heatmap")
            category_scores["smart_money"] += 5
            print(f"[CHECKLIST] ✅ Spoofing/heatmap exhaustion detected")
        else:
            print(f"[CHECKLIST] ❌ Spoofing/heatmap: not detected")
        
        # III. MICROSTRUCTURE CONTEXT (8 optional boosts)
        print(f"[CHECKLIST] === MICROSTRUCTURE CONTEXT ===")
        
        # 9. VWAP pinning active
        if signals.get('vwap_pinning') is True:
            fulfilled_conditions.append("vwap_pinning")
            category_scores["microstructure"] += 5
            print(f"[CHECKLIST] ✅ VWAP pinning detected")
        else:
            print(f"[CHECKLIST] ❌ VWAP pinning: not detected")
            
        # 10. Fractal echo squeeze (mini squeeze pattern)
        if signals.get('fractal_momentum_echo') is True or signals.get('substructure_squeeze') is True:
            fulfilled_conditions.append("fractal_squeeze")
            category_scores["microstructure"] += 5
            print(f"[CHECKLIST] ✅ Fractal echo/squeeze detected")
        else:
            print(f"[CHECKLIST] ❌ Fractal squeeze: not detected")
            
        # 11. Time clustering (other tokens from sector also active)
        if signals.get('sector_clustering') is True:
            fulfilled_conditions.append("time_clustering")
            category_scores["microstructure"] += 5
            print(f"[CHECKLIST] ✅ Time clustering detected")
        else:
            print(f"[CHECKLIST] ❌ Time clustering: not detected")
            
        # 12. Positive tag: listing, presale, airdrop
        event_tag = signals.get('event_tag', '').lower()
        positive_tags = ['listing', 'presale', 'airdrop', 'partnership', 'cex_listed']
        if event_tag in positive_tags:
            fulfilled_conditions.append("positive_tag")
            category_scores["microstructure"] += 5
            print(f"[CHECKLIST] ✅ Positive event tag: {event_tag}")
        else:
            print(f"[CHECKLIST] ❌ Positive tag: {event_tag} (not positive)")
            
        # 13. Orderbook anomaly
        if signals.get('orderbook_anomaly') is True:
            fulfilled_conditions.append("orderbook_anomaly")
            category_scores["microstructure"] += 5
            print(f"[CHECKLIST] ✅ Orderbook anomaly detected")
        else:
            print(f"[CHECKLIST] ❌ Orderbook anomaly: not detected")
            
        # 14. Whale execution pattern
        if signals.get('whale_sequence') is True:
            fulfilled_conditions.append("whale_sequence")
            category_scores["microstructure"] += 5
            print(f"[CHECKLIST] ✅ Whale execution pattern detected")
        else:
            print(f"[CHECKLIST] ❌ Whale sequence: not detected")
            
        # 15. Gas pressure / blockspace friction
        if signals.get('gas_pressure') is True:
            fulfilled_conditions.append("gas_pressure")
            category_scores["microstructure"] += 5
            print(f"[CHECKLIST] ✅ Gas pressure detected")
        else:
            print(f"[CHECKLIST] ❌ Gas pressure: not detected")
            
        # 16. DEX pool divergence
        if signals.get('dex_divergence') is True:
            fulfilled_conditions.append("dex_divergence")
            category_scores["microstructure"] += 5
            print(f"[CHECKLIST] ✅ DEX pool divergence detected")
        else:
            print(f"[CHECKLIST] ❌ DEX divergence: not detected")
        
        # IV. ANTI-FAKE FILTERS (4 conditions)
        print(f"[CHECKLIST] === ANTI-FAKE FILTERS ===")
        
        # 17. No social hype (pure accumulation)
        if signals.get('pure_accumulation') is True or signals.get('social_spike') is False:
            fulfilled_conditions.append("no_social_hype")
            category_scores["anti_fake_filters"] += 5
            print(f"[CHECKLIST] ✅ No social hype (pure accumulation)")
        else:
            print(f"[CHECKLIST] ❌ Social hype detected")
            
        # 18. RSI < 65 (not overbought)
        if isinstance(rsi_value, (int, float)) and rsi_value < 65:
            fulfilled_conditions.append("not_overbought")
            category_scores["anti_fake_filters"] += 5
            print(f"[CHECKLIST] ✅ Not overbought: RSI {rsi_value}")
        else:
            print(f"[CHECKLIST] ❌ Overbought: RSI {rsi_value}")
            
        # 19. No risky tags (exploit, rug, delisting, unlock)
        risky_tags = ['exploit', 'rug', 'delisting', 'unlock', 'drama']
        if event_tag not in risky_tags:
            fulfilled_conditions.append("no_risky_tags")
            category_scores["anti_fake_filters"] += 5
            print(f"[CHECKLIST] ✅ No risky tags")
        else:
            print(f"[CHECKLIST] ❌ Risky tag detected: {event_tag}")
            
        # 20. Execution intent confirmed
        if signals.get('execution_intent') is True:
            fulfilled_conditions.append("execution_intent")
            category_scores["anti_fake_filters"] += 5
            print(f"[CHECKLIST] ✅ Execution intent confirmed")
        else:
            print(f"[CHECKLIST] ❌ Execution intent: not confirmed")
        
        # Calculate final score
        total_score = sum(category_scores.values())
        condition_count = len(fulfilled_conditions)
        
        print(f"[CHECKLIST v3.0] === SUMMARY ===")
        print(f"[CHECKLIST v3.0] Total score: {total_score}/100")
        print(f"[CHECKLIST v3.0] Conditions fulfilled: {condition_count}/20")
        print(f"[CHECKLIST v3.0] Technical: {category_scores['technical_structure']}/20")
        print(f"[CHECKLIST v3.0] Smart Money: {category_scores['smart_money']}/20")
        print(f"[CHECKLIST v3.0] Microstructure: {category_scores['microstructure']}/40")
        print(f"[CHECKLIST v3.0] Anti-Fake: {category_scores['anti_fake_filters']}/20")
        print(f"[CHECKLIST v3.0] Fulfilled: {fulfilled_conditions}")
        
        # Determine confidence level
        if condition_count >= 7:
            confidence_level = "High Confidence Setup"
            print(f"[CHECKLIST v3.0] 🔥 HIGH CONFIDENCE SETUP ({condition_count}/20 conditions)")
        elif condition_count >= 5:
            confidence_level = "Pre-Pump Trigger"
            print(f"[CHECKLIST v3.0] ⚡ PRE-PUMP TRIGGER ({condition_count}/20 conditions)")
        else:
            confidence_level = "Insufficient Conditions"
            print(f"[CHECKLIST v3.0] ❌ INSUFFICIENT CONDITIONS ({condition_count}/20)")
        
        # Store additional metadata separately (not in category_scores dict)
        metadata = {
            "confidence_level": confidence_level,
            "condition_count": condition_count
        }
        
        return total_score, fulfilled_conditions, category_scores
        
    except Exception as e:
        logger.error(f"Error in compute_checklist_score: {e}")
        return 0, [], {"technical_structure": 0, "smart_money": 0, "microstructure": 0, "anti_fake_filters": 0}

def get_checklist_summary(fulfilled_conditions: List[str], category_scores: Dict[str, int]) -> Dict[str, Any]:
    """
    Generate checklist summary for reports and logs
    
    Args:
        fulfilled_conditions: List of fulfilled condition names
        category_scores: Scores by category
        
    Returns:
        dict: Summary data for JSON reports
    """
    
    condition_count = len(fulfilled_conditions)
    total_score = sum(v for k, v in category_scores.items() if isinstance(v, int))
    
    # Determine setup type
    if condition_count >= 7:
        setup_type = "High Confidence Setup"
        setup_icon = "🔥"
    elif condition_count >= 5:
        setup_type = "Pre-Pump Trigger"
        setup_icon = "⚡"
    else:
        setup_type = "Insufficient Conditions"
        setup_icon = "❌"
    
    return {
        "checklist_score": total_score,
        "checklist_summary": fulfilled_conditions,
        "condition_count": condition_count,
        "setup_type": setup_type,
        "setup_icon": setup_icon,
        "category_breakdown": {
            "technical_structure": category_scores.get("technical_structure", 0),
            "smart_money": category_scores.get("smart_money", 0),
            "microstructure": category_scores.get("microstructure", 0),
            "anti_fake_filters": category_scores.get("anti_fake_filters", 0)
        },
        "checklist_percentage": round((total_score / 100) * 100, 1)
    }

def format_checklist_for_alert(checklist_score: int, fulfilled_conditions: List[str], 
                             category_scores: Dict[str, int]) -> str:
    """
    Format checklist results for Telegram alerts
    
    Args:
        checklist_score: Total checklist score
        fulfilled_conditions: List of fulfilled conditions
        category_scores: Scores by category
        
    Returns:
        str: Formatted text for alerts
    """
    
    condition_count = len(fulfilled_conditions)
    
    # Setup type
    if condition_count >= 7:
        setup_type = "🔥 HIGH CONFIDENCE"
    elif condition_count >= 5:
        setup_type = "⚡ PRE-PUMP TRIGGER"
    else:
        setup_type = "❌ INSUFFICIENT"
    
    # Category breakdown
    tech = category_scores.get("technical_structure", 0)
    smart = category_scores.get("smart_money", 0)
    micro = category_scores.get("microstructure", 0)
    filters = category_scores.get("anti_fake_filters", 0)
    
    # Top fulfilled conditions (max 5 for alert)
    top_conditions = fulfilled_conditions[:5]
    conditions_text = ", ".join(top_conditions) if top_conditions else "None"
    
    return f"""📋 **Checklist v3.0:** {checklist_score}/100 ({condition_count}/20)
🎯 **Setup:** {setup_type}
📊 **Categories:** Tech:{tech} | Smart:{smart} | Micro:{micro} | Filter:{filters}
✅ **Top Signals:** {conditions_text}"""