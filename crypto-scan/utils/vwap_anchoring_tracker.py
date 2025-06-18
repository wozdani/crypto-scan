"""
VWAP Anchoring Tracker - Dynamic Price Magnet Analysis
Sprawdza, czy VWAP służy jako dynamiczny magnes (średnia transakcyjna whales) – wzmacnia scoring
"""

import numpy as np
from datetime import datetime, timezone, timedelta
import json
import os


def calculate_vwap_levels(data, periods=[20, 50, 100]):
    """Calculate multiple VWAP timeframes for anchoring analysis"""
    vwap_levels = {}
    
    for period in periods:
        if len(data) >= period:
            recent_data = data[-period:]
            
            volumes = np.array([candle[5] for candle in recent_data])
            typical_prices = np.array([(candle[2] + candle[3] + candle[4]) / 3 for candle in recent_data])
            
            total_volume = np.sum(volumes)
            if total_volume > 0:
                vwap = np.sum(typical_prices * volumes) / total_volume
                vwap_levels[f"vwap_{period}"] = vwap
            else:
                vwap_levels[f"vwap_{period}"] = 0
        else:
            vwap_levels[f"vwap_{period}"] = 0
    
    return vwap_levels


def analyze_price_vwap_interaction(data, current_price):
    """Analyze how price interacts with VWAP levels"""
    if len(data) < 20:
        return {
            "interaction_type": "insufficient_data",
            "magnet_strength": 0,
            "distance_pct": 0
        }
    
    # Calculate multiple VWAP levels
    vwap_levels = calculate_vwap_levels(data)
    
    # Focus on 20-period VWAP for anchoring analysis
    vwap_20 = vwap_levels.get("vwap_20", 0)
    
    if vwap_20 <= 0:
        return {
            "interaction_type": "invalid_vwap",
            "magnet_strength": 0,
            "distance_pct": 0
        }
    
    # Calculate distance from VWAP
    distance_pct = ((current_price - vwap_20) / vwap_20) * 100
    
    # Analyze recent price behavior around VWAP
    recent_candles = data[-10:]  # Last 10 candles
    touches = 0
    rejections = 0
    
    for candle in recent_candles:
        low = candle[3]
        high = candle[2]
        close = candle[4]
        
        # Check if price touched VWAP area (±0.5%)
        vwap_zone_upper = vwap_20 * 1.005
        vwap_zone_lower = vwap_20 * 0.995
        
        if low <= vwap_zone_upper and high >= vwap_zone_lower:
            touches += 1
            
            # Check for rejection (close away from VWAP after touching)
            if close > vwap_zone_upper or close < vwap_zone_lower:
                rejections += 1
    
    # Determine interaction type
    interaction_type = "neutral"
    magnet_strength = 0
    
    if touches >= 3:  # Frequent VWAP interaction
        if rejections >= 2:
            interaction_type = "strong_magnet"
            magnet_strength = min(15, touches * 3)  # Up to 15 points
        else:
            interaction_type = "weak_magnet"
            magnet_strength = min(8, touches * 2)   # Up to 8 points
    elif abs(distance_pct) < 1.0:  # Very close to VWAP
        interaction_type = "proximity_magnet"
        magnet_strength = 5
    
    return {
        "interaction_type": interaction_type,
        "magnet_strength": magnet_strength,
        "distance_pct": round(distance_pct, 2),
        "touches_count": touches,
        "rejections_count": rejections,
        "vwap_20": vwap_20
    }


def detect_vwap_whale_activity(data, symbol):
    """Detect if whales are using VWAP as execution anchor"""
    if len(data) < 20:
        return {
            "whale_anchoring": False,
            "anchoring_score": 0,
            "evidence": []
        }
    
    vwap_levels = calculate_vwap_levels(data)
    vwap_20 = vwap_levels.get("vwap_20", 0)
    
    if vwap_20 <= 0:
        return {
            "whale_anchoring": False,
            "anchoring_score": 0,
            "evidence": ["invalid_vwap"]
        }
    
    recent_candles = data[-20:]
    evidence = []
    anchoring_score = 0
    
    # Look for whale execution patterns around VWAP
    high_volume_at_vwap = 0
    vwap_bounces = 0
    
    for i, candle in enumerate(recent_candles[1:], 1):
        prev_candle = recent_candles[i-1]
        
        open_price = candle[1]
        high = candle[2]
        low = candle[3]
        close = candle[4]
        volume = candle[5]
        prev_volume = prev_candle[5]
        
        # Check for high volume near VWAP
        vwap_zone_upper = vwap_20 * 1.01
        vwap_zone_lower = vwap_20 * 0.99
        
        price_in_vwap_zone = (low <= vwap_zone_upper and high >= vwap_zone_lower)
        volume_spike = volume > (prev_volume * 1.5) if prev_volume > 0 else False
        
        if price_in_vwap_zone and volume_spike:
            high_volume_at_vwap += 1
            evidence.append(f"high_volume_at_vwap_candle_{i}")
        
        # Check for VWAP bounce (price rejection from VWAP level)
        if (low <= vwap_20 <= high) and (close > vwap_20 * 1.005):  # Bounced up from VWAP
            vwap_bounces += 1
            evidence.append(f"vwap_bounce_up_candle_{i}")
        elif (low <= vwap_20 <= high) and (close < vwap_20 * 0.995):  # Rejected at VWAP
            evidence.append(f"vwap_rejection_candle_{i}")
    
    # Calculate anchoring score
    if high_volume_at_vwap >= 3:
        anchoring_score += 10
        evidence.append("frequent_volume_at_vwap")
    elif high_volume_at_vwap >= 2:
        anchoring_score += 6
    
    if vwap_bounces >= 2:
        anchoring_score += 8
        evidence.append("multiple_vwap_bounces")
    elif vwap_bounces >= 1:
        anchoring_score += 4
    
    # Check for sustained price action above/below VWAP
    recent_closes = [c[4] for c in recent_candles[-5:]]
    above_vwap_count = sum(1 for close in recent_closes if close > vwap_20)
    
    if above_vwap_count >= 4:  # 4/5 closes above VWAP
        anchoring_score += 5
        evidence.append("sustained_above_vwap")
    elif above_vwap_count <= 1:  # 1/5 closes above VWAP
        anchoring_score += 3
        evidence.append("sustained_below_vwap")
    
    whale_anchoring = anchoring_score >= 8
    
    return {
        "whale_anchoring": whale_anchoring,
        "anchoring_score": anchoring_score,
        "evidence": evidence,
        "high_volume_events": high_volume_at_vwap,
        "vwap_bounces": vwap_bounces
    }


def analyze_vwap_trend_alignment(data, trend_direction="up"):
    """Analyze if VWAP supports the current trend direction"""
    if len(data) < 50:
        return {
            "alignment": "insufficient_data",
            "alignment_score": 0
        }
    
    # Calculate VWAP slope over different periods
    vwap_20_current = calculate_vwap_levels(data[-20:])["vwap_20"]
    vwap_20_previous = calculate_vwap_levels(data[-40:-20])["vwap_20"]
    
    if vwap_20_current <= 0 or vwap_20_previous <= 0:
        return {
            "alignment": "invalid_data",
            "alignment_score": 0
        }
    
    # Calculate VWAP trend
    vwap_change_pct = ((vwap_20_current - vwap_20_previous) / vwap_20_previous) * 100
    
    alignment_score = 0
    alignment = "neutral"
    
    if trend_direction == "up":
        if vwap_change_pct > 1.0:  # VWAP rising >1%
            alignment = "strong_aligned"
            alignment_score = 10
        elif vwap_change_pct > 0.3:  # VWAP rising >0.3%
            alignment = "aligned"
            alignment_score = 6
        elif vwap_change_pct > 0:  # VWAP slightly rising
            alignment = "weak_aligned"
            alignment_score = 3
        else:  # VWAP falling while trend up
            alignment = "divergent"
            alignment_score = -5
    
    return {
        "alignment": alignment,
        "alignment_score": alignment_score,
        "vwap_change_pct": round(vwap_change_pct, 3)
    }


def compute_vwap_anchoring_score(data, symbol, trend_direction="up"):
    """
    Main function to compute VWAP anchoring score for trend enhancement
    
    Args:
        data: OHLCV candle data
        symbol: token symbol
        trend_direction: expected trend direction
    
    Returns:
        dict: {
            "vwap_anchoring_active": bool,
            "anchoring_score": int (0-25),
            "magnet_analysis": dict,
            "whale_analysis": dict,
            "trend_alignment": dict
        }
    """
    print(f"[VWAP ANCHOR] Analyzing VWAP anchoring for {symbol}")
    
    if len(data) < 20:
        return {
            "vwap_anchoring_active": False,
            "anchoring_score": 0,
            "magnet_analysis": {"interaction_type": "insufficient_data"},
            "whale_analysis": {"whale_anchoring": False},
            "trend_alignment": {"alignment": "insufficient_data"}
        }
    
    current_price = data[-1][4]
    
    # Analyze price-VWAP interaction (magnet effect)
    magnet_analysis = analyze_price_vwap_interaction(data, current_price)
    
    # Detect whale anchoring activity
    whale_analysis = detect_vwap_whale_activity(data, symbol)
    
    # Check trend alignment
    trend_alignment = analyze_vwap_trend_alignment(data, trend_direction)
    
    # Calculate total anchoring score
    total_score = 0
    
    # Magnet strength component (0-15 points)
    total_score += magnet_analysis.get("magnet_strength", 0)
    
    # Whale anchoring component (0-10 points, but cap contribution)
    whale_score = min(10, whale_analysis.get("anchoring_score", 0))
    total_score += whale_score
    
    # Trend alignment component (0-10 points, or negative for divergence)
    alignment_score = trend_alignment.get("alignment_score", 0)
    total_score += max(0, alignment_score)  # Only positive alignment
    
    # Cap total score at 25
    total_score = min(25, total_score)
    
    anchoring_active = total_score >= 8  # Minimum threshold for active anchoring
    
    if anchoring_active:
        print(f"[VWAP ANCHOR] {symbol}: ACTIVE anchoring detected (score: {total_score}/25)")
        print(f"[VWAP ANCHOR]   Magnet: {magnet_analysis['interaction_type']} ({magnet_analysis['magnet_strength']}pts)")
        print(f"[VWAP ANCHOR]   Whale: {whale_analysis['whale_anchoring']} ({whale_score}pts)")
        print(f"[VWAP ANCHOR]   Alignment: {trend_alignment['alignment']} ({alignment_score}pts)")
    else:
        print(f"[VWAP ANCHOR] {symbol}: No significant anchoring (score: {total_score}/25)")
    
    return {
        "vwap_anchoring_active": anchoring_active,
        "anchoring_score": total_score,
        "magnet_analysis": magnet_analysis,
        "whale_analysis": whale_analysis,
        "trend_alignment": trend_alignment
    }


def save_vwap_anchoring_data(symbol, anchoring_result):
    """Save VWAP anchoring analysis for tracking"""
    try:
        anchoring_data = {
            "symbol": symbol,
            "anchoring_result": anchoring_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Save to VWAP anchoring file
        anchoring_file = os.path.join("data", "vwap_anchoring.json")
        
        data = []
        if os.path.exists(anchoring_file):
            with open(anchoring_file, 'r') as f:
                data = json.load(f)
        
        data.append(anchoring_data)
        
        # Keep only last 100 entries
        data = data[-100:]
        
        with open(anchoring_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[VWAP ANCHOR] Anchoring data saved for {symbol}")
        
    except Exception as e:
        print(f"[VWAP ANCHOR] Save error: {e}")