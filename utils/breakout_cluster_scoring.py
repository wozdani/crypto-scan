"""
Breakout Cluster Scoring - Sector-based Token Analysis
Gromadzenie i scoring wielu tokenów z tego samego sektora aktywnych w tym samym czasie
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict


# Sector mapping for cryptocurrency tokens
SECTOR_MAPPING = {
    # DeFi protocols
    "UNIUSDT": "defi", "AAVEUSDT": "defi", "COMPUSDT": "defi", "CRVUSDT": "defi",
    "SUSHIUSDT": "defi", "1INCHUSDT": "defi", "MKRUSDT": "defi", "YFIUSDT": "defi",
    
    # Layer 1 blockchains
    "ETHUSDT": "layer1", "ADAUSDT": "layer1", "DOTUSDT": "layer1", "AVAXUSDT": "layer1",
    "SOLUSDT": "layer1", "ALGOUSDT": "layer1", "ATOMUSDT": "layer1", "NEARUSDT": "layer1",
    
    # Layer 2 solutions
    "MATICUSDT": "layer2", "OPUSDT": "layer2", "ARBUSDT": "layer2", "LRCUSDT": "layer2",
    
    # Gaming/Metaverse
    "AXSUSDT": "gaming", "SANDUSDT": "gaming", "MANAUSDT": "gaming", "ENJUSDT": "gaming",
    "GALAUSDT": "gaming", "CHRUSDT": "gaming", "ALICEUSDT": "gaming",
    
    # AI/Data
    "FETUSDT": "ai", "OCEANUSDT": "ai", "AGIXUSDT": "ai", "RNDROUSDT": "ai",
    
    # Meme tokens
    "DOGEUSDT": "meme", "SHIBUSDT": "meme", "PEPEUSDT": "meme", "FLOKIUSDT": "meme",
    
    # Infrastructure
    "LINKUSDT": "infrastructure", "FILUSDT": "infrastructure", "ARUSDT": "infrastructure",
    
    # Privacy
    "XMRUSDT": "privacy", "ZCASHUSDT": "privacy", "SCRTUSDT": "privacy"
}


def get_token_sector(symbol):
    """Get sector for given token symbol"""
    return SECTOR_MAPPING.get(symbol, "other")


def load_cluster_activity():
    """Load recent cluster activity from file"""
    try:
        cluster_file = os.path.join("data", "cluster_activity.json")
        
        if not os.path.exists(cluster_file):
            return {}
        
        with open(cluster_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[CLUSTER] Error loading cluster activity: {e}")
        return {}


def save_cluster_activity(cluster_data):
    """Save cluster activity to file"""
    try:
        cluster_file = os.path.join("data", "cluster_activity.json")
        
        with open(cluster_file, 'w') as f:
            json.dump(cluster_data, f, indent=2)
            
    except Exception as e:
        print(f"[CLUSTER] Error saving cluster activity: {e}")


def record_token_activity(symbol, trend_score, trend_signals):
    """Record token activity for cluster analysis"""
    try:
        cluster_data = load_cluster_activity()
        
        current_time = datetime.now(timezone.utc)
        sector = get_token_sector(symbol)
        
        # Initialize sector data if not exists
        if sector not in cluster_data:
            cluster_data[sector] = {}
        
        # Record token activity
        cluster_data[sector][symbol] = {
            "trend_score": trend_score,
            "trend_signals": trend_signals,
            "timestamp": current_time.isoformat(),
            "sector": sector
        }
        
        # Clean old entries (older than 2 hours)
        cutoff_time = current_time - timedelta(hours=2)
        
        for sector_name in list(cluster_data.keys()):
            for token in list(cluster_data[sector_name].keys()):
                try:
                    token_time = datetime.fromisoformat(cluster_data[sector_name][token]["timestamp"].replace('Z', '+00:00'))
                    if token_time < cutoff_time:
                        del cluster_data[sector_name][token]
                except (KeyError, ValueError):
                    del cluster_data[sector_name][token]
            
            # Remove empty sectors
            if not cluster_data[sector_name]:
                del cluster_data[sector_name]
        
        save_cluster_activity(cluster_data)
        
        print(f"[CLUSTER] Recorded {symbol} activity in {sector} sector (score: {trend_score})")
        
    except Exception as e:
        print(f"[CLUSTER] Error recording token activity: {e}")


def analyze_sector_cluster(symbol, trend_score):
    """
    Analyze if multiple tokens from same sector are active simultaneously
    
    Args:
        symbol: current token symbol
        trend_score: current token's trend score
    
    Returns:
        dict: {
            "cluster_active": bool,
            "cluster_score": int (0-25),
            "sector": str,
            "active_tokens": list,
            "cluster_strength": str,
            "time_window": str
        }
    """
    print(f"[CLUSTER] Analyzing sector cluster for {symbol}")
    
    sector = get_token_sector(symbol)
    cluster_data = load_cluster_activity()
    
    if sector not in cluster_data:
        return {
            "cluster_active": False,
            "cluster_score": 0,
            "sector": sector,
            "active_tokens": [],
            "cluster_strength": "none",
            "time_window": "2 hours"
        }
    
    # Find active tokens in same sector within time window
    current_time = datetime.now(timezone.utc)
    time_window = timedelta(minutes=30)  # 30-minute cluster window
    
    active_tokens = []
    total_score = 0
    
    for token, data in cluster_data[sector].items():
        try:
            token_time = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
            time_diff = current_time - token_time
            
            if time_diff <= time_window:
                active_tokens.append({
                    "symbol": token,
                    "score": data["trend_score"],
                    "signals": len(data.get("trend_signals", [])),
                    "minutes_ago": int(time_diff.total_seconds() / 60)
                })
                total_score += data["trend_score"]
                
        except (KeyError, ValueError):
            continue
    
    # Calculate cluster score
    cluster_score = 0
    cluster_strength = "none"
    
    active_count = len(active_tokens)
    
    if active_count >= 2:  # At least 2 tokens in cluster
        # Base cluster score
        cluster_score = min(10 + (active_count * 3), 25)  # 10-25 points
        
        # Bonus for high average score
        if active_count > 0:
            avg_score = total_score / active_count
            if avg_score >= 40:
                cluster_score += 5
        
        # Determine cluster strength
        if active_count >= 4:
            cluster_strength = "strong"
        elif active_count >= 3:
            cluster_strength = "moderate"
        else:
            cluster_strength = "weak"
    
    cluster_active = cluster_score > 0
    
    if cluster_active:
        print(f"[CLUSTER] {sector.upper()} CLUSTER ACTIVE: {active_count} tokens, score {cluster_score}")
        for token in active_tokens:
            print(f"[CLUSTER]   {token['symbol']}: {token['score']}/50 ({token['minutes_ago']}min ago)")
    else:
        print(f"[CLUSTER] No cluster activity in {sector} sector")
    
    return {
        "cluster_active": cluster_active,
        "cluster_score": cluster_score,
        "sector": sector,
        "active_tokens": active_tokens,
        "cluster_strength": cluster_strength,
        "time_window": "30 minutes"
    }


def get_sector_momentum(sector, hours=2):
    """
    Get overall momentum for a specific sector
    
    Args:
        sector: sector name
        hours: time window in hours
    
    Returns:
        dict: Sector momentum analysis
    """
    cluster_data = load_cluster_activity()
    
    if sector not in cluster_data:
        return {
            "sector": sector,
            "momentum": "neutral",
            "active_tokens": 0,
            "avg_score": 0,
            "total_activity": 0
        }
    
    current_time = datetime.now(timezone.utc)
    time_window = timedelta(hours=hours)
    
    recent_tokens = []
    
    for token, data in cluster_data[sector].items():
        try:
            token_time = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
            if current_time - token_time <= time_window:
                recent_tokens.append(data["trend_score"])
        except (KeyError, ValueError):
            continue
    
    if not recent_tokens:
        momentum = "neutral"
        avg_score = 0
    else:
        avg_score = sum(recent_tokens) / len(recent_tokens)
        
        if avg_score >= 40:
            momentum = "strong"
        elif avg_score >= 30:
            momentum = "moderate"
        else:
            momentum = "weak"
    
    return {
        "sector": sector,
        "momentum": momentum,
        "active_tokens": len(recent_tokens),
        "avg_score": round(avg_score, 1),
        "total_activity": len(recent_tokens)
    }


def get_all_sectors_overview():
    """Get overview of all sector activities"""
    cluster_data = load_cluster_activity()
    
    sector_overview = {}
    
    for sector in set(SECTOR_MAPPING.values()):
        momentum_data = get_sector_momentum(sector)
        sector_overview[sector] = momentum_data
    
    # Sort by activity level
    sorted_sectors = sorted(
        sector_overview.items(),
        key=lambda x: x[1]["active_tokens"],
        reverse=True
    )
    
    return dict(sorted_sectors)


def compute_cluster_boost(symbol, trend_score, trend_signals):
    """
    Compute cluster boost for trend scoring
    
    Args:
        symbol: token symbol
        trend_score: base trend score
        trend_signals: list of active trend signals
    
    Returns:
        dict: {
            "cluster_boost": int (0-25),
            "cluster_analysis": dict,
            "boosted_score": int
        }
    """
    # Record current token activity
    record_token_activity(symbol, trend_score, trend_signals)
    
    # Analyze cluster
    cluster_analysis = analyze_sector_cluster(symbol, trend_score)
    
    cluster_boost = cluster_analysis["cluster_score"]
    boosted_score = min(50, trend_score + cluster_boost)  # Cap at 50
    
    return {
        "cluster_boost": cluster_boost,
        "cluster_analysis": cluster_analysis,
        "boosted_score": boosted_score
    }