import json
import os
from datetime import datetime, timedelta

def compute_ppwcs(signals: dict, previous_score: int = 0) -> int:
    """
    PPWCS 2.5: Pre-Pump Weighted Composite Score (0-100 points)
    Enhanced multi-stage analysis with context awareness
    """
    try:
        score = 0

        # --- STAGE -2.1: Micro-anomaly Detection ---
        if signals.get("whale_activity"):
            score += 15
        if signals.get("dex_inflow"):
            score += 15
        if signals.get("volume_spike"):
            score += 10
        if signals.get("orderbook_anomaly"):
            score += 10
        if signals.get("social_spike"):
            score += 5

        # --- Pure Accumulation Bonus ---
        # Whale + DEX inflow without social noise
        if (signals.get("whale_activity") and 
            signals.get("dex_inflow") and 
            not signals.get("social_spike")):
            score += 5

        # --- STAGE -2.2: News/Tag Analysis ---
        tag = signals.get("event_tag")
        boost_tags = {
            "listing": 10, 
            "partnership": 10, 
            "cex_listed": 7, 
            "presale": 7, 
            "airdrop": 3
        }
        risk_tags = {
            "exploit": -15, 
            "rug": -100, 
            "delisting": -100, 
            "unlock": -10, 
            "drama": -10
        }

        if tag:
            tag_lower = tag.lower()
            if tag_lower in boost_tags:
                score += boost_tags[tag_lower]
            elif tag_lower in risk_tags:
                score += risk_tags[tag_lower]

        if signals.get("event_risk"):
            score -= 15

        # --- STAGE -1: Compression ---
        if signals.get("compressed"):
            score += 15

        # --- STAGE 1G: Breakout Detection ---
        if signals.get("stage1g_active"):
            trigger_type = signals.get("stage1g_trigger_type")
            if trigger_type == "classic":
                score += 10
            elif trigger_type == "tag_boost":
                score += 7

        # --- Context/Timing Analysis ---
        current_hour = datetime.utcnow().hour
        # Market hours boost (6-18 UTC typically better for crypto)
        if 6 <= current_hour <= 18:
            score += 3

        # Optional: Heatmap exhaustion detection
        if signals.get("heatmap_exhaustion"):
            score += 3

        # --- Scaling to 0-100 range ---
        score = max(0, min(score, 100))

        # --- Trailing Logic: Only accept significant improvements ---
        if previous_score and score < previous_score + 5:
            return previous_score

        return score
        
    except Exception as e:
        print(f"❌ Error computing PPWCS: {e}")
        return 0

def get_previous_score(symbol):
    """
    Get the previous PPWCS score for trailing logic
    """
    try:
        # Try to get from recent scores file
        scores_file = os.path.join("data", "ppwcs_scores.json")
        if os.path.exists(scores_file):
            with open(scores_file, "r") as f:
                scores = json.load(f)
                return scores.get(symbol, 0)
        return 0
    except Exception as e:
        return 0

def save_score(symbol, score):
    """
    Save current PPWCS score for future trailing logic
    """
    try:
        scores_file = os.path.join("data", "ppwcs_scores.json")
        scores = {}
        
        if os.path.exists(scores_file):
            with open(scores_file, "r") as f:
                scores = json.load(f)
        
        scores[symbol] = score
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(scores_file), exist_ok=True)
        
        with open(scores_file, "w") as f:
            json.dump(scores, f, indent=2)
            
    except Exception as e:
        print(f"Error saving score for {symbol}: {e}")

def should_alert(symbol, score):
    """
    Determine if an alert should be sent based on score and recent alert history
    """
    try:
        # Minimum score threshold
        if score < 60:
            return False
            
        # Check recent alerts to avoid spam
        recent_alerts = get_recent_alerts(symbol, hours=4)
        
        # If there was a recent alert with similar or higher score, don't alert again
        for alert in recent_alerts:
            if alert.get('score', 0) >= score - 10:  # 10 point tolerance
                return False
                
        # If score is very high (90+), always alert regardless of recent history
        if score >= 90:
            return True
            
        # For scores 80-89, allow alerts every 2 hours
        if score >= 80:
            recent_high_alerts = get_recent_alerts(symbol, hours=2)
            return len(recent_high_alerts) == 0
            
        # For scores 70-79, allow alerts every 4 hours
        if score >= 70:
            return len(recent_alerts) == 0
            
        # For scores 60-69, allow alerts every 8 hours
        recent_medium_alerts = get_recent_alerts(symbol, hours=8)
        return len(recent_medium_alerts) == 0
        
    except Exception as e:
        print(f"❌ Error checking alert conditions for {symbol}: {e}")
        return False

def apply_alert_decay(symbol, score):
    """
    Apply time-based decay to reduce score for symbols that have been alerting frequently
    """
    try:
        recent_alerts = get_recent_alerts(symbol, hours=24)
        
        if len(recent_alerts) == 0:
            return score
            
        # Calculate decay factor based on alert frequency
        decay_factor = min(len(recent_alerts) * 0.05, 0.3)  # Max 30% decay
        decayed_score = score * (1 - decay_factor)
        
        return max(decayed_score, score * 0.7)  # Minimum 70% of original score
        
    except Exception as e:
        print(f"❌ Error applying alert decay for {symbol}: {e}")
        return score

def get_recent_alerts(symbol, hours=4):
    """
    Get recent alerts for a symbol within specified hours
    """
    try:
        alerts_file = "data/alerts/alerts_history.json"
        if not os.path.exists(alerts_file):
            return []
            
        with open(alerts_file, 'r') as f:
            alerts = json.load(f)
            
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_alerts = []
        for alert in alerts:
            try:
                alert_time = datetime.fromisoformat(alert.get('timestamp', ''))
                if alert_time > cutoff_time and symbol in alert.get('message', ''):
                    # Extract score from message if possible
                    message = alert.get('message', '')
                    if 'PPWCS:' in message:
                        score_part = message.split('PPWCS:')[1].split()[0]
                        try:
                            alert['score'] = float(score_part)
                        except:
                            alert['score'] = 0
                    recent_alerts.append(alert)
            except:
                continue
                
        return recent_alerts
        
    except Exception as e:
        print(f"❌ Error getting recent alerts for {symbol}: {e}")
        return []

def log_ppwcs_score(symbol, score):
    """
    Log PPWCS score to file for tracking and analysis
    """
    try:
        score_entry = {
            'symbol': symbol,
            'score': score,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Ensure scores directory exists
        os.makedirs("data/scores", exist_ok=True)
        
        # Load existing scores or create new list
        scores_file = "data/scores/ppwcs_scores.json"
        if os.path.exists(scores_file):
            with open(scores_file, 'r') as f:
                scores = json.load(f)
        else:
            scores = []
            
        scores.append(score_entry)
        
        # Keep only last 10000 scores
        if len(scores) > 10000:
            scores = scores[-10000:]
            
        with open(scores_file, 'w') as f:
            json.dump(scores, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error logging PPWCS score for {symbol}: {e}")

def get_symbol_stats(symbol, days=7):
    """
    Get statistics for a symbol over specified days
    """
    try:
        scores_file = "data/scores/ppwcs_scores.json"
        if not os.path.exists(scores_file):
            return None
            
        with open(scores_file, 'r') as f:
            scores = json.load(f)
            
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        symbol_scores = []
        for entry in scores:
            try:
                entry_time = datetime.fromisoformat(entry.get('timestamp', ''))
                if entry_time > cutoff_time and entry.get('symbol') == symbol:
                    symbol_scores.append(entry.get('score', 0))
            except:
                continue
                
        if not symbol_scores:
            return None
            
        return {
            'symbol': symbol,
            'count': len(symbol_scores),
            'avg_score': sum(symbol_scores) / len(symbol_scores),
            'max_score': max(symbol_scores),
            'min_score': min(symbol_scores),
            'latest_score': symbol_scores[-1] if symbol_scores else 0
        }
        
    except Exception as e:
        print(f"❌ Error getting stats for {symbol}: {e}")
        return None

def get_top_performers(limit=10, hours=24):
    """
    Get top performing symbols by PPWCS score in recent hours
    """
    try:
        scores_file = "data/scores/ppwcs_scores.json"
        if not os.path.exists(scores_file):
            return []
            
        with open(scores_file, 'r') as f:
            scores = json.load(f)
            
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Group by symbol and get max score
        symbol_max_scores = {}
        for entry in scores:
            try:
                entry_time = datetime.fromisoformat(entry.get('timestamp', ''))
                if entry_time > cutoff_time:
                    symbol = entry.get('symbol')
                    score = entry.get('score', 0)
                    if symbol not in symbol_max_scores or score > symbol_max_scores[symbol]['score']:
                        symbol_max_scores[symbol] = {
                            'symbol': symbol,
                            'score': score,
                            'timestamp': entry.get('timestamp')
                        }
            except:
                continue
                
        # Sort by score and return top performers
        top_performers = sorted(
            symbol_max_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        return top_performers[:limit]
        
    except Exception as e:
        print(f"❌ Error getting top performers: {e}")
        return []
