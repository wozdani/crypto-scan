import json
import os
from datetime import datetime, timedelta

def compute_ppwcs(symbol, signals, compressed, stage1g_active, event_tags):
    """
    Compute Pre-Pump Warning Composite Score (PPWCS)
    Range: 0-100 points
    """
    try:
        score = 0
        
        # Stage -2 signals contribution (30 points max)
        if signals:
            signal_weight = {
                'volume_spike': 10,
                'price_stability': 5,
                'accumulation_pattern': 10,
                'market_cap_growth': 5
            }
            
            for signal, active in signals.items():
                if active and signal in signal_weight:
                    score += signal_weight[signal]
        
        # Stage -1 compressed signals contribution (25 points max)
        if compressed:
            # Strength component (0-10 points)
            strength = compressed.get('strength', 0)
            score += min(strength / 10, 10)
            
            # Confidence component (0-10 points)
            confidence = compressed.get('confidence', 0)
            score += min(confidence / 10, 10)
            
            # Binary signals (5 points total)
            if compressed.get('momentum', False):
                score += 2
            if compressed.get('volume_confirmed', False):
                score += 2
            if compressed.get('technical_alignment', False):
                score += 1
        
        # Stage 1G contribution (20 points max)
        if stage1g_active:
            score += 20
        
        # Event tags contribution (25 points max)
        if event_tags:
            tag_weights = {
                'ascending_triangle': 8,
                'volume_breakout': 10,
                'support_bounce': 5,
                'whale_accumulation': 12,
                'social_buzz_increase': 3
            }
            
            for tag in event_tags:
                if tag in tag_weights:
                    score += tag_weights[tag]
        
        # Cap score at 100
        score = min(score, 100)
        
        # Apply time-based decay for recent alerts
        score = apply_alert_decay(symbol, score)
        
        return round(score, 1)
        
    except Exception as e:
        print(f"❌ Error computing PPWCS for {symbol}: {e}")
        return 0

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
