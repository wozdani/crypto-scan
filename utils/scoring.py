import json
import os
from datetime import datetime, timedelta, timezone

def score_stage_minus2_1(data):
    """
    Stage -2.1 scoring based on detector count and combinations
    PPWCS 2.6 implementation - FIXED LOGIC
    """
    # Poprawiona logika: tylko explicit True counts + compressed jako aktywny detektor
    detectors = [
        "whale_activity", "dex_inflow", "orderbook_anomaly", 
        "volume_spike", "vwap_pinning", "spoofing", 
        "cluster_slope", "heatmap_exhaustion", "social_spike",
        "compressed"  # DODANE: compressed jako pełnoprawny detektor Stage -2.1
    ]
    
    active_detectors = []
    for detector in detectors:
        value = data.get(detector, False)
        if value is True:  # Strict True check
            active_detectors.append(detector)
    
    count = len(active_detectors)
    print(f"[SCORING DEBUG] Stage -2.1 active detectors: {active_detectors} (count: {count})")

    # NOWE WAGI - znacznie wyższe dla silnych sygnałów
    score = 0
    
    # Individual signal scores - PODWYŻSZONE
    if data.get("whale_activity") is True:
        score += 12
        print(f"[SCORING DEBUG] Whale activity: +12")
    if data.get("volume_spike") is True:
        score += 12
        print(f"[SCORING DEBUG] Volume spike: +12")
    if data.get("dex_inflow") is True:
        score += 10
        print(f"[SCORING DEBUG] DEX inflow: +10")
    if data.get("orderbook_anomaly") is True:
        score += 8
        print(f"[SCORING DEBUG] Orderbook anomaly: +8")
    if data.get("vwap_pinning") is True:
        score += 6
        print(f"[SCORING DEBUG] VWAP pinning: +6")
    if data.get("spoofing") is True:
        score += 6
        print(f"[SCORING DEBUG] Spoofing: +6")
    if data.get("heatmap_exhaustion") is True:
        score += 6
        print(f"[SCORING DEBUG] Heatmap exhaustion: +6")
    if data.get("cluster_slope") is True:
        score += 5
        print(f"[SCORING DEBUG] Cluster slope: +5")
    if data.get("social_spike") is True:
        score += 4
        print(f"[SCORING DEBUG] Social spike: +4")
    if data.get("compressed") is True:
        score += 10
        print(f"[SCORING DEBUG] Compressed: +10")

    # COMBO BONUSY - jeszcze wyższe
    if data.get("whale_activity") is True and data.get("dex_inflow") is True:
        score += 8
        print(f"[SCORING DEBUG] Whale+DEX combo: +8")
    if data.get("volume_spike") is True and data.get("dex_inflow") is True:
        score += 6
        print(f"[SCORING DEBUG] Volume+DEX combo: +6")
    if data.get("volume_spike") is True and data.get("spoofing") is True:
        score += 6
        print(f"[SCORING DEBUG] Volume+Spoofing combo: +6")
    if data.get("vwap_pinning") is True and data.get("orderbook_anomaly") is True:
        score += 5
        print(f"[SCORING DEBUG] VWAP+Orderbook combo: +5")

    return score

def score_stage_1g(data):
    """
    Stage 1g quality filter scoring
    PPWCS 2.6 implementation - FIXED LOGIC
    """
    score = 0
    if data.get("squeeze") is True:
        score += 6
        print(f"[SCORING DEBUG] Stage 1g squeeze: +6")
    if data.get("stealth_acc") is True:
        score += 6
        print(f"[SCORING DEBUG] Stage 1g stealth_acc: +6")
    if data.get("fake_reject") is True:
        score -= 4
        print(f"[SCORING DEBUG] Stage 1g fake_reject: -4")
    if data.get("vwap_pinning") is True:
        score += 3
        print(f"[SCORING DEBUG] Stage 1g vwap_pinning: +3")
    if data.get("liquidity_box") is True:
        score += 3
        print(f"[SCORING DEBUG] Stage 1g liquidity_box: +3")
    if data.get("RSI_flatline") is True and data.get("inflow") is True:
        score += 3
        print(f"[SCORING DEBUG] Stage 1g RSI_flatline+inflow: +3")
    if data.get("fractal_echo") is True:
        score += 2
        print(f"[SCORING DEBUG] Stage 1g fractal_echo: +2")
    return score

def compute_ppwcs(signals: dict, previous_score: int = 0) -> tuple[int, int, int]:
    """
    PPWCS 2.6: Pre-Pump Weighted Composite Score (0-100 points)
    Enhanced multi-stage analysis with new scoring algorithms
    """
    if not isinstance(signals, dict):
        print(f"⚠️ signals nie jest dict: {signals}")
        return 0, 0, 0

    try:
        # Debug INPUT SIGNALS and active detectors - FIXED LOGIC
        print(f"[PPWCS DEBUG] INPUT SIGNALS: {signals}")
        
        # Poprawiona logika: tylko explicit True, nie False/None/0.0/""
        active = []
        for k, v in signals.items():
            print(f"[PPWCS DEBUG] Signal {k} = {v} (type: {type(v)})")
            if v is True:  # Strict True check
                active.append(k)
        
        print(f"[PPWCS DEBUG] Active detectors (corrected): {active}")
        
        ppwcs_structure = 0  # Stage -2.1, compressed, news, pure accumulation
        ppwcs_quality = 0    # Stage 1g

        # --- STAGE -2.1: Micro-anomaly Detection (New Algorithm) ---
        stage_minus2_1_score = score_stage_minus2_1(signals)
        print(f"[PPWCS DEBUG] Stage -2.1 score: {stage_minus2_1_score}")
        ppwcs_structure += stage_minus2_1_score

        # --- STAGE -2.2: News/Tag Analysis (Updated Tags) ---
        tag = signals.get("event_tag")
        tag_scores = {
            "listing": 10, 
            "partnership": 10, 
            "presale": 5, 
            "cex_listed": 5, 
            "airdrop": 0,  # pomocniczy do Stage 1g
            "mint": 0,     # neutralne
            "burn": 0,     # neutralne  
            "lock": 0,     # neutralne
            "exploit": -15, # blokujące
            "rug": -15,    # blokujące
            "delisting": -15, # blokujące
            "drama": -10,
            "unlock": -10
        }

        if tag and isinstance(tag, str):
            tag_lower = tag.lower()
            if tag_lower in tag_scores:
                tag_score = tag_scores[tag_lower]
                print(f"[PPWCS DEBUG] Event tag: {tag} → {tag_score}")
                ppwcs_structure += tag_score
                
                # Blokujące tagi - zwróć bardzo niski score
                if tag_score <= -15:
                    return max(0, ppwcs_structure), 0, 0

        # --- COMBO VOLUME INFLOW BONUS ---
        if signals.get("combo_volume_inflow") is True:
            ppwcs_structure += 5
            print(f"[PPWCS DEBUG] Combo volume+inflow: +5")

        # --- STAGE -1: Compression Filter ---
        # USUNIĘTE: compressed już liczony w Stage -2.1 jako pełnoprawny detektor
        # Nie dodajemy duplikatu bonusu

        # --- Pure Accumulation Bonus ---
        # Whale + DEX inflow bez social spike - FIXED LOGIC
        if (signals.get("whale_activity") is True and 
            signals.get("dex_inflow") is True and 
            signals.get("social_spike") is not True):
            ppwcs_structure += 5
            print(f"[PPWCS DEBUG] Pure accumulation bonus: +5")

        # --- RSI FLATLINE QUALITY BONUS ---
        if signals.get("RSI_flatline") is True:
            ppwcs_quality += 5
            print(f"[PPWCS DEBUG] RSI flatline quality: +5")

        # --- STAGE 1G: Breakout Detection (Version 2.0) ---
        if signals.get("stage1g_active") is True:
            stage1g_score = score_stage_1g(signals)
            ppwcs_quality += stage1g_score
            print(f"[PPWCS DEBUG] Stage 1g score: {stage1g_score}")

        final_score = ppwcs_structure + ppwcs_quality
        print(f"[PPWCS DEBUG] Final: structure={ppwcs_structure}, quality={ppwcs_quality}, total={final_score}")
        
        return max(0, final_score), max(0, ppwcs_structure), max(0, ppwcs_quality)
        
    except Exception as e:
        print(f"❌ Error computing PPWCS: {e}")
        return 0, 0, 0

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
            
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
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
            'timestamp': datetime.now(timezone.utc).isoformat()
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
            
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        
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
            
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
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
