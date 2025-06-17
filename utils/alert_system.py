"""
Alert System with Dynamic Updates
Handles crypto pre-pump alerts with cache-based update logic
"""

import os
import time
import json
import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple, List

# Global state for trailing scores and cooldowns
trailing_scores = {}
last_alert_time = {}

# Setup logging
logger = logging.getLogger(__name__)

def load_cooldown_tracker():
    """Load cooldown tracker from file"""
    try:
        if os.path.exists("data/cooldown_tracker.json"):
            with open("data/cooldown_tracker.json", "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading cooldown tracker: {e}")
        return {}

def save_cooldown_tracker(tracker):
    """Save cooldown tracker to file"""
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/cooldown_tracker.json", "w") as f:
            json.dump(tracker, f, default=str)
    except Exception as e:
        logger.error(f"Error saving cooldown tracker: {e}")

def get_feedback_icon(score):
    """Get quality icon based on feedback score"""
    if score >= 8:
        return "🔥"  # Excellent
    elif score >= 6:
        return "✅"  # Good
    elif score >= 4:
        return "⚠️"   # Caution
    else:
        return "❌"  # Poor

def check_cooldown(token):
    """Check if token is in cooldown period"""
    tracker = load_cooldown_tracker()
    if token in tracker:
        last_time = datetime.fromisoformat(tracker[token])
        now = datetime.now(timezone.utc)
        return (now - last_time).total_seconds() < 3600  # 1 hour cooldown
    return False

def update_cooldown(token):
    """Update cooldown timestamp for token"""
    tracker = load_cooldown_tracker()
    tracker[token] = datetime.now(timezone.utc).isoformat()
    save_cooldown_tracker(tracker)

def determine_alert_level(ppwcs_score, stage1g_quality=0):
    """
    Determine alert level based on PPWCS score and Stage 1g quality
    PPWCS 2.6: Stage 1g quality > 12 allows alerts at 60-69 range
    """
    # Stage 1g quality boost: quality > 12 allows watchlist promotion
    quality_boost = stage1g_quality > 12
    
    if ppwcs_score >= 80:
        return "strong_alert", "🚨 STRONG ALERT"
    elif ppwcs_score >= 70:
        return "pre_pump_active", "⚠️ PRE-PUMP WATCH"
    elif ppwcs_score >= 60 and quality_boost:
        return "pre_pump_active", "⚠️ PRE-PUMP WATCH (Quality Boost)"
    elif ppwcs_score >= 60:
        return "watchlist", "📊 WATCHLIST"
    else:
        return "none", ""

def calculate_stage1g_quality(signals):
    """
    Calculate Stage 1g quality score for PPWCS 2.6
    Returns: int (0-20+ points)
    """
    from utils.scoring import score_stage_1g
    
    if signals.get("stage1g_active"):
        return score_stage_1g(signals)
    return 0

def forecast_take_profit_levels(signals):
    """Generate TP levels based on signal strength"""
    base_tp = {
        'TP1': 15,
        'TP2': 35,
        'TP3': 65,
        'TrailingTP': 100
    }
    
    # Adjust based on signal strength
    signal_count = sum([1 for k, v in signals.items() if v and isinstance(v, bool)])
    multiplier = 1 + (signal_count * 0.1)
    
    return {k: round(v * multiplier) for k, v in base_tp.items()}

def calculate_risk_reward_ratio(signals, tp_forecast):
    """Calculate risk/reward ratio"""
    return round(tp_forecast['TP1'] / 8, 1)  # Assuming 8% risk

def save_alert_cache_to_file(active_alerts: dict, filename="data/alerts_cache.json"):
    """Save active alerts cache to file with datetime serialization"""
    try:
        os.makedirs("data", exist_ok=True)
        
        # Convert datetime to string (ISO format) for JSON serialization
        serializable_alerts = {}
        for symbol, data in active_alerts.items():
            serializable_alerts[symbol] = {
                "timestamp": data["timestamp"].isoformat(),
                "ppwcs": data["ppwcs"],
                "signals": data["signals"]
            }

        with open(filename, "w") as f:
            json.dump(serializable_alerts, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving alert cache: {e}")

def load_alert_cache_from_file(filename="data/alerts_cache.json"):
    """Load active alerts cache from file with datetime parsing"""
    try:
        with open(filename, "r") as f:
            raw = json.load(f)

        # Convert timestamp from ISO string back to datetime
        parsed = {}
        for symbol, data in raw.items():
            parsed[symbol] = {
                "timestamp": datetime.fromisoformat(data["timestamp"]),
                "ppwcs": data["ppwcs"],
                "signals": data["signals"]
            }

        return parsed
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.error(f"Error loading alerts cache: {e}")
        return {}

def update_alert_cache(symbol, new_signals: dict, ppwcs_score: float, active_alerts: dict):
    """
    Update alert cache entry after sending alert or alert update
    
    Args:
        symbol: e.g. 'PEPEUSDT'
        new_signals: current signals dict, e.g. {"dex_inflow": True, "spoofing": False}
        ppwcs_score: current PPWCS scoring
        active_alerts: global dict with alert cache
    
    Saves:
        - timestamp: time of last alert or update
        - ppwcs: last score
        - signals: last signal state
    """
    active_alerts[symbol] = {
        "timestamp": datetime.now(timezone.utc),
        "ppwcs": ppwcs_score,
        "signals": new_signals.copy()
    }

def should_update_alert(symbol, new_signals: dict, active_alerts: dict, ppwcs_score: float):
    """
    Determine if alert for given symbol should be updated
    
    Args:
        symbol: token symbol
        new_signals: dict with new signals (e.g. {"dex_inflow": True, "spoofing": True})
        active_alerts: dict with saved active alerts and their timestamps
        ppwcs_score: current scoring result

    Returns:
        - update_needed: bool
        - reason: str
    """
    now = datetime.now(timezone.utc)
    cooldown = timedelta(minutes=60)
    significant_keys = {"dex_inflow", "spoofing", "event_tag", "stealth_inflow"}

    # If token has no active alert - new alert
    if symbol not in active_alerts:
        return False, "no_active_alert"

    last_alert_time = active_alerts[symbol]["timestamp"]
    last_ppwcs = active_alerts[symbol].get("ppwcs", 0)

    # If an hour has passed - new alert
    if now - last_alert_time > cooldown:
        return False, "cooldown_expired"

    # If new significant signal appeared - update alert
    for key in significant_keys:
        if new_signals.get(key) and not active_alerts[symbol]["signals"].get(key):
            return True, f"new_signal: {key}"

    # If PPWCS increased significantly (by ≥5)
    if ppwcs_score - last_ppwcs >= 5:
        return True, "ppwcs_rise"

    # Otherwise - no update needed
    return False, "no_update_needed"

def send_telegram_alert(token, ppwcs_score, stage_signals, tp_forecast, stage1g_trigger_type=None, 
                       gpt_feedback=None, feedback_score=None, is_update=False, new_signals=None, update_reason=None):
    """Enhanced Telegram alert with Polish formatting, update support, and GPT feedback"""
    try:
        from utils.alert_utils import get_alert_level, get_alert_level_text, should_send_telegram_alert
        
        alert_level = get_alert_level(ppwcs_score)
        
        if not should_send_telegram_alert(alert_level):
            return False  # Score too low for alerts
            
        alert_level_text = get_alert_level_text(alert_level)
        active_signals = [k for k, v in stage_signals.items() if v and isinstance(v, bool)]
        signals_text = ', '.join(active_signals) if active_signals else "None"

        # Alert prefix based on update status
        if is_update:
            alert_prefix = f"🔄 *ALERT UPDATE* - {alert_level_text}"
            if new_signals:
                signals_text += f" | NEW: {', '.join(new_signals)}"
        else:
            alert_prefix = alert_level_text
        
        text = f"""{alert_prefix}
📈 Token: *{token}*
🧠 Score: *{ppwcs_score:.1f} / 100*"""

        # Add update reason if this is an update
        if is_update and update_reason:
            text += f"\n🔄 Update: {update_reason.replace(':', ' - ')}"

        text += f"""

🎯 TP Forecast:
• TP1: +{tp_forecast['TP1']}%
• TP2: +{tp_forecast['TP2']}%
• TP3: +{tp_forecast['TP3']}%
• Trailing TP: +{tp_forecast['TrailingTP']}%

🔬 Signals: {signals_text}"""

        if stage1g_trigger_type:
            text += f"\n🧩 Trigger: {stage1g_trigger_type}"

        # Add GPT feedback for strong alerts (PPWCS >= 80)
        if gpt_feedback and ppwcs_score >= 80:
            icon = get_feedback_icon(feedback_score or 0)
            text += f"\n\n🤖 GPT Feedback {icon}:\n{gpt_feedback}"

        text += f"\n\n🕒 UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"

        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.error("Telegram credentials not configured")
            return False

        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        
        print(f"✅ Alert sent for {token}")
        update_cooldown(token)
        return True
        
    except Exception as e:
        print(f"❌ Error sending Telegram alert: {e}")
        return False

def process_alert(token, ppwcs_score, signals, gpt_analysis=None):
    """Main alert processing function with dynamic updates"""
    try:
        # Load active alerts cache
        active_alerts = load_alert_cache_from_file()
        
        # Determine if alert should be sent
        if ppwcs_score < 70:
            # Only log to watchlist for scores 60-69
            if ppwcs_score >= 60:
                log_to_watchlist(token, ppwcs_score, signals)
            return False
        
        # Check if update is needed
        should_update, update_reason = should_update_alert(token, signals, active_alerts, ppwcs_score)
        
        if should_update:
            # Update existing alert
            new_signals_list = []
            for key in ["dex_inflow", "spoofing", "event_tag", "stealth_inflow"]:
                if signals.get(key) and not active_alerts[token]["signals"].get(key):
                    new_signals_list.append(key)
            
            # Generate TP forecast
            tp_forecast = forecast_take_profit_levels(signals)
            stage1g_trigger_type = signals.get('stage1g_trigger_type') if signals.get('stage1g_active') else None
            
            # Send updated alert
            feedback_score = signals.get('feedback_score')
            success = send_telegram_alert(
                token, ppwcs_score, signals, tp_forecast, stage1g_trigger_type, 
                gpt_analysis, feedback_score, is_update=True, new_signals=new_signals_list,
                update_reason=update_reason
            )
            
            if success:
                # Update cache
                update_alert_cache(token, signals, ppwcs_score, active_alerts)
                save_alert_cache_to_file(active_alerts)
                
                print(f"🔄 UPDATED alert sent for {token} (Score: {ppwcs_score}, Reason: {update_reason})")
                return True
            else:
                print(f"❌ Failed to send updated alert for {token}")
                return False
                
        elif token not in active_alerts:
            # New alert - check traditional cooldown
            if check_cooldown(token):
                print(f"⏱️ {token} in cooldown period, skipping alert")
                return False
                
            # Generate TP forecast
            tp_forecast = forecast_take_profit_levels(signals)
            stage1g_trigger_type = signals.get('stage1g_trigger_type') if signals.get('stage1g_active') else None
            
            # Send new alert
            feedback_score = signals.get('feedback_score')
            success = send_telegram_alert(
                token, ppwcs_score, signals, tp_forecast, stage1g_trigger_type, 
                gpt_analysis, feedback_score, is_update=False
            )
            
            if success:
                # Add to active alerts cache
                update_alert_cache(token, signals, ppwcs_score, active_alerts)
                save_alert_cache_to_file(active_alerts)
                
                print(f"📢 NEW alert sent for {token} (Score: {ppwcs_score})")
                return True
            else:
                print(f"❌ Failed to send new alert for {token}")
                return False
        else:
            print(f"⏭️ {token} has active alert but no significant updates ({update_reason})")
            return False
            
    except Exception as e:
        print(f"❌ Error processing alert for {token}: {e}")
        return False

def log_to_watchlist(token, ppwcs_score, signals):
    """Log watchlist entries (60-69 score) to CSV"""
    try:
        watchlist_file = os.path.join("data", "watchlist.csv")
        os.makedirs(os.path.dirname(watchlist_file), exist_ok=True)
        
        now = datetime.now(timezone.utc)
        active_signals = [k for k, v in signals.items() if v and isinstance(v, bool)]
        
        # Create header if file doesn't exist
        if not os.path.exists(watchlist_file):
            with open(watchlist_file, "w") as f:
                f.write("timestamp,token,ppwcs_score,active_signals\n")
        
        # Append entry
        with open(watchlist_file, "a") as f:
            f.write(f"{now.isoformat()},{token},{ppwcs_score},\"{';'.join(active_signals)}\"\n")
            
        print(f"📋 {token} added to watchlist (Score: {ppwcs_score})")
        
    except Exception as e:
        print(f"❌ Error logging to watchlist: {e}")

def get_alert_statistics():
    """Get alert statistics for dashboard"""
    try:
        active_alerts = load_alert_cache_from_file()
        now = datetime.now(timezone.utc)
        
        # Count active alerts (within 2 hours)
        active_count = 0
        for symbol, data in active_alerts.items():
            if now - data["timestamp"] <= timedelta(hours=2):
                active_count += 1
        
        return {
            "active_alerts": active_count,
            "total_cached": len(active_alerts)
        }
    except Exception as e:
        logger.error(f"Error getting alert statistics: {e}")
        return {"active_alerts": 0, "total_cached": 0}