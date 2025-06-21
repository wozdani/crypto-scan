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

def send_alert(symbol, ppwcs, checklist_score, checklist_summary, signals):
    """
    Enhanced alert function with checklist_score integration
    Changes alert content based on structure quality
    """
    try:
        alert_lines = []

        # Header with enhanced structure assessment
        alert_lines.append(f"🚨 **PRE-PUMP ALERT** – {symbol}")
        alert_lines.append(f"PPWCS Score: {ppwcs}/100")
        alert_lines.append(f"Checklist Score: {checklist_score}/100")

        # Structure quality assessment
        if checklist_score >= 70:
            alert_lines.append("✅ Struktura: bardzo silna (setup high-confidence)")
        elif checklist_score >= 50:
            alert_lines.append("⚠️ Struktura: akceptowalna, ale warto monitorować")
        else:
            alert_lines.append("❗ Uwaga: słaba struktura – możliwy fałszywy sygnał")

        # Active signals section
        alert_lines.append("\n📡 Aktywne sygnały:")
        for k, v in signals.items():
            if isinstance(v, bool) and v:
                alert_lines.append(f"• {k}")
            elif isinstance(v, str) and v.strip():
                alert_lines.append(f"• {k}: {v}")
        
        # Structure setup summary
        if checklist_summary:
            alert_lines.append("\n🧠 Struktura setupu:")
            # Show first 8 conditions, then summarize rest
            if len(checklist_summary) <= 8:
                alert_lines.append(" + ".join(checklist_summary))
            else:
                main_conditions = " + ".join(checklist_summary[:6])
                additional_count = len(checklist_summary) - 6
                alert_lines.append(f"{main_conditions} + {additional_count} more")

        # Add chart link
        alert_lines.append(f"\n🔗 Sprawdź wykres: https://www.bybit.com/en-US/trade/spot/{symbol}")
        
        # Add timestamp
        alert_lines.append(f"\n🕒 UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

        # Prepare final message
        final_msg = "\n".join(alert_lines)
        
        # Send to Telegram if configured
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if bot_token and chat_id:
            # Escape Markdown special characters to prevent 400 errors
            escaped_msg = final_msg.replace('*', '\\*').replace('_', '\\_').replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)')
            
            payload = {
                "chat_id": chat_id,
                "text": escaped_msg,
                "parse_mode": "Markdown"
            }

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            
            print(f"✅ Enhanced alert sent for {symbol}")
            print(final_msg)
            return True
        else:
            # Fallback to console output
            print("📢 ENHANCED ALERT (Telegram not configured):")
            print(final_msg)
            return True
            
    except Exception as e:
        logger.error(f"Error sending enhanced alert for {symbol}: {e}")
        return False

def send_telegram_alert(token, ppwcs_score, stage_signals, tp_forecast, stage1g_trigger_type=None, 
                       gpt_feedback=None, feedback_score=None, is_update=False, new_signals=None, update_reason=None):
    """
    Enhanced Telegram alert that uses new send_alert function with checklist integration
    Maintained for backward compatibility
    """
    try:
        # Extract checklist data from signals
        checklist_score = stage_signals.get('checklist_score', 0)
        checklist_summary = stage_signals.get('checklist_summary', [])
        
        # Use new enhanced alert function
        success = send_alert(token, ppwcs_score, checklist_score, checklist_summary, stage_signals)
        
        # Add TP forecast and GPT feedback for legacy compatibility
        if success and (tp_forecast or gpt_feedback):
            additional_lines = []
            
            if tp_forecast:
                additional_lines.append("\n🎯 TP Forecast:")
                additional_lines.append(f"• TP1: +{tp_forecast['TP1']}%")
                additional_lines.append(f"• TP2: +{tp_forecast['TP2']}%") 
                additional_lines.append(f"• TP3: +{tp_forecast['TP3']}%")
                additional_lines.append(f"• Trailing TP: +{tp_forecast['TrailingTP']}%")
            
            if gpt_feedback and ppwcs_score >= 80:
                icon = get_feedback_icon(feedback_score or 0)
                additional_lines.append(f"\n🤖 GPT Feedback {icon}:\n{gpt_feedback}")
            
            if additional_lines:
                additional_msg = "\n".join(additional_lines)
                
                bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
                chat_id = os.getenv('TELEGRAM_CHAT_ID')
                
                if bot_token and chat_id:
                    payload = {
                        "chat_id": chat_id,
                        "text": additional_msg,
                        "parse_mode": "Markdown"
                    }
                    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    requests.post(url, data=payload, timeout=10)
        
        return success
        
    except Exception as e:
        print(f"❌ Error sending Telegram alert: {e}")
        return False

def process_alert(token, ppwcs_score, signals, gpt_analysis=None):
    """Main alert processing function with dynamic updates"""
    try:
        # Extract checklist data for alert generation
        checklist_score = signals.get('checklist_score', 0)
        checklist_summary = signals.get('checklist_summary', [])
        
        # Send alert using the working send_alert function
        alert_success = send_alert(token, ppwcs_score, checklist_score, checklist_summary, signals)
        
        return alert_success
            
    except Exception as e:
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