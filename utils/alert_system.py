import os
import json
import requests
from datetime import datetime, timedelta
from utils.take_profit_engine import forecast_take_profit_levels, calculate_risk_reward_ratio

COOLDOWN_FILE = os.path.join("data", "cooldown_tracker.json")
COOLDOWN_MINUTES = 60

def load_cooldown_tracker():
    """Load cooldown tracker from file"""
    try:
        if os.path.exists(COOLDOWN_FILE):
            with open(COOLDOWN_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def save_cooldown_tracker(tracker):
    """Save cooldown tracker to file"""
    try:
        os.makedirs(os.path.dirname(COOLDOWN_FILE), exist_ok=True)
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(tracker, f, indent=2)
    except Exception as e:
        print(f"Error saving cooldown tracker: {e}")

def check_cooldown(token):
    """Check if token is in cooldown period"""
    tracker = load_cooldown_tracker()
    
    if token not in tracker:
        return False
        
    try:
        last_alert = datetime.fromisoformat(tracker[token])
        now = datetime.utcnow()
        return (now - last_alert).total_seconds() < (COOLDOWN_MINUTES * 60)
    except Exception:
        return False

def update_cooldown(token):
    """Update cooldown timestamp for token"""
    tracker = load_cooldown_tracker()
    tracker[token] = datetime.utcnow().isoformat()
    save_cooldown_tracker(tracker)

def determine_alert_level(ppwcs_score):
    """Determine alert level based on PPWCS score"""
    if ppwcs_score >= 80:
        return "strong_alert", "🚨 STRONG ALERT"
    elif ppwcs_score >= 70:
        return "pre_pump_active", "⚠️ PRE-PUMP WATCH"
    elif ppwcs_score >= 60:
        return "watchlist", "📊 WATCHLIST"
    else:
        return "none", ""

def format_alert_message(token, ppwcs_score, signals, tp_forecast, risk_reward):
    """Format comprehensive alert message with TP levels"""
    now = datetime.utcnow()
    alert_level, alert_emoji = determine_alert_level(ppwcs_score)
    
    # Extract active signals
    active_signals = []
    if signals.get('whale_activity'):
        active_signals.append('Whale Activity')
    if signals.get('dex_inflow'):
        active_signals.append('DEX Inflow')
    if signals.get('volume_spike'):
        active_signals.append('Volume Spike')
    if signals.get('orderbook_anomaly'):
        active_signals.append('Orderbook Anomaly')
    if signals.get('social_spike'):
        active_signals.append('Social Spike')
    if signals.get('stage1g_active'):
        trigger_type = signals.get('stage1g_trigger_type', 'unknown')
        active_signals.append(f'Stage 1G ({trigger_type})')
    if signals.get('event_tag'):
        active_signals.append(f"Event: {signals['event_tag']}")
    
    signals_text = ', '.join(active_signals) if active_signals else 'Multiple micro-signals'
    
    alert_text = f"""
{alert_emoji}
**Token:** `{token}`
**Score:** {ppwcs_score} / 100
**Confidence:** {tp_forecast.get('confidence', 'medium')}

**🎯 Take Profit Levels:**
TP1: +{tp_forecast['TP1']}% (R:R {risk_reward['RR_TP1']})
TP2: +{tp_forecast['TP2']}% (R:R {risk_reward['RR_TP2']})
TP3: +{tp_forecast['TP3']}% (R:R {risk_reward['RR_TP3']})
Trailing TP: +{tp_forecast['TrailingTP']}%

**🛡️ Risk Management:**
Stop Loss: -{risk_reward['stop_loss']}%
Position Size: {risk_reward['recommended_position_size']}

**📊 Active Signals:** {signals_text}

🕒 UTC: {now.strftime('%Y-%m-%d %H:%M')}
"""
    
    return alert_text.strip()

def send_telegram_alert(message):
    """Send alert to Telegram"""
    try:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            print("❌ Telegram credentials not configured")
            return False
            
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        return True
        
    except Exception as e:
        print(f"❌ Error sending Telegram alert: {e}")
        return False

def process_alert(token, ppwcs_score, signals, gpt_analysis=None):
    """Main alert processing function"""
    try:
        # Determine if alert should be sent
        if ppwcs_score < 70:
            # Only log to watchlist for scores 60-69
            if ppwcs_score >= 60:
                log_to_watchlist(token, ppwcs_score, signals)
            return False
            
        # Check cooldown
        if check_cooldown(token):
            print(f"⏱️ {token} in cooldown period, skipping alert")
            return False
            
        # Generate TP forecast
        tp_forecast = forecast_take_profit_levels(signals)
        risk_reward = calculate_risk_reward_ratio(signals, tp_forecast)
        
        # Format alert message
        alert_message = format_alert_message(token, ppwcs_score, signals, tp_forecast, risk_reward)
        
        # Add GPT analysis for strong alerts
        if ppwcs_score >= 80 and gpt_analysis:
            alert_message += f"\n\n🤖 **GPT Analysis:**\n{gpt_analysis}"
        
        # Send alert
        success = send_telegram_alert(alert_message)
        
        if success:
            update_cooldown(token)
            alert_level, _ = determine_alert_level(ppwcs_score)
            print(f"📢 {alert_level.upper()} sent for {token} (Score: {ppwcs_score})")
            return True
        else:
            print(f"❌ Failed to send alert for {token}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing alert for {token}: {e}")
        return False

def log_to_watchlist(token, ppwcs_score, signals):
    """Log watchlist entries (60-69 score) to CSV"""
    try:
        watchlist_file = os.path.join("data", "watchlist.csv")
        os.makedirs(os.path.dirname(watchlist_file), exist_ok=True)
        
        now = datetime.utcnow()
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
        tracker = load_cooldown_tracker()
        now = datetime.utcnow()
        
        recent_alerts = []
        for token, timestamp_str in tracker.items():
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                if (now - timestamp).total_seconds() < (24 * 3600):  # Last 24 hours
                    recent_alerts.append({
                        'token': token,
                        'timestamp': timestamp,
                        'hours_ago': round((now - timestamp).total_seconds() / 3600, 1)
                    })
            except Exception:
                continue
                
        return {
            'total_recent_alerts': len(recent_alerts),
            'recent_alerts': sorted(recent_alerts, key=lambda x: x['timestamp'], reverse=True)[:10]
        }
        
    except Exception as e:
        print(f"Error getting alert statistics: {e}")
        return {'total_recent_alerts': 0, 'recent_alerts': []}