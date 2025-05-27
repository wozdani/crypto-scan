import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_alert(message):
    """Send alert message to Telegram chat"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials not configured")
        return False
        
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            # Log successful alert
            log_alert(message, "success")
            return True
        else:
            print(f"❌ Telegram API error: {response.status_code} - {response.text}")
            log_alert(message, "failed", response.text)
            return False
            
    except Exception as e:
        print(f"❌ Failed to send Telegram alert: {e}")
        log_alert(message, "error", str(e))
        return False

def log_alert(message, status, error=None):
    """Log alert attempts to file"""
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "status": status,
            "error": error
        }
        
        # Ensure alerts directory exists
        os.makedirs("data/alerts", exist_ok=True)
        
        # Load existing alerts or create new list
        alerts_file = "data/alerts/alerts_history.json"
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        else:
            alerts = []
            
        alerts.append(log_entry)
        
        # Keep only last 1000 alerts
        if len(alerts) > 1000:
            alerts = alerts[-1000:]
            
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
            
    except Exception as e:
        print(f"⚠️ Failed to log alert: {e}")

def send_system_status(status_message):
    """Send system status update"""
    formatted_message = f"🔧 *System Status*\n{status_message}"
    return send_alert(formatted_message)

def send_error_alert(error_message, context=""):
    """Send error alert with context"""
    formatted_message = f"🚨 *Error Alert*\n{error_message}"
    if context:
        formatted_message += f"\n\n*Context:* {context}"
    return send_alert(formatted_message)

def test_telegram_connection():
    """Test Telegram bot connection"""
    if not TELEGRAM_BOT_TOKEN:
        return False, "Bot token not configured"
        
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            bot_info = response.json()
            if bot_info.get("ok"):
                return True, f"Connected as @{bot_info['result']['username']}"
            else:
                return False, "Bot token invalid"
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        return False, str(e)
