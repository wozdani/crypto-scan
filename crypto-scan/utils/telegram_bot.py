import os
import json
import requests
from datetime import datetime, timezone
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


def get_feedback_summary_with_reasons(weights_before: dict, weights_after: dict, reasons: dict) -> str:
    """
    Generate formatted feedback summary with weight change explanations
    
    Args:
        weights_before: Original weights
        weights_after: Updated weights
        reasons: Explanations for changes
        
    Returns:
        Formatted string for Telegram
    """
    try:
        summary_lines = []
        
        for key in weights_after:
            before = weights_before.get(key, 0)
            after = weights_after[key]
            delta = after - before
            
            if abs(delta) >= 0.01:  # Only show significant changes
                arrow = "🔼" if delta > 0 else "🔽"
                reason = reasons.get(key, "Automatyczna korekta na podstawie analizy")
                
                # Format feature name for display
                display_name = key.replace("_", " ").title()
                
                summary_lines.append(
                    f"- {arrow} *{display_name}*: {before:.3f} → {after:.3f}\n   _{reason}_"
                )
        
        if summary_lines:
            from datetime import datetime
            header = f"\n📊 *Feedback Loop - Aktualizacja Wag* ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}):"
            return header + "\n" + "\n".join(summary_lines)
        
        return ""
        
    except Exception as e:
        print(f"❌ Error generating feedback summary: {e}")
        return ""


def send_trend_alert(message: str, feedback_summary: str = "") -> bool:
    """Send trend-mode alert to Telegram with optional feedback summary"""
    try:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_TREND_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
        
        if not token or not chat_id:
            print("⚠️ Telegram configuration missing (token/chat_id)")
            return False
        
        # Combine message with feedback summary
        full_message = message
        if feedback_summary:
            full_message += f"\n\n{feedback_summary}"
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": full_message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, data=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ Trend alert sent successfully")
            return True
        elif response.status_code == 400:
            # Try without markdown if parsing fails
            payload["parse_mode"] = None
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code == 200:
                print("✅ Trend alert sent (fallback mode)")
                return True
            else:
                print(f"❌ Telegram error even without markdown: {response.status_code}")
                return False
        else:
            print(f"❌ Telegram error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception sending trend alert: {e}")
        return False

def log_alert(message, status, error=None):
    """Log alert attempts to file"""
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

def format_alert(symbol, score, tags, compressed, stage1g_active, gpt_analysis=None):
    """Format alert message with detailed information and optional GPT analysis"""
    tag_line = ", ".join(tags) if tags else "Brak"
    compression_status = "✅" if compressed.get('momentum', False) and compressed.get('technical_alignment', False) else "❌"
    stage1g_status = "✅" if stage1g_active else "❌"

    message = f"""🚨 *{symbol}* – *PPWCS: {score}*
*Tags:* `{tag_line}`
*Compressed:* {compression_status}
*Stage 1g:* {stage1g_status}"""

    # Add GPT analysis if available
    if gpt_analysis and score >= 80:
        risk = gpt_analysis.get('risk_assessment', 'Unknown')
        confidence = gpt_analysis.get('confidence_level', 0)
        prediction = gpt_analysis.get('price_prediction', 'Unknown')
        entry = gpt_analysis.get('entry_recommendation', 'Unknown')
        
        message += f"""

🤖 *AI Analysis:*
*Risk:* {risk} | *Confidence:* {confidence}%
*Prediction:* {prediction}
*Entry:* {entry}"""

    message += "\n\n#PrePumpAlert"
    
    return message.strip()

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
