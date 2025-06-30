"""
TJDE Trend-Mode Alert System
Dedicated alert system for TJDE scores - completely independent from PPWCS
"""
import os
import requests
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

def send_tjde_trend_alert(alert_data: Dict) -> bool:
    """
    Send dedicated TJDE trend-mode alert via Telegram
    Completely separate from PPWCS alert system
    
    Args:
        alert_data: Dictionary containing TJDE alert information
        
    Returns:
        bool: True if alert sent successfully
    """
    try:
        symbol = alert_data.get("symbol", "UNKNOWN")
        tjde_score = alert_data.get("tjde_score", 0.0)
        tjde_decision = alert_data.get("tjde_decision", "unknown")
        original_decision = alert_data.get("original_decision", tjde_decision)
        alert_level = alert_data.get("alert_level", 2)
        reasoning = alert_data.get("reasoning", [])
        price = alert_data.get("price", 0.0)
        volume_24h = alert_data.get("volume_24h", 0.0)
        
        # 🔒 CRITICAL FIX: Nie wysyłaj alertów dla "unknown" decision
        if tjde_decision in ["unknown", "none", None, ""]:
            print(f"[TJDE ALERT BLOCK] {symbol}: Decision is '{tjde_decision}' - alert blocked to prevent false signals")
            return False
        
        # Build alert message
        alert_lines = []
        
        # Header with TJDE focus
        if alert_level >= 3:
            alert_lines.append(f"🔥 **TJDE HIGH ALERT** – {symbol}")
        elif alert_level >= 2:
            alert_lines.append(f"⚠️ **TJDE ALERT** – {symbol}")
        else:
            alert_lines.append(f"📊 **TJDE SIGNAL** – {symbol}")
        
        # TJDE Score and Decision
        alert_lines.append(f"TJDE Score: {tjde_score:.3f}")
        alert_lines.append(f"Decision: {tjde_decision.upper()}")
        
        # Show decision enhancement if changed
        if original_decision != tjde_decision:
            alert_lines.append(f"Enhanced: {original_decision} → {tjde_decision}")
        
        # Alert level indicator
        level_indicators = {1: "🟡 Low", 2: "🟠 Medium", 3: "🔴 High"}
        alert_lines.append(f"Alert Level: {level_indicators.get(alert_level, '⚪ Unknown')}")
        
        # Market data
        alert_lines.append(f"Price: ${price:.6f}")
        if volume_24h > 0:
            volume_str = f"${volume_24h:,.0f}" if volume_24h >= 1000 else f"${volume_24h:.2f}"
            alert_lines.append(f"Volume 24h: {volume_str}")
        
        # Reasoning (why alert was triggered)
        if reasoning:
            alert_lines.append("\n🧠 Alert Reasoning:")
            for reason in reasoning[:3]:  # Show max 3 reasons
                alert_lines.append(f"• {reason}")
            
            if len(reasoning) > 3:
                alert_lines.append(f"• +{len(reasoning) - 3} more reasons...")
        
        # Trading link
        alert_lines.append(f"\n🔗 Trade: https://www.bybit.com/en-US/trade/spot/{symbol}")
        
        # Timestamp
        alert_lines.append(f"\n🕒 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        # Add trend-mode identifier
        alert_lines.append("\n📈 TJDE Trend-Mode System")
        
        # Prepare final message
        final_msg = "\n".join(alert_lines)
        
        # Send to Telegram if configured
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if bot_token and chat_id:
            # Escape Markdown special characters
            escaped_msg = final_msg.replace('*', '\\*').replace('_', '\\_').replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)')
            
            payload = {
                "chat_id": chat_id,
                "text": escaped_msg,
                "parse_mode": "Markdown"
            }

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            
            print(f"✅ TJDE alert sent for {symbol}")
            print(final_msg)
            return True
        else:
            # Fallback to console output
            print("📢 TJDE ALERT (Telegram not configured):")
            print(final_msg)
            return True
            
    except Exception as e:
        logging.error(f"Error sending TJDE alert for {symbol}: {e}")
        print(f"❌ TJDE alert failed for {symbol}: {e}")
        return False

def log_tjde_alert_history(alert_data: Dict) -> bool:
    """
    Log TJDE alert to history file for analysis
    
    Args:
        alert_data: Alert data dictionary
        
    Returns:
        bool: True if logged successfully
    """
    try:
        os.makedirs("logs", exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": alert_data.get("symbol"),
            "tjde_score": alert_data.get("tjde_score"),
            "tjde_decision": alert_data.get("tjde_decision"),
            "alert_level": alert_data.get("alert_level"),
            "reasoning_count": len(alert_data.get("reasoning", [])),
            "price": alert_data.get("price"),
            "volume_24h": alert_data.get("volume_24h"),
            "alert_type": "tjde_trend_mode"
        }
        
        # Append to JSONL log file
        with open("logs/tjde_alerts_history.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return True
        
    except Exception as e:
        logging.error(f"Error logging TJDE alert history: {e}")
        return False

def check_tjde_alert_cooldown(symbol: str, cooldown_minutes: int = 60) -> bool:
    """
    Check if symbol is in alert cooldown period
    
    Args:
        symbol: Trading symbol
        cooldown_minutes: Cooldown period in minutes
        
    Returns:
        bool: True if in cooldown (should skip alert)
    """
    try:
        cooldown_file = "data/tjde_cooldowns.json"
        
        if not os.path.exists(cooldown_file):
            return False
        
        with open(cooldown_file, 'r') as f:
            cooldowns = json.load(f)
        
        if symbol not in cooldowns:
            return False
        
        last_alert_time = datetime.fromisoformat(cooldowns[symbol])
        time_diff = datetime.now() - last_alert_time
        
        return time_diff.total_seconds() < (cooldown_minutes * 60)
        
    except Exception as e:
        logging.error(f"Error checking TJDE cooldown for {symbol}: {e}")
        return False

def set_tjde_alert_cooldown(symbol: str) -> bool:
    """
    Set alert cooldown for symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        bool: True if cooldown set successfully
    """
    try:
        os.makedirs("data", exist_ok=True)
        cooldown_file = "data/tjde_cooldowns.json"
        
        # Load existing cooldowns
        cooldowns = {}
        if os.path.exists(cooldown_file):
            with open(cooldown_file, 'r') as f:
                cooldowns = json.load(f)
        
        # Set current time as cooldown
        cooldowns[symbol] = datetime.now().isoformat()
        
        # Save updated cooldowns
        with open(cooldown_file, 'w') as f:
            json.dump(cooldowns, f, indent=2)
        
        return True
        
    except Exception as e:
        logging.error(f"Error setting TJDE cooldown for {symbol}: {e}")
        return False

def send_tjde_trend_alert_with_cooldown(alert_data: Dict) -> bool:
    """
    Send TJDE alert with automatic cooldown management
    
    Args:
        alert_data: Alert data dictionary
        
    Returns:
        bool: True if alert sent successfully
    """
    symbol = alert_data.get("symbol", "UNKNOWN")
    
    # Check cooldown
    if check_tjde_alert_cooldown(symbol):
        print(f"[TJDE COOLDOWN] {symbol}: Alert skipped due to cooldown")
        return False
    
    # Send alert
    alert_success = send_tjde_trend_alert(alert_data)
    
    if alert_success:
        # Set cooldown and log
        set_tjde_alert_cooldown(symbol)
        log_tjde_alert_history(alert_data)
        print(f"[TJDE ALERT] {symbol}: Alert sent and cooldown set")
    
    return alert_success

def get_tjde_alert_stats() -> Dict:
    """
    Get statistics about TJDE alerts
    
    Returns:
        dict: Alert statistics
    """
    try:
        stats = {
            "total_alerts": 0,
            "alerts_today": 0,
            "avg_score": 0.0,
            "level_distribution": {1: 0, 2: 0, 3: 0},
            "top_symbols": {}
        }
        
        log_file = "logs/tjde_alerts_history.jsonl"
        if not os.path.exists(log_file):
            return stats
        
        today = datetime.now().date()
        scores = []
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    stats["total_alerts"] += 1
                    
                    # Check if today
                    alert_date = datetime.fromisoformat(entry["timestamp"]).date()
                    if alert_date == today:
                        stats["alerts_today"] += 1
                    
                    # Score tracking
                    score = entry.get("tjde_score", 0.0)
                    scores.append(score)
                    
                    # Level distribution
                    level = entry.get("alert_level", 2)
                    if level in stats["level_distribution"]:
                        stats["level_distribution"][level] += 1
                    
                    # Symbol tracking
                    symbol = entry.get("symbol", "UNKNOWN")
                    stats["top_symbols"][symbol] = stats["top_symbols"].get(symbol, 0) + 1
                    
                except json.JSONDecodeError:
                    continue
        
        # Calculate average score
        if scores:
            stats["avg_score"] = sum(scores) / len(scores)
        
        # Top 5 symbols
        stats["top_symbols"] = dict(sorted(stats["top_symbols"].items(), key=lambda x: x[1], reverse=True)[:5])
        
        return stats
        
    except Exception as e:
        logging.error(f"Error getting TJDE alert stats: {e}")
        return {"error": str(e)}