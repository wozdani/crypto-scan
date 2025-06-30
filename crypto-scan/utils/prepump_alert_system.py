"""
Pre-Pump Alert System - Early Entry Detection
Specialized alert system for pre-pump phase using Unified TJDE Engine
"""

import os
import requests
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

def send_prepump_alert(alert_data: Dict) -> bool:
    """
    Send dedicated PRE-PUMP EARLY ENTRY alert via Telegram
    
    Args:
        alert_data: Dictionary containing pre-pump alert information
        
    Returns:
        bool: True if alert sent successfully
    """
    try:
        symbol = alert_data.get("symbol", "UNKNOWN")
        final_score = alert_data.get("final_score", 0.0)
        decision = alert_data.get("decision", "unknown")
        quality_grade = alert_data.get("quality_grade", "unknown")
        market_phase = alert_data.get("market_phase", "unknown")
        components = alert_data.get("components", {})
        score_breakdown = alert_data.get("score_breakdown", {})
        
        # 🔒 CRITICAL: Only send alerts for valid pre-pump decisions
        if decision not in ["early_entry", "monitor"]:
            print(f"[PRE-PUMP ALERT BLOCK] {symbol}: Decision '{decision}' not valid for pre-pump alerts")
            return False
        
        if market_phase != "pre-pump":
            print(f"[PRE-PUMP ALERT BLOCK] {symbol}: Market phase '{market_phase}' is not pre-pump")
            return False
        
        # Build comprehensive pre-pump alert message
        alert_lines = []
        
        # Header with pre-pump focus
        if decision == "early_entry":
            alert_lines.append(f"🚨 **PRE-PUMP EARLY ENTRY** – {symbol}")
            alert_lines.append(f"🎯 **Early opportunity detected before breakout**")
        else:
            alert_lines.append(f"📊 **PRE-PUMP MONITOR** – {symbol}")
            alert_lines.append(f"👀 **Building pressure - watch for entry**")
        
        # Core metrics
        alert_lines.append(f"**TJDE Score:** {final_score:.3f}")
        alert_lines.append(f"**Quality:** {quality_grade.upper()}")
        alert_lines.append(f"**Phase:** {market_phase}")
        
        # Component breakdown for pre-pump analysis
        alert_lines.append("\n**📊 PRE-PUMP ANALYSIS:**")
        
        # Volume structure
        volume_structure = components.get("volume_structure", 0.0)
        volume_contrib = score_breakdown.get("volume_structure", {}).get("contribution", 0.0)
        alert_lines.append(f"• Volume Building: {volume_structure:.2f} (contrib: +{volume_contrib:.3f})")
        
        # CLIP confidence
        clip_confidence = components.get("clip_confidence", 0.0)
        clip_contrib = score_breakdown.get("clip_confidence", {}).get("contribution", 0.0)
        if clip_confidence > 0:
            alert_lines.append(f"• Vision AI: {clip_confidence:.2f} (contrib: +{clip_contrib:.3f})")
        
        # GPT label match
        gpt_match = components.get("gpt_label_match", 0.0)
        gpt_contrib = score_breakdown.get("gpt_label_match", {}).get("contribution", 0.0)
        if gpt_match > 0.5:
            alert_lines.append(f"• Pattern Recognition: {gpt_match:.2f} (contrib: +{gpt_contrib:.3f})")
        
        # Liquidity behavior
        liquidity = components.get("liquidity_behavior", 0.0)
        liquidity_contrib = score_breakdown.get("liquidity_behavior", {}).get("contribution", 0.0)
        alert_lines.append(f"• Liquidity Balance: {liquidity:.2f} (contrib: +{liquidity_contrib:.3f})")
        
        # Price compression (heatmap)
        compression = components.get("heatmap_window", 0.0)
        compression_contrib = score_breakdown.get("heatmap_window", {}).get("contribution", 0.0)
        alert_lines.append(f"• Price Compression: {compression:.2f} (contrib: +{compression_contrib:.3f})")
        
        # Pre-breakout structure
        structure = components.get("pre_breakout_structure", 0.0)
        structure_contrib = score_breakdown.get("pre_breakout_structure", {}).get("contribution", 0.0)
        alert_lines.append(f"• Breakout Setup: {structure:.2f} (contrib: +{structure_contrib:.3f})")
        
        # Market data if available
        price = alert_data.get("price", 0.0)
        volume_24h = alert_data.get("volume_24h", 0.0)
        
        if price > 0:
            alert_lines.append(f"\n**💰 MARKET DATA:**")
            alert_lines.append(f"• Price: ${price:.6f}")
            if volume_24h > 0:
                alert_lines.append(f"• Volume 24h: ${volume_24h:,.0f}")
        
        # Trading context
        alert_lines.append(f"\n**📈 STRATEGY:**")
        if decision == "early_entry":
            alert_lines.append(f"• Position: Consider early entry before crowd")
            alert_lines.append(f"• Risk: Moderate (pre-breakout timing)")
            alert_lines.append(f"• Reward: High (entry before momentum)")
        else:
            alert_lines.append(f"• Position: Monitor for confirmation")
            alert_lines.append(f"• Watch: Volume expansion + price action")
            alert_lines.append(f"• Entry: Wait for breakout confirmation")
        
        # Links
        alert_lines.append(f"\n**🔗 QUICK ACCESS:**")
        alert_lines.append(f"📊 [TradingView](https://www.tradingview.com/chart/?symbol=BINANCE:{symbol})")
        alert_lines.append(f"💱 [Bybit](https://www.bybit.com/trade/usdt/{symbol})")
        
        # Timestamp
        alert_lines.append(f"\n🕐 {datetime.now().strftime('%H:%M:%S UTC')}")
        
        # Join message
        final_msg = "\n".join(alert_lines)
        
        # Send via Telegram if configured
        telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        if telegram_token and telegram_chat_id:
            telegram_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            
            payload = {
                'chat_id': telegram_chat_id,
                'text': final_msg,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(telegram_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ PRE-PUMP alert sent for {symbol}")
                return True
            else:
                print(f"❌ PRE-PUMP Telegram API error: {response.status_code}")
                return False
        else:
            # Fallback to console output
            print("📢 PRE-PUMP ALERT (Telegram not configured):")
            print(final_msg)
            return True
            
    except Exception as e:
        logging.error(f"Error sending PRE-PUMP alert for {symbol}: {e}")
        print(f"❌ PRE-PUMP alert failed for {symbol}: {e}")
        return False

def log_prepump_alert_history(alert_data: Dict) -> bool:
    """
    Log PRE-PUMP alert to history file for analysis
    
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
            "final_score": alert_data.get("final_score"),
            "decision": alert_data.get("decision"),
            "market_phase": alert_data.get("market_phase"),
            "quality_grade": alert_data.get("quality_grade"),
            "components": alert_data.get("components", {}),
            "price": alert_data.get("price"),
            "volume_24h": alert_data.get("volume_24h"),
            "alert_type": "PRE_PUMP_EARLY_ENTRY"
        }
        
        # Append to history file
        with open("logs/prepump_alerts_history.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        print(f"[PRE-PUMP LOG] {alert_data.get('symbol')}: Alert logged to history")
        return True
        
    except Exception as e:
        print(f"[PRE-PUMP LOG ERROR] {alert_data.get('symbol', 'UNKNOWN')}: {e}")
        return False

def send_prepump_alert_with_cooldown(alert_data: Dict) -> bool:
    """
    Send PRE-PUMP alert with automatic cooldown management
    
    Args:
        alert_data: Alert data dictionary
        
    Returns:
        bool: True if alert sent successfully
    """
    symbol = alert_data.get("symbol", "UNKNOWN")
    
    # Check cooldown (pre-pump uses longer cooldown - 2 hours)
    if check_prepump_alert_cooldown(symbol):
        print(f"[PRE-PUMP COOLDOWN] {symbol}: Alert skipped due to cooldown")
        return False
    
    # Send alert
    alert_success = send_prepump_alert(alert_data)
    
    if alert_success:
        # Set cooldown and log
        set_prepump_alert_cooldown(symbol)
        log_prepump_alert_history(alert_data)
        print(f"[PRE-PUMP ALERT] {symbol}: Alert sent and cooldown set")
    
    return alert_success

def check_prepump_alert_cooldown(symbol: str) -> bool:
    """Check if symbol is in PRE-PUMP alert cooldown (2 hours)"""
    try:
        cooldown_file = "data/prepump_alert_cooldowns.json"
        
        if not os.path.exists(cooldown_file):
            return False
        
        with open(cooldown_file, 'r') as f:
            cooldowns = json.load(f)
        
        if symbol not in cooldowns:
            return False
        
        last_alert = datetime.fromisoformat(cooldowns[symbol])
        cooldown_hours = 2  # 2 hour cooldown for pre-pump
        time_diff = datetime.now() - last_alert
        
        return time_diff.total_seconds() < (cooldown_hours * 3600)
        
    except Exception as e:
        print(f"[PRE-PUMP COOLDOWN ERROR] {symbol}: {e}")
        return False

def set_prepump_alert_cooldown(symbol: str):
    """Set PRE-PUMP alert cooldown for symbol"""
    try:
        os.makedirs("data", exist_ok=True)
        cooldown_file = "data/prepump_alert_cooldowns.json"
        
        # Load existing cooldowns
        cooldowns = {}
        if os.path.exists(cooldown_file):
            with open(cooldown_file, 'r') as f:
                cooldowns = json.load(f)
        
        # Set cooldown
        cooldowns[symbol] = datetime.now().isoformat()
        
        # Save updated cooldowns
        with open(cooldown_file, 'w') as f:
            json.dump(cooldowns, f, indent=2)
        
        print(f"[PRE-PUMP COOLDOWN] {symbol}: 2-hour cooldown set")
        
    except Exception as e:
        print(f"[PRE-PUMP COOLDOWN ERROR] {symbol}: {e}")

def get_prepump_alert_stats() -> Dict:
    """
    Get statistics about PRE-PUMP alerts
    
    Returns:
        dict: Alert statistics
    """
    try:
        stats = {
            "total_alerts": 0,
            "early_entry_alerts": 0,
            "monitor_alerts": 0,
            "symbols_alerted": [],
            "avg_score": 0.0,
            "quality_distribution": {},
            "last_24h_alerts": 0
        }
        
        history_file = "logs/prepump_alerts_history.jsonl"
        if not os.path.exists(history_file):
            return stats
        
        scores = []
        recent_alerts = 0
        cutoff_time = datetime.now().timestamp() - (24 * 3600)  # 24 hours ago
        
        with open(history_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    stats["total_alerts"] += 1
                    
                    # Decision type
                    decision = entry.get("decision", "unknown")
                    if decision == "early_entry":
                        stats["early_entry_alerts"] += 1
                    elif decision == "monitor":
                        stats["monitor_alerts"] += 1
                    
                    # Symbol tracking
                    symbol = entry.get("symbol")
                    if symbol and symbol not in stats["symbols_alerted"]:
                        stats["symbols_alerted"].append(symbol)
                    
                    # Score tracking
                    score = entry.get("final_score", 0.0)
                    if score > 0:
                        scores.append(score)
                    
                    # Quality distribution
                    quality = entry.get("quality_grade", "unknown")
                    stats["quality_distribution"][quality] = stats["quality_distribution"].get(quality, 0) + 1
                    
                    # Recent alerts
                    timestamp = entry.get("timestamp", "")
                    if timestamp:
                        entry_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                        if entry_time > cutoff_time:
                            recent_alerts += 1
                
                except json.JSONDecodeError:
                    continue
        
        # Calculate averages
        if scores:
            stats["avg_score"] = sum(scores) / len(scores)
        
        stats["last_24h_alerts"] = recent_alerts
        
        return stats
        
    except Exception as e:
        print(f"[PRE-PUMP STATS ERROR]: {e}")
        return {"error": str(e)}