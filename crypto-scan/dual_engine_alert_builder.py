#!/usr/bin/env python3
"""
Dual Engine Alert Builder - Specialized alert system for separated TJDE + Stealth decisions
🎯 New Alert Logic: Trend Alert | Stealth Alert | Hybrid Alert | Watch
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def build_dual_engine_alert(result: Dict) -> Optional[Dict]:
    """
    Build alert message for dual engine result
    
    Args:
        result: Dual engine decision result
        
    Returns:
        Complete alert data or None if no alert needed
    """
    symbol = result["symbol"]
    final_decision = result.get("final_decision", "wait")
    alert_type = result.get("alert_type", "[⏳ WAIT]")
    
    # Check if alert should be generated
    alert_eligible_decisions = [
        "hybrid_alert", "trend_alert", "stealth_alert", 
        "dual_watch", "trend_watch", "stealth_watch"
    ]
    
    if final_decision not in alert_eligible_decisions:
        print(f"[DUAL ALERT] {symbol}: Decision '{final_decision}' - no alert needed")
        return None
    
    print(f"[DUAL ALERT] {symbol}: Building {alert_type} alert")
    
    # Format message components
    message_parts = format_dual_alert_message(result)
    
    # Determine priority
    priority_score = result.get("priority_boost", 0.0)
    combined_priority = result.get("combined_priority", "low")
    
    # Calculate delay based on priority
    if final_decision == "hybrid_alert":
        delay_seconds = 0  # Immediate
        priority_level = "CRITICAL"
    elif final_decision in ["trend_alert", "stealth_alert"]:
        delay_seconds = 5  # High priority
        priority_level = "HIGH"
    elif "watch" in final_decision:
        delay_seconds = 30  # Medium priority
        priority_level = "MEDIUM"
    else:
        delay_seconds = 60  # Low priority
        priority_level = "LOW"
    
    # Build complete alert
    alert_data = {
        "symbol": symbol,
        "alert_type": alert_type,
        "final_decision": final_decision,
        "priority_level": priority_level,
        "delay_seconds": delay_seconds,
        "priority_boost": priority_score,
        
        # Scores
        "tjde_score": result.get("tjde_score", 0.0),
        "stealth_score": result.get("stealth_score", 0.0),
        
        # Message
        "header": message_parts["header"],
        "content": message_parts["content"],
        "footer": message_parts["footer"],
        
        # Context
        "market_phase": result.get("market_phase", "unknown"),
        "reasoning": result.get("reasoning", []),
        "timestamp": datetime.now().isoformat()
    }
    
    return alert_data


def format_dual_alert_message(result: Dict) -> Dict:
    """Format message components for dual engine alert"""
    symbol = result["symbol"]
    tjde_score = result.get("tjde_score", 0.0)
    stealth_score = result.get("stealth_score", 0.0)
    final_decision = result.get("final_decision", "wait")
    alert_type = result.get("alert_type", "[⏳ WAIT]")
    
    # Build header with appropriate emoji
    priority_emoji = get_priority_emoji(final_decision)
    header = f"{priority_emoji} {alert_type} {symbol}"
    
    # Build core content
    score_line = f"TJDE: {tjde_score:.3f} | Stealth: {stealth_score:.3f}"
    
    # Engine analysis
    tjde_analysis = get_tjde_analysis(result)
    stealth_analysis = get_stealth_analysis(result)
    
    content_lines = [score_line]
    if tjde_analysis:
        content_lines.append(f"📈 {tjde_analysis}")
    if stealth_analysis:
        content_lines.append(f"🔍 {stealth_analysis}")
    
    # Add decision reasoning (top 2)
    reasoning = result.get("reasoning", [])
    if reasoning:
        content_lines.extend([f"• {reason}" for reason in reasoning[:2]])
    
    content = "\n".join(content_lines)
    
    # Build footer
    market_phase = result.get("market_phase", "unknown")
    timestamp = datetime.now().strftime("%H:%M")
    footer = f"Phase: {market_phase} | {timestamp}"
    
    return {
        "header": header,
        "content": content,
        "footer": footer
    }


def get_priority_emoji(final_decision: str) -> str:
    """Get priority emoji for decision"""
    emoji_map = {
        "hybrid_alert": "🚨",
        "trend_alert": "🔥",
        "stealth_alert": "🕵️",
        "dual_watch": "👀",
        "trend_watch": "📈",
        "stealth_watch": "🔍",
        "wait": "⏳"
    }
    return emoji_map.get(final_decision, "🎯")


def get_tjde_analysis(result: Dict) -> str:
    """Generate TJDE analysis summary"""
    tjde_score = result.get("tjde_score", 0.0)
    tjde_decision = result.get("tjde_decision", "wait")
    
    if tjde_score >= 0.7:
        return f"Strong trend setup ({tjde_decision})"
    elif tjde_score >= 0.4:
        return f"Moderate trend potential ({tjde_decision})"
    elif tjde_score > 0.0:
        return f"Weak trend signals ({tjde_decision})"
    else:
        return ""


def get_stealth_analysis(result: Dict) -> str:
    """Generate Stealth analysis summary"""
    stealth_score = result.get("stealth_score", 0.0)
    stealth_decision = result.get("stealth_decision", "none")
    
    if stealth_score >= 0.75:
        return f"Strong smart money activity ({stealth_decision})"
    elif stealth_score >= 0.5:
        return f"Moderate smart money signals ({stealth_decision})"
    elif stealth_score > 0.0:
        return f"Weak stealth signals ({stealth_decision})"
    else:
        return ""


def send_dual_engine_alert(alert_data: Dict) -> bool:
    """
    Send dual engine alert via Telegram
    
    Args:
        alert_data: Formatted alert data
        
    Returns:
        True if sent successfully
    """
    try:
        # Build complete message
        message = f"{alert_data['header']}\n\n{alert_data['content']}\n\n{alert_data['footer']}"
        
        # Log alert
        log_dual_alert(alert_data)
        
        # Send via Telegram (implement actual sending)
        success = send_telegram_alert(message, alert_data)
        
        if success:
            print(f"[DUAL ALERT SENT] {alert_data['symbol']}: {alert_data['alert_type']}")
        else:
            print(f"[DUAL ALERT FAILED] {alert_data['symbol']}: Telegram send failed")
            
        return success
        
    except Exception as e:
        logger.error(f"Failed to send dual engine alert: {e}")
        return False


def send_telegram_alert(message: str, alert_data: Dict) -> bool:
    """Send alert via Telegram - implement with your bot token"""
    # This is a placeholder - implement with actual Telegram bot
    print(f"[TELEGRAM MOCK] Sending: {message}")
    return True


def log_dual_alert(alert_data: Dict):
    """Log dual engine alert"""
    try:
        log_file = "data/dual_engine_alerts.jsonl"
        os.makedirs("data", exist_ok=True)
        
        log_entry = {
            "symbol": alert_data["symbol"],
            "timestamp": alert_data["timestamp"],
            "final_decision": alert_data["final_decision"],
            "priority_level": alert_data["priority_level"],
            "tjde_score": alert_data["tjde_score"],
            "stealth_score": alert_data["stealth_score"],
            "delay_seconds": alert_data["delay_seconds"]
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    except Exception as e:
        logger.error(f"Failed to log dual alert: {e}")


def should_send_alert(result: Dict) -> bool:
    """
    Determine if alert should be sent based on dual engine result
    
    Args:
        result: Dual engine decision result
        
    Returns:
        True if alert should be sent
    """
    final_decision = result.get("final_decision", "wait")
    tjde_score = result.get("tjde_score", 0.0)
    stealth_score = result.get("stealth_score", 0.0)
    
    # Always alert on hybrid
    if final_decision == "hybrid_alert":
        return True
    
    # Alert on strong individual engine signals
    if final_decision == "trend_alert" and tjde_score >= 0.7:
        return True
        
    if final_decision == "stealth_alert" and stealth_score >= 0.75:
        return True
    
    # Alert on watch conditions with high scores
    if "watch" in final_decision:
        if tjde_score >= 0.5 or stealth_score >= 0.6:
            return True
    
    return False


# === GLOBAL CONVENIENCE FUNCTIONS ===

def process_dual_engine_alert(result: Dict) -> bool:
    """
    Complete dual engine alert processing
    
    Args:
        result: Dual engine decision result
        
    Returns:
        True if alert was processed and sent
    """
    if not should_send_alert(result):
        return False
    
    alert_data = build_dual_engine_alert(result)
    if not alert_data:
        return False
    
    return send_dual_engine_alert(alert_data)


def get_alert_statistics() -> Dict:
    """Get dual engine alert statistics"""
    try:
        log_file = "data/dual_engine_alerts.jsonl"
        if not os.path.exists(log_file):
            return {"total_alerts": 0}
        
        with open(log_file, 'r') as f:
            alerts = [json.loads(line) for line in f]
        
        stats = {
            "total_alerts": len(alerts),
            "hybrid_alerts": len([a for a in alerts if a["final_decision"] == "hybrid_alert"]),
            "trend_alerts": len([a for a in alerts if a["final_decision"] == "trend_alert"]),
            "stealth_alerts": len([a for a in alerts if a["final_decision"] == "stealth_alert"]),
            "last_24h": len([a for a in alerts if (datetime.now() - datetime.fromisoformat(a["timestamp"])).total_seconds() < 86400])
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get alert statistics: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("Dual Engine Alert Builder - Separated TJDE + Stealth Alert System")
    print("✅ Trend Alerts: Independent TJDE analysis")
    print("✅ Stealth Alerts: Independent smart money detection")
    print("✅ Hybrid Alerts: Combined signal alignment")
    print("✅ Priority System: Context-aware alert timing")