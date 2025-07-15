#!/usr/bin/env python3
"""
DiamondWhale AI - Telegram Alert Notification System
Stage 4/7: Diamond Alert Telegram Notification

Dedykowany system alertów dla DiamondWhale AI z unikalnym stylem i brandingiem.
Alerts mają mocniejszy branding, większą wagę i oznaczenia 🧠 + 💎.
"""

import os
import requests
from datetime import datetime
from typing import Dict, Any, Optional
import json
import time


def get_telegram_credentials() -> tuple:
    """
    Pobiera credentials do Telegram Bot API
    
    Returns:
        tuple: (bot_token, chat_id)
    """
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("[DIAMOND ALERT] ⚠️ Missing Telegram credentials")
        return None, None
    
    return bot_token, chat_id


def get_dominant_detector_emoji(dominant_detector: str) -> str:
    """
    Zwraca emoji dla dominant detector
    
    Args:
        dominant_detector: Nazwa dominant detector
        
    Returns:
        str: Emoji reprezentujący detector
    """
    emoji_map = {
        "whale_ping": "🐋",
        "whaleclip": "🧠", 
        "diamond": "💎",
        "unknown": "🔍"
    }
    
    return emoji_map.get(dominant_detector.lower(), "🔍")


def format_confidence_indicator(confidence: str) -> str:
    """
    Formatuje wskaźnik confidence z emoji
    
    Args:
        confidence: Poziom confidence (HIGH/MEDIUM/LOW)
        
    Returns:
        str: Sformatowany wskaźnik confidence
    """
    confidence_map = {
        "HIGH": "🔥 HIGH",
        "MEDIUM": "⚡ MEDIUM", 
        "LOW": "⚠️ LOW",
        "ERROR": "❌ ERROR"
    }
    
    return confidence_map.get(confidence.upper(), f"❓ {confidence}")


def format_trigger_reasons(reasons: list) -> str:
    """
    Formatuje trigger reasons z odpowiednimi emoji
    
    Args:
        reasons: Lista trigger reasons
        
    Returns:
        str: Sformatowane reasons
    """
    if not reasons:
        return "• No specific triggers detected"
    
    formatted_reasons = []
    for reason in reasons:
        # Dodaj bullet point jeśli go nie ma
        if not reason.startswith("•") and not reason.startswith("🔹"):
            reason = f"• {reason}"
        formatted_reasons.append(reason)
    
    return "\n".join(formatted_reasons)


def format_score_breakdown(decision_breakdown: Dict[str, float]) -> str:
    """
    Formatuje breakdown scores dla alertu
    
    Args:
        decision_breakdown: Dictionary z breakdown scores
        
    Returns:
        str: Sformatowany breakdown
    """
    if not decision_breakdown:
        return "• No breakdown available"
    
    breakdown_lines = []
    
    # Formatuj scores z emoji
    if "whale_ping_score" in decision_breakdown:
        whale_score = decision_breakdown["whale_ping_score"]
        breakdown_lines.append(f"🐋 Whale Ping: {whale_score:.3f}")
    
    if "whaleclip_score" in decision_breakdown:
        clip_score = decision_breakdown["whaleclip_score"]
        breakdown_lines.append(f"🧠 WhaleCLIP: {clip_score:.3f}")
    
    if "diamond_score" in decision_breakdown:
        diamond_score = decision_breakdown["diamond_score"]
        breakdown_lines.append(f"💎 Diamond AI: {diamond_score:.3f}")
    
    if "fused_score" in decision_breakdown:
        fused_score = decision_breakdown["fused_score"]
        breakdown_lines.append(f"⚡ Fused Score: {fused_score:.3f}")
    
    return "\n".join(breakdown_lines) if breakdown_lines else "• No detailed breakdown"


def send_diamond_alert(token: str, chat_id: str, symbol: str, scores_dict: Dict[str, Any]) -> bool:
    """
    Wysyła dedykowany Diamond Alert na Telegram
    
    Args:
        token: Telegram bot token
        chat_id: Telegram chat ID
        symbol: Symbol tokena (np. "PENGUUSDT")
        scores_dict: Dictionary z wynikami simulate_diamond_decision()
        
    Returns:
        bool: True jeśli wysłano pomyślnie, False w przeciwnym razie
    """
    try:
        # Przygotuj podstawowe dane
        decision = scores_dict.get("decision", "UNKNOWN")
        fused_score = scores_dict.get("fused_score", 0.0)
        confidence = scores_dict.get("confidence", "UNKNOWN")
        dominant_detector = scores_dict.get("dominant_detector", "unknown")
        trigger_reasons = scores_dict.get("trigger_reasons", [])
        decision_breakdown = scores_dict.get("decision_breakdown", {})
        
        # Formatuj komponenty alertu
        confidence_indicator = format_confidence_indicator(confidence)
        dominant_emoji = get_dominant_detector_emoji(dominant_detector)
        formatted_reasons = format_trigger_reasons(trigger_reasons)
        formatted_breakdown = format_score_breakdown(decision_breakdown)
        
        # Timestamp UTC
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # === DIAMOND ALERT MESSAGE TEMPLATE ===
        message = f"""🚨💎 [DIAMOND ALERT] {symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 Fused Score: {fused_score:.3f}
📊 Decision: {decision}
🎯 Confidence: {confidence_indicator}
{dominant_emoji} Dominant: {dominant_detector.upper()}

📡 Trigger Reasons:
{formatted_reasons}

📈 Score Breakdown:
{formatted_breakdown}

🧠 Detector: DiamondWhale AI - Temporal Graph + QIRL
📅 Timestamp: {timestamp}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⛏️ On-chain pattern suggests possible whale accumulation.
💎 Advanced temporal graph analysis detected anomalies.
"""

        # Przygotuj parametry API
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        # Parametry z obsługą Markdown
        params = {
            "chat_id": chat_id,
            "text": message.strip(),
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        # Wyślij alert
        print(f"[DIAMOND ALERT] 💎 Sending alert for {symbol}...")
        print(f"[DIAMOND ALERT] Decision: {decision}, Score: {fused_score:.3f}, Confidence: {confidence}")
        
        response = requests.post(url, json=params, timeout=10)
        
        if response.status_code == 200:
            print(f"[DIAMOND ALERT] ✅ Alert sent successfully for {symbol}")
            
            # Zapisz log alertu
            save_diamond_alert_log(symbol, scores_dict, timestamp)
            return True
        else:
            print(f"[DIAMOND ALERT] ❌ Failed to send alert: {response.status_code}")
            print(f"[DIAMOND ALERT] Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"[DIAMOND ALERT] ❌ Exception during alert sending: {type(e).__name__}: {e}")
        return False


def save_diamond_alert_log(symbol: str, scores_dict: Dict[str, Any], timestamp: str):
    """
    Zapisuje log Diamond Alert do pliku
    
    Args:
        symbol: Symbol tokena
        scores_dict: Dictionary z wynikami
        timestamp: Timestamp alertu
    """
    try:
        log_entry = {
            "symbol": symbol,
            "timestamp": timestamp,
            "alert_type": "DIAMOND",
            "decision": scores_dict.get("decision"),
            "fused_score": scores_dict.get("fused_score"),
            "confidence": scores_dict.get("confidence"),
            "dominant_detector": scores_dict.get("dominant_detector"),
            "trigger_reasons": scores_dict.get("trigger_reasons", []),
            "decision_breakdown": scores_dict.get("decision_breakdown", {})
        }
        
        # Zapisz do pliku log
        log_file = "cache/diamond_alerts_log.json"
        
        # Wczytaj istniejące logi
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
        
        # Dodaj nowy log
        logs.append(log_entry)
        
        # Ogranicz do ostatnich 1000 alertów
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Zapisz zaktualizowane logi
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"[DIAMOND ALERT] 📝 Log saved for {symbol}")
        
    except Exception as e:
        print(f"[DIAMOND ALERT] ⚠️ Failed to save log: {e}")


def send_diamond_alert_auto(symbol: str, scores_dict: Dict[str, Any]) -> bool:
    """
    Automatyczne wysyłanie Diamond Alert z credentials z environment
    
    Args:
        symbol: Symbol tokena
        scores_dict: Dictionary z wynikami simulate_diamond_decision()
        
    Returns:
        bool: True jeśli wysłano pomyślnie
    """
    # Pobierz credentials
    bot_token, chat_id = get_telegram_credentials()
    
    if not bot_token or not chat_id:
        print(f"[DIAMOND ALERT] ⚠️ Cannot send alert for {symbol} - missing credentials")
        return False
    
    # Wyślij alert
    return send_diamond_alert(bot_token, chat_id, symbol, scores_dict)


def get_diamond_alert_stats() -> Dict[str, Any]:
    """
    Pobiera statystyki Diamond Alerts
    
    Returns:
        dict: Statystyki alertów
    """
    try:
        log_file = "cache/diamond_alerts_log.json"
        
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        # Oblicz statystyki
        total_alerts = len(logs)
        decisions = {}
        confidence_levels = {}
        dominant_detectors = {}
        
        for log in logs:
            # Decisions
            decision = log.get("decision", "UNKNOWN")
            decisions[decision] = decisions.get(decision, 0) + 1
            
            # Confidence levels
            confidence = log.get("confidence", "UNKNOWN")
            confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1
            
            # Dominant detectors
            detector = log.get("dominant_detector", "unknown")
            dominant_detectors[detector] = dominant_detectors.get(detector, 0) + 1
        
        # Ostatnie 24h
        current_time = time.time()
        last_24h = [log for log in logs if (current_time - time.mktime(time.strptime(log["timestamp"], '%Y-%m-%d %H:%M:%S UTC'))) < 86400]
        
        return {
            "total_alerts": total_alerts,
            "alerts_24h": len(last_24h),
            "decisions": decisions,
            "confidence_levels": confidence_levels,
            "dominant_detectors": dominant_detectors,
            "last_updated": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "total_alerts": 0,
            "alerts_24h": 0,
            "decisions": {},
            "confidence_levels": {},
            "dominant_detectors": {},
            "last_updated": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }
    except Exception as e:
        print(f"[DIAMOND STATS] ⚠️ Error getting stats: {e}")
        return {"error": str(e)}


def test_diamond_alert_system():
    """Testuje system Diamond Alerts"""
    print("🔍 Testing Diamond Alert System...")
    
    # Test data
    test_scores = {
        "decision": "TRIGGER",
        "fused_score": 0.847,
        "confidence": "HIGH",
        "dominant_detector": "diamond",
        "trigger_reasons": [
            "💎 Diamond Temporal Anomaly",
            "🧠 WhaleCLIP Confidence",
            "🐋 Whale Ping Detected"
        ],
        "decision_breakdown": {
            "whale_ping_score": 0.6,
            "whaleclip_score": 0.8,
            "diamond_score": 0.9,
            "fused_score": 0.847
        }
    }
    
    # Test formatowania
    print("✅ Testing message formatting...")
    
    confidence_indicator = format_confidence_indicator(test_scores["confidence"])
    print(f"  Confidence: {confidence_indicator}")
    
    dominant_emoji = get_dominant_detector_emoji(test_scores["dominant_detector"])
    print(f"  Dominant emoji: {dominant_emoji}")
    
    formatted_reasons = format_trigger_reasons(test_scores["trigger_reasons"])
    print(f"  Formatted reasons:\n{formatted_reasons}")
    
    formatted_breakdown = format_score_breakdown(test_scores["decision_breakdown"])
    print(f"  Formatted breakdown:\n{formatted_breakdown}")
    
    # Test statystyk
    print("✅ Testing stats...")
    stats = get_diamond_alert_stats()
    print(f"  Stats: {stats}")
    
    print("🔥 Diamond Alert System test completed!")


if __name__ == "__main__":
    test_diamond_alert_system()