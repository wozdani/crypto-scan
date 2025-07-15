#!/usr/bin/env python3
"""
CaliforniumWhale AI Alert System - Stage 4/7 → UNIFIED INTEGRATION
DEPRECATED: Use alerts/unified_telegram_alerts.py instead
This module provides backward compatibility wrapper
"""

import os
import json
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_californium_alert(symbol: str, californium_score: float, 
                          market_data: Dict = None, additional_context: Dict = None) -> bool:
    """
    UNIFIED INTEGRATION: CaliforniumWhale AI Alert Wrapper
    Redirects to unified alert system for consistent messaging
    
    Args:
        symbol: Token symbol
        californium_score: CaliforniumWhale AI score
        market_data: Market data context
        additional_context: Additional analysis context
        
    Returns:
        True if alert sent successfully via unified system
    """
    try:
        from alerts.unified_telegram_alerts import send_stealth_alert
        
        # Przygotuj komentarz na podstawie score
        if californium_score >= 0.85:
            comment = "High-confidence mastermind traced 🧠💎"
            action = "IMMEDIATE HIGH PRIORITY WATCH 🚀"
        elif californium_score >= 0.75:
            comment = "Mastermind activity traced 🧠"
            action = "HIGH PRIORITY WATCH 🔍"
        else:
            comment = "Temporal graph mastermind pattern 📊"
            action = "Monitor & Evaluate"
        
        # Przygotuj dodatkowe dane z kontekstu
        additional_data = {}
        if market_data:
            additional_data.update({
                "price_usd": market_data.get("price_usd"),
                "volume_24h": market_data.get("volume_24h")
            })
        
        if additional_context:
            additional_data.update({
                "stealth_score": additional_context.get("stealth_score"),
                "active_signals": additional_context.get("active_signals"),
                "diamond_score": additional_context.get("diamond_score"),
                "data_coverage": additional_context.get("data_coverage")
            })
        
        # Wyślij przez unified alert system
        return send_stealth_alert(
            symbol, "CaliforniumWhale AI", californium_score,
            comment, action, additional_data
        )
        
    except ImportError:
        logger.error("[CALIFORNIUM WRAPPER] Unified alerts not available")
        return False
    except Exception as e:
        logger.error(f"[CALIFORNIUM WRAPPER] Alert error: {e}")
        return False

class CaliforniumAlertManager:
    """
    Manager for CaliforniumWhale AI alerts with Telegram integration
    """
    
    def __init__(self):
        """Initialize CaliforniumAlertManager with environment variables"""
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.alert_threshold = 0.7  # Score > 0.7 triggers alert
        self.alerts_sent = 0
        self.cooldown_minutes = 60  # 1 hour cooldown per symbol
        self.last_alerts = {}  # Track last alert time per symbol
        
    def check_californium_alert_trigger(self, symbol: str, californium_score: float, 
                                      market_data: Dict = None) -> bool:
        """
        Check if CaliforniumWhale AI score triggers alert
        
        Args:
            symbol: Token symbol
            californium_score: CaliforniumWhale AI score (0.0-1.0)
            market_data: Optional market data for context
            
        Returns:
            True if alert should be sent
        """
        if californium_score <= self.alert_threshold:
            return False
            
        # Check cooldown
        current_time = datetime.now().timestamp()
        last_alert_time = self.last_alerts.get(symbol, 0)
        
        if current_time - last_alert_time < (self.cooldown_minutes * 60):
            time_remaining = int((self.cooldown_minutes * 60) - (current_time - last_alert_time))
            logger.info(f"[CALIFORNIUM COOLDOWN] {symbol}: {time_remaining}s remaining")
            return False
            
        return True
    
    def format_californium_alert_message(self, symbol: str, californium_score: float, 
                                       market_data: Dict = None, 
                                       additional_context: Dict = None) -> str:
        """
        Format CaliforniumWhale AI alert message for Telegram
        
        Args:
            symbol: Token symbol
            californium_score: CaliforniumWhale AI score
            market_data: Market data context
            additional_context: Additional analysis context
            
        Returns:
            Formatted Telegram message
        """
        # Main alert header
        message = f"🚨 *CALIFORNIUM ALERT* - {symbol}\n"
        message += f"🧠 *Mastermind activity detected!*\n\n"
        
        # CaliforniumWhale AI analysis
        message += f"📊 *CaliforniumWhale Score:* `{californium_score:.3f}`\n"
        message += f"🔍 *Source:* CaliforniumWhale AI\n"
        message += f"⚡ *Confidence:* {'🔥 HIGH' if californium_score > 0.85 else '⚡ MEDIUM'}\n\n"
        
        # Market context if available
        if market_data:
            price = market_data.get('price_usd', 0)
            volume = market_data.get('volume_24h', 0)
            if price > 0:
                message += f"💰 *Price:* ${price:.6f}\n"
            if volume > 0:
                message += f"📈 *Volume 24h:* ${volume:,.0f}\n"
        
        # Temporal Graph Analysis details
        message += f"\n🔮 *Temporal Graph Analysis:*\n"
        message += f"• TGN Pattern Recognition: ✅\n"
        message += f"• QIRL Decision Engine: ✅\n"
        message += f"• Whale Accumulation: {'🐋 DETECTED' if californium_score > 0.8 else '🔍 POSSIBLE'}\n"
        
        # Additional context
        if additional_context:
            stealth_score = additional_context.get('stealth_score', 0)
            if stealth_score > 0:
                message += f"• Combined Stealth Score: `{stealth_score:.3f}`\n"
                
            active_signals = additional_context.get('active_signals', [])
            if active_signals:
                signals_text = ', '.join(active_signals)
                message += f"• Active Signals: {signals_text}\n"
        
        # Warning and action
        message += f"\n⚠️ *High confidence pump signal*\n"
        message += f"🎯 *Action:* Monitor for entry opportunity\n"
        message += f"⏰ *Alert Level:* IMMEDIATE\n\n"
        
        # Timestamp
        utc_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        message += f"🕒 *UTC:* {utc_time}"
        
        return message
    
    def send_californium_telegram_alert(self, symbol: str, californium_score: float,
                                      market_data: Dict = None,
                                      additional_context: Dict = None) -> bool:
        """
        Send CaliforniumWhale AI alert via Telegram
        
        Args:
            symbol: Token symbol
            californium_score: CaliforniumWhale AI score
            market_data: Market data context
            additional_context: Additional analysis context
            
        Returns:
            True if alert sent successfully
        """
        if not self.bot_token or not self.chat_id:
            logger.warning("[CALIFORNIUM ALERT] Missing Telegram credentials")
            return False
            
        if not self.check_californium_alert_trigger(symbol, californium_score, market_data):
            return False
            
        try:
            # Format alert message
            message = self.format_californium_alert_message(
                symbol, californium_score, market_data, additional_context
            )
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                # Update tracking
                self.last_alerts[symbol] = datetime.now().timestamp()
                self.alerts_sent += 1
                
                logger.info(f"[CALIFORNIUM ALERT] ✅ Alert sent for {symbol} (score: {californium_score:.3f})")
                
                # Save alert to history
                self.save_alert_history(symbol, californium_score, market_data, additional_context)
                
                return True
            else:
                logger.error(f"[CALIFORNIUM ALERT] ❌ Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"[CALIFORNIUM ALERT] ❌ Error sending alert: {e}")
            return False
    
    def save_alert_history(self, symbol: str, californium_score: float,
                          market_data: Dict = None, additional_context: Dict = None):
        """Save alert to history file"""
        try:
            # Ensure cache directory exists
            os.makedirs("crypto-scan/cache", exist_ok=True)
            
            alert_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "californium_score": californium_score,
                "market_data": market_data or {},
                "additional_context": additional_context or {},
                "alert_type": "californium_whale_ai"
            }
            
            history_file = "crypto-scan/cache/californium_alert_history.json"
            
            # Load existing history
            history = []
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                except:
                    history = []
            
            # Add new alert
            history.append(alert_record)
            
            # Keep only last 100 alerts
            if len(history) > 100:
                history = history[-100:]
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            logger.info(f"[CALIFORNIUM ALERT] Alert history saved for {symbol}")
            
        except Exception as e:
            logger.error(f"[CALIFORNIUM ALERT] Error saving alert history: {e}")
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get CaliforniumWhale AI alert statistics"""
        try:
            history_file = "crypto-scan/cache/californium_alert_history.json"
            
            if not os.path.exists(history_file):
                return {
                    "total_alerts": 0,
                    "alerts_24h": 0,
                    "unique_symbols": 0,
                    "avg_score": 0.0,
                    "last_alert": None
                }
            
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            if not history:
                return {
                    "total_alerts": 0,
                    "alerts_24h": 0,
                    "unique_symbols": 0,
                    "avg_score": 0.0,
                    "last_alert": None
                }
            
            # Calculate 24h cutoff
            cutoff_24h = (datetime.now(timezone.utc).timestamp() - 24 * 3600) * 1000
            
            alerts_24h = [
                alert for alert in history
                if datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00')).timestamp() * 1000 > cutoff_24h
            ]
            
            unique_symbols = len(set(alert['symbol'] for alert in history))
            avg_score = sum(alert['californium_score'] for alert in history) / len(history)
            
            return {
                "total_alerts": len(history),
                "alerts_24h": len(alerts_24h),
                "unique_symbols": unique_symbols,
                "avg_score": round(avg_score, 3),
                "last_alert": history[-1] if history else None
            }
            
        except Exception as e:
            logger.error(f"[CALIFORNIUM ALERT] Error getting statistics: {e}")
            return {
                "total_alerts": 0,
                "alerts_24h": 0,
                "unique_symbols": 0,
                "avg_score": 0.0,
                "last_alert": None,
                "error": str(e)
            }

# Global instance
_californium_alert_manager = None

def get_californium_alert_manager() -> CaliforniumAlertManager:
    """Get singleton CaliforniumAlertManager instance"""
    global _californium_alert_manager
    if _californium_alert_manager is None:
        _californium_alert_manager = CaliforniumAlertManager()
    return _californium_alert_manager

def send_californium_alert(symbol: str, californium_score: float,
                         market_data: Dict = None, additional_context: Dict = None) -> bool:
    """
    Convenience function to send CaliforniumWhale AI alert
    
    Args:
        symbol: Token symbol  
        californium_score: CaliforniumWhale AI score (0.0-1.0)
        market_data: Optional market data
        additional_context: Optional additional context
        
    Returns:
        True if alert sent successfully
    """
    manager = get_californium_alert_manager()
    return manager.send_californium_telegram_alert(symbol, californium_score, market_data, additional_context)

def test_californium_alert_system():
    """Test CaliforniumWhale AI alert system"""
    print("🧪 Testing CaliforniumWhale AI Alert System...")
    
    manager = get_californium_alert_manager()
    
    # Test data
    test_symbol = "ETHUSDT"
    test_score = 0.85
    test_market_data = {
        "price_usd": 2500.0,
        "volume_24h": 1500000000
    }
    test_context = {
        "stealth_score": 1.2,
        "active_signals": ["californium_whale_detection", "whale_ping"]
    }
    
    # Test alert formatting
    message = manager.format_californium_alert_message(
        test_symbol, test_score, test_market_data, test_context
    )
    print("📱 Sample Alert Message:")
    print(message)
    print()
    
    # Test alert trigger logic
    should_trigger = manager.check_californium_alert_trigger(test_symbol, test_score)
    print(f"🎯 Alert trigger test: {should_trigger}")
    
    # Test statistics
    stats = manager.get_alert_statistics()
    print(f"📊 Alert statistics: {stats}")
    
    print("✅ CaliforniumWhale AI Alert System test completed")

if __name__ == "__main__":
    test_californium_alert_system()