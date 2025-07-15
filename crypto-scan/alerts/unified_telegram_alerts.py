"""
Unified Telegram Alert System - Stage 8/7
System jednolitych alertów Telegram dla wszystkich detektorów stealthowych

Wspiera:
- WhaleCLIP (wizualne rozpoznanie stylu portfeli + embeddingi CLIP)
- DiamondWhale AI (Temporal GNN + Quantum-Inspired RL) 
- CaliforniumWhale AI (Temporal GNN + Mastermind tracing + thresholding)
- Classic Stealth Engine (whale_ping + dex_inflow + spoofing_layers)
"""

import requests
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTelegramAlerts:
    """
    Unified Telegram Alert System for all stealth detectors
    """
    
    def __init__(self):
        """Initialize unified alert system with environment variables"""
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.alert_history_file = "cache/unified_alert_history.json"
        self.cooldown_seconds = 1800  # 30 minutes cooldown per token per detector
        self.alert_counts = {}
        
        # Detector configurations
        self.detector_configs = {
            "CaliforniumWhale AI": {
                "emoji": "🧠",
                "hashtag": "#CaliforniumAI",
                "threshold": 0.70,
                "default_comment": "Mastermind traced",
                "default_action": "HIGH PRIORITY WATCH 🔍"
            },
            "DiamondWhale AI": {
                "emoji": "💎", 
                "hashtag": "#DiamondAI",
                "threshold": 0.70,
                "default_comment": "Temporal whale pattern 📊",
                "default_action": "Watch & Evaluate"
            },
            "WhaleCLIP": {
                "emoji": "📷",
                "hashtag": "#WhaleCLIP", 
                "threshold": 0.70,
                "default_comment": "Visual whale signature",
                "default_action": "Potential Pre-Pump Detected"
            },
            "Classic Stealth Engine": {
                "emoji": "🐳",
                "hashtag": "#StealthEngine",
                "threshold": 0.70, 
                "default_comment": "Stealth pattern: whale_ping + DEX inflow",
                "default_action": "LONG Opportunity?"
            },
            "Fusion Engine": {
                "emoji": "🚀",
                "hashtag": "#FusionEngine",
                "threshold": 0.65,
                "default_comment": "Multi-detector consensus",
                "default_action": "HIGH CONFIDENCE SIGNAL"
            }
        }
        
        logger.info("[UNIFIED ALERTS] Initialized with environment variables")
    
    def check_alert_cooldown(self, token_symbol: str, detector_name: str, current_score: float = 0.0) -> bool:
        """
        Check if alert is within cooldown period (intelligent cooldown for high scores)
        
        Args:
            token_symbol: Token symbol
            detector_name: Name of detector
            current_score: Current score for dynamic cooldown calculation
            
        Returns:
            True if alert can be sent, False if in cooldown
        """
        cooldown_key = f"{token_symbol}_{detector_name}"
        current_time = time.time()
        
        if cooldown_key in self.alert_counts:
            last_alert_time = self.alert_counts[cooldown_key].get('last_alert', 0)
            last_score = self.alert_counts[cooldown_key].get('last_score', 0.0)
            time_elapsed = current_time - last_alert_time
            
            # 🎯 INTELLIGENT COOLDOWN: Reduced cooldown for exceptional signals
            base_cooldown = self.cooldown_seconds  # 30 minutes default
            
            # Dynamic cooldown based on current score strength
            if current_score >= 1.5:
                # Very high score: 5 minute cooldown only
                dynamic_cooldown = 300  # 5 minutes
                logger.info(f"[COOLDOWN SMART] {detector_name} → {token_symbol} | High score {current_score:.3f}, reduced cooldown: 5min")
            elif current_score >= 1.0:
                # High score: 10 minute cooldown
                dynamic_cooldown = 600  # 10 minutes
                logger.info(f"[COOLDOWN SMART] {detector_name} → {token_symbol} | Good score {current_score:.3f}, reduced cooldown: 10min")
            elif current_score >= 0.8:
                # Medium-high score: 15 minute cooldown
                dynamic_cooldown = 900  # 15 minutes
                logger.info(f"[COOLDOWN SMART] {detector_name} → {token_symbol} | Medium score {current_score:.3f}, reduced cooldown: 15min")
            else:
                # Standard score: full cooldown
                dynamic_cooldown = base_cooldown
                logger.info(f"[COOLDOWN SMART] {detector_name} → {token_symbol} | Standard score {current_score:.3f}, normal cooldown: 30min")
            
            # Check if score significantly improved from last alert
            if current_score > last_score + 0.3:
                # Score improved significantly: allow immediate re-alert
                logger.info(f"[COOLDOWN SMART] {detector_name} → {token_symbol} | Score improved {last_score:.3f} → {current_score:.3f}, bypassing cooldown")
                return True
            
            # Apply dynamic cooldown
            if time_elapsed < dynamic_cooldown:
                remaining = dynamic_cooldown - time_elapsed
                logger.info(f"[COOLDOWN SMART] {detector_name} → {token_symbol} | Alert blocked, {remaining:.0f}s remaining (score: {current_score:.3f})")
                return False
        
        return True
    
    def update_alert_history(self, token_symbol: str, detector_name: str, score: float, 
                           comment: str, action: str, sent_successfully: bool):
        """Update alert history and cooldown tracking"""
        cooldown_key = f"{token_symbol}_{detector_name}"
        current_time = time.time()
        
        # Update cooldown tracking
        if cooldown_key not in self.alert_counts:
            self.alert_counts[cooldown_key] = {}
        
        self.alert_counts[cooldown_key].update({
            'last_alert': current_time,
            'total_alerts': self.alert_counts[cooldown_key].get('total_alerts', 0) + 1,
            'last_score': score,
            'last_success': sent_successfully
        })
        
        # Save to history file
        try:
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'token_symbol': token_symbol,
                'detector_name': detector_name,
                'score': score,
                'comment': comment,
                'action': action,
                'sent_successfully': sent_successfully,
                'unix_timestamp': current_time
            }
            
            # Load existing history
            history = []
            if os.path.exists(self.alert_history_file):
                with open(self.alert_history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new record
            history.append(alert_record)
            
            # Keep only last 1000 alerts
            if len(history) > 1000:
                history = history[-1000:]
            
            # Save updated history
            os.makedirs(os.path.dirname(self.alert_history_file), exist_ok=True)
            with open(self.alert_history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"[HISTORY ERROR] Failed to save alert history: {e}")
    
    def send_stealth_alert(self, token_symbol: str, detector_name: str, score: float, 
                          comment: str = None, action: str = None, 
                          additional_data: Dict = None, consensus_decision: str = None, 
                          consensus_enabled: bool = False) -> bool:
        """
        Universal stealth alert function for all detectors with consensus validation
        
        Args:
            token_symbol: Symbol of the token (e.g., "RSRUSDT")
            detector_name: Name of detector that generated alert
            score: Score value (0.0-1.0 or higher)
            comment: Custom comment (uses default if None)
            action: Suggested action (uses default if None)
            additional_data: Additional data to include in alert
            consensus_decision: Consensus decision (BUY/HOLD/AVOID)
            consensus_enabled: Whether consensus is available
            
        Returns:
            True if alert sent successfully, False otherwise
        """
        
        # 🔐 CRITICAL CONSENSUS DECISION CHECK FIRST - NAJWAŻNIEJSZE SPRAWDZENIE
        if consensus_enabled and consensus_decision:
            if consensus_decision != "BUY":
                logger.info(f"[UNIFIED CONSENSUS BLOCK] {token_symbol} → Consensus decision {consensus_decision} blocks alert ({detector_name}, score={score:.3f})")
                return False  # Blokuj alert jeśli consensus != BUY
            else:
                logger.info(f"[UNIFIED CONSENSUS PASS] {token_symbol} → Consensus decision BUY allows alert ({detector_name}, score={score:.3f})")
        else:
            # Fallback - bez consensus, sprawdź score threshold
            if score < 4.0:
                logger.info(f"[UNIFIED NO CONSENSUS] {token_symbol} → No consensus, score {score:.3f} < 4.0 threshold - blocking alert ({detector_name})")
                return False
            else:
                logger.info(f"[UNIFIED FALLBACK] {token_symbol} → No consensus, high score {score:.3f} >= 4.0 allows alert ({detector_name})")
        
        # Validate credentials
        if not self.bot_token or not self.chat_id:
            logger.error("[ALERT ERROR] Missing Telegram credentials")
            return False
        
        # Get detector config
        if detector_name not in self.detector_configs:
            logger.warning(f"[ALERT WARNING] Unknown detector: {detector_name}")
            config = {
                "emoji": "🔍",
                "hashtag": f"#{detector_name.replace(' ', '').replace('-', '')}",
                "threshold": 0.70,
                "default_comment": "Detection signal",
                "default_action": "Monitor closely"
            }
        else:
            config = self.detector_configs[detector_name]
        
        # Use defaults if not provided
        if comment is None:
            comment = config["default_comment"]
        if action is None:
            action = config["default_action"]
        
        # Check intelligent cooldown with score
        if not self.check_alert_cooldown(token_symbol, detector_name, score):
            return False
        
        # Format message
        message = self._format_alert_message(
            token_symbol, detector_name, score, comment, action, config, additional_data
        )
        
        # Send alert
        success = self._send_telegram_message(message)
        
        # Update history
        self.update_alert_history(token_symbol, detector_name, score, comment, action, success)
        
        if success:
            logger.info(f"[ALERT SENT] {detector_name} → {token_symbol} | Score: {score:.3f}")
        else:
            logger.error(f"[ALERT FAILED] {detector_name} → {token_symbol}")
        
        return success
    
    def _format_alert_message(self, token_symbol: str, detector_name: str, score: float,
                             comment: str, action: str, config: Dict, 
                             additional_data: Dict = None) -> str:
        """Format the alert message for Telegram"""
        
        # Main alert message
        message = (
            f"🚨 Stealth Alert Detected 🚨\n\n"
            f"{config['emoji']} **Token:** `{token_symbol}`\n"
            f"🔍 **Detector:** {detector_name}\n"
            f"📈 **Score:** `{score:.3f}`\n"
            f"💬 **Comment:** {comment}\n"
            f"⚡ **Suggested Action:** {action}\n"
        )
        
        # Add additional data if provided
        if additional_data:
            message += "\n📊 **Additional Data:**\n"
            for key, value in additional_data.items():
                if isinstance(value, float):
                    message += f"• {key}: `{value:.3f}`\n"
                else:
                    message += f"• {key}: `{value}`\n"
        
        # Add timestamp and hashtag
        timestamp = datetime.now().strftime("%H:%M:%S UTC")
        message += f"\n⏰ **Time:** {timestamp}\n"
        message += f"\n{config['hashtag']} #StealthAlert"
        
        return message
    
    def _send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            params = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"[TELEGRAM ERROR] Failed to send message: {e}")
            return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        try:
            if not os.path.exists(self.alert_history_file):
                return {"total_alerts": 0, "detectors": {}}
            
            with open(self.alert_history_file, 'r') as f:
                history = json.load(f)
            
            stats = {
                "total_alerts": len(history),
                "successful_alerts": sum(1 for alert in history if alert.get('sent_successfully', False)),
                "detectors": {},
                "recent_24h": 0,
                "top_tokens": {}
            }
            
            # Calculate detector stats
            current_time = time.time()
            for alert in history:
                detector = alert['detector_name']
                if detector not in stats['detectors']:
                    stats['detectors'][detector] = {
                        'total': 0,
                        'successful': 0,
                        'avg_score': 0.0
                    }
                
                stats['detectors'][detector]['total'] += 1
                if alert.get('sent_successfully', False):
                    stats['detectors'][detector]['successful'] += 1
                
                # Recent 24h count
                if current_time - alert.get('unix_timestamp', 0) < 86400:
                    stats['recent_24h'] += 1
                
                # Top tokens
                token = alert['token_symbol']
                if token not in stats['top_tokens']:
                    stats['top_tokens'][token] = 0
                stats['top_tokens'][token] += 1
            
            # Calculate average scores
            for detector in stats['detectors']:
                detector_alerts = [alert for alert in history if alert['detector_name'] == detector]
                if detector_alerts:
                    avg_score = sum(alert['score'] for alert in detector_alerts) / len(detector_alerts)
                    stats['detectors'][detector]['avg_score'] = avg_score
            
            return stats
            
        except Exception as e:
            logger.error(f"[STATS ERROR] Failed to get statistics: {e}")
            return {"error": str(e)}

# Global instance
_unified_alerts = None

def get_unified_alerts() -> UnifiedTelegramAlerts:
    """Get singleton instance of unified alerts"""
    global _unified_alerts
    if _unified_alerts is None:
        _unified_alerts = UnifiedTelegramAlerts()
    return _unified_alerts

def send_stealth_alert(token_symbol: str, detector_name: str, score: float, 
                      comment: str = None, action: str = None, 
                      additional_data: Dict = None) -> bool:
    """
    Convenience function for sending stealth alerts
    
    Args:
        token_symbol: Symbol of the token (e.g., "RSRUSDT")
        detector_name: Name of detector that generated alert
        score: Score value (0.0-1.0 or higher)
        comment: Custom comment (uses default if None)
        action: Suggested action (uses default if None)
        additional_data: Additional data to include in alert
        
    Returns:
        True if alert sent successfully, False otherwise
    """
    alerts = get_unified_alerts()
    return alerts.send_stealth_alert(token_symbol, detector_name, score, comment, action, additional_data)

def get_alert_statistics() -> Dict[str, Any]:
    """Get unified alert statistics"""
    alerts = get_unified_alerts()
    return alerts.get_alert_statistics()

def test_unified_alerts():
    """Test unified alert system"""
    print("🧪 Testing Unified Telegram Alert System...")
    
    # Test all detectors
    test_cases = [
        ("BTCUSDT", "CaliforniumWhale AI", 0.851, None, None),
        ("ETHUSDT", "DiamondWhale AI", 0.743, "Strong temporal pattern", "IMMEDIATE WATCH"),
        ("BNBUSDT", "WhaleCLIP", 0.892, "Whale-style transactions detected", None),
        ("ADAUSDT", "Classic Stealth Engine", 0.723, None, None),
        ("SOLUSDT", "Fusion Engine", 0.678, "Multi-detector consensus", "HIGH CONFIDENCE")
    ]
    
    for token, detector, score, comment, action in test_cases:
        success = send_stealth_alert(token, detector, score, comment, action)
        print(f"✅ {detector} → {token}: {'SUCCESS' if success else 'FAILED'}")
    
    # Print statistics
    stats = get_alert_statistics()
    print(f"\n📊 Alert Statistics:")
    print(f"• Total alerts: {stats.get('total_alerts', 0)}")
    print(f"• Successful: {stats.get('successful_alerts', 0)}")
    print(f"• Recent 24h: {stats.get('recent_24h', 0)}")
    
    print("✅ Unified Alert System test completed!")

if __name__ == "__main__":
    test_unified_alerts()