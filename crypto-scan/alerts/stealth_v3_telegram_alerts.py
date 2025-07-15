"""
Stealth Engine v3 Telegram Alert System
Nowoczesny, kontekstowy i modularny format alertów dla Stealth Engine v3 z AI detektorami i consensus logic
"""

import requests
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StealthV3TelegramAlerts:
    """
    Stealth Engine v3 Telegram Alert System
    Nowoczesny format z modularnym breakdown detektorów i consensus decision logic
    """
    
    def __init__(self):
        """Initialize Stealth v3 alert system"""
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.alert_history_file = "cache/stealth_v3_alert_history.json"
        self.cooldown_seconds = 1800  # 30 minutes default
        self.alert_counts = {}
        
        logger.info("[STEALTH V3 ALERTS] Initialized new modular alert system")
    
    def send_stealth_v3_alert(self, symbol: str, detector_results: Dict[str, bool], 
                             consensus_data: Dict[str, Any], meta_data: Dict[str, Any] = None) -> bool:
        """
        Wyślij nowoczesny Stealth v3 alert z pełnym breakdown detektorów i consensus logic
        
        Args:
            symbol: Symbol tokena (np. "FUELUSDT")
            detector_results: Dict z rezultatami detektorów {"whale_ping": True, "dex_inflow": True, ...}
            consensus_data: Dict z danymi consensus {"decision": "BUY", "votes": "3/4", "confidence": 1.945, "feedback_adjust": 0.124}
            meta_data: Opcjonalne meta dane {"trust_addresses": 3, "coverage": 100.0, "californium_score": 0.434}
            
        Returns:
            True jeśli alert wysłany pomyślnie
        """
        
        # Validate credentials
        if not self.bot_token or not self.chat_id:
            logger.error("[STEALTH V3] Missing Telegram credentials")
            return False
        
        # Check cooldown for symbol
        if not self._check_symbol_cooldown(symbol, consensus_data.get('confidence', 0.0)):
            return False
        
        # Format message using new v3 template
        message = self._format_stealth_v3_message(symbol, detector_results, consensus_data, meta_data)
        
        # Send alert
        success = self._send_telegram_message(message)
        
        # Update history
        self._update_alert_history(symbol, detector_results, consensus_data, success)
        
        if success:
            logger.info(f"[STEALTH V3 SENT] {symbol} | Consensus: {consensus_data.get('decision', 'UNKNOWN')} | Confidence: {consensus_data.get('confidence', 0.0):.3f}")
        else:
            logger.error(f"[STEALTH V3 FAILED] {symbol}")
        
        return success
    
    def _format_stealth_v3_message(self, symbol: str, detector_results: Dict[str, bool], 
                                  consensus_data: Dict[str, Any], meta_data: Dict[str, Any] = None) -> str:
        """Format nowy Stealth v3 alert message"""
        
        # Extract data with defaults
        decision = consensus_data.get('decision', 'UNKNOWN')
        votes = consensus_data.get('votes', '0/4')
        confidence = consensus_data.get('confidence', 0.0)
        feedback_adjust = consensus_data.get('feedback_adjust', 0.0)
        
        # Meta data with defaults
        if meta_data is None:
            meta_data = {}
        trust_addresses = meta_data.get('trust_addresses', 0)
        coverage = meta_data.get('coverage', 0.0)
        historical_support = meta_data.get('historical_support', 'Unknown')
        californium_score = meta_data.get('californium_score', 0.0)
        
        # Generate active detectors list for reason
        active_detectors = []
        detector_mapping = {
            'whale_ping': 'whale ping',
            'dex_inflow': 'dex inflow', 
            'mastermind_tracing': 'mastermind tracing',
            'orderbook_anomaly': 'orderbook anomaly',
            'whaleclip_vision': 'WhaleCLIP vision'
        }
        
        for detector, active in detector_results.items():
            if active and detector in detector_mapping:
                active_detectors.append(detector_mapping[detector])
        
        reason_text = f"Detected stealth accumulation via {', '.join(active_detectors) if active_detectors else 'multiple signals'}."
        
        # Decision emoji based on consensus
        decision_emoji = "✅" if decision == "BUY" else "⚠️" if decision == "HOLD" else "❌"
        
        # Timestamp
        timestamp_utc = datetime.utcnow().strftime("%H:%M:%S UTC")
        
        # Format message
        message = f"""🚨 Stealth Alert: {symbol} 🚨

🧠 Detectors:
• 🐳 whale_ping:           {"✅" if detector_results.get('whale_ping', False) else "❌"}
• 💧 dex_inflow:           {"✅" if detector_results.get('dex_inflow', False) else "❌"}
• 🧠 mastermind_tracing:   {"✅" if detector_results.get('mastermind_tracing', False) else "❌"}
• 📊 orderbook_anomaly:    {"✅" if detector_results.get('orderbook_anomaly', False) else "❌"}
• 🛰️ WhaleCLIP (Vision):   {"✅" if detector_results.get('whaleclip_vision', False) else "❌"}

📡 Consensus Decision: {decision_emoji} {decision} ({votes} agents)
🎯 Confidence Score:    {confidence:.3f}
🔁 Feedback Adjusted:   {feedback_adjust:+.3f}

🔍 Reason:
{reason_text}

📊 Meta:
• Trust addresses:        {trust_addresses}
• Data coverage:         {coverage:.1f}%
• Historical support:    {historical_support}
• Californium Score:     {californium_score:.3f}

⏰ Time: {timestamp_utc}

#StealthEngine #WhaleCLIP #ConsensusAI #PumpRadar"""

        return message
    
    def _check_symbol_cooldown(self, symbol: str, confidence: float) -> bool:
        """Check cooldown for symbol with intelligent adjustment based on confidence"""
        current_time = time.time()
        
        if symbol in self.alert_counts:
            last_alert_time = self.alert_counts[symbol].get('last_alert', 0)
            time_elapsed = current_time - last_alert_time
            
            # Intelligent cooldown based on confidence
            if confidence >= 2.0:
                # Bardzo wysoki confidence: 5 minut
                dynamic_cooldown = 300
            elif confidence >= 1.5:
                # Wysoki confidence: 10 minut
                dynamic_cooldown = 600
            elif confidence >= 1.0:
                # Średni-wysoki confidence: 15 minut
                dynamic_cooldown = 900
            else:
                # Standardowy confidence: pełny cooldown
                dynamic_cooldown = self.cooldown_seconds
            
            if time_elapsed < dynamic_cooldown:
                logger.info(f"[STEALTH V3 COOLDOWN] {symbol} | {time_elapsed:.0f}s < {dynamic_cooldown}s required")
                return False
        
        # Update cooldown
        self.alert_counts[symbol] = {
            'last_alert': current_time,
            'last_confidence': confidence
        }
        
        return True
    
    def _send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            params = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"  # Using HTML for better emoji support
            }
            
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"[STEALTH V3 TELEGRAM ERROR] Failed to send message: {e}")
            return False
    
    def _update_alert_history(self, symbol: str, detector_results: Dict[str, bool], 
                             consensus_data: Dict[str, Any], success: bool):
        """Update alert history"""
        try:
            # Load existing history
            history = []
            if os.path.exists(self.alert_history_file):
                with open(self.alert_history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new alert
            alert_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'unix_timestamp': time.time(),
                'symbol': symbol,
                'detector_results': detector_results,
                'consensus_data': consensus_data,
                'sent_successfully': success
            }
            
            history.append(alert_data)
            
            # Keep only last 1000 alerts
            if len(history) > 1000:
                history = history[-1000:]
            
            # Save updated history
            os.makedirs(os.path.dirname(self.alert_history_file), exist_ok=True)
            with open(self.alert_history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"[STEALTH V3 HISTORY ERROR] Failed to save alert history: {e}")
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get Stealth v3 alert statistics"""
        try:
            if not os.path.exists(self.alert_history_file):
                return {"total_alerts": 0, "decisions": {}, "detectors": {}}
            
            with open(self.alert_history_file, 'r') as f:
                history = json.load(f)
            
            stats = {
                "total_alerts": len(history),
                "successful_alerts": sum(1 for alert in history if alert.get('sent_successfully', False)),
                "decisions": {},
                "detectors": {},
                "recent_24h": 0,
                "avg_confidence": 0.0
            }
            
            # Calculate stats
            current_time = time.time()
            total_confidence = 0.0
            
            for alert in history:
                # Decision breakdown
                decision = alert.get('consensus_data', {}).get('decision', 'UNKNOWN')
                if decision not in stats['decisions']:
                    stats['decisions'][decision] = 0
                stats['decisions'][decision] += 1
                
                # Detector breakdown
                detector_results = alert.get('detector_results', {})
                for detector, active in detector_results.items():
                    if detector not in stats['detectors']:
                        stats['detectors'][detector] = {'active': 0, 'total': 0}
                    stats['detectors'][detector]['total'] += 1
                    if active:
                        stats['detectors'][detector]['active'] += 1
                
                # Recent 24h
                if current_time - alert.get('unix_timestamp', 0) < 86400:
                    stats['recent_24h'] += 1
                
                # Confidence tracking
                confidence = alert.get('consensus_data', {}).get('confidence', 0.0)
                total_confidence += confidence
            
            # Average confidence
            if len(history) > 0:
                stats['avg_confidence'] = total_confidence / len(history)
            
            return stats
            
        except Exception as e:
            logger.error(f"[STEALTH V3 STATS ERROR] Failed to get statistics: {e}")
            return {"error": str(e)}

# Global instance
_stealth_v3_alerts = None

def get_stealth_v3_alerts() -> StealthV3TelegramAlerts:
    """Get singleton instance of Stealth v3 alerts"""
    global _stealth_v3_alerts
    if _stealth_v3_alerts is None:
        _stealth_v3_alerts = StealthV3TelegramAlerts()
    return _stealth_v3_alerts

def send_stealth_v3_alert(symbol: str, detector_results: Dict[str, bool], 
                         consensus_data: Dict[str, Any], meta_data: Dict[str, Any] = None) -> bool:
    """
    Convenience function for sending Stealth v3 alerts
    
    Args:
        symbol: Symbol tokena (np. "FUELUSDT")
        detector_results: Dict z rezultatami detektorów
        consensus_data: Dict z danymi consensus
        meta_data: Opcjonalne meta dane
        
    Returns:
        True jeśli alert wysłany pomyślnie
    """
    return get_stealth_v3_alerts().send_stealth_v3_alert(symbol, detector_results, consensus_data, meta_data)

def test_stealth_v3_alert_system():
    """Test Stealth v3 alert system"""
    # Test data
    test_symbol = "FUELUSDT"
    test_detector_results = {
        'whale_ping': True,
        'dex_inflow': True,
        'mastermind_tracing': False,
        'orderbook_anomaly': False,
        'whaleclip_vision': True
    }
    test_consensus_data = {
        'decision': 'BUY',
        'votes': '3/4',
        'confidence': 1.945,
        'feedback_adjust': 0.124
    }
    test_meta_data = {
        'trust_addresses': 3,
        'coverage': 100.0,
        'historical_support': 'Yes',
        'californium_score': 0.434
    }
    
    # Send test alert
    result = send_stealth_v3_alert(test_symbol, test_detector_results, test_consensus_data, test_meta_data)
    
    if result:
        logger.info("[STEALTH V3 TEST] Test alert sent successfully")
    else:
        logger.error("[STEALTH V3 TEST] Test alert failed")
    
    # Get statistics
    stats = get_stealth_v3_alerts().get_alert_statistics()
    logger.info(f"[STEALTH V3 TEST] Current stats: {stats}")
    
    return result

if __name__ == "__main__":
    test_stealth_v3_alert_system()