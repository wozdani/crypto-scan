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

# Import watchlist functionality
try:
    from .stealth_v3_watchlist_alerts import send_watchlist_alert
except ImportError:
    # Fallback for direct import
    try:
        from stealth_v3_watchlist_alerts import send_watchlist_alert
    except ImportError:
        def send_watchlist_alert(watchlist_data: Dict[str, Any]) -> bool:
            """Fallback function if watchlist alerts not available"""
            print(f"[WATCHLIST FALLBACK] Watchlist alerts not available for {watchlist_data.get('symbol', 'UNKNOWN')}")
            return False

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
    
    def send_stealth_v3_alert(self, symbol: str, detector_results: Dict[str, Any], 
                             consensus_data: Dict[str, Any], meta_data: Dict[str, Any] = None) -> bool:
        """
        Wyślij nowoczesny Stealth v3 alert z pełnym breakdown detektorów i consensus logic
        
        Args:
            symbol: Symbol tokena (np. "FUELUSDT")
            detector_results: Dict z rezultatami detektorów {"whale_ping": 1.000, "dex_inflow": 0.000, "whaleclip_vision": 1.000, ...}
            consensus_data: Dict z danymi consensus {"decision": "BUY", "votes": ["BUY", "BUY", "HOLD", "BUY"], "confidence": 1.945, "feedback_adjust": 0.124}
            meta_data: Opcjonalne meta dane {"trust_addresses": 3, "coverage": 100.0, "californium_score": 0.434, "diamond_score": 0.555}
            
        Returns:
            True jeśli alert wysłany pomyślnie
        """
        
        # 🔐 CRITICAL CONSENSUS DECISION CHECK FIRST - NAJWAŻNIEJSZE SPRAWDZENIE
        consensus_decision = consensus_data.get('decision', 'HOLD')
        if consensus_decision != "BUY":
            logger.info(f"[STEALTH V3 CONSENSUS BLOCK] {symbol} → Consensus decision {consensus_decision} blocks alert")
            return False  # Blokuj alert jeśli consensus != BUY
        else:
            logger.info(f"[STEALTH V3 CONSENSUS PASS] {symbol} → Consensus decision BUY allows alert")
        
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
    
    def _format_stealth_v3_message(self, symbol: str, detector_results: Dict[str, Any], 
                                  consensus_data: Dict[str, Any], meta_data: Dict[str, Any] = None) -> str:
        """Format nowy Stealth v3 alert message z comprehensive diagnostic breakdown"""
        
        # Extract data with defaults
        decision = consensus_data.get('decision', 'UNKNOWN')
        votes = consensus_data.get('votes', [])
        confidence = consensus_data.get('confidence', 0.0)
        feedback_adjust = consensus_data.get('feedback_adjust', 0.0)
        
        # Meta data with defaults
        if meta_data is None:
            meta_data = {}
        trust_addresses = meta_data.get('trust_addresses', 0)
        coverage = meta_data.get('coverage', 0.0)
        historical_support = meta_data.get('historical_support', 'Unknown')
        californium_score = meta_data.get('californium_score', 0.0)
        diamond_score = meta_data.get('diamond_score', 0.0)
        
        # Calculate total score z detector breakdown
        total_score = 0.0
        active_detector_names = []
        
        # Extract detector scores (obsługuje zarówno bool jak i float)
        whale_ping_score = self._extract_detector_value(detector_results.get('whale_ping', 0))
        dex_inflow_score = self._extract_detector_value(detector_results.get('dex_inflow', 0))
        mastermind_score = self._extract_detector_value(detector_results.get('mastermind_tracing', 0))
        orderbook_score = self._extract_detector_value(detector_results.get('orderbook_anomaly', 0))
        whaleclip_score = self._extract_detector_value(detector_results.get('whaleclip_vision', 0))
        
        # Suma score breakdown
        total_score = whale_ping_score + dex_inflow_score + mastermind_score + orderbook_score + whaleclip_score + californium_score + diamond_score + feedback_adjust
        
        # Generate active detectors list dla pattern recognition
        if whale_ping_score > 0: active_detector_names.append("Whale Ping")
        if dex_inflow_score > 0: active_detector_names.append("DEX Inflow")
        if mastermind_score > 0: active_detector_names.append("Mastermind")
        if orderbook_score > 0: active_detector_names.append("Orderbook")
        if whaleclip_score > 0: active_detector_names.append("WhaleCLIP")
        if californium_score > 0: active_detector_names.append("CaliforniumWhale")
        if diamond_score > 0: active_detector_names.append("DiamondWhale")
        
        # Smart pattern detection
        if len(active_detector_names) >= 3:
            pattern_text = f"Multi-detector consensus ({'+'.join(active_detector_names[:3])})"
        elif len(active_detector_names) == 2:
            pattern_text = f"{' + '.join(active_detector_names)} consensus"
        elif len(active_detector_names) == 1:
            pattern_text = f"{active_detector_names[0]} dominant signal"
        else:
            pattern_text = "Composite stealth signals"
        
        # Process agent votes z proper formatting
        if isinstance(votes, list) and len(votes) > 0:
            buy_votes = votes.count('BUY')
            hold_votes = votes.count('HOLD')
            avoid_votes = votes.count('AVOID')
            total_votes = len(votes)
            
            # Consensus analysis
            if buy_votes >= total_votes * 0.75:
                consensus_text = f"✅ {buy_votes}/{total_votes} BUY (Strong Consensus)"
            elif buy_votes > total_votes * 0.5:
                consensus_text = f"✅ {buy_votes}/{total_votes} BUY (Majority)"
            elif hold_votes > buy_votes:
                consensus_text = f"⚠️ {hold_votes}/{total_votes} HOLD (Caution)"
            else:
                consensus_text = f"❌ {avoid_votes}/{total_votes} AVOID (Rejection)"
            
            votes_detail = f"[{', '.join(votes)}] → {consensus_text}"
        else:
            # Fallback dla starych formatów
            votes_detail = f"Confidence: {confidence:.3f}"
        
        # Decision action na podstawie score i decision
        if decision == "BUY" and confidence >= 2.0:
            action_text = "STRONG LONG 🚀"
        elif decision == "BUY" and confidence >= 1.5:
            action_text = "LONG Opportunity"
        elif decision == "MONITOR":
            action_text = "High-priority watch 🔍"
        else:
            action_text = "Monitor closely"
        
        # Timestamp
        timestamp_utc = datetime.utcnow().strftime("%H:%M:%S UTC")
        
        # Format enhanced message z complete diagnostic breakdown
        message = f"""🚨 Stealth Alert Detected 🚨

🐳 Token: {symbol}
🔍 Detectors: {', '.join(active_detector_names) if active_detector_names else "None active"}
📈 Score: {total_score:.3f}
💬 Pattern: {pattern_text}

🗳️ Agent Votes: {votes_detail}
🤖 Final Decision: {action_text}

📊 Score Breakdown:
• 🐳 Whale Ping: {whale_ping_score:.3f}
• 💧 DEX Inflow: {dex_inflow_score:.3f}
• 🧠 Mastermind: {mastermind_score:.3f}
• 📊 Orderbook: {orderbook_score:.3f}
• 🛰️ WhaleCLIP: {whaleclip_score:.3f}
• 🔮 Californium: {californium_score:.3f}
• 💎 Diamond: {diamond_score:.3f}
• 🔁 Feedback Boost: {feedback_adjust:+.3f}

📋 Meta Analysis:
• Trust addresses: {trust_addresses}
• Data coverage: {coverage:.1f}%
• Historical support: {historical_support}

⏰ Time: {timestamp_utc}

#StealthEngine #WhaleDetection #RLConsensus"""

        return message
    
    def _extract_detector_value(self, detector_value) -> float:
        """Extract numeric value from detector result (supports bool, float, or dict)"""
        if isinstance(detector_value, bool):
            return 1.0 if detector_value else 0.0
        elif isinstance(detector_value, (int, float)):
            return float(detector_value)
        elif isinstance(detector_value, dict) and 'score' in detector_value:
            return float(detector_value['score'])
        else:
            return 0.0
    
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
    """Test Enhanced Stealth v3 alert system z numerical diagnostic transparency"""
    print("🧪 Testing Enhanced Stealth v3 Alert System...")
    print("📊 Features: Numerical detector scores, agent vote breakdown, comprehensive diagnostics")
    
    # Test scenario 1: High confidence multi-detector alert
    test_symbol = "TESTUSDT"
    test_detector_results = {
        'whale_ping': 1.000,
        'dex_inflow': 0.750,
        'mastermind_tracing': 0.434,  # CaliforniumWhale score
        'orderbook_anomaly': 0.650,
        'whaleclip_vision': 0.000,
        'diamond_ai': 0.555  # DiamondWhale score
    }
    test_consensus_data = {
        'decision': 'BUY',
        'votes': ['BUY', 'BUY', 'HOLD', 'BUY'],  # Proper agent votes array
        'confidence': 3.389,  # Total score jako confidence
        'feedback_adjust': 0.124
    }
    test_meta_data = {
        'trust_addresses': 3,
        'coverage': 100.0,
        'historical_support': 'Strong',
        'californium_score': 0.434,
        'diamond_score': 0.555
    }
    
    # Send high confidence test alert
    result1 = send_stealth_v3_alert(test_symbol, test_detector_results, test_consensus_data, test_meta_data)
    
    if result1:
        logger.info("[STEALTH V3 TEST] High confidence alert sent successfully")
    else:
        logger.error("[STEALTH V3 TEST] High confidence alert failed")
    
    # Test scenario 2: Low confidence scenario
    print("\n🧪 Testing Low Confidence Scenario...")
    
    low_symbol = "LOWUSDT"
    low_detector_results = {
        'whale_ping': 0.150,
        'dex_inflow': 0.200,
        'mastermind_tracing': 0.000,
        'orderbook_anomaly': 0.000,
        'whaleclip_vision': 0.100,
        'diamond_ai': 0.000
    }
    low_consensus_data = {
        'decision': 'WATCH',
        'votes': ['HOLD', 'AVOID', 'HOLD', 'AVOID'],
        'confidence': 0.450,
        'feedback_adjust': -0.050
    }
    
    result2 = send_stealth_v3_alert(low_symbol, low_detector_results, low_consensus_data, test_meta_data)
    
    if result2:
        logger.info("[STEALTH V3 TEST] Low confidence alert sent successfully")
    else:
        logger.info("[STEALTH V3 TEST] Low confidence alert blocked (expected)")
    
    # Get statistics
    stats = get_stealth_v3_alerts().get_alert_statistics()
    logger.info(f"[STEALTH V3 TEST] Current stats: {stats}")
    
    return result1 or result2

if __name__ == "__main__":
    test_stealth_v3_alert_system()