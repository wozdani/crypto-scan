#!/usr/bin/env python3
"""
Alert Manager - System alertów z powiadomieniami Telegram
Oparty o anomaly score z GNN + decyzję RL Agent

Wysyła alerty, jeśli GNN + RL wykryją potencjalny pump.
"""

import os
import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from rl_agent import RLAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_alert(token: str, chat_id: str, message: str) -> bool:
    """
    Wysyła alert na Telegram.
    
    Args:
        token: Bot token Telegram
        chat_id: ID chatu docelowego
        message: Treść wiadomości
        
    Returns:
        True jeśli wysłano pomyślnie, False w przeciwnym razie
    """
    if not token or not chat_id:
        logger.warning("[TELEGRAM] Missing bot token or chat ID")
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id, 
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info(f"[TELEGRAM] Alert sent successfully")
            return True
        else:
            logger.error(f"[TELEGRAM ERROR] HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"[TELEGRAM ERROR] {e}")
        return False

def escape_markdown(text: str) -> str:
    """
    Escape special characters for Telegram Markdown.
    
    Args:
        text: Text to escape
        
    Returns:
        Escaped text safe for Markdown
    """
    special_chars = ['*', '_', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

def format_alert_message(symbol: str, anomaly_scores: Dict[str, float], 
                        rl_prediction: Dict[str, Any], 
                        market_data: Dict[str, Any] = None) -> str:
    """
    Formatuje wiadomość alertu Telegram.
    
    Args:
        symbol: Symbol tokena
        anomaly_scores: Wyniki anomaly detection z GNN
        rl_prediction: Predykcja z RL Agent
        market_data: Opcjonalne dane rynkowe
        
    Returns:
        Sformatowana wiadomość dla Telegram
    """
    # Header
    message = f"🚨 *STEALTH ENGINE GNN ALERT*\n\n"
    message += f"📊 *Token:* `{escape_markdown(symbol)}`\n"
    
    # Market data if available
    if market_data:
        price = market_data.get('price_usd', 0)
        volume = market_data.get('volume_24h', 0)
        message += f"💰 *Price:* ${price:.4f}\n"
        message += f"📈 *Volume 24h:* ${volume:,.0f}\n\n"
    
    # RL Agent decision
    confidence = rl_prediction.get('confidence', 0)
    q_values = rl_prediction.get('q_values', [0, 0])
    message += f"🤖 *RL Decision:* {'ALERT' if rl_prediction.get('should_alert') else 'HOLD'}\n"
    message += f"🎯 *Confidence:* {confidence:.3f}\n"
    message += f"⚖️ *Q\\-Values:* \\[{q_values[0]:.3f}, {q_values[1]:.3f}\\]\n\n"
    
    # Top suspicious addresses
    high_risk_addresses = [(addr, score) for addr, score in anomaly_scores.items() if score > 0.7]
    medium_risk_addresses = [(addr, score) for addr, score in anomaly_scores.items() if 0.4 <= score <= 0.7]
    
    if high_risk_addresses:
        message += f"🔴 *High Risk Addresses* \\(>{0.7}\\):\n"
        for addr, score in sorted(high_risk_addresses, key=lambda x: x[1], reverse=True)[:5]:
            addr_short = f"{addr[:6]}...{addr[-4:]}" if len(addr) > 10 else addr
            message += f"• `{escape_markdown(addr_short)}`: {score:.3f}\n"
        message += "\n"
    
    if medium_risk_addresses:
        message += f"🟡 *Medium Risk Addresses* \\({0.4}\\-{0.7}\\):\n"
        for addr, score in sorted(medium_risk_addresses, key=lambda x: x[1], reverse=True)[:3]:
            addr_short = f"{addr[:6]}...{addr[-4:]}" if len(addr) > 10 else addr
            message += f"• `{escape_markdown(addr_short)}`: {score:.3f}\n"
        message += "\n"
    
    # Summary statistics
    total_addresses = len(anomaly_scores)
    avg_score = sum(anomaly_scores.values()) / max(total_addresses, 1)
    message += f"📊 *Summary:*\n"
    message += f"• Total addresses: {total_addresses}\n"
    message += f"• Average anomaly: {avg_score:.3f}\n"
    message += f"• High risk count: {len(high_risk_addresses)}\n"
    message += f"• Medium risk count: {len(medium_risk_addresses)}\n\n"
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message += f"⏰ *Time:* {escape_markdown(timestamp)}"
    
    return message

def process_alert_decision(symbol: str, anomaly_scores: Dict[str, float], 
                          rl_agent: RLAgent, market_data: Dict[str, Any] = None,
                          token: str = None, chat_id: str = None) -> Dict[str, Any]:
    """
    Przekształca anomaly_scores w decyzję alertu → wysyła alert jeśli RL to zatwierdzi.
    
    Args:
        symbol: Symbol tokena
        anomaly_scores: Wyniki z detect_graph_anomalies()
        rl_agent: Instancja RLAgent
        market_data: Opcjonalne dane rynkowe
        token: Telegram bot token (opcjonalnie, użyje env var)
        chat_id: Telegram chat ID (opcjonalnie, użyje env var)
        
    Returns:
        Dictionary z rezultatem przetwarzania alertu
    """
    logger.info(f"[ALERT PROCESS] {symbol}: Processing {len(anomaly_scores)} anomaly scores")
    
    # Use environment variables if not provided
    bot_token = token or TELEGRAM_BOT_TOKEN
    target_chat_id = chat_id or TELEGRAM_CHAT_ID
    
    # Convert anomaly scores to list for RL agent
    score_values = list(anomaly_scores.values())
    
    # Get RL agent prediction
    rl_prediction = rl_agent.predict_alert(score_values, symbol=symbol)
    
    result = {
        'symbol': symbol,
        'anomaly_scores': anomaly_scores,
        'rl_prediction': rl_prediction,
        'alert_sent': False,
        'alert_message': None,
        'telegram_success': False,
        'timestamp': datetime.now().isoformat()
    }
    
    # Check if RL agent recommends alert
    if rl_prediction.get('should_alert', False):
        # Check if we have high-risk addresses
        high_risk_count = sum(1 for score in anomaly_scores.values() if score > 0.7)
        medium_risk_count = sum(1 for score in anomaly_scores.values() if score >= 0.4)
        
        if high_risk_count > 0 or medium_risk_count >= 2:
            # Format alert message
            alert_message = format_alert_message(symbol, anomaly_scores, rl_prediction, market_data)
            result['alert_message'] = alert_message
            result['alert_sent'] = True
            
            # Send Telegram alert
            if bot_token and target_chat_id:
                telegram_success = send_telegram_alert(bot_token, target_chat_id, alert_message)
                result['telegram_success'] = telegram_success
                
                logger.info(f"[ALERT DECISION] {symbol}: ALERT SENT "
                           f"(high_risk: {high_risk_count}, medium_risk: {medium_risk_count}, "
                           f"telegram: {'✅' if telegram_success else '❌'})")
            else:
                logger.warning(f"[ALERT DECISION] {symbol}: Alert prepared but no Telegram credentials")
        else:
            logger.info(f"[ALERT DECISION] {symbol}: RL recommends alert but risk scores too low "
                       f"(high: {high_risk_count}, medium: {medium_risk_count})")
    else:
        logger.info(f"[ALERT DECISION] {symbol}: RL recommends HOLD - no alert sent")
    
    return result

def process_batch_alerts(tokens_data: List[Dict[str, Any]], rl_agent: RLAgent) -> List[Dict[str, Any]]:
    """
    Przetwarza batch alertów dla wielu tokenów.
    
    Args:
        tokens_data: Lista słowników z danymi tokenów
        rl_agent: Instancja RLAgent
        
    Returns:
        Lista rezultatów przetwarzania alertów
    """
    results = []
    alerts_sent = 0
    
    logger.info(f"[BATCH ALERTS] Processing {len(tokens_data)} tokens")
    
    for token_data in tokens_data:
        symbol = token_data.get('symbol')
        anomaly_scores = token_data.get('anomaly_scores', {})
        market_data = token_data.get('market_data', {})
        
        if not symbol or not anomaly_scores:
            logger.warning(f"[BATCH ALERTS] Skipping incomplete token data: {symbol}")
            continue
        
        try:
            result = process_alert_decision(symbol, anomaly_scores, rl_agent, market_data)
            results.append(result)
            
            if result.get('alert_sent'):
                alerts_sent += 1
                
        except Exception as e:
            logger.error(f"[BATCH ALERTS] Error processing {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    logger.info(f"[BATCH ALERTS] Completed: {alerts_sent} alerts sent from {len(tokens_data)} tokens")
    
    return results

def save_alert_history(alert_results: List[Dict[str, Any]], 
                      history_file: str = "cache/alert_history_gnn.json"):
    """
    Zapisuje historię alertów do pliku.
    
    Args:
        alert_results: Lista rezultatów alertów
        history_file: Ścieżka do pliku historii
    """
    try:
        # Load existing history
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        # Add new results
        history.extend(alert_results)
        
        # Keep only last 1000 entries
        history = history[-1000:]
        
        # Save updated history
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"[ALERT HISTORY] Saved {len(alert_results)} new alerts to {history_file}")
        
    except Exception as e:
        logger.error(f"[ALERT HISTORY] Error saving: {e}")

def get_alert_stats(history_file: str = "cache/alert_history_gnn.json") -> Dict[str, Any]:
    """
    Pobiera statystyki alertów z historii.
    
    Args:
        history_file: Ścieżka do pliku historii
        
    Returns:
        Słownik ze statystykami
    """
    try:
        if not os.path.exists(history_file):
            return {'total_alerts': 0, 'telegram_success_rate': 0}
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        total_alerts = len([h for h in history if h.get('alert_sent')])
        telegram_success = len([h for h in history if h.get('telegram_success')])
        
        stats = {
            'total_entries': len(history),
            'total_alerts': total_alerts,
            'telegram_success': telegram_success,
            'telegram_success_rate': telegram_success / max(total_alerts, 1),
            'recent_24h': len([h for h in history 
                              if (datetime.now() - datetime.fromisoformat(h.get('timestamp', '2000-01-01T00:00:00'))).days < 1])
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"[ALERT STATS] Error: {e}")
        return {'error': str(e)}

def test_alert_manager():
    """Test Alert Manager functionality"""
    print("\n🧪 Testing Alert Manager...")
    
    # Initialize RL Agent
    rl_agent = RLAgent(learning_rate=0.1, epsilon=0.1)
    
    # Test anomaly scores
    test_anomaly_scores = {
        "0x1234...abcd": 0.85,  # High risk
        "0x5678...efgh": 0.65,  # Medium risk
        "0x9abc...ijkl": 0.45,  # Medium risk
        "0xdefg...mnop": 0.12   # Low risk
    }
    
    test_market_data = {
        'price_usd': 1.2345,
        'volume_24h': 1500000
    }
    
    # Test alert processing
    result = process_alert_decision(
        symbol="TESTUSDT",
        anomaly_scores=test_anomaly_scores,
        rl_agent=rl_agent,
        market_data=test_market_data
    )
    
    print(f"✅ Alert decision result: {result['alert_sent']}")
    if result.get('alert_message'):
        print(f"✅ Alert message generated (length: {len(result['alert_message'])})")
    
    # Test batch processing
    tokens_data = [
        {
            'symbol': 'TOKEN1USDT',
            'anomaly_scores': {"0x1111": 0.9, "0x2222": 0.8},
            'market_data': {'price_usd': 2.5, 'volume_24h': 2000000}
        },
        {
            'symbol': 'TOKEN2USDT',
            'anomaly_scores': {"0x3333": 0.3, "0x4444": 0.2},
            'market_data': {'price_usd': 0.5, 'volume_24h': 500000}
        }
    ]
    
    batch_results = process_batch_alerts(tokens_data, rl_agent)
    print(f"✅ Batch processing: {len(batch_results)} results")
    
    # Test alert history
    save_alert_history(batch_results)
    stats = get_alert_stats()
    print(f"✅ Alert stats: {stats}")
    
    # Test message formatting
    test_rl_prediction = {
        'should_alert': True,
        'confidence': 0.85,
        'q_values': [0.2, 0.8]
    }
    
    formatted_message = format_alert_message(
        "TESTUSDT", test_anomaly_scores, test_rl_prediction, test_market_data
    )
    print(f"✅ Formatted message length: {len(formatted_message)}")
    
    print("🎉 Alert Manager test completed!")

if __name__ == "__main__":
    test_alert_manager()