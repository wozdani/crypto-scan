"""
Stealth V3 Watchlist Alert System
Wyśle watchlist alerty dla tokenów z score 0.4-0.6 + HOLD decision
zgodnie z propozycją Szefira
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import requests

def send_watchlist_alert(watchlist_data: Dict[str, Any]) -> bool:
    """
    Wyślij watchlist alert na Telegram dla tokena w zakresie 0.4-0.6
    
    Args:
        watchlist_data: Dane watchlist zawierające symbol, score, decision, etc.
        
    Returns:
        bool: True jeśli alert wysłany pomyślnie
    """
    try:
        # Pobierz konfigurację Telegram
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not bot_token or not chat_id:
            print("[WATCHLIST ALERT] Brak konfiguracji Telegram")
            return False
            
        # Sprawdź cooldown
        if _is_watchlist_cooldown(watchlist_data["symbol"]):
            print(f"[WATCHLIST ALERT] {watchlist_data['symbol']}: W cooldown")
            return False
            
        # Przygotuj wiadomość watchlist
        message = _format_watchlist_message(watchlist_data)
        
        # Wyślij przez Telegram API
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            _update_watchlist_cooldown(watchlist_data["symbol"])
            _save_watchlist_history(watchlist_data)
            return True
        else:
            print(f"[WATCHLIST ALERT] Telegram API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[WATCHLIST ALERT ERROR] {e}")
        return False

def _format_watchlist_message(watchlist_data: Dict[str, Any]) -> str:
    """
    Formatuj wiadomość watchlist alert
    
    Args:
        watchlist_data: Dane watchlist
        
    Returns:
        str: Sformatowana wiadomość HTML
    """
    symbol = watchlist_data["symbol"]
    score = watchlist_data["score"]
    decision = watchlist_data["decision"]
    active_detectors = watchlist_data["active_detectors"]
    
    # Emoji na podstawie score
    if score >= 0.6:
        emoji = "👀"
        level = "HIGH WATCH"
    elif score >= 0.5:
        emoji = "📊"
        level = "MEDIUM WATCH"
    else:
        emoji = "⚠️"
        level = "LOW WATCH"
        
    message = f"""
{emoji} <b>WATCHLIST ALERT</b> - {level}

🎯 <b>Symbol:</b> {symbol}
📊 <b>Score:</b> {score:.3f}
🤖 <b>Decision:</b> {decision}
🔍 <b>Active Detectors:</b> {active_detectors}/3

💡 <b>Status:</b> Token w obserwacji
📈 <b>Recommendation:</b> Monitor for changes
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S UTC')}

<i>Watchlist alerts są wysyłane dla tokenów z score 0.4-0.6 + HOLD decision</i>
"""
    
    # Dodaj informacje o GNN i mastermind jeśli dostępne
    if watchlist_data.get("gnn_score"):
        message += f"\n🧠 <b>GNN Score:</b> {watchlist_data['gnn_score']:.3f}"
        
    if watchlist_data.get("mastermind_addresses", 0) > 0:
        message += f"\n🎭 <b>Mastermind:</b> {watchlist_data['mastermind_addresses']} addresses"
        
    return message.strip()

def _is_watchlist_cooldown(symbol: str) -> bool:
    """
    Sprawdź czy token jest w cooldown dla watchlist alertów
    
    Args:
        symbol: Symbol tokena
        
    Returns:
        bool: True jeśli w cooldown
    """
    try:
        cooldown_file = "crypto-scan/cache/watchlist_cooldown.json"
        
        if not os.path.exists(cooldown_file):
            return False
            
        with open(cooldown_file, 'r') as f:
            cooldown_data = json.load(f)
            
        if symbol in cooldown_data:
            last_alert = cooldown_data[symbol]
            current_time = time.time()
            
            # Cooldown 2 godziny dla watchlist alertów
            if current_time - last_alert < 7200:  # 2 hours
                return True
                
        return False
        
    except Exception:
        return False

def _update_watchlist_cooldown(symbol: str):
    """
    Aktualizuj cooldown dla watchlist alertów
    
    Args:
        symbol: Symbol tokena
    """
    try:
        cooldown_file = "crypto-scan/cache/watchlist_cooldown.json"
        
        # Załaduj istniejące dane
        cooldown_data = {}
        if os.path.exists(cooldown_file):
            with open(cooldown_file, 'r') as f:
                cooldown_data = json.load(f)
                
        # Aktualizuj timestamp
        cooldown_data[symbol] = time.time()
        
        # Zapisz
        os.makedirs(os.path.dirname(cooldown_file), exist_ok=True)
        with open(cooldown_file, 'w') as f:
            json.dump(cooldown_data, f, indent=2)
            
    except Exception as e:
        print(f"[WATCHLIST COOLDOWN ERROR] {e}")

def _save_watchlist_history(watchlist_data: Dict[str, Any]):
    """
    Zapisz historię watchlist alertów
    
    Args:
        watchlist_data: Dane watchlist
    """
    try:
        history_file = "crypto-scan/cache/watchlist_history.json"
        
        # Załaduj istniejącą historię
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
                
        # Dodaj nowy wpis
        history_entry = {
            "symbol": watchlist_data["symbol"],
            "score": watchlist_data["score"],
            "decision": watchlist_data["decision"],
            "active_detectors": watchlist_data["active_detectors"],
            "timestamp": datetime.now().isoformat(),
            "unix_timestamp": time.time()
        }
        
        history.append(history_entry)
        
        # Zachowaj ostatnie 1000 wpisów
        if len(history) > 1000:
            history = history[-1000:]
            
        # Zapisz
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        print(f"[WATCHLIST HISTORY ERROR] {e}")

def get_watchlist_stats() -> Dict[str, Any]:
    """
    Pobierz statystyki watchlist alertów
    
    Returns:
        Dict[str, Any]: Statystyki
    """
    try:
        history_file = "crypto-scan/cache/watchlist_history.json"
        
        if not os.path.exists(history_file):
            return {
                "total_alerts": 0,
                "last_24h": 0,
                "top_symbols": [],
                "score_distribution": {}
            }
            
        with open(history_file, 'r') as f:
            history = json.load(f)
            
        # Oblicz statystyki
        total_alerts = len(history)
        current_time = time.time()
        last_24h = len([h for h in history if current_time - h["unix_timestamp"] < 86400])
        
        # Top symbole
        symbol_counts = {}
        for entry in history:
            symbol = entry["symbol"]
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Rozkład score
        score_ranges = {
            "0.4-0.45": 0,
            "0.45-0.5": 0,
            "0.5-0.55": 0,
            "0.55-0.6": 0,
            "0.6-0.65": 0,
            "0.65-0.7": 0
        }
        
        for entry in history:
            score = entry["score"]
            if score < 0.45:
                score_ranges["0.4-0.45"] += 1
            elif score < 0.5:
                score_ranges["0.45-0.5"] += 1
            elif score < 0.55:
                score_ranges["0.5-0.55"] += 1
            elif score < 0.6:
                score_ranges["0.55-0.6"] += 1
            elif score < 0.65:
                score_ranges["0.6-0.65"] += 1
            else:
                score_ranges["0.65-0.7"] += 1
                
        return {
            "total_alerts": total_alerts,
            "last_24h": last_24h,
            "top_symbols": top_symbols,
            "score_distribution": score_ranges
        }
        
    except Exception as e:
        print(f"[WATCHLIST STATS ERROR] {e}")
        return {
            "total_alerts": 0,
            "last_24h": 0,
            "top_symbols": [],
            "score_distribution": {}
        }

# Compatibility function for backwards compatibility
def send_stealth_v3_watchlist_alert(watchlist_data: Dict[str, Any]) -> bool:
    """Compatibility wrapper for watchlist alerts"""
    return send_watchlist_alert(watchlist_data)