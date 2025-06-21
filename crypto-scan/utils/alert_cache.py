"""
Alert Cache Management System
Handles dynamic alert updates and active alert tracking
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

ALERT_CACHE_FILE = "data/alerts_cache.json"

def load_alert_cache() -> Dict[str, Any]:
    """Load active alerts cache from file"""
    try:
        if os.path.exists(ALERT_CACHE_FILE):
            with open(ALERT_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"⚠️ Error loading alert cache: {e}")
        return {}

def save_alert_cache(cache: Dict[str, Any]):
    """Save active alerts cache to file"""
    try:
        os.makedirs(os.path.dirname(ALERT_CACHE_FILE), exist_ok=True)
        with open(ALERT_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, default=str)
    except Exception as e:
        print(f"❌ Error saving alert cache: {e}")

def is_alert_active(symbol: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check if symbol has an active alert
    Returns: (is_active, alert_data)
    """
    cache = load_alert_cache()
    
    if symbol not in cache:
        return False, None
        
    alert_data = cache[symbol]
    created_at = datetime.fromisoformat(alert_data['created_at'])
    now = datetime.now(timezone.utc)
    
    # Alert expires after 2 hours
    if now - created_at > timedelta(hours=2):
        remove_active_alert(symbol)
        return False, None
        
    return True, alert_data

def add_active_alert(symbol: str, ppwcs_score: float, signals: Dict[str, Any], 
                    alert_level: str, initial_signals: List[str]):
    """Add symbol to active alerts cache"""
    cache = load_alert_cache()
    
    cache[symbol] = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'last_updated': datetime.now(timezone.utc).isoformat(),
        'ppwcs_score': ppwcs_score,
        'alert_level': alert_level,
        'initial_signals': initial_signals,
        'current_signals': list(signals.keys()),
        'update_count': 0,
        'last_telegram_sent': datetime.now(timezone.utc).isoformat()
    }
    
    save_alert_cache(cache)


def remove_active_alert(symbol: str):
    """Remove symbol from active alerts cache"""
    cache = load_alert_cache()
    
    if symbol in cache:
        del cache[symbol]
        save_alert_cache(cache)
        print(f"🗑️ Removed active alert for {symbol}")

def detect_new_signals(symbol: str, current_signals: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Detect if there are new signals compared to cached alert
    Returns: (has_new_signals, list_of_new_signals)
    """
    is_active, alert_data = is_alert_active(symbol)
    
    if not is_active or not alert_data:
        return False, []
    
    # Get currently active signals (True values only)
    current_active = [k for k, v in current_signals.items() if v is True]
    
    # Get previously detected signals
    previous_signals = alert_data.get('current_signals', [])
    
    # Find new signals
    new_signals = [sig for sig in current_active if sig not in previous_signals]
    
    # Special signals that always trigger updates
    priority_signals = ['dex_inflow', 'stealth_inflow', 'event_tag', 'spoofing', 
                       'whale_sequence', 'heatmap_exhaustion', 'fake_reject']
    
    # Check if any new signal is a priority signal
    priority_new = [sig for sig in new_signals if sig in priority_signals]
    
    has_new = len(new_signals) > 0 or len(priority_new) > 0
    
    if has_new:
        print(f"🔄 New signals detected for {symbol}: {new_signals}")
        if priority_new:
            print(f"🚨 Priority signals: {priority_new}")
    
    return has_new, new_signals

def should_update_alert(symbol: str, new_signals: Dict[str, Any], ppwcs_score: float) -> Tuple[bool, str]:
    """
    Określa, czy alert dla danego symbolu powinien zostać zaktualizowany.
    
    Args:
        symbol: symbol tokena
        new_signals: dict z nowymi sygnałami (np. {"dex_inflow": True, "spoofing": True})
        ppwcs_score: aktualny wynik scoringu
    
    Returns:
        tuple: (update_needed, reason)
    """
    cache = load_alert_cache()
    now = datetime.now(timezone.utc)
    cooldown = timedelta(minutes=60)
    significant_keys = {"dex_inflow", "spoofing", "event_tag", "stealth_inflow", "whale_sequence", "heatmap_exhaustion"}

    # Jeśli token nie ma aktywnego alertu – nowy alert
    if symbol not in cache:
        return False, "no_active_alert"

    alert_data = cache[symbol]
    last_alert_time = datetime.fromisoformat(alert_data['created_at'])
    last_ppwcs = alert_data.get('ppwcs_score', 0)
    previous_signals = {k: True for k in alert_data.get('current_signals', [])}

    # Jeśli minęła godzina – nowy alert (expired, nie update)
    if now - last_alert_time > cooldown:
        return False, "cooldown_expired"

    # Jeśli pojawił się nowy istotny sygnał – zaktualizuj alert
    for key in significant_keys:
        if new_signals.get(key) and not previous_signals.get(key):
            return True, f"new_signal:{key}"

    # Jeśli PPWCS wzrosło znacząco (o ≥5)
    if ppwcs_score - last_ppwcs >= 5:
        return True, "ppwcs_rise"

    # Sprawdź cooldown dla aktualizacji (minimum 15 minut między aktualizacjami)
    last_updated = datetime.fromisoformat(alert_data.get('last_updated', alert_data['created_at']))
    if now - last_updated < timedelta(minutes=15):
        return False, "update_cooldown"

    # Inaczej – brak potrzeby aktualizacji
    return False, "no_update_needed"

def update_active_alert(symbol: str, signals: Dict[str, Any], ppwcs_score: float, 
                       new_signals: List[str]) -> Dict[str, Any]:
    """
    Update existing active alert with new signals and score
    Returns: updated alert data
    """
    cache = load_alert_cache()
    
    if symbol not in cache:
        return {}
    
    # Update alert data
    if symbol not in cache:
        return {}
        
    alert_data = cache[symbol]
    alert_data['last_updated'] = datetime.now(timezone.utc).isoformat()
    alert_data['ppwcs_score'] = max(alert_data.get('ppwcs_score', 0), ppwcs_score)  # Keep highest score
    alert_data['current_signals'] = [k for k, v in signals.items() if v is True]
    alert_data['update_count'] = alert_data.get('update_count', 0) + 1
    alert_data['latest_new_signals'] = new_signals
    
    cache[symbol] = alert_data
    save_alert_cache(cache)
    

    
    return alert_data

def get_active_alerts_summary() -> Dict[str, Any]:
    """Get summary of all active alerts"""
    cache = load_alert_cache()
    now = datetime.now(timezone.utc)
    
    active_count = 0
    expired_symbols = []
    
    for symbol, alert_data in cache.items():
        created_at = datetime.fromisoformat(alert_data['created_at'])
        if now - created_at <= timedelta(hours=2):
            active_count += 1
        else:
            expired_symbols.append(symbol)
    
    # Clean up expired alerts
    for symbol in expired_symbols:
        remove_active_alert(symbol)
    
    return {
        'active_count': active_count,
        'expired_removed': len(expired_symbols),
        'total_symbols': len(cache)
    }

def cleanup_expired_alerts():
    """Remove expired alerts from cache"""
    cache = load_alert_cache()
    now = datetime.now(timezone.utc)
    expired_symbols = []
    
    for symbol, alert_data in cache.items():
        created_at = datetime.fromisoformat(alert_data['created_at'])
        if now - created_at > timedelta(hours=2):
            expired_symbols.append(symbol)
    
    for symbol in expired_symbols:
        del cache[symbol]
    
    if expired_symbols:
        save_alert_cache(cache)
        print(f"🧹 Cleaned up {len(expired_symbols)} expired alerts")
    
    return len(expired_symbols)