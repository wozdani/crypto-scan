"""
Trend Mode Alert Cache System
Prevents spamming of trend mode alerts
"""

import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Set


class TrendAlertCache:
    """Manages trend mode alert cooldowns"""
    
    def __init__(self, cache_file: str = "data/trend_alert_cache.json", cooldown_minutes: int = 60):
        self.cache_file = cache_file
        self.cooldown_minutes = cooldown_minutes
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
    
    def _load_cache(self) -> Dict:
        """Load alert cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading trend alert cache: {e}")
        return {}
    
    def _save_cache(self, cache: Dict):
        """Save alert cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving trend alert cache: {e}")
    
    def _cleanup_expired(self, cache: Dict) -> Dict:
        """Remove expired entries from cache"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=self.cooldown_minutes)
        cutoff_str = cutoff_time.isoformat()
        
        cleaned = {}
        for symbol, last_alert_time in cache.items():
            if last_alert_time > cutoff_str:
                cleaned[symbol] = last_alert_time
        
        return cleaned
    
    def already_alerted_recently(self, symbol: str, alert_type: str = "trend_mode") -> bool:
        """
        Check if symbol was already alerted recently
        
        Args:
            symbol: Trading symbol
            alert_type: Type of alert (default: "trend_mode")
            
        Returns:
            bool: True if already alerted within cooldown period
        """
        cache = self._load_cache()
        cache = self._cleanup_expired(cache)
        
        cache_key = f"{symbol}_{alert_type}"
        return cache_key in cache
    
    def mark_alert_sent(self, symbol: str, alert_type: str = "trend_mode"):
        """
        Mark that an alert was sent for symbol
        
        Args:
            symbol: Trading symbol
            alert_type: Type of alert (default: "trend_mode")
        """
        cache = self._load_cache()
        cache = self._cleanup_expired(cache)
        
        cache_key = f"{symbol}_{alert_type}"
        cache[cache_key] = datetime.now(timezone.utc).isoformat()
        
        self._save_cache(cache)
    
    def get_cooldown_status(self, symbol: str, alert_type: str = "trend_mode") -> Dict:
        """
        Get cooldown status for symbol
        
        Args:
            symbol: Trading symbol
            alert_type: Type of alert
            
        Returns:
            dict: {"on_cooldown": bool, "minutes_remaining": int}
        """
        cache = self._load_cache()
        cache_key = f"{symbol}_{alert_type}"
        
        if cache_key not in cache:
            return {"on_cooldown": False, "minutes_remaining": 0}
        
        last_alert = datetime.fromisoformat(cache[cache_key].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        elapsed = now - last_alert
        
        if elapsed.total_seconds() >= self.cooldown_minutes * 60:
            return {"on_cooldown": False, "minutes_remaining": 0}
        
        remaining_seconds = (self.cooldown_minutes * 60) - elapsed.total_seconds()
        remaining_minutes = int(remaining_seconds / 60)
        
        return {"on_cooldown": True, "minutes_remaining": remaining_minutes}


# Global instance
trend_alert_cache = TrendAlertCache()