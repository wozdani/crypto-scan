"""
LRU store for last 10 tokens with persistent storage
"""
import json
import os
from typing import Dict, Any, Optional
from collections import OrderedDict
import time

class Last10Store:
    """LRU cache for last 10 tokens with JSON persistence"""
    
    def __init__(self, capacity: int = 10, storage_path: str = "state/last10_tokens.json"):
        self.capacity = capacity
        self.storage_path = storage_path
        self._data = OrderedDict()
        self._load_from_disk()
    
    def _load_from_disk(self) -> None:
        """Load data from JSON file if exists"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                # Restore order (most recent first)
                self._data = OrderedDict(data.items())
            except (json.JSONDecodeError, FileNotFoundError):
                self._data = OrderedDict()
    
    def _save_to_disk(self) -> None:
        """Persist data to JSON file"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(dict(self._data), f, indent=2)
    
    def record(self, symbol: str, ts: float, features_by_detector: Dict[str, Dict[str, Any]]) -> None:
        """
        Record token with active detectors only
        
        Args:
            symbol: Token symbol (e.g., "BTCUSDT")
            ts: Timestamp (float)
            features_by_detector: Dict of {detector_name: {feature_dict}} for ACTIVE detectors only
        """
        # Remove if already exists (to update position)
        if symbol in self._data:
            del self._data[symbol]
        
        # Add new record at end (most recent)
        self._data[symbol] = {
            "ts": int(ts),
            "detectors": list(features_by_detector.keys()),
            "feats": features_by_detector
        }
        
        # Maintain LRU capacity
        while len(self._data) > self.capacity:
            # Remove oldest (first item)
            self._data.popitem(last=False)
        
        # Persist to disk
        self._save_to_disk()
    
    def get_last10(self) -> Dict[str, Dict[str, Any]]:
        """
        Get last 10 tokens
        
        Returns:
            Dict of {symbol: {ts: int, detectors: [str], feats: {detector: {...}}}}
        """
        return dict(self._data)
    
    def clear(self) -> None:
        """Clear all data"""
        self._data.clear()
        self._save_to_disk()

# Global singleton instance
_last10_store = None

def get_last10_store() -> Last10Store:
    """Get global Last10Store instance"""
    global _last10_store
    if _last10_store is None:
        _last10_store = Last10Store()
    return _last10_store