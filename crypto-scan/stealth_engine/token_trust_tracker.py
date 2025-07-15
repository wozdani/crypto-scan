#!/usr/bin/env python3
"""
Token Trust Tracker - STAGE 13 Implementation
Tracks wallet address trust scores based on historical prediction accuracy
"""

import json
import os
import threading
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta, timezone

class TokenTrustTracker:
    """
    Tracker for wallet address trust scores within individual tokens
    Each token maintains independent trust profiles
    """
    
    def __init__(self, cache_dir: str = "crypto-scan/cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "token_trust_scores.json")
        self.lock = threading.Lock()
        self.trust_data = self._load_trust_data()
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    def _load_trust_data(self) -> Dict:
        """Load trust data from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"[TOKEN TRUST] Error loading trust data: {e}")
            return {}
    
    def _save_trust_data(self):
        """Save trust data to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.trust_data, f, indent=2)
        except Exception as e:
            print(f"[TOKEN TRUST] Error saving trust data: {e}")
    
    def update_token_trust(self, token: str, addresses: List[str], source: str = "stealth"):
        """
        Update trust scores for addresses detected in token signals
        
        Args:
            token: Token symbol
            addresses: List of wallet addresses
            source: Signal source (whale_ping, dex_inflow, etc.)
        """
        with self.lock:
            if token not in self.trust_data:
                self.trust_data[token] = {
                    "addresses": {},
                    "total_signals": 0,
                    "recognized_addresses": 0
                }
            
            token_data = self.trust_data[token]
            current_time = datetime.now(timezone.utc).isoformat()
            
            for address in addresses:
                if address not in token_data["addresses"]:
                    token_data["addresses"][address] = {
                        "signal_count": 0,
                        "first_seen": current_time,
                        "last_seen": current_time,
                        "sources": []
                    }
                
                addr_data = token_data["addresses"][address]
                addr_data["signal_count"] += 1
                addr_data["last_seen"] = current_time
                
                if source not in addr_data["sources"]:
                    addr_data["sources"].append(source)
            
            token_data["total_signals"] += 1
            self._save_trust_data()
    
    def compute_trust_boost(self, token: str, detected_addresses: List[str]) -> float:
        """
        Compute trust boost based on address recognition ratio
        
        Args:
            token: Token symbol
            detected_addresses: Currently detected addresses
            
        Returns:
            Trust boost value (0.0-0.25)
        """
        with self.lock:
            if token not in self.trust_data or not detected_addresses:
                return 0.0
            
            token_data = self.trust_data[token]
            known_addresses = token_data["addresses"]
            
            if not known_addresses:
                return 0.0
            
            # Calculate recognition ratio
            recognized_count = 0
            for addr in detected_addresses:
                if addr in known_addresses:
                    recognized_count += 1
            
            recognition_ratio = recognized_count / len(detected_addresses)
            
            # Calculate average signal history for recognized addresses
            avg_history = 1.0
            if recognized_count > 0:
                total_signals = sum(
                    known_addresses[addr]["signal_count"] 
                    for addr in detected_addresses 
                    if addr in known_addresses
                )
                avg_history = total_signals / recognized_count
            
            # Trust boost formula: ratio × (avg_history-1) × 0.1, max 0.25
            trust_boost = min(recognition_ratio * (avg_history - 1) * 0.1, 0.25)
            
            return max(0.0, trust_boost)
    
    def get_token_trust_stats(self, token: str) -> Dict:
        """Get trust statistics for a token"""
        with self.lock:
            if token not in self.trust_data:
                return {
                    "total_addresses": 0,
                    "total_signals": 0,
                    "recognized_addresses": 0,
                    "recognition_ratio": 0.0
                }
            
            token_data = self.trust_data[token]
            total_addresses = len(token_data["addresses"])
            recognized_addresses = sum(
                1 for addr_data in token_data["addresses"].values() 
                if addr_data["signal_count"] > 1
            )
            
            recognition_ratio = (
                recognized_addresses / total_addresses 
                if total_addresses > 0 else 0.0
            )
            
            return {
                "total_addresses": total_addresses,
                "total_signals": token_data["total_signals"],
                "recognized_addresses": recognized_addresses,
                "recognition_ratio": recognition_ratio
            }
    
    def get_trust_statistics(self) -> Dict:
        """Get overall trust statistics"""
        with self.lock:
            total_tokens = len(self.trust_data)
            total_addresses = 0
            total_signals = 0
            recognized_addresses = 0
            
            for token_data in self.trust_data.values():
                total_addresses += len(token_data["addresses"])
                total_signals += token_data["total_signals"]
                recognized_addresses += sum(
                    1 for addr_data in token_data["addresses"].values()
                    if addr_data["signal_count"] > 1
                )
            
            return {
                "total_tokens": total_tokens,
                "total_addresses": total_addresses,
                "total_signals": total_signals,
                "recognized_addresses": recognized_addresses,
                "recognition_ratio": (
                    recognized_addresses / total_addresses 
                    if total_addresses > 0 else 0.0
                )
            }
    
    def cleanup_old_data(self, days: int = 30):
        """Cleanup old trust data"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        with self.lock:
            tokens_to_remove = []
            
            for token, token_data in self.trust_data.items():
                addresses_to_remove = []
                
                for addr, addr_data in token_data["addresses"].items():
                    if addr_data["last_seen"] < cutoff_str:
                        addresses_to_remove.append(addr)
                
                for addr in addresses_to_remove:
                    del token_data["addresses"][addr]
                
                if not token_data["addresses"]:
                    tokens_to_remove.append(token)
            
            for token in tokens_to_remove:
                del self.trust_data[token]
            
            self._save_trust_data()

# Global instance
_token_trust_tracker = None

def get_token_trust_tracker() -> TokenTrustTracker:
    """Get global TokenTrustTracker instance"""
    global _token_trust_tracker
    if _token_trust_tracker is None:
        _token_trust_tracker = TokenTrustTracker()
    return _token_trust_tracker

# Convenience functions
def update_token_trust(token: str, addresses: List[str], source: str = "stealth"):
    """Update token trust scores"""
    tracker = get_token_trust_tracker()
    tracker.update_token_trust(token, addresses, source)

def compute_trust_boost(token: str, detected_addresses: List[str]) -> float:
    """Compute trust boost for token"""
    tracker = get_token_trust_tracker()
    return tracker.compute_trust_boost(token, detected_addresses)

def get_token_trust_stats(token: str) -> Dict:
    """Get trust stats for token"""
    tracker = get_token_trust_tracker()
    return tracker.get_token_trust_stats(token)

def get_trust_statistics() -> Dict:
    """Get overall trust statistics"""
    tracker = get_token_trust_tracker()
    return tracker.get_trust_statistics()