"""
Address Tracker for Stealth Engine
Tracks repeated addresses in whale_ping and dex_inflow signals to detect accumulation patterns
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path

class AddressTracker:
    """
    Tracks wallet addresses appearing in stealth signals
    Detects repeated accumulation patterns by same addresses
    """
    
    def __init__(self, history_file: str = "cache/address_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.address_history = self._load_history()
    
    def _load_history(self) -> Dict:
        """Load address history from JSON file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"[ADDRESS TRACKER] Error loading history: {e}")
            return {}
    
    def _save_history(self):
        """Save address history to JSON file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.address_history, f, indent=2)
        except Exception as e:
            print(f"[ADDRESS TRACKER] Error saving history: {e}")
    
    def record_address_activity(self, token: str, address: str, usd_value: float, 
                               source: str, timestamp: str = None):
        """
        Record address activity for a token
        
        Args:
            token: Token symbol (e.g., "BTCUSDT")
            address: Wallet address
            usd_value: USD value of transaction/order
            source: Source of signal ("dex_inflow" or "whale_ping")
            timestamp: ISO timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Initialize token history if not exists
        if token not in self.address_history:
            self.address_history[token] = {}
        
        # Initialize address history if not exists
        if address not in self.address_history[token]:
            self.address_history[token][address] = []
        
        # Add new activity record
        activity = {
            "timestamp": timestamp,
            "usd_value": usd_value,
            "source": source
        }
        
        self.address_history[token][address].append(activity)
        
        print(f"[ADDRESS TRACKER] Recorded {source} activity: {token} | {address[:8]}... | ${usd_value:,.0f}")
        
        # Save updated history
        self._save_history()
    
    def get_repeated_addresses_boost(self, token: str, current_addresses: List[str], 
                                   history_days: int = 7) -> Tuple[float, Dict]:
        """
        Calculate boost score based on repeated address appearances
        
        Args:
            token: Token symbol
            current_addresses: List of addresses active in current scan
            history_days: Days to look back in history
            
        Returns:
            Tuple of (boost_score, details_dict)
        """
        if token not in self.address_history:
            return 0.0, {"repeated_addresses": 0, "details": []}
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=history_days)
        cutoff_iso = cutoff_date.isoformat()
        
        repeated_addresses = []
        total_boost = 0.0
        
        for address in current_addresses:
            if address in self.address_history[token]:
                # Filter activities within time window
                recent_activities = [
                    activity for activity in self.address_history[token][address]
                    if activity["timestamp"] >= cutoff_iso
                ]
                
                if len(recent_activities) > 1:  # Address appeared more than once
                    repeated_addresses.append({
                        "address": address,
                        "appearances": len(recent_activities),
                        "total_value": sum(a["usd_value"] for a in recent_activities),
                        "sources": list(set(a["source"] for a in recent_activities))
                    })
                    
                    # Boost calculation: +0.2 per repeated address, max +0.6
                    address_boost = min(0.2 * (len(recent_activities) - 1), 0.2)
                    total_boost += address_boost
        
        # Cap total boost at 0.6
        total_boost = min(total_boost, 0.6)
        
        details = {
            "repeated_addresses": len(repeated_addresses),
            "boost_score": total_boost,
            "details": repeated_addresses,
            "history_days": history_days
        }
        
        if total_boost > 0:
            print(f"[ADDRESS TRACKER] Repeated address boost for {token}: +{total_boost:.2f} ({len(repeated_addresses)} addresses)")
        
        return total_boost, details
    
    def cleanup_old_records(self, days_to_keep: int = 30):
        """
        Remove records older than specified days to keep file size manageable
        
        Args:
            days_to_keep: Number of days to keep in history
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_iso = cutoff_date.isoformat()
        
        cleaned_count = 0
        
        for token in list(self.address_history.keys()):
            for address in list(self.address_history[token].keys()):
                # Filter out old activities
                old_activities = self.address_history[token][address]
                new_activities = [
                    activity for activity in old_activities
                    if activity["timestamp"] >= cutoff_iso
                ]
                
                cleaned_count += len(old_activities) - len(new_activities)
                
                if new_activities:
                    self.address_history[token][address] = new_activities
                else:
                    # Remove address if no recent activities
                    del self.address_history[token][address]
            
            # Remove token if no addresses left
            if not self.address_history[token]:
                del self.address_history[token]
        
        if cleaned_count > 0:
            print(f"[ADDRESS TRACKER] Cleaned {cleaned_count} old records (>{days_to_keep} days)")
            self._save_history()
    
    def get_address_statistics(self, token: str = None, days: int = 7) -> Dict:
        """
        Get statistics about address activities
        
        Args:
            token: Specific token (None for all tokens)
            days: Days to analyze
            
        Returns:
            Statistics dictionary
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff_date.isoformat()
        
        stats = {
            "total_addresses": 0,
            "repeated_addresses": 0,
            "total_activities": 0,
            "tokens_analyzed": [],
            "top_addresses": []
        }
        
        tokens_to_analyze = [token] if token else list(self.address_history.keys())
        
        address_activity_count = {}
        
        for token_symbol in tokens_to_analyze:
            if token_symbol not in self.address_history:
                continue
                
            stats["tokens_analyzed"].append(token_symbol)
            
            for address, activities in self.address_history[token_symbol].items():
                recent_activities = [
                    a for a in activities if a["timestamp"] >= cutoff_iso
                ]
                
                if recent_activities:
                    stats["total_addresses"] += 1
                    stats["total_activities"] += len(recent_activities)
                    
                    if len(recent_activities) > 1:
                        stats["repeated_addresses"] += 1
                    
                    # Track address activity for top addresses
                    addr_key = f"{address[:8]}...({token_symbol})"
                    address_activity_count[addr_key] = len(recent_activities)
        
        # Get top 5 most active addresses
        sorted_addresses = sorted(
            address_activity_count.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        stats["top_addresses"] = [
            {"address": addr, "activities": count}
            for addr, count in sorted_addresses
        ]
        
        return stats

# Global instance
address_tracker = AddressTracker()