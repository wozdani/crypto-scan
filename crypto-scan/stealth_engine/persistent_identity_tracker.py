#!/usr/bin/env python3
"""
Persistent Identity Tracker - STAGE 14 Implementation
Tracks wallet identity scores based on prediction accuracy
"""

import json
import os
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class PersistentIdentityTracker:
    """
    Tracks wallet identity scores and prediction accuracy
    Provides boost calculation based on historical performance
    """
    
    def __init__(self, cache_dir: str = "crypto-scan/cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "wallet_identity_score.json")
        self.lock = threading.Lock()
        self.identity_data = self._load_identity_data()
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    def _load_identity_data(self) -> Dict:
        """Load identity data from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"[IDENTITY TRACKER] Error loading identity data: {e}")
            return {}
    
    def _save_identity_data(self):
        """Save identity data to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.identity_data, f, indent=2)
        except Exception as e:
            print(f"[IDENTITY TRACKER] Error saving identity data: {e}")
    
    def update_wallet_identity(self, address: str, token: str, prediction_success: bool, score_impact: float = 1.0):
        """
        Update wallet identity score based on prediction outcome
        
        Args:
            address: Wallet address
            token: Token symbol where prediction was made
            prediction_success: Whether prediction was successful
            score_impact: Impact weight for this prediction
        """
        with self.lock:
            current_time = datetime.now().isoformat()
            
            if address not in self.identity_data:
                self.identity_data[address] = {
                    "score": 0.0,
                    "total_predictions": 0,
                    "successful_predictions": 0,
                    "success_rate": 0.0,
                    "last_seen": current_time,
                    "last_token": token,
                    "prediction_history": []
                }
            
            wallet_data = self.identity_data[address]
            
            # Update basic stats
            wallet_data["total_predictions"] += 1
            if prediction_success:
                wallet_data["successful_predictions"] += 1
            
            wallet_data["success_rate"] = (
                wallet_data["successful_predictions"] / wallet_data["total_predictions"]
            )
            wallet_data["last_seen"] = current_time
            wallet_data["last_token"] = token
            
            # Add to prediction history (keep last 50)
            wallet_data["prediction_history"].append({
                "token": token,
                "success": prediction_success,
                "timestamp": current_time,
                "impact": score_impact
            })
            
            if len(wallet_data["prediction_history"]) > 50:
                wallet_data["prediction_history"] = wallet_data["prediction_history"][-50:]
            
            # Update identity score based on success rate and impact
            if wallet_data["total_predictions"] >= 3:  # Minimum predictions for scoring
                base_score = wallet_data["success_rate"] * score_impact
                wallet_data["score"] = min(base_score, 1.0)
            
            self._save_identity_data()
    
    def get_identity_boost(self, addresses: List[str]) -> float:
        """
        Calculate identity boost based on wallet recognition and performance
        
        Args:
            addresses: List of detected wallet addresses
            
        Returns:
            Identity boost value (0.0-0.2)
        """
        if not addresses:
            return 0.0
        
        with self.lock:
            # Limit to 50 addresses max for performance
            addresses = addresses[:50]
            
            total_boost = 0.0
            recognized_count = 0
            
            for address in addresses:
                if address in self.identity_data:
                    wallet_data = self.identity_data[address]
                    
                    # Only count wallets with sufficient history
                    if wallet_data["total_predictions"] >= 3:
                        recognized_count += 1
                        
                        # Calculate boost based on score (avg_score × 0.05, max 0.2)
                        wallet_boost = min(wallet_data["score"] * 0.05, 0.2)
                        total_boost += wallet_boost
            
            if recognized_count == 0:
                return 0.0
            
            # Calculate recognition ratio boost
            recognition_ratio = recognized_count / len(addresses)
            
            # Average boost with recognition ratio weight
            avg_boost = total_boost / recognized_count
            final_boost = avg_boost * recognition_ratio
            
            return min(final_boost, 0.2)
    
    def get_wallet_identity_stats(self, address: str) -> Optional[Dict]:
        """Get identity statistics for specific wallet"""
        with self.lock:
            if address not in self.identity_data:
                return None
            
            return self.identity_data[address].copy()
    
    def get_top_identity_wallets(self, limit: int = 20) -> List[Dict]:
        """Get top performing wallets by identity score"""
        with self.lock:
            wallets = []
            
            for address, data in self.identity_data.items():
                if data["total_predictions"] >= 3:  # Minimum predictions
                    wallets.append({
                        "address": address,
                        "score": data["score"],
                        "success_rate": data["success_rate"],
                        "total_predictions": data["total_predictions"],
                        "last_token": data["last_token"],
                        "last_seen": data["last_seen"]
                    })
            
            # Sort by score descending
            wallets.sort(key=lambda x: x["score"], reverse=True)
            return wallets[:limit]
    
    def get_identity_statistics(self) -> Dict:
        """Get overall identity tracking statistics"""
        with self.lock:
            total_wallets = len(self.identity_data)
            total_predictions = sum(
                data["total_predictions"] for data in self.identity_data.values()
            )
            total_successful = sum(
                data["successful_predictions"] for data in self.identity_data.values()
            )
            
            overall_success_rate = (
                total_successful / total_predictions 
                if total_predictions > 0 else 0.0
            )
            
            high_score_wallets = sum(
                1 for data in self.identity_data.values() 
                if data["score"] > 0.5 and data["total_predictions"] >= 3
            )
            
            # Active wallets in last 24h
            cutoff_time = (datetime.now() - timedelta(hours=24)).isoformat()
            active_wallets_24h = sum(
                1 for data in self.identity_data.values() 
                if data["last_seen"] > cutoff_time
            )
            
            return {
                "total_wallets": total_wallets,
                "total_predictions": total_predictions,
                "successful_predictions": total_successful,
                "overall_success_rate": overall_success_rate,
                "high_score_wallets": high_score_wallets,
                "active_wallets_24h": active_wallets_24h
            }
    
    def cleanup_old_wallets(self, days: int = 30):
        """Cleanup old wallet data"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        with self.lock:
            wallets_to_remove = []
            
            for address, data in self.identity_data.items():
                if data["last_seen"] < cutoff_str:
                    wallets_to_remove.append(address)
            
            for address in wallets_to_remove:
                del self.identity_data[address]
            
            self._save_identity_data()

# Global instance
_persistent_identity_tracker = None

def get_persistent_identity_tracker() -> PersistentIdentityTracker:
    """Get global PersistentIdentityTracker instance"""
    global _persistent_identity_tracker
    if _persistent_identity_tracker is None:
        _persistent_identity_tracker = PersistentIdentityTracker()
    return _persistent_identity_tracker

# Convenience functions
def update_wallet_identity(address: str, token: str, prediction_success: bool, score_impact: float = 1.0):
    """Update wallet identity score"""
    tracker = get_persistent_identity_tracker()
    tracker.update_wallet_identity(address, token, prediction_success, score_impact)

def get_identity_boost(addresses: List[str]) -> float:
    """Get identity boost for addresses"""
    tracker = get_persistent_identity_tracker()
    return tracker.get_identity_boost(addresses)

def get_wallet_identity_stats(address: str) -> Optional[Dict]:
    """Get wallet identity stats"""
    tracker = get_persistent_identity_tracker()
    return tracker.get_wallet_identity_stats(address)

def get_top_identity_wallets(limit: int = 20) -> List[Dict]:
    """Get top identity wallets"""
    tracker = get_persistent_identity_tracker()
    return tracker.get_top_identity_wallets(limit)

def get_identity_statistics() -> Dict:
    """Get identity statistics"""
    tracker = get_persistent_identity_tracker()
    return tracker.get_identity_statistics()