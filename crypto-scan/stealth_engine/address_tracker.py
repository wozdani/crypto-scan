"""
Address Tracker v2 for Stealth Engine - Cross-Token Analysis
Tracks repeated addresses across multiple tokens to detect coordinated accumulation patterns
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class AddressTracker:
    """
    Tracks wallet addresses appearing in stealth signals across multiple tokens
    Detects repeated accumulation patterns and cross-token coordination
    """
    
    def __init__(self, history_file: str = "cache/address_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.address_history = self._load_history()
        self._migrate_to_new_format()
    
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
    
    def _migrate_to_new_format(self):
        """Migrate old token-centric format to new address-centric format"""
        if not self.address_history:
            return
            
        # Check if already in new format
        sample_key = next(iter(self.address_history.keys()))
        if sample_key in self.address_history:
            sample_value = self.address_history[sample_key]
            if isinstance(sample_value, list) and sample_value and "token" in sample_value[0]:
                print(f"[ADDRESS TRACKER] Already in new format: {len(self.address_history)} addresses")
                return  # Already in new format
        
        # Convert old format {token: {address: [activities]}} to new format {address: [activities with token]}
        new_format = {}
        migrated_count = 0
        
        for token, addresses in self.address_history.items():
            if isinstance(addresses, dict):
                for address, activities in addresses.items():
                    if address not in new_format:
                        new_format[address] = []
                    
                    # Add token field to each activity
                    for activity in activities:
                        if isinstance(activity, dict):
                            if "token" not in activity:
                                activity["token"] = token
                            new_format[address].append(activity)
                            migrated_count += 1
        
        if new_format:
            self.address_history = new_format
            self._save_history()
            print(f"[ADDRESS TRACKER] Migrated to new format: {len(new_format)} addresses, {migrated_count} activities")

    def record_address_activity(self, token: str, address: str, usd_value: float, 
                               source: str, timestamp: str = None):
        """
        Record address activity for a token (NEW FORMAT: address-centric)
        
        Args:
            token: Token symbol (e.g., "BTCUSDT")
            address: Wallet address
            usd_value: USD value of transaction/order
            source: Source of activity ("dex_inflow" or "whale_ping")
            timestamp: ISO timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Initialize address if not exists (NEW FORMAT)
        if address not in self.address_history:
            self.address_history[address] = []
        
        # Add new activity record with token field
        activity = {
            "token": token,
            "timestamp": timestamp,
            "usd_value": usd_value,
            "source": source
        }
        
        self.address_history[address].append(activity)
        
        print(f"[ADDRESS TRACKER] Recorded {source} activity: {token} | {address[:8]}... | ${usd_value:,.0f}")
        
        # Save updated history
        self._save_history()
    
    def get_repeated_addresses_boost(self, token: str, current_addresses: List[str], 
                                   history_days: int = 7) -> Tuple[float, Dict]:
        """
        Calculate boost score based on repeated address appearances on SAME token
        
        Args:
            token: Token symbol
            current_addresses: List of addresses active in current scan
            history_days: Days to look back in history
            
        Returns:
            Tuple of (boost_score, details_dict)
        """
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=history_days)
        cutoff_iso = cutoff_date.isoformat()
        
        repeated_addresses = []
        total_boost = 0.0
        
        for address in current_addresses:
            if address in self.address_history:
                # Filter activities for this token within time window
                token_activities = [
                    activity for activity in self.address_history[address]
                    if activity.get("token") == token and activity["timestamp"] >= cutoff_iso
                ]
                
                if len(token_activities) > 1:  # Address appeared more than once on this token
                    repeated_addresses.append({
                        "address": address,
                        "appearances": len(token_activities),
                        "total_value": sum(a["usd_value"] for a in token_activities),
                        "sources": list(set(a["source"] for a in token_activities))
                    })
                    
                    # Boost calculation: +0.2 per repeated address, max +0.6
                    address_boost = min(0.2 * (len(token_activities) - 1), 0.2)
                    total_boost += address_boost
        
        # Cap total boost at 0.6
        total_boost = min(total_boost, 0.6)
        
        if total_boost > 0:
            print(f"[ADDRESS TRACKER] Repeated address boost for {token}: +{total_boost:.2f} ({len(repeated_addresses)} addresses)")
        
        return total_boost, {
            "repeated_addresses": len(repeated_addresses),
            "details": repeated_addresses,
            "boost_score": total_boost
        }
    
    def get_cross_token_activity_boost(self, current_token: str, current_addresses: List[str], 
                                     history_days: int = 7, window_hours: int = 48) -> Tuple[float, Dict]:
        """
        Calculate boost score based on cross-token address activity
        
        Args:
            current_token: Token currently being analyzed
            current_addresses: List of addresses active in current scan
            history_days: Days to look back in history
            window_hours: Hours window for cross-token correlation
            
        Returns:
            Tuple of (boost_score, details_dict)
        """
        # Calculate cutoff dates
        cutoff_date = datetime.now() - timedelta(days=history_days)
        window_cutoff = datetime.now() - timedelta(hours=window_hours)
        cutoff_iso = cutoff_date.isoformat()
        window_iso = window_cutoff.isoformat()
        
        cross_token_details = []
        total_boost = 0.0
        
        for address in current_addresses:
            if address in self.address_history:
                # Get all activities for this address within history window
                all_activities = [
                    activity for activity in self.address_history[address]
                    if activity["timestamp"] >= cutoff_iso
                ]
                
                # Find activities on other tokens within correlation window
                other_tokens = set()
                for activity in all_activities:
                    if (activity.get("token") != current_token and 
                        activity["timestamp"] >= window_iso):
                        other_tokens.add(activity.get("token"))
                
                if len(other_tokens) >= 1:  # Address active on other tokens
                    # Calculate unique token-address pairs
                    token_pairs = len(other_tokens) + 1  # +1 for current token
                    
                    cross_token_details.append({
                        "address": address,
                        "tokens_involved": list(other_tokens) + [current_token],
                        "token_pairs": token_pairs,
                        "total_value": sum(a["usd_value"] for a in all_activities),
                        "window_activities": len([a for a in all_activities if a["timestamp"] >= window_iso])
                    })
                    
                    # Boost calculation: +0.2 per unique token-address pair, max +0.6
                    pair_boost = min(0.2 * (token_pairs - 1), 0.6)
                    total_boost += pair_boost
        
        # Cap total boost at 0.6
        total_boost = min(total_boost, 0.6)
        
        if total_boost > 0:
            print(f"[ADDRESS TRACKER] Cross-token activity boost for {current_token}: +{total_boost:.2f} ({len(cross_token_details)} addresses)")
        
        return total_boost, {
            "cross_token_addresses": len(cross_token_details),
            "details": cross_token_details,
            "boost_score": total_boost,
            "window_hours": window_hours
        }
    
    def _cleanup_old_records(self, days_to_keep: int = 7):
        """
        Clean up old address records
        
        Args:
            days_to_keep: Number of days to keep in history
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_iso = cutoff_date.isoformat()
        
        cleaned_count = 0
        
        for address in list(self.address_history.keys()):
            # Filter out old activities
            old_activities = self.address_history[address]
            new_activities = [
                activity for activity in old_activities
                if activity["timestamp"] >= cutoff_iso
            ]
            
            cleaned_count += len(old_activities) - len(new_activities)
            
            if new_activities:
                self.address_history[address] = new_activities
            else:
                # Remove address if no recent activities
                del self.address_history[address]
        
        if cleaned_count > 0:
            print(f"[ADDRESS TRACKER] Cleaned {cleaned_count} old records (>{days_to_keep} days)")
            self._save_history()
    
    def get_velocity_analysis(self, current_token: str, current_addresses: List[str], 
                             window_minutes: int = 60) -> Tuple[float, Dict]:
        """
        🆕 PHASE 3/5: Time-Based Velocity Tracking
        
        Analizuje prędkość akumulacji adresów w czasie - im szybsza sekwencja
        aktywności tego samego adresu, tym wyższy boost velocity
        
        Args:
            current_token: Aktualny token
            current_addresses: Lista aktualnych adresów
            window_minutes: Okno czasowe do analizy velocity (domyślnie 60 minut)
            
        Returns:
            Tuple[velocity_boost_score, velocity_details]
        """
        try:
            if not current_addresses:
                return 0.0, {"velocity_addresses": 0, "details": []}
                
            velocity_details = []
            total_velocity_boost = 0.0
            
            current_time = datetime.now()
            window_start = current_time - timedelta(minutes=window_minutes)
            
            for address in current_addresses:
                if address not in self.address_history:
                    continue
                    
                # Pobierz wszystkie aktywności tego adresu w oknie czasowym
                activities = self.address_history[address]
                recent_activities = []
                
                for activity in activities:
                    activity_time = datetime.fromisoformat(activity["timestamp"])
                    if activity_time >= window_start:
                        recent_activities.append(activity)
                
                # Oblicz velocity boost na podstawie częstotliwości aktywności
                if len(recent_activities) >= 2:
                    # Sortuj chronologicznie
                    recent_activities.sort(key=lambda x: x["timestamp"])
                    
                    # Oblicz średni odstęp między aktywnościami
                    time_gaps = []
                    for i in range(1, len(recent_activities)):
                        prev_time = datetime.fromisoformat(recent_activities[i-1]["timestamp"])
                        curr_time = datetime.fromisoformat(recent_activities[i]["timestamp"])
                        gap_minutes = (curr_time - prev_time).total_seconds() / 60
                        time_gaps.append(gap_minutes)
                    
                    avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 60
                    
                    # Velocity boost: im mniejszy gap, tym wyższy boost
                    # Formuła: max(0, 0.3 - (avg_gap / 60) * 0.2)
                    velocity_boost = max(0, 0.3 - (avg_gap / 60) * 0.2)
                    
                    # Bonus za wysoką wartość transakcji
                    total_value = sum(act["usd_value"] for act in recent_activities)
                    if total_value > 100000:  # $100k+
                        velocity_boost *= 1.5
                    elif total_value > 50000:  # $50k+
                        velocity_boost *= 1.2
                    
                    total_velocity_boost += velocity_boost
                    
                    velocity_details.append({
                        "address": address,
                        "activities_count": len(recent_activities),
                        "avg_gap_minutes": round(avg_gap, 1),
                        "total_value": total_value,
                        "velocity_boost": round(velocity_boost, 3),
                        "tokens_involved": list(set(act["token"] for act in recent_activities))
                    })
            
            # Maksymalny boost na poziomie 0.8
            final_boost = min(total_velocity_boost, 0.8)
            
            print(f"[ADDRESS TRACKER] Velocity analysis for {current_token}: +{final_boost:.2f} ({len(velocity_details)} addresses)")
            
            return final_boost, {
                "velocity_addresses": len(velocity_details),
                "details": velocity_details,
                "window_minutes": window_minutes,
                "total_boost": round(final_boost, 3)
            }
            
        except Exception as e:
            print(f"[ADDRESS TRACKER] Velocity analysis error: {e}")
            return 0.0, {"velocity_addresses": 0, "details": [], "error": str(e)}

    def get_address_statistics(self, token: str = None, days: int = 7) -> Dict:
        """
        Get statistics about address activities
        
        Args:
            token: Specific token (None for all tokens)
            days: Days to analyze
            
        Returns:
            Dictionary with statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff_date.isoformat()
        
        stats = {
            "total_addresses": len(self.address_history),
            "tokens_analyzed": set(),
            "activities_count": 0,
            "total_usd_value": 0.0,
            "sources": {"dex_inflow": 0, "whale_ping": 0}
        }
        
        for address, activities in self.address_history.items():
            for activity in activities:
                if activity["timestamp"] >= cutoff_iso:
                    if token is None or activity.get("token") == token:
                        stats["activities_count"] += 1
                        stats["total_usd_value"] += activity.get("usd_value", 0)
                        stats["tokens_analyzed"].add(activity.get("token"))
                        
                        source = activity.get("source", "unknown")
                        if source in stats["sources"]:
                            stats["sources"][source] += 1
        
        stats["tokens_analyzed"] = list(stats["tokens_analyzed"])
        stats["unique_addresses"] = len([
            addr for addr, activities in self.address_history.items()
            if any(a["timestamp"] >= cutoff_iso for a in activities)
        ])
        
        return stats