"""
🎯 Stage 13: Token Trust Score System
System zaufania do tokena na podstawie przeszłych trafień portfeli

Mechanizm:
- Tracking adresów portfeli w sygnałach whale_ping i dex_inflow
- Automatyczne zwiększanie trust score gdy te same adresy powracają
- Dynamiczny boost scoring bazowany na historii trafień
- Redukcja fałszywych alertów przez preferencje dla sprawdzonych portfeli

File: stealth_engine/utils/trust_tracker.py
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
# Global constants
TRUST_PATH = "cache/token_trust_scores.json"
MAX_TRUST_BOOST = 0.25  # Maksymalny boost dla trust score
BASE_TRUST_BOOST = 0.1  # Bazowy boost dla rozpoznanych portfeli

class TokenTrustTracker:
    """
    🎯 Token Trust Score Manager
    
    Tracks address performance across tokens and provides trust-based scoring boosts
    """
    
    def __init__(self):
        self.trust_data = {}
        self.load_trust_scores()
    
    def load_trust_scores(self) -> Dict:
        """
        📂 Załaduj trust scores z cache
        """
        if os.path.exists(TRUST_PATH):
            try:
                with open(TRUST_PATH, 'r') as f:
                    self.trust_data = json.load(f)
                    print(f"[TRUST TRACKER] Loaded trust data for {len(self.trust_data)} tokens")
            except Exception as e:
                print(f"[TRUST TRACKER ERROR] Failed to load trust scores: {e}")
                self.trust_data = {}
        else:
            self.trust_data = {}
            print(f"[TRUST TRACKER] Initialized empty trust data")
        
        return self.trust_data
    
    def save_trust_scores(self):
        """
        💾 Zapisz trust scores do cache
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(TRUST_PATH), exist_ok=True)
            
            with open(TRUST_PATH, 'w') as f:
                json.dump(self.trust_data, f, indent=2)
                
            print(f"[TRUST TRACKER] Saved trust data for {len(self.trust_data)} tokens")
            
        except Exception as e:
            print(f"[TRUST TRACKER ERROR] Failed to save trust scores: {e}")
    
    def update_token_trust(self, token: str, addresses: List[str], signal_type: str = "general"):
        """
        📈 Aktualizuj trust score dla tokena z wykrytymi adresami
        
        Args:
            token: Symbol tokena
            addresses: Lista wykrytych adresów portfeli
            signal_type: Typ sygnału (whale_ping, dex_inflow, general)
        """
        if not addresses:
            return
        
# Initialize token data if not exists
        if token not in self.trust_data:
            self.trust_data[token] = {
                "addresses": {},
                "signal_history": {},
                "last_updated": "",
                "total_signals": 0
            }
        
        token_data = self.trust_data[token]
        
        # Update address counts
        for addr in addresses:
            if addr not in token_data["addresses"]:
                token_data["addresses"][addr] = 0
            token_data["addresses"][addr] += 1
        
        # Update signal history
        if signal_type not in token_data["signal_history"]:
            token_data["signal_history"][signal_type] = 0
        token_data["signal_history"][signal_type] += 1
        
        # Update metadata
        token_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        token_data["total_signals"] += 1
        
        self.save_trust_scores()
        
        print(f"[TRUST UPDATE] {token}: Updated trust for {len(addresses)} addresses "
              f"(signal: {signal_type}, total: {token_data['total_signals']})")
    
    def compute_trust_boost(self, token: str, addresses: List[str]) -> float:
        """
        🔢 Oblicz trust boost dla tokena na podstawie wykrytych adresów
        
        Args:
            token: Symbol tokena
            addresses: Lista aktualnie wykrytych adresów
            
        Returns:
            float: Trust boost (0.0 - 0.25)
        """
        if not addresses or token not in self.trust_data:
            return 0.0
        
        token_data = self.trust_data[token]
        known_addresses = token_data["addresses"]
        
        # Count recognized addresses
        recognized_count = 0
        total_history_score = 0
        
        for addr in addresses:
            if addr in known_addresses:
                history_count = known_addresses[addr]
                # Only count as "recognized" if seen multiple times (trust requires repetition)
                if history_count > 1:
                    recognized_count += 1
                    # Weight by historical frequency
                    history_score = min(history_count, 5)  # Cap at 5 occurrences
                    total_history_score += history_score
        
        if recognized_count == 0:
            return 0.0
        
        # Calculate ratio of recognized addresses
        recognition_ratio = recognized_count / len(addresses)
        
        # Calculate average historical score
        avg_history_score = total_history_score / recognized_count
        
        # Compute trust boost based on truly repeated addresses
        # Base boost for recognized addresses + bonus for high recognition ratio + bonus for historical frequency
        trust_boost = (
            BASE_TRUST_BOOST * recognition_ratio +  # 0.0 - 0.1
            0.1 * recognition_ratio +  # 0.0 - 0.1 for high recognition
            0.05 * min(avg_history_score / 5.0, 1.0)  # 0.0 - 0.05 for frequency
        )
        
        # Cap at maximum
        trust_boost = min(trust_boost, MAX_TRUST_BOOST)
        
        print(f"[TRUST BOOST] {token}: {recognized_count}/{len(addresses)} addresses recognized "
              f"(ratio: {recognition_ratio:.2f}, avg_history: {avg_history_score:.1f}) → boost: {trust_boost:.3f}")
        
        return trust_boost
    
    def get_token_trust_stats(self, token: str) -> Dict:
        """
        📊 Pobierz statystyki trust dla tokena
        """
        if token not in self.trust_data:
            return {
                "total_addresses": 0,
                "total_signals": 0,
                "last_updated": None,
                "top_addresses": []
            }
        
        token_data = self.trust_data[token]
        
        # Sort addresses by frequency
        sorted_addresses = sorted(
            token_data["addresses"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "total_addresses": len(token_data["addresses"]),
            "total_signals": token_data.get("total_signals", 0),
            "last_updated": token_data.get("last_updated"),
            "top_addresses": sorted_addresses[:5],  # Top 5 most frequent
            "signal_history": token_data.get("signal_history", {})
        }
    
    def get_trust_statistics(self, address: str = None) -> Dict:
        """
        📊 Pobierz statystyki trust - z cache dla pojedynczych adresów
        
        Args:
            address: Opcjonalny adres do sprawdzenia (z cache)
        """
        if address:
            # Quick cache lookup for single address
            cache_key = address[:10] + "..."
            
            # Search in cached data
            for token, data in self.trust_data.items():
                if address in data["addresses"]:
                    return {
                        "trust": min(data["addresses"][address] / 10.0, 1.0),  # Normalize to 0-1
                        "predictions": data["addresses"][address],
                        "token": token,
                        "cached": True
                    }
            
            # Address not found in cache
            return {
                "trust": 0.0,
                "predictions": 0,
                "token": None,
                "cached": True
            }
        
        # Full statistics calculation (original behavior)
        total_tokens = len(self.trust_data)
        total_addresses = sum(len(data["addresses"]) for data in self.trust_data.values())
        total_signals = sum(data.get("total_signals", 0) for data in self.trust_data.values())
        
        # Find most trusted tokens
        token_trust_scores = []
        for token, data in self.trust_data.items():
            avg_address_frequency = (
                sum(data["addresses"].values()) / len(data["addresses"]) 
                if data["addresses"] else 0
            )
            token_trust_scores.append((token, avg_address_frequency, len(data["addresses"])))
        
        # Sort by average frequency
        token_trust_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_tokens": total_tokens,
            "total_unique_addresses": total_addresses,
            "total_signals_tracked": total_signals,
            "most_trusted_tokens": token_trust_scores[:10],
            "avg_addresses_per_token": total_addresses / total_tokens if total_tokens > 0 else 0
        }
    
    def cleanup_old_data(self, max_age_days: int = 30):
        """
        🧹 Oczyść stare dane trust (opcjonalnie)
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        
        cleaned_tokens = []
        
        for token, data in list(self.trust_data.items()):
            last_updated = data.get("last_updated")
            if last_updated:
                try:
                    update_date = datetime.datetime.fromisoformat(last_updated)
                    if update_date < cutoff_date:
                        del self.trust_data[token]
                        cleaned_tokens.append(token)
                except ValueError:
                    # Invalid date format, remove
                    del self.trust_data[token]
                    cleaned_tokens.append(token)
        
        if cleaned_tokens:
            self.save_trust_scores()
            print(f"[TRUST CLEANUP] Removed {len(cleaned_tokens)} old token trust records")

# Global instance
_trust_tracker = None

def get_trust_tracker() -> TokenTrustTracker:
    """Pobierz globalną instancję Trust Tracker"""
    global _trust_tracker
    if _trust_tracker is None:
        _trust_tracker = TokenTrustTracker()
    return _trust_tracker

# Convenience functions
def load_trust_scores() -> Dict:
    """📂 Convenience function: Załaduj trust scores z cache"""
    tracker = get_trust_tracker()
    return tracker.trust_data

def save_trust_scores(data: Dict):
    """💾 Convenience function: Zapisz trust scores do cache (legacy compatibility)"""
    tracker = get_trust_tracker()
    tracker.trust_data = data
    tracker.save_trust_scores()

def update_token_trust(token: str, addresses: List[str], signal_type: str = "general"):
    """📈 Convenience function: Aktualizuj trust score dla tokena"""
    tracker = get_trust_tracker()
    tracker.update_token_trust(token, addresses, signal_type)

def compute_trust_boost(token: str, addresses: List[str]) -> float:
    """🔢 Convenience function: Oblicz trust boost dla tokena"""
    tracker = get_trust_tracker()
    return tracker.compute_trust_boost(token, addresses)

def get_token_trust_stats(token: str) -> Dict:
    """📊 Convenience function: Pobierz statystyki trust dla tokena"""
    tracker = get_trust_tracker()
    return tracker.get_token_trust_stats(token)

def get_trust_statistics(address: str = None) -> Dict:
    """📊 Convenience function: Pobierz statystyki trust z cache dla pojedynczych adresów"""
    tracker = get_trust_tracker()
    return tracker.get_trust_statistics(address)

def cleanup_trust_data(max_age_days: int = 30):
    """🧹 Convenience function: Oczyść stare dane trust"""
    tracker = get_trust_tracker()
    tracker.cleanup_old_data(max_age_days)