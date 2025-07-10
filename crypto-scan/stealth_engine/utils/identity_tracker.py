"""
🎯 Stage 14: Persistent Identity Scoring System
Identyfikacja powtarzalnych sygnatur aktywności portfeli

Mechanizm:
- Tracking portfeli w whale_ping i dex_inflow signals
- Permanentna reputacja na podstawie skuteczności predykcji
- Identity boost dla portfeli z wysoką skutecznością historyczną
- Feedback loop - portfele które trafnie "kupowały przed ruchem" zyskują trwałe zaufanie

File: stealth_engine/utils/identity_tracker.py
"""

import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class PersistentIdentityTracker:
    """
    🎯 Persistent Identity Scoring Manager
    
    Tracks wallet identity scores based on successful pre-pump predictions
    """
    
    def __init__(self, cache_file: str = "cache/wallet_identity_score.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self.identity_scores = self._load_identity_scores()
    
    def _load_identity_scores(self) -> Dict:
        """
        📂 Załaduj identity scores z cache
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    print(f"[IDENTITY TRACKER] Loaded identity data for {len(data)} wallets")
                    return data
            print("[IDENTITY TRACKER] Initialized empty identity data")
            return {}
        except Exception as e:
            print(f"[IDENTITY TRACKER] Error loading identity scores: {e}")
            return {}
    
    def _save_identity_scores(self):
        """
        💾 Zapisz identity scores do cache
        """
        try:
            with self._lock:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.identity_scores, f, indent=2)
                print(f"[IDENTITY TRACKER] Saved identity data for {len(self.identity_scores)} wallets")
        except Exception as e:
            print(f"[IDENTITY TRACKER] Error saving identity scores: {e}")
    
    def update_wallet_identity(self, wallets: List[str], token: str, success: bool = True):
        """
        📈 Aktualizuj identity score dla portfeli
        
        Args:
            wallets: Lista adresów portfeli
            token: Symbol tokena
            success: Czy predykcja była skuteczna (default: True dla manual update)
        """
        if not wallets:
            return
        
        current_time = datetime.utcnow().isoformat()
        
        with self._lock:
            for wallet in wallets:
                if wallet not in self.identity_scores:
                    self.identity_scores[wallet] = {
                        "score": 0,
                        "last_seen": "",
                        "last_token": "",
                        "total_predictions": 0,
                        "successful_predictions": 0
                    }
                
                # Aktualizuj statystyki
                self.identity_scores[wallet]["total_predictions"] += 1
                if success:
                    self.identity_scores[wallet]["score"] += 1
                    self.identity_scores[wallet]["successful_predictions"] += 1
                
                # Aktualizuj metadane
                self.identity_scores[wallet]["last_seen"] = current_time
                self.identity_scores[wallet]["last_token"] = token
        
        self._save_identity_scores()
        
        success_wallets = len([w for w in wallets if success])
        print(f"[IDENTITY UPDATE] {token}: Updated identity for {len(wallets)} wallets "
              f"(successful: {success_wallets})")
    
    def get_identity_boost(self, wallets: List[str]) -> float:
        """
        🔢 Oblicz identity boost dla portfeli
        
        PERFORMANCE OPTIMIZED: Zmniejszone logi debug dla szybszego przetwarzania
        
        Args:
            wallets: Lista adresów portfeli
            
        Returns:
            float: Identity boost (0.0 - 0.2)
        """
        if not wallets:
            return 0.0
        
        # PERFORMANCE BOOST: Limit do 20 adresów dla szybszego przetwarzania
        if len(wallets) > 20:
            wallets = wallets[:20]
        
        total_score = 0
        recognized_wallets = 0
        
        for wallet in wallets:
            if wallet in self.identity_scores:
                score = self.identity_scores[wallet]["score"]
                total_score += score
                recognized_wallets += 1
        
        if recognized_wallets == 0:
            return 0.0
        
        # Oblicz średni identity score
        avg_score = total_score / recognized_wallets
        
        # Boost formula: min(avg_score * 0.05, 0.2) - max +0.2 boost
        identity_boost = min(avg_score * 0.05, 0.2)
        
        return identity_boost
    
    def get_wallet_identity_stats(self, wallet: str) -> Dict:
        """
        📊 Pobierz statystyki identity dla portfela
        """
        if wallet not in self.identity_scores:
            return {
                "wallet": wallet,
                "score": 0,
                "total_predictions": 0,
                "successful_predictions": 0,
                "success_rate": 0.0,
                "last_seen": None,
                "last_token": None
            }
        
        data = self.identity_scores[wallet]
        success_rate = (data["successful_predictions"] / data["total_predictions"] 
                       if data["total_predictions"] > 0 else 0.0)
        
        return {
            "wallet": wallet,
            "score": data["score"],
            "total_predictions": data["total_predictions"],
            "successful_predictions": data["successful_predictions"],
            "success_rate": success_rate,
            "last_seen": data["last_seen"],
            "last_token": data["last_token"]
        }
    
    def get_top_identity_wallets(self, limit: int = 10) -> List[Dict]:
        """
        🏆 Pobierz top portfele według identity score
        """
        wallets_data = []
        
        for wallet, data in self.identity_scores.items():
            success_rate = (data["successful_predictions"] / data["total_predictions"] 
                           if data["total_predictions"] > 0 else 0.0)
            
            wallets_data.append({
                "wallet": wallet,
                "score": data["score"],
                "total_predictions": data["total_predictions"],
                "successful_predictions": data["successful_predictions"],
                "success_rate": success_rate,
                "last_seen": data["last_seen"],
                "last_token": data["last_token"]
            })
        
        # Sortuj według score, potem success_rate
        wallets_data.sort(key=lambda x: (x["score"], x["success_rate"]), reverse=True)
        
        return wallets_data[:limit]
    
    def get_identity_statistics(self) -> Dict:
        """
        📊 Pobierz kompletne statystyki identity trackera
        """
        if not self.identity_scores:
            return {
                "total_wallets": 0,
                "total_predictions": 0,
                "total_successful": 0,
                "overall_success_rate": 0.0,
                "high_score_wallets": 0,  # score >= 5
                "active_wallets_24h": 0
            }
        
        total_predictions = sum(data["total_predictions"] for data in self.identity_scores.values())
        total_successful = sum(data["successful_predictions"] for data in self.identity_scores.values())
        overall_success_rate = total_successful / total_predictions if total_predictions > 0 else 0.0
        
        high_score_wallets = len([data for data in self.identity_scores.values() if data["score"] >= 5])
        
        # Aktywne portfele w ostatnich 24h
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        active_wallets_24h = 0
        for data in self.identity_scores.values():
            if data["last_seen"]:
                try:
                    last_seen = datetime.fromisoformat(data["last_seen"])
                    if last_seen > cutoff_time:
                        active_wallets_24h += 1
                except:
                    pass
        
        return {
            "total_wallets": len(self.identity_scores),
            "total_predictions": total_predictions,
            "total_successful": total_successful,
            "overall_success_rate": overall_success_rate,
            "high_score_wallets": high_score_wallets,
            "active_wallets_24h": active_wallets_24h
        }
    
    def cleanup_old_data(self, max_age_days: int = 30):
        """
        🧹 Oczyść stare dane identity (opcjonalnie)
        """
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        removed_count = 0
        
        with self._lock:
            wallets_to_remove = []
            for wallet, data in self.identity_scores.items():
                if data["last_seen"]:
                    try:
                        last_seen = datetime.fromisoformat(data["last_seen"])
                        if last_seen < cutoff_time and data["score"] < 2:  # Keep high-score wallets
                            wallets_to_remove.append(wallet)
                    except:
                        pass
            
            for wallet in wallets_to_remove:
                del self.identity_scores[wallet]
                removed_count += 1
        
        if removed_count > 0:
            self._save_identity_scores()
            print(f"[IDENTITY CLEANUP] Removed {removed_count} old wallet entries")


# Global instance
_identity_tracker = None


def get_identity_tracker() -> PersistentIdentityTracker:
    """Pobierz globalną instancję Identity Tracker"""
    global _identity_tracker
    if _identity_tracker is None:
        _identity_tracker = PersistentIdentityTracker()
    return _identity_tracker


# Convenience functions for easy integration
def load_identity_scores() -> Dict:
    """📂 Convenience function: Załaduj identity scores z cache"""
    return get_identity_tracker().identity_scores


def save_identity_scores(data: Dict):
    """💾 Convenience function: Zapisz identity scores do cache (legacy compatibility)"""
    tracker = get_identity_tracker()
    tracker.identity_scores = data
    tracker._save_identity_scores()


def update_wallet_identity(wallets: List[str], token: str, success: bool = True):
    """📈 Convenience function: Aktualizuj identity score dla portfeli"""
    get_identity_tracker().update_wallet_identity(wallets, token, success)


def get_identity_boost(wallets: List[str]) -> float:
    """🔢 Convenience function: Oblicz identity boost dla portfeli"""
    print(f"[DEBUG IDENTITY] Global function called with {len(wallets) if wallets else 0} wallets")
    try:
        result = get_identity_tracker().get_identity_boost(wallets)
        print(f"[DEBUG IDENTITY] Global function returning {result:.3f}")
        return result
    except Exception as e:
        print(f"[DEBUG IDENTITY] ERROR in global function: {e}")
        return 0.0


def get_wallet_identity_stats(wallet: str) -> Dict:
    """📊 Convenience function: Pobierz statystyki identity dla portfela"""
    return get_identity_tracker().get_wallet_identity_stats(wallet)


def get_top_identity_wallets(limit: int = 10) -> List[Dict]:
    """🏆 Convenience function: Pobierz top portfele według identity score"""
    return get_identity_tracker().get_top_identity_wallets(limit)


def get_identity_statistics() -> Dict:
    """📊 Convenience function: Pobierz kompletne statystyki identity"""
    return get_identity_tracker().get_identity_statistics()


def cleanup_identity_data(max_age_days: int = 30):
    """🧹 Convenience function: Oczyść stare dane identity"""
    get_identity_tracker().cleanup_old_data(max_age_days)