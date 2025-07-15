#!/usr/bin/env python3
"""
🛰️ STAGE 12: Token Clusters - Satelitarny skan stealthowych bliźniaków
🎯 Cel: Wykrywanie i skanowanie tokenów powiązanych z tokenami generującymi silne stealth alerty

📊 Funkcjonalności:
- Ręczne mapowanie tokenów bliźniaczych
- Dynamiczne wykrywanie powiązań (future expansion)
- Satelitarny skan po wykryciu stealth alertu
- Integracja z systemem alertów
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import threading
import time

# 🗺️ RĘCZNE MAPOWANIE TOKENÓW BLIŹNIACZYCH
# Tokeny historycznie skorelowane lub fundamentalnie powiązane
STEALTH_TWINS = {
    # MEME Ecosystem
    "PEPEUSDT": ["DOGEUSDT", "SHIBUSDT", "FLOKIUSDT", "CHEEMSUSDT"],
    "DOGEUSDT": ["PEPEUSDT", "SHIBUSDT", "FLOKIUSDT"],
    "SHIBUSDT": ["DOGEUSDT", "PEPEUSDT", "FLOKIUSDT"],
    "FLOKIUSDT": ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT"],
    
    # DeFi Ecosystem
    "UNIUSDT": ["SUSHIUSDT", "CAKEUSDT", "1INCHUSDT"],
    "SUSHIUSDT": ["UNIUSDT", "CAKEUSDT", "1INCHUSDT"], 
    "CAKEUSDT": ["UNIUSDT", "SUSHIUSDT", "1INCHUSDT"],
    "1INCHUSDT": ["UNIUSDT", "SUSHIUSDT", "CAKEUSDT"],
    
    # Layer 1 Blockchain
    "ETHUSDT": ["BNBUSDT", "ADAUSDT", "SOLUSDT"],
    "BNBUSDT": ["ETHUSDT", "ADAUSDT", "SOLUSDT"],
    "ADAUSDT": ["ETHUSDT", "BNBUSDT", "SOLUSDT"],
    "SOLUSDT": ["ETHUSDT", "BNBUSDT", "ADAUSDT"],
    
    # Gaming Ecosystem
    "AXSUSDT": ["SLPUSDT", "SANDUSDT", "MANAUSDT"],
    "SLPUSDT": ["AXSUSDT", "SANDUSDT", "MANAUSDT"],
    "SANDUSDT": ["AXSUSDT", "SLPUSDT", "MANAUSDT"],
    "MANAUSDT": ["AXSUSDT", "SLPUSDT", "SANDUSDT"],
    
    # Storage/Utility
    "FILUSDT": ["ARUSDT", "STORJUSDT"],
    "ARUSDT": ["FILUSDT", "STORJUSDT"],
    "STORJUSDT": ["FILUSDT", "ARUSDT"],
    
    # Smaller Altcoins
    "RLBUSDT": ["XENUSDT", "HEXUSDT"],
    "XENUSDT": ["RLBUSDT", "HEXUSDT"],
    "HEXUSDT": ["RLBUSDT", "XENUSDT"],
    
    # AI/Meta Tokens
    "TURBOUSDT": ["POPCATUSDT", "WENUSDT"],
    "POPCATUSDT": ["TURBOUSDT", "WENUSDT"],
    "WENUSDT": ["TURBOUSDT", "POPCATUSDT"],
    
    # Oracle/Data
    "LINKUSDT": ["BANDUSDT", "APIUSDT"],
    "BANDUSDT": ["LINKUSDT", "APIUSDT"],
    "APIUSDT": ["LINKUSDT", "BANDUSDT"]
}

@dataclass
class SatelliteResult:
    """Wynik satelitarnego skanu"""
    symbol: str
    triggered_by: str
    stealth_score: float
    scan_timestamp: str
    success: bool
    error_message: Optional[str] = None

class TokenClusterManager:
    """
    🛰️ Manager klastrów tokenów dla satelitarnego skanu
    """
    
    def __init__(self):
        self.cache_file = "cache/satellite_scan_results.json"
        self.results_cache = self._load_results_cache()
        self.lock = threading.Lock()
        
    def _load_results_cache(self) -> Dict:
        """Załaduj cache wyników satelitarnych skanów"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[SATELLITE CACHE] Failed to load cache: {e}")
        
        return {
            "scan_history": [],
            "twin_performance": {},
            "last_cleanup": datetime.now(timezone.utc).isoformat()
        }
    
    def _save_results_cache(self):
        """Zapisz cache wyników"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.results_cache, f, indent=2)
        except Exception as e:
            print(f"[SATELLITE CACHE] Failed to save cache: {e}")
    
    def get_stealth_twins(self, symbol: str) -> List[str]:
        """
        🎯 Pobierz listę tokenów bliźniaczych dla podanego symbolu
        
        Args:
            symbol: Symbol tokenu (np. "PEPEUSDT", "1000000PEPEUSDT")
            
        Returns:
            Lista symboli tokenów bliźniaczych
        """
        # Normalizuj symbol (usuń prefiksy typu 1000000)
        normalized_symbol = symbol.replace("1000000", "").replace("1000", "")
        
        twins = STEALTH_TWINS.get(normalized_symbol, [])
        
        # Filtruj, żeby nie zwracać siebie samego
        filtered_twins = [twin for twin in twins if twin != normalized_symbol]
        
        print(f"[SATELLITE TWINS] {symbol} → {len(filtered_twins)} twins: {filtered_twins}")
        
        return filtered_twins
    
    def record_satellite_scan(self, result: SatelliteResult):
        """
        📝 Zapisz wynik satelitarnego skanu
        """
        with self.lock:
            self.results_cache["scan_history"].append({
                "symbol": result.symbol,
                "triggered_by": result.triggered_by,
                "stealth_score": result.stealth_score,
                "scan_timestamp": result.scan_timestamp,
                "success": result.success,
                "error_message": result.error_message
            })
            
            # Utrzymuj maksymalnie 1000 wyników
            if len(self.results_cache["scan_history"]) > 1000:
                self.results_cache["scan_history"] = self.results_cache["scan_history"][-1000:]
            
            self._save_results_cache()
            
            print(f"[SATELLITE RECORD] {result.symbol} scan recorded (triggered by {result.triggered_by})")
    
    def update_twin_performance(self, original_symbol: str, twin_symbol: str, 
                               performance_score: float):
        """
        📊 Aktualizuj wydajność powiązania twin tokenów
        """
        with self.lock:
            key = f"{original_symbol}->{twin_symbol}"
            
            if key not in self.results_cache["twin_performance"]:
                self.results_cache["twin_performance"][key] = {
                    "total_scans": 0,
                    "successful_scans": 0,
                    "avg_performance": 0.0,
                    "last_update": datetime.now(timezone.utc).isoformat()
                }
            
            perf_data = self.results_cache["twin_performance"][key]
            perf_data["total_scans"] += 1
            
            if performance_score > 0.5:  # Threshold dla "successful"
                perf_data["successful_scans"] += 1
            
            # Aktualizuj średnią wydajność
            old_avg = perf_data["avg_performance"]
            new_avg = (old_avg * (perf_data["total_scans"] - 1) + performance_score) / perf_data["total_scans"]
            perf_data["avg_performance"] = round(new_avg, 3)
            perf_data["last_update"] = datetime.now(timezone.utc).isoformat()
            
            self._save_results_cache()
            
            print(f"[TWIN PERFORMANCE] {key}: {perf_data['successful_scans']}/{perf_data['total_scans']} "
                  f"(avg: {perf_data['avg_performance']:.3f})")
    
    def get_best_twin_pairs(self, limit: int = 10) -> List[Tuple[str, Dict]]:
        """
        🏆 Pobierz najlepsze pary twin tokenów na podstawie wydajności
        """
        twin_data = self.results_cache.get("twin_performance", {})
        
        # Sortuj według success rate i średniej wydajności
        sorted_pairs = sorted(
            twin_data.items(),
            key=lambda x: (
                x[1]["successful_scans"] / max(x[1]["total_scans"], 1),
                x[1]["avg_performance"]
            ),
            reverse=True
        )
        
        return sorted_pairs[:limit]
    
    def get_satellite_statistics(self) -> Dict:
        """
        📊 Pobierz statystyki satelitarnego skanu
        """
        scan_history = self.results_cache.get("scan_history", [])
        twin_performance = self.results_cache.get("twin_performance", {})
        
        # Ostatnie 24h
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_scans = [
            scan for scan in scan_history
            if datetime.fromisoformat(scan["scan_timestamp"]) > recent_cutoff
        ]
        
        successful_recent = sum(1 for scan in recent_scans if scan["success"])
        
        return {
            "total_satellite_scans": len(scan_history),
            "recent_24h_scans": len(recent_scans),
            "recent_24h_success_rate": successful_recent / len(recent_scans) if recent_scans else 0.0,
            "total_twin_pairs": len(twin_performance),
            "top_performing_pairs": self.get_best_twin_pairs(5),
            "cluster_coverage": len(STEALTH_TWINS),
            "last_scan": scan_history[-1]["scan_timestamp"] if scan_history else None
        }
    
    def should_trigger_satellite_scan(self, symbol: str, stealth_score: float) -> bool:
        """
        🚦 Określ czy należy uruchomić satelitarny skan
        
        Args:
            symbol: Symbol tokenu
            stealth_score: Score stealth dla tokenu
            
        Returns:
            True jeśli należy uruchomić satelitarny skan
        """
        # Podstawowy threshold
        if stealth_score < 3.5:
            return False
        
        # Sprawdź czy token ma zdefiniowanych twins
        twins = self.get_stealth_twins(symbol)
        if not twins:
            return False
        
        # Sprawdź czy nie był już skanowany ostatnio (cooldown 1h)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_scans = [
            scan for scan in self.results_cache.get("scan_history", [])
            if (scan["triggered_by"] == symbol and 
                datetime.fromisoformat(scan["scan_timestamp"]) > recent_cutoff)
        ]
        
        if recent_scans:
            print(f"[SATELLITE COOLDOWN] {symbol} satellite scan on cooldown (last: {len(recent_scans)} scans)")
            return False
        
        print(f"[SATELLITE TRIGGER] {symbol} qualifies for satellite scan (score: {stealth_score:.2f})")
        return True
    
    def cleanup_old_data(self, max_age_days: int = 30):
        """
        🧹 Oczyść stare dane satelitarne
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        
        with self.lock:
            original_count = len(self.results_cache["scan_history"])
            
            # Filtruj stare skany
            self.results_cache["scan_history"] = [
                scan for scan in self.results_cache["scan_history"]
                if datetime.fromisoformat(scan["scan_timestamp"]) > cutoff_date
            ]
            
            removed_count = original_count - len(self.results_cache["scan_history"])
            
            # Aktualizuj timestamp cleanup
            self.results_cache["last_cleanup"] = datetime.now(timezone.utc).isoformat()
            
            self._save_results_cache()
            
            print(f"[SATELLITE CLEANUP] Removed {removed_count} old satellite scan records")
    
    def reset_cooldown_for_testing(self, symbol: str):
        """
        🧪 Reset cooldown dla testów - usuń recent scans dla danego symbolu
        """
        with self.lock:
            original_count = len(self.results_cache["scan_history"])
            
            # Usuń skany dla tego symbolu
            self.results_cache["scan_history"] = [
                scan for scan in self.results_cache["scan_history"]
                if scan["triggered_by"] != symbol
            ]
            
            removed_count = original_count - len(self.results_cache["scan_history"])
            self._save_results_cache()
            
            print(f"[SATELLITE TEST RESET] Removed {removed_count} recent scans for {symbol}")
    
    def clear_all_satellite_data(self):
        """
        🧪 Wyczyść wszystkie dane satelitarne (tylko dla testów)
        """
        with self.lock:
            self.results_cache = {
                "scan_history": [],
                "twin_performance": {},
                "last_cleanup": datetime.now(timezone.utc).isoformat()
            }
            self._save_results_cache()
            print(f"[SATELLITE TEST CLEAR] All satellite data cleared for testing")

# Global instance
_token_cluster_manager = None

def get_token_cluster_manager() -> TokenClusterManager:
    """Pobierz globalną instancję Token Cluster Manager"""
    global _token_cluster_manager
    if _token_cluster_manager is None:
        _token_cluster_manager = TokenClusterManager()
    return _token_cluster_manager

def get_stealth_twins(symbol: str) -> List[str]:
    """
    🎯 Convenience function: Pobierz tokeny bliźniacze
    """
    manager = get_token_cluster_manager()
    return manager.get_stealth_twins(symbol)

def should_trigger_satellite_scan(symbol: str, stealth_score: float) -> bool:
    """
    🚦 Convenience function: Sprawdź czy uruchomić satelitarny skan
    """
    manager = get_token_cluster_manager()
    return manager.should_trigger_satellite_scan(symbol, stealth_score)

def record_satellite_result(symbol: str, triggered_by: str, stealth_score: float, 
                           success: bool, error_message: str = None):
    """
    📝 Convenience function: Zapisz wynik satelitarnego skanu
    """
    manager = get_token_cluster_manager()
    result = SatelliteResult(
        symbol=symbol,
        triggered_by=triggered_by,
        stealth_score=stealth_score,
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
        success=success,
        error_message=error_message
    )
    manager.record_satellite_scan(result)

def get_satellite_statistics() -> Dict:
    """
    📊 Convenience function: Pobierz statystyki satelitarne
    """
    manager = get_token_cluster_manager()
    return manager.get_satellite_statistics()

def update_twin_performance(original_symbol: str, twin_symbol: str, performance_score: float):
    """
    📊 Convenience function: Aktualizuj wydajność twin pary
    """
    manager = get_token_cluster_manager()
    manager.update_twin_performance(original_symbol, twin_symbol, performance_score)

def reset_cooldown_for_testing(symbol: str):
    """
    🧪 Convenience function: Reset cooldown dla testów
    """
    manager = get_token_cluster_manager()
    manager.reset_cooldown_for_testing(symbol)

def clear_all_satellite_data():
    """
    🧪 Convenience function: Wyczyść wszystkie dane satelitarne (tylko dla testów)
    """
    manager = get_token_cluster_manager()
    manager.clear_all_satellite_data()