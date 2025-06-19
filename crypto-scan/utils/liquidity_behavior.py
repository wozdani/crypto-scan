"""
Liquidity Behavior Detector - Stage -2.1
Wykrywa nietypowe zachowania płynności w orderbooku sygnalizujące cichą akumulację
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import time

class LiquidityBehaviorAnalyzer:
    """Analizator zachowań płynności w orderbooku"""
    
    def __init__(self):
        self.data_dir = "data/orderbooks"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def store_orderbook_snapshot(self, symbol: str, orderbook_data: Dict) -> bool:
        """
        Przechowuje snapshot orderbooku dla symbolu
        
        Args:
            symbol: Symbol trading (np. 'BTCUSDT')
            orderbook_data: Dane orderbooku z API Bybit
            
        Returns:
            bool: True jeśli snapshot został zapisany pomyślnie
        """
        try:
            filepath = os.path.join(self.data_dir, f"{symbol}.json")
            
            # Przygotuj nowy snapshot
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "bids": orderbook_data.get("bids", [])[:10],  # Top 10 bidów
                "asks": orderbook_data.get("asks", [])[:10],  # Top 10 asków
                "price": float(orderbook_data.get("bids", [["0"]])[0][0]) if orderbook_data.get("bids") else 0
            }
            
            # Wczytaj istniejące snapshoty
            snapshots = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    snapshots = data.get("snapshots", [])
            
            # Dodaj nowy snapshot
            snapshots.append(snapshot)
            
            # Zachowaj tylko ostatnie 3 snapshoty (15 minut historii)
            snapshots = snapshots[-3:]
            
            # Zapisz do pliku
            with open(filepath, 'w') as f:
                json.dump({
                    "symbol": symbol,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "snapshots": snapshots
                }, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"❌ Błąd podczas zapisywania snapshot orderbooku dla {symbol}: {e}")
            return False
    
    def get_snapshots(self, symbol: str) -> List[Dict]:
        """
        Pobiera snapshoty orderbooku dla symbolu
        
        Args:
            symbol: Symbol trading
            
        Returns:
            List[Dict]: Lista snapshotów orderbooku
        """
        try:
            filepath = os.path.join(self.data_dir, f"{symbol}.json")
            
            if not os.path.exists(filepath):
                return []
                
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get("snapshots", [])
                
        except Exception as e:
            print(f"❌ Błąd podczas wczytywania snapshotów dla {symbol}: {e}")
            return []

def detect_layered_bids(snapshots: List[Dict], price_tolerance: float = 0.005) -> Tuple[bool, Dict]:
    """
    Wykrywa warstwowanie bidów - ≥3 bidy w bliskim zasięgu cenowym
    
    Args:
        snapshots: Lista snapshotów orderbooku
        price_tolerance: Tolerancja cenowa (0.5% domyślnie)
        
    Returns:
        Tuple[bool, Dict]: (czy_wykryto, szczegóły)
    """
    if len(snapshots) < 2:
        return False, {}
        
    try:
        latest_snapshot = snapshots[-1]
        bids = latest_snapshot.get("bids", [])
        
        if len(bids) < 3:
            return False, {}
            
        # Sprawdź czy top 3 bidy są w bliskim zasięgu
        top_bids = bids[:3]
        highest_bid = float(top_bids[0][0])
        
        layered_count = 0
        total_volume = 0
        
        for bid_price, bid_volume in top_bids:
            bid_price = float(bid_price)
            bid_volume = float(bid_volume)
            
            # Sprawdź czy bid jest w tolerancji cenowej
            if abs(bid_price - highest_bid) / highest_bid <= price_tolerance:
                layered_count += 1
                total_volume += bid_volume
        
        detected = layered_count >= 3
        
        details = {
            "layered_bids_count": layered_count,
            "total_layered_volume": total_volume,
            "highest_bid": highest_bid,
            "price_range": f"±{price_tolerance*100:.1f}%"
        }
        
        return detected, details
        
    except Exception as e:
        print(f"❌ Błąd w detect_layered_bids: {e}")
        return False, {}

def detect_pinned_orders(snapshots: List[Dict], stability_threshold: int = 3) -> Tuple[bool, Dict]:
    """
    Wykrywa utrzymujące się poziomy - te same poziomy przez ≥3 snapshoty
    
    Args:
        snapshots: Lista snapshotów orderbooku
        stability_threshold: Minimalna liczba snapshotów z tym samym poziomem
        
    Returns:
        Tuple[bool, Dict]: (czy_wykryto, szczegóły)
    """
    if len(snapshots) < stability_threshold:
        return False, {}
        
    try:
        # Zbierz poziomy bidów z każdego snapshotu
        bid_levels = []
        for snapshot in snapshots:
            bids = snapshot.get("bids", [])
            if bids:
                # Weź top 3 poziomy bidów
                levels = [float(bid[0]) for bid in bids[:3]]
                bid_levels.append(levels)
        
        if len(bid_levels) < stability_threshold:
            return False, {}
            
        # Znajdź poziomy, które powtarzają się we wszystkich snapshotach
        stable_levels = []
        tolerance = 0.001  # 0.1% tolerancja
        
        for level in bid_levels[0]:  # Poziomy z pierwszego snapshotu
            stable_count = 1
            
            for other_levels in bid_levels[1:]:
                # Sprawdź czy poziom występuje w innych snapshotach
                for other_level in other_levels:
                    if abs(level - other_level) / level <= tolerance:
                        stable_count += 1
                        break
            
            if stable_count >= stability_threshold:
                stable_levels.append(level)
        
        detected = len(stable_levels) > 0
        
        details = {
            "stable_levels_count": len(stable_levels),
            "stable_levels": stable_levels,
            "snapshots_analyzed": len(snapshots),
            "stability_threshold": stability_threshold
        }
        
        return detected, details
        
    except Exception as e:
        print(f"❌ Błąd w detect_pinned_orders: {e}")
        return False, {}

def detect_void_reaction(snapshots: List[Dict], volume_threshold: float = 0.3) -> Tuple[bool, Dict]:
    """
    Wykrywa void reaction - zniknięcie dużego aska bez znaczącej zmiany ceny
    
    Args:
        snapshots: Lista snapshotów orderbooku
        volume_threshold: Próg zmian wolumenu (30% domyślnie)
        
    Returns:
        Tuple[bool, Dict]: (czy_wykryto, szczegóły)
    """
    if len(snapshots) < 2:
        return False, {}
        
    try:
        prev_snapshot = snapshots[-2]
        curr_snapshot = snapshots[-1]
        
        prev_asks = prev_snapshot.get("asks", [])
        curr_asks = curr_snapshot.get("asks", [])
        
        if not prev_asks or not curr_asks:
            return False, {}
            
        # Sprawdź zmianę wolumenu na najlepszym asku
        prev_ask_volume = float(prev_asks[0][1])
        curr_ask_volume = float(curr_asks[0][1])
        
        # Sprawdź zmianę ceny
        prev_price = prev_snapshot.get("price", 0)
        curr_price = curr_snapshot.get("price", 0)
        
        if prev_price == 0 or curr_price == 0:
            return False, {}
            
        volume_change = abs(curr_ask_volume - prev_ask_volume) / prev_ask_volume
        price_change = abs(curr_price - prev_price) / prev_price
        
        # Void reaction: duża zmiana wolumenu, mała zmiana ceny
        detected = volume_change > volume_threshold and price_change < 0.01  # <1% zmiana ceny
        
        details = {
            "volume_change_pct": volume_change * 100,
            "price_change_pct": price_change * 100,
            "prev_ask_volume": prev_ask_volume,
            "curr_ask_volume": curr_ask_volume,
            "price_stability": price_change < 0.01
        }
        
        return detected, details
        
    except Exception as e:
        print(f"❌ Błąd w detect_void_reaction: {e}")
        return False, {}

def detect_fractal_pullback(snapshots: List[Dict]) -> Tuple[bool, Dict]:
    """
    Wykrywa fraktalne cofnięcia - powtarzające się fillowanie bidów na podobnych poziomach
    
    Args:
        snapshots: Lista snapshotów orderbooku
        
    Returns:
        Tuple[bool, Dict]: (czy_wykryto, szczegóły)
    """
    if len(snapshots) < 3:
        return False, {}
        
    try:
        # Analiza zmian w top bidach między snapshotami
        bid_changes = []
        
        for i in range(1, len(snapshots)):
            prev_bids = snapshots[i-1].get("bids", [])
            curr_bids = snapshots[i].get("bids", [])
            
            if prev_bids and curr_bids:
                prev_top_bid = float(prev_bids[0][0])
                curr_top_bid = float(curr_bids[0][0])
                
                change = {
                    "timestamp": snapshots[i].get("timestamp"),
                    "prev_bid": prev_top_bid,
                    "curr_bid": curr_top_bid,
                    "change_pct": abs(curr_top_bid - prev_top_bid) / prev_top_bid
                }
                bid_changes.append(change)
        
        if len(bid_changes) < 2:
            return False, {}
            
        # Szukaj podobnych poziomów cofnięć
        similar_pullbacks = 0
        tolerance = 0.002  # 0.2% tolerancja
        
        for i, change1 in enumerate(bid_changes):
            for change2 in bid_changes[i+1:]:
                # Sprawdź czy poziomy bidów są podobne
                level_similarity = abs(change1["curr_bid"] - change2["curr_bid"]) / change1["curr_bid"]
                
                if level_similarity <= tolerance:
                    similar_pullbacks += 1
        
        detected = similar_pullbacks >= 1  # Przynajmniej jedna para podobnych poziomów
        
        details = {
            "similar_pullbacks_count": similar_pullbacks,
            "total_changes_analyzed": len(bid_changes),
            "tolerance_pct": tolerance * 100,
            "pattern_strength": similar_pullbacks / max(len(bid_changes) - 1, 1)
        }
        
        return detected, details
        
    except Exception as e:
        print(f"❌ Błąd w detect_fractal_pullback: {e}")
        return False, {}

def detect_liquidity_behavior(symbol: str, snapshots: List[Dict] = None) -> Tuple[bool, Dict]:
    """
    Główna funkcja wykrywania zachowań płynności
    
    Args:
        symbol: Symbol trading
        snapshots: Opcjonalna lista snapshotów (jeśli None, pobiera z pliku)
        
    Returns:
        Tuple[bool, Dict]: (czy_wykryto, szczegółowe_dane)
    """
    try:
        # Pobierz snapshoty jeśli nie podano
        if snapshots is None:
            analyzer = LiquidityBehaviorAnalyzer()
            snapshots = analyzer.get_snapshots(symbol)
        
        if len(snapshots) < 2:
            return False, {"error": "Insufficient snapshots", "snapshots_count": len(snapshots)}
        
        # Uruchom wszystkie detektory zachowań
        layered_detected, layered_details = detect_layered_bids(snapshots)
        pinned_detected, pinned_details = detect_pinned_orders(snapshots)
        void_detected, void_details = detect_void_reaction(snapshots)
        fractal_detected, fractal_details = detect_fractal_pullback(snapshots)
        
        # Zlicz aktywne zachowania
        active_behaviors = sum([
            layered_detected,
            pinned_detected, 
            void_detected,
            fractal_detected
        ])
        
        # Liquidity Behavior wykryty jeśli ≥2 z 4 zachowań
        liquidity_behavior_detected = active_behaviors >= 2
        
        # Przygotuj szczegółowe dane
        behavior_details = {
            "liquidity_behavior_detected": liquidity_behavior_detected,
            "active_behaviors_count": active_behaviors,
            "snapshots_analyzed": len(snapshots),
            "detection_timestamp": datetime.now(timezone.utc).isoformat(),
            
            "layered_bids": {
                "detected": layered_detected,
                "details": layered_details
            },
            "pinned_orders": {
                "detected": pinned_detected,
                "details": pinned_details
            },
            "void_reaction": {
                "detected": void_detected,
                "details": void_details
            },
            "fractal_pullback": {
                "detected": fractal_detected,
                "details": fractal_details
            }
        }
        
        return liquidity_behavior_detected, behavior_details
        
    except Exception as e:
        print(f"❌ Błąd w detect_liquidity_behavior dla {symbol}: {e}")
        return False, {"error": str(e)}

# Global instance
liquidity_analyzer = LiquidityBehaviorAnalyzer()