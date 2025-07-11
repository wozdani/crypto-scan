"""
Heatmap Vacuum Detector - Detektor Próżni Likwidacyjnej
Wykrywa zanik zleceń ask na poziomach 1-2% powyżej ceny (vacuum zone)
Identyfikuje sytuacje gdy smart money "czyści drogę" pod przyszły ruch w górę
"""

import requests
import json
import time
from datetime import datetime, timezone

def get_orderbook_heatmap_bybit(symbol: str, depth: int = 200) -> dict:
    """
    Pobiera głęboki orderbook z Bybit API dla analizy heatmap
    
    Args:
        symbol: Symbol trading pair (np. 'BTCUSDT')
        depth: Głębokość orderbooka (default 25 poziomów)
        
    Returns:
        dict: orderbook z ceną, bids i asks
    """
    url = "https://api.bybit.com/v5/market/orderbook"
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": depth
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data["retCode"] != 0:
            print(f"❌ Bybit API error for {symbol}: {data['retMsg']}")
            return {}

        result = data["result"]
        bids = [[float(item[0]), float(item[1])] for item in result["b"]]
        asks = [[float(item[0]), float(item[1])] for item in result["a"]]
        
        # Oblicz mid price
        mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0
        
        return {
            "price": mid_price,
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000)
        }
    except Exception as e:
        print(f"❌ Exception fetching heatmap orderbook for {symbol}: {e}")
        return {}


def collect_heatmap_snapshots(symbol: str, count: int = 3, interval: int = 300) -> list[dict]:
    """
    Zbiera serię snapshotów orderbooka dla analizy heatmap vacuum
    
    Args:
        symbol: Symbol trading pair
        count: Liczba snapshotów do zebrania (default 3)
        interval: Odstęp między snapshotami w sekundach (default 5 min)
        
    Returns:
        list: lista snapshotów orderbooka z głęboką heatmap
    """
    snapshots = []
    
    for i in range(count):
        snapshot = get_orderbook_heatmap_bybit(symbol, depth=200)
        if snapshot:
            snapshots.append(snapshot)
            print(f"🗺️ Collected heatmap snapshot {i+1}/{count} for {symbol}")
            
            # Wait before next snapshot (except for last one)
            if i < count - 1:
                time.sleep(min(interval, 30))  # Max 30s dla testów
        else:
            print(f"⚠️ Failed to collect heatmap snapshot {i+1} for {symbol}")
            break
    
    return snapshots


def detect_heatmap_vacuum(orderbook_snapshots: list[dict]) -> tuple[bool, str, dict]:
    """
    Wykrywa zanik zleceń ask 1–2% powyżej ceny (tworzenie vacuum zone).
    
    Args:
        orderbook_snapshots: lista dictów z orderbook data
        
    Returns:
        tuple: (bool, str, dict) - (wykryto_vacuum, opis, szczegóły)
    """
    if len(orderbook_snapshots) < 3:
        return False, "Za mało snapshotów dla analizy heatmap vacuum", {}

    disappearance_ratio_threshold = 0.4  # 40% zanik wolumenu
    initial_price = orderbook_snapshots[0]["price"]
    
    if initial_price <= 0:
        return False, "Nieprawidłowa cena dla analizy vacuum", {}

    vacuum_counts = 0
    total_levels = 0
    vacuum_details = []
    
    # Analizuj poziomy ask 1-2% powyżej ceny w pierwszym snapshot
    initial_snapshot = orderbook_snapshots[0]
    final_snapshot = orderbook_snapshots[-1]
    
    if not initial_snapshot.get("asks") or not final_snapshot.get("asks"):
        return False, "Brak danych ask w snapshotach", {}
    
    # Definiuj vacuum zone: 1.01% - 1.02% powyżej początkowej ceny
    vacuum_zone_min = initial_price * 1.01
    vacuum_zone_max = initial_price * 1.02
    final_price = orderbook_snapshots[-1]["price"]
    
    for level in initial_snapshot["asks"]:
        ask_price = level[0]
        initial_volume = level[1]
        
        # Sprawdź czy poziom jest w vacuum zone
        if vacuum_zone_min <= ask_price <= vacuum_zone_max:
            total_levels += 1
            
            # Znajdź odpowiadający poziom w końcowym snapshot
            final_volume = 0
            for final_level in final_snapshot["asks"]:
                if abs(final_level[0] - ask_price) / ask_price < 0.001:  # 0.1% tolerance
                    final_volume = final_level[1]
                    break
            
            # Oblicz zanik wolumenu
            volume_reduction = (initial_volume - final_volume) / initial_volume if initial_volume > 0 else 0
            
            vacuum_details.append({
                "ask_price": ask_price,
                "initial_volume": initial_volume,
                "final_volume": final_volume,
                "volume_reduction_pct": round(volume_reduction * 100, 2),
                "vacuum_detected": volume_reduction >= disappearance_ratio_threshold
            })
            
            # Sprawdź czy wystąpił zanik ≥40%
            if volume_reduction >= disappearance_ratio_threshold:
                vacuum_counts += 1
    
    # Warunki detekcji vacuum:
    # 1. Co najmniej 3 poziomy w vacuum zone
    # 2. Co najmniej 2 poziomy z zanikiem ≥40%
    vacuum_detected = (total_levels >= 3 and vacuum_counts >= 2)
    
    # Oblicz dodatkowe metryki
    avg_reduction = 0
    if vacuum_details:
        avg_reduction = sum(detail["volume_reduction_pct"] for detail in vacuum_details) / len(vacuum_details)
    
    vacuum_intensity = vacuum_counts / total_levels if total_levels > 0 else 0
    
    details = {
        "vacuum_zone_min_price": vacuum_zone_min,
        "vacuum_zone_max_price": vacuum_zone_max,
        "total_levels_analyzed": total_levels,
        "vacuum_levels_count": vacuum_counts,
        "vacuum_intensity": round(vacuum_intensity, 3),
        "avg_volume_reduction_pct": round(avg_reduction, 2),
        "disappearance_threshold_pct": disappearance_ratio_threshold * 100,
        "vacuum_details": vacuum_details,
        "snapshots_analyzed": len(orderbook_snapshots),
        "price_at_analysis": initial_price
    }
    
    if vacuum_detected:
        if vacuum_intensity >= 0.8:  # ≥80% poziomów z vacuum
            description = f"Silna próżnia likwidacyjna - {vacuum_counts}/{total_levels} poziomów wyczyszczonych"
        elif vacuum_intensity >= 0.6:  # ≥60% poziomów z vacuum
            description = f"Umiarkowana próżnia likwidacyjna - {vacuum_counts}/{total_levels} poziomów wyczyszczonych"
        else:
            description = f"Słaba próżnia likwidacyjna - {vacuum_counts}/{total_levels} poziomów wyczyszczonych"
    else:
        if total_levels < 3:
            description = f"Brak vacuum - za mało poziomów w strefie ({total_levels}/3)"
        else:
            description = f"Brak vacuum - niewystarczający zanik ({vacuum_counts}/{total_levels} poziomów)"
    
    return vacuum_detected, description, details


def calculate_heatmap_vacuum_score(detection_result: tuple) -> int:
    """
    Oblicza punktację dla Heatmap Vacuum Detection
    
    Args:
        detection_result: wynik z detect_heatmap_vacuum()
        
    Returns:
        int: punkty do dodania (0-10)
    """
    detected, description, details = detection_result
    
    if not detected:
        return 0
    
    vacuum_intensity = details.get("vacuum_intensity", 0)
    avg_reduction = details.get("avg_volume_reduction_pct", 0)
    
    # Bazowy score za wykrycie vacuum
    base_score = 10
    
    # Dodatkowy bonus za intensywność (można zwiększyć do 12-15 w przyszłości)
    if vacuum_intensity >= 0.8 and avg_reduction >= 60:  # Very strong vacuum
        base_score = 10  # Keep at max for now
    elif vacuum_intensity >= 0.6 and avg_reduction >= 50:  # Strong vacuum
        base_score = 10
    elif vacuum_intensity >= 0.4 and avg_reduction >= 40:  # Moderate vacuum
        base_score = 8
    else:  # Weak vacuum
        base_score = 6
    
    return min(base_score, 10)  # Maksymalnie 10 punktów zgodnie ze specyfikacją


def analyze_heatmap_vacuum_detailed(orderbook_snapshots: list[dict]) -> dict:
    """
    Szczegółowa analiza heatmap vacuum z dodatkowymi metrykami
    
    Args:
        orderbook_snapshots: lista snapshotów orderbooka
        
    Returns:
        dict: szczegółowe dane o vacuum zone
    """
    if len(orderbook_snapshots) < 3:
        return {"error": "Za mało snapshotów dla szczegółowej analizy"}
    
    # Pobierz wyniki podstawowej detekcji
    detected, description, basic_details = detect_heatmap_vacuum(orderbook_snapshots)
    
    # Dodatkowe analizy
    price_progression = [snapshot["price"] for snapshot in orderbook_snapshots]
    price_trend = "rosnący" if price_progression[-1] > price_progression[0] else "malejący"
    price_change_pct = ((price_progression[-1] - price_progression[0]) / price_progression[0]) * 100
    
    # Analiza całkowitego wolumenu ask w vacuum zone
    total_volumes = []
    for snapshot in orderbook_snapshots:
        total_vacuum_volume = 0
        price = snapshot["price"]
        vacuum_min = price * 1.01
        vacuum_max = price * 1.02
        
        for ask_level in snapshot["asks"]:
            if vacuum_min <= ask_level[0] <= vacuum_max:
                total_vacuum_volume += ask_level[1]
        
        total_volumes.append(total_vacuum_volume)
    
    total_volume_change = 0
    if total_volumes[0] > 0:
        total_volume_change = ((total_volumes[-1] - total_volumes[0]) / total_volumes[0]) * 100
    
    return {
        **basic_details,
        "description": description,
        "detected": detected,
        "price_progression": [round(p, 6) for p in price_progression],
        "price_trend": price_trend,
        "price_change_pct": round(price_change_pct, 3),
        "total_vacuum_volumes": [round(v, 2) for v in total_volumes],
        "total_volume_change_pct": round(total_volume_change, 2),
        "vacuum_strength": "silna" if basic_details.get("vacuum_intensity", 0) >= 0.8 else 
                         "umiarkowana" if basic_details.get("vacuum_intensity", 0) >= 0.6 else "słaba"
    }


def create_mock_heatmap_snapshots() -> list[dict]:
    """
    Tworzy przykładowe snapshoty orderbooka z heatmap vacuum dla testów
    """
    snapshots = []
    base_price = 50000.0
    
    # Snapshot 1: pełna heatmap powyżej ceny
    snapshots.append({
        "price": base_price,
        "bids": [[49950, 0.5], [49900, 0.3], [49850, 0.2]],
        "asks": [
            [50100, 0.4],  # +0.2%
            [50250, 0.8],  # +0.5% - vacuum zone start
            [50500, 1.2],  # +1.0% - vacuum zone
            [50600, 0.9],  # +1.2% - vacuum zone  
            [50750, 0.7],  # +1.5% - vacuum zone
            [50900, 0.6],  # +1.8% - vacuum zone
            [51000, 1.1],  # +2.0% - vacuum zone end
            [51200, 0.5]   # +2.4%
        ],
        "timestamp": 1000000
    })
    
    # Snapshot 2: częściowy zanik w vacuum zone
    snapshots.append({
        "price": base_price * 1.003,  # cena wzrosła o 0.3%
        "bids": [[49965, 0.5], [49915, 0.3], [49865, 0.2]],
        "asks": [
            [50115, 0.4],  # +0.2%
            [50265, 0.8],  # +0.5% - vacuum zone start  
            [50515, 0.7],  # +1.0% - vacuum zone (lekki zanik)
            [50615, 0.5],  # +1.2% - vacuum zone (większy zanik)
            [50765, 0.3],  # +1.5% - vacuum zone (duży zanik) 
            [50915, 0.2],  # +1.8% - vacuum zone (duży zanik)
            [51015, 1.1],  # +2.0% - vacuum zone end (bez zmian)
            [51215, 0.5]   # +2.4%
        ],
        "timestamp": 1000300
    })
    
    # Snapshot 3: znaczący vacuum w strefie +1-2%
    snapshots.append({
        "price": base_price * 1.005,  # cena wzrosła o 0.5%
        "bids": [[49975, 0.5], [49925, 0.3], [49875, 0.2]],
        "asks": [
            [50125, 0.4],  # +0.2%
            [50275, 0.8],  # +0.5% - vacuum zone start
            [50525, 0.3],  # +1.0% - vacuum zone (60% zanik: 1.2→0.3)
            [50625, 0.2],  # +1.2% - vacuum zone (78% zanik: 0.9→0.2)
            [50775, 0.1],  # +1.5% - vacuum zone (86% zanik: 0.7→0.1)
            [50925, 0.1],  # +1.8% - vacuum zone (83% zanik: 0.6→0.1)
            [51025, 1.1],  # +2.0% - vacuum zone end (bez zmian)
            [51225, 0.5]   # +2.4%
        ],
        "timestamp": 1000600
    })
    
    return snapshots


def test_heatmap_vacuum_with_mock_data():
    """
    Test funkcji z przykładowymi danymi
    """
    print("🧪 Testing Heatmap Vacuum Detector with mock data\n")
    
    # Test 1: Perfect vacuum scenario
    vacuum_snapshots = create_mock_heatmap_snapshots()
    detected1, desc1, details1 = detect_heatmap_vacuum(vacuum_snapshots)
    score1 = calculate_heatmap_vacuum_score((detected1, desc1, details1))
    
    print(f"🗺️ Perfect Vacuum Test:")
    print(f"   Detected: {detected1}")
    print(f"   Description: {desc1}")
    print(f"   Vacuum levels: {details1['vacuum_levels_count']}/{details1['total_levels_analyzed']}")
    print(f"   Vacuum intensity: {details1['vacuum_intensity']:.3f}")
    print(f"   Avg volume reduction: {details1['avg_volume_reduction_pct']:.1f}%")
    print(f"   Score: {score1}/10")
    
    # Test 2: No vacuum scenario (strong asks remain)
    no_vacuum_snapshots = create_mock_heatmap_snapshots()
    # Modify to show no significant ask disappearance
    for i in range(1, len(no_vacuum_snapshots)):
        # Keep ask volumes high (no vacuum)
        no_vacuum_snapshots[i]["asks"] = [
            [50125, 0.4],
            [50275, 0.8],
            [50525, 1.1],  # No reduction
            [50625, 0.8],  # No reduction  
            [50775, 0.6],  # No reduction
            [50925, 0.5],  # No reduction
            [51025, 1.1],
            [51225, 0.5]
        ]
    
    detected2, desc2, details2 = detect_heatmap_vacuum(no_vacuum_snapshots)
    score2 = calculate_heatmap_vacuum_score((detected2, desc2, details2))
    
    print(f"\n📊 No Vacuum Test:")
    print(f"   Detected: {detected2}")
    print(f"   Description: {desc2}")
    print(f"   Vacuum levels: {details2['vacuum_levels_count']}/{details2['total_levels_analyzed']}")
    print(f"   Vacuum intensity: {details2['vacuum_intensity']:.3f}")
    print(f"   Avg volume reduction: {details2['avg_volume_reduction_pct']:.1f}%")
    print(f"   Score: {score2}/10")
    
    print("\n✅ Heatmap Vacuum Detector tests completed!")


if __name__ == "__main__":
    test_heatmap_vacuum_with_mock_data()