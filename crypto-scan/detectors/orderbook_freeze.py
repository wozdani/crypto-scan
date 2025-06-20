"""
Orderbook Freeze Detector - Detektor Zamrożonego Ask-side
Wykrywa sytuacje gdy ask-side w orderbooku zamiera podczas wzrostu ceny
Identyfikuje ukrytą akceptację trendu przez brak oporu
"""

import requests
import json
import time
from datetime import datetime, timezone

def get_orderbook_snapshot_bybit(symbol: str) -> dict:
    """
    Pobiera snapshot orderbooka z Bybit API
    
    Args:
        symbol: Symbol trading pair (np. 'BTCUSDT')
        
    Returns:
        dict: snapshot orderbooka z ceną, bids i asks
    """
    url = "https://api.bybit.com/v5/market/orderbook"
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": 5  # top 5 poziomów
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
        
        # Oblicz mid price jako przybliżoną cenę
        mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0
        
        return {
            "price": mid_price,
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000)
        }
    except Exception as e:
        print(f"❌ Exception fetching orderbook for {symbol}: {e}")
        return {}


def collect_orderbook_snapshots(symbol: str, count: int = 3, interval: int = 300) -> list[dict]:
    """
    Zbiera serię snapshotów orderbooka
    
    Args:
        symbol: Symbol trading pair
        count: Liczba snapshotów do zebrania
        interval: Odstęp między snapshotami w sekundach (default 5 min)
        
    Returns:
        list: lista snapshotów orderbooka
    """
    snapshots = []
    
    for i in range(count):
        snapshot = get_orderbook_snapshot_bybit(symbol)
        if snapshot:
            snapshots.append(snapshot)
            print(f"📊 Collected orderbook snapshot {i+1}/{count} for {symbol}")
            
            # Wait before next snapshot (except for last one)
            if i < count - 1:
                time.sleep(min(interval, 30))  # Max 30s dla testów
        else:
            print(f"⚠️ Failed to collect snapshot {i+1} for {symbol}")
            break
    
    return snapshots


def detect_orderbook_freeze(orderbook_snapshots: list[dict]) -> tuple[bool, str, dict]:
    """
    Wykrywa zamrożony ask-side w orderbooku przy rosnącej cenie.
    
    Args:
        orderbook_snapshots: lista dictów, każdy ma "price", "bids", "asks"
        
    Returns:
        tuple: (bool, str, dict) - (wykryto_freeze, opis, szczegóły)
    """
    if len(orderbook_snapshots) < 3:
        return False, "Za mało snapshotów orderbooka do analizy freeze", {}

    ask_movement_detected = False
    ask_movements = []
    price_changes = []
    
    # Sprawdź ruch ask-side między snapshotami
    for i in range(1, len(orderbook_snapshots)):
        prev_snapshot = orderbook_snapshots[i-1]
        curr_snapshot = orderbook_snapshots[i]
        
        prev_asks = prev_snapshot["asks"]
        curr_asks = curr_snapshot["asks"]
        
        if not prev_asks or not curr_asks:
            continue
        
        # Porównaj top 3 poziomy ask
        snapshot_movement = False
        level_movements = []
        
        for level in range(min(3, len(prev_asks), len(curr_asks))):
            prev_price = prev_asks[level][0]
            curr_price = curr_asks[level][0]
            movement = abs(curr_price - prev_price) / prev_price if prev_price > 0 else 0
            
            level_movements.append({
                "level": level,
                "prev_price": prev_price,
                "curr_price": curr_price,
                "movement_pct": movement * 100
            })
            
            # Jeśli ruch >1% na dowolnym poziomie, ask się porusza
            if movement > 0.01:
                snapshot_movement = True
                ask_movement_detected = True
        
        ask_movements.append({
            "snapshot_pair": f"{i-1}-{i}",
            "movement_detected": snapshot_movement,
            "level_details": level_movements
        })
        
        # Oblicz zmianę ceny między snapshotami
        prev_price = prev_snapshot["price"]
        curr_price = curr_snapshot["price"]
        price_change = (curr_price - prev_price) / prev_price if prev_price > 0 else 0
        price_changes.append(price_change)
    
    # Sprawdź całkowity wzrost ceny
    price_start = orderbook_snapshots[0]["price"]
    price_end = orderbook_snapshots[-1]["price"]
    total_price_delta = (price_end - price_start) / price_start if price_start > 0 else 0
    
    # Warunki freeze: cena wzrosła >0.5% ale ask nie poruszał się
    price_threshold = 0.005  # 0.5%
    freeze_detected = (total_price_delta > price_threshold and not ask_movement_detected)
    
    details = {
        "snapshots_count": len(orderbook_snapshots),
        "total_price_change_pct": round(total_price_delta * 100, 3),
        "ask_movement_detected": ask_movement_detected,
        "ask_movements": ask_movements,
        "price_changes": [round(pc * 100, 3) for pc in price_changes],
        "price_start": price_start,
        "price_end": price_end,
        "freeze_threshold_pct": price_threshold * 100,
        "analysis_duration_minutes": len(orderbook_snapshots) * 5  # assuming 5min intervals
    }
    
    if freeze_detected:
        if total_price_delta > 0.02:  # >2% wzrost
            description = f"Silny orderbook freeze - cena +{round(total_price_delta*100, 2)}% bez ruchu ask"
        elif total_price_delta > 0.01:  # >1% wzrost
            description = f"Umiarkowany orderbook freeze - cena +{round(total_price_delta*100, 2)}% bez ruchu ask"
        else:
            description = f"Słaby orderbook freeze - cena +{round(total_price_delta*100, 2)}% bez ruchu ask"
    else:
        if ask_movement_detected:
            description = "Brak freeze - ask-side aktywny podczas wzrostu"
        else:
            description = f"Brak freeze - niewystarczający wzrost ceny ({round(total_price_delta*100, 2)}%)"
    
    return freeze_detected, description, details


def calculate_orderbook_freeze_score(detection_result: tuple) -> int:
    """
    Oblicza punktację dla Orderbook Freeze Detection
    
    Args:
        detection_result: wynik z detect_orderbook_freeze()
        
    Returns:
        int: punkty do dodania (0-15)
    """
    detected, description, details = detection_result
    
    if not detected:
        return 0
    
    total_price_change = details.get("total_price_change_pct", 0)
    
    # Bazowy score za wykrycie freeze
    base_score = 15
    
    # Dodatkowy bonus za silniejszy wzrost ceny
    if total_price_change > 3.0:  # >3%
        base_score += 0  # już maksymalny
    elif total_price_change > 2.0:  # >2%
        base_score = 15
    elif total_price_change > 1.0:  # >1%
        base_score = 12
    else:  # 0.5-1%
        base_score = 10
    
    return min(base_score, 15)  # Maksymalnie 15 punktów


def analyze_orderbook_freeze_detailed(orderbook_snapshots: list[dict]) -> dict:
    """
    Szczegółowa analiza orderbook freeze
    
    Args:
        orderbook_snapshots: lista snapshotów orderbooka
        
    Returns:
        dict: szczegółowe dane o freeze
    """
    if len(orderbook_snapshots) < 3:
        return {"error": "Za mało snapshotów orderbooka"}
    
    # Pobierz wyniki podstawowej detekcji
    detected, description, basic_details = detect_orderbook_freeze(orderbook_snapshots)
    
    # Dodatkowe analizy
    bid_stability = []
    ask_stability = []
    spread_changes = []
    
    for i in range(1, len(orderbook_snapshots)):
        prev = orderbook_snapshots[i-1]
        curr = orderbook_snapshots[i]
        
        if prev["bids"] and curr["bids"] and prev["asks"] and curr["asks"]:
            # Analiza stabilności bid
            bid_change = abs(curr["bids"][0][0] - prev["bids"][0][0]) / prev["bids"][0][0]
            bid_stability.append(bid_change)
            
            # Analiza stabilności ask
            ask_change = abs(curr["asks"][0][0] - prev["asks"][0][0]) / prev["asks"][0][0]
            ask_stability.append(ask_change)
            
            # Analiza spread
            prev_spread = prev["asks"][0][0] - prev["bids"][0][0]
            curr_spread = curr["asks"][0][0] - curr["bids"][0][0]
            spread_change = (curr_spread - prev_spread) / prev_spread if prev_spread > 0 else 0
            spread_changes.append(spread_change)
    
    return {
        **basic_details,
        "description": description,
        "detected": detected,
        "bid_stability": [round(bs * 100, 3) for bs in bid_stability],
        "ask_stability": [round(ask_s * 100, 3) for ask_s in ask_stability],
        "spread_changes": [round(sc * 100, 3) for sc in spread_changes],
        "avg_bid_stability": round(sum(bid_stability) / len(bid_stability) * 100, 3) if bid_stability else 0,
        "avg_ask_stability": round(sum(ask_stability) / len(ask_stability) * 100, 3) if ask_stability else 0,
        "avg_spread_change": round(sum(spread_changes) / len(spread_changes) * 100, 3) if spread_changes else 0
    }


def create_mock_orderbook_snapshots() -> list[dict]:
    """
    Tworzy przykładowe snapshoty orderbooka dla testów
    """
    snapshots = []
    base_price = 0.023
    
    # Snapshot 1: start
    snapshots.append({
        "price": base_price,
        "bids": [[0.0229, 120], [0.0228, 80], [0.0227, 45]],
        "asks": [[0.0231, 100], [0.0232, 75], [0.0233, 60]],
        "timestamp": int(time.time() * 1000)
    })
    
    # Snapshot 2: cena wzrosła, ask nie poruszył się
    snapshots.append({
        "price": base_price * 1.007,  # +0.7%
        "bids": [[0.02297, 115], [0.02287, 85], [0.02277, 50]],  # bids podążają
        "asks": [[0.0231, 100], [0.0232, 75], [0.0233, 60]],     # asks zamrożone
        "timestamp": int(time.time() * 1000) + 300000
    })
    
    # Snapshot 3: dalsza wzrost, ask nadal nieruchomy
    snapshots.append({
        "price": base_price * 1.012,  # +1.2% total
        "bids": [[0.02304, 110], [0.02294, 90], [0.02284, 55]],  # bids dalej podążają
        "asks": [[0.0231, 100], [0.0232, 75], [0.0233, 60]],     # asks nadal zamrożone
        "timestamp": int(time.time() * 1000) + 600000
    })
    
    return snapshots


def test_orderbook_freeze_with_mock_data():
    """
    Test funkcji z przykładowymi danymi
    """
    print("🧪 Testing Orderbook Freeze Detector with mock data\n")
    
    # Test 1: Perfect freeze scenario
    freeze_snapshots = create_mock_orderbook_snapshots()
    detected1, desc1, details1 = detect_orderbook_freeze(freeze_snapshots)
    score1 = calculate_orderbook_freeze_score((detected1, desc1, details1))
    
    print(f"Perfect freeze scenario: {detected1}")
    print(f"Description: {desc1}")
    print(f"Price change: {details1['total_price_change_pct']:.3f}%")
    print(f"Ask movement: {details1['ask_movement_detected']}")
    print(f"Score: {score1}/15")
    
    # Test 2: Active ask scenario (no freeze)
    active_snapshots = create_mock_orderbook_snapshots()
    # Modify to show ask movement
    active_snapshots[1]["asks"] = [[0.02315, 95], [0.02325, 70], [0.02335, 55]]  # asks moved
    active_snapshots[2]["asks"] = [[0.02320, 90], [0.02330, 65], [0.02340, 50]]  # asks moved again
    
    detected2, desc2, details2 = detect_orderbook_freeze(active_snapshots)
    score2 = calculate_orderbook_freeze_score((detected2, desc2, details2))
    
    print(f"\nActive ask scenario: {detected2}")
    print(f"Description: {desc2}")
    print(f"Price change: {details2['total_price_change_pct']:.3f}%")
    print(f"Ask movement: {details2['ask_movement_detected']}")
    print(f"Score: {score2}/15")
    
    print("\n✅ Orderbook Freeze Detector tests completed!")


if __name__ == "__main__":
    test_orderbook_freeze_with_mock_data()