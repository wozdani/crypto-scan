from utils.data_fetchers import get_all_data
from utils.token_price import get_token_price_usd
from utils.contracts import get_or_fetch_token_contract, normalize_token_name
from utils.whale_detector import detect_whale_transfers
from utils.token_map_loader import load_token_map
from utils.orderbook_anomaly import detect_orderbook_anomaly
from utils.heatmap_exhaustion import detect_heatmap_exhaustion, analyze_orderbook_exhaustion
from utils.orderbook_spoofing import detect_orderbook_spoofing, analyze_orderbook_walls
from utils.vwap_pinning import detect_vwap_pinning, analyze_vwap_control, get_recent_market_data
from utils.volume_cluster_slope import detect_volume_cluster_slope, get_recent_candle_data, analyze_volume_price_dynamics
from stages.stage_minus2_2 import detect_stage_minus2_2
from stages.stage_1g import detect_stage_1g
import numpy as np
import json
import os
import requests

def detect_volume_spike(symbol):
    """
    Wykrywa nagly skok wolumenu: Z-score > 2.5 lub 3x Średnia z poprzednich 4 świec.
    """
    data = get_all_data(symbol)
    if not data or not data["prev_candle"]:
        return False, 0.0

    try:
        current_volume = float(data["volume"])
        prev_volume = float(data["prev_candle"][5])

        # Dla testow uzywamy tylko jednej wczesniejszej swiecy (w przyszlosci 4)
        volumes = [prev_volume, current_volume]
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        z_score = (current_volume -
                   mean_volume) / std_volume if std_volume > 0 else 0
        spike_detected = z_score > 2.5 or current_volume > 3 * prev_volume

        return spike_detected, current_volume
    except Exception as e:
        print(f"âťŚ laÄ…d w volume spike dla {symbol}: {e}")
        return False, 0.0

def get_dex_inflow(symbol):
    try:
        from utils.contracts import get_or_fetch_token_contract
        
        # Pobierz kontrakt tokena (z mapy lub CoinGecko)
        token_info = get_or_fetch_token_contract(symbol)
        
        if not token_info:
            return 0.0

        chain = token_info["chain"].lower()
        
        # Sprawdź czy chain jest obsługiwany
        if not is_chain_supported(chain):
            print(f"⚠️ Token {symbol} ({chain}) pominięty w DEX inflow – brak klucza API")
            return 0.0

        address = token_info["address"]

        # Dobor odpowiedniego API i klucza
        if chain == "ethereum":
            base_url = "https://api.etherscan.io/api"
            api_key = os.getenv("ETHERSCAN_API_KEY")
        elif chain == "bsc":
            base_url = "https://api.bscscan.com/api"
            api_key = os.getenv("BSCSCAN_API_KEY")
        elif chain == "arbitrum":
            base_url = "https://api.arbiscan.io/api"
            api_key = os.getenv("ARBISCAN_API_KEY")
        elif chain == "polygon":
            base_url = "https://api.polygonscan.com/api"
            api_key = os.getenv("POLYGONSCAN_API_KEY")
        elif chain == "optimism":
            base_url = "https://api-optimistic.etherscan.io/api"
            api_key = os.getenv("OPTIMISMSCAN_API_KEY")
        else:
            print(f"âš ď¸Ź Chain {chain} nieobsĹ‚ugiwany jeszcze")
            return 0.0

        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",
            "apikey": api_key,
        }

        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
        if data["status"] != "1":
            print(f"âš ď¸Ź Brakk wynikow inflow dla {symbol}: {data.get('message')}")
            return 0.0

        # Liczymy sume wplywow do kontraktu
        inflow_sum = 0.0
        for tx in data["result"][:10]:
            if tx["to"].lower() == address.lower():
                inflow_sum += int(tx["value"]) / (10 ** 18)

        return inflow_sum
    except Exception as e:
        print(f"âťŚ laÄ…d inflow {symbol}: {e}")
        return 0.0

def is_chain_supported(chain: str) -> bool:
    """Sprawdza czy chain jest obsługiwany (ma dostępny klucz API)"""
    chain = chain.lower()
    if chain == "ethereum" and os.getenv("ETHERSCAN_API_KEY"):
        return True
    if chain == "bsc" and os.getenv("BSCSCAN_API_KEY"):
        return True
    if chain == "arbitrum" and os.getenv("ARBISCAN_API_KEY"):
        return True
    if chain == "polygon" and os.getenv("POLYGONSCAN_API_KEY"):
        return True
    if chain == "optimism" and os.getenv("OPTIMISMSCAN_API_KEY"):
        return True
    return False

def detect_whale_tx(symbol):
    from utils.contracts import get_or_fetch_token_contract
    
    # Pobierz kontrakt tokena (z mapy lub CoinGecko)
    token_info = get_or_fetch_token_contract(symbol)
    
    if not token_info:
        return False
    
    chain = token_info["chain"].lower()
    
    # Sprawdź czy chain jest obsługiwany
    if not is_chain_supported(chain):
        print(f"⚠️ Token {symbol} ({chain}) pominięty – brak klucza API")
        return False

    address = token_info["address"]
    chain = token_info["chain"].lower()

    # Dobór API key i endpointów
    if chain == "ethereum":
        base_url = "https://api.etherscan.io/api"
        api_key = os.getenv("ETHERSCAN_API_KEY")
    elif chain == "bsc":
        base_url = "https://api.bscscan.com/api"
        api_key = os.getenv("BSCSCAN_API_KEY")
    elif chain == "arbitrum":
        base_url = "https://api.arbiscan.io/api"
        api_key = os.getenv("ARBISCAN_API_KEY")
    else:
        print(f"⚠️ Chain {chain} nieobsługiwany")
        return False

    # Zapytanie o transakcje
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "desc",
        "apikey": api_key,
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        if data["status"] != "1":
            return False

        price_usd = get_token_price_usd(symbol)
        if price_usd is None:
            return False

        for tx in data["result"][:10]:
            if tx["to"].lower() == address.lower():
                token_amount = int(tx["value"]) / (10 ** 18)
                usd_value = token_amount * price_usd
                if usd_value > 50000:
                    return True

        return False

    except Exception as e:
        print(f"❌ Błąd whale TX {symbol}: {e}")
        return False

def detect_stage_minus2_1(symbol):
    """
    Glowna funkcja detekcji Stage â€“2.1 (mikroanomalii).
    Zwraca:
        - stage2_pass (czy przechodzi dalej)
        - signals (slownik aktywnych mikroanomalii)
        - dex_inflow_volume (float)
        - stage1g_active (czy Stage 1G jest aktywny)
    """
    # Get market data for analysis
    data = get_all_data(symbol)
    
    signals = {
        "whale_activity": False,
        "orderbook_anomaly": False,
        "volume_spike": False,
        "dex_inflow": False,
        "heatmap_exhaustion": False,
        "spoofing_suspected": False,
        "vwap_pinned": False,
        "volume_slope_up": False,
        "event_tag": None,
        "event_score": 0,
        "event_risk": False,
        "stage1g_active": False,
        "stage1g_trigger_type": None,
    }

    # Realna logika volume spike
    volume_spike_detected, volume = detect_volume_spike(symbol)
    if volume_spike_detected:
        signals["volume_spike"] = True

    # Whale transaction detection
    token_map = load_token_map()
    whale_detected, whale_usd = detect_whale_transfers(symbol, token_map)
    signals["whale_activity"] = whale_detected
    if whale_detected:
        print(f"âś… Stage â€“2.1: Whale TX dla {symbol} = ${whale_usd:.2f}")

    # Heatmap exhaustion detection
    exhaustion_data = analyze_orderbook_exhaustion(symbol)
    signals["heatmap_exhaustion"] = detect_heatmap_exhaustion({
        "ask_wall_disappeared": exhaustion_data.get("ask_wall_disappeared", False),
        "volume_spike": signals["volume_spike"],
        "whale_activity": signals["whale_activity"]
    })
    if signals["heatmap_exhaustion"]:
        print(f"âś… Stage â€“2.1: Heatmap exhaustion dla {symbol}")

    # Orderbook spoofing detection
    wall_data = analyze_orderbook_walls(symbol)
    signals["spoofing_suspected"] = detect_orderbook_spoofing({
        "ask_wall_appeared": wall_data.get("ask_wall_appeared", False),
        "ask_wall_disappeared": wall_data.get("ask_wall_disappeared", False),
        "ask_wall_lifetime_sec": wall_data.get("ask_wall_lifetime_sec", 0),
        "whale_activity": signals["whale_activity"],
        "volume_spike": signals["volume_spike"]
    })
    if signals["spoofing_suspected"]:
        print(f"âś… Stage â€“2.1: Orderbook spoofing dla {symbol}")

    # VWAP pinning detection
    vwap_market_data = get_recent_market_data(symbol)
    signals["vwap_pinned"] = detect_vwap_pinning(vwap_market_data)
    if signals["vwap_pinned"]:
        print(f"âś… Stage â€“2.1: VWAP pinning dla {symbol}")

    # Volume cluster slope detection
    candle_data = get_recent_candle_data(symbol)
    signals["volume_slope_up"] = detect_volume_cluster_slope(candle_data)
    if signals["volume_slope_up"]:
        print(f"âś… Stage â€“2.1: Volume cluster slope dla {symbol}")

    # Orderbook anomaly detection
    signals["orderbook_anomaly"] = detect_orderbook_anomaly(symbol)

    # Stage -2.2: Event tag detection
    tag, tag_score, risk_flag = detect_stage_minus2_2(symbol)
    signals["event_tag"] = tag
    signals["event_score"] = tag_score
    signals["event_risk"] = risk_flag

    # DEX inflow detection
    inflow = get_dex_inflow(symbol)
    if inflow > 0.3:  # â¬…ď¸Ź prĂłg 0.3 ETH / BNB / etc.
        signals["dex_inflow"] = True
    else:
        inflow = 0.0



    # Stage 1G detection with news tag integration
    stage1g_active, trigger_type = detect_stage_1g(symbol, data, signals["event_tag"])
    signals["stage1g_active"] = stage1g_active
    signals["stage1g_trigger_type"] = trigger_type
    
    # Finalna decyzja: aktywowany jesli co najmniej 1 aktywny sygnal boolean
    boolean_signals = [
        signals["whale_activity"], 
        signals["orderbook_anomaly"],
        signals["volume_spike"],
        signals["dex_inflow"]
    ]
    stage2_pass = any(boolean_signals) or signals["event_tag"] is not None

    return stage2_pass, signals, inflow, stage1g_active
