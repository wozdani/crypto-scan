from utils.data_fetchers import get_all_data
from utils.token_price import get_token_price_usd
from utils.whale_detector import detect_whale_transfers
from utils.token_map_loader import load_token_map
from utils.orderbook_anomaly import detect_orderbook_anomaly
from utils.social_detector import detect_social_spike
import numpy as np
import json
import os
import requests

def detect_volume_spike(symbol):
    """
    Wykrywa nagły skok wolumenu: Z-score > 2.5 lub 3x średnia z poprzednich 4 świec.
    """
    data = get_all_data(symbol)
    if not data or not data["prev_candle"]:
        return False, 0.0

    try:
        current_volume = float(data["volume"])
        prev_volume = float(data["prev_candle"][5])

        # Dla testów używamy tylko jednej wcześniejszej świecy (w przyszłości 4)
        volumes = [prev_volume, current_volume]
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        z_score = (current_volume -
                   mean_volume) / std_volume if std_volume > 0 else 0
        spike_detected = z_score > 2.5 or current_volume > 3 * prev_volume

        return spike_detected, current_volume
    except Exception as e:
        print(f"❌ Błąd w volume spike dla {symbol}: {e}")
        return False, 0.0

def get_dex_inflow(symbol):
    try:
        with open("token_contract_map.json", "r") as f:
            token_map = json.load(f)

        token_info = token_map.get(symbol)
        if not token_info:
            print(f"⚠️ Brak kontraktu dla {symbol}")
            return 0.0

        address = token_info["address"]
        chain = token_info["chain"].lower()

        # Dobór odpowiedniego API i klucza
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
            print(f"⚠️ Chain {chain} nieobsługiwany jeszcze")
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
            print(f"⚠️ Brak wyników inflow dla {symbol}: {data.get('message')}")
            return 0.0

        # Liczymy sumę wpływów do kontraktu
        inflow_sum = 0.0
        for tx in data["result"][:10]:
            if tx["to"].lower() == address.lower():
                inflow_sum += int(tx["value"]) / (10 ** 18)

        return inflow_sum
    except Exception as e:
        print(f"❌ Błąd inflow {symbol}: {e}")
        return 0.0

def detect_whale_tx(symbol):
    try:
        with open("token_contract_map.json", "r") as f:
            token_map = json.load(f)

        token_info = token_map.get(symbol)
        if not token_info:
            print(f"⚠️ Brak kontraktu dla {symbol}")
            return False

        address = token_info["address"]
        chain = token_info["chain"].lower()

        # Dobór API key
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
    Główna funkcja detekcji Stage –2.1 (mikroanomalii).
    Zwraca:
        - stage2_pass (czy przechodzi dalej)
        - signals (słownik aktywnych mikroanomalii)
        - dex_inflow_volume (float)
    """
    signals = {
        "social_spike": False,
        "whale_activity": False,
        "orderbook_anomaly": False,
        "volume_spike": False,
        "dex_inflow": False,
    }

    # Social spike detection
    signals["social_spike"] = detect_social_spike(symbol)

    # Realna logika volume spike
    volume_spike_detected, volume = detect_volume_spike(symbol)
    if volume_spike_detected:
        signals["volume_spike"] = True

    # Whale transaction detection
    token_map = load_token_map()
    whale_detected, whale_usd = detect_whale_transfers(symbol, token_map)
    signals["whale_activity"] = whale_detected
    if whale_detected:
        print(f"✅ Stage –2.1: Whale TX dla {symbol} = ${whale_usd:.2f}")

    # Orderbook anomaly detection
    signals["orderbook_anomaly"] = detect_orderbook_anomaly(symbol)

    # DEX inflow detection
    inflow = get_dex_inflow(symbol)
    if inflow > 0.3:  # ⬅️ próg 0.3 ETH / BNB / etc.
        signals["dex_inflow"] = True
    else:
        inflow = 0.0

    # Finalna decyzja: aktywowany jeśli co najmniej 1 aktywny sygnał
    stage2_pass = any(signals.values())

    return stage2_pass, signals, inflow
