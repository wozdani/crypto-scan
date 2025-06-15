from utils.contracts import get_contract
from utils.data_fetchers import get_market_data, get_all_data
from utils.whale_detector import detect_whale_tx
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
import traceback
import logging

logger = logging.getLogger(__name__)

def detect_volume_spike(symbol, data):
    """
    Wykrywa nagly skok wolumenu: Z-score > 2.5 lub 3x Średnia z poprzednich 4 świec.
    """
    market_data = get_all_data(symbol)
    if not market_data or not isinstance(market_data, dict):
        print(f"❌ [detect_volume_spike] Brak danych lub nieprawidłowy typ dla {symbol}: {type(market_data)}")
        return False, 0.0
    
    if not market_data.get("prev_candle"):
        return False, 0.0

    try:
        current_volume = float(market_data["volume"])
        prev_volume = float(market_data["prev_candle"][5])

        # Dla testow uzywamy tylko jednej wczesniejszej swiecy (w przyszlosci 4)
        volumes = [prev_volume, current_volume]
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        z_score = (current_volume -
                   mean_volume) / std_volume if std_volume > 0 else 0
        spike_detected = z_score > 2.5 or current_volume > 3 * prev_volume

        return spike_detected, current_volume
    except Exception as e:
        logger.error(f"âťŚ laÄ…d w volume spike dla {symbol}: {e}")
        return False, 0.0

def get_dex_inflow(symbol, data):
    try:
        token_info = get_contract(symbol)
        if not isinstance(token_info, dict):
            print(f"❌ token_info nie jest dict dla {symbol}: {type(token_info)} → {token_info}")
            return 0.0
        if token_info is None:
            print(f"⚠️ Brak kontraktu dla {symbol}")
            return 0.0
        
        print(f"✅ Kontrakt dla {symbol}: {token_info}")

        if not token_info:
            print(f"⛔ Token {symbol} nie istnieje w cache CoinGecko")
            return 0.0
        if "chain" not in token_info:
            print(f"⚠️ Brak danych chain w token_info dla {symbol}")
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
        elif chain == "tron":
            base_url = "https://api.trongrid.io/v1"
            api_key = os.getenv("TRONGRID_API_KEY")
        else:
            print(f"⚠️ Chain {chain} nieobsługiwany ")
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

        try:
            response = requests.get(base_url, params=params, timeout=10)

            try:
                data = response.json()
            except Exception as e:
                logger.error(f"❌ [inflow] Błąd dekodowania JSON dla {symbol}: {e}")
                return 0.0

            if not isinstance(data, dict):
                logger.error(f"❌ [inflow] Odpowiedź nie jest dict dla {symbol}: {type(data)} → {data}")
                return 0.0

            if data.get("status") != "1":
                logger.warning(f"❌ [inflow] Status != 1 dla {symbol}: {data.get('message', 'brak komunikatu')}")
                return 0.0

            result = data.get("result")
            if not isinstance(result, list):
                logger.error(f"❌ [inflow] Nieprawidłowy typ pola 'result' dla {symbol}: {type(result)} → {result}")
                return 0.0

            # Liczymy sumę wpływów do kontraktu
            inflow_sum = 0.0
            for tx in result[:10]:
                try:
                    # Sprawdź czy tx jest dictionary
                    if not isinstance(tx, dict):
                        logger.warning(f"⚠️ [inflow] tx nie jest dict dla {symbol}: {type(tx)} → {tx}")
                        continue
                    
                    # Dodatkowa walidacja przed użyciem .get()
                    tx_to = tx.get("to", "") if isinstance(tx, dict) else ""
                    tx_value = tx.get("value", 0) if isinstance(tx, dict) else 0
                    
                    if isinstance(tx_to, str) and tx_to.lower() == address.lower():
                        inflow_sum += int(tx_value) / (10 ** 18)
                except Exception as e:
                    logger.warning(f"⚠️ [inflow] Błąd w transakcji {tx} dla {symbol}: {e}")

            return inflow_sum

        except Exception as e:
            logger.error(f"❌ [inflow] Błąd krytyczny dla {symbol}: {e}")
            return 0.0
    
    except Exception as e:
        print(f"❌ Błąd przy pobieraniu kontraktu dla {symbol}: {e}")
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
    if chain == "tron" and os.getenv("TRONGRID_API_KEY"):
        return True
    return False

def detect_dex_inflow_anomaly(symbol, price_usd=None):
    """Wykrywa anomalie w napływie DEX - uproszczona wersja"""
    try:
        inflow_usd = get_dex_inflow(symbol, {})
        if isinstance(inflow_usd, (int, float)) and inflow_usd > 0:
            # Próg dla anomalii - możesz dostosować
            threshold = 10000  # $10k USD
            return inflow_usd if inflow_usd > threshold else 0.0
        return 0.0
    except Exception as e:
        logger.error(f"❌ Błąd w detect_dex_inflow_anomaly dla {symbol}: {e}")
        return 0.0

def detect_event_tag(symbol):
    """Wykrywa tagi eventów - uproszczona wersja"""
    try:
        # Podstawowa implementacja - możesz rozszerzyć
        return False, 0.0, "low"
    except Exception as e:
        logger.error(f"❌ Błąd w detect_event_tag dla {symbol}: {e}")
        return False, 0.0, "low"

def detect_stage_minus2_1(symbol, price_usd=None):
    try:
        result = get_market_data(symbol)
        if not isinstance(result, tuple) or len(result) != 4:
            print(f"❌ Dane rynkowe {symbol} nieprawidłowe: {result}")
            return False, {}, 0.0, False

        success, data, price_usd, compressed = result

        if not success or not isinstance(data, dict):
            print(f"❌ [get_market_data] {symbol} → oczekiwano dict, otrzymano: {type(data)} → przerwanie etapu")
            return False, {}, 0.0, False

        if not isinstance(data, dict):
            logger.error(f"❌ [detect_stage_minus2_1] Zmienna 'data' nie jest dict dla {symbol}: {type(data)} → {data}")
            return False, {}, 0.0, False

        # Wczytaj kontrakt
        contract_data = get_contract(symbol)
        if not contract_data:
            return False, {}, 0.0, False
        
        # Sprawdź czy contract_data jest dict
        if not isinstance(contract_data, dict):
            print(f"❌ [detect_stage_minus2_1] contract_data nie jest dict dla {symbol}: {type(contract_data)} → {contract_data}")
            return False, {}, 0.0, False
            
        print(f"✅ Kontrakt dla {symbol}: {contract_data}")

        # Detektory Stage –2.1
        volume_spike_active, _ = detect_volume_spike(symbol, data)
        vwap_pinned, _ = detect_vwap_pinning(symbol, data)
        volume_slope_up, _ = detect_volume_cluster_slope(data)
        heatmap_exhaustion, _ = detect_heatmap_exhaustion(symbol)
        spoofing_suspected, _ = detect_orderbook_spoofing(symbol)
        orderbook_anomaly, _ = detect_orderbook_anomaly(symbol)

        # Whale activity
        whale_activity = False
        if price_usd:
            whale_result = detect_whale_tx(symbol, price_usd=price_usd)
            # detect_whale_tx zwraca tuple (bool, float)
            if isinstance(whale_result, tuple) and len(whale_result) >= 1:
                whale_activity = whale_result[0]
            else:
                whale_activity = whale_result
        else:
            print(f"⚠️ Brak ceny USD dla {symbol} (price_usd={price_usd}) – pomijam whale tx.")

        # DEX inflow
        inflow_usd = detect_dex_inflow_anomaly(symbol, price_usd=price_usd)

        # Event tags (Stage –2.2)
        event_tag, event_score, event_risk = detect_event_tag(symbol)

        # Stage 1g – tylko jeśli spełnione warunki wstępne
        stage1g_active, stage1g_trigger_type = detect_stage_1g(symbol, data)

        signals = {
            "whale_activity": whale_activity,
            "orderbook_anomaly": orderbook_anomaly,
            "volume_spike": volume_spike_active,
            "dex_inflow": inflow_usd > 0,
            "heatmap_exhaustion": heatmap_exhaustion,
            "spoofing_suspected": spoofing_suspected,
            "vwap_pinned": vwap_pinned,
            "volume_slope_up": volume_slope_up,
            "event_tag": event_tag,
            "event_score": event_score,
            "event_risk": event_risk,
            "stage1g_active": stage1g_active,
            "stage1g_trigger_type": stage1g_trigger_type
        }

        # Czy aktywować Stage 2.1 (Stage –2 w strategii)
        stage2_pass = any([
            whale_activity,
            orderbook_anomaly,
            volume_spike_active,
            inflow_usd > 0,
            heatmap_exhaustion,
            spoofing_suspected,
            vwap_pinned,
            volume_slope_up
        ])

        return stage2_pass, signals, inflow_usd, stage1g_active

    except Exception as e:
        logger.error(f"❌ Błąd krytyczny w detect_stage_minus2_1 dla {symbol}: {e}")
        logger.debug("🔍 Traceback:\n" + traceback.format_exc())
        return False, {}, 0.0, False