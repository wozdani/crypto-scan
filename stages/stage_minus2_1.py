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
from stages.stage_minus1 import detect_stage_minus1_compression
from utils.custom_detectors import detect_stealth_acc, detect_rsi_flatline, get_rsi_from_data
import numpy as np
import json
import os
import requests
import traceback
import logging

logger = logging.getLogger(__name__)

# Volume spike cooldown tracker
volume_spike_cooldown = {}

def detect_volume_spike(symbol, data):
    """
    Wykrywa nagly skok wolumenu w ostatnich 2-3 świecach z cooldown (45 min)
    Analizuje każdą z ostatnich 3 świec względem wcześniejszych 4
    """
    print(f"[DEBUG] detect_volume_spike({symbol}) - START")
    
    # Sprawdź cooldown (3 świece = 45 minut)
    import time
    current_time = time.time()
    cooldown_period = 45 * 60  # 45 minut w sekundach
    
    if symbol in volume_spike_cooldown:
        time_since_spike = current_time - volume_spike_cooldown[symbol]
        if time_since_spike < cooldown_period:
            remaining = int((cooldown_period - time_since_spike) / 60)
            print(f"[DEBUG] {symbol} w cooldown ({remaining}min pozostało)")
            return False
    
    # Sprawdź recent_volumes z data (preferred method)
    recent_volumes = data.get("recent_volumes", [])
    print(f"[DEBUG] {symbol} recent_volumes: {recent_volumes}")
    
    if len(recent_volumes) >= 7:  # Potrzebujemy co najmniej 7 świec (4 bazowe + 3 do sprawdzenia)
        # Sprawdź każdą z ostatnich 3 świec względem wcześniejszych 4
        for i in range(-3, 0):  # -3, -2, -1 (ostatnie 3 świece)
            # Porównaj świecę i z średnią z 4 wcześniejszych świec
            base_start = i - 4
            base_end = i
            
            if base_start >= -len(recent_volumes):
                base_volumes = recent_volumes[base_start:base_end]
                current_volume = recent_volumes[i]
                
                if len(base_volumes) >= 4:
                    avg_base = sum(base_volumes) / len(base_volumes)
                    spike_threshold = avg_base * 2.5
                    
                    print(f"[DEBUG] {symbol} świeca[{i}]: {current_volume:,.0f} vs avg: {avg_base:,.0f} (threshold: {spike_threshold:,.0f})")
                    
                    if current_volume > spike_threshold:
                        volume_spike_cooldown[symbol] = current_time
                        print(f"[VOLUME SPIKE] {symbol} spike w świecy[{i}]: {current_volume:,.0f} > {spike_threshold:,.0f}")
                        return True, current_volume
        
        print(f"[DEBUG] {symbol} brak spike w ostatnich 3 świecach")
        return False, 0.0
    
    # Sprawdź pojedynczą świecę jako fallback
    elif len(recent_volumes) >= 5:
        avg = sum(recent_volumes[-5:-1]) / 4
        current = recent_volumes[-1]
        spike_threshold = avg * 2.5
        
        print(f"[DEBUG] {symbol} fallback - current: {current:,.0f} vs avg: {avg:,.0f} (threshold: {spike_threshold:,.0f})")
        
        if current > spike_threshold:
            volume_spike_cooldown[symbol] = current_time
            print(f"[VOLUME SPIKE] {symbol} fallback spike: {current:,.0f} > {spike_threshold:,.0f}")
            return True, current
        return False, 0.0
    
    # Fallback do market_data jeśli brak recent_volumes
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
        chain = token_info["chain"].lower() if isinstance(token_info.get("chain"), str) else ""

        
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
    if not isinstance(chain, str):
        return False
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
    """Wykrywa anomalie w napływie DEX - zwiększona czułość"""
    print("RUNNING: detect_dex_inflow_anomaly")
    try:
        inflow_usd = get_dex_inflow(symbol, {})
        if isinstance(inflow_usd, (int, float)) and inflow_usd > 0:
            # Oblicz market cap dla dynamicznego progu
            market_cap = price_usd * 1000000 if price_usd else 50000000  # fallback 50M
            
            # Zwiększona czułość: 1.2% market cap (było 2.5%)
            threshold = market_cap * 0.012
            weak_threshold = market_cap * 0.006  # 0.6% dla słabych sygnałów
            
            if inflow_usd > threshold:
                print(f"[DEX INFLOW] Strong anomaly: ${inflow_usd:,.0f} > ${threshold:,.0f}")
                return inflow_usd
            elif inflow_usd > weak_threshold:
                print(f"[DEX INFLOW] Weak anomaly: ${inflow_usd:,.0f} > ${weak_threshold:,.0f}")
                return inflow_usd * 0.5  # Słabszy sygnał, ale nadal aktywny
                
        return 0.0
    except Exception as e:
        logger.error(f"❌ Błąd w detect_dex_inflow_anomaly dla {symbol}: {e}")
        return 0.0

def detect_social_spike(symbol):
    """Detect social media activity spike"""
    print("RUNNING: detect_social_spike")
    try:
        # Simple simulation based on symbol characteristics
        # In production, this would connect to Twitter/Reddit APIs
        symbol_base = symbol.replace("USDT", "").lower()
        
        # Simulate higher social activity for certain patterns
        social_indicators = [
            len(symbol_base) <= 4,  # Short names get more attention
            any(char in symbol_base for char in ['ai', 'dog', 'cat', 'meme']),
            symbol_base.endswith(('coin', 'token'))
        ]
        
        # Basic heuristic: if 2+ indicators, simulate social spike
        spike_detected = sum(social_indicators) >= 2
        
        if spike_detected:
            print(f"[SOCIAL SPIKE] Detected for {symbol} (indicators: {sum(social_indicators)})")
        
        return spike_detected
    except Exception as e:
        logger.error(f"❌ Błąd w detect_social_spike dla {symbol}: {e}")
        return False

def detect_event_tag(symbol):
    """Detect event tags from Stage -2.2 integration"""
    try:
        from stages.stage_minus2_2 import detect_stage_minus2_2
        tag, score_boost, risk_flag = detect_stage_minus2_2(symbol)
        
        if tag:
            return tag
        
        return None
    except Exception as e:
        logger.error(f"❌ Błąd w detect_event_tag dla {symbol}: {e}")
        return None

def detect_stage_minus2_1(symbol, price_usd=None):
    print(f"[DEBUG] === STAGE -2.1 ANALYSIS START: {symbol} ===")
    print(f"[DEBUG] Input price_usd: {price_usd}")
    
    try:
        result = get_market_data(symbol)
        print(f"[DEBUG] get_market_data result type: {type(result)}, length: {len(result) if isinstance(result, (tuple, list)) else 'N/A'}")
        
        if not isinstance(result, tuple) or len(result) != 4:
            print(f"❌ Dane rynkowe {symbol} nieprawidłowe: {result}")
            return False, {}, 0.0, False

        success, data, price_usd, compressed = result
        print(f"[DEBUG] Market data unpacked - success: {success}, data type: {type(data)}, price_usd: {price_usd}")

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
        try:
            volume_spike_result = detect_volume_spike(symbol, data)
            if isinstance(volume_spike_result, tuple):
                volume_spike_active, _ = volume_spike_result
            else:
                volume_spike_active = bool(volume_spike_result)
        except Exception as e:
            logger.error(f"❌ [detect_stage_minus2_1] Błąd w detect_volume_spike dla {symbol}: {e}")
            volume_spike_active = False
            
        try:
            vwap_result = detect_vwap_pinning(symbol, data)
            if isinstance(vwap_result, tuple):
                vwap_pinned, _ = vwap_result
            else:
                vwap_pinned = bool(vwap_result)
        except Exception as e:
            logger.error(f"❌ [detect_stage_minus2_1] Błąd w detect_vwap_pinning dla {symbol}: {e}")
            vwap_pinned = False
            
        try:
            cluster_result = detect_volume_cluster_slope(data)
            if isinstance(cluster_result, tuple):
                volume_slope_up, _ = cluster_result
            else:
                volume_slope_up = bool(cluster_result)
        except Exception as e:
            logger.error(f"❌ [detect_stage_minus2_1] Błąd w detect_volume_cluster_slope dla {symbol}: {e}")
            volume_slope_up = False
            
        try:
            heatmap_exhaustion, _ = detect_heatmap_exhaustion(symbol)
        except Exception as e:
            logger.error(f"❌ [detect_stage_minus2_1] Błąd w detect_heatmap_exhaustion dla {symbol}: {e}")
            heatmap_exhaustion = False
            
        try:
            spoofing_suspected, _ = detect_orderbook_spoofing(symbol)
        except Exception as e:
            logger.error(f"❌ [detect_stage_minus2_1] Błąd w detect_orderbook_spoofing dla {symbol}: {e}")
            spoofing_suspected = False
            
        try:
            orderbook_anomaly, _ = detect_orderbook_anomaly(symbol)
        except Exception as e:
            logger.error(f"❌ [detect_stage_minus2_1] Błąd w detect_orderbook_anomaly dla {symbol}: {e}")
            orderbook_anomaly = False

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
        
        # Social spike detection
        social_spike_active = detect_social_spike(symbol)

        # Event tags (Stage –2.2)
        event_tag = detect_event_tag(symbol)

        # Stage 1g – tylko jeśli spełnione warunki wstępne
        stage1g_active, stage1g_trigger_type = detect_stage_1g(symbol, data, event_tag)

        # Przygotuj sygnały Stage -2.1 dla PPWCS 2.6
        stage_minus2_1_signals = {
            "whale_activity": whale_activity,
            "dex_inflow": inflow_usd > 0,
            "orderbook_anomaly": orderbook_anomaly,
            "volume_spike": volume_spike_active,
            "vwap_pinning": vwap_pinned,
            "spoofing": spoofing_suspected,
            "cluster_slope": volume_slope_up,
            "heatmap_exhaustion": heatmap_exhaustion,
            "social_spike": social_spike_active
        }

        # Stage -1: Compression Filter (PPWCS 2.6)
        compressed = detect_stage_minus1_compression(symbol, stage_minus2_1_signals)
        print(f"[DEBUG] {symbol} compression result: {compressed}")

        # Get RSI value for RSI flatline detection
        rsi_value = get_rsi_from_data(data)
        print(f"[DEBUG] {symbol} RSI value: {rsi_value}")

        signals = {
            # Stage -2.1 Core Detectors (PPWCS 2.6)
            "whale_activity": whale_activity,
            "dex_inflow": inflow_usd > 0,
            "orderbook_anomaly": orderbook_anomaly,
            "volume_spike": volume_spike_active,
            "spoofing": spoofing_suspected,
            "vwap_pinning": vwap_pinned,
            "heatmap_exhaustion": heatmap_exhaustion,
            "cluster_slope": volume_slope_up,
            "social_spike": social_spike_active,
            
            # Stage Processing
            "event_tag": event_tag,
            "stage1g_active": stage1g_active,
            "stage1g_trigger_type": stage1g_trigger_type,
            "compressed": compressed,
            
            # Additional PPWCS 2.6 Fields
            "spoofing_suspected": spoofing_suspected,
            "volume_slope_up": volume_slope_up,
            "pure_accumulation": whale_activity and inflow_usd > 0 and not False,  # whale+DEX without social
            "inflow_usd": inflow_usd,
            
            # Custom Detectors (PPWCS 2.6 Enhanced)
            "squeeze": False,
            "stealth_acc": detect_stealth_acc({"whale_activity": whale_activity, "dex_inflow": inflow_usd > 0, "social_spike": social_spike_active}),
            "fake_reject": False,
            "liquidity_box": False,
            "RSI_flatline": detect_rsi_flatline(rsi_value, {"dex_inflow": inflow_usd > 0, "whale_activity": whale_activity}),
            "fractal_echo": False,
            "news_boost": event_tag is not None and isinstance(event_tag, str) and event_tag.lower() in ["listing", "partnership", "presale", "cex_listed"],
            "inflow": inflow_usd > 0
        }

        # Debug wszystkich detektorów przed finalną decyzją
        detector_results = {
            "whale_activity": whale_activity,
            "orderbook_anomaly": orderbook_anomaly, 
            "volume_spike_active": volume_spike_active,
            "dex_inflow": inflow_usd > 0,
            "heatmap_exhaustion": heatmap_exhaustion,
            "spoofing_suspected": spoofing_suspected,
            "vwap_pinned": vwap_pinned,
            "volume_slope_up": volume_slope_up,
            "social_spike_active": social_spike_active
        }
        
        print(f"[DEBUG] {symbol} detector results: {detector_results}")
        print(f"[DEBUG] {symbol} inflow_usd: {inflow_usd}")
        
        # PPWCS boost logic - combo signals
        combo_volume_inflow = volume_spike_active and inflow_usd > 0
        if combo_volume_inflow:
            print(f"💥 {symbol} = COMBO signal volume + inflow")
        
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
        
        print(f"[DEBUG] {symbol} stage2_pass result: {stage2_pass}")
        print(f"[DEBUG] === STAGE -2.1 ANALYSIS END: {symbol} ===")

        return stage2_pass, signals, inflow_usd, stage1g_active

    except Exception as e:
        logger.error(f"❌ Błąd krytyczny w detect_stage_minus2_1 dla {symbol}: {e}")
        logger.debug("🔍 Traceback:\n" + traceback.format_exc())
        return False, {}, 0.0, False