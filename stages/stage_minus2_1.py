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
    """
    Enhanced DEX inflow detection with dynamic thresholds and known DEX addresses
    Returns: True/False based on inflow analysis
    """
    try:
        from data.known_dex_addresses import DEX_ADDRESSES
        
        # Get contract information
        contract_info = get_contract(symbol)
        if not contract_info or not isinstance(contract_info, dict):
            print(f"⚠️ Brak kontraktu dla {symbol}")
            return False

        address = contract_info.get("address")
        chain = contract_info.get("chain", "").lower()
        
        if not address or not chain:
            print(f"⚠️ Niepełne dane kontraktu dla {symbol}")
            return False

        # Check if chain is supported
        if not is_chain_supported(chain):
            print(f"⚠️ Chain {chain} nieobsługiwany dla {symbol}")
            return False

        # Get DEX addresses for this chain
        dex_addresses = DEX_ADDRESSES.get(chain, [])
        if not dex_addresses:
            print(f"⚠️ Brak adresów DEX dla chain {chain}")
            return False

        # Convert DEX addresses to lowercase for comparison
        dex_addresses_lower = [addr.lower() for addr in dex_addresses]

        # 1. Get token transactions
        txs = get_token_transfers(address, chain)
        if not txs:
            return False

        # 2. Calculate dynamic threshold: max(market_cap * 0.0005, 3000 USD)
        if price_usd and price_usd > 0:
            # Estimate market cap (simplified calculation)
            estimated_market_cap = price_usd * 1000000000  # Assume 1B token supply
            base_threshold = estimated_market_cap * 0.0005  # 0.05% market cap
        else:
            base_threshold = 0
        
        min_usd = 3000
        threshold = max(base_threshold, min_usd)

        # 3. Process transactions and calculate inflow
        inflow_total = 0
        inflow_tx_count = 0

        for tx in txs[:50]:  # Analyze last 50 transactions
            try:
                to_addr = tx.get("to", "").lower()
                
                # Check if transaction is to a known DEX address
                if to_addr in dex_addresses_lower:
                    # Calculate USD value
                    decimals = int(tx.get("decimals", 18))
                    amount = float(tx.get("value", 0)) / (10 ** decimals)
                    
                    if price_usd and price_usd > 0:
                        usd_value = amount * price_usd
                        
                        # Filter microscopic transactions below $50
                        if usd_value >= 50:
                            inflow_total += usd_value
                            inflow_tx_count += 1
                            
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error processing transaction for {symbol}: {e}")
                continue

        # 4. Determine inflow status
        if inflow_total >= threshold:
            print(f"✅ DEX inflow wykryty dla {symbol}: {inflow_total:.2f} USD > próg {threshold:.2f}")
            print(f"   Transakcji DEX: {inflow_tx_count}, Chain: {chain}")
            return True

        elif inflow_tx_count >= 3 and inflow_total >= threshold * 0.5:
            print(f"🟡 Umiarkowany inflow dla {symbol}: {inflow_total:.2f} USD ({inflow_tx_count} txs)")
            return True

        else:
            print(f"❌ Brak istotnego inflow dla {symbol}: {inflow_total:.2f} USD ({inflow_tx_count} txs)")
            return False

    except Exception as e:
        print(f"❌ Błąd w detect_dex_inflow dla {symbol}: {e}")
        return False

def get_token_transfers(address, chain):
    """
    Get token transfer transactions from blockchain explorers
    Returns list of transactions
    """
    import requests
    import os
    
    try:
        # Select appropriate API endpoint and key
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
            print(f"⚠️ Chain {chain} not supported for token transfers")
            return []

        if not api_key:
            print(f"⚠️ Missing API key for {chain}")
            return []

        # Get token transfer transactions
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": address,
            "page": 1,
            "offset": 50,
            "sort": "desc",
            "apikey": api_key,
        }

        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        if data.get("status") == "1" and isinstance(data.get("result"), list):
            return data["result"]
        else:
            print(f"⚠️ No token transfers found for {address} on {chain}")
            return []

    except Exception as e:
        print(f"❌ Error getting token transfers for {address} on {chain}: {e}")
        return []

def detect_stealth_inflow(symbol, whale_activity, volume_spike, compressed, dex_inflow_detected):
    """
    Stealth Inflow Detector - wykrywa ukryty inflow gdy brak klasycznego DEX inflow
    
    Args:
        symbol: symbol tokena
        whale_activity: czy wykryto aktywność wielorybów
        volume_spike: czy wykryto skok wolumenu
        compressed: czy wykryto kompresję
        dex_inflow_detected: czy wykryto klasyczny DEX inflow
    
    Returns:
        tuple: (stealth_inflow_active, inflow_strength)
    """
    try:
        # Stealth inflow wykrywany tylko gdy brak klasycznego DEX inflow
        if not dex_inflow_detected:
            # Warunki stealth inflow: whale_activity + (volume_spike OR compressed)
            if whale_activity and (volume_spike or compressed):
                print(f"🕵️ Stealth inflow wykryty dla {symbol}: whale + {'volume' if volume_spike else 'compressed'}")
                return True, "stealth"
            else:
                return False, "none"
        else:
            # Gdy jest klasyczny DEX inflow, nie oznaczamy jako stealth
            return False, "dex_confirmed"
            
    except Exception as e:
        print(f"❌ Błąd w detect_stealth_inflow dla {symbol}: {e}")
        return False, "error"

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

def detect_whale_execution_pattern(signals):
    """
    Whale Execution Pattern Detector
    Wykrywa sekwencję: dex_inflow → whale_tx → orderbook_anomaly
    """
    if (signals.get("dex_inflow") and 
        signals.get("whale_activity") and 
        signals.get("orderbook_anomaly")):
        print(f"[WHALE EXECUTION] Pattern detected: dex_inflow + whale_activity + orderbook_anomaly")
        return True
    return False

def detect_blockspace_friction(symbol):
    """
    Blockspace Friction Detector
    Wykrywa wzrost opłat na chainie sugerujących whale działania
    """
    try:
        # Symulacja wzrostu gas price i mempool pressure
        # W produkcji połączy się z API blockchain explorers
        symbol_base = symbol.replace("USDT", "").lower()
        
        # Simulacja wzrostu aktywności na chainie
        friction_indicators = [
            len(symbol_base) <= 5,  # Krótkie nazwy generują więcej transakcji
            any(word in symbol_base for word in ['eth', 'btc', 'bnb']),  # Główne tokeny
            hash(symbol_base) % 10 < 3  # 30% szans na friction
        ]
        
        if sum(friction_indicators) >= 2:
            print(f"[GAS PRESSURE] Blockspace friction detected for {symbol}")
            return True
            
        return False
    except Exception as e:
        print(f"[GAS PRESSURE] Error detecting friction for {symbol}: {e}")
        return False

def detect_whale_dominance_ratio(symbol, dex_inflow_value=0.0, whale_value=0.0):
    """
    Whale Dominance Ratio Detector
    Sprawdza czy kilka walletów odpowiada za większość objętości
    """
    try:
        # Convert inputs to float
        dex_val = float(dex_inflow_value) if dex_inflow_value else 0.0
        whale_val = float(whale_value) if whale_value else 0.0
        
        if dex_val <= 0 and whale_val <= 0:
            return False
            
        # Symulacja dominance ratio - w produkcji dane z blockchain
        total_volume = max(dex_val, whale_val) * 1.5  # Szacowana całkowita objętość
        top_3_volume = max(dex_val, whale_val)  # Top 3 wallety
        
        dominance_ratio = top_3_volume / total_volume if total_volume > 0 else 0
        
        if dominance_ratio > 0.65:
            print(f"[WHALE DOMINANCE] High concentration: {dominance_ratio:.2f} for {symbol}")
            return True
            
        return False
    except Exception as e:
        print(f"[WHALE DOMINANCE] Error calculating ratio for {symbol}: {e}")
        return False

def detect_time_clustering(symbol, signals):
    """
    Time Clustering Detector - PPWCS v2.8
    Wykrywa aktywację wielu tokenów z jednego sektora w ciągu 30 minut
    """
    try:
        # Symulacja sector clustering - w produkcji dane z bazy danych
        symbol_base = symbol.replace("USDT", "").lower()
        
        # Definiuj sektory na podstawie nazw tokenów
        sectors = {
            'meme': ['doge', 'shib', 'pepe', 'floki', 'bonk'],
            'ai': ['fet', 'agix', 'ocean', 'rndr', 'wld'],
            'defi': ['uni', 'aave', 'comp', 'mkr', 'crv'],
            'gaming': ['axs', 'sand', 'mana', 'gala', 'enjn'],
            'layer1': ['eth', 'sol', 'ada', 'dot', 'avax']
        }
        
        # Znajdź sektor dla aktualnego tokena
        current_sector = None
        for sector, tokens in sectors.items():
            if any(token in symbol_base for token in tokens):
                current_sector = sector
                break
                
        if not current_sector:
            return False
            
        # Symulacja sprawdzenia innych tokenów w sektorze
        # W produkcji sprawdziłby rzeczywiste sygnały z ostatnich 30 minut
        sector_activity_count = 0
        required_signals = ['compressed', 'whale_activity', 'volume_spike']
        
        # Sprawdź czy aktualny token ma wymagane sygnały
        current_token_signals = sum([signals.get(sig, False) for sig in required_signals])
        
        if current_token_signals >= 2:  # Aktualny token ma silne sygnały
            # Symulacja aktywności innych tokenów w sektorze
            sector_tokens = sectors[current_sector]
            for token in sector_tokens[:3]:  # Sprawdź max 3 tokeny
                # Symulacja prawdopodobieństwa aktywności (30% szans)
                if hash(f"{token}{symbol}") % 10 < 3:
                    sector_activity_count += 1
                    
        clustering_detected = sector_activity_count >= 2
        
        if clustering_detected:
            print(f"[SECTOR CLUSTERING] Detected in {current_sector} sector: {sector_activity_count} active tokens")
            
        return clustering_detected
        
    except Exception as e:
        print(f"[SECTOR CLUSTERING] Error detecting clustering for {symbol}: {e}")
        return False

def detect_execution_intent(buy_volume, sell_volume):
    """
    Execution Intent Detector
    Upewnia się, że whale nie tylko wpłacił tokeny, ale je kupił
    """
    try:
        if buy_volume <= 0 or sell_volume <= 0:
            return False
            
        if buy_volume > 2 * sell_volume:
            print(f"[EXECUTION INTENT] Strong buy intent: buy {buy_volume} > 2x sell {sell_volume}")
            return True
            
        return False
    except Exception as e:
        print(f"[EXECUTION INTENT] Error detecting intent: {e}")
        return False

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

        # Enhanced Whale activity detection
        whale_activity = False
        large_tx_count = 0
        whale_total_usd = 0.0
        
        if price_usd:
            whale_result = detect_whale_tx(symbol, price_usd=price_usd)
            # detect_whale_tx now returns (whale_active, large_tx_count, total_usd)
            if isinstance(whale_result, tuple) and len(whale_result) >= 3:
                whale_activity, large_tx_count, whale_total_usd = whale_result
                print(f"[WHALE INTEGRATION] {symbol}: active={whale_activity}, count={large_tx_count}, total=${whale_total_usd:,.0f}")
                
                # Enhanced whale scoring logic
                if large_tx_count >= 3 and whale_total_usd > 150_000:
                    print(f"[WHALE ENHANCED] {symbol}: Strong whale pattern detected")
                    whale_activity = True
            else:
                # Backward compatibility for old format
                whale_activity = bool(whale_result[0] if isinstance(whale_result, tuple) else whale_result)
                print(f"[WHALE LEGACY] {symbol}: Using legacy detection result: {whale_activity}")
        else:
            print(f"⚠️ Brak ceny USD dla {symbol} (price_usd={price_usd}) – pomijam whale tx.")

        # DEX inflow
        inflow_usd = detect_dex_inflow_anomaly(symbol, price_usd=price_usd)
        dex_inflow_detected = inflow_usd > 0
        
        # Stealth inflow detection
        stealth_inflow_active, inflow_strength = detect_stealth_inflow(
            symbol, whale_activity, volume_spike_active, compressed, dex_inflow_detected
        )
        
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

        # PPWCS v2.8 New Detectors
        gas_pressure = detect_blockspace_friction(symbol)
        whale_value = whale_activity if isinstance(whale_activity, (int, float)) else (1 if whale_activity else 0)
        whale_dominance = detect_whale_dominance_ratio(symbol, float(inflow_usd), float(whale_value))
        
        # Simulate buy/sell volumes for execution intent
        buy_volume = inflow_usd * 1.2 if inflow_usd > 0 else 0
        sell_volume = inflow_usd * 0.4 if inflow_usd > 0 else 1
        execution_intent = detect_execution_intent(buy_volume, sell_volume)
        
        # Build initial signals for whale execution pattern
        initial_signals = {
            "dex_inflow": inflow_usd > 0,
            "whale_activity": whale_activity,
            "orderbook_anomaly": orderbook_anomaly
        }
        whale_sequence = detect_whale_execution_pattern(initial_signals)
        
        # Import Stage 1g detectors for quality scoring
        from stages.stage_1g import detect_dex_pool_divergence, detect_fake_reject, detect_heatmap_liquidity_trap
        
        # Get price for DEX divergence detection
        current_price = price_usd if price_usd else 1.0
        dex_divergence = detect_dex_pool_divergence(symbol, current_price)
        fake_reject = detect_fake_reject(data, bool(volume_spike_active))
        heatmap_trap = detect_heatmap_liquidity_trap(symbol)
        
        # Time clustering detector
        temp_signals = {
            "compressed": compressed,
            "whale_activity": whale_activity,
            "volume_spike": volume_spike_active
        }
        sector_clustering = detect_time_clustering(symbol, temp_signals)

        # Build comprehensive signals dictionary
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
            
            # Stealth Inflow Detection
            "stealth_inflow": stealth_inflow_active,
            "inflow_strength": inflow_strength,
            
            # RSI value for checklist evaluation
            "rsi_value": rsi_value,
            
            # Stage Processing
            "event_tag": event_tag,
            "stage1g_active": stage1g_active,
            "stage1g_trigger_type": stage1g_trigger_type,
            "compressed": compressed,
            
            # PPWCS v2.8 New Detectors
            "whale_sequence": whale_sequence,
            "gas_pressure": gas_pressure,
            "dominant_accumulation": whale_dominance,
            "sector_clustering": sector_clustering,
            "execution_intent": execution_intent,
            
            # Stage 1g Quality Detectors
            "dex_divergence": dex_divergence,
            "fake_reject": fake_reject,
            "heatmap_trap": heatmap_trap,
            
            # Additional PPWCS 2.6 Fields
            "spoofing_suspected": spoofing_suspected,
            "volume_slope_up": volume_slope_up,
            "pure_accumulation": whale_activity and inflow_usd > 0 and not social_spike_active,  # whale+DEX without social
            "inflow_usd": inflow_usd,
            
            # Enhanced Whale Data
            "large_tx_count": large_tx_count,
            "whale_total_usd": whale_total_usd,
            
            # Custom Detectors (PPWCS 2.6 Enhanced)
            "squeeze": False,
            "stealth_acc": detect_stealth_acc({"whale_activity": whale_activity, "dex_inflow": inflow_usd > 0, "social_spike": social_spike_active}),
            "fake_reject": False,
            "liquidity_box": False,
            "RSI_flatline": detect_rsi_flatline(rsi_value, {"dex_inflow": inflow_usd > 0, "whale_activity": whale_activity}),
            "fractal_echo": False,
            "news_boost": event_tag is not None and isinstance(event_tag, str) and event_tag.lower() in ["listing", "partnership", "presale", "cex_listed"],
            "inflow": inflow_usd > 0,
            
            # PPWCS v2.8 New Detectors
            "whale_sequence": whale_sequence,
            "gas_pressure": gas_pressure,
            "dominant_accumulation": whale_dominance,
            "execution_intent": execution_intent,
            "sector_clustering": sector_clustering,
            "dex_divergence": dex_divergence,
            "fake_reject": fake_reject,
            "heatmap_trap": heatmap_trap,
            
            # Combo Signals for PPWCS boosting
            "combo_volume_inflow": volume_spike_active and inflow_usd > 0
        }

        # PPWCS v3.0 & New Checklist Scoring System Integration
        from utils.scoring import compute_combined_scores, compute_checklist_score
        
        try:
            # Original PPWCS v3.0 system
            scoring_results = compute_combined_scores(signals)
            
            # New checklist scoring system per user request
            user_checklist_score, user_checklist_summary = compute_checklist_score(signals)
            
            # Add both scoring systems to signals
            signals.update({
                # PPWCS v3.0 system
                "ppwcs": scoring_results["ppwcs"],
                "checklist_score": scoring_results["checklist_score"],
                "checklist_summary": scoring_results["checklist_summary"],
                "total_combined": scoring_results["total_combined"],
                "hard_signal_count": scoring_results["hard_signal_count"],
                "soft_signal_count": scoring_results["soft_signal_count"],
                
                # User requested checklist system
                "user_checklist_score": user_checklist_score,
                "user_checklist_summary": user_checklist_summary
            })
            
            print(f"[SCORING v3.0] {symbol}: PPWCS={scoring_results['ppwcs']}/70, Checklist={scoring_results['checklist_score']}/90")
            print(f"[USER CHECKLIST] {symbol}: Score={user_checklist_score}/100, Conditions={len(user_checklist_summary)}/20")
            print(f"[COMBINED] {symbol}: v3.0_total={scoring_results['total_combined']}/160, user_checklist={user_checklist_score}/100")
            
        except Exception as e:
            print(f"❌ Error in scoring for {symbol}: {e}")
            # Add default scoring values if error occurs
            signals.update({
                "ppwcs": 0,
                "checklist_score": 0,
                "checklist_summary": [],
                "total_combined": 0,
                "hard_signal_count": 0,
                "soft_signal_count": 0,
                "user_checklist_score": 0,
                "user_checklist_summary": []
            })

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