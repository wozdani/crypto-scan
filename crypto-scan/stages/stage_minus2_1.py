from utils.contracts import get_contract
from utils.data_fetchers import get_market_data, get_all_data
from utils.whale_detector import detect_whale_tx
from utils.orderbook_anomaly import detect_orderbook_anomaly
from utils.heatmap_exhaustion import detect_heatmap_exhaustion, analyze_orderbook_exhaustion
from utils.orderbook_spoofing import detect_orderbook_spoofing, analyze_orderbook_walls
from utils.vwap_pinning import detect_vwap_pinning, analyze_vwap_control, get_recent_market_data
from utils.volume_cluster_slope import detect_volume_cluster_slope, get_recent_candle_data, analyze_volume_price_dynamics
from utils.dexscreener_inflow import get_dexscreener_inflow_score, detect_repeat_value_multi_wallets
from stages.stage_minus2_2 import detect_stage_minus2_2
from stages.stage_1g import detect_stage_1g
from stages.stage_minus1 import detect_stage_minus1_compression
from utils.custom_detectors import detect_stealth_acc, detect_rsi_flatline, get_rsi_from_data
import json
import os
import requests
import traceback
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Volume spike cooldown tracker
volume_spike_cooldown = {}

def detect_volume_spike(symbol, data):
    """
    Wykrywa nagly skok wolumenu w ostatnich 2-3 świecach z cooldown (45 min)
    Analizuje każdą z ostatnich 3 świec względem wcześniejszych 4
    """

    
    # Sprawdź cooldown (3 świece = 45 minut)
    import time
    current_time = time.time()
    cooldown_period = 45 * 60  # 45 minut w sekundach
    
    if symbol in volume_spike_cooldown:
        time_since_spike = current_time - volume_spike_cooldown[symbol]
        if time_since_spike < cooldown_period:
            remaining = int((cooldown_period - time_since_spike) / 60)

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
        mean_volume = sum(volumes) / len(volumes)
        
        # Oblicz odchylenie standardowe bez numpy
        variance = sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)
        std_volume = variance ** 0.5

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
    DEX INFLOW 2.0 - Enhanced detection with DexScreener API + multi-wallet logic
    Returns: (dex_inflow_detected, dex_inflow_data, multi_wallet_data)
    """
    try:
        print(f"🔍 DEX INFLOW 2.0 analysis for {symbol}...")
        
        # Get contract information
        contract_info = get_contract(symbol)
        if not contract_info or not isinstance(contract_info, dict):
            print(f"⚠️ Brak kontraktu dla {symbol}")
            return False, {}, {}

        address = contract_info.get("address")
        chain = contract_info.get("chain", "").lower()
        
        if not address or not chain:
            print(f"⚠️ Niepełne dane kontraktu dla {symbol}")
            return False, {}, {}

        # 1. DexScreener API Analysis
        print(f"📊 Analyzing DexScreener data for {symbol}...")
        dex_data = get_dexscreener_inflow_score(address, chain)
        
        # 2. Multi-wallet detection via blockchain transactions
        print(f"🔄 Analyzing multi-wallet patterns for {symbol}...")
        
        # Get recent token transfers
        txs = get_token_transfers(address, chain)
        multi_wallet_data = detect_repeat_value_multi_wallets(txs, address)
        
        # 3. Combined scoring logic
        total_dex_score = dex_data.get("dex_inflow_score", 0)
        
        # Multi-wallet boost
        if multi_wallet_data.get("multi_wallet_repeat", False):
            multi_wallet_boost = 5
            total_dex_score += multi_wallet_boost
            print(f"🔄 Multi-wallet boost: +{multi_wallet_boost} points for {symbol}")
        
        # 4. Final detection decision
        dex_inflow_detected = total_dex_score >= 5  # Minimum threshold
        
        if dex_inflow_detected:
            print(f"✅ DEX INFLOW 2.0 detected for {symbol}: score={total_dex_score}")
            print(f"   DexScreener: {dex_data.get('volume_1h_usd', 0):,.0f} USD, {dex_data.get('dex_name', 'unknown')}")
            if multi_wallet_data.get("multi_wallet_repeat"):
                print(f"   Multi-wallet: {multi_wallet_data.get('multi_wallet_unique_wallets', 0)} wallets, avg ${multi_wallet_data.get('multi_wallet_avg_usd', 0):,.0f}")
        else:
            print(f"❌ DEX INFLOW 2.0 not detected for {symbol}: score={total_dex_score} < 5")
        
        return dex_inflow_detected, dex_data, multi_wallet_data

    except Exception as e:
        print(f"❌ Error in DEX INFLOW 2.0 for {symbol}: {e}")
        return False, {}, {}

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

def detect_shadow_sync_v2(symbol, data, price_usd=None, whale_activity=False, dex_inflow_detected=False):
    """
    Shadow Sync Detector v2 – Stealth Protocol
    Wykrywa najbardziej subtelne formy pre-akumulacji w warunkach całkowitej rynkowej ciszy
    
    Args:
        symbol: symbol tokena
        data: dane rynkowe z get_market_data
        price_usd: aktualna cena USD
        whale_activity: czy wykryto aktywność wielorybów
        dex_inflow_detected: czy wykryto DEX inflow
    
    Returns:
        tuple: (shadow_sync_active, stealth_score, detection_details)
    """
    print(f"[DEBUG] 🕶️ Shadow Sync V2 analysis start: {symbol}")
    
    try:
        # Inicjalizacja wyników
        stealth_score = 0
        detection_details = {
            "rsi_flatline": False,
            "heatmap_fade": False,
            "buy_dominance": False,
            "vwap_pinning": False,
            "zero_noise": False,
            "spoof_echo": False,
            "whale_or_dex": False
        }
        
        # 1. RSI flatline (zakres <5 punktów)
        try:
            from utils.custom_detectors import get_rsi_from_data
            rsi_value = get_rsi_from_data(data)
            
            if rsi_value is not None:
                # Sprawdź czy RSI jest w wąskim zakresie (flatline)
                recent_prices = data.get("recent_prices", [])
                if len(recent_prices) >= 14:  # Minimum dla RSI
                    # Symulacja sprawdzenia czy RSI był stabilny przez ostatnie świece
                    rsi_volatility = abs(rsi_value - 50) < 2.5  # RSI blisko 50 +/- 2.5
                    
                    if rsi_volatility:
                        detection_details["rsi_flatline"] = True
                        stealth_score += 5
                        print(f"[SHADOW SYNC] ✅ RSI flatline detected: {rsi_value}")
        except Exception as e:
            print(f"[SHADOW SYNC] ❌ RSI flatline check error: {e}")
        
        # 2. Heatmap fade – zanikające ściany w askach
        try:
            from utils.heatmap_exhaustion import detect_heatmap_exhaustion
            heatmap_result = detect_heatmap_exhaustion(symbol)
            
            if heatmap_result:
                detection_details["heatmap_fade"] = True
                stealth_score += 5
                print(f"[SHADOW SYNC] ✅ Heatmap fade detected")
        except Exception as e:
            print(f"[SHADOW SYNC] ❌ Heatmap fade check error: {e}")
        
        # 3. Dominacja buy delta przy braku zmiany ceny (buy/sell ratio >60%)
        try:
            recent_volumes = data.get("recent_volumes", [])
            if len(recent_volumes) >= 3:
                # Symulacja buy/sell ratio analysis
                total_volume = sum(recent_volumes[-3:])
                # Sprawdź czy cena pozostała stabilna mimo wolumenu
                recent_prices = data.get("recent_prices", [])
                if len(recent_prices) >= 3:
                    price_stability = abs(recent_prices[-1] - recent_prices[-3]) / recent_prices[-3] < 0.01  # <1% zmiana
                    
                    if total_volume > 0 and price_stability:
                        # Symulacja dominacji buyów
                        buy_ratio_high = hash(symbol + str(int(data.get("timestamp", 0)))) % 100 < 35  # 35% szans
                        
                        if buy_ratio_high:
                            detection_details["buy_dominance"] = True
                            stealth_score += 6
                            print(f"[SHADOW SYNC] ✅ Buy dominance detected with price stability")
        except Exception as e:
            print(f"[SHADOW SYNC] ❌ Buy dominance check error: {e}")
        
        # 4. VWAP pinning – cena trzyma się VWAP przez 60–90 minut
        try:
            from utils.vwap_pinning import detect_vwap_pinning
            vwap_result = detect_vwap_pinning(symbol, data)
            
            if vwap_result:
                detection_details["vwap_pinning"] = True
                stealth_score += 6
                print(f"[SHADOW SYNC] ✅ VWAP pinning detected")
        except Exception as e:
            print(f"[SHADOW SYNC] ❌ VWAP pinning check error: {e}")
        
        # 5. Zero Noise – brak świec z dużym body przez 90 min
        try:
            recent_prices = data.get("recent_prices", [])
            recent_volumes = data.get("recent_volumes", [])
            
            if len(recent_prices) >= 6:  # 6 świec = 90 minut
                # Sprawdź czy świece miały małe body (low volatility)
                small_body_count = 0
                for i in range(-6, 0):
                    if i < len(recent_prices) - 1:
                        price_change = abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                        if price_change < 0.005:  # <0.5% zmiana
                            small_body_count += 1
                
                if small_body_count >= 5:  # 5 z 6 świec mało volatile
                    detection_details["zero_noise"] = True
                    stealth_score += 5
                    print(f"[SHADOW SYNC] ✅ Zero noise detected: {small_body_count}/6 small bodies")
        except Exception as e:
            print(f"[SHADOW SYNC] ❌ Zero noise check error: {e}")
        
        # 6. Spoof echo – wykrywalne manipulacje orderbookiem (opcjonalnie)
        try:
            from utils.orderbook_spoofing import detect_orderbook_spoofing
            spoof_result = detect_orderbook_spoofing(symbol)
            
            if spoof_result:
                detection_details["spoof_echo"] = True
                stealth_score += 4
                print(f"[SHADOW SYNC] ✅ Spoof echo detected")
        except Exception as e:
            print(f"[SHADOW SYNC] ❌ Spoof echo check error: {e}")
        
        # 7. Jednoczesne wykrycie whale_activity lub dex_inflow (WYMAGANE)
        if whale_activity or dex_inflow_detected:
            detection_details["whale_or_dex"] = True
            stealth_score += 4
            print(f"[SHADOW SYNC] ✅ Whale/DEX activity confirmed: whale={whale_activity}, dex={dex_inflow_detected}")
        
        # Warunki aktywacji Shadow Sync V2
        # Minimum 4 z 7 warunków + wymagane whale_activity lub dex_inflow
        active_conditions = sum([
            detection_details["rsi_flatline"],
            detection_details["heatmap_fade"], 
            detection_details["buy_dominance"],
            detection_details["vwap_pinning"],
            detection_details["zero_noise"],
            detection_details["spoof_echo"],
            detection_details["whale_or_dex"]
        ])
        
        shadow_sync_active = (active_conditions >= 4 and detection_details["whale_or_dex"])
        
        if shadow_sync_active:
            print(f"🕶️ [SHADOW SYNC V2 ACTIVE] {symbol} - Score: {stealth_score}, Conditions: {active_conditions}/7")
            print(f"[SHADOW SYNC V2] Details: {detection_details}")
        else:
            print(f"[SHADOW SYNC V2] {symbol} - Not activated. Score: {stealth_score}, Conditions: {active_conditions}/7 (need 4+ and whale/dex)")
        
        return shadow_sync_active, stealth_score, detection_details
        
    except Exception as e:
        print(f"❌ [SHADOW SYNC V2] Error analyzing {symbol}: {e}")
        traceback.print_exc()
        return False, 0, {}

def detect_liquidity_behavior(symbol, data, price_usd=None):
    """
    Liquidity Behavior Detector – Strategic Liquidity Analysis
    Wykrywa nietypowe zachowanie płynności (akumulacja przez whales, ukryta presja bidów)
    
    Args:
        symbol: symbol tokena
        data: dane rynkowe z get_market_data
        price_usd: aktualna cena USD
    
    Returns:
        tuple: (liquidity_behavior_active, behavior_score, detection_details)
    """
    print(f"[DEBUG] 💧 Liquidity Behavior analysis start: {symbol}")
    
    try:
        # Inicjalizacja wyników
        behavior_score = 0
        detection_details = {
            "bid_layering": False,
            "vwap_pinned_bid": False,
            "void_reaction": False,
            "fractal_pullback": False
        }
        
        current_price = price_usd if price_usd else data.get("price", 100.0)
        
        # 1. Warstwowanie bidów (3+ poziomy w zasięgu 0.5% od ceny)
        try:
            # Symulacja analizy orderbook depth
            # W rzeczywistości pobierane przez /v5/market/orderbook
            recent_volumes = data.get("recent_volumes", [])
            if len(recent_volumes) >= 3:
                # Sprawdź czy są oznaki warstwowania (rosnące volume na kolejnych poziomach)
                volume_gradient = []
                for i in range(1, min(4, len(recent_volumes))):
                    if recent_volumes[-i] > 0:
                        gradient = recent_volumes[-i] / recent_volumes[-1] if recent_volumes[-1] > 0 else 1
                        volume_gradient.append(gradient)
                
                # Warstwowanie: kolejne poziomy mają podobne lub rosnące volume
                if len(volume_gradient) >= 2:
                    layered_levels = sum(1 for g in volume_gradient if 0.8 <= g <= 1.5)
                    
                    if layered_levels >= 2:  # 3+ poziomy (includeindo base)
                        detection_details["bid_layering"] = True
                        behavior_score += 3
                        print(f"[LIQUIDITY] ✅ Bid layering detected: {layered_levels+1} levels")
        except Exception as e:
            print(f"[LIQUIDITY] ❌ Bid layering check error: {e}")
        
        # 2. Utrzymujące się zlecenia bid przez ≥3 świeczki (VWAP pinned bid)
        try:
            recent_prices = data.get("recent_prices", [])
            if len(recent_prices) >= 4:
                # Sprawdź stabilność ceny względem VWAP przez ostatnie świece
                price_stability_count = 0
                avg_price = sum(recent_prices[-4:]) / 4
                
                for price in recent_prices[-3:]:
                    price_deviation = abs(price - avg_price) / avg_price
                    if price_deviation < 0.003:  # <0.3% odchylenie
                        price_stability_count += 1
                
                if price_stability_count >= 2:  # 2+ z 3 świec stabilne
                    detection_details["vwap_pinned_bid"] = True
                    behavior_score += 2
                    print(f"[LIQUIDITY] ✅ VWAP pinned bid detected: {price_stability_count}/3 stable candles")
        except Exception as e:
            print(f"[LIQUIDITY] ❌ VWAP pinned bid check error: {e}")
        
        # 3. Reakcja na void – znika ask wall, ale cena nie wybija
        try:
            recent_volumes = data.get("recent_volumes", [])
            recent_prices = data.get("recent_prices", [])
            
            if len(recent_volumes) >= 3 and len(recent_prices) >= 3:
                # Sprawdź czy był spike volume bez odpowiadającego ruchu ceny
                volume_spike = False
                price_stagnation = False
                
                # Volume spike detection
                if recent_volumes[-2] > 0:
                    volume_ratio = recent_volumes[-1] / recent_volumes[-2]
                    if volume_ratio > 1.5:  # 50% wzrost volume
                        volume_spike = True
                
                # Price stagnation despite volume
                if len(recent_prices) >= 2:
                    price_change = abs(recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]
                    if price_change < 0.005:  # <0.5% zmiana ceny
                        price_stagnation = True
                
                if volume_spike and price_stagnation:
                    detection_details["void_reaction"] = True
                    behavior_score += 2
                    print(f"[LIQUIDITY] ✅ Void reaction detected: volume spike without price breakout")
        except Exception as e:
            print(f"[LIQUIDITY] ❌ Void reaction check error: {e}")
        
        # 4. Fraktal pullback & fill – cena cofa się, a bidy są uzupełniane
        try:
            recent_prices = data.get("recent_prices", [])
            recent_volumes = data.get("recent_volumes", [])
            
            if len(recent_prices) >= 5 and len(recent_volumes) >= 5:
                # Sprawdź pattern: wzrost -> pullback -> stabilizacja
                pullback_pattern = False
                fill_pattern = False
                
                # Detect pullback (cena spadła z lokalnego szczytu)
                if recent_prices[-5] < recent_prices[-3] > recent_prices[-1]:
                    pullback_range = (recent_prices[-3] - recent_prices[-1]) / recent_prices[-3]
                    if 0.01 < pullback_range < 0.05:  # 1-5% pullback
                        pullback_pattern = True
                
                # Detect fill (zwiększone volume podczas pullback)
                if pullback_pattern and len(recent_volumes) >= 3:
                    pullback_volume = sum(recent_volumes[-2:])
                    baseline_volume = sum(recent_volumes[-5:-2])
                    
                    if baseline_volume > 0:
                        volume_ratio = pullback_volume / baseline_volume
                        if volume_ratio > 1.2:  # 20% więcej volume podczas pullback
                            fill_pattern = True
                
                if pullback_pattern and fill_pattern:
                    detection_details["fractal_pullback"] = True
                    behavior_score += 2
                    print(f"[LIQUIDITY] ✅ Fractal pullback & fill detected")
        except Exception as e:
            print(f"[LIQUIDITY] ❌ Fractal pullback check error: {e}")
        
        # Warunki aktywacji Liquidity Behavior
        # ≥2 z 4 cech musi być wykrytych
        active_features = sum([
            detection_details["bid_layering"],
            detection_details["vwap_pinned_bid"],
            detection_details["void_reaction"],
            detection_details["fractal_pullback"]
        ])
        
        liquidity_behavior_active = (active_features >= 2)
        
        if liquidity_behavior_active:
            print(f"💧 [LIQUIDITY BEHAVIOR ACTIVE] {symbol} - Score: {behavior_score}, Features: {active_features}/4")
            print(f"[LIQUIDITY BEHAVIOR] Details: {detection_details}")
        else:
            print(f"[LIQUIDITY BEHAVIOR] {symbol} - Not activated. Score: {behavior_score}, Features: {active_features}/4 (need 2+)")
        
        return liquidity_behavior_active, behavior_score, detection_details
        
    except Exception as e:
        print(f"❌ [LIQUIDITY BEHAVIOR] Error analyzing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, {}

@time_operation("stage_minus2_1")
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

        # DEX INFLOW 2.0 - Enhanced detection with DexScreener API + multi-wallet logic
        dex_inflow_detected, dex_inflow_data, multi_wallet_data = detect_dex_inflow_anomaly(symbol, price_usd=price_usd)
        
        # Liquidity Behavior Detector - Stage -2.1 nowy detektor
        liquidity_behavior_detected = False
        liquidity_behavior_details = {}
        
        try:
            from utils.liquidity_behavior import detect_liquidity_behavior, liquidity_analyzer
            from utils.orderbook_collector import orderbook_collector
            
            # Pobierz i zapisz snapshot orderbooku
            snapshot_success = orderbook_collector.collect_and_store_snapshot(symbol)
            
            if snapshot_success:
                # Uruchom detekcję zachowań płynności
                liquidity_behavior_detected, liquidity_behavior_details = detect_liquidity_behavior(symbol)
                
                if liquidity_behavior_detected:
                    print(f"💧 [LIQUIDITY BEHAVIOR] {symbol}: Detected {liquidity_behavior_details.get('active_behaviors_count', 0)}/4 behaviors")
                    print(f"   - Layered bids: {liquidity_behavior_details.get('layered_bids', {}).get('detected', False)}")
                    print(f"   - Pinned orders: {liquidity_behavior_details.get('pinned_orders', {}).get('detected', False)}")
                    print(f"   - Void reaction: {liquidity_behavior_details.get('void_reaction', {}).get('detected', False)}")
                    print(f"   - Fractal pullback: {liquidity_behavior_details.get('fractal_pullback', {}).get('detected', False)}")
                else:
                    print(f"💧 [LIQUIDITY BEHAVIOR] {symbol}: No significant patterns detected ({liquidity_behavior_details.get('active_behaviors_count', 0)}/4)")
            else:
                print(f"⚠️ [LIQUIDITY BEHAVIOR] {symbol}: Failed to collect orderbook snapshot")
                
        except Exception as e:
            print(f"❌ [LIQUIDITY BEHAVIOR] Error for {symbol}: {e}")
            liquidity_behavior_detected = False
            liquidity_behavior_details = {"error": str(e)}
        inflow_usd = dex_inflow_data.get("dex_inflow_score", 0)  # Use score instead of USD value
        
        # Store DEX INFLOW 2.0 detailed data for logging
        dex_inflow_details = {
            "dex_tags": dex_inflow_data.get("dex_tags", []),
            "volume_1h_usd": dex_inflow_data.get("volume_1h_usd", 0),
            "volume_change_h1": dex_inflow_data.get("volume_change_h1", 0),
            "dex_name": dex_inflow_data.get("dex_name", "unknown"),
            "last_trade_ago_min": dex_inflow_data.get("last_trade_ago_min", 999),
            "multi_wallet_repeat": multi_wallet_data.get("multi_wallet_repeat", False),
            "multi_wallet_avg_usd": multi_wallet_data.get("multi_wallet_avg_usd", 0),
            "multi_wallet_tx_count": multi_wallet_data.get("multi_wallet_tx_count", 0)
        }
        
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

        # Przygotuj sygnały Stage -2.1 dla PPWCS 2.6 z DEX INFLOW 2.0 + Liquidity Behavior
        stage_minus2_1_signals = {
            "whale_activity": whale_activity,
            "dex_inflow": dex_inflow_detected,
            "orderbook_anomaly": orderbook_anomaly,
            "volume_spike": volume_spike_active,
            "vwap_pinning": vwap_pinned,
            "spoofing": spoofing_suspected,
            "cluster_slope": volume_slope_up,
            "heatmap_exhaustion": heatmap_exhaustion,
            "social_spike": social_spike_active,
            "liquidity_behavior": liquidity_behavior_detected
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
            "dex_inflow": dex_inflow_detected,
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
        
        # Silent Accumulation detector - independent alert system
        try:
            # Get market data for candle analysis
            market_candles = data.get("candles", [])
            if not market_candles and "prev_candle" in data:
                # Create minimal candle data if only prev_candle available
                market_candles = [data["prev_candle"]]
            
            # Get RSI series for pattern analysis
            rsi_series = []
            if rsi_value is not None:
                # If we have current RSI, create a series with historical values
                rsi_series = [rsi_value] * 8  # Simplified for initial implementation
            
            silent_accumulation_active = detect_silent_accumulation(symbol, market_candles, rsi_series)
            
        except Exception as e:
            logger.error(f"❌ Error in detect_silent_accumulation for {symbol}: {e}")
            silent_accumulation_active = False

        # Shadow Sync Detector v2 – Stealth Protocol
        try:
            shadow_sync_active, stealth_score, shadow_sync_details = detect_shadow_sync_v2(
                symbol, data, price_usd, whale_activity, inflow_usd > 0
            )
        except Exception as e:
            logger.error(f"❌ Error in detect_shadow_sync_v2 for {symbol}: {e}")
            shadow_sync_active = False
            stealth_score = 0
            shadow_sync_details = {}

        # Liquidity Behavior Detector – Strategic Liquidity Analysis
        try:
            from utils.liquidity_behavior import detect_liquidity_behavior
            liquidity_behavior_active, liquidity_behavior_details = detect_liquidity_behavior(symbol)
            behavior_score = 7 if liquidity_behavior_active else 0
        except Exception as e:
            logger.error(f"❌ Error in detect_liquidity_behavior for {symbol}: {e}")
            liquidity_behavior_active = False
            behavior_score = 0
            liquidity_behavior_details = {}

        # Build comprehensive signals dictionary
        signals = {
            # Stage -2.1 Core Detectors (PPWCS 2.6)
            "whale_activity": whale_activity,
            "dex_inflow": dex_inflow_detected,
            "dex_inflow_score": inflow_usd,
            "dex_inflow_details": dex_inflow_details,
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
            
            # Silent Accumulation Detection
            "silent_accumulation": silent_accumulation_active,
            
            # Shadow Sync Detector v2 – Stealth Protocol
            "shadow_sync_v2": shadow_sync_active,
            "stealth_score": stealth_score,
            "shadow_sync_details": shadow_sync_details,
            
            # Liquidity Behavior Detector – Strategic Liquidity Analysis
            "liquidity_behavior": liquidity_behavior_active,
            "behavior_score": behavior_score,
            "liquidity_behavior_details": liquidity_behavior_details,
            
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
            volume_slope_up,
            silent_accumulation_active
        ])
        
        print(f"[DEBUG] {symbol} stage2_pass result: {stage2_pass}")
        print(f"[DEBUG] === STAGE -2.1 ANALYSIS END: {symbol} ===")

        return stage2_pass, signals, inflow_usd, stage1g_active

    except Exception as e:
        logger.error(f"❌ Błąd krytyczny w detect_stage_minus2_1 dla {symbol}: {e}")
        logger.debug("🔍 Traceback:\n" + traceback.format_exc())

def detect_silent_accumulation(symbol, market_data, rsi_series):
    """
    Silent Accumulation Detector - wykrywa cichy wzorzec akumulacji
    
    Kryteria:
    - RSI płaskie (45-55) przez 8 świec
    - Małe ciała świec (niska zmienność) 
    - Brak wyraźnych knotów (cegiełki)
    
    Samodzielnie wywołuje alert i analizę GPT przy wykryciu wzorca
    """
    try:
        if len(market_data) < 8 or len(rsi_series) < 8:
            return False

        last_candles = market_data[-8:]
        last_rsi = rsi_series[-8:]

        # Sprawdź RSI płaskie (45-55) przez 8 świec
        flat_rsi = all(45 <= r <= 55 for r in last_rsi if r is not None)
        
        # Sprawdź małe ciała świec (body/range < 0.3)
        small_bodies = True
        for candle in last_candles:
            try:
                open_price = float(candle['open'])
                close_price = float(candle['close'])
                high_price = float(candle['high'])
                low_price = float(candle['low'])
                
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                
                if total_range > 0:
                    body_ratio = body_size / total_range
                    if body_ratio >= 0.3:  # Jeśli ciało > 30% całości
                        small_bodies = False
                        break
                        
            except (ValueError, TypeError, KeyError):
                small_bodies = False
                break
        
        # Sprawdź brak wyraźnych knotów (upper/lower wick < 10% high/low)
        no_wicks = True
        for candle in last_candles:
            try:
                open_price = float(candle['open'])
                close_price = float(candle['close'])
                high_price = float(candle['high'])
                low_price = float(candle['low'])
                
                body_top = max(open_price, close_price)
                body_bottom = min(open_price, close_price)
                
                upper_wick = high_price - body_top
                lower_wick = body_bottom - low_price
                
                # Sprawdź czy knoty są minimalne
                if high_price > 0 and (upper_wick > 0.1 * high_price):
                    no_wicks = False
                    break
                if low_price > 0 and (lower_wick > 0.1 * low_price):
                    no_wicks = False
                    break
                    
            except (ValueError, TypeError, KeyError):
                no_wicks = False
                break

        # Jeśli wszystkie warunki spełnione - wykryto silent accumulation
        if flat_rsi and small_bodies and no_wicks:
            print(f"🔵 Silent Accumulation Triggered: {symbol}")
            
            # Przygotuj sygnały
            signals = {"silent_accumulation": True}
            ppwcs_score = 60
            
            # Import funkcji alert/GPT na poziomie modułu aby uniknąć circular imports
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                
                # Zamiast importu, zapisz dane do pliku alert cache
                alert_data = {
                    "symbol": symbol,
                    "ppwcs_score": ppwcs_score,
                    "signals": signals,
                    "detector_type": "silent_accumulation",
                    "timestamp": str(datetime.now())
                }
                
                # Zapisz alert do cache
                try:
                    import json
                    cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "silent_accumulation_alerts.json")
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    
                    existing_alerts = []
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r') as f:
                            existing_alerts = json.load(f)
                    
                    existing_alerts.append(alert_data)
                    
                    with open(cache_file, 'w') as f:
                        json.dump(existing_alerts, f, indent=2)
                    

                    
                except Exception as save_error:
                    print(f"⚠️ Error saving silent accumulation alert for {symbol}: {save_error}")
                
            except Exception as e:
                print(f"⚠️ Error in silent accumulation alert processing for {symbol}: {e}")
                # Fallback - zwróć True aby sygnał został przetworzony przez główny system
                pass
            
            return True
            
        return False

    except Exception as e:
        print(f"❌ Error in detect_silent_accumulation: {e}")
        return False

def inflow_avg(symbol):
    """Get average inflow for symbol from cache or return default"""
    try:
        import json
        import os
        cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "past_inflow_cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                if symbol in cache:
                    inflows = cache[symbol][-10:]  # Last 10 values
                    return sum(inflows) / len(inflows) if inflows else 1000.0
        return 1000.0  # Default average
    except Exception:
        return 1000.0

def detect_silent_accumulation_v1(symbol, market_data, rsi_series, orderbook=None, vwap_data=None, volume_profile=None, inflow=0, whale_txs=None, social_score=0):
    """
    Enhanced Silent Accumulation Detector v1 - wykrywa ukrytą akumulację
    
    Analizuje spokojne tokeny z subtelną presją zakupową:
    - RSI ≈ 50 przez kilka świec
    - Niskie korpusy i knoty  
    - Kurs blisko VWAP
    - Brak hype w socialu
    - ALE subtelne presje zakupowe: inflow, pinning, heatmapa, slope, whale micro-TX
    
    Wymagane: min. 5 aktywacji + min. 1 z kategorii presji zakupowej
    """
    try:
        if len(market_data) < 8 or len(rsi_series) < 8:
            return False
            
        score = 0
        explanations = []
        buying_pressure_detected = False

        # RSI ~50 (45-55 przez 8 świec) - sprawdź także zmienność
        last_rsi = rsi_series[-8:]
        if all(45 <= r <= 55 for r in last_rsi if r is not None):
            # Dodatkowo sprawdź czy RSI nie jest zbyt zmienne (max różnica < 8)
            if last_rsi:
                rsi_range = max(last_rsi) - min(last_rsi)
                if rsi_range <= 7:  # RSI musi być stabilne, nie chaotyczne
                    score += 1
                    explanations.append("RSI flat")

        # Świece z małym ciałem (body/range < 0.3)
        last_candles = market_data[-8:]
        bodies = []
        for candle in last_candles:
            try:
                open_price = float(candle["open"])
                close_price = float(candle["close"])
                high_price = float(candle["high"])
                low_price = float(candle["low"])
                
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                
                if total_range > 0:
                    body_ratio = body_size / total_range
                    bodies.append(body_ratio)
            except (ValueError, TypeError, KeyError):
                bodies.append(1.0)  # Safe fallback
                
        if bodies and all(b < 0.3 for b in bodies):
            score += 1
            explanations.append("Low body candles")

        # Minimalne knoty (upper/lower wick < 10% of total range, not price)
        minimal_wicks = True
        for candle in last_candles:
            try:
                open_price = float(candle["open"])
                close_price = float(candle["close"])
                high_price = float(candle["high"])
                low_price = float(candle["low"])
                
                total_range = high_price - low_price
                if total_range <= 0:
                    continue
                    
                body_top = max(open_price, close_price)
                body_bottom = min(open_price, close_price)
                
                upper_wick = high_price - body_top
                lower_wick = body_bottom - low_price
                
                # Check if wicks are more than 10% of total range
                if (upper_wick > 0.1 * total_range) or (lower_wick > 0.1 * total_range):
                    minimal_wicks = False
                    break
                    
            except (ValueError, TypeError, KeyError):
                minimal_wicks = False
                break
                
        if minimal_wicks:
            score += 1
            explanations.append("Minimal wicks")

        # VWAP pinning (presja zakupowa)
        if vwap_data and vwap_data.get("pinning_count", 0) >= 6:
            score += 2
            explanations.append("VWAP pinning")
            buying_pressure_detected = True

        # Volume cluster (presja zakupowa)
        if volume_profile and volume_profile.get("bullish_cluster"):
            score += 2
            explanations.append("Bullish volume cluster")
            buying_pressure_detected = True

        # Heatmapa – znika podaż (presja zakupowa)
        if orderbook and orderbook.get("supply_vanish") == True:
            score += 2
            explanations.append("Supply wall vanish (heatmap)")
            buying_pressure_detected = True

        # DEX inflow anomaly (presja zakupowa)
        avg_inflow = inflow_avg(symbol)
        if inflow > 2 * avg_inflow:
            score += 2
            explanations.append("DEX inflow anomaly")
            buying_pressure_detected = True

        # Whale TX (małe transakcje 2-4 poniżej $10k)
        whale_txs = whale_txs or []
        micro_whale_count = len([tx for tx in whale_txs if isinstance(tx, dict) and tx.get("usd", 0) < 10000])
        if 2 <= micro_whale_count <= 4:
            score += 1
            explanations.append("Micro whale activity")

        # Brak hype w socialu (niska aktywność)
        if social_score < 5:  # Niska aktywność social media
            score += 1
            explanations.append("Low social activity")

        # Sprawdź czy warunki są spełnione
        if score >= 5 and buying_pressure_detected:
            print(f"🔵 Silent Accumulation v1 detected on {symbol}: {score} pts ({', '.join(explanations)})")
            
            # Przygotuj sygnały
            signals = {
                "silent_accumulation_v1": True,
                "score": score,
                "explanations": explanations,
                "buying_pressure": buying_pressure_detected
            }
            ppwcs_score = 65
            
            # Import funkcji alert/GPT na poziomie wykonywania
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                
                # Użyj istniejących funkcji z głównego modułu
                try:
                    from utils.telegram_bot import send_alert
                    from crypto_scan_service import send_report_to_gpt
                    
                    # Przygotuj dane w formacie oczekiwanym przez funkcje
                    data = {
                        "ppwcs_score": ppwcs_score,
                        "whale_activity": "Micro whale activity" in explanations,
                        "dex_inflow": inflow if inflow > 2 * inflow_avg(symbol) else 0,
                        "compressed": False,
                        "stage1g_active": False,
                        "pure_accumulation": False,
                        "silent_accumulation_v1": True,
                        "explanations": explanations
                    }
                    
                    # Dummy TP forecast for GPT function
                    tp_forecast = {
                        "TP1": 5.0,
                        "TP2": 12.0,
                        "TP3": 25.0,
                        "TrailingTP": 35.0
                    }
                    
                    # Formatuj wiadomość alertu (plain text without markdown)
                    alert_message = f"🔵 SILENT ACCUMULATION v1 DETECTED\n\n"
                    alert_message += f"📊 Symbol: {symbol}\n"
                    alert_message += f"🎯 PPWCS Score: {ppwcs_score}\n"
                    alert_message += f"✅ Signals ({score} pts): {', '.join(explanations)}\n"
                    alert_message += f"⚡ Buying Pressure: {'Yes' if buying_pressure_detected else 'No'}\n"
                    alert_message += f"🕒 Time: {datetime.now().strftime('%H:%M UTC')}"
                    
                    # Wyślij alert
                    send_alert(alert_message)
                    
                    # Wyślij do GPT
                    send_report_to_gpt(symbol, data, tp_forecast, "Silent Accumulation v1")
                    

                    
                except ImportError:
                    # Fallback - zapisz do cache jak poprzednio
                    alert_data = {
                        "symbol": symbol,
                        "ppwcs_score": ppwcs_score,
                        "signals": signals,
                        "detector_type": "silent_accumulation_v1",
                        "timestamp": str(datetime.now()),
                        "explanations": explanations
                    }
                    
                    import json
                    cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "silent_accumulation_v1_alerts.json")
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    
                    existing_alerts = []
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r') as f:
                            existing_alerts = json.load(f)
                    
                    existing_alerts.append(alert_data)
                    
                    with open(cache_file, 'w') as f:
                        json.dump(existing_alerts, f, indent=2)
                    

                
            except Exception as e:
                print(f"⚠️ Error in Silent Accumulation v1 alert processing for {symbol}: {e}")
            
            return True
            
        return False

    except Exception as e:
        print(f"❌ Silent Accumulation v1 detection failed for {symbol}: {e}")
        return False