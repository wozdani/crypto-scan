"""
Stealth Signal Detector
Definicje i detekcja sygnałów z otoczenia rynku bez analizy wykresów

Sygnały obejmują:
- Orderbook manipulation detection
- Volume pattern analysis
- DEX inflow tracking  
- Spoofing detection
- Market microstructure patterns
"""

import asyncio
from typing import Dict, List, Optional
import statistics
import time


class StealthSignal:
    """
    Klasa reprezentująca pojedynczy sygnał stealth
    Zgodnie ze specyfikacją użytkownika
    """
    def __init__(self, name: str, active: bool, strength: float = 1.0):
        self.name = name
        self.active = active
        self.strength = strength if active else 0.0


class StealthSignalDetector:
    """
    Detektor sygnałów stealth pre-pump
    Analizuje otoczenie rynku bez polegania na wykresach
    """
    
    def __init__(self):
        """Inicjalizacja detektora sygnałów"""
        from .address_tracker import AddressTracker
        self.address_tracker = AddressTracker()
        
        self.signal_definitions = {
            # ORDERBOOK SIGNALS
            'orderbook_imbalance': {
                'description': 'Znacząca asymetria bid/ask w książce zleceń',
                'category': 'orderbook',
                'weight_default': 0.15
            },
            'large_bid_walls': {
                'description': 'Duże mury bid wspierające cenę',
                'category': 'orderbook', 
                'weight_default': 0.12
            },
            'ask_wall_removal': {
                'description': 'Usunięcie dużych murów ask',
                'category': 'orderbook',
                'weight_default': 0.18
            },
            'spoofing_detected': {
                'description': 'Wykrycie manipulacji spoofing',
                'category': 'manipulation',
                'weight_default': -0.25  # Negatywny sygnał
            },
            
            # VOLUME SIGNALS
            'volume_spike': {
                'description': 'Nagły wzrost wolumenu',
                'category': 'volume',
                'weight_default': 0.20
            },
            'volume_accumulation': {
                'description': 'Stopniowa akumulacja wolumenu',
                'category': 'volume',
                'weight_default': 0.14
            },
            'unusual_volume_profile': {
                'description': 'Nietypowy profil wolumenu na poziomach cen',
                'category': 'volume',
                'weight_default': 0.16
            },
            
            # DEX SIGNALS
            'dex_inflow_spike': {
                'description': 'Zwiększony napływ do DEX',
                'category': 'dex',
                'weight_default': 0.22
            },
            'whale_accumulation': {
                'description': 'Akumulacja przez wieloryby',
                'category': 'whale',
                'weight_default': 0.19
            },
            
            # MICROSTRUCTURE SIGNALS
            'bid_ask_spread_tightening': {
                'description': 'Zawężenie spread bid-ask',
                'category': 'microstructure',
                'weight_default': 0.10
            },
            'order_flow_pressure': {
                'description': 'Presja w przepływie zleceń',
                'category': 'microstructure',
                'weight_default': 0.13
            },
            'liquidity_absorption': {
                'description': 'Absorpcja płynności na kluczowych poziomach',
                'category': 'liquidity',
                'weight_default': 0.17
            },
            'repeated_address_boost': {
                'description': 'Boost za powtarzające się adresy w sygnałach (+0.2 per adres, max +0.6)',
                'category': 'accumulation',
                'weight_default': 0.25
            },
            'velocity_boost': {
                'description': 'Boost za szybkie sekwencje aktywności adresów (velocity tracking)',
                'category': 'accumulation',
                'weight_default': 0.18
            },
            'inflow_momentum_boost': {
                'description': 'Boost za przyspieszającą aktywność adresów (momentum inflow)',
                'category': 'accumulation',
                'weight_default': 0.15
            },
            'source_reliability_boost': {
                'description': 'Boost za adresy o wysokiej reputacji (smart money)',
                'category': 'accumulation',
                'weight_default': 0.12
            },
            'cross_token_activity_boost': {
                'description': 'Boost za korelację adresów między różnymi tokenami',
                'category': 'accumulation',
                'weight_default': 0.12
            },
            'multi_address_group_activity': {
                'description': 'Detekcja skoordynowanej aktywności grup adresów (3+ adresy w 72h)',
                'category': 'accumulation',
                'weight_default': 0.15
            }
        }
        
        print(f"[STEALTH SIGNALS] Initialized {len(self.signal_definitions)} signal definitions")
    
    def get_active_stealth_signals(self, token_data: Dict) -> List[StealthSignal]:
        """
        Funkcja główna wykrywająca aktywne sygnały stealth
        Zgodnie ze specyfikacją użytkownika
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        print(f"[DEBUG FLOW] {symbol} - get_active_stealth_signals() starting signal detection...")
        signals = []
        
        # Lista funkcji sygnałów do sprawdzenia z error handling
        signal_functions = [
            ("whale_ping", self.check_whale_ping),
            ("spoofing_layers", self.check_spoofing_layers), 
            ("dex_inflow", self.check_dex_inflow),
            ("orderbook_anomaly", self.check_orderbook_anomaly),
            ("volume_slope", self.check_volume_slope),
            ("ghost_orders", self.check_ghost_orders),
            ("event_tag", self.check_event_tag),
            ("orderbook_imbalance_stealth", self.check_orderbook_imbalance_stealth),
            ("large_bid_walls_stealth", self.check_large_bid_walls_stealth),
            ("ask_wall_removal", self.check_ask_wall_removal),
            ("volume_spike_stealth", self.check_volume_spike_stealth),
            ("bid_ask_spread_tightening_stealth", self.check_bid_ask_spread_tightening_stealth),
            ("liquidity_absorption", self.check_liquidity_absorption),
            ("repeated_address_boost", self.check_repeated_address_boost),
            ("velocity_boost", self.check_velocity_boost),
            ("inflow_momentum_boost", self.check_inflow_momentum_boost),
            ("source_reliability_boost", self.check_source_reliability_boost),
            ("cross_token_activity_boost", self.check_cross_token_activity_boost),
            ("multi_address_group_activity", self.check_multi_address_group_activity)
        ]
        
        print(f"[DEBUG FLOW] {symbol} - processing {len(signal_functions)} signal functions...")
        
        # PERFORMANCE OPTIMIZED: Sprawdź każdy sygnał z obsługą błędów i ograniczonymi logami
        for i, (signal_name, signal_func) in enumerate(signal_functions):
            try:
                signal = signal_func(token_data)
                
                if signal is not None:
                    signals.append(signal)
                else:
                    signals.append(StealthSignal(signal_name, False, 0.0))
            except Exception as e:
                print(f"[STEALTH ERROR] {symbol}: Failed to check {signal_name}: {e}")
                # Dodaj pustý sygnał aby utrzymać spójność
                signals.append(StealthSignal(signal_name, False, 0.0))
        
        print(f"[DEBUG FLOW] {symbol} - get_active_stealth_signals() completed with {len(signals)} signals")
        return signals
    
    def get_dynamic_whale_threshold(self, orderbook: dict) -> float:
        """
        Oblicza dynamiczny próg detekcji whale_ping w USD
        na podstawie mediany wielkości zleceń w orderbooku.
        
        Returns:
            float: Dynamiczny próg w USD (mediana_wielkości_zlecenia × 20)
        """
        try:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            # Zbierz wszystkie wielkości zleceń (size w USD)
            sizes_usd = []
            
            # Extract sizes from bids (convert to USD value)
            for bid in bids:
                try:
                    if isinstance(bid, dict) and 'price' in bid and 'size' in bid:
                        price = float(bid['price'])
                        size = float(bid['size'])
                        sizes_usd.append(price * size)
                    elif isinstance(bid, (list, tuple)) and len(bid) >= 2:
                        price = float(bid[0])
                        size = float(bid[1])
                        sizes_usd.append(price * size)
                except (ValueError, TypeError, IndexError):
                    continue
            
            # Extract sizes from asks (convert to USD value)
            for ask in asks:
                try:
                    if isinstance(ask, dict) and 'price' in ask and 'size' in ask:
                        price = float(ask['price'])
                        size = float(ask['size'])
                        sizes_usd.append(price * size)
                    elif isinstance(ask, (list, tuple)) and len(ask) >= 2:
                        price = float(ask[0])
                        size = float(ask[1])
                        sizes_usd.append(price * size)
                except (ValueError, TypeError, IndexError):
                    continue
            
            if not sizes_usd:
                return 50_000  # fallback na 50k USD
            
            # Oblicz medianę wielkości zleceń w USD
            sorted_sizes = sorted(sizes_usd)
            median_size_usd = sorted_sizes[len(sorted_sizes) // 2]
            
            # Próg = mediana × 20 (mnożnik do wykrycia dużych zleceń)
            dynamic_threshold = median_size_usd * 20
            
            # Minimalne zabezpieczenie - nie mniej niż $5k
            return max(dynamic_threshold, 5_000)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] get_dynamic_whale_threshold error: {e}")
            return 50_000

    def check_whale_ping(self, token_data: Dict) -> StealthSignal:
        """
        Whale ping detector - wykrycie wielorybów przez duże zlecenia
        Dynamiczna wersja dopasowana do wolumenu tokena
        """
        FUNC_NAME = "whale_ping"
        symbol = token_data.get("symbol", "UNKNOWN")
        
        try:
            orderbook = token_data.get("orderbook", {})
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            # Pobierz średni wolumen 15m dla dynamicznego progu
            candles_15m = token_data.get("candles_15m", [])
            if candles_15m and len(candles_15m) > 0:
                # Oblicz średni wolumen z ostatnich 15m świec
                volumes = []
                for candle in candles_15m[-8:]:  # ostatnie 8 świec
                    if isinstance(candle, dict) and "volume" in candle:
                        volumes.append(float(candle["volume"]))
                    elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                        volumes.append(float(candle[5]))  # volume jest na pozycji 5
                avg_volume_15m = sum(volumes) / len(volumes) if volumes else 0
            else:
                avg_volume_15m = 0
            
            if not bids or not asks:
                print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → bids={len(bids)}, asks={len(asks)}, insufficient_data=True")
                return StealthSignal("whale_ping", False, 0.0)
            
            # Handle different orderbook formats (list vs dict)
            if isinstance(bids, dict):
                try:
                    bids_list = []
                    for key in sorted(bids.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0, reverse=True):
                        if isinstance(bids[key], list) and len(bids[key]) >= 2:
                            bids_list.append(bids[key])
                    bids = bids_list if bids_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] whale_ping bids conversion error for {symbol}: {e}")
                    bids = []
            
            if isinstance(asks, dict):
                try:
                    asks_list = []
                    for key in sorted(asks.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0):
                        if isinstance(asks[key], list) and len(asks[key]) >= 2:
                            asks_list.append(asks[key])
                    asks = asks_list if asks_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] whale_ping asks conversion error for {symbol}: {e}")
                    asks = []
            
            # Handle case where bids/asks are lists of dicts with 'price' and 'size' keys
            if isinstance(bids, list) and len(bids) > 0 and isinstance(bids[0], dict):
                try:
                    bids_list = []
                    for bid in bids:
                        if isinstance(bid, dict) and 'price' in bid and 'size' in bid:
                            bids_list.append([bid['price'], bid['size']])
                        elif isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            bids_list.append(bid)
                    bids = bids_list if bids_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] whale_ping bids dict-to-list conversion error for {symbol}: {e}")
                    bids = []
            
            if isinstance(asks, list) and len(asks) > 0 and isinstance(asks[0], dict):
                try:
                    asks_list = []
                    for ask in asks:
                        if isinstance(ask, dict) and 'price' in ask and 'size' in ask:
                            asks_list.append([ask['price'], ask['size']])
                        elif isinstance(ask, (list, tuple)) and len(ask) >= 2:
                            asks_list.append(ask)
                    asks = asks_list if asks_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] whale_ping asks dict-to-list conversion error for {symbol}: {e}")
                    asks = []
            
            # Verify we have valid orderbook data after conversion
            if not bids or not asks:
                print(f"[STEALTH DEBUG] whale_ping for {symbol}: no valid bids/asks after conversion")
                return StealthSignal("whale_ping", False, 0.0)
            
            # Additional validation for data structure after conversion
            if not isinstance(bids[0], (list, tuple)) or len(bids[0]) < 2:
                print(f"[STEALTH DEBUG] whale_ping for {symbol}: invalid bid structure after conversion: {type(bids[0])}, content: {bids[0]}")
                return StealthSignal("whale_ping", False, 0.0)
            
            if not isinstance(asks[0], (list, tuple)) or len(asks[0]) < 2:
                print(f"[STEALTH DEBUG] whale_ping for {symbol}: invalid ask structure after conversion: {type(asks[0])}, content: {asks[0]}")
                return StealthSignal("whale_ping", False, 0.0)
            
            # Oblicz dynamiczny próg na podstawie mediany wielkości zleceń
            threshold = self.get_dynamic_whale_threshold(orderbook)
            
            # === MICROCAP ADAPTATION ===
            # For tokens with very small max orders, adapt threshold
            max_order_check = 0
            for bid in bids:
                try:
                    price = float(bid[0])
                    size = float(bid[1])
                    usd_value = price * size
                    max_order_check = max(max_order_check, usd_value)
                except:
                    pass
            for ask in asks:
                try:
                    price = float(ask[0])
                    size = float(ask[1])
                    usd_value = price * size
                    max_order_check = max(max_order_check, usd_value)
                except:
                    pass
            
            if max_order_check < 200:  # Microcap token detection
                # Use proportional threshold for micro tokens
                adapted_threshold = max(50, max_order_check * 4)
                print(f"[MICROCAP THRESHOLD] {symbol}: max_order=${max_order_check:.2f} → adapted threshold=${adapted_threshold:.2f} (was ${threshold:.2f})")
                threshold = adapted_threshold
            
            # Oblicz wszystkie zlecenia USD
            all_orders = []
            for bid in bids:
                try:
                    price = float(bid[0])
                    size = float(bid[1])
                    usd_value = price * size
                    all_orders.append({"price": price, "size": size, "usd_value": usd_value})
                except (ValueError, TypeError, IndexError):
                    continue
            
            for ask in asks:
                try:
                    price = float(ask[0])
                    size = float(ask[1])
                    usd_value = price * size
                    all_orders.append({"price": price, "size": size, "usd_value": usd_value})
                except (ValueError, TypeError, IndexError):
                    continue
            
            # Znajdź największe zlecenie w USD
            max_order_usd = max([o["usd_value"] for o in all_orders], default=0)
            
            # INPUT LOG - wszystkie dane wejściowe
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → max_order_usd=${max_order_usd:.2f}, threshold=${threshold:.2f}, bids={len(bids)}, asks={len(asks)}")
            
            # Warunek aktywacji z dynamicznym progiem
            active = max_order_usd > threshold
            
            # Strength: max_order / (threshold * 2)
            strength = min(max_order_usd / (threshold * 2), 1.0) if threshold > 0 else 0.0
            
            # MID LOG - kluczowe obliczenia pośrednie
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] MID → ratio={max_order_usd/threshold:.3f}, initial_strength={strength:.3f}")
            
            # 🔧 FIX: Microcap bonus przy niskim spread
            # Dla tokenów z bardzo niskim spread (<0.3%) i małym max_order_usd, daj bonus aktywacyjny
            if max_order_usd < 50 and len(bids) > 0 and len(asks) > 0:
                try:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    spread_pct = (best_ask - best_bid) / best_bid * 100
                    
                    if spread_pct < 0.3:  # Bardzo niski spread
                        # Aktywuj whale_ping z bonusem
                        active = True
                        strength = max(strength, 0.15)  # Minimum strength dla tight spread
                        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] MICROCAP BONUS → spread={spread_pct:.3f}% <0.3%, activated with strength={strength:.3f}")
                except:
                    pass
            
            # REAL WHALE ADDRESS DETECTION - blockchain-based whale identification
            if active and max_order_usd > 0:
                try:
                    # Import real blockchain scanner for whale detection
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    parent_dir = os.path.dirname(current_dir)
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    
                    from utils.blockchain_scanners import get_whale_transfers
                    from utils.contracts import get_contract
                    from .address_tracker import AddressTracker
                    
                    # Get real whale transfers from blockchain
                    contract_info = get_contract(symbol)
                    real_whale_addresses = []
                    
                    if contract_info:
                        # Get real whale transfers using blockchain API
                        whale_transfers = get_whale_transfers(symbol, min_usd=threshold)
                        real_whale_addresses = [t['from'] for t in whale_transfers if t['value_usd'] >= threshold][:10]
                        
                        # Track REAL addresses from blockchain
                        tracker = AddressTracker()
                        for real_address in real_whale_addresses:
                            tracker.record_address_activity(
                                token=symbol,
                                address=real_address,  # REAL ADDRESS from blockchain
                                usd_value=max_order_usd,
                                source="whale_ping_real"
                            )
                        
                        print(f"[STEALTH DEBUG] whale_ping real address tracking for {symbol}: {len(real_whale_addresses)} real whale addresses")
                    
                    # If no real addresses found, skip advanced tracking
                    if not real_whale_addresses:
                        print(f"[STEALTH DEBUG] whale_ping for {symbol}: no real whale addresses found, using orderbook detection only")
                
                except Exception as addr_e:
                    print(f"[STEALTH DEBUG] whale_ping real address tracking error for {symbol}: {addr_e}")
                    real_whale_addresses = []
                
                # REAL WHALE MEMORY SYSTEM - using blockchain addresses
                try:
                    from utils.whale_memory import update_whale_memory, is_repeat_whale, get_repeat_whale_boost
                    from stealth_engine.address_trust_manager import record_address_prediction, get_address_boost
                    
                    # Use real addresses if available, otherwise fall back to orderbook-based detection
                    addresses_to_process = real_whale_addresses if real_whale_addresses else []
                    
                    total_repeat_boost = 0.0
                    repeat_whales_found = 0
                    
                    # Process real whale addresses from blockchain
                    for real_address in addresses_to_process[:5]:  # Top 5 real addresses
                        # Aktualizuj pamięć wieloryba z RZECZYWISTYM adresem
                        repeat_count = update_whale_memory(symbol, real_address, source="whale_ping_real")
                        
                        # Sprawdź czy to powtarzający się wieloryb
                        if is_repeat_whale(symbol, real_address):
                            repeat_boost = get_repeat_whale_boost(symbol, real_address)
                            total_repeat_boost += repeat_boost * 0.3  # Max 30% boost per address
                            repeat_whales_found += 1
                    
                    # Apply combined boost (limited to 0.8 max)
                    if total_repeat_boost > 0:
                        final_repeat_boost = min(0.8, total_repeat_boost)
                        strength = min(1.0, strength + final_repeat_boost)
                        print(f"[WHALE MEMORY] {symbol} repeat whales detected! Count: {repeat_whales_found}, Total boost: +{final_repeat_boost:.3f} → strength: {strength:.3f}")
                        
                        # Etap 4: Zwiększ priorytet tokena w kolejce skanowania
                        try:
                            from utils.token_priority_manager import update_token_priority
                            priority_boost = 10 + (final_repeat_boost * 25)  # Enhanced priority for real addresses
                            update_token_priority(symbol, priority_boost, "whale_ping_real_repeat")
                        except Exception as priority_e:
                            print(f"[TOKEN PRIORITY] Error updating whale_ping priority for {symbol}: {priority_e}")
                    
                    # Etap 5: Rejestruj aktywność adresu dla multi-address detection
                    try:
                        from stealth_engine.multi_address_detector import record_address_activity
                        for real_address in addresses_to_process[:10]:
                            record_address_activity(symbol, real_address)
                        print(f"[MULTI-ADDRESS] Recorded whale_ping activity for {symbol}: {len(addresses_to_process)} real addresses")
                    except Exception as multi_e:
                        print(f"[MULTI-ADDRESS ERROR] whale_ping for {symbol}: {multi_e}")
                    
                    # Etap 6: Trust Scoring z RZECZYWISTYMI adresami
                    try:
                        total_trust_boost = 0.0
                        for real_address in addresses_to_process[:5]:
                            # Rejestruj predykcję rzeczywistego adresu
                            record_address_prediction(symbol, real_address)
                            
                            # Pobierz trust boost na podstawie historycznej skuteczności
                            trust_boost = get_address_boost(real_address)
                            if trust_boost > 0:
                                total_trust_boost += trust_boost
                        
                        if total_trust_boost > 0:
                            final_trust_boost = min(0.5, total_trust_boost)  # Cap at 0.5
                            strength = min(1.0, strength + final_trust_boost)
                            print(f"[TRUST BOOST] {symbol} whale_ping: Applied +{final_trust_boost:.3f} trust boost from real addresses → strength: {strength:.3f}")
                        
                    except Exception as trust_e:
                        print(f"[TRUST BOOST ERROR] whale_ping for {symbol}: {trust_e}")
                    
                    # Stage 13: Token Trust Score z RZECZYWISTYMI adresami
                    try:
                        from stealth_engine.utils.trust_tracker import update_token_trust, compute_trust_boost
                        
                        # Oblicz trust boost na podstawie historii RZECZYWISTYCH adresów
                        token_trust_boost = compute_trust_boost(symbol, addresses_to_process[:10])
                        if token_trust_boost > 0:
                            strength = min(1.0, strength + token_trust_boost)
                            print(f"[TOKEN TRUST] {symbol} whale_ping: Applied +{token_trust_boost:.3f} token trust boost from real addresses → strength: {strength:.3f}")
                        
                        # Aktualizuj historię trust z RZECZYWISTYMI adresami
                        update_token_trust(symbol, addresses_to_process[:10], "whale_ping_real")
                        
                    except Exception as token_trust_e:
                        print(f"[TOKEN TRUST ERROR] whale_ping for {symbol}: {token_trust_e}")
                    
                    # Stage 14: Persistent Identity Scoring z RZECZYWISTYMI portfelami
                    try:
                        print(f"[DEBUG WHALE_PING] {symbol} - Starting identity boost calculation...")
                        from stealth_engine.utils.identity_tracker import get_identity_boost, update_wallet_identity
                        
                        # SAFETY LIMIT: Maksymalnie 10 adresów aby zapobiec zawieszeniu
                        safe_addresses = addresses_to_process[:10]
                        print(f"[DEBUG WHALE_PING] {symbol} - Processing {len(safe_addresses)} addresses for identity boost")
                        
                        # Oblicz identity boost na podstawie RZECZYWISTYCH portfeli
                        identity_boost = get_identity_boost(safe_addresses)
                        print(f"[DEBUG WHALE_PING] {symbol} - Identity boost calculated: {identity_boost:.3f}")
                        
                        if identity_boost > 0:
                            strength = min(1.0, strength + identity_boost)
                            print(f"[IDENTITY BOOST] {symbol} whale_ping: Applied +{identity_boost:.3f} identity boost from real wallets → strength: {strength:.3f}")
                        
                        print(f"[DEBUG WHALE_PING] {symbol} - Identity boost calculation completed")
                        
                    except Exception as identity_e:
                        print(f"[IDENTITY BOOST ERROR] whale_ping for {symbol}: {identity_e}")
                        print(f"[DEBUG WHALE_PING] {symbol} - Identity boost calculation failed, continuing...")
                    
                    # Etap 7: Trigger Alert System z RZECZYWISTYMI adresami
                    try:
                        from stealth_engine.trigger_alert_system import check_smart_money_trigger, apply_smart_money_boost
                        from stealth_engine.address_trust_manager import get_trust_manager
                        
                        # Sprawdź czy wykryto trigger addresses (smart money) wśród RZECZYWISTYCH adresów
                        trust_manager = get_trust_manager()
                        is_trigger, trigger_addresses = check_smart_money_trigger(addresses_to_process[:10], trust_manager)
                        
                        if is_trigger:
                            # Zastosuj trigger boost - minimum score 3.0 dla instant alert
                            boosted_strength, priority_alert = apply_smart_money_boost(
                                symbol, strength, trigger_addresses, "whale_ping_real"
                            )
                            strength = boosted_strength
                            
                            print(f"[TRIGGER ALERT] 🚨 {symbol} WHALE PING: Smart money detected! "
                                  f"Strength boosted to {strength:.3f} (priority alert: {priority_alert})")
                        
                    except Exception as trigger_e:
                        print(f"[TRIGGER ALERT ERROR] whale_ping for {symbol}: {trigger_e}")
                    
                except Exception as memory_e:
                    print(f"[WHALE MEMORY] Error for {symbol}: {memory_e}")
            
            # RESULT LOG - końcowa decyzja i siła sygnału
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] RESULT → active={active}, strength={strength:.3f}")
            
            return StealthSignal("whale_ping", active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] ERROR → {type(e).__name__}: {e}")
            return StealthSignal("whale_ping", False, 0.0)

    
    def check_spoofing_layers(self, token_data: Dict) -> StealthSignal:
        """
        Spoofing layers detector - detekcja warstwowania zleceń
        Matematycznie precyzyjna implementacja zgodna z user specification v2
        """
        FUNC_NAME = "spoofing_layers"
        symbol = token_data.get("symbol", "UNKNOWN")
        orderbook = token_data.get("orderbook", {})
        
        if not orderbook:
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → orderbook=None, insufficient_data=True")
            return StealthSignal("spoofing_layers", False, 0.0)
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        # INPUT LOG - dane wejściowe
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → bids={len(bids)}, asks={len(asks)}")
        
        # === MICROCAP ADAPTATION ===
        # For microcap tokens, use relaxed requirements
        min_levels_required = 1 if len(bids) <= 1 and len(asks) <= 1 else 3
        
        if len(bids) < min_levels_required and len(asks) < min_levels_required:
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] RESULT → active=False, strength=0.000 (insufficient_levels)")
            return StealthSignal("spoofing_layers", False, 0.0)
        
        try:
            def analyze_side_spoofing(orders, side_name):
                if len(orders) < 1:
                    return False, 0.0, 0.0, 0.0
                
                # === SAFE ORDERBOOK PARSING ===
                # Handle different orderbook formats (list vs dict)
                parsed_orders = []
                
                for order in orders:
                    try:
                        if isinstance(order, list) and len(order) >= 2:
                            # Format: [price, volume]
                            price = float(order[0])
                            volume = float(order[1])
                            parsed_orders.append((price, volume))
                        elif isinstance(order, dict):
                            # Format: {'price': X, 'size': Y}
                            price = float(order.get('price', 0))
                            volume = float(order.get('size', 0))
                            parsed_orders.append((price, volume))
                        else:
                            continue
                    except (ValueError, TypeError, KeyError, IndexError):
                        continue
                
                if len(parsed_orders) < 3:
                    return False, 0.0, 0.0, 0.0
                
                # Oblicz total volume dla tej strony
                total_side_volume = sum(volume for price, volume in parsed_orders)
                layers_volume = 0.0
                layer_count = 0
                
                # Sprawdź pierwsze 10 poziomów
                for i in range(min(len(parsed_orders), 10)):
                    if i == 0:
                        continue  # Skip first level
                    
                    base_price, _ = parsed_orders[0]
                    current_price, current_volume = parsed_orders[i]
                    
                    # Sprawdź odległość <0.2% między sobą
                    price_diff_pct = abs(current_price - base_price) / base_price * 100
                    
                    if price_diff_pct < 0.2:
                        # Sprawdź czy wolumen stanowi >5% całkowitego bid/ask sum
                        volume_pct = current_volume / total_side_volume
                        if volume_pct > 0.05:
                            layers_volume += current_volume
                            layer_count += 1
                    else:
                        break
                
                # Warunek aktywacji: co najmniej 3 bidy lub aski w odległości <0.2%
                is_spoofing = layer_count >= 3
                
                # Strength: min(1.0, (layers_volume / total_side_volume))
                strength = min(1.0, layers_volume / total_side_volume) if total_side_volume > 0 else 0.0
                
                return is_spoofing, strength, layers_volume, total_side_volume
            
            # Analizuj obie strony
            bid_spoofing, bid_strength, bid_layers_vol, bid_total_vol = analyze_side_spoofing(bids, "bids")
            ask_spoofing, ask_strength, ask_layers_vol, ask_total_vol = analyze_side_spoofing(asks, "asks")
            
            # MID LOG - kluczowe obliczenia pośrednie
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] MID → bid_spoofing={bid_spoofing}, ask_spoofing={ask_spoofing}, bid_strength={bid_strength:.3f}, ask_strength={ask_strength:.3f}")
            
            # Aktywacja jeśli którakolwiek strona ma spoofing
            is_active = bid_spoofing or ask_spoofing
            
            # Wybierz wyższą strength
            strength = max(bid_strength, ask_strength)
            
            # RESULT LOG - końcowa decyzja i siła sygnału
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] RESULT → active={is_active}, strength={strength:.3f}")
            
            return StealthSignal("spoofing_layers", is_active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] ERROR → {type(e).__name__}: {e}")
            return StealthSignal("spoofing_layers", False, 0.0)
    
    def check_volume_slope(self, token_data: Dict) -> StealthSignal:
        """
        Wolumen rosnący bez zmiany ceny
        """
        FUNC_NAME = "volume_slope"
        symbol = token_data.get("symbol", "UNKNOWN")
        slope = token_data.get("volume_slope_up", False)
        
        # INPUT LOG
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → volume_slope_up={slope}")
        
        strength = 1.0 if slope else 0.0
        
        # RESULT LOG
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] RESULT → active={slope}, strength={strength:.3f}")
        
        return StealthSignal("volume_slope", slope, strength)
    
    def check_ghost_orders(self, token_data: Dict) -> StealthSignal:
        """
        Martwe poziomy z nietypową aktywnością
        """
        FUNC_NAME = "ghost_orders"
        symbol = token_data.get("symbol", "UNKNOWN")
        ghost = token_data.get("ghost_orders", False)
        
        # INPUT LOG
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → ghost_orders={ghost}")
        
        strength = 1.0 if ghost else 0.0
        
        # RESULT LOG
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] RESULT → active={ghost}, strength={strength:.3f}")
        
        return StealthSignal("ghost_orders", ghost, strength)
    
    def check_dex_inflow(self, token_data: Dict) -> StealthSignal:
        """
        DEX inflow detector - REAL BLOCKCHAIN DATA detection of DEX inflows
        Uses authentic blockchain transfer data instead of mock addresses
        """
        FUNC_NAME = "dex_inflow"
        symbol = token_data.get("symbol", "UNKNOWN")
        
        try:
            # Import real blockchain scanner
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from utils.blockchain_scanners import get_token_transfers_last_24h, load_known_exchange_addresses
            from utils.contracts import get_contract
            
            # Get real contract info for the token
            contract_info = get_contract(symbol)
            if not contract_info:
                # Fallback to legacy inflow data if no contract found
                inflow_usd = token_data.get("dex_inflow", 0)
                inflow_history = token_data.get("dex_inflow_history", [])[-8:]
                avg_recent = sum(inflow_history) / len(inflow_history) if inflow_history else 0
                
                print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → fallback_mode, inflow_usd=${inflow_usd:.2f}, avg_recent=${avg_recent:.2f}")
                
                spike_detected = inflow_usd > avg_recent * 2 and inflow_usd > 1000
                strength = min(inflow_usd / (avg_recent * 5 + 1), 0.8) if avg_recent > 0 else 0.0
                
                print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] RESULT → active={spike_detected}, strength={strength:.3f} (fallback)")
                return StealthSignal("dex_inflow", spike_detected, strength)
            
            # Get real blockchain transfers in last 24h
            real_transfers = get_token_transfers_last_24h(
                symbol=symbol,
                chain=contract_info['chain'],
                contract_address=contract_info['address']
            )
            
            # Load known exchange addresses
            known_exchanges = load_known_exchange_addresses()
            exchange_addresses = known_exchanges.get(contract_info['chain'], [])
            dex_routers = known_exchanges.get('dex_routers', {}).get(contract_info['chain'], [])
            all_known_addresses = set(addr.lower() for addr in exchange_addresses + dex_routers)
            
            # Calculate real DEX inflow from blockchain data
            total_inflow_usd = 0
            inflow_usd = 0  # CRITICAL FIX: Initialize inflow_usd variable
            real_addresses = []
            
            for transfer in real_transfers:
                # Check if transfer is TO a known exchange/DEX address
                if transfer['to'] in all_known_addresses:
                    total_inflow_usd += transfer['value_usd']
                    if transfer['from'] not in real_addresses:
                        real_addresses.append(transfer['from'])
            
            # Calculate historical average (use recent data as baseline)
            historical_baseline = token_data.get("dex_inflow_history", [])[-8:]
            avg_recent = sum(historical_baseline) / len(historical_baseline) if historical_baseline else 1000
            
            # INPUT LOG - real blockchain data
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → real_transfers={len(real_transfers)}, total_inflow_usd=${total_inflow_usd:.2f}, unique_addresses={len(real_addresses)}, avg_baseline=${avg_recent:.2f}")
            
            # CRITICAL FIX: Set inflow_usd for metadata consistency
            inflow_usd = total_inflow_usd
            
            # Enhanced detection logic with real data
            spike_detected = total_inflow_usd > max(avg_recent * 2, 2000) and len(real_addresses) >= 2
            
            # Strength calculation based on real inflow magnitude
            strength = min(total_inflow_usd / (avg_recent * 3 + 1), 0.8) if avg_recent > 0 else 0.0
            
            # Bonus for multiple unique addresses (sign of coordinated activity)
            if len(real_addresses) >= 5:
                strength = min(1.0, strength + 0.2)
            elif len(real_addresses) >= 3:
                strength = min(1.0, strength + 0.1)
            
            # MID LOG - calculation details
            ratio = total_inflow_usd / (avg_recent * 2) if avg_recent > 0 else 0
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] MID → spike_ratio={ratio:.3f}, address_count={len(real_addresses)}, enhanced_strength={strength:.3f}")
            
            # Address tracking with REAL addresses from blockchain
            if spike_detected and real_addresses:
                try:
                    from .address_tracker import AddressTracker
                    tracker = AddressTracker()
                    
                    # Record each real address that contributed to the inflow
                    for real_address in real_addresses[:10]:  # Limit to top 10 addresses
                        tracker.record_address_activity(
                            token=symbol,
                            address=real_address,  # REAL ADDRESS from blockchain
                            usd_value=total_inflow_usd / len(real_addresses),  # Distributed value
                            source="dex_inflow_real"
                        )
                        
                except Exception as addr_e:
                    print(f"[STEALTH DEBUG] dex_inflow real address tracking error for {symbol}: {addr_e}")
                
                # REAL WHALE MEMORY SYSTEM - używamy prawdziwych adresów z blockchain
                try:
                    from utils.whale_memory import update_whale_memory, is_repeat_whale, get_repeat_whale_boost
                    from stealth_engine.address_trust_manager import record_address_prediction, get_address_boost
                    
                    total_repeat_boost = 0.0
                    repeat_whales_found = 0
                    
                    # Analizuj każdy rzeczywisty adres z blockchain
                    for real_address in real_addresses[:5]:  # Top 5 addresses
                        # Aktualizuj pamięć wieloryba z RZECZYWISTYM adresem
                        repeat_count = update_whale_memory(symbol, real_address, source="dex_inflow_real")
                        
                        # Sprawdź czy to powtarzający się wieloryb
                        if is_repeat_whale(symbol, real_address):
                            repeat_boost = get_repeat_whale_boost(symbol, real_address)
                            total_repeat_boost += repeat_boost * 0.25  # Max 25% boost dla DEX per address
                            repeat_whales_found += 1
                    
                    # Zastosuj łączny boost (ale ograniczony do 0.6 max)
                    if total_repeat_boost > 0:
                        final_repeat_boost = min(0.6, total_repeat_boost)
                        strength = min(1.0, strength + final_repeat_boost)
                        print(f"[WHALE MEMORY] {symbol} repeat DEX whales detected! Count: {repeat_whales_found}, Total boost: +{final_repeat_boost:.3f} → strength: {strength:.3f}")
                        
                        # Etap 4: Zwiększ priorytet tokena w kolejce skanowania
                        try:
                            from utils.token_priority_manager import update_token_priority
                            priority_boost = 8 + (final_repeat_boost * 20)  # Enhanced priority for real addresses
                            update_token_priority(symbol, priority_boost, "dex_inflow_real_repeat")
                        except Exception as priority_e:
                            print(f"[TOKEN PRIORITY] Error updating dex_inflow priority for {symbol}: {priority_e}")
                    
                    # Etap 5: Rejestruj aktywność adresu dla multi-address detection
                    try:
                        from stealth_engine.multi_address_detector import record_address_activity
                        for real_address in real_addresses[:10]:
                            record_address_activity(symbol, real_address)
                        print(f"[MULTI-ADDRESS] Recorded dex_inflow activity for {symbol}: {len(real_addresses)} real addresses")
                    except Exception as multi_e:
                        print(f"[MULTI-ADDRESS ERROR] dex_inflow for {symbol}: {multi_e}")
                    
                    # Etap 6: Trust Scoring z RZECZYWISTYMI adresami
                    try:
                        total_trust_boost = 0.0
                        for real_address in real_addresses[:5]:
                            # Rejestruj predykcję rzeczywistego adresu
                            record_address_prediction(symbol, real_address)
                            
                            # Pobierz trust boost na podstawie historycznej skuteczności
                            trust_boost = get_address_boost(real_address)
                            if trust_boost > 0:
                                total_trust_boost += trust_boost
                        
                        if total_trust_boost > 0:
                            final_trust_boost = min(0.4, total_trust_boost)  # Cap at 0.4
                            strength = min(1.0, strength + final_trust_boost)
                            print(f"[TRUST BOOST] {symbol} dex_inflow: Applied +{final_trust_boost:.3f} trust boost from real addresses → strength: {strength:.3f}")
                        
                    except Exception as trust_e:
                        print(f"[TRUST BOOST ERROR] dex_inflow for {symbol}: {trust_e}")
                    
                    # Stage 13: Token Trust Score - WITH TIMEOUT PROTECTION
                    try:
                        print(f"[DEBUG TOKEN_TRUST] {symbol} - Starting token trust calculation with timeout...")
                        
                        # Import with timeout protection
                        import signal
                        
                        def token_trust_timeout_handler(signum, frame):
                            raise TimeoutError("token_trust timeout")
                        
                        try:
                            # Set 1-second timeout for token trust
                            signal.signal(signal.SIGALRM, token_trust_timeout_handler)
                            signal.alarm(1)
                            
                            from .token_trust_tracker import update_token_trust, compute_trust_boost
                            trust_boost = compute_trust_boost(symbol, real_addresses[:3])  # Top 3 addresses only
                            
                            signal.alarm(0)  # Cancel timeout
                            
                            if trust_boost > 0:
                                strength = min(1.0, strength + trust_boost)
                                print(f"[TOKEN TRUST] {symbol} → trust boost +{trust_boost:.3f} → strength: {strength:.3f}")
                            else:
                                print(f"[TOKEN TRUST] {symbol} → no trust boost applied")
                                
                        except TimeoutError:
                            signal.alarm(0)
                            print(f"[TOKEN TRUST TIMEOUT] {symbol} - using emergency skip (1s timeout)")
                        
                        print(f"[DEBUG FLOW] {symbol} - Token trust completed, proceeding...")
                        
                    except Exception as token_trust_e:
                        print(f"[TOKEN TRUST ERROR] dex_inflow for {symbol}: {token_trust_e}")
                    
                    # Stage 14: Persistent Identity Scoring - WITH TIMEOUT PROTECTION  
                    try:
                        print(f"[DEBUG DEX_INFLOW] {symbol} - Starting identity boost calculation with timeout protection...")
                        
                        # Import with timeout protection
                        import signal
                        
                        def identity_timeout_handler(signum, frame):
                            raise TimeoutError("identity_boost timeout")
                        
                        try:
                            # Set 1-second timeout for identity boost
                            signal.signal(signal.SIGALRM, identity_timeout_handler) 
                            signal.alarm(1)
                            
                            from .persistent_identity_tracker import get_identity_boost
                            identity_boost = get_identity_boost(real_addresses[:3])  # Top 3 addresses only
                            
                            signal.alarm(0)  # Cancel timeout
                            
                            if identity_boost > 0:
                                strength = min(1.0, strength + identity_boost)
                                print(f"[IDENTITY BOOST] {symbol} → identity boost +{identity_boost:.3f} → strength: {strength:.3f}")
                            else:
                                print(f"[IDENTITY BOOST] {symbol} → no identity boost applied")
                                
                        except TimeoutError:
                            signal.alarm(0)
                            print(f"[IDENTITY TIMEOUT] {symbol} - using emergency skip (1s timeout)")
                        
                        print(f"[DEBUG FLOW] {symbol} - Identity boost completed, proceeding to Trigger Alert System...")
                        
                    except Exception as identity_e:
                        print(f"[IDENTITY BOOST ERROR] dex_inflow for {symbol}: {identity_e}")
                        print(f"[DEBUG DEX_INFLOW] {symbol} - Identity boost calculation failed, continuing...")
                        print(f"[DEBUG FLOW] {symbol} - After identity_boost ERROR, proceeding to Trigger Alert System...")
                    
                    # Etap 7: Trigger Alert System - WILL BE HANDLED BY UNIVERSAL SYSTEM BELOW
                    print(f"[DEBUG FLOW] {symbol} - Trigger Alert System will be handled by UNIVERSAL system below")
                    
                except Exception as memory_e:
                    print(f"[WHALE MEMORY] DEX error for {symbol}: {memory_e}")
                    print(f"[DEBUG FLOW] {symbol} - Whale memory ERROR, continuing to result...")
            
            # TRIGGER ALERT SYSTEM - ALWAYS RUN (not dependent on spike_detected)
            print(f"[DEBUG FLOW] {symbol} - Starting UNIVERSAL Trigger Alert System (independent of spike detection)...")
            try:
                from stealth_engine.trigger_alert_system import check_smart_money_trigger, apply_smart_money_boost
                from stealth_engine.address_trust_manager import get_trust_manager
                print(f"[DEBUG FLOW] {symbol} - Trigger imports successful")
                
                # Sprawdź czy wykryto trigger addresses nawet bez spike detection
                if real_addresses:
                    trust_manager = get_trust_manager()
                    print(f"[DEBUG FLOW] {symbol} - Trust manager loaded")
                    
                    try:
                        print(f"[DEBUG FLOW] {symbol} - Calling check_smart_money_trigger with {len(real_addresses[:10])} addresses...")
                        is_trigger, trigger_addresses = check_smart_money_trigger(real_addresses[:10], trust_manager)
                        print(f"[DEBUG FLOW] {symbol} - Smart money check completed: {is_trigger}")
                        
                        if is_trigger:
                            # Zastosuj trigger boost - minimum score 3.0 dla instant alert
                            print(f"[DEBUG FLOW] {symbol} - Applying smart money boost...")
                            boosted_strength, priority_alert = apply_smart_money_boost(
                                symbol, strength, trigger_addresses, "dex_inflow"
                            )
                            strength = boosted_strength
                            spike_detected = True  # Force spike detection dla smart money
                            
                            print(f"[TRIGGER ALERT] 🚨 {symbol} DEX INFLOW: Smart money detected! "
                                  f"Strength boosted to {strength:.3f} (priority alert: {priority_alert})")
                        else:
                            print(f"[DEBUG FLOW] {symbol} - No smart money trigger detected")
                            
                    except Exception as smart_money_e:
                        print(f"[SMART_MONEY ERROR] {symbol} - check_smart_money_trigger failed: {smart_money_e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[DEBUG FLOW] {symbol} - No real addresses available for trigger analysis")
                    
                print(f"[DEBUG FLOW] {symbol} - UNIVERSAL Trigger Alert System completed")
                
            except Exception as trigger_e:
                print(f"[UNIVERSAL TRIGGER ERROR] dex_inflow for {symbol}: {trigger_e}")
                import traceback
                traceback.print_exc()
                print(f"[DEBUG FLOW] {symbol} - UNIVERSAL Trigger Alert System ERROR, continuing...")
            
            print(f"[DEBUG FLOW] {symbol} - All DEX inflow processing completed, preparing result...")
            
            # RESULT LOG - końcowa decyzja i siła sygnału
            print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] RESULT → active={spike_detected}, strength={strength:.3f}")
            print(f"[DEBUG FLOW] {symbol} - DEX inflow function returning signal")
            
            # CRITICAL FIX: Return StealthSignal object (was missing!)
            result = StealthSignal(
                name=FUNC_NAME,
                active=spike_detected,
                strength=strength
            )
            print(f"[SCAN END] {symbol} - DEX inflow function completed successfully")
            return result
        
        except Exception as e:
            print(f"[STEALTH ERROR] DEX inflow for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            print(f"[SCAN END] {symbol} - DEX inflow function failed with exception")
            return StealthSignal("dex_inflow", False, 0.0)
    
    def check_event_tag(self, token_data: Dict) -> StealthSignal:
        """
        Event tag detection - unlock tokenów / airdrop
        """
        FUNC_NAME = "event_tag"
        symbol = token_data.get("symbol", "UNKNOWN")
        tag = token_data.get("event_tag", None)
        
        # INPUT LOG
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → event_tag={tag}")
        
        active = tag is not None
        strength = 1.0 if active else 0.0
        
        # RESULT LOG
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] RESULT → active={active}, strength={strength:.3f}")
        
        return StealthSignal("event_tag", active, strength)
    
    def check_orderbook_imbalance_stealth(self, token_data: Dict) -> StealthSignal:
        """
        Sprawdź asymetrię orderbook - wersja stealth
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        orderbook = token_data.get('orderbook', {})
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        print(f"[STEALTH DEBUG] orderbook_imbalance_stealth for {symbol}: checking orderbook asymmetry...")
        
        if not bids or not asks:
            print(f"[STEALTH DEBUG] orderbook_imbalance_stealth for {symbol}: insufficient orderbook data")
            return StealthSignal("orderbook_imbalance", False, 0.0)
        
        try:
            # Handle different orderbook formats (list vs dict) with safe processing
            if isinstance(bids, dict):
                try:
                    bids_list = []
                    for key in sorted(bids.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0, reverse=True):
                        if isinstance(bids[key], list) and len(bids[key]) >= 2:
                            bids_list.append(bids[key])
                    bids = bids_list if bids_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] orderbook_imbalance bids conversion error for {symbol}: {e}")
                    print(f"[STEALTH DEBUG] orderbook_imbalance bids format for {symbol}: {type(bids)}, keys: {list(bids.keys()) if isinstance(bids, dict) else 'N/A'}")
                    bids = []
            
            if isinstance(asks, dict):
                try:
                    asks_list = []
                    for key in sorted(asks.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0):
                        if isinstance(asks[key], list) and len(asks[key]) >= 2:
                            asks_list.append(asks[key])
                    asks = asks_list if asks_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] orderbook_imbalance asks conversion error for {symbol}: {e}")
                    print(f"[STEALTH DEBUG] orderbook_imbalance asks format for {symbol}: {type(asks)}, keys: {list(asks.keys()) if isinstance(asks, dict) else 'N/A'}")
                    asks = []
            
            # Handle case where bids is a list of dicts with 'price' and 'size' keys
            if isinstance(bids, list) and len(bids) > 0 and isinstance(bids[0], dict):
                try:
                    bids_list = []
                    for bid in bids:
                        if isinstance(bid, dict) and 'price' in bid and 'size' in bid:
                            bids_list.append([bid['price'], bid['size']])
                        elif isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            bids_list.append(bid)
                    bids = bids_list if bids_list else []
                    print(f"[STEALTH DEBUG] orderbook_imbalance for {symbol}: converted dict format bids to list format, count: {len(bids)}")
                except Exception as e:
                    print(f"[STEALTH DEBUG] orderbook_imbalance bids dict-to-list conversion error for {symbol}: {e}")
                    bids = []
            
            # Handle case where asks is a list of dicts with 'price' and 'size' keys
            if isinstance(asks, list) and len(asks) > 0 and isinstance(asks[0], dict):
                try:
                    asks_list = []
                    for ask in asks:
                        if isinstance(ask, dict) and 'price' in ask and 'size' in ask:
                            asks_list.append([ask['price'], ask['size']])
                        elif isinstance(ask, (list, tuple)) and len(ask) >= 2:
                            asks_list.append(ask)
                    asks = asks_list if asks_list else []
                    print(f"[STEALTH DEBUG] orderbook_imbalance for {symbol}: converted dict format asks to list format, count: {len(asks)}")
                except Exception as e:
                    print(f"[STEALTH DEBUG] orderbook_imbalance asks dict-to-list conversion error for {symbol}: {e}")
                    asks = []
            
            # Verify we have valid data after conversion
            if not bids or not asks:
                print(f"[STEALTH DEBUG] orderbook_imbalance for {symbol}: no valid bids/asks after conversion")
                return StealthSignal("orderbook_imbalance", False, 0.0)
            
            # Additional validation for data structure after conversion
            if not isinstance(bids[0], (list, tuple)) or len(bids[0]) < 2:
                print(f"[STEALTH DEBUG] orderbook_imbalance for {symbol}: invalid bid structure after conversion: {type(bids[0])}, content: {bids[0]}")
                return StealthSignal("orderbook_imbalance", False, 0.0)
            
            if not isinstance(asks[0], (list, tuple)) or len(asks[0]) < 2:
                print(f"[STEALTH DEBUG] orderbook_imbalance for {symbol}: invalid ask structure after conversion: {type(asks[0])}, content: {asks[0]}")
                return StealthSignal("orderbook_imbalance", False, 0.0)
            
            # Oblicz siłę bid vs ask z enhanced validation dla każdego poziomu
            bid_strength = 0
            ask_strength = 0
            
            # Safe calculation for bid strength
            for i, bid in enumerate(bids[:5]):
                if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                    try:
                        bid_strength += float(bid[1])
                    except (ValueError, TypeError) as e:
                        print(f"[STEALTH DEBUG] orderbook_imbalance for {symbol}: invalid bid[{i}][1] value: {bid[1]}")
                else:
                    print(f"[STEALTH DEBUG] orderbook_imbalance for {symbol}: invalid bid[{i}] structure: {type(bid)}, content: {bid}")
            
            # Safe calculation for ask strength  
            for i, ask in enumerate(asks[:5]):
                if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                    try:
                        ask_strength += float(ask[1])
                    except (ValueError, TypeError) as e:
                        print(f"[STEALTH DEBUG] orderbook_imbalance for {symbol}: invalid ask[{i}][1] value: {ask[1]}")
                else:
                    print(f"[STEALTH DEBUG] orderbook_imbalance for {symbol}: invalid ask[{i}] structure: {type(ask)}, content: {ask}")
            
            if bid_strength + ask_strength == 0:
                return StealthSignal("orderbook_imbalance", False, 0.0)
            
            imbalance_ratio = abs(bid_strength - ask_strength) / (bid_strength + ask_strength)
            active = imbalance_ratio > 0.6  # 60% threshold
            strength = min(imbalance_ratio, 1.0)
            
            print(f"[STEALTH DEBUG] orderbook_imbalance_stealth: bid_strength={bid_strength:.0f}, ask_strength={ask_strength:.0f}, imbalance_ratio={imbalance_ratio:.3f}")
            if active:
                print(f"[STEALTH DEBUG] orderbook_imbalance_stealth DETECTED: imbalance_ratio={imbalance_ratio:.3f} > 0.6")
            return StealthSignal("orderbook_imbalance", active, strength)
        except Exception as e:
            print(f"[STEALTH DEBUG] orderbook_imbalance_stealth error for {symbol}: {e}")
            print(f"[STEALTH DEBUG] orderbook_imbalance_stealth error details for {symbol}: bids_type={type(bids)}, asks_type={type(asks)}")
            return StealthSignal("orderbook_imbalance", False, 0.0)
    
    def check_large_bid_walls_stealth(self, token_data: Dict) -> StealthSignal:
        """
        Wykryj duże mury bid wspierające cenę - wersja stealth
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        orderbook = token_data.get('orderbook', {})
        bids = orderbook.get('bids', [])
        
        print(f"[STEALTH DEBUG] large_bid_walls: checking bid wall sizes...")
        
        # === MICROCAP ADAPTATION ===
        # For tokens with minimal orderbook, use dynamic thresholds
        volume_threshold = 5.0 if len(bids) <= 1 else 10.0  # Lower for microcap tokens
        
        # ENHANCED: Adaptive logic for low-liquidity tokens
        token_price = token_data.get("price_usd", 1.0)
        is_microcap = token_price < 1.0  # Tokens under $1 are considered microcap
        
        min_levels_required = 1 if is_microcap else 3  # Relaxed requirement for microcap
        
        if len(bids) < min_levels_required:
            print(f"[STEALTH DEBUG] large_bid_walls: insufficient bid levels ({len(bids)} < {min_levels_required}) for {'microcap' if is_microcap else 'regular'} token")
            return StealthSignal("large_bid_walls", False, 0.0)
        
        try:
            # Sprawdź czy są duże bidy z adaptive threshold based on token type
            volume_threshold = 5.0 if is_microcap else 10.0  # Lower threshold for microcap
            max_levels_to_check = min(len(bids), 3 if not is_microcap else 1)  # Check fewer levels for microcap
            
            large_bids = 0
            for i, bid in enumerate(bids[:max_levels_to_check]):
                if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                    try:
                        if float(bid[1]) > volume_threshold:
                            large_bids += 1
                    except (ValueError, TypeError) as e:
                        print(f"[STEALTH DEBUG] large_bid_walls for {symbol}: invalid bid[{i}][1] value: {bid[1]}")
                else:
                    print(f"[STEALTH DEBUG] large_bid_walls for {symbol}: invalid bid[{i}] structure: {type(bid)}, content: {bid}")
            
            # Adaptive activation criteria
            required_large_bids = 1 if is_microcap else 2
            active = large_bids >= required_large_bids
            strength = large_bids / max_levels_to_check
            
            if active:
                print(f"[STEALTH DEBUG] large_bid_walls DETECTED for {symbol}: {large_bids}/3 large bids (>10.0 volume)")
            
            print(f"[STEALTH DEBUG] large_bid_walls result for {symbol}: active={active}, strength={strength:.3f}, large_bids={large_bids}/3")
            return StealthSignal("large_bid_walls", active, strength)
        except Exception as e:
            print(f"[STEALTH DEBUG] large_bid_walls error for {symbol}: {e}")
            return StealthSignal("large_bid_walls", False, 0.0)
    
    def check_ask_wall_removal(self, token_data: Dict) -> StealthSignal:
        """
        Wykryj usunięcie murów ask (placeholder - wymaga historycznych danych)
        """
        FUNC_NAME = "ask_wall_removal"
        symbol = token_data.get("symbol", "UNKNOWN")
        # Placeholder - w rzeczywistości potrzebne są dane historyczne orderbook
        active = token_data.get("ask_walls_removed", False)
        
        # INPUT LOG
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → ask_walls_removed={active}")
        
        strength = 1.0 if active else 0.0
        
        # RESULT LOG
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] RESULT → active={active}, strength={strength:.3f}")
        
        return StealthSignal("ask_wall_removal", active, strength)
    
    def check_volume_spike_stealth(self, token_data: Dict) -> StealthSignal:
        """
        Volume spike detector - nagłe zwiększenie wolumenu na 15M świecy
        Matematycznie precyzyjna implementacja zgodna z user specification v2
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        candles_15m = token_data.get('candles_15m', [])
        volume_24h = token_data.get('volume_24h', 0)
        
        print(f"[STEALTH DEBUG] volume_spike for {symbol}: candles_15m={len(candles_15m)}, volume_24h={volume_24h}")
        print(f"[STEALTH INPUT] volume_spike received candles_15m with {len(candles_15m)} entries")
        
        if len(candles_15m) < 4:
            print(f"[STEALTH DEBUG] volume_spike for {symbol}: insufficient candle data ({len(candles_15m)}/4)")
            print(f"[STEALTH DEBUG] volume_spike DIAGNOSTIC for {symbol}: candles_15m type={type(candles_15m)}, content preview={str(candles_15m)[:200]}...")
            return StealthSignal("volume_spike", False, 0.0)
        
        try:
            # Pobierz ostatnie 4 świece zgodnie ze specyfikacją
            recent_candles = candles_15m[-4:]
            volumes = []
            
            for candle in recent_candles:
                try:
                    if isinstance(candle, dict):
                        volume = float(candle.get('volume', 0))
                    elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                        volume = float(candle[5])  # Volume na pozycji 5
                    else:
                        volume = 0
                    volumes.append(volume)
                except (ValueError, TypeError, IndexError):
                    volumes.append(0)
            
            if len(volumes) < 4 or volumes[-1] == 0:
                print(f"[STEALTH DEBUG] volume_spike for {symbol}: invalid volume data")
                return StealthSignal("volume_spike", False, 0.0)
            
            # Warunek aktywacji: vol_current > 2 × avg([vol_1, vol_2, vol_3])
            vol_current = volumes[-1]  # Ostatnia świeca
            vol_prev = volumes[:-1]    # Poprzednie 3 świece
            
            # Oblicz avg_volume z poprzednich 3 świec
            avg_volume = sum(vol_prev) / len(vol_prev) if vol_prev else 1
            
            # Warunek aktywacji
            is_active = vol_current > 2 * avg_volume
            
            # Strength: min(1.0, vol_current / avg_volume - 1)
            strength = min(1.0, vol_current / avg_volume - 1) if is_active else 0.0
            
            print(f"[STEALTH DEBUG] volume_spike: vol_current={vol_current:.0f}, avg_volume={avg_volume:.0f}, ratio={vol_current/avg_volume:.2f}")
            if is_active:
                print(f"[STEALTH DEBUG] volume_spike DETECTED: vol_current={vol_current:.0f} > 2×avg_volume={avg_volume:.0f}")
            return StealthSignal("volume_spike", is_active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] volume_spike error for {symbol}: {e}")
            return StealthSignal("volume_spike", False, 0.0)
    
    def check_bid_ask_spread_tightening_stealth(self, token_data: Dict) -> StealthSignal:
        """
        Wykryj zwężenie spreadu bid-ask - wersja stealth
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        orderbook = token_data.get('orderbook', {})
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        print(f"[STEALTH DEBUG] spread_tightening for {symbol}: checking bid-ask spread compression...")
        
        if not bids or not asks:
            print(f"[STEALTH DEBUG] spread_tightening for {symbol}: insufficient orderbook data")
            return StealthSignal("bid_ask_spread_tightening", False, 0.0)
        
        try:
            # Handle different orderbook formats (list vs dict) with safe processing
            symbol = token_data.get("symbol", "UNKNOWN")
            
            if isinstance(bids, dict):
                try:
                    bids_list = []
                    for key in sorted(bids.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0, reverse=True):
                        if isinstance(bids[key], list) and len(bids[key]) >= 2:
                            bids_list.append(bids[key])
                    bids = bids_list if bids_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] spread_tightening bids conversion error for {symbol}: {e}")
                    print(f"[STEALTH DEBUG] spread_tightening bids format for {symbol}: {type(bids)}, keys: {list(bids.keys()) if isinstance(bids, dict) else 'N/A'}")
                    bids = []
            
            if isinstance(asks, dict):
                try:
                    asks_list = []
                    for key in sorted(asks.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0):
                        if isinstance(asks[key], list) and len(asks[key]) >= 2:
                            asks_list.append(asks[key])
                    asks = asks_list if asks_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] spread_tightening asks conversion error for {symbol}: {e}")
                    print(f"[STEALTH DEBUG] spread_tightening asks format for {symbol}: {type(asks)}, keys: {list(asks.keys()) if isinstance(asks, dict) else 'N/A'}")
                    asks = []
            
            # Handle case where bids is a list of dicts with 'price' and 'size' keys
            if isinstance(bids, list) and len(bids) > 0 and isinstance(bids[0], dict):
                try:
                    bids_list = []
                    for bid in bids:
                        if isinstance(bid, dict) and 'price' in bid and 'size' in bid:
                            bids_list.append([bid['price'], bid['size']])
                        elif isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            bids_list.append(bid)
                    bids = bids_list if bids_list else []
                    print(f"[STEALTH DEBUG] spread_tightening for {symbol}: converted dict format bids to list format, count: {len(bids)}")
                except Exception as e:
                    print(f"[STEALTH DEBUG] spread_tightening bids dict-to-list conversion error for {symbol}: {e}")
                    bids = []
            
            # Handle case where asks is a list of dicts with 'price' and 'size' keys
            if isinstance(asks, list) and len(asks) > 0 and isinstance(asks[0], dict):
                try:
                    asks_list = []
                    for ask in asks:
                        if isinstance(ask, dict) and 'price' in ask and 'size' in ask:
                            asks_list.append([ask['price'], ask['size']])
                        elif isinstance(ask, (list, tuple)) and len(ask) >= 2:
                            asks_list.append(ask)
                    asks = asks_list if asks_list else []
                    print(f"[STEALTH DEBUG] spread_tightening for {symbol}: converted dict format asks to list format, count: {len(asks)}")
                except Exception as e:
                    print(f"[STEALTH DEBUG] spread_tightening asks dict-to-list conversion error for {symbol}: {e}")
                    asks = []
            
            # Verify we have valid data after conversion
            if not bids or not asks:
                print(f"[STEALTH DEBUG] spread_tightening for {symbol}: no valid bids/asks after conversion")
                return StealthSignal("bid_ask_spread_tightening", False, 0.0)
            
            # Additional validation for data structure after conversion
            if not isinstance(bids[0], (list, tuple)) or len(bids[0]) < 2:
                print(f"[STEALTH DEBUG] spread_tightening for {symbol}: invalid bid structure after conversion: {type(bids[0])}, content: {bids[0]}")
                return StealthSignal("bid_ask_spread_tightening", False, 0.0)
            
            if not isinstance(asks[0], (list, tuple)) or len(asks[0]) < 2:
                print(f"[STEALTH DEBUG] spread_tightening for {symbol}: invalid ask structure after conversion: {type(asks[0])}, content: {asks[0]}")
                return StealthSignal("bid_ask_spread_tightening", False, 0.0)
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            if best_bid == 0:
                return StealthSignal("bid_ask_spread_tightening", False, 0.0)
            
            spread_percentage = (best_ask - best_bid) / best_bid * 100
            
            # Tight spread < 0.1%
            active = spread_percentage < 0.1
            strength = max(0.0, (0.1 - spread_percentage) / 0.1) if active else 0.0
            
            print(f"[STEALTH DEBUG] spread_tightening: spread_percentage={spread_percentage:.4f}%")
            if active:
                print(f"[STEALTH DEBUG] spread_tightening DETECTED: spread_percentage={spread_percentage:.4f}% < 0.1%")
            return StealthSignal("bid_ask_spread_tightening", active, strength)
        except Exception as e:
            print(f"[STEALTH DEBUG] spread_tightening error for {symbol}: {e}")
            print(f"[STEALTH DEBUG] spread_tightening error details for {symbol}: bids_type={type(bids)}, asks_type={type(asks)}")
            if hasattr(e, '__traceback__'):
                import traceback
                print(f"[STEALTH DEBUG] spread_tightening traceback for {symbol}: {traceback.format_exc()}")
            return StealthSignal("bid_ask_spread_tightening", False, 0.0)
    
    def check_liquidity_absorption(self, token_data: Dict) -> StealthSignal:
        """
        Wykryj absorpcję płynności (placeholder)
        """
        FUNC_NAME = "liquidity_absorption"
        symbol = token_data.get("symbol", "UNKNOWN")
        # Placeholder - wymaga analizy zmian w orderbook
        active = token_data.get("liquidity_absorbed", False)
        
        # INPUT LOG
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] INPUT → liquidity_absorbed={active}")
        
        strength = 1.0 if active else 0.0
        
        # RESULT LOG
        print(f"[STEALTH DEBUG] [{symbol}] [{FUNC_NAME}] RESULT → active={active}, strength={strength:.3f}")
        
        return StealthSignal("liquidity_absorption", active, strength)
    
    def check_orderbook_anomaly(self, token_data: Dict) -> StealthSignal:
        """
        Orderbook anomaly detector - anomalie w spreadzie i balansie bid/ask
        Matematycznie precyzyjna implementacja zgodna z user specification v2
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        orderbook = token_data.get("orderbook", {})
        
        print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: checking spread and imbalance...")
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: insufficient orderbook data")
            return StealthSignal("orderbook_anomaly", False, 0.0)
        
        try:
            # Handle different orderbook formats (list vs dict) with safe processing
            if isinstance(bids, dict):
                try:
                    bids_list = []
                    for key in sorted(bids.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0, reverse=True):
                        if isinstance(bids[key], list) and len(bids[key]) >= 2:
                            bids_list.append(bids[key])
                    bids = bids_list if bids_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] orderbook_anomaly bids conversion error for {symbol}: {e}")
                    bids = []
            
            if isinstance(asks, dict):
                try:
                    asks_list = []
                    for key in sorted(asks.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0):
                        if isinstance(asks[key], list) and len(asks[key]) >= 2:
                            asks_list.append(asks[key])
                    asks = asks_list if asks_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] orderbook_anomaly asks conversion error for {symbol}: {e}")
                    asks = []
            
            # Handle case where bids is a list of dicts with 'price' and 'size' keys
            if isinstance(bids, list) and len(bids) > 0 and isinstance(bids[0], dict):
                try:
                    bids_list = []
                    for bid in bids:
                        if isinstance(bid, dict) and 'price' in bid and 'size' in bid:
                            bids_list.append([bid['price'], bid['size']])
                        elif isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            bids_list.append(bid)
                    bids = bids_list if bids_list else []
                    print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: converted dict format bids to list format, count: {len(bids)}")
                except Exception as e:
                    print(f"[STEALTH DEBUG] orderbook_anomaly bids dict-to-list conversion error for {symbol}: {e}")
                    bids = []
            
            # Handle case where asks is a list of dicts with 'price' and 'size' keys
            if isinstance(asks, list) and len(asks) > 0 and isinstance(asks[0], dict):
                try:
                    asks_list = []
                    for ask in asks:
                        if isinstance(ask, dict) and 'price' in ask and 'size' in ask:
                            asks_list.append([ask['price'], ask['size']])
                        elif isinstance(ask, (list, tuple)) and len(ask) >= 2:
                            asks_list.append(ask)
                    asks = asks_list if asks_list else []
                    print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: converted dict format asks to list format, count: {len(asks)}")
                except Exception as e:
                    print(f"[STEALTH DEBUG] orderbook_anomaly asks dict-to-list conversion error for {symbol}: {e}")
                    asks = []
            
            # Verify we have valid data after conversion
            if not bids or not asks:
                print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: no valid bids/asks after conversion")
                return StealthSignal("orderbook_anomaly", False, 0.0)
            
            # Additional validation for data structure after conversion
            if not isinstance(bids[0], (list, tuple)) or len(bids[0]) < 2:
                print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: invalid bid structure after conversion: {type(bids[0])}, content: {bids[0]}")
                return StealthSignal("orderbook_anomaly", False, 0.0)
            
            if not isinstance(asks[0], (list, tuple)) or len(asks[0]) < 2:
                print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: invalid ask structure after conversion: {type(asks[0])}, content: {asks[0]}")
                return StealthSignal("orderbook_anomaly", False, 0.0)
            
            # Oblicz spread_pct i imbalance_pct zgodnie ze specyfikacją
            bid_price = float(bids[0][0])
            ask_price = float(asks[0][0])
            mid_price = (bid_price + ask_price) / 2
            
            # spread_pct = (ask_price - bid_price) / mid_price
            spread_pct = (ask_price - bid_price) / mid_price
            
            # Oblicz total bids i total asks volume z enhanced validation
            total_bids = 0
            total_asks = 0
            
            # Safe calculation for total bids
            for i, bid in enumerate(bids[:10]):
                if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                    try:
                        total_bids += float(bid[1])
                    except (ValueError, TypeError) as e:
                        print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: invalid bid[{i}][1] value: {bid[1]}")
                else:
                    print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: invalid bid[{i}] structure: {type(bid)}, content: {bid}")
            
            # Safe calculation for total asks
            for i, ask in enumerate(asks[:10]):
                if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                    try:
                        total_asks += float(ask[1])
                    except (ValueError, TypeError) as e:
                        print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: invalid ask[{i}][1] value: {ask[1]}")
                else:
                    print(f"[STEALTH DEBUG] orderbook_anomaly for {symbol}: invalid ask[{i}] structure: {type(ask)}, content: {ask}")
            
            # imbalance_pct = abs(total_bids - total_asks) / (total_bids + total_asks)
            total_volume = total_bids + total_asks
            imbalance_pct = abs(total_bids - total_asks) / total_volume if total_volume > 0 else 0.0
            
            # Warunek aktywacji: spread_pct < 0.0005 and imbalance_pct > 0.85
            is_active = spread_pct < 0.0005 and imbalance_pct > 0.85
            
            # Strength: min(1.0, imbalance_pct × 2)
            strength = min(1.0, imbalance_pct * 2) if is_active else 0.0
            
            print(f"[STEALTH DEBUG] orderbook_anomaly: spread_pct={spread_pct:.6f}, imbalance_pct={imbalance_pct:.3f}")
            if is_active:
                print(f"[STEALTH DEBUG] orderbook_anomaly DETECTED: spread_pct={spread_pct:.6f} < 0.0005, imbalance_pct={imbalance_pct:.3f} > 0.85")
            return StealthSignal("orderbook_anomaly", is_active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] orderbook_anomaly error for {symbol}: {e}")
            return StealthSignal("orderbook_anomaly", False, 0.0)

    async def detect_all_signals(self, symbol: str, market_data: Dict) -> List[Dict]:
        """
        Wykryj wszystkie sygnały stealth dla tokena
        
        Args:
            symbol: Symbol tokena
            market_data: Dane rynkowe (orderbook, volume, candles)
            
        Returns:
            Lista słowników z sygnałami: {signal_name, active, strength, details}
        """
        signals = []
        
        try:
            # ORDERBOOK SIGNALS
            orderbook_signals = await self.detect_orderbook_signals(symbol, market_data)
            signals.extend(orderbook_signals)
            
            # VOLUME SIGNALS 
            volume_signals = await self.detect_volume_signals(symbol, market_data)
            signals.extend(volume_signals)
            
            # DEX SIGNALS
            dex_signals = await self.detect_dex_signals(symbol, market_data)
            signals.extend(dex_signals)
            
            # MICROSTRUCTURE SIGNALS
            microstructure_signals = await self.detect_microstructure_signals(symbol, market_data)
            signals.extend(microstructure_signals)
            
            active_count = sum(1 for s in signals if s['active'])
            print(f"[SIGNALS] {symbol}: {active_count}/{len(signals)} signals active")
            
            return signals
            
        except Exception as e:
            print(f"[SIGNAL ERROR] {symbol}: {e}")
            return []
    
    async def detect_orderbook_signals(self, symbol: str, market_data: Dict) -> List[Dict]:
        """Wykryj sygnały z analizy orderbook"""
        signals = []
        orderbook = market_data.get('orderbook', {})
        
        if not orderbook:
            # Zwróć nieaktywne sygnały
            for signal_name in ['orderbook_imbalance', 'large_bid_walls', 'ask_wall_removal', 'spoofing_detected']:
                signals.append({
                    'signal_name': signal_name,
                    'active': False,
                    'strength': 0.0,
                    'details': 'No orderbook data'
                })
            return signals
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        # 1. ORDERBOOK IMBALANCE
        imbalance_signal = self.detect_orderbook_imbalance(bids, asks)
        signals.append(imbalance_signal)
        
        # 2. LARGE BID WALLS
        bid_walls_signal = self.detect_large_bid_walls(bids)
        signals.append(bid_walls_signal)
        
        # 3. ASK WALL REMOVAL (wymagałby historii orderbook)
        ask_removal_signal = {
            'signal_name': 'ask_wall_removal',
            'active': False,
            'strength': 0.0,
            'details': 'Requires orderbook history - future implementation'
        }
        signals.append(ask_removal_signal)
        
        # 4. SPOOFING DETECTION
        spoofing_signal = self.detect_spoofing(bids, asks)
        signals.append(spoofing_signal)
        
        return signals
    
    def detect_orderbook_imbalance(self, bids: List, asks: List) -> Dict:
        """Wykryj asymetrię bid/ask w orderbook"""
        if not bids or not asks:
            return {
                'signal_name': 'orderbook_imbalance',
                'active': False,
                'strength': 0.0,
                'details': 'Insufficient orderbook data'
            }
        
        try:
            # Oblicz całkowity wolumen bid vs ask (pierwsze 10 poziomów)
            total_bid_volume = sum(float(bid[1]) for bid in bids[:10])
            total_ask_volume = sum(float(ask[1]) for ask in asks[:10])
            
            if total_ask_volume == 0:
                ratio = 10.0  # Max imbalance
            else:
                ratio = total_bid_volume / total_ask_volume
            
            # Sygnał aktywny gdy ratio > 2.0 (bid dominuje)
            if ratio > 2.0:
                strength = min(1.0, ratio / 5.0)  # Normalizacja do 1.0
                return {
                    'signal_name': 'orderbook_imbalance',
                    'active': True,
                    'strength': strength,
                    'details': f'Bid/Ask ratio: {ratio:.2f}'
                }
            else:
                return {
                    'signal_name': 'orderbook_imbalance',
                    'active': False,
                    'strength': 0.0,
                    'details': f'Normal ratio: {ratio:.2f}'
                }
                
        except Exception as e:
            return {
                'signal_name': 'orderbook_imbalance',
                'active': False,
                'strength': 0.0,
                'details': f'Error: {e}'
            }
    
    def detect_large_bid_walls(self, bids: List) -> Dict:
        """Wykryj duże mury bid wspierające cenę"""
        if not bids or len(bids) < 5:
            return {
                'signal_name': 'large_bid_walls',
                'active': False,
                'strength': 0.0,
                'details': 'Insufficient bid data'
            }
        
        try:
            # Analiza wolumenu na poziomach bid
            bid_volumes = [float(bid[1]) for bid in bids[:10]]
            avg_volume = statistics.mean(bid_volumes)
            max_volume = max(bid_volumes)
            
            # Wykryj znacząco większe zlecenia (>3x średnia)
            if max_volume > avg_volume * 3.0:
                strength = min(1.0, max_volume / (avg_volume * 5.0))
                return {
                    'signal_name': 'large_bid_walls',
                    'active': True,
                    'strength': strength,
                    'details': f'Large bid: {max_volume:.0f} vs avg {avg_volume:.0f}'
                }
            else:
                return {
                    'signal_name': 'large_bid_walls',
                    'active': False,
                    'strength': 0.0,
                    'details': f'Normal bid sizes: max {max_volume:.0f}'
                }
                
        except Exception as e:
            return {
                'signal_name': 'large_bid_walls',
                'active': False,
                'strength': 0.0,
                'details': f'Error: {e}'
            }
    
    def detect_spoofing(self, bids: List, asks: List) -> Dict:
        """Wykryj potencjalne spoofing w orderbook"""
        if not bids or not asks:
            return {
                'signal_name': 'spoofing_detected',
                'active': False,
                'strength': 0.0,
                'details': 'No orderbook data'
            }
        
        try:
            # Heurystyka: bardzo duże zlecenia daleko od spread
            bid_prices = [float(bid[0]) for bid in bids[:5]]
            bid_volumes = [float(bid[1]) for bid in bids[:5]]
            
            ask_prices = [float(ask[0]) for ask in asks[:5]]
            ask_volumes = [float(ask[1]) for ask in asks[:5]]
            
            # Sprawdź czy są bardzo duże zlecenia daleko od best bid/ask
            best_bid = bid_prices[0] if bid_prices else 0
            best_ask = ask_prices[0] if ask_prices else float('inf')
            
            avg_bid_volume = statistics.mean(bid_volumes) if bid_volumes else 0
            avg_ask_volume = statistics.mean(ask_volumes) if ask_volumes else 0
            
            spoofing_strength = 0.0
            
            # Sprawdź duże zlecenia bid daleko od ceny
            for i, (price, volume) in enumerate(zip(bid_prices, bid_volumes)):
                if i > 0 and volume > avg_bid_volume * 5.0:  # Bardzo duże zlecenie
                    distance_from_best = (best_bid - price) / best_bid if best_bid > 0 else 0
                    if distance_from_best > 0.02:  # >2% od best bid
                        spoofing_strength = max(spoofing_strength, 0.7)
            
            # Sprawdź duże zlecenia ask daleko od ceny
            for i, (price, volume) in enumerate(zip(ask_prices, ask_volumes)):
                if i > 0 and volume > avg_ask_volume * 5.0:  # Bardzo duże zlecenie
                    distance_from_best = (price - best_ask) / best_ask if best_ask > 0 else 0
                    if distance_from_best > 0.02:  # >2% od best ask
                        spoofing_strength = max(spoofing_strength, 0.7)
            
            if spoofing_strength > 0.5:
                return {
                    'signal_name': 'spoofing_detected',
                    'active': True,
                    'strength': spoofing_strength,
                    'details': 'Large orders far from spread detected'
                }
            else:
                return {
                    'signal_name': 'spoofing_detected',
                    'active': False,
                    'strength': 0.0,
                    'details': 'No spoofing patterns detected'
                }
                
        except Exception as e:
            return {
                'signal_name': 'spoofing_detected',
                'active': False,
                'strength': 0.0,
                'details': f'Error: {e}'
            }
    
    async def detect_volume_signals(self, symbol: str, market_data: Dict) -> List[Dict]:
        """Wykryj sygnały z analizy wolumenu"""
        signals = []
        
        candles_15m = market_data.get('candles_15m', [])
        candles_5m = market_data.get('candles_5m', [])
        current_volume = market_data.get('volume_24h', 0)
        
        # 1. VOLUME SPIKE
        volume_spike_signal = self.detect_volume_spike(candles_15m, current_volume)
        signals.append(volume_spike_signal)
        
        # 2. VOLUME ACCUMULATION
        volume_accumulation_signal = self.detect_volume_accumulation(candles_15m)
        signals.append(volume_accumulation_signal)
        
        # 3. UNUSUAL VOLUME PROFILE
        unusual_profile_signal = self.detect_unusual_volume_profile(candles_5m)
        signals.append(unusual_profile_signal)
        
        return signals
    
    def detect_volume_spike(self, candles_15m: List, current_volume: float) -> Dict:
        """Wykryj nagły wzrost wolumenu"""
        if not candles_15m or len(candles_15m) < 10:
            return {
                'signal_name': 'volume_spike',
                'active': False,
                'strength': 0.0,
                'details': 'Insufficient candle data'
            }
        
        try:
            # Pobierz wolumeny z ostatnich świec
            recent_volumes = []
            for candle in candles_15m[-10:]:
                if isinstance(candle, dict):
                    volume = candle.get('volume', 0)
                elif isinstance(candle, list) and len(candle) >= 6:
                    volume = candle[5]  # volume jest na pozycji 5
                else:
                    continue
                recent_volumes.append(float(volume))
            
            if len(recent_volumes) < 5:
                return {
                    'signal_name': 'volume_spike',
                    'active': False,
                    'strength': 0.0,
                    'details': 'Not enough volume data'
                }
            
            # Porównaj ostatnią świecę z średnią
            latest_volume = recent_volumes[-1]
            avg_volume = statistics.mean(recent_volumes[:-1])
            
            if avg_volume > 0:
                volume_ratio = latest_volume / avg_volume
                
                # Spike gdy ostatnia świeca >2x średnia
                if volume_ratio > 2.0:
                    strength = min(1.0, volume_ratio / 5.0)
                    return {
                        'signal_name': 'volume_spike',
                        'active': True,
                        'strength': strength,
                        'details': f'Volume spike: {volume_ratio:.2f}x normal'
                    }
            
            return {
                'signal_name': 'volume_spike',
                'active': False,
                'strength': 0.0,
                'details': 'No significant volume spike'
            }
            
        except Exception as e:
            return {
                'signal_name': 'volume_spike',
                'active': False,
                'strength': 0.0,
                'details': f'Error: {e}'
            }
    
    def detect_volume_accumulation(self, candles_15m: List) -> Dict:
        """Wykryj stopniową akumulację wolumenu"""
        if not candles_15m or len(candles_15m) < 20:
            return {
                'signal_name': 'volume_accumulation',
                'active': False,
                'strength': 0.0,
                'details': 'Insufficient data for accumulation analysis'
            }
        
        try:
            # Pobierz wolumeny z ostatnich 20 świec
            volumes = []
            for candle in candles_15m[-20:]:
                if isinstance(candle, dict):
                    volume = candle.get('volume', 0)
                elif isinstance(candle, list) and len(candle) >= 6:
                    volume = candle[5]
                else:
                    continue
                volumes.append(float(volume))
            
            if len(volumes) < 15:
                return {
                    'signal_name': 'volume_accumulation',
                    'active': False,
                    'strength': 0.0,
                    'details': 'Not enough volume data'
                }
            
            # Porównaj średnią z pierwszej i drugiej połowy
            first_half = volumes[:len(volumes)//2]
            second_half = volumes[len(volumes)//2:]
            
            avg_first = statistics.mean(first_half)
            avg_second = statistics.mean(second_half)
            
            if avg_first > 0:
                accumulation_ratio = avg_second / avg_first
                
                # Akumulacja gdy druga połowa >1.5x pierwsza
                if accumulation_ratio > 1.5:
                    strength = min(1.0, accumulation_ratio / 3.0)
                    return {
                        'signal_name': 'volume_accumulation',
                        'active': True,
                        'strength': strength,
                        'details': f'Volume accumulation: {accumulation_ratio:.2f}x growth'
                    }
            
            return {
                'signal_name': 'volume_accumulation',
                'active': False,
                'strength': 0.0,
                'details': 'No volume accumulation detected'
            }
            
        except Exception as e:
            return {
                'signal_name': 'volume_accumulation',
                'active': False,
                'strength': 0.0,
                'details': f'Error: {e}'
            }
    
    def detect_unusual_volume_profile(self, candles_5m: List) -> Dict:
        """Wykryj nietypowy profil wolumenu"""
        # Placeholder - wymagałby bardziej zaawansowanej analizy VWAP/volume profile
        return {
            'signal_name': 'unusual_volume_profile',
            'active': False,
            'strength': 0.0,
            'details': 'Volume profile analysis - future implementation'
        }
    
    async def detect_dex_signals(self, symbol: str, market_data: Dict) -> List[Dict]:
        """Wykryj sygnały z analizy DEX"""
        # Placeholder - wymagałby integracji z DEX APIs
        signals = [
            {
                'signal_name': 'dex_inflow_spike',
                'active': False,
                'strength': 0.0,
                'details': 'DEX analysis - future implementation'
            },
            {
                'signal_name': 'whale_accumulation',
                'active': False,
                'strength': 0.0,
                'details': 'Whale tracking - future implementation'
            }
        ]
        return signals
    
    async def detect_microstructure_signals(self, symbol: str, market_data: Dict) -> List[Dict]:
        """Wykryj sygnały z mikrostruktury rynku"""
        signals = []
        orderbook = market_data.get('orderbook', {})
        
        # 1. BID-ASK SPREAD TIGHTENING
        spread_signal = self.detect_spread_tightening(orderbook)
        signals.append(spread_signal)
        
        # 2. ORDER FLOW PRESSURE
        flow_signal = self.detect_order_flow_pressure(orderbook)
        signals.append(flow_signal)
        
        # 3. LIQUIDITY ABSORPTION
        liquidity_signal = self.detect_liquidity_absorption(orderbook)
        signals.append(liquidity_signal)
        
        return signals
    
    def detect_spread_tightening(self, orderbook: Dict) -> Dict:
        """Wykryj zawężenie spread bid-ask"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return {
                'signal_name': 'bid_ask_spread_tightening',
                'active': False,
                'strength': 0.0,
                'details': 'No orderbook data'
            }
        
        try:
            # Handle different orderbook formats (list vs dict) with safe processing
            if isinstance(bids, dict):
                try:
                    bids_list = []
                    for key in sorted(bids.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0, reverse=True):
                        if isinstance(bids[key], list) and len(bids[key]) >= 2:
                            bids_list.append(bids[key])
                    bids = bids_list if bids_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] detect_microstructure bids conversion error: {e}")
                    bids = []
            
            if isinstance(asks, dict):
                try:
                    asks_list = []
                    for key in sorted(asks.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0):
                        if isinstance(asks[key], list) and len(asks[key]) >= 2:
                            asks_list.append(asks[key])
                    asks = asks_list if asks_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] detect_microstructure asks conversion error: {e}")
                    asks = []
            
            # Verify we have valid data after conversion
            if not bids or not asks:
                print(f"[STEALTH DEBUG] detect_microstructure: no valid bids/asks after conversion")
                return {
                    'signal_name': 'bid_ask_spread_tightening',
                    'active': False,
                    'strength': 0.0,
                    'details': 'No valid orderbook data after conversion'
                }
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100 if best_bid > 0 else 0
            
            # Tight spread (<0.1%) jest pozytywnym sygnałem
            if spread_pct < 0.1:
                strength = max(0.0, (0.1 - spread_pct) / 0.1)
                return {
                    'signal_name': 'bid_ask_spread_tightening',
                    'active': True,
                    'strength': strength,
                    'details': f'Tight spread: {spread_pct:.3f}%'
                }
            else:
                return {
                    'signal_name': 'bid_ask_spread_tightening',
                    'active': False,
                    'strength': 0.0,
                    'details': f'Normal spread: {spread_pct:.3f}%'
                }
                
        except Exception as e:
            return {
                'signal_name': 'bid_ask_spread_tightening',
                'active': False,
                'strength': 0.0,
                'details': f'Error: {e}'
            }
    
    def detect_order_flow_pressure(self, orderbook: Dict) -> Dict:
        """Wykryj presję w przepływie zleceń"""
        # Placeholder - wymagałby danych time & sales
        return {
            'signal_name': 'order_flow_pressure',
            'active': False,
            'strength': 0.0,
            'details': 'Order flow analysis - requires time & sales data'
        }
    
    def detect_liquidity_absorption(self, orderbook: Dict) -> Dict:
        """Wykryj absorpcję płynności"""
        # Placeholder - wymagałby historii orderbook
        return {
            'signal_name': 'liquidity_absorption',
            'active': False,
            'strength': 0.0,
            'details': 'Liquidity absorption - requires orderbook history'
        }
    
    def check_repeated_address_boost(self, token_data: Dict) -> StealthSignal:
        """
        Repeated Address Boost - wykrycie powtarzalnych schematów akumulacji
        przez te same adresy w sygnałach whale_ping i dex_inflow
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        
        try:
            print(f"[STEALTH DEBUG] repeated_address_boost for {symbol}: checking address patterns...")
            
            # Użyj istniejącej instancji self.address_tracker
            
            # Pobierz aktywne adresy z ostatnich sygnałów (symulowane na podstawie aktualnych wartości)
            current_addresses = []
            
            # Dodaj adresy z dex_inflow jeśli był aktywny
            inflow_usd = token_data.get("dex_inflow", 0)
            if inflow_usd > 1000:
                address = f"dex_{symbol.lower()}_{int(inflow_usd)}"[:42]
                current_addresses.append(address)
                # Zapisz aktywność adresu w historii
                self.address_tracker.record_address_activity(
                    token=symbol,
                    address=address,
                    usd_value=inflow_usd,
                    source="dex_inflow"
                )
            
            # Dodaj adresy z whale_ping na podstawie orderbook
            orderbook = token_data.get("orderbook", {})
            if orderbook.get("bids") and orderbook.get("asks"):
                try:
                    bids = orderbook.get("bids", [])
                    asks = orderbook.get("asks", [])
                    
                    # Oblicz max order value
                    all_orders = []
                    for bid in bids[:10]:  # Sprawdź top 10 bids for better whale detection
                        if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            price = float(bid[0])
                            size = float(bid[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    for ask in asks[:10]:  # Sprawdź top 10 asks for better whale detection
                        if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                            price = float(ask[0])
                            size = float(ask[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    max_order = max(all_orders, default=0)
                    if max_order > 10000:  # Próg dla whale orders
                        address = f"whale_{symbol.lower()}_{int(max_order)}"[:42]
                        current_addresses.append(address)
                        # Zapisz aktywność adresu w historii
                        self.address_tracker.record_address_activity(
                            token=symbol,
                            address=address,
                            usd_value=max_order,
                            source="whale_ping"
                        )
                        
                except Exception as e:
                    print(f"[STEALTH DEBUG] repeated_address_boost orderbook processing error for {symbol}: {e}")
            
            if not current_addresses:
                return StealthSignal("repeated_address_boost", False, 0.0)
            
            # Oblicz boost na podstawie powtarzających się adresów
            boost_score, details = self.address_tracker.get_repeated_addresses_boost(
                token=symbol,
                current_addresses=current_addresses,
                history_days=7
            )
            
            # 🔧 FIX: Obniż próg aktywacji - aktywuj przy 1+ adresach
            # Jeśli mamy jakiekolwiek adresy, daj minimum boost 0.05
            if len(current_addresses) >= 1 and boost_score == 0:
                boost_score = 0.05 * len(current_addresses)  # Minimum boost za obecność adresów
            
            active = boost_score > 0 or len(current_addresses) >= 1  # Aktywuj jeśli są adresy
            strength = min(boost_score / 0.6, 1.0)  # Normalizuj do 0-1 (max boost 0.6)
            
            print(f"[STEALTH DEBUG] repeated_address_boost: addresses={len(current_addresses)}, boost_score={boost_score:.2f}, active={active}")
            if active:
                print(f"[STEALTH DEBUG] repeated_address_boost DETECTED: {details['repeated_addresses']} repeated addresses, boost={boost_score:.2f}")
            
            return StealthSignal("repeated_address_boost", active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] repeated_address_boost error for {symbol}: {e}")
            return StealthSignal("repeated_address_boost", False, 0.0)
    
    def check_velocity_boost(self, token_data: Dict) -> StealthSignal:
        """
        🆕 PHASE 3/5: Time-Based Velocity Tracking
        
        Wykrywa szybkie sekwencje aktywności adresów w czasie - im szybsza
        akumulacja przez te same adresy, tym wyższy boost velocity
        
        Args:
            token_data: Dane rynkowe tokena
            
        Returns:
            StealthSignal z velocity boost score
        """
        try:
            symbol = token_data.get("symbol", "UNKNOWN")
            
            # Identyfikacja aktywnych adresów
            current_addresses = []
            
            # Sprawdź dex_inflow addresses
            if "dex_inflow" in token_data and token_data["dex_inflow"] > 1000:
                dex_value = token_data["dex_inflow"]
                dex_address = f"dex_{symbol.lower().replace('usdt', '')}_{int(dex_value)}"
                current_addresses.append(dex_address)
                
                # Rejestruj aktywność
                self.address_tracker.record_address_activity(
                    address=dex_address,
                    token=symbol,
                    usd_value=dex_value,
                    source="dex_inflow"
                )
            
            # Sprawdź whale_ping addresses
            if "volume_24h" in token_data and token_data["volume_24h"] > 10000:
                whale_value = token_data["volume_24h"] * 0.05  # 5% volume jako whale ping
                whale_address = f"whale_{symbol.lower().replace('usdt', '')}_{int(whale_value)}"
                current_addresses.append(whale_address)
                
                # Rejestruj aktywność
                self.address_tracker.record_address_activity(
                    address=whale_address,
                    token=symbol,
                    usd_value=whale_value,
                    source="whale_ping"
                )
            
            print(f"[STEALTH DEBUG] velocity_boost for {symbol}: checking velocity patterns...")
            
            # Analiza velocity
            velocity_boost, velocity_details = self.address_tracker.get_velocity_analysis(
                current_token=symbol,
                current_addresses=current_addresses,
                window_minutes=60
            )
            
            # 🔧 FIX: Dodaj minimum boost dla 2+ adresów w krótkim czasie
            if len(current_addresses) >= 2 and velocity_boost == 0:
                velocity_boost = 0.05 * len(current_addresses)  # Minimum boost za szybką aktywność
                print(f"[STEALTH DEBUG] velocity_boost: Added minimum boost for {len(current_addresses)} addresses: +{velocity_boost:.2f}")
            
            # Próg aktywacji: velocity_boost > 0.05 (obniżony z 0.1)
            is_active = velocity_boost > 0.05
            strength = min(velocity_boost, 1.0)  # Normalizuj do 0-1
            
            print(f"[STEALTH DEBUG] velocity_boost: addresses={len(current_addresses)}, boost_score={velocity_boost:.2f}, active={is_active}")
            
            if is_active:
                print(f"[STEALTH DEBUG] velocity_boost DETECTED: {velocity_details['velocity_addresses']} velocity addresses, boost={velocity_boost:.2f}")
            
            return StealthSignal(
                name="velocity_boost",
                active=is_active,
                strength=strength
            )
            
        except Exception as e:
            print(f"[STEALTH DEBUG] velocity_boost error: {e}")
            return StealthSignal(
                name="velocity_boost",
                active=False,
                strength=0.0
            )

    def check_inflow_momentum_boost(self, token_data: Dict) -> StealthSignal:
        """
        🆕 PHASE 4/5: Momentum Inflow Booster
        
        Wykrywa przyspieszającą aktywność adresów - jak szybko adresy z dex_inflow 
        lub whale_ping przesyłają środki w czasie. Duży strumień w krótkim okresie = sygnał FOMO/akumulacji
        
        Args:
            token_data: Dane rynkowe tokena
            
        Returns:
            StealthSignal z momentum boost score
        """
        try:
            symbol = token_data.get("symbol", "UNKNOWN")
            
            # Identyfikacja aktywnych adresów
            current_addresses = []
            
            # Sprawdź dex_inflow addresses
            if "dex_inflow" in token_data and token_data["dex_inflow"] > 1000:
                dex_value = token_data["dex_inflow"]
                dex_address = f"dex_{symbol.lower().replace('usdt', '')}_{int(dex_value)}"
                current_addresses.append(dex_address)
                
                # Rejestruj aktywność
                self.address_tracker.record_address_activity(
                    address=dex_address,
                    token=symbol,
                    usd_value=dex_value,
                    source="dex_inflow"
                )
            
            # Sprawdź whale_ping addresses na podstawie orderbook
            orderbook = token_data.get("orderbook", {})
            if orderbook.get("bids") and orderbook.get("asks"):
                try:
                    bids = orderbook.get("bids", [])
                    asks = orderbook.get("asks", [])
                    
                    # Oblicz max order value
                    all_orders = []
                    for bid in bids[:10]:  # Sprawdź top 10 bids for better whale detection
                        if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            price = float(bid[0])
                            size = float(bid[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    for ask in asks[:10]:  # Sprawdź top 10 asks for better whale detection
                        if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                            price = float(ask[0])
                            size = float(ask[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    max_order = max(all_orders, default=0)
                    if max_order > 10000:  # Próg dla whale orders
                        whale_address = f"whale_{symbol.lower().replace('usdt', '')}_{int(max_order)}"
                        current_addresses.append(whale_address)
                        
                        # Rejestruj aktywność
                        self.address_tracker.record_address_activity(
                            address=whale_address,
                            token=symbol,
                            usd_value=max_order,
                            source="whale_ping"
                        )
                        
                except Exception as e:
                    print(f"[STEALTH DEBUG] inflow_momentum_boost orderbook processing error for {symbol}: {e}")
            
            print(f"[STEALTH DEBUG] inflow_momentum_boost for {symbol}: checking momentum patterns...")
            
            if not current_addresses:
                print(f"[STEALTH DEBUG] inflow_momentum_boost: no addresses found for {symbol}")
                return StealthSignal("inflow_momentum_boost", False, 0.0)
            
            # Analiza momentum
            momentum_boost, details = self.address_tracker.compute_inflow_momentum_boost(
                current_token=symbol,
                current_addresses=current_addresses
            )
            
            active = momentum_boost > 0.0
            strength = min(momentum_boost, 1.0)  # Normalizuj do 0-1
            
            print(f"[STEALTH DEBUG] inflow_momentum_boost: addresses={len(current_addresses)}, boost_score={momentum_boost:.2f}, active={active}")
            if active:
                print(f"[STEALTH DEBUG] inflow_momentum_boost DETECTED: {details['momentum_addresses']} momentum addresses, boost={momentum_boost:.2f}")
                
            return StealthSignal("inflow_momentum_boost", active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] inflow_momentum_boost error for {symbol}: {e}")
            return StealthSignal("inflow_momentum_boost", False, 0.0)

    def check_source_reliability_boost(self, token_data: Dict) -> StealthSignal:
        """
        🆕 PHASE 5/5: Dynamic Source Reliability
        
        Wykrywa adresy o wysokiej reputacji (smart money) które wcześniej dały trafne sygnały
        Adresy z historycznie trafnymi sygnałami otrzymują wyższe współczynniki wagowe
        
        Args:
            token_data: Dane rynkowe tokena
            
        Returns:
            StealthSignal z source reliability boost score
        """
        try:
            symbol = token_data.get("symbol", "UNKNOWN")
            
            # Identyfikacja aktywnych adresów
            current_addresses = []
            
            # Sprawdź dex_inflow addresses
            if "dex_inflow" in token_data and token_data["dex_inflow"] > 1000:
                dex_value = token_data["dex_inflow"]
                dex_address = f"dex_{symbol.lower().replace('usdt', '')}_{int(dex_value)}"
                current_addresses.append(dex_address)
            
            # Sprawdź whale_ping addresses na podstawie orderbook
            orderbook = token_data.get("orderbook", {})
            if orderbook.get("bids") and orderbook.get("asks"):
                try:
                    bids = orderbook.get("bids", [])
                    asks = orderbook.get("asks", [])
                    
                    # Oblicz max order value
                    all_orders = []
                    for bid in bids[:10]:  # Sprawdź top 10 bids for better whale detection
                        if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            price = float(bid[0])
                            size = float(bid[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    for ask in asks[:10]:  # Sprawdź top 10 asks for better whale detection
                        if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                            price = float(ask[0])
                            size = float(ask[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    max_order = max(all_orders, default=0)
                    if max_order > 10000:  # Próg dla whale orders
                        whale_address = f"whale_{symbol.lower().replace('usdt', '')}_{int(max_order)}"
                        current_addresses.append(whale_address)
                        
                except Exception as e:
                    print(f"[STEALTH DEBUG] source_reliability_boost orderbook processing error for {symbol}: {e}")
            
            print(f"[STEALTH DEBUG] source_reliability_boost for {symbol}: checking address reputation...")
            
            if not current_addresses:
                print(f"[STEALTH DEBUG] source_reliability_boost: no addresses found for {symbol}")
                return StealthSignal("source_reliability_boost", False, 0.0)
            
            # Analiza reputacji adresów
            reputation_boost, details = self.address_tracker.compute_reputation_boost(
                current_token=symbol,
                current_addresses=current_addresses
            )
            
            # 🔧 FIX: Sprawdzenie trafnych predykcji w ostatnich 24h
            if len(current_addresses) >= 1 and reputation_boost == 0:
                # Sprawdź czy jakieś adresy miały trafne predykcje w ostatnich 24h
                try:
                    from .address_trust_manager import AddressTrustManager
                    trust_manager = AddressTrustManager()
                    
                    successful_predictions = 0
                    for address in current_addresses:
                        # Sprawdź statystyki zaufania dla adresu
                        trust_stats = trust_manager.get_trust_statistics(address)
                        if trust_stats and trust_stats.get('success_rate', 0) > 0.5:  # >50% success rate
                            successful_predictions += 1
                    
                    if successful_predictions > 0:
                        reputation_boost = 0.1 * successful_predictions  # Bonus za trafne predykcje
                        print(f"[STEALTH DEBUG] source_reliability_boost: Found {successful_predictions} addresses with successful predictions → boost={reputation_boost:.2f}")
                except Exception as e:
                    print(f"[STEALTH DEBUG] source_reliability_boost: Trust check error: {e}")
            
            active = reputation_boost > 0.0
            strength = min(reputation_boost / 0.30, 1.0)  # Normalizuj do 0-1 (max boost 0.30)
            
            print(f"[STEALTH DEBUG] source_reliability_boost: addresses={len(current_addresses)}, boost_score={reputation_boost:.2f}, active={active}")
            if active:
                print(f"[STEALTH DEBUG] source_reliability_boost DETECTED: {details['reputation_addresses']} reputation addresses, boost={reputation_boost:.2f}")
                
            return StealthSignal("source_reliability_boost", active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] source_reliability_boost error for {symbol}: {e}")
            return StealthSignal("source_reliability_boost", False, 0.0)

    def check_cross_token_activity_boost(self, token_data: Dict) -> StealthSignal:
        """
        🆕 PUNKT 2/5: Cross-Token Activity Boost
        
        Wykrywa aktywność tych samych adresów na różnych tokenach w krótkim czasie
        Sugeruje szeroką akcję akumulacyjną przez tego samego gracza
        
        Args:
            token_data: Dane rynkowe tokena
            
        Returns:
            StealthSignal z cross-token boost score
        """
        try:
            from .address_tracker import AddressTracker
            
            symbol = token_data.get("symbol", "UNKNOWN")
            
            print(f"[STEALTH DEBUG] cross_token_activity_boost for {symbol}: checking cross-token patterns...")
            
            # Inicjalizuj tracker
            address_tracker = AddressTracker()
            
            # Pobierz aktywne adresy z obecnego skanu
            current_addresses = []
            
            # Dodaj adresy z dex_inflow jeśli był aktywny
            inflow_usd = token_data.get("dex_inflow", 0)
            if inflow_usd > 1000:
                address = f"dex_{symbol.lower()}_{int(inflow_usd)}"[:42]
                current_addresses.append(address)
                # Zapisz aktywność adresu w historii
                self.address_tracker.record_address_activity(
                    token=symbol,
                    address=address,
                    usd_value=inflow_usd,
                    source="dex_inflow"
                )
            
            # Dodaj adresy z whale_ping jeśli był aktywny
            orderbook = token_data.get("orderbook", {})
            if orderbook:
                try:
                    bids = orderbook.get("bids", [])
                    asks = orderbook.get("asks", [])
                    
                    all_orders = []
                    
                    # Sprawdź bidy
                    for bid in bids[:5]:  # Top 5 levels
                        if isinstance(bid, list) and len(bid) >= 2:
                            price = float(bid[0])
                            size = float(bid[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    # Sprawdź aski
                    for ask in asks[:5]:  # Top 5 levels
                        if isinstance(ask, list) and len(ask) >= 2:
                            price = float(ask[0])
                            size = float(ask[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    max_order = max(all_orders, default=0)
                    if max_order > 10000:  # Próg dla whale orders
                        address = f"whale_{symbol.lower()}_{int(max_order)}"[:42]
                        current_addresses.append(address)
                        # Zapisz aktywność adresu w historii
                        self.address_tracker.record_address_activity(
                            token=symbol,
                            address=address,
                            usd_value=max_order,
                            source="whale_ping"
                        )
                        
                except Exception as e:
                    print(f"[STEALTH DEBUG] cross_token_activity_boost orderbook error: {e}")
            
            # 🎯 KLUCZOWA FUNKCJA: Sprawdź aktywność cross-tokenową
            boost_score, details = address_tracker.get_cross_token_activity_boost(
                current_token=symbol,
                current_addresses=current_addresses,
                history_days=7,
                window_hours=48  # 48h okno korelacji
            )
            
            # Przelicz boost score na strength (0-1)
            strength = min(boost_score / 0.6, 1.0)  # Max boost 0.6 → strength 1.0
            
            is_active = boost_score > 0
            
            print(f"[STEALTH DEBUG] cross_token_activity_boost: addresses={len(current_addresses)}, boost_score={boost_score:.2f}, active={is_active}")
            
            if is_active:
                print(f"[STEALTH DEBUG] cross_token_activity_boost DETECTED: {details['cross_token_addresses']} cross-token addresses, boost={boost_score:.2f}")
            
            return StealthSignal(
                name="cross_token_activity_boost",
                active=is_active,
                strength=strength
            )
            
        except Exception as e:
            print(f"[STEALTH DEBUG] cross_token_activity_boost error: {e}")
            return StealthSignal(
                name="cross_token_activity_boost",
                active=False,
                strength=0.0
            )
    
    def check_multi_address_group_activity(self, token_data: Dict) -> StealthSignal:
        """
        🆕 STAGE 5: Multi-Address Group Activity Detection
        CRITICAL FIX: Add timeout protection to prevent hanging
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"multi_address_group_activity timeout for {symbol}")
        
        try:
            # EMERGENCY TIMEOUT: 2-second timeout for multi-address operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(2)
            
            print(f"[MULTI-ADDRESS DEBUG] Checking group activity for {symbol}...")
            
            # Import detektor multi-address z error handling
            try:
                from stealth_engine.multi_address_detector import detect_group_activity, get_group_statistics
            except ImportError as e:
                print(f"[MULTI-ADDRESS ERROR] Import failed for {symbol}: {e}")
                signal.alarm(0)
                return StealthSignal("multi_address_group_activity", False, 0.0)
            
            # Wykryj grupową aktywność dla tego tokena z timeout protection
            active_group, unique_addresses, total_events, group_intensity = detect_group_activity(symbol)
            
            # Strength based na intensywność grupowej aktywności
            strength = min(group_intensity, 1.0)
            
            # Debug output
            if active_group:
                print(f"[MULTI-ADDRESS DETECTED] {symbol}: {unique_addresses} addresses, {total_events} events, intensity={group_intensity:.3f}")
            else:
                print(f"[MULTI-ADDRESS DEBUG] {symbol}: No coordinated group activity detected")
            
            signal.alarm(0)  # Cancel timeout
            return StealthSignal("multi_address_group_activity", active_group, strength)
            
        except TimeoutError:
            signal.alarm(0)
            print(f"[MULTI-ADDRESS TIMEOUT] {symbol} - group activity check timed out, returning fallback")
            return StealthSignal("multi_address_group_activity", False, 0.0)
        except Exception as e:
            signal.alarm(0)
            print(f"[MULTI-ADDRESS ERROR] Group activity check failed for {symbol}: {e}")
            return StealthSignal("multi_address_group_activity", False, 0.0)

    def get_signal_definitions(self) -> Dict:
        """Pobierz definicje wszystkich sygnałów"""
        return self.signal_definitions