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
        
        # Sprawdź każdy sygnał z obsługą błędów
        for signal_name, signal_func in signal_functions:
            try:
                signal = signal_func(token_data)
                if signal is not None:
                    signals.append(signal)
                else:
                    print(f"[STEALTH WARNING] {symbol}: {signal_name} returned None, creating empty signal")
                    signals.append(StealthSignal(signal_name, False, 0.0))
            except Exception as e:
                print(f"[STEALTH ERROR] {symbol}: Failed to check {signal_name}: {e}")
                # Dodaj pustý sygnał aby utrzymać spójność
                signals.append(StealthSignal(signal_name, False, 0.0))
        
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
        symbol = token_data.get("symbol", "UNKNOWN")
        
        try:
            print(f"[STEALTH DEBUG] whale_ping for {symbol}: checking whale activity...")
            
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
                print(f"[STEALTH DEBUG] whale_ping for {symbol}: insufficient orderbook data")
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
            
            # Warunek aktywacji z dynamicznym progiem
            active = max_order_usd > threshold
            
            # Strength: max_order / (threshold * 2)
            strength = min(max_order_usd / (threshold * 2), 1.0) if threshold > 0 else 0.0
            
            # Address tracking dla whale ping
            if active and max_order_usd > 0:
                try:
                    from .address_tracker import address_tracker
                    # Symulujemy adres na podstawie whale order (w rzeczywistości byłby to prawdziwy adres z orderbook)
                    mock_address = f"whale_{symbol.lower()}_{int(max_order_usd)}"[:42]  # Symulacja adresu
                    self.address_tracker.record_address_activity(
                        token=symbol,
                        address=mock_address,
                        usd_value=max_order_usd,
                        source="whale_ping"
                    )
                except Exception as addr_e:
                    print(f"[STEALTH DEBUG] whale_ping address tracking error for {symbol}: {addr_e}")
                
                # Whale Memory System - zapamiętaj adres wieloryba
                try:
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    parent_dir = os.path.dirname(current_dir)
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    from utils.whale_memory import update_whale_memory, is_repeat_whale, get_repeat_whale_boost
                    from stealth_engine.address_trust_manager import record_address_prediction, get_address_boost
                    mock_address = f"whale_{symbol.lower()}_{int(max_order_usd)}"[:42]
                    
                    # Aktualizuj pamięć wieloryba
                    repeat_count = update_whale_memory(symbol, mock_address, source="whale_ping")
                    
                    # Sprawdź czy to powtarzający się wieloryb
                    if is_repeat_whale(symbol, mock_address):
                        repeat_boost = get_repeat_whale_boost(symbol, mock_address)
                        # Zwiększ strength o bonus za powtarzającego się wieloryba
                        strength = min(1.0, strength + repeat_boost * 0.3)  # Max 30% boost
                        print(f"[WHALE MEMORY] {symbol} repeat whale detected! Boost: +{repeat_boost*0.3:.2f} → strength: {strength:.3f}")
                        
                        # Etap 4: Zwiększ priorytet tokena w kolejce skanowania
                        try:
                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            parent_dir = os.path.dirname(current_dir)
                            if parent_dir not in sys.path:
                                sys.path.insert(0, parent_dir)
                            from utils.token_priority_manager import update_token_priority
                            
                            # Boost priorytetu w zależności od repeat_boost (0.2-1.0 → 10-20 priority)
                            priority_boost = 10 + (repeat_boost * 10)  # 10-20 boost range
                            update_token_priority(symbol, priority_boost, "whale_ping_repeat")
                        except Exception as priority_e:
                            print(f"[TOKEN PRIORITY] Error updating whale_ping priority for {symbol}: {priority_e}")
                    
                    # Etap 5: Rejestruj aktywność adresu dla multi-address detection
                    try:
                        from stealth_engine.multi_address_detector import record_address_activity
                        record_address_activity(symbol, mock_address)
                        print(f"[MULTI-ADDRESS] Recorded whale_ping activity for {symbol}: {mock_address}")
                    except Exception as multi_e:
                        print(f"[MULTI-ADDRESS ERROR] whale_ping for {symbol}: {multi_e}")
                    
                    # Etap 6: Trust Scoring - rejestruj predykcję i oblicz boost
                    try:
                        # Rejestruj predykcję adresu dla feedback loop
                        record_address_prediction(symbol, mock_address)
                        
                        # Pobierz trust boost na podstawie historycznej skuteczności
                        trust_boost = get_address_boost(mock_address)
                        if trust_boost > 0:
                            strength = min(1.0, strength + trust_boost)
                            print(f"[TRUST BOOST] {symbol} whale_ping: Applied +{trust_boost:.3f} trust boost → strength: {strength:.3f}")
                        
                    except Exception as trust_e:
                        print(f"[TRUST BOOST ERROR] whale_ping for {symbol}: {trust_e}")
                    
                    # Etap 7: Trigger Alert System - smart money detection
                    try:
                        from stealth_engine.trigger_alert_system import check_smart_money_trigger, apply_smart_money_boost
                        from stealth_engine.address_trust_manager import get_trust_manager
                        
                        # Sprawdź czy wykryto trigger addresses (smart money)
                        trust_manager = get_trust_manager()
                        is_trigger, trigger_addresses = check_smart_money_trigger([mock_address], trust_manager)
                        
                        if is_trigger:
                            # Zastosuj trigger boost - minimum score 3.0 dla instant alert
                            boosted_strength, priority_alert = apply_smart_money_boost(
                                symbol, strength, trigger_addresses, "whale_ping"
                            )
                            strength = boosted_strength
                            
                            print(f"[TRIGGER ALERT] 🚨 {symbol} WHALE PING: Smart money detected! "
                                  f"Strength boosted to {strength:.3f} (priority alert: {priority_alert})")
                        
                    except Exception as trigger_e:
                        print(f"[TRIGGER ALERT ERROR] whale_ping for {symbol}: {trigger_e}")
                    
                except Exception as memory_e:
                    print(f"[WHALE MEMORY] Error for {symbol}: {memory_e}")
            
            print(f"[STEALTH DEBUG] whale_ping: max_order=${max_order_usd:.0f}, dynamic_threshold=${threshold:.0f}, active={active}")
            if active:
                print(f"[STEALTH DEBUG] whale_ping DETECTED: max_order=${max_order_usd:.0f} > dynamic_threshold=${threshold:.0f}")
            return StealthSignal("whale_ping", active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] whale_ping error for {symbol}: {e}")
            return StealthSignal("whale_ping", False, 0.0)

    
    def check_spoofing_layers(self, token_data: Dict) -> StealthSignal:
        """
        Spoofing layers detector - detekcja warstwowania zleceń
        Matematycznie precyzyjna implementacja zgodna z user specification v2
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        orderbook = token_data.get("orderbook", {})
        
        print(f"[STEALTH DEBUG] spoofing_layers for {symbol}: checking layered orders...")
        
        if not orderbook:
            print(f"[STEALTH DEBUG] spoofing_layers for {symbol}: no orderbook data")
            return StealthSignal("spoofing_layers", False, 0.0)
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if len(bids) < 3 and len(asks) < 3:
            print(f"[STEALTH DEBUG] spoofing_layers: insufficient levels")
            return StealthSignal("spoofing_layers", False, 0.0)
        
        try:
            def analyze_side_spoofing(orders, side_name):
                if len(orders) < 3:
                    return False, 0.0, 0.0, 0.0
                
                # Oblicz total volume dla tej strony
                total_side_volume = sum(float(order[1]) for order in orders)
                layers_volume = 0.0
                layer_count = 0
                
                # Sprawdź pierwsze 10 poziomów
                for i in range(min(len(orders), 10)):
                    if i == 0:
                        continue  # Skip first level
                    
                    base_price = float(orders[0][0])
                    current_price = float(orders[i][0])
                    current_volume = float(orders[i][1])
                    
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
            
            # Aktywacja jeśli którakolwiek strona ma spoofing
            is_active = bid_spoofing or ask_spoofing
            
            # Wybierz wyższą strength
            strength = max(bid_strength, ask_strength)
            
            if len(bids) >= 3 and len(asks) >= 3:
                bid_layer_ratio = bids[0][1] / bids[1][1] if len(bids) > 1 and bids[1][1] > 0 else 0
                ask_layer_ratio = asks[0][1] / asks[1][1] if len(asks) > 1 and asks[1][1] > 0 else 0
                spoof_detected = bid_layer_ratio > 5 or ask_layer_ratio > 5
                print(f"[STEALTH DEBUG] spoofing_layers: bid_ratio={bid_layer_ratio:.2f}, ask_ratio={ask_layer_ratio:.2f}, detected={spoof_detected}")
            
            if is_active:
                side_detected = "bids" if bid_spoofing else "asks"
                layers_vol = bid_layers_vol if bid_spoofing else ask_layers_vol
                total_vol = bid_total_vol if bid_spoofing else ask_total_vol
                print(f"[STEALTH DEBUG] spoofing_layers DETECTED: {side_detected} layers_volume={layers_vol:.0f}, total_side_volume={total_vol:.0f}")
            
            return StealthSignal("spoofing_layers", is_active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] spoofing_layers error for {symbol}: {e}")
            return StealthSignal("spoofing_layers", False, 0.0)
    
    def check_volume_slope(self, token_data: Dict) -> StealthSignal:
        """
        Wolumen rosnący bez zmiany ceny
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        slope = token_data.get("volume_slope_up", False)
        
        print(f"[STEALTH DEBUG] volume_slope: volume_slope_up={slope}")
        return StealthSignal("volume_slope", slope, 1.0 if slope else 0.0)
    
    def check_ghost_orders(self, token_data: Dict) -> StealthSignal:
        """
        Martwe poziomy z nietypową aktywnością
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        ghost = token_data.get("ghost_orders", False)
        
        print(f"[STEALTH DEBUG] ghost_orders: ghost_orders={ghost}")
        return StealthSignal("ghost_orders", ghost, 1.0 if ghost else 0.0)
    
    def check_dex_inflow(self, token_data: Dict) -> StealthSignal:
        """
        DEX inflow detector - detekcja nagłych napływów DEX względem historii
        Dynamiczna wersja z kontekstem historycznym + tracking adresów
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        
        try:
            print(f"[STEALTH DEBUG] dex_inflow for {symbol}: checking on-chain inflow...")
            
            # Pobierz wartość dex_inflow z token_data
            inflow_usd = token_data.get("dex_inflow", 0)
            
            # Pobierz historię dex_inflow (ostatnie 8 wartości)
            inflow_history = token_data.get("dex_inflow_history", [])[-8:]
            
            # Oblicz średnią z ostatnich wartości
            avg_recent = sum(inflow_history) / len(inflow_history) if inflow_history else 0
            
            # Warunek aktywacji: inflow > avg_recent * 2 AND inflow > 1000
            spike_detected = inflow_usd > avg_recent * 2 and inflow_usd > 1000
            
            # Strength: min(inflow_usd / (avg_recent * 5 + 1), 1.0) - zmniejszona by zostawić miejsce na boost
            strength = min(inflow_usd / (avg_recent * 5 + 1), 0.8) if avg_recent > 0 else 0.0
            
            # Address tracking dla DEX inflow
            if spike_detected and inflow_usd > 0:
                try:
                    from .address_tracker import address_tracker
                    # Symulujemy adres na podstawie danych DEX inflow (w rzeczywistości byłby to prawdziwy adres z blockchain)
                    mock_address = f"dex_{symbol.lower()}_{int(inflow_usd)}"[:42]  # Symulacja adresu
                    self.address_tracker.record_address_activity(
                        token=symbol,
                        address=mock_address,
                        usd_value=inflow_usd,
                        source="dex_inflow"
                    )
                except Exception as addr_e:
                    print(f"[STEALTH DEBUG] dex_inflow address tracking error for {symbol}: {addr_e}")
                
                # Whale Memory System - zapamiętaj adres DEX inflow
                try:
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    parent_dir = os.path.dirname(current_dir)
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    from utils.whale_memory import update_whale_memory, is_repeat_whale, get_repeat_whale_boost
                    from stealth_engine.address_trust_manager import record_address_prediction, get_address_boost
                    mock_address = f"dex_{symbol.lower()}_{int(inflow_usd)}"[:42]
                    
                    # Aktualizuj pamięć wieloryba
                    repeat_count = update_whale_memory(symbol, mock_address, source="dex_inflow")
                    
                    # Sprawdź czy to powtarzający się wieloryb
                    if is_repeat_whale(symbol, mock_address):
                        repeat_boost = get_repeat_whale_boost(symbol, mock_address)
                        # Zwiększ strength o bonus za powtarzającego się wieloryba
                        strength = min(1.0, strength + repeat_boost * 0.25)  # Max 25% boost dla DEX
                        print(f"[WHALE MEMORY] {symbol} repeat DEX whale detected! Boost: +{repeat_boost*0.25:.2f} → strength: {strength:.3f}")
                        
                        # Etap 4: Zwiększ priorytet tokena w kolejce skanowania
                        try:
                            from utils.token_priority_manager import update_token_priority
                            
                            # Boost priorytetu w zależności od repeat_boost (0.2-1.0 → 8-16 priority dla DEX)
                            priority_boost = 8 + (repeat_boost * 8)  # 8-16 boost range (nieco mniej niż whale_ping)
                            update_token_priority(symbol, priority_boost, "dex_inflow_repeat")
                        except Exception as priority_e:
                            print(f"[TOKEN PRIORITY] Error updating dex_inflow priority for {symbol}: {priority_e}")
                    
                    # Etap 5: Rejestruj aktywność adresu dla multi-address detection
                    try:
                        from stealth_engine.multi_address_detector import record_address_activity
                        record_address_activity(symbol, mock_address)
                        print(f"[MULTI-ADDRESS] Recorded dex_inflow activity for {symbol}: {mock_address}")
                    except Exception as multi_e:
                        print(f"[MULTI-ADDRESS ERROR] dex_inflow for {symbol}: {multi_e}")
                    
                    # Etap 6: Trust Scoring - rejestruj predykcję i oblicz boost
                    try:
                        # Rejestruj predykcję adresu dla feedback loop
                        record_address_prediction(symbol, mock_address)
                        
                        # Pobierz trust boost na podstawie historycznej skuteczności
                        trust_boost = get_address_boost(mock_address)
                        if trust_boost > 0:
                            strength = min(1.0, strength + trust_boost)
                            print(f"[TRUST BOOST] {symbol} dex_inflow: Applied +{trust_boost:.3f} trust boost → strength: {strength:.3f}")
                        
                    except Exception as trust_e:
                        print(f"[TRUST BOOST ERROR] dex_inflow for {symbol}: {trust_e}")
                    
                    # Etap 7: Trigger Alert System - smart money detection
                    try:
                        from stealth_engine.trigger_alert_system import check_smart_money_trigger, apply_smart_money_boost
                        from stealth_engine.address_trust_manager import get_trust_manager
                        
                        # Sprawdź czy wykryto trigger addresses (smart money)
                        trust_manager = get_trust_manager()
                        is_trigger, trigger_addresses = check_smart_money_trigger([mock_address], trust_manager)
                        
                        if is_trigger:
                            # Zastosuj trigger boost - minimum score 3.0 dla instant alert
                            boosted_strength, priority_alert = apply_smart_money_boost(
                                symbol, strength, trigger_addresses, "dex_inflow"
                            )
                            strength = boosted_strength
                            
                            print(f"[TRIGGER ALERT] 🚨 {symbol} DEX INFLOW: Smart money detected! "
                                  f"Strength boosted to {strength:.3f} (priority alert: {priority_alert})")
                        
                    except Exception as trigger_e:
                        print(f"[TRIGGER ALERT ERROR] dex_inflow for {symbol}: {trigger_e}")
                    
                except Exception as memory_e:
                    print(f"[WHALE MEMORY] DEX error for {symbol}: {memory_e}")
            
            print(f"[STEALTH DEBUG] dex_inflow: inflow=${inflow_usd}, avg_recent=${avg_recent:.0f}, spike_detected={spike_detected}")
            if spike_detected:
                print(f"[STEALTH DEBUG] dex_inflow DETECTED: inflow=${inflow_usd} > avg_recent*2=${avg_recent*2:.0f} AND inflow > 1000")
            return StealthSignal("dex_inflow", spike_detected, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] dex_inflow error for {symbol}: {e}")
            return StealthSignal("dex_inflow", False, 0.0)
    
    def check_event_tag(self, token_data: Dict) -> StealthSignal:
        """
        Event tag detection - unlock tokenów / airdrop
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        tag = token_data.get("event_tag", None)
        
        print(f"[STEALTH DEBUG] event_tag: event_tag={tag}")
        return StealthSignal("event_tag", tag is not None, 1.0 if tag else 0.0)
    
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
        
        if len(bids) < 3:
            print(f"[STEALTH DEBUG] large_bid_walls: insufficient bid levels ({len(bids)} < 3)")
            return StealthSignal("large_bid_walls", False, 0.0)
        
        try:
            # Sprawdź czy są duże bidy w top 3 poziomach z enhanced validation
            large_bids = 0
            for i, bid in enumerate(bids[:3]):
                if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                    try:
                        if float(bid[1]) > 10.0:
                            large_bids += 1
                    except (ValueError, TypeError) as e:
                        print(f"[STEALTH DEBUG] large_bid_walls for {symbol}: invalid bid[{i}][1] value: {bid[1]}")
                else:
                    print(f"[STEALTH DEBUG] large_bid_walls for {symbol}: invalid bid[{i}] structure: {type(bid)}, content: {bid}")
            
            active = large_bids >= 2
            strength = large_bids / 3.0
            
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
        symbol = token_data.get("symbol", "UNKNOWN")
        # Placeholder - w rzeczywistości potrzebne są dane historyczne orderbook
        active = token_data.get("ask_walls_removed", False)
        
        print(f"[STEALTH DEBUG] ask_wall_removal: ask_walls_removed={active}")
        return StealthSignal("ask_wall_removal", active, 1.0 if active else 0.0)
    
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
        symbol = token_data.get("symbol", "UNKNOWN")
        # Placeholder - wymaga analizy zmian w orderbook
        active = token_data.get("liquidity_absorbed", False)
        
        print(f"[STEALTH DEBUG] liquidity_absorption: liquidity_absorbed={active}")
        return StealthSignal("liquidity_absorption", active, 1.0 if active else 0.0)
    
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
                    for bid in bids[:3]:  # Sprawdź top 3 bids
                        if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            price = float(bid[0])
                            size = float(bid[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    for ask in asks[:3]:  # Sprawdź top 3 asks
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
            
            active = boost_score > 0
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
            
            # Próg aktywacji: velocity_boost > 0.1
            is_active = velocity_boost > 0.1
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
                    for bid in bids[:3]:  # Sprawdź top 3 bids
                        if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            price = float(bid[0])
                            size = float(bid[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    for ask in asks[:3]:  # Sprawdź top 3 asks
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
                    for bid in bids[:3]:  # Sprawdź top 3 bids
                        if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            price = float(bid[0])
                            size = float(bid[1])
                            usd_value = price * size
                            all_orders.append(usd_value)
                    
                    for ask in asks[:3]:  # Sprawdź top 3 asks
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
        
        Wykrywa skoordynowaną aktywność grup adresów na tym samym tokenie.
        Jeśli 3+ różnych adresów pojawia się w transakcjach na ten sam token 
        więcej niż X razy w 72 godzinach, system traktuje to jako "koordynowaną aktywność"
        lub wzór akumulacji przez większą grupę (fundusz, market maker, whale consortium).
        
        Args:
            token_data: Dane rynkowe tokena
            
        Returns:
            StealthSignal z oceną intensywności grup adresów
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        
        try:
            print(f"[MULTI-ADDRESS DEBUG] Checking group activity for {symbol}...")
            
            # Import detektor multi-address z error handling
            try:
                from stealth_engine.multi_address_detector import detect_group_activity, get_group_statistics
            except ImportError as e:
                print(f"[MULTI-ADDRESS ERROR] Import failed for {symbol}: {e}")
                return StealthSignal("multi_address_group_activity", False, 0.0)
            
            # Wykryj grupową aktywność dla tego tokena
            active_group, unique_addresses, total_events, group_intensity = detect_group_activity(symbol)
            
            # Dodatkowe statystyki dla debugowania
            try:
                stats = get_group_statistics(symbol)
                print(f"[MULTI-ADDRESS DEBUG] {symbol} group stats: unique_addresses={unique_addresses}, total_events={total_events}, group_intensity={group_intensity:.3f}")
                print(f"[MULTI-ADDRESS DEBUG] {symbol} detailed stats: {stats}")
            except Exception as stats_e:
                print(f"[MULTI-ADDRESS ERROR] Stats error for {symbol}: {stats_e}")
            
            # Strength based na intensywność grupowej aktywności
            # group_intensity jest już w zakresie 0.0-1.0 
            strength = min(group_intensity, 1.0)
            
            # Debug output
            if active_group:
                print(f"[MULTI-ADDRESS DETECTED] {symbol}: {unique_addresses} addresses, {total_events} events, intensity={group_intensity:.3f}")
            else:
                print(f"[MULTI-ADDRESS DEBUG] {symbol}: No coordinated group activity detected")
            
            return StealthSignal("multi_address_group_activity", active_group, strength)
            
        except Exception as e:
            print(f"[MULTI-ADDRESS ERROR] Group activity check failed for {symbol}: {e}")
            return StealthSignal("multi_address_group_activity", False, 0.0)

    def get_signal_definitions(self) -> Dict:
        """Pobierz definicje wszystkich sygnałów"""
        return self.signal_definitions