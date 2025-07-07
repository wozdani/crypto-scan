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
            ("liquidity_absorption", self.check_liquidity_absorption)
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
    
    def check_whale_ping(self, token_data: Dict) -> StealthSignal:
        """
        Whale ping detector - wykrycie wielorybów przez duże zlecenia
        Matematycznie precyzyjna implementacja zgodna z user specification v2
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        orderbook = token_data.get("orderbook", {})
        
        print(f"[STEALTH DEBUG] whale_ping for {symbol}: checking orderbook for large orders...")
        
        if not orderbook:
            print(f"[STEALTH DEBUG] whale_ping for {symbol}: no orderbook data")
            return StealthSignal("whale_ping", False, 0.0)
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            print(f"[STEALTH DEBUG] whale_ping for {symbol}: insufficient orderbook data")
            return StealthSignal("whale_ping", False, 0.0)
        
        try:
            # Handle different orderbook formats (list vs dict)
            if isinstance(bids, dict):
                # Convert dict format to list format with safe key processing
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
                # Convert dict format to list format with safe key processing
                try:
                    asks_list = []
                    for key in sorted(asks.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0):
                        if isinstance(asks[key], list) and len(asks[key]) >= 2:
                            asks_list.append(asks[key])
                    asks = asks_list if asks_list else []
                except Exception as e:
                    print(f"[STEALTH DEBUG] whale_ping asks conversion error for {symbol}: {e}")
                    asks = []
            
            # Verify we have valid orderbook data after conversion
            if not bids or not asks:
                print(f"[STEALTH DEBUG] whale_ping for {symbol}: no valid bids/asks after conversion")
                return StealthSignal("whale_ping", False, 0.0)
            
            # Now process as list format - Oblicz max_bid_usd i max_ask_usd zgodnie ze specyfikacją
            best_bid_price = float(bids[0][0])
            best_bid_size = float(bids[0][1])
            max_bid_usd = best_bid_price * best_bid_size
            
            best_ask_price = float(asks[0][0])
            best_ask_size = float(asks[0][1])
            max_ask_usd = best_ask_price * best_ask_size
            
            # Warunek aktywacji: max(max_bid_usd, max_ask_usd) > 100_000
            max_order_usd = max(max_bid_usd, max_ask_usd)
            is_whale = max_order_usd > 100_000
            
            # Strength: logarytmicznie min(1.0, log10(max_order_usd / 10_000) / 2)
            if is_whale:
                import math
                strength = min(1.0, math.log10(max_order_usd / 10_000) / 2)
                print(f"[STEALTH DEBUG] whale_ping DETECTED for {symbol}: max_order_usd=${max_order_usd:,.0f} > $100,000")
            else:
                strength = 0.0
            
            print(f"[STEALTH DEBUG] whale_ping result for {symbol}: active={is_whale}, strength={strength:.3f}, max_order_usd=${max_order_usd:,.0f}")
            return StealthSignal("whale_ping", is_whale, strength)
            
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
            print(f"[STEALTH DEBUG] spoofing_layers for {symbol}: insufficient layers")
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
            
            if is_active:
                side_detected = "bids" if bid_spoofing else "asks"
                layers_vol = bid_layers_vol if bid_spoofing else ask_layers_vol
                total_vol = bid_total_vol if bid_spoofing else ask_total_vol
                print(f"[STEALTH DEBUG] spoofing_layers DETECTED for {symbol}: {side_detected} layers_volume={layers_vol:.0f}, total_side_volume={total_vol:.0f}")
            
            print(f"[STEALTH DEBUG] spoofing_layers result for {symbol}: active={is_active}, strength={strength:.3f}")
            return StealthSignal("spoofing_layers", is_active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] spoofing_layers error for {symbol}: {e}")
            return StealthSignal("spoofing_layers", False, 0.0)
    
    def check_volume_slope(self, token_data: Dict) -> StealthSignal:
        """
        Wolumen rosnący bez zmiany ceny
        """
        slope = token_data.get("volume_slope_up", False)
        return StealthSignal("volume_slope", slope, 1.0 if slope else 0.0)
    
    def check_ghost_orders(self, token_data: Dict) -> StealthSignal:
        """
        Martwe poziomy z nietypową aktywnością
        """
        ghost = token_data.get("ghost_orders", False)
        return StealthSignal("ghost_orders", ghost, 1.0 if ghost else 0.0)
    
    def check_dex_inflow(self, token_data: Dict) -> StealthSignal:
        """
        DEX inflow detector - detekcja przepływu tokenów on-chain do CEX/DEX
        Matematycznie precyzyjna implementacja zgodna z user specification v2
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        inflow_value = token_data.get("dex_inflow", 0)
        
        print(f"[STEALTH DEBUG] dex_inflow for {symbol}: checking inflow value={inflow_value}")
        
        if inflow_value is None:
            inflow_value = 0
        
        try:
            inflow_usd = float(inflow_value)
        except:
            inflow_usd = 0.0
        
        # Warunek aktywacji: inflow_usd > 30_000
        is_active = inflow_usd > 30_000
        
        # Strength: min(1.0, inflow_usd / 100_000)
        if is_active:
            strength = min(1.0, inflow_usd / 100_000)
            print(f"[STEALTH DEBUG] dex_inflow DETECTED for {symbol}: inflow_usd=${inflow_usd:,.0f} > $30,000")
        else:
            strength = 0.0
            print(f"[STEALTH DEBUG] dex_inflow BELOW THRESHOLD for {symbol}: inflow_usd=${inflow_usd:,.0f} <= $30,000")
        
        print(f"[STEALTH DEBUG] dex_inflow result for {symbol}: active={is_active}, strength={strength:.3f}, inflow_usd=${inflow_usd:,.0f}")
        return StealthSignal("dex_inflow", is_active, strength)
    
    def check_event_tag(self, token_data: Dict) -> StealthSignal:
        """
        Event tag detection - unlock tokenów / airdrop
        """
        tag = token_data.get("event_tag", None)
        return StealthSignal("event_tag", tag is not None, 1.0 if tag else 0.0)
    
    def check_orderbook_imbalance_stealth(self, token_data: Dict) -> StealthSignal:
        """
        Sprawdź asymetrię orderbook - wersja stealth
        """
        orderbook = token_data.get('orderbook', {})
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return StealthSignal("orderbook_imbalance", False, 0.0)
        
        try:
            # Handle different orderbook formats (list vs dict)
            if isinstance(bids, dict):
                bids_list = [bids[key] for key in sorted(bids.keys(), key=lambda x: float(x) if x.isdigit() else 0, reverse=True)]
                bids = bids_list
            
            if isinstance(asks, dict):
                asks_list = [asks[key] for key in sorted(asks.keys(), key=lambda x: float(x) if x.isdigit() else 0)]
                asks = asks_list
            
            # Oblicz siłę bid vs ask
            bid_strength = sum(float(bid[1]) for bid in bids[:5])
            ask_strength = sum(float(ask[1]) for ask in asks[:5])
            
            if bid_strength + ask_strength == 0:
                return StealthSignal("orderbook_imbalance", False, 0.0)
            
            imbalance_ratio = abs(bid_strength - ask_strength) / (bid_strength + ask_strength)
            active = imbalance_ratio > 0.6  # 60% threshold
            strength = min(imbalance_ratio, 1.0)
            
            return StealthSignal("orderbook_imbalance", active, strength)
        except:
            return StealthSignal("orderbook_imbalance", False, 0.0)
    
    def check_large_bid_walls_stealth(self, token_data: Dict) -> StealthSignal:
        """
        Wykryj duże mury bid wspierające cenę - wersja stealth
        """
        orderbook = token_data.get('orderbook', {})
        bids = orderbook.get('bids', [])
        
        if len(bids) < 3:
            return StealthSignal("large_bid_walls", False, 0.0)
        
        try:
            # Sprawdź czy są duże bidy w top 3 poziomach
            large_bids = sum(1 for bid in bids[:3] if float(bid[1]) > 10.0)
            active = large_bids >= 2
            strength = large_bids / 3.0
            
            return StealthSignal("large_bid_walls", active, strength)
        except:
            return StealthSignal("large_bid_walls", False, 0.0)
    
    def check_ask_wall_removal(self, token_data: Dict) -> StealthSignal:
        """
        Wykryj usunięcie murów ask (placeholder - wymaga historycznych danych)
        """
        # Placeholder - w rzeczywistości potrzebne są dane historyczne orderbook
        active = token_data.get("ask_walls_removed", False)
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
            if is_active:
                strength = min(1.0, vol_current / avg_volume - 1)
                print(f"[STEALTH DEBUG] volume_spike DETECTED for {symbol}: vol_current={vol_current:.0f} > 2×avg_volume={avg_volume:.0f}")
            else:
                strength = 0.0
                
            print(f"[STEALTH DEBUG] volume_spike result for {symbol}: active={is_active}, strength={strength:.3f}, vol_current={vol_current:.0f}, avg_volume={avg_volume:.0f}")
            return StealthSignal("volume_spike", is_active, strength)
            
        except Exception as e:
            print(f"[STEALTH DEBUG] volume_spike error for {symbol}: {e}")
            return StealthSignal("volume_spike", False, 0.0)
    
    def check_bid_ask_spread_tightening_stealth(self, token_data: Dict) -> StealthSignal:
        """
        Wykryj zwężenie spreadu bid-ask - wersja stealth
        """
        orderbook = token_data.get('orderbook', {})
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return StealthSignal("bid_ask_spread_tightening", False, 0.0)
        
        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            if best_bid == 0:
                return StealthSignal("bid_ask_spread_tightening", False, 0.0)
            
            spread_percentage = (best_ask - best_bid) / best_bid * 100
            
            # Tight spread < 0.1%
            active = spread_percentage < 0.1
            strength = max(0.0, (0.1 - spread_percentage) / 0.1) if active else 0.0
            
            return StealthSignal("bid_ask_spread_tightening", active, strength)
        except:
            return StealthSignal("bid_ask_spread_tightening", False, 0.0)
    
    def check_liquidity_absorption(self, token_data: Dict) -> StealthSignal:
        """
        Wykryj absorpcję płynności (placeholder)
        """
        # Placeholder - wymaga analizy zmian w orderbook
        active = token_data.get("liquidity_absorbed", False)
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
            # Handle different orderbook formats (list vs dict)
            if isinstance(bids, dict):
                # Convert dict format to list format
                bids_list = [bids[key] for key in sorted(bids.keys(), key=lambda x: float(x) if x.isdigit() else 0, reverse=True)]
                bids = bids_list
            
            if isinstance(asks, dict):
                # Convert dict format to list format
                asks_list = [asks[key] for key in sorted(asks.keys(), key=lambda x: float(x) if x.isdigit() else 0)]
                asks = asks_list
            
            # Oblicz spread_pct i imbalance_pct zgodnie ze specyfikacją
            bid_price = float(bids[0][0])
            ask_price = float(asks[0][0])
            mid_price = (bid_price + ask_price) / 2
            
            # spread_pct = (ask_price - bid_price) / mid_price
            spread_pct = (ask_price - bid_price) / mid_price
            
            # Oblicz total bids i total asks volume
            total_bids = sum(float(bid[1]) for bid in bids[:10])  # Top 10 levels
            total_asks = sum(float(ask[1]) for ask in asks[:10])  # Top 10 levels
            
            # imbalance_pct = abs(total_bids - total_asks) / (total_bids + total_asks)
            total_volume = total_bids + total_asks
            imbalance_pct = abs(total_bids - total_asks) / total_volume if total_volume > 0 else 0.0
            
            # Warunek aktywacji: spread_pct < 0.0005 and imbalance_pct > 0.85
            is_active = spread_pct < 0.0005 and imbalance_pct > 0.85
            
            # Strength: min(1.0, imbalance_pct × 2)
            if is_active:
                strength = min(1.0, imbalance_pct * 2)
                print(f"[STEALTH DEBUG] orderbook_anomaly DETECTED for {symbol}: spread_pct={spread_pct:.6f} < 0.0005, imbalance_pct={imbalance_pct:.3f} > 0.85")
            else:
                strength = 0.0
            
            print(f"[STEALTH DEBUG] orderbook_anomaly result for {symbol}: active={is_active}, strength={strength:.3f}, spread_pct={spread_pct:.6f}, imbalance_pct={imbalance_pct:.3f}")
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
            # Handle different orderbook formats (list vs dict)
            if isinstance(bids, dict):
                bids_list = [bids[key] for key in sorted(bids.keys(), key=lambda x: float(x) if x.isdigit() else 0, reverse=True)]
                bids = bids_list
            
            if isinstance(asks, dict):
                asks_list = [asks[key] for key in sorted(asks.keys(), key=lambda x: float(x) if x.isdigit() else 0)]
                asks = asks_list
            
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
    
    def get_signal_definitions(self) -> Dict:
        """Pobierz definicje wszystkich sygnałów"""
        return self.signal_definitions