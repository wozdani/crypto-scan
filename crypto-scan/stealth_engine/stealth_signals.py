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
        signals = []
        
        # Lista głównych sygnałów stealth - zaktualizowana z nowymi definicjami
        signals.append(self.check_whale_ping(token_data))
        signals.append(self.check_spoofing_layers(token_data))
        signals.append(self.check_dex_inflow(token_data))
        signals.append(self.check_orderbook_anomaly(token_data))  # Nowa funkcja
        signals.append(self.check_volume_slope(token_data))
        signals.append(self.check_ghost_orders(token_data))
        signals.append(self.check_event_tag(token_data))
        
        # Dodaj podstawowe detektory z istniejącej implementacji
        signals.append(self.check_orderbook_imbalance_stealth(token_data))
        signals.append(self.check_large_bid_walls_stealth(token_data))
        signals.append(self.check_ask_wall_removal(token_data))
        signals.append(self.check_volume_spike_stealth(token_data))
        signals.append(self.check_bid_ask_spread_tightening_stealth(token_data))
        signals.append(self.check_liquidity_absorption(token_data))
        
        return signals
    
    def check_whale_ping(self, token_data: Dict) -> StealthSignal:
        """
        Whale ping detector - wykrycie wielorybów przez duże zlecenia (>= $100,000)
        Nowa implementacja zgodna z user specification
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        orderbook = token_data.get("orderbook", {})
        
        print(f"[STEALTH DEBUG] whale_ping for {symbol}: checking orderbook for large orders...")
        
        if not orderbook:
            print(f"[STEALTH DEBUG] whale_ping for {symbol}: no orderbook data")
            return StealthSignal("whale_ping", False, 0.0)
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        whale_threshold_usd = 100_000  # $100k threshold
        
        active = False
        max_order_value = 0
        
        # Sprawdź obie strony orderbook
        for side_name, side in [("bids", bids), ("asks", asks)]:
            for price, size in side:
                try:
                    order_value = float(price) * float(size)
                    max_order_value = max(max_order_value, order_value)
                    
                    if order_value >= whale_threshold_usd:
                        active = True
                        print(f"[STEALTH DEBUG] whale_ping DETECTED for {symbol}: {side_name} order ${order_value:,.0f} >= ${whale_threshold_usd:,.0f}")
                        break
                except Exception as e:
                    continue
            if active:
                break
        
        strength = 1.0 if active else 0.0
        print(f"[STEALTH DEBUG] whale_ping result for {symbol}: active={active}, strength={strength}, max_order=${max_order_value:,.0f}")
        return StealthSignal("whale_ping", active, strength)
    
    def check_spoofing_layers(self, token_data: Dict) -> StealthSignal:
        """
        Spoofing layers detector - wykrycie warstw spoofingu przez zlecenia w wąskim zakresie cenowym
        Nowa implementacja zgodna z user specification
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        orderbook = token_data.get("orderbook", {})
        
        print(f"[STEALTH DEBUG] spoofing_layers for {symbol}: checking layered orders...")
        
        if not orderbook:
            print(f"[STEALTH DEBUG] spoofing_layers for {symbol}: no orderbook data")
            return StealthSignal("spoofing_layers", False, 0.0)
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        spoof_layers_threshold = 3
        price_tolerance = 0.0025  # 0.25% price range
        
        def has_layers(side, side_name):
            layers = 0
            prev_price = None
            for price, _ in side:
                try:
                    price = float(price)
                    if prev_price is not None and abs(price - prev_price) / price < price_tolerance:
                        layers += 1
                        if layers >= spoof_layers_threshold:
                            print(f"[STEALTH DEBUG] spoofing_layers DETECTED for {symbol}: {layers+1} {side_name} layers within {price_tolerance*100:.2f}% range")
                            return True
                    else:
                        layers = 1
                    prev_price = price
                except:
                    continue
            return False
        
        active = has_layers(bids, "bid") or has_layers(asks, "ask")
        strength = 0.8 if active else 0.0
        
        print(f"[STEALTH DEBUG] spoofing_layers result for {symbol}: active={active}, strength={strength}")
        return StealthSignal("spoofing_layers", active, strength)
    
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
        DEX inflow detection - wykrycie napływu do portfeli DEX
        Nowa implementacja zgodna z user specification
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        inflow_value = token_data.get("dex_inflow", 0)
        
        print(f"[STEALTH DEBUG] dex_inflow for {symbol}: checking inflow value={inflow_value}")
        
        # Konwersja do liczby jeśli potrzebna
        if isinstance(inflow_value, str):
            try:
                inflow_value = float(inflow_value)
            except:
                inflow_value = 0
        
        active = False
        strength = 0.0
        
        if inflow_value == 0:
            active = False
            strength = 0.0
            print(f"[STEALTH DEBUG] dex_inflow for {symbol}: no inflow detected")
        elif inflow_value > 25000:
            active = True
            strength = 1.0
            print(f"[STEALTH DEBUG] dex_inflow HIGH for {symbol}: ${inflow_value:,.0f} > $25,000")
        elif inflow_value > 10000:
            active = True
            strength = 0.7
            print(f"[STEALTH DEBUG] dex_inflow MEDIUM for {symbol}: ${inflow_value:,.0f} > $10,000")
        elif inflow_value > 5000:
            active = True
            strength = 0.4
            print(f"[STEALTH DEBUG] dex_inflow LOW for {symbol}: ${inflow_value:,.0f} > $5,000")
        else:
            active = False
            strength = 0.0
            print(f"[STEALTH DEBUG] dex_inflow BELOW THRESHOLD for {symbol}: ${inflow_value:,.0f} <= $5,000")
        
        print(f"[STEALTH DEBUG] dex_inflow result for {symbol}: active={active}, strength={strength}")
        return StealthSignal("dex_inflow", active, strength)
    
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
        Wykryj spike w wolumenie - wersja stealth
        """
        symbol = token_data.get("symbol", "UNKNOWN")
        candles_15m = token_data.get('candles_15m', [])
        volume_24h = token_data.get('volume_24h', 0)
        
        # 🔍 DEBUG: Szczegółowe debugowanie volume spike + FIX 2 validation
        print(f"[STEALTH DEBUG] volume_spike for {symbol}: candles_15m={len(candles_15m)}, volume_24h={volume_24h}")
        print(f"[STEALTH INPUT] volume_spike received candles_15m with {len(candles_15m)} entries")
        
        # 🛡️ FIX 2: Enhanced data validation - pokazuj szczegóły
        if len(candles_15m) < 4:
            print(f"[STEALTH DEBUG] volume_spike for {symbol}: insufficient candle data ({len(candles_15m)}/4)")
            print(f"[STEALTH DEBUG] volume_spike DIAGNOSTIC for {symbol}: candles_15m type={type(candles_15m)}, content preview={str(candles_15m)[:200]}...")
            return StealthSignal("volume_spike", False, 0.0)
        
        try:
            # Porównaj ostatni wolumen z średnią z 3 poprzednich
            recent_volumes = []
            for candle in candles_15m[-4:]:
                if isinstance(candle, dict):
                    volume = candle.get('volume', 0)
                else:
                    volume = candle[5] if len(candle) > 5 else 0
                recent_volumes.append(float(volume))
            
            current_volume = recent_volumes[-1]
            avg_volume = sum(recent_volumes[:-1]) / 3
            
            print(f"[STEALTH DEBUG] volume_spike for {symbol}: volumes={recent_volumes}, current={current_volume}, avg={avg_volume}")
            
            if avg_volume == 0:
                print(f"[STEALTH DEBUG] volume_spike for {symbol}: zero average volume - skipping")
                return StealthSignal("volume_spike", False, 0.0)
            
            volume_ratio = current_volume / avg_volume
            active = volume_ratio > 1.8  # Lowered threshold for testing (was 2.0)
            strength = min(volume_ratio / 3.0, 1.0)  # Scale 0-1
            
            print(f"[STEALTH DEBUG] volume_spike result for {symbol}: ratio={volume_ratio:.2f}, active={active}, strength={strength}")
            return StealthSignal("volume_spike", active, strength)
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
        Orderbook anomaly detector - wykrycie anomalii w spread bid/ask lub ekstremalnej nierównowagi
        Nowa implementacja zgodna z user specification
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
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            # Oblicz spread jako procent
            spread = abs(best_ask - best_bid) / best_ask
            
            # Oblicz imbalance jako ratio liczby bidów do asków
            imbalance = len(bids) / max(len(asks), 1)
            
            active = False
            strength = 0.0
            
            # Sprawdź szeroki spread (>2%)
            if spread > 0.02:
                active = True
                strength = 0.6
                print(f"[STEALTH DEBUG] orderbook_anomaly WIDE SPREAD for {symbol}: {spread*100:.2f}% > 2%")
            
            # Sprawdź ekstremalną nierównowagę (>2.5x więcej bidów niż asków)
            elif imbalance > 2.5:
                active = True
                strength = 0.7
                print(f"[STEALTH DEBUG] orderbook_anomaly IMBALANCE for {symbol}: {imbalance:.2f}x > 2.5x")
            
            # Jeśli oba warunki spełnione, zwiększ strength
            if spread > 0.02 and imbalance > 2.5:
                strength = 0.8
                print(f"[STEALTH DEBUG] orderbook_anomaly BOTH CONDITIONS for {symbol}: spread={spread*100:.2f}%, imbalance={imbalance:.2f}x")
            
            print(f"[STEALTH DEBUG] orderbook_anomaly result for {symbol}: active={active}, strength={strength}")
            return StealthSignal("orderbook_anomaly", active, strength)
            
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