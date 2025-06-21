"""
Trend Mode 2.0 Score Engine - nowy model scoringu z trzema kategoriami detektorów

Model zaprojektowany dla rzeczywistych warunków rynkowych gdzie rzadko aktywują się 
wszystkie detektory jednocześnie.

Kategorie detektorów:
- Core Signals (kluczowe): +10 punktów
- Helper Signals (pomocnicze): +5 punktów  
- Negative Signals (blokujące): -10 punktów

Poziomy alertów:
- Level 1 Watchlist: ≥25 punktów, min. 2 detektory
- Level 2 Active Entry: ≥40 punktów, min. 1 core detector
- Level 3 Confirmed Trend: ≥60 punktów, min. 2 core detektory
"""

import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone


class TrendMode2ScoreEngine:
    """Engine do obliczania Trend Mode 2.0 scoring z nowym modelem"""
    
    def __init__(self):
        # Definicje kategorii detektorów
        self.core_detectors = {
            'uptrend_15m': 10,
            'pullback_flow': 10, 
            'calm_before_trend': 10,
            'orderbook_freeze': 10,
            'heatmap_vacuum': 10
        }
        
        self.helper_detectors = {
            'vwap_pinning': 5,
            'human_flow': 5,
            'micro_echo': 5,
            'sr_structure_score': 5,  # when > 7
            'volume_flow_consistency': 5
        }
        
        self.negative_detectors = {
            'chaotic_flow': -10,
            'ask_domination': -10,  # when > 80x
            'fake_pulse': -10,
            'high_volatility': -10,
            'no_uptrend': -10,
            'strong_sell_pressure': -10
        }
    
    def analyze_symbol_detectors(self, symbol: str, market_data: Dict, orderbook_data: Dict = None) -> Dict:
        """
        Analizuje symbol przez wszystkie detektory Trend Mode 2.0
        
        Args:
            symbol: Trading symbol
            market_data: Dane rynkowe (candles, prices, volumes)
            orderbook_data: Dane orderbook
            
        Returns:
            dict: Wyniki analizy wszystkich detektorów
        """
        detector_results = {
            'core_active': {},
            'helper_active': {},
            'negative_active': {},
            'raw_scores': {}
        }
        
        try:
            candles = market_data.get('candles', [])
            prices = [float(c[4]) for c in candles] if candles else []
            volumes = [float(c[5]) for c in candles] if candles else []
            
            # === CORE DETECTORS ===
            
            # 1. Uptrend 15M - potwierdzony trend z wyższej perspektywy
            detector_results['core_active']['uptrend_15m'] = self._detect_uptrend_15m(prices)
            
            # 2. Pullback Flow - idealne miejsce wejścia po korekcie
            detector_results['core_active']['pullback_flow'] = self._detect_pullback_flow(
                symbol, candles, orderbook_data
            )
            
            # 3. Calm Before Trend - stabilizacja napięcia przed impulsem
            detector_results['core_active']['calm_before_trend'] = self._detect_calm_before_trend(prices)
            
            # 4. Orderbook Freeze - whale control bez zmiany ceny
            detector_results['core_active']['orderbook_freeze'] = self._detect_orderbook_freeze(orderbook_data)
            
            # 5. Heatmap Vacuum - brak podaży powyżej (4/5 poziomów czyste)
            detector_results['core_active']['heatmap_vacuum'] = self._detect_heatmap_vacuum(
                prices, orderbook_data
            )
            
            # === HELPER DETECTORS ===
            
            # 1. VWAP Pinning
            detector_results['helper_active']['vwap_pinning'] = self._detect_vwap_pinning(prices, volumes)
            
            # 2. Human Flow - ludzki rytm
            detector_results['helper_active']['human_flow'] = self._detect_human_flow(prices)
            
            # 3. Micro Echo
            detector_results['helper_active']['micro_echo'] = self._detect_micro_echo(prices)
            
            # 4. S/R Structure Score > 7
            sr_score = self._calculate_sr_structure_score(prices)
            detector_results['helper_active']['sr_structure_score'] = sr_score > 7
            detector_results['raw_scores']['sr_structure_score'] = sr_score
            
            # 5. Volume Flow Consistency - stały napływ volume
            detector_results['helper_active']['volume_flow_consistency'] = self._detect_volume_flow_consistency(volumes)
            
            # === NEGATIVE DETECTORS ===
            
            # 1. Chaotic Flow - dużo zmian kierunku (≥40%)
            detector_results['negative_active']['chaotic_flow'] = self._detect_chaotic_flow(prices)
            
            # 2. Ask Domination > 80x
            ask_ratio = self._calculate_ask_domination(orderbook_data)
            detector_results['negative_active']['ask_domination'] = ask_ratio > 80
            detector_results['raw_scores']['ask_domination_ratio'] = ask_ratio
            
            # 3. Fake Pulse
            detector_results['negative_active']['fake_pulse'] = self._detect_fake_pulse(prices, volumes)
            
            # 4. High Volatility
            detector_results['negative_active']['high_volatility'] = self._detect_high_volatility(prices)
            
            # 5. No Uptrend
            detector_results['negative_active']['no_uptrend'] = not detector_results['core_active']['uptrend_15m']
            
            # 6. Strong Sell Pressure w orderbooku
            detector_results['negative_active']['strong_sell_pressure'] = self._detect_strong_sell_pressure(orderbook_data)
            
        except Exception as e:
            print(f"⚠️ Error analyzing detectors for {symbol}: {e}")
            
        return detector_results
    
    def calculate_trend_mode_2_score(self, detector_results: Dict) -> Tuple[int, Dict]:
        """
        Oblicza finalne Trend Mode 2.0 score na podstawie wyników detektorów
        
        Args:
            detector_results: Wyniki z analyze_symbol_detectors
            
        Returns:
            tuple: (final_score, score_breakdown)
        """
        score = 0
        score_breakdown = {
            'core_points': 0,
            'helper_points': 0,
            'negative_points': 0,
            'active_core': [],
            'active_helper': [],
            'active_negative': [],
            'total_detectors': 0
        }
        
        # Oblicz punkty z core detectors
        for detector, active in detector_results['core_active'].items():
            if active:
                points = self.core_detectors[detector]
                score += points
                score_breakdown['core_points'] += points
                score_breakdown['active_core'].append(detector)
                score_breakdown['total_detectors'] += 1
        
        # Oblicz punkty z helper detectors
        for detector, active in detector_results['helper_active'].items():
            if active:
                points = self.helper_detectors[detector]
                score += points
                score_breakdown['helper_points'] += points
                score_breakdown['active_helper'].append(detector)
                score_breakdown['total_detectors'] += 1
        
        # Oblicz punkty z negative detectors
        for detector, active in detector_results['negative_active'].items():
            if active:
                points = self.negative_detectors[detector]
                score += points
                score_breakdown['negative_points'] += points
                score_breakdown['active_negative'].append(detector)
                score_breakdown['total_detectors'] += 1
        
        # Zabezpieczenie przed ujemnym score
        score = max(0, score)
        score_breakdown['final_score'] = score
        
        return score, score_breakdown
    
    def determine_alert_level(self, score: int, score_breakdown: Dict) -> Tuple[int, str]:
        """
        Określa poziom alertu na podstawie score i aktywnych detektorów
        
        Args:
            score: Final trend mode score
            score_breakdown: Breakdown punktów
            
        Returns:
            tuple: (alert_level, description)
        """
        core_count = len(score_breakdown['active_core'])
        total_detectors = score_breakdown['total_detectors']
        negative_count = len(score_breakdown['active_negative'])
        
        # Sprawdź negatywne detektory - mogą blokować alerty
        if negative_count >= 2:
            return 0, f"Blocked by {negative_count} negative detectors"
        
        # Level 3 - Confirmed Trend (≥60 punktów, min. 2 core)
        if score >= 60 and core_count >= 2:
            return 3, f"Confirmed trend: {core_count} core detectors, {score} points"
        
        # Level 2 - Active Entry (≥40 punktów, min. 1 core)
        if score >= 40 and core_count >= 1:
            return 2, f"Active entry: {core_count} core detector(s), {score} points"
        
        # Level 1 - Watchlist (≥25 punktów, min. 2 detektory łącznie)
        if score >= 25 and total_detectors >= 2:
            return 1, f"Watchlist: {total_detectors} detectors, {score} points"
        
        # No alert
        return 0, f"No alert: {score} points, {total_detectors} detectors"
    
    def process_symbol_trend_mode_2(self, symbol: str, market_data: Dict, orderbook_data: Dict = None) -> Dict:
        """
        Główna funkcja przetwarzania symbolu przez Trend Mode 2.0
        
        Args:
            symbol: Trading symbol
            market_data: Dane rynkowe
            orderbook_data: Dane orderbook
            
        Returns:
            dict: Kompletny wynik analizy Trend Mode 2.0
        """
        try:
            # 1. Analizuj przez wszystkie detektory
            detector_results = self.analyze_symbol_detectors(symbol, market_data, orderbook_data)
            
            # 2. Oblicz score
            score, score_breakdown = self.calculate_trend_mode_2_score(detector_results)
            
            # 3. Określ poziom alertu
            alert_level, alert_description = self.determine_alert_level(score, score_breakdown)
            
            # 4. Przygotuj kompletny wynik
            result = {
                'symbol': symbol,
                'trend_mode_2_score': score,
                'score_breakdown': score_breakdown,
                'detector_results': detector_results,
                'alert_level': alert_level,
                'alert_description': alert_description,
                'trend_active': alert_level >= 2,  # Level 2+ = trend aktywny
                'entry_recommended': alert_level >= 2,
                'timestamp': int(time.time()),
                'analysis_summary': self._create_analysis_summary(score, score_breakdown, alert_level)
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error in Trend Mode 2.0 processing for {symbol}: {e}")
            return {
                'symbol': symbol,
                'trend_mode_2_score': 0,
                'alert_level': 0,
                'trend_active': False,
                'error': str(e)
            }
    
    def _create_analysis_summary(self, score: int, score_breakdown: Dict, alert_level: int) -> List[str]:
        """Tworzy czytelne podsumowanie analizy"""
        summary = []
        
        # Podstawowe info
        summary.append(f"Trend Mode 2.0 Score: {score}/100")
        summary.append(f"Alert Level: {alert_level}/3")
        
        # Aktywne detektory
        if score_breakdown['active_core']:
            summary.append(f"Core: {', '.join(score_breakdown['active_core'])}")
        
        if score_breakdown['active_helper']:
            summary.append(f"Helper: {', '.join(score_breakdown['active_helper'])}")
        
        if score_breakdown['active_negative']:
            summary.append(f"⚠️ Negative: {', '.join(score_breakdown['active_negative'])}")
        
        # Punkty
        if score_breakdown['core_points'] > 0:
            summary.append(f"Core points: +{score_breakdown['core_points']}")
        
        if score_breakdown['helper_points'] > 0:
            summary.append(f"Helper points: +{score_breakdown['helper_points']}")
        
        if score_breakdown['negative_points'] < 0:
            summary.append(f"Negative points: {score_breakdown['negative_points']}")
        
        return summary
    
    # === DETECTOR IMPLEMENTATIONS ===
    
    def _detect_uptrend_15m(self, prices: List[float]) -> bool:
        """Wykryj potwierdzony uptrend na 15M"""
        if len(prices) < 20:
            return False
        
        recent = prices[-12:]  # Ostatnie 3h (12x15M)
        rising_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        return rising_count >= 7  # 7/11 = ~63% wzrostów
    
    def _detect_pullback_flow(self, symbol: str, candles: List, orderbook_data: Dict) -> bool:
        """Wykryj pullback flow pattern"""
        try:
            from detectors.pullback_flow_pattern import pullback_flow_pattern
            
            if not candles or len(candles) < 24:
                return False
            
            candles_15m = [str(c[4]) for c in candles[-48:]] if len(candles) >= 48 else [str(c[4]) for c in candles]
            candles_5m = [str(c[4]) for c in candles[-12:]] if len(candles) >= 12 else [str(c[4]) for c in candles]
            
            # Fallback orderbook jeśli nie ma danych
            if not orderbook_data:
                orderbook_data = {"ask_volumes": [1000, 950, 900], "bid_volumes": [800, 900, 1000]}
            
            result = pullback_flow_pattern(symbol, candles_15m, candles_5m, orderbook_data)
            return result.get('pullback_trigger', False)
        except:
            return False
    
    def _detect_calm_before_trend(self, prices: List[float]) -> bool:
        """Wykryj calm before trend pattern"""
        if len(prices) < 20:
            return False
        
        recent = prices[-10:]
        volatility = max(recent) - min(recent)
        avg_price = sum(recent) / len(recent)
        volatility_pct = (volatility / avg_price) * 100 if avg_price > 0 else 0
        
        return volatility_pct < 2.0  # Niska zmienność < 2%
    
    def _detect_orderbook_freeze(self, orderbook_data: Dict) -> bool:
        """Wykryj orderbook freeze - whale control"""
        if not orderbook_data:
            return False
        
        ask_volumes = orderbook_data.get('ask_volumes', [])
        bid_volumes = orderbook_data.get('bid_volumes', [])
        
        if len(ask_volumes) < 3 or len(bid_volumes) < 3:
            return False
        
        # Sprawdź stabilność volumes (małe zmiany)
        ask_changes = [abs(ask_volumes[i] - ask_volumes[i-1]) / ask_volumes[i-1] 
                      for i in range(1, len(ask_volumes)) if ask_volumes[i-1] > 0]
        
        bid_changes = [abs(bid_volumes[i] - bid_volumes[i-1]) / bid_volumes[i-1] 
                      for i in range(1, len(bid_volumes)) if bid_volumes[i-1] > 0]
        
        avg_ask_change = sum(ask_changes) / len(ask_changes) if ask_changes else 1
        avg_bid_change = sum(bid_changes) / len(bid_changes) if bid_changes else 1
        
        return avg_ask_change < 0.05 and avg_bid_change < 0.05  # < 5% change
    
    def _detect_heatmap_vacuum(self, prices: List[float], orderbook_data: Dict) -> bool:
        """Wykryj heatmap vacuum - brak podaży powyżej"""
        if len(prices) < 10:
            return False
        
        current_price = prices[-1]
        recent_highs = [max(prices[i:i+5]) for i in range(len(prices)-10, len(prices)-5)]
        
        # Sprawdź czy current price jest blisko lub powyżej ostatnich highs
        resistance_levels = [h for h in recent_highs if h > current_price * 1.002]
        
        # Vacuum = mało poziomów resistance
        return len(resistance_levels) <= 1
    
    def _detect_vwap_pinning(self, prices: List[float], volumes: List[float]) -> bool:
        """Wykryj VWAP pinning"""
        if len(prices) < 20 or len(volumes) < 20:
            return False
        
        # Prosta kalkulacja VWAP
        recent_prices = prices[-20:]
        recent_volumes = volumes[-20:]
        
        vwap = sum(p * v for p, v in zip(recent_prices, recent_volumes)) / sum(recent_volumes) if sum(recent_volumes) > 0 else 0
        current_price = prices[-1]
        
        # Pinning = cena blisko VWAP (±0.2%)
        return abs(current_price - vwap) / vwap < 0.002 if vwap > 0 else False
    
    def _detect_human_flow(self, prices: List[float]) -> bool:
        """Wykryj human flow - ludzki rytm"""
        try:
            from detectors.human_flow import detect_human_like_flow
            result = detect_human_like_flow(prices)
            return result[0] if isinstance(result, tuple) else False
        except:
            return False
    
    def _detect_micro_echo(self, prices: List[float]) -> bool:
        """Wykryj micro echo patterns"""
        try:
            from detectors.micro_echo import detect_micro_echo
            result = detect_micro_echo("", prices)  # Symbol nie jest używany
            return result[0] if isinstance(result, tuple) else False
        except:
            return False
    
    def _calculate_sr_structure_score(self, prices: List[float]) -> float:
        """Oblicz S/R structure score"""
        if len(prices) < 20:
            return 0
        
        # Prosta analiza S/R na podstawie pivot points
        highs = []
        lows = []
        
        for i in range(2, len(prices)-2):
            # Local high
            if prices[i] > prices[i-1] and prices[i] > prices[i+1] and prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                highs.append(prices[i])
            
            # Local low  
            if prices[i] < prices[i-1] and prices[i] < prices[i+1] and prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                lows.append(prices[i])
        
        # Score na podstawie liczby wyraźnych poziomów
        return min(len(highs) + len(lows), 10)  # Max 10 punktów
    
    def _detect_volume_flow_consistency(self, volumes: List[float]) -> bool:
        """Wykryj stały napływ volume"""
        if len(volumes) < 10:
            return False
        
        recent = volumes[-10:]
        avg_volume = sum(recent) / len(recent)
        
        # Sprawdź czy ostatnie 5 okresów ma volume powyżej średniej
        last_5 = recent[-5:]
        above_avg = sum(1 for v in last_5 if v > avg_volume * 0.8)
        
        return above_avg >= 4  # 4/5 ostatnich powyżej 80% średniej
    
    def _detect_chaotic_flow(self, prices: List[float]) -> bool:
        """Wykryj chaotic flow - dużo zmian kierunku"""
        if len(prices) < 20:
            return False
        
        direction_changes = 0
        for i in range(2, len(prices)):
            prev_direction = prices[i-1] > prices[i-2]
            curr_direction = prices[i] > prices[i-1]
            if prev_direction != curr_direction:
                direction_changes += 1
        
        change_ratio = direction_changes / (len(prices) - 2)
        return change_ratio >= 0.4  # ≥40% zmian kierunku
    
    def _calculate_ask_domination(self, orderbook_data: Dict) -> float:
        """Oblicz ask domination ratio"""
        if not orderbook_data:
            return 0
        
        ask_volumes = orderbook_data.get('ask_volumes', [])
        bid_volumes = orderbook_data.get('bid_volumes', [])
        
        if not ask_volumes or not bid_volumes:
            return 0
        
        total_ask = sum(ask_volumes)
        total_bid = sum(bid_volumes)
        
        return total_ask / total_bid if total_bid > 0 else 100
    
    def _detect_fake_pulse(self, prices: List[float], volumes: List[float]) -> bool:
        """Wykryj fake pulse - wzrost ceny bez volume"""
        if len(prices) < 5 or len(volumes) < 5:
            return False
        
        recent_prices = prices[-5:]
        recent_volumes = volumes[-5:]
        
        price_increase = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = recent_volumes[-1]
        
        # Fake pulse = wzrost ceny >1% ale volume poniżej średniej
        return price_increase > 0.01 and current_volume < avg_volume * 0.7
    
    def _detect_high_volatility(self, prices: List[float]) -> bool:
        """Wykryj wysoką zmienność"""
        if len(prices) < 20:
            return False
        
        recent = prices[-20:]
        volatility = (max(recent) - min(recent)) / min(recent) if min(recent) > 0 else 0
        
        return volatility > 0.05  # >5% volatility
    
    def _detect_strong_sell_pressure(self, orderbook_data: Dict) -> bool:
        """Wykryj silną presję sprzedaży"""
        if not orderbook_data:
            return False
        
        ask_ratio = self._calculate_ask_domination(orderbook_data)
        return ask_ratio > 5.0  # Ask volume 5x większy niż bid


# Globalna instancja dla całego systemu
trend_mode_2_engine = TrendMode2ScoreEngine()