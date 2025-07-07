"""
PrePump Engine v2 – Stealth AI
Główny silnik scoringu bez analizy wykresów

Analizuje sygnały z otoczenia rynku:
- Orderbook manipulation detection
- Volume pattern analysis  
- DEX inflow tracking
- Spoofing detection
- Market microstructure signals
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os

from .stealth_signals import StealthSignalDetector, StealthSignal
from .stealth_weights import StealthWeightManager
from .stealth_feedback import StealthFeedbackSystem


@dataclass
class StealthResult:
    """Wynik analizy Stealth Engine"""
    symbol: str
    stealth_score: float
    decision: str  # 'enter', 'wait', 'avoid'
    confidence: float
    active_signals: List[str]
    signal_breakdown: Dict[str, float]
    risk_assessment: str
    timestamp: float


class StealthEngine:
    """
    PrePump Engine v2 – Stealth AI
    
    Główny silnik analizy sygnałów pre-pump bez wykresów
    Koncentruje się na sygnałach z otoczenia rynku
    """
    
    def __init__(self, config_path: str = "crypto-scan/cache"):
        """
        Inicjalizacja Stealth Engine
        
        Args:
            config_path: Ścieżka do katalogu z konfiguracją i cache
        """
        self.config_path = config_path
        self.ensure_directories()
        
        # Inicjalizacja komponentów
        self.signal_detector = StealthSignalDetector()
        self.weight_manager = StealthWeightManager(config_path)
        self.feedback_system = StealthFeedbackSystem(config_path)
        
        # Ładowanie wag
        self.weights = self.weight_manager.load_weights()
        
        print(f"[STEALTH ENGINE] Initialized with {len(self.weights)} signal weights")
    
    def ensure_directories(self):
        """Upewnij się że wymagane katalogi istnieją"""
        os.makedirs(self.config_path, exist_ok=True)
        os.makedirs(f"{self.config_path}/stealth_feedback", exist_ok=True)
    
    async def analyze_token(self, symbol: str, market_data: Dict) -> Optional[StealthResult]:
        """
        Główna funkcja analizy tokena przez Stealth Engine
        
        Args:
            symbol: Symbol tokena (np. 'BTCUSDT')
            market_data: Dane rynkowe zawierające:
                - price: aktualna cena
                - volume_24h: wolumen 24h
                - orderbook: książka zleceń
                - candles_15m: świece 15m (dla volume analysis)
                - candles_5m: świece 5m (dla micro patterns)
                
        Returns:
            StealthResult z oceną stealth_score i decyzją
        """
        try:
            start_time = time.time()
            
            # KROK 1: Detekcja sygnałów stealth
            signals = await self.signal_detector.detect_all_signals(symbol, market_data)
            
            if not signals:
                print(f"[STEALTH] {symbol}: No signals detected")
                return None
            
            # KROK 2: Obliczenie stealth_score z wagami
            stealth_score, signal_breakdown = self.calculate_stealth_score(signals)
            
            # KROK 3: Decyzja na podstawie score i confidence
            decision, confidence = self.make_decision(stealth_score, signals)
            
            # KROK 4: Ocena ryzyka
            risk_assessment = self.assess_risk(signals, stealth_score)
            
            # KROK 5: Utworzenie wyniku
            result = StealthResult(
                symbol=symbol,
                stealth_score=stealth_score,
                decision=decision,
                confidence=confidence,
                active_signals=[sig['signal_name'] for sig in signals if sig['active']],
                signal_breakdown=signal_breakdown,
                risk_assessment=risk_assessment,
                timestamp=time.time()
            )
            
            # KROK 6: Logging dla feedback loop
            self.feedback_system.log_prediction(result, market_data)
            
            elapsed = time.time() - start_time
            print(f"[STEALTH] {symbol}: score={stealth_score:.3f} ({decision}) in {elapsed:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"[STEALTH ERROR] {symbol}: {e}")
            return None
    
    def calculate_stealth_score(self, signals: List[Dict]) -> Tuple[float, Dict[str, float]]:
        """
        Oblicz stealth_score na podstawie aktywnych sygnałów i wag
        
        Args:
            signals: Lista wykrytych sygnałów
            
        Returns:
            Tuple (stealth_score, breakdown)
        """
        total_score = 0.0
        signal_breakdown = {}
        active_weight_sum = 0.0
        
        for signal in signals:
            signal_name = signal['signal_name']
            strength = signal['strength']
            active = signal['active']
            
            if active and signal_name in self.weights:
                weight = self.weights[signal_name]
                contribution = strength * weight
                total_score += contribution
                active_weight_sum += weight
                
                signal_breakdown[signal_name] = contribution
                
        # Normalizacja przez sumę aktywnych wag
        if active_weight_sum > 0:
            stealth_score = min(1.0, total_score / active_weight_sum)
        else:
            stealth_score = 0.0
            
        return stealth_score, signal_breakdown
    
    def make_decision(self, stealth_score: float, signals: List[Dict]) -> Tuple[str, float]:
        """
        Podejmij decyzję handlową na podstawie stealth_score
        
        Args:
            stealth_score: Obliczony wynik stealth
            signals: Lista sygnałów (do oceny confidence)
            
        Returns:
            Tuple (decision, confidence)
        """
        # Liczba aktywnych sygnałów wpływa na confidence
        active_signals = [s for s in signals if s['active']]
        signal_count = len(active_signals)
        
        # Base confidence na podstawie liczby sygnałów
        base_confidence = min(1.0, signal_count / 5.0)  # Max confidence przy 5+ sygnałach
        
        # Decyzja na podstawie stealth_score
        if stealth_score >= 0.7:
            decision = 'enter'
            confidence = base_confidence * 0.9  # Wysokie confidence dla strong signals
        elif stealth_score >= 0.4:
            decision = 'wait'
            confidence = base_confidence * 0.7  # Średnie confidence dla medium signals
        else:
            decision = 'avoid'
            confidence = base_confidence * 0.5  # Niskie confidence dla weak signals
            
        return decision, confidence
    
    def assess_risk(self, signals: List[Dict], stealth_score: float) -> str:
        """
        Oceń ryzyko na podstawie wykrytych sygnałów
        
        Args:
            signals: Lista wykrytych sygnałów
            stealth_score: Stealth score
            
        Returns:
            Risk assessment string
        """
        risk_signals = []
        
        # Sprawdź sygnały wysokiego ryzyka
        for signal in signals:
            if signal['active'] and signal['signal_name'] in ['spoofing_detected', 'manipulation_warning']:
                risk_signals.append(signal['signal_name'])
        
        if risk_signals:
            return f"HIGH_RISK: {', '.join(risk_signals)}"
        elif stealth_score < 0.3:
            return "LOW_SIGNAL_QUALITY"
        elif stealth_score > 0.6:
            return "FAVORABLE_CONDITIONS"
        else:
            return "MODERATE_RISK"
    
    async def batch_analyze(self, tokens_data: List[Tuple[str, Dict]]) -> List[StealthResult]:
        """
        Analizuj wiele tokenów równolegle
        
        Args:
            tokens_data: Lista tupli (symbol, market_data)
            
        Returns:
            Lista StealthResult
        """
        tasks = []
        for symbol, market_data in tokens_data:
            task = self.analyze_token(symbol, market_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtruj poprawne wyniki
        valid_results = []
        for result in results:
            if isinstance(result, StealthResult):
                valid_results.append(result)
        
        return valid_results
    
    def update_weights_from_feedback(self):
        """Aktualizuj wagi na podstawie feedback loop"""
        try:
            updated_weights = self.feedback_system.calculate_updated_weights()
            if updated_weights:
                self.weights.update(updated_weights)
                self.weight_manager.save_weights(self.weights)
                print(f"[STEALTH] Updated {len(updated_weights)} weights from feedback")
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] {e}")
    
    async def analyze_token_stealth_v2(self, symbol: str, market_data: Dict) -> Optional[StealthResult]:
        """
        Analizuj token używając nowej architektury StealthSignal
        Implementuje specyfikację użytkownika z funkcją get_active_stealth_signals
        
        Args:
            symbol: Symbol tokena
            market_data: Dane rynkowe
            
        Returns:
            StealthResult lub None jeśli błąd
        """
        start_time = time.time()
        
        try:
            # Użyj nowej funkcji get_active_stealth_signals
            stealth_signals = self.signal_detector.get_active_stealth_signals(market_data)
            # Konwertuj StealthSignal do formatu używanego przez calculate_stealth_score
            signals_dict = []
            for sig in stealth_signals:
                if hasattr(sig, 'name') and hasattr(sig, 'active') and hasattr(sig, 'strength'):
                    signals_dict.append({
                        'signal_name': sig.name,
                        'active': sig.active,
                        'strength': sig.strength,
                        'details': f'Stealth signal strength: {sig.strength:.3f}'
                    })
            
            # Oblicz stealth score
            stealth_score, signal_breakdown = self.calculate_stealth_score(signals_dict)
            
            # Podejmij decyzję
            decision, confidence = self.make_decision(stealth_score, signals_dict)
            
            # Risk assessment
            risk_assessment = self.assess_risk(signals_dict, stealth_score)
            
            # Lista aktywnych sygnałów
            active_signals = [sig.name for sig in stealth_signals if sig.active]
            
            # Utwórz StealthResult
            result = StealthResult(
                symbol=symbol,
                stealth_score=stealth_score,
                decision=decision,
                confidence=confidence,
                active_signals=active_signals,
                signal_breakdown=signal_breakdown,
                risk_assessment=risk_assessment,
                timestamp=time.time()
            )
            
            # Log to feedback system
            if decision in ['enter', 'wait'] and stealth_score > 0.3:
                self.feedback_system.log_prediction(result, market_data)
            
            analysis_time = time.time() - start_time
            print(f"[STEALTH v2] {symbol}: score={stealth_score:.3f} ({decision}) in {analysis_time:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"[STEALTH v2 ERROR] {symbol}: {e}")
            return None
    
    def get_engine_stats(self) -> Dict:
        """Pobierz statystyki silnika"""
        return {
            'total_signals': len(self.weights),
            'active_weights': sum(1 for w in self.weights.values() if w > 0),
            'feedback_predictions': self.feedback_system.get_stats(),
            'config_path': self.config_path
        }


# Global instance dla łatwego importu
stealth_engine = None

def get_stealth_engine(config_path: str = "crypto-scan/cache") -> StealthEngine:
    """Pobierz singleton instance Stealth Engine"""
    global stealth_engine
    if stealth_engine is None:
        stealth_engine = StealthEngine(config_path)
    return stealth_engine


async def analyze_token_stealth(symbol: str, market_data: Dict) -> Optional[StealthResult]:
    """
    Convenience function dla analizy pojedynczego tokena
    
    Args:
        symbol: Symbol tokena
        market_data: Dane rynkowe
        
    Returns:
        StealthResult lub None
    """
    engine = get_stealth_engine()
    return await engine.analyze_token(symbol, market_data)


def compute_stealth_score(token_data: Dict) -> Dict:
    """
    🎯 GŁÓWNA FUNKCJA FINALIZACJI SCORE - zgodnie z user specification v2
    
    Finalne wyliczenie score dla tokena na podstawie aktywnych sygnałów Stealth 
    z nowymi matematycznie precyzyjnymi warunkami aktywacji i bonusem +0.025 za każdą regułę.
    
    Args:
        token_data: Dane rynkowe tokena (orderbook, volume, DEX, etc.)
        
    Returns:
        dict: {
            "score": float,       # Zsumowany score z aktywnych sygnałów + bonus
            "active_signals": list  # Lista nazw aktywnych sygnałów
        }
    """
    try:
        # Import lokalny aby uniknąć circular imports
        from .stealth_signals import StealthSignalDetector
        from .stealth_weights import load_weights
        
        symbol = token_data.get("symbol", "UNKNOWN")
        
        # LOG: Rozpoczęcie analizy Stealth Engine
        print(f"[STEALTH] Checking token: {symbol}...")
        
        # Walidacja tickera - zablokuj STEALTH jeśli ticker nieprawidłowy
        price = token_data.get("price", 0)
        volume_24h = token_data.get("volume_24h", 0)
        
        if price == 0 or volume_24h == 0:
            print(f"[STEALTH SKIPPED] {symbol}: Invalid ticker data (price={price}, volume={volume_24h}) - blocking STEALTH analysis")
            return {
                "score": 0.0,
                "active_signals": [],
                "skipped": "invalid_ticker"
            }
        
        # Załaduj wagi dynamiczne i wyloguj
        weights = load_weights()
        print(f"[STEALTH WEIGHTS] Loaded {len(weights)} dynamic weights from feedback system")
        
        # Utwórz detektor sygnałów
        detector = StealthSignalDetector()
        
        # Debug danych wejściowych przed analizą sygnałów
        candles_15m = token_data.get("candles_15m", [])
        candles_5m = token_data.get("candles_5m", [])
        orderbook = token_data.get("orderbook", {})
        dex_inflow = token_data.get("dex_inflow")
        
        print(f"[STEALTH INPUT] {symbol} data validation:")
        print(f"  - candles_15m: {len(candles_15m)} candles")
        print(f"  - candles_5m: {len(candles_5m)} candles") 
        print(f"  - orderbook: {bool(orderbook)} (bids: {len(orderbook.get('bids', []))}, asks: {len(orderbook.get('asks', []))})")
        print(f"  - dex_inflow: {dex_inflow} (type: {type(dex_inflow)})")
        
        # Enhanced debug logging dla kluczowych wartości (zgodnie ze specyfikacją)
        if orderbook:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            if bids and asks:
                max_bid_usd = float(bids[0][0]) * float(bids[0][1])
                max_ask_usd = float(asks[0][0]) * float(asks[0][1])
                max_order_usd = max(max_bid_usd, max_ask_usd)
                
                bid_price = float(bids[0][0])
                ask_price = float(asks[0][0])
                mid_price = (bid_price + ask_price) / 2
                spread_pct = (ask_price - bid_price) / mid_price
                
                total_bids = sum(float(bid[1]) for bid in bids[:10])
                total_asks = sum(float(ask[1]) for ask in asks[:10])
                total_volume = total_bids + total_asks
                imbalance_pct = abs(total_bids - total_asks) / total_volume if total_volume > 0 else 0.0
                
                print(f"[STEALTH DEBUG] {symbol} orderbook metrics:")
                print(f"  - max_order_usd: ${max_order_usd:,.0f}")
                print(f"  - spread_pct: {spread_pct:.6f}")
                print(f"  - imbalance_pct: {imbalance_pct:.3f}")
        
        # Pobierz aktywne sygnały z detektorów (zgodnie z user specification)
        try:
            signals = detector.get_active_stealth_signals(token_data)
            print(f"[STEALTH DEBUG] {symbol}: Successfully got {len(signals)} signals from detector")
        except Exception as e:
            print(f"[STEALTH ERROR] {symbol}: Failed to get signals from detector: {e}")
            return {
                "score": 0.0,
                "active_signals": [],
                "error": f"signal_detection_failed: {e}"
            }
        
        # Załaduj aktualne wagi (mogą być dostrojone przez feedback loop)
        weights = load_weights()
        
        # Analizuj każdy sygnał
        signal_status = {}
        
        for signal in signals:
            signal_status[signal.name] = getattr(signal, 'active', False)
        
        # Wyloguj kluczowe sygnały
        whale_ping = signal_status.get('whale_ping', False)
        spoofing_layers = signal_status.get('spoofing_layers', False) 
        volume_spike = signal_status.get('volume_spike', False)
        orderbook_anomaly = signal_status.get('orderbook_anomaly', False)
        dex_inflow_active = signal_status.get('dex_inflow', False)
        
        print(f"[STEALTH] Detected signals for {symbol}: whale={whale_ping}, spoofing={spoofing_layers}, volume_spike={volume_spike}, orderbook={orderbook_anomaly}, dex={dex_inflow_active}")
        
        score = 0.0
        used_signals = []
        total_signals = len(signals)
        available_signals = 0  # Sygnały z danymi (niezależnie od aktywności)
        
        # Oblicz score tylko z aktywnych sygnałów + liczenie dostępności
        for signal in signals:
            # 🔍 FIX 3: Sprawdź czy sygnał ma dane (nie jest placeholder) - poprawiona logika
            has_data = True
            if signal.name in ['dex_inflow']:
                # DEX inflow ma dane jeśli wartość nie jest None lub False
                dex_value = token_data.get('dex_inflow')
                has_data = dex_value is not None
                if not has_data:
                    print(f"[STEALTH DATA] {symbol}: {signal.name} - no DEX data available")
            elif signal.name in ['spoofing_layers', 'large_bid_walls', 'orderbook_imbalance']:
                # Orderbook sygnały mają dane jeśli orderbook istnieje
                orderbook_data = token_data.get('orderbook', {})
                has_data = bool(orderbook_data.get('bids')) and bool(orderbook_data.get('asks'))
                if not has_data:
                    print(f"[STEALTH DATA] {symbol}: {signal.name} - no orderbook data available")
            elif signal.name in ['volume_spike']:
                # Volume spike ma dane jeśli są świece
                candles_data = token_data.get('candles_15m', [])
                has_data = len(candles_data) >= 4
                if not has_data:
                    print(f"[STEALTH DATA] {symbol}: {signal.name} - insufficient candle data ({len(candles_data)}/4)")
            
            if has_data:
                available_signals += 1
            
            if hasattr(signal, 'active') and signal.active:
                # Pobierz wagę dla tego sygnału (fallback na 1.0)
                weight = weights.get(signal.name, 1.0)
                
                # Wkład sygnału = waga * siła sygnału
                contribution = weight * signal.strength
                score += contribution
                used_signals.append(signal.name)
                
                # LOG: Każdy aktywny sygnał
                if signal.strength > 0:
                    print(f"[STEALTH] Signal {signal.name}: strength={signal.strength:.3f}, weight={weight:.3f}, contribution=+{contribution:.3f}")
        
        # 🧠 PARTIAL SCORING MECHANISM - zgodnie z user request
        data_coverage = available_signals / total_signals if total_signals > 0 else 0
        
        # Jeśli mało danych dostępnych (≤50%), zastosuj scaling
        if data_coverage <= 0.5 and available_signals >= 3:
            # Proporcjonalne przeliczenie - scale up score based on data coverage
            scaling_factor = min(2.0, 1.0 / data_coverage) if data_coverage > 0 else 1.0
            original_score = score
            score = score * scaling_factor
            print(f"[STEALTH PARTIAL] {symbol}: Low data coverage ({data_coverage:.1%}), scaling {original_score:.3f} → {score:.3f} (factor: {scaling_factor:.2f})")
        
        # Dodaj bonus zgodnie z nową specyfikacją: +0.025 za każdą aktywną regułę
        active_rules_bonus = len(used_signals) * 0.025
        score += active_rules_bonus
        
        # Dodaj minimalny baseline score jeśli wykryto jakiekolwiek pozytywne sygnały
        if len(used_signals) > 0 and score < 0.5:
            baseline_bonus = 0.3 * len(used_signals) / total_signals
            score += baseline_bonus
            print(f"[STEALTH PARTIAL] {symbol}: Added baseline bonus +{baseline_bonus:.3f} for {len(used_signals)} active signals")
        
        # LOG: Finalna decyzja scoringowa z nową implementacją bonusu
        decision = "strong" if score >= 3.0 else "weak" if score >= 1.0 else "none"
        partial_note = f" (partial: {available_signals}/{total_signals} signals)" if data_coverage < 0.8 else ""
        
        print(f"[STEALTH SCORING] {symbol} final calculation:")
        print(f"  Base score: {score - active_rules_bonus:.3f}")
        print(f"  Active rules bonus: {len(used_signals)} × 0.025 = +{active_rules_bonus:.3f}")
        print(f"  Final score: {score:.3f}")
        print(f"[STEALTH] Final signal for {symbol} → Score: {score:.3f}, Decision: {decision}, Active: {len(used_signals)} signals{partial_note}")
        
        return {
            "score": round(score, 3),
            "active_signals": used_signals,
            "data_coverage": round(data_coverage, 2),
            "partial_scoring": data_coverage < 0.8
        }
        
    except Exception as e:
        import traceback
        print(f"[COMPUTE STEALTH SCORE ERROR] {symbol}: Exception occurred: {type(e).__name__}: {e}")
        print(f"[COMPUTE STEALTH SCORE ERROR] {symbol}: Traceback: {traceback.format_exc()}")
        return {
            "score": 0.0,
            "active_signals": [],
            "error": f"compute_stealth_error: {e}"
        }


def classify_stealth_alert(stealth_score: float) -> Optional[str]:
    """
    🚨 KLASYFIKACJA ALERTÓW - zgodnie z user specification
    
    Określa typ alertu na podstawie stealth score:
    - strong_stealth_alert: score > 4.0
    - medium_alert: score > 2.5  
    - None: score <= 2.5
    
    Args:
        stealth_score: Score z compute_stealth_score()
        
    Returns:
        str lub None: Typ alertu
    """
    if stealth_score > 4.0:
        return "strong_stealth_alert"
    elif stealth_score > 2.5:
        return "medium_alert"
    else:
        return None


def analyze_token_with_stealth_score(symbol: str, token_data: Dict) -> Dict:
    """
    🔍 KOMPLETNA ANALIZA TOKENA - convenience function
    
    Łączy compute_stealth_score() z klasyfikacją alertów
    
    Args:
        symbol: Symbol tokena
        token_data: Dane rynkowe
        
    Returns:
        dict: Kompletny wynik analizy Stealth
    """
    try:
        # Oblicz stealth score
        stealth_result = compute_stealth_score(token_data)
        
        # Klasyfikuj alert
        alert_type = classify_stealth_alert(stealth_result["score"])
        
        return {
            "symbol": symbol,
            "stealth_score": stealth_result["score"], 
            "active_signals": stealth_result["active_signals"],
            "alert_type": alert_type,
            "signal_count": len(stealth_result["active_signals"]),
            "timestamp": time.time()
        }
        
    except Exception as e:
        print(f"[STEALTH ANALYSIS ERROR] {symbol}: {e}")
        return {
            "symbol": symbol,
            "stealth_score": 0.0,
            "active_signals": [],
            "alert_type": None,
            "signal_count": 0,
            "timestamp": time.time(),
            "error": str(e)
        }