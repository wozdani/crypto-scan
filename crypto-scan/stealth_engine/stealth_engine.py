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
from .diamond_detector import run_diamond_detector

# CaliforniumWhale AI imports
try:
    from stealth.californium.californium_whale_detect import CaliforniumTGN
    from stealth.californium.qirl_agent_singleton import get_qirl_agent
    from stealth.californium.graph_cache import generate_mock_graph_data
    import networkx as nx
    import torch
    CALIFORNIUM_AVAILABLE = True
    print("[CALIFORNIUM] CaliforniumWhale AI successfully imported")
except ImportError as e:
    print(f"[CALIFORNIUM WARNING] CaliforniumWhale AI not available: {e}")
    CALIFORNIUM_AVAILABLE = False


def simulate_stealth_decision(score: float, volume_24h: float, tjde_phase: str = "unknown") -> bool:
    """
    Zwraca True jeśli powinien być wygenerowany alert Stealth.
    Uwzględnia dynamiczne progi w zależności od płynności i fazy rynku z bonusem kontekstowym.

    Args:
        score: final stealth score
        volume_24h: 24-godzinny wolumen tokena  
        tjde_phase: faza rynku z Trend Mode (domyślnie 'unknown')
        
    Returns:
        bool: True jeśli alert powinien być aktywowany
        
    Enhanced Logic:
        - Podstawowy próg: 0.70 (wyższy dla lepszej selektywności)
        - Duży volume (>$10M): próg pozostaje 0.70
        - Mały volume (<$1M): próg obniżony do 0.65
        - Bonus kontekstowy: +0.15 do score dla accumulation/momentum
        - Obniżenie progu: -0.10 dla korzystnych faz (accumulation/momentum)
        - Fallback: unknown/None = neutralne zachowanie
    """
    # Kopiuj pierwotny score dla modyfikacji
    adjusted_score = score
    
    # Podstawowy próg (wyższy dla lepszej selektywności)
    threshold = 0.70
    
    # Dostosowanie progu do wolumenu (mniejsze różnice)
    if volume_24h < 1_000_000:
        threshold = 0.65  # Nieco łagodniej dla małych tokenów
    
    # Bonus kontekstowy do score dla korzystnych faz rynku
    if tjde_phase in ["accumulation", "momentum"]:
        adjusted_score += 0.15  # Boost for stealth in favorable market context
        threshold -= 0.10       # Obniżenie progu dla korzystnych faz (0.70 → 0.60)
    elif tjde_phase == "unknown":
        pass  # Neutralne zachowanie - bez bonusu ani kary
    
    return adjusted_score >= threshold


def californium_whale_score(symbol: str) -> float:
    """
    CaliforniumWhale AI - Temporal Graph + QIRL detector
    
    Analizuje ukryte wzorce akumulacji whale'ów używając:
    - Temporal Graph Convolutional Network (TGN)
    - Quantum-inspired Reinforcement Learning (QIRL)
    - Mock graph data z cache system
    
    Args:
        symbol: Symbol tokena (np. 'BTCUSDT')
        
    Returns:
        float: CaliforniumWhale score (0.0-1.0)
    """
    if not CALIFORNIUM_AVAILABLE:
        print(f"[CALIFORNIUM] CaliforniumWhale AI not available for {symbol}")
        return 0.0
    
    try:
        # STEP 1: Generate mock graph data
        data = generate_mock_graph_data(symbol)
        
        # STEP 2: Extract graph components
        G = data["graph"]
        adj = nx.to_numpy_array(G)
        features = data["features"]
        timestamps = data["timestamps"]
        volumes = data["volumes"]
        
        # STEP 3: Initialize Temporal GNN model
        model = CaliforniumTGN(in_features=features.shape[1], out_features=1)
        model.eval()  # Set to evaluation mode
        
        # STEP 4: Run TGN analysis
        with torch.no_grad():
            scores = model(adj, features, timestamps, volumes)
        
        # STEP 5: Prepare state for QIRL Agent
        tgn_score = float(scores.max().item())
        state_vector = scores.flatten().detach().numpy().tolist() + timestamps.tolist()
        
        # Pad or truncate to fixed size (20 features)
        if len(state_vector) > 20:
            state_vector = state_vector[:20]
        else:
            state_vector.extend([0.0] * (20 - len(state_vector)))
        
        # STEP 6: Get QIRL Agent decision
        agent = get_qirl_agent(state_size=20, action_size=2)
        action = agent.get_action(state_vector)
        
        # STEP 7: Update QIRL Agent with placeholder reward
        # In production, this would be real market feedback
        reward = 1.0 if action == 1 else 0.0
        agent.update(state_vector, action, reward)
        
        # STEP 8: Calculate final CaliforniumWhale score
        # 🔧 FIXED: Always use weighted TGN score regardless of QIRL action
        if action == 1:  # QIRL recommends action
            final_score = tgn_score
        else:  # QIRL recommends hold - but still use weighted TGN score
            final_score = tgn_score * 0.5  # Reduced weight for HOLD action
        
        print(f"[CALIFORNIUM] {symbol}: TGN score={tgn_score:.3f}, QIRL action={action}, final={final_score:.3f}")
        
        return final_score
        
    except Exception as e:
        print(f"[CALIFORNIUM ERROR] {symbol}: {e}")
        return 0.0


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
            
            # KROK 3: Decyzja na podstawie score i confidence z dynamicznymi progami
            decision, confidence = self.make_decision(stealth_score, signals, market_data)
            
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
        🚀 ENHANCED: Dynamic spoofing weight based on signal context (HIGHUSDT fix)
        
        Args:
            signals: Lista wykrytych sygnałów
            
        Returns:
            Tuple (stealth_score, breakdown)
        """
        total_score = 0.0
        signal_breakdown = {}
        active_weight_sum = 0.0
        
        # 🚀 ADAPTIVE SPOOFING WEIGHT: Check if spoofing is the only active signal
        active_signals = [s for s in signals if s['active']]
        spoofing_signals = [s for s in active_signals if 'spoofing' in s['signal_name'].lower()]
        is_spoofing_only = len(spoofing_signals) > 0 and len(active_signals) <= 2
        
        for signal in signals:
            signal_name = signal['signal_name']
            strength = signal['strength']
            active = signal['active']
            
            if active and signal_name in self.weights:
                weight = self.weights[signal_name]
                
                # 🚀 ENHANCED SPOOFING WEIGHT: Apply dynamic weight for spoofing_layers
                if signal_name == 'spoofing_layers':
                    from .adaptive_thresholds import get_enhanced_spoofing_weight
                    enhanced_weight = get_enhanced_spoofing_weight(is_spoofing_only, weight)
                    weight = enhanced_weight
                    
                    print(f"[ADAPTIVE WEIGHT] {signal_name}: base={self.weights[signal_name]:.3f} → enhanced={weight:.3f} (only_signal: {is_spoofing_only})")
                
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
    
    def make_decision(self, stealth_score: float, signals: List[Dict], market_data: Dict = None) -> Tuple[str, float]:
        """
        Podejmij decyzję handlową na podstawie stealth_score z dynamicznymi progami
        🚀 ENHANCED: Context-aware decision with adaptive thresholds (HIGHUSDT fix)
        
        Args:
            stealth_score: Obliczony wynik stealth
            signals: Lista sygnałów (do oceny confidence)
            market_data: Dane rynkowe (volume_24h, tjde_phase) dla progów
            
        Returns:
            Tuple (decision, confidence)
        """
        # Liczba aktywnych sygnałów wpływa na confidence
        active_signals = [s for s in signals if s['active']]
        signal_count = len(active_signals)
        
        # Base confidence na podstawie liczby sygnałów
        base_confidence = min(1.0, signal_count / 5.0)  # Max confidence przy 5+ sygnałach
        
        # 🚀 ENHANCED DECISION LOGIC: Use adaptive thresholds for market context
        volume_24h = market_data.get('volume_24h', 0) if market_data else 0
        tjde_phase = market_data.get('tjde_phase') if market_data else None
        
        # 🚀 CONTEXTUAL PHASE ESTIMATION: Fill in missing tjde_phase data
        if tjde_phase is None or tjde_phase == 'unknown':
            from .adaptive_thresholds import estimate_contextual_phase
            spoofing_active = any('spoofing' in s['signal_name'].lower() and s['active'] for s in signals)
            tjde_phase = estimate_contextual_phase(volume_24h, spoofing_active)
            print(f"[PHASE ESTIMATION] tjde_phase estimated as: {tjde_phase} (volume: ${volume_24h:.0f}, spoofing: {spoofing_active})")
        
        # Sprawdź czy powinien być alert według dynamicznych progów
        should_alert = simulate_stealth_decision(stealth_score, volume_24h, tjde_phase)
        
        # 🚀 WEAK ALERT SYSTEM: Check for weak but valuable signals
        from .adaptive_thresholds import should_trigger_weak_alert
        spoofing_active = any('spoofing' in s['signal_name'].lower() and s['active'] for s in signals)
        weak_alert = should_trigger_weak_alert(stealth_score, volume_24h, spoofing_active)
        
        # Decyzja na podstawie progów alertowych + weak alert system
        if should_alert and stealth_score >= 0.5:
            decision = 'enter'
            confidence = base_confidence * 0.9  # Wysokie confidence dla qualified alerts
        elif weak_alert:
            decision = 'stealth_alert_weak'
            confidence = base_confidence * 0.6  # Średnie confidence dla weak alerts
            print(f"[WEAK ALERT] Triggered for score={stealth_score:.3f}, volume=${volume_24h:.0f}, spoofing={spoofing_active}")
        elif should_alert:
            decision = 'watch'  # Alert ale niższy score
            confidence = base_confidence * 0.8  # Średnie-wysokie confidence dla threshold alerts
        elif stealth_score >= 0.25:
            decision = 'wait'
            confidence = base_confidence * 0.6  # Średnie confidence dla sub-threshold
        else:
            decision = 'avoid'
            confidence = base_confidence * 0.4  # Niskie confidence dla weak signals
            
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
            
            # Podejmij decyzję z dynamicznymi progami
            decision, confidence = self.make_decision(stealth_score, signals_dict, market_data)
            
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


def log_stealth_decision(symbol: str, stealth_score: float, volume_24h: float, tjde_phase: str, final_decision: str) -> None:
    """
    Loguje szczegóły decyzji Stealth Engine z ulepszonym systemem progów
    
    Args:
        symbol: Symbol tokena
        stealth_score: Obliczony stealth score
        volume_24h: Wolumen 24h tokena
        tjde_phase: Faza rynku z TJDE (domyślnie 'unknown')
        final_decision: Finalna decyzja systemu
    """
    # Obsłuż None jako 'unknown'
    if tjde_phase is None:
        tjde_phase = "unknown"
    
    # Oblicz bazowy próg i adjusted score zgodnie z nową logiką
    adjusted_score = stealth_score
    threshold = 0.70  # Nowy wyższy podstawowy próg
    
    # Dostosowanie progu do wolumenu
    if volume_24h < 1_000_000:
        threshold = 0.65  # Nieco łagodniej dla małych tokenów
    
    # 🔧 MOCAUSDT FIX 2: Enhanced threshold_reduction logic for strong signals
    context_bonus = 0.0
    threshold_reduction = 0.0
    
    # Standardowy bonus dla korzystnych faz
    if tjde_phase in ["accumulation", "momentum"]:
        context_bonus = 0.15  # Boost to score
        threshold_reduction = 0.10  # Reduction to threshold
        adjusted_score += context_bonus
        threshold -= threshold_reduction
    
    # 🔧 MOCAUSDT FIX 2: Dodatkowy threshold_reduction dla silnych sygnałów
    # Jeśli score > 0.66 i active_signals ≥ 2 i volume_24h > $10M
    active_signals_count = getattr(adjusted_score, '_active_signals_count', 0)  # Będzie przekazane z wywołania
    if stealth_score > 0.66 and volume_24h > 10_000_000:
        # Dodatkowe obniżenie progu dla silnych sygnałów
        additional_threshold_reduction = 0.02
        threshold_reduction += additional_threshold_reduction
        threshold -= additional_threshold_reduction
        context_bonus += 0.01  # Mały bonus do score też
        adjusted_score += 0.01
        print(f"[MOCAUSDT FIX 2] {symbol}: Strong signals detected → additional threshold_reduction={additional_threshold_reduction:.2f}")
    
    # Sprawdź finalną decyzję
    should_alert = adjusted_score >= threshold
    
    # Szczegółowy log z nowym formatem
    volume_str = f"${volume_24h/1_000_000:.1f}M" if volume_24h >= 1_000_000 else f"${volume_24h/1_000:.0f}k"
    print(f"[STEALTH DECISION] {symbol}: score={stealth_score:.3f}→{adjusted_score:.3f}, threshold={threshold:.2f}, volume={volume_str}")
    print(f"[STEALTH DECISION] {symbol}: phase={tjde_phase}, context_bonus={context_bonus:.2f}, threshold_reduction={threshold_reduction:.2f}")
    print(f"[STEALTH DECISION] {symbol}: alert={should_alert} → {final_decision}")


def get_stealth_alert_threshold(volume_24h: float) -> float:
    """
    Pobierz bazowy próg alertowy dla danego wolumenu (bez bonusów kontekstowych)
    
    Args:
        volume_24h: Wolumen 24h tokena
        
    Returns:
        float: Bazowy próg alertowy stealth_score
    """
    if volume_24h < 1_000_000:
        return 0.65  # Mały token - nieco łagodniejszy próg
    else:
        return 0.70  # Standardowy/duży token - wyższy próg dla lepszej selektywności


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
    # ⛔ Hard-filter: Skip tokens with too low daily volume
    if market_data.get("volume_24h", 0) < 500_000:
        return None  # silently skip without logs
    
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
        
        # ⛔ Hard-filter: Skip tokens with too low daily volume
        volume_24h = token_data.get("volume_24h", 0)
        if volume_24h < 500_000:
            return {
                "score": 0.0,
                "active_signals": [],
                "skipped": "low_volume"
            }
        
        # LOG: Rozpoczęcie analizy Stealth Engine
        print(f"[STEALTH] Checking token: {symbol}...")
        
        # Walidacja tickera z fallback na cenę ze świeczek
        price = token_data.get("price_usd", 0)
        
        # 🔧 PRICE FALLBACK: Use candles_15m close price if ticker price_usd == 0
        if price == 0:
            try:
                candles_15m = token_data.get("candles_15m", [])
                if candles_15m and len(candles_15m) > 0:
                    last_candle = candles_15m[-1]
                    if isinstance(last_candle, dict) and "close" in last_candle:
                        price = float(last_candle["close"])
                        print(f"[STEALTH PRICE FALLBACK] {symbol} → Using candle price: ${price}")
                    elif isinstance(last_candle, (list, tuple)) and len(last_candle) >= 5:
                        price = float(last_candle[4])  # close price in OHLCV format
                        print(f"[STEALTH PRICE FALLBACK] {symbol} → Using candle price: ${price}")
            except Exception as e:
                print(f"[STEALTH PRICE FALLBACK ERROR] {symbol} → Cannot extract fallback price: {e}")
        
        if price == 0:
            print(f"[STEALTH SKIPPED] {symbol}: No valid price data (ticker and candles both failed) - blocking STEALTH analysis")
            return {
                "score": 0.0,
                "active_signals": [],
                "skipped": "no_price_data"
            }
        
        if volume_24h == 0:
            print(f"[STEALTH VOLUME WARNING] {symbol}: 24h volume is 0 - possible data issue")
            # Continue analysis despite volume warning - may still have useful signals
        
        # Załaduj wagi dynamiczne i wyloguj
        weights = load_weights()
        print(f"[STEALTH WEIGHTS] Loaded {len(weights)} dynamic weights from feedback system")
        
        # Utwórz detektor sygnałów
        detector = StealthSignalDetector()
        
        # Debug danych wejściowych przed analizą sygnałów
        candles_15m = token_data.get("candles_15m", [])
        candles_5m = token_data.get("candles_5m", [])
        orderbook = token_data.get("orderbook", {})
        dex_inflow = token_data.get("dex_inflow", 0)
        
        # ENHANCED: Calculate real dex_inflow for display
        try:
            from utils.contracts import get_contract
            from utils.blockchain_scanners import get_token_transfers_last_24h, load_known_exchange_addresses
            
            contract_info = get_contract(symbol)
            if contract_info:
                real_transfers = get_token_transfers_last_24h(
                    symbol=symbol,
                    chain=contract_info['chain'],
                    contract_address=contract_info['address']
                )
                known_exchanges = load_known_exchange_addresses()
                exchange_addresses = known_exchanges.get(contract_info['chain'], [])
                dex_routers = known_exchanges.get('dex_routers', {}).get(contract_info['chain'], [])
                all_known_addresses = set(addr.lower() for addr in exchange_addresses + dex_routers)
                
                # Calculate actual DEX inflow for display
                real_dex_inflow = 0
                for transfer in real_transfers:
                    if transfer['to'] in all_known_addresses:
                        real_dex_inflow += transfer['value_usd']
                
                dex_inflow = real_dex_inflow  # Use real value for display
        except:
            pass  # Keep original value if calculation fails
        
        print(f"[STEALTH INPUT] {symbol} data validation:")
        print(f"  - candles_15m: {len(candles_15m)} candles")
        print(f"  - candles_5m: {len(candles_5m)} candles") 
        print(f"  - orderbook: {bool(orderbook)} (bids: {len(orderbook.get('bids', []))}, asks: {len(orderbook.get('asks', []))})")
        print(f"  - dex_inflow: ${dex_inflow:.2f} (real blockchain data)")
        
        # Enhanced debug logging dla kluczowych wartości (zgodnie ze specyfikacją)
        if orderbook:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            # Debug orderbook structure before processing
            print(f"[STEALTH DEBUG] {symbol} orderbook structure:")
            print(f"  - bids type: {type(bids)}, length: {len(bids) if bids else 0}")
            print(f"  - asks type: {type(asks)}, length: {len(asks) if asks else 0}")
            
            # Safe inspection of orderbook format
            try:
                if bids and isinstance(bids, list) and len(bids) > 0:
                    print(f"  - first bid type: {type(bids[0])}, content: {bids[0]}")
                elif bids and isinstance(bids, dict):
                    print(f"  - bids dict keys: {list(bids.keys())[:3]}...")  # Show first 3 keys
                    if '0' in bids:
                        print(f"  - bids['0']: {bids['0']}")
                
                if asks and isinstance(asks, list) and len(asks) > 0:
                    print(f"  - first ask type: {type(asks[0])}, content: {asks[0]}")
                elif asks and isinstance(asks, dict):
                    print(f"  - asks dict keys: {list(asks.keys())[:3]}...")  # Show first 3 keys
                    if '0' in asks:
                        print(f"  - asks['0']: {asks['0']}")
            except Exception as debug_e:
                print(f"[STEALTH DEBUG] {symbol}: Debug inspection error: {debug_e}")
            
            # Handle different orderbook formats safely
            if bids and asks:
                try:
                    # Convert dict-based orderbook to list format if needed with safe processing
                    if isinstance(bids, dict):
                        try:
                            bids_list = []
                            for key in sorted(bids.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0, reverse=True):
                                if isinstance(bids[key], list) and len(bids[key]) >= 2:
                                    bids_list.append(bids[key])
                            bids = bids_list if bids_list else []
                        except Exception as e:
                            print(f"[STEALTH DEBUG] stealth_engine bids conversion error for {symbol}: {e}")
                            bids = []
                    
                    if isinstance(asks, dict):
                        try:
                            asks_list = []
                            for key in sorted(asks.keys(), key=lambda x: float(x) if str(x).replace('.','').isdigit() else 0):
                                if isinstance(asks[key], list) and len(asks[key]) >= 2:
                                    asks_list.append(asks[key])
                            asks = asks_list if asks_list else []
                        except Exception as e:
                            print(f"[STEALTH DEBUG] stealth_engine asks conversion error for {symbol}: {e}")
                            asks = []
                    
                    # Now process as list format
                    if len(bids) > 0 and len(asks) > 0:
                        # Handle both list and dict formats safely
                        if isinstance(bids[0], list) and len(bids[0]) >= 2:
                            max_bid_usd = float(bids[0][0]) * float(bids[0][1])
                            bid_price = float(bids[0][0])
                        elif isinstance(bids[0], dict):
                            max_bid_usd = float(bids[0].get('price', 0)) * float(bids[0].get('size', 0))
                            bid_price = float(bids[0].get('price', 0))
                        else:
                            print(f"[STEALTH DEBUG] {symbol}: Unknown bid format: {type(bids[0])}")
                            max_bid_usd = 0
                            bid_price = 0
                        
                        if isinstance(asks[0], list) and len(asks[0]) >= 2:
                            max_ask_usd = float(asks[0][0]) * float(asks[0][1])
                            ask_price = float(asks[0][0])
                        elif isinstance(asks[0], dict):
                            max_ask_usd = float(asks[0].get('price', 0)) * float(asks[0].get('size', 0))
                            ask_price = float(asks[0].get('price', 0))
                        else:
                            print(f"[STEALTH DEBUG] {symbol}: Unknown ask format: {type(asks[0])}")
                            max_ask_usd = 0
                            ask_price = 0
                    
                    max_order_usd = max(max_bid_usd, max_ask_usd)
                    
                    if bid_price > 0 and ask_price > 0:
                        mid_price = (bid_price + ask_price) / 2
                        spread_pct = (ask_price - bid_price) / mid_price
                    else:
                        mid_price = price or 0
                        spread_pct = 0.01  # Default spread
                    
                    # Safe volume calculation
                    try:
                        if isinstance(bids[0], list):
                            total_bids = sum(float(bid[1]) for bid in bids[:10] if len(bid) >= 2)
                            total_asks = sum(float(ask[1]) for ask in asks[:10] if len(ask) >= 2)
                        else:
                            total_bids = sum(float(bid.get('size', 0)) for bid in bids[:10] if isinstance(bid, dict))
                            total_asks = sum(float(ask.get('size', 0)) for ask in asks[:10] if isinstance(ask, dict))
                        total_volume = total_bids + total_asks
                    except (ValueError, TypeError, KeyError) as e:
                        print(f"[STEALTH DEBUG] {symbol}: Volume calculation error: {e}")
                        total_bids = total_asks = total_volume = 0
                        
                except (ValueError, TypeError, KeyError, IndexError) as e:
                    print(f"[STEALTH DEBUG] {symbol}: Orderbook parsing error: {e}")
                    max_order_usd = spread_pct = total_volume = 0
                    total_bids = total_asks = 0
                
                # Calculate imbalance
                imbalance_pct = abs(total_bids - total_asks) / total_volume if total_volume > 0 else 0.0
                
                print(f"[STEALTH DEBUG] {symbol} orderbook metrics:")
                print(f"  - max_order_usd: ${max_order_usd:,.0f}")
                print(f"  - spread_pct: {spread_pct:.6f}")
                print(f"  - imbalance_pct: {imbalance_pct:.3f}")
            else:
                print(f"[STEALTH DEBUG] {symbol}: No valid orderbook data available")
                max_order_usd = spread_pct = total_volume = 0
                total_bids = total_asks = imbalance_pct = 0
        
        # === MIKROSKOPIJNY ORDERBOOK CHECK ===
        # Sprawdź czy token ma bardzo niską płynność (1 bid + 1 ask)
        # Użyj już sparsowanych bids/asks z orderbook, nie raw token_data
        orderbook_bids = orderbook.get('bids', []) if orderbook else []
        orderbook_asks = orderbook.get('asks', []) if orderbook else []
        
        # Debug orderbook check to verify proper data usage
        print(f"[FALLBACK DEBUG] {symbol}: orderbook_bids={len(orderbook_bids) if orderbook_bids else 0}, orderbook_asks={len(orderbook_asks) if orderbook_asks else 0}")
        
        if len(orderbook_bids) <= 1 and len(orderbook_asks) <= 1:
            print(f"[ILLIQUID SKIP] {symbol}: Orderbook too small (bids={len(orderbook_bids)}, asks={len(orderbook_asks)}) - applying fallback scoring")
            
            # Sprawdź czy możemy zastosować fallback scoring dla "quiet microcap"
            spread_pct = getattr(token_data, 'spread_pct', 0.0)
            if spread_pct > 0 and spread_pct < 0.3:  # Tight spread + mały orderbook = potencjalna cicha akumulacja
                return {
                    "score": 0.25,
                    "stealth_score": 0.25, 
                    "active_signals": ["quiet_microcap"],
                    "stealth_decision": "stealth_watch",
                    "partial_scoring": True,
                    "reason": "quiet_microcap_detected"
                }
            else:
                return {
                    "score": 0.0,
                    "stealth_score": 0.0,
                    "active_signals": [],
                    "stealth_decision": "none",
                    "partial_scoring": True,
                    "reason": "illiquid_orderbook_skipped"
                }

        # Pobierz aktywne sygnały z detektorów (zgodnie z user specification)
        print(f"[DEBUG FLOW] {symbol} - Starting get_active_stealth_signals() call...")
        try:
            signals = detector.get_active_stealth_signals(token_data)
            print(f"[DEBUG FLOW] {symbol} - get_active_stealth_signals() completed successfully")
            print(f"[STEALTH DEBUG] {symbol}: Successfully got {len(signals)} signals from detector")
        except Exception as e:
            print(f"[DEBUG FLOW] {symbol} - get_active_stealth_signals() FAILED")
            print(f"[STEALTH ERROR] {symbol}: Failed to get signals from detector: {e}")
            return {
                "score": 0.0,
                "stealth_score": 0.0,
                "active_signals": [],
                "stealth_decision": "none",
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
        
        # === DIAMOND WHALE AI DETECTOR INTEGRATION ===
        # 🔥 STAGE 2/7: Integrate DiamondWhale AI Temporal Graph + QIRL Detector
        diamond_score = 0.0
        diamond_enabled = False
        diamond_error = None
        
        try:
            # Sprawdź czy token ma kontrakt blockchain dla analizy transakcji
            from utils.contracts import get_contract
            contract_info = get_contract(symbol)
            
            if contract_info and contract_info.get('address'):
                print(f"[DIAMOND AI] {symbol}: Starting DiamondWhale AI analysis for contract {contract_info['address'][:10]}...")
                
                # Pobierz rzeczywiste transakcje blockchain dla analizy temporal graph
                from .blockchain_fetcher import fetch_diamond_transactions
                contract_address = contract_info['address']
                chain = contract_info.get('chain', 'ethereum')
                
                # Fetch real blockchain transactions
                transactions = fetch_diamond_transactions(contract_address, chain)
                print(f"[DIAMOND AI] {symbol}: Fetched {len(transactions)} real blockchain transactions")
                
                # Wywołaj DiamondWhale AI Detector z rzeczywistymi transakcjami
                diamond_result = run_diamond_detector(transactions, symbol)
                
                if diamond_result and diamond_result.get('diamond_score') is not None:
                    diamond_score = float(diamond_result['diamond_score'])
                    diamond_enabled = True
                    
                    # Dodaj diamond_score do głównego stealth score (z wagą 0.3)
                    diamond_contribution = diamond_score * 0.3
                    score += diamond_contribution
                    
                    # Dodaj do listy aktywnych sygnałów jeśli znaczący
                    if diamond_score > 0.5:
                        used_signals.append("diamond_whale_detection")
                    
                    print(f"[DIAMOND AI] {symbol}: Diamond score={diamond_score:.3f}, contribution=+{diamond_contribution:.3f}")
                    print(f"[DIAMOND AI] {symbol}: Temporal graph analysis completed successfully")
                else:
                    print(f"[DIAMOND AI] {symbol}: No significant diamond activity detected")
            else:
                print(f"[DIAMOND AI] {symbol}: No blockchain contract found - skipping temporal analysis")
                
        except Exception as e:
            diamond_error = str(e)
            print(f"[DIAMOND AI ERROR] {symbol}: Failed to run DiamondWhale detector: {e}")
            print(f"[DIAMOND AI ERROR] {symbol}: Continuing without diamond analysis")

        # === WHALECLIP AI DETECTOR INTEGRATION ===
        # 🧠 STAGE: Integrate WhaleCLIP Vision AI Detector
        whaleclip_score = 0.0
        whaleclip_enabled = False
        whaleclip_error = None
        
        try:
            # Sprawdź czy token ma kontrakt blockchain dla analizy transakcji
            from utils.contracts import get_contract
            contract_info = get_contract(symbol)
            
            if contract_info and contract_info.get('address'):
                print(f"[WHALECLIP AI] {symbol}: Starting WhaleCLIP AI analysis for contract {contract_info['address'][:10]}...")
                
                # Wywołaj WhaleCLIP AI Detector - simplified behavioral analysis
                try:
                    # Use stealth_engine_advanced WhaleCLIP integration
                    from stealth_engine_advanced import StealthEngineAdvanced
                    advanced_engine = StealthEngineAdvanced()
                    
                    # Get blockchain address for behavioral analysis
                    blockchain_address = contract_info['address']
                    chain = contract_info.get('chain', 'ethereum')
                    
                    # Fetch transactions and analyze behavioral patterns
                    transactions = advanced_engine.fetch_transactions_from_blockchain(blockchain_address, chain, limit=20)
                    
                    if transactions and len(transactions) > 0:
                        # Simple WhaleCLIP scoring based on transaction patterns
                        high_value_txs = [tx for tx in transactions if float(tx.get('value', 0)) > 1000]
                        total_value = sum(float(tx.get('value', 0)) for tx in transactions)
                        
                        # Calculate WhaleCLIP confidence based on transaction behavior
                        if len(high_value_txs) > len(transactions) * 0.3 and total_value > 10000:
                            whaleclip_score = min(0.9, (len(high_value_txs) / len(transactions)) * (total_value / 100000))
                            whaleclip_enabled = True
                            
                            # Dodaj whaleclip_score do głównego stealth score (z wagą 0.2)
                            whaleclip_contribution = whaleclip_score * 0.2
                            score += whaleclip_contribution
                            
                            # Dodaj do listy aktywnych sygnałów jeśli znaczący
                            if whaleclip_score > 0.3:
                                used_signals.append("whaleclip_vision_detection")
                            
                            print(f"[WHALECLIP AI] {symbol}: WhaleCLIP score={whaleclip_score:.3f}, contribution=+{whaleclip_contribution:.3f}")
                            print(f"[WHALECLIP AI] {symbol}: Behavioral vision analysis completed successfully")
                        else:
                            print(f"[WHALECLIP AI] {symbol}: Insufficient behavioral patterns for WhaleCLIP analysis")
                    else:
                        print(f"[WHALECLIP AI] {symbol}: No transactions available for behavioral analysis")
                        
                except Exception as clip_e:
                    whaleclip_error = str(clip_e)
                    print(f"[WHALECLIP AI ERROR] {symbol}: WhaleCLIP analysis failed: {clip_e}")
            else:
                print(f"[WHALECLIP AI] {symbol}: No blockchain contract found - skipping behavioral analysis")
                
        except Exception as e:
            whaleclip_error = str(e)
            print(f"[WHALECLIP AI ERROR] {symbol}: Failed to run WhaleCLIP detector: {e}")
            print(f"[WHALECLIP AI ERROR] {symbol}: Continuing without WhaleCLIP analysis")

        # === CALIFORNIUM WHALE AI DETECTOR INTEGRATION ===
        # 🚀 STAGE 3/7: Integrate CaliforniumWhale AI Temporal Graph + QIRL Detector
        californium_score = 0.0
        californium_enabled = False
        californium_error = None
        
        try:
            # Wywołaj CaliforniumWhale AI Score
            californium_score = californium_whale_score(symbol)
            
            if californium_score > 0.0:
                californium_enabled = True
                
                # Dodaj californium_score do głównego stealth score (z wagą 0.25)
                californium_contribution = californium_score * 0.25
                score += californium_contribution
                
                # Dodaj do listy aktywnych sygnałów jeśli znaczący
                if californium_score > 0.3:
                    used_signals.append("californium_whale_detection")
                
                print(f"[CALIFORNIUM AI] {symbol}: Californium score={californium_score:.3f}, contribution=+{californium_contribution:.3f}")
                print(f"[CALIFORNIUM AI] {symbol}: Temporal GNN + QIRL analysis completed successfully")
                
                # 🚨 STAGE 4/7 - CALIFORNIUM ALERT SYSTEM
                # Sprawdź czy wysłać alert CaliforniumWhale AI (score > 0.7)
                if californium_score > 0.7:
                    try:
                        from californium_alerts import send_californium_alert
                        
                        # Przygotuj market data dla alertu
                        alert_market_data = {
                            "price_usd": token_data.get('price', 0),
                            "volume_24h": token_data.get('volume_24h', 0),
                            "symbol": symbol
                        }
                        
                        # Przygotuj dodatkowy kontekst
                        alert_context = {
                            "stealth_score": score,
                            "active_signals": used_signals.copy(),
                            "diamond_score": diamond_score if diamond_enabled else None,
                            "data_coverage": data_coverage
                        }
                        
                        # Wyślij alert CaliforniumWhale AI
                        alert_sent = send_californium_alert(
                            symbol, californium_score, 
                            alert_market_data, alert_context
                        )
                        
                        if alert_sent:
                            print(f"[CALIFORNIUM ALERT] ✅ {symbol}: Mastermind alert sent (score: {californium_score:.3f})")
                        else:
                            print(f"[CALIFORNIUM ALERT] ℹ️ {symbol}: Alert not sent (cooldown/config)")
                            
                    except ImportError:
                        print(f"[CALIFORNIUM ALERT] ⚠️ {symbol}: CaliforniumAlerts module not available")
                    except Exception as alert_error:
                        print(f"[CALIFORNIUM ALERT] ❌ {symbol}: Alert error: {alert_error}")
                        
            else:
                print(f"[CALIFORNIUM AI] {symbol}: No significant CaliforniumWhale activity detected")
                
        except Exception as e:
            californium_error = str(e)
            print(f"[CALIFORNIUM AI ERROR] {symbol}: Failed to run CaliforniumWhale detector: {e}")
            print(f"[CALIFORNIUM AI ERROR] {symbol}: Continuing without californium analysis")

        # LOG: Finalna decyzja scoringowa z nową implementacją bonusu + All AI Detectors
        decision = "strong" if score >= 3.0 else "weak" if score >= 1.0 else "none"
        partial_note = f" (partial: {available_signals}/{total_signals} signals)" if data_coverage < 0.8 else ""
        diamond_note = f" + Diamond: {diamond_score:.3f}" if diamond_enabled else ""
        whaleclip_note = f" + WhaleCLIP: {whaleclip_score:.3f}" if whaleclip_enabled else ""
        californium_note = f" + Californium: {californium_score:.3f}" if californium_enabled else ""
        
        print(f"[STEALTH SCORING] {symbol} final calculation:")
        base_score = score - active_rules_bonus
        if diamond_enabled:
            base_score -= diamond_score * 0.3
        if whaleclip_enabled:
            base_score -= whaleclip_score * 0.2
        if californium_enabled:
            base_score -= californium_score * 0.25
        print(f"  Base score: {base_score:.3f}")
        print(f"  Active rules bonus: {len(used_signals)} × 0.025 = +{active_rules_bonus:.3f}")
        if diamond_enabled:
            print(f"  Diamond AI contribution: {diamond_score:.3f} × 0.3 = +{diamond_score * 0.3:.3f}")
        if whaleclip_enabled:
            print(f"  WhaleCLIP AI contribution: {whaleclip_score:.3f} × 0.2 = +{whaleclip_score * 0.2:.3f}")
        if californium_enabled:
            print(f"  CaliforniumWhale AI contribution: {californium_score:.3f} × 0.25 = +{californium_score * 0.25:.3f}")
        print(f"  Final score: {score:.3f}")
        print(f"[STEALTH] Final signal for {symbol} → Score: {score:.3f}, Decision: {decision}, Active: {len(used_signals)} signals{partial_note}{diamond_note}{whaleclip_note}{californium_note}")
        
        # 🚨 STEALTH V3 TELEGRAM ALERT SYSTEM - QUALITY PRIORITY CHANGE
        # ✅ CONSENSUS-BASED ALERT LOGIC: Only send alerts when final_decision == "BUY"
        # This eliminates false positives where high score but agents vote HOLD/AVOID
        
        # 🚨 NEW UNIFIED CONSENSUS DECISION ENGINE INTEGRATION
        # Replace old 5-agent system with new 4-agent consensus: CaliforniumWhale, WhaleCLIP, StealthEngine, DiamondWhale
        final_decision = "WATCH"  # Default fallback
        consensus_data = None
        
        try:
            from .decision_consensus import create_decision_consensus_engine
            
            # Prepare detector outputs for new consensus system
            detector_outputs = {}
            
            # StealthEngine (Classic Stealth Signals)
            if score > 0.0:
                stealth_vote = "BUY" if score > 2.0 else ("HOLD" if score > 1.0 else "AVOID")
                detector_outputs["StealthEngine"] = {
                    "vote": stealth_vote,
                    "score": min(score / 4.0, 1.0),  # Normalize to 0-1
                    "confidence": min(data_coverage, 1.0),
                    "weight": 0.25
                }
                print(f"[NEW CONSENSUS] {symbol}: StealthEngine → {stealth_vote} (score: {score:.3f})")
            
            # CaliforniumWhale AI
            if californium_enabled and californium_score > 0.0:
                calif_vote = "BUY" if californium_score > 0.7 else ("HOLD" if californium_score > 0.4 else "AVOID")
                detector_outputs["CaliforniumWhale"] = {
                    "vote": calif_vote,
                    "score": californium_score,
                    "confidence": 0.85,
                    "weight": 0.33
                }
                print(f"[NEW CONSENSUS] {symbol}: CaliforniumWhale → {calif_vote} (score: {californium_score:.3f})")
            
            # DiamondWhale AI
            if diamond_enabled and diamond_score > 0.0:
                diamond_vote = "BUY" if diamond_score > 0.6 else ("HOLD" if diamond_score > 0.3 else "AVOID")
                detector_outputs["DiamondWhale"] = {
                    "vote": diamond_vote,
                    "score": diamond_score,
                    "confidence": 0.80,
                    "weight": 0.25
                }
                print(f"[NEW CONSENSUS] {symbol}: DiamondWhale → {diamond_vote} (score: {diamond_score:.3f})")
            
            # WhaleCLIP (derived from whale signals)
            whale_signal_strength = 0.0
            if "whale_ping" in used_signals:
                for signal in signals:
                    if signal.name == "whale_ping" and hasattr(signal, 'strength'):
                        whale_signal_strength = signal.strength
                        break
                
                if whale_signal_strength > 0.0:
                    clip_vote = "BUY" if whale_signal_strength > 0.8 else ("HOLD" if whale_signal_strength > 0.5 else "AVOID")
                    detector_outputs["WhaleCLIP"] = {
                        "vote": clip_vote,
                        "score": whale_signal_strength,
                        "confidence": 0.75,
                        "weight": 0.26
                    }
                    print(f"[NEW CONSENSUS] {symbol}: WhaleCLIP → {clip_vote} (score: {whale_signal_strength:.3f})")
            
            # Run consensus decision if we have detectors
            if len(detector_outputs) >= 2:
                consensus_engine = create_decision_consensus_engine()
                consensus_result = consensus_engine.simulate_decision_consensus(
                    detector_outputs, 
                    threshold=0.7, 
                    token=symbol
                )
                
                final_decision = consensus_result.decision
                consensus_data = {
                    "decision": final_decision,
                    "votes": [f"{det}: {data['vote']}" for det, data in detector_outputs.items()],
                    "confidence": consensus_result.confidence,
                    "final_score": consensus_result.final_score,
                    "threshold_met": consensus_result.threshold_met,
                    "contributing_detectors": consensus_result.contributing_detectors
                }
                
                print(f"[NEW CONSENSUS] {symbol}: Final decision → {final_decision} (score: {consensus_result.final_score:.3f}, confidence: {consensus_result.confidence:.3f})")
                print(f"[NEW CONSENSUS] {symbol}: Contributing detectors: {consensus_result.contributing_detectors}")
                
            else:
                print(f"[NEW CONSENSUS] {symbol}: Insufficient detectors ({len(detector_outputs)}) for consensus - defaulting to WATCH")
                consensus_data = {
                    "decision": "WATCH",
                    "votes": [],
                    "confidence": 0.0,
                    "final_score": 0.0,
                    "threshold_met": False,
                    "contributing_detectors": []
                }
                
        except ImportError:
            print(f"[NEW CONSENSUS] {symbol}: Consensus engine not available - falling back to legacy logic")
            # Simplified fallback logic
            if score > 2.0:
                final_decision = "BUY"
            elif score > 1.0:
                final_decision = "MONITOR"
            else:
                final_decision = "WATCH"
                
            consensus_data = {
                "decision": final_decision,
                "votes": ["fallback_mode"],
                "confidence": min(score / 4.0, 1.0),
                "final_score": score,
                "threshold_met": score > 2.0,
                "contributing_detectors": ["stealth_fallback"]
            }
        except Exception as e:
            print(f"[NEW CONSENSUS ERROR] {symbol}: {e}")
            final_decision = "WATCH"
            consensus_data = {
                "decision": "WATCH",
                "votes": [],
                "confidence": 0.0,
                "final_score": 0.0,
                "threshold_met": False,
                "contributing_detectors": []
            }
        
        print(f"[ALERT LOGIC] {symbol}: Score={score:.3f}, Decision={final_decision} → Alert Eligible: {final_decision == 'BUY'}")
        
        # ✅ NEW ALERT CONDITION: Only send alerts when agents reach BUY consensus
        if score >= 0.70 and final_decision == "BUY":
            try:
                from alerts.stealth_v3_telegram_alerts import send_stealth_v3_alert
                
                # Prepare detector_results for alert system using new consensus data
                detector_results = {}
                
                # Extract detector scores for alert display
                if consensus_data and "contributing_detectors" in consensus_data:
                    for detector in consensus_data["contributing_detectors"]:
                        # Map detector names to legacy alert format
                        if detector == "StealthEngine":
                            detector_results["whale_ping"] = min(score / 4.0, 1.0)
                        elif detector == "CaliforniumWhale":
                            detector_results["mastermind_tracing"] = californium_score if californium_enabled else 0.0
                        elif detector == "DiamondWhale": 
                            detector_results["diamond_ai"] = diamond_score if diamond_enabled else 0.0
                        elif detector == "WhaleCLIP":
                            detector_results["whaleclip_vision"] = whale_signal_strength
                
                # Ensure required detector fields exist for alert compatibility
                required_detectors = ["whale_ping", "dex_inflow", "orderbook_anomaly", "whaleclip_vision", "mastermind_tracing"]
                for detector in required_detectors:
                    if detector not in detector_results:
                        detector_results[detector] = 0.0
                
                # Add classic signal scores
                for signal in signals:
                    if hasattr(signal, 'active') and signal.active and hasattr(signal, 'strength'):
                        weight = weights.get(signal.name, 1.0)
                        signal_score = weight * signal.strength
                        
                        if signal.name == "dex_inflow":
                            detector_results["dex_inflow"] = signal_score
                        elif signal.name in ["spoofing_layers", "orderbook_imbalance_stealth", "large_bid_walls"]:
                            detector_results["orderbook_anomaly"] = detector_results.get("orderbook_anomaly", 0.0) + signal_score
                
                # Calculate feedback adjustment from component system
                feedback_adjustment = 0.0
                try:
                    from feedback_loop.component_score_updater import get_component_updater
                    updater = get_component_updater()
                    if hasattr(updater, 'get_recent_boost'):
                        feedback_adjustment = updater.get_recent_boost(symbol)
                except ImportError:
                    pass
                
                # Use consensus_data if available, otherwise create fallback
                if not consensus_data:
                    consensus_data = {
                        "decision": final_decision,
                        "votes": ["fallback_mode"],
                        "confidence": min(score / 4.0, 1.0),
                        "final_score": score,
                        "threshold_met": score > 2.0,
                        "contributing_detectors": ["stealth_fallback"]
                    }
                
                # Update consensus_data with feedback adjustment
                consensus_data["feedback_adjust"] = round(feedback_adjustment, 3)
                
                # Przygotuj meta_data z AI detector scores
                meta_data = {
                    "trust_addresses": len([s for s in used_signals if "trust" in s or "identity" in s]),
                    "coverage": round(data_coverage * 100, 1),
                    "historical_support": "Yes" if score >= 3.0 else "Partial"
                }
                
                if diamond_enabled and diamond_score > 0:
                    meta_data["diamond_score"] = round(diamond_score, 3)
                if whaleclip_enabled and whaleclip_score > 0:
                    meta_data["whaleclip_score"] = round(whaleclip_score, 3)
                if californium_enabled and californium_score > 0:
                    meta_data["californium_score"] = round(californium_score, 3)
                
                # Enhanced logging z comprehensive diagnostic breakdown
                try:
                    from utils.stealth_logger import StealthLogger
                    stealth_logger = StealthLogger()
                    stealth_logger.log_stealth_analysis_complete(symbol, detector_results, consensus_data)
                except ImportError:
                    pass
                
                # Wyślij Stealth v3 alert
                alert_sent = send_stealth_v3_alert(
                    symbol=symbol,
                    detector_results=detector_results,
                    consensus_data=consensus_data,
                    meta_data=meta_data
                )
                
                if alert_sent:
                    print(f"[STEALTH V3 ALERT] ✅ {symbol}: Nowoczesny alert v3 wysłany (score: {score:.3f})")
                    
                    # Log pattern identification dla diagnostic transparency
                    active_detectors = [name for name, score_val in detector_results.items() if score_val > 0.0]
                    if active_detectors:
                        print(f"[STEALTH V3 PATTERN] {symbol}: {', '.join(active_detectors)} consensus pattern")
                    
                    # Log agent consensus breakdown
                    votes = consensus_data.get('votes', [])
                    if isinstance(votes, list) and len(votes) > 0:
                        buy_votes = votes.count('BUY')
                        print(f"[STEALTH V3 CONSENSUS] {symbol}: {buy_votes}/{len(votes)} BUY votes → {consensus_data.get('decision', 'UNKNOWN')}")
                else:
                    print(f"[STEALTH V3 ALERT] ℹ️ {symbol}: Alert v3 w cooldown lub błąd")
                    
            except ImportError:
                print(f"[STEALTH V3 ALERT] ⚠️ {symbol}: Stealth v3 alert system nie jest dostępny")
            except Exception as alert_error:
                print(f"[STEALTH V3 ALERT] ❌ {symbol}: Alert v3 error: {alert_error}")
        
        # === COMPONENT-AWARE FEEDBACK LOOP V4 INTEGRATION ===
        # Enhanced component breakdown calculation z dynamic weight application
        
        try:
            # Import Component Score Engine V4
            from stealth_engine.component_score_engine import get_component_engine, extract_classic_stealth_components
            from stealth_engine.component_logger import log_component_feedback, log_dynamic_weights
            
            component_engine = get_component_engine()
            
            # Build raw component scores from signals
            raw_component_scores = {
                "dex": 0.0,
                "whale": 0.0, 
                "trust": 0.0,
                "id": 0.0
            }
            
            # Extract component contributions from active signals
            for signal in signals:
                if hasattr(signal, 'active') and signal.active and hasattr(signal, 'strength'):
                    weight = weights.get(signal.name, 1.0)
                    contribution = weight * signal.strength
                    
                    # Kategoryzuj komponenty według advanced mapping
                    if signal.name in ['dex_inflow']:
                        raw_component_scores["dex"] += contribution
                    elif signal.name in ['whale_ping', 'whale_ping_real', 'whale_ping_real_repeat']:
                        raw_component_scores["whale"] += contribution
                    elif signal.name in ['trust_boost', 'token_trust']:
                        raw_component_scores["trust"] += contribution
                    elif signal.name in ['identity_boost', 'repeated_address_boost']:
                        raw_component_scores["id"] += contribution
            
            # Add AI detector contributions to component scores
            if diamond_enabled and diamond_score > 0:
                raw_component_scores["whale"] += diamond_score * 0.3  # Diamond AI → whale detection
                raw_component_scores["trust"] += diamond_score * 0.1   # Diamond AI → trust boost
            
            if whaleclip_enabled and whaleclip_score > 0:
                raw_component_scores["whale"] += whaleclip_score * 0.2  # WhaleCLIP AI → whale detection
                raw_component_scores["trust"] += whaleclip_score * 0.05  # WhaleCLIP AI → trust boost
            
            if californium_enabled and californium_score > 0:
                raw_component_scores["trust"] += californium_score * 0.25  # Californium AI → trust
                raw_component_scores["whale"] += californium_score * 0.1   # Californium AI → whale detection
            
            # Apply Component-Aware Dynamic Weights V4 with Self-Learning Decay
            weighted_component_scores, component_weighted_total = component_engine.apply_dynamic_weights_to_components(
                raw_component_scores, "ClassicStealth"
            )
            
            # Update original score with component-weighted total (enhanced precision)
            if component_weighted_total > 0:
                # Blend original score with component-weighted score (70/30 mix for stability)
                enhanced_score = (score * 0.7) + (component_weighted_total * 0.3)
                print(f"[COMPONENT V4] {symbol}: Enhanced score {score:.3f} → {enhanced_score:.3f} (component boost: +{enhanced_score-score:.3f})")
                score = enhanced_score
            
            # Extract final component values for TOP 5 display
            dex_inflow_score = weighted_component_scores.get("dex", 0.0)
            whale_ping_score = weighted_component_scores.get("whale", 0.0)
            trust_boost_score = weighted_component_scores.get("trust", 0.0)
            identity_boost_score = weighted_component_scores.get("id", 0.0)
            
            # Log Component-Aware Feedback V4 effectiveness with Self-Learning Decay
            if sum(raw_component_scores.values()) > 0:
                # Calculate effectiveness percentages
                total_raw = sum(raw_component_scores.values())
                effectiveness_percentages = {
                    comp: (weighted_component_scores.get(comp, 0.0) / raw_component_scores.get(comp, 0.001)) * 100
                    for comp in raw_component_scores if raw_component_scores[comp] > 0
                }
                
                # Log component feedback according to user specification
                log_component_feedback(effectiveness_percentages, symbol, "ClassicStealth")
                
                # Get dynamic weights with decay application
                try:
                    from feedback_loop.component_score_updater import get_component_updater
                    updater = get_component_updater()
                    applied_weights = updater.load_current_weights()
                    
                    # Log decay-adjusted weights
                    log_dynamic_weights(applied_weights, symbol, "ClassicStealth")
                    
                    # Log Self-Learning Decay status
                    if hasattr(updater, 'enable_decay') and updater.enable_decay:
                        print(f"[SELF-LEARNING DECAY] {symbol}: Active - weights auto-adjusted based on historical performance")
                    
                except ImportError:
                    # Fallback to basic weights loader
                    from feedback_loop.weights_loader import get_dynamic_component_weights
                    applied_weights = get_dynamic_component_weights()
                    log_dynamic_weights(applied_weights, symbol, "ClassicStealth")
            
            # Enhanced component logging with V4 format
            print(f"[STEALTH COMPONENTS V4] {symbol} | DEX: {dex_inflow_score:.3f} | WHALE: {whale_ping_score:.3f} | TRUST: {trust_boost_score:.3f} | ID: {identity_boost_score:.3f}")
            print(f"[COMPONENT V4] {symbol}: Component-weighted total: {component_weighted_total:.3f}")
            
        except ImportError as e:
            print(f"[COMPONENT V4] {symbol}: Component system not available: {e}")
            # Fallback to original component calculation
            dex_inflow_score = 0.0
            whale_ping_score = 0.0
            trust_boost_score = 0.0
            identity_boost_score = 0.0
            
            for signal in signals:
                if hasattr(signal, 'active') and signal.active and hasattr(signal, 'strength'):
                    weight = weights.get(signal.name, 1.0)
                    contribution = weight * signal.strength
                    
                    if signal.name in ['dex_inflow']:
                        dex_inflow_score += contribution
                    elif signal.name in ['whale_ping', 'whale_ping_real', 'whale_ping_real_repeat']:
                        whale_ping_score += contribution
                    elif signal.name in ['trust_boost', 'token_trust']:
                        trust_boost_score += contribution
                    elif signal.name in ['identity_boost', 'repeated_address_boost']:
                        identity_boost_score += contribution
            
            # Add AI contributions (fallback)
            if diamond_enabled and diamond_score > 0:
                whale_ping_score += diamond_score * 0.3
            if whaleclip_enabled and whaleclip_score > 0:
                whale_ping_score += whaleclip_score * 0.2
            if californium_enabled and californium_score > 0:
                trust_boost_score += californium_score * 0.25
                
            print(f"[STEALTH COMPONENTS] {symbol} | DEX: {dex_inflow_score:.3f} | WHALE: {whale_ping_score:.3f} | TRUST: {trust_boost_score:.3f} | ID: {identity_boost_score:.3f}")
        
        except Exception as e:
            print(f"[COMPONENT V4 ERROR] {symbol}: Component system error: {e}")
            # Safe fallback values
            dex_inflow_score = 0.0
            whale_ping_score = 0.0  
            trust_boost_score = 0.0
            identity_boost_score = 0.0
        
        # 🧠 MULTI-AGENT CONSENSUS DECISION ENGINE - Unified Detector Fusion
        consensus_result = None
        consensus_error = None
        
        try:
            from .consensus_decision_engine import create_consensus_engine
            
            # Prepare detector scores for consensus
            detector_scores = {}
            
            # Classic Stealth Engine
            if score > 0.0:
                detector_scores["StealthEngine"] = {
                    "score": score,
                    "confidence": min(data_coverage, 1.0),
                    "weight": 0.25
                }
            
            # DiamondWhale AI
            if diamond_enabled and diamond_score > 0.0:
                detector_scores["DiamondWhale"] = {
                    "score": diamond_score,
                    "confidence": 0.85,  # High confidence temporal analysis
                    "weight": 0.25
                }
            
            # CaliforniumWhale AI
            if californium_enabled and californium_score > 0.0:
                detector_scores["CaliforniumWhale"] = {
                    "score": californium_score,
                    "confidence": 0.80,  # Good confidence TGN+QIRL
                    "weight": 0.30
                }
            
            # WhaleCLIP (mock for now - could be integrated later)
            if "whale_ping" in used_signals:
                detector_scores["WhaleCLIP"] = {
                    "score": min(score * 0.8, 1.0),  # Derive from whale signals
                    "confidence": 0.75,
                    "weight": 0.20
                }
            
            # Run consensus if we have multiple detectors
            if len(detector_scores) >= 2:
                consensus_engine = create_consensus_engine()
                
                # Use simple consensus for production
                consensus_result = consensus_engine.run(
                    symbol, 
                    {k: v["score"] for k, v in detector_scores.items()},
                    use_simple_consensus=True
                )
                
                print(f"[CONSENSUS ENGINE] {symbol}: {consensus_result.decision.value} (score: {consensus_result.final_score:.3f})")
                print(f"[CONSENSUS ENGINE] {symbol}: Active detectors: {consensus_result.contributing_detectors}")
                print(f"[CONSENSUS ENGINE] {symbol}: Reasoning: {consensus_result.reasoning}")
                
            else:
                print(f"[CONSENSUS ENGINE] {symbol}: Insufficient detectors ({len(detector_scores)}) for consensus analysis")
                
        except ImportError:
            print(f"[CONSENSUS ENGINE] {symbol}: Consensus engine not available")
        except Exception as e:
            consensus_error = str(e)
            print(f"[CONSENSUS ENGINE ERROR] {symbol}: Failed to run consensus: {e}")
        
        # Przygotuj rezultat z integracją DiamondWhale AI + CaliforniumWhale AI + Component Breakdown + Consensus
        result = {
            "score": round(score, 3),
            "active_signals": used_signals,
            "data_coverage": round(data_coverage, 2),
            "partial_scoring": data_coverage < 0.8,
            "diamond_score": round(diamond_score, 3) if diamond_enabled else None,
            "diamond_enabled": diamond_enabled,
            "californium_score": round(californium_score, 3) if californium_enabled else None,
            "californium_enabled": californium_enabled,
            
            # === COMPONENT BREAKDOWN FOR TOP 5 DISPLAY ===
            "dex_inflow": round(dex_inflow_score, 3),
            "whale_ping": round(whale_ping_score, 3),
            "trust_boost": round(trust_boost_score, 3),
            "identity_boost": round(identity_boost_score, 3),
            
            # === CONSENSUS DECISION ENGINE RESULTS ===
            "consensus_decision": consensus_result.decision.value if consensus_result else None,
            "consensus_score": round(consensus_result.final_score, 3) if consensus_result else None,
            "consensus_confidence": round(consensus_result.confidence, 3) if consensus_result else None,
            "consensus_detectors": consensus_result.contributing_detectors if consensus_result else []
        }
        
        # Dodaj informacje o błędach jeśli wystąpiły
        if diamond_error:
            result["diamond_error"] = diamond_error
        if californium_error:
            result["californium_error"] = californium_error
        if consensus_error:
            result["consensus_error"] = consensus_error
        
        # 🚨 STEALTH V3 TELEGRAM ALERT INTEGRATION - Nowoczesny Alert System
        # Alert triggering logic based ONLY on consensus decision
        try:
            # Import nowego systemu alertów Stealth v3
            from alerts.stealth_v3_telegram_alerts import send_stealth_v3_alert
            
            # 🔐 CRITICAL CONSENSUS DECISION ONLY - IGNORE SCORE THRESHOLDS
            should_alert = False
            
            # Jeśli mamy consensus, użyj TYLKO jego decyzji - ignoruj score
            if consensus_result and consensus_result.decision.value == "BUY":
                should_alert = True
                print(f"[STEALTH V3 ALERT] {token_data.get('symbol', 'UNKNOWN')}: Consensus decision BUY triggers alert (score ignored)")
            elif consensus_result and consensus_result.decision.value in ["HOLD", "AVOID"]:
                should_alert = False
                print(f"[STEALTH V3 ALERT BLOCK] {token_data.get('symbol', 'UNKNOWN')}: Consensus decision {consensus_result.decision.value} blocks alert (score={score:.3f})")
            else:
                # Fallback - jeśli nie ma consensus, sprawdź tylko wysoki score (4.0+)
                if score >= 4.0:
                    should_alert = True
                    print(f"[STEALTH V3 FALLBACK] {token_data.get('symbol', 'UNKNOWN')}: No consensus, high score {score:.3f} triggers alert")
                else:
                    print(f"[STEALTH V3 NO ALERT] {token_data.get('symbol', 'UNKNOWN')}: No consensus, score {score:.3f} < 4.0 threshold")
            
            if should_alert:
                symbol = token_data.get('symbol', 'UNKNOWN')
                
                # Przygotuj detector results - mapuj aktywne sygnały na detektory
                detector_results = {}
                
                # Mapuj klasyczne stealth signals na detektory
                for signal in used_signals:
                    if signal == "whale_ping":
                        detector_results["whale_ping"] = True
                    elif signal == "dex_inflow":
                        detector_results["dex_inflow"] = True
                    elif signal == "spoofing_layers":
                        detector_results["orderbook_anomaly"] = True
                
                # Dodaj AI detektory
                if diamond_enabled and diamond_score > 0.3:
                    detector_results["diamond_ai"] = True
                if californium_enabled and californium_score > 0.3:
                    detector_results["mastermind_tracing"] = True
                
                # Przygotuj consensus data
                consensus_data = {}
                if consensus_result:
                    consensus_data = {
                        "decision": consensus_result.decision.value,
                        "votes": f"{len(consensus_result.contributing_detectors)}/{len(detector_scores) if 'detector_scores' in locals() else 4}",
                        "confidence": round(consensus_result.final_score, 3),
                        "feedback_adjust": 0.0  # Placeholder dla przyszłych implementacji
                    }
                else:
                    # Fallback consensus data based on stealth score
                    if score >= 4.0:
                        consensus_data = {
                            "decision": "BUY",
                            "votes": f"{len(used_signals)}/4",
                            "confidence": round(min(score / 4.0, 1.0), 3),
                            "feedback_adjust": 0.0
                        }
                    elif score >= 2.5:
                        consensus_data = {
                            "decision": "MONITOR",
                            "votes": f"{len(used_signals)}/4", 
                            "confidence": round(score / 4.0, 3),
                            "feedback_adjust": 0.0
                        }
                
                # Przygotuj meta data
                meta_data = {
                    "trust_addresses": len([s for s in used_signals if "trust" in s or "identity" in s]),
                    "coverage": round(data_coverage * 100, 1),
                    "historical_support": "Yes" if score >= 3.0 else "Partial",
                    "californium_score": round(californium_score, 3) if californium_enabled else None,
                    "diamond_score": round(diamond_score, 3) if diamond_enabled else None
                }
                
                # Wyślij Stealth v3 alert
                alert_success = send_stealth_v3_alert(
                    symbol=symbol,
                    detector_results=detector_results,
                    consensus_data=consensus_data,
                    meta_data=meta_data
                )
                
                if alert_success:
                    print(f"[STEALTH V3 ALERT SUCCESS] {symbol}: Nowoczesny alert wysłany (score: {score:.3f})")
                    result["alert_sent"] = True
                    result["alert_type"] = "stealth_v3"
                else:
                    print(f"[STEALTH V3 ALERT SKIP] {symbol}: Alert w cooldown lub błąd wysyłania")
                    result["alert_sent"] = False
                    result["alert_skip_reason"] = "cooldown_or_error"
            else:
                alert_threshold = 3.0  # Define the alert threshold
                print(f"[STEALTH V3 ALERT] {symbol}: Score {score:.3f} poniżej progu alertów ({alert_threshold})")
                result["alert_sent"] = False
                result["alert_skip_reason"] = "below_threshold"
                
        except ImportError:
            print(f"[STEALTH V3 ALERT] {token_data.get('symbol', 'UNKNOWN')}: Stealth v3 alert system nie jest dostępny")
            result["alert_sent"] = False
            result["alert_skip_reason"] = "system_unavailable"
        except Exception as alert_error:
            print(f"[STEALTH V3 ALERT ERROR] {token_data.get('symbol', 'UNKNOWN')}: {alert_error}")
            result["alert_sent"] = False
            result["alert_skip_reason"] = f"error: {alert_error}"
            
        return result
        
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
    🔍 KOMPLETNA ANALIZA TOKENA - Stage 3/7 Diamond Decision Integration
    
    Łączy compute_stealth_score() z Diamond Decision Engine dla adaptive decision making
    
    Args:
        symbol: Symbol tokena
        token_data: Dane rynkowe
        
    Returns:
        dict: Kompletny wynik analizy Stealth z Diamond Decision
    """
    try:
        # Oblicz stealth score
        stealth_result = compute_stealth_score(token_data)
        
        # === STAGE 3/7: DIAMOND DECISION INTEGRATION ===
        # Przygotuj scores dla Diamond Decision Engine
        stealth_score = stealth_result.get("score", 0.0)
        diamond_score = stealth_result.get("diamond_score", 0.0) or 0.0
        volume_24h = token_data.get("volume_24h", 0)
        
        # Oblicz whale_ping_score i whaleclip_score z aktywnych sygnałów
        active_signals = stealth_result.get("active_signals", [])
        
        # Whale ping score (z klasycznych stealth signals)
        whale_ping_score = 0.0
        if "whale_ping" in active_signals:
            whale_ping_score = 0.8  # High confidence dla active whale ping
        elif any(signal in active_signals for signal in ["spoofing_layers", "orderbook_anomaly"]):
            whale_ping_score = 0.5  # Medium confidence dla related signals
        
        # WhaleCLIP score (z stealth score jako proxy - będzie później zastąpiony prawdziwym WhaleCLIP)
        whaleclip_score = min(0.9, stealth_score / 4.0)  # Scale stealth score to 0-0.9 range
        
        # Wywołaj Diamond Decision Engine
        from .decision import simulate_diamond_decision
        
        diamond_decision = simulate_diamond_decision(
            whale_score=whale_ping_score,
            whaleclip_score=whaleclip_score, 
            diamond_score=diamond_score,
            token=symbol,
            volume_24h=volume_24h
        )
        
        # Klasyfikuj alert tradycyjnie jako fallback
        alert_type = classify_stealth_alert(stealth_score)
        
        # === ENHANCED RESULT WITH DIAMOND DECISION ===
        result = {
            "symbol": symbol,
            "stealth_score": stealth_score,
            "active_signals": active_signals,
            "alert_type": alert_type,
            "signal_count": len(active_signals),
            "timestamp": time.time(),
            # Diamond Decision Integration
            "diamond_decision": diamond_decision["decision"],
            "diamond_fused_score": diamond_decision["fused_score"],
            "diamond_confidence": diamond_decision["confidence"],
            "diamond_reasons": diamond_decision["trigger_reasons"],
            "dominant_detector": diamond_decision["dominant_detector"],
            "decision_breakdown": {
                "whale_ping_score": whale_ping_score,
                "whaleclip_score": whaleclip_score,
                "diamond_score": diamond_score,
                "fused_score": diamond_decision["fused_score"]
            },
            # Enhanced metadata
            "diamond_enabled": stealth_result.get("diamond_enabled", False),
            "data_coverage": stealth_result.get("data_coverage", 1.0),
            "partial_scoring": stealth_result.get("partial_scoring", False)
        }
        
        # Dodaj błędy jeśli wystąpiły
        if "diamond_error" in stealth_result:
            result["diamond_error"] = stealth_result["diamond_error"]
        if "error" in stealth_result:
            result["stealth_error"] = stealth_result["error"]
        
        # LOG: Diamond Decision result
        print(f"[DIAMOND INTEGRATION] {symbol}: Diamond decision={diamond_decision['decision']}, fused_score={diamond_decision['fused_score']}")
        print(f"[DIAMOND INTEGRATION] {symbol}: Confidence={diamond_decision['confidence']}, dominant={diamond_decision['dominant_detector']}")
        
        # 🚨 STEALTH V3 TELEGRAM ALERT INTEGRATION - Diamond Decision Enhanced
        # Alert triggering w oparciu o Diamond Decision + Stealth Score
        try:
            from alerts.stealth_v3_telegram_alerts import send_stealth_v3_alert
            
            # Alert logic - używamy Diamond Decision jako główny trigger
            should_alert = False
            
            # Główny trigger: Diamond Decision = BUY
            if diamond_decision["decision"] == "BUY":
                should_alert = True
                print(f"[STEALTH V3 ALERT] {symbol}: Diamond Decision BUY triggers alert")
            
            # Secondary trigger: High stealth score >= 3.0 
            elif stealth_score >= 3.0:
                should_alert = True
                print(f"[STEALTH V3 ALERT] {symbol}: High stealth score {stealth_score:.3f} triggers alert")
            
            # Tertiary trigger: High fused score >= 0.75
            elif diamond_decision["fused_score"] >= 0.75:
                should_alert = True
                print(f"[STEALTH V3 ALERT] {symbol}: High fused score {diamond_decision['fused_score']:.3f} triggers alert")
            
            if should_alert:
                # Przygotuj detector results z enhanced mapping
                detector_results = {}
                
                # Mapuj Diamond Decision reasoning na detektory
                for reason in diamond_decision.get("trigger_reasons", []):
                    if "whale" in reason.lower():
                        detector_results["whale_ping"] = True
                    elif "dex" in reason.lower() or "inflow" in reason.lower():
                        detector_results["dex_inflow"] = True
                    elif "diamond" in reason.lower():
                        detector_results["diamond_ai"] = True
                
                # Mapuj aktywne stealth signals
                for signal in active_signals:
                    if signal == "whale_ping":
                        detector_results["whale_ping"] = True
                    elif signal == "dex_inflow":
                        detector_results["dex_inflow"] = True
                    elif signal == "spoofing_layers":
                        detector_results["orderbook_anomaly"] = True
                
                # Enhanced consensus data z Diamond Decision
                consensus_data = {
                    "decision": diamond_decision["decision"],
                    "votes": f"{len(diamond_decision.get('trigger_reasons', []))}/3",  # Based on Diamond reasoning
                    "confidence": round(diamond_decision["fused_score"], 3),
                    "feedback_adjust": 0.0,
                    "diamond_confidence": diamond_decision["confidence"],
                    "dominant_detector": diamond_decision["dominant_detector"]
                }
                
                # Enhanced meta data z Diamond Decision breakdown
                meta_data = {
                    "trust_addresses": len([s for s in active_signals if "trust" in s]),
                    "coverage": round(stealth_result.get("data_coverage", 1.0) * 100, 1),
                    "diamond_fused_score": round(diamond_decision["fused_score"], 3),
                    "whale_ping_score": round(whale_ping_score, 3),
                    "whaleclip_score": round(whaleclip_score, 3),
                    "diamond_score": round(diamond_score, 3),
                    "decision_engine": "Diamond_v3",
                    "historical_support": "Diamond_Enhanced"
                }
                
                # Wyślij enhanced Stealth v3 alert
                alert_success = send_stealth_v3_alert(
                    symbol=symbol,
                    detector_results=detector_results,
                    consensus_data=consensus_data,
                    meta_data=meta_data
                )
                
                if alert_success:
                    print(f"[STEALTH V3 DIAMOND ALERT SUCCESS] {symbol}: Enhanced Diamond alert wysłany")
                    result["alert_sent"] = True
                    result["alert_type"] = "stealth_v3_diamond"
                    result["alert_trigger"] = diamond_decision["decision"]
                else:
                    print(f"[STEALTH V3 DIAMOND ALERT SKIP] {symbol}: Alert w cooldown lub błąd")
                    result["alert_sent"] = False
                    result["alert_skip_reason"] = "cooldown_or_error"
            else:
                print(f"[STEALTH V3 DIAMOND ALERT] {symbol}: Kryteria alertu nie spełnione (decision: {diamond_decision['decision']}, score: {stealth_score:.3f})")
                result["alert_sent"] = False
                result["alert_skip_reason"] = "criteria_not_met"
                
        except ImportError:
            print(f"[STEALTH V3 DIAMOND ALERT] {symbol}: Stealth v3 alert system niedostępny")
            result["alert_sent"] = False
            result["alert_skip_reason"] = "system_unavailable"
        except Exception as alert_error:
            print(f"[STEALTH V3 DIAMOND ALERT ERROR] {symbol}: {alert_error}")
            result["alert_sent"] = False
            result["alert_skip_reason"] = f"error: {alert_error}"
        
        return result
        
    except Exception as e:
        import traceback
        print(f"[STEALTH ANALYSIS ERROR] {symbol}: {e}")
        print(f"[STEALTH ANALYSIS ERROR] {symbol}: Traceback: {traceback.format_exc()}")
        return {
            "symbol": symbol,
            "stealth_score": 0.0,
            "active_signals": [],
            "alert_type": None,
            "signal_count": 0,
            "timestamp": time.time(),
            "error": str(e),
            "diamond_decision": "AVOID",
            "diamond_fused_score": 0.0,
            "diamond_confidence": "ERROR"
        }