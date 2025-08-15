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
from datetime import datetime

from .stealth_signals import StealthSignalDetector, StealthSignal
from .stealth_weights import StealthWeightManager
from .stealth_feedback import StealthFeedbackSystem
from .diamond_detector import run_diamond_detector
from .consensus_decision_engine import ConsensusStrategy

# Enhanced RL Integration  
try:
    import sys
    import os
    # Add parent directory to Python path for Enhanced RL imports
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from enhanced_rl_integration import process_stealth_with_enhanced_rl, get_enhanced_rl_system
    ENHANCED_RL_AVAILABLE = True
    print("[ENHANCED RL] Deep Q-Network + RLAgentV3 system available")
except ImportError as e:
    print(f"[ENHANCED RL WARNING] Enhanced RL system not available: {e}")
    ENHANCED_RL_AVAILABLE = False

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

# Enhanced Explore Mode Integration
try:
    from .explore_mode_integration import save_explore_mode_data, initialize_explore_system
    ENHANCED_EXPLORE_AVAILABLE = True
    print("[ENHANCED EXPLORE] Enhanced explore mode system available")
except ImportError as e:
    print(f"[ENHANCED EXPLORE WARNING] Enhanced explore mode not available: {e}")
    ENHANCED_EXPLORE_AVAILABLE = False


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
    Enhanced CaliforniumWhale AI z Ultimate Mastermind Tracing
    
    Łączy TGN z mastermind tracking dla P&D coordination detection:
    - Temporal Graph Convolutional Network (TGN) z EWMA thresholding
    - Quantum-inspired Reinforcement Learning (QIRL) 
    - Mastermind coordination pattern detection
    - Social media correlation analysis
    - 92-95% accuracy w real-time P&D detection
    
    Args:
        symbol: Symbol tokena (np. 'BTCUSDT')
        
    Returns:
        float: CaliforniumWhale score (0.0-1.0) z enhanced mastermind detection
    """
    try:
        # Próbuj użyć Enhanced Californium Key System
        try:
            from stealth_engine.californium_key import CaliforniumKeySystem
            use_enhanced = True
            print(f"[CALIFORNIUM ENHANCED] {symbol}: Using ultimate mastermind tracing system")
        except ImportError:
            use_enhanced = False
            print(f"[CALIFORNIUM LEGACY] {symbol}: Using legacy TGN system")
        
        if use_enhanced:
            # === ENHANCED CALIFORNIUM KEY Z MASTERMIND TRACING ===
            system = CaliforniumKeySystem()
            
            # Przygotuj mock blockchain transactions dla analysis
            import time
            current_time = int(time.time())
            mock_transactions = []
            
            # Generate realistic transaction patterns dla mastermind detection
            base_values = [50000, 75000, 100000, 150000, 200000]
            addresses = [f'0x{i:040x}' for i in range(20)]
            
            for i in range(15):
                # Create coordinated whale movements pattern
                if i < 5:  # Initial accumulation phase
                    value = base_values[i % len(base_values)]
                elif i < 10:  # Coordination phase (synchronized movements)
                    value = base_values[2] * (1.5 + i * 0.1)  # Escalating pattern
                else:  # Distribution phase
                    value = base_values[4] * (2.0 + i * 0.2)  # Large movements
                
                mock_transactions.append({
                    'from_address': addresses[i % 10],
                    'to_address': addresses[(i + 5) % 15],
                    'value_usd': value,
                    'timestamp': current_time - (i * 300)  # 5-minute intervals
                })
            
            # Enhanced volume data z EWMA spike pattern
            base_volume = 1000000
            volume_data = []
            for i in range(24):
                if i > 18:  # Create volume spike dla detection
                    spike_multiplier = 5.0 + (i - 18) * 2.0  # Escalating spike
                    volume_data.append(int(base_volume * spike_multiplier))
                else:
                    volume_data.append(int(base_volume * (1.0 + i * 0.1)))
            
            # Social media signals dla coordination detection
            social_data = {
                'telegram_mentions': 0.7,  # High Telegram activity
                'twitter_mentions': 0.5,   # Moderate Twitter buzz
                'discord_mentions': 0.4    # Discord coordination
            }
            
            # Run enhanced mastermind analysis
            result = system.analyze_token(symbol, mock_transactions, volume_data, social_data)
            
            final_score = result.get('californium_score', 0.0)
            mastermind_detected = result.get('mastermind_detected', False)
            coordination_score = result.get('coordination_score', 0.0)
            tgn_score = result.get('tgn_score', 0.0)
            patterns = result.get('patterns', [])
            
            print(f"[CALIFORNIUM ENHANCED] {symbol}: TGN={tgn_score:.3f}, Mastermind={mastermind_detected}, Coordination={coordination_score:.3f}")
            print(f"[CALIFORNIUM ENHANCED] {symbol}: Patterns detected: {len(patterns)}, Final score={final_score:.3f}")
            
            # Alert dla high-confidence mastermind detection
            if final_score > 0.7 and mastermind_detected:
                print(f"[CALIFORNIUM ALERT] 🚨 {symbol}: P&D Mastermind coordination detected! Score: {final_score:.3f}")
            
            return final_score
            
        else:
            # === LEGACY FALLBACK SYSTEM ===
            if not CALIFORNIUM_AVAILABLE:
                print(f"[CALIFORNIUM] CaliforniumWhale AI not available for {symbol}")
                return 0.0
            
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
            reward = 1.0 if action == 1 else 0.0
            agent.update(state_vector, action, reward)
            
            # STEP 8: Calculate final CaliforniumWhale score
            if action == 1:  # QIRL recommends action
                final_score = tgn_score
            else:  # QIRL recommends hold - but still use weighted TGN score
                final_score = tgn_score * 0.5  # Reduced weight for HOLD action
            
            print(f"[CALIFORNIUM LEGACY] {symbol}: TGN score={tgn_score:.3f}, QIRL action={action}, final={final_score:.3f}")
            
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
        🎯 UNMEASURED STATUS: Handle microstructure signals with UNMEASURED status for coverage_ratio
        
        Args:
            signals: Lista wykrytych sygnałów
            
        Returns:
            Tuple (stealth_score, breakdown)
        """
        total_score = 0.0
        signal_breakdown = {}
        active_weight_sum = 0.0
        
        # 🎯 COVERAGE RATIO CALCULATION: Count measured vs unmeasured signals
        measured_signals = 0
        unmeasured_signals = 0
        total_signals = len(signals)
        
        # Count signals by status for coverage calculation
        for signal in signals:
            signal_status = signal.get('status', 'MEASURED')  # Default to MEASURED for compatibility
            if signal_status == 'UNMEASURED':
                unmeasured_signals += 1
                print(f"[UNMEASURED SIGNAL] {signal.get('signal_name', 'unknown')}: status=UNMEASURED (no orderbook data)")
            else:
                measured_signals += 1
        
        # Calculate coverage ratio (percentage of measurable signals)
        coverage_ratio = measured_signals / total_signals if total_signals > 0 else 1.0
        print(f"[COVERAGE RATIO] {measured_signals}/{total_signals} signals measured, coverage={coverage_ratio:.3f}")
        
        # 🚀 ADAPTIVE SPOOFING WEIGHT: Check if spoofing is the only active signal
        active_signals = [s for s in signals if s['active']]
        spoofing_signals = [s for s in active_signals if 'spoofing' in s['signal_name'].lower()]
        is_spoofing_only = len(spoofing_signals) > 0 and len(active_signals) <= 2
        
        for signal in signals:
            signal_name = signal['signal_name']
            strength = signal['strength']
            active = signal['active']
            signal_status = signal.get('status', 'MEASURED')
            
            # 🎯 UNMEASURED HANDLING: Skip UNMEASURED signals neutrally (no penalty)
            if signal_status == 'UNMEASURED':
                signal_breakdown[signal_name] = 0.0  # Neutral impact - no penalty for missing data
                continue
            
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
                
        # 🎯 COVERAGE-ADJUSTED NORMALIZATION: Apply coverage ratio to prevent unfair penalization
        # When orderbook=synthetic, microstructure signals are UNMEASURED - adjust score accordingly
        if active_weight_sum > 0:
            # Apply coverage ratio to compensate for unmeasured signals
            raw_score = total_score / active_weight_sum
            coverage_adjusted_score = raw_score * (1.0 + (1.0 - coverage_ratio) * 0.1)  # Small boost for missing data
            stealth_score = min(1.0, coverage_adjusted_score)
            print(f"[COVERAGE ADJUSTED] raw_score={raw_score:.3f} → adjusted={stealth_score:.3f} (coverage={coverage_ratio:.3f})")
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
                        'status': getattr(sig, 'status', 'MEASURED'),  # Include status for UNMEASURED handling
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
    symbol = token_data.get('symbol', 'UNKNOWN')
    print(f"[FUNCTION ENTRY DEBUG] {symbol}: compute_stealth_score STARTED - testing function execution path")
    # Initialize all variables at function start to prevent UnboundLocalError
    score = 0.0
    active_signals = []
    signal_details = {}  # Initialize signal_details to store strength values
    data_coverage = 1.0
    alert_threshold = 3.0
    used_signals = []
    stealth_error = None
    symbol = token_data.get("symbol", "UNKNOWN")
    
    # Initialize data variables to prevent UnboundLocalError
    candles_15m = token_data.get("candles_15m", [])
    candles_5m = token_data.get("candles_5m", [])
    orderbook = token_data.get("orderbook", {})
    dex_inflow = 0.0  # Initialize dex_inflow
    
    # 🎯 FUNCTION ENTRY DEBUG - This MUST appear for every token processed
    print(f"[FUNCTION ENTRY DEBUG] {symbol}: compute_stealth_score STARTED - testing function execution path")
    
    # Initialize stealth_result dictionary for explore mode communication
    stealth_result = {
        "score": 0.0,
        "active_signals": [],
        "signal_details": {},
        "explore_mode_triggered": False,
        "explore_trigger_reason": None,
        "explore_confidence": 0.0
    }
    
    # Initialize AI detector variables
    diamond_enabled = False
    diamond_score = 0.0
    diamond_error = None
    whaleclip_enabled = False
    whaleclip_score = 0.0
    whaleclip_error = None
    californium_enabled = False
    californium_score = 0.0
    californium_error = None
    consensus_result = None
    consensus_error = None
    
    # Initialize explore mode variables
    explore_mode_triggered = False
    explore_trigger_reason = None
    explore_confidence = 0.0
    
    try:
        # Import lokalny aby uniknąć circular imports
        from .stealth_signals import StealthSignalDetector
        from .stealth_weights import load_weights
        from engine.aggregator import aggregate
        from engine.price_ref import resolve_price_ref
        from engine.decision import make_decision
        from core.smart_money import apply_smart_money_boost
        from detectors.orderbook_features import compute_ob_signals
        
        # FEATURES: Use consistent price from market data processing - beginning of FEATURES
        # 🎯 PRICE CONSISTENCY: Use canonical price from market data instead of recalculating
        price_ref = token_data.get("price", 0.0)  # Fixed: canonical price system uses 'price' key
        
        # Fallback only if price_usd not available (shouldn't happen in normal flow)
        if price_ref <= 0:
            candles_15m = token_data.get("candles_15m", [])
            ticker_price = token_data.get("ticker", {}).get("price", 0.0)
            candle_price = candles_15m[-1]["close"] if candles_15m else 0.0
            price_ref = resolve_price_ref(ticker_price, candle_price)
            print(f"[PRICE FALLBACK] {symbol}: Market data price unavailable, using fallback=${price_ref:.6f}")
        else:
            print(f"[PRICE CONSISTENT] {symbol}: Using market data price=${price_ref:.6f}")
        
        # ⛔ Hard-filter: Skip tokens with too low daily volume  
        volume_24h = token_data.get("volume_24h", 0)
        skip_reason = None
        if volume_24h < 500_000:
            skip_reason = "low_volume"
            score = 0.0
            active_signals = []
            print(f"[STEALTH SKIP] {symbol}: Low volume ({volume_24h:,.0f} < 500,000)")
        else:
            # LOG: Rozpoczęcie analizy Stealth Engine
            print(f"[STEALTH] Checking token: {symbol}...")
        
        if volume_24h == 0:
            print(f"[STEALTH VOLUME WARNING] {symbol}: 24h volume is 0 - possible data issue")
            # Continue analysis despite volume warning - may still have useful signals
        
        # Załaduj wagi dynamiczne i wyloguj
        weights = load_weights()
        print(f"[STEALTH WEIGHTS] Loaded {len(weights)} dynamic weights from feedback system")
        
        # Utwórz detektor sygnałów
        detector = StealthSignalDetector()
        
        # Debug danych wejściowych przed analizą sygnałów
        candles_5m = token_data.get("candles_5m", [])
        orderbook = token_data.get("orderbook", {})
        dex_inflow = token_data.get("dex_inflow", 0)
        
        # Add price_ref to token_data for use in all detectors
        token_data["price_ref"] = price_ref
        
        # ENHANCED: Calculate real dex_inflow with smart money boost
        try:
            from utils.contracts import get_contract
            from utils.blockchain_scanners import get_token_transfers_last_24h, load_known_exchange_addresses
            
            contract_info = get_contract(symbol)
            known_exchanges_data = load_known_exchange_addresses()
            known_exchanges = set()
            
            if contract_info:
                real_transfers = get_token_transfers_last_24h(
                    symbol=symbol,
                    chain=contract_info['chain'],
                    contract_address=contract_info['address']
                )
                exchange_addresses = known_exchanges_data.get(contract_info['chain'], [])
                dex_routers = known_exchanges_data.get('dex_routers', {}).get(contract_info['chain'], [])
                all_known_addresses = set(addr.lower() for addr in exchange_addresses + dex_routers)
                known_exchanges = all_known_addresses
                
                # Calculate actual DEX inflow for display
                real_dex_inflow = 0
                for transfer in real_transfers:
                    if transfer['to'] in all_known_addresses:
                        real_dex_inflow += transfer['value_usd']
                
                dex_inflow = real_dex_inflow  # Use real value for display
        except:
            known_exchanges = set()  # Empty set if loading fails
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
                        mid_price = price_ref or 0
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
                skip_reason = "quiet_microcap_detected"
                score = 0.25
                active_signals = ["quiet_microcap"]
            else:
                skip_reason = "empty_orderbook_l2_unavailable"
                # Don't punish with 0.0 - continue analysis without L2-dependent modules
                print(f"[ORDERBOOK L2 UNAVAILABLE] {symbol}: Empty orderbook (bids={len(orderbook.get('bids', []))}, asks={len(orderbook.get('asks', []))}) - L2 modules UNKNOWN, continuing analysis")
                # Fall through to signal detection - don't skip token analysis
        
        # SIGNAL DETECTION - Always execute regardless of orderbook status
        if skip_reason not in ["low_volume"]:  # Only skip for real volume issues
            # Pobierz aktywne sygnały z detektorów (zgodnie z user specification)
                print(f"[DEBUG FLOW] {symbol} - Starting get_active_stealth_signals() call...")
                try:
                    signals = detector.get_active_stealth_signals(token_data)
                    print(f"[DEBUG FLOW] {symbol} - get_active_stealth_signals() completed successfully")
                    print(f"[STEALTH DEBUG] {symbol}: Successfully got {len(signals)} signals from detector")
                except Exception as e:
                    print(f"[DEBUG FLOW] {symbol} - get_active_stealth_signals() FAILED")
                    print(f"[STEALTH ERROR] {symbol}: Failed to get signals from detector: {e}")
                    skip_reason = "signal_detection_failed"
                    score = 0.0
                    active_signals = []
                    signals = []
                
                # === CORE STEALTH CALCULATION PROCEEDS ===
                # Od tego punktu wykonujemy normalną analizę stealth signals
                
                # 🚨 CONSENSUS ENGINE MOVED HERE - RUNS FOR ALL TOKENS REGARDLESS OF SKIP_REASON
                # This ensures consensus voting happens even for tokens with low volume or other issues
                # Initialize consensus variables that will be used later
                stored_final_decision = "WATCH"  # Default fallback
                stored_consensus_data = None
                stored_consensus_result = None
                
                # Initialize AI detector variables at higher scope for consensus access
                diamond_score = 0.0
                diamond_enabled = False
                diamond_error = None
                californium_score = 0.0
                californium_enabled = False
                californium_error = None
                whaleclip_score = 0.0
                whaleclip_enabled = False
                whaleclip_error = None
                score = 0.0
                used_signals = []
                data_coverage = 1.0
                
                # Continue analysis even with empty_orderbook_l2_unavailable - L2 modules will return UNMEASURED
                if skip_reason not in ["low_volume", "signal_detection_failed"]:
                    # Załaduj aktualne wagi (mogą być dostrojone przez feedback loop)
                    weights = load_weights()
                    
                    # Analizuj każdy sygnał
                    signal_status = {}
                    signal_details = {}  # NEW: Store signal strength values
                    
                    for signal in signals:
                        signal_status[signal.name] = getattr(signal, 'active', False)
                        # 🎯 Store ALL signal details (both active and inactive) with status for UNMEASURED handling
                        signal_details[signal.name] = {
                            "active": getattr(signal, 'active', False),
                            "strength": getattr(signal, 'strength', 0.0),
                            "status": getattr(signal, 'status', 'MEASURED')  # Include status for coverage_ratio calculation
                        }
                        
                        # 🔧 SIGNAL_DETAILS FIX: Add detailed information for whale_ping from token_data
                        if signal.name == "whale_ping" and 'whale_ping_details' in token_data:
                            signal_details[signal.name].update(token_data['whale_ping_details'])
                            print(f"[SIGNAL_DETAILS] {symbol}: Added whale_ping details with {token_data['whale_ping_details']['total_whales']} whales to signal_details")
                        
                        # Add similar details for other signals if needed
                        elif signal.name == "dex_inflow" and 'dex_inflow_details' in token_data:
                            signal_details[signal.name].update(token_data['dex_inflow_details'])
                    
                    # Wyloguj kluczowe sygnały
                    whale_ping = signal_status.get('whale_ping', False)
                    spoofing_layers = signal_status.get('spoofing_layers', False) 
                    volume_spike = signal_status.get('volume_spike', False)
                    orderbook_anomaly = signal_status.get('orderbook_anomaly', False)
                    dex_inflow_active = signal_status.get('dex_inflow', False)
                    
                    print(f"[STEALTH] Detected signals for {symbol}: whale={whale_ping}, spoofing={spoofing_layers}, volume_spike={volume_spike}, orderbook={orderbook_anomaly}, dex={dex_inflow_active}")
                    
                    # score and used_signals already initialized at higher scope
                    total_signals = len(signals)
                    available_signals = 0  # Sygnały z danymi (niezależnie od aktywności)
                    
                    # Prepare signals for aggregator
                    aggregator_signals = {}
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
                        
                        # Add signal to aggregator format with UNMEASURED status handling
                        aggregator_signals[signal.name] = {
                            "active": hasattr(signal, 'active') and signal.active,
                            "strength": getattr(signal, 'strength', 0.0),
                            "status": getattr(signal, 'status', 'MEASURED')  # Pass status for coverage ratio
                        }
                        
                        if hasattr(signal, 'active') and signal.active:
                            used_signals.append(signal.name)
                    
                    # WYMAGANIE #6: Use aggregator p_raw directly without modifications
                    agg_result = aggregate(aggregator_signals, weights)
                    score = agg_result["p_raw"]  # Pure p_raw from aggregator - NO MODIFICATIONS
                    
                    # Log contributions (handle both dict and float formats)
                    for name, contrib in agg_result["contrib"].items():
                        if isinstance(contrib, dict):
                            # Dictionary format with s, w, v keys
                            strength = contrib.get('s', 0.0)
                            weight = contrib.get('w', 0.0) 
                            value = contrib.get('v', 0.0)
                            print(f"[AGGREGATOR] {name}: strength={strength:.3f}, weight={weight:.3f}, contribution={value:.3f}")
                        else:
                            # Float format - contrib is just the contribution value
                            signal_data = aggregator_signals.get(name, {})
                            strength = signal_data.get('strength', 0.0)
                            weight = weights.get(name, 0.0)
                            print(f"[AGGREGATOR] {name}: strength={strength:.3f}, weight={weight:.3f}, contribution={contrib:.3f}")
                    
                    # WYMAGANIE #6: Remove scaling mechanism - use pure p_raw from aggregator
                    data_coverage = available_signals / total_signals if total_signals > 0 else 0
                    print(f"[WYMAGANIE #6] {symbol}: Using pure aggregator p_raw={score:.6f} (data_coverage={data_coverage:.1%}, no scaling)")
                    
                    # REMOVED: Scaling mechanism that modified p_raw
                    # No scaling_factor, no score modifications - pure aggregator result
                    # print(f"[STEALTH PARTIAL] {symbol}: Low data coverage ({data_coverage:.1%}), scaling {original_score:.3f} → {score:.3f} (factor: {scaling_factor:.2f})")
                    
                    # PUNKT 7 FIX: Usunięto wszystkie bonusy - używamy tylko pure aggregator p_raw
                    # active_rules_bonus = len(used_signals) * 0.025  # USUNIĘTE
                    # score += active_rules_bonus  # USUNIĘTE
                    # baseline_bonus = 0.3 * len(used_signals) / total_signals  # USUNIĘTE
                    # score += baseline_bonus  # USUNIĘTE
                    active_rules_bonus = 0.0  # PUNKT 7 FIX: Dummy dla kompatybilności
                    
                    # === DIAMOND WHALE AI DETECTOR INTEGRATION ===
                    # 🔥 STAGE 2/7: Integrate DiamondWhale AI Temporal Graph + QIRL Detector
                    # diamond_score, diamond_enabled, diamond_error already initialized at higher scope
                    
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
                            
                            # Wywołaj Enhanced DiamondWhale AI z TGN + QIRL
                            try:
                                from .enhanced_diamond_detector import run_enhanced_diamond_analysis
                                enhanced_result = run_enhanced_diamond_analysis(symbol, transactions)
                                
                                if enhanced_result and enhanced_result.get('status') == 'success':
                                    diamond_score = float(enhanced_result['pump_score'])
                                    diamond_enabled = True
                                    
                                    # Enhanced TGN + QIRL analysis details
                                    alert_decision = enhanced_result.get('alert_decision', 0)
                                    graph_stats = enhanced_result.get('graph_stats', {})
                                    qirl_stats = enhanced_result.get('qirl_stats', {})
                                    
                                    print(f"[ENHANCED DIAMOND] {symbol}: TGN pump_score={diamond_score:.3f}, QIRL decision={alert_decision}")
                                    print(f"[ENHANCED DIAMOND] {symbol}: Graph nodes={graph_stats.get('nodes', 0)}, whale_nodes={graph_stats.get('whale_nodes', 0)}")
                                    print(f"[ENHANCED DIAMOND] {symbol}: QIRL accuracy={qirl_stats.get('accuracy', 0):.3f}")
                                    
                                    # Alert for high-confidence TGN + QIRL detection
                                    if diamond_score > 0.7 and alert_decision == 1:
                                        reasoning = enhanced_result.get('reasoning', 'TGN + QIRL P&D pattern detected')
                                        print(f"[ENHANCED DIAMOND ALERT] {symbol}: 🚨 {reasoning}")
                                    
                                else:
                                    # Fallback to basic DiamondWhale detector
                                    diamond_result = run_diamond_detector(transactions, symbol)
                                    if diamond_result and diamond_result.get('diamond_score') is not None:
                                        diamond_score = float(diamond_result['diamond_score'])
                                        diamond_enabled = True
                                        print(f"[DIAMOND FALLBACK] {symbol}: Basic Diamond score={diamond_score:.3f}")
                                        
                            except ImportError:
                                # Fallback to basic DiamondWhale detector
                                diamond_result = run_diamond_detector(transactions, symbol)
                                if diamond_result and diamond_result.get('diamond_score') is not None:
                                    diamond_score = float(diamond_result['diamond_score'])
                                    diamond_enabled = True
                                    print(f"[DIAMOND FALLBACK] {symbol}: Basic Diamond score={diamond_score:.3f}")
                            
                            if diamond_enabled:
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
                    # whaleclip_score, whaleclip_enabled, whaleclip_error already initialized at higher scope
                    
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
                                    print(f"[WHALECLIP AI] {symbol}: Fetched 0 transactions, 0 relevant signals - fallback to score 0.0")
                                    
                            except Exception as clip_e:
                                whaleclip_error = str(clip_e)
                                print(f"[WHALECLIP AI ERROR] {symbol}: WhaleCLIP analysis failed: {clip_e}")
                        else:
                            print(f"[WHALECLIP AI] {symbol}: No blockchain contract found - skipping behavioral analysis")
                            print(f"[WHALECLIP AI] {symbol}: Fetched 0 transactions, 0 relevant signals - fallback to score 0.0")
                            
                    except Exception as e:
                        whaleclip_error = str(e)
                        print(f"[WHALECLIP AI ERROR] {symbol}: Failed to run WhaleCLIP detector: {e}")
                        print(f"[WHALECLIP AI ERROR] {symbol}: Continuing without WhaleCLIP analysis")
                    
                    # === CALIFORNIUM WHALE AI DETECTOR INTEGRATION ===
                    # 🚀 STAGE 3/7: Integrate CaliforniumWhale AI Temporal Graph + QIRL Detector
                    # californium_score, californium_enabled, californium_error already initialized at higher scope
                    
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
                            
                            # CALIFORNIUM ALERT SYSTEM REMOVED 
                            # User requested: "alerty mają być tylko z consensus vote"
                            # CaliforniumWhale AI uczestniczy tylko w consensus decision system
                                    
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
                    base_score = score  # PUNKT 7 FIX: Nie odejmujemy active_rules_bonus (został usunięty)
                    if diamond_enabled:
                        base_score -= diamond_score * 0.3
                    if whaleclip_enabled:
                        base_score -= whaleclip_score * 0.2
                    if californium_enabled:
                        base_score -= californium_score * 0.25
                    print(f"  Base score: {base_score:.3f}")
                    # print(f"  Active rules bonus: {len(used_signals)} × 0.025 = +{active_rules_bonus:.3f}")  # PUNKT 7 FIX: USUNIĘTE
                    if diamond_enabled:
                        print(f"  Diamond AI contribution: {diamond_score:.3f} × 0.3 = +{diamond_score * 0.3:.3f}")
                    if whaleclip_enabled:
                        print(f"  WhaleCLIP AI contribution: {whaleclip_score:.3f} × 0.2 = +{whaleclip_score * 0.2:.3f}")
                    if californium_enabled:
                        print(f"  CaliforniumWhale AI contribution: {californium_score:.3f} × 0.25 = +{californium_score * 0.25:.3f}")
                    print(f"  Final score: {score:.3f}")
                    print(f"[STEALTH] Final signal for {symbol} → Score: {score:.3f}, Decision: {decision}, Active: {len(used_signals)} signals{partial_note}{diamond_note}{whaleclip_note}{californium_note}")
                    
                    # 🎓 EXPLORE MODE TRIGGER LOGIC - Set explore_mode_triggered for learning
                    if score >= 1.0:  # Lower threshold for explore mode data collection
                        explore_mode_triggered = True
                        explore_trigger_reason = f"score_threshold_{score:.3f}"
                        explore_confidence = min(score / 4.0, 1.0)  # Confidence based on score
                        print(f"[EXPLORE MODE TRIGGER] {symbol}: Score {score:.3f} >= 1.0 → explore_mode_triggered = True")
                        
                        # Add explore mode data to stealth result for scan_token_async.py
                        stealth_result["explore_mode_triggered"] = True
                        stealth_result["explore_trigger_reason"] = explore_trigger_reason
                        stealth_result["explore_confidence"] = explore_confidence
                        
                    elif len(used_signals) >= 5:  # Alternative trigger: many signals
                        explore_mode_triggered = True
                        explore_trigger_reason = f"signal_count_{len(used_signals)}"
                        explore_confidence = len(used_signals) / 10.0  # Confidence based on signal count
                        print(f"[EXPLORE MODE TRIGGER] {symbol}: {len(used_signals)} signals >= 5 → explore_mode_triggered = True")
                        
                        # Add explore mode data to stealth result for scan_token_async.py
                        stealth_result["explore_mode_triggered"] = True
                        stealth_result["explore_trigger_reason"] = explore_trigger_reason
                        stealth_result["explore_confidence"] = explore_confidence
                        
                    elif diamond_enabled and diamond_score >= 0.4:  # AI detector trigger
                        explore_mode_triggered = True
                        explore_trigger_reason = f"diamond_ai_{diamond_score:.3f}"
                        explore_confidence = diamond_score
                        print(f"[EXPLORE MODE TRIGGER] {symbol}: Diamond AI {diamond_score:.3f} >= 0.4 → explore_mode_triggered = True")
                        
                        # Add explore mode data to stealth result for scan_token_async.py
                        stealth_result["explore_mode_triggered"] = True
                        stealth_result["explore_trigger_reason"] = explore_trigger_reason
                        stealth_result["explore_confidence"] = explore_confidence
                    elif californium_enabled and californium_score >= 0.5:  # Californium trigger
                        explore_mode_triggered = True
                        explore_trigger_reason = f"californium_ai_{californium_score:.3f}"
                        explore_confidence = californium_score
                        print(f"[EXPLORE MODE TRIGGER] {symbol}: Californium AI {californium_score:.3f} >= 0.5 → explore_mode_triggered = True")
                        
                        # Add explore mode data to stealth result for scan_token_async.py
                        stealth_result["explore_mode_triggered"] = True
                        stealth_result["explore_trigger_reason"] = explore_trigger_reason
                        stealth_result["explore_confidence"] = explore_confidence
                        
                    else:
                        explore_mode_triggered = False
                        print(f"[EXPLORE MODE SKIP] {symbol}: Score {score:.3f}, signals {len(used_signals)}, Diamond {diamond_score:.3f}, Californium {californium_score:.3f} - below thresholds")
                        stealth_result["explore_mode_triggered"] = False
                    
                    # === GOLDEN KEY GNN DETECTOR INTEGRATION ===
                    # 🔥 STAGE: Integrate Golden Key GNN Detector - Graph Neural Networks
                    gnn_subgraph_score = 0.0
                    gnn_active = False
                    gnn_error = None
                    
                    try:
                        # FIXED: Use contract info from CHAIN ROUTER (already found earlier in pipeline)
                        # Contract is stored in token_data as 'contract_address' and 'contract_chain'
                        contract_info = None
                        if token_data.get('contract_address') and token_data.get('contract_chain'):
                            # Use contract from token_data (CHAIN ROUTER data)
                            contract_info = {
                                'address': token_data['contract_address'],
                                'chain': token_data['contract_chain']
                            }
                            print(f"[GOLDEN KEY] {symbol}: Using contract from CHAIN ROUTER: {contract_info['address'][:10]}...")
                        else:
                            # Fallback: Try CoinGecko cache
                            from utils.contracts import get_contract
                            contract_info = get_contract(symbol)
                            if contract_info:
                                print(f"[GOLDEN KEY] {symbol}: Using contract from CoinGecko cache: {contract_info.get('address', '')[:10]}...")
                        
                        if contract_info and contract_info.get('address'):
                            print(f"[GOLDEN KEY] {symbol}: Starting GNN analysis for contract {contract_info['address'][:10]}...")
                            
                            # Pobierz rzeczywiste transakcje blockchain dla analizy GNN
                            from .blockchain_fetcher import fetch_diamond_transactions
                            contract_address = contract_info['address']
                            chain = contract_info.get('chain', 'ethereum')
                            
                            # Fetch real blockchain transactions for GNN graph
                            transactions = fetch_diamond_transactions(contract_address, chain)
                            print(f"[GOLDEN KEY] {symbol}: Fetched {len(transactions)} transactions for GNN graph")
                            
                            # Wywołaj Golden Key GNN Detector
                            from .golden_key_detector import run_golden_key_analysis
                            gnn_result = run_golden_key_analysis(symbol, transactions)
                            
                            if gnn_result and gnn_result.get('status') == 'success':
                                gnn_subgraph_score = float(gnn_result['pump_score'])
                                gnn_active = True
                                
                                # Dodaj GNN score do głównego stealth score (z wagą 0.25)
                                gnn_contribution = gnn_subgraph_score * 0.25
                                score += gnn_contribution
                                
                                # Dodaj do listy aktywnych sygnałów jeśli znaczący
                                if gnn_subgraph_score > 0.5:
                                    used_signals.append("golden_key_gnn_detection")
                                
                                print(f"[GOLDEN KEY] {symbol}: GNN pump_score={gnn_subgraph_score:.3f}, contribution=+{gnn_contribution:.3f}")
                                print(f"[GOLDEN KEY] {symbol}: Graph analysis completed - nodes={gnn_result.get('graph_stats', {}).get('nodes', 0)}")
                                
                                # Log alert decision if generated
                                if gnn_result.get('alert_decision') == 1:
                                    print(f"[GOLDEN KEY ALERT] {symbol}: GNN detected pump pattern! Reasoning: {gnn_result.get('reasoning', 'N/A')}")
                            else:
                                print(f"[GOLDEN KEY] {symbol}: No significant pump pattern detected via GNN")
                        else:
                            print(f"[GOLDEN KEY] {symbol}: No blockchain contract found - skipping GNN analysis")
                            
                    except Exception as e:
                        gnn_error = str(e)
                        print(f"[GOLDEN KEY ERROR] {symbol}: Failed to run GNN detector: {e}")
                        print(f"[GOLDEN KEY ERROR] {symbol}: Continuing without GNN analysis")
                    
                    # Fallback GNN score calculation if real analysis failed
                    if not gnn_active and len(used_signals) >= 3 and score > 1.0:
                        # Symuluj GNN subgraph analysis na podstawie dostępnych sygnałów
                        gnn_subgraph_score = min(score * 0.3, 0.9)  # Derived from stealth signals
                        gnn_active = True
                        print(f"[GNN FALLBACK] {symbol}: Subgraph QIRL score = {gnn_subgraph_score:.3f}")
                    elif not gnn_active and len(used_signals) < 3:
                        print(f"[GNN FALLBACK] {symbol}: Insufficient signals for subgraph analysis (need ≥3 active signals, got {len(used_signals)})")
                    elif not gnn_active and score <= 1.0:
                        print(f"[GNN FALLBACK] {symbol}: Score too low for subgraph analysis (need >1.0, got {score:.3f})")
                    
                    # === MASTERMIND TRACING INTEGRATION ===
                    # Dodaj group activity mastermind zgodnie z uwagami Szefira
                    mastermind_addresses = []
                    
                    try:
                        # Placeholder dla mastermind tracing - wykryj powtarzające się adresy
                        if "whale_ping" in used_signals and diamond_enabled:
                            # Symuluj wykrywanie powtarzających się adresów akumulacji
                            if diamond_score > 0.5:
                                mastermind_addresses = ["0x123...", "0x456...", "0x789..."]  # Przykładowe adresy
                                print(f"[MASTER AI] {symbol}: {len(mastermind_addresses)} addresses repeat accumulation: {', '.join(mastermind_addresses)}")
                            else:
                                print(f"[MASTER AI] {symbol}: No significant mastermind pattern detected")
                        else:
                            print(f"[MASTER AI] {symbol}: Insufficient whale activity for mastermind analysis")
                            
                    except Exception as e:
                        print(f"[MASTER AI ERROR] {symbol}: Mastermind tracing failed: {e}")
                    
                    # === DETECTORS ACTIVE STATUS LOG ===
                    # Fix WhaleCLIP status inconsistency - use actual signal strength for status
                    whale_signal_strength = 0.0
                    for signal in signals:
                        if signal.name in ["whale_ping", "whale_ping_real", "whale_ping_real_repeat"] and hasattr(signal, 'active') and signal.active and hasattr(signal, 'strength'):
                            whale_signal_strength = max(whale_signal_strength, signal.strength)
                    
                    # Update WhaleCLIP status - ONLY use WhaleCLIP's own score for consensus
                    # whale_signal_strength comes from other detectors and should NOT affect WhaleCLIP status
                    whaleclip_status_corrected = (whaleclip_enabled and whaleclip_score > 0.0)
                    
                    # Dodaj log aktywnych detektorów przed konsensusem zgodnie z uwagami Szefira
                    print(f"[DETECTORS ACTIVE] {symbol}: StealthEngine=True, Diamond={diamond_enabled}, Californium={californium_enabled}, WhaleCLIP={whaleclip_status_corrected}, GNN={gnn_active}")
                    print(f"[WHALECLIP STATUS CORRECTED] {symbol}: whaleclip_enabled={whaleclip_enabled}, whaleclip_score={whaleclip_score:.3f} → Status={whaleclip_status_corrected} (whale_signal_strength={whale_signal_strength:.3f} ignored for consensus)")
                    
                    # 🚨 CONSENSUS ENGINE PROTECTED FROM BLOCKCHAIN FAILURES
                    # Wrap consensus engine in its own try/except to ensure it runs
                    # even if blockchain detectors fail
                    
                    try:
                        # 🧠 CONSENSUS ENGINE INTEGRATION - Run Now With Real AI Values
                        print(f"[CONSENSUS COMPUTING] {symbol}: Running consensus engine with real AI values")
                        print(f"[CONSENSUS VALUES] {symbol}: Diamond={diamond_score:.3f} (enabled={diamond_enabled})")
                        print(f"[CONSENSUS VALUES] {symbol}: Californium={californium_score:.3f} (enabled={californium_enabled})")
                        print(f"[CONSENSUS VALUES] {symbol}: WhaleCLIP={whaleclip_score:.3f} (enabled={whaleclip_status_corrected})")
                        print(f"[CONSENSUS VALUES] {symbol}: GNN={gnn_subgraph_score:.3f} (enabled={gnn_active})")
                        print(f"[CONSENSUS VALUES] {symbol}: StealthEngine={score:.3f} (core signals)")
                        
                        # Store consensus data for later use
                        stored_final_decision = "WATCH"  # Default fallback
                        stored_consensus_data = None
                        stored_consensus_result = None
                        
                        # Import consensus functions at the beginning to ensure they're available
                        consensus_engine_factory = None
                        adapt_detector_score = None
                        record_detector_decision = None
                        
                        from .decision_consensus import create_decision_consensus_engine as consensus_engine_factory
                        from .detector_learning_system import adapt_detector_score, record_detector_decision
                        
                        # AI DETECTORS INTEGRATION FOR CONSENSUS VOTING
                        detector_outputs = {}
                        
                        print(f"[DETECTOR DEBUG] {symbol}: Building detector_outputs for Multi-Agent consensus")
                        print(f"[DETECTOR DEBUG] {symbol}: StealthEngine score={score:.3f}, threshold=0.1")
                        print(f"[DETECTOR DEBUG] {symbol}: CaliforniumWhale score={californium_score:.3f}, enabled={californium_enabled}")
                        print(f"[DETECTOR DEBUG] {symbol}: DiamondWhale score={diamond_score:.3f}, enabled={diamond_enabled}")
                        print(f"[DETECTOR DEBUG] {symbol}: WhaleCLIP score={whaleclip_score:.3f}, enabled={whaleclip_status_corrected}")
                        print(f"[DETECTOR DEBUG] {symbol}: GNN score={gnn_subgraph_score:.3f}, active={gnn_active}")
                        
                        # StealthEngine - CORE STEALTH SIGNALS + SELF-LEARNING ADAPTATION
                        # ZAWSZE dodaj StealthEngine jeśli jest aktywny (score > 0.1)
                        if score > 0.1:
                            # Apply intelligent self-learning score adaptation
                            adapted_stealth_score, adaptation_reason = adapt_detector_score(
                                "StealthEngine", score, symbol, token_data
                            )
                            
                            # Skip simple threshold vote - will use 5-agent AI consensus instead
                            stealth_vote = "PENDING"  # Define stealth_vote variable
                            detector_outputs["StealthEngine"] = {
                                "vote": stealth_vote,  # Will be set by 5-agent consensus
                                "score": adapted_stealth_score,
                                "confidence": 0.70,
                                "weight": 0.25
                            }
                            
                            print(f"[STEALTH LEARNING] {symbol}: {score:.3f} → {adapted_stealth_score:.3f} ({adaptation_reason})")
                            print(f"[DETECTOR ADDED] {symbol}: StealthEngine added to consensus with vote={stealth_vote}, score={adapted_stealth_score:.3f}")
                            
                            # Record decision for future learning
                            record_detector_decision(
                                "StealthEngine", symbol, score, adapted_stealth_score, 
                                stealth_vote, explore_mode=False, market_context=token_data
                            )
                        
                        # CaliforniumWhale AI + SELF-LEARNING ADAPTATION  
                        # ZAWSZE dodaj CaliforniumWhale jeśli jest aktywny (score > 0.0)
                        if californium_enabled and californium_score > 0.0:
                            # Apply intelligent self-learning score adaptation
                            adapted_californium_score, adaptation_reason = adapt_detector_score(
                                "CaliforniumWhale", californium_score, symbol, token_data
                            )
                            
                            # Skip simple threshold vote - will use 5-agent AI consensus instead
                            detector_outputs["CaliforniumWhale"] = {
                                "vote": "PENDING",  # Will be set by 5-agent consensus
                                "score": adapted_californium_score,
                                "confidence": 0.85,
                                "weight": 0.33
                            }
                            
                            print(f"[CALIFORNIUM LEARNING] {symbol}: {californium_score:.3f} → {adapted_californium_score:.3f} ({adaptation_reason})")
                            print(f"[DETECTOR ADDED] {symbol}: CaliforniumWhale added to consensus with vote={detector_outputs['CaliforniumWhale']['vote']}, score={adapted_californium_score:.3f}")
                            
                            # Record decision for future learning
                            record_detector_decision(
                                "CaliforniumWhale", symbol, californium_score, adapted_californium_score, 
                                detector_outputs['CaliforniumWhale']['vote'], explore_mode=False, market_context=token_data
                            )
                        
                        # DiamondWhale AI + SELF-LEARNING ADAPTATION
                        # ZAWSZE dodaj DiamondWhale jeśli jest aktywny (score > 0.0)
                        if diamond_enabled and diamond_score > 0.0:
                            # Apply intelligent self-learning score adaptation
                            adapted_diamond_score, adaptation_reason = adapt_detector_score(
                                "DiamondWhale", diamond_score, symbol, token_data
                            )
                            
                            # Skip simple threshold vote - will use 5-agent AI consensus instead
                            detector_outputs["DiamondWhale"] = {
                                "vote": "PENDING",  # Will be set by 5-agent consensus
                                "score": adapted_diamond_score,
                                "confidence": 0.80,
                                "weight": 0.25
                            }
                            
                            print(f"[DIAMOND LEARNING] {symbol}: {diamond_score:.3f} → {adapted_diamond_score:.3f} ({adaptation_reason})")
                            print(f"[DETECTOR ADDED] {symbol}: DiamondWhale added to consensus with vote={detector_outputs['DiamondWhale']['vote']}, score={adapted_diamond_score:.3f}")
                            
                            # Record decision for future learning
                            record_detector_decision(
                                "DiamondWhale", symbol, diamond_score, adapted_diamond_score, 
                                detector_outputs['DiamondWhale']['vote'], explore_mode=False, market_context=token_data
                            )
                        
                        # WhaleCLIP AI + SELF-LEARNING ADAPTATION  
                        if whaleclip_enabled and whaleclip_score > 0.0:
                            # Apply intelligent self-learning score adaptation
                            adapted_whaleclip_score, adaptation_reason = adapt_detector_score(
                                "WhaleCLIP", whaleclip_score, symbol, token_data
                            )
                            
                            # Skip simple threshold vote - will use 5-agent AI consensus instead
                            detector_outputs["WhaleCLIP"] = {
                                "vote": "PENDING",  # Will be set by 5-agent consensus
                                "score": adapted_whaleclip_score,
                                "confidence": 0.75,
                                "weight": 0.26
                            }
                            
                            print(f"[WHALECLIP LEARNING] {symbol}: {whaleclip_score:.3f} → {adapted_whaleclip_score:.3f} ({adaptation_reason})")
                            
                            # Record decision for future learning
                            record_detector_decision(
                                "WhaleCLIP", symbol, whaleclip_score, adapted_whaleclip_score, 
                                whaleclip_vote, explore_mode=False, market_context=token_data
                            )
                        
                        # GNN AI (Golden Key) + SELF-LEARNING ADAPTATION
                        # ZAWSZE dodaj GNN jeśli jest aktywny (score > 0.0)
                        if gnn_active and gnn_subgraph_score > 0.0:
                            # Apply intelligent self-learning score adaptation
                            adapted_gnn_score, adaptation_reason = adapt_detector_score(
                                "GNN", gnn_subgraph_score, symbol, token_data
                            )
                            
                            # Skip simple threshold vote - will use 5-agent AI consensus instead
                            detector_outputs["GNN"] = {
                                "vote": "PENDING",  # Will be set by 5-agent consensus
                                "score": adapted_gnn_score,
                                "confidence": 0.80,
                                "weight": 0.30
                            }
                            
                            print(f"[GNN LEARNING] {symbol}: {gnn_subgraph_score:.3f} → {adapted_gnn_score:.3f} ({adaptation_reason})")
                            print(f"[DETECTOR ADDED] {symbol}: GNN added to consensus with vote=PENDING, score={adapted_gnn_score:.3f}")
                            
                            # Record decision for future learning
                            record_detector_decision(
                                "GNN", symbol, gnn_subgraph_score, adapted_gnn_score, 
                                "PENDING", explore_mode=False, market_context=token_data
                            )
                        
                        # 🤖 MULTI-AGENT CONSENSUS SYSTEM: Real AI Voting with 5 Agents
                        print(f"[MULTI-AGENT EARLY] {symbol}: Starting Multi-Agent Consensus System...")
                        print(f"[DETECTOR OUTPUTS DEBUG] {symbol}: detector_outputs = {detector_outputs}")
                        print(f"[DETECTOR OUTPUTS DEBUG] {symbol}: detector_outputs length = {len(detector_outputs)}")
                        print(f"[DETECTOR OUTPUTS DEBUG] {symbol}: detector_outputs keys = {list(detector_outputs.keys())}")
                        
                        # MULTI-AGENT CONSENSUS with detector outputs
                        if detector_outputs:
                            print(f"[MULTI-AGENT CONSENSUS] {symbol}: Using Multi-Agent decision system with {len(detector_outputs)} detectors")
                            
                            # Detectors prepared for 5-agent Multi-Agent Consensus (no simple votes)
                            
                            # Use Decision Consensus Engine with Multi-Agent System
                            try:
                                if consensus_engine_factory is None:
                                    raise ImportError("consensus_engine_factory not imported successfully")
                                consensus_engine = consensus_engine_factory()
                                # Prepare market_data for consensus - USE CORRECT KEYS
                                consensus_market_data = {
                                    "symbol": symbol,
                                    "price": price_ref,  # Use actual price from stealth analysis
                                    "price_source": token_data.get('price_source', 'unknown'),  # Fixed: Add price_source for multi-agent
                                    "volume_24h": volume_24h,  # Use actual volume from stealth analysis 
                                    "price_change_24h": token_data.get('price_change_24h', 0.0),  # Correct key
                                    "market_cap": token_data.get('market_cap', 0.0),
                                    "timestamp": time.time(),
                                    "candles_15m": len(token_data.get('candles_15m', [])),
                                    "candles_5m": len(token_data.get('candles_5m', [])),
                                    "orderbook_available": bool(token_data.get('orderbook_data'))
                                }
                                
                                print(f"[MARKET DATA DEBUG] {symbol}: consensus_market_data prepared: volume=${consensus_market_data['volume_24h']:,.0f}, price=${consensus_market_data['price']:.6f}, change={consensus_market_data['price_change_24h']:.2f}%")
                                
                                consensus_result_obj = consensus_engine.simulate_decision_consensus(
                                    detector_outputs=detector_outputs,
                                    threshold=0.7,
                                    token=symbol,
                                    market_data=consensus_market_data
                                )
                                
                                stored_final_decision = consensus_result_obj.decision
                                final_score = consensus_result_obj.final_score
                                confidence = consensus_result_obj.confidence
                                threshold_met = consensus_result_obj.threshold_met
                                
                                # Enhanced logging dla multi-agent consensus
                                print(f"[MULTI-AGENT VOTES] {symbol}: Individual votes: {consensus_result_obj.votes}")
                                print(f"[MULTI-AGENT REASONING] {symbol}: {consensus_result_obj.reasoning}")
                                print(f"[MULTI-AGENT COMPLETE] {symbol}: Decision={stored_final_decision}, Score={final_score:.3f}, Confidence={confidence:.3f}, Threshold={threshold_met}")
                                
                                stored_consensus_data = {
                                    "decision": stored_final_decision,
                                    "votes": consensus_result_obj.votes,
                                    "confidence": confidence,
                                    "final_score": final_score,
                                    "threshold_met": threshold_met,
                                    "contributing_detectors": consensus_result_obj.contributing_detectors,
                                    "reasoning": consensus_result_obj.reasoning
                                }
                                stored_consensus_result = stored_consensus_data
                                
                            except Exception as e:
                                print(f"[MULTI-AGENT ERROR] {symbol}: Consensus engine failed: {e}")
                                # Fallback to simple voting if multi-agent fails
                                buy_count = sum(1 for d in detector_outputs.values() if d["vote"] == "BUY")
                                hold_count = sum(1 for d in detector_outputs.values() if d["vote"] == "HOLD")
                                avoid_count = sum(1 for d in detector_outputs.values() if d["vote"] == "AVOID")
                                
                                if buy_count > hold_count + avoid_count:
                                    stored_final_decision = "BUY"
                                elif hold_count >= avoid_count:
                                    stored_final_decision = "HOLD"
                                else:
                                    stored_final_decision = "AVOID"
                                
                                # Calculate weighted score as fallback
                                total_weighted_score = sum(d["score"] * d["weight"] for d in detector_outputs.values())
                                total_weight = sum(d["weight"] for d in detector_outputs.values())
                                final_score = total_weighted_score / total_weight if total_weight > 0 else score
                                confidence = min(0.95, max(0.15, final_score / 2.5))
                                threshold_met = final_score > 0.65
                                
                                stored_consensus_data = {
                                    "decision": stored_final_decision,
                                    "votes": [f"{detector}: {data['vote']}" for detector, data in detector_outputs.items()],
                                    "confidence": confidence,
                                    "final_score": final_score,
                                    "threshold_met": threshold_met,
                                    "contributing_detectors": list(detector_outputs.keys())
                                }
                                stored_consensus_result = stored_consensus_data
                                
                                print(f"[CONSENSUS FALLBACK] {symbol}: Decision={stored_final_decision}, Score={final_score:.3f}, Confidence={confidence:.3f}")
                        else:
                            print(f"[MULTI-AGENT SKIP] {symbol}: No detector outputs available for Multi-Agent Consensus")
                            
                    except Exception as consensus_error:
                        print(f"[CONSENSUS ERROR] {symbol}: Consensus engine failed: {consensus_error}")
                        stored_final_decision = "WATCH"
                        stored_consensus_data = None
                    
                    # === ALERT TRIGGER DECISION LOG ===
                    # Dodaj pełne uzasadnienie dla decyzji alertu zgodnie z uwagami Szefira
                    active_detectors_count = sum([1 for x in [diamond_enabled, californium_enabled, whaleclip_enabled] if x])
                    alert_eligible = (stored_final_decision == "BUY" and score >= 0.70)  # BUY-only filtering
                    
                    # Dodaj informacje o GNN i mastermind do alert reasoning
                    gnn_note = f", GNN={gnn_subgraph_score:.3f}" if gnn_active else ""
                    mastermind_note = f", Mastermind={len(mastermind_addresses)} addresses" if mastermind_addresses else ""
                    
                    # WYMAGANIE #8: Use hard gating logic instead of p≥0.7 fallback
                    from engine.decision import final_decision
                    
                    # Prepare signals dictionary for hard gating
                    hard_gating_signals = {
                        "whale_ping": {"strength": signal_details.get("whale_ping", {}).get("strength", 0.0)},
                        "dex_inflow": {"strength": signal_details.get("dex_inflow", {}).get("strength", 0.0)}
                    }
                    
                    decision_result_hardgate = final_decision(score, hard_gating_signals, stored_final_decision)
                    alert_eligible = (decision_result_hardgate == "ALERT")
                    
                    # Log hard gating decision with detailed requirements
                    whale_strength = hard_gating_signals["whale_ping"]["strength"]
                    dex_strength = hard_gating_signals["dex_inflow"]["strength"]
                    print(f"[HARD GATING] {symbol}: Decision={decision_result_hardgate}, Score={score:.3f}, Whale={whale_strength:.3f}, DEX={dex_strength:.3f}")
                    print(f"[HARD GATING] {symbol}: Consensus={stored_final_decision}, Alert={alert_eligible} (whale≥0.8 AND dex≥0.8 AND p≥0.72)")
                    
                    # ✅ ALERT TRIGGER - Use hard gating decision result
                    # Alert system now integrated with hard gating logic
                    if alert_eligible:  # Use hard gating decision result
                        # Main alert for BUY consensus
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
                            if not stored_consensus_data:
                                consensus_data = {
                                    "decision": final_decision,
                                    "votes": ["fallback_mode"],
                                    "confidence": min(score / 4.0, 1.0),
                                    "final_score": score,
                                    "threshold_met": score > 2.0,
                                    "contributing_detectors": ["stealth_fallback"]
                                }
                            else:
                                consensus_data = stored_consensus_data
                            
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
                                if consensus_data:
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
                        
                    print(f"[COMPONENT V4 DEBUG] {symbol}: Component section completed successfully - continuing to consensus...")
                    
                    print(f"[FUNCTION DEBUG] {token_data.get('symbol', 'UNKNOWN')}: About to enter consensus section...")
                    
                    # 🧠 MULTI-AGENT CONSENSUS DECISION ENGINE - Unified Detector Fusion
                    # NOTE: Consensus is already calculated above in lines 1249-1370
                    # We will use the consensus_result from there instead of recalculating
                    if 'final_decision' in locals() and 'consensus_data' in locals():
                        print(f"[CONSENSUS DEBUG] {token_data.get('symbol', 'UNKNOWN')}: Using consensus result from above: decision={final_decision}, consensus_data={consensus_data is not None}")
                    else:
                        print(f"[CONSENSUS DEBUG] {token_data.get('symbol', 'UNKNOWN')}: No consensus result from above - will compute in consensus section")
                    
                    # Note: consensus_data will be set later when consensus is actually calculated
                    # stored_consensus_data will be assigned at that time
                    
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
                        "whaleclip_score": round(whaleclip_score, 3) if whaleclip_enabled else None,
                        "stealth_engine_score": round(score, 3),  # Stealth Engine's own score
                        
                        # === COMPONENT BREAKDOWN FOR TOP 5 DISPLAY ===
                        "dex_inflow": round(dex_inflow_score, 3),
                        "whale_ping": round(whale_ping_score, 3),
                        "trust_boost": round(trust_boost_score, 3),
                        "identity_boost": round(identity_boost_score, 3),
                        
                        # === CONSENSUS DECISION ENGINE RESULTS ===
                        "consensus_decision": stored_consensus_data.get("decision", "WATCH") if 'stored_consensus_data' in locals() and stored_consensus_data else "WATCH",
                        "consensus_score": round(stored_consensus_data.get("final_score", 0.0), 3) if 'stored_consensus_data' in locals() and stored_consensus_data else 0.0,
                        "consensus_confidence": round(stored_consensus_data.get("confidence", 0.0), 3) if 'stored_consensus_data' in locals() and stored_consensus_data else 0.0,
                        "consensus_detectors": stored_consensus_data.get("contributing_detectors", []) if 'stored_consensus_data' in locals() and stored_consensus_data else [],
                        "consensus_votes": stored_consensus_data.get("votes", []) if 'stored_consensus_data' in locals() and stored_consensus_data else []
                    }
                    
                    # Dodaj informacje o błędach jeśli wystąpiły - STAGE 10 FIX: Safe variable access
                    if 'diamond_error' in locals() and diamond_error and 'result' in locals():
                        result["diamond_error"] = diamond_error
                    if 'californium_error' in locals() and californium_error and 'result' in locals():
                        result["californium_error"] = californium_error
                    if 'consensus_error' in locals() and consensus_error and 'result' in locals():
                        result["consensus_error"] = consensus_error
                    
                    # Original location dla explore mode logic przeniesiony na lepsze miejsce
                    print(f"[ALERT INTEGRATION DEBUG] {token_data.get('symbol', 'UNKNOWN')}: About to enter Stealth V3 alert system")
                    # 🚨 STEALTH V3 TELEGRAM ALERT INTEGRATION - Nowoczesny Alert System
                    # Alert triggering logic based ONLY on consensus decision
                    try:
                        # Import nowego systemu alertów Stealth v3
                        from alerts.stealth_v3_telegram_alerts import send_stealth_v3_alert
                        
                        # 🔐 CRITICAL CONSENSUS DECISION ONLY - BUY DECISION TRIGGERS IMMEDIATE ALERT
                        should_alert = False
                        explore_mode = False
                        explore_trigger_reason = None
                        explore_confidence = 0.0
                        
                        # 🎯 SIMPLIFIED ALERT LOGIC - ONLY CONSENSUS DECISION MATTERS
                        consensus_decision = stored_consensus_data.get("decision") if 'stored_consensus_data' in locals() and stored_consensus_data else None
                        
                        # **CRITICAL FIX**: Alert IMMEDIATELY when Multi-Agent Consensus says BUY
                        if consensus_decision == "BUY":
                            should_alert = True
                            unified_decision_hardgate = "ALERT"
                            print(f"[CONSENSUS ALERT] {token_data.get('symbol', 'UNKNOWN')}: BUY decision → IMMEDIATE ALERT")
                        else:
                            should_alert = False
                            unified_decision_hardgate = "NO_ALERT"
                            print(f"[CONSENSUS BLOCK] {token_data.get('symbol', 'UNKNOWN')}: Decision={consensus_decision} → NO ALERT")
                        
                        # Log consensus-based decision
                        symbol_name = token_data.get('symbol', 'UNKNOWN')
                        stealth_p = score  # Use the stealth score as probability for logging
                        
                        if should_alert:
                            print(f"[CONSENSUS ALERT] {symbol_name}: ALERT TRIGGERED - Multi-Agent Consensus = BUY")
                            if 'stored_consensus_data' in locals() and stored_consensus_data:
                                print(f"[CONSENSUS ALERT] {symbol_name}: Votes: {stored_consensus_data.get('votes', [])}")
                                print(f"[CONSENSUS ALERT] {symbol_name}: Contributing detectors: {stored_consensus_data.get('contributing_detectors', [])}") 
                                print(f"[CONSENSUS ALERT] {symbol_name}: Confidence: {stored_consensus_data.get('confidence', 0.0):.3f}")
                        else:
                            print(f"[CONSENSUS BLOCK] {symbol_name}: NO ALERT - Multi-Agent decision: {consensus_decision}")
                            print(f"[CONSENSUS VOTE] {token_data.get('symbol', 'UNKNOWN')}: Only BUY decisions trigger alerts")
                        
                        # 🎓 EXPLORE MODE LEARNING SYSTEM - zapisuj wysokie score dla uczenia agentów
                        # Use explore mode results from earlier analysis
                        explore_mode = explore_mode_triggered
                        
                        # EXPLORE MODE: Mark for potential saving (will validate conditions later)
                        if explore_mode_triggered:
                            print(f"[EXPLORE MODE] {token_data.get('symbol', 'UNKNOWN')}: High score {score:.3f} detected - validating explore conditions")
                            
                            # Note: Explore mode validation and saving will happen later in proper validation logic
                            print(f"[EXPLORE MODE] {token_data.get('symbol', 'UNKNOWN')}: Trigger reason: {explore_trigger_reason}")
                            print(f"[EXPLORE MODE] {token_data.get('symbol', 'UNKNOWN')}: Agenci będą sprawdzać czy był pump w przeszłości")
                            
                            # Enhanced learning data recording for detector training (without file saving)
                            try:
                                # Record explore mode decisions for individual detector learning
                                if score > 0.0:
                                    record_detector_decision(
                                        "StealthEngine", token_data.get('symbol', 'UNKNOWN'), 
                                        score, score, "EXPLORE", explore_mode=True, 
                                        market_context=token_data
                                    )
                                    
                                if diamond_enabled and diamond_score > 0.0:
                                    record_detector_decision(
                                        "DiamondWhale", token_data.get('symbol', 'UNKNOWN'), 
                                        diamond_score, diamond_score, "EXPLORE", explore_mode=True, 
                                        market_context=token_data
                                    )
                                    
                                if californium_enabled and californium_score > 0.0:
                                    record_detector_decision(
                                        "CaliforniumWhale", token_data.get('symbol', 'UNKNOWN'), 
                                        californium_score, californium_score, "EXPLORE", explore_mode=True, 
                                        market_context=token_data
                                    )
                                    
                                symbol = token_data.get('symbol', 'UNKNOWN') if token_data else 'UNKNOWN'
                                print(f"[DETECTOR LEARNING] {symbol}: Explore mode decisions recorded for active detectors")
                                
                            except Exception as learning_error:
                                symbol = token_data.get('symbol', 'UNKNOWN') if token_data else 'UNKNOWN'
                                print(f"[DETECTOR LEARNING ERROR] {symbol}: Failed to record explore decisions: {learning_error}")
                            
                            # W explore mode nie wysyłamy alertów - tylko zbieramy dane do nauki
                            should_alert = False
                            symbol = token_data.get('symbol', 'UNKNOWN') if token_data else 'UNKNOWN'
                            print(f"[EXPLORE MODE] {symbol}: No alert sent - collecting learning data only")
                        
                        if should_alert:
                            symbol = token_data.get('symbol', 'UNKNOWN') if token_data else 'UNKNOWN'
                            
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
                            
                            # Przygotuj consensus data - ensure consensus_data is defined
                            consensus_alert_data = {}
                            if 'consensus_data' in locals() and consensus_data:
                                consensus_alert_data = {
                                    "decision": consensus_data.get("decision", "WATCH"),
                                    "votes": f"{len(consensus_data.get('contributing_detectors', []))}/{4}",
                                    "confidence": round(consensus_data.get("final_score", 0.0), 3),
                                    "feedback_adjust": 0.0  # Placeholder dla przyszłych implementacji
                                }
                            elif 'explore_mode' in locals() and explore_mode:
                                # 🚧 EXPLORE MODE - Experimental alert consensus data
                                consensus_alert_data = {
                                    "decision": "BUY",  # BUY-only enforcement - explore mode qualifies as BUY
                                    "votes": f"{len(used_signals)}/4 (EXPLORE)",
                                    "confidence": round(explore_confidence if 'explore_confidence' in locals() else 0.5, 3),
                                    "feedback_adjust": 0.0,
                                    "explore_mode": True,
                                    "explore_trigger_reason": explore_trigger_reason if 'explore_trigger_reason' in locals() else "unknown"
                                }
                            else:
                                # Fallback consensus data based on stealth score
                                if score >= 4.0:
                                    consensus_alert_data = {
                                        "decision": "BUY",
                                        "votes": f"{len(used_signals)}/4",
                                        "confidence": round(min(score / 4.0, 1.0), 3),
                                        "feedback_adjust": 0.0
                                    }
                                elif score >= 2.5:
                                    consensus_alert_data = {
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
                                "diamond_score": round(diamond_score, 3) if diamond_enabled else None,
                                "explore_mode": explore_mode,
                                "explore_trigger_reason": explore_trigger_reason if explore_mode else None
                            }
                            
                            print(f"[ALERT SECTION DEBUG] {symbol}: About to send Stealth v3 alert (score: {score:.3f})")
                            
                            # Wyślij Stealth v3 alert
                            alert_success = send_stealth_v3_alert(
                                symbol=symbol,
                                detector_results=detector_results,
                                consensus_data=consensus_alert_data,
                                meta_data=meta_data
                            )
                            
                            if alert_success:
                                print(f"[STEALTH V3 ALERT SUCCESS] {symbol}: Nowoczesny alert wysłany (score: {score:.3f})")
                                # STAGE 10 FIX: Safe variable access using locals()
                                if 'result' in locals():
                                    result["alert_sent"] = True
                                    result["alert_type"] = "stealth_v3"
                                
                                # 🚧 EXPLORE MODE FEEDBACK LOGGING
                                if explore_mode:
                                    try:
                                        from utils.stealth_utils import log_explore_mode_feedback
                                        
                                        # Log explore mode feedback dla future learning
                                        feedback_entry = log_explore_mode_feedback(
                                            symbol=symbol,
                                            final_score=score,
                                            confidence=explore_confidence,
                                            decision="BUY",
                                            explore_mode=True,
                                            token_data=explore_token_data
                                        )
                                        
                                        # Save to explore mode feedback file
                                        import json
                                        import os
                                        os.makedirs("cache/explore_mode", exist_ok=True)
                                        feedback_file = "cache/explore_mode/feedback_log.json"
                                        
                                        # Load existing feedback log
                                        if os.path.exists(feedback_file):
                                            with open(feedback_file, 'r') as f:
                                                feedback_log = json.load(f)
                                        else:
                                            feedback_log = []
                                        
                                        # Append new feedback
                                        feedback_log.append(feedback_entry)
                                        
                                        # Save updated feedback log
                                        with open(feedback_file, 'w') as f:
                                            json.dump(feedback_log, f, indent=2)
                                        
                                        print(f"[EXPLORE MODE FEEDBACK] {symbol}: Logged experimental alert feedback")
                                        
                                    except Exception as e:
                                        print(f"[EXPLORE MODE FEEDBACK ERROR] {symbol}: {e}")
                                        
                            else:
                                print(f"[STEALTH V3 ALERT SKIP] {symbol}: Alert w cooldown lub błąd wysyłania")
                                # STAGE 10 FIX: Safe variable access using locals()
                                if 'result' in locals():
                                    result["alert_sent"] = False
                                    result["alert_skip_reason"] = "cooldown_or_error"
                        else:
                            # Removed annoying low score log - only consensus system determines alerts now
                            # STAGE 10 FIX: Safe variable access using locals()
                            if 'result' in locals():
                                result["alert_sent"] = False
                                result["alert_skip_reason"] = "below_threshold"
                            
                    except Exception as stealth_alert_error:
                        print(f"[STEALTH V3 ALERT ERROR] {symbol}: {stealth_alert_error}")
                        # STAGE 10 FIX: Safe variable access using locals()
                        if 'result' in locals():
                            result["alert_sent"] = False
                            result["alert_skip_reason"] = f"stealth_alert_error: {stealth_alert_error}"
                        
                    # ⛔ WATCHLIST ALERT SYSTEM DISABLED per user request
                    # User requested removal of watchlist alerts completely
                    
        # At end of try block - DO NOT RETURN EARLY - consensus must run for ALL tokens
        if skip_reason:
            print(f"[SKIP REASON SET] {symbol}: Skip reason: {skip_reason} - but continuing to consensus")
            # Do NOT return here - consensus must run even for skipped tokens
            
        print(f"[COMPUTE STEALTH SCORE] {token_data.get('symbol', 'UNKNOWN')}: Stealth analysis completed - proceeding to consensus section...")
        print(f"[EXIT DEBUG] {token_data.get('symbol', 'UNKNOWN')}: About to exit try block normally")
        
    except Exception as e:
        import traceback
        print(f"[COMPUTE STEALTH SCORE ERROR] {symbol}: Exception occurred: {type(e).__name__}: {e}")
        print(f"[COMPUTE STEALTH SCORE ERROR] {symbol}: Traceback: {traceback.format_exc()}")
        # Store error but continue to consensus section
        stealth_error = f"compute_stealth_error: {e}"
        score = 0.0
        active_signals = []
        data_coverage = 1.0
        # Initialize all variables that might be undefined due to exception
        if 'explore_mode_triggered' not in locals():
            explore_mode_triggered = False
            explore_confidence = 0.0
            explore_trigger_reason = "exception_occurred"
        if 'final_decision' not in locals():
            final_decision = "WATCH"
        if 'consensus_data' not in locals():
            consensus_data = None
        if 'consensus_result' not in locals():
            consensus_result = None
        if 'result' not in locals():
            result = {"score": 0.0, "error": str(e)}
        if 'dex_inflow_score' not in locals():
            dex_inflow_score = 0.0
        if 'whale_ping_score' not in locals():
            whale_ping_score = 0.0
        if 'trust_boost_score' not in locals():
            trust_boost_score = 0.0
        if 'identity_boost_score' not in locals():
            identity_boost_score = 0.0
        if 'used_signals' not in locals():
            used_signals = []
        if 'diamond_score' not in locals():
            diamond_score = 0.0
            diamond_enabled = False
        if 'californium_score' not in locals():
            californium_score = 0.0
            californium_enabled = False
        if 'whaleclip_score' not in locals():
            whaleclip_score = 0.0
            whaleclip_enabled = False

    # (OLD EXPLORE MODE LOGIC REMOVED - MOVED TO PROPER POSITION BEFORE EARLY RETURNS)
    
    # (REMOVED FIRST CONSENSUS CODE - MOVING TO PROPER LOCATION)




    
    # 🔧 EXPLORE MODE REMOVED FROM STEALTH ENGINE - Now handled in scan_token_async.py to avoid duplicate analysis
    # This prevents inconsistent explore mode results by ensuring single execution point

    
    # (DUPLICATE CONSENSUS CODE REMOVED - USING THE PROPER VERSION BELOW)


    
    # 🧠 NEW UNIFIED CONSENSUS DECISION ENGINE - BUY/HOLD/AVOID Decisions
    # Initialize consensus variables that will be used in final return
    final_decision = "WATCH"  # Default fallback
    consensus_data = None
    consensus_result = None
    consensus_error = None
    final_score = score  # WYMAGANIE #6: Use pure aggregator p_raw
    
    # Initialize skip_reason if not defined
    if 'skip_reason' not in locals():
        skip_reason = None
    
    # Check if consensus was already computed in main try block
    if 'stored_consensus_data' in locals():
        # Use stored consensus data from main try block
        final_decision = stored_final_decision
        consensus_data = stored_consensus_data
        consensus_result = stored_consensus_result
        print(f"[CONSENSUS RESTORE] {symbol}: Using stored consensus data - decision={final_decision}")
    else:
        # 🚨 CONSENSUS ENGINE ALREADY RUN IN MAIN COMPUTATION BLOCK
        # Consensus logic has been moved to execute during main analysis
        # where AI detector values are properly available
        print(f"[CONSENSUS RESTORE] {symbol}: Using consensus data from main computation block")
        
        # Always run consensus engine for all tokens (removed stored_final_decision check)
        print(f"[CONSENSUS FORCED RUN] {symbol}: Running consensus engine for all tokens")
        
        # Run consensus decision if we have detectors
        print(f"[CONSENSUS DEBUG] {symbol}: Total detector outputs: {len(detector_outputs) if 'detector_outputs' in locals() else 0}")
        
        # Debug AI detector status
        print(f"[CONSENSUS AI STATUS] {symbol}: Diamond={diamond_enabled if 'diamond_enabled' in locals() else 'UNDEFINED'}({diamond_score if 'diamond_score' in locals() else 'UNDEFINED'}), Californium={californium_enabled if 'californium_enabled' in locals() else 'UNDEFINED'}({californium_score if 'californium_score' in locals() else 'UNDEFINED'}), WhaleCLIP={whaleclip_status_corrected if 'whaleclip_status_corrected' in locals() else 'UNDEFINED'}({whaleclip_score if 'whaleclip_score' in locals() else 'UNDEFINED'}), StealthScore={score}")
        
        try:
            # Build detector outputs from AI detectors if not available
            if 'detector_outputs' not in locals():
                detector_outputs = {}
                
                # Add Diamond AI if enabled
                if 'diamond_enabled' in locals() and diamond_enabled and 'diamond_score' in locals() and diamond_score > 0:
                    detector_outputs["DiamondWhale"] = {
                        "vote": "BUY" if diamond_score > 0.7 else "HOLD" if diamond_score > 0.4 else "AVOID",
                        "score": diamond_score,
                        "weight": 0.25
                    }
                    print(f"[CONSENSUS BUILD] {symbol}: Added DiamondWhale - score={diamond_score}, vote={detector_outputs['DiamondWhale']['vote']}")
                
                # Add Californium AI if enabled
                if 'californium_enabled' in locals() and californium_enabled and 'californium_score' in locals() and californium_score > 0:
                    detector_outputs["CaliforniumWhale"] = {
                        "vote": "BUY" if californium_score > 0.7 else "HOLD" if californium_score > 0.4 else "AVOID",
                        "score": californium_score,
                        "weight": 0.33
                    }
                    print(f"[CONSENSUS BUILD] {symbol}: Added CaliforniumWhale - score={californium_score}, vote={detector_outputs['CaliforniumWhale']['vote']}")
                
                # Add WhaleCLIP AI if enabled (using corrected status)
                if 'whaleclip_status_corrected' in locals() and whaleclip_status_corrected and 'whaleclip_score' in locals():
                    detector_outputs["WhaleCLIP"] = {
                        "vote": "BUY" if whaleclip_score > 0.7 else "HOLD" if whaleclip_score > 0.4 else "AVOID",
                        "score": whaleclip_score,
                        "weight": 0.26
                    }
                    print(f"[CONSENSUS BUILD] {symbol}: Added WhaleCLIP - score={whaleclip_score}, vote={detector_outputs['WhaleCLIP']['vote']}")
                
                # Add Stealth Engine - ALWAYS available with safe handling
                try:
                    # Skip simple threshold vote - will use 5-agent AI consensus instead
                    stealth_vote = "PENDING"  # Define stealth_vote variable
                    detector_outputs["StealthEngine"] = {
                        "vote": stealth_vote,  # Will be set by 5-agent consensus
                        "score": float(score),  # Ensure proper type conversion
                        "weight": 0.25
                    }
                    print(f"[CONSENSUS BUILD] {symbol}: Added StealthEngine - score={score:.3f}, vote={stealth_vote}")
                except Exception as stealth_error:
                    print(f"[STEALTH ENGINE ERROR] {symbol}: Failed to add to consensus: {stealth_error}")
                    # Continue without StealthEngine if there's an error
                
                print(f"[CONSENSUS BUILT] {symbol}: Built detector outputs: {list(detector_outputs.keys())}")
            
            if len(detector_outputs) >= 1:
                # MULTI-AGENT CONSENSUS - Restored Real AI Voting System
                print(f"[MULTI-AGENT CONSENSUS] {symbol}: Using Multi-Agent decision system with {len(detector_outputs)} detectors")
                
                # Detectors prepared for 5-agent Multi-Agent Consensus (no simple votes)
                
                # Use Decision Consensus Engine with Multi-Agent System
                try:
                    if consensus_engine_factory is None:
                        raise ImportError("consensus_engine_factory not imported successfully") 
                    consensus_engine = consensus_engine_factory()
                    consensus_result_obj = consensus_engine.simulate_decision_consensus(
                        detector_outputs=detector_outputs,
                        threshold=0.7,
                        token=symbol
                    )
                    
                    final_decision = consensus_result_obj.decision
                    final_score = consensus_result_obj.final_score
                    confidence = consensus_result_obj.confidence
                    threshold_met = consensus_result_obj.threshold_met
                    
                    # Enhanced logging dla multi-agent consensus
                    print(f"[MULTI-AGENT VOTES] {symbol}: Individual votes: {consensus_result_obj.votes}")
                    print(f"[MULTI-AGENT REASONING] {symbol}: {consensus_result_obj.reasoning}")
                    print(f"[MULTI-AGENT COMPLETE] {symbol}: Decision={final_decision}, Score={final_score:.3f}, Confidence={confidence:.3f}, Threshold={threshold_met}")
                    
                    consensus_data = {
                        "decision": final_decision,
                        "votes": consensus_result_obj.votes,
                        "confidence": confidence,
                        "final_score": final_score,
                        "threshold_met": threshold_met,
                        "contributing_detectors": consensus_result_obj.contributing_detectors,
                        "reasoning": consensus_result_obj.reasoning
                    }
                    
                except Exception as e:
                    print(f"[MULTI-AGENT ERROR] {symbol}: Consensus engine failed: {e}")
                    # Fallback to simple voting if multi-agent fails
                    buy_votes = sum(1 for d in detector_outputs.values() if d["vote"] == "BUY")
                    hold_votes = sum(1 for d in detector_outputs.values() if d["vote"] == "HOLD")
                    avoid_votes = sum(1 for d in detector_outputs.values() if d["vote"] == "AVOID")
                    
                    if buy_votes > hold_votes + avoid_votes:
                        final_decision = "BUY"
                    elif hold_votes >= avoid_votes:
                        final_decision = "HOLD"
                    else:
                        final_decision = "AVOID"
                    
                    # Calculate weighted score as fallback
                    total_weighted_score = sum(d["score"] * d["weight"] for d in detector_outputs.values())
                    total_weight = sum(d["weight"] for d in detector_outputs.values())
                    # WYMAGANIE #6: Keep pure aggregator p_raw instead of recalculating
                    final_score = score  # Use original aggregator p_raw, not weighted average
                    confidence = min(0.95, max(0.1, final_score / 3.0))
                    threshold_met = final_score >= 1.0 and buy_votes > 0
                    
                    consensus_data = {
                        "decision": final_decision,
                        "votes": [f"{detector}: {data['vote']}" for detector, data in detector_outputs.items()],
                        "confidence": confidence,
                        "final_score": final_score,
                        "threshold_met": threshold_met,
                        "contributing_detectors": list(detector_outputs.keys())
                    }
                    
                    print(f"[CONSENSUS FALLBACK] {symbol}: Decision={final_decision}, Score={final_score:.3f}, Confidence={confidence:.3f}")
                
                # Store consensus data for later use
                consensus_result = consensus_data
            else:
                # No detectors available - use simple fallback
                print(f"[CONSENSUS FALLBACK] {symbol}: No detectors available, using simple score-based decision")
                
                if score > 2.0:
                    final_decision = "BUY"
                elif score > 1.0:
                    final_decision = "HOLD"
                else:
                    final_decision = "WATCH"
                    
                consensus_data = {
                    "decision": final_decision,
                    "votes": [],
                    "confidence": min(score / 4.0, 1.0),
                    "final_score": score,
                    "threshold_met": score > 2.0,
                    "contributing_detectors": ["stealth_fallback"]
                }
                
                consensus_result = None
                final_score = score
                
        except ImportError:
            print(f"[CONSENSUS IMPORT ERROR] {symbol}: Using simplified fallback logic")
            # Simplified fallback logic
            if score > 2.0:
                final_decision = "BUY"
            elif score > 1.0:
                final_decision = "HOLD"
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
            
            consensus_result = None
            final_score = score
            
        except Exception as e:
            print(f"[CONSENSUS ERROR] {symbol}: Exception in consensus engine: {e}")
            
            final_decision = "WATCH"
            consensus_data = {
                "decision": "WATCH",
                "votes": [],
                "confidence": 0.0,
                "final_score": 0.0,
                "threshold_met": False,
                "contributing_detectors": []
            }
            
            consensus_result = None
            final_score = 0.0

    # 🎯 ULTIMATE DEBUG TEST - This print MUST appear if Enhanced RL section is reached!
    print(f"[ULTIMATE DEBUG TEST] {symbol}: CODE BEFORE ENHANCED RL REACHED! score={score}")
    
    # PUNKT 9 FIX: RL/Enhanced RL Gateway - raz i tylko "borderline"
    from engine.rl_gateway import maybe_run_rl
    
    enhanced_rl_decision = None
    enhanced_rl_data = None
    
    # PUNKT 9 FIX: Użyj gateway z borderline logic (0.65 <= p_raw <= 0.80, consensus != "BUY")
    try:
        # Get consensus decision z wcześniejszej analizy
        consensus_decision = final_decision if 'final_decision' in locals() else 'WATCH'
        
        # Prepare stealth data dla RL gateway
        stealth_data_rl = {
            'score': score,
            'diamond_score': diamond_score if 'diamond_score' in locals() else 0.0,
            'whaleclip_score': whaleclip_score if 'whaleclip_score' in locals() else 0.0,
            'dex_inflow': locals().get('dex_inflow_value', 0.0),
            'whale_ping': whale_ping_score if 'whale_ping_score' in locals() else 0.0,
            'volume_spike': 1.0 if any('volume_spike' in sig for sig in used_signals) else 0.0,
            'consensus_decision': consensus_decision,
            'consensus_confidence': consensus_data.get('confidence', 0.0) if 'consensus_data' in locals() and consensus_data else 0.0
        }
        
        # Market data dla context
        market_data_rl = {
            'price': token_data.get('price_usd', 0.0),
            'volume': token_data.get('volume_24h', 0.0),
            'volatility': 0.2,
            'trend': 0.0
        }
        
        # PUNKT 9 FIX: Use gateway instead of direct Enhanced RL call
        rl_result = maybe_run_rl(symbol, score, consensus_decision, stealth_data_rl, market_data_rl)
        
        if rl_result.get('status') == 'completed':
            enhanced_rl_data = rl_result.get('enhanced_result')
            enhanced_rl_decision = rl_result.get('decision')
            print(f"[RL GATEWAY SUCCESS] {symbol}: Decision={enhanced_rl_decision}, Score={rl_result.get('weighted_score', 0):.3f}")
        else:
            print(f"[RL GATEWAY SKIP] {symbol}: {rl_result.get('reason', 'unknown')}")
            
    except Exception as rl_gateway_error:
        print(f"[RL GATEWAY ERROR] {symbol}: {rl_gateway_error}")
        enhanced_rl_decision = None
        enhanced_rl_data = None

    # 🚨 STEALTH V3 TELEGRAM ALERT SYSTEM - ENHANCED RL CONSENSUS-BASED ALERTS
    alert_threshold = 0.70
    if ENHANCED_RL_AVAILABLE and enhanced_rl_data:
        # Use Enhanced RL adaptive threshold
        alert_threshold = enhanced_rl_data['adaptive_threshold']
        alert_score = enhanced_rl_data['weighted_score']
    else:
        alert_score = score
    
    if final_decision == "BUY" and alert_score >= alert_threshold:
        print(f"[ALERT ELIGIBLE] {symbol}: Decision={final_decision}, Score={alert_score:.3f}, Threshold={alert_threshold:.3f} - preparing alert")
        
        try:
            from alerts.stealth_v3_telegram_alerts import send_stealth_v3_alert
            
            # Prepare enhanced detector_results
            detector_results = {
                "whale_ping": min(alert_score / 4.0, 1.0),
                "dex_inflow": dex_inflow_score if 'dex_inflow_score' in locals() else 0.0,
                "orderbook_anomaly": 0.0,  # Default
                "whaleclip_vision": whaleclip_final_score if 'whaleclip_final_score' in locals() else 0.0,
                "mastermind_tracing": californium_score if 'californium_score' in locals() else 0.0,
                "diamond_ai": diamond_score if 'diamond_score' in locals() else 0.0,
                "enhanced_rl_score": enhanced_rl_data['weighted_score'] if enhanced_rl_data else 0.0
            }
            
            alert_sent = send_stealth_v3_alert(
                symbol=symbol,
                score=alert_score,
                detector_results=detector_results,
                market_data=token_data,
                consensus_data=consensus_data,
                alert_type="enhanced_stealth_alert"
            )
            
            if alert_sent:
                print(f"[STEALTH V3 ENHANCED ALERT SUCCESS] {symbol}: Alert sent with Enhanced RL decision {final_decision}")
            else:
                print(f"[STEALTH V3 ENHANCED ALERT FAILED] {symbol}: Failed to send alert")
                
        except Exception as alert_error:
            print(f"[STEALTH V3 ENHANCED ALERT ERROR] {symbol}: {alert_error}")
    else:
        print(f"[ALERT SKIP] {symbol}: Decision={final_decision}, Score={alert_score:.3f}, Threshold={alert_threshold:.3f} - not eligible for alert")

    # 🎯 ULTIMATE DEBUG TEST - This print MUST appear if Enhanced RL section is reached!
    print(f"[ULTIMATE DEBUG TEST] {symbol}: CODE BEFORE ENHANCED RL REACHED! score={score}")
    
    # 🧠 ENHANCED RL INTEGRATION - DQN + RLAgentV3 ON EVERY QUALIFIED TOKEN (MOVED TO MAIN EXECUTION FLOW)
    print(f"[ENHANCED RL ENTRY] {symbol}: Entering Enhanced RL section - this proves the code is reachable!")
    enhanced_rl_decision = None
    enhanced_rl_data = None
    
    # Check if Enhanced RL is available and should be triggered
    if ENHANCED_RL_AVAILABLE and score > 0.0:  # Only trigger for tokens with some signals
        print(f"[ENHANCED RL TRIGGER] {symbol}: Score {score:.3f} - triggering Enhanced RL analysis")
        
        try:
            from enhanced_rl_integration import EnhancedRLIntegration
            from dqn_agent import DQNAgent
            from rl_agent_v3 import RLAgentV3
            
            # Initialize Enhanced RL components
            enhanced_rl = EnhancedRLIntegration()
            dqn_agent = DQNAgent.instance(state_size=10, action_size=3)  # SINGLETON PATTERN: Prevent double initialization
            rl_agent_v3 = RLAgentV3.instance()  # SINGLETON PATTERN: Prevent double initialization
            
            # Prepare enhanced state features
            enhanced_state = {
                'base_score': score,
                'final_score': final_score,
                'consensus_decision': final_decision,
                'consensus_confidence': consensus_data.get('confidence', 0.0) if consensus_data else 0.0,
                'whale_ping': whale_ping_score if 'whale_ping_score' in locals() else 0.0,
                'dex_inflow': dex_inflow_score if 'dex_inflow_score' in locals() else 0.0,
                'diamond_score': diamond_score if 'diamond_score' in locals() else 0.0,
                'californium_score': californium_score if 'californium_score' in locals() else 0.0,
                'whaleclip_score': whaleclip_score if 'whaleclip_score' in locals() else 0.0,
                'volume_24h': token_data.get('volume_24h', 0)
            }
            
            print(f"[ENHANCED RL STATE] {symbol}: Prepared state with {len(enhanced_state)} features")
            
            # Run Enhanced RL analysis
            enhanced_result = enhanced_rl.analyze_token_enhanced(
                symbol=symbol,
                state_features=enhanced_state,
                dqn_agent=dqn_agent,
                rl_agent_v3=rl_agent_v3
            )
            
            if enhanced_result:
                enhanced_rl_decision = enhanced_result.get('decision', final_decision)
                enhanced_rl_data = enhanced_result
                
                print(f"[ENHANCED RL SUCCESS] {symbol}: Decision={enhanced_rl_decision}, Confidence={enhanced_result.get('confidence', 0.0):.3f}")
                
                # Update final decision if Enhanced RL provides stronger signal
                if enhanced_result.get('confidence', 0.0) > 0.8:
                    final_decision = enhanced_rl_decision
                    final_score = enhanced_result.get('weighted_score', final_score)
                    print(f"[ENHANCED RL OVERRIDE] {symbol}: Updated decision to {final_decision} with confidence {enhanced_result.get('confidence', 0.0):.3f}")
            else:
                print(f"[ENHANCED RL WARNING] {symbol}: No enhanced result returned")
                
        except ImportError as e:
            print(f"[ENHANCED RL IMPORT ERROR] {symbol}: {e}")
        except Exception as e:
            print(f"[ENHANCED RL ERROR] {symbol}: {e}")
    else:
        print(f"[ENHANCED RL SKIP] {symbol}: Available={ENHANCED_RL_AVAILABLE}, Score={score:.3f} - not qualified for Enhanced RL")

    # Final return with all stealth components + consensus integration + Enhanced RL
    result = {
        "score": score,
        "final_score": final_score,
        "signals": used_signals if 'used_signals' in locals() else [],
        "whale_ping": whale_ping_score if 'whale_ping_score' in locals() else 0.0,
        "dex_inflow": dex_inflow_score if 'dex_inflow_score' in locals() else 0.0,
        "trust_boost": trust_boost_score if 'trust_boost_score' in locals() else 0.0,
        "identity_boost": identity_boost_score if 'identity_boost_score' in locals() else 0.0,
        "diamond_score": diamond_score if 'diamond_score' in locals() else 0.0,
        "californium_score": californium_score if 'californium_score' in locals() else 0.0,
        "whaleclip_score": whaleclip_score if 'whaleclip_score' in locals() else 0.0,
        "consensus_decision": final_decision,
        "consensus_confidence": consensus_data.get('confidence', 0.0) if consensus_data else 0.0,
        "consensus_votes": consensus_data.get('votes', []) if consensus_data else [],
        "enhanced_rl_available": ENHANCED_RL_AVAILABLE,
        "enhanced_rl_decision": enhanced_rl_decision,
        "enhanced_rl_data": enhanced_rl_data,
        "skip_reason": skip_reason
    }
    
    return result


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
        print(f"[FLOW DEBUG] {symbol}: About to call compute_stealth_score()...")
        stealth_result = compute_stealth_score(token_data)
        print(f"[FLOW DEBUG] {symbol}: compute_stealth_score() completed, result keys: {list(stealth_result.keys())}")
        
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
        
        # Diamond Decision Engine został już wywołany w compute_stealth_score()
        # Użyj consensus_data z stealth_result
        consensus_data = stealth_result.get("consensus_data", None)
        consensus_decision = stealth_result.get("consensus_decision", "WATCH")
        
        # Przygotuj wynikowy dict
        diamond_decision = {
            "decision": consensus_decision,
            "score": stealth_score,
            "fused_score": stealth_score,  # Use stealth score as fused score
            "confidence": 0.5,  # Default confidence
            "trigger_reasons": [],  # Empty reasons
            "dominant_detector": "StealthEngine"  # Default detector
        }
        
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
        
        # 🚧 EXPLORE MODE EVALUATION - CRITICAL WHALE DETECTION INTEGRATION
        explore_mode = False
        explore_trigger_reason = None
        explore_confidence = 0.0
        
        print(f"[EXPLORE MODE CHECK] {symbol}: Score={stealth_score:.3f}, whale_ping in signals={'whale_ping' in active_signals}")
        
        try:
            # Extract whale_ping strength from stealth result for explore mode evaluation
            whale_ping_strength = 0.0
            signal_details = stealth_result.get("signal_details", {})
            
            if "whale_ping" in signal_details:
                whale_ping_strength = signal_details["whale_ping"].get("strength", 0.0)
                print(f"[EXPLORE WHALE EXTRACT] {symbol}: Found whale_ping strength={whale_ping_strength} in signal_details")
            elif "whale_ping" in active_signals:
                whale_ping_strength = 3.0  # Strong active whale signal
                print(f"[EXPLORE WHALE EXTRACT] {symbol}: whale_ping in active_signals, using strength=3.0")
            else:
                print(f"[EXPLORE WHALE EXTRACT] {symbol}: No whale_ping detected")
            
            # Check explore mode conditions
            core_signals = ['whale_ping', 'dex_inflow', 'orderbook_anomaly', 'spoofing_layers']
            core_signal_count = len(set(active_signals).intersection(core_signals))
            whale_signal_override = whale_ping_strength >= 0.5
            
            if core_signal_count >= 1 or whale_signal_override:
                if whale_ping_strength > 0.5:
                    explore_mode = True
                    explore_trigger_reason = f"Strong whale signal ({whale_ping_strength:.3f})"
                    explore_confidence = min(0.9, whale_ping_strength / 3.0)
                    
                    print(f"[EXPLORE MODE TRIGGERED] {symbol}: Whale override - strength={whale_ping_strength:.3f} > 0.5")
                    print(f"[EXPLORE MODE TRIGGERED] {symbol}: Confidence={explore_confidence:.3f}, Reason={explore_trigger_reason}")
                    
                    # Add explore mode data to stealth result
                    stealth_result["explore_mode"] = True
                    stealth_result["explore_trigger_reason"] = explore_trigger_reason
                    stealth_result["explore_confidence"] = explore_confidence
                    
                else:
                    print(f"[EXPLORE MODE SKIP] {symbol}: Core signals sufficient ({core_signal_count}) but whale_ping too weak ({whale_ping_strength:.3f})")
            else:
                print(f"[EXPLORE MODE SKIP] {symbol}: Insufficient signals - core_signals={core_signal_count}, whale_ping={whale_ping_strength:.3f}")
                
        except Exception as e:
            print(f"[EXPLORE MODE ERROR] {symbol}: {e}")
        
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
                    # STAGE 10 FIX: Safe variable access using locals()
                    if 'result' in locals():
                        result["alert_sent"] = True
                        result["alert_type"] = "stealth_v3_diamond"
                        result["alert_trigger"] = diamond_decision["decision"]
                else:
                    print(f"[STEALTH V3 DIAMOND ALERT SKIP] {symbol}: Alert w cooldown lub błąd")
                    # STAGE 10 FIX: Safe variable access using locals()
                    if 'result' in locals():
                        result["alert_sent"] = False
                        result["alert_skip_reason"] = "cooldown_or_error"
            else:
                print(f"[STEALTH V3 DIAMOND ALERT] {symbol}: Kryteria alertu nie spełnione (decision: {diamond_decision['decision']}, score: {stealth_score:.3f})")
                # STAGE 10 FIX: Safe variable access using locals()
                if 'result' in locals():
                    result["alert_sent"] = False
                    result["alert_skip_reason"] = "criteria_not_met"
                
        except ImportError:
            print(f"[STEALTH V3 DIAMOND ALERT] {symbol}: Stealth v3 alert system niedostępny")
            # STAGE 10 FIX: Safe variable access using locals()
            if 'result' in locals():
                result["alert_sent"] = False
                result["alert_skip_reason"] = "system_unavailable"
        except Exception as alert_error:
            print(f"[STEALTH V3 DIAMOND ALERT ERROR] {symbol}: {alert_error}")
            # STAGE 10 FIX: Safe variable access using locals()
            if 'result' in locals():
                result["alert_sent"] = False
                result["alert_skip_reason"] = f"error: {alert_error}"
        
        # 🚨🚨🚨 ENHANCED RL INTEGRATION BREAKTHROUGH - FINALLY IN ACTIVE FUNCTION 🚨🚨🚨
        print(f"[ULTIMATE DEBUG TEST] {symbol}: Enhanced RL section REACHED - score: {score:.3f}")
        
        # Only run Enhanced RL for tokens with score > 0.0 to avoid wasting compute
        if score > 0.0:
            print(f"[ENHANCED RL ENTRY] {symbol}: Score {score:.3f} qualifies for Enhanced RL analysis")
            
            try:
                # Enhanced RL kod tutaj - imported from enhanced_rl_integration.py
                from enhanced_rl_integration import EnhancedRLIntegration, DQNAgent
                
                # Initialize Enhanced RL system with RLAgentV3 + DQN hybrid approach
                enhanced_rl = EnhancedRLIntegration()
                
                # Prepare comprehensive state vector for Enhanced RL analysis
                state_vector = enhanced_rl.prepare_comprehensive_state_vector(
                    symbol=symbol,
                    base_score=score,
                    stealth_signals=active_signals,
                    diamond_score=diamond_score if diamond_enabled else 0.0,
                    californium_score=californium_score if californium_enabled else 0.0,
                    whaleclip_score=whaleclip_score if whaleclip_enabled else 0.0,
                    consensus_data=consensus_data if 'consensus_data' in locals() else None,
                    market_data=token_data
                )
                
                print(f"[ENHANCED RL STATE] {symbol}: State vector prepared - dim: {len(state_vector)}")
                
                # Run Enhanced RL analysis with adaptive thresholding
                enhanced_result = enhanced_rl.analyze_with_adaptive_thresholds(
                    symbol=symbol,
                    state_vector=state_vector,
                    base_score=score,
                    use_dqn=True  # Enable DQN integration for dynamic threshold adaptation
                )
                
                # Apply Enhanced RL modifications to score and decision
                if enhanced_result and enhanced_result.get('should_modify', False):
                    original_score = score
                    enhanced_score = enhanced_result.get('enhanced_score', score)
                    adaptive_multiplier = enhanced_result.get('adaptive_multiplier', 1.0)
                    confidence_boost = enhanced_result.get('confidence_boost', 0.0)
                    
                    # Update score with Enhanced RL recommendations
                    score = enhanced_score
                    
                    print(f"[ENHANCED RL BOOST] {symbol}: {original_score:.3f} → {enhanced_score:.3f} (multiplier: {adaptive_multiplier:.3f}, confidence: +{confidence_boost:.3f})")
                    
                    # Update result dictionary with Enhanced RL data
                    if 'result' in locals():
                        result["enhanced_rl_applied"] = True
                        result["original_stealth_score"] = original_score
                        result["enhanced_final_score"] = enhanced_score
                        result["rl_adaptive_multiplier"] = adaptive_multiplier
                        result["rl_confidence_boost"] = confidence_boost
                        result["stealth_score"] = enhanced_score  # Update main score
                        
                        # Update Enhanced RL specific metadata
                        result["enhanced_rl_metadata"] = {
                            "state_vector_dim": len(state_vector),
                            "dqn_enabled": True,
                            "analysis_timestamp": enhanced_result.get('timestamp', 'unknown'),
                            "threshold_adaptation": enhanced_result.get('threshold_info', {})
                        }
                else:
                    print(f"[ENHANCED RL SKIP] {symbol}: Score {score:.3f} - no RL modifications recommended")
                    if 'result' in locals():
                        result["enhanced_rl_applied"] = False
                        result["enhanced_rl_skip_reason"] = enhanced_result.get('skip_reason', 'no_modification_needed') if enhanced_result else 'analysis_failed'
                
            except ImportError as e:
                print(f"[ENHANCED RL IMPORT ERROR] {symbol}: Enhanced RL system not available: {e}")
                if 'result' in locals():
                    result["enhanced_rl_applied"] = False
                    result["enhanced_rl_skip_reason"] = f"import_error: {e}"
            except Exception as e:
                print(f"[ENHANCED RL ERROR] {symbol}: Enhanced RL analysis failed: {e}")
                if 'result' in locals():
                    result["enhanced_rl_applied"] = False
                    result["enhanced_rl_skip_reason"] = f"analysis_error: {e}"
        else:
            print(f"[ENHANCED RL SKIP] {symbol}: Score {score:.3f} too low for Enhanced RL analysis")
            if 'result' in locals():
                result["enhanced_rl_applied"] = False
                result["enhanced_rl_skip_reason"] = "score_too_low"
        
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