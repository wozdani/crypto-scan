"""
Stealth Feedback System v3
Mechanizm samouczenia wag sygnałów StealthSignal

Automatyczne dostrajanie wag na podstawie skuteczności alertów z przeszłości:
- Po 2-6h sprawdza efektywność alertu
- Wzmacnia wagi skutecznych sygnałów (+X% cena)
- Osłabia wagi nieskutecznych sygnałów (spadek/brak ruchu)
"""

import json
import os
import time
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from .stealth_weights import update_weight

class StealthFeedbackSystem:
    """
    System feedback loop v3 dla automatycznego uczenia wag sygnałów
    Implementuje mechanizm samouczenia na podstawie skuteczności alertów
    """
    
    def __init__(self, feedback_dir: str = "crypto-scan/cache/stealth_feedback"):
        """
        Inicjalizacja systemu feedback v3
        
        Args:
            feedback_dir: Katalog do przechowywania danych feedback
        """
        self.feedback_dir = feedback_dir
        self.predictions_file = os.path.join(feedback_dir, "predictions.json")
        self.history_file = os.path.join(feedback_dir, "history.json")
        self.feedback_log_file = os.path.join(feedback_dir, "feedback_log.json")
        self.performance_file = os.path.join(feedback_dir, "performance.json")
        
        # Konfiguracja feedback loop
        self.evaluation_hours = [2, 6]  # Sprawdzaj skuteczność po 2h i 6h
        self.success_threshold = 0.02   # +2% = sukces
        self.failure_threshold = -0.01  # -1% = porażka
        
        # Upewnij się że katalogi istnieją
        self.ensure_directories()
        
        print(f"[STEALTH FEEDBACK] Initialized feedback system v3 in {feedback_dir}")
    
    def ensure_directories(self):
        """Upewnij się że wymagane katalogi istnieją"""
        os.makedirs(self.feedback_dir, exist_ok=True)
    
    def log_prediction(self, stealth_result, market_data: Dict) -> str:
        """
        Zaloguj predykcję do systemu feedback v3
        
        Args:
            stealth_result: Wynik analizy stealth
            market_data: Dane rynkowe w momencie predykcji
            
        Returns:
            ID predykcji dla trackingu
        """
        try:
            prediction_id = f"{stealth_result.symbol}_{int(time.time())}"
            
            # Przygotuj dane aktywnych sygnałów
            active_signal_names = []
            signal_strengths = {}
            
            if hasattr(stealth_result, 'active_signals') and stealth_result.active_signals:
                active_signal_names = stealth_result.active_signals
            
            if hasattr(stealth_result, 'signal_breakdown') and stealth_result.signal_breakdown:
                signal_strengths = stealth_result.signal_breakdown
            
            prediction_data = {
                "prediction_id": prediction_id,
                "symbol": stealth_result.symbol,
                "timestamp": stealth_result.timestamp,
                "timestamp_human": datetime.fromtimestamp(stealth_result.timestamp).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "stealth_score": stealth_result.stealth_score,
                "decision": stealth_result.decision,
                "confidence": stealth_result.confidence,
                "active_signals": active_signal_names,
                "signal_strengths": signal_strengths,
                "risk_assessment": stealth_result.risk_assessment,
                "initial_price": market_data.get("price", 0.0),
                "volume_24h": market_data.get("volume_24h", 0.0),
                "price_change_24h": market_data.get("price_change_24h", 0.0),
                "evaluation_pending": True,
                "evaluation_attempts": 0,
                "feedback_applied": False
            }
            
            # Załaduj istniejące predykcje
            predictions = self.load_predictions()
            predictions[prediction_id] = prediction_data
            
            # Zapisz do pliku
            with open(self.predictions_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
            
            print(f"[STEALTH FEEDBACK] Logged prediction: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to log prediction: {e}")
            return ""
    
    def evaluate_pending_predictions(self) -> Dict:
        """
        Oceń oczekujące predykcje i zastosuj feedback do wag
        
        Returns:
            Statystyki ewaluacji
        """
        try:
            predictions = self.load_predictions()
            current_time = time.time()
            
            evaluated_count = 0
            successful_count = 0
            failed_count = 0
            feedback_applied_count = 0
            
            for prediction_id, prediction in predictions.items():
                if not prediction.get("evaluation_pending", False):
                    continue
                
                # Sprawdź czy minął czas na ewaluację (2h lub 6h)
                prediction_time = prediction.get("timestamp", 0)
                time_elapsed_hours = (current_time - prediction_time) / 3600
                
                if time_elapsed_hours >= 2 and not prediction.get("feedback_applied", False):
                    # Pobierz aktualną cenę i oceń skuteczność
                    outcome = self.evaluate_prediction_outcome(prediction)
                    
                    if outcome is not None:
                        # Zastosuj feedback do wag sygnałów
                        if self.apply_feedback_to_signals(prediction, outcome):
                            feedback_applied_count += 1
                            prediction["feedback_applied"] = True
                            
                        if outcome > 0:
                            successful_count += 1
                        else:
                            failed_count += 1
                        
                        # Oznacz jako ocenioną
                        prediction["evaluation_pending"] = False
                        prediction["outcome"] = outcome
                        prediction["evaluated_at"] = current_time
                        prediction["evaluated_at_human"] = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S UTC")
                        
                        evaluated_count += 1
            
            # Zapisz zaktualizowane predykcje
            if evaluated_count > 0:
                with open(self.predictions_file, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, indent=2, ensure_ascii=False)
                
                print(f"[STEALTH FEEDBACK] Evaluated {evaluated_count} predictions: {successful_count} successful, {failed_count} failed")
                print(f"[STEALTH FEEDBACK] Applied feedback to {feedback_applied_count} predictions")
            
            return {
                "evaluated": evaluated_count,
                "successful": successful_count,
                "failed": failed_count,
                "feedback_applied": feedback_applied_count
            }
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to evaluate predictions: {e}")
            return {"evaluated": 0, "successful": 0, "failed": 0, "feedback_applied": 0}
    
    def evaluate_prediction_outcome(self, prediction: Dict) -> Optional[float]:
        """
        Oceń skuteczność pojedynczej predykcji
        
        Args:
            prediction: Dane predykcji
            
        Returns:
            Outcome [-1.0, +1.0] lub None jeśli błąd
        """
        try:
            symbol = prediction.get("symbol", "")
            initial_price = prediction.get("initial_price", 0.0)
            
            if not symbol or initial_price <= 0:
                return None
            
            # Pobierz aktualną cenę z Bybit API
            current_price = self.fetch_current_price(symbol)
            if current_price is None:
                return None
            
            # Oblicz zmianę ceny w procentach
            price_change_pct = (current_price - initial_price) / initial_price
            
            # Określ outcome na podstawie zmiany ceny
            if price_change_pct >= self.success_threshold:
                # Sukces: +2% lub więcej
                outcome = min(1.0, price_change_pct / 0.1)  # Normalizuj do [0, 1.0] przy 10% zysku
            elif price_change_pct <= self.failure_threshold:
                # Porażka: -1% lub więcej strat
                outcome = max(-1.0, price_change_pct / 0.05)  # Normalizuj do [-1.0, 0] przy 5% stracie
            else:
                # Neutralny: między -1% a +2%
                outcome = price_change_pct / 0.02  # Lekka kara/nagroda
            
            print(f"[STEALTH FEEDBACK] {symbol}: {initial_price:.6f} → {current_price:.6f} ({price_change_pct*100:+.2f}%) = outcome {outcome:.3f}")
            
            return outcome
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to evaluate outcome: {e}")
            return None
    
    def fetch_current_price(self, symbol: str) -> Optional[float]:
        """
        Pobierz aktualną cenę z Bybit API
        
        Args:
            symbol: Symbol tokena (np. BTCUSDT)
            
        Returns:
            Aktualna cena lub None jeśli błąd
        """
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {
                "category": "spot",
                "symbol": symbol
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                    ticker = data["result"]["list"][0]
                    return float(ticker.get("lastPrice", 0))
            
            return None
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to fetch price for {symbol}: {e}")
            return None
    
    def apply_feedback_to_signals(self, prediction: Dict, outcome: float) -> bool:
        """
        Zastosuj feedback do wag aktywnych sygnałów
        
        Args:
            prediction: Dane predykcji
            outcome: Skuteczność [-1.0, +1.0]
            
        Returns:
            True jeśli feedback został zastosowany
        """
        try:
            active_signals = prediction.get("active_signals", [])
            signal_strengths = prediction.get("signal_strengths", {})
            
            if not active_signals:
                return False
            
            feedback_applied = 0
            feedback_details = []
            
            for signal_name in active_signals:
                # Pobierz siłę sygnału (domyślnie 1.0)
                signal_strength = signal_strengths.get(signal_name, 1.0)
                
                # Oblicz delta dla wagi: outcome * siła sygnału * współczynnik uczenia
                learning_rate = 0.1  # Współczynnik uczenia
                delta = learning_rate * outcome * signal_strength
                
                # Zastosuj update do wagi
                if update_weight(signal_name, delta):
                    feedback_applied += 1
                    feedback_details.append({
                        "signal": signal_name,
                        "strength": signal_strength,
                        "delta": delta,
                        "outcome": outcome
                    })
            
            # Zaloguj feedback do feedback_log
            self.log_feedback_application(prediction, outcome, feedback_details)
            
            print(f"[STEALTH FEEDBACK] Applied feedback to {feedback_applied}/{len(active_signals)} signals (outcome: {outcome:.3f})")
            
            return feedback_applied > 0
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to apply feedback: {e}")
            return False
    
    def log_feedback_application(self, prediction: Dict, outcome: float, feedback_details: List[Dict]):
        """
        Zaloguj zastosowanie feedback do historii
        
        Args:
            prediction: Dane predykcji
            outcome: Skuteczność
            feedback_details: Szczegóły zastosowanego feedback
        """
        try:
            # Załaduj istniejący log
            feedback_log = []
            if os.path.exists(self.feedback_log_file):
                with open(self.feedback_log_file, 'r', encoding='utf-8') as f:
                    feedback_log = json.load(f)
            
            # Dodaj nowy wpis
            log_entry = {
                "timestamp": time.time(),
                "timestamp_human": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "prediction_id": prediction.get("prediction_id", ""),
                "symbol": prediction.get("symbol", ""),
                "outcome": outcome,
                "feedback_details": feedback_details,
                "initial_price": prediction.get("initial_price", 0),
                "stealth_score": prediction.get("stealth_score", 0),
                "decision": prediction.get("decision", "")
            }
            
            feedback_log.append(log_entry)
            
            # Zachowaj tylko ostatnie 1000 wpisów
            if len(feedback_log) > 1000:
                feedback_log = feedback_log[-1000:]
            
            # Zapisz log
            with open(self.feedback_log_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_log, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to log feedback application: {e}")
    
    def load_predictions(self) -> Dict:
        """Załaduj istniejące predykcje"""
        try:
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to load predictions: {e}")
            return {}
    
    def get_feedback_stats(self) -> Dict:
        """
        Pobierz statystyki systemu feedback v3
        
        Returns:
            Słownik ze statystykami
        """
        try:
            predictions = self.load_predictions()
            pending = sum(1 for p in predictions.values() if p.get("evaluation_pending", True))
            evaluated = sum(1 for p in predictions.values() if not p.get("evaluation_pending", True))
            successful = sum(1 for p in predictions.values() if not p.get("evaluation_pending", True) and p.get("outcome", 0) > 0)
            
            # Statystyki feedback log
            feedback_entries = 0
            if os.path.exists(self.feedback_log_file):
                with open(self.feedback_log_file, 'r', encoding='utf-8') as f:
                    feedback_log = json.load(f)
                    feedback_entries = len(feedback_log)
            
            return {
                "total_predictions": len(predictions),
                "pending_evaluation": pending,
                "evaluated_predictions": evaluated,
                "successful_predictions": successful,
                "success_rate": successful / evaluated if evaluated > 0 else 0,
                "feedback_log_entries": feedback_entries,
                "feedback_dir": self.feedback_dir
            }
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to get stats: {e}")
            return {
                "total_predictions": 0,
                "pending_evaluation": 0,
                "evaluated_predictions": 0,
                "successful_predictions": 0,
                "success_rate": 0,
                "feedback_log_entries": 0,
                "feedback_dir": self.feedback_dir
            }


# Convenience functions dla łatwego użycia
def apply_feedback_to_signal(token: str, signals: List, outcome: float):
    """
    Zastosuj feedback do sygnałów (zgodnie z oryginalną specyfikacją)
    
    Args:
        token: Symbol tokena (np. "XYZUSDT")
        signals: Lista StealthSignal użytych w momencie alertu
        outcome: Efektywność alertu [-1.0, +1.0]
    """
    try:
        learning_rate = 0.1
        
        for signal in signals:
            if hasattr(signal, 'active') and signal.active:
                signal_strength = getattr(signal, 'strength', 1.0)
                delta = learning_rate * outcome * signal_strength
                
                signal_name = getattr(signal, 'name', str(signal))
                update_weight(signal_name, delta)
                
                print(f"[STEALTH FEEDBACK] {token}: Updated {signal_name} by {delta:+.4f} (outcome: {outcome:.3f}, strength: {signal_strength:.3f})")
        
    except Exception as e:
        print(f"[STEALTH FEEDBACK ERROR] Failed to apply feedback for {token}: {e}")