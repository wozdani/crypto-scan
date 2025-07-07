"""
Stealth Feedback System
System uczenia wag przez feedback loop dla Stealth Engine

Analizuje skuteczność sygnałów na podstawie rzeczywistych wyników
i automatycznie dostosowuje wagi dla lepszej dokładności predykcji
"""

import json
import os
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics


@dataclass
class StealthPrediction:
    """Predykcja Stealth Engine do ewaluacji"""
    prediction_id: str
    symbol: str
    stealth_score: float
    decision: str
    confidence: float
    active_signals: List[str]
    signal_breakdown: Dict[str, float]
    price_at_prediction: float
    timestamp: float
    evaluated: bool = False
    price_after_2h: Optional[float] = None
    price_after_6h: Optional[float] = None
    success_2h: Optional[bool] = None
    success_6h: Optional[bool] = None


class StealthFeedbackSystem:
    """
    System uczenia wag sygnałów przez feedback loop
    Ewaluuje skuteczność predykcji i dostosowuje wagi
    """
    
    def __init__(self, config_path: str = "crypto-scan/cache"):
        """
        Inicjalizacja systemu feedback
        
        Args:
            config_path: Ścieżka do katalogu z konfiguracją
        """
        self.config_path = config_path
        self.feedback_dir = os.path.join(config_path, "stealth_feedback")
        self.predictions_file = os.path.join(self.feedback_dir, "predictions.jsonl")
        self.performance_file = os.path.join(self.feedback_dir, "signal_performance.json")
        self.weight_updates_file = os.path.join(self.feedback_dir, "weight_updates.json")
        
        self.ensure_directories()
        
        # Konfiguracja ewaluacji
        self.min_price_change_threshold = 0.02  # 2% min zmiana dla sukcesu
        self.evaluation_hours = [2, 6]  # Ewaluacja po 2h i 6h
        self.min_predictions_for_update = 10  # Min predykcji do aktualizacji wag
        
        print(f"[STEALTH FEEDBACK] Initialized feedback system in {self.feedback_dir}")
    
    def ensure_directories(self):
        """Upewnij się że wymagane katalogi istnieją"""
        os.makedirs(self.feedback_dir, exist_ok=True)
    
    def log_prediction(self, stealth_result, market_data: Dict) -> str:
        """
        Zaloguj predykcję do systemu feedback
        
        Args:
            stealth_result: Wynik z StealthEngine
            market_data: Dane rynkowe (zawierające cenę)
            
        Returns:
            ID predykcji
        """
        try:
            # Utwórz unikalny ID predykcji
            prediction_id = f"{stealth_result.symbol}_{int(time.time())}"
            
            # Pobierz cenę z market_data
            current_price = market_data.get('price', 0)
            if current_price == 0:
                # Spróbuj pobrać z ticker_data
                ticker_data = market_data.get('ticker_data', {})
                if isinstance(ticker_data, dict):
                    current_price = float(ticker_data.get('lastPrice', 0))
            
            # Utwórz predykcję
            prediction = StealthPrediction(
                prediction_id=prediction_id,
                symbol=stealth_result.symbol,
                stealth_score=stealth_result.stealth_score,
                decision=stealth_result.decision,
                confidence=stealth_result.confidence,
                active_signals=stealth_result.active_signals,
                signal_breakdown=stealth_result.signal_breakdown,
                price_at_prediction=current_price,
                timestamp=stealth_result.timestamp
            )
            
            # Zapisz do pliku JSONL
            with open(self.predictions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(prediction), ensure_ascii=False) + '\n')
            
            print(f"[STEALTH FEEDBACK] Logged prediction: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to log prediction: {e}")
            return ""
    
    def load_pending_predictions(self) -> List[StealthPrediction]:
        """
        Załaduj predykcje oczekujące na ewaluację
        
        Returns:
            Lista nieewaluowanych predykcji
        """
        predictions = []
        
        if not os.path.exists(self.predictions_file):
            return predictions
        
        try:
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pred_data = json.loads(line)
                        prediction = StealthPrediction(**pred_data)
                        
                        # Dodaj tylko nieewaluowane predykcje starsze niż 6h
                        if not prediction.evaluated:
                            age_hours = (time.time() - prediction.timestamp) / 3600
                            if age_hours >= 6:  # Minimum 6h dla pełnej ewaluacji
                                predictions.append(prediction)
            
            print(f"[STEALTH FEEDBACK] Loaded {len(predictions)} pending predictions")
            return predictions
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to load predictions: {e}")
            return []
    
    async def evaluate_predictions(self) -> Dict:
        """
        Ewaluuj oczekujące predykcje
        
        Returns:
            Statystyki ewaluacji
        """
        pending_predictions = self.load_pending_predictions()
        
        if len(pending_predictions) < self.min_predictions_for_update:
            print(f"[STEALTH FEEDBACK] Not enough predictions for evaluation: {len(pending_predictions)}")
            return {'evaluated': 0, 'reason': 'insufficient_data'}
        
        evaluated_count = 0
        
        for prediction in pending_predictions:
            try:
                # Pobierz ceny historyczne
                price_2h = await self.fetch_historical_price(prediction.symbol, prediction.timestamp + 2*3600)
                price_6h = await self.fetch_historical_price(prediction.symbol, prediction.timestamp + 6*3600)
                
                # Oceń sukces
                success_2h = self.evaluate_success(prediction, price_2h)
                success_6h = self.evaluate_success(prediction, price_6h)
                
                # Aktualizuj predykcję
                prediction.price_after_2h = price_2h
                prediction.price_after_6h = price_6h
                prediction.success_2h = success_2h
                prediction.success_6h = success_6h
                prediction.evaluated = True
                
                evaluated_count += 1
                
            except Exception as e:
                print(f"[STEALTH FEEDBACK ERROR] Failed to evaluate {prediction.prediction_id}: {e}")
        
        # Zapisz zaktualizowane predykcje
        if evaluated_count > 0:
            self.save_evaluated_predictions(pending_predictions)
            print(f"[STEALTH FEEDBACK] Evaluated {evaluated_count} predictions")
        
        return {'evaluated': evaluated_count}
    
    async def fetch_historical_price(self, symbol: str, timestamp: float) -> Optional[float]:
        """
        Pobierz historyczną cenę tokena
        
        Args:
            symbol: Symbol tokena
            timestamp: Timestamp (unix)
            
        Returns:
            Cena lub None jeśli nie udało się pobrać
        """
        try:
            # Import Bybit API
            import aiohttp
            
            # Konwertuj timestamp na milisekundy
            end_time = int(timestamp * 1000)
            start_time = end_time - 3600000  # 1h przed
            
            # API call do Bybit klines
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                'category': 'spot',
                'symbol': symbol,
                'interval': '15',
                'start': start_time,
                'end': end_time,
                'limit': 10
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                            # Pobierz ostatnią dostępną cenę
                            klines = data['result']['list']
                            if klines:
                                # Kline format: [timestamp, open, high, low, close, volume, turnover]
                                last_kline = klines[0]  # Najbardziej aktualna
                                close_price = float(last_kline[4])
                                return close_price
            
            return None
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK] Failed to fetch price for {symbol}: {e}")
            return None
    
    def evaluate_success(self, prediction: StealthPrediction, price_after: Optional[float]) -> Optional[bool]:
        """
        Oceń czy predykcja była skuteczna
        
        Args:
            prediction: Predykcja do oceny
            price_after: Cena po określonym czasie
            
        Returns:
            True/False dla sukcesu/porażki, None jeśli brak danych
        """
        if price_after is None or prediction.price_at_prediction == 0:
            return None
        
        price_change = (price_after - prediction.price_at_prediction) / prediction.price_at_prediction
        
        # Logika sukcesu na podstawie decyzji
        if prediction.decision == 'enter':
            # Sukces dla 'enter' gdy cena wzrosła >2%
            return price_change >= self.min_price_change_threshold
        elif prediction.decision == 'avoid':
            # Sukces dla 'avoid' gdy cena spadła lub wzrosła <2%
            return price_change < self.min_price_change_threshold
        elif prediction.decision == 'wait':
            # 'wait' to neutralna predykcja - sukces przy małych zmianach
            return abs(price_change) < self.min_price_change_threshold
        
        return None
    
    def save_evaluated_predictions(self, predictions: List[StealthPrediction]):
        """Zapisz ewaluowane predykcje z powrotem do pliku"""
        try:
            # Załaduj wszystkie predykcje
            all_predictions = []
            
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            pred_data = json.loads(line)
                            all_predictions.append(StealthPrediction(**pred_data))
            
            # Aktualizuj ewaluowane predykcje
            predictions_dict = {p.prediction_id: p for p in predictions}
            
            for i, pred in enumerate(all_predictions):
                if pred.prediction_id in predictions_dict:
                    all_predictions[i] = predictions_dict[pred.prediction_id]
            
            # Zapisz wszystkie predykcje
            with open(self.predictions_file, 'w', encoding='utf-8') as f:
                for pred in all_predictions:
                    f.write(json.dumps(asdict(pred), ensure_ascii=False) + '\n')
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to save predictions: {e}")
    
    def calculate_signal_performance(self) -> Dict[str, Dict]:
        """
        Oblicz wydajność każdego sygnału na podstawie ewaluowanych predykcji
        
        Returns:
            Słownik z wydajnością sygnałów
        """
        try:
            # Załaduj wszystkie ewaluowane predykcje
            evaluated_predictions = []
            
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            pred_data = json.loads(line)
                            prediction = StealthPrediction(**pred_data)
                            if prediction.evaluated and prediction.success_6h is not None:
                                evaluated_predictions.append(prediction)
            
            if not evaluated_predictions:
                return {}
            
            # Analiza wydajności per sygnał
            signal_stats = {}
            
            for prediction in evaluated_predictions:
                for signal_name in prediction.active_signals:
                    if signal_name not in signal_stats:
                        signal_stats[signal_name] = {
                            'total_predictions': 0,
                            'successful_predictions': 0,
                            'success_rate': 0.0,
                            'avg_confidence': 0.0,
                            'confidences': []
                        }
                    
                    stats = signal_stats[signal_name]
                    stats['total_predictions'] += 1
                    stats['confidences'].append(prediction.confidence)
                    
                    if prediction.success_6h:
                        stats['successful_predictions'] += 1
            
            # Oblicz finalne statystyki
            for signal_name, stats in signal_stats.items():
                if stats['total_predictions'] > 0:
                    stats['success_rate'] = stats['successful_predictions'] / stats['total_predictions']
                    stats['avg_confidence'] = statistics.mean(stats['confidences'])
                
                # Usuń surowe listy confidence
                del stats['confidences']
            
            # Zapisz wydajność
            with open(self.performance_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'signal_performance': signal_stats,
                    'last_updated': time.time(),
                    'total_evaluated_predictions': len(evaluated_predictions)
                }, f, indent=2)
            
            print(f"[STEALTH FEEDBACK] Calculated performance for {len(signal_stats)} signals")
            return signal_stats
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to calculate performance: {e}")
            return {}
    
    def calculate_updated_weights(self) -> Dict[str, float]:
        """
        Oblicz zaktualizowane wagi na podstawie wydajności sygnałów
        
        Returns:
            Słownik z nowymi wagami
        """
        signal_performance = self.calculate_signal_performance()
        
        if not signal_performance:
            return {}
        
        updated_weights = {}
        
        for signal_name, performance in signal_performance.items():
            success_rate = performance['success_rate']
            total_predictions = performance['total_predictions']
            
            # Wymagaj minimum predykcji dla wiarygodności
            if total_predictions < 5:
                continue
            
            # Oblicz nową wagę na podstawie success rate
            if success_rate >= 0.6:
                # Dobry sygnał - zwiększ wagę
                weight_multiplier = 1.0 + (success_rate - 0.6) * 0.5  # Max 1.2x dla 80% success
            elif success_rate <= 0.4:
                # Słaby sygnał - zmniejsz wagę
                weight_multiplier = 0.5 + success_rate  # Min 0.5x dla 0% success
            else:
                # Neutralny sygnał - mała korekta
                weight_multiplier = 0.8 + success_rate * 0.4  # 0.8x-1.0x dla 40-60%
            
            updated_weights[signal_name] = weight_multiplier
        
        # Zapisz aktualizacje
        if updated_weights:
            weight_update_data = {
                'weight_multipliers': updated_weights,
                'calculated_at': time.time(),
                'based_on_signals': len(signal_performance),
                'performance_summary': signal_performance
            }
            
            with open(self.weight_updates_file, 'w', encoding='utf-8') as f:
                json.dump(weight_update_data, f, indent=2)
            
            print(f"[STEALTH FEEDBACK] Calculated weight updates for {len(updated_weights)} signals")
        
        return updated_weights
    
    def get_stats(self) -> Dict:
        """Pobierz statystyki systemu feedback"""
        try:
            total_predictions = 0
            evaluated_predictions = 0
            
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            total_predictions += 1
                            pred_data = json.loads(line)
                            if pred_data.get('evaluated', False):
                                evaluated_predictions += 1
            
            return {
                'total_predictions': total_predictions,
                'evaluated_predictions': evaluated_predictions,
                'pending_predictions': total_predictions - evaluated_predictions,
                'feedback_dir': self.feedback_dir,
                'files_exist': {
                    'predictions': os.path.exists(self.predictions_file),
                    'performance': os.path.exists(self.performance_file),
                    'weight_updates': os.path.exists(self.weight_updates_file)
                }
            }
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Failed to get stats: {e}")
            return {}