"""
Feedback Evaluator - Module 5 Feedback Loop
Porównuje predykcje z rzeczywistością i ocenia skuteczność systemu
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
import os

def was_prediction_successful(entry: Dict, price_after_2h: float, price_after_6h: float) -> Dict:
    """
    Ocenia czy predykcja była skuteczna na podstawie rzeczywistego ruchu ceny
    
    Args:
        entry: Wpis predykcji z cache
        price_after_2h: Cena po 2 godzinach
        price_after_6h: Cena po 6 godzinach
        
    Returns:
        Słownik z wynikiem ewaluacji
    """
    try:
        initial_price = entry["price"]
        label = entry.get("label", "unknown").lower()
        confidence = entry.get("confidence", 0.0)
        tjde_score = entry.get("tjde_score", 0.0)
        decision = entry.get("decision", "wait").lower()
        
        # Określ oczekiwany kierunek na podstawie labela i decyzji
        expected_direction = determine_expected_direction(label, decision, tjde_score)
        
        # Oblicz rzeczywiste zmiany ceny
        change_2h = (price_after_2h - initial_price) / initial_price if initial_price > 0 else 0
        change_6h = (price_after_6h - initial_price) / initial_price if initial_price > 0 else 0
        
        # Oceń sukces na różnych horyzontach czasowych
        success_2h = evaluate_direction_success(expected_direction, change_2h, confidence)
        success_6h = evaluate_direction_success(expected_direction, change_6h, confidence)
        
        # Ogólna ocena sukcesu (priorytet dla 2h, ale uwzględnia 6h)
        overall_success = success_2h["successful"] or (success_6h["successful"] and abs(change_6h) > abs(change_2h))
        
        # Oblicz score quality (czy high score rzeczywiście oznaczał dobry ruch)
        score_quality = evaluate_score_quality(tjde_score, max(abs(change_2h), abs(change_6h)))
        
        return {
            "successful": overall_success,
            "expected_direction": expected_direction,
            "actual_change_2h": round(change_2h, 4),
            "actual_change_6h": round(change_6h, 4),
            "success_2h": success_2h,
            "success_6h": success_6h,
            "score_quality": score_quality,
            "confidence_accuracy": evaluate_confidence_accuracy(confidence, overall_success),
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"[FEEDBACK EVALUATOR ERROR] Failed to evaluate prediction: {e}")
        return {
            "successful": False,
            "error": str(e),
            "evaluation_timestamp": datetime.now().isoformat()
        }

def determine_expected_direction(label: str, decision: str, tjde_score: float) -> str:
    """
    Określa oczekiwany kierunek ruchu na podstawie labela, decyzji i score
    
    Args:
        label: AI label pattern
        decision: Decyzja systemu
        tjde_score: TJDE score
        
    Returns:
        Oczekiwany kierunek: "up", "down", "neutral"
    """
    # Bullish patterns
    bullish_patterns = [
        "pullback", "pullback_continuation", "breakout", "breakout_pattern",
        "momentum_follow", "trend_continuation", "bullish", "support_bounce"
    ]
    
    # Bearish patterns  
    bearish_patterns = [
        "reversal", "reversal_pattern", "breakdown", "bearish", 
        "exhaustion", "resistance_rejection", "bear_trap"
    ]
    
    # Neutral patterns
    neutral_patterns = [
        "range", "consolidation", "consolidation_squeeze", "chaos", 
        "sideways", "unknown", "no_clear_pattern"
    ]
    
    # Prioritize decision over label
    if decision in ["enter", "scalp_entry"] and tjde_score > 0.65:
        return "up"
    elif decision == "avoid" and tjde_score < 0.40:
        return "down"
    
    # Check label patterns
    if any(pattern in label for pattern in bullish_patterns):
        return "up"
    elif any(pattern in label for pattern in bearish_patterns):
        return "down"
    elif any(pattern in label for pattern in neutral_patterns):
        return "neutral"
    
    # Default based on TJDE score
    if tjde_score > 0.70:
        return "up"
    elif tjde_score < 0.40:
        return "down"
    else:
        return "neutral"

def evaluate_direction_success(expected_direction: str, actual_change: float, confidence: float) -> Dict:
    """
    Ocenia sukces kierunku na podstawie oczekiwanego kierunku i rzeczywistej zmiany
    
    Args:
        expected_direction: Oczekiwany kierunek
        actual_change: Rzeczywista zmiana ceny (fraction)
        confidence: Poziom pewności predykcji
        
    Returns:
        Wynik ewaluacji kierunku
    """
    # Progi sukcesu zależne od confidence
    min_move_threshold = 0.010 if confidence < 0.7 else 0.015  # 1.0% lub 1.5%
    strong_move_threshold = 0.025  # 2.5%
    
    if expected_direction == "up":
        if actual_change >= strong_move_threshold:
            return {"successful": True, "quality": "strong", "threshold_met": True}
        elif actual_change >= min_move_threshold:
            return {"successful": True, "quality": "moderate", "threshold_met": True}
        elif actual_change > 0:
            return {"successful": True, "quality": "weak", "threshold_met": False}
        else:
            return {"successful": False, "quality": "failed", "threshold_met": False}
            
    elif expected_direction == "down":
        if actual_change <= -strong_move_threshold:
            return {"successful": True, "quality": "strong", "threshold_met": True}
        elif actual_change <= -min_move_threshold:
            return {"successful": True, "quality": "moderate", "threshold_met": True}
        elif actual_change < 0:
            return {"successful": True, "quality": "weak", "threshold_met": False}
        else:
            return {"successful": False, "quality": "failed", "threshold_met": False}
            
    else:  # neutral
        if abs(actual_change) <= min_move_threshold:
            return {"successful": True, "quality": "correct", "threshold_met": True}
        elif abs(actual_change) <= strong_move_threshold:
            return {"successful": False, "quality": "moderate_miss", "threshold_met": False}
        else:
            return {"successful": False, "quality": "strong_miss", "threshold_met": False}

def evaluate_score_quality(tjde_score: float, max_price_move: float) -> Dict:
    """
    Ocenia jakość TJDE score w kontekście rzeczywistego ruchu ceny
    
    Args:
        tjde_score: TJDE final score
        max_price_move: Maksymalny ruch ceny (absolute value)
        
    Returns:
        Ocena jakości score
    """
    # Oczekiwane korelacje
    if tjde_score > 0.80:  # Very high score
        expected_min_move = 0.020
    elif tjde_score > 0.70:  # High score
        expected_min_move = 0.015
    elif tjde_score > 0.60:  # Medium score
        expected_min_move = 0.010
    else:  # Low score
        expected_min_move = 0.005
    
    if max_price_move >= expected_min_move:
        quality = "good"
        score_accuracy = min(1.0, max_price_move / expected_min_move)
    else:
        quality = "poor"
        score_accuracy = max_price_move / expected_min_move if expected_min_move > 0 else 0
    
    return {
        "quality": quality,
        "score_accuracy": round(score_accuracy, 3),
        "expected_move": expected_min_move,
        "actual_move": max_price_move
    }

def evaluate_confidence_accuracy(confidence: float, was_successful: bool) -> Dict:
    """
    Ocenia dokładność poziomu confidence AI
    
    Args:
        confidence: Poziom pewności AI (0.0-1.0)
        was_successful: Czy predykcja była skuteczna
        
    Returns:
        Ocena dokładności confidence
    """
    # Oczekiwana skuteczność na różnych poziomach confidence
    if confidence >= 0.85:
        expected_success_rate = 0.80
    elif confidence >= 0.75:
        expected_success_rate = 0.70
    elif confidence >= 0.65:
        expected_success_rate = 0.60
    else:
        expected_success_rate = 0.50
    
    # Pojedyncza predykcja: 1 jeśli successful, 0 jeśli nie
    actual_success_rate = 1.0 if was_successful else 0.0
    
    accuracy = abs(expected_success_rate - actual_success_rate)
    
    return {
        "expected_success_rate": expected_success_rate,
        "actual_success": actual_success_rate,
        "accuracy_error": round(accuracy, 3),
        "quality": "good" if accuracy < 0.2 else "moderate" if accuracy < 0.4 else "poor"
    }

def fetch_current_price(symbol: str) -> Optional[float]:
    """
    Pobiera aktualną cenę symbolu z Bybit API
    
    Args:
        symbol: Symbol trading (np. BTCUSDT)
        
    Returns:
        Aktualna cena lub None jeśli błąd
    """
    try:
        # Używaj Bybit API do pobrania ceny
        url = "https://api.bybit.com/v5/market/tickers"
        params = {
            "category": "linear",
            "symbol": symbol
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            ticker = data["result"]["list"][0]
            price = float(ticker.get("lastPrice", 0))
            if price > 0:
                print(f"[PRICE FETCH] {symbol}: Current price {price}")
                return price
                
        print(f"[PRICE FETCH] {symbol}: No price data in API response")
        return None
        
    except Exception as e:
        print(f"[PRICE FETCH] {symbol}: API error - {e}")
        return None

def fetch_price_at(symbol: str, timestamp: datetime) -> Optional[float]:
    """
    Pobiera cenę symbolu w określonym czasie z historycznych danych Bybit
    
    Args:
        symbol: Symbol trading (np. BTCUSDT)
        timestamp: Czas dla którego pobieramy cenę
        
    Returns:
        Cena w określonym czasie lub None jeśli błąd
    """
    try:
        # Konwertuj timestamp na milliseconds
        timestamp_ms = int(timestamp.timestamp() * 1000)
        
        # Pobierz świeczkę 1-minutową dla danego czasu
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": "1",  # 1 minute
            "start": timestamp_ms,
            "end": timestamp_ms + 60000,  # +1 minute
            "limit": 1
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            candle = data["result"]["list"][0]
            # Format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
            close_price = float(candle[4])  # Close price
            
            if close_price > 0:
                print(f"[HISTORICAL PRICE] {symbol} at {timestamp.isoformat()}: {close_price}")
                return close_price
                
        # Fallback: pobierz najbliższą dostępną cenę
        return fetch_nearest_historical_price(symbol, timestamp)
        
    except Exception as e:
        print(f"[HISTORICAL PRICE ERROR] {symbol} at {timestamp.isoformat()}: {e}")
        return fetch_nearest_historical_price(symbol, timestamp)

def fetch_nearest_historical_price(symbol: str, target_timestamp: datetime) -> Optional[float]:
    """
    Fallback: pobiera najbliższą dostępną cenę historyczną
    
    Args:
        symbol: Symbol trading
        target_timestamp: Docelowy czas
        
    Returns:
        Najbliższa dostępna cena lub None
    """
    try:
        # Rozszerz okno czasowe do ±30 minut
        start_time = target_timestamp - timedelta(minutes=30)
        end_time = target_timestamp + timedelta(minutes=30)
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": "5",  # 5 minutes for broader coverage
            "start": start_ms,
            "end": end_ms,
            "limit": 20
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            # Znajdź najbliższą świeczkę
            candles = data["result"]["list"]
            if candles:
                # Znajdź najbliższą świeczkę czasowo
                best_candle = None
                best_time_diff = float('inf')
                
                for candle in candles:
                    candle_time = datetime.fromtimestamp(int(candle[0]) / 1000)
                    time_diff = abs((candle_time - target_timestamp).total_seconds())
                    
                    if time_diff < best_time_diff:
                        best_time_diff = time_diff
                        best_candle = candle
                
                if best_candle:
                    close_price = float(best_candle[4])
                    candle_time = datetime.fromtimestamp(int(best_candle[0]) / 1000)
                    time_diff_min = best_time_diff / 60
                    
                    print(f"[FALLBACK PRICE] {symbol}: Found price {close_price} at {candle_time.isoformat()} ({time_diff_min:.1f}min from target)")
                    return close_price
                
        # Jeśli nie ma dokładnych danych, spróbuj poszerzyć zakres do ±2h
        extended_start = target_timestamp - timedelta(hours=2)
        extended_end = target_timestamp + timedelta(hours=2)
        
        extended_start_ms = int(extended_start.timestamp() * 1000)
        extended_end_ms = int(extended_end.timestamp() * 1000)
        
        # Ostatnia próba z szerszym zakresem
        try:
            params["start"] = extended_start_ms
            params["end"] = extended_end_ms
            params["limit"] = 50
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                    candles = data["result"]["list"]
                    if candles:
                        # Użyj środkowej świecy jako przybliżenie
                        middle_candle = candles[len(candles)//2]
                        fallback_price = float(middle_candle[4])
                        print(f"[FALLBACK PRICE EXTENDED] {symbol}: Using extended range price {fallback_price}")
                        return fallback_price
        except:
            pass
            
        print(f"[FALLBACK PRICE] {symbol}: No historical data available around {target_timestamp.isoformat()}")
        return None
        
    except Exception as e:
        print(f"[FALLBACK PRICE ERROR] {symbol}: {e}")
        return None

def evaluate_predictions_batch(predictions: List[Dict]) -> List[Dict]:
    """
    Ocenia batch predykcji pobierając rzeczywiste ceny historyczne
    
    Args:
        predictions: Lista predykcji do oceny
        
    Returns:
        Lista wyników ewaluacji z prawdziwymi danymi cenowymi
    """
    results = []
    
    for prediction in predictions:
        try:
            symbol = prediction["symbol"]
            prediction_time = datetime.fromisoformat(prediction["timestamp"].replace('Z', '+00:00'))
            
            # Oblicz czasy dla których potrzebujemy ceny
            time_after_2h = prediction_time + timedelta(hours=2)
            time_after_6h = prediction_time + timedelta(hours=6)
            
            print(f"[FEEDBACK EVALUATION] {symbol}: Fetching historical prices...")
            print(f"  Prediction time: {prediction_time.isoformat()}")
            print(f"  Price after 2h:  {time_after_2h.isoformat()}")
            print(f"  Price after 6h:  {time_after_6h.isoformat()}")
            
            # Pobierz rzeczywiste ceny historyczne
            price_after_2h = fetch_price_at(symbol, time_after_2h)
            price_after_6h = fetch_price_at(symbol, time_after_6h)
            
            if price_after_2h is None and price_after_6h is None:
                print(f"[FEEDBACK EVALUATION] {symbol}: Could not fetch historical prices")
                continue
            
            # Użyj dostępnych cen (fallback jeśli jedna z nich nie jest dostępna)
            if price_after_2h is None:
                price_after_2h = price_after_6h
            if price_after_6h is None:
                price_after_6h = price_after_2h
            
            # Skip if no prices available
            if price_after_2h is None or price_after_6h is None:
                print(f"[FEEDBACK EVALUATION] {symbol}: Could not fetch current price")
                continue
            
            # Oceń predykcję z rzeczywistymi danymi
            evaluation = was_prediction_successful(prediction, float(price_after_2h), float(price_after_6h))
            evaluation["symbol"] = symbol
            evaluation["timestamp"] = prediction["timestamp"]
            evaluation["price_after_2h"] = price_after_2h
            evaluation["price_after_6h"] = price_after_6h
            evaluation["evaluation_timestamp"] = datetime.now().isoformat()
            
            results.append(evaluation)
            
            success_status = "SUCCESS" if evaluation['successful'] else "FAILED"
            print(f"[FEEDBACK EVALUATION] {symbol}: {success_status} (2h: {price_after_2h}, 6h: {price_after_6h})")
            
        except Exception as e:
            try:
                symbol = prediction.get("symbol", "UNKNOWN")
            except:
                symbol = "UNKNOWN"
            print(f"[FEEDBACK EVALUATION] Failed: {symbol} - {e}")
            continue
    
    return results

def calculate_label_performance(evaluations: List[Dict]) -> Dict[str, Dict]:
    """
    Oblicza statystyki skuteczności dla każdego labela
    
    Args:
        evaluations: Lista wyników ewaluacji
        
    Returns:
        Statystyki skuteczności per label
    """
    label_stats = {}
    
    for evaluation in evaluations:
        # Pobierz dane z oryginalnej predykcji jeśli dostępne
        # W przeciwnym razie użyj dostępnych danych z evaluation
        label = evaluation.get("label", "unknown")
        successful = evaluation.get("successful", False)
        
        if label not in label_stats:
            label_stats[label] = {
                "total_predictions": 0,
                "successful_predictions": 0,
                "success_rate": 0.0,
                "confidence_avg": 0.0,
                "confidence_sum": 0.0
            }
        
        label_stats[label]["total_predictions"] += 1
        if successful:
            label_stats[label]["successful_predictions"] += 1
        
        # Aktualizuj confidence average jeśli dostępne
        confidence = evaluation.get("confidence", 0.0)
        label_stats[label]["confidence_sum"] += confidence
    
    # Oblicz końcowe statystyki
    for label, stats in label_stats.items():
        total = stats["total_predictions"]
        if total > 0:
            stats["success_rate"] = round(stats["successful_predictions"] / total, 3)
            stats["confidence_avg"] = round(stats["confidence_sum"] / total, 3)
    
    return label_stats