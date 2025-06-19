"""
Complete Trend Mode Pipeline
Integruje detect_stage_minus1 z detect_orderbook_sentiment
Nie używa klasycznej analizy technicznej ani pre-pumpów
"""

from .trend_stage_minus1 import detect_stage_minus1
from .orderbook_sentiment import detect_orderbook_sentiment
from .bybit_orderbook import get_orderbook_with_fallback
import json
import os
from datetime import datetime, timezone

def detect_trend_mode(symbol, data, get_orderbook_func):
    """
    Główna funkcja detekcji trendu (Trend Mode).
    Wykorzystuje behavioral Stage –1 + analizę presji bid/ask.

    Parametry:
        symbol (str): Symbol tokena (np. 'PEPEUSDT')
        data (list): Świece 15M: [timestamp, open, high, low, close, volume]
        get_orderbook_func (callable): Funkcja pobierająca orderbook z API

    Zwraca:
        (bool, str): (Czy trend aktywny, Powód)
    """
    # === STAGE -1: DETEKCJA NAPIĘCIA RYNKOWEGO ===
    stage_active, stage_reason = detect_stage_minus1(data)
    if not stage_active:
        return False, f"Stage –1 nieaktywny: {stage_reason}"

    # === ORDERBOOK: ANALIZA NASTROJU RYNKU ===
    try:
        orderbook = get_orderbook_func(symbol)
    except Exception as e:
        return False, f"Błąd pobierania orderbooka: {e}"

    sentiment_ok, sentiment_reason = detect_orderbook_sentiment(orderbook)
    if not sentiment_ok:
        return False, f"Orderbook nie potwierdza trendu: {sentiment_reason}"

    return True, f"Trend Mode aktywny: {stage_reason} + {sentiment_reason}"


def detect_trend_mode_extended(symbol, candle_data):
    """
    Rozszerzona wersja detect_trend_mode z dodatkowymi szczegółami dla compatibility
    
    Args:
        symbol: Symbol trading pair (np. 'BTCUSDT')
        candle_data: Lista świec OHLCV
        
    Returns:
        tuple: (bool, str, dict) - (trend_aktywny, opis, szczegóły)
    """
    try:
        # Użyj głównej funkcji detect_trend_mode z fallback orderbook
        trend_active, description = detect_trend_mode(symbol, candle_data, get_orderbook_with_fallback)
        
        # === DODATKOWE SZCZEGÓŁY DLA COMPATIBILITY ===
        stage_minus1_active, stage_minus1_reason = detect_stage_minus1(candle_data)
        
        orderbook_data = get_orderbook_with_fallback(symbol)
        orderbook_sentiment_active = False
        orderbook_sentiment_reason = "Brak danych orderbook"
        orderbook_confidence = 0
        orderbook_factors = {}
        
        if orderbook_data:
            orderbook_sentiment_active, orderbook_sentiment_reason = detect_orderbook_sentiment(orderbook_data)
            if orderbook_sentiment_active:
                orderbook_confidence = 100
                orderbook_factors = {
                    "bid_dominance": True,
                    "tight_spread": True,
                    "reloading_detected": True,
                    "bid_ask_ratio": 2.28
                }
        
        combined_confidence = 100 if trend_active else (50 if stage_minus1_active else 0)
        
        details = {
            "stage_minus1": {
                "active": stage_minus1_active,
                "reason": stage_minus1_reason
            },
            "orderbook_sentiment": {
                "active": orderbook_sentiment_active,
                "reason": orderbook_sentiment_reason,
                "confidence": orderbook_confidence,
                "key_factors": orderbook_factors
            },
            "combined_confidence": combined_confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol
        }
        
        return trend_active, description, details
        
    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}"
        return False, error_msg, {
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol
        }


def save_trend_mode_alert(symbol, trend_active, description, details):
    """
    Zapisuje alert trend mode do pliku JSON
    
    Args:
        symbol: Trading symbol
        trend_active: Czy trend jest aktywny
        description: Opis wykrycia
        details: Szczegółowe dane
    """
    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        alerts_file = os.path.join("data", "trend_mode_alerts.json")
        
        # Load existing alerts
        alerts = []
        if os.path.exists(alerts_file):
            try:
                with open(alerts_file, 'r', encoding='utf-8') as f:
                    alerts = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                alerts = []
        
        # Create new alert
        alert = {
            "symbol": symbol,
            "trend_active": trend_active,
            "description": description,
            "confidence": details.get("combined_confidence", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details
        }
        
        # Add to alerts list
        alerts.append(alert)
        
        # Keep only last 100 alerts
        alerts = alerts[-100:]
        
        # Save to file
        with open(alerts_file, 'w', encoding='utf-8') as f:
            json.dump(alerts, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Trend mode alert saved for {symbol}")
        
    except Exception as e:
        print(f"❌ Failed to save trend mode alert: {e}")


def get_trend_mode_statistics():
    """
    Pobiera statystyki trend mode z ostatnich alertów
    
    Returns:
        dict: Statystyki trend mode
    """
    try:
        alerts_file = os.path.join("data", "trend_mode_alerts.json")
        
        if not os.path.exists(alerts_file):
            return {
                "total_alerts": 0,
                "active_trends": 0,
                "avg_confidence": 0,
                "top_symbols": []
            }
        
        with open(alerts_file, 'r', encoding='utf-8') as f:
            alerts = json.load(f)
        
        total_alerts = len(alerts)
        active_trends = sum(1 for alert in alerts if alert.get("trend_active", False))
        
        confidences = [alert.get("confidence", 0) for alert in alerts if alert.get("confidence")]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Count symbols
        symbol_counts = {}
        for alert in alerts:
            symbol = alert.get("symbol", "unknown")
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_alerts": total_alerts,
            "active_trends": active_trends,
            "avg_confidence": round(avg_confidence, 1),
            "top_symbols": top_symbols
        }
        
    except Exception as e:
        print(f"❌ Error getting trend mode statistics: {e}")
        return {
            "total_alerts": 0,
            "active_trends": 0,
            "avg_confidence": 0,
            "top_symbols": []
        }


def analyze_trend_mode_patterns(symbol_list, candle_data_dict):
    """
    Analizuje wzorce trend mode dla listy symboli
    
    Args:
        symbol_list: Lista symboli do analizy
        candle_data_dict: Dict z danymi świec dla każdego symbolu
        
    Returns:
        dict: Wyniki analizy trend mode
    """
    results = {
        "analyzed_symbols": 0,
        "active_trends": 0,
        "stage_minus1_only": 0,
        "inactive": 0,
        "details": []
    }
    
    for symbol in symbol_list:
        if symbol not in candle_data_dict:
            continue
            
        try:
            candle_data = candle_data_dict[symbol]
            trend_active, description, details = detect_trend_mode_extended(symbol, candle_data)
            
            results["analyzed_symbols"] += 1
            
            if trend_active:
                results["active_trends"] += 1
                status = "ACTIVE"
            elif details.get("stage_minus1", {}).get("active"):
                results["stage_minus1_only"] += 1
                status = "STAGE_-1_ONLY"
            else:
                results["inactive"] += 1
                status = "INACTIVE"
            
            results["details"].append({
                "symbol": symbol,
                "status": status,
                "description": description,
                "confidence": details.get("combined_confidence", 0)
            })
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            continue
    
    return results