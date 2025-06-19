"""
Complete Trend Mode Pipeline
Integruje detect_stage_minus1 z detect_orderbook_sentiment
Nie używa klasycznej analizy technicznej ani pre-pumpów
"""

from .trend_stage_minus1 import detect_stage_minus1
from .orderbook_sentiment import detect_orderbook_sentiment, get_orderbook_sentiment_summary
from .bybit_orderbook import get_orderbook_with_fallback, validate_orderbook_data
import json
import os
from datetime import datetime, timezone

def detect_trend_mode(symbol, candle_data):
    """
    Kompletny pipeline trend_mode łączący analizę rytmu i sentymentu orderbook
    
    Args:
        symbol: Symbol trading pair (np. 'BTCUSDT')
        candle_data: Lista świec OHLCV
        
    Returns:
        tuple: (bool, str, dict) - (trend_aktywny, opis, szczegóły)
    """
    try:
        # Krok 1: Analiza rytmu rynku (Stage -1)
        stage_minus1_active, stage_minus1_reason = detect_stage_minus1(candle_data)
        
        if not stage_minus1_active:
            return False, f"Trend nieaktywny: {stage_minus1_reason}", {
                "stage_minus1": {
                    "active": False,
                    "reason": stage_minus1_reason
                },
                "orderbook_sentiment": None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Krok 2: Analiza sentymentu orderbook
        orderbook = get_orderbook_with_fallback(symbol)
        
        if not orderbook or not validate_orderbook_data(orderbook):
            return False, f"Stage -1 aktywny ale brak danych orderbook: {stage_minus1_reason}", {
                "stage_minus1": {
                    "active": True,
                    "reason": stage_minus1_reason
                },
                "orderbook_sentiment": {
                    "active": False,
                    "reason": "Brak danych orderbook"
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Krok 3: Sprawdź sentiment orderbook
        sentiment_active, orderbook_reason = detect_orderbook_sentiment(orderbook)
        sentiment_summary = get_orderbook_sentiment_summary(orderbook)
        
        if sentiment_active:
            # Oba warunki spełnione - trend mode aktywny
            combined_reason = f"Trend aktywny + {stage_minus1_reason} + {orderbook_reason}"
            
            details = {
                "stage_minus1": {
                    "active": True,
                    "reason": stage_minus1_reason
                },
                "orderbook_sentiment": {
                    "active": True,
                    "reason": orderbook_reason,
                    "confidence": sentiment_summary["confidence"],
                    "key_factors": sentiment_summary["key_factors"]
                },
                "combined_confidence": min(90 + sentiment_summary["confidence"] // 10, 100),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol
            }
            
            return True, combined_reason, details
        else:
            # Stage -1 aktywny ale orderbook nie wspiera
            return False, f"Stage -1 aktywny ale {orderbook_reason}", {
                "stage_minus1": {
                    "active": True,
                    "reason": stage_minus1_reason
                },
                "orderbook_sentiment": {
                    "active": False,
                    "reason": orderbook_reason,
                    "confidence": sentiment_summary["confidence"]
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        error_msg = f"Błąd w pipeline trend_mode: {str(e)}"
        return False, error_msg, {
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
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
    if not trend_active:
        return  # Zapisuj tylko aktywne trendy
    
    alert_data = {
        "symbol": symbol,
        "trend_active": trend_active,
        "description": description,
        "stage_minus1_reason": details["stage_minus1"]["reason"],
        "orderbook_reason": details["orderbook_sentiment"]["reason"],
        "confidence": details.get("combined_confidence", 0),
        "timestamp": details["timestamp"]
    }
    
    # Zapisz do pliku
    os.makedirs("data", exist_ok=True)
    alerts_file = "data/trend_mode_alerts.json"
    
    try:
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r', encoding='utf-8') as f:
                existing_alerts = json.load(f)
        else:
            existing_alerts = []
        
        existing_alerts.append(alert_data)
        
        # Zachowaj tylko ostatnie 100 alertów
        if len(existing_alerts) > 100:
            existing_alerts = existing_alerts[-100:]
        
        with open(alerts_file, 'w', encoding='utf-8') as f:
            json.dump(existing_alerts, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Trend mode alert saved for {symbol}")
        
    except Exception as e:
        print(f"❌ Error saving trend mode alert: {e}")

def get_trend_mode_statistics():
    """
    Pobiera statystyki trend mode z ostatnich alertów
    
    Returns:
        dict: Statystyki trend mode
    """
    alerts_file = "data/trend_mode_alerts.json"
    
    if not os.path.exists(alerts_file):
        return {
            "total_alerts": 0,
            "active_trends": 0,
            "avg_confidence": 0,
            "top_symbols": []
        }
    
    try:
        with open(alerts_file, 'r', encoding='utf-8') as f:
            alerts = json.load(f)
        
        if not alerts:
            return {
                "total_alerts": 0,
                "active_trends": 0,
                "avg_confidence": 0,
                "top_symbols": []
            }
        
        active_alerts = [a for a in alerts if a.get("trend_active")]
        confidences = [a.get("confidence", 0) for a in active_alerts]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Top symbole
        symbol_counts = {}
        for alert in active_alerts:
            symbol = alert.get("symbol", "UNKNOWN")
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_alerts": len(alerts),
            "active_trends": len(active_alerts),
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
        "full_trend_mode": 0,
        "details": []
    }
    
    for symbol in symbol_list:
        if symbol not in candle_data_dict:
            continue
            
        candle_data = candle_data_dict[symbol]
        trend_active, description, details = detect_trend_mode(symbol, candle_data)
        
        results["analyzed_symbols"] += 1
        
        if details.get("stage_minus1", {}).get("active"):
            if trend_active:
                results["full_trend_mode"] += 1
                results["active_trends"] += 1
                save_trend_mode_alert(symbol, True, description, details)
            else:
                results["stage_minus1_only"] += 1
        
        results["details"].append({
            "symbol": symbol,
            "trend_active": trend_active,
            "description": description,
            "confidence": details.get("combined_confidence", 0)
        })
    
    return results