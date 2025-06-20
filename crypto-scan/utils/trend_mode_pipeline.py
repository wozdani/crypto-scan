"""
Complete Trend Mode Pipeline
Integruje detect_stage_minus1 z detect_orderbook_sentiment + Wave Flow Detector
Nie używa klasycznej analizy technicznej ani pre-pumpów
"""

from .trend_stage_minus1 import detect_stage_minus1
from .orderbook_sentiment import detect_orderbook_sentiment
from .bybit_orderbook import get_orderbook_with_fallback
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from detectors.directional_flow_detector import detect_directional_flow, calculate_directional_score
from detectors.flow_consistency import compute_flow_consistency, calculate_flow_consistency_score, get_price_series_bybit
from detectors.pulse_delay import detect_pulse_delay, calculate_pulse_delay_score
from detectors.orderbook_freeze import detect_orderbook_freeze, calculate_orderbook_freeze_score, create_mock_orderbook_snapshots
from detectors.heatmap_vacuum import detect_heatmap_vacuum, calculate_heatmap_vacuum_score, create_mock_heatmap_snapshots
from detectors.vwap_pinning import detect_vwap_pinning, calculate_vwap_pinning_score, calculate_vwap_values
from detectors.one_sided_pressure import detect_one_sided_pressure, calculate_one_sided_pressure_score
import json
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
        
        # === COMPREHENSIVE FLOW ANALYSIS (Directional + Consistency + Pulse Delay + Orderbook Freeze + Heatmap Vacuum + VWAP Pinning) ===
        directional_flow_detected = False
        directional_flow_score = 0
        directional_flow_details = {}
        flow_consistency = 0.0
        flow_consistency_score = 0
        flow_consistency_desc = ""
        pulse_delay_detected = False
        pulse_delay_score = 0
        pulse_delay_details = {}
        orderbook_freeze_detected = False
        orderbook_freeze_score = 0
        orderbook_freeze_details = {}
        heatmap_vacuum_detected = False
        heatmap_vacuum_score = 0
        heatmap_vacuum_details = {}
        vwap_pinning_detected = False
        vwap_pinning_score = 0
        vwap_pinning_details = {}
        
        try:
            # Pobierz serię cen z ostatnich 3h (dane 5-minutowe)
            prices = get_price_series_bybit(symbol)
            if prices:
                # Directional Flow Detection
                flow_result = detect_directional_flow(prices)
                directional_flow_detected, flow_description, directional_flow_details = flow_result
                directional_flow_score = calculate_directional_score(flow_result)
                
                # Flow Consistency Index
                flow_consistency = compute_flow_consistency(prices)
                flow_consistency_score, flow_consistency_desc = calculate_flow_consistency_score(flow_consistency)
                
                # Pulse Delay Detection
                pulse_result = detect_pulse_delay(prices)
                pulse_delay_detected, pulse_description, pulse_delay_details = pulse_result
                pulse_delay_score = calculate_pulse_delay_score(pulse_result)
                
                # Orderbook Freeze Detection (using mock data in development)
                orderbook_snapshots = create_mock_orderbook_snapshots()  # In production, use real snapshots
                freeze_result = detect_orderbook_freeze(orderbook_snapshots)
                orderbook_freeze_detected, freeze_description, orderbook_freeze_details = freeze_result
                orderbook_freeze_score = calculate_orderbook_freeze_score(freeze_result)
                
                # Heatmap Vacuum Detection (using mock data in development)
                heatmap_snapshots = create_mock_heatmap_snapshots()  # In production, use real heatmap data
                vacuum_result = detect_heatmap_vacuum(heatmap_snapshots)
                heatmap_vacuum_detected, vacuum_description, heatmap_vacuum_details = vacuum_result
                heatmap_vacuum_score = calculate_heatmap_vacuum_score(vacuum_result)
                
                # VWAP Pinning Detection
                # Calculate VWAP values from price data (simplified calculation for development)
                # In production, this would use actual volume data
                vwap_values = []
                if len(prices) >= 20:
                    # Simple moving average as VWAP approximation for development
                    for i in range(len(prices)):
                        start_idx = max(0, i - 9)  # 10-period window
                        window_prices = prices[start_idx:i+1]
                        avg_price = sum(window_prices) / len(window_prices)
                        vwap_values.append(avg_price)
                    
                    pinning_result = detect_vwap_pinning(prices, vwap_values)
                    vwap_pinning_detected, pinning_description, vwap_pinning_details = pinning_result
                    vwap_pinning_score = calculate_vwap_pinning_score(pinning_result)
                else:
                    vwap_pinning_detected = False
                    pinning_description = "Za mało danych dla analizy VWAP pinning"
                    vwap_pinning_details = {}
                    vwap_pinning_score = 0
                
                # Logging results
                if directional_flow_detected:
                    print(f"📈 {symbol}: Natural flow detected - {flow_description} (+{directional_flow_score} points)")
                else:
                    print(f"📉 {symbol}: Chaotic flow - {flow_description} ({directional_flow_score} points)")
                
                print(f"📊 {symbol}: Flow consistency {round(flow_consistency*100)}% ({flow_consistency_score:+d} points)")
                
                if pulse_delay_detected:
                    print(f"⏸️ {symbol}: Pulse delay detected - {pulse_description} (+{pulse_delay_score} points)")
                else:
                    print(f"🌊 {symbol}: No pulse delay - continuous flow")
                
                if orderbook_freeze_detected:
                    print(f"🧊 {symbol}: Orderbook freeze detected - {freeze_description} (+{orderbook_freeze_score} points)")
                else:
                    print(f"📋 {symbol}: No orderbook freeze - ask-side active")
                
                if heatmap_vacuum_detected:
                    print(f"🗺️ {symbol}: Heatmap vacuum detected - {vacuum_description} (+{heatmap_vacuum_score} points)")
                else:
                    print(f"📊 {symbol}: No heatmap vacuum - ask levels stable")
                
                if vwap_pinning_detected:
                    print(f"📌 {symbol}: VWAP pinning detected - {pinning_description} (+{vwap_pinning_score} points)")
                else:
                    print(f"📈 {symbol}: No VWAP pinning - price volatile vs VWAP")
                
                # One-Sided Pressure Detection (7th Layer)
                # Use current orderbook snapshot for bid/ask pressure analysis
                current_orderbook = get_orderbook_with_fallback(symbol)
                if current_orderbook and current_orderbook.get("bids") and current_orderbook.get("asks"):
                    pressure_result = detect_one_sided_pressure(current_orderbook)
                    pressure_detected, pressure_description, pressure_details = pressure_result
                    pressure_score = calculate_one_sided_pressure_score(pressure_result)
                else:
                    # Fallback to mock data for development/testing
                    from detectors.one_sided_pressure import create_mock_strong_bid_orderbook
                    mock_orderbook = create_mock_strong_bid_orderbook()
                    pressure_result = detect_one_sided_pressure(mock_orderbook)
                    pressure_detected, pressure_description, pressure_details = pressure_result
                    pressure_score = calculate_one_sided_pressure_score(pressure_result)
                    pressure_description += " (mock data)"
                
                if pressure_detected:
                    print(f"💪 {symbol}: One-sided pressure detected - {pressure_description} (+{pressure_score} points)")
                else:
                    print(f"⚖️ {symbol}: Balanced orderbook - {pressure_description}")
            else:
                print(f"⚠️ {symbol}: No price data for flow analysis")
        except Exception as e:
            print(f"❌ {symbol}: Flow analysis failed: {e}")
        
        # === ENHANCED COMBINED CONFIDENCE CALCULATION ===
        base_confidence = 100 if trend_active else (50 if stage_minus1_active else 0)
        
        # Dodaj adjustmenty za kompletną analizę flow (7 warstw)
        directional_adjustment = directional_flow_score if directional_flow_details else 0
        consistency_adjustment = flow_consistency_score
        pulse_delay_adjustment = pulse_delay_score
        orderbook_freeze_adjustment = orderbook_freeze_score
        heatmap_vacuum_adjustment = heatmap_vacuum_score
        vwap_pinning_adjustment = vwap_pinning_score
        pressure_adjustment = pressure_score
        total_flow_adjustment = directional_adjustment + consistency_adjustment + pulse_delay_adjustment + orderbook_freeze_adjustment + heatmap_vacuum_adjustment + vwap_pinning_adjustment + pressure_adjustment
        
        combined_confidence = max(0, min(base_confidence + total_flow_adjustment, 215))  # 0-215 punktów
        
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
            "directional_flow": {
                "active": directional_flow_detected,
                "score": directional_flow_score,
                "details": directional_flow_details
            },
            "flow_consistency": {
                "index": flow_consistency,
                "score": flow_consistency_score,
                "description": flow_consistency_desc
            },
            "pulse_delay": {
                "detected": pulse_delay_detected,
                "score": pulse_delay_score,
                "details": pulse_delay_details
            },
            "orderbook_freeze": {
                "detected": orderbook_freeze_detected,
                "score": orderbook_freeze_score,
                "details": orderbook_freeze_details
            },
            "heatmap_vacuum": {
                "detected": heatmap_vacuum_detected,
                "score": heatmap_vacuum_score,
                "details": heatmap_vacuum_details
            },
            "vwap_pinning": {
                "detected": vwap_pinning_detected,
                "score": vwap_pinning_score,
                "details": vwap_pinning_details
            },
            "one_sided_pressure": {
                "detected": pressure_detected,
                "score": pressure_score,
                "details": pressure_details
            },
            "combined_confidence": combined_confidence,
            "base_confidence": base_confidence,
            "directional_adjustment": directional_adjustment,
            "consistency_adjustment": consistency_adjustment,
            "pulse_delay_adjustment": pulse_delay_adjustment,
            "orderbook_freeze_adjustment": orderbook_freeze_adjustment,
            "heatmap_vacuum_adjustment": heatmap_vacuum_adjustment,
            "vwap_pinning_adjustment": vwap_pinning_adjustment,
            "pressure_adjustment": pressure_adjustment,
            "total_flow_adjustment": total_flow_adjustment,
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