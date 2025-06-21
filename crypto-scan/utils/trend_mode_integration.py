"""
Trend Mode Integration Helper - funkcje pomocnicze dla integracji z głównym systemem
Obsługuje zarówno legacy Trend Mode jak i nowy Trend Mode 2.0
"""

from detectors.pullback_flow_pattern import pullback_flow_pattern
from detectors.trend_mode_sr import detect_sr_trend_mode
from utils.trend_mode_alert_engine import trend_alert_engine
from utils.trend_mode_score_engine import trend_mode_2_engine
import time


def collect_real_modules_data(symbol, candle_data, orderbook_data=None):
    """
    Zbiera rzeczywiste dane z modułów dla enhanced trend scoring
    
    Args:
        symbol: Trading symbol
        candle_data: Lista świec OHLCV
        orderbook_data: Opcjonalne dane orderbook
        
    Returns:
        dict: Dane z różnych modułów
    """
    modules_data = {}
    
    try:
        # 1. Pullback Flow Pattern
        if candle_data and len(candle_data) >= 24:
            # Przygotuj dane dla pullback detection
            candles_15m = [str(candle[4]) for candle in candle_data[-96:]] if len(candle_data) >= 96 else [str(candle[4]) for candle in candle_data]
            candles_5m = [str(candle[4]) for candle in candle_data[-12:]] if len(candle_data) >= 12 else [str(candle[4]) for candle in candle_data]
            
            # Mock orderbook jeśli nie ma prawdziwego
            if not orderbook_data:
                orderbook_data = {
                    "ask_volumes": [1000, 950, 900],
                    "bid_volumes": [800, 900, 1000]
                }
            
            pullback_result = pullback_flow_pattern(symbol, candles_15m, candles_5m, orderbook_data)
            modules_data["pullback_flow"] = pullback_result
        
        # 2. Flow Consistency (oblicz na podstawie świec)
        if candle_data and len(candle_data) >= 20:
            prices = [float(candle[4]) for candle in candle_data[-20:]]
            
            # Prosta analiza consistency
            up_moves = 0
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    up_moves += 1
            
            flow_consistency = (up_moves / (len(prices) - 1)) * 100 if len(prices) > 1 else 0
            modules_data["flow_consistency"] = flow_consistency
        
        # 3. Orderbook Analysis
        if orderbook_data:
            ask_volumes = orderbook_data.get("ask_volumes", [])
            bid_volumes = orderbook_data.get("bid_volumes", [])
            
            if ask_volumes and bid_volumes:
                latest_bid = bid_volumes[-1] if bid_volumes else 0
                latest_ask = ask_volumes[-1] if ask_volumes else 1
                bid_ask_ratio = latest_bid / latest_ask if latest_ask > 0 else 0
                
                modules_data["orderbook"] = {
                    "bid_ask_ratio": bid_ask_ratio,
                    "bid_dominance": bid_ask_ratio >= 1.2
                }
        
        # 4. One-Sided Pressure (na podstawie volume i price action)
        if candle_data and len(candle_data) >= 5:
            recent_volumes = [float(candle[5]) for candle in candle_data[-5:]]
            recent_closes = [float(candle[4]) for candle in candle_data[-5:]]
            
            # Sprawdź czy volume rośnie z ceną
            volume_trend = sum(1 for i in range(1, len(recent_volumes)) if recent_volumes[i] > recent_volumes[i-1])
            price_trend = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] > recent_closes[i-1])
            
            one_sided_pressure = volume_trend >= 3 and price_trend >= 3  # 3/4 ostatnich okresów
            modules_data["one_sided_pressure"] = one_sided_pressure
        
        # 5. Heatmap Vacuum (wykryj niską resistance)
        if candle_data and len(candle_data) >= 10:
            highs = [float(candle[2]) for candle in candle_data[-10:]]
            current_high = highs[-1]
            
            # Sprawdź czy current high jest blisko poprzednich resistance levels
            resistance_levels = [h for h in highs[:-1] if h > current_high * 1.002]  # 0.2% above
            heatmap_vacuum = len(resistance_levels) < 2  # Mało resistance levels
            modules_data["heatmap_vacuum"] = heatmap_vacuum
        
    except Exception as e:
        print(f"⚠️ Error collecting modules data for {symbol}: {e}")
    
    return modules_data


def process_trend_mode_2(symbol, market_data, orderbook_data=None):
    """
    Przetwórz symbol przez nowy Trend Mode 2.0 system
    
    Args:
        symbol: Trading symbol
        market_data: Dane rynkowe (candles, prices, volumes)
        orderbook_data: Opcjonalne dane orderbook
        
    Returns:
        dict: Kompletny wynik Trend Mode 2.0
    """
    try:
        # 1. Analizuj przez Trend Mode 2.0 engine
        trend_result = trend_mode_2_engine.process_symbol_trend_mode_2(
            symbol, market_data, orderbook_data
        )
        
        # 2. Przetwórz przez Alert Engine dla trailing logic
        alert_data = trend_alert_engine.process_trend_mode_2_alert(symbol, trend_result)
        
        # 3. Dodaj alert data do wyniku
        trend_result['alert_generated'] = alert_data is not None
        trend_result['alert_data'] = alert_data
        
        return trend_result
        
    except Exception as e:
        print(f"⚠️ Error in Trend Mode 2.0 for {symbol}: {e}")
        return {
            "symbol": symbol,
            "trend_mode_2_score": 0,
            "alert_level": 0,
            "trend_active": False,
            "alert_generated": False,
            "error": str(e)
        }

def process_enhanced_trend_mode(symbol, candle_data, base_trend_result=None):
    """
    Przetwórz symbol przez Enhanced Trend Mode z Alert Engine
    
    Args:
        symbol: Trading symbol
        candle_data: Lista świec OHLCV
        base_trend_result: Opcjonalny wynik z detect_sr_trend_mode
        
    Returns:
        dict: Kompletny wynik enhanced trend mode
    """
    try:
        # 1. Podstawowy trend mode jeśli nie podano
        if not base_trend_result:
            base_trend_result = detect_sr_trend_mode(symbol)
        
        base_score = base_trend_result.get('trend_score', 0)
        
        # 2. Zbierz dane z modułów
        modules_data = collect_real_modules_data(symbol, candle_data)
        
        # 3. Przetwórz przez Alert Engine
        alert_data = trend_alert_engine.process_trend_mode_alert(
            symbol, base_score, modules_data
        )
        
        # 4. Oblicz enhanced score
        enhanced_score, score_breakdown = trend_alert_engine.compute_enhanced_trend_score(
            symbol, base_score, modules_data
        )
        
        # 5. Przygotuj wynik
        result = {
            "symbol": symbol,
            "base_score": base_score,
            "enhanced_score": enhanced_score,
            "score_breakdown": score_breakdown,
            "trend_active": enhanced_score >= 70 or alert_data is not None,
            "alert_generated": alert_data is not None,
            "alert_data": alert_data,
            "modules_data": modules_data,
            "base_trend_result": base_trend_result,
            "timestamp": int(time.time())
        }
        
        return result
        
    except Exception as e:
        print(f"⚠️ Error in enhanced trend mode for {symbol}: {e}")
        return {
            "symbol": symbol,
            "base_score": 0,
            "enhanced_score": 0,
            "trend_active": False,
            "alert_generated": False,
            "error": str(e)
        }


def get_trend_mode_2_summary(trend_mode_2_result):
    """
    Stwórz czytelne podsumowanie Trend Mode 2.0
    
    Args:
        trend_mode_2_result: Wynik z process_trend_mode_2
        
    Returns:
        list: Lista stringów z podsumowaniem
    """
    summary = []
    
    score = trend_mode_2_result.get("trend_mode_2_score", 0)
    alert_level = trend_mode_2_result.get("alert_level", 0)
    score_breakdown = trend_mode_2_result.get("score_breakdown", {})
    
    # Podstawowe info
    level_names = {0: "No Alert", 1: "Watchlist", 2: "Active Entry", 3: "Confirmed"}
    level_name = level_names.get(alert_level, "Unknown")
    summary.append(f"Trend Mode 2.0: {score}/100 (Level {alert_level}: {level_name})")
    
    # Aktywne detektory
    core_active = score_breakdown.get('active_core', [])
    helper_active = score_breakdown.get('active_helper', [])
    negative_active = score_breakdown.get('active_negative', [])
    
    if core_active:
        summary.append(f"🔵 Core: {', '.join(core_active)} (+{score_breakdown.get('core_points', 0)})")
    
    if helper_active:
        summary.append(f"🟢 Helper: {', '.join(helper_active)} (+{score_breakdown.get('helper_points', 0)})")
    
    if negative_active:
        summary.append(f"🔴 Warning: {', '.join(negative_active)} ({score_breakdown.get('negative_points', 0)})")
    
    # Alert info
    if trend_mode_2_result.get("alert_generated"):
        alert_data = trend_mode_2_result.get("alert_data", {})
        summary.append(f"🚨 ALERT: {alert_data.get('reason', 'unknown')}")
        summary.append(f"Priority: {alert_data.get('priority', 'unknown')}")
    
    return summary

def get_trend_mode_summary(enhanced_result):
    """
    Stwórz czytelne podsumowanie Enhanced Trend Mode
    
    Args:
        enhanced_result: Wynik z process_enhanced_trend_mode
        
    Returns:
        list: Lista stringów z podsumowaniem
    """
    summary = []
    
    enhanced_score = enhanced_result.get("enhanced_score", 0)
    base_score = enhanced_result.get("base_score", 0)
    score_breakdown = enhanced_result.get("score_breakdown", {})
    active_modules = score_breakdown.get("active_modules", [])
    
    # Podstawowe info
    summary.append(f"Enhanced Score: {enhanced_score}/100 (base: {base_score})")
    summary.append(f"Active Modules: {len(active_modules)}")
    
    # Alert info
    if enhanced_result.get("alert_generated"):
        alert_data = enhanced_result.get("alert_data", {})
        summary.append(f"🚨 ALERT: {alert_data.get('reason', 'unknown')}")
        summary.append(f"Priority: {alert_data.get('priority', 'unknown')}")
    
    # Top modules
    if active_modules:
        top_modules = active_modules[:3]  # Top 3
        summary.append(f"Top modules: {', '.join(top_modules)}")
    
    # Pullback info
    modules_data = enhanced_result.get("modules_data", {})
    pullback_data = modules_data.get("pullback_flow", {})
    if pullback_data.get("pullback_trigger"):
        confidence = pullback_data.get("confidence_score", 0)
        summary.append(f"Pullback trigger: {confidence}% confidence")
    
    return summary