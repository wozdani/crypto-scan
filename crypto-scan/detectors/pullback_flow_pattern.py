"""
Pullback Flow Pattern Detector - wykrycie końca korekty i potencjalnego wejścia

Analizuje:
- Trend spadkowy na 15M (ostatnie 24h)
- Sygnały odwrócenia na 5M (malejący ask pressure, rosnący bid)
- Potencjalna próżnia lub freeze w orderbookach
"""

import time


def pullback_flow_pattern(symbol, candles_15m, candles_5m, orderbook_data):
    """
    Pullback Flow Pattern Detector - wykrycie końca korekty i potencjalnego wejścia
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        candles_15m: Lista świec 15M (ostatnie 96 = 24h)
        candles_5m: Lista świec 5M (ostatnie 12 = 1h)
        orderbook_data: Dane orderbooka z trzema punktami czasowymi
        
    Returns:
        dict: Wynik analizy pullback pattern
    """
    try:
        # Sprawdź czy mamy wystarczające dane
        if not candles_15m or len(candles_15m) < 24:
            return {"pullback_trigger": False, "error": "insufficient_15m_data"}
        
        if not candles_5m or len(candles_5m) < 12:
            return {"pullback_trigger": False, "error": "insufficient_5m_data"}
            
        if not orderbook_data:
            return {"pullback_trigger": False, "error": "insufficient_orderbook_data"}
        
        # === 1. ANALIZA TRENDU SPADKOWEGO NA 15M (24h) ===
        # Używamy dostępnych świec 15M
        available_candles = min(len(candles_15m), 96)
        recent_candles = candles_15m[-available_candles:]
        
        # Policz czerwone świece w dostępnym okresie
        red_candles = 0
        total_change = 0
        
        if len(recent_candles) >= 2:
            start_price = float(recent_candles[0])
            current_price = float(recent_candles[-1])
            
            for i in range(1, len(recent_candles)):
                prev_price = float(recent_candles[i-1])
                curr_price = float(recent_candles[i])
                if curr_price < prev_price:
                    red_candles += 1
            
            # Oblicz procentowy spadek
            decline_pct = (start_price - current_price) / start_price * 100 if start_price > 0 else 0
            red_ratio = red_candles / (len(recent_candles) - 1) if len(recent_candles) > 1 else 0
        else:
            decline_pct = 0
            red_ratio = 0
        
        # Warunki trendu spadkowego
        is_downtrend_15m = (red_ratio >= 0.55 and decline_pct >= 2.0) or decline_pct >= 5.0
        
        # === 2. ANALIZA SYGNAŁÓW ODWRÓCENIA NA 5M ===
        green_candles_1h = 0
        if len(candles_5m) >= 2:
            for i in range(1, len(candles_5m)):
                prev_price = float(candles_5m[i-1])
                curr_price = float(candles_5m[i])
                if curr_price > prev_price:
                    green_candles_1h += 1
            
            green_ratio_1h = green_candles_1h / (len(candles_5m) - 1)
        else:
            green_ratio_1h = 0
        
        # === 3. ANALIZA ORDERBOOK PRESSURE ===
        ask_volumes = orderbook_data.get("ask_volumes", [])
        bid_volumes = orderbook_data.get("bid_volumes", [])
        
        if len(ask_volumes) >= 3 and len(bid_volumes) >= 3:
            # Sprawdź trend ask pressure (powinien maleć)
            ask_trend_declining = (ask_volumes[0] > ask_volumes[1] > ask_volumes[2])
            
            # Sprawdź trend bid pressure (powinien rosnąć)
            bid_trend_rising = (bid_volumes[0] < bid_volumes[1] < bid_volumes[2])
            
            # Oblicz bid/ask ratio dla siły
            current_bid_ask_ratio = bid_volumes[2] / ask_volumes[2] if ask_volumes[2] > 0 else 0
            
            # Wykryj potencjalną próżnię (bardzo niski ask pressure)
            avg_ask = sum(ask_volumes) / len(ask_volumes)
            ask_vacuum = ask_volumes[2] < avg_ask * 0.3
            
            # Wykryj orderbook freeze (bardzo małe zmiany)
            ask_change = abs(ask_volumes[2] - ask_volumes[1]) / ask_volumes[1] if ask_volumes[1] > 0 else 0
            bid_change = abs(bid_volumes[2] - bid_volumes[1]) / bid_volumes[1] if bid_volumes[1] > 0 else 0
            orderbook_freeze = ask_change < 0.1 and bid_change < 0.1
        else:
            ask_trend_declining = False
            bid_trend_rising = False
            current_bid_ask_ratio = 0
            ask_vacuum = False
            orderbook_freeze = False
        
        # === 4. SCORING I TRIGGERS ===
        confidence_score = 0
        entry_triggers = []
        
        # Podstawowe warunki (40 punktów)
        if is_downtrend_15m:
            confidence_score += 25
            entry_triggers.append("downtrend_15m_confirmed")
            
        if green_ratio_1h >= 0.6:  # 60%+ zielonych świec w ostatniej godzinie
            confidence_score += 15
            entry_triggers.append("reversal_signals_5m")
        
        # Orderbook conditions (40 punktów)
        if ask_trend_declining:
            confidence_score += 15
            entry_triggers.append("ask_pressure_fading")
            
        if bid_trend_rising:
            confidence_score += 15
            entry_triggers.append("bid_absorption")
            
        if current_bid_ask_ratio >= 1.2:  # Bid silniejszy niż ask
            confidence_score += 10
            entry_triggers.append("bid_dominance")
        
        # Bonus conditions (20 punktów)
        if ask_vacuum:
            confidence_score += 10
            entry_triggers.append("ask_vacuum_detected")
            
        if orderbook_freeze:
            confidence_score += 10
            entry_triggers.append("orderbook_freeze")
        
        # === 5. FINALNE DECISION ===
        # Wymagamy min. 50 punktów i przynajmniej 3 triggery
        pullback_trigger = confidence_score >= 50 and len(entry_triggers) >= 3
        
        return {
            "pullback_trigger": pullback_trigger,
            "confidence_score": confidence_score,
            "entry_trigger": " + ".join(entry_triggers) if entry_triggers else "no_triggers",
            "downtrend_15m": is_downtrend_15m,
            "decline_pct": round(decline_pct, 2),
            "red_ratio": round(red_ratio, 3),
            "green_ratio_1h": round(green_ratio_1h, 3),
            "bid_ask_ratio": round(current_bid_ask_ratio, 3),
            "ask_trend_declining": ask_trend_declining,
            "bid_trend_rising": bid_trend_rising,
            "vacuum_detected": ask_vacuum,
            "freeze_detected": orderbook_freeze,
            "analysis_timestamp": int(time.time())
        }
        
    except Exception as e:
        return {
            "pullback_trigger": False,
            "error": f"analysis_failed: {str(e)}",
            "confidence_score": 0
        }


def calculate_pullback_flow_score(pullback_result):
    """
    Oblicz score dla pullback flow pattern w systemie PPWCS
    
    Args:
        pullback_result: Wynik z pullback_flow_pattern()
        
    Returns:
        int: Score 0-20 punktów dla PPWCS
    """
    if not pullback_result or not pullback_result.get("pullback_trigger"):
        return 0
    
    confidence = pullback_result.get("confidence_score", 0)
    
    # Mapowanie confidence score (0-100) na PPWCS points (0-20)
    if confidence >= 80:
        return 20  # Bardzo silny sygnał
    elif confidence >= 70:
        return 15  # Silny sygnał
    elif confidence >= 60:
        return 12  # Umiarkowany sygnał
    elif confidence >= 50:
        return 8   # Słaby sygnał
    else:
        return 0   # Brak sygnału


def detect_pullback_flow(symbol, candles_15m=None, candles_5m=None, orderbook_data=None):
    """
    Wrapper function kompatybilny z istniejącym systemem detektorów
    
    Returns:
        tuple: (is_detected, confidence_score, details)
    """
    try:
        result = pullback_flow_pattern(symbol, candles_15m or [], candles_5m or [], orderbook_data or {})
        
        is_detected = result.get("pullback_trigger", False)
        confidence_score = result.get("confidence_score", 0)
        
        details = {
            "entry_trigger": result.get("entry_trigger", ""),
            "downtrend_15m": result.get("downtrend_15m", False),
            "decline_pct": result.get("decline_pct", 0),
            "bid_ask_ratio": result.get("bid_ask_ratio", 0)
        }
        
        return (is_detected, confidence_score, details)
        
    except Exception as e:
        return (False, 0, {"error": str(e)})