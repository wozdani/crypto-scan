def forecast_take_profit_levels(signals: dict) -> dict:
    """
    Estimates TP1 / TP2 / TP3 and trailing TP levels (%) based on signal quality.
    Dynamic calculation considering multiple market factors.
    """
    try:
        # Base TP levels (%)
        base_tp1, base_tp2, base_tp3 = 6, 15, 30
        
        # Extract signal data
        ppwcs = signals.get("ppwcs_score", 0)
        rsi = signals.get("rsi", 50)
        delta_flow = signals.get("delta_flow_strength", 0)
        heatmap_score = signals.get("bookmap_heatmap_score", 0)
        orderbook_imbalance = signals.get("orderbook_imbalance", 0)
        breakout_type = signals.get("type_of_breakout", "unknown")
        time_tag = signals.get("time_tag", "other")
        
        # Start with base multiplier
        multiplier = 1.0
        
        # --- PPWCS Score Impact ---
        if ppwcs > 85:
            multiplier += 0.4  # Exceptional signal quality
        elif ppwcs > 75:
            multiplier += 0.2  # High signal quality
        elif ppwcs < 60:
            multiplier -= 0.2  # Lower confidence
        
        # --- RSI Momentum Impact ---
        if rsi > 68:
            multiplier += 0.15  # Strong momentum
        elif rsi < 50:
            multiplier -= 0.1   # Weak momentum
        
        # --- Flow and Market Depth Impact ---
        if delta_flow > 1.5:
            multiplier += 0.1   # Strong buying pressure
        if heatmap_score > 1.2:
            multiplier += 0.1   # Good absorption/lack of supply
        if orderbook_imbalance > 1.5:
            multiplier += 0.1   # Strong bid/ask imbalance
        
        # --- Breakout Type Impact ---
        if breakout_type == "squeeze_breakout":
            multiplier += 0.1   # Compression breakouts tend to be stronger
        elif breakout_type == "reject_impulse":
            multiplier -= 0.05  # Rejection patterns may have less follow-through
        
        # --- Timing Impact ---
        if time_tag in ["before_15", "after_15"]:
            multiplier += 0.05  # Better timing windows
        
        # --- Calculate TP levels ---
        tp1 = round(base_tp1 * multiplier, 1)
        tp2 = round(base_tp2 * multiplier, 1)
        tp3 = round(base_tp3 * multiplier, 1)
        
        # Trailing TP: Activated after TP2, set at 66% of TP2 level
        trailing_tp = round(tp2 * 0.66, 1)
        
        # Ensure minimum levels (safety bounds)
        tp1 = max(tp1, 3.0)
        tp2 = max(tp2, 8.0)
        tp3 = max(tp3, 15.0)
        trailing_tp = max(trailing_tp, 5.0)
        
        # Ensure maximum levels (realistic bounds)
        tp1 = min(tp1, 15.0)
        tp2 = min(tp2, 40.0)
        tp3 = min(tp3, 80.0)
        trailing_tp = min(trailing_tp, 30.0)
        
        return {
            "TP1": tp1,
            "TP2": tp2,
            "TP3": tp3,
            "TrailingTP": trailing_tp,
            "multiplier": round(multiplier, 2),
            "confidence": get_forecast_confidence(ppwcs, multiplier)
        }
        
    except Exception as e:
        print(f"❌ Error forecasting TP levels: {e}")
        # Return conservative defaults on error
        return {
            "TP1": 6.0,
            "TP2": 15.0,
            "TP3": 30.0,
            "TrailingTP": 10.0,
            "multiplier": 1.0,
            "confidence": "low"
        }

def get_forecast_confidence(ppwcs_score: float, multiplier: float) -> str:
    """
    Determine forecast confidence level based on signal quality
    """
    if ppwcs_score >= 85 and multiplier >= 1.3:
        return "very_high"
    elif ppwcs_score >= 75 and multiplier >= 1.15:
        return "high"
    elif ppwcs_score >= 60 and multiplier >= 1.0:
        return "medium"
    else:
        return "low"

def calculate_risk_reward_ratio(signals: dict, tp_levels: dict) -> dict:
    """
    Calculate risk/reward ratios for each TP level
    Assumes 3-5% stop loss based on signal quality
    """
    try:
        ppwcs = signals.get("ppwcs_score", 0)
        
        # Dynamic stop loss based on signal quality
        if ppwcs >= 80:
            stop_loss = 3.0  # Tighter SL for high quality signals
        elif ppwcs >= 60:
            stop_loss = 4.0  # Medium SL
        else:
            stop_loss = 5.0  # Wider SL for lower quality
        
        # Calculate R:R ratios
        rr_tp1 = round(tp_levels["TP1"] / stop_loss, 2)
        rr_tp2 = round(tp_levels["TP2"] / stop_loss, 2)
        rr_tp3 = round(tp_levels["TP3"] / stop_loss, 2)
        
        return {
            "stop_loss": stop_loss,
            "RR_TP1": rr_tp1,
            "RR_TP2": rr_tp2,
            "RR_TP3": rr_tp3,
            "recommended_position_size": get_position_size_recommendation(ppwcs)
        }
        
    except Exception as e:
        print(f"❌ Error calculating risk/reward: {e}")
        return {
            "stop_loss": 4.0,
            "RR_TP1": 1.5,
            "RR_TP2": 3.75,
            "RR_TP3": 7.5,
            "recommended_position_size": "small"
        }

def get_position_size_recommendation(ppwcs_score: float) -> str:
    """
    Recommend position size based on signal confidence
    """
    if ppwcs_score >= 85:
        return "large"      # 3-5% of portfolio
    elif ppwcs_score >= 75:
        return "medium"     # 2-3% of portfolio
    elif ppwcs_score >= 60:
        return "small"      # 1-2% of portfolio
    else:
        return "micro"      # 0.5-1% of portfolio