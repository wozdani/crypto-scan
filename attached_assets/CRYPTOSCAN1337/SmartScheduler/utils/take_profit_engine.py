def forecast_take_profit_levels(signals: dict) -> dict:
    """
    Enhanced TP forecasting with RSI adjustments, whale activity, and smart trailing logic.
    Returns TP levels based on PPWCS, RSI, inflow, whale activity and market structure.
    """
    try:
        # Extract enhanced signal data
        ppwcs = signals.get("ppwcs_score", 0)
        rsi = signals.get("rsi", 50) 
        whale_activity = signals.get("whale_activity", False)
        dex_inflow = signals.get("dex_inflow", 0)
        compressed = signals.get("compressed", False)
        stage1g_active = signals.get("stage1g_active", False)
        pure_accumulation = signals.get("pure_accumulation", False)
        
        # Base TP levels (as decimals for calculation)
        tp1 = 0.06  # 6%
        tp2 = 0.14  # 14%
        tp3 = 0.27  # 27%
        trailing_tp = "adaptive"
        
        # Strong signal boost - exceptional conditions
        if ppwcs >= 85 and compressed and stage1g_active:
            tp1 += 0.02  # +2%
            tp2 += 0.04  # +4%
            tp3 += 0.08  # +8%
            trailing_tp = "aggressive trail"
        
        # RSI momentum exhaustion adjustment
        if rsi > 70:
            tp3 -= 0.04  # Reduce TP3 when momentum exhausted
            trailing_tp = "early lock"
        
        # Whale activity + high inflow boost
        if whale_activity and dex_inflow > 100000:
            tp3 += 0.05  # +5% for whale confirmation
            trailing_tp = "trail after TP2"
        
        # Pure accumulation bonus
        if pure_accumulation:
            tp1 += 0.01  # +1%
            tp2 += 0.02  # +2%
        
        # PPWCS score scaling
        if ppwcs > 85:
            multiplier = 1.4  # Exceptional signal
        elif ppwcs > 75:
            multiplier = 1.2  # High signal quality
        elif ppwcs < 60:
            multiplier = 0.8  # Lower confidence
        else:
            multiplier = 1.0
        
        # Apply multiplier to final TP levels
        tp1 *= multiplier
        tp2 *= multiplier  
        tp3 *= multiplier
        
        # Convert to percentages and round
        final_tp1 = round(tp1 * 100, 1)
        final_tp2 = round(tp2 * 100, 1) 
        final_tp3 = round(tp3 * 100, 1)
        
        # Ensure minimum levels (safety bounds)
        final_tp1 = max(final_tp1, 4.0)
        final_tp2 = max(final_tp2, 10.0)
        final_tp3 = max(final_tp3, 20.0)
        
        # Ensure maximum levels (realistic bounds)
        final_tp1 = min(final_tp1, 12.0)
        final_tp2 = min(final_tp2, 25.0)
        final_tp3 = min(final_tp3, 45.0)
        
        # Calculate trailing TP as percentage value
        if isinstance(trailing_tp, str):
            trailing_tp_value = round(final_tp2 * 0.66, 1)  # 66% of TP2
        else:
            trailing_tp_value = trailing_tp
        
        return {
            "TP1": final_tp1,
            "TP2": final_tp2,
            "TP3": final_tp3,
            "TrailingTP": trailing_tp_value,
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