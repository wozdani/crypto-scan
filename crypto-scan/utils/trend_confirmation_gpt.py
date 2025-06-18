"""
Trend Confirmation GPT - AI-powered Trend Quality Assessment
Wysłanie danych do GPT w celu oceny jakości trendu (opcjonalnie: tylko dla score ≥ 40)
"""

import os
import openai
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def format_trend_data_for_gpt(symbol, trend_score, trend_signals, candle_data, tp_levels=None):
    """Format trend data for GPT analysis"""
    
    # Extract key metrics from recent candles
    if len(candle_data) >= 5:
        recent_candles = candle_data[-5:]
        
        # Price action summary
        opens = [c[1] for c in recent_candles]
        highs = [c[2] for c in recent_candles]
        lows = [c[3] for c in recent_candles]
        closes = [c[4] for c in recent_candles]
        volumes = [c[5] for c in recent_candles]
        
        price_change_5c = ((closes[-1] - closes[0]) / closes[0]) * 100
        volume_trend = "rising" if volumes[-1] > volumes[0] else "falling"
        
        # Calculate average candle body size
        body_sizes = [abs(c[4] - c[1]) / c[1] * 100 for c in recent_candles]
        avg_body_size = sum(body_sizes) / len(body_sizes)
        
        candle_summary = {
            "price_change_5_candles": round(price_change_5c, 2),
            "volume_trend": volume_trend,
            "avg_body_size_pct": round(avg_body_size, 2),
            "current_price": closes[-1],
            "highest_in_5": max(highs),
            "lowest_in_5": min(lows)
        }
    else:
        candle_summary = {"error": "insufficient_data"}
    
    # Format trend signals
    signals_formatted = []
    for signal in trend_signals:
        signals_formatted.append(signal.replace("_", " ").title())
    
    # Prepare structured data
    trend_data = {
        "symbol": symbol,
        "trend_score": trend_score,
        "max_score": 50,
        "active_signals": signals_formatted,
        "signal_count": len(trend_signals),
        "candle_analysis": candle_summary,
        "tp_levels": tp_levels or {"TP1": "N/A", "TP2": "N/A", "TP3": "N/A"}
    }
    
    return trend_data


def analyze_trend_with_gpt(symbol, trend_score, trend_signals, candle_data, tp_levels=None):
    """
    Send trend data to GPT for quality assessment
    
    Args:
        symbol: token symbol
        trend_score: trend score (0-50)
        trend_signals: list of active trend signals
        candle_data: OHLCV candle data
        tp_levels: take profit levels from trailing TP engine
    
    Returns:
        dict: {
            "gpt_assessment": str,
            "quality_score": int (0-100),
            "risk_level": str,
            "continuation_probability": int (0-100),
            "recommendations": list,
            "confidence": str
        }
    """
    print(f"[TREND GPT] Analyzing trend quality for {symbol} (score: {trend_score}/50)")
    
    if not OPENAI_API_KEY:
        return {
            "gpt_assessment": "GPT analysis unavailable - API key not configured",
            "quality_score": 0,
            "risk_level": "unknown",
            "continuation_probability": 0,
            "recommendations": [],
            "confidence": "low"
        }
    
    # Only analyze trends with score >= 40 (configurable threshold)
    GPT_ANALYSIS_THRESHOLD = 40
    
    if trend_score < GPT_ANALYSIS_THRESHOLD:
        return {
            "gpt_assessment": f"Trend score {trend_score} below GPT analysis threshold ({GPT_ANALYSIS_THRESHOLD})",
            "quality_score": trend_score * 2,  # Convert to 0-100 scale
            "risk_level": "moderate",
            "continuation_probability": trend_score,
            "recommendations": ["Monitor for stronger signals"],
            "confidence": "low"
        }
    
    try:
        # Format data for GPT
        trend_data = format_trend_data_for_gpt(symbol, trend_score, trend_signals, candle_data, tp_levels)
        
        # Create comprehensive prompt
        prompt = f"""Analyze this cryptocurrency trend continuation setup:

Symbol: {symbol}
Trend Score: {trend_score}/50 points
Active Signals: {', '.join(trend_signals)}

Price Action (Last 5 Candles):
- Price Change: {trend_data['candle_analysis'].get('price_change_5_candles', 'N/A')}%
- Volume Trend: {trend_data['candle_analysis'].get('volume_trend', 'N/A')}
- Average Candle Body: {trend_data['candle_analysis'].get('avg_body_size_pct', 'N/A')}%

Take Profit Levels:
- TP1: {tp_levels.get('TP1', 'N/A')}%
- TP2: {tp_levels.get('TP2', 'N/A')}%
- TP3: {tp_levels.get('TP3', 'N/A')}%

Evaluate this trend using professional trading principles (Minervini, Raschke, Grimes methodology):

1. Quality Score (0-100): Rate the overall trend quality
2. Risk Level (low/moderate/high): Assess potential downside risk
3. Continuation Probability (0-100%): Likelihood of sustained upward movement
4. Key Recommendations: 2-3 specific trading recommendations

Respond in JSON format:
{{
  "quality_score": number,
  "risk_level": "low|moderate|high",
  "continuation_probability": number,
  "recommendations": ["rec1", "rec2", "rec3"],
  "assessment": "detailed analysis in Polish (2-3 sentences)"
}}"""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional cryptocurrency trend analyst specializing in continuation patterns. Provide precise, actionable analysis in Polish."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            timeout=20
        )
        
        # Parse GPT response
        gpt_result = json.loads(response.choices[0].message.content)
        
        # Determine confidence level
        quality_score = gpt_result.get("quality_score", 0)
        
        if quality_score >= 80:
            confidence = "high"
        elif quality_score >= 60:
            confidence = "medium"
        else:
            confidence = "low"
        
        result = {
            "gpt_assessment": gpt_result.get("assessment", "No assessment provided"),
            "quality_score": quality_score,
            "risk_level": gpt_result.get("risk_level", "moderate"),
            "continuation_probability": gpt_result.get("continuation_probability", 50),
            "recommendations": gpt_result.get("recommendations", []),
            "confidence": confidence
        }
        
        print(f"[TREND GPT] {symbol}: Quality={quality_score}/100, Risk={result['risk_level']}, Continuation={result['continuation_probability']}%")
        
        return result
        
    except Exception as e:
        print(f"[TREND GPT] Error analyzing {symbol}: {e}")
        
        return {
            "gpt_assessment": f"GPT analysis error: {str(e)}",
            "quality_score": trend_score * 2,  # Fallback scoring
            "risk_level": "moderate",
            "continuation_probability": trend_score,
            "recommendations": ["Manual review recommended"],
            "confidence": "low"
        }


def save_gpt_trend_analysis(symbol, gpt_result, trend_score):
    """Save GPT trend analysis to file"""
    try:
        analysis_data = {
            "symbol": symbol,
            "trend_score": trend_score,
            "gpt_analysis": gpt_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "trend_confirmation"
        }
        
        # Save to trend analyses file
        analyses_file = os.path.join("data", "gpt_trend_analyses.json")
        
        analyses = []
        if os.path.exists(analyses_file):
            with open(analyses_file, 'r') as f:
                analyses = json.load(f)
        
        analyses.append(analysis_data)
        
        # Keep only last 50 analyses
        analyses = analyses[-50:]
        
        with open(analyses_file, 'w') as f:
            json.dump(analyses, f, indent=2)
        
        print(f"[TREND GPT] Analysis saved for {symbol}")
        
    except Exception as e:
        print(f"[TREND GPT] Save error: {e}")


def get_recent_gpt_trend_analyses(hours=24, limit=10):
    """Get recent GPT trend analyses for dashboard"""
    try:
        analyses_file = os.path.join("data", "gpt_trend_analyses.json")
        
        if not os.path.exists(analyses_file):
            return []
        
        with open(analyses_file, 'r') as f:
            all_analyses = json.load(f)
        
        # Filter by time and limit
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_analyses = []
        
        for analysis in all_analyses:
            try:
                analysis_time = datetime.fromisoformat(analysis['timestamp'].replace('Z', '+00:00'))
                if analysis_time >= cutoff_time:
                    recent_analyses.append(analysis)
            except (KeyError, ValueError):
                continue
        
        # Sort by timestamp (newest first) and limit
        recent_analyses.sort(key=lambda x: x['timestamp'], reverse=True)
        return recent_analyses[:limit]
        
    except Exception as e:
        print(f"[TREND GPT] Error loading analyses: {e}")
        return []


def compute_gpt_trend_boost(symbol, trend_score, trend_signals, candle_data, tp_levels=None):
    """
    Compute GPT-based trend boost for high-quality trends
    
    Returns:
        dict: {
            "gpt_boost": int (0-10),
            "gpt_analysis": dict,
            "boosted_score": int
        }
    """
    # Analyze with GPT
    gpt_result = analyze_trend_with_gpt(symbol, trend_score, trend_signals, candle_data, tp_levels)
    
    # Calculate GPT boost based on quality score
    gpt_boost = 0
    quality_score = gpt_result.get("quality_score", 0)
    
    if quality_score >= 90:
        gpt_boost = 10  # Maximum boost for exceptional quality
    elif quality_score >= 80:
        gpt_boost = 7   # High quality boost
    elif quality_score >= 70:
        gpt_boost = 5   # Moderate quality boost
    elif quality_score >= 60:
        gpt_boost = 3   # Minor quality boost
    
    boosted_score = min(50, trend_score + gpt_boost)  # Cap at 50
    
    # Save analysis if significant
    if trend_score >= 35:
        save_gpt_trend_analysis(symbol, gpt_result, trend_score)
    
    return {
        "gpt_boost": gpt_boost,
        "gpt_analysis": gpt_result,
        "boosted_score": boosted_score
    }