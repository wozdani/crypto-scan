"""
AI Label Pipeline - Complete Vision Analysis Integration
Main pipeline function that combines CLIP, GPT, and heatmap analysis for comprehensive pattern recognition
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def prepare_ai_label(symbol: str, chart_path: str, heatmap_path: Optional[str], 
                    candles: List[Dict], orderbook: Dict, htf_context: str = "unknown") -> Dict:
    """
    Complete AI-EYE pipeline that integrates CLIP + GPT analysis
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        chart_path: Path to TradingView chart image
        heatmap_path: Path to orderbook heatmap (optional)
        candles: List of recent candles for context
        orderbook: Orderbook data with bids/asks
        htf_context: Higher timeframe context description
        
    Returns:
        Dictionary with complete AI analysis including label, phase, confidence
    """
    try:
        logger.info(f"[AI-EYE] 🧠 Starting comprehensive vision analysis for {symbol}")
        
        # === STEP 1: CLIP VISUAL ANALYSIS ===
        try:
            from .clip_predictor import get_clip_prediction
            
            if os.path.exists(chart_path):
                clip_label, clip_confidence = get_clip_prediction(chart_path)
                logger.info(f"[AI-EYE CLIP] {symbol}: {clip_label} (confidence: {clip_confidence})")
            else:
                logger.warning(f"[AI-EYE] ❌ Chart not found: {chart_path}")
                clip_label, clip_confidence = "chart_missing", 0.0
                
        except Exception as e:
            logger.error(f"[AI-EYE CLIP] ❌ CLIP analysis failed: {e}")
            clip_label, clip_confidence = "clip_error", 0.0
            
        # === STEP 2: GENERATE HEATMAP ===
        if not heatmap_path and orderbook:
            try:
                from .heatmap_generator import generate_simple_heatmap
                heatmap_path = generate_simple_heatmap(symbol, orderbook)
                logger.info(f"[AI-EYE HEATMAP] Generated: {heatmap_path}")
            except Exception as e:
                logger.warning(f"[AI-EYE HEATMAP] ❌ Generation failed: {e}")
                heatmap_path = None
                
        # === STEP 3: EXTRACT MARKET CONTEXT ===
        volume_info = _analyze_volume_context(candles)
        price_position = _analyze_price_position(candles)
        
        # === STEP 4: BUILD CONTEXT FOR GPT ===
        context = {
            "symbol": symbol,
            "volume_info": volume_info,
            "price_position": price_position,
            "htf_context": htf_context,
            "clip_label": clip_label,
            "clip_confidence": clip_confidence,
            "has_heatmap": heatmap_path is not None,
            "orderbook_levels": len(orderbook.get('bids', [])) + len(orderbook.get('asks', []))
        }
        
        logger.info(f"[AI-EYE CONTEXT] {symbol}: volume={volume_info}, price={price_position}, htf={htf_context}")
        
        # === STEP 5: GPT CONTEXTUAL ANALYSIS ===
        try:
            from .gpt_labeler import gpt_label_with_context
            gpt_result = gpt_label_with_context(context)
            
            # Enhance with CLIP data
            final_result = gpt_result.copy()
            final_result.update({
                "clip_label": clip_label,
                "clip_confidence": clip_confidence,
                "context": context,
                "heatmap_path": heatmap_path,
                "analysis_source": "ai_eye_pipeline"
            })
            
            logger.info(f"[AI-EYE COMPLETE] {symbol}: {final_result.get('label')} "
                       f"(phase: {final_result.get('phase')}, confidence: {final_result.get('confidence')})")
            
            return final_result
            
        except Exception as e:
            logger.error(f"[AI-EYE GPT] ❌ GPT analysis failed: {e}")
            
            # Fallback to CLIP-only analysis
            return _clip_fallback_analysis(symbol, clip_label, clip_confidence, context)
            
    except Exception as e:
        logger.error(f"[AI-EYE] ❌ Pipeline failed for {symbol}: {e}")
        return _error_fallback(symbol, str(e))

def _analyze_volume_context(candles: List[Dict]) -> str:
    """Analyze volume trend from recent candles"""
    try:
        if not candles or len(candles) < 3:
            return "insufficient_data"
            
        # Extract volumes from last 5 candles
        recent_candles = candles[-5:] if len(candles) >= 5 else candles
        volumes = []
        
        for candle in recent_candles:
            if isinstance(candle, dict):
                vol = candle.get('volume', 0)
            elif isinstance(candle, list) and len(candle) >= 6:
                vol = candle[5]  # Volume is typically at index 5
            else:
                continue
                
            try:
                volumes.append(float(vol))
            except (ValueError, TypeError):
                continue
                
        if len(volumes) < 2:
            return "no_volume_data"
            
        # Analyze trend
        recent_avg = sum(volumes[-2:]) / 2 if len(volumes) >= 2 else volumes[-1]
        earlier_avg = sum(volumes[:-2]) / len(volumes[:-2]) if len(volumes) > 2 else volumes[0]
        
        volume_change = (recent_avg / earlier_avg - 1) * 100 if earlier_avg > 0 else 0
        
        if volume_change > 20:
            return "volume_surge"
        elif volume_change > 5:
            return "volume_increasing"
        elif volume_change < -20:
            return "volume_declining"
        elif volume_change < -5:
            return "volume_decreasing"
        else:
            return "volume_stable"
            
    except Exception as e:
        logger.error(f"[VOLUME ANALYSIS] ❌ Failed: {e}")
        return "volume_analysis_error"

def _analyze_price_position(candles: List[Dict]) -> str:
    """Analyze price position relative to recent action"""
    try:
        if not candles or len(candles) < 3:
            return "insufficient_data"
            
        # Get last 3 candles for analysis
        recent_candles = candles[-3:]
        
        # Extract price data
        prices = []
        for candle in recent_candles:
            if isinstance(candle, dict):
                close = candle.get('close', 0)
                high = candle.get('high', 0)
                low = candle.get('low', 0)
            elif isinstance(candle, list) and len(candle) >= 5:
                close = candle[4]  # Close price
                high = candle[2]   # High price
                low = candle[3]    # Low price
            else:
                continue
                
            try:
                prices.append({
                    'close': float(close),
                    'high': float(high),
                    'low': float(low)
                })
            except (ValueError, TypeError):
                continue
                
        if len(prices) < 2:
            return "no_price_data"
            
        current = prices[-1]
        previous = prices[-2] if len(prices) >= 2 else prices[-1]
        
        # Calculate key levels
        recent_high = max(p['high'] for p in prices)
        recent_low = min(p['low'] for p in prices)
        price_range = recent_high - recent_low
        
        current_close = current['close']
        
        # Determine position
        if price_range > 0:
            position_pct = (current_close - recent_low) / price_range
            
            if current_close > previous['high']:
                return "breakout_above"
            elif current_close < previous['low']:
                return "breakdown_below"
            elif position_pct > 0.8:
                return "near_resistance"
            elif position_pct < 0.2:
                return "near_support"
            elif 0.4 <= position_pct <= 0.6:
                return "middle_range"
            else:
                return "trending_move"
        else:
            return "tight_range"
            
    except Exception as e:
        logger.error(f"[PRICE ANALYSIS] ❌ Failed: {e}")
        return "price_analysis_error"

def _clip_fallback_analysis(symbol: str, clip_label: str, clip_confidence: float, context: Dict) -> Dict:
    """Fallback analysis using only CLIP data when GPT fails"""
    
    # Map CLIP labels to phases
    phase_mapping = {
        "pullback": "trend",
        "breakout": "trend", 
        "accumulation": "accumulation",
        "exhaustion": "distribution",
        "retest": "trend",
        "early_trend": "trend",
        "range": "consolidation",
        "chaos": "consolidation"
    }
    
    fallback_phase = phase_mapping.get(clip_label, "consolidation")
    
    logger.info(f"[AI-EYE FALLBACK] {symbol}: Using CLIP-only analysis")
    
    return {
        "label": clip_label,
        "phase": fallback_phase,
        "confidence": min(0.6, clip_confidence),  # Cap fallback confidence
        "reasoning": f"CLIP-only fallback analysis - {clip_label} pattern detected",
        "clip_label": clip_label,
        "clip_confidence": clip_confidence,
        "context": context,
        "analysis_source": "clip_fallback"
    }

def _error_fallback(symbol: str, error_msg: str) -> Dict:
    """Final fallback when all analysis fails"""
    
    logger.warning(f"[AI-EYE ERROR] {symbol}: Using error fallback")
    
    return {
        "label": "analysis_failed",
        "phase": "unknown",
        "confidence": 0.0,
        "reasoning": f"AI analysis failed: {error_msg}",
        "clip_label": "error",
        "clip_confidence": 0.0,
        "context": {"error": error_msg},
        "analysis_source": "error_fallback"
    }

def test_ai_label_pipeline():
    """Test the complete AI label pipeline"""
    
    # Sample test data
    test_candles = [
        {"open": 50000, "high": 50500, "low": 49800, "close": 50200, "volume": 1000000},
        {"open": 50200, "high": 50800, "low": 50000, "close": 50600, "volume": 1200000},
        {"open": 50600, "high": 51000, "low": 50400, "close": 50900, "volume": 1500000}
    ]
    
    test_orderbook = {
        "bids": [["50800", "5.0"], ["50750", "3.2"], ["50700", "2.1"]],
        "asks": [["50900", "4.1"], ["50950", "2.8"], ["51000", "3.5"]]
    }
    
    # Test the pipeline
    result = prepare_ai_label(
        symbol="TESTUSDT",
        chart_path="test_chart.png",  # Won't exist, will test fallback
        heatmap_path=None,
        candles=test_candles,
        orderbook=test_orderbook,
        htf_context="bullish trend on 4H timeframe"
    )
    
    print("🧪 AI-EYE Pipeline Test Results:")
    print("=" * 40)
    print(f"Label: {result.get('label')}")
    print(f"Phase: {result.get('phase')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Reasoning: {result.get('reasoning')}")
    print(f"Source: {result.get('analysis_source')}")

if __name__ == "__main__":
    test_ai_label_pipeline()