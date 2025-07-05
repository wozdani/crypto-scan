"""
HTF Phase Detector - Higher Timeframe Market Structure Analysis
Detects market phases (uptrend, downtrend, range) on higher timeframes to provide macro context
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def detect_htf_phase(htf_candles: List[Dict], timeframe: str = "1H") -> Dict:
    """
    Wykrywa fazę rynku HTF na podstawie ostatnich świec z enhanced analysis
    
    Args:
        htf_candles: Lista świec HTF w formacie dict lub list
        timeframe: Timeframe identifier for logging
        
    Returns:
        dict: {
            "phase": str,           # "uptrend", "downtrend", "range", "consolidation"
            "strength": float,      # 0.0-1.0 siła trendu
            "direction": str,       # "bullish", "bearish", "neutral"
            "volatility": float,    # Normalized volatility measure
            "confidence": float,    # Confidence in phase detection
            "trend_quality": str,   # "strong", "weak", "choppy"
            "support_levels": List[float],
            "resistance_levels": List[float]
        }
    """
    try:
        if not htf_candles or len(htf_candles) < 10:
            logger.warning(f"[HTF PHASE] Insufficient HTF candles: {len(htf_candles) if htf_candles else 0}")
            return _default_htf_phase()
            
        # Convert candles to standardized format
        formatted_candles = _format_candles(htf_candles)
        
        if len(formatted_candles) < 10:
            logger.warning(f"[HTF PHASE] Insufficient formatted candles: {len(formatted_candles)}")
            return _default_htf_phase()
            
        # Use last 20-50 candles for analysis (depending on availability)
        analysis_window = min(50, len(formatted_candles))
        candles = formatted_candles[-analysis_window:]
        
        logger.info(f"[HTF PHASE] Analyzing {len(candles)} {timeframe} candles")
        
        # Extract price data
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        volumes = [c.get('volume', 1000000) for c in candles]  # Default volume if missing
        
        # === PHASE DETECTION ANALYSIS ===
        
        # 1. Trend direction analysis
        trend_analysis = _analyze_trend_direction(closes)
        
        # 2. Volatility and range analysis
        volatility_analysis = _analyze_volatility(highs, lows, closes)
        
        # 3. Structure analysis (higher highs/lower lows)
        structure_analysis = _analyze_market_structure(highs, lows, closes)
        
        # 4. Volume trend analysis
        volume_analysis = _analyze_volume_trend(volumes)
        
        # 5. Support/Resistance level detection
        sr_levels = _detect_key_levels(highs, lows, closes)
        
        # === PHASE CLASSIFICATION ===
        
        phase_result = _classify_market_phase(
            trend_analysis,
            volatility_analysis, 
            structure_analysis,
            volume_analysis,
            sr_levels
        )
        
        # Add timeframe context
        phase_result['timeframe'] = timeframe
        phase_result['candles_analyzed'] = len(candles)
        
        logger.info(f"[HTF PHASE] {timeframe}: {phase_result['phase']} "
                   f"(strength: {phase_result['strength']:.2f}, "
                   f"confidence: {phase_result['confidence']:.2f})")
        
        return phase_result
        
    except Exception as e:
        logger.error(f"[HTF PHASE ERROR] Phase detection failed: {e}")
        return _default_htf_phase()

def _format_candles(candles: List) -> List[Dict]:
    """Convert various candle formats to standardized dict format"""
    formatted = []
    
    for candle in candles:
        try:
            if isinstance(candle, dict):
                # Already in dict format
                if all(key in candle for key in ['open', 'high', 'low', 'close']):
                    formatted.append(candle)
            elif isinstance(candle, (list, tuple)) and len(candle) >= 5:
                # List format: [timestamp, open, high, low, close, volume]
                formatted.append({
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]) if len(candle) > 5 else 1000000
                })
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"[HTF FORMAT] Skipping invalid candle: {e}")
            continue
            
    return formatted

def _analyze_trend_direction(closes: List[float]) -> Dict:
    """Analyze overall trend direction using multiple methods"""
    
    # 1. Simple price change over window
    price_change = (closes[-1] / closes[0] - 1) * 100 if closes[0] > 0 else 0
    
    # 2. Moving average slope
    if len(closes) >= 20:
        ma20_start = np.mean(closes[:10])
        ma20_end = np.mean(closes[-10:])
        ma_slope = (ma20_end / ma20_start - 1) * 100 if ma20_start > 0 else 0
    else:
        ma_slope = price_change
    
    # 3. Linear regression slope
    x = np.arange(len(closes))
    if len(closes) >= 3:
        try:
            slope, _ = np.polyfit(x, closes, 1)
            normalized_slope = (slope / closes[0]) * 100 if closes[0] > 0 else 0
        except:
            normalized_slope = 0
    else:
        normalized_slope = 0
    
    # Combine indicators
    trend_score = (price_change * 0.4 + ma_slope * 0.4 + normalized_slope * 0.2)
    
    return {
        'price_change_pct': price_change,
        'ma_slope_pct': ma_slope,
        'regression_slope': normalized_slope,
        'combined_trend_score': trend_score
    }

def _analyze_volatility(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Analyze market volatility and range characteristics"""
    
    # True Range calculation
    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        true_ranges.append(tr)
    
    # Average True Range
    atr = np.mean(true_ranges) if true_ranges else 0
    
    # Price volatility (standard deviation)
    price_volatility = np.std(closes) if len(closes) > 1 else 0
    
    # Normalized volatility
    avg_price = np.mean(closes)
    normalized_volatility = (price_volatility / avg_price) * 100 if avg_price > 0 else 0
    
    # Range expansion/contraction
    recent_ranges = [highs[i] - lows[i] for i in range(max(0, len(highs)-10), len(highs))]
    earlier_ranges = [highs[i] - lows[i] for i in range(max(0, len(highs)-20), max(0, len(highs)-10))]
    
    if earlier_ranges and recent_ranges:
        range_change = (np.mean(recent_ranges) / np.mean(earlier_ranges) - 1) * 100
    else:
        range_change = 0
    
    return {
        'atr': atr,
        'price_volatility': price_volatility,
        'normalized_volatility': normalized_volatility,
        'range_change_pct': range_change
    }

def _analyze_market_structure(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Analyze market structure for higher highs/lower lows pattern"""
    
    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    
    window = min(5, len(highs) // 4)  # Adaptive window
    
    for i in range(window, len(highs) - window):
        # Swing high: current high is highest in window
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append((i, highs[i]))
            
        # Swing low: current low is lowest in window  
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append((i, lows[i]))
    
    # Analyze swing patterns
    higher_highs = 0
    lower_highs = 0
    higher_lows = 0
    lower_lows = 0
    
    # Count higher/lower highs
    if len(swing_highs) >= 2:
        for i in range(1, len(swing_highs)):
            if swing_highs[i][1] > swing_highs[i-1][1]:
                higher_highs += 1
            else:
                lower_highs += 1
                
    # Count higher/lower lows
    if len(swing_lows) >= 2:
        for i in range(1, len(swing_lows)):
            if swing_lows[i][1] > swing_lows[i-1][1]:
                higher_lows += 1
            else:
                lower_lows += 1
    
    # Structure score
    total_swings = higher_highs + lower_highs + higher_lows + lower_lows
    if total_swings > 0:
        bullish_structure = (higher_highs + higher_lows) / total_swings
        bearish_structure = (lower_highs + lower_lows) / total_swings
    else:
        bullish_structure = 0.5
        bearish_structure = 0.5
    
    return {
        'swing_highs': len(swing_highs),
        'swing_lows': len(swing_lows),
        'higher_highs': higher_highs,
        'lower_highs': lower_highs,
        'higher_lows': higher_lows,
        'lower_lows': lower_lows,
        'bullish_structure_score': bullish_structure,
        'bearish_structure_score': bearish_structure
    }

def _analyze_volume_trend(volumes: List[float]) -> Dict:
    """Analyze volume trend supporting price movement"""
    
    if len(volumes) < 5:
        return {'volume_trend': 'insufficient_data', 'volume_strength': 0.0}
    
    # Volume moving averages
    recent_vol = np.mean(volumes[-5:])
    earlier_vol = np.mean(volumes[-15:-5]) if len(volumes) >= 15 else np.mean(volumes[:-5])
    
    volume_change = (recent_vol / earlier_vol - 1) * 100 if earlier_vol > 0 else 0
    
    # Volume trend classification
    if volume_change > 20:
        volume_trend = 'increasing_strong'
        volume_strength = 0.8
    elif volume_change > 5:
        volume_trend = 'increasing'
        volume_strength = 0.6
    elif volume_change < -20:
        volume_trend = 'decreasing_strong'
        volume_strength = 0.2
    elif volume_change < -5:
        volume_trend = 'decreasing'
        volume_strength = 0.4
    else:
        volume_trend = 'stable'
        volume_strength = 0.5
    
    return {
        'volume_trend': volume_trend,
        'volume_change_pct': volume_change,
        'volume_strength': volume_strength
    }

def _detect_key_levels(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Detect key support and resistance levels"""
    
    current_price = closes[-1]
    
    # Find recent highs and lows for S/R levels
    recent_highs = sorted(set(highs[-20:]), reverse=True)[:3]  # Top 3 recent highs
    recent_lows = sorted(set(lows[-20:]))[:3]  # Bottom 3 recent lows
    
    # Filter levels close to current price (within 10%)
    resistance_levels = [h for h in recent_highs if h > current_price and h < current_price * 1.1]
    support_levels = [l for l in recent_lows if l < current_price and l > current_price * 0.9]
    
    # Price position analysis
    if resistance_levels:
        distance_to_resistance = min(resistance_levels) - current_price
        resistance_proximity = distance_to_resistance / current_price
    else:
        resistance_proximity = 0.1  # No nearby resistance
        
    if support_levels:
        distance_to_support = current_price - max(support_levels)
        support_proximity = distance_to_support / current_price
    else:
        support_proximity = 0.1  # No nearby support
    
    return {
        'resistance_levels': resistance_levels,
        'support_levels': support_levels,
        'resistance_proximity': resistance_proximity,
        'support_proximity': support_proximity,
        'current_price': current_price
    }

def _classify_market_phase(trend_analysis: Dict, volatility_analysis: Dict, 
                          structure_analysis: Dict, volume_analysis: Dict,
                          sr_levels: Dict) -> Dict:
    """Classify market phase based on all analysis components"""
    
    trend_score = trend_analysis['combined_trend_score']
    volatility = volatility_analysis['normalized_volatility']
    bullish_structure = structure_analysis['bullish_structure_score']
    bearish_structure = structure_analysis['bearish_structure_score']
    volume_strength = volume_analysis['volume_strength']
    
    # Phase classification logic
    if abs(trend_score) < 2 and volatility < 3:
        # Low volatility, sideways movement
        phase = "consolidation"
        strength = min(0.7, volatility / 3)
        direction = "neutral"
        trend_quality = "tight_range"
        
    elif abs(trend_score) < 5 and volatility > 5:
        # High volatility, no clear direction
        phase = "range"
        strength = min(0.8, volatility / 10)
        direction = "neutral"
        trend_quality = "choppy"
        
    elif trend_score > 5:
        # Clear uptrend
        phase = "uptrend"
        strength = min(0.95, trend_score / 20)
        direction = "bullish"
        
        if bullish_structure > 0.7 and volume_strength > 0.6:
            trend_quality = "strong"
        elif bullish_structure > 0.6:
            trend_quality = "moderate"
        else:
            trend_quality = "weak"
            
    elif trend_score < -5:
        # Clear downtrend
        phase = "downtrend"
        strength = min(0.95, abs(trend_score) / 20)
        direction = "bearish"
        
        if bearish_structure > 0.7 and volume_strength > 0.6:
            trend_quality = "strong"
        elif bearish_structure > 0.6:
            trend_quality = "moderate"
        else:
            trend_quality = "weak"
            
    else:
        # Uncertain/transitional phase
        phase = "range"
        strength = 0.4
        direction = "neutral"
        trend_quality = "uncertain"
    
    # Calculate confidence based on consistency of signals
    confidence_factors = []
    
    # Trend consistency
    if abs(trend_score) > 10:
        confidence_factors.append(0.9)
    elif abs(trend_score) > 5:
        confidence_factors.append(0.7)
    else:
        confidence_factors.append(0.4)
    
    # Structure consistency
    max_structure = max(bullish_structure, bearish_structure)
    confidence_factors.append(max_structure)
    
    # Volume confirmation
    confidence_factors.append(volume_strength)
    
    confidence = np.mean(confidence_factors)
    
    return {
        'phase': phase,
        'strength': round(strength, 3),
        'direction': direction,
        'volatility': round(volatility, 2),
        'confidence': round(confidence, 3),
        'trend_quality': trend_quality,
        'support_levels': sr_levels['support_levels'],
        'resistance_levels': sr_levels['resistance_levels'],
        'trend_score': round(trend_score, 2),
        'analysis_components': {
            'trend': trend_analysis,
            'volatility': volatility_analysis,
            'structure': structure_analysis,
            'volume': volume_analysis,
            'levels': sr_levels
        }
    }

def _default_htf_phase() -> Dict:
    """Return default HTF phase when analysis fails"""
    return {
        'phase': 'range',
        'strength': 0.3,
        'direction': 'neutral',
        'volatility': 5.0,
        'confidence': 0.2,
        'trend_quality': 'insufficient_data',
        'support_levels': [],
        'resistance_levels': [],
        'trend_score': 0.0,
        'timeframe': 'unknown',
        'candles_analyzed': 0
    }

def test_htf_phase_detection():
    """Test HTF phase detection with sample data"""
    
    # Create sample uptrend data
    uptrend_candles = []
    base_price = 50000
    
    for i in range(30):
        # Gradual uptrend with some noise
        trend_component = i * 50  # +50 per candle
        noise = (i % 3 - 1) * 100  # Some volatility
        
        price = base_price + trend_component + noise
        
        uptrend_candles.append({
            'timestamp': 1640995200 + i * 3600,  # 1H candles
            'open': price - 25,
            'high': price + 75,
            'low': price - 75,
            'close': price + 25,
            'volume': 1000000 + i * 50000
        })
    
    print("🧪 Testing HTF Phase Detection:")
    print("=" * 40)
    
    result = detect_htf_phase(uptrend_candles, "1H")
    
    print(f"Phase: {result['phase']}")
    print(f"Strength: {result['strength']:.3f}")
    print(f"Direction: {result['direction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Trend Quality: {result['trend_quality']}")
    print(f"Trend Score: {result['trend_score']:.2f}")
    print(f"Candles Analyzed: {result['candles_analyzed']}")

if __name__ == "__main__":
    test_htf_phase_detection()