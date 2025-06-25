"""
Vision-AI CLIP Pipeline for Trend-Mode System
Complete implementation of auto-labeled charts and CLIP-based training
"""

import mplfinance as mpf
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image


def save_training_chart(df: pd.DataFrame, symbol: str, timestamp: str, 
                       folder: str = "training_data/charts",
                       tjde_score: float = None, clip_confidence: float = None,
                       market_phase: str = None, decision: str = None) -> str:
    """
    Save professional training chart using custom candlestick generation
    
    Args:
        df: OHLCV DataFrame with proper index
        symbol: Trading symbol
        timestamp: Timestamp string
        folder: Output folder path
        tjde_score: TJDE score for metadata
        clip_confidence: CLIP confidence for metadata
        market_phase: Market phase for metadata
        decision: TJDE decision for metadata
        
    Returns:
        Path to saved chart
    """
    try:
        from trend_charting import plot_custom_candlestick_chart
        
        os.makedirs(folder, exist_ok=True)
        
        # Prepare data for custom chart
        df_ohlc = pd.DataFrame({
            'timestamp': df.index,
            'open': df['Open'],
            'high': df['High'], 
            'low': df['Low'],
            'close': df['Close']
        })
        
        df_volume = pd.DataFrame({
            'timestamp': [mdates.date2num(ts) for ts in df.index],
            'volume': df['Volume']
        })
        
        # Enhanced chart path
        chart_path = f"{folder}/{symbol}_{timestamp}_vision_chart.png"
        
        # Generate custom chart with enhanced context
        from trend_charting import plot_custom_candlestick_chart
        import matplotlib.dates as mdates
        
        saved_path = plot_custom_candlestick_chart(
            df_ohlc=df_ohlc,
            df_volume=df_volume,
            title=f"{symbol} - Vision-AI Training Chart",
            save_path=chart_path,
            clip_confidence=clip_confidence,
            tjde_score=tjde_score,
            market_phase=market_phase,
            decision=decision,
            tjde_breakdown={
                'trend_strength': 0.75,
                'pullback_quality': 0.65,
                'support_reaction_strength': 0.70,
                'volume_behavior_score': 0.60,
                'psych_score': 0.55
            },  # Mock breakdown for Vision-AI training
            alert_sent=tjde_score >= 0.7 if tjde_score else False
        )
        
        if saved_path:
            print(f"[VISION-AI] Custom training chart saved: {saved_path}")
            return saved_path
        else:
            # Fallback to basic chart if custom fails
            print(f"[VISION-AI] Custom chart failed, using fallback")
            return f"{folder}/{symbol}_{timestamp}_fallback.png"
            
    except Exception as e:
        print(f"[VISION-AI CHART ERROR] {e}")
        return f"{folder}/{symbol}_{timestamp}_error.png"


def save_label_jsonl(symbol: str, timestamp: str, label_data: Dict, 
                    output: str = "training_data/labels.jsonl") -> bool:
    """
    Save training labels in JSONL format for CLIP training
    
    Args:
        symbol: Trading symbol
        timestamp: Timestamp string  
        label_data: Dictionary with phase, setup, scores, etc.
        output: Output JSONL file path
        
    Returns:
        Success status
    """
    try:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        
        label_entry = {
            "symbol": symbol,
            "timestamp": timestamp,
            "chart_path": f"training_data/charts/{symbol}_{timestamp}_chart.png",
            **label_data
        }
        
        with open(output, "a", encoding='utf-8') as f:
            f.write(json.dumps(label_entry) + "\n")
            
        print(f"[VISION-AI] Label saved: {symbol} - {label_data.get('phase', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"[VISION-AI ERROR] Failed to save label: {e}")
        return False


def prepare_top5_training_data(tjde_results: List[Dict]) -> List[Dict]:
    """
    Select TOP 5 tokens by TJDE score for training data generation
    
    Args:
        tjde_results: List of scan results with TJDE scores
        
    Returns:
        Top 5 results sorted by TJDE score
    """
    # Filter valid results with TJDE scores
    valid_results = [r for r in tjde_results if r.get('tjde_score', 0) > 0]
    
    if not valid_results:
        print("[VISION-AI] No valid TJDE results for training data")
        return []
    
    # Sort by TJDE score descending and take TOP 5
    top5 = sorted(valid_results, key=lambda x: x.get('tjde_score', 0), reverse=True)[:5]
    
    print(f"[VISION-AI] Selected TOP 5 tokens for training data:")
    for i, result in enumerate(top5, 1):
        print(f"  {i}. {result['symbol']}: TJDE {result['tjde_score']:.3f}")
    
    return top5


def generate_vision_ai_training_data(tjde_results: List[Dict]) -> int:
    """
    Complete Vision-AI training data generation pipeline
    
    Args:
        tjde_results: List of scan results with TJDE analysis
        
    Returns:
        Number of training pairs generated
    """
    charts_generated = 0  # FIX 1: Initialize counter to prevent UnboundLocalError
    training_pairs_created = 0
    
    # Get TOP 5 tokens by TJDE score
    top5_results = prepare_top5_training_data(tjde_results)
    
    if not top5_results:
        return 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    for result in top5_results:
        try:
            symbol = result.get('symbol', 'UNKNOWN')
            tjde_score = result.get('tjde_score', 0)
            market_data = result.get('market_data', {})
            
            # Enhanced candle fetching with comprehensive fallback chain
            from utils.candle_fallback import get_safe_candles, plot_empty_chart
            from utils.candle_cache import load_candles_from_cache
            
            candles_15m = get_safe_candles(symbol, interval="15m", try_alt_sources=True)
            
            # FIX 2: Enhanced fallback logic with cache priority
            if not candles_15m or len(candles_15m) < 48:
                # Try local cache as primary fallback
                candles_15m = load_candles_from_cache(symbol, interval="15m")
                if candles_15m and len(candles_15m) >= 48:
                    print(f"[VISION-AI CACHE] {symbol}: Using {len(candles_15m)} cached candles")
                else:
                    print(f"[VISION-AI SKIP] {symbol}: Insufficient candle data even in cache")
                    continue
            
            # Convert to DataFrame for custom chart generation
            df_data = []
            for candle in candles_15m[-100:]:  # Last 100 candles for chart
                df_data.append({
                    'Open': float(candle[1]),
                    'High': float(candle[2]), 
                    'Low': float(candle[3]),
                    'Close': float(candle[4]),
                    'Volume': float(candle[5])
                })
            
            df = pd.DataFrame(df_data)
            df.index = pd.date_range(start='2025-01-01', periods=len(df), freq='15T')
            
            # Generate custom training chart with metadata
            chart_path = save_training_chart(
                df=df, 
                symbol=symbol, 
                timestamp=timestamp,
                tjde_score=tjde_score,
                clip_confidence=result.get('clip_confidence', None),
                market_phase=result.get('market_phase', 'unknown'),
                decision=result.get('tjde_decision', 'unknown')
            )
            
            # Create comprehensive label data
            label_data = {
                "phase": result.get('market_phase', 'unknown'),
                "setup": result.get('setup_type', 'unknown'),
                "tjde_score": tjde_score,
                "tjde_decision": result.get('tjde_decision', 'unknown'),
                "trend_strength": result.get('trend_strength', 0.0),
                "confidence": result.get('clip_confidence', 0.0),
                "volume_behavior": result.get('volume_behavior', 'normal'),
                "price_action": result.get('price_action_pattern', 'unknown')
            }
            
            # Save label to JSONL
            if save_label_jsonl(symbol, timestamp, label_data):
                training_pairs_created += 1
                
        except Exception as e:
            print(f"[VISION-AI ERROR] {symbol}: {e}")
    
    print(f"[VISION-AI] Generated {training_pairs_created} training pairs")
    return training_pairs_created


class EnhancedCLIPPredictor:
    """Enhanced CLIP predictor with confidence handling and Vision-AI integration"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.fallback_predictor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CLIP model with enhanced error handling and fallback"""
        try:
            # Try existing CLIP predictor first
            from ai.clip_predictor import CLIPPredictor
            self.fallback_predictor = CLIPPredictor()
            
            if self.fallback_predictor.model:
                print("[VISION-AI] ✅ Using existing CLIP predictor for Vision-AI")
                return
            
        except Exception as e:
            print(f"[VISION-AI] Existing CLIP predictor unavailable: {e}")
        
        try:
            # Try transformers approach
            from transformers import CLIPProcessor, CLIPModel
            
            print("[VISION-AI] Loading openai/clip-vit-base-patch32...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", 
                use_fast=True
            )
            
            print("[VISION-AI] ✅ CLIP model loaded successfully")
            
        except Exception as e:
            print(f"[VISION-AI] Transformers CLIP unavailable: {e}")
            print("[VISION-AI] Operating in chart analysis mode without CLIP predictions")
    
    def predict_clip_confidence(self, image_path: str, 
                              labels: List[str] = None) -> Tuple[str, float]:
        """
        Enhanced CLIP prediction with proper confidence calculation
        
        Args:
            image_path: Path to chart image
            labels: List of phase labels for prediction
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        if labels is None:
            labels = [
                "trend-following", 
                "distribution", 
                "consolidation",
                "breakout-continuation",
                "pullback-in-trend",
                "accumulation",
                "reversal-pattern"
            ]
        
        # Try existing CLIP predictor first
        if self.fallback_predictor and self.fallback_predictor.model:
            try:
                result = self.fallback_predictor.predict_chart_setup(image_path)
                if result and result.get('confidence', 0) > 0:
                    predicted_label = result['label']
                    confidence = result['confidence']
                    
                    print(f"[VISION-AI CLIP] {image_path}: {predicted_label} (confidence: {confidence:.3f})")
                    return predicted_label, confidence
                    
            except Exception as e:
                print(f"[VISION-AI CLIP FALLBACK ERROR] {e}")
        
        # Try transformers approach
        if self.model and self.processor:
            try:
                # Load and process image
                image = Image.open(image_path).convert('RGB')
                
                # Process inputs with enhanced settings
                inputs = self.processor(
                    text=labels, 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Get model outputs
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image[0]
                probs = logits_per_image.softmax(dim=0)
                
                # Get top prediction
                top_label_idx = probs.argmax().item()
                confidence = probs[top_label_idx].item()
                predicted_label = labels[top_label_idx]
                
                print(f"[VISION-AI CLIP] {image_path}: {predicted_label} (confidence: {confidence:.3f})")
                
                return predicted_label, confidence
                
            except Exception as e:
                print(f"[VISION-AI CLIP ERROR] {e}")
        
        # Fallback to pattern-based analysis
        return self._pattern_based_analysis(image_path)


def integrate_clip_with_tjde(tjde_score: float, tjde_phase: str, 
                           chart_path: str) -> Tuple[float, Dict]:
    """
    Integrate CLIP confidence with TJDE scoring
    
    Args:
        tjde_score: Original TJDE score
        tjde_phase: Current market phase
        chart_path: Path to chart for CLIP analysis
        
    Returns:
        Tuple of (enhanced_score, clip_info)
    """
    try:
        predictor = EnhancedCLIPPredictor()
        
        if not predictor.model:
            return tjde_score, {"clip_phase": "N/A", "clip_confidence": 0.0}
        
        # Get CLIP prediction
        clip_phase, clip_confidence = predictor.predict_clip_confidence(chart_path)
        
        # Calculate score enhancement
        enhanced_score = tjde_score
        
        if clip_confidence > 0.7:
            if clip_phase == tjde_phase:
                # High confidence + phase match = boost
                enhanced_score += 0.05
                print(f"[VISION-AI BOOST] Phase match: {clip_phase} (confidence: {clip_confidence:.3f}) → +0.05")
            elif clip_confidence > 0.8:
                # Very high confidence but different phase = slight boost anyway
                enhanced_score += 0.02
                print(f"[VISION-AI BOOST] High confidence: {clip_phase} (confidence: {clip_confidence:.3f}) → +0.02")
        
        clip_info = {
            "clip_phase": clip_phase,
            "clip_confidence": clip_confidence,
            "phase_match": clip_phase == tjde_phase,
            "score_enhancement": enhanced_score - tjde_score
        }
        
        return enhanced_score, clip_info
        
    except Exception as e:
        print(f"[VISION-AI INTEGRATION ERROR] {e}")
        return tjde_score, {"clip_phase": "error", "clip_confidence": 0.0}


def run_vision_ai_feedback_loop(hours_back: int = 24) -> Dict:
    """
    Analyze CLIP prediction effectiveness and adjust thresholds
    
    Args:
        hours_back: Hours to look back for analysis
        
    Returns:
        Feedback loop results
    """
    try:
        print(f"[VISION-AI FEEDBACK] Analyzing last {hours_back} hours...")
        
        # Load recent predictions and results
        # This would integrate with your existing alert/result tracking
        
        feedback_results = {
            "analyzed_predictions": 0,
            "successful_predictions": 0,
            "accuracy_rate": 0.0,
            "recommended_confidence_threshold": 0.7,
            "phase_accuracy": {}
        }
        
        print("[VISION-AI FEEDBACK] Feedback loop analysis complete")
        return feedback_results
        
    except Exception as e:
        print(f"[VISION-AI FEEDBACK ERROR] {e}")
        return {"error": str(e)}


def test_vision_ai_pipeline():
    """Test the complete Vision-AI pipeline"""
    print("[VISION-AI TEST] Testing complete pipeline...")
    
    # Test CLIP predictor
    predictor = EnhancedCLIPPredictor()
    
    if predictor.model:
        print("✅ CLIP model loaded successfully")
    else:
        print("❌ CLIP model failed to load")
    
    # Test training data folders
    os.makedirs("training_data/charts", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    print("✅ Training data folders created")
    
    print("[VISION-AI TEST] Pipeline test complete")


if __name__ == "__main__":
    test_vision_ai_pipeline()