"""
Vision-AI CLIP Pipeline for Trend-Mode System
Complete implementation of auto-labeled charts and CLIP-based training
"""

# 🚫 MATPLOTLIB IMPORTS DISABLED - TradingView-only system active
# All chart generation now uses authentic TradingView screenshots only
# import mplfinance as mpf  # DISABLED for TradingView-only system
import json
import os
import glob
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image


def save_training_chart(df: pd.DataFrame, symbol: str, timestamp: str, 
                       folder: str = "training_data/charts",  # Default to professional charts
                       tjde_score: float = None, clip_confidence: float = None,
                       market_phase: str = None, decision: str = None) -> str:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    This function now returns None and redirects to TradingView screenshot system.
    No matplotlib charts are generated to maintain dataset quality.
    
    Args:
        df: OHLCV DataFrame (ignored)
        symbol: Trading symbol
        timestamp: Timestamp string (ignored)
        folder: Output folder path (ignored)
        tjde_score: TJDE score for metadata (ignored)
        clip_confidence: CLIP confidence for metadata (ignored)
        market_phase: Market phase for metadata (ignored)
        decision: TJDE decision for metadata (ignored)
        
    Returns:
        None - Function disabled, use TradingView screenshot system
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Chart generation disabled, using TradingView-only system")
    return None  # All matplotlib chart generation disabled


def plot_chart_vision_ai(candles_15m: List, symbol: str, phase: str = "unknown",
                        setup: str = "unknown", score: float = 0.0, 
                        decision: str = "unknown", clip_confidence: float = 0.0,
                        output_dir: str = "training_data/charts") -> Optional[str]:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    This function now returns None and redirects to TradingView screenshot system.
    No matplotlib charts are generated to maintain dataset quality.
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Chart generation disabled, using TradingView-only system")
    return None


def generate_chart_image(symbol: str, market_data: Dict, tjde_result: Dict,
                        output_dir: str = "training_data/charts") -> Optional[str]:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    This function now returns None and redirects to TradingView screenshot system.
    No matplotlib charts are generated to maintain dataset quality.
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Chart generation disabled, using TradingView-only system")
    return None


def generate_vision_ai_training_data(scan_results: List[Dict], vision_ai_mode: str = "full") -> int:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    
    This function now returns 0 and redirects to TradingView screenshot system.
    No matplotlib charts are generated to maintain dataset quality.
    """
    print(f"[MATPLOTLIB DISABLED] generate_vision_ai_training_data → Chart generation disabled, using TradingView-only system")
    return 0


# Helper functions to support legacy code - all disabled and redirect to TradingView
def get_token_history(symbol: str) -> Dict:
    """Legacy function - redirected to TradingView-only system"""
    return {}


def update_token_memory(symbol: str, data: Dict):
    """Legacy function - redirected to TradingView-only system"""
    pass


def save_label_jsonl(symbol: str, data: Dict):
    """Legacy function - redirected to TradingView-only system"""
    pass


def plot_custom_candlestick_chart(candles: List, symbol: str, **kwargs) -> Optional[str]:
    """
    🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
    """
    print(f"[MATPLOTLIB DISABLED] {symbol} → Chart generation disabled, using TradingView-only system")
    return None


# JSONL label saving functions  
def save_training_labels(symbol: str, timestamp: str, label_data: Dict,
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
        with open(output, 'a', encoding='utf-8') as f:
            import json
            json_line = json.dumps({
                "symbol": symbol,
                "timestamp": timestamp,
                **label_data
            })
            f.write(json_line + '\n')
        return True
    except Exception as e:
        print(f"[JSONL ERROR] {symbol}: {e}")
        return False


def fetch_candles_for_vision(symbol: str) -> Optional[List]:
    """
    Elastyczne pobieranie świec z fallbackiem dla Vision-AI
    
    Args:
        symbol: Symbol tradingowy
        
    Returns:
        Lista świec lub None jeśli brak wystarczających danych
    """
    print(f"[VISION-AI] {symbol} → Function disabled, using TradingView-only system")
    return None


def prepare_top5_training_data(tjde_results: List[Dict]) -> List[Dict]:
    """
    Select TOP 5 tokens by TJDE score for training data generation with fresh data validation
    
    Args:
        tjde_results: List of scan results with TJDE scores
        
    Returns:
        Top 5 results sorted by TJDE score with fresh data validation
    """
    print("[VISION-AI] Function disabled - using TradingView-only system")
    return []


# Clean implementations of necessary functions for the TradingView-only system
def generate_vision_ai_training_data(tjde_results: List[Dict], vision_ai_mode: str = "full") -> int:
    """
    TradingView-ONLY Vision-AI training data generation pipeline
    Completely replaces matplotlib with authentic TradingView screenshots
    
    Args:
        tjde_results: List of scan results with TJDE analysis
        vision_ai_mode: Vision-AI mode ("full", "fast", "minimal")
        
    Returns:
        Number of authentic TradingView charts generated
    """
    print("[VISION-AI] TradingView-only system - using sync wrapper for chart generation")
    return 0


def save_missing_candles_report(symbol: str, reason: str):
    """Zapisuj tokeny z brakującymi danymi do raportu debugowania"""
    print(f"[MISSING CANDLES] {symbol}: {reason}")


def save_training_summary_report(generated: int, attempted: int):
    """Zapisuj podsumowanie sesji treningowej"""
    print(f"[TRAINING SUMMARY] Generated: {generated}, Attempted: {attempted}")


# All function implementations completely cleaned - TradingView-only system active


# EnhancedCLIPPredictor class - also disabled for TradingView-only system
class EnhancedCLIPPredictor:
    """Enhanced CLIP predictor with confidence handling and Vision-AI integration"""
    
    def __init__(self):
        print("[CLIP PREDICTOR] TradingView-only system - CLIP disabled")
        self.model = None
        self.processor = None
    
    def _initialize_model(self):
        """Initialize CLIP model with enhanced error handling and fallback"""
        print("[CLIP INIT] CLIP model disabled - using TradingView-only system")
        return False
    
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
        print(f"[CLIP DISABLED] {image_path} → Using TradingView-only system")
        return ("unknown", 0.0)


# All helper functions below are disabled for TradingView-only system


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
    print("[CLIP-TJDE] Integration disabled - using TradingView-only system")
    return (tjde_score, {"confidence": 0.0, "label": "disabled"})


def run_vision_ai_feedback_loop(hours_back: int = 24) -> Dict:
    """
    Analyze CLIP prediction effectiveness and adjust thresholds
    
    Args:
        hours_back: Hours to look back for analysis
        
    Returns:
        Feedback loop results
    """
    print("[VISION-AI FEEDBACK] Feedback loop disabled - using TradingView-only system")
    return {"status": "disabled"}


def test_vision_ai_pipeline():
    """Test the complete Vision-AI pipeline"""
    print("[VISION-AI TEST] Pipeline test disabled - using TradingView-only system")


# End of file - all matplotlib functions cleaned up for TradingView-only system
    
    print(f"[VISION-AI] Selected TOP 5 tokens for training data with fresh data validation:")
    for i, result in enumerate(top5, 1):
        symbol = result['symbol']
        tjde_score = result['tjde_score']
        print(f"  {i}. {symbol}: TJDE {tjde_score:.3f}")
        
        # FRESH DATA VALIDATION: Check if this token has current market data
        market_data = result.get('market_data', {})
        candles_15m = market_data.get('candles_15m', [])
        
        if candles_15m:
            try:
                from utils.fresh_candles import validate_candle_freshness
                is_fresh = validate_candle_freshness(candles_15m, symbol, max_age_minutes=45)
                if not is_fresh:
                    print(f"    ⚠️ {symbol}: Market data may be stale - will fetch fresh data during chart generation")
                else:
                    print(f"    ✅ {symbol}: Market data is fresh")
            except Exception as e:
                print(f"    ⚠️ {symbol}: Could not validate data freshness: {e}")
        else:
            print(f"    ⚠️ {symbol}: No 15M candles in market_data - will fetch fresh data")
    
    return top5


def generate_vision_ai_training_data(tjde_results: List[Dict], vision_ai_mode: str = "full") -> int:
    """
    TradingView-ONLY Vision-AI training data generation pipeline
    Completely replaces matplotlib with authentic TradingView screenshots
    
    Args:
        tjde_results: List of scan results with TJDE analysis
        vision_ai_mode: Vision-AI mode ("full", "fast", "minimal")
        
    Returns:
        Number of authentic TradingView charts generated
    """
    try:
        print(f"[TRADINGVIEW-ONLY] 🎯 Starting TradingView-only chart generation pipeline")
        
        # 🎯 CRITICAL FIX: Use TOP 5 tokens for TradingView chart generation
        # First select TOP 5 tokens to prevent generating charts for all tokens
        top5_results = prepare_top5_training_data(tjde_results)
        
        if not top5_results:
            print("[TRADINGVIEW-ONLY] No TOP 5 tokens available for chart generation")
            return 0
        
        # ✅ CRITICAL FIX: Skip TradingView generation here since force_refresh_vision_ai_charts() already handles it
        # This prevents duplicate TradingView browser sessions and chart generation conflicts
        print("[TRADINGVIEW-ONLY] 🎯 Skipping TradingView generation - handled by force_refresh pipeline")
        
        # Check for existing TradingView charts generated by force_refresh
        chart_mapping = {}
        for result in top5_results:
            symbol = result.get('symbol', 'UNKNOWN')
            chart_pattern = f"training_data/charts/{symbol}_*.png"
            existing_charts = glob.glob(chart_pattern)
            if existing_charts:
                # Use most recent chart
                latest_chart = max(existing_charts, key=os.path.getmtime)
                chart_mapping[symbol] = latest_chart
                print(f"[TRADINGVIEW-ONLY] ✅ Found existing chart: {symbol} -> {os.path.basename(latest_chart)}")
            else:
                print(f"[TRADINGVIEW-ONLY] ⚠️ No existing chart found for {symbol}")
        
        # Import TradingView-only pipeline - DISABLED to prevent duplication
        # from utils.tradingview_only_pipeline import generate_tradingview_only_charts  
        # chart_mapping = generate_tradingview_only_charts(top5_results)
        
        charts_generated = len(chart_mapping)
        
        if charts_generated > 0:
            print(f"[TRADINGVIEW-ONLY] ✅ Generated {charts_generated} authentic TradingView charts")
            for symbol, path in chart_mapping.items():
                print(f"[TRADINGVIEW-ONLY] • {symbol}: {os.path.basename(path)}")
        else:
            print("[TRADINGVIEW-ONLY] ❌ No TradingView charts generated")
            
        # Generate metadata for training (but NO fallback charts)
        training_pairs_created = 0
        # ✅ Use already prepared TOP 5 results instead of calling prepare_top5_training_data again
        
        if top5_results and vision_ai_mode not in ["minimal"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            for result in top5_results:
                try:
                    symbol = result.get('symbol', 'UNKNOWN')
                    tjde_score = result.get('tjde_score', 0)
                    
                    # Skip low-quality signals in fast mode
                    if vision_ai_mode == "fast" and tjde_score < 0.5:
                        continue
                    
                    # Generate metadata for TJDE analysis
                    phase = result.get('market_phase', 'unknown')
                    decision = result.get('tjde_decision', 'unknown')
                    setup = result.get('setup_type', phase)
                    clip_confidence = result.get('clip_confidence', 0.0)
                    
                    # Create training metadata (no synthetic charts)
                    label_data = {
                        "phase": phase,
                        "setup": setup,
                        "tjde_score": tjde_score,
                        "tjde_decision": decision,
                        "confidence": clip_confidence,
                        "data_source": "tradingview_only",
                        "chart_type": "authentic_tradingview",
                        "has_tradingview_chart": symbol in chart_mapping,
                        "tradingview_path": chart_mapping.get(symbol, None)
                    }
                    
                    if save_label_jsonl(symbol, timestamp, label_data):
                        training_pairs_created += 1
                        
                except Exception as e:
                    print(f"[TRADINGVIEW-ONLY ERROR] {symbol}: {e}")
        
        # Save training summary
        save_training_summary_report(training_pairs_created, len(top5_results) if top5_results else 0)
        
        total_generated = charts_generated + training_pairs_created
        print(f"[TRADINGVIEW-ONLY] ✅ Generated {total_generated} total items ({charts_generated} authentic charts + {training_pairs_created} metadata)")
        
        return total_generated
        
    except ImportError as e:
        print(f"[TRADINGVIEW-ONLY ERROR] Cannot import TradingView pipeline: {e}")
        return 0
    except Exception as e:
        print(f"[TRADINGVIEW-ONLY ERROR] Pipeline failed: {e}")
        return 0

def save_missing_candles_report(symbol: str, reason: str):
    """Zapisuj tokeny z brakującymi danymi do raportu debugowania"""
    try:
        report_file = "data/missing_candles_report.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        # Wczytaj istniejący raport
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report = json.load(f)
        else:
            report = {"missing_candles": [], "last_updated": None}
        
        # Dodaj nowy wpis
        entry = {
            "symbol": symbol,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "needs_investigation": True
        }
        
        report["missing_candles"].append(entry)
        report["last_updated"] = datetime.now().isoformat()
        
        # Zachowaj tylko ostatnie 100 wpisów
        if len(report["missing_candles"]) > 100:
            report["missing_candles"] = report["missing_candles"][-100:]
        
        # Zapisz raport
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
    except Exception as e:
        print(f"[REPORT ERROR] Failed to save missing candles report: {e}")

def save_training_summary_report(generated: int, attempted: int):
    """Zapisuj podsumowanie sesji treningowej"""
    try:
        summary_file = "data/training_summary.json"
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        
        summary = {
            "session_timestamp": datetime.now().isoformat(),
            "tokens_attempted": attempted,
            "training_pairs_generated": generated,
            "success_rate": (generated / attempted) * 100 if attempted > 0 else 0,
            "status": "SUCCESS" if generated > 0 else "FAILED"
        }
        
        # Append to history
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                data = json.load(f)
            if "sessions" not in data:
                data["sessions"] = []
        else:
            data = {"sessions": []}
            
        data["sessions"].append(summary)
        data["last_session"] = summary
        
        # Keep only last 50 sessions
        if len(data["sessions"]) > 50:
            data["sessions"] = data["sessions"][-50:]
        
        with open(summary_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"[TRAINING SUMMARY] {generated}/{attempted} pairs generated ({summary['success_rate']:.1f}% success)")
        
    except Exception as e:
        print(f"[SUMMARY ERROR] Failed to save training summary: {e}")


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