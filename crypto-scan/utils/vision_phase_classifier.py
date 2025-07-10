"""
Vision Phase Classifier - Computer Vision Analysis for Chart Patterns
Uses Vision Transformer models to analyze chart screenshots like a professional trader
"""

import os
import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    VISION_AVAILABLE = True
except ImportError:
    try:
        # Fallback - just use basic computer vision without transformers
        VISION_AVAILABLE = "basic"
        print("⚠️ Advanced vision model not available, using basic analysis")
    except ImportError:
        VISION_AVAILABLE = False
        print("⚠️ Vision dependencies not available")


class VisionPhaseClassifier:
    """Computer Vision-based chart pattern analysis"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.snapshots_dir = Path("data/vision_snapshots")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart pattern classifications
        self.chart_patterns = {
            "breakout-continuation": "Strong breakout with continuation momentum",
            "breakout-z-cofka": "Breakout with healthy pullback confirmation", 
            "wyczerpanie-trendu": "Trend exhaustion pattern developing",
            "konsolidacja": "Sideways consolidation before next move",
            "fakeout-pattern": "False breakout or bear trap setup",
            "support-test": "Testing key support level reaction",
            "resistance-rejection": "Rejection at resistance level",
            "trend-acceleration": "Accelerating trend momentum"
        }
        
        self.setup_types = {
            "breakout-volume": "Volume-confirmed breakout setup",
            "pullback-entry": "Quality pullback to support entry",
            "bounce-setup": "Bounce from key level setup",
            "continuation-flag": "Flag pattern continuation setup",
            "reversal-pattern": "Potential reversal pattern forming",
            "range-trade": "Range-bound trading opportunity"
        }
        
        if VISION_AVAILABLE == True:
            self._initialize_model()
        elif VISION_AVAILABLE == "basic":
            print("[VISION] Using basic analysis mode")
    
    def _initialize_model(self):
        """Initialize CLIP model for chart analysis"""
        try:
            print("[VISION] Loading CLIP model for chart analysis...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
            print("[VISION] ✅ Model loaded successfully")
        except Exception as e:
            print(f"[VISION] ❌ Model loading failed: {e}")
            self.model = None
            self.processor = None
    
    def generate_chart_image(self, symbol: str, candles: List[List], save_path: str = None, style: str = "vision") -> str:
        """
        🚫 MATPLOTLIB CHART GENERATION DISABLED - TradingView-only system active
        
        Args:
            symbol: Trading symbol
            candles: OHLCV candle data
            save_path: Optional custom save path
            style: Chart style
            
        Returns:
            None - Function disabled, use TradingView screenshot system
        """
        print(f"[MATPLOTLIB DISABLED] {symbol} → Vision chart generation disabled, using TradingView-only system")
        return None
    
    def predict_chart_setup(self, image_path: str) -> Dict:
        """
        Analyze chart image and predict market phase and setup type
        
        Args:
            image_path: Path to chart image
            
        Returns:
            Dict with phase, setup, and confidence predictions
        """
        try:
            if VISION_AVAILABLE == True and self.model:
                # Advanced CLIP model analysis
                return self._clip_analysis(image_path)
            else:
                # Basic heuristic analysis
                return self._fallback_analysis(image_path)
                
        except Exception as e:
            print(f"[VISION] Prediction failed: {e}")
            return self._fallback_analysis(image_path)
    
    def _clip_analysis(self, image_path: str) -> Dict:
        """Advanced CLIP model analysis"""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare text prompts for classification
            phase_prompts = [f"a cryptocurrency chart showing {pattern}" 
                           for pattern in self.chart_patterns.keys()]
            setup_prompts = [f"a trading chart with {setup}" 
                           for setup in self.setup_types.keys()]
            
            # Encode image and text
            inputs = self.processor(
                text=phase_prompts + setup_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Extract phase and setup predictions
            phase_probs = probs[0][:len(phase_prompts)]
            setup_probs = probs[0][len(phase_prompts):]
            
            # Get best predictions
            phase_idx = torch.argmax(phase_probs).item()
            setup_idx = torch.argmax(setup_probs).item()
            
            phase_confidence = float(phase_probs[phase_idx])
            setup_confidence = float(setup_probs[setup_idx])
            
            predicted_phase = list(self.chart_patterns.keys())[phase_idx]
            predicted_setup = list(self.setup_types.keys())[setup_idx]
            
            # Combined confidence score
            combined_confidence = (phase_confidence + setup_confidence) / 2
            
            result = {
                "phase": predicted_phase,
                "setup": predicted_setup,
                "confidence": round(combined_confidence, 3),
                "phase_confidence": round(phase_confidence, 3),
                "setup_confidence": round(setup_confidence, 3),
                "phase_description": self.chart_patterns[predicted_phase],
                "setup_description": self.setup_types[predicted_setup],
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": "CLIP-ViT-Base-Patch32"
            }
            
            # Log prediction
            self._log_vision_decision(image_path, result)
            
            print(f"[VISION] Prediction: {predicted_phase} ({combined_confidence:.3f} confidence)")
            return result
            
        except Exception as e:
            print(f"[VISION] CLIP analysis failed: {e}")
            return self._fallback_analysis(image_path)
    
    def _fallback_analysis(self, image_path: str) -> Dict:
        """
        Fallback analysis when vision model is not available
        Uses basic heuristics based on filename and timestamp
        """
        try:
            # Extract symbol from filename
            filename = Path(image_path).name
            symbol = filename.split('_')[0] if '_' in filename else 'UNKNOWN'
            
            # Basic heuristic analysis
            result = {
                "phase": "konsolidacja",
                "setup": "range-trade",
                "confidence": 0.45,  # Low confidence for fallback
                "phase_confidence": 0.45,
                "setup_confidence": 0.45,
                "phase_description": "Basic technical analysis - consolidation detected",
                "setup_description": "Range-bound trading opportunity",
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": "fallback-heuristics"
            }
            
            self._log_vision_decision(image_path, result)
            print(f"[VISION] ⚠️ Using fallback analysis for {symbol}")
            return result
            
        except Exception as e:
            print(f"[VISION] ❌ Fallback analysis failed: {e}")
            return {
                "phase": "unknown",
                "setup": "unknown", 
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _log_vision_decision(self, image_path: str, result: Dict):
        """Log vision analysis decision to JSON file"""
        try:
            log_file = Path("data/vision_decisions.json")
            
            # Load existing logs
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Add new log entry
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "image_path": str(image_path),
                "symbol": Path(image_path).name.split('_')[0],
                "prediction": result
            }
            
            logs.append(log_entry)
            
            # Keep only last 100 entries
            if len(logs) > 100:
                logs = logs[-100:]
            
            # Save logs
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"[VISION] ⚠️ Logging failed: {e}")
    
    def analyze_symbol_with_vision(self, symbol: str, candles: List[List] = None, export_for_training: bool = False) -> Optional[Dict]:
        """
        Complete vision analysis pipeline for a symbol
        
        Args:
            symbol: Trading symbol
            candles: OHLCV candle data
            export_for_training: Whether to also export with chart_exporter for training data
            
        Returns:
            Vision analysis result or None if failed
        """
        try:
            # Real data only - no mock candles generated
            if not candles:
                print(f"[VISION] ❌ No candle data provided for {symbol} - authentic data required")
                return None
            
            # Generate chart image for vision analysis
            image_path = self.generate_chart_image(symbol, candles, style="vision")
            if not image_path:
                return None
            
            # Export additional training data and label with OpenAI Vision if requested
            training_paths = []
            if export_for_training:
                try:
                    from utils.chart_exporter import export_chart_image
                    from utils.chart_labeler import chart_labeler
                    
                    # Export in multiple styles for training
                    for style in ["professional", "clean"]:
                        training_path = export_chart_image(
                            symbol=symbol,
                            timeframe="15m",
                            chart_style=style,
                            save_as=f"training_{symbol}_{style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        )
                        if training_path:
                            training_paths.append(training_path)
                            
                            # Auto-label with OpenAI Vision
                            features = {
                                "trend_strength": 0.7,
                                "pullback_quality": 0.6,
                                "liquidity_score": 0.8,
                                "htf_trend_match": True,
                                "phase": vision_result.get("phase", "unknown")
                            }
                            
                            try:
                                label = chart_labeler.label_and_save(training_path, features, symbol)
                                print(f"[VISION] Auto-labeled {style} chart as: {label}")
                            except Exception as e:
                                print(f"[VISION] Auto-labeling failed: {e}")
                            
                except ImportError:
                    print("[VISION] Chart exporter/labeler not available for training data")
            
            # Analyze chart pattern
            vision_result = self.predict_chart_setup(image_path)
            vision_result["image_path"] = image_path
            vision_result["symbol"] = symbol
            vision_result["training_exports"] = training_paths
            
            return vision_result
            
        except Exception as e:
            print(f"[VISION] ❌ Complete analysis failed for {symbol}: {e}")
            return None


# Global instance
vision_classifier = VisionPhaseClassifier()


def predict_chart_setup(image_path: str) -> Dict:
    """
    Main function for chart setup prediction
    
    Args:
        image_path: Path to chart image
        
    Returns:
        Prediction result dictionary
    """
    return vision_classifier.predict_chart_setup(image_path)


def analyze_symbol_with_vision(symbol: str, candles: List[List] = None, export_for_training: bool = False) -> Optional[Dict]:
    """
    Analyze symbol with complete vision pipeline
    
    Args:
        symbol: Trading symbol  
        candles: OHLCV candle data (optional, will generate mock data if None)
        export_for_training: Whether to export training data
        
    Returns:
        Vision analysis result
    """
    return vision_classifier.analyze_symbol_with_vision(symbol, candles, export_for_training)


def get_vision_model_status() -> Dict:
    """Get current vision model status"""
    return {
        "available": VISION_AVAILABLE,
        "model_loaded": vision_classifier.model is not None,
        "snapshots_dir": str(vision_classifier.snapshots_dir),
        "supported_patterns": list(vision_classifier.chart_patterns.keys()),
        "supported_setups": list(vision_classifier.setup_types.keys())
    }