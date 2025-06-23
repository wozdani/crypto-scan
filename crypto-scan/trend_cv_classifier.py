#!/usr/bin/env python3
"""
Trend Computer Vision Classifier
Uses trained PyTorch model for real-time chart pattern classification
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ChartPatternCNN(nn.Module):
    """CNN model for chart pattern recognition (same as training)"""
    
    def __init__(self, num_classes=6, pretrained=True):
        super(ChartPatternCNN, self).__init__()
        
        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class TrendCVClassifier:
    """Computer Vision classifier for trend analysis"""
    
    def __init__(self, model_path: str = "data/chart_training/models/chart_pattern_model.pth"):
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_map = {}
        self.id_to_label = {}
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if available
        self.load_model()
    
    def load_model(self) -> bool:
        """Load trained model from checkpoint"""
        try:
            if not self.model_path.exists():
                print(f"[CV CLASSIFIER] Model not found: {self.model_path}")
                print("[CV CLASSIFIER] Run train_chart_model.py to create model")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model
            self.model = ChartPatternCNN(num_classes=checkpoint['num_classes'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mapping
            self.label_map = checkpoint.get('label_map', {
                'breakout_continuation': 0,
                'pullback_setup': 1,
                'range': 2,
                'fakeout': 3,
                'exhaustion': 4,
                'retest_confirmation': 5
            })
            self.id_to_label = {v: k for k, v in self.label_map.items()}
            
            print(f"[CV CLASSIFIER] ✅ Model loaded successfully")
            print(f"[CV CLASSIFIER] Supported classes: {list(self.label_map.keys())}")
            return True
            
        except Exception as e:
            print(f"[CV CLASSIFIER] ❌ Failed to load model: {e}")
            return False
    
    def classify_chart(self, image_path: str) -> Dict:
        """
        Classify chart pattern from image
        
        Args:
            image_path: Path to chart image
            
        Returns:
            Dict with predicted_phase and confidence
        """
        try:
            if self.model is None:
                return {
                    "predicted_phase": "unknown",
                    "confidence": 0.0,
                    "error": "Model not loaded"
                }
            
            if not os.path.exists(image_path):
                return {
                    "predicted_phase": "unknown",
                    "confidence": 0.0,
                    "error": "Image file not found"
                }
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                predicted_phase = self.id_to_label[predicted_class.item()]
                confidence_score = confidence.item()
            
            result = {
                "predicted_phase": predicted_phase,
                "confidence": round(confidence_score, 3),
                "all_probabilities": {
                    self.id_to_label[i]: round(prob.item(), 3) 
                    for i, prob in enumerate(probabilities[0])
                }
            }
            
            print(f"[CV CLASSIFIER] 🎯 {os.path.basename(image_path)}: {predicted_phase} ({confidence_score:.3f})")
            return result
            
        except Exception as e:
            print(f"[CV CLASSIFIER] ❌ Classification failed: {e}")
            return {
                "predicted_phase": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def classify_from_candles(self, symbol: str, candles: List, temp_dir: str = "temp") -> Dict:
        """
        Classify pattern by generating chart from candle data
        
        Args:
            symbol: Trading symbol
            candles: OHLCV candle data
            temp_dir: Temporary directory for chart generation
            
        Returns:
            Classification result
        """
        try:
            # Create temporary chart
            from utils.chart_generator import create_pattern_chart
            
            temp_path = Path(temp_dir)
            temp_path.mkdir(exist_ok=True)
            
            # Generate chart image
            chart_path = temp_path / f"{symbol}_temp_classification.png"
            
            # Use chart generator with candle data
            success = self._create_temp_chart(candles, symbol, str(chart_path))
            
            if not success:
                return {
                    "predicted_phase": "unknown",
                    "confidence": 0.0,
                    "error": "Failed to generate chart"
                }
            
            # Classify the generated chart
            result = self.classify_chart(str(chart_path))
            
            # Clean up temp file
            try:
                chart_path.unlink()
            except:
                pass
            
            return result
            
        except Exception as e:
            print(f"[CV CLASSIFIER] ❌ Candle classification failed: {e}")
            return {
                "predicted_phase": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_temp_chart(self, candles: List, symbol: str, save_path: str) -> bool:
        """Create temporary chart for classification"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            
            # Process candle data
            times = [datetime.fromtimestamp(c[0] / 1000) for c in candles]
            opens = [float(c[1]) for c in candles]
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            closes = [float(c[4]) for c in candles]
            volumes = [float(c[5]) for c in candles]
            
            # Create chart (same format as training data)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[4, 1], facecolor='white')
            
            # Candlesticks
            for i in range(len(times)):
                color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
                ax1.plot([times[i], times[i]], [lows[i], highs[i]], color=color, linewidth=1, alpha=0.8)
                body_height = abs(closes[i] - opens[i])
                body_bottom = min(opens[i], closes[i])
                ax1.bar(times[i], body_height, bottom=body_bottom, color=color, width=0.6, alpha=0.9)
            
            # EMA20
            if len(closes) >= 20:
                ema20 = self._calculate_ema(closes, 20)
                if ema20:
                    ax1.plot(times[19:], ema20, label="EMA20", color='#2196f3', linewidth=2)
                    ax1.legend(loc='upper left')
            
            # Styling
            ax1.set_title(f"{symbol} - Chart Pattern Analysis", fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_ylabel('Price', fontsize=12)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # Volume
            volume_colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' for i in range(len(closes))]
            ax2.bar(times, volumes, color=volume_colors, alpha=0.7, width=0.6)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Format axes
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            
            return True
            
        except Exception as e:
            print(f"[CV CLASSIFIER] ❌ Temp chart creation failed: {e}")
            plt.close('all')
            return False
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[List[float]]:
        """Calculate EMA for chart"""
        try:
            if len(prices) < period:
                return None
            
            ema = []
            multiplier = 2 / (period + 1)
            ema.append(sum(prices[:period]) / period)
            
            for i in range(period, len(prices)):
                ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
                ema.append(ema_value)
            
            return ema
        except Exception:
            return None
    
    def batch_classify(self, image_directory: str) -> Dict:
        """Classify multiple charts in a directory"""
        try:
            image_dir = Path(image_directory)
            image_files = list(image_dir.glob("*.png"))
            
            if not image_files:
                return {"error": "No PNG files found in directory"}
            
            results = {}
            
            for image_file in image_files:
                result = self.classify_chart(str(image_file))
                results[image_file.name] = result
            
            return results
            
        except Exception as e:
            return {"error": f"Batch classification failed: {e}"}
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return {
            "model_loaded": self.model is not None,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "supported_classes": list(self.label_map.keys()) if self.label_map else [],
            "num_classes": len(self.label_map) if self.label_map else 0
        }


# Global classifier instance
cv_classifier = TrendCVClassifier()


def classify_chart(image_path: str) -> Dict:
    """
    Main function for chart classification
    
    Args:
        image_path: Path to chart image
        
    Returns:
        Dict with predicted_phase and confidence
    """
    return cv_classifier.classify_chart(image_path)


def classify_from_candles(symbol: str, candles: List) -> Dict:
    """
    Classify pattern from candle data
    
    Args:
        symbol: Trading symbol
        candles: OHLCV candle data
        
    Returns:
        Classification result
    """
    return cv_classifier.classify_from_candles(symbol, candles)


def main():
    """Test the CV classifier"""
    print("🔍 Trend Computer Vision Classifier")
    print("=" * 40)
    
    # Get model info
    info = cv_classifier.get_model_info()
    print(f"Model Status: {'✅ Loaded' if info['model_loaded'] else '❌ Not Loaded'}")
    
    if info['model_loaded']:
        print(f"Model Path: {info['model_path']}")
        print(f"Device: {info['device']}")
        print(f"Classes: {info['supported_classes']}")
        
        # Test classification on available charts
        charts_dir = "data/chart_training/charts"
        if Path(charts_dir).exists():
            results = cv_classifier.batch_classify(charts_dir)
            
            print(f"\n🎯 Classification Results:")
            for filename, result in list(results.items())[:5]:  # Show first 5
                if 'error' not in result:
                    print(f"  {filename}: {result['predicted_phase']} ({result['confidence']:.3f})")
        else:
            print(f"\n⚠️ No charts found in {charts_dir}")
    else:
        print("❌ Model not available. Run train_chart_model.py first.")


if __name__ == "__main__":
    main()