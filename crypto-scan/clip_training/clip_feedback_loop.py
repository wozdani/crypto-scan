"""
CLIP Feedback Loop - Automatic Model Fine-tuning
Automatycznie analizuje skuteczność predykcji CLIP i dostraja model
"""

import os
import json
import torch
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from .clip_model import get_clip_model
from .clip_trainer import get_clip_trainer
from .clip_predictor import get_clip_predictor

logger = logging.getLogger(__name__)

class CLIPFeedbackLoop:
    """CLIP feedback loop for automatic model improvement"""
    
    def __init__(self):
        """Initialize feedback loop components"""
        self.clip_model = get_clip_model()
        self.clip_trainer = get_clip_trainer()
        self.clip_predictor = get_clip_predictor()
        
        # Paths for feedback data
        self.history_path = Path("logs/auto_label_session_history.json")
        self.tjde_results_path = Path("data/results")
        self.feedback_log_path = Path("logs/clip_feedback_log.json")
        self.model_path = Path("models/clip_model_latest.pt")
        
        # Create directories
        for path in [self.history_path.parent, self.tjde_results_path, self.feedback_log_path.parent]:
            path.mkdir(exist_ok=True)
        
        logger.info("CLIP Feedback Loop initialized")
    
    def load_feedback_history(self) -> List[Dict]:
        """Load prediction history from auto-labeling sessions"""
        if not self.history_path.exists():
            logger.warning(f"History file not found: {self.history_path}")
            return []
        
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            logger.info(f"Loaded {len(history)} prediction records from history")
            return history
            
        except Exception as e:
            logger.error(f"Error loading feedback history: {e}")
            return []
    
    def load_latest_tjde_results(self) -> Dict:
        """Load latest TJDE results for comparison"""
        try:
            # Find latest TJDE results file
            result_files = list(self.tjde_results_path.glob("tjde_results_*.json"))
            
            if not result_files:
                logger.warning("No TJDE results files found")
                return {}
            
            latest_file = sorted(result_files, reverse=True)[0]
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                results_list = json.load(f)
            
            # Convert to dict with symbol as key
            results_dict = {}
            for entry in results_list:
                if isinstance(entry, dict) and 'symbol' in entry:
                    results_dict[entry['symbol']] = entry
            
            logger.info(f"Loaded TJDE results for {len(results_dict)} symbols from {latest_file.name}")
            return results_dict
            
        except Exception as e:
            logger.error(f"Error loading TJDE results: {e}")
            return {}
    
    def analyze_prediction_success(self, training_data: List[Dict], tjde_results: Dict) -> List[Dict]:
        """
        Analyze CLIP prediction accuracy against TJDE decisions
        
        Args:
            training_data: List of CLIP predictions with labels
            tjde_results: TJDE decision results
            
        Returns:
            List of incorrect predictions for retraining
        """
        incorrect_predictions = []
        correct_predictions = 0
        total_predictions = 0
        
        # Define label-to-decision mapping for accuracy analysis
        label_decision_mapping = {
            "breakout-continuation": ["consider_entry", "join_trend"],
            "volume-backed breakout": ["consider_entry", "join_trend"],
            "bullish momentum": ["consider_entry", "join_trend"],
            "trending-up": ["consider_entry"],
            "pullback-in-trend": ["consider_entry"],
            "range-accumulation": ["consider_entry"],
            "consolidation": ["consider_entry", "avoid"],
            "trending-down": ["avoid"],
            "bearish momentum": ["avoid"],
            "trend-reversal": ["avoid"],
            "exhaustion pattern": ["avoid"],
            "fake-breakout": ["avoid"]
        }
        
        for item in training_data:
            try:
                symbol = item.get("symbol", "")
                clip_label = item.get("label", "")
                
                if not symbol or not clip_label:
                    continue
                
                # Get TJDE result for this symbol
                tjde_result = tjde_results.get(symbol)
                if not tjde_result:
                    continue
                
                actual_decision = tjde_result.get("decision", "unknown")
                total_predictions += 1
                
                # Parse CLIP label (handle multi-label format)
                primary_label = clip_label.split(" | ")[0].strip()
                
                # Check if prediction matches expected decision
                expected_decisions = label_decision_mapping.get(primary_label, [])
                
                is_correct = actual_decision in expected_decisions
                
                if is_correct:
                    correct_predictions += 1
                else:
                    # Add to incorrect predictions for retraining
                    incorrect_item = {
                        "symbol": symbol,
                        "filename": item.get("filename", ""),
                        "clip_prediction": primary_label,
                        "tjde_decision": actual_decision,
                        "expected_decisions": expected_decisions,
                        "full_label": clip_label,
                        "confidence": item.get("confidence", 0),
                        "timestamp": item.get("timestamp", "")
                    }
                    incorrect_predictions.append(incorrect_item)
                
            except Exception as e:
                logger.error(f"Error analyzing prediction for {item}: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        logger.info(f"Prediction analysis: {correct_predictions}/{total_predictions} correct ({accuracy:.3f} accuracy)")
        logger.info(f"Found {len(incorrect_predictions)} incorrect predictions for retraining")
        
        return incorrect_predictions
    
    def create_corrected_training_samples(self, incorrect_predictions: List[Dict]) -> List[Tuple[str, str]]:
        """
        Create corrected training samples from incorrect predictions
        
        Args:
            incorrect_predictions: List of incorrect CLIP predictions
            
        Returns:
            List of (image_path, corrected_label) tuples
        """
        corrected_samples = []
        
        # Define correction mapping based on TJDE decisions
        decision_to_label_mapping = {
            "join_trend": "breakout-continuation",
            "consider_entry": "pullback-in-trend", 
            "avoid": "trend-reversal"
        }
        
        for prediction in incorrect_predictions:
            try:
                # Find chart image
                filename = prediction.get("filename", "")
                symbol = prediction.get("symbol", "")
                
                # Try different chart locations
                chart_locations = [
                    f"data/training/charts/{filename}.png",
                    f"charts/{filename}.png",
                    f"exports/{filename}.png",
                    f"training_data/clip/{filename}.png"
                ]
                
                chart_path = None
                for location in chart_locations:
                    if os.path.exists(location):
                        chart_path = location
                        break
                
                if not chart_path:
                    logger.warning(f"Chart not found for {filename}")
                    continue
                
                # Generate corrected label based on TJDE decision
                tjde_decision = prediction.get("tjde_decision", "")
                corrected_label = decision_to_label_mapping.get(tjde_decision, "consolidation")
                
                corrected_samples.append((chart_path, corrected_label))
                
            except Exception as e:
                logger.error(f"Error creating corrected sample for {prediction}: {e}")
                continue
        
        logger.info(f"Created {len(corrected_samples)} corrected training samples")
        return corrected_samples
    
    def retrain_on_incorrect_predictions(self, corrected_samples: List[Tuple[str, str]]) -> bool:
        """
        Retrain CLIP model on corrected samples
        
        Args:
            corrected_samples: List of (image_path, corrected_label) tuples
            
        Returns:
            True if retraining was successful
        """
        if not corrected_samples:
            logger.info("No samples for retraining")
            return True
        
        try:
            logger.info(f"Starting CLIP retraining on {len(corrected_samples)} corrected samples")
            
            # Prepare training pairs for CLIP trainer
            training_pairs = []
            for image_path, corrected_label in corrected_samples:
                training_pairs.append((image_path, corrected_label))
            
            # Use existing CLIP trainer for fine-tuning
            if hasattr(self.clip_trainer, '_train_on_pairs'):
                results = self.clip_trainer._train_on_pairs(training_pairs, num_epochs=1)
                
                if results.get('num_batches', 0) > 0:
                    logger.info(f"Retraining completed: {results.get('avg_loss', 0):.4f} avg loss")
                    
                    # Save improved model
                    self.clip_model.save_model()
                    return True
                else:
                    logger.warning("No batches processed during retraining")
                    return False
            else:
                logger.warning("CLIP trainer does not support direct retraining")
                return False
                
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            return False
    
    def log_feedback_session(self, session_data: Dict):
        """Log feedback session results"""
        try:
            # Load existing feedback log
            feedback_log = []
            if self.feedback_log_path.exists():
                with open(self.feedback_log_path, 'r', encoding='utf-8') as f:
                    feedback_log = json.load(f)
            
            # Add new session
            session_entry = {
                "timestamp": datetime.now().isoformat(),
                "total_predictions": session_data.get("total_predictions", 0),
                "correct_predictions": session_data.get("correct_predictions", 0),
                "accuracy": session_data.get("accuracy", 0),
                "incorrect_count": session_data.get("incorrect_count", 0),
                "retrained_samples": session_data.get("retrained_samples", 0),
                "retraining_success": session_data.get("retraining_success", False)
            }
            
            feedback_log.append(session_entry)
            
            # Keep only last 100 sessions
            if len(feedback_log) > 100:
                feedback_log = feedback_log[-100:]
            
            # Save updated log
            with open(self.feedback_log_path, 'w', encoding='utf-8') as f:
                json.dump(feedback_log, f, indent=2)
            
            logger.info(f"Feedback session logged: {session_entry}")
            
        except Exception as e:
            logger.error(f"Error logging feedback session: {e}")
    
    def run_feedback_loop(self, hours_back: int = 24) -> Dict:
        """
        Run complete feedback loop analysis and retraining
        
        Args:
            hours_back: How many hours back to analyze predictions
            
        Returns:
            Feedback loop results
        """
        logger.info("Starting CLIP feedback loop analysis")
        
        try:
            # Load prediction history and TJDE results
            prediction_history = self.load_feedback_history()
            tjde_results = self.load_latest_tjde_results()
            
            if not prediction_history:
                return {"success": False, "error": "No prediction history available"}
            
            if not tjde_results:
                return {"success": False, "error": "No TJDE results available"}
            
            # Filter recent predictions
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_predictions = []
            
            for pred in prediction_history:
                try:
                    pred_time = datetime.fromisoformat(pred.get("timestamp", ""))
                    if pred_time >= cutoff_time:
                        recent_predictions.append(pred)
                except:
                    # Include predictions without valid timestamp
                    recent_predictions.append(pred)
            
            logger.info(f"Analyzing {len(recent_predictions)} recent predictions")
            
            # Analyze prediction accuracy
            incorrect_predictions = self.analyze_prediction_success(recent_predictions, tjde_results)
            
            total_analyzed = len(recent_predictions)
            correct_count = total_analyzed - len(incorrect_predictions)
            accuracy = correct_count / total_analyzed if total_analyzed > 0 else 0
            
            # Create corrected training samples
            corrected_samples = self.create_corrected_training_samples(incorrect_predictions)
            
            # Retrain if we have enough incorrect samples
            retraining_success = False
            if len(corrected_samples) >= 3:  # Minimum samples for meaningful retraining
                retraining_success = self.retrain_on_incorrect_predictions(corrected_samples)
            else:
                logger.info(f"Not enough samples for retraining ({len(corrected_samples)} < 3)")
            
            # Log session results
            session_data = {
                "total_predictions": total_analyzed,
                "correct_predictions": correct_count,
                "accuracy": accuracy,
                "incorrect_count": len(incorrect_predictions),
                "retrained_samples": len(corrected_samples),
                "retraining_success": retraining_success
            }
            
            self.log_feedback_session(session_data)
            
            result = {
                "success": True,
                "analyzed_predictions": total_analyzed,
                "accuracy": accuracy,
                "incorrect_predictions": len(incorrect_predictions),
                "corrected_samples": len(corrected_samples),
                "retraining_performed": retraining_success,
                "session_data": session_data
            }
            
            logger.info(f"Feedback loop completed: {accuracy:.3f} accuracy, {len(corrected_samples)} samples retrained")
            return result
            
        except Exception as e:
            logger.error(f"Error in feedback loop: {e}")
            return {"success": False, "error": str(e)}


def run_daily_feedback_loop() -> Dict:
    """
    Convenience function to run daily feedback loop
    
    Returns:
        Feedback loop results
    """
    feedback_loop = CLIPFeedbackLoop()
    return feedback_loop.run_feedback_loop(hours_back=24)

def run_weekly_feedback_loop() -> Dict:
    """
    Convenience function to run weekly feedback loop
    
    Returns:
        Feedback loop results
    """
    feedback_loop = CLIPFeedbackLoop()
    return feedback_loop.run_feedback_loop(hours_back=168)  # 7 days

def main():
    """Test feedback loop functionality"""
    print("Testing CLIP Feedback Loop")
    print("=" * 40)
    
    try:
        feedback_loop = CLIPFeedbackLoop()
        
        # Test loading components
        print(f"✅ Feedback loop initialized")
        print(f"   CLIP model: {'loaded' if feedback_loop.clip_model.initialized else 'not loaded'}")
        print(f"   History path: {feedback_loop.history_path}")
        print(f"   TJDE results path: {feedback_loop.tjde_results_path}")
        
        # Test loading data
        history = feedback_loop.load_feedback_history()
        tjde_results = feedback_loop.load_latest_tjde_results()
        
        print(f"\nData availability:")
        print(f"   Prediction history: {len(history)} records")
        print(f"   TJDE results: {len(tjde_results)} symbols")
        
        if history and tjde_results:
            print(f"\n🔄 Running feedback loop test...")
            
            # Run feedback loop
            results = feedback_loop.run_feedback_loop(hours_back=24)
            
            if results.get("success"):
                print(f"✅ Feedback loop test completed")
                print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
                print(f"   Analyzed: {results.get('analyzed_predictions', 0)} predictions")
                print(f"   Incorrect: {results.get('incorrect_predictions', 0)}")
                print(f"   Retrained: {results.get('corrected_samples', 0)} samples")
            else:
                print(f"❌ Feedback loop failed: {results.get('error')}")
        else:
            print(f"\n⚠️ Insufficient data for feedback loop test")
            print(f"   Need both prediction history and TJDE results")
        
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main()