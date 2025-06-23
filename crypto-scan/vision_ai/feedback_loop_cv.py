#!/usr/bin/env python3
"""
Computer Vision Feedback Loop
Analyzes effectiveness of CV predictions and updates model performance
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CVFeedbackLoop:
    """Manages feedback loop for Computer Vision predictions"""
    
    def __init__(self):
        # Directory structure
        self.feedback_dir = Path("data/cv_feedback_logs")
        self.predictions_dir = Path("data/vision_ai/predictions")
        
        # Create directories
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Feedback analysis parameters
        self.analysis_windows = {
            "short": 2,   # 2 hours
            "medium": 6,  # 6 hours
            "long": 24    # 24 hours
        }
        
        # Movement thresholds for success evaluation
        self.movement_thresholds = {
            "breakout_with_pullback": 2.0,    # +2% for breakouts
            "trend_continuation": 1.5,        # +1.5% for continuations
            "exhaustion": -1.0,               # -1% for exhaustion
            "fakeout": -0.5,                  # -0.5% for fakeouts
            "range_accumulation": 0.5,        # ±0.5% for range
            "support_retest": 1.0,            # +1% for support bounce
            "resistance_rejection": -1.0      # -1% for resistance reject
        }
    
    def analyze_prediction_success(
        self, 
        symbol: str, 
        prediction: Dict, 
        hours_elapsed: int = 2
    ) -> Optional[Dict]:
        """
        Analyze if a prediction was successful based on price movement
        
        Args:
            symbol: Trading symbol
            prediction: CV prediction data
            hours_elapsed: Hours to analyze after prediction
            
        Returns:
            Success analysis results
        """
        try:
            prediction_time = datetime.fromisoformat(prediction["timestamp"])
            analysis_time = prediction_time + timedelta(hours=hours_elapsed)
            
            # Get price data for the analysis period
            price_data = self.get_price_movement(symbol, prediction_time, analysis_time)
            
            if not price_data:
                return None
            
            # Determine expected movement based on setup type
            setup_type = prediction.get("setup", "unknown")
            expected_threshold = self.movement_thresholds.get(setup_type, 1.0)
            
            # Calculate actual movement
            actual_movement = price_data["price_change_percent"]
            
            # Determine success based on setup type and direction
            success = self.evaluate_prediction_success(
                setup_type, 
                expected_threshold, 
                actual_movement
            )
            
            # Create feedback record
            feedback = {
                "symbol": symbol,
                "prediction_timestamp": prediction["timestamp"],
                "analysis_timestamp": datetime.now().isoformat(),
                "hours_elapsed": hours_elapsed,
                "setup_type": setup_type,
                "phase_type": prediction.get("phase", "unknown"),
                "prediction_confidence": prediction.get("confidence", 0.0),
                "expected_threshold": expected_threshold,
                "actual_movement": actual_movement,
                "success": success,
                "price_data": price_data,
                "analysis_window": f"{hours_elapsed}h"
            }
            
            # Save feedback log
            self.save_feedback_log(symbol, feedback)
            
            print(f"[CV FEEDBACK] {symbol}: {setup_type} -> {actual_movement:.2f}% ({'✅' if success else '❌'})")
            return feedback
            
        except Exception as e:
            print(f"[CV FEEDBACK] Analysis failed for {symbol}: {e}")
            return None
    
    def evaluate_prediction_success(
        self, 
        setup_type: str, 
        expected_threshold: float, 
        actual_movement: float
    ) -> bool:
        """Evaluate if prediction was successful based on setup expectations"""
        
        if setup_type in ["breakout_with_pullback", "trend_continuation", "support_retest"]:
            # Bullish setups - expect positive movement
            return actual_movement >= expected_threshold
            
        elif setup_type in ["exhaustion", "fakeout", "resistance_rejection"]:
            # Bearish setups - expect negative movement
            return actual_movement <= expected_threshold
            
        elif setup_type == "range_accumulation":
            # Range setups - expect limited movement
            return abs(actual_movement) <= abs(expected_threshold)
            
        else:
            # Unknown setup - use absolute threshold
            return abs(actual_movement) >= abs(expected_threshold)
    
    def get_price_movement(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Optional[Dict]:
        """
        Get price movement data for analysis period
        
        Args:
            symbol: Trading symbol
            start_time: Start of analysis period
            end_time: End of analysis period
            
        Returns:
            Price movement data or None if failed
        """
        try:
            # In production, this would use real price data from Bybit API
            # For now, we'll simulate price movement analysis
            
            # Calculate time difference
            time_diff = (end_time - start_time).total_seconds() / 3600  # hours
            
            # Simulate price movement (in production, use real API)
            # This is a placeholder - replace with actual Bybit API calls
            
            import random
            import numpy as np
            
            # Generate realistic price movement simulation
            # Based on symbol characteristics
            base_volatility = 0.02  # 2% base volatility per hour
            volatility = base_volatility * np.sqrt(time_diff)
            
            # Simulate price change
            price_change = np.random.normal(0, volatility * 100)  # Convert to percentage
            
            # Add some bias based on symbol patterns (simulation)
            if "BTC" in symbol:
                price_change *= 0.8  # Less volatile
            elif "ETH" in symbol:
                price_change *= 1.0  # Normal volatility
            else:
                price_change *= 1.2  # More volatile altcoins
            
            return {
                "symbol": symbol,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "time_window_hours": time_diff,
                "price_change_percent": round(price_change, 2),
                "data_source": "simulated",  # In production: "bybit_api"
                "note": "Replace with real Bybit API data in production"
            }
            
        except Exception as e:
            print(f"[CV FEEDBACK] Price data retrieval failed: {e}")
            return None
    
    def save_feedback_log(self, symbol: str, feedback: Dict):
        """Save feedback log to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp}.json"
            
            feedback_file = self.feedback_dir / filename
            
            with open(feedback_file, 'w') as f:
                json.dump(feedback, f, indent=2)
            
            print(f"[CV FEEDBACK] Saved log: {filename}")
            
        except Exception as e:
            print(f"[CV FEEDBACK] Failed to save feedback log: {e}")
    
    def analyze_pending_predictions(self, hours_back: int = 24) -> Dict:
        """Analyze all predictions from the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Find prediction files to analyze
            prediction_files = list(self.predictions_dir.glob("*.json"))
            
            analyzed_count = 0
            success_count = 0
            results = []
            
            for pred_file in prediction_files:
                try:
                    with open(pred_file, 'r') as f:
                        prediction = json.load(f)
                    
                    # Check if prediction is within analysis window
                    pred_time = datetime.fromisoformat(prediction["timestamp"])
                    
                    if pred_time < cutoff_time:
                        continue  # Too old
                    
                    # Check if enough time has elapsed for analysis (min 2 hours)
                    elapsed_hours = (datetime.now() - pred_time).total_seconds() / 3600
                    
                    if elapsed_hours < 2:
                        continue  # Too recent
                    
                    # Extract symbol from filename
                    symbol = pred_file.name.split('_')[0]
                    
                    # Analyze prediction
                    feedback = self.analyze_prediction_success(
                        symbol, 
                        prediction, 
                        hours_elapsed=min(int(elapsed_hours), 6)  # Max 6 hours
                    )
                    
                    if feedback:
                        analyzed_count += 1
                        if feedback["success"]:
                            success_count += 1
                        results.append(feedback)
                
                except Exception as e:
                    print(f"[CV FEEDBACK] Error analyzing {pred_file.name}: {e}")
                    continue
            
            # Calculate success rate
            success_rate = (success_count / analyzed_count) if analyzed_count > 0 else 0.0
            
            analysis_summary = {
                "analysis_timestamp": datetime.now().isoformat(),
                "hours_analyzed": hours_back,
                "predictions_analyzed": analyzed_count,
                "successful_predictions": success_count,
                "success_rate": round(success_rate, 3),
                "feedback_logs": len(results)
            }
            
            print(f"[CV FEEDBACK] Analysis completed:")
            print(f"  Analyzed: {analyzed_count} predictions")
            print(f"  Success rate: {success_rate:.1%}")
            
            return {
                "summary": analysis_summary,
                "results": results
            }
            
        except Exception as e:
            print(f"[CV FEEDBACK] Batch analysis failed: {e}")
            return {"error": str(e)}
    
    def update_cv_model_weights(self, feedback_results: List[Dict]) -> Dict:
        """
        Update CV model weights based on feedback results
        Currently logs performance, future versions will adjust model
        """
        try:
            if not feedback_results:
                return {"status": "no_data"}
            
            # Analyze performance by setup type
            setup_performance = {}
            
            for result in feedback_results:
                setup_type = result["setup_type"]
                success = result["success"]
                confidence = result["prediction_confidence"]
                
                if setup_type not in setup_performance:
                    setup_performance[setup_type] = {
                        "total": 0,
                        "successful": 0,
                        "avg_confidence": 0.0,
                        "confidences": []
                    }
                
                setup_performance[setup_type]["total"] += 1
                if success:
                    setup_performance[setup_type]["successful"] += 1
                
                setup_performance[setup_type]["confidences"].append(confidence)
            
            # Calculate performance metrics
            for setup_type, perf in setup_performance.items():
                perf["success_rate"] = perf["successful"] / perf["total"]
                perf["avg_confidence"] = sum(perf["confidences"]) / len(perf["confidences"])
                del perf["confidences"]  # Remove raw data for cleaner output
            
            # Create weight update recommendations
            weight_updates = {
                "timestamp": datetime.now().isoformat(),
                "setup_performance": setup_performance,
                "recommendations": self.generate_weight_recommendations(setup_performance),
                "status": "log_only",  # Future: "weights_updated"
                "note": "Model weight updates not yet implemented"
            }
            
            # Save weight update log
            updates_file = self.feedback_dir / f"weight_updates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(updates_file, 'w') as f:
                json.dump(weight_updates, f, indent=2)
            
            print(f"[CV FEEDBACK] Weight analysis completed:")
            for setup_type, perf in setup_performance.items():
                print(f"  {setup_type}: {perf['success_rate']:.1%} success rate")
            
            return weight_updates
            
        except Exception as e:
            print(f"[CV FEEDBACK] Weight update failed: {e}")
            return {"error": str(e)}
    
    def generate_weight_recommendations(self, performance: Dict) -> Dict:
        """Generate recommendations for model weight adjustments"""
        recommendations = {}
        
        for setup_type, perf in performance.items():
            success_rate = perf["success_rate"]
            avg_confidence = perf["avg_confidence"]
            
            if success_rate > 0.7 and avg_confidence < 0.6:
                # Good performance but low confidence - increase confidence
                recommendations[setup_type] = "increase_confidence_weight"
            elif success_rate < 0.4 and avg_confidence > 0.7:
                # Poor performance but high confidence - decrease confidence
                recommendations[setup_type] = "decrease_confidence_weight"
            elif success_rate > 0.8:
                # Excellent performance - maintain or slightly increase
                recommendations[setup_type] = "maintain_weights"
            else:
                # Average performance - monitor
                recommendations[setup_type] = "monitor_performance"
        
        return recommendations
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about feedback logs"""
        try:
            feedback_files = list(self.feedback_dir.glob("*.json"))
            
            if not feedback_files:
                return {"total_feedback_logs": 0}
            
            # Analyze feedback logs
            total_logs = len(feedback_files)
            success_count = 0
            setup_stats = {}
            
            for log_file in feedback_files:
                try:
                    with open(log_file, 'r') as f:
                        feedback = json.load(f)
                    
                    if feedback.get("success"):
                        success_count += 1
                    
                    setup_type = feedback.get("setup_type", "unknown")
                    if setup_type not in setup_stats:
                        setup_stats[setup_type] = {"total": 0, "successful": 0}
                    
                    setup_stats[setup_type]["total"] += 1
                    if feedback.get("success"):
                        setup_stats[setup_type]["successful"] += 1
                        
                except:
                    continue
            
            # Calculate success rates per setup
            for setup_type, stats in setup_stats.items():
                stats["success_rate"] = stats["successful"] / stats["total"]
            
            return {
                "total_feedback_logs": total_logs,
                "overall_success_rate": success_count / total_logs if total_logs > 0 else 0.0,
                "setup_performance": setup_stats,
                "feedback_dir": str(self.feedback_dir)
            }
            
        except Exception as e:
            return {"error": str(e)}


def main():
    """Test CV feedback loop"""
    print("🔄 Computer Vision Feedback Loop")
    print("=" * 40)
    
    feedback_loop = CVFeedbackLoop()
    
    # Analyze recent predictions
    print("Analyzing recent predictions...")
    analysis = feedback_loop.analyze_pending_predictions(hours_back=48)
    
    if "error" not in analysis:
        summary = analysis["summary"]
        print(f"✅ Analysis completed:")
        print(f"  Predictions analyzed: {summary['predictions_analyzed']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        
        # Update model weights based on feedback
        if analysis["results"]:
            print("\nUpdating model weights...")
            weight_updates = feedback_loop.update_cv_model_weights(analysis["results"])
            
            if "error" not in weight_updates:
                print("✅ Weight analysis completed")
    
    # Show feedback statistics
    stats = feedback_loop.get_feedback_stats()
    print(f"\n📊 Feedback Statistics:")
    print(f"  Total logs: {stats.get('total_feedback_logs', 0)}")
    print(f"  Overall success: {stats.get('overall_success_rate', 0):.1%}")


if __name__ == "__main__":
    main()