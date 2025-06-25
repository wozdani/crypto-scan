"""
Phase 3: Vision-AI Feedback Loop and Model Evaluation
Evaluates CLIP + GPT effectiveness and creates adaptive feedback system
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from token_context_memory import TokenContextMemory


class VisionAIEvaluator:
    """Evaluates and provides feedback for Vision-AI model performance"""
    
    def __init__(self, feedback_dir: str = "feedback"):
        self.feedback_dir = feedback_dir
        self.reports_dir = os.path.join(feedback_dir, "reports")
        self.ensure_feedback_structure()
        
    def ensure_feedback_structure(self):
        """Create feedback directory structure"""
        os.makedirs(self.feedback_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def collect_historical_predictions(self, days_back: int = 7) -> List[Dict]:
        """Collect historical predictions for evaluation"""
        context_memory = TokenContextMemory()
        all_history = context_memory.load_token_history()
        
        cutoff_time = datetime.now() - timedelta(days=days_back)
        cutoff_str = cutoff_time.isoformat()
        
        predictions = []
        
        for symbol, entries in all_history.items():
            for entry in entries:
                if (entry.get("timestamp", "1970-01-01T00:00:00") > cutoff_str and
                    entry.get("verdict") is not None and
                    entry.get("clip_prediction") != "unknown"):
                    
                    predictions.append({
                        "symbol": symbol,
                        "timestamp": entry.get("timestamp"),
                        "clip_prediction": entry.get("clip_prediction"),
                        "clip_confidence": entry.get("clip_confidence", 0.0),
                        "setup_type": entry.get("setup_type"),
                        "gpt_comment": entry.get("gpt_comment", ""),
                        "decision": entry.get("decision"),
                        "tjde_score": entry.get("tjde_score", 0.0),
                        "verdict": entry.get("verdict"),
                        "result_after_2h": entry.get("result_after_2h"),
                        "result_after_6h": entry.get("result_after_6h")
                    })
        
        print(f"[VISION EVAL] Collected {len(predictions)} predictions from last {days_back} days")
        return predictions
    
    def evaluate_clip_accuracy(self, predictions: List[Dict]) -> Dict:
        """Evaluate CLIP prediction accuracy by category"""
        clip_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0, "confidences": []})
        
        for pred in predictions:
            clip_prediction = pred["clip_prediction"]
            verdict = pred["verdict"]
            confidence = pred["clip_confidence"]
            
            # Skip avoided decisions for accuracy calculation
            if verdict == "avoided":
                continue
                
            clip_stats[clip_prediction]["total"] += 1
            clip_stats[clip_prediction]["confidences"].append(confidence)
            
            if verdict == "correct":
                clip_stats[clip_prediction]["correct"] += 1
            else:
                clip_stats[clip_prediction]["wrong"] += 1
        
        # Calculate accuracy metrics
        accuracy_metrics = {}
        overall_correct = 0
        overall_total = 0
        
        for clip_type, stats in clip_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                avg_confidence = np.mean(stats["confidences"]) if stats["confidences"] else 0.0
                
                accuracy_metrics[clip_type] = {
                    "accuracy": accuracy,
                    "correct": stats["correct"],
                    "wrong": stats["wrong"],
                    "total": stats["total"],
                    "avg_confidence": avg_confidence,
                    "confidence_range": [min(stats["confidences"]), max(stats["confidences"])] if stats["confidences"] else [0, 0]
                }
                
                overall_correct += stats["correct"]
                overall_total += stats["total"]
        
        # Overall accuracy
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
        accuracy_metrics["overall"] = {
            "accuracy": overall_accuracy,
            "correct": overall_correct,
            "wrong": overall_total - overall_correct,
            "total": overall_total
        }
        
        print(f"[CLIP ACCURACY] Overall: {overall_accuracy:.1%} ({overall_correct}/{overall_total})")
        return accuracy_metrics
    
    def evaluate_setup_accuracy(self, predictions: List[Dict]) -> Dict:
        """Evaluate accuracy by setup type"""
        setup_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
        
        for pred in predictions:
            setup_type = pred["setup_type"]
            verdict = pred["verdict"]
            
            if verdict == "avoided":
                continue
                
            setup_stats[setup_type]["total"] += 1
            
            if verdict == "correct":
                setup_stats[setup_type]["correct"] += 1
            else:
                setup_stats[setup_type]["wrong"] += 1
        
        # Calculate setup accuracy
        setup_metrics = {}
        for setup, stats in setup_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                setup_metrics[setup] = {
                    "accuracy": accuracy,
                    "correct": stats["correct"],
                    "wrong": stats["wrong"],
                    "total": stats["total"]
                }
        
        return setup_metrics
    
    def analyze_confidence_vs_accuracy(self, predictions: List[Dict]) -> Dict:
        """Analyze relationship between confidence and accuracy"""
        confidence_bins = np.arange(0.0, 1.1, 0.1)
        bin_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for pred in predictions:
            if pred["verdict"] == "avoided":
                continue
                
            confidence = pred["clip_confidence"]
            verdict = pred["verdict"]
            
            # Find confidence bin
            bin_idx = np.digitize(confidence, confidence_bins) - 1
            bin_idx = max(0, min(bin_idx, len(confidence_bins) - 2))
            bin_label = f"{confidence_bins[bin_idx]:.1f}-{confidence_bins[bin_idx + 1]:.1f}"
            
            bin_stats[bin_label]["total"] += 1
            if verdict == "correct":
                bin_stats[bin_label]["correct"] += 1
        
        # Calculate accuracy per bin
        confidence_analysis = {}
        for bin_label, stats in bin_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                confidence_analysis[bin_label] = {
                    "accuracy": accuracy,
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "sample_size": stats["total"]
                }
        
        return confidence_analysis
    
    def calculate_feedback_modifiers(self, clip_accuracy: Dict) -> Dict:
        """Calculate feedback modifiers based on historical accuracy"""
        modifiers = {}
        
        for clip_type, metrics in clip_accuracy.items():
            if clip_type == "overall":
                continue
                
            accuracy = metrics["accuracy"]
            total_samples = metrics["total"]
            
            # Only apply modifiers if we have sufficient samples
            if total_samples >= 3:
                if accuracy < 0.3:  # Poor performance
                    modifier = 0.6  # Reduce confidence by 40%
                elif accuracy < 0.5:  # Below average
                    modifier = 0.8  # Reduce confidence by 20%
                elif accuracy > 0.8:  # Excellent performance
                    modifier = 1.2  # Boost confidence by 20%
                elif accuracy > 0.6:  # Good performance
                    modifier = 1.1  # Boost confidence by 10%
                else:  # Average performance
                    modifier = 1.0  # No change
            else:
                modifier = 1.0  # Not enough data
            
            modifiers[clip_type] = {
                "modifier": modifier,
                "reason": f"accuracy={accuracy:.1%}, samples={total_samples}",
                "accuracy": accuracy,
                "total_samples": total_samples
            }
        
        return modifiers
    
    def generate_error_analysis_charts(self, predictions: List[Dict], clip_accuracy: Dict):
        """Generate error analysis visualization charts"""
        try:
            # Chart 1: Accuracy Heatmap by Prediction Type
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            clip_types = []
            accuracies = []
            sample_sizes = []
            
            for clip_type, metrics in clip_accuracy.items():
                if clip_type != "overall" and metrics["total"] > 0:
                    clip_types.append(clip_type)
                    accuracies.append(metrics["accuracy"])
                    sample_sizes.append(metrics["total"])
            
            if clip_types:
                colors = ['red' if acc < 0.5 else 'orange' if acc < 0.7 else 'green' for acc in accuracies]
                bars = plt.bar(range(len(clip_types)), accuracies, color=colors, alpha=0.7)
                
                # Add sample size labels
                for i, (bar, size) in enumerate(zip(bars, sample_sizes)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'n={size}', ha='center', va='bottom', fontsize=8)
                
                plt.title('CLIP Prediction Accuracy by Type')
                plt.ylabel('Accuracy')
                plt.xlabel('Prediction Type')
                plt.xticks(range(len(clip_types)), [t.replace('_', '\n') for t in clip_types], rotation=45)
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
            
            # Chart 2: Confidence vs Accuracy
            plt.subplot(2, 2, 2)
            confidences = [p["clip_confidence"] for p in predictions if p["verdict"] != "avoided"]
            verdicts = [1 if p["verdict"] == "correct" else 0 for p in predictions if p["verdict"] != "avoided"]
            
            if confidences and verdicts:
                plt.scatter(confidences, verdicts, alpha=0.6, s=30)
                plt.title('Confidence vs Accuracy')
                plt.xlabel('CLIP Confidence')
                plt.ylabel('Correct (1) / Wrong (0)')
                plt.grid(True, alpha=0.3)
                
                # Add trend line
                if len(confidences) > 1:
                    z = np.polyfit(confidences, verdicts, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(confidences), max(confidences), 100)
                    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # Chart 3: Error Distribution by Setup Type
            plt.subplot(2, 2, 3)
            error_counts = Counter()
            
            for pred in predictions:
                if pred["verdict"] == "wrong":
                    setup = pred["setup_type"]
                    error_counts[setup] += 1
            
            if error_counts:
                setups = list(error_counts.keys())
                errors = list(error_counts.values())
                
                plt.bar(range(len(setups)), errors, color='red', alpha=0.7)
                plt.title('Error Count by Setup Type')
                plt.ylabel('Number of Errors')
                plt.xlabel('Setup Type')
                plt.xticks(range(len(setups)), [s.replace('_', '\n') for s in setups], rotation=45)
                plt.grid(True, alpha=0.3)
            
            # Chart 4: Performance Over Time
            plt.subplot(2, 2, 4)
            
            # Group predictions by day
            daily_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
            
            for pred in predictions:
                if pred["verdict"] != "avoided":
                    date = pred["timestamp"][:10]  # Extract date part
                    daily_accuracy[date]["total"] += 1
                    if pred["verdict"] == "correct":
                        daily_accuracy[date]["correct"] += 1
            
            if daily_accuracy:
                dates = sorted(daily_accuracy.keys())
                accuracies = [daily_accuracy[d]["correct"] / daily_accuracy[d]["total"] if daily_accuracy[d]["total"] > 0 else 0 for d in dates]
                
                plt.plot(range(len(dates)), accuracies, 'bo-', linewidth=2, markersize=6)
                plt.title('Daily Accuracy Trend')
                plt.ylabel('Accuracy')
                plt.xlabel('Days')
                plt.xticks(range(len(dates)), [d[5:] for d in dates], rotation=45)  # Show MM-DD
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = os.path.join(self.reports_dir, f"vision_ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[VISION CHARTS] Error analysis saved: {chart_path}")
            return chart_path
            
        except Exception as e:
            print(f"[VISION CHARTS ERROR] {e}")
            return None
    
    def run_complete_evaluation(self, days_back: int = 7) -> Dict:
        """Run complete Vision-AI evaluation and generate feedback report"""
        print(f"[VISION EVAL] Starting complete evaluation ({days_back} days)")
        
        # Collect predictions
        predictions = self.collect_historical_predictions(days_back)
        
        if not predictions:
            print("[VISION EVAL] No predictions found for evaluation")
            return {
                "status": "no_data",
                "message": "No historical predictions available for evaluation"
            }
        
        # Evaluate accuracies
        clip_accuracy = self.evaluate_clip_accuracy(predictions)
        setup_accuracy = self.evaluate_setup_accuracy(predictions)
        confidence_analysis = self.analyze_confidence_vs_accuracy(predictions)
        feedback_modifiers = self.calculate_feedback_modifiers(clip_accuracy)
        
        # Generate charts
        chart_path = self.generate_error_analysis_charts(predictions, clip_accuracy)
        
        # Create comprehensive report
        report = {
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_period_days": days_back,
            "total_predictions": len(predictions),
            "clip_accuracy": clip_accuracy,
            "setup_accuracy": setup_accuracy,
            "confidence_analysis": confidence_analysis,
            "feedback_modifiers": feedback_modifiers,
            "chart_path": chart_path,
            "recommendations": self.generate_recommendations(clip_accuracy, setup_accuracy),
            "performance_summary": self.generate_performance_summary(clip_accuracy, setup_accuracy)
        }
        
        # Save report
        report_path = os.path.join(self.feedback_dir, "vision_feedback_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[VISION EVAL] Complete report saved: {report_path}")
        
        # Print summary
        self.print_evaluation_summary(report)
        
        return report
    
    def generate_recommendations(self, clip_accuracy: Dict, setup_accuracy: Dict) -> List[str]:
        """Generate actionable recommendations based on evaluation"""
        recommendations = []
        
        overall_accuracy = clip_accuracy.get("overall", {}).get("accuracy", 0.0)
        
        if overall_accuracy < 0.5:
            recommendations.append("Overall CLIP accuracy is below 50% - consider retraining the model")
        
        # Check individual prediction types
        for clip_type, metrics in clip_accuracy.items():
            if clip_type == "overall":
                continue
                
            if metrics["accuracy"] < 0.3 and metrics["total"] >= 3:
                recommendations.append(f"'{clip_type}' predictions are highly inaccurate ({metrics['accuracy']:.1%}) - review training data")
        
        # Check setup types
        for setup_type, metrics in setup_accuracy.items():
            if metrics["accuracy"] < 0.4 and metrics["total"] >= 3:
                recommendations.append(f"'{setup_type}' setups show poor performance ({metrics['accuracy']:.1%}) - analyze pattern recognition")
        
        if not recommendations:
            recommendations.append("Model performance is within acceptable ranges")
        
        return recommendations
    
    def generate_performance_summary(self, clip_accuracy: Dict, setup_accuracy: Dict) -> Dict:
        """Generate performance summary metrics"""
        overall_acc = clip_accuracy.get("overall", {}).get("accuracy", 0.0)
        
        # Performance grade
        if overall_acc >= 0.8:
            grade = "Excellent"
        elif overall_acc >= 0.7:
            grade = "Good"
        elif overall_acc >= 0.5:
            grade = "Average"
        elif overall_acc >= 0.3:
            grade = "Below Average"
        else:
            grade = "Poor"
        
        # Best and worst performing categories
        clip_perfs = [(k, v["accuracy"]) for k, v in clip_accuracy.items() if k != "overall" and v["total"] >= 2]
        setup_perfs = [(k, v["accuracy"]) for k, v in setup_accuracy.items() if v["total"] >= 2]
        
        best_clip = max(clip_perfs, key=lambda x: x[1]) if clip_perfs else ("N/A", 0)
        worst_clip = min(clip_perfs, key=lambda x: x[1]) if clip_perfs else ("N/A", 0)
        
        best_setup = max(setup_perfs, key=lambda x: x[1]) if setup_perfs else ("N/A", 0)
        worst_setup = min(setup_perfs, key=lambda x: x[1]) if setup_perfs else ("N/A", 0)
        
        return {
            "overall_accuracy": overall_acc,
            "performance_grade": grade,
            "best_clip_prediction": {"type": best_clip[0], "accuracy": best_clip[1]},
            "worst_clip_prediction": {"type": worst_clip[0], "accuracy": worst_clip[1]},
            "best_setup_type": {"type": best_setup[0], "accuracy": best_setup[1]},
            "worst_setup_type": {"type": worst_setup[0], "accuracy": worst_setup[1]}
        }
    
    def print_evaluation_summary(self, report: Dict):
        """Print evaluation summary to console"""
        print("\n" + "="*60)
        print("VISION-AI EVALUATION SUMMARY")
        print("="*60)
        
        summary = report["performance_summary"]
        print(f"Overall Performance: {summary['performance_grade']} ({summary['overall_accuracy']:.1%})")
        print(f"Evaluation Period: {report['evaluation_period_days']} days")
        print(f"Total Predictions: {report['total_predictions']}")
        
        print(f"\nBest CLIP Prediction: {summary['best_clip_prediction']['type']} ({summary['best_clip_prediction']['accuracy']:.1%})")
        print(f"Worst CLIP Prediction: {summary['worst_clip_prediction']['type']} ({summary['worst_clip_prediction']['accuracy']:.1%})")
        
        print(f"\nBest Setup Type: {summary['best_setup_type']['type']} ({summary['best_setup_type']['accuracy']:.1%})")
        print(f"Worst Setup Type: {summary['worst_setup_type']['type']} ({summary['worst_setup_type']['accuracy']:.1%})")
        
        print("\nRecommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print("\nFeedback Modifiers Applied:")
        for clip_type, mod in report["feedback_modifiers"].items():
            print(f"  {clip_type}: {mod['modifier']:.1f}x ({mod['reason']})")
        
        print("="*60)


def apply_vision_feedback_modifiers(clip_features: Dict, feedback_modifiers: Dict) -> Dict:
    """Apply Vision-AI feedback modifiers to CLIP features"""
    if not clip_features or not feedback_modifiers:
        return clip_features
    
    modified_features = clip_features.copy()
    clip_prediction = clip_features.get("clip_trend_match", "unknown")
    
    if clip_prediction in feedback_modifiers:
        modifier_data = feedback_modifiers[clip_prediction]
        modifier = modifier_data["modifier"]
        
        # Apply modifier to confidence
        original_confidence = clip_features.get("clip_confidence", 0.0)
        modified_confidence = original_confidence * modifier
        modified_confidence = max(0.0, min(1.0, modified_confidence))  # Keep in valid range
        
        modified_features["clip_confidence"] = modified_confidence
        modified_features["clip_accuracy_modifier"] = modifier
        modified_features["feedback_applied"] = True
        
        if modifier != 1.0:
            print(f"[VISION FEEDBACK] {clip_prediction}: confidence {original_confidence:.3f} → {modified_confidence:.3f} (×{modifier:.1f})")
    
    return modified_features


def load_vision_feedback_modifiers() -> Dict:
    """Load latest Vision-AI feedback modifiers"""
    try:
        feedback_path = "feedback/vision_feedback_report.json"
        if os.path.exists(feedback_path):
            with open(feedback_path, 'r') as f:
                report = json.load(f)
                return report.get("feedback_modifiers", {})
    except Exception as e:
        print(f"[VISION FEEDBACK] Error loading modifiers: {e}")
    
    return {}


def test_vision_ai_evaluation():
    """Test the Vision-AI evaluation system"""
    print("Testing Phase 3: Vision-AI Feedback Loop...")
    
    try:
        # Create test historical data
        context_memory = TokenContextMemory()
        test_symbol = "VISIONEVAL"
        
        # Create diverse test predictions with outcomes
        test_predictions = [
            {
                "decision": "consider_entry",
                "final_score": 0.75,
                "clip_features": {
                    "clip_confidence": 0.8,
                    "clip_trend_match": "pullback",
                    "clip_setup_type": "support-bounce"
                },
                "gpt_comment": "Strong pullback setup with volume confirmation",
                "market_price": 100.0,
                "perception_sync": True
            },
            {
                "decision": "consider_entry",
                "final_score": 0.65,
                "clip_features": {
                    "clip_confidence": 0.6,
                    "clip_trend_match": "breakout",
                    "clip_setup_type": "volume-backed"
                },
                "gpt_comment": "Breakout pattern with increasing volume",
                "market_price": 105.0,
                "perception_sync": True
            },
            {
                "decision": "avoid",
                "final_score": 0.3,
                "clip_features": {
                    "clip_confidence": 0.4,
                    "clip_trend_match": "consolidation",
                    "clip_setup_type": "range-bound"
                },
                "gpt_comment": "Sideways movement without clear direction",
                "market_price": 98.0,
                "perception_sync": True
            }
        ]
        
        # Add predictions to memory
        for pred in test_predictions:
            context_memory.add_decision_entry(test_symbol, pred)
        
        # Simulate outcomes by updating entries
        history = context_memory.load_token_history()
        if test_symbol in history:
            # Make some correct and some wrong
            history[test_symbol][0].update({
                "verdict": "correct",
                "result_after_2h": "+2.5%",
                "result_after_6h": "+5.2%"
            })
            history[test_symbol][1].update({
                "verdict": "wrong",
                "result_after_2h": "-1.2%",
                "result_after_6h": "-2.8%"
            })
            history[test_symbol][2].update({
                "verdict": "avoided",
                "result_after_2h": "N/A",
                "result_after_6h": "N/A"
            })
            
            context_memory.save_token_history(history)
        
        # Run evaluation
        evaluator = VisionAIEvaluator()
        report = evaluator.run_complete_evaluation(days_back=1)
        
        # Test feedback modifier application
        test_clip_features = {
            "clip_confidence": 0.7,
            "clip_trend_match": "pullback",
            "clip_setup_type": "support-bounce"
        }
        
        feedback_modifiers = load_vision_feedback_modifiers()
        modified_features = apply_vision_feedback_modifiers(test_clip_features, feedback_modifiers)
        
        print("\nFeedback Modifier Test:")
        print(f"Original confidence: {test_clip_features['clip_confidence']:.3f}")
        print(f"Modified confidence: {modified_features.get('clip_confidence', 0):.3f}")
        print(f"Modifier applied: {modified_features.get('clip_accuracy_modifier', 1.0):.2f}")
        
        # Cleanup
        if test_symbol in history:
            del history[test_symbol]
            context_memory.save_token_history(history)
        
        print("\n✅ Phase 3 Vision-AI Feedback Loop working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 3 test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_vision_ai_evaluation()