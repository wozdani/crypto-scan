"""
Vision-AI Feedback Loop for CLIP Model Learning
Evaluates CLIP prediction accuracy and updates training dataset
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def evaluate_clip_predictions(hours_back: int = 24) -> Dict:
    """
    Evaluate CLIP prediction accuracy against actual alert outcomes
    
    Args:
        hours_back: Hours to look back for evaluation
        
    Returns:
        Evaluation results and corrections
    """
    if not os.path.exists("training_dataset.jsonl"):
        print("[CLIP FEEDBACK] No training dataset found")
        return {}
    
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    
    corrections = []
    total_predictions = 0
    correct_predictions = 0
    
    # Load and evaluate predictions
    updated_lines = []
    
    try:
        with open("training_dataset.jsonl", "r") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                clip_pred = data.get("clip_prediction")
                
                if not clip_pred:
                    updated_lines.append(line.strip())
                    continue
                
                # Parse timestamp
                timestamp_str = data.get("timestamp", "")
                try:
                    entry_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    updated_lines.append(line.strip())
                    continue
                
                # Skip if too old
                if entry_time < cutoff_time:
                    updated_lines.append(line.strip())
                    continue
                
                total_predictions += 1
                
                # Evaluate prediction accuracy based on score and decision
                actual_score = data.get("score", 0)
                actual_decision = data.get("decision", "avoid")
                clip_decision = clip_pred.get("decision")
                clip_confidence = clip_pred.get("confidence", 0)
                
                # Determine if CLIP was correct
                was_correct = evaluate_prediction_accuracy(
                    clip_decision, actual_decision, actual_score, clip_confidence
                )
                
                if was_correct:
                    correct_predictions += 1
                
                # Determine alert outcome
                alert_outcome = determine_alert_outcome(actual_score, actual_decision)
                
                # Update data entry
                data["was_correct"] = was_correct
                data["alert_outcome"] = alert_outcome
                
                # Add to corrections if wrong
                if not was_correct:
                    correction_data = {
                        "symbol": data.get("symbol"),
                        "clip_prediction": clip_decision,
                        "actual_decision": actual_decision,
                        "actual_score": actual_score,
                        "image_path": data.get("image_path"),
                        "corrected_label": actual_decision
                    }
                    corrections.append(correction_data)
                    
                    # Generate GPT explanation for CLIP error
                    try:
                        from gpt_commentary import explain_clip_misclassification
                        
                        image_path = data.get("image_path")
                        if image_path and os.path.exists(image_path):
                            explanation = explain_clip_misclassification(
                                image_path,
                                actual_decision,
                                clip_decision,
                                data.get("symbol", "UNKNOWN")
                            )
                            if explanation:
                                correction_data["gpt_explanation"] = explanation
                                
                    except ImportError:
                        pass
                    except Exception as e:
                        print(f"[GPT CLIP ERROR] Failed to explain misclassification: {e}")
                
                updated_lines.append(json.dumps(data))
        
        # Write updated dataset
        with open("training_dataset.jsonl", "w") as f:
            for line in updated_lines:
                f.write(line + "\n")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        results = {
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "corrections": corrections,
            "evaluation_time": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"[CLIP FEEDBACK] Evaluated {total_predictions} predictions")
        print(f"[CLIP FEEDBACK] Accuracy: {accuracy:.2%}")
        print(f"[CLIP FEEDBACK] Corrections needed: {len(corrections)}")
        
        # Save evaluation results
        with open("logs/clip_feedback_evaluation.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"[CLIP FEEDBACK ERROR] {e}")
        return {}


def evaluate_prediction_accuracy(clip_decision: str, actual_decision: str, 
                                actual_score: float, clip_confidence: float) -> bool:
    """
    Determine if CLIP prediction was correct based on actual outcomes
    
    Args:
        clip_decision: CLIP predicted decision
        actual_decision: Actual TJDE decision
        actual_score: Actual TJDE score
        clip_confidence: CLIP confidence level
        
    Returns:
        True if prediction was accurate
    """
    # High confidence predictions should match closely
    if clip_confidence > 0.8:
        return clip_decision == actual_decision
    
    # Medium confidence - allow some flexibility
    if clip_confidence > 0.6:
        # Consider similar decisions as correct
        similar_decisions = {
            "consider_entry": ["consider_entry", "join_trend"],
            "avoid": ["avoid", "strong_avoid"],
            "join_trend": ["consider_entry", "join_trend"]
        }
        
        expected_decisions = similar_decisions.get(clip_decision, [clip_decision])
        return actual_decision in expected_decisions
    
    # Low confidence - evaluate based on score alignment
    if clip_decision == "consider_entry" and actual_score >= 0.6:
        return True
    elif clip_decision == "avoid" and actual_score < 0.5:
        return True
    
    return False


def determine_alert_outcome(score: float, decision: str) -> str:
    """
    Determine alert outcome category for learning
    
    Args:
        score: TJDE score
        decision: TJDE decision
        
    Returns:
        Outcome category
    """
    if decision in ["join_trend", "strong_entry"] and score >= 0.75:
        return "high_confidence_entry"
    elif decision == "consider_entry" and score >= 0.6:
        return "moderate_entry"
    elif decision == "avoid" and score < 0.4:
        return "correct_avoidance"
    elif decision == "strong_avoid" and score < 0.3:
        return "strong_avoidance"
    else:
        return "uncertain"


def generate_clip_retraining_data(corrections: List[Dict]) -> int:
    """
    Generate corrected training samples for CLIP retraining
    
    Args:
        corrections: List of incorrect predictions to correct
        
    Returns:
        Number of retraining samples generated
    """
    if not corrections:
        return 0
    
    os.makedirs("data/clip_retraining", exist_ok=True)
    
    retraining_samples = []
    
    for correction in corrections:
        image_path = correction.get("image_path")
        if not image_path or not os.path.exists(image_path):
            continue
        
        # Create corrected training sample
        sample = {
            "image_path": image_path,
            "original_prediction": correction.get("clip_prediction"),
            "correct_label": correction.get("corrected_label"),
            "symbol": correction.get("symbol"),
            "correction_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        retraining_samples.append(sample)
    
    # Save retraining dataset
    retraining_path = f"data/clip_retraining/corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    with open(retraining_path, "w") as f:
        for sample in retraining_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"[CLIP RETRAINING] Generated {len(retraining_samples)} correction samples")
    print(f"[CLIP RETRAINING] Saved to: {retraining_path}")
    
    return len(retraining_samples)


def run_vision_feedback_cycle() -> Dict:
    """
    Run complete Vision-AI feedback cycle
    
    Returns:
        Feedback cycle results
    """
    print("[VISION FEEDBACK] Starting feedback cycle...")
    
    # Evaluate CLIP predictions
    evaluation_results = evaluate_clip_predictions(hours_back=24)
    
    if not evaluation_results:
        return {"status": "failed", "reason": "evaluation_failed"}
    
    # Generate retraining data if needed
    corrections = evaluation_results.get("corrections", [])
    retraining_count = 0
    
    if len(corrections) >= 3:  # Minimum threshold for retraining
        retraining_count = generate_clip_retraining_data(corrections)
    
    results = {
        "status": "completed",
        "evaluation": evaluation_results,
        "retraining_samples": retraining_count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    print(f"[VISION FEEDBACK] Cycle complete: {evaluation_results['accuracy']:.2%} accuracy")
    
    return results


if __name__ == "__main__":
    run_vision_feedback_cycle()