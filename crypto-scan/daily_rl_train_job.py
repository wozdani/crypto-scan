#!/usr/bin/env python3
"""
Daily RLAgentV3 Training Job - Automated Learning from Feedback Loop
Automated training system for RLAgentV3 based on feedback data collected during production operations
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import sys

# Add crypto-scan to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_agent_v3 import RLAgentV3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_rl_v3_from_feedback(
    feedback_folder: str = "feedback_logs",
    weight_path: str = "cache/rl_agent_v3_stealth_weights.json",
    training_log_path: str = "cache/training_log.jsonl"
) -> Dict[str, Any]:
    """
    Train RLAgentV3 from feedback data collected during production operations
    
    Args:
        feedback_folder: Directory containing feedback JSONL files
        weight_path: Path to save/load agent weights
        training_log_path: Path to save training history
        
    Returns:
        Dictionary with training statistics
    """
    logger.info("[DAILY RL TRAIN] Starting automated training from feedback data")
    
    try:
        # Initialize RLAgentV3 using SINGLETON PATTERN
        agent = RLAgentV3.instance(weight_path=weight_path)
        logger.info(f"[DAILY RL TRAIN] Initialized RLAgentV3 with weights: {agent.weights}")
        
        total_updates = 0
        pump_positive = 0
        pump_negative = 0
        
        # Ensure feedback folder exists
        if not os.path.exists(feedback_folder):
            logger.warning(f"[DAILY RL TRAIN] Feedback folder not found: {feedback_folder}")
            os.makedirs(feedback_folder, exist_ok=True)
            return {"status": "no_data", "total_updates": 0}
        
        # Process all feedback files
        feedback_files = [f for f in os.listdir(feedback_folder) if f.endswith(".jsonl")]
        logger.info(f"[DAILY RL TRAIN] Found {len(feedback_files)} feedback files")
        
        for file in feedback_files:
            file_path = os.path.join(feedback_folder, file)
            logger.info(f"[DAILY RL TRAIN] Processing file: {file}")
            
            try:
                with open(file_path, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            entry = json.loads(line.strip())
                            
                            # Skip entries without evaluation
                            if entry.get("pump_occurred") is None:
                                continue
                            
                            # Extract inputs for RLAgentV3
                            inputs = {
                                "gnn": round(float(entry.get("gnn_score", 0.0)), 3),
                                "whaleClip": round(float(entry.get("whale_clip_confidence", 0.0)), 3),
                                "dexInflow": int(bool(entry.get("dex_inflow", False)))
                            }
                            
                            # Determine reward based on pump outcome
                            pump_occurred = entry.get("pump_occurred", False)
                            reward = 1.0 if pump_occurred else -1.0
                            
                            # Update agent weights
                            agent.update_weights(inputs, reward)
                            total_updates += 1
                            
                            if pump_occurred:
                                pump_positive += 1
                            else:
                                pump_negative += 1
                            
                            if total_updates % 10 == 0:
                                logger.info(f"[DAILY RL TRAIN] Processed {total_updates} updates...")
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"[DAILY RL TRAIN] JSON error in {file}:{line_num}: {e}")
                        except Exception as e:
                            logger.warning(f"[DAILY RL TRAIN] Error processing entry in {file}:{line_num}: {e}")
                            
            except Exception as e:
                logger.error(f"[DAILY RL TRAIN] Error reading file {file}: {e}")
        
        # Save updated weights
        agent.save()
        logger.info(f"[DAILY RL TRAIN] Saved updated weights to {weight_path}")
        
        # Create training log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_updates": total_updates,
            "pump_positive": pump_positive,
            "pump_negative": pump_negative,
            "success_rate": round(pump_positive / max(1, total_updates) * 100, 2),
            "weights": agent.weights.copy(),
            "booster_effectiveness": agent.get_booster_importance_ranking(),
            "training_metadata": {
                "feedback_files_processed": len(feedback_files),
                "learning_rate": agent.learning_rate,
                "decay_factor": agent.decay,
                "weight_bounds": [agent.min_weight, agent.max_weight]
            }
        }
        
        # Save training log
        os.makedirs(os.path.dirname(training_log_path), exist_ok=True)
        with open(training_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        logger.info(f"[DAILY RL TRAIN] Training complete:")
        logger.info(f"  • Total updates: {total_updates}")
        logger.info(f"  • Positive outcomes: {pump_positive}")
        logger.info(f"  • Negative outcomes: {pump_negative}")
        logger.info(f"  • Success rate: {log_entry['success_rate']}%")
        logger.info(f"  • Updated weights: {agent.weights}")
        
        return {
            "status": "success",
            "total_updates": total_updates,
            "pump_positive": pump_positive,
            "pump_negative": pump_negative,
            "success_rate": log_entry['success_rate'],
            "weights": agent.weights,
            "booster_effectiveness": agent.get_booster_importance_ranking()
        }
        
    except Exception as e:
        logger.error(f"[DAILY RL TRAIN] Critical error during training: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e), "total_updates": 0}

def get_training_history(training_log_path: str = "cache/training_log.jsonl") -> List[Dict[str, Any]]:
    """
    Get training history from log file
    
    Args:
        training_log_path: Path to training log file
        
    Returns:
        List of training entries
    """
    history = []
    
    if not os.path.exists(training_log_path):
        return history
    
    try:
        with open(training_log_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    history.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"[TRAINING HISTORY] Error reading history: {e}")
    
    return history

def get_training_statistics(training_log_path: str = "cache/training_log.jsonl") -> Dict[str, Any]:
    """
    Get comprehensive training statistics
    
    Args:
        training_log_path: Path to training log file
        
    Returns:
        Dictionary with training statistics
    """
    history = get_training_history(training_log_path)
    
    if not history:
        return {"status": "no_data", "total_sessions": 0}
    
    latest = history[-1]
    total_sessions = len(history)
    total_updates = sum(entry.get("total_updates", 0) for entry in history)
    
    # Calculate weight evolution
    weight_evolution = {}
    for entry in history:
        for booster, weight in entry.get("weights", {}).items():
            if booster not in weight_evolution:
                weight_evolution[booster] = []
            weight_evolution[booster].append(weight)
    
    return {
        "status": "success",
        "total_sessions": total_sessions,
        "total_updates": total_updates,
        "latest_session": latest,
        "weight_evolution": weight_evolution,
        "first_session": history[0]["timestamp"] if history else None,
        "last_session": latest["timestamp"],
        "average_updates_per_session": round(total_updates / total_sessions, 1) if total_sessions > 0 else 0
    }

def main():
    """Main function for manual execution"""
    print("🧠 Daily RLAgentV3 Training Job")
    print("=" * 50)
    
    result = train_rl_v3_from_feedback()
    
    if result["status"] == "success":
        print(f"✅ Training completed successfully!")
        print(f"   • Updates processed: {result['total_updates']}")
        print(f"   • Success rate: {result['success_rate']}%")
        print(f"   • Final weights: {result['weights']}")
    elif result["status"] == "no_data":
        print("⚠️ No feedback data found for training")
    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    
    return result["status"] == "success"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)