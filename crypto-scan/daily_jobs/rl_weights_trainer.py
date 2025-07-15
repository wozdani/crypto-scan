#!/usr/bin/env python3
"""
Stage 7/7: RL Feedback Trainer - Daily Training System
Automatyczny system treningu RLAgentV4 na podstawie skuteczności alertów z poprzednich dni
"""

import json
import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RLAgentV4 and fusion components
try:
    from stealth_engine.rl.fusion_rl_agent import get_rl_fusion_agent
    from stealth_engine.fusion_layer import get_fusion_engine
    FUSION_AVAILABLE = True
    logger.info("[RL TRAINER] Fusion components imported successfully")
except ImportError as e:
    logger.warning(f"[RL TRAINER] Fusion components not available: {e}")
    FUSION_AVAILABLE = False

class RLFeedbackTrainer:
    """
    Daily RL Feedback Trainer for automatic weight optimization
    """
    
    def __init__(self, cache_path: str = "crypto-scan/cache"):
        """Initialize RL Feedback Trainer"""
        self.cache_path = cache_path
        self.alert_history_file = os.path.join(cache_path, "fusion_history.json")
        self.weights_backup_file = os.path.join(cache_path, "rl_fusion", "weights_backup.json")
        self.training_log_file = os.path.join(cache_path, "rl_fusion", "daily_training.jsonl")
        self.weight_history_file = os.path.join("crypto-scan", "logs", "rl_weight_history.jsonl")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.weights_backup_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.training_log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.weight_history_file), exist_ok=True)
    
    def load_recent_alerts(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Load recent alerts from fusion history for training
        
        Args:
            hours_back: How many hours back to look for alerts
            
        Returns:
            List of recent alerts with outcomes
        """
        try:
            if not os.path.exists(self.alert_history_file):
                logger.warning(f"[RL TRAINER] No fusion history file found: {self.alert_history_file}")
                return []
            
            with open(self.alert_history_file, 'r') as f:
                all_alerts = json.load(f)
            
            # Filter alerts from last N hours
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            recent_alerts = []
            
            for alert in all_alerts:
                try:
                    alert_time = datetime.fromisoformat(alert.get('timestamp', '').replace('Z', '+00:00'))
                    if alert_time >= cutoff_time:
                        recent_alerts.append(alert)
                except (ValueError, KeyError) as e:
                    logger.warning(f"[RL TRAINER] Invalid alert timestamp: {e}")
                    continue
            
            logger.info(f"[RL TRAINER] Loaded {len(recent_alerts)} alerts from last {hours_back} hours")
            return recent_alerts
            
        except Exception as e:
            logger.error(f"[RL TRAINER] Error loading alerts: {e}")
            return []
    
    def evaluate_alert_success(self, alert: Dict[str, Any]) -> tuple[bool, str]:
        """
        Evaluate if an alert was successful based on available data
        
        Args:
            alert: Alert data with fusion scores and metadata
            
        Returns:
            Tuple of (success: bool, outcome_description: str)
        """
        # For now, we'll use fusion score and confidence as success indicators
        # In production, this would check actual market outcomes
        
        fusion_score = alert.get('fusion_score', 0.0)
        confidence = alert.get('confidence', 'UNKNOWN')
        should_alert = alert.get('should_alert', False)
        
        # Mock success evaluation based on signal strength
        # In real implementation, this would check if pump occurred after alert
        if should_alert and fusion_score > 0.8 and confidence in ['VERY_HIGH', 'HIGH']:
            success = True
            outcome = f"high_confidence_alert_strong_signal"
        elif should_alert and fusion_score > 0.65:
            success = True if fusion_score > 0.75 else False
            outcome = f"medium_confidence_alert_score_{fusion_score:.3f}"
        elif should_alert and fusion_score <= 0.65:
            success = False
            outcome = f"weak_signal_false_positive_score_{fusion_score:.3f}"
        else:
            success = False
            outcome = f"no_alert_generated_score_{fusion_score:.3f}"
        
        return success, outcome
    
    def extract_detector_scores(self, alert: Dict[str, Any]) -> List[float]:
        """
        Extract individual detector scores from alert data
        
        Args:
            alert: Alert data with detector details
            
        Returns:
            List of [californium_score, diamond_score, whaleclip_score]
        """
        details = alert.get('details', {})
        
        californium_score = details.get('californium', 0.0)
        diamond_score = details.get('diamond', 0.0) 
        whaleclip_score = details.get('whaleclip', 0.0)
        
        return [californium_score, diamond_score, whaleclip_score]
    
    def train_from_recent_alerts(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Train RLAgentV4 based on recent alert outcomes
        
        Args:
            hours_back: Hours back to look for training data
            
        Returns:
            Training summary statistics
        """
        logger.info(f"[RL TRAINER] Starting daily training from last {hours_back} hours")
        
        # Load recent alerts
        recent_alerts = self.load_recent_alerts(hours_back)
        
        if not recent_alerts:
            logger.warning("[RL TRAINER] No recent alerts found for training")
            return {
                "status": "no_data",
                "alerts_processed": 0,
                "training_samples": 0
            }
        
        # Prepare training data
        training_samples = []
        successful_alerts = 0
        failed_alerts = 0
        
        for alert in recent_alerts:
            detector_scores = self.extract_detector_scores(alert)
            success, outcome = self.evaluate_alert_success(alert)
            
            if success:
                successful_alerts += 1
            else:
                failed_alerts += 1
            
            training_samples.append({
                "scores": detector_scores,
                "reward": 1.0 if success else -1.0,
                "outcome": outcome,
                "timestamp": alert.get('timestamp'),
                "fusion_score": alert.get('fusion_score', 0.0)
            })
        
        # Train RLAgentV4 with batch data
        if training_samples and not FUSION_AVAILABLE:
            logger.warning("[RL TRAINER] Fusion components not available - simulating training")
            return {
                "status": "simulated",
                "alerts_processed": len(recent_alerts),
                "training_samples": len(training_samples),
                "successful_alerts": successful_alerts,
                "failed_alerts": failed_alerts,
                "message": "Training simulated - fusion components not available"
            }
        
        if training_samples and FUSION_AVAILABLE:
            try:
                rl_agent = get_rl_fusion_agent()
                
                # Perform batch training
                for sample in training_samples:
                    # Use direct agent training method if available
                    detector_scores = sample["scores"]
                    reward = sample["reward"]
                    
                    # Try batch_update if available
                    if hasattr(rl_agent, 'batch_update'):
                        rl_agent.batch_update([{
                            "detector_scores": detector_scores,
                            "reward": reward,
                            "outcome": sample["outcome"]
                        }])
                    elif hasattr(rl_agent, 'update'):
                        # Fallback to individual update
                        rl_agent.update(detector_scores, reward > 0)
                
                # Get updated statistics
                training_stats = rl_agent.get_training_statistics()
                
                # Get final weights for logging
                try:
                    final_weights = rl_agent.get_current_weights()
                    if not final_weights or len(final_weights) < 3:
                        final_weights = [0.5, 0.3, 0.2]  # fallback
                except:
                    final_weights = [0.5, 0.3, 0.2]  # fallback
                
                # Save weight history for visualization
                self.save_weight_history(final_weights)
                
                logger.info(f"[RL TRAINER] Training completed: {len(training_samples)} samples")
                logger.info(f"[RL TRAINER] Success/Failure: {successful_alerts}/{failed_alerts}")
                logger.info(f"[RL TRAINER] Updated success rate: {training_stats.get('success_rate', 0):.1f}%")
                
                # Save training log with weights
                training_stats["final_weights"] = {
                    "californium": final_weights[0],
                    "diamond": final_weights[1], 
                    "whaleclip": final_weights[2]
                }
                self.save_training_log(training_samples, training_stats)
                
                return {
                    "status": "success",
                    "alerts_processed": len(recent_alerts),
                    "training_samples": len(training_samples),
                    "successful_alerts": successful_alerts,
                    "failed_alerts": failed_alerts,
                    "success_rate": training_stats.get('success_rate', 0),
                    "total_updates": training_stats.get('total_updates', 0)
                }
                
            except Exception as e:
                logger.error(f"[RL TRAINER] Training failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "alerts_processed": len(recent_alerts),
                    "training_samples": len(training_samples)
                }
        
        else:
            logger.warning("[RL TRAINER] No valid training samples generated")
            return {
                "status": "no_samples",
                "alerts_processed": len(recent_alerts),
                "training_samples": 0
            }
    
    def save_training_log(self, training_samples: List[Dict], training_stats: Dict):
        """Save training session log"""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "training_samples": len(training_samples),
                "successful_samples": sum(1 for s in training_samples if s["reward"] > 0),
                "failed_samples": sum(1 for s in training_samples if s["reward"] <= 0),
                "training_stats": training_stats,
                "sample_outcomes": [s["outcome"] for s in training_samples]
            }
            
            with open(self.training_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            logger.info(f"[RL TRAINER] Training log saved to {self.training_log_file}")
            
        except Exception as e:
            logger.error(f"[RL TRAINER] Failed to save training log: {e}")
    
    def save_weight_history(self, final_weights: List[float]):
        """Save weight evolution history for visualization"""
        try:
            # Ensure logs directory exists
            os.makedirs(os.path.dirname(self.weight_history_file), exist_ok=True)
            
            # Prepare weight history entry
            weight_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "californium": final_weights[0] if len(final_weights) > 0 else 0.5,
                "diamond": final_weights[1] if len(final_weights) > 1 else 0.3,
                "whaleclip": final_weights[2] if len(final_weights) > 2 else 0.2,
                "total_weight": sum(final_weights) if final_weights else 1.0
            }
            
            # Append to JSONL file
            with open(self.weight_history_file, 'a') as f:
                f.write(json.dumps(weight_entry) + '\n')
            
            logger.info(f"[RL TRAINER] Weight history saved: C:{weight_entry['californium']:.3f}, D:{weight_entry['diamond']:.3f}, W:{weight_entry['whaleclip']:.3f}")
            
        except Exception as e:
            logger.error(f"[RL TRAINER] Failed to save weight history: {e}")
    
    def backup_current_weights(self):
        """Create backup of current weights before training"""
        try:
            if not FUSION_AVAILABLE:
                logger.warning("[RL TRAINER] Fusion components not available for backup")
                return
                
            fusion_engine = get_fusion_engine()
            rl_agent = get_rl_fusion_agent()
            
            backup_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fusion_weights": getattr(fusion_engine, 'weights', {}),
                "fusion_threshold": getattr(fusion_engine, 'threshold', 0.65),
                "rl_training_stats": rl_agent.get_training_statistics() if hasattr(rl_agent, 'get_training_statistics') else {}
            }
            
            with open(self.weights_backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"[RL TRAINER] Weights backup saved")
            
        except Exception as e:
            logger.error(f"[RL TRAINER] Failed to backup weights: {e}")

def run_daily_training(hours_back: int = 24):
    """
    Main function to run daily RL training
    
    Args:
        hours_back: Hours back to look for training data
    """
    logger.info("="*60)
    logger.info("[RL TRAINER] Starting daily RL weights training")
    logger.info("="*60)
    
    trainer = RLFeedbackTrainer()
    
    # Backup current weights
    trainer.backup_current_weights()
    
    # Run training
    results = trainer.train_from_recent_alerts(hours_back)
    
    # Log results
    logger.info("[RL TRAINER] Training Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("="*60)
    logger.info("[RL TRAINER] Daily training completed")
    logger.info("="*60)
    
    return results

def test_rl_trainer():
    """Test RL trainer functionality"""
    print("🧪 Testing RL Feedback Trainer...")
    
    trainer = RLFeedbackTrainer()
    
    # Test alert loading
    alerts = trainer.load_recent_alerts(hours_back=168)  # 1 week
    print(f"📊 Loaded {len(alerts)} recent alerts")
    
    if alerts:
        # Test training with real data
        results = trainer.train_from_recent_alerts(hours_back=168)
        print(f"🎯 Training Results: {results}")
    else:
        print("⚠️ No alerts found for testing")
    
    print("✅ RL Trainer test completed")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_rl_trainer()
    else:
        # Run daily training (default: last 24 hours)
        hours = int(sys.argv[1]) if len(sys.argv) > 1 else 24
        run_daily_training(hours)