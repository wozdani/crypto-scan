#!/usr/bin/env python3
"""
Automatyczny trener agenta RL na podstawie feedback_logs
Trenuje Q-table na rzeczywistych wynikach alertów z production data
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from rl_agent_v2 import RLAgentV2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLFeedbackTrainer:
    """
    Automatyczny trener RL Agent na podstawie danych feedback
    """
    
    def __init__(self, feedback_folder: str = "feedback_logs"):
        """
        Initialize RL feedback trainer
        
        Args:
            feedback_folder: Folder containing feedback JSONL files
        """
        self.feedback_folder = feedback_folder
        self.agent = RLAgentV2(
            learning_rate=0.1,
            decay=0.99,
            epsilon=0.2,  # Start with moderate exploration
            epsilon_min=0.01,
            epsilon_decay=0.995,
            q_path="cache/trained_qtable_v2.json"
        )
        
    def load_feedback_data(self, max_age_days: int = 30) -> List[Dict]:
        """
        Load all feedback data from JSONL files
        
        Args:
            max_age_days: Maximum age of data to consider (days)
            
        Returns:
            List of feedback entries with known outcomes
        """
        feedback_entries = []
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        if not os.path.exists(self.feedback_folder):
            logger.warning(f"[FEEDBACK TRAINER] Feedback folder {self.feedback_folder} not found")
            return []
        
        for filename in os.listdir(self.feedback_folder):
            if not filename.endswith(".jsonl"):
                continue
            
            filepath = os.path.join(self.feedback_folder, filename)
            logger.info(f"[FEEDBACK TRAINER] Loading data from {filename}")
            
            try:
                with open(filepath, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            entry = json.loads(line.strip())
                            
                            # Skip entries without known outcome
                            if entry.get("pump_occurred") is None:
                                continue
                            
                            # Check age limit
                            entry_time = datetime.fromisoformat(entry["timestamp"])
                            if entry_time < cutoff_date:
                                continue
                            
                            feedback_entries.append(entry)
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"[FEEDBACK TRAINER] Skipping invalid JSON in {filename}:{line_num}")
                        except Exception as e:
                            logger.warning(f"[FEEDBACK TRAINER] Error processing entry in {filename}:{line_num}: {e}")
                            
            except Exception as e:
                logger.error(f"[FEEDBACK TRAINER] Error reading {filename}: {e}")
        
        logger.info(f"[FEEDBACK TRAINER] Loaded {len(feedback_entries)} feedback entries with known outcomes")
        return feedback_entries
    
    def create_state_from_entry(self, entry: Dict) -> Tuple[float, float, int]:
        """
        Create RL state from feedback entry
        
        Args:
            entry: Feedback entry dictionary
            
        Returns:
            State tuple (gnn_score, whale_clip_conf, dex_inflow)
        """
        gnn_score = round(entry.get("gnn_score", 0.0), 2)
        whale_clip_conf = round(entry.get("whale_clip_confidence", 0.0), 2)
        dex_inflow = int(entry.get("dex_inflow", False))
        
        return (gnn_score, whale_clip_conf, dex_inflow)
    
    def determine_action_from_entry(self, entry: Dict) -> int:
        """
        Determine action taken from feedback entry
        
        Args:
            entry: Feedback entry dictionary
            
        Returns:
            Action (1 = alert sent, 0 = no alert)
        """
        final_score = entry.get("final_score", 0.0)
        return 1 if final_score >= 0.7 else 0
    
    def calculate_reward_from_entry(self, entry: Dict) -> float:
        """
        Calculate reward from feedback entry outcome
        
        Args:
            entry: Feedback entry dictionary
            
        Returns:
            Reward value
        """
        pump_occurred = entry.get("pump_occurred", False)
        
        # Enhanced reward system for RLAgentV2
        if pump_occurred:
            base_reward = 1.5  # Higher base reward for successful pumps
            
            # Bonus for strong price movements
            price_change_1h = entry.get("price_change_1h", 0.0)
            if price_change_1h > 10.0:
                # Very strong pump - maximum bonus
                return base_reward + 1.0
            elif price_change_1h > 5.0:
                # Strong pump - good bonus
                return base_reward + 0.5
            elif price_change_1h > 2.0:
                # Moderate pump - small bonus
                return base_reward + 0.2
            
            return base_reward
        else:
            # Penalty for false alerts
            base_penalty = -1.0
            
            # Additional penalty for strong dumps after false alert
            price_change_1h = entry.get("price_change_1h", 0.0)
            if price_change_1h < -5.0:
                return base_penalty - 0.5
            
            return base_penalty
    
    def train_from_feedback(self, max_age_days: int = 30) -> Dict[str, any]:
        """
        Train RL agent from feedback data
        
        Args:
            max_age_days: Maximum age of data to consider
            
        Returns:
            Training statistics
        """
        logger.info(f"[FEEDBACK TRAINER] Starting RL training from feedback data")
        
        # Load feedback data
        feedback_entries = self.load_feedback_data(max_age_days)
        
        if not feedback_entries:
            logger.warning(f"[FEEDBACK TRAINER] No feedback data found for training")
            return {
                "training_successful": False,
                "reason": "no_data",
                "entries_processed": 0
            }
        
        # Training statistics
        stats = {
            "entries_processed": 0,
            "positive_rewards": 0,
            "negative_rewards": 0,
            "unique_states": set(),
            "action_distribution": {0: 0, 1: 0},
            "decision_breakdown": {},
            "reward_total": 0.0
        }
        
        # Process each feedback entry
        for entry in feedback_entries:
            try:
                # Create state, action, reward
                state = self.create_state_from_entry(entry)
                action = self.determine_action_from_entry(entry)
                reward = self.calculate_reward_from_entry(entry)
                
                # Update Q-table
                self.agent.update(state, action, reward)
                
                # Update statistics
                stats["entries_processed"] += 1
                stats["unique_states"].add(state)
                stats["action_distribution"][action] += 1
                stats["reward_total"] += reward
                
                if reward > 0:
                    stats["positive_rewards"] += 1
                else:
                    stats["negative_rewards"] += 1
                
                # Track decision types
                decision = entry.get("decision", "unknown")
                stats["decision_breakdown"][decision] = stats["decision_breakdown"].get(decision, 0) + 1
                
                logger.debug(f"[FEEDBACK TRAINER] Processed: {entry['token']} - State: {state}, Action: {action}, Reward: {reward}")
                
            except Exception as e:
                logger.warning(f"[FEEDBACK TRAINER] Error processing entry for {entry.get('token', 'unknown')}: {e}")
        
        # Convert set to count for JSON serialization
        stats["unique_states"] = len(stats["unique_states"])
        stats["average_reward"] = stats["reward_total"] / max(stats["entries_processed"], 1)
        stats["success_rate"] = stats["positive_rewards"] / max(stats["entries_processed"], 1)
        stats["training_successful"] = True
        
        logger.info(f"[FEEDBACK TRAINER] Training complete:")
        logger.info(f"   • Entries processed: {stats['entries_processed']}")
        logger.info(f"   • Unique states: {stats['unique_states']}")
        logger.info(f"   • Positive rewards: {stats['positive_rewards']}")
        logger.info(f"   • Success rate: {stats['success_rate']:.2%}")
        logger.info(f"   • Average reward: {stats['average_reward']:.3f}")
        
        return stats
    
    def save_trained_model(self, output_path: str = "cache/trained_qtable.json") -> bool:
        """
        Save trained Q-table to file
        
        Args:
            output_path: Path to save trained model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert Q-table for JSON serialization
            serializable_qtable = {}
            for state, actions in self.agent.q_table.items():
                # Convert state tuple to string key
                state_key = f"{state[0]},{state[1]},{state[2]}"
                # Handle different action value types
                if isinstance(actions, list):
                    # RL Agent uses list format [no_alert_value, alert_value]
                    serializable_qtable[state_key] = {"0": float(actions[0]), "1": float(actions[1])}
                elif isinstance(actions, dict):
                    serializable_qtable[state_key] = {str(k): float(v) for k, v in actions.items()}
                else:
                    # Fallback for other types
                    serializable_qtable[state_key] = {"0": 0.0, "1": float(actions) if actions else 0.0}
            
            # Save with metadata
            model_data = {
                "q_table": serializable_qtable,
                "metadata": {
                    "trained_at": datetime.now().isoformat(),
                    "states_count": len(self.agent.q_table),
                    "learning_rate": self.agent.lr,
                    "epsilon": self.agent.epsilon
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"[FEEDBACK TRAINER] Saved trained Q-table to {output_path}")
            logger.info(f"   • States in Q-table: {len(self.agent.q_table)}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FEEDBACK TRAINER] Failed to save model to {output_path}: {e}")
            return False
    
    def load_pretrained_model(self, model_path: str = "cache/trained_qtable.json") -> bool:
        """
        Load pre-trained Q-table
        
        Args:
            model_path: Path to trained model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                logger.warning(f"[FEEDBACK TRAINER] No pre-trained model found at {model_path}")
                return False
            
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            
            # Restore Q-table
            q_table = {}
            for state_key, actions in model_data["q_table"].items():
                # Convert string key back to tuple
                state_parts = state_key.split(',')
                state = (float(state_parts[0]), float(state_parts[1]), int(state_parts[2]))
                q_table[state] = {int(k): v for k, v in actions.items()}
            
            self.agent.q_table = q_table
            
            metadata = model_data.get("metadata", {})
            logger.info(f"[FEEDBACK TRAINER] Loaded pre-trained model from {model_path}")
            logger.info(f"   • Trained at: {metadata.get('trained_at', 'unknown')}")
            logger.info(f"   • States loaded: {len(q_table)}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FEEDBACK TRAINER] Failed to load model from {model_path}: {e}")
            return False

def train_agent_from_feedback(feedback_folder: str = "feedback_logs", 
                            model_output_path: str = "cache/trained_qtable.json",
                            max_age_days: int = 30) -> Dict[str, any]:
    """
    Main function to train RL agent from feedback data
    
    Args:
        feedback_folder: Folder containing feedback JSONL files
        model_output_path: Path to save trained model
        max_age_days: Maximum age of data to consider
        
    Returns:
        Training statistics
    """
    trainer = RLFeedbackTrainer(feedback_folder)
    
    # Load any existing pre-trained model
    trainer.load_pretrained_model(model_output_path)
    
    # Train from feedback
    stats = trainer.train_from_feedback(max_age_days)
    
    # Save updated model
    if stats.get("training_successful"):
        trainer.save_trained_model(model_output_path)
        stats["model_saved"] = True
        stats["model_path"] = model_output_path
    
    return stats

def test_feedback_trainer():
    """Test feedback trainer functionality"""
    print("🧠 FEEDBACK RL TRAINER TEST")
    print("=" * 40)
    
    try:
        # Test training
        stats = train_agent_from_feedback()
        
        print(f"✅ Training completed: {stats.get('training_successful', False)}")
        print(f"📊 Entries processed: {stats.get('entries_processed', 0)}")
        print(f"🎯 Success rate: {stats.get('success_rate', 0):.2%}")
        print(f"💾 Model saved: {stats.get('model_saved', False)}")
        
        if stats.get("decision_breakdown"):
            print(f"📋 Decision breakdown:")
            for decision, count in stats["decision_breakdown"].items():
                print(f"   • {decision}: {count}")
        
        return stats.get("training_successful", False)
        
    except Exception as e:
        print(f"❌ FEEDBACK TRAINER TEST ERROR: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test
        success = test_feedback_trainer()
        print(f"\n🎯 TEST RESULT: {'✅ PASS' if success else '❌ FAIL'}")
    else:
        # Run training
        stats = train_agent_from_feedback()
        
        print(f"🧠 RL FEEDBACK TRAINING COMPLETE")
        print(f"   📊 Processed: {stats.get('entries_processed', 0)} entries")
        print(f"   🎯 Success rate: {stats.get('success_rate', 0):.2%}")
        print(f"   💾 Model saved: {stats.get('model_saved', False)}")
        
        if not stats.get("training_successful"):
            print(f"   ⚠️ Training failed: {stats.get('reason', 'unknown')}")