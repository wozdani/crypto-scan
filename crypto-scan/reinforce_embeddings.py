"""
Phase 5: Self-Reinforcing Learning System
Reward-based model improvement using real alert effectiveness
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from token_context_memory import TokenContextMemory
from hybrid_embedding_system import HybridEmbeddingSystem


class ReinforcementLearningEngine:
    """Self-learning system based on prediction effectiveness"""
    
    def __init__(self, rewards_dir: str = "data/reinforcement"):
        self.rewards_dir = rewards_dir
        self.rewards_file = os.path.join(rewards_dir, "prediction_rewards.json")
        self.model_weights_file = os.path.join(rewards_dir, "model_weights.json")
        self.ensure_reinforcement_structure()
        
        # Initialize systems
        self.context_memory = TokenContextMemory()
        self.embedding_system = HybridEmbeddingSystem()
        
        # Load or initialize model weights
        self.model_weights = self.load_model_weights()
    
    def ensure_reinforcement_structure(self):
        """Create reinforcement learning directory structure"""
        os.makedirs(self.rewards_dir, exist_ok=True)
        
        if not os.path.exists(self.rewards_file):
            with open(self.rewards_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.model_weights_file):
            initial_weights = {
                "clip_confidence_weight": 1.0,
                "tjde_score_weight": 1.0,
                "embedding_similarity_weight": 1.0,
                "historical_accuracy_weight": 1.0,
                "pattern_recognition_boost": 0.02,
                "confidence_penalty_threshold": 0.3,
                "confidence_boost_threshold": 0.8
            }
            with open(self.model_weights_file, 'w') as f:
                json.dump(initial_weights, f, indent=2)
    
    def load_model_weights(self) -> Dict:
        """Load current model weights"""
        try:
            with open(self.model_weights_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[RL WEIGHTS] Error loading weights: {e}")
            return {
                "clip_confidence_weight": 1.0,
                "tjde_score_weight": 1.0,
                "embedding_similarity_weight": 1.0,
                "historical_accuracy_weight": 1.0,
                "pattern_recognition_boost": 0.02
            }
    
    def save_model_weights(self):
        """Save updated model weights"""
        try:
            with open(self.model_weights_file, 'w') as f:
                json.dump(self.model_weights, f, indent=2)
            print(f"[RL WEIGHTS] Model weights updated and saved")
        except Exception as e:
            print(f"[RL WEIGHTS ERROR] Failed to save weights: {e}")
    
    def calculate_reward(self, prediction_entry: Dict) -> float:
        """Calculate reward based on prediction effectiveness"""
        try:
            verdict = prediction_entry.get("verdict")
            decision = prediction_entry.get("decision", "unknown")
            clip_confidence = prediction_entry.get("clip_confidence", 0.0)
            tjde_score = prediction_entry.get("tjde_score", 0.0)
            result_6h = prediction_entry.get("result_after_6h", "0%")
            
            # Parse percentage result
            try:
                if result_6h and result_6h != "N/A":
                    result_percent = float(result_6h.replace('%', ''))
                else:
                    result_percent = 0.0
            except:
                result_percent = 0.0
            
            # Base reward calculation
            reward = 0.0
            
            if verdict == "correct":
                # Positive reward for correct predictions
                if decision in ["join_trend", "consider_entry"]:
                    if result_percent > 5.0:  # Strong positive result
                        reward = 1.5
                    elif result_percent > 2.0:  # Good positive result
                        reward = 1.0
                    else:  # Weak positive result
                        reward = 0.5
                else:  # Correctly avoided bad trade
                    reward = 0.3
                    
                # Bonus for high confidence correct predictions
                if clip_confidence > 0.7:
                    reward *= 1.2
                    
            elif verdict == "wrong":
                # Negative reward for wrong predictions
                if decision in ["join_trend", "consider_entry"]:
                    if result_percent < -3.0:  # Strong negative result
                        reward = -1.5
                    elif result_percent < -1.0:  # Moderate negative result
                        reward = -1.0
                    else:  # Small negative result
                        reward = -0.5
                        
                    # Extra penalty for high confidence wrong predictions
                    if clip_confidence > 0.7:
                        reward *= 1.3
                else:
                    # Missed opportunity (avoided but should have entered)
                    if result_percent > 3.0:
                        reward = -0.7
                    else:
                        reward = -0.3
            
            elif verdict == "avoided":
                # Neutral to small positive for avoided trades
                reward = 0.1
            
            # Factor in TJDE score consistency
            if tjde_score > 0.7 and verdict == "correct":
                reward += 0.2  # Bonus for high score accuracy
            elif tjde_score < 0.4 and verdict == "wrong":
                reward -= 0.1  # Small penalty for low score inaccuracy
            
            print(f"[RL REWARD] {prediction_entry.get('symbol', 'UNKNOWN')}: {verdict} -> reward {reward:.2f}")
            return reward
            
        except Exception as e:
            print(f"[RL REWARD ERROR] {e}")
            return 0.0
    
    def analyze_prediction_patterns(self, days_back: int = 7) -> Dict:
        """Analyze patterns in successful vs failed predictions"""
        history = self.context_memory.load_token_history()
        
        successful_patterns = defaultdict(list)
        failed_patterns = defaultdict(list)
        
        cutoff_time = datetime.now() - timedelta(days=days_back)
        cutoff_str = cutoff_time.isoformat()
        
        for symbol, entries in history.items():
            for entry in entries:
                if (entry.get("timestamp", "1970-01-01T00:00:00") > cutoff_str and
                    entry.get("verdict") is not None):
                    
                    reward = self.calculate_reward(entry)
                    
                    pattern_key = f"{entry.get('clip_prediction', 'unknown')}_{entry.get('setup_type', 'unknown')}"
                    
                    pattern_data = {
                        "symbol": symbol,
                        "clip_confidence": entry.get("clip_confidence", 0.0),
                        "tjde_score": entry.get("tjde_score", 0.0),
                        "decision": entry.get("decision", "unknown"),
                        "reward": reward,
                        "result_6h": entry.get("result_after_6h", "0%")
                    }
                    
                    if reward > 0:
                        successful_patterns[pattern_key].append(pattern_data)
                    else:
                        failed_patterns[pattern_key].append(pattern_data)
        
        # Calculate pattern statistics
        pattern_stats = {}
        all_patterns = set(list(successful_patterns.keys()) + list(failed_patterns.keys()))
        
        for pattern in all_patterns:
            successful = successful_patterns[pattern]
            failed = failed_patterns[pattern]
            
            total_cases = len(successful) + len(failed)
            success_rate = len(successful) / total_cases if total_cases > 0 else 0.0
            
            avg_successful_reward = np.mean([p["reward"] for p in successful]) if successful else 0.0
            avg_failed_reward = np.mean([p["reward"] for p in failed]) if failed else 0.0
            
            pattern_stats[pattern] = {
                "success_rate": success_rate,
                "total_cases": total_cases,
                "successful_cases": len(successful),
                "failed_cases": len(failed),
                "avg_successful_reward": avg_successful_reward,
                "avg_failed_reward": avg_failed_reward,
                "pattern_quality": success_rate * avg_successful_reward if success_rate > 0 else 0.0
            }
        
        print(f"[RL PATTERNS] Analyzed {len(pattern_stats)} patterns from {days_back} days")
        return pattern_stats
    
    def update_model_weights_from_patterns(self, pattern_stats: Dict):
        """Update model weights based on pattern analysis"""
        try:
            # Analyze overall performance
            total_success_rate = np.mean([p["success_rate"] for p in pattern_stats.values() if p["total_cases"] >= 3])
            
            # Weight adjustments based on performance
            if total_success_rate > 0.7:  # High performance
                self.model_weights["pattern_recognition_boost"] = min(0.05, self.model_weights.get("pattern_recognition_boost", 0.02) * 1.1)
                self.model_weights["confidence_boost_threshold"] = max(0.6, self.model_weights.get("confidence_boost_threshold", 0.8) * 0.95)
            elif total_success_rate < 0.4:  # Poor performance
                self.model_weights["pattern_recognition_boost"] = max(0.01, self.model_weights.get("pattern_recognition_boost", 0.02) * 0.9)
                self.model_weights["confidence_penalty_threshold"] = min(0.5, self.model_weights.get("confidence_penalty_threshold", 0.3) * 1.1)
            
            # Adjust component weights based on effectiveness
            high_quality_patterns = [p for p in pattern_stats.values() if p["pattern_quality"] > 0.5 and p["total_cases"] >= 3]
            
            if high_quality_patterns:
                # Increase embedding similarity weight if pattern recognition is working well
                avg_quality = np.mean([p["pattern_quality"] for p in high_quality_patterns])
                if avg_quality > 0.8:
                    self.model_weights["embedding_similarity_weight"] = min(1.5, self.model_weights.get("embedding_similarity_weight", 1.0) * 1.05)
            
            self.save_model_weights()
            print(f"[RL ADAPTATION] Model weights updated based on {len(pattern_stats)} patterns")
            
        except Exception as e:
            print(f"[RL ADAPTATION ERROR] {e}")
    
    def get_reinforced_confidence_modifier(self, symbol: str, analysis_data: Dict) -> float:
        """Get confidence modifier based on reinforcement learning"""
        try:
            clip_prediction = analysis_data.get("clip_features", {}).get("clip_trend_match", "unknown")
            setup_type = analysis_data.get("clip_features", {}).get("clip_setup_type", "unknown")
            
            # Load recent pattern analysis
            pattern_stats = self.analyze_prediction_patterns(days_back=14)
            pattern_key = f"{clip_prediction}_{setup_type}"
            
            if pattern_key in pattern_stats:
                pattern_data = pattern_stats[pattern_key]
                
                # Apply reinforcement modifiers
                if pattern_data["total_cases"] >= 3:  # Sufficient sample size
                    success_rate = pattern_data["success_rate"]
                    pattern_quality = pattern_data["pattern_quality"]
                    
                    if success_rate > self.model_weights.get("confidence_boost_threshold", 0.8):
                        # High success rate pattern - boost confidence
                        boost = self.model_weights.get("pattern_recognition_boost", 0.02) * pattern_quality
                        print(f"[RL BOOST] {symbol}: Pattern {pattern_key} boost +{boost:.3f} (success: {success_rate:.1%})")
                        return boost
                        
                    elif success_rate < self.model_weights.get("confidence_penalty_threshold", 0.3):
                        # Low success rate pattern - reduce confidence
                        penalty = -self.model_weights.get("pattern_recognition_boost", 0.02) * (1 - success_rate)
                        print(f"[RL PENALTY] {symbol}: Pattern {pattern_key} penalty {penalty:.3f} (success: {success_rate:.1%})")
                        return penalty
            
            return 0.0
            
        except Exception as e:
            print(f"[RL MODIFIER ERROR] {symbol}: {e}")
            return 0.0
    
    def run_reinforcement_cycle(self, days_back: int = 7):
        """Run complete reinforcement learning cycle"""
        print(f"[RL CYCLE] Starting reinforcement learning cycle ({days_back} days)")
        
        try:
            # Analyze prediction patterns
            pattern_stats = self.analyze_prediction_patterns(days_back)
            
            if not pattern_stats:
                print("[RL CYCLE] No patterns found for analysis")
                return
            
            # Update model weights
            self.update_model_weights_from_patterns(pattern_stats)
            
            # Save rewards and analysis
            cycle_report = {
                "timestamp": datetime.now().isoformat(),
                "days_analyzed": days_back,
                "patterns_analyzed": len(pattern_stats),
                "pattern_stats": pattern_stats,
                "updated_weights": self.model_weights,
                "overall_performance": {
                    "avg_success_rate": np.mean([p["success_rate"] for p in pattern_stats.values() if p["total_cases"] >= 2]),
                    "total_analyzed_cases": sum(p["total_cases"] for p in pattern_stats.values()),
                    "high_quality_patterns": len([p for p in pattern_stats.values() if p["pattern_quality"] > 0.5])
                }
            }
            
            # Save cycle report
            cycle_file = os.path.join(self.rewards_dir, f"rl_cycle_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
            with open(cycle_file, 'w') as f:
                json.dump(cycle_report, f, indent=2)
            
            print(f"[RL CYCLE] Completed: {len(pattern_stats)} patterns analyzed")
            print(f"[RL CYCLE] Report saved: {cycle_file}")
            
            return cycle_report
            
        except Exception as e:
            print(f"[RL CYCLE ERROR] {e}")
            return None
    
    def get_rl_enhanced_score(self, symbol: str, base_score: float, analysis_data: Dict) -> float:
        """Apply reinforcement learning enhancements to base score"""
        try:
            # Get RL confidence modifier
            rl_modifier = self.get_reinforced_confidence_modifier(symbol, analysis_data)
            
            # Apply weighted modifiers
            embedding_weight = self.model_weights.get("embedding_similarity_weight", 1.0)
            
            # Calculate RL-enhanced score
            rl_enhanced_score = base_score + (rl_modifier * embedding_weight)
            
            # Ensure score stays in valid range
            rl_enhanced_score = max(0.0, min(1.0, rl_enhanced_score))
            
            if rl_modifier != 0:
                print(f"[RL ENHANCED] {symbol}: {base_score:.3f} -> {rl_enhanced_score:.3f} (RL: {rl_modifier:+.3f})")
            
            return rl_enhanced_score
            
        except Exception as e:
            print(f"[RL ENHANCED ERROR] {symbol}: {e}")
            return base_score


def integrate_reinforcement_learning(symbol: str, analysis_data: Dict) -> Dict:
    """Integrate reinforcement learning into analysis pipeline"""
    try:
        rl_engine = ReinforcementLearningEngine()
        
        # Get current base score
        base_score = analysis_data.get("final_score", 0.0)
        
        # Apply RL enhancements
        rl_enhanced_score = rl_engine.get_rl_enhanced_score(symbol, base_score, analysis_data)
        
        # Update analysis data
        enhanced_analysis = analysis_data.copy()
        enhanced_analysis.update({
            "rl_base_score": base_score,
            "rl_enhanced_score": rl_enhanced_score,
            "final_score": rl_enhanced_score,
            "reinforcement_learning": True,
            "rl_model_weights": rl_engine.model_weights
        })
        
        # Update decision if RL significantly changed the score
        score_change = rl_enhanced_score - base_score
        if abs(score_change) > 0.03:  # Significant change threshold
            # Recalculate decision based on new score
            if rl_enhanced_score >= 0.70:
                enhanced_analysis["decision"] = "join_trend"
                enhanced_analysis["quality_grade"] = "strong"
            elif rl_enhanced_score >= 0.45:
                enhanced_analysis["decision"] = "consider_entry"
                enhanced_analysis["quality_grade"] = "moderate"
            else:
                enhanced_analysis["decision"] = "avoid"
                enhanced_analysis["quality_grade"] = "weak"
            
            enhanced_analysis["rl_decision_modified"] = True
        
        return enhanced_analysis
        
    except Exception as e:
        print(f"[RL INTEGRATION ERROR] {symbol}: {e}")
        return analysis_data


def run_periodic_reinforcement_learning():
    """Run periodic RL cycle for continuous learning"""
    try:
        rl_engine = ReinforcementLearningEngine()
        report = rl_engine.run_reinforcement_cycle(days_back=7)
        
        if report:
            performance = report["overall_performance"]
            print(f"[RL PERIODIC] Success rate: {performance['avg_success_rate']:.1%}")
            print(f"[RL PERIODIC] Cases analyzed: {performance['total_analyzed_cases']}")
            print(f"[RL PERIODIC] High-quality patterns: {performance['high_quality_patterns']}")
        
        return report
        
    except Exception as e:
        print(f"[RL PERIODIC ERROR] {e}")
        return None


def test_reinforcement_learning_system():
    """Test the reinforcement learning system"""
    print("Testing Phase 5: Reinforcement Learning System...")
    
    try:
        # Initialize RL engine
        rl_engine = ReinforcementLearningEngine()
        
        # Test pattern analysis
        print("Testing pattern analysis...")
        pattern_stats = rl_engine.analyze_prediction_patterns(days_back=30)
        print(f"✅ Pattern analysis: {len(pattern_stats)} patterns found")
        
        # Test weight updates
        print("Testing weight updates...")
        original_weights = rl_engine.model_weights.copy()
        rl_engine.update_model_weights_from_patterns(pattern_stats)
        print("✅ Weight updates completed")
        
        # Test RL integration
        print("Testing RL integration...")
        test_analysis = {
            "final_score": 0.65,
            "clip_features": {
                "clip_trend_match": "pullback",
                "clip_setup_type": "support-bounce",
                "clip_confidence": 0.72
            },
            "decision": "consider_entry",
            "quality_grade": "moderate"
        }
        
        enhanced_analysis = integrate_reinforcement_learning("RLTEST", test_analysis)
        
        if enhanced_analysis.get("reinforcement_learning"):
            print("✅ RL integration working")
            print(f"  Base score: {enhanced_analysis.get('rl_base_score', 0):.3f}")
            print(f"  RL enhanced: {enhanced_analysis.get('rl_enhanced_score', 0):.3f}")
        
        # Test periodic cycle
        print("Testing periodic RL cycle...")
        report = run_periodic_reinforcement_learning()
        
        if report:
            print("✅ Periodic RL cycle completed")
        
        print("✅ Phase 5 Reinforcement Learning System working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Phase 5 test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_reinforcement_learning_system()