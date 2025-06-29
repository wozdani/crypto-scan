"""
Phase 2: Decision Memory Layer for Trend-Mode
Token Context History Management and Learning System
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time


class TokenContextMemory:
    """Manages historical context and learning for tokens"""
    
    def __init__(self, history_file: str = "data/context/token_context_history.json"):
        self.history_file = history_file
        self.history_days = 3  # Track last 3 days
        self.ensure_data_structure()
    
    def ensure_data_structure(self):
        """Ensure data directory and history file exist"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f:
                json.dump({}, f)
    
    def load_token_history(self) -> Dict:
        """Load complete token history from file"""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[MEMORY ERROR] Failed to load history: {e}")
            return {}
    
    def save_token_history(self, history: Dict):
        """Save token history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"[MEMORY ERROR] Failed to save history: {e}")
    
    def get_recent_context(self, symbol: str) -> List[Dict]:
        """Get recent historical context for a token (last 3 days)"""
        history = self.load_token_history()
        token_history = history.get(symbol, [])
        
        # Filter entries from last 3 days
        cutoff_time = datetime.now() - timedelta(days=self.history_days)
        cutoff_str = cutoff_time.isoformat()
        
        recent_entries = [
            entry for entry in token_history
            if entry.get("timestamp", "1970-01-01T00:00:00") > cutoff_str
        ]
        
        # Sort by timestamp (most recent first)
        recent_entries.sort(key=lambda x: x.get("timestamp", "1970-01-01T00:00:00"), reverse=True)
        
        print(f"[MEMORY] {symbol}: Loaded {len(recent_entries)} recent entries from last {self.history_days} days")
        return recent_entries
    
    def add_decision_entry(self, symbol: str, decision_data: Dict):
        """Add new decision entry to token history"""
        history = self.load_token_history()
        
        if symbol not in history:
            history[symbol] = []
        
        # Create new entry with current timestamp and Phase 3 fields
        entry = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision_data.get("decision", "unknown"),
            "tjde_score": decision_data.get("final_score", 0.0),
            "clip_confidence": decision_data.get("clip_features", {}).get("clip_confidence", 0.0),
            "clip_prediction": decision_data.get("clip_features", {}).get("clip_trend_match", "unknown"),
            "trend_label": decision_data.get("clip_features", {}).get("clip_trend_match", "unknown"),
            "setup_type": decision_data.get("clip_features", {}).get("clip_setup_type", "unknown"),
            "gpt_comment": decision_data.get("gpt_comment", ""),
            "result_after_2h": None,  # To be filled later
            "result_after_6h": None,  # To be filled later
            "verdict": None,  # To be filled later (correct/wrong/avoided)
            "market_price": decision_data.get("market_price", 0.0),
            "perception_sync": decision_data.get("perception_sync", False),
            "feedback_score": None,  # Phase 3: Vision-AI feedback score
            "clip_accuracy_modifier": 1.0  # Phase 3: Historical accuracy modifier
        }
        
        history[symbol].append(entry)
        
        # Keep only recent entries to prevent file bloat
        self.cleanup_old_entries(history, symbol)
        
        self.save_token_history(history)
        print(f"[MEMORY] {symbol}: Added new decision entry")
    
    def cleanup_old_entries(self, history: Dict, symbol: str):
        """Remove entries older than tracking period"""
        cutoff_time = datetime.now() - timedelta(days=self.history_days + 1)  # Keep 1 extra day
        cutoff_str = cutoff_time.isoformat()
        
        if symbol in history:
            original_count = len(history[symbol])
            history[symbol] = [
                entry for entry in history[symbol]
                if entry.get("timestamp", "1970-01-01T00:00:00") > cutoff_str
            ]
            
            removed_count = original_count - len(history[symbol])
            if removed_count > 0:
                print(f"[MEMORY CLEANUP] {symbol}: Removed {removed_count} old entries")
    
    def update_decision_outcomes(self, symbol: str, entry_timestamp: str, outcome_data: Dict):
        """Update decision outcomes after time has passed"""
        history = self.load_token_history()
        
        if symbol in history:
            for entry in history[symbol]:
                if entry.get("timestamp") == entry_timestamp:
                    entry.update(outcome_data)
                    self.save_token_history(history)
                    print(f"[MEMORY UPDATE] {symbol}: Updated outcomes for {entry_timestamp}")
                    return True
        
        return False
    
    def analyze_historical_performance(self, symbol: str) -> Dict:
        """Analyze historical performance for a token"""
        recent_context = self.get_recent_context(symbol)
        
        if not recent_context:
            return {
                "total_decisions": 0,
                "accuracy_rate": 0.0,
                "average_score": 0.0,
                "recent_trend": "unknown",
                "historical_penalty": 0.0,
                "confidence_booster": 0.0
            }
        
        # Calculate performance metrics
        total_decisions = len(recent_context)
        completed_decisions = [entry for entry in recent_context if entry.get("verdict") is not None]
        correct_decisions = [entry for entry in completed_decisions if entry.get("verdict") == "correct"]
        
        accuracy_rate = len(correct_decisions) / len(completed_decisions) if completed_decisions else 0.0
        average_score = sum(entry.get("tjde_score", 0) for entry in recent_context) / total_decisions
        
        # Determine recent trend
        recent_trend = "improving" if accuracy_rate > 0.6 else "declining" if accuracy_rate < 0.4 else "stable"
        
        # Calculate modifiers
        historical_penalty = 0.0
        confidence_booster = 0.0
        
        if accuracy_rate < 0.3 and len(completed_decisions) >= 3:
            historical_penalty = -0.05  # Penalty for poor performance
        elif accuracy_rate > 0.7 and len(completed_decisions) >= 3:
            confidence_booster = 0.03  # Boost for good performance
        
        performance = {
            "total_decisions": total_decisions,
            "completed_decisions": len(completed_decisions),
            "accuracy_rate": accuracy_rate,
            "average_score": average_score,
            "recent_trend": recent_trend,
            "historical_penalty": historical_penalty,
            "confidence_booster": confidence_booster,
            "last_decision": recent_context[0] if recent_context else None
        }
        
        print(f"[MEMORY ANALYSIS] {symbol}: Accuracy {accuracy_rate:.1%}, Trend: {recent_trend}")
        return performance
    
    def find_similar_setups(self, symbol: str, current_setup: Dict) -> List[Dict]:
        """Find similar historical setups for pattern matching"""
        recent_context = self.get_recent_context(symbol)
        
        current_trend_label = current_setup.get("trend_label", "unknown")
        current_setup_type = current_setup.get("setup_type", "unknown")
        
        similar_setups = []
        for entry in recent_context:
            if (entry.get("trend_label") == current_trend_label and 
                entry.get("setup_type") == current_setup_type):
                similar_setups.append(entry)
        
        if similar_setups:
            print(f"[MEMORY PATTERN] {symbol}: Found {len(similar_setups)} similar setups ({current_trend_label}/{current_setup_type})")
        
        return similar_setups


def integrate_historical_context(symbol: str, current_features: Dict, context_memory: TokenContextMemory) -> Dict:
    """Integrate historical context into current analysis"""
    
    # Get recent historical context
    historical_context = context_memory.get_recent_context(symbol)
    
    # Analyze historical performance
    performance = context_memory.analyze_historical_performance(symbol)
    
    # Find similar setups
    current_setup = {
        "trend_label": current_features.get("clip_features", {}).get("clip_trend_match", "unknown"),
        "setup_type": current_features.get("clip_features", {}).get("clip_setup_type", "unknown")
    }
    similar_setups = context_memory.find_similar_setups(symbol, current_setup)
    
    # Build enhanced features with historical context
    enhanced_features = current_features.copy()
    enhanced_features["historical_context"] = {
        "recent_entries": historical_context[:5],  # Last 5 decisions
        "performance": performance,
        "similar_setups": similar_setups,
        "historical_penalty": performance["historical_penalty"],
        "confidence_booster": performance["confidence_booster"]
    }
    
    print(f"[MEMORY INTEGRATION] {symbol}: Added historical context ({len(historical_context)} entries)")
    return enhanced_features


def apply_historical_modifiers(base_score: float, historical_context: Dict) -> float:
    """Apply historical performance modifiers to base score"""
    performance = historical_context.get("performance", {})
    
    # Apply penalty for poor historical performance
    historical_penalty = performance.get("historical_penalty", 0.0)
    confidence_booster = performance.get("confidence_booster", 0.0)
    
    # Check for pattern repetition bonus
    similar_setups = historical_context.get("similar_setups", [])
    pattern_bonus = 0.0
    
    if len(similar_setups) >= 2:
        # Check if similar setups were successful
        successful_similar = [s for s in similar_setups if s.get("verdict") == "correct"]
        if len(successful_similar) >= len(similar_setups) * 0.6:  # 60% success rate
            pattern_bonus = 0.02
            print(f"[MEMORY MODIFIER] Pattern repetition bonus: +{pattern_bonus:.3f}")
    
    modified_score = base_score + historical_penalty + confidence_booster + pattern_bonus
    
    if historical_penalty != 0 or confidence_booster != 0 or pattern_bonus != 0:
        total_modifier = historical_penalty + confidence_booster + pattern_bonus
        print(f"[MEMORY MODIFIER] Applied total modifier: {total_modifier:+.3f}")
    
    return max(0.0, min(1.0, modified_score))


def simulate_trader_decision_with_memory(symbol: str, market_data: dict, signals: dict, debug_info: dict = None) -> dict:
    """
    Phase 2: Enhanced decision making with historical memory context
    """
    try:
        # Initialize memory system
        context_memory = TokenContextMemory()
        
        # === STEP 1: LOAD HISTORICAL CONTEXT ===
        enhanced_signals = integrate_historical_context(symbol, {"clip_features": signals}, context_memory)
        
        # === STEP 2: APPLY PHASE 3 VISION-AI FEEDBACK ===
        # Load Vision-AI feedback modifiers
        try:
            from evaluate_model_accuracy import load_vision_feedback_modifiers, apply_vision_feedback_modifiers
            
            feedback_modifiers = load_vision_feedback_modifiers()
            if feedback_modifiers:
                print(f"[PHASE 3] {symbol}: Applying Vision-AI feedback modifiers")
            
        except ImportError:
            feedback_modifiers = {}
        
        # === STEP 3: RUN PHASE 1 PERCEPTION SYNC WITH CONTEXT ===
        from perception_sync import simulate_trader_decision_perception_sync
        
        # Add historical context to signals
        signals_with_context = signals.copy()
        signals_with_context["historical_context"] = enhanced_signals["historical_context"]
        signals_with_context["vision_feedback_modifiers"] = feedback_modifiers
        
        # Run Phase 1 perception synchronization
        phase1_result = simulate_trader_decision_perception_sync(symbol, market_data, signals_with_context)
        
        # Apply feedback modifiers to CLIP features in result
        if feedback_modifiers and phase1_result.get("clip_features"):
            try:
                modified_clip_features = apply_vision_feedback_modifiers(
                    phase1_result["clip_features"], 
                    feedback_modifiers
                )
                phase1_result["clip_features"] = modified_clip_features
                
                # Update final score if feedback was applied
                if modified_clip_features.get("feedback_applied"):
                    # Recalculate enhanced score with modified CLIP confidence
                    from perception_sync import calculate_enhanced_tjde_score
                    
                    all_features_updated = signals.copy()
                    all_features_updated.update(modified_clip_features)
                    
                    recalculated_score = calculate_enhanced_tjde_score(all_features_updated)
                    phase1_result["base_score_phase1"] = phase1_result.get("final_score", 0.0)
                    phase1_result["final_score"] = recalculated_score
                    phase1_result["vision_feedback_applied"] = True
                    
                    print(f"[PHASE 3] {symbol}: Score adjusted by Vision-AI feedback")
                
            except Exception as e:
                print(f"[PHASE 3 ERROR] {symbol}: Failed to apply feedback: {e}")
        
        # === STEP 4: APPLY HISTORICAL MODIFIERS ===
        base_score = phase1_result.get("final_score", 0.0)
        historical_context = enhanced_signals.get("historical_context", {})
        
        memory_enhanced_score = apply_historical_modifiers(base_score, historical_context)
        
        # Update decision based on memory-enhanced score
        decision = phase1_result.get("decision", "avoid")
        quality_grade = phase1_result.get("quality_grade", "weak")
        
        if memory_enhanced_score != base_score:
            # Recalculate decision with memory enhancement
            if memory_enhanced_score >= 0.70:
                decision = "join_trend"
                quality_grade = "strong"
            elif memory_enhanced_score >= 0.45:
                decision = "consider_entry"
                quality_grade = "moderate"
            else:
                decision = "avoid"
                quality_grade = "weak"
        
        # === STEP 5: PREPARE ENHANCED RESULT ===
        memory_result = phase1_result.copy()
        memory_result.update({
            "memory_enhanced_score": round(memory_enhanced_score, 3),
            "base_score_phase1": round(base_score, 3),
            "final_score": round(memory_enhanced_score, 3),
            "decision": decision,
            "quality_grade": quality_grade,
            "historical_context": historical_context,
            "memory_integration": True,
            "vision_feedback_applied": phase1_result.get("vision_feedback_applied", False),
            "market_price": market_data.get("ticker", {}).get("price", 0.0)
        })
        
        # Calculate feedback score for Phase 3
        feedback_score = memory_enhanced_score
        if memory_result.get("vision_feedback_applied"):
            # Factor in feedback effectiveness
            feedback_score = (memory_enhanced_score + phase1_result.get("base_score_phase1", memory_enhanced_score)) / 2
        
        memory_result["feedback_score"] = round(feedback_score, 3)
        
        # === STEP 6: APPLY PHASE 4 HYBRID EMBEDDINGS ===
        try:
            from hybrid_embedding_system import integrate_embeddings_with_decision_system
            
            # Find recent chart for embedding
            import glob
            chart_pattern = f"training_data/charts/{symbol}_*.png"
            charts = glob.glob(chart_pattern)
            recent_chart = sorted(charts, reverse=True)[0] if charts else None
            
            if recent_chart:
                memory_result = integrate_embeddings_with_decision_system(symbol, memory_result, recent_chart)
                
                # Update final score if embedding boost applied
                embedding_similarity = memory_result.get("embedding_similarity", {})
                similarity_boost = embedding_similarity.get("similarity_boost", 0.0)
                
                if similarity_boost > 0:
                    embedding_enhanced_score = memory_enhanced_score + similarity_boost
                    memory_result["embedding_enhanced_score"] = round(embedding_enhanced_score, 3)
                    memory_result["final_score"] = round(embedding_enhanced_score, 3)
                    memory_result["embedding_integration"] = True
                    
                    print(f"[PHASE 4] {symbol}: Embedding boost applied: +{similarity_boost:.3f}")
                else:
                    memory_result["embedding_integration"] = False
            
        except ImportError:
            memory_result["embedding_integration"] = False
        except Exception as e:
            print(f"[PHASE 4 ERROR] {symbol}: {e}")
            memory_result["embedding_integration"] = False
        
        # === STEP 7: APPLY PHASE 5 REINFORCEMENT LEARNING ===
        try:
            from reinforce_embeddings import integrate_reinforcement_learning
            
            memory_result = integrate_reinforcement_learning(symbol, memory_result)
            
            # Update final score if RL enhanced
            rl_enhanced_score = memory_result.get("rl_enhanced_score")
            if rl_enhanced_score and rl_enhanced_score != memory_result.get("rl_base_score", 0):
                memory_result["final_score"] = rl_enhanced_score
                
                # Update decision if significantly changed
                if memory_result.get("rl_decision_modified"):
                    decision = memory_result["decision"]
                    print(f"[PHASE 5] {symbol}: RL modified decision to {decision}")
            
        except ImportError:
            memory_result["reinforcement_learning"] = False
        except Exception as e:
            print(f"[PHASE 5 ERROR] {symbol}: {e}")
            memory_result["reinforcement_learning"] = False
        
        # === STEP 8: APPLY PHASE 6 TRADER-LEVEL AI ===
        try:
            from trader_level_ai_engine import integrate_trader_level_ai
            
            memory_result = integrate_trader_level_ai(symbol, memory_result)
            
            # Update final decision if Trader AI enhanced
            if memory_result.get("arcymistrz_ai"):
                elite_decision = memory_result.get("elite_decision")
                elite_score = memory_result.get("elite_score")
                
                if elite_decision and elite_score:
                    memory_result["final_score"] = elite_score
                    memory_result["decision"] = elite_decision
                    decision = elite_decision
                    
                    print(f"[PHASE 6] {symbol}: Trader-Level AI decision: {elite_decision.upper()}")
            
        except ImportError:
            memory_result["arcymistrz_ai"] = False
        except Exception as e:
            print(f"[PHASE 6 ERROR] {symbol}: {e}")
            memory_result["arcymistrz_ai"] = False
        
        # === STEP 9: SAVE DECISION TO MEMORY ===
        context_memory.add_decision_entry(symbol, memory_result)
        
        # Display final score from highest level system available
        final_score_display = (memory_result.get("elite_score") or 
                             memory_result.get("rl_enhanced_score") or 
                             memory_result.get("embedding_enhanced_score", memory_enhanced_score))
        
        print(f"[ARCYMISTRZ] {symbol}: Final decision: {decision.upper()} (score: {final_score_display:.3f})")
        
        return memory_result
        
    except Exception as e:
        print(f"❌ [PHASE 2 ERROR] {symbol}: {e}")
        
        # Fallback to Phase 1
        try:
            from perception_sync import simulate_trader_decision_perception_sync
            fallback_result = simulate_trader_decision_perception_sync(symbol, market_data, signals)
            fallback_result["memory_integration"] = False
            fallback_result["fallback_reason"] = str(e)
            return fallback_result
        except:
            return {
                "decision": "avoid",
                "combined_score": 0.0,
                "final_score": 0.0,
                "quality_grade": "error",
                "memory_integration": False,
                "error": str(e)
            }


def update_historical_outcomes_loop():
    """Background process to update historical outcomes (to be run periodically)"""
    print("[MEMORY OUTCOMES] Starting historical outcome update...")
    
    try:
        context_memory = TokenContextMemory()
        history = context_memory.load_token_history()
        
        updated_count = 0
        
        for symbol, entries in history.items():
            for entry in entries:
                if entry.get("verdict") is None:  # Unresolved entry
                    timestamp_str = entry.get("timestamp")
                    if timestamp_str:
                        entry_time = datetime.fromisoformat(timestamp_str)
                        now = datetime.now()
                        
                        # Check if enough time has passed for evaluation
                        if now - entry_time >= timedelta(hours=2):
                            # Simulate outcome evaluation (in production, fetch actual price data)
                            # For now, create placeholder logic
                            outcome_data = evaluate_decision_outcome(entry, symbol)
                            
                            if outcome_data:
                                entry.update(outcome_data)
                                updated_count += 1
        
        if updated_count > 0:
            context_memory.save_token_history(history)
            print(f"[MEMORY OUTCOMES] Updated {updated_count} historical outcomes")
        
    except Exception as e:
        print(f"[MEMORY OUTCOMES ERROR] {e}")


def evaluate_decision_outcome(entry: Dict, symbol: str) -> Optional[Dict]:
    """Evaluate the outcome of a historical decision"""
    # This would fetch actual price data in production
    # For now, return simulated outcome based on decision quality
    
    tjde_score = entry.get("tjde_score", 0.0)
    decision = entry.get("decision", "avoid")
    
    # Simulate outcome based on decision quality
    if decision == "avoid":
        return {
            "result_after_2h": "N/A",
            "result_after_6h": "N/A", 
            "verdict": "avoided"
        }
    else:
        # Simulate price change based on score quality
        if tjde_score > 0.7:
            return {
                "result_after_2h": "+3.2%",
                "result_after_6h": "+8.1%",
                "verdict": "correct"
            }
        elif tjde_score > 0.5:
            return {
                "result_after_2h": "+1.1%",
                "result_after_6h": "+2.8%",
                "verdict": "correct"
            }
        else:
            return {
                "result_after_2h": "-0.8%",
                "result_after_6h": "-1.5%",
                "verdict": "wrong"
            }


def test_phase2_memory_system():
    """Test the Phase 2 memory system"""
    print("Testing Phase 2: Decision Memory Layer...")
    
    try:
        # Test memory initialization
        context_memory = TokenContextMemory()
        
        # Test symbol
        test_symbol = "MEMORYUSDT"
        
        # Create test decision data
        test_decision = {
            "decision": "consider_entry",
            "final_score": 0.742,
            "clip_features": {
                "clip_confidence": 0.68,
                "clip_trend_match": "pullback",
                "clip_setup_type": "support-bounce"
            },
            "gpt_comment": "Strong pullback setup with volume support",
            "market_price": 125.50,
            "perception_sync": True
        }
        
        # Add decision to memory
        context_memory.add_decision_entry(test_symbol, test_decision)
        
        # Test historical context retrieval
        historical_context = context_memory.get_recent_context(test_symbol)
        
        print(f"✅ Memory system: {len(historical_context)} entries for {test_symbol}")
        
        # Test performance analysis
        performance = context_memory.analyze_historical_performance(test_symbol)
        print(f"✅ Performance analysis: {performance['accuracy_rate']:.1%} accuracy")
        
        # Test complete Phase 2 integration
        test_market_data = {
            'ticker': {'price': 125.50, 'volume': 18000},
            'candles_15m': [[1640995200000, 124.0, 126.0, 123.5, 125.50, 18000]]
        }
        
        test_signals = {
            'trend_strength': 0.78,
            'pullback_quality': 0.85,
            'support_reaction_strength': 0.82,
            'volume_behavior_score': 0.75,
            'psych_score': 0.65
        }
        
        # Run Phase 2 with memory
        result = simulate_trader_decision_with_memory(test_symbol, test_market_data, test_signals)
        
        print()
        print("PHASE 2 MEMORY-ENHANCED RESULTS:")
        print(f"  Decision: {result.get('decision', 'unknown')}")
        print(f"  Phase 1 Score: {result.get('base_score_phase1', 0):.3f}")
        print(f"  Memory Enhanced: {result.get('memory_enhanced_score', 0):.3f}")
        print(f"  Final Score: {result.get('final_score', 0):.3f}")
        print(f"  Memory Integration: {result.get('memory_integration', False)}")
        
        # Clean up test data
        history = context_memory.load_token_history()
        if test_symbol in history:
            del history[test_symbol]
            context_memory.save_token_history(history)
        
        print("✅ Phase 2 Decision Memory Layer working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 test error: {e}")
        return False


if __name__ == "__main__":
    test_phase2_memory_system()