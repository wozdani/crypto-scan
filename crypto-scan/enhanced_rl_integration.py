#!/usr/bin/env python3
"""
🧠 ENHANCED RL INTEGRATION - DQN + RLAgentV3 Combined System
Integracja zaawansowanego Deep Q-Network z istniejącym RLAgentV3 system
"""

import os
import json
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Import istniejących systemów
from rl_agent_v3 import RLAgentV3
from dqn_agent import DQNCryptoIntegration

# Import feedback loop z względną ścieżką
try:
    from feedback_loop import TJDEFeedbackLoop
    FEEDBACK_LOOP_AVAILABLE = True
except ImportError:
    print("[ENHANCED RL] Traditional feedback loop not available, using simplified version")
    FEEDBACK_LOOP_AVAILABLE = False


class EnhancedRLSystem:
    """
    🎯 Unified system łączący:
    - RLAgentV3 (adaptive booster weights)
    - DQN Agent (dynamic threshold adaptation)
    - Feedback Loop (learning z outcomes)
    """
    
    def __init__(self):
        print("[ENHANCED RL] Initializing unified RL system...")
        
        # Initialize istniejące systemy
        self.rl_agent_v3 = RLAgentV3(
            booster_names=("gnn", "whaleClip", "dexInflow", "whale_ping", "volume_spike"),
            learning_rate=0.05,
            weight_path="crypto-scan/cache/rl_agent_v3_stealth_weights.json"
        )
        
        # Initialize DQN system
        self.dqn_integration = DQNCryptoIntegration()
        
        # Initialize traditional feedback loop if available
        if FEEDBACK_LOOP_AVAILABLE:
            self.feedback_loop = TJDEFeedbackLoop()
        else:
            self.feedback_loop = None
        
        # Enhanced state tracking
        self.alert_outcomes = []
        self.performance_metrics = {
            'total_alerts': 0,
            'successful_alerts': 0,
            'avg_profit': 0.0,
            'threshold_adaptations': 0,
            'weight_adaptations': 0
        }
        
        # Threading dla background processes
        self.training_thread = None
        self.should_train = True
        
        print("[ENHANCED RL] System initialized successfully")
        self._log_system_status()
    
    def _log_system_status(self):
        """Log current system status"""
        print(f"[ENHANCED RL STATUS]")
        print(f"  • RLAgentV3 weights: {self.rl_agent_v3.weights}")
        print(f"  • DQN threshold: {self.dqn_integration.agent.current_threshold:.3f}")
        print(f"  • Training episodes: {len(self.dqn_integration.agent.memory)}")
        if self.feedback_loop:
            print(f"  • Feedback data: {len(self.feedback_loop.feedback_data)} entries")
        else:
            print(f"  • Feedback loop: Simplified mode")
    
    def process_stealth_detection(self, symbol: str, stealth_data: Dict[str, Any], 
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 Main integration point - process stealth detection z unified RL intelligence
        
        Args:
            symbol: Token symbol
            stealth_data: Wyniki stealth engine (signals, scores, etc.)
            market_data: Market data dla context
            
        Returns:
            Enhanced decision z adaptive thresholds i weights
        """
        print(f"[ENHANCED RL] Processing {symbol}...")
        
        # Extract components dla RLAgentV3
        booster_inputs = {
            'gnn': stealth_data.get('diamond_score', 0.0),
            'whaleClip': stealth_data.get('whaleclip_score', 0.0),
            'dexInflow': min(stealth_data.get('dex_inflow', 0.0) / 1000.0, 3.0),  # Normalize
            'whale_ping': stealth_data.get('whale_ping', 0.0),
            'volume_spike': stealth_data.get('volume_spike', 0.0)
        }
        
        # Compute weighted score z RLAgentV3
        weighted_score = self.rl_agent_v3.compute_final_score(booster_inputs)
        
        # Get adaptive threshold z DQN
        current_threshold = self.dqn_integration.agent.current_threshold
        
        # Enhanced decision logic
        decision_data = {
            'symbol': symbol,
            'base_score': stealth_data.get('score', 0.0),
            'weighted_score': weighted_score,
            'adaptive_threshold': current_threshold,
            'booster_contributions': booster_inputs,
            'raw_stealth_data': stealth_data,
            'market_context': market_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Make decision based na adaptive threshold
        if weighted_score >= current_threshold:
            decision_data['decision'] = 'BUY'
            decision_data['confidence'] = min(1.0, weighted_score / current_threshold)
        else:
            decision_data['decision'] = 'HOLD'
            decision_data['confidence'] = weighted_score / current_threshold
        
        print(f"[ENHANCED RL] {symbol}: Score={weighted_score:.3f}, Threshold={current_threshold:.3f}, Decision={decision_data['decision']}")
        
        return decision_data
    
    def record_alert_outcome(self, symbol: str, alert_data: Dict[str, Any], 
                           outcome: bool, pump_percentage: float = 0.0,
                           time_to_pump_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        🎯 Record alert outcome i trigger learning w obu systemach
        
        Args:
            symbol: Token symbol
            alert_data: Original alert data z process_stealth_detection()
            outcome: True = pump occurred, False = no pump
            pump_percentage: Percentage pump (if any)
            time_to_pump_minutes: Minutes do pump (if occurred)
            
        Returns:
            Learning results z obu systemów
        """
        print(f"[ENHANCED RL] Recording outcome for {symbol}: {'SUCCESS' if outcome else 'FAILED'}")
        
        # Update performance metrics
        self.performance_metrics['total_alerts'] += 1
        if outcome:
            self.performance_metrics['successful_alerts'] += 1
            self.performance_metrics['avg_profit'] = (
                self.performance_metrics['avg_profit'] * (self.performance_metrics['successful_alerts'] - 1) + pump_percentage
            ) / self.performance_metrics['successful_alerts']
        
        # Prepare reward dla systems
        reward = self._calculate_unified_reward(outcome, pump_percentage, time_to_pump_minutes)
        
        # Update RLAgentV3 weights
        rl_v3_result = self._update_rl_agent_v3(alert_data, reward)
        
        # Update DQN system
        dqn_result = self._update_dqn_system(alert_data, outcome, reward)
        
        # Update traditional feedback loop
        feedback_result = self._update_feedback_loop(symbol, alert_data, outcome, pump_percentage)
        
        # Store outcome dla analysis
        outcome_record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'alert_data': alert_data,
            'outcome': outcome,
            'pump_percentage': pump_percentage,
            'time_to_pump_minutes': time_to_pump_minutes,
            'reward': reward,
            'rl_v3_update': rl_v3_result,
            'dqn_update': dqn_result,
            'feedback_update': feedback_result
        }
        
        self.alert_outcomes.append(outcome_record)
        
        # Periodic model saving
        if len(self.alert_outcomes) % 10 == 0:
            self._save_models()
        
        print(f"[ENHANCED RL] Learning completed for {symbol}")
        return outcome_record
    
    def _calculate_unified_reward(self, outcome: bool, pump_percentage: float, 
                                time_to_pump_minutes: Optional[int]) -> float:
        """Calculate sophisticated reward dla learning systems"""
        if not outcome:
            return -5.0  # Penalty za false positive
        
        base_reward = 10.0  # Base reward za correct prediction
        
        # Pump magnitude bonus
        if pump_percentage > 0:
            magnitude_bonus = min(pump_percentage * 2.0, 20.0)  # Max 20 points
            base_reward += magnitude_bonus
        
        # Time bonus - earlier detection = higher reward
        if time_to_pump_minutes is not None:
            time_bonus = max(0, (120 - time_to_pump_minutes) / 120 * 10.0)  # Max 10 points
            base_reward += time_bonus
        
        return base_reward
    
    def _update_rl_agent_v3(self, alert_data: Dict[str, Any], reward: float) -> Dict[str, Any]:
        """Update RLAgentV3 weights based na outcome"""
        try:
            booster_inputs = alert_data['booster_contributions']
            
            # Update weights
            self.rl_agent_v3.update_weights(
                inputs=booster_inputs,
                reward=reward,
                alert_metadata=alert_data
            )
            
            self.performance_metrics['weight_adaptations'] += 1
            
            return {
                'status': 'success',
                'new_weights': self.rl_agent_v3.weights.copy(),
                'update_count': self.rl_agent_v3.update_count,
                'success_rate': self.rl_agent_v3.successful_alerts / max(1, self.rl_agent_v3.update_count)
            }
        except Exception as e:
            print(f"[ENHANCED RL ERROR] RLAgentV3 update failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _update_dqn_system(self, alert_data: Dict[str, Any], outcome: bool, reward: float) -> Dict[str, Any]:
        """Update DQN system z alert outcome"""
        try:
            # Extract detector performance
            detector_performance = {
                'stealth_engine': alert_data['base_score'] / 5.0,  # Normalize
                'weighted_score': alert_data['weighted_score'] / 3.0  # Normalize
            }
            
            # Process outcome z DQN integration
            changes = self.dqn_integration.process_alert_outcome(
                symbol=alert_data['symbol'],
                score=alert_data['weighted_score'],
                alert_sent=True,
                outcome=outcome,
                detector_performance=detector_performance,
                market_data=alert_data['market_context']
            )
            
            if 'threshold' in changes:
                self.performance_metrics['threshold_adaptations'] += 1
            
            return {
                'status': 'success',
                'changes': changes,
                'new_threshold': self.dqn_integration.agent.current_threshold,
                'epsilon': self.dqn_integration.agent.epsilon
            }
        except Exception as e:
            print(f"[ENHANCED RL ERROR] DQN update failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _update_feedback_loop(self, symbol: str, alert_data: Dict[str, Any], 
                            outcome: bool, pump_percentage: float) -> Dict[str, Any]:
        """Update traditional feedback loop"""
        if not self.feedback_loop:
            return {'status': 'skipped', 'reason': 'feedback_loop_not_available'}
            
        try:
            # Record w traditional feedback system
            self.feedback_loop.record_alert_result(
                symbol=symbol,
                phase="stealth_v3",
                score=alert_data['weighted_score'],
                decision=alert_data['decision'],
                score_components=alert_data['booster_contributions'],
                entry_price=alert_data['market_context'].get('price', 0.0),
                alert_time=alert_data['timestamp'],
                was_successful=outcome,
                profit_loss_pct=pump_percentage if outcome else 0.0
            )
            
            return {'status': 'success', 'recorded': True}
        except Exception as e:
            print(f"[ENHANCED RL ERROR] Feedback loop update failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _save_models(self):
        """Save all model states"""
        try:
            # Save RLAgentV3
            self.rl_agent_v3.save_weights()
            
            # Save DQN model
            self.dqn_integration.agent.save_model()
            
            # Save performance metrics
            metrics_path = "crypto-scan/cache/enhanced_rl_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            
            print(f"[ENHANCED RL] Models saved successfully")
        except Exception as e:
            print(f"[ENHANCED RL ERROR] Model saving failed: {e}")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current system parameters dla debugging"""
        return {
            'rl_agent_v3_weights': self.rl_agent_v3.weights,
            'dqn_threshold': self.dqn_integration.agent.current_threshold,
            'dqn_epsilon': self.dqn_integration.agent.epsilon,
            'performance_metrics': self.performance_metrics,
            'total_outcomes': len(self.alert_outcomes),
            'memory_size': len(self.dqn_integration.agent.memory)
        }
    
    def start_background_training(self):
        """Start background training thread"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(target=self._background_training_loop)
            self.training_thread.daemon = True
            self.training_thread.start()
            print("[ENHANCED RL] Background training started")
    
    def _background_training_loop(self):
        """Background training loop dla continuous improvement"""
        while self.should_train:
            try:
                # Train DQN co 5 minutes
                if len(self.dqn_integration.agent.memory) >= 64:
                    loss = self.dqn_integration.agent.replay()
                    if loss > 0:
                        print(f"[ENHANCED RL BACKGROUND] DQN training loss: {loss:.4f}")
                
                # Sleep 5 minutes
                time.sleep(300)
                
            except Exception as e:
                print(f"[ENHANCED RL ERROR] Background training error: {e}")
                time.sleep(60)
    
    def stop_background_training(self):
        """Stop background training"""
        self.should_train = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=10)
        print("[ENHANCED RL] Background training stopped")


# 🎯 Global instance dla easy access
_enhanced_rl_system = None

def get_enhanced_rl_system() -> EnhancedRLSystem:
    """Get global enhanced RL system instance"""
    global _enhanced_rl_system
    if _enhanced_rl_system is None:
        _enhanced_rl_system = EnhancedRLSystem()
        _enhanced_rl_system.start_background_training()
    return _enhanced_rl_system

def process_stealth_with_enhanced_rl(symbol: str, stealth_data: Dict[str, Any], 
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    🎯 Main entry point dla stealth detection z Enhanced RL
    Use this w stealth_engine.py instead of basic scoring
    """
    system = get_enhanced_rl_system()
    return system.process_stealth_detection(symbol, stealth_data, market_data)

def record_enhanced_rl_outcome(symbol: str, alert_data: Dict[str, Any], 
                             outcome: bool, pump_percentage: float = 0.0,
                             time_to_pump_minutes: Optional[int] = None) -> Dict[str, Any]:
    """
    🎯 Record alert outcome dla Enhanced RL learning
    Use this w feedback mechanisms
    """
    system = get_enhanced_rl_system()
    return system.record_alert_outcome(symbol, alert_data, outcome, pump_percentage, time_to_pump_minutes)


if __name__ == "__main__":
    # Test standalone
    print("🧠 Enhanced RL Integration Test")
    print("=" * 50)
    
    system = EnhancedRLSystem()
    
    # Simulate stealth detection
    test_stealth_data = {
        'score': 2.5,
        'diamond_score': 0.6,
        'whaleclip_score': 0.8,
        'dex_inflow': 150000,
        'whale_ping': 2.0,
        'volume_spike': 1.2
    }
    
    test_market_data = {
        'price': 0.5,
        'volume': 1000000,
        'volatility': 0.3,
        'trend': 0.1
    }
    
    # Process detection
    result = system.process_stealth_detection("TESTUSDT", test_stealth_data, test_market_data)
    print(f"Decision result: {result}")
    
    # Simulate outcomes
    for i, outcome in enumerate([True, False, True, True, False]):
        pump_pct = np.random.uniform(2.0, 15.0) if outcome else 0.0
        time_to_pump = np.random.randint(30, 180) if outcome else None
        
        outcome_result = system.record_alert_outcome(
            f"TEST{i}USDT", result, outcome, pump_pct, time_to_pump
        )
        print(f"Outcome {i}: {outcome_result['outcome']}")
    
    # Show final parameters
    params = system.get_current_parameters()
    print("\nFinal System Parameters:")
    print(f"RLAgentV3 weights: {params['rl_agent_v3_weights']}")
    print(f"DQN threshold: {params['dqn_threshold']:.3f}")
    print(f"Performance: {params['performance_metrics']}")
    
    system.stop_background_training()
    print("\nTest completed!")