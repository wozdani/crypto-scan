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
    TJDEFeedbackLoop = None


class EnhancedRLIntegration:
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
    
    def prepare_comprehensive_state_vector(self, symbol: str, base_score: float, 
                                         stealth_signals: List[str], diamond_score: float = 0.0,
                                         californium_score: float = 0.0, whaleclip_score: float = 0.0,
                                         consensus_data = None, market_data = None) -> np.ndarray:
        """
        Prepare comprehensive state vector for Enhanced RL analysis
        """
        try:
            state_vector_size = 25
            state_vector = np.zeros(state_vector_size)
            
            # Index 0-4: Core scores (normalized to 0-1)
            state_vector[0] = min(base_score / 5.0, 1.0)  # Base stealth score
            state_vector[1] = min(diamond_score, 1.0)      # Diamond AI score
            state_vector[2] = min(californium_score, 1.0)  # Californium score
            state_vector[3] = min(whaleclip_score, 1.0)    # WhaleCLIP score
            
            # Index 4: Signal count (normalized)
            state_vector[4] = min(len(stealth_signals) / 10.0, 1.0)
            
            # Index 5-14: Key stealth signals (binary indicators)
            signal_mapping = {
                'whale_ping': 5, 'dex_inflow': 6, 'volume_spike': 7, 
                'spoofing_layers': 8, 'large_bid_walls': 9, 'repeated_address_boost': 10,
                'velocity_boost': 11, 'inflow_momentum_boost': 12, 'multi_address_group_activity': 13,
                'orderbook_imbalance': 14
            }
            
            for signal in stealth_signals:
                if signal in signal_mapping:
                    state_vector[signal_mapping[signal]] = 1.0
            
            # Index 15-19: Market data features
            if market_data:
                state_vector[15] = min(market_data.get('volume_24h', 0) / 10000000, 1.0)  # Volume normalized
                state_vector[16] = abs(market_data.get('price_change_24h', 0)) / 100.0    # Price change
                state_vector[17] = min(market_data.get('price', 0) / 100.0, 1.0)         # Price normalized
                
            # Index 18-19: Consensus features
            if consensus_data:
                state_vector[18] = consensus_data.get('final_score', 0.0)
                state_vector[19] = len(consensus_data.get('contributing_detectors', [])) / 4.0
            
            # Index 20-24: Time-based features
            current_hour = datetime.now().hour
            state_vector[20] = current_hour / 24.0  # Hour of day
            state_vector[21] = len(self.alert_outcomes) / 1000.0  # Historical context
            state_vector[22] = self.performance_metrics.get('avg_profit', 0.0) / 100.0  # Performance history
            state_vector[23] = min(time.time() % 86400 / 86400.0, 1.0)  # Time of day cycle
            state_vector[24] = 1.0 if len(stealth_signals) > 5 else 0.0  # High signal count indicator
            
            return state_vector
            
        except Exception as e:
            print(f"[ENHANCED RL] State vector preparation error: {e}")
            return np.zeros(25)
    
    def analyze_with_adaptive_thresholds(self, symbol: str, state_vector: np.ndarray, 
                                       base_score: float, use_dqn: bool = True) -> Dict[str, Any]:
        """
        Run Enhanced RL analysis with adaptive thresholding
        """
        try:
            analysis_start = time.time()
            
            # Initialize result
            result = {
                'symbol': symbol,
                'timestamp': analysis_start,
                'should_modify': False,
                'enhanced_score': base_score,
                'adaptive_multiplier': 1.0,
                'confidence_boost': 0.0,
                'threshold_info': {},
                'skip_reason': None
            }
            
            # DQN Analysis (if enabled and available)
            dqn_action = None
            dqn_confidence = 0.0
            
            if use_dqn and self.dqn_integration and hasattr(self.dqn_integration, 'agent'):
                try:
                    # Use DQN agent for threshold adaptation
                    current_threshold = self.dqn_integration.agent.current_threshold
                    
                    # Simple DQN-like logic based on state vector
                    signal_strength = np.sum(state_vector[5:15])  # Count active signals
                    score_ratio = base_score / max(current_threshold, 0.1)
                    
                    if signal_strength >= 3 and score_ratio >= 1.2:
                        dqn_action = 4  # STRONG_BUY
                        dqn_confidence = min(1.0, score_ratio * 0.8)
                    elif signal_strength >= 2 and score_ratio >= 0.8:
                        dqn_action = 3  # BUY
                        dqn_confidence = min(1.0, score_ratio * 0.6)
                    else:
                        dqn_action = 2  # HOLD
                        dqn_confidence = 0.3
                    
                    result['threshold_info']['dqn_action'] = int(dqn_action)
                    result['threshold_info']['dqn_confidence'] = round(dqn_confidence, 3)
                    result['threshold_info']['current_threshold'] = round(current_threshold, 3)
                    
                except Exception as e:
                    print(f"[ENHANCED RL] DQN analysis error: {e}")
            
            # RLAgentV3 Analysis (if available)
            rl_recommendation = None
            rl_confidence = 0.0
            
            if self.rl_agent_v3:
                try:
                    # Get current weights and compute enhancement
                    weights = self.rl_agent_v3.weights
                    
                    # Simple weighted decision based on current performance
                    if base_score >= 2.0:
                        rl_recommendation = 'BUY'
                        rl_confidence = min(1.0, base_score / 4.0)
                    elif base_score >= 1.0:
                        rl_recommendation = 'HOLD'
                        rl_confidence = base_score / 2.0
                    else:
                        rl_recommendation = 'AVOID'
                        rl_confidence = 0.3
                        
                    result['threshold_info']['rl_action'] = rl_recommendation
                    result['threshold_info']['rl_confidence'] = round(rl_confidence, 3)
                
                except Exception as e:
                    print(f"[ENHANCED RL] RLAgentV3 analysis error: {e}")
            
            # Hybrid Decision Making
            should_enhance = False
            adaptive_multiplier = 1.0
            confidence_boost = 0.0
            
            # DQN-based enhancement
            if dqn_action is not None:
                if dqn_action >= 3 and dqn_confidence > 0.6:  # Strong buy signals
                    should_enhance = True
                    adaptive_multiplier = 1.0 + (dqn_confidence * 0.5)
                    confidence_boost = dqn_confidence * 0.3
                elif dqn_action <= 1 and dqn_confidence > 0.6:  # Strong sell signals
                    should_enhance = True
                    adaptive_multiplier = max(0.5, 1.0 - (dqn_confidence * 0.3))
                    confidence_boost = -dqn_confidence * 0.2
            
            # RLAgentV3-based enhancement
            if rl_recommendation and rl_confidence > 0.5:
                if rl_recommendation in ['BUY', 'STRONG_BUY']:
                    should_enhance = True
                    adaptive_multiplier = max(adaptive_multiplier, 1.0 + (rl_confidence * 0.4))
                    confidence_boost = max(confidence_boost, rl_confidence * 0.25)
                elif rl_recommendation in ['SELL', 'STRONG_SELL']:
                    should_enhance = True
                    adaptive_multiplier = min(adaptive_multiplier, max(0.6, 1.0 - (rl_confidence * 0.2)))
                    confidence_boost = min(confidence_boost, -rl_confidence * 0.15)
            
            # Apply score modifications if enhancement is recommended
            if should_enhance:
                enhanced_score = base_score * adaptive_multiplier + confidence_boost
                enhanced_score = max(0.0, min(enhanced_score, 10.0))  # Clamp to reasonable range
                
                result.update({
                    'should_modify': True,
                    'enhanced_score': round(enhanced_score, 3),
                    'adaptive_multiplier': round(adaptive_multiplier, 3),
                    'confidence_boost': round(confidence_boost, 3)
                })
                
                # Update performance metrics
                self.performance_metrics['successful_adaptations'] += 1
            else:
                result['skip_reason'] = 'no_significant_enhancement_detected'
            
            # Update analysis history
            self.alert_outcomes.append({
                'symbol': symbol,
                'timestamp': analysis_start,
                'base_score': base_score,
                'enhanced_score': result['enhanced_score'],
                'should_modify': should_enhance,
                'dqn_action': dqn_action,
                'rl_recommendation': rl_recommendation
            })
            
            # Keep history manageable
            if len(self.alert_outcomes) > 1000:
                self.alert_outcomes = self.alert_outcomes[-500:]
            
            # Update performance metrics
            self.performance_metrics['total_alerts'] += 1
            
            return result
            
        except Exception as e:
            print(f"[ENHANCED RL] Analysis error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': time.time(),
                'should_modify': False,
                'enhanced_score': base_score,
                'skip_reason': f'analysis_error: {e}'
            }


# 🎯 Global instance dla easy access
_enhanced_rl_system = None

def get_enhanced_rl_system() -> EnhancedRLIntegration:
    """Get global enhanced RL system instance"""
    global _enhanced_rl_system
    if _enhanced_rl_system is None:
        _enhanced_rl_system = EnhancedRLIntegration()
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
    
    system = EnhancedRLIntegration()
    
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