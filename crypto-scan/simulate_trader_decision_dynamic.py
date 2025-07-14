#!/usr/bin/env python3
"""
Dynamic Trader Decision System - RLAgentV3 Powered Alert Decision Engine
Intelligent alert decisions based on adaptive booster weighting and learned experience
"""

import os
import sys
import logging
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

# Add crypto-scan to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_agent_v3 import RLAgentV3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_trader_decision_dynamic(inputs: Dict[str, float],
                                   rl_agent: RLAgentV3,
                                   threshold: float = 0.7,
                                   debug: bool = False) -> Tuple[str, float, Dict[str, Any]]:
    """
    Intelligent alert decision based on RLAgentV3 adaptive weights
    
    Args:
        inputs: Signal inputs {"gnn": 0.6, "whaleClip": 0.9, "dexInflow": 1.0}
        rl_agent: Trained RLAgentV3 instance with learned weights
        threshold: Alert threshold (default: 0.7)
        debug: Enable debug logging
        
    Returns:
        Tuple of (decision, final_score, detailed_analysis)
    """
    
    # Compute weighted final score using learned weights
    final_score = rl_agent.compute_final_score(inputs)
    
    # Get alert quality prediction from RL agent
    prediction = rl_agent.predict_alert_quality(inputs)
    
    # Get booster importance ranking
    importance_ranking = rl_agent.get_booster_importance_ranking()
    
    # Determine decision based on score and prediction
    if final_score >= threshold:
        decision = "ALERT_TRIGGERED"
        confidence = "HIGH" if prediction["confidence"] >= 0.7 else "MEDIUM"
    elif final_score >= threshold * 0.8:  # 80% of threshold
        decision = "ALERT_CONSIDER"
        confidence = "MEDIUM" if prediction["confidence"] >= 0.5 else "LOW"
    else:
        decision = "AVOID"
        confidence = "LOW"
    
    # Identify dominant booster
    dominant_booster = None
    max_contribution = 0
    for booster, value in inputs.items():
        if booster in rl_agent.weights:
            contribution = value * rl_agent.weights[booster]
            if contribution > max_contribution:
                max_contribution = contribution
                dominant_booster = booster
    
    # Create detailed analysis
    detailed_analysis = {
        "final_score": final_score,
        "threshold": threshold,
        "decision": decision,
        "confidence": confidence,
        "prediction": prediction,
        "dominant_booster": dominant_booster,
        "booster_contributions": {},
        "importance_ranking": importance_ranking,
        "weights_used": rl_agent.weights.copy(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Calculate individual booster contributions
    for booster, value in inputs.items():
        if booster in rl_agent.weights:
            weight = rl_agent.weights[booster]
            contribution = value * weight
            detailed_analysis["booster_contributions"][booster] = {
                "value": value,
                "weight": weight,
                "contribution": contribution,
                "percentage": (contribution / final_score * 100) if final_score > 0 else 0
            }
    
    # Debug logging
    if debug:
        logger.info(f"[DECISION RL-V3] === DYNAMIC TRADER DECISION ===")
        logger.info(f"[DECISION RL-V3] Inputs: {inputs}")
        logger.info(f"[DECISION RL-V3] Learned Weights: {rl_agent.weights}")
        logger.info(f"[DECISION RL-V3] Final Score: {final_score:.3f} (threshold: {threshold})")
        logger.info(f"[DECISION RL-V3] Decision: {decision} (confidence: {confidence})")
        logger.info(f"[DECISION RL-V3] Dominant Booster: {dominant_booster}")
        logger.info(f"[DECISION RL-V3] Prediction: {prediction['recommendation']}")
        
        logger.info(f"[DECISION RL-V3] Booster Contributions:")
        for booster, details in detailed_analysis["booster_contributions"].items():
            logger.info(f"  • {booster}: {details['contribution']:.3f} "
                       f"({details['percentage']:.1f}%) = {details['value']:.3f} × {details['weight']:.3f}")
        
        logger.info(f"[DECISION RL-V3] Importance Ranking:")
        for i, (booster, importance, stats) in enumerate(importance_ranking, 1):
            logger.info(f"  {i}. {booster}: {importance:.3f} "
                       f"(effectiveness: {stats['effectiveness']*100:.1f}%)")
    
    return decision, final_score, detailed_analysis

def create_rl_agent_for_stealth_engine(weight_path: str = "cache/rl_agent_v3_weights.json",
                                     booster_names: Tuple[str, ...] = ("gnn", "whaleClip", "dexInflow")) -> RLAgentV3:
    """
    Create and initialize RLAgentV3 for Stealth Engine integration
    
    Args:
        weight_path: Path to saved weights
        booster_names: Names of signal boosters
        
    Returns:
        Initialized RLAgentV3 instance
    """
    agent = RLAgentV3(
        booster_names=booster_names,
        learning_rate=0.05,  # Production learning rate
        decay=0.995,
        weight_path=weight_path,
        min_weight=0.1,
        max_weight=5.0,
        normalize_weights=True
    )
    
    logger.info(f"[RL AGENT INIT] Initialized RLAgentV3 with {len(agent.weights)} boosters")
    logger.info(f"[RL AGENT INIT] Current weights: {agent.weights}")
    
    # Get training statistics
    stats = agent.get_training_statistics()
    if stats["update_count"] > 0:
        logger.info(f"[RL AGENT INIT] Training stats: {stats['update_count']} updates, "
                   f"{stats['success_rate']:.1f}% success rate")
    
    return agent

def process_stealth_signals_with_rl(gnn_anomaly_scores: Dict[str, float],
                                   whale_clip_confidence: float,
                                   dex_inflow_detected: bool,
                                   rl_agent: RLAgentV3,
                                   threshold: float = 0.7,
                                   debug: bool = False) -> Dict[str, Any]:
    """
    Process stealth signals using RLAgentV3 for intelligent decision making
    
    Args:
        gnn_anomaly_scores: Dictionary of address anomaly scores
        whale_clip_confidence: WhaleCLIP confidence score (0-1)
        dex_inflow_detected: Boolean indicating DEX inflow detection
        rl_agent: Trained RLAgentV3 instance
        threshold: Alert threshold
        debug: Enable debug logging
        
    Returns:
        Complete analysis with decision and metadata
    """
    
    # Prepare inputs for RL agent
    inputs = {
        "gnn": max(gnn_anomaly_scores.values()) if gnn_anomaly_scores else 0.0,
        "whaleClip": whale_clip_confidence,
        "dexInflow": 1.0 if dex_inflow_detected else 0.0
    }
    
    # Get dynamic decision
    decision, final_score, detailed_analysis = simulate_trader_decision_dynamic(
        inputs=inputs,
        rl_agent=rl_agent,
        threshold=threshold,
        debug=debug
    )
    
    # Enhanced analysis with stealth context
    enhanced_analysis = {
        **detailed_analysis,
        "gnn_analysis": {
            "anomaly_scores": gnn_anomaly_scores,
            "max_score": inputs["gnn"],
            "high_risk_addresses": len([s for s in gnn_anomaly_scores.values() if s >= 0.8]),
            "medium_risk_addresses": len([s for s in gnn_anomaly_scores.values() if 0.6 <= s < 0.8])
        },
        "whale_analysis": {
            "clip_confidence": whale_clip_confidence,
            "confidence_level": "HIGH" if whale_clip_confidence >= 0.8 else 
                              "MEDIUM" if whale_clip_confidence >= 0.5 else "LOW"
        },
        "dex_analysis": {
            "inflow_detected": dex_inflow_detected,
            "inflow_strength": 1.0 if dex_inflow_detected else 0.0
        }
    }
    
    return enhanced_analysis

def update_rl_agent_from_outcome(rl_agent: RLAgentV3,
                                inputs: Dict[str, float],
                                pump_occurred: bool,
                                price_change_24h: float = None,
                                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Update RLAgentV3 weights based on alert outcome
    
    Args:
        rl_agent: RLAgentV3 instance to update
        inputs: Original inputs used for decision
        pump_occurred: Whether pump actually occurred
        price_change_24h: 24h price change percentage
        metadata: Additional outcome metadata
        
    Returns:
        Update statistics and analysis
    """
    
    # Determine reward based on outcome
    if pump_occurred:
        # Successful pump prediction
        reward = 1.0
        if price_change_24h and price_change_24h > 20:  # Extra reward for big pumps
            reward = 1.2
    else:
        # False alert
        reward = -1.0
        if price_change_24h and price_change_24h < -5:  # Extra penalty for dumps
            reward = -1.2
    
    # Update weights
    old_weights = rl_agent.weights.copy()
    rl_agent.update_weights(inputs, reward, metadata)
    new_weights = rl_agent.weights.copy()
    
    # Save updated weights
    rl_agent.save_weights()
    
    # Calculate weight changes
    weight_changes = {}
    for booster in old_weights:
        weight_changes[booster] = new_weights[booster] - old_weights[booster]
    
    # Get updated statistics
    stats = rl_agent.get_training_statistics()
    
    update_analysis = {
        "reward": reward,
        "pump_occurred": pump_occurred,
        "price_change_24h": price_change_24h,
        "old_weights": old_weights,
        "new_weights": new_weights,
        "weight_changes": weight_changes,
        "training_stats": stats,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"[RL UPDATE] Reward: {reward:+.1f}, Pump: {pump_occurred}")
    logger.info(f"[RL UPDATE] Weight changes: {weight_changes}")
    logger.info(f"[RL UPDATE] Success rate: {stats['success_rate']:.1f}%")
    
    return update_analysis

def test_dynamic_decision_system():
    """Test the dynamic decision system"""
    print("🧪 Testing Dynamic Trader Decision System")
    
    # Create test RL agent
    agent = RLAgentV3(
        booster_names=("gnn", "whaleClip", "dexInflow"),
        learning_rate=0.1,
        weight_path=None  # Don't save during test
    )
    
    # Train with some scenarios to establish weights
    training_scenarios = [
        ({"gnn": 0.9, "whaleClip": 0.8, "dexInflow": 1.0}, 1.0),  # Strong pump
        ({"gnn": 0.2, "whaleClip": 0.1, "dexInflow": 0.8}, -1.0),  # False alert
        ({"gnn": 0.85, "whaleClip": 0.9, "dexInflow": 0.2}, 1.0),  # Whale pump
    ]
    
    for inputs, reward in training_scenarios:
        agent.update_weights(inputs, reward)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Strong institutional signal",
            "inputs": {"gnn": 0.92, "whaleClip": 0.88, "dexInflow": 0.3},
            "expected": "ALERT_TRIGGERED"
        },
        {
            "name": "Weak signals",
            "inputs": {"gnn": 0.3, "whaleClip": 0.2, "dexInflow": 0.1},
            "expected": "AVOID"
        },
        {
            "name": "DEX only signal",
            "inputs": {"gnn": 0.1, "whaleClip": 0.05, "dexInflow": 1.0},
            "expected": "AVOID"
        }
    ]
    
    print(f"\n📊 Testing decision scenarios:")
    for scenario in test_scenarios:
        decision, score, analysis = simulate_trader_decision_dynamic(
            inputs=scenario["inputs"],
            rl_agent=agent,
            threshold=0.7,
            debug=True
        )
        
        print(f"\n✅ {scenario['name']}:")
        print(f"   Decision: {decision} (score: {score:.3f})")
        print(f"   Dominant: {analysis['dominant_booster']}")
        print(f"   Confidence: {analysis['confidence']}")
    
    return True

if __name__ == "__main__":
    test_dynamic_decision_system()