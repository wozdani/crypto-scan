#!/usr/bin/env python3
"""
Post-Decision Calibration System
Tracks decision outcomes and applies Platt scaling for probability calibration
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)

class PostDecisionTracker:
    """Tracks decision outcomes and calibrates probabilities without hard thresholds"""
    
    def __init__(self, data_path: str = "crypto-scan/data"):
        self.data_path = data_path
        self.decisions_file = os.path.join(data_path, "decision_outcomes.json")
        self.agent_reliability_file = os.path.join(data_path, "agent_reliability.json")
        self.calibration_file = os.path.join(data_path, "probability_calibration.json")
        
        # EMA smoothing factor for agent reliability updates
        self.ema_alpha = 0.1
        
        # Ensure data directory exists
        os.makedirs(data_path, exist_ok=True)
        
    def record_decision(self, symbol: str, final_probs: Dict[str, float], 
                       agent_opinions: List[Dict], timestamp: Optional[datetime] = None):
        """Record a decision for later outcome tracking"""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        decision_record = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "final_probs": final_probs,
            "dominant_action": max(final_probs, key=final_probs.get),
            "confidence": max(final_probs.values()),
            "agent_opinions": agent_opinions,
            "outcome": None,  # To be filled later
            "latency_mins": None,  # To be filled when outcome is determined
            "verified": False
        }
        
        # Load existing decisions
        decisions = self._load_decisions()
        decisions.append(decision_record)
        
        # Keep only last 1000 decisions to manage size
        if len(decisions) > 1000:
            decisions = decisions[-1000:]
            
        self._save_decisions(decisions)
        logger.info(f"Recorded decision for {symbol}: {decision_record['dominant_action']} ({decision_record['confidence']:.3f})")
        
    def update_outcome(self, symbol: str, timestamp: datetime, outcome: str, 
                      price_change_percent: float, verification_delay_mins: int):
        """Update decision outcome after verification period"""
        decisions = self._load_decisions()
        
        for decision in decisions:
            if (decision["symbol"] == symbol and 
                decision["timestamp"] == timestamp.isoformat() and 
                not decision["verified"]):
                
                # Determine outcome classification based on soft criteria
                decision["outcome"] = self._classify_outcome(
                    decision["dominant_action"], 
                    decision["confidence"], 
                    price_change_percent
                )
                decision["latency_mins"] = verification_delay_mins
                decision["price_change_percent"] = price_change_percent
                decision["verified"] = True
                
                logger.info(f"Updated outcome for {symbol}: {decision['outcome']} "
                           f"(price_change: {price_change_percent:.2f}%, latency: {verification_delay_mins}min)")
                break
                
        self._save_decisions(decisions)
        
        # Update agent reliability using EMA
        self._update_agent_reliability(decisions[-1])
        
        # Periodically recalibrate probabilities
        if len([d for d in decisions if d["verified"]]) % 20 == 0:
            self._recalibrate_probabilities()
            
    def _classify_outcome(self, dominant_action: str, confidence: float, 
                         price_change_percent: float) -> str:
        """Classify outcome using soft criteria (no hard thresholds)"""
        # Soft classification based on price movement and action
        abs_change = abs(price_change_percent)
        
        if dominant_action == "BUY":
            if price_change_percent > 2.0:  # Significant positive movement
                return "TP"  # True Positive
            elif price_change_percent < -1.0:  # Negative movement
                return "FP"  # False Positive
            else:
                return "TN" if confidence < 0.6 else "FP"  # Soft boundary
                
        elif dominant_action == "AVOID":
            if price_change_percent < -1.0:  # Price dropped as predicted
                return "TP"
            elif price_change_percent > 2.0:  # Missed opportunity
                return "FN"  # False Negative
            else:
                return "TN"
                
        else:  # HOLD or ABSTAIN
            if abs_change < 1.5:  # Stable price
                return "TP"
            else:
                return "FN" if abs_change > 3.0 else "TN"
                
    def _update_agent_reliability(self, decision: Dict):
        """Update agent reliability using EMA without hard thresholds"""
        reliability_data = self._load_agent_reliability()
        
        # Calculate success score (soft, not binary)
        outcome = decision["outcome"]
        success_score = {
            "TP": 1.0,
            "TN": 0.8,
            "FP": 0.2,
            "FN": 0.1
        }.get(outcome, 0.5)
        
        # Update each agent's reliability using EMA
        for opinion in decision["agent_opinions"]:
            agent_name = opinion.get("agent_name", "unknown")
            current_reliability = reliability_data.get(agent_name, 0.6)
            
            # EMA update: new_reliability = alpha * success_score + (1-alpha) * old_reliability
            new_reliability = (self.ema_alpha * success_score + 
                             (1 - self.ema_alpha) * current_reliability)
            
            reliability_data[agent_name] = max(0.1, min(0.95, new_reliability))  # Soft bounds
            
        reliability_data["last_update"] = datetime.utcnow().isoformat()
        self._save_agent_reliability(reliability_data)
        
    def _recalibrate_probabilities(self):
        """Apply Platt scaling/isotonic regression to calibrate BUY probabilities"""
        decisions = [d for d in self._load_decisions() if d["verified"]]
        
        if len(decisions) < 30:  # Need sufficient data
            return
            
        # Extract features and targets
        buy_probs = []
        binary_outcomes = []
        
        for decision in decisions[-100:]:  # Use last 100 decisions
            buy_prob = decision["final_probs"].get("BUY", 0.0)
            outcome = decision["outcome"]
            
            buy_probs.append(buy_prob)
            # Binary outcome: 1 for successful BUY-related decisions
            binary_outcomes.append(1 if outcome == "TP" and decision["dominant_action"] == "BUY" else 0)
            
        if len(set(binary_outcomes)) < 2:  # Need both positive and negative examples
            return
            
        X = np.array(buy_probs).reshape(-1, 1)
        y = np.array(binary_outcomes)
        
        try:
            # Apply Platt scaling (logistic regression)
            platt_model = LogisticRegression()
            platt_model.fit(X, y)
            
            # Apply isotonic regression for monotonic calibration
            isotonic_model = IsotonicRegression(out_of_bounds='clip')
            isotonic_model.fit(buy_probs, binary_outcomes)
            
            # Save calibration models (simplified - store parameters)
            calibration_data = {
                "platt_coef": platt_model.coef_[0].tolist(),
                "platt_intercept": float(platt_model.intercept_[0]),
                "isotonic_points": len(isotonic_model.X_thresholds_),
                "last_calibration": datetime.utcnow().isoformat(),
                "calibration_size": len(decisions)
            }
            
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
                
            logger.info(f"Recalibrated probabilities using {len(decisions)} decisions")
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            
    def get_calibrated_probability(self, raw_buy_prob: float) -> float:
        """Apply calibration to raw BUY probability"""
        if not os.path.exists(self.calibration_file):
            return raw_buy_prob  # No calibration available
            
        try:
            with open(self.calibration_file, 'r') as f:
                calibration_data = json.load(f)
                
            # Apply Platt scaling
            coef = calibration_data["platt_coef"][0]
            intercept = calibration_data["platt_intercept"]
            
            # Logistic function: 1 / (1 + exp(-(coef * x + intercept)))
            linear_score = coef * raw_buy_prob + intercept
            calibrated_prob = 1.0 / (1.0 + np.exp(-linear_score))
            
            return float(np.clip(calibrated_prob, 0.01, 0.99))  # Soft bounds
            
        except Exception as e:
            logger.error(f"Calibration application failed: {e}")
            return raw_buy_prob
            
    def get_agent_reliability(self, agent_name: str) -> float:
        """Get current reliability for agent"""
        reliability_data = self._load_agent_reliability()
        return reliability_data.get(agent_name, 0.6)  # Default reliability
        
    def _load_decisions(self) -> List[Dict]:
        """Load decision history"""
        if os.path.exists(self.decisions_file):
            try:
                with open(self.decisions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading decisions: {e}")
        return []
        
    def _save_decisions(self, decisions: List[Dict]):
        """Save decision history"""
        try:
            with open(self.decisions_file, 'w') as f:
                json.dump(decisions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving decisions: {e}")
            
    def _load_agent_reliability(self) -> Dict[str, float]:
        """Load agent reliability scores"""
        if os.path.exists(self.agent_reliability_file):
            try:
                with open(self.agent_reliability_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading agent reliability: {e}")
        return {}
        
    def _save_agent_reliability(self, reliability_data: Dict[str, float]):
        """Save agent reliability scores"""
        try:
            with open(self.agent_reliability_file, 'w') as f:
                json.dump(reliability_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving agent reliability: {e}")

# Singleton instance for global use
post_decision_tracker = PostDecisionTracker()