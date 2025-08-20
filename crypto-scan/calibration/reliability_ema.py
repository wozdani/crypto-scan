#!/usr/bin/env python3
"""
EMA-based Reliability Tracking for Agents
Updates agent reliability using exponential moving average based on outcomes
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from contracts.agent_contracts import ReliabilityUpdate

logger = logging.getLogger(__name__)

class ReliabilityTracker:
    """Tracks and updates agent reliability using EMA"""
    
    def __init__(self, storage_path: str = "state/agent_reliability.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.alpha = 0.1  # EMA smoothing factor
        self.min_reliability = 0.1  # Soft lower bound
        self.max_reliability = 0.95  # Soft upper bound
        self.agents = self._load_state()
        
    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        """Load agent reliability state from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded reliability data for {len(data)} agents")
                    return data
        except Exception as e:
            logger.error(f"Failed to load reliability state: {e}")
        
        # Initialize default agents
        default_agents = {
            "ANALYZER": {"reliability": 0.6, "update_count": 0, "last_update": None},
            "REASONER": {"reliability": 0.6, "update_count": 0, "last_update": None},
            "VOTER": {"reliability": 0.75, "update_count": 0, "last_update": None},  # Higher baseline
            "DEBATER": {"reliability": 0.6, "update_count": 0, "last_update": None}
        }
        logger.info("Initialized default agent reliabilities")
        return default_agents
    
    def _save_state(self):
        """Save agent reliability state to disk"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.agents, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save reliability state: {e}")
    
    def get_agent_reliability(self, agent_name: str) -> float:
        """Get current reliability for an agent"""
        if agent_name not in self.agents:
            # Initialize new agent with default reliability
            self.agents[agent_name] = {
                "reliability": 0.6,
                "update_count": 0, 
                "last_update": None
            }
            logger.info(f"Initialized new agent: {agent_name}")
            
        return self.agents[agent_name]["reliability"]
    
    def update_agent_reliability(self, agent_name: str, success_score: float, 
                               symbol: str = "unknown") -> ReliabilityUpdate:
        """
        Update agent reliability using EMA
        success_score: 0.0 (complete failure) to 1.0 (perfect success)
        """
        current_reliability = self.get_agent_reliability(agent_name)
        
        # EMA update: new_rel = alpha * success_score + (1-alpha) * old_rel
        new_reliability = self.alpha * success_score + (1 - self.alpha) * current_reliability
        
        # Apply soft bounds (no hard thresholds)
        new_reliability = max(self.min_reliability, min(self.max_reliability, new_reliability))
        
        # Update state
        old_reliability = current_reliability
        self.agents[agent_name].update({
            "reliability": new_reliability,
            "update_count": self.agents[agent_name]["update_count"] + 1,
            "last_update": datetime.utcnow().isoformat(),
            "last_success_score": success_score,
            "last_symbol": symbol
        })
        
        # Save to disk
        self._save_state()
        
        # Create update record
        update_record = ReliabilityUpdate(
            agent_name=agent_name,
            old_reliability=old_reliability,
            new_reliability=new_reliability,
            success_score=success_score,
            alpha=self.alpha,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"[RELIABILITY] {agent_name}: {old_reliability:.3f} → {new_reliability:.3f} "
                   f"(success={success_score:.3f}, α={self.alpha})")
        
        return update_record
    
    def get_all_reliabilities(self) -> Dict[str, float]:
        """Get current reliabilities for all agents"""
        return {name: data["reliability"] for name, data in self.agents.items()}
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed stats for an agent"""
        if agent_name not in self.agents:
            return {"error": f"Agent {agent_name} not found"}
        
        data = self.agents[agent_name]
        return {
            "reliability": data["reliability"],
            "update_count": data["update_count"],
            "last_update": data.get("last_update"),
            "last_success_score": data.get("last_success_score"),
            "last_symbol": data.get("last_symbol"),
            "reliability_category": self._categorize_reliability(data["reliability"])
        }
    
    def _categorize_reliability(self, reliability: float) -> str:
        """Categorize reliability level (descriptive, no hard thresholds)"""
        if reliability >= 0.8:
            return "high"
        elif reliability >= 0.6:
            return "moderate"
        elif reliability >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def decay_reliabilities(self, decay_factor: float = 0.02):
        """
        Apply slow decay to all reliabilities to prevent over-confidence
        Called periodically (e.g., daily) to maintain adaptability
        """
        decayed_count = 0
        
        for agent_name, data in self.agents.items():
            old_rel = data["reliability"]
            
            # Soft decay toward neutral value (0.6)
            neutral_value = 0.6
            new_rel = old_rel * (1 - decay_factor) + neutral_value * decay_factor
            
            if abs(new_rel - old_rel) > 0.001:  # Only update if significant change
                data["reliability"] = new_rel
                decayed_count += 1
        
        if decayed_count > 0:
            self._save_state()
            logger.info(f"Applied reliability decay to {decayed_count} agents (factor={decay_factor})")
    
    def reset_agent(self, agent_name: str):
        """Reset agent reliability to default (emergency use only)"""
        if agent_name in self.agents:
            old_rel = self.agents[agent_name]["reliability"]
            self.agents[agent_name] = {
                "reliability": 0.6,
                "update_count": 0,
                "last_update": datetime.utcnow().isoformat(),
                "reset_reason": "manual_reset"
            }
            self._save_state()
            logger.warning(f"Reset {agent_name} reliability: {old_rel:.3f} → 0.600")

# Global instance
reliability_tracker = ReliabilityTracker()