#!/usr/bin/env python3
"""
Clean Decision Consensus Engine - Multi-Agent system DISABLED for individual tokens
Only essential functions for fallback to weighted voting when Multi-Agent is disabled
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import AlertDecision and ConsensusResult from consensus_decision_engine
try:
    from .consensus_decision_engine import AlertDecision, ConsensusResult, ConsensusStrategy
    CONSENSUS_CLASSES_AVAILABLE = True
    print("[DECISION CONSENSUS CLEAN] AlertDecision and ConsensusResult imported successfully")
except ImportError as e:
    CONSENSUS_CLASSES_AVAILABLE = False
    print(f"[DECISION CONSENSUS CLEAN] Import failed: {e}")
    
    # Fallback definitions
    from enum import Enum
    
    class AlertDecision(Enum):
        ALERT = "ALERT"
        NO_ALERT = "NO_ALERT"  
        WATCH = "WATCH"
        ESCALATE = "ESCALATE"
    
    class ConsensusStrategy(Enum):
        MAJORITY_VOTE = "majority_vote"
    
    @dataclass
    class ConsensusResult:
        decision: AlertDecision
        final_score: float
        confidence: float
        strategy_used: ConsensusStrategy
        contributing_detectors: List[str] 
        reasoning: str
        consensus_strength: float
        timestamp: str

@dataclass
class DetectorResult:
    """Structure for single detector result"""
    vote: str  # "BUY", "HOLD", "AVOID"
    score: float  # 0.0 - 1.0
    weight: float  # detector weight
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

class DecisionConsensusEngine:
    """
    Clean Decision Consensus Engine
    Multi-Agent system DISABLED for individual tokens - only simple weighted voting
    """
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.decision_history = []
        print("[CONSENSUS ENGINE CLEAN] Initialized with Multi-Agent DISABLED for individual tokens")
    
    def simulate_decision_consensus(
        self, 
        detector_outputs: Dict[str, Dict[str, Any]], 
        threshold: float = 0.7,
        token: str = "UNKNOWN"
    ) -> Optional[ConsensusResult]:
        """
        🚫 MULTI-AGENT CONSENSUS DISABLED FOR INDIVIDUAL TOKENS
        This function returns None to force fallback to simple weighted voting
        Multi-Agent consensus will ONLY run for LAST10 batch processing
        """
        print(f"[MULTI-AGENT DISABLED] {token}: simulate_decision_consensus DISABLED for individual tokens")
        print(f"[MULTI-AGENT DISABLED] {token}: Multi-Agent consensus will ONLY run for LAST10 batch processing")
        print(f"[MULTI-AGENT DISABLED] {token}: Returning None to force fallback to simple weighted voting")
        return None

def create_decision_consensus_engine() -> DecisionConsensusEngine:
    """Factory function for Decision Consensus Engine"""
    return DecisionConsensusEngine()

def test_decision_consensus():
    """Test function for Decision Consensus Engine"""
    print("🧪 Testing Decision Consensus Engine - Multi-Agent DISABLED...")
    
    engine = create_decision_consensus_engine()
    
    # Test case: All calls should return None (Multi-Agent disabled)
    print("\n🔬 Test: Multi-Agent Disabled Check")
    detector_outputs = {
        "StealthEngine": {"vote": "BUY", "score": 0.85, "weight": 0.25}
    }
    
    result = engine.simulate_decision_consensus(detector_outputs, threshold=0.7, token="TESTUSDT")
    
    if result is None:
        print("✅ Test PASSED: Multi-Agent correctly disabled, returned None")
    else:
        print("❌ Test FAILED: Multi-Agent should return None")
    
    print("🧪 Test completed")

if __name__ == "__main__":
    test_decision_consensus()