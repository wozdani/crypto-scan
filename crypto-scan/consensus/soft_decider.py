#!/usr/bin/env python3
"""
Soft Decider - Bradley-Terry aggregation without hard thresholds
"""

import math
import logging
from typing import List, Dict, Any
from datetime import datetime
from contracts.agent_contracts import AgentResponse, ConsensusResult, normalize_action_probs

logger = logging.getLogger(__name__)

class SoftDecider:
    """Implements Bradley-Terry soft aggregation for agent consensus"""
    
    def __init__(self):
        self.eps = 1e-6  # Small epsilon to avoid log(0)
        self.actions = ["BUY", "HOLD", "AVOID", "ABSTAIN"]
    
    def aggregate(self, agent_responses: List[AgentResponse], symbol: str = "unknown") -> ConsensusResult:
        """
        Aggregate agent responses using Bradley-Terry soft consensus
        No hard thresholds - uses weighted log-probabilities
        """
        if not agent_responses:
            return self._fallback_consensus(symbol)
        
        start_time = datetime.utcnow()
        
        # Initialize scores and accumulators
        scores = {a: 0.0 for a in self.actions}
        weight_sum = 0.0
        epistemic_sum = 0.0
        aleatoric_sum = 0.0
        all_evidence = []
        
        # Process each agent response
        for response in agent_responses:
            try:
                # Extract probabilities
                probs = response.action_probs.dict()
                
                # Calculate weight: reliability * (1 - epistemic_uncertainty)
                reliability = response.calibration_hint.reliability
                epistemic = response.uncertainty.epistemic
                weight = reliability * (1.0 - epistemic)
                weight_sum += weight
                
                # Accumulate uncertainties
                epistemic_sum += epistemic
                aleatoric_sum += response.uncertainty.aleatoric
                
                # Collect evidence
                for evidence in response.evidence:
                    all_evidence.append({
                        "name": evidence.name,
                        "direction": evidence.direction,
                        "strength": evidence.strength
                    })
                
                # Weighted log-probabilities (Bradley-Terry model)
                for action in self.actions:
                    prob = max(probs.get(action, 0.0), self.eps)
                    scores[action] += weight * math.log(prob)
                    
            except Exception as e:
                logger.error(f"Error processing agent response for {symbol}: {e}")
                continue
        
        # Normalize by total weight
        if weight_sum > 0:
            for action in self.actions:
                scores[action] /= weight_sum
        
        # Apply softmax to get final probabilities
        max_score = max(scores.values())
        exp_scores = {a: math.exp(scores[a] - max_score) for a in self.actions}
        total_exp = sum(exp_scores.values())
        
        if total_exp == 0:
            final_probs = {a: 0.25 for a in self.actions}  # Uniform fallback
        else:
            final_probs = {a: exp_scores[a] / total_exp for a in self.actions}
        
        # Calculate entropy
        entropy = -sum(p * math.log(max(p, self.eps)) for p in final_probs.values()) / math.log(len(self.actions))
        
        # Soft entropy-based ABSTAIN adjustment (no hard threshold)
        if entropy > 0.75:
            abstain_boost = min(0.07, 0.07 * (entropy - 0.75) / 0.25)
            final_probs["ABSTAIN"] = min(1.0, final_probs["ABSTAIN"] + abstain_boost)
            
            # Renormalize
            total = sum(final_probs.values())
            if total > 0:
                final_probs = {a: final_probs[a] / total for a in self.actions}
        
        # Calculate global uncertainties
        num_agents = max(len(agent_responses), 1)
        global_epistemic = epistemic_sum / num_agents
        global_aleatoric = aleatoric_sum / num_agents
        
        # Extract top evidence
        top_evidence = self._extract_top_evidence(all_evidence)
        
        # Determine dominant action and confidence
        dominant_action = max(final_probs, key=final_probs.get)
        confidence = final_probs[dominant_action]
        
        # Generate rationale
        rationale = self._generate_rationale(dominant_action, confidence, entropy, len(agent_responses))
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.info(f"[CONSENSUS] {symbol}: {dominant_action} ({confidence:.3f}), "
                   f"entropy={entropy:.2f}, agents={len(agent_responses)}, "
                   f"time={processing_time:.1f}ms")
        
        return ConsensusResult(
            final_probs=normalize_action_probs(final_probs),
            dominant_action=dominant_action,
            confidence=confidence,
            entropy=entropy,
            top_evidence=top_evidence,
            uncertainty_global={
                "epistemic": global_epistemic,
                "aleatoric": global_aleatoric
            },
            rationale=rationale
        )
    
    def _extract_top_evidence(self, all_evidence: List[Dict]) -> List[str]:
        """Extract top evidence by strength"""
        try:
            # Sort by strength, then by direction priority (pro > neutral > con)
            direction_priority = {"pro": 3, "neutral": 2, "con": 1}
            
            sorted_evidence = sorted(
                all_evidence,
                key=lambda x: (x.get("strength", 0), direction_priority.get(x.get("direction", "neutral"), 0)),
                reverse=True
            )
            
            # Get top 3-5 evidence names
            top_names = [e.get("name", "unknown") for e in sorted_evidence[:5]]
            
            # Remove duplicates while preserving order
            unique_names = []
            for name in top_names:
                if name not in unique_names:
                    unique_names.append(name)
                    
            return unique_names[:5]
            
        except Exception as e:
            logger.error(f"Error extracting top evidence: {e}")
            return ["consensus_analysis", "soft_reasoning"]
    
    def _generate_rationale(self, dominant_action: str, confidence: float, 
                          entropy: float, agent_count: int) -> str:
        """Generate consensus rationale"""
        rationale = f"Bradley-Terry consensus: {dominant_action} ({confidence:.3f})"
        
        if entropy > 0.8:
            rationale += f", high uncertainty (H={entropy:.2f})"
        elif entropy < 0.4:
            rationale += f", low uncertainty (H={entropy:.2f})"
        else:
            rationale += f", moderate uncertainty (H={entropy:.2f})"
            
        rationale += f", {agent_count} agents"
        
        if confidence < 0.4:
            rationale += " - weak consensus"
        elif confidence > 0.6:
            rationale += " - strong consensus"
        
        return rationale
    
    def _fallback_consensus(self, symbol: str) -> ConsensusResult:
        """Fallback consensus when no agent responses available"""
        logger.warning(f"No agent responses for {symbol}, using fallback consensus")
        
        return ConsensusResult(
            final_probs=normalize_action_probs({"BUY": 0.1, "HOLD": 0.6, "AVOID": 0.2, "ABSTAIN": 0.1}),
            dominant_action="HOLD",
            confidence=0.6,
            entropy=0.85,
            top_evidence=["no_agent_responses", "fallback_consensus"],
            uncertainty_global={"epistemic": 0.9, "aleatoric": 0.5},
            rationale="Fallback consensus due to no agent responses available"
        )

# Global instance
soft_decider = SoftDecider()