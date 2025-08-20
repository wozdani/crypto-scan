# integration/multi_agent_integration.py
"""
Multi-Agent Consensus Integration for Main Scanner
Integrates 4-agent probabilistic consensus system into existing scanning pipeline
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import the new consensus system
from consensus.agents import run_analyzer, run_reasoner, run_voter, run_debater
from consensus.decider import aggregate
from consensus.contracts import FinalDecision, AgentOpinion

logger = logging.getLogger(__name__)

class MultiAgentScanner:
    """Integrates multi-agent consensus into main scanning pipeline"""
    
    def __init__(self):
        self.enabled = True
        
    def process_token_consensus(self, symbol: str, detector_breakdown: Dict[str, Any], 
                               meta_dict: Dict[str, Any], trust_dict: Dict[str, Any],
                               history_dict: Dict[str, Any], perf_dict: Dict[str, Any]) -> Optional[FinalDecision]:
        """
        Process token through multi-agent consensus system
        
        Args:
            symbol: Token symbol (e.g., "BTCUSDT")
            detector_breakdown: Results from all detectors
            meta_dict: Market metadata (price, volume, etc.)
            trust_dict: Trust scores and whale information
            history_dict: Historical patterns and trends
            perf_dict: Performance metrics and statistics
            
        Returns:
            FinalDecision with consensus probabilities or None if error
        """
        if not self.enabled:
            return None
            
        try:
            start_time = datetime.utcnow()
            
            # Build payload for agents
            payload = {
                "symbol": symbol,
                "detector_breakdown": detector_breakdown,
                "meta": meta_dict,
                "trust": trust_dict,
                "history": history_dict,
                "perf": perf_dict,
                "timestamp": start_time.isoformat() + "Z"
            }
            
            logger.info(f"[MULTI-AGENT] Processing {symbol} with {len(detector_breakdown)} detectors")
            
            # Run all 4 agents in sequence (could be parallelized later)
            opinions = []
            
            try:
                analyzer_opinion = run_analyzer(payload)
                opinions.append(analyzer_opinion)
                logger.debug(f"[MULTI-AGENT] {symbol} Analyzer: {analyzer_opinion.action_probs}")
            except Exception as e:
                logger.error(f"[MULTI-AGENT] {symbol} Analyzer failed: {e}")
                return None
            
            try:
                reasoner_opinion = run_reasoner(payload)
                opinions.append(reasoner_opinion)
                logger.debug(f"[MULTI-AGENT] {symbol} Reasoner: {reasoner_opinion.action_probs}")
            except Exception as e:
                logger.error(f"[MULTI-AGENT] {symbol} Reasoner failed: {e}")
                return None
            
            try:
                voter_opinion = run_voter(payload)
                opinions.append(voter_opinion)
                logger.debug(f"[MULTI-AGENT] {symbol} Voter: {voter_opinion.action_probs}")
            except Exception as e:
                logger.error(f"[MULTI-AGENT] {symbol} Voter failed: {e}")
                return None
            
            try:
                debater_opinion = run_debater(payload)
                opinions.append(debater_opinion)
                logger.debug(f"[MULTI-AGENT] {symbol} Debater: {debater_opinion.action_probs}")
            except Exception as e:
                logger.error(f"[MULTI-AGENT] {symbol} Debater failed: {e}")
                return None
            
            # Aggregate opinions using miękka agregacja
            final_decision: FinalDecision = aggregate(opinions)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Log final decision
            dominant_action = max(final_decision.final_probs, key=final_decision.final_probs.get)
            confidence = final_decision.final_probs[dominant_action]
            
            logger.info(f"[MULTI-AGENT] {symbol} Final: {dominant_action} ({confidence:.3f}) "
                       f"in {processing_time:.1f}ms")
            
            return final_decision
            
        except Exception as e:
            logger.error(f"[MULTI-AGENT] Error processing {symbol}: {e}")
            return None
    
    def should_trigger_alert(self, final_decision: FinalDecision, min_confidence: float = 0.7) -> bool:
        """
        Determine if consensus decision should trigger an alert
        
        Args:
            final_decision: Result from multi-agent consensus
            min_confidence: Minimum confidence threshold for alerts
            
        Returns:
            True if should alert, False otherwise
        """
        try:
            # Get dominant action and confidence
            dominant_action = max(final_decision.final_probs, key=final_decision.final_probs.get)
            confidence = final_decision.final_probs[dominant_action]
            
            # Only trigger alerts for BUY decisions above confidence threshold
            if dominant_action == "BUY" and confidence >= min_confidence:
                return True
                
            # Could add logic for high-confidence AVOID warnings
            # if dominant_action == "AVOID" and confidence >= 0.8:
            #     return True  # Warning alert
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining alert trigger: {e}")
            return False
    
    def format_alert_message(self, symbol: str, final_decision: FinalDecision) -> str:
        """
        Format alert message from consensus decision
        
        Args:
            symbol: Token symbol
            final_decision: Consensus result
            
        Returns:
            Formatted alert message
        """
        try:
            dominant_action = max(final_decision.final_probs, key=final_decision.final_probs.get)
            confidence = final_decision.final_probs[dominant_action]
            
            # Build probability summary
            prob_summary = " | ".join([
                f"{action}: {prob:.2f}" 
                for action, prob in final_decision.final_probs.items()
                if prob >= 0.15  # Only show significant probabilities
            ])
            
            # Build evidence summary
            evidence_summary = " | ".join(final_decision.top_evidence[:3])
            
            message = f"🤖 Multi-Agent Consensus: {symbol}\n"
            message += f"📊 Decision: {dominant_action} ({confidence:.1%})\n"
            message += f"🎯 Probabilities: {prob_summary}\n"
            message += f"🔍 Key Evidence: {evidence_summary}\n"
            message += f"💭 Rationale: {final_decision.rationale}"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting alert message: {e}")
            return f"Multi-Agent Alert: {symbol} - {dominant_action} (Error formatting details)"

# Global instance for main scanner integration
multi_agent_scanner = MultiAgentScanner()