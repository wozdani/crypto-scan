#!/usr/bin/env python3
"""
Feedback Integration Module
Integruje system uczenia się z głównym skannerem TJDE.

Funkcje:
- Automatyczne zapisywanie wyników alertów
- Integracja z systemem alertów Telegram
- Okresowe uruchamianie feedback loop
- Monitoring skuteczności decyzji
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import asyncio

# Import feedback loop (with error handling for circular imports)
try:
    from feedback_loop import TJDEFeedbackLoop, record_trading_result
except ImportError:
    print("[FEEDBACK INTEGRATION] Warning: Could not import feedback_loop module")
    TJDEFeedbackLoop = None
    record_trading_result = None

class FeedbackIntegration:
    """
    Integration layer between TJDE scanner and feedback learning system
    """
    
    def __init__(self):
        self.feedback_system = TJDEFeedbackLoop() if TJDEFeedbackLoop else None
        self.last_learning_run = None
        self.learning_interval_hours = 6  # Run learning every 6 hours
        print("[FEEDBACK INTEGRATION] Feedback integration initialized")
    
    def record_tjde_alert(
        self,
        symbol: str,
        tjde_result: Dict[str, Any],
        current_price: float,
        market_phase: str = "unknown"
    ) -> bool:
        """
        Record TJDE alert for learning purposes
        
        Args:
            symbol: Trading symbol
            tjde_result: Complete TJDE analysis result
            current_price: Current market price
            market_phase: Detected market phase
            
        Returns:
            True if recorded successfully
        """
        
        if not self.feedback_system:
            return False
        
        try:
            # Extract key information from TJDE result
            score = tjde_result.get("final_score", 0.0)
            decision = tjde_result.get("enhanced_decision", "unknown")
            
            # Extract score components from breakdown
            score_components = {}
            if "tjde_breakdown" in tjde_result:
                breakdown = tjde_result["tjde_breakdown"]
                
                # Map breakdown components to feedback system format
                component_mapping = {
                    "pre_breakout_structure": ["structure_score", "consolidation_score", "breakout_structure"],
                    "volume_structure": ["volume_score", "volume_analysis", "volume_trend"],
                    "liquidity_behavior": ["liquidity_score", "orderbook_score", "liquidity_analysis"],
                    "clip_confidence": ["clip_confidence", "vision_confidence"],
                    "gpt_label_match": ["gpt_match", "label_consistency", "gpt_confidence"],
                    "heatmap_window": ["heatmap_score", "resistance_gap"],
                    "orderbook_setup": ["orderbook_quality", "bid_strength"],
                    "market_phase_modifier": ["phase_bonus", "market_context"]
                }
                
                for feedback_component, breakdown_keys in component_mapping.items():
                    for key in breakdown_keys:
                        if key in breakdown:
                            score_components[feedback_component] = breakdown[key]
                            break
                    else:
                        score_components[feedback_component] = 0.0
            
            # Ensure we have reasonable default components
            if not score_components:
                # Create default component distribution based on score
                base_weight = score / 8  # Distribute evenly across 8 components
                score_components = {
                    "pre_breakout_structure": base_weight * 1.2,  # Slightly higher
                    "volume_structure": base_weight * 1.1,
                    "liquidity_behavior": base_weight * 0.9,
                    "clip_confidence": base_weight * 0.8,
                    "gpt_label_match": base_weight * 0.8,
                    "heatmap_window": base_weight * 0.7,
                    "orderbook_setup": base_weight * 0.6,
                    "market_phase_modifier": base_weight * 0.5
                }
            
            # Record the alert
            self.feedback_system.record_alert_result(
                symbol=symbol,
                phase=market_phase,
                score=score,
                decision=decision,
                score_components=score_components,
                entry_price=current_price,
                alert_time=datetime.now().isoformat(),
                was_successful=None,  # Will be determined later
                hours_tracked=2  # Track for 2 hours
            )
            
            print(f"[FEEDBACK INTEGRATION] Recorded alert: {symbol} {decision} {score:.3f}")
            return True
            
        except Exception as e:
            print(f"[FEEDBACK INTEGRATION] Error recording alert {symbol}: {e}")
            return False
    
    def should_run_learning(self) -> bool:
        """Check if it's time to run learning cycle"""
        if not self.last_learning_run:
            return True
        
        time_since_last = datetime.now() - self.last_learning_run
        return time_since_last.total_seconds() > (self.learning_interval_hours * 3600)
    
    def run_learning_cycle(self) -> Optional[Dict[str, Any]]:
        """
        Run feedback learning cycle if conditions are met
        
        Returns:
            Learning results or None if not run
        """
        
        if not self.feedback_system:
            return None
        
        if not self.should_run_learning():
            return None
        
        try:
            print("[FEEDBACK INTEGRATION] Starting learning cycle...")
            result = self.feedback_system.run_feedback_loop()
            self.last_learning_run = datetime.now()
            
            if result.get("profiles_updated", 0) > 0:
                print(f"[FEEDBACK INTEGRATION] Learning completed: {result['profiles_updated']} profiles updated")
            else:
                print("[FEEDBACK INTEGRATION] Learning completed: No profile updates")
            
            return result
            
        except Exception as e:
            print(f"[FEEDBACK INTEGRATION] Error in learning cycle: {e}")
            return None
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status"""
        
        if not self.feedback_system:
            return {"status": "disabled", "reason": "feedback_system_not_available"}
        
        # Count pending and completed results
        pending_count = sum(1 for r in self.feedback_system.feedback_data if r.get("status") == "pending")
        completed_count = sum(1 for r in self.feedback_system.feedback_data if r.get("status") == "completed")
        
        # Calculate time since last learning
        time_since_learning = None
        if self.last_learning_run:
            time_since_learning = (datetime.now() - self.last_learning_run).total_seconds() / 3600
        
        return {
            "status": "active",
            "pending_results": pending_count,
            "completed_results": completed_count,
            "total_results": len(self.feedback_system.feedback_data),
            "hours_since_last_learning": time_since_learning,
            "next_learning_in_hours": max(0, self.learning_interval_hours - (time_since_learning or 0)) if time_since_learning else 0,
            "learning_ready": self.should_run_learning()
        }
    
    def update_pending_results(self) -> int:
        """
        Update pending trading results
        
        Returns:
            Number of results updated
        """
        
        if not self.feedback_system:
            return 0
        
        try:
            before_count = sum(1 for r in self.feedback_system.feedback_data if r.get("status") == "pending")
            self.feedback_system.update_pending_results()
            after_count = sum(1 for r in self.feedback_system.feedback_data if r.get("status") == "pending")
            
            updated = before_count - after_count
            if updated > 0:
                print(f"[FEEDBACK INTEGRATION] Updated {updated} pending results")
            
            return updated
            
        except Exception as e:
            print(f"[FEEDBACK INTEGRATION] Error updating pending results: {e}")
            return 0

# Global instance for easy access
_feedback_integration = None

def get_feedback_integration() -> FeedbackIntegration:
    """Get global feedback integration instance"""
    global _feedback_integration
    if _feedback_integration is None:
        _feedback_integration = FeedbackIntegration()
    return _feedback_integration

def record_tjde_decision(
    symbol: str,
    tjde_result: Dict[str, Any],
    current_price: float,
    market_phase: str = "unknown"
) -> bool:
    """
    Convenience function to record TJDE decision
    Can be called from scanner modules
    """
    integration = get_feedback_integration()
    return integration.record_tjde_alert(symbol, tjde_result, current_price, market_phase)

def run_periodic_learning() -> Optional[Dict[str, Any]]:
    """
    Run learning cycle if ready
    Can be called from main scanner loop
    """
    integration = get_feedback_integration()
    return integration.run_learning_cycle()

def get_learning_status() -> Dict[str, Any]:
    """Get current learning system status"""
    integration = get_feedback_integration()
    return integration.get_learning_status()

def update_trading_results() -> int:
    """Update pending trading results"""
    integration = get_feedback_integration()
    return integration.update_pending_results()

async def periodic_feedback_worker():
    """
    Async worker for periodic feedback operations
    Can be run as background task
    """
    integration = get_feedback_integration()
    
    while True:
        try:
            # Update pending results every 30 minutes
            integration.update_pending_results()
            
            # Run learning cycle if ready
            integration.run_learning_cycle()
            
            # Wait 30 minutes before next check
            await asyncio.sleep(30 * 60)
            
        except Exception as e:
            print(f"[FEEDBACK INTEGRATION] Error in periodic worker: {e}")
            await asyncio.sleep(5 * 60)  # Wait 5 minutes on error

if __name__ == "__main__":
    # Test feedback integration
    integration = FeedbackIntegration()
    status = integration.get_learning_status()
    print(f"[FEEDBACK INTEGRATION] Status: {status}")