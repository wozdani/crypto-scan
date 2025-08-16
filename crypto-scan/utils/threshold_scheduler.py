"""
Threshold-Aware Learning Scheduler

Periodic updates of detector weights based on threshold awareness data.
Runs daily to optimize detector weights using 0.7 threshold understanding.
"""

import schedule
import time
import threading
from datetime import datetime
from typing import Dict, Any


class ThresholdLearningScheduler:
    """
    Scheduler for threshold-aware weight updates
    """
    
    def __init__(self):
        self.running = False
        self.thread = None
        
    def start_scheduler(self):
        """Start the threshold learning scheduler"""
        if self.running:
            print("[THRESHOLD SCHEDULER] Already running")
            return
            
        self.running = True
        
        # Schedule daily weight updates at 03:00 UTC
        schedule.every().day.at("03:00").do(self._update_detector_weights)
        
        # Schedule periodic saves every 2 hours
        schedule.every(2).hours.do(self._save_threshold_data)
        
        # Start scheduler thread
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        
        print("[THRESHOLD SCHEDULER] ✅ Started threshold-aware learning scheduler")
        print("[THRESHOLD SCHEDULER] - Daily weight updates at 03:00 UTC")
        print("[THRESHOLD SCHEDULER] - Threshold tracking saves every 2 hours")
        
    def stop_scheduler(self):
        """Stop the threshold learning scheduler"""
        self.running = False
        schedule.clear()
        print("[THRESHOLD SCHEDULER] Stopped threshold-aware learning scheduler")
        
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    def _update_detector_weights(self):
        """Update detector weights based on threshold awareness"""
        try:
            from ..stealth_engine.consensus_decision_engine import ConsensusDecisionEngine
            
            print("[THRESHOLD SCHEDULER] Starting daily weight update...")
            
            # Create consensus engine and update weights
            consensus_engine = ConsensusDecisionEngine()
            updated_weights = consensus_engine.update_weights_with_threshold_awareness(days=7)
            
            # Save updated threshold data
            consensus_engine.save_threshold_tracking()
            
            print(f"[THRESHOLD SCHEDULER] ✅ Updated {len(updated_weights)} detector weights")
            for detector, weight in updated_weights.items():
                print(f"  📊 {detector}: {weight:.4f}")
                
        except Exception as e:
            print(f"[THRESHOLD SCHEDULER ERROR] Failed to update weights: {e}")
            
    def _save_threshold_data(self):
        """Save threshold tracking data"""
        try:
            from ..stealth_engine.consensus_decision_engine import ConsensusDecisionEngine
            
            consensus_engine = ConsensusDecisionEngine()
            consensus_engine.save_threshold_tracking()
            
            print("[THRESHOLD SCHEDULER] ✅ Saved threshold tracking data")
            
        except Exception as e:
            print(f"[THRESHOLD SCHEDULER ERROR] Failed to save threshold data: {e}")


# Global scheduler instance
_threshold_scheduler = None


def get_threshold_scheduler() -> ThresholdLearningScheduler:
    """Get global threshold learning scheduler instance"""
    global _threshold_scheduler
    if _threshold_scheduler is None:
        _threshold_scheduler = ThresholdLearningScheduler()
    return _threshold_scheduler


def start_threshold_learning():
    """Start the threshold-aware learning system"""
    scheduler = get_threshold_scheduler()
    scheduler.start_scheduler()


def stop_threshold_learning():
    """Stop the threshold-aware learning system"""
    scheduler = get_threshold_scheduler()
    scheduler.stop_scheduler()