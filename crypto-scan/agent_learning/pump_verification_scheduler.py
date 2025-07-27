#!/usr/bin/env python3
"""
Pump Verification Scheduler
Automatycznie uruchamia pump verification co 6 godzin
Integruje się z agent learning system
"""

import threading
import time
import schedule
from datetime import datetime
import sys
import os

# Add crypto-scan to path
sys.path.append('/home/runner/workspace/crypto-scan')

from agent_learning.pump_verification import PumpVerificationSystem

class PumpVerificationScheduler:
    def __init__(self):
        self.verifier = PumpVerificationSystem()
        self.is_running = False
        self.scheduler_thread = None
        
    def run_verification_job(self):
        """Job function dla scheduled verification"""
        try:
            print(f"[PUMP SCHEDULER] Starting scheduled verification at {datetime.now()}")
            results = self.verifier.run_verification_cycle()
            
            if results:
                print(f"[PUMP SCHEDULER] Verified {len(results)} pump outcomes")
                # Log important results
                for result in results:
                    symbol = result["symbol"]
                    pump_level = result["pump_classification"]["pump_level"]
                    pump_pct = result["pump_classification"]["pump_percentage"]
                    agents_correct = result["agents_accuracy"]["correct_decision"]
                    
                    status = "✅ CORRECT" if agents_correct else "❌ WRONG"
                    print(f"[PUMP SCHEDULER] {symbol}: {pump_level} ({pump_pct:.1f}%) - Agents {status}")
            else:
                print("[PUMP SCHEDULER] No pumps ready for verification")
                
        except Exception as e:
            print(f"[PUMP SCHEDULER ERROR] During verification: {e}")
    
    def start_scheduler(self):
        """Start pump verification scheduler"""
        if self.is_running:
            print("[PUMP SCHEDULER] Already running")
            return
            
        # Schedule verification every 6 hours
        schedule.every(6).hours.do(self.run_verification_job)
        
        # Also run immediately at startup (check for any pending)
        schedule.every().minute.do(self.run_verification_job).tag('startup')
        
        self.is_running = True
        
        def scheduler_worker():
            """Worker thread dla scheduler"""
            print("[PUMP SCHEDULER] Started - will verify pumps every 6 hours")
            
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                    
                    # Remove startup job after first run
                    if schedule.get_jobs('startup'):
                        schedule.clear('startup')
                        
                except Exception as e:
                    print(f"[PUMP SCHEDULER ERROR] In scheduler loop: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        self.scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
        self.scheduler_thread.start()
        
        print("[PUMP SCHEDULER] Background scheduler started")
    
    def stop_scheduler(self):
        """Stop scheduler"""
        self.is_running = False
        schedule.clear()
        print("[PUMP SCHEDULER] Stopped")
    
    def force_verification(self):
        """Force immediate verification (for testing)"""
        print("[PUMP SCHEDULER] Running forced verification...")
        return self.run_verification_job()

# Global scheduler instance
pump_scheduler = PumpVerificationScheduler()

def start_pump_verification_scheduler():
    """Start pump verification scheduler"""
    pump_scheduler.start_scheduler()

def stop_pump_verification_scheduler():
    """Stop pump verification scheduler"""
    pump_scheduler.stop_scheduler()

def force_pump_verification():
    """Force immediate pump verification"""
    return pump_scheduler.force_verification()

if __name__ == "__main__":
    # For testing
    print("Starting pump verification scheduler...")
    start_pump_verification_scheduler()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping scheduler...")
        stop_pump_verification_scheduler()