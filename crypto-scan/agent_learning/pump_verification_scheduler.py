#!/usr/bin/env python3
"""
Pump Verification Scheduler
Scheduler dla weryfikacji pump po 2h i automatyczne nazewnictwo plików explore mode.
Uruchamia się co 2h i sprawdza czy były pumpy.
"""

import schedule
import time
import threading
from datetime import datetime
import json
import os
from typing import Dict, List
try:
    from .explore_file_manager import ExploreFileManager
except ImportError:
    from explore_file_manager import ExploreFileManager

class PumpVerificationScheduler:
    def __init__(self):
        self.file_manager = ExploreFileManager()
        self.running = False
        self.scheduler_thread = None
        
    def verification_job(self):
        """
        Job function uruchamiany co 2h dla weryfikacji pump.
        """
        try:
            print(f"[PUMP SCHEDULER] Starting scheduled 2h verification at {datetime.now()}")
            
            # Uruchom cykl weryfikacji
            results = self.file_manager.run_verification_cycle()
            
            # Log wyników
            self.log_verification_results(results)
            
            print(f"[PUMP SCHEDULER] Scheduled verification complete: {results}")
            
        except Exception as e:
            print(f"[PUMP SCHEDULER ERROR] Verification job failed: {e}")
    
    def log_verification_results(self, results: Dict):
        """
        Zapisz wyniki weryfikacji do pliku log.
        
        Args:
            results: Wyniki z run_verification_cycle
        """
        try:
            log_file = "crypto-scan/cache/pump_verification_log.json"
            
            # Load existing log
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {"verification_history": []}
            
            # Add new entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "job_type": "scheduled_2h_verification"
            }
            
            log_data["verification_history"].append(log_entry)
            
            # Keep only last 100 entries
            if len(log_data["verification_history"]) > 100:
                log_data["verification_history"] = log_data["verification_history"][-100:]
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"[PUMP SCHEDULER LOG ERROR] Failed to log results: {e}")
    
    def start_scheduler(self):
        """
        Uruchom scheduler w tle.
        """
        if self.running:
            print("[PUMP SCHEDULER] Already running")
            return
        
        print("[PUMP SCHEDULER] Starting 2h pump verification scheduler...")
        
        # Zaplanuj job co 2 godziny
        schedule.every(2).hours.do(self.verification_job)
        
        # Dodaj job o każdej parzystej godzinie UTC (co 2h)
        for hour in ["00:00", "02:00", "04:00", "06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"]:
            schedule.every().day.at(hour).do(self.verification_job)
        
        self.running = True
        
        # Uruchom w oddzielnym wątku
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        print("[PUMP SCHEDULER] Scheduler started successfully")
        print("[PUMP SCHEDULER] Next verification times: Every 2 hours (00:00, 02:00, 04:00, 06:00, 08:00, 10:00, 12:00, 14:00, 16:00, 18:00, 20:00, 22:00 UTC)")
    
    def _run_scheduler(self):
        """
        Wewnętrzna funkcja dla wątku scheduler.
        """
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"[PUMP SCHEDULER ERROR] Scheduler thread error: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def stop_scheduler(self):
        """
        Zatrzymaj scheduler.
        """
        print("[PUMP SCHEDULER] Stopping scheduler...")
        self.running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        schedule.clear()
        print("[PUMP SCHEDULER] Scheduler stopped")
    
    def manual_verification(self) -> Dict:
        """
        Uruchom weryfikację manualnie (na żądanie).
        
        Returns:
            Wyniki weryfikacji
        """
        print("[PUMP SCHEDULER] Running manual verification...")
        
        results = self.file_manager.run_verification_cycle()
        self.log_verification_results(results)
        
        print(f"[PUMP SCHEDULER] Manual verification complete: {results}")
        return results
    
    def get_scheduler_status(self) -> Dict:
        """
        Sprawdź status scheduler.
        
        Returns:
            Status scheduler i statystyki
        """
        next_run = None
        if schedule.jobs:
            next_run = min(job.next_run for job in schedule.jobs)
        
        return {
            "running": self.running,
            "next_run": next_run.isoformat() if next_run else None,
            "scheduled_jobs": len(schedule.jobs),
            "thread_alive": self.scheduler_thread.is_alive() if self.scheduler_thread else False
        }

# Global scheduler instance
_scheduler_instance = None

def get_scheduler() -> PumpVerificationScheduler:
    """
    Singleton pattern dla scheduler.
    
    Returns:
        Global scheduler instance
    """
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = PumpVerificationScheduler()
    return _scheduler_instance

def start_pump_verification_scheduler():
    """
    Start global pump verification scheduler.
    """
    scheduler = get_scheduler()
    scheduler.start_scheduler()

def stop_pump_verification_scheduler():
    """
    Stop global pump verification scheduler.
    """
    scheduler = get_scheduler()
    scheduler.stop_scheduler()

def manual_pump_verification() -> Dict:
    """
    Run manual pump verification.
    
    Returns:
        Verification results
    """
    scheduler = get_scheduler()
    return scheduler.manual_verification()

if __name__ == "__main__":
    # Test scheduler
    scheduler = PumpVerificationScheduler()
    
    print("Testing manual verification...")
    results = scheduler.manual_verification()
    print(f"Results: {results}")
    
    print("\nTesting scheduler start...")
    scheduler.start_scheduler()
    
    try:
        print("Scheduler running... Press Ctrl+C to stop")
        while True:
            status = scheduler.get_scheduler_status()
            print(f"Status: {status}")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        scheduler.stop_scheduler()