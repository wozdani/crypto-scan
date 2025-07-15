#!/usr/bin/env python3
"""
Daily Diamond Feedback Evaluation Job
Stage 5/7: Automatic evaluation of Diamond alerts effectiveness

Uruchamia się codziennie o 02:30 UTC i ewaluuje wszystkie pending Diamond alerts.
Aktualizuje QIRLAgent na podstawie real trading outcomes.
"""

import os
import sys
import schedule
import time
import logging
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feedback.feedback_loop_diamond import (
    get_diamond_feedback_loop, 
    run_diamond_daily_evaluation,
    evaluate_diamond_alerts_after_delay
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_daily_diamond_feedback_evaluation():
    """
    Codzienne uruchamianie Diamond Feedback Loop evaluation
    """
    logger.info("🔥 [DIAMOND FEEDBACK] Starting daily evaluation cycle")
    
    try:
        # Get feedback loop instance
        feedback_loop = get_diamond_feedback_loop()
        
        # Run comprehensive evaluation
        results = feedback_loop.run_daily_evaluation()
        
        # Log results
        logger.info(f"[DIAMOND FEEDBACK] Daily evaluation completed:")
        logger.info(f"  📊 Evaluated alerts: {results.get('evaluated', 0)}")
        logger.info(f"  ✅ Success rate: {results.get('success_rate', 0.0):.1%}")
        logger.info(f"  🎯 Total successful: {results.get('total_successful', 0)}")
        logger.info(f"  ❌ Total failed: {results.get('total_failed', 0)}")
        logger.info(f"  ⏳ Pending evaluations: {results.get('pending_evaluations', 0)}")
        
        # Get comprehensive statistics
        stats = feedback_loop.get_feedback_statistics()
        logger.info(f"[DIAMOND FEEDBACK] Overall system statistics:")
        logger.info(f"  📈 Total alerts: {stats.get('total_alerts', 0)}")
        logger.info(f"  📊 Evaluated alerts: {stats.get('evaluated_alerts', 0)}")
        logger.info(f"  🎯 Overall success rate: {stats.get('success_rate', 0.0):.1%}")
        logger.info(f"  📅 Last 24h alerts: {stats.get('alerts_24h', 0)}")
        
        # Save evaluation timestamp
        evaluation_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
            "stats": stats,
            "evaluation_type": "daily_scheduled"
        }
        
        # Write evaluation record
        os.makedirs("cache", exist_ok=True)
        eval_file = "cache/diamond_feedback_evaluations.jsonl"
        
        import json
        with open(eval_file, "a") as f:
            f.write(f"{json.dumps(evaluation_record)}\n")
        
        logger.info(f"[DIAMOND FEEDBACK] ✅ Daily evaluation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"[DIAMOND FEEDBACK] ❌ Daily evaluation failed: {e}")
        return False


def run_hourly_diamond_feedback_check():
    """
    Godzinne sprawdzenie pending alerts (jeśli jest >5 pending)
    """
    try:
        feedback_loop = get_diamond_feedback_loop()
        stats = feedback_loop.get_feedback_statistics()
        
        pending_count = stats.get('pending_evaluations', 0)
        
        if pending_count >= 5:
            logger.info(f"[DIAMOND FEEDBACK] 🔄 Hourly check: {pending_count} pending alerts - running evaluation")
            
            # Run evaluation for alerts older than 1 hour
            results = evaluate_diamond_alerts_after_delay(delay_minutes=60, threshold=0.05)
            
            logger.info(f"[DIAMOND FEEDBACK] Hourly evaluation: {results.get('evaluated', 0)} alerts processed")
            return True
        else:
            logger.info(f"[DIAMOND FEEDBACK] 💤 Hourly check: only {pending_count} pending alerts - skipping evaluation")
            return False
            
    except Exception as e:
        logger.error(f"[DIAMOND FEEDBACK] ❌ Hourly check failed: {e}")
        return False


def schedule_diamond_feedback_jobs():
    """
    Planuje automatyczne uruchamianie Diamond Feedback jobs
    """
    logger.info("📅 Scheduling Diamond Feedback evaluation jobs...")
    
    # Daily evaluation at 02:30 UTC
    schedule.every().day.at("02:30").do(run_daily_diamond_feedback_evaluation)
    
    # Hourly check for pending alerts
    schedule.every().hour.at(":00").do(run_hourly_diamond_feedback_check)
    
    logger.info("✅ Diamond Feedback jobs scheduled:")
    logger.info("  • Daily evaluation: 02:30 UTC")
    logger.info("  • Hourly pending check: every hour at :00")


def run_feedback_scheduler():
    """
    Main scheduler loop for Diamond Feedback evaluation
    """
    logger.info("🚀 [DIAMOND FEEDBACK] Starting scheduler...")
    
    # Schedule jobs
    schedule_diamond_feedback_jobs()
    
    # Initial run for testing
    logger.info("🧪 [DIAMOND FEEDBACK] Running initial evaluation test...")
    run_hourly_diamond_feedback_check()
    
    # Main scheduler loop
    logger.info("⏰ [DIAMOND FEEDBACK] Scheduler running - waiting for scheduled jobs...")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            logger.info("🛑 [DIAMOND FEEDBACK] Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"[DIAMOND FEEDBACK] ❌ Scheduler error: {e}")
            time.sleep(300)  # Wait 5 minutes on error


def manual_evaluation():
    """
    Manual evaluation function for testing
    """
    print("🔧 MANUAL DIAMOND FEEDBACK EVALUATION")
    print("=" * 50)
    
    try:
        # Run daily evaluation
        print("📊 Running daily evaluation...")
        results = run_daily_diamond_feedback_evaluation()
        
        if results:
            print("✅ Manual evaluation completed successfully")
        else:
            print("❌ Manual evaluation failed")
            
        # Get current statistics
        feedback_loop = get_diamond_feedback_loop()
        stats = feedback_loop.get_feedback_statistics()
        
        print("\n📈 CURRENT STATISTICS:")
        print(f"  Total alerts: {stats.get('total_alerts', 0)}")
        print(f"  Evaluated alerts: {stats.get('evaluated_alerts', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0.0):.1%}")
        print(f"  Pending evaluations: {stats.get('pending_evaluations', 0)}")
        print(f"  Last 24h alerts: {stats.get('alerts_24h', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual evaluation error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        # Manual evaluation mode
        manual_evaluation()
    else:
        # Scheduled mode
        run_feedback_scheduler()