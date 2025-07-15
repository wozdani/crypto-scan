#!/usr/bin/env python3
"""
Stage 6/7: DiamondWhale AI Scheduler
Daily Training + Feedback Loop Automation

Automatycznie uruchamia:
- Feedback loop dla DiamondWhale AI (evaluate_alerts_after_delay)
- Zapis checkpointa modelu RL oraz logów decyzji
- QIRL Agent training na podstawie real trading outcomes
"""

import os
import sys
import schedule
import time
import logging
import threading
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from feedback.feedback_loop_diamond import (
    evaluate_diamond_alerts_after_delay,
    run_diamond_daily_evaluation,
    get_diamond_feedback_loop
)

# Import RLAgentV4 trainer for Stage 7/7
try:
    from daily_jobs.rl_weights_trainer import run_daily_training
    RL_TRAINER_AVAILABLE = True
    logger.info("✅ [DIAMOND SCHEDULER] RLAgentV4 trainer imported successfully")
except ImportError as e:
    RL_TRAINER_AVAILABLE = False
    logger.warning(f"⚠️ [DIAMOND SCHEDULER] RLAgentV4 trainer not available: {e}")


def job_rl_weights_trainer():
    """
    Stage 7/7: Daily RLAgentV4 training job
    Uruchamia się codziennie o 01:45 UTC (15 minut przed DiamondWhale)
    """
    current_time = datetime.utcnow().isoformat()
    logger.info(f"🧠 [DIAMOND SCHEDULER] Stage 7/7: Starting RLAgentV4 training @ {current_time}")
    
    try:
        if not RL_TRAINER_AVAILABLE:
            logger.warning("[DIAMOND SCHEDULER] ⚠️ RLAgentV4 trainer not available - skipping")
            return False
        
        # Run daily RLAgentV4 training
        logger.info("[DIAMOND SCHEDULER] Running RLAgentV4 daily training...")
        training_results = run_daily_training(hours_back=24)
        
        logger.info(f"[DIAMOND SCHEDULER] RLAgentV4 training results:")
        logger.info(f"  📊 Status: {training_results.get('status', 'unknown')}")
        logger.info(f"  🎯 Alerts processed: {training_results.get('alerts_processed', 0)}")
        logger.info(f"  🧠 Training samples: {training_results.get('training_samples', 0)}")
        logger.info(f"  ✅ Successful alerts: {training_results.get('successful_alerts', 0)}")
        logger.info(f"  ❌ Failed alerts: {training_results.get('failed_alerts', 0)}")
        
        if training_results.get('status') == 'success':
            logger.info(f"  📈 Success rate: {training_results.get('success_rate', 0):.1f}%")
            logger.info(f"  🔄 Total updates: {training_results.get('total_updates', 0)}")
        
        # Save scheduler execution log
        rl_scheduler_log = {
            "timestamp": current_time,
            "training_results": training_results,
            "execution_type": "scheduled_daily_rl_training",
            "stage": "7/7"
        }
        
        # Write scheduler log
        os.makedirs("cache/scheduler", exist_ok=True)
        rl_scheduler_log_file = "cache/scheduler/rl_training_scheduler_log.jsonl"
        
        import json
        with open(rl_scheduler_log_file, "a") as f:
            f.write(f"{json.dumps(rl_scheduler_log)}\n")
        
        logger.info(f"[DIAMOND SCHEDULER] ✅ RLAgentV4 training completed successfully")
        logger.info(f"[DIAMOND SCHEDULER] Training log saved: {rl_scheduler_log_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"[DIAMOND SCHEDULER] ❌ RLAgentV4 training failed: {e}")
        return False


def job_feedback_loop():
    """
    Daily DiamondWhale AI feedback loop job
    Uruchamia się codziennie o 02:00 UTC
    """
    current_time = datetime.utcnow().isoformat()
    logger.info(f"🔥 [DIAMOND SCHEDULER] Starting DiamondWhale feedback loop @ {current_time}")
    
    try:
        # Get feedback loop instance
        feedback_loop = get_diamond_feedback_loop()
        
        # Run comprehensive daily evaluation
        logger.info("[DIAMOND SCHEDULER] Running daily evaluation...")
        daily_results = run_diamond_daily_evaluation()
        
        logger.info(f"[DIAMOND SCHEDULER] Daily evaluation results:")
        logger.info(f"  📊 Evaluated alerts: {daily_results.get('evaluated', 0)}")
        logger.info(f"  ✅ Success rate: {daily_results.get('success_rate', 0.0):.1%}")
        logger.info(f"  🎯 Successful alerts: {daily_results.get('total_successful', 0)}")
        logger.info(f"  ❌ Failed alerts: {daily_results.get('total_failed', 0)}")
        
        # Run specific delay-based evaluation (60 minutes)
        logger.info("[DIAMOND SCHEDULER] Running 60-minute delay evaluation...")
        delay_results = evaluate_diamond_alerts_after_delay(delay_minutes=60, threshold=0.05)
        
        logger.info(f"[DIAMOND SCHEDULER] Delay evaluation results:")
        logger.info(f"  📊 Evaluated: {delay_results.get('evaluated', 0)}")
        logger.info(f"  ✅ Success rate: {delay_results.get('success_rate', 0.0):.1%}")
        
        # Get QIRL agent statistics after training
        qirl_agent = feedback_loop.initialize_qirl_agent()
        if qirl_agent:
            stats = qirl_agent.get_statistics()
            logger.info(f"[DIAMOND SCHEDULER] QIRL Agent statistics:")
            logger.info(f"  🤖 Total decisions: {stats.get('total_decisions', 0)}")
            logger.info(f"  🎯 Accuracy: {stats.get('accuracy', 0.0):.1f}%")
            logger.info(f"  💾 Memory size: {stats.get('memory_size', 0)}")
        
        # Save scheduler execution log
        scheduler_log = {
            "timestamp": current_time,
            "daily_results": daily_results,
            "delay_results": delay_results,
            "qirl_stats": stats if qirl_agent else {},
            "execution_type": "scheduled_daily"
        }
        
        # Write scheduler log
        os.makedirs("cache/scheduler", exist_ok=True)
        scheduler_log_file = "cache/scheduler/diamond_scheduler_log.jsonl"
        
        import json
        with open(scheduler_log_file, "a") as f:
            f.write(f"{json.dumps(scheduler_log)}\n")
        
        logger.info(f"[DIAMOND SCHEDULER] ✅ Daily job completed successfully")
        logger.info(f"[DIAMOND SCHEDULER] Scheduler log saved: {scheduler_log_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"[DIAMOND SCHEDULER] ❌ Daily job failed: {e}")
        return False


def job_hourly_check():
    """
    Hourly check for pending alerts evaluation
    Uruchamia się co godzinę o :30
    """
    try:
        logger.info("[DIAMOND SCHEDULER] Running hourly pending alerts check...")
        
        feedback_loop = get_diamond_feedback_loop()
        stats = feedback_loop.get_feedback_statistics()
        
        pending_count = stats.get('pending_evaluations', 0)
        
        if pending_count >= 3:  # Lower threshold for more frequent evaluation
            logger.info(f"[DIAMOND SCHEDULER] 🔄 Found {pending_count} pending alerts - running evaluation")
            
            # Run evaluation for alerts older than 1 hour
            results = evaluate_diamond_alerts_after_delay(delay_minutes=60, threshold=0.05)
            
            logger.info(f"[DIAMOND SCHEDULER] Hourly evaluation: {results.get('evaluated', 0)} alerts processed")
            return True
        else:
            logger.info(f"[DIAMOND SCHEDULER] 💤 Only {pending_count} pending alerts - skipping evaluation")
            return False
            
    except Exception as e:
        logger.error(f"[DIAMOND SCHEDULER] ❌ Hourly check failed: {e}")
        return False


def job_model_checkpoint():
    """
    Model checkpoint job - saves QIRL agent state
    Uruchamia się codziennie o 02:15 UTC (15 minut po feedback loop)
    """
    try:
        logger.info("[DIAMOND SCHEDULER] Creating model checkpoint...")
        
        feedback_loop = get_diamond_feedback_loop()
        qirl_agent = feedback_loop.initialize_qirl_agent()
        
        if qirl_agent:
            # Save agent checkpoint
            checkpoint_dir = "cache/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            checkpoint_file = f"{checkpoint_dir}/qirl_agent_checkpoint_{timestamp}.json"
            
            # Get agent state
            agent_state = {
                "timestamp": datetime.utcnow().isoformat(),
                "statistics": qirl_agent.get_statistics(),
                "memory_size": qirl_agent.get_statistics().get('memory_size', 0),
                "accuracy": qirl_agent.get_statistics().get('accuracy', 0.0),
                "total_decisions": qirl_agent.get_statistics().get('total_decisions', 0),
                "checkpoint_type": "daily_scheduled"
            }
            
            import json
            with open(checkpoint_file, "w") as f:
                json.dump(agent_state, f, indent=2)
            
            logger.info(f"[DIAMOND SCHEDULER] ✅ Checkpoint saved: {checkpoint_file}")
            logger.info(f"[DIAMOND SCHEDULER] Agent accuracy: {agent_state['accuracy']:.1f}%")
            logger.info(f"[DIAMOND SCHEDULER] Agent decisions: {agent_state['total_decisions']}")
            
            return True
        else:
            logger.warning("[DIAMOND SCHEDULER] ⚠️ QIRL agent not available for checkpoint")
            return False
            
    except Exception as e:
        logger.error(f"[DIAMOND SCHEDULER] ❌ Checkpoint failed: {e}")
        return False


def schedule_diamond_jobs():
    """
    Planuje wszystkie Diamond Scheduler jobs
    """
    logger.info("📅 [DIAMOND SCHEDULER] Scheduling DiamondWhale AI jobs...")
    
    # Stage 7/7: Daily RLAgentV4 training at 01:45 UTC (before feedback loop)
    if RL_TRAINER_AVAILABLE:
        schedule.every().day.at("01:45").do(job_rl_weights_trainer)
        logger.info("✅ [DIAMOND SCHEDULER] Stage 7/7: RLAgentV4 training scheduled")
    else:
        logger.warning("⚠️ [DIAMOND SCHEDULER] RLAgentV4 trainer not available - skipping scheduling")
    
    # Daily feedback loop at 02:00 UTC
    schedule.every().day.at("02:00").do(job_feedback_loop)
    
    # Model checkpoint at 02:15 UTC (after feedback loop)
    schedule.every().day.at("02:15").do(job_model_checkpoint)
    
    # Hourly check for pending alerts at :30
    schedule.every().hour.at(":30").do(job_hourly_check)
    
    logger.info("✅ [DIAMOND SCHEDULER] Jobs scheduled:")
    if RL_TRAINER_AVAILABLE:
        logger.info("  • Stage 7/7 RLAgentV4 training: 01:45 UTC")
    logger.info("  • Daily feedback loop: 02:00 UTC")
    logger.info("  • Model checkpoint: 02:15 UTC")
    logger.info("  • Hourly pending check: every hour at :30")


def run_scheduler():
    """
    Main scheduler loop dla DiamondWhale AI
    Uruchamia się jako background thread
    """
    logger.info("🚀 [DIAMOND SCHEDULER] Starting DiamondWhale AI scheduler...")
    
    # Schedule jobs
    schedule_diamond_jobs()
    
    # Initial test run
    logger.info("🧪 [DIAMOND SCHEDULER] Running initial test...")
    job_hourly_check()
    
    # Main scheduler loop
    logger.info("⏰ [DIAMOND SCHEDULER] Scheduler running - waiting for scheduled jobs...")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            logger.info("🛑 [DIAMOND SCHEDULER] Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"[DIAMOND SCHEDULER] ❌ Scheduler error: {e}")
            time.sleep(300)  # Wait 5 minutes on error


def start_diamond_scheduler_thread():
    """
    Uruchamia Diamond Scheduler jako daemon thread
    Do użycia w crypto_scan_service.py
    """
    logger.info("🔗 [DIAMOND SCHEDULER] Starting as background thread...")
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    logger.info("✅ [DIAMOND SCHEDULER] Background thread started")
    return scheduler_thread


def manual_run():
    """
    Manual run dla testowania
    """
    print("🔧 MANUAL DIAMOND SCHEDULER RUN")
    print("=" * 50)
    
    try:
        # Test feedback loop
        print("📊 Running feedback loop...")
        feedback_result = job_feedback_loop()
        
        # Test model checkpoint
        print("💾 Creating model checkpoint...")
        checkpoint_result = job_model_checkpoint()
        
        # Test hourly check
        print("⏰ Running hourly check...")
        hourly_result = job_hourly_check()
        
        print(f"\n✅ RESULTS:")
        print(f"  Feedback loop: {'✅' if feedback_result else '❌'}")
        print(f"  Model checkpoint: {'✅' if checkpoint_result else '❌'}")
        print(f"  Hourly check: {'✅' if hourly_result else '❌'}")
        
        return all([feedback_result, checkpoint_result, hourly_result])
        
    except Exception as e:
        print(f"❌ Manual run error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        # Manual run mode
        manual_run()
    else:
        # Scheduled mode
        run_scheduler()