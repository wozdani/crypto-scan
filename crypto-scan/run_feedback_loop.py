#!/usr/bin/env python3
"""
Run Feedback Loop - Uruchamia samouczący się feedback loop

Analizuje skuteczność alertów i automatycznie aktualizuje wagi scoringowe
"""

import os
import sys
import schedule
import time
from datetime import datetime, timezone

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.score_accuracy_analysis import analyze_alert_performance, save_performance_analysis
from utils.update_advanced_weights import run_feedback_loop


def run_daily_feedback_analysis():
    """
    Uruchamia dzienną analizę feedback i aktualizację wag
    """
    print(f"\n🔄 Starting daily feedback analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Analyze alert performance
        print("📊 Step 1: Analyzing alert performance...")
        
        alerts_path = "data/alerts/alerts_history.json"
        logs, success_rate = analyze_alert_performance(
            alerts_path=alerts_path,
            candle_loader=None,  # Use simulated data for now - replace with real candle loader
            hours_to_check=2,
            success_threshold=2.0
        )
        
        if not logs:
            print("⚠️ No alerts found for analysis")
            return
        
        print(f"📈 Performance Analysis Results:")
        print(f"  Total alerts analyzed: {len(logs)}")
        print(f"  Success rate: {success_rate:.1%}")
        
        # Step 2: Save performance analysis
        performance_file = "logs/tjde_performance_analysis.json"
        save_performance_analysis(logs, success_rate, performance_file)
        
        # Step 3: Update weights based on performance
        print("🧠 Step 3: Updating adaptive weights...")
        
        weights_updated = run_feedback_loop(performance_file)
        
        if weights_updated:
            print("✅ Feedback loop completed successfully - weights updated")
        else:
            print("⚠️ Feedback loop completed but no weights were updated")
        
        # Step 4: Log feedback summary
        _log_feedback_summary(len(logs), success_rate, weights_updated)
        
    except Exception as e:
        print(f"❌ Error in daily feedback analysis: {e}")
        import traceback
        traceback.print_exc()


def run_weekly_deep_analysis():
    """
    Uruchamia tygodniową głęboką analizę systemu
    """
    print(f"\n🔬 Starting weekly deep analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Analyze longer period with more aggressive weight adjustments
        alerts_path = "data/alerts/alerts_history.json"
        logs, success_rate = analyze_alert_performance(
            alerts_path=alerts_path,
            hours_to_check=6,  # Longer timeframe
            success_threshold=3.0  # Higher success threshold
        )
        
        if logs:
            # More aggressive learning for weekly updates
            from utils.update_advanced_weights import update_weights_based_on_performance
            update_weights_based_on_performance(
                logs, 
                learning_rate=0.02  # Double the normal learning rate
            )
            
            print(f"✅ Weekly deep analysis complete - {len(logs)} alerts analyzed")
        
    except Exception as e:
        print(f"❌ Error in weekly analysis: {e}")


def _log_feedback_summary(alerts_count: int, success_rate: float, weights_updated: bool):
    """Log feedback loop summary"""
    try:
        import json
        
        os.makedirs("logs", exist_ok=True)
        
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alerts_analyzed": alerts_count,
            "success_rate": round(success_rate, 4),
            "weights_updated": weights_updated,
            "feedback_type": "daily_automatic"
        }
        
        with open("logs/feedback_loop_history.jsonl", "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(summary)}\n")
        
    except Exception as e:
        print(f"⚠️ Error logging feedback summary: {e}")


def schedule_feedback_loops():
    """
    Planuje automatyczne uruchamianie feedback loops
    """
    print("📅 Scheduling automatic feedback loops...")
    
    # Daily analysis at 06:00
    schedule.every().day.at("06:00").do(run_daily_feedback_analysis)
    
    # Weekly analysis on Sunday at 03:00
    schedule.every().sunday.at("03:00").do(run_weekly_deep_analysis)
    
    print("✅ Feedback loops scheduled:")
    print("  • Daily analysis: 06:00 UTC")
    print("  • Weekly deep analysis: Sunday 03:00 UTC")


def run_immediate_feedback():
    """
    Uruchamia natychmiastową analizę feedback (do testów)
    """
    print("🚀 Running immediate feedback analysis...")
    run_daily_feedback_analysis()


if __name__ == "__main__":
    print("🧠 TJDE Adaptive Feedback Loop System")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "run":
            run_immediate_feedback()
        elif command == "schedule":
            schedule_feedback_loops()
            print("🔄 Running scheduled feedback loop system...")
            print("Press Ctrl+C to stop")
            
            try:
                while True:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                print("\n🛑 Feedback loop system stopped")
        elif command == "test":
            # Test mode with sample data
            print("🧪 Running in test mode...")
            run_immediate_feedback()
        else:
            print(f"❌ Unknown command: {command}")
            print("Usage: python run_feedback_loop.py [run|schedule|test]")
    else:
        print("Available commands:")
        print("  run      - Run immediate feedback analysis")
        print("  schedule - Start scheduled feedback system")
        print("  test     - Run test mode with sample data")
        print("\nExample: python run_feedback_loop.py run")