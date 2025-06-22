#!/usr/bin/env python3
"""
Feedback Integration - Integruje feedback_loop_v2 z systemem TJDE

Automatyczne logowanie alertów i wywołanie feedback analysis
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any


def log_tjde_alert_for_feedback(symbol: str, decision_result: Dict, features: Dict) -> bool:
    """
    Loguje alert TJDE do systemu feedback v2
    
    Args:
        symbol: Trading symbol
        decision_result: Wynik z simulate_trader_decision_advanced()
        features: Features użyte w analizie
        
    Returns:
        bool: True jeśli log został zapisany
    """
    try:
        alert_log_file = "logs/alerts_history.jsonl"
        os.makedirs(os.path.dirname(alert_log_file), exist_ok=True)
        
        # Prepare alert data for feedback system
        alert_data = {
            "symbol": symbol,
            "decision": decision_result.get("decision", "unknown"),
            "final_score": decision_result.get("final_score", 0.0),
            "quality_grade": decision_result.get("quality_grade", "unknown"),
            "confidence": decision_result.get("confidence", 0.0),
            "alert_type": "TJDE_adaptive",
            "used_features": decision_result.get("used_features", features),
            "context_modifiers": decision_result.get("context_modifiers", []),
            "market_phase": decision_result.get("market_phase", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logged_for_feedback": True
        }
        
        # Append to JSONL file
        with open(alert_log_file, "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(alert_data, ensure_ascii=False)}\n")
        
        print(f"[FEEDBACK INTEGRATION] Alert logged for feedback analysis: {symbol}")
        return True
        
    except Exception as e:
        print(f"[FEEDBACK INTEGRATION] Error logging alert: {e}")
        return False


def run_periodic_feedback_analysis() -> bool:
    """
    Uruchamia okresową analizę feedback
    
    Returns:
        bool: True jeśli analiza się powiodła
    """
    try:
        import subprocess
        
        # Run feedback_loop_v2 analysis
        result = subprocess.run(
            ["python", "feedback/feedback_loop_v2.py", "analyze"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode == 0:
            print("[FEEDBACK INTEGRATION] Periodic feedback analysis completed successfully")
            print(result.stdout)
            return True
        else:
            print(f"[FEEDBACK INTEGRATION] Feedback analysis failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[FEEDBACK INTEGRATION] Error running feedback analysis: {e}")
        return False


def schedule_daily_feedback():
    """
    Planuje codzienną analizę feedback (do integracji z cron lub scheduler)
    """
    import schedule
    import time
    
    def daily_feedback_job():
        print("[FEEDBACK INTEGRATION] Running scheduled daily feedback analysis...")
        run_periodic_feedback_analysis()
    
    # Schedule daily at 2 AM
    schedule.every().day.at("02:00").do(daily_feedback_job)
    
    print("[FEEDBACK INTEGRATION] Daily feedback analysis scheduled for 02:00")
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour


def get_feedback_stats() -> Dict[str, Any]:
    """
    Pobiera statystyki feedback systemu
    
    Returns:
        Dict ze statystykami
    """
    try:
        alert_log_file = "logs/alerts_history.jsonl"
        feedback_history_file = "logs/feedback_v2_history.json"
        
        stats = {
            "total_alerts_logged": 0,
            "tjde_alerts": 0,
            "feedback_sessions": 0,
            "last_feedback_analysis": None,
            "current_success_rate": None
        }
        
        # Count alerts
        if os.path.exists(alert_log_file):
            with open(alert_log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        stats["total_alerts_logged"] += 1
                        try:
                            alert_data = json.loads(line)
                            if alert_data.get("alert_type") == "TJDE_adaptive":
                                stats["tjde_alerts"] += 1
                        except:
                            continue
        
        # Get feedback history
        if os.path.exists(feedback_history_file):
            with open(feedback_history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
                sessions = history.get("feedback_sessions", [])
                stats["feedback_sessions"] = len(sessions)
                
                if sessions:
                    latest = sessions[-1]
                    stats["last_feedback_analysis"] = latest.get("timestamp")
                    feedback_info = latest.get("feedback_info", {})
                    stats["current_success_rate"] = feedback_info.get("success_rate")
        
        return stats
        
    except Exception as e:
        print(f"[FEEDBACK INTEGRATION] Error getting stats: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "stats":
            stats = get_feedback_stats()
            print("📊 Feedback Integration Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        elif command == "run":
            success = run_periodic_feedback_analysis()
            if success:
                print("✅ Feedback analysis completed")
            else:
                print("❌ Feedback analysis failed")
                
        elif command == "schedule":
            print("🕒 Starting scheduled feedback system...")
            schedule_daily_feedback()
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: stats, run, schedule")
    else:
        print("Feedback Integration Commands:")
        print("  stats    - Show feedback system statistics")
        print("  run      - Run immediate feedback analysis")
        print("  schedule - Start scheduled daily feedback")
        print("\nExample: python utils/feedback_integration.py stats")