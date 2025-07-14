#!/usr/bin/env python3
"""
Feedback Loop Logger - System śledzenia skuteczności alertów
Zapisuje każdy alert i jego parametry do późniejszej analizy skuteczności
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def log_alert_feedback(token: str,
                       gnn_score: float,
                       whale_clip_conf: float,
                       dex_inflow_flag: bool,
                       decision: str,
                       final_score: float,
                       alert_time: str,
                       pump_occurred: Optional[bool] = None,
                       additional_data: Optional[Dict[str, Any]] = None):
    """
    Zapisuje parametry alertu + wynik po czasie (jeśli znany).
    
    Args:
        token: Symbol tokena
        gnn_score: Score z GNN anomaly detection
        whale_clip_conf: Confidence z WhaleCLIP
        dex_inflow_flag: Czy wykryto DEX inflow
        decision: Decyzja strategiczna (STRONG_SIGNAL, etc.)
        final_score: Końcowy score decyzyjny
        alert_time: Timestamp alertu
        pump_occurred: Czy nastąpił pump (None = nieznane)
        additional_data: Dodatkowe dane kontekstowe
    """
    try:
        # Ensure feedback logs directory exists
        os.makedirs("feedback_logs", exist_ok=True)
        
        # Create token-specific log file
        safe_token = token.replace("/", "_").replace("\\", "_")
        path = f"feedback_logs/{safe_token}_alerts.jsonl"
        
        # Prepare feedback data
        data = {
            "timestamp": alert_time,
            "token": token,
            "gnn_score": round(float(gnn_score), 4),
            "whale_clip_confidence": round(float(whale_clip_conf), 4),
            "dex_inflow": bool(dex_inflow_flag),
            "decision": str(decision),
            "final_score": round(float(final_score), 4),
            "pump_occurred": pump_occurred,  # None jeśli nie wiadomo jeszcze
            "evaluation_pending": pump_occurred is None,
            "additional_data": additional_data or {}
        }
        
        # Write to JSONL file
        with open(path, "a", encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        logger.info(f"[FEEDBACK LOG] Saved alert feedback for {token}: {decision} (score: {final_score:.3f})")
        
        # Also log to master feedback file
        master_path = "feedback_logs/master_alerts.jsonl"
        with open(master_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
        return True
        
    except Exception as e:
        logger.error(f"[FEEDBACK LOG] Error saving feedback for {token}: {e}")
        return False

def update_alert_outcome(token: str, alert_time: str, pump_occurred: bool, 
                        price_change_1h: Optional[float] = None,
                        price_change_3h: Optional[float] = None,
                        price_change_24h: Optional[float] = None):
    """
    Aktualizuje istniejący wpis alertu z informacją o wystąpieniu pump'a.
    
    Args:
        token: Symbol tokena
        alert_time: Timestamp oryginalnego alertu
        pump_occurred: Czy nastąpił pump
        price_change_1h: Zmiana ceny po 1h (opcjonalnie)
        price_change_3h: Zmiana ceny po 3h (opcjonalnie)
        price_change_24h: Zmiana ceny po 24h (opcjonalnie)
    """
    try:
        safe_token = token.replace("/", "_").replace("\\", "_")
        path = f"feedback_logs/{safe_token}_alerts.jsonl"
        
        if not os.path.exists(path):
            logger.warning(f"[FEEDBACK UPDATE] No feedback file found for {token}")
            return False
        
        # Read all entries
        updated_lines = []
        found_alert = False
        
        with open(path, "r", encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    
                    # Check if this is the alert we want to update
                    if data.get("timestamp") == alert_time and data.get("evaluation_pending"):
                        # Update with outcome
                        data["pump_occurred"] = pump_occurred
                        data["evaluation_pending"] = False
                        data["evaluation_timestamp"] = datetime.utcnow().isoformat()
                        
                        # Add price changes if provided
                        if price_change_1h is not None:
                            data["price_change_1h"] = round(float(price_change_1h), 4)
                        if price_change_3h is not None:
                            data["price_change_3h"] = round(float(price_change_3h), 4)
                        if price_change_24h is not None:
                            data["price_change_24h"] = round(float(price_change_24h), 4)
                        
                        found_alert = True
                        logger.info(f"[FEEDBACK UPDATE] Updated {token} alert outcome: pump={pump_occurred}")
                    
                    updated_lines.append(json.dumps(data, ensure_ascii=False) + "\n")
        
        if found_alert:
            # Write back updated data
            with open(path, "w", encoding='utf-8') as f:
                f.writelines(updated_lines)
            
            # Also update master file
            update_master_file_outcome(token, alert_time, pump_occurred, 
                                     price_change_1h, price_change_3h, price_change_24h)
            return True
        else:
            logger.warning(f"[FEEDBACK UPDATE] Alert not found for {token} at {alert_time}")
            return False
            
    except Exception as e:
        logger.error(f"[FEEDBACK UPDATE] Error updating feedback for {token}: {e}")
        return False

def update_master_file_outcome(token: str, alert_time: str, pump_occurred: bool,
                              price_change_1h: Optional[float] = None,
                              price_change_3h: Optional[float] = None,
                              price_change_24h: Optional[float] = None):
    """Aktualizuje master feedback file z wynikiem alertu."""
    try:
        master_path = "feedback_logs/master_alerts.jsonl"
        
        if not os.path.exists(master_path):
            return False
        
        updated_lines = []
        found_alert = False
        
        with open(master_path, "r", encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    
                    if (data.get("token") == token and 
                        data.get("timestamp") == alert_time and 
                        data.get("evaluation_pending")):
                        
                        data["pump_occurred"] = pump_occurred
                        data["evaluation_pending"] = False
                        data["evaluation_timestamp"] = datetime.utcnow().isoformat()
                        
                        if price_change_1h is not None:
                            data["price_change_1h"] = round(float(price_change_1h), 4)
                        if price_change_3h is not None:
                            data["price_change_3h"] = round(float(price_change_3h), 4)
                        if price_change_24h is not None:
                            data["price_change_24h"] = round(float(price_change_24h), 4)
                        
                        found_alert = True
                    
                    updated_lines.append(json.dumps(data, ensure_ascii=False) + "\n")
        
        if found_alert:
            with open(master_path, "w", encoding='utf-8') as f:
                f.writelines(updated_lines)
        
        return found_alert
        
    except Exception as e:
        logger.error(f"[FEEDBACK MASTER] Error updating master file: {e}")
        return False

def get_feedback_statistics(token: Optional[str] = None) -> Dict[str, Any]:
    """
    Pobiera statystyki feedback dla tokena lub wszystkich.
    
    Args:
        token: Symbol tokena (None = wszystkie)
        
    Returns:
        Dictionary ze statystykami feedback
    """
    try:
        stats = {
            "total_alerts": 0,
            "evaluated_alerts": 0,
            "successful_pumps": 0,
            "failed_pumps": 0,
            "pending_evaluation": 0,
            "success_rate": 0.0,
            "decision_breakdown": {},
            "score_distribution": []
        }
        
        if token:
            # Token-specific stats
            safe_token = token.replace("/", "_").replace("\\", "_")
            path = f"feedback_logs/{safe_token}_alerts.jsonl"
            
            if not os.path.exists(path):
                return stats
            
            with open(path, "r", encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        stats["total_alerts"] += 1
                        
                        decision = data.get("decision", "UNKNOWN")
                        stats["decision_breakdown"][decision] = stats["decision_breakdown"].get(decision, 0) + 1
                        stats["score_distribution"].append(data.get("final_score", 0.0))
                        
                        if data.get("evaluation_pending", True):
                            stats["pending_evaluation"] += 1
                        else:
                            stats["evaluated_alerts"] += 1
                            if data.get("pump_occurred", False):
                                stats["successful_pumps"] += 1
                            else:
                                stats["failed_pumps"] += 1
        else:
            # All tokens stats from master file
            master_path = "feedback_logs/master_alerts.jsonl"
            
            if os.path.exists(master_path):
                with open(master_path, "r", encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            stats["total_alerts"] += 1
                            
                            decision = data.get("decision", "UNKNOWN")
                            stats["decision_breakdown"][decision] = stats["decision_breakdown"].get(decision, 0) + 1
                            stats["score_distribution"].append(data.get("final_score", 0.0))
                            
                            if data.get("evaluation_pending", True):
                                stats["pending_evaluation"] += 1
                            else:
                                stats["evaluated_alerts"] += 1
                                if data.get("pump_occurred", False):
                                    stats["successful_pumps"] += 1
                                else:
                                    stats["failed_pumps"] += 1
        
        # Calculate success rate
        if stats["evaluated_alerts"] > 0:
            stats["success_rate"] = round(stats["successful_pumps"] / stats["evaluated_alerts"], 4)
        
        return stats
        
    except Exception as e:
        logger.error(f"[FEEDBACK STATS] Error getting feedback statistics: {e}")
        return stats

def get_pending_evaluations() -> list:
    """Pobiera listę alertów oczekujących na ewaluację."""
    try:
        pending = []
        master_path = "feedback_logs/master_alerts.jsonl"
        
        if os.path.exists(master_path):
            with open(master_path, "r", encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        if data.get("evaluation_pending", True):
                            pending.append(data)
        
        return pending
        
    except Exception as e:
        logger.error(f"[FEEDBACK PENDING] Error getting pending evaluations: {e}")
        return []

def test_feedback_loop():
    """Test feedback loop functionality"""
    print("🧠 FEEDBACK LOOP LOGGER TEST")
    print("=" * 50)
    
    # Test logging
    test_time = datetime.utcnow().isoformat()
    
    result1 = log_alert_feedback(
        token="TESTUSDT",
        gnn_score=0.85,
        whale_clip_conf=0.92,
        dex_inflow_flag=True,
        decision="STRONG_SIGNAL",
        final_score=1.0,
        alert_time=test_time,
        additional_data={"test": True}
    )
    
    print(f"✅ Alert logging: {'SUCCESS' if result1 else 'FAILED'}")
    
    # Test statistics
    stats = get_feedback_statistics("TESTUSDT")
    print(f"✅ Statistics: {stats['total_alerts']} alerts found")
    
    # Test update
    result2 = update_alert_outcome("TESTUSDT", test_time, True, price_change_1h=5.2)
    print(f"✅ Outcome update: {'SUCCESS' if result2 else 'FAILED'}")
    
    # Test pending
    pending = get_pending_evaluations()
    print(f"✅ Pending evaluations: {len(pending)} found")
    
    print("\n🎯 FEEDBACK LOOP READY FOR INTEGRATION!")

if __name__ == "__main__":
    test_feedback_loop()