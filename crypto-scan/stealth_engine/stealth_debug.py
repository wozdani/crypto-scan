#!/usr/bin/env python3
"""
Stealth Debug System - Zaawansowane debugowanie dla Stealth Engine
Szczegółowe logowanie, analiza błędów i monitoring wydajności
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

class StealthDebugLogger:
    """Zaawansowany logger dla Stealth Engine z kategoryzacją i poziomami"""
    
    def __init__(self, log_dir: str = "logs/stealth_debug"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Aktywne sesje debugowania
        self.active_sessions = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_analyses": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "slowest_analysis": {"symbol": None, "time": 0.0},
            "fastest_analysis": {"symbol": None, "time": float('inf')}
        }
    
    def start_debug_session(self, symbol: str) -> str:
        """Rozpoczyna sesję debugowania dla symbolu"""
        session_id = f"{symbol}_{int(time.time())}"
        timestamp = datetime.now()
        
        self.active_sessions[session_id] = {
            "symbol": symbol,
            "start_time": timestamp,
            "steps": [],
            "metrics": {},
            "errors": []
        }
        
        self.log_debug(symbol, "SESSION_START", f"Debug session {session_id} started", session_id)
        return session_id
    
    def log_debug_step(self, session_id: str, step_name: str, data: Any, execution_time: Optional[float] = None):
        """Loguje krok w sesji debugowania"""
        if session_id not in self.active_sessions:
            return
        
        timestamp = datetime.now()
        step_data = {
            "step": step_name,
            "timestamp": timestamp.isoformat(),
            "data": data,
            "execution_time": execution_time
        }
        
        self.active_sessions[session_id]["steps"].append(step_data)
        
        # Log do pliku
        symbol = self.active_sessions[session_id]["symbol"]
        self.log_debug(symbol, f"STEP_{step_name.upper()}", data, session_id)
    
    def log_debug_error(self, session_id: str, error_type: str, error_message: str, context: Dict = None):
        """Loguje błąd w sesji debugowania"""
        if session_id not in self.active_sessions:
            return
        
        error_data = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.active_sessions[session_id]["errors"].append(error_data)
        
        symbol = self.active_sessions[session_id]["symbol"]
        self.log_debug(symbol, "ERROR", f"{error_type}: {error_message}", session_id)
    
    def end_debug_session(self, session_id: str) -> str:
        """Kończy sesję debugowania i zapisuje pełny raport"""
        if session_id not in self.active_sessions:
            return ""
        
        session = self.active_sessions[session_id]
        end_time = datetime.now()
        total_time = (end_time - session["start_time"]).total_seconds()
        
        session["end_time"] = end_time
        session["total_time"] = total_time
        
        # Update performance metrics
        self.update_performance_metrics(session["symbol"], total_time)
        
        # Save session report
        report_path = self.save_session_report(session_id)
        
        # Cleanup
        del self.active_sessions[session_id]
        
        self.log_debug(session["symbol"], "SESSION_END", f"Session completed in {total_time:.3f}s, report: {report_path}")
        
        return report_path
    
    def log_debug(self, symbol: str, level: str, message: Any, session_id: str = None):
        """Główna funkcja logowania debug"""
        timestamp = datetime.now()
        
        # Format log entry
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "level": level,
            "message": str(message),
            "session_id": session_id
        }
        
        # Daily log file
        date_str = timestamp.strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"stealth_debug_{date_str}.log")
        
        # Append to daily log
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] {symbol} {level}: {message}\n")
        
        # Symbol-specific detailed log
        symbol_log_file = os.path.join(self.log_dir, f"{symbol}_detailed.json")
        
        # Load existing entries
        entries = []
        if os.path.exists(symbol_log_file):
            try:
                with open(symbol_log_file, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
            except:
                entries = []
        
        # Add new entry
        entries.append(log_entry)
        
        # Keep only last 100 entries per symbol
        if len(entries) > 100:
            entries = entries[-100:]
        
        # Save updated entries
        with open(symbol_log_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    
    def save_session_report(self, session_id: str) -> str:
        """Zapisuje szczegółowy raport sesji debugowania"""
        if session_id not in self.active_sessions:
            return ""
        
        session = self.active_sessions[session_id]
        
        # Generate comprehensive report
        report = {
            "session_id": session_id,
            "symbol": session["symbol"],
            "start_time": session["start_time"].isoformat(),
            "end_time": session["end_time"].isoformat(),
            "total_time": session["total_time"],
            "steps_count": len(session["steps"]),
            "errors_count": len(session["errors"]),
            "steps": session["steps"],
            "errors": session["errors"],
            "metrics": session.get("metrics", {}),
            "performance_summary": {
                "time_per_step": session["total_time"] / max(len(session["steps"]), 1),
                "success_rate": 1.0 - (len(session["errors"]) / max(len(session["steps"]), 1)),
                "classification": self.classify_session_performance(session)
            }
        }
        
        # Save report
        report_file = os.path.join(self.log_dir, f"session_report_{session_id}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_file
    
    def update_performance_metrics(self, symbol: str, execution_time: float):
        """Aktualizuje metryki wydajności"""
        self.performance_metrics["total_analyses"] += 1
        self.performance_metrics["total_time"] += execution_time
        self.performance_metrics["average_time"] = self.performance_metrics["total_time"] / self.performance_metrics["total_analyses"]
        
        # Update extremes
        if execution_time > self.performance_metrics["slowest_analysis"]["time"]:
            self.performance_metrics["slowest_analysis"] = {"symbol": symbol, "time": execution_time}
        
        if execution_time < self.performance_metrics["fastest_analysis"]["time"]:
            self.performance_metrics["fastest_analysis"] = {"symbol": symbol, "time": execution_time}
    
    def classify_session_performance(self, session: Dict) -> str:
        """Klasyfikuje wydajność sesji"""
        total_time = session["total_time"]
        errors_count = len(session["errors"])
        steps_count = len(session["steps"])
        
        if errors_count > 0:
            return "error"
        elif total_time > 2.0:
            return "slow"
        elif total_time < 0.1:
            return "fast"
        elif steps_count < 5:
            return "incomplete"
        else:
            return "normal"
    
    def get_performance_report(self) -> Dict:
        """Zwraca raport wydajności"""
        return {
            "performance_metrics": self.performance_metrics,
            "active_sessions": len(self.active_sessions),
            "log_directory": self.log_dir
        }

# Global debug logger instance
debug_logger = StealthDebugLogger()

@contextmanager
def stealth_debug_session(symbol: str):
    """Context manager dla sesji debugowania"""
    session_id = debug_logger.start_debug_session(symbol)
    start_time = time.time()
    
    try:
        yield session_id
    except Exception as e:
        debug_logger.log_debug_error(session_id, "CONTEXT_ERROR", str(e))
        raise
    finally:
        execution_time = time.time() - start_time
        debug_logger.log_debug_step(session_id, "TOTAL_EXECUTION", {"execution_time": execution_time}, execution_time)
        debug_logger.end_debug_session(session_id)

def log_stealth_debug(symbol: str, stealth_score: float, active_signals: List[str], 
                     signal_details: Dict = None, weights: Dict = None, 
                     processing_time: float = None):
    """
    Szczegółowe logowanie wyników Stealth Engine
    
    Args:
        symbol: Symbol tokena
        stealth_score: Końcowy score
        active_signals: Lista aktywnych sygnałów
        signal_details: Szczegóły sygnałów
        weights: Użyte wagi
        processing_time: Czas przetwarzania
    """
    
    timestamp = datetime.now()
    
    # Przygotuj szczegółowe dane
    debug_data = {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "stealth_score": stealth_score,
        "active_signals": active_signals,
        "signals_count": len(active_signals),
        "signal_details": signal_details or {},
        "weights_used": weights or {},
        "processing_time": processing_time,
        "score_breakdown": {
            "score_per_signal": stealth_score / max(len(active_signals), 1),
            "score_category": "high" if stealth_score >= 4.0 else "medium" if stealth_score >= 2.5 else "low"
        }
    }
    
    # Log standardowy
    debug_logger.log_debug(symbol, "ANALYSIS", debug_data)
    
    # Osobny plik dla tego symbolu z timestamp
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    debug_file = os.path.join(debug_logger.log_dir, f"{symbol}_{timestamp_str}_stealth_debug.json")
    
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)
    
    print(f"[STEALTH DEBUG] {symbol} → Debug data saved to {debug_file}")

def log_signal_analysis(symbol: str, signal_name: str, signal_strength: float, 
                       signal_data: Dict, session_id: str = None):
    """Loguje analizę pojedynczego sygnału"""
    
    analysis_data = {
        "signal_name": signal_name,
        "signal_strength": signal_strength,
        "signal_data": signal_data,
        "timestamp": datetime.now().isoformat()
    }
    
    if session_id:
        debug_logger.log_debug_step(session_id, f"SIGNAL_{signal_name.upper()}", analysis_data)
    else:
        debug_logger.log_debug(symbol, f"SIGNAL_{signal_name.upper()}", analysis_data)

def log_weight_application(symbol: str, weights_applied: Dict, session_id: str = None):
    """Loguje zastosowanie wag"""
    
    weight_data = {
        "weights_applied": weights_applied,
        "total_weights": len(weights_applied),
        "average_weight": sum(weights_applied.values()) / max(len(weights_applied), 1),
        "timestamp": datetime.now().isoformat()
    }
    
    if session_id:
        debug_logger.log_debug_step(session_id, "WEIGHT_APPLICATION", weight_data)
    else:
        debug_logger.log_debug(symbol, "WEIGHT_APPLICATION", weight_data)

def get_debug_statistics(hours: int = 24) -> Dict:
    """Zwraca statystyki debugowania z ostatnich godzin"""
    
    # Scan log files from last N hours
    cutoff_time = datetime.now().timestamp() - (hours * 3600)
    
    stats = {
        "total_analyses": 0,
        "symbols_analyzed": set(),
        "error_count": 0,
        "average_score": 0.0,
        "score_distribution": {"high": 0, "medium": 0, "low": 0},
        "most_active_signals": {},
        "performance": debug_logger.get_performance_report()
    }
    
    # Process recent log files
    for filename in os.listdir(debug_logger.log_dir):
        if filename.endswith("_stealth_debug.json"):
            filepath = os.path.join(debug_logger.log_dir, filename)
            
            try:
                file_mtime = os.path.getmtime(filepath)
                if file_mtime < cutoff_time:
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    debug_data = json.load(f)
                
                # Update statistics
                stats["total_analyses"] += 1
                stats["symbols_analyzed"].add(debug_data["symbol"])
                
                score = debug_data.get("stealth_score", 0)
                stats["average_score"] += score
                
                # Score distribution
                if score >= 4.0:
                    stats["score_distribution"]["high"] += 1
                elif score >= 2.5:
                    stats["score_distribution"]["medium"] += 1
                else:
                    stats["score_distribution"]["low"] += 1
                
                # Active signals
                for signal in debug_data.get("active_signals", []):
                    stats["most_active_signals"][signal] = stats["most_active_signals"].get(signal, 0) + 1
                
            except Exception as e:
                stats["error_count"] += 1
    
    # Finalize calculations
    if stats["total_analyses"] > 0:
        stats["average_score"] /= stats["total_analyses"]
    
    stats["symbols_analyzed"] = len(stats["symbols_analyzed"])
    
    # Sort signals by frequency
    stats["most_active_signals"] = dict(sorted(stats["most_active_signals"].items(), 
                                             key=lambda x: x[1], reverse=True)[:10])
    
    return stats

def cleanup_old_debug_logs(days: int = 7):
    """Czyści stare logi debug starsze niż N dni"""
    
    cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
    cleaned_count = 0
    
    for filename in os.listdir(debug_logger.log_dir):
        filepath = os.path.join(debug_logger.log_dir, filename)
        
        try:
            file_mtime = os.path.getmtime(filepath)
            if file_mtime < cutoff_time:
                os.remove(filepath)
                cleaned_count += 1
        except Exception as e:
            print(f"[DEBUG CLEANUP] Error removing {filepath}: {e}")
    
    print(f"[DEBUG CLEANUP] Cleaned {cleaned_count} old debug files")
    return cleaned_count

def test_stealth_debug():
    """Test funkcji stealth debug"""
    
    print("🧪 Testing Stealth Debug System...")
    
    # Test basic logging
    log_stealth_debug(
        symbol="DEBUGTEST",
        stealth_score=3.7,
        active_signals=["dex_inflow", "whale_activity_tracking"],
        signal_details={"dex_inflow": 0.8, "whale_activity_tracking": 0.6},
        weights={"dex_inflow": 0.22, "whale_activity_tracking": 0.18},
        processing_time=0.15
    )
    
    # Test session management
    with stealth_debug_session("SESSIONTEST") as session_id:
        debug_logger.log_debug_step(session_id, "signal_analysis", {"signals_detected": 3})
        debug_logger.log_debug_step(session_id, "score_calculation", {"final_score": 4.2})
        log_signal_analysis("SESSIONTEST", "test_signal", 0.75, {"test": "data"}, session_id)
    
    # Test statistics
    stats = get_debug_statistics(1)
    print(f"   📊 Debug stats: {stats['total_analyses']} analyses, {stats['symbols_analyzed']} symbols")
    
    # Test performance report
    perf_report = debug_logger.get_performance_report()
    print(f"   ⏱️ Performance: {perf_report['performance_metrics']['total_analyses']} total analyses")
    
    print("   ✅ Stealth Debug System operational")

if __name__ == "__main__":
    test_stealth_debug()