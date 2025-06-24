"""
Performance Monitor - Track and optimize slow operations
Identifies bottlenecks in real-time scanning
"""
import time
import json
import os
from typing import Dict, List
from datetime import datetime

class PerformanceMonitor:
    """Monitor and track performance of scanning operations"""
    
    def __init__(self):
        self.slow_operations = []
        self.performance_log = "logs/performance.json"
        os.makedirs("logs", exist_ok=True)
        
    def log_slow_operation(self, symbol: str, operation: str, duration: float, details: Dict = None):
        """Log slow operations for analysis"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "operation": operation,
            "duration": duration,
            "details": details or {}
        }
        
        self.slow_operations.append(entry)
        
        # Save to file periodically
        if len(self.slow_operations) % 10 == 0:
            self.save_performance_log()
            
    def save_performance_log(self):
        """Save performance log to file"""
        try:
            existing_data = []
            if os.path.exists(self.performance_log):
                with open(self.performance_log, "r") as f:
                    existing_data = json.load(f)
            
            existing_data.extend(self.slow_operations)
            
            with open(self.performance_log, "w") as f:
                json.dump(existing_data, f, indent=2)
                
            self.slow_operations = []  # Clear after saving
            
        except Exception as e:
            print(f"[PERF] Failed to save performance log: {e}")
    
    def analyze_bottlenecks(self) -> Dict:
        """Analyze performance data to identify bottlenecks"""
        if not os.path.exists(self.performance_log):
            return {}
            
        try:
            with open(self.performance_log, "r") as f:
                data = json.load(f)
            
            # Analyze by operation type
            operation_stats = {}
            for entry in data:
                op = entry["operation"]
                duration = entry["duration"]
                
                if op not in operation_stats:
                    operation_stats[op] = {
                        "count": 0,
                        "total_time": 0,
                        "max_time": 0,
                        "avg_time": 0
                    }
                
                operation_stats[op]["count"] += 1
                operation_stats[op]["total_time"] += duration
                operation_stats[op]["max_time"] = max(operation_stats[op]["max_time"], duration)
            
            # Calculate averages
            for op, stats in operation_stats.items():
                stats["avg_time"] = stats["total_time"] / stats["count"]
            
            return operation_stats
            
        except Exception as e:
            print(f"[PERF] Failed to analyze bottlenecks: {e}")
            return {}

# Global performance monitor
perf_monitor = PerformanceMonitor()

def time_operation(operation_name: str):
    """Decorator to time operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log slow operations (>1s)
                if duration > 1.0:
                    symbol = args[0] if args else "unknown"
                    perf_monitor.log_slow_operation(
                        symbol, 
                        operation_name, 
                        duration,
                        {"args_count": len(args), "kwargs_count": len(kwargs)}
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                symbol = args[0] if args else "unknown"
                perf_monitor.log_slow_operation(
                    symbol,
                    f"{operation_name}_ERROR",
                    duration,
                    {"error": str(e)}
                )
                raise
        return wrapper
    return decorator

def log_scan_performance(symbols_processed: int, total_duration: float, successful: int):
    """Log overall scan performance"""
    avg_per_symbol = total_duration / max(symbols_processed, 1)
    
    perf_monitor.log_slow_operation(
        "SCAN_SUMMARY",
        "full_scan",
        total_duration,
        {
            "symbols_processed": symbols_processed,
            "successful": successful,
            "avg_per_symbol": avg_per_symbol,
            "symbols_per_second": symbols_processed / max(total_duration, 0.1)
        }
    )

def get_performance_summary() -> str:
    """Get human-readable performance summary"""
    stats = perf_monitor.analyze_bottlenecks()
    if not stats:
        return "No performance data available"
    
    summary = "🚀 Performance Analysis:\n"
    for operation, data in sorted(stats.items(), key=lambda x: x[1]["avg_time"], reverse=True):
        summary += f"  {operation}: {data['avg_time']:.2f}s avg ({data['count']} calls, max: {data['max_time']:.2f}s)\n"
    
    return summary