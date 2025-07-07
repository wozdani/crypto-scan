#!/usr/bin/env python3
"""
Stealth Utils - Zestaw narzędzi pomocniczych dla Stealth Engine
Metadata management, cache operations, export utilities i monitoring
"""

import os
import json
import time
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

class StealthMetadataManager:
    """Manager metadanych dla alertów Stealth Engine"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.metadata_file = os.path.join(cache_dir, "stealth_metadata.json")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing metadata
        self.metadata = self.load_metadata()
    
    def load_metadata(self) -> Dict:
        """Ładuje istniejące metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[METADATA ERROR] Failed to load metadata: {e}")
                return self.create_default_metadata()
        else:
            return self.create_default_metadata()
    
    def create_default_metadata(self) -> Dict:
        """Tworzy domyślną strukturę metadata"""
        return {
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_alerts": 0,
            "alert_counters": {
                "strong_stealth_alert": 0,
                "medium_alert": 0,
                "weak_alert": 0
            },
            "symbol_statistics": {},
            "signal_statistics": {},
            "performance_metrics": {
                "total_processing_time": 0.0,
                "average_processing_time": 0.0,
                "fastest_analysis": {"symbol": None, "time": float('inf')},
                "slowest_analysis": {"symbol": None, "time": 0.0}
            },
            "daily_summaries": {},
            "export_history": []
        }
    
    def save_metadata(self):
        """Zapisuje metadata do pliku"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        try:
            # Create backup
            if os.path.exists(self.metadata_file):
                backup_file = f"{self.metadata_file}.backup"
                shutil.copy2(self.metadata_file, backup_file)
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[METADATA ERROR] Failed to save metadata: {e}")
    
    def record_alert(self, symbol: str, stealth_score: float, active_signals: List[str], 
                    alert_type: str, processing_time: float = None):
        """Rejestruje nowy alert w metadata"""
        
        # Update counters
        self.metadata["total_alerts"] += 1
        self.metadata["alert_counters"][alert_type] = self.metadata["alert_counters"].get(alert_type, 0) + 1
        
        # Update symbol statistics
        if symbol not in self.metadata["symbol_statistics"]:
            self.metadata["symbol_statistics"][symbol] = {
                "total_alerts": 0,
                "scores": [],
                "alert_types": {},
                "first_alert": datetime.now().isoformat(),
                "last_alert": None
            }
        
        symbol_stats = self.metadata["symbol_statistics"][symbol]
        symbol_stats["total_alerts"] += 1
        symbol_stats["scores"].append(stealth_score)
        symbol_stats["alert_types"][alert_type] = symbol_stats["alert_types"].get(alert_type, 0) + 1
        symbol_stats["last_alert"] = datetime.now().isoformat()
        
        # Update signal statistics
        for signal in active_signals:
            if signal not in self.metadata["signal_statistics"]:
                self.metadata["signal_statistics"][signal] = {
                    "total_occurrences": 0,
                    "symbols": set(),
                    "average_contribution": 0.0,
                    "last_seen": None
                }
            
            signal_stats = self.metadata["signal_statistics"][signal]
            signal_stats["total_occurrences"] += 1
            
            # Handle both set and list for symbols
            if isinstance(signal_stats["symbols"], set):
                signal_stats["symbols"].add(symbol)
                signal_stats["symbols"] = list(signal_stats["symbols"])  # Convert to list
            elif isinstance(signal_stats["symbols"], list):
                if symbol not in signal_stats["symbols"]:
                    signal_stats["symbols"].append(symbol)
            
            signal_stats["last_seen"] = datetime.now().isoformat()
        
        # Update performance metrics
        if processing_time:
            perf = self.metadata["performance_metrics"]
            perf["total_processing_time"] += processing_time
            
            if self.metadata["total_alerts"] > 0:
                perf["average_processing_time"] = perf["total_processing_time"] / self.metadata["total_alerts"]
            
            if processing_time < perf["fastest_analysis"]["time"]:
                perf["fastest_analysis"] = {"symbol": symbol, "time": processing_time}
            
            if processing_time > perf["slowest_analysis"]["time"]:
                perf["slowest_analysis"] = {"symbol": symbol, "time": processing_time}
        
        # Update daily summary
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.metadata["daily_summaries"]:
            self.metadata["daily_summaries"][today] = {
                "alerts_count": 0,
                "symbols": set(),
                "alert_types": {},
                "top_signals": {}
            }
        
        daily = self.metadata["daily_summaries"][today]
        daily["alerts_count"] += 1
        
        # Handle both set and list for daily symbols
        if isinstance(daily["symbols"], set):
            daily["symbols"].add(symbol)
            daily["symbols"] = list(daily["symbols"])  # Convert to list
        elif isinstance(daily["symbols"], list):
            if symbol not in daily["symbols"]:
                daily["symbols"].append(symbol)
        
        daily["alert_types"][alert_type] = daily["alert_types"].get(alert_type, 0) + 1
        
        for signal in active_signals:
            daily["top_signals"][signal] = daily["top_signals"].get(signal, 0) + 1
        
        # Save updated metadata
        self.save_metadata()
    
    def get_symbol_history(self, symbol: str) -> Dict:
        """Pobiera historię alertów dla symbolu"""
        return self.metadata["symbol_statistics"].get(symbol, {})
    
    def get_top_symbols(self, limit: int = 10) -> List[Dict]:
        """Zwraca top symbole z największą liczbą alertów"""
        symbols = []
        
        for symbol, stats in self.metadata["symbol_statistics"].items():
            symbols.append({
                "symbol": symbol,
                "total_alerts": stats["total_alerts"],
                "average_score": sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0,
                "last_alert": stats["last_alert"]
            })
        
        return sorted(symbols, key=lambda x: x["total_alerts"], reverse=True)[:limit]
    
    def get_top_signals(self, limit: int = 10) -> List[Dict]:
        """Zwraca najczęściej występujące sygnały"""
        signals = []
        
        for signal, stats in self.metadata["signal_statistics"].items():
            signals.append({
                "signal": signal,
                "total_occurrences": stats["total_occurrences"],
                "unique_symbols": len(stats["symbols"]) if isinstance(stats["symbols"], list) else len(stats["symbols"]),
                "last_seen": stats["last_seen"]
            })
        
        return sorted(signals, key=lambda x: x["total_occurrences"], reverse=True)[:limit]
    
    def get_daily_summary(self, date: str = None) -> Dict:
        """Pobiera podsumowanie dzienne"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        return self.metadata["daily_summaries"].get(date, {})
    
    def export_metadata(self, format: str = "json") -> str:
        """Eksportuje metadata do pliku"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            export_file = f"exports/stealth_metadata_export_{timestamp}.json"
            os.makedirs("exports", exist_ok=True)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            import csv
            export_file = f"exports/stealth_metadata_export_{timestamp}.csv"
            os.makedirs("exports", exist_ok=True)
            
            # Export symbol statistics as CSV
            with open(export_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Symbol", "Total Alerts", "Average Score", "Last Alert"])
                
                for symbol, stats in self.metadata["symbol_statistics"].items():
                    avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
                    writer.writerow([symbol, stats["total_alerts"], f"{avg_score:.3f}", stats["last_alert"]])
        
        # Record export
        self.metadata["export_history"].append({
            "timestamp": datetime.now().isoformat(),
            "format": format,
            "file": export_file
        })
        
        self.save_metadata()
        print(f"[METADATA EXPORT] Exported to {export_file}")
        
        return export_file

class StealthCacheManager:
    """Manager cache'u dla Stealth Engine"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def cache_analysis_result(self, symbol: str, result: Dict, ttl_hours: int = 1):
        """Cache'uje wynik analizy z TTL"""
        cache_file = os.path.join(self.cache_dir, f"stealth_cache_{symbol}.json")
        
        cache_data = {
            "symbol": symbol,
            "result": result,
            "cached_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
            "ttl_hours": ttl_hours
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    def get_cached_result(self, symbol: str) -> Optional[Dict]:
        """Pobiera cache'owany wynik jeśli jest aktualny"""
        cache_file = os.path.join(self.cache_dir, f"stealth_cache_{symbol}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if expired
            expires_at = datetime.fromisoformat(cache_data["expires_at"])
            if datetime.now() > expires_at:
                os.remove(cache_file)  # Clean up expired cache
                return None
            
            return cache_data["result"]
            
        except Exception as e:
            print(f"[CACHE ERROR] Failed to load cache for {symbol}: {e}")
            return None
    
    def cleanup_expired_cache(self):
        """Czyści wygasłe cache'e"""
        cleaned_count = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.startswith("stealth_cache_") and filename.endswith(".json"):
                cache_file = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    expires_at = datetime.fromisoformat(cache_data["expires_at"])
                    if datetime.now() > expires_at:
                        os.remove(cache_file)
                        cleaned_count += 1
                        
                except Exception as e:
                    print(f"[CACHE CLEANUP] Error processing {cache_file}: {e}")
        
        print(f"[CACHE CLEANUP] Cleaned {cleaned_count} expired cache files")
        return cleaned_count

def compress_old_logs(days_old: int = 7, compression_ratio: float = 0.1):
    """Kompresuje stare logi dla oszczędności miejsca"""
    
    cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
    compressed_count = 0
    
    log_dirs = ["logs", "logs/stealth_debug", "data"]
    
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            continue
        
        for filename in os.listdir(log_dir):
            if not (filename.endswith('.json') or filename.endswith('.log') or filename.endswith('.jsonl')):
                continue
            
            filepath = os.path.join(log_dir, filename)
            
            try:
                file_mtime = os.path.getmtime(filepath)
                if file_mtime < cutoff_time:
                    # Compress file
                    compressed_file = f"{filepath}.gz"
                    
                    with open(filepath, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Verify compression and remove original
                    if os.path.exists(compressed_file):
                        original_size = os.path.getsize(filepath)
                        compressed_size = os.path.getsize(compressed_file)
                        
                        if compressed_size < original_size * compression_ratio:
                            os.remove(filepath)
                            compressed_count += 1
                            print(f"[COMPRESSION] {filename}: {original_size} → {compressed_size} bytes")
                        else:
                            os.remove(compressed_file)  # Remove if compression not effective
                            
            except Exception as e:
                print(f"[COMPRESSION ERROR] Failed to compress {filepath}: {e}")
    
    print(f"[COMPRESSION] Compressed {compressed_count} old log files")
    return compressed_count

def generate_stealth_report(hours: int = 24, export_format: str = "json") -> str:
    """Generuje kompleksowy raport działania Stealth Engine"""
    
    # Initialize managers
    metadata_mgr = StealthMetadataManager()
    
    # Collect data
    report_data = {
        "report_generated": datetime.now().isoformat(),
        "time_period_hours": hours,
        "metadata_summary": {
            "total_alerts": metadata_mgr.metadata.get("total_alerts", 0),
            "alert_distribution": metadata_mgr.metadata.get("alert_counters", {}),
            "performance_metrics": metadata_mgr.metadata.get("performance_metrics", {})
        },
        "top_symbols": metadata_mgr.get_top_symbols(15),
        "top_signals": metadata_mgr.get_top_signals(15),
        "daily_activity": {}
    }
    
    # Add recent daily summaries
    for i in range(min(hours // 24 + 1, 7)):  # Last week max
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_summary = metadata_mgr.get_daily_summary(date)
        if daily_summary:
            report_data["daily_activity"][date] = daily_summary
    
    # Load recent alerts for analysis
    from stealth_engine.stealth_labels import load_stealth_labels
    recent_labels = load_stealth_labels(hours=hours)
    
    if recent_labels:
        # Analyze label patterns
        label_distribution = {}
        confidence_scores = []
        
        for label_data in recent_labels:
            label = label_data.get("stealth_label", "unknown")
            label_distribution[label] = label_distribution.get(label, 0) + 1
            confidence_scores.append(label_data.get("label_confidence", 0))
        
        report_data["label_analysis"] = {
            "total_labeled_alerts": len(recent_labels),
            "label_distribution": dict(sorted(label_distribution.items(), key=lambda x: x[1], reverse=True)),
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "confidence_histogram": {
                "high_confidence": sum(1 for c in confidence_scores if c >= 0.8),
                "medium_confidence": sum(1 for c in confidence_scores if 0.5 <= c < 0.8),
                "low_confidence": sum(1 for c in confidence_scores if c < 0.5)
            }
        }
    
    # Export report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if export_format == "json":
        report_file = f"exports/stealth_comprehensive_report_{timestamp}.json"
        os.makedirs("exports", exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"[STEALTH REPORT] Comprehensive report generated: {report_file}")
    return report_file

def monitor_stealth_health() -> Dict:
    """Monitoruje stan zdrowia Stealth Engine"""
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "components": {},
        "alerts": [],
        "recommendations": []
    }
    
    # Check metadata integrity
    try:
        metadata_mgr = StealthMetadataManager()
        health_status["components"]["metadata"] = "healthy"
        
        # Check if alerts are being generated
        recent_alerts = metadata_mgr.metadata.get("total_alerts", 0)
        if recent_alerts == 0:
            health_status["alerts"].append("No alerts generated yet")
            health_status["recommendations"].append("Ensure Stealth Engine is integrated with scanner")
        
    except Exception as e:
        health_status["components"]["metadata"] = f"error: {e}"
        health_status["overall_status"] = "warning"
    
    # Check cache health
    try:
        cache_mgr = StealthCacheManager()
        cache_files = len([f for f in os.listdir(cache_mgr.cache_dir) if f.startswith("stealth_cache_")])
        health_status["components"]["cache"] = f"healthy ({cache_files} cached items)"
        
    except Exception as e:
        health_status["components"]["cache"] = f"error: {e}"
        health_status["overall_status"] = "warning"
    
    # Check log files
    try:
        log_dirs = ["logs", "logs/stealth_debug", "data"]
        total_log_files = 0
        
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                total_log_files += len([f for f in os.listdir(log_dir) if f.endswith(('.log', '.json', '.jsonl'))])
        
        health_status["components"]["logging"] = f"healthy ({total_log_files} log files)"
        
    except Exception as e:
        health_status["components"]["logging"] = f"error: {e}"
        health_status["overall_status"] = "warning"
    
    # Check storage space
    try:
        total_size = 0
        for root, dirs, files in os.walk("."):
            for file in files:
                if any(keyword in root for keyword in ["stealth", "cache", "logs", "data"]):
                    total_size += os.path.getsize(os.path.join(root, file))
        
        size_mb = total_size / (1024 * 1024)
        health_status["components"]["storage"] = f"healthy ({size_mb:.1f} MB used)"
        
        if size_mb > 500:  # 500 MB threshold
            health_status["alerts"].append(f"High storage usage: {size_mb:.1f} MB")
            health_status["recommendations"].append("Consider running cleanup utilities")
        
    except Exception as e:
        health_status["components"]["storage"] = f"error: {e}"
    
    # Set overall status based on components
    if any("error" in status for status in health_status["components"].values()):
        health_status["overall_status"] = "error"
    elif health_status["alerts"]:
        health_status["overall_status"] = "warning"
    
    return health_status

def test_stealth_utils():
    """Test wszystkich funkcji stealth utils"""
    
    print("🧪 Testing Stealth Utils System...")
    
    # Test metadata manager
    metadata_mgr = StealthMetadataManager()
    metadata_mgr.record_alert("UTILTEST", 4.1, ["dex_inflow", "whale_activity_tracking"], "strong_stealth_alert", 0.12)
    
    top_symbols = metadata_mgr.get_top_symbols(5)
    print(f"   📊 Top symbols: {len(top_symbols)} entries")
    
    # Test cache manager
    cache_mgr = StealthCacheManager()
    test_result = {"score": 3.5, "signals": ["test"]}
    cache_mgr.cache_analysis_result("CACHETEST", test_result, 1)
    
    cached = cache_mgr.get_cached_result("CACHETEST")
    if cached:
        print(f"   💾 Cache working: {cached['score']}")
    
    # Test health monitoring
    health = monitor_stealth_health()
    print(f"   🏥 Health status: {health['overall_status']}")
    
    # Test report generation
    report_file = generate_stealth_report(1, "json")
    if os.path.exists(report_file):
        print(f"   📄 Report generated: {os.path.basename(report_file)}")
    
    print("   ✅ Stealth Utils System operational")

# Global instances
metadata_manager = StealthMetadataManager()
cache_manager = StealthCacheManager()

if __name__ == "__main__":
    test_stealth_utils()