"""
Global Scan Error Reporting System for Trend-Mode Scanner
Collects all errors during scan cycles and provides comprehensive summaries
"""

import time
from datetime import datetime
from typing import List, Dict, Any
from threading import Lock

# Global error collection system
scan_errors: List[str] = []
error_lock = Lock()
scan_start_time = None
scan_session_id = None

def initialize_scan_session():
    """Initialize a new scan session with error tracking"""
    global scan_errors, scan_start_time, scan_session_id
    
    with error_lock:
        scan_errors = []
        scan_start_time = time.time()
        scan_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    print(f"🔄 [SCAN SESSION] Initialized error tracking: {scan_session_id}")

def log_scan_error(message: str, category: str = "GENERAL"):
    """
    Log an error to the global error collection
    
    Args:
        message: Error message to log
        category: Error category (TOKEN, GLOBAL, API, CLIP, CHART, etc.)
    """
    global scan_errors
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_error = f"[{timestamp}] [{category}] {message}"
    
    with error_lock:
        scan_errors.append(formatted_error)
    
    # Also print to console for immediate visibility
    print(f"❌ {formatted_error}")

def log_token_error(symbol: str, error_type: str, details: str):
    """Log token-specific error with symbol context"""
    log_scan_error(f"{symbol} → {error_type}: {details}", "TOKEN")

def log_api_error(symbol: str, api_name: str, details: str):
    """Log API-related errors"""
    log_scan_error(f"{symbol} → {api_name} API failed: {details}", "API")

def log_clip_error(symbol: str, details: str):
    """Log CLIP prediction errors"""
    log_scan_error(f"{symbol} → CLIP error: {details}", "CLIP")

def log_chart_error(symbol: str, details: str):
    """Log chart generation errors"""
    log_scan_error(f"{symbol} → Chart generation failed: {details}", "CHART")

def log_global_error(component: str, details: str):
    """Log system-wide errors"""
    log_scan_error(f"{component} → {details}", "GLOBAL")

def log_warning(symbol: str, warning_type: str, details: str):
    """Log warnings (non-critical issues)"""
    log_scan_error(f"{symbol} → ⚠️ {warning_type}: {details}", "WARNING")

def generate_error_summary() -> Dict[str, Any]:
    """
    Generate comprehensive error summary for the current scan session
    
    Returns:
        Dictionary with error statistics and categorized errors
    """
    global scan_errors, scan_start_time, scan_session_id
    
    with error_lock:
        total_errors = len(scan_errors)
        
        if total_errors == 0:
            return {
                "session_id": scan_session_id,
                "total_errors": 0,
                "scan_duration": time.time() - scan_start_time if scan_start_time else 0,
                "categories": {},
                "errors": []
            }
        
        # Categorize errors
        categories = {}
        for error in scan_errors:
            # Extract category from error message
            if "] [" in error:
                category = error.split("] [")[1].split("]")[0]
                categories[category] = categories.get(category, 0) + 1
        
        return {
            "session_id": scan_session_id,
            "total_errors": total_errors,
            "scan_duration": time.time() - scan_start_time if scan_start_time else 0,
            "categories": categories,
            "errors": scan_errors.copy()
        }

def print_error_summary():
    """Print formatted error summary to console"""
    summary = generate_error_summary()
    
    print("\n" + "="*60)
    print("📋 [SCAN ERROR SUMMARY]")
    print("="*60)
    
    if summary["total_errors"] == 0:
        print("✅ No errors during scan.")
        print(f"⏱️ Scan duration: {summary['scan_duration']:.1f}s")
        return
    
    print(f"📊 Session: {summary['session_id']}")
    print(f"⏱️ Duration: {summary['scan_duration']:.1f}s")
    print(f"❌ Total errors: {summary['total_errors']}")
    
    # Print category breakdown
    if summary["categories"]:
        print("\n📊 [ERROR CATEGORIES]")
        for category, count in sorted(summary["categories"].items()):
            print(f"   {category}: {count} errors")
    
    # Print detailed error list
    print(f"\n📝 [DETAILED ERROR LOG]")
    for idx, error in enumerate(summary["errors"], 1):
        # Remove timestamp and category for cleaner display
        clean_error = error
        if "] [" in error and "]" in error:
            parts = error.split("] ", 2)
            if len(parts) >= 3:
                clean_error = parts[2]
        
        print(f"{idx:3}. {clean_error}")
    
    print("="*60)

def save_error_report(filepath: str = None):
    """
    Save error report to file
    
    Args:
        filepath: Optional custom filepath. Defaults to logs/scan_errors_{session_id}.json
    """
    import json
    import os
    
    summary = generate_error_summary()
    
    if not filepath:
        os.makedirs("logs", exist_ok=True)
        filepath = f"logs/scan_errors_{summary['session_id']}.json"
    
    try:
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"💾 Error report saved: {filepath}")
    except Exception as e:
        print(f"❌ Failed to save error report: {e}")

def get_error_count() -> int:
    """Get current error count"""
    with error_lock:
        return len(scan_errors)

def has_errors() -> bool:
    """Check if any errors have been logged"""
    return get_error_count() > 0

def clear_errors():
    """Clear all accumulated errors (use with caution)"""
    global scan_errors
    with error_lock:
        scan_errors.clear()
    print("🧹 Error log cleared")

# Convenience functions for common error patterns
def log_fallback_used(symbol: str, primary_source: str, fallback_source: str):
    """Log when fallback data source is used"""
    log_warning(symbol, "Fallback Source", f"{primary_source} failed → using {fallback_source}")

def log_insufficient_data(symbol: str, data_type: str, reason: str):
    """Log when insufficient data prevents processing"""
    log_warning(symbol, "Insufficient Data", f"{data_type} - {reason}")

def log_prediction_skipped(symbol: str, predictor_type: str, reason: str):
    """Log when prediction is skipped"""
    log_warning(symbol, "Prediction Skipped", f"{predictor_type} - {reason}")

def log_processing_timeout(symbol: str, operation: str, timeout_duration: float):
    """Log when operation times out"""
    log_token_error(symbol, "Timeout", f"{operation} exceeded {timeout_duration}s")

def log_validation_failure(symbol: str, validation_type: str, details: str):
    """Log when data validation fails"""
    log_token_error(symbol, "Validation Failed", f"{validation_type} - {details}")