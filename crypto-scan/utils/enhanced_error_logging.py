"""
Enhanced Error Logging System
Provides detailed error analysis and categorization for debugging
"""

import json
import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any

class ErrorLogger:
    """Enhanced error logging with categorization and analysis"""
    
    def __init__(self, log_file: str = "logs/enhanced_errors.jsonl"):
        self.log_file = log_file
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Ensure log directory exists"""
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        except Exception as e:
            print(f"[ERROR LOG SETUP] Failed to create log directory: {e}")
    
    def log_api_error(self, symbol: str, endpoint: str, error_type: str, 
                     response_data: Any = None, exception: Exception = None):
        """Log detailed API error with context"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "error_category": "api_error",
            "endpoint": endpoint,
            "error_type": error_type,
            "response_data": str(response_data) if response_data else None,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_message": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception else None
        }
        
        self._write_log_entry(error_entry)
        
        # Enhanced console logging
        print(f"[API ERROR DETAIL] {symbol}:")
        print(f"  ├─ Endpoint: {endpoint}")
        print(f"  ├─ Error Type: {error_type}")
        if response_data:
            print(f"  ├─ Response: {str(response_data)[:200]}{'...' if len(str(response_data)) > 200 else ''}")
        if exception:
            print(f"  └─ Exception: {type(exception).__name__}: {str(exception)}")
    
    def log_data_validation_error(self, symbol: str, data_type: str, 
                                 expected: str, received: Any, reason: str = None):
        """Log data validation errors with details"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "error_category": "data_validation",
            "data_type": data_type,
            "expected": expected,
            "received_type": type(received).__name__,
            "received_value": str(received) if received is not None else "None",
            "reason": reason or "Data validation failed"
        }
        
        self._write_log_entry(error_entry)
        
        print(f"[DATA VALIDATION ERROR] {symbol}:")
        print(f"  ├─ Data Type: {data_type}")
        print(f"  ├─ Expected: {expected}")
        print(f"  ├─ Received: {type(received).__name__} = {str(received)[:100]}")
        if reason:
            print(f"  └─ Reason: {reason}")
    
    def log_processing_error(self, symbol: str, processing_stage: str, 
                           error_details: Dict, exception: Exception = None):
        """Log processing errors with stage information"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "error_category": "processing_error",
            "processing_stage": processing_stage,
            "error_details": error_details,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_message": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception else None
        }
        
        self._write_log_entry(error_entry)
        
        print(f"[PROCESSING ERROR] {symbol} at {processing_stage}:")
        for key, value in error_details.items():
            print(f"  ├─ {key}: {value}")
        if exception:
            print(f"  └─ Exception: {type(exception).__name__}: {str(exception)}")
    
    def _write_log_entry(self, entry: Dict):
        """Write log entry to JSONL file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"[ERROR LOG WRITE] Failed to write log entry: {e}")
    
    def analyze_error_patterns(self, hours_back: int = 24) -> Dict:
        """Analyze error patterns from recent logs"""
        try:
            if not os.path.exists(self.log_file):
                return {"error": "No log file found"}
            
            errors = []
            cutoff_time = datetime.now().timestamp() - (hours_back * 3600)
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry.get("timestamp", "")).timestamp()
                        if entry_time > cutoff_time:
                            errors.append(entry)
                    except:
                        continue
            
            # Analyze patterns
            analysis = {
                "total_errors": len(errors),
                "error_categories": {},
                "frequent_symbols": {},
                "common_error_types": {},
                "endpoint_failures": {}
            }
            
            for error in errors:
                # Category analysis
                category = error.get("error_category", "unknown")
                analysis["error_categories"][category] = analysis["error_categories"].get(category, 0) + 1
                
                # Symbol analysis
                symbol = error.get("symbol", "unknown")
                analysis["frequent_symbols"][symbol] = analysis["frequent_symbols"].get(symbol, 0) + 1
                
                # Error type analysis
                error_type = error.get("error_type", error.get("exception_type", "unknown"))
                analysis["common_error_types"][error_type] = analysis["common_error_types"].get(error_type, 0) + 1
                
                # Endpoint analysis
                if "endpoint" in error:
                    endpoint = error["endpoint"]
                    analysis["endpoint_failures"][endpoint] = analysis["endpoint_failures"].get(endpoint, 0) + 1
            
            # Sort by frequency
            for key in ["frequent_symbols", "common_error_types", "endpoint_failures"]:
                analysis[key] = dict(sorted(analysis[key].items(), key=lambda x: x[1], reverse=True))
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

# Global error logger instance
_error_logger = None

def get_error_logger() -> ErrorLogger:
    """Get global error logger instance"""
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorLogger()
    return _error_logger

def log_api_error(symbol: str, endpoint: str, error_type: str, 
                 response_data: Any = None, exception: Exception = None):
    """Convenience function for API error logging"""
    get_error_logger().log_api_error(symbol, endpoint, error_type, response_data, exception)

def log_data_validation_error(symbol: str, data_type: str, expected: str, 
                            received: Any, reason: str = None):
    """Convenience function for data validation error logging"""
    get_error_logger().log_data_validation_error(symbol, data_type, expected, received, reason)

def log_processing_error(symbol: str, processing_stage: str, 
                       error_details: Dict, exception: Exception = None):
    """Convenience function for processing error logging"""
    get_error_logger().log_processing_error(symbol, processing_stage, error_details, exception)

def analyze_recent_errors(hours_back: int = 24) -> Dict:
    """Analyze recent error patterns"""
    return get_error_logger().analyze_error_patterns(hours_back)

def main():
    """Test enhanced error logging"""
    logger = get_error_logger()
    
    # Test API error
    logger.log_api_error("TESTUSDT", "/v5/market/tickers", "HTTP_403", 
                        {"error": "Forbidden"}, Exception("API access denied"))
    
    # Test data validation error
    logger.log_data_validation_error("TESTUSDT", "ticker_data", "dict with lastPrice", 
                                   None, "API returned empty response")
    
    # Test processing error
    logger.log_processing_error("TESTUSDT", "candle_processing", 
                              {"candles_received": 0, "candles_expected": 100})
    
    # Analyze patterns
    analysis = logger.analyze_error_patterns(1)
    print("Error Analysis:", json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()