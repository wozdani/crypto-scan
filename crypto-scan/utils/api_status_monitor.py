#!/usr/bin/env python3
"""
API Status Monitor for Bybit Authentication Issues
Tracks and manages API authentication failures to prevent unnecessary retry attempts
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

@dataclass
class APIStatus:
    """API status tracking"""
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    failure_codes: List[int] = None
    is_blocked: bool = False
    block_until: Optional[float] = None
    
    def __post_init__(self):
        if self.failure_codes is None:
            self.failure_codes = []

class APIStatusMonitor:
    """Monitor and manage API authentication status"""
    
    def __init__(self, cache_file: str = "data/api_status_cache.json"):
        self.cache_file = cache_file
        self.status_cache: Dict[str, APIStatus] = {}
        self.session_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "blocked_requests": 0,
            "auth_failures": 0
        }
        self._load_cache()
    
    def _load_cache(self):
        """Load API status cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for endpoint, status_data in data.get("endpoints", {}).items():
                        self.status_cache[endpoint] = APIStatus(
                            last_success=status_data.get("last_success"),
                            last_failure=status_data.get("last_failure"),
                            consecutive_failures=status_data.get("consecutive_failures", 0),
                            failure_codes=status_data.get("failure_codes", []),
                            is_blocked=status_data.get("is_blocked", False),
                            block_until=status_data.get("block_until")
                        )
                    self.session_stats = data.get("session_stats", self.session_stats)
                print(f"[API MONITOR] Loaded cache with {len(self.status_cache)} endpoints")
        except Exception as e:
            print(f"[API MONITOR] Cache load error: {e}")
    
    def _save_cache(self):
        """Save API status cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache_data = {
                "endpoints": {},
                "session_stats": self.session_stats,
                "last_updated": time.time()
            }
            
            for endpoint, status in self.status_cache.items():
                cache_data["endpoints"][endpoint] = {
                    "last_success": status.last_success,
                    "last_failure": status.last_failure,
                    "consecutive_failures": status.consecutive_failures,
                    "failure_codes": status.failure_codes,
                    "is_blocked": status.is_blocked,
                    "block_until": status.block_until
                }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            print(f"[API MONITOR] Cache save error: {e}")
    
    def record_request(self, endpoint: str, status_code: int, response_data: Optional[Dict] = None):
        """Record API request result"""
        current_time = time.time()
        self.session_stats["total_requests"] += 1
        
        if endpoint not in self.status_cache:
            self.status_cache[endpoint] = APIStatus()
        
        status = self.status_cache[endpoint]
        
        if status_code == 200 and response_data and self._is_valid_response(response_data):
            # Successful request
            status.last_success = current_time
            status.consecutive_failures = 0
            status.is_blocked = False
            status.block_until = None
            self.session_stats["successful_requests"] += 1
            print(f"[API SUCCESS] {endpoint}: HTTP {status_code}")
            
        else:
            # Failed request
            status.last_failure = current_time
            status.consecutive_failures += 1
            if status_code not in status.failure_codes:
                status.failure_codes.append(status_code)
            self.session_stats["failed_requests"] += 1
            
            # Track authentication failures
            if status_code == 403:
                self.session_stats["auth_failures"] += 1
                print(f"[API AUTH FAILURE] {endpoint}: HTTP 403 - Auth issue detected")
            
            # Block endpoint if too many consecutive failures
            if status.consecutive_failures >= 5:
                block_duration = min(300, status.consecutive_failures * 30)  # Max 5 minutes
                status.is_blocked = True
                status.block_until = current_time + block_duration
                print(f"[API BLOCKED] {endpoint}: Blocked for {block_duration}s after {status.consecutive_failures} failures")
        
        self._save_cache()
    
    def _is_valid_response(self, response_data: Dict) -> bool:
        """Check if response contains valid data"""
        if not response_data:
            return False
        
        result = response_data.get("result")
        if not result:
            return False
        
        # Check for valid ticker data
        if "list" in result and result["list"]:
            return True
        
        # Check for valid candle data
        if "list" in result and isinstance(result["list"], list):
            return True
        
        # Check for valid orderbook data
        if "b" in result and "a" in result:
            return True
        
        return False
    
    def should_skip_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint should be skipped due to blocking"""
        if endpoint not in self.status_cache:
            return False
        
        status = self.status_cache[endpoint]
        current_time = time.time()
        
        # Check if block has expired
        if status.is_blocked and status.block_until and current_time > status.block_until:
            status.is_blocked = False
            status.block_until = None
            print(f"[API UNBLOCK] {endpoint}: Block expired, allowing requests")
            return False
        
        if status.is_blocked:
            remaining = int(status.block_until - current_time) if status.block_until else 0
            print(f"[API SKIP] {endpoint}: Blocked for {remaining}s more")
            self.session_stats["blocked_requests"] += 1
            return True
        
        return False
    
    def get_endpoint_status(self, endpoint: str) -> Dict:
        """Get detailed status for endpoint"""
        if endpoint not in self.status_cache:
            return {"status": "unknown", "never_accessed": True}
        
        status = self.status_cache[endpoint]
        current_time = time.time()
        
        result = {
            "endpoint": endpoint,
            "consecutive_failures": status.consecutive_failures,
            "failure_codes": status.failure_codes,
            "is_blocked": status.is_blocked
        }
        
        if status.last_success:
            result["last_success"] = datetime.fromtimestamp(status.last_success).isoformat()
            result["success_age_minutes"] = (current_time - status.last_success) / 60
        
        if status.last_failure:
            result["last_failure"] = datetime.fromtimestamp(status.last_failure).isoformat()
            result["failure_age_minutes"] = (current_time - status.last_failure) / 60
        
        if status.block_until:
            result["block_expires"] = datetime.fromtimestamp(status.block_until).isoformat()
            result["block_remaining_seconds"] = max(0, status.block_until - current_time)
        
        return result
    
    def get_session_summary(self) -> Dict:
        """Get session statistics summary"""
        total = self.session_stats["total_requests"]
        success_rate = (self.session_stats["successful_requests"] / total * 100) if total > 0 else 0
        
        blocked_endpoints = sum(1 for status in self.status_cache.values() if status.is_blocked)
        auth_failure_rate = (self.session_stats["auth_failures"] / total * 100) if total > 0 else 0
        
        return {
            "session_stats": self.session_stats.copy(),
            "success_rate_percent": round(success_rate, 1),
            "auth_failure_rate_percent": round(auth_failure_rate, 1),
            "blocked_endpoints": blocked_endpoints,
            "total_endpoints_tracked": len(self.status_cache),
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations based on API status"""
        recommendations = []
        
        auth_failures = self.session_stats["auth_failures"]
        total_requests = self.session_stats["total_requests"]
        
        if auth_failures > 10:
            recommendations.append("High number of HTTP 403 errors - check API credentials")
        
        if total_requests > 0:
            auth_rate = auth_failures / total_requests
            if auth_rate > 0.5:
                recommendations.append("Over 50% authentication failures - API keys may be invalid")
            elif auth_rate > 0.1:
                recommendations.append("Significant authentication issues - verify API permissions")
        
        blocked_count = sum(1 for status in self.status_cache.values() if status.is_blocked)
        if blocked_count > 5:
            recommendations.append(f"{blocked_count} endpoints blocked - consider API rate limiting")
        
        return recommendations

# Global instance
_api_monitor = None

def get_api_monitor() -> APIStatusMonitor:
    """Get global API status monitor instance"""
    global _api_monitor
    if _api_monitor is None:
        _api_monitor = APIStatusMonitor()
    return _api_monitor

def record_api_request(endpoint: str, status_code: int, response_data: Optional[Dict] = None):
    """Convenience function to record API request"""
    monitor = get_api_monitor()
    monitor.record_request(endpoint, status_code, response_data)

def should_skip_api_endpoint(endpoint: str) -> bool:
    """Convenience function to check if endpoint should be skipped"""
    monitor = get_api_monitor()
    return monitor.should_skip_endpoint(endpoint)

def get_api_status_summary() -> Dict:
    """Convenience function to get API status summary"""
    monitor = get_api_monitor()
    return monitor.get_session_summary()

def main():
    """Test API status monitor"""
    monitor = APIStatusMonitor()
    
    # Simulate some API requests
    monitor.record_request("tickers", 403, None)
    monitor.record_request("tickers", 403, None)
    monitor.record_request("orderbook", 200, {"result": {"b": [], "a": []}})
    
    print(f"Should skip tickers: {monitor.should_skip_endpoint('tickers')}")
    print(f"Should skip orderbook: {monitor.should_skip_endpoint('orderbook')}")
    
    summary = monitor.get_session_summary()
    print(f"Session summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()