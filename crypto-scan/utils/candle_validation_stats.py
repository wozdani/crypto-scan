#!/usr/bin/env python3
"""
Candle Validation Statistics Tracker
Monitors and reports effectiveness of candle validation system
"""

import json
import os
from datetime import datetime
from typing import Dict, List

class CandleValidationStats:
    """Track candle validation statistics"""
    
    def __init__(self, stats_file: str = "data/candle_validation_stats.json"):
        self.stats_file = stats_file
        self.session_stats = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "total_scanned": 0,
            "candle_skipped": 0,
            "candle_passed": 0,
            "invalid_15m": 0,
            "invalid_5m": 0,
            "geographic_blocked": 0,
            "api_errors": 0,
            "processed_successfully": 0,
            "skip_reasons": {}
        }
        
    def record_skip(self, symbol: str, reason: str, candles_15m: int = 0, candles_5m: int = 0):
        """Record a token skip with reason"""
        self.session_stats["total_scanned"] += 1
        self.session_stats["candle_skipped"] += 1
        
        # Categorize skip reasons
        if "15M candles" in reason:
            self.session_stats["invalid_15m"] += 1
        elif "5M candles" in reason:
            self.session_stats["invalid_5m"] += 1
        elif "Invalid" in reason and "candle data" in reason:
            self.session_stats["api_errors"] += 1
        elif "403" in reason or "geographical" in reason:
            self.session_stats["geographic_blocked"] += 1
            
        # Track detailed reasons
        if reason not in self.session_stats["skip_reasons"]:
            self.session_stats["skip_reasons"][reason] = 0
        self.session_stats["skip_reasons"][reason] += 1
        
    def record_pass(self, symbol: str, candles_15m: int, candles_5m: int):
        """Record a successful candle validation"""
        self.session_stats["total_scanned"] += 1
        self.session_stats["candle_passed"] += 1
        
    def record_processed(self, symbol: str):
        """Record a successfully processed token"""
        self.session_stats["processed_successfully"] += 1
        
    def get_session_summary(self) -> Dict:
        """Get current session statistics"""
        total = self.session_stats["total_scanned"]
        if total == 0:
            return self.session_stats
            
        skip_rate = (self.session_stats["candle_skipped"] / total) * 100
        pass_rate = (self.session_stats["candle_passed"] / total) * 100
        
        summary = self.session_stats.copy()
        summary.update({
            "skip_rate_pct": round(skip_rate, 1),
            "pass_rate_pct": round(pass_rate, 1),
            "efficiency_gain": f"Skipped {self.session_stats['candle_skipped']} tokens early, saved processing time",
            "end_time": datetime.now().isoformat()
        })
        
        return summary
        
    def save_session_stats(self):
        """Save session statistics to file"""
        summary = self.get_session_summary()
        
        # Load existing stats
        all_stats = []
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    all_stats = json.load(f)
            except:
                all_stats = []
                
        # Add current session
        all_stats.append(summary)
        
        # Keep only last 10 sessions
        all_stats = all_stats[-10:]
        
        # Save updated stats
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        with open(self.stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
            
    def print_session_report(self):
        """Print detailed session report"""
        summary = self.get_session_summary()
        
        print(f"\n📊 Candle Validation Report - Session {summary['session_id']}")
        print("=" * 60)
        print(f"Total Scanned: {summary['total_scanned']}")
        print(f"✅ Passed Validation: {summary['candle_passed']} ({summary['pass_rate_pct']}%)")
        print(f"❌ Skipped Validation: {summary['candle_skipped']} ({summary['skip_rate_pct']}%)")
        print(f"🚀 Successfully Processed: {summary['processed_successfully']}")
        
        print(f"\nSkip Breakdown:")
        print(f"  • Insufficient 15M: {summary['invalid_15m']}")
        print(f"  • Insufficient 5M: {summary['invalid_5m']}")
        print(f"  • API Errors: {summary['api_errors']}")
        print(f"  • Geographic Blocks: {summary['geographic_blocked']}")
        
        if summary['skip_reasons']:
            print(f"\nTop Skip Reasons:")
            sorted_reasons = sorted(summary['skip_reasons'].items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_reasons[:5]:
                print(f"  • {reason}: {count}")
                
        print(f"\n⚡ Efficiency: {summary['efficiency_gain']}")

# Global stats tracker instance
_stats_tracker = None

def get_stats_tracker() -> CandleValidationStats:
    """Get global statistics tracker"""
    global _stats_tracker
    if _stats_tracker is None:
        _stats_tracker = CandleValidationStats()
    return _stats_tracker

def record_candle_skip(symbol: str, reason: str, candles_15m: int = 0, candles_5m: int = 0):
    """Record a candle validation skip"""
    tracker = get_stats_tracker()
    tracker.record_skip(symbol, reason, candles_15m, candles_5m)

def record_candle_pass(symbol: str, candles_15m: int, candles_5m: int):
    """Record a successful candle validation"""
    tracker = get_stats_tracker()
    tracker.record_pass(symbol, candles_15m, candles_5m)

def record_token_processed(symbol: str):
    """Record a successfully processed token"""
    tracker = get_stats_tracker()
    tracker.record_processed(symbol)

def print_validation_report():
    """Print current validation report"""
    tracker = get_stats_tracker()
    tracker.print_session_report()

def save_validation_stats():
    """Save current validation statistics"""
    tracker = get_stats_tracker()
    tracker.save_session_stats()

if __name__ == "__main__":
    # Test the statistics system
    print("🧪 Testing Candle Validation Statistics...")
    
    tracker = CandleValidationStats()
    
    # Simulate some validation events
    tracker.record_skip("TEST1USDT", "Insufficient 15M candles (5/20)", 5, 100)
    tracker.record_skip("TEST2USDT", "Insufficient 5M candles (30/60)", 50, 30)
    tracker.record_skip("TEST3USDT", "Invalid 15M candle data - API error")
    tracker.record_pass("TEST4USDT", 50, 150)
    tracker.record_pass("TEST5USDT", 96, 288)
    tracker.record_processed("TEST4USDT")
    tracker.record_processed("TEST5USDT")
    
    tracker.print_session_report()
    print("\n✅ Statistics system test completed!")