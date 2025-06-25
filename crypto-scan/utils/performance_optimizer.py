#!/usr/bin/env python3
"""
Performance Optimizer for Async Scanner
Implements fast mode optimizations and bottleneck reduction
"""

import time
import asyncio
from typing import Dict, List, Optional

class PerformanceOptimizer:
    """Optimizes async scanning performance"""
    
    def __init__(self):
        self.scan_metrics = {
            'tokens_processed': 0,
            'total_time': 0,
            'api_calls': 0,
            'disk_writes': 0,
            'bottlenecks': []
        }
    
    def start_scan_timer(self):
        """Start performance measurement"""
        self.scan_start = time.time()
        return self.scan_start
    
    def end_scan_timer(self, tokens_count: int):
        """End performance measurement and calculate metrics"""
        total_time = time.time() - self.scan_start
        self.scan_metrics.update({
            'tokens_processed': tokens_count,
            'total_time': total_time,
            'tokens_per_second': tokens_count / total_time if total_time > 0 else 0
        })
        
        # Identify bottlenecks
        if total_time > 15:
            self.scan_metrics['bottlenecks'].append('scan_timeout')
        if self.scan_metrics['tokens_per_second'] < 50:
            self.scan_metrics['bottlenecks'].append('low_throughput')
            
        return self.scan_metrics
    
    def optimize_concurrency(self, total_tokens: int, target_time: int = 15) -> int:
        """Calculate optimal concurrency for target completion time"""
        # Estimate based on target: 750 tokens in 15s = 50 tokens/s
        # With I/O wait, aim for 80-100 concurrent workers
        if total_tokens <= 200:
            return min(total_tokens, 40)
        elif total_tokens <= 500:
            return min(total_tokens, 60)
        else:
            return min(total_tokens, 80)
    
    def should_skip_heavy_operations(self, score: float, fast_mode: bool = False) -> Dict[str, bool]:
        """Determine which operations to skip based on score and mode"""
        return {
            'skip_tjde': fast_mode and score < 40,
            'skip_chart_generation': fast_mode and score < 35,
            'skip_clip_analysis': fast_mode and score < 30,
            'skip_detailed_logging': fast_mode,
            'skip_disk_writes': fast_mode and score < 25
        }
    
    def log_performance_summary(self):
        """Log performance summary"""
        metrics = self.scan_metrics
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   Tokens: {metrics['tokens_processed']}")
        print(f"   Time: {metrics['total_time']:.1f}s (Target: <15s)")
        print(f"   Speed: {metrics['tokens_per_second']:.1f} tokens/s (Target: >50)")
        print(f"   API calls: {metrics['api_calls']}")
        
        if metrics['bottlenecks']:
            print(f"   Bottlenecks: {', '.join(metrics['bottlenecks'])}")
        else:
            print("   ✅ Performance targets met")

# Global optimizer instance
performance_optimizer = PerformanceOptimizer()

def get_performance_optimizer():
    """Get global performance optimizer"""
    return performance_optimizer

async def optimize_batch_processing(items: List, batch_size: int = 20, delay: float = 0.01):
    """Optimize batch processing with controlled delays"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch
        if delay > 0:
            await asyncio.sleep(delay)

def reduce_disk_io_overhead(data: Dict, fast_mode: bool = False) -> bool:
    """Determine if disk write should be performed"""
    if fast_mode:
        # Only write high-value results in fast mode
        tjde_score = data.get('tjde_score', 0)
        ppwcs_score = data.get('ppwcs_score', 0)
        return tjde_score >= 0.6 or ppwcs_score >= 60
    return True

def optimize_memory_usage():
    """Clean up memory to prevent bottlenecks"""
    import gc
    gc.collect()