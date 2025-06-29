"""
Performance Optimizer for Crypto Scanner
Optimizes scan performance to meet <15s target for 750+ tokens
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

# Import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from crypto_scan_service import log_warning
except ImportError:
    def log_warning(label, exception=None, additional_info=None):
        print(f"⚠️ [{label}] {exception} - {additional_info}")

class PerformanceOptimizer:
    """
    Optimizes scanning performance through various strategies
    """
    
    def __init__(self):
        self.target_time = 15  # Target scan time in seconds
        self.max_concurrent = 400  # Very aggressive concurrency for 752 tokens
        self.volume_threshold = 100000  # Minimum volume threshold
        self.batch_size = 50  # Process in batches
        
    def prioritize_high_volume_tokens(self, symbols: List[str], limit: int = 752) -> List[str]:
        """
        Prioritize tokens by volume to focus on most active markets
        
        Args:
            symbols: List of all symbols
            limit: Maximum symbols to process
            
        Returns:
            Prioritized symbol list
        """
        try:
            # Load volume data from cache if available
            volume_data = {}
            cache_path = "data/volume_cache.json"
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        volume_data = json.load(f)
                except Exception as e:
                    log_warning("VOLUME CACHE LOAD ERROR", e)
            
            # If no volume data, return top symbols by alphabetical order (fallback)
            if not volume_data:
                print(f"[PERFORMANCE] No volume data available, using first {limit} symbols")
                return symbols[:limit]
            
            # Sort symbols by volume
            prioritized = []
            for symbol in symbols:
                volume = volume_data.get(symbol, 0)
                if volume >= self.volume_threshold:
                    prioritized.append((symbol, volume))
            
            # Sort by volume descending
            prioritized.sort(key=lambda x: x[1], reverse=True)
            
            # Return top symbols
            result = [symbol for symbol, volume in prioritized[:limit]]
            
            print(f"[PERFORMANCE] Prioritized {len(result)} high-volume tokens (min volume: ${self.volume_threshold:,})")
            
            # Fill remaining slots with other symbols if needed
            if len(result) < limit:
                remaining = [s for s in symbols if s not in result]
                result.extend(remaining[:limit - len(result)])
            
            return result
            
        except Exception as e:
            log_warning("VOLUME PRIORITIZATION ERROR", e)
            return symbols[:limit]
    
    def optimize_concurrency_for_target(self, symbol_count: int) -> int:
        """
        Calculate optimal concurrency for target time
        
        Args:
            symbol_count: Number of symbols to process
            
        Returns:
            Optimal concurrent connections
        """
        # Target: process all symbols in <15s
        # Assume each token takes ~0.5s average
        target_per_second = symbol_count / self.target_time
        
        # Calculate required concurrency (with overhead factor)
        optimal_concurrency = int(target_per_second * 2.5)  # 2.5x overhead for safety
        
        # Cap at reasonable limits for 752 token processing
        optimal_concurrency = min(optimal_concurrency, 400)  # Max 400 concurrent for 752 tokens
        optimal_concurrency = max(optimal_concurrency, 50)   # Min 50 concurrent
        
        print(f"[PERFORMANCE] Target: {symbol_count} tokens in <{self.target_time}s")
        print(f"[PERFORMANCE] Optimal concurrency: {optimal_concurrency} connections")
        
        return optimal_concurrency
    
    def create_performance_config(self, symbols: List[str]) -> Dict:
        """
        Create optimized configuration for scanning
        
        Args:
            symbols: List of symbols to scan
            
        Returns:
            Performance configuration
        """
        # Prioritize high-volume tokens - Process all 752 symbols
        prioritized_symbols = self.prioritize_high_volume_tokens(symbols, 752)
        
        # Calculate optimal concurrency
        optimal_concurrency = self.optimize_concurrency_for_target(len(prioritized_symbols))
        
        config = {
            'symbols': prioritized_symbols,
            'max_concurrent': optimal_concurrency,
            'batch_size': self.batch_size,
            'fast_mode': True,
            'skip_detailed_analysis': False,  # Keep TJDE analysis
            'timeout_per_token': 3.0,  # 3 second timeout per token
            'enable_caching': True,
            'parallel_processing': True
        }
        
        return config
    
    def estimate_scan_time(self, config: Dict) -> float:
        """
        Estimate scan completion time based on configuration
        
        Args:
            config: Performance configuration
            
        Returns:
            Estimated time in seconds
        """
        symbol_count = len(config['symbols'])
        concurrency = config['max_concurrent']
        timeout_per_token = config['timeout_per_token']
        
        # Conservative estimate: assume 50% of tokens hit timeout
        avg_time_per_token = timeout_per_token * 0.5
        
        # Calculate with parallel processing
        estimated_time = (symbol_count * avg_time_per_token) / concurrency
        
        # Add overhead for startup, cache building, etc.
        estimated_time += 3.0  # 3 second overhead
        
        return estimated_time
    
    def monitor_performance(self, start_time: float, processed_count: int, total_count: int):
        """
        Monitor and report performance during scan
        
        Args:
            start_time: Scan start timestamp
            processed_count: Number of tokens processed
            total_count: Total tokens to process
        """
        elapsed_time = time.time() - start_time
        
        if processed_count > 0:
            rate = processed_count / elapsed_time
            estimated_total_time = total_count / rate
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"[PERFORMANCE] Progress: {processed_count}/{total_count} tokens")
            print(f"[PERFORMANCE] Rate: {rate:.1f} tokens/second")
            print(f"[PERFORMANCE] Elapsed: {elapsed_time:.1f}s, ETA: {remaining_time:.1f}s")
            
            # Warning if behind target
            if elapsed_time > self.target_time * (processed_count / total_count):
                print(f"[PERFORMANCE WARNING] Behind target pace")
    
    def optimize_async_scanner_config(self) -> Dict:
        """
        Get optimized configuration for AsyncCryptoScanner
        
        Returns:
            Optimized scanner configuration
        """
        return {
            'max_concurrent': 250,  # Very high concurrency
            'request_timeout': 2.0,  # Fast timeout
            'fast_mode': True,
            'enable_compression': True,
            'connection_pool_size': 400,
            'connection_pool_maxsize': 400
        }
    
    def should_skip_detailed_analysis(self, symbol: str, basic_score: float) -> bool:
        """
        Determine if detailed TJDE analysis should be skipped for performance
        
        Args:
            symbol: Trading symbol
            basic_score: Basic screening score
            
        Returns:
            True if detailed analysis should be skipped
        """
        # Skip detailed analysis for very low scores to save time
        if basic_score < 15:  # PPWCS < 15 unlikely to generate alerts
            return True
        
        # Always do detailed analysis for high priority tokens
        return False
    
    def batch_process_symbols(self, symbols: List[str], batch_size: int = None) -> List[List[str]]:
        """
        Split symbols into processing batches
        
        Args:
            symbols: List of symbols
            batch_size: Size of each batch
            
        Returns:
            List of symbol batches
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        batches = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batches.append(batch)
        
        print(f"[PERFORMANCE] Split {len(symbols)} symbols into {len(batches)} batches")
        return batches
    
    def save_performance_metrics(self, metrics: Dict):
        """
        Save performance metrics for analysis
        
        Args:
            metrics: Performance metrics dictionary
        """
        try:
            os.makedirs("data/performance", exist_ok=True)
            metrics_path = f"data/performance/scan_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            print(f"[PERFORMANCE] Metrics saved: {metrics_path}")
            
        except Exception as e:
            log_warning("PERFORMANCE METRICS SAVE ERROR", e)

# Global optimizer instance
_performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer

def optimize_scan_performance(symbols: List[str]) -> Dict:
    """
    Get optimized configuration for high-performance scanning
    
    Args:
        symbols: List of symbols to scan
        
    Returns:
        Optimized configuration
    """
    optimizer = get_performance_optimizer()
    return optimizer.create_performance_config(symbols)

def main():
    """Test performance optimization"""
    print("⚡ PERFORMANCE OPTIMIZER TEST")
    print("=" * 50)
    
    # Test with sample symbols
    test_symbols = [f"TEST{i}USDT" for i in range(500)]
    
    optimizer = PerformanceOptimizer()
    config = optimizer.create_performance_config(test_symbols)
    
    print(f"📊 Configuration:")
    print(f"• Symbols to process: {len(config['symbols'])}")
    print(f"• Max concurrent: {config['max_concurrent']}")
    print(f"• Batch size: {config['batch_size']}")
    print(f"• Fast mode: {config['fast_mode']}")
    
    estimated_time = optimizer.estimate_scan_time(config)
    print(f"\n⏱️ Estimated scan time: {estimated_time:.1f}s")
    
    if estimated_time <= 15:
        print("✅ Configuration meets <15s target")
    else:
        print("⚠️ Configuration exceeds 15s target")

if __name__ == "__main__":
    main()