"""
TradingView Async Event Loop Fix
Resolves asyncio.run() conflicts when already in event loop environment
"""

import asyncio
import threading
import os
import sys
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from crypto_scan_service import log_warning
except ImportError:
    def log_warning(label, exception=None, additional_info=None):
        print(f"⚠️ [{label}] {exception} - {additional_info}")

try:
    from .tradingview_screenshot import TradingViewScreenshotGenerator, PLAYWRIGHT_AVAILABLE
except ImportError:
    try:
        from tradingview_screenshot import TradingViewScreenshotGenerator, PLAYWRIGHT_AVAILABLE
    except ImportError:
        PLAYWRIGHT_AVAILABLE = False
        log_warning("PLAYWRIGHT IMPORT", None, "TradingView screenshots unavailable")

class TradingViewAsyncFix:
    """
    Fixes asyncio event loop conflicts for TradingView screenshot generation
    Uses thread-based execution to avoid nested event loop issues
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def generate_charts_sync_safe(
        self, 
        tjde_results: List[Dict], 
        min_score: float = 0.5, 
        max_symbols: int = 5
    ) -> List[str]:
        """
        Thread-safe TradingView chart generation that works within existing event loops
        
        Args:
            tjde_results: TJDE analysis results
            min_score: Minimum TJDE score
            max_symbols: Maximum symbols to process
            
        Returns:
            List of generated screenshot paths
        """
        if not PLAYWRIGHT_AVAILABLE:
            log_warning("PLAYWRIGHT NOT AVAILABLE", None, "Cannot generate TradingView charts")
            return []
        
        try:
            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                print("[TRADINGVIEW FIX] Running in existing event loop - using thread-based generation")
                
                # Use thread to avoid event loop conflict
                future = self.executor.submit(self._run_in_new_thread, tjde_results, min_score, max_symbols)
                screenshot_paths = future.result(timeout=120)  # 2 minute timeout
                
                return screenshot_paths
                
            except RuntimeError:
                # No event loop running - safe to use asyncio.run()
                print("[TRADINGVIEW FIX] No event loop detected - using direct async execution")
                return asyncio.run(self._generate_charts_async(tjde_results, min_score, max_symbols))
                
        except Exception as e:
            log_warning("TRADINGVIEW ASYNC FIX ERROR", e, "Chart generation failed")
            return []
    
    def _run_in_new_thread(self, tjde_results: List[Dict], min_score: float, max_symbols: int) -> List[str]:
        """
        Run chart generation in a new thread with its own event loop
        """
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(
                    self._generate_charts_async(tjde_results, min_score, max_symbols)
                )
            finally:
                loop.close()
                
        except Exception as e:
            log_warning("TRADINGVIEW THREAD ERROR", e, "Thread-based generation failed")
            return []
    
    async def _generate_charts_async(
        self, 
        tjde_results: List[Dict], 
        min_score: float, 
        max_symbols: int
    ) -> List[str]:
        """
        Async chart generation with proper error handling
        """
        try:
            # Filter and sort results
            eligible = [r for r in tjde_results if r.get('tjde_score', 0) >= min_score]
            eligible.sort(key=lambda x: x.get('tjde_score', 0), reverse=True)
            top_results = eligible[:max_symbols]
            
            if not top_results:
                print("[TRADINGVIEW FIX] No eligible tokens for screenshot generation")
                return []
            
            print(f"[TRADINGVIEW FIX] Generating charts for {len(top_results)} tokens:")
            for i, result in enumerate(top_results, 1):
                symbol = result.get('symbol', 'UNKNOWN')
                score = result.get('tjde_score', 0)
                decision = result.get('tjde_decision', 'unknown')
                phase = result.get('market_phase', 'unknown')
                print(f"  {i}. {symbol}: TJDE {score:.3f} | {decision} | {phase}")
            
            screenshot_paths = []
            
            # Use async context manager for browser lifecycle
            async with TradingViewScreenshotGenerator() as generator:
                
                # Process each symbol
                for result in top_results:
                    symbol = result.get('symbol', 'UNKNOWN')
                    tjde_score = result.get('tjde_score', 0)
                    market_phase = result.get('market_phase', 'unknown')
                    decision = result.get('tjde_decision', 'unknown')
                    
                    screenshot_path = await generator.generate_tradingview_screenshot(
                        symbol=symbol,
                        tjde_score=tjde_score,
                        market_phase=market_phase,
                        decision=decision
                    )
                    
                    if screenshot_path and os.path.exists(screenshot_path):
                        screenshot_paths.append(screenshot_path)
                        print(f"[TRADINGVIEW FIX] ✅ {symbol}: Generated {os.path.basename(screenshot_path)}")
                    else:
                        print(f"[TRADINGVIEW FIX] ❌ {symbol}: Screenshot generation failed")
                    
                    # Small delay between captures
                    await asyncio.sleep(1)
            
            print(f"[TRADINGVIEW FIX] ✅ Generated {len(screenshot_paths)}/{len(top_results)} charts")
            return screenshot_paths
            
        except Exception as e:
            log_warning("TRADINGVIEW ASYNC GENERATION ERROR", e, "Async generation failed")
            return []
    
    def generate_single_chart(
        self, 
        symbol: str, 
        tjde_score: float, 
        market_phase: str, 
        tjde_decision: str
    ) -> Optional[str]:
        """
        Generate single TradingView chart for a specific symbol
        
        Args:
            symbol: Trading symbol
            tjde_score: TJDE score
            market_phase: Market phase
            tjde_decision: TJDE decision
            
        Returns:
            Path to generated chart or None if failed
        """
        try:
            if not PLAYWRIGHT_AVAILABLE:
                log_warning("PLAYWRIGHT UNAVAILABLE", None, "Cannot generate TradingView screenshots")
                return None
            
            # Create a single result entry
            result = {
                'symbol': symbol,
                'tjde_score': tjde_score,
                'market_phase': market_phase,
                'tjde_decision': tjde_decision
            }
            
            # Use existing batch generation with single item
            charts = self.generate_charts_sync_safe([result], min_score=0.0, max_symbols=1)
            
            return charts[0] if charts else None
            
        except Exception as e:
            log_warning("SINGLE CHART GENERATION ERROR", e, f"Failed to generate chart for {symbol}")
            return None
    
    def __del__(self):
        """Cleanup thread pool"""
        try:
            self.executor.shutdown(wait=False)
        except:
            pass

# Global instance
_tradingview_async_fix = None

def get_tradingview_async_fix() -> TradingViewAsyncFix:
    """Get global TradingView async fix instance"""
    global _tradingview_async_fix
    if _tradingview_async_fix is None:
        _tradingview_async_fix = TradingViewAsyncFix()
    return _tradingview_async_fix

def generate_tradingview_charts_safe(
    tjde_results: List[Dict], 
    min_score: float = 0.5, 
    max_symbols: int = 5
) -> List[str]:
    """
    Safe TradingView chart generation that works in any async context
    
    Args:
        tjde_results: TJDE analysis results
        min_score: Minimum TJDE score threshold
        max_symbols: Maximum number of symbols to process
        
    Returns:
        List of generated screenshot paths
    """
    fix = get_tradingview_async_fix()
    return fix.generate_charts_sync_safe(tjde_results, min_score, max_symbols)

def test_tradingview_fix():
    """Test the async fix with sample data"""
    test_results = [
        {
            'symbol': 'BTCUSDT',
            'tjde_score': 0.75,
            'market_phase': 'trend-following',
            'tjde_decision': 'consider_entry'
        },
        {
            'symbol': 'ETHUSDT',
            'tjde_score': 0.65,
            'market_phase': 'pullback',
            'tjde_decision': 'avoid'
        }
    ]
    
    screenshots = generate_tradingview_charts_safe(test_results, 0.5, 2)
    print(f"[TRADINGVIEW FIX TEST] Generated {len(screenshots)} screenshots")
    return len(screenshots) > 0

if __name__ == "__main__":
    success = test_tradingview_fix()
    print(f"[TRADINGVIEW FIX] Test {'passed' if success else 'failed'}")