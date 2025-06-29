"""
TradingView Synchronous Wrapper - Fixes Event Loop Conflicts
Resolves 'Cannot run the event loop while another loop is running' and async context manager issues
"""

import asyncio
import subprocess
import sys
import json
import os
from typing import Optional, Dict, List
import logging
from datetime import datetime

class TradingViewSyncWrapper:
    """Synchronous wrapper for TradingView screenshot generation to avoid event loop conflicts"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def generate_tradingview_screenshot(
        self, 
        symbol: str, 
        phase: str = "unknown", 
        setup: str = "unknown", 
        score: float = 0.0,
        output_dir: str = "training_data/charts"
    ) -> Optional[str]:
        """
        Generate TradingView screenshot using subprocess to avoid event loop conflicts
        
        Args:
            symbol: Trading symbol
            phase: Market phase
            setup: Setup type
            score: TJDE score
            output_dir: Output directory
            
        Returns:
            Path to generated screenshot or None if failed
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Check symbol validity for TradingView first
            if not self._is_valid_tradingview_symbol(symbol):
                self.logger.warning(f"[TV SYMBOL] {symbol} → Invalid for TradingView, skipping")
                return None
            
            # Format filename with proper structure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{phase}-{setup}_score-{int(score*1000)}_{timestamp}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Use subprocess to run async TradingView generation
            result = self._run_tradingview_subprocess(symbol, output_path)
            
            if result and os.path.exists(output_path):
                self.logger.info(f"[TV SUCCESS] {symbol} → {filename}")
                return output_path
            else:
                self.logger.warning(f"[TV FAILED] {symbol} → No screenshot generated")
                return None
                
        except Exception as e:
            self.logger.error(f"[TV ERROR] {symbol} → {e}")
            return None
    
    def _is_valid_tradingview_symbol(self, symbol: str) -> bool:
        """
        Check if symbol is valid for TradingView
        Uses cached validation to avoid repeated checks
        """
        try:
            # Load cache
            cache_file = "data/tv_symbol_cache.json"
            cache = {}
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            
            # Check cache first
            if symbol in cache:
                return cache[symbol]
            
            # For now, use simple validation - can be enhanced with actual TradingView API check
            # Most BYBIT symbols work if they exist on Bybit
            is_valid = len(symbol) > 3 and symbol.endswith('USDT')
            
            # Cache result
            cache[symbol] = is_valid
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"[TV VALIDATION] {symbol} → {e}")
            return False
    
    def _run_tradingview_subprocess(self, symbol: str, output_path: str) -> bool:
        """
        Run TradingView screenshot generation in subprocess to avoid event loop conflicts
        """
        try:
            # Create a simple script for subprocess execution
            script_content = f'''
import asyncio
from playwright.async_api import async_playwright

async def capture_tradingview_screenshot():
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Navigate to TradingView using intelligent symbol mapping
            from .tradingview_symbol_mapper import map_to_tradingview
            tv_symbol = map_to_tradingview(symbol) or f"BINANCE:{symbol}"
            url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Wait for chart to load
            await page.wait_for_selector("canvas", timeout=20000)
            await page.wait_for_timeout(5000)  # Additional wait for chart rendering
            
            # Take screenshot
            await page.screenshot(path="{output_path}", full_page=False, quality=95)
            
            await browser.close()
            return True
            
        except Exception as e:
            print(f"Screenshot error: {{e}}")
            return False

if __name__ == "__main__":
    result = asyncio.run(capture_tradingview_screenshot())
    exit(0 if result else 1)
'''
            
            # Write temporary script
            temp_script = f"temp_tv_script_{symbol}.py"
            with open(temp_script, 'w') as f:
                f.write(script_content)
            
            # Run subprocess
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Cleanup
            if os.path.exists(temp_script):
                os.remove(temp_script)
            
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"[TV SUBPROCESS] {symbol} → {e}")
            return False
    
    def generate_multiple_screenshots(self, tokens: List[Dict]) -> List[str]:
        """
        Generate screenshots for multiple tokens
        
        Args:
            tokens: List of token dictionaries with symbol, phase, setup, score
            
        Returns:
            List of generated screenshot paths
        """
        generated_paths = []
        
        for token_info in tokens:
            symbol = token_info.get('symbol', '')
            phase = token_info.get('phase', 'unknown')
            setup = token_info.get('setup', 'unknown') 
            score = token_info.get('score', 0.0)
            
            screenshot_path = self.generate_tradingview_screenshot(
                symbol, phase, setup, score
            )
            
            if screenshot_path:
                generated_paths.append(screenshot_path)
        
        return generated_paths

# Global instance
_tv_wrapper = None

def get_tradingview_wrapper() -> TradingViewSyncWrapper:
    """Get global TradingView wrapper instance"""
    global _tv_wrapper
    if _tv_wrapper is None:
        _tv_wrapper = TradingViewSyncWrapper()
    return _tv_wrapper

def generate_tradingview_screenshot_sync(
    symbol: str, 
    phase: str = "unknown", 
    setup: str = "unknown", 
    score: float = 0.0
) -> Optional[str]:
    """
    Convenience function for sync TradingView screenshot generation
    Fixes event loop conflicts and async context manager issues
    """
    wrapper = get_tradingview_wrapper()
    return wrapper.generate_tradingview_screenshot(symbol, phase, setup, score)

if __name__ == "__main__":
    # Test the wrapper
    wrapper = TradingViewSyncWrapper()
    result = wrapper.generate_tradingview_screenshot("BTCUSDT", "trend-following", "breakout", 0.75)
    print(f"Test result: {result}")