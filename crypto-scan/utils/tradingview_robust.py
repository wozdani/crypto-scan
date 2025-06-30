#!/usr/bin/env python3
"""
Robust TradingView Screenshot System
Enhanced reliability with timeout management and fallback strategies
"""

import asyncio
import os
from datetime import datetime
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright, Browser, Page
from .multi_exchange_resolver import get_multi_exchange_resolver

class RobustTradingViewGenerator:
    """Enhanced TradingView screenshot generator with robust error handling"""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        
    async def __aenter__(self):
        """Initialize browser with enhanced reliability"""
        try:
            print("[ROBUST TV] Initializing enhanced browser context...")
            
            # Initialize Playwright
            self.playwright = await async_playwright().start()
            
            # Enhanced browser args for stability
            browser_args = [
                '--headless=new',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions',
                '--no-first-run',
                '--disable-default-apps',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=TranslateUI',
                '--disable-gpu-sandbox',
                '--memory-pressure-off'
            ]
            
            # Try system Chromium first
            chromium_path = '/nix/store/zi4f80l169xlmivz8vja8wlphq74qqk0-chromium-125.0.6422.141/bin/chromium'
            
            if os.path.exists(chromium_path):
                print(f"[ROBUST TV] Using system Chromium: {chromium_path}")
                self.browser = await self.playwright.chromium.launch(
                    executable_path=chromium_path,
                    headless=True,
                    args=browser_args
                )
            else:
                print("[ROBUST TV] Using default Playwright Chromium")
                self.browser = await self.playwright.chromium.launch(
                    headless=True,
                    args=browser_args
                )
            
            # Create context with minimal settings
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            # Create page
            self.page = await context.new_page()
            
            # Set reasonable timeouts
            self.page.set_default_timeout(30000)  # 30 seconds max
            self.page.set_default_navigation_timeout(20000)  # 20 seconds for navigation
            
            print("[ROBUST TV] Browser context ready")
            return self
            
        except Exception as e:
            print(f"[ROBUST TV ERROR] Browser initialization failed: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean shutdown"""
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            print(f"[ROBUST TV] Cleanup error: {e}")
    
    def _resolve_symbol(self, symbol: str) -> tuple[str, str]:
        """Resolve symbol to TradingView format with exchange info"""
        try:
            # Try multi-exchange resolver first
            resolver = get_multi_exchange_resolver()
            if resolver:
                result = resolver.resolve_tradingview_symbol(symbol)
                if result:
                    tv_symbol, exchange = result
                    return tv_symbol, exchange
            
            # Fallback to BINANCE
            return f"BINANCE:{symbol}", "BINANCE"
            
        except Exception as e:
            print(f"[ROBUST TV] Symbol resolution error for {symbol}: {e}")
            return f"BINANCE:{symbol}", "BINANCE"
    
    async def generate_screenshot(
        self, 
        symbol: str, 
        tjde_score: float = 0.0,
        decision: str = "unknown"
    ) -> Optional[str]:
        """Generate TradingView screenshot with enhanced reliability"""
        
        # 🔒 CRITICAL: TOP5 HARD BLOCK - Prevent training data generation outside TOP 5
        try:
            from .top5_selector import should_generate_training_data
            if not should_generate_training_data(symbol, tjde_score):
                print(f"🚫 [TOP5 BLOCK] {symbol}: TradingView screenshot BLOCKED - not in TOP 5 (TJDE: {tjde_score:.3f})")
                return None
        except ImportError:
            print(f"🚫 [TOP5 BLOCK] {symbol}: TOP5 selector unavailable - blocking screenshot to maintain dataset quality")
            return None
        except Exception as e:
            print(f"🚫 [TOP5 BLOCK] {symbol}: TOP5 check failed ({e}) - blocking screenshot")
            return None
        
        try:
            # Resolve symbol
            tv_symbol, exchange = self._resolve_symbol(symbol)
            print(f"[ROBUST TV] {symbol} → {tv_symbol} ({exchange})")
            
            # Generate filename with timestamp for freshness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            score_formatted = str(int(tjde_score * 1000)) if tjde_score > 0 else "000"
            filename = f"{symbol}_{exchange}_score-{score_formatted}_{timestamp}.png"
            
            # Ensure output directory
            os.makedirs("training_data/charts", exist_ok=True)
            file_path = f"training_data/charts/{filename}"
            
            # Load TradingView URL
            url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}&interval=15"
            print(f"[ROBUST TV] Loading: {url}")
            
            # Navigate with timeout
            try:
                await self.page.goto(url, wait_until='domcontentloaded', timeout=15000)
                print("[ROBUST TV] Page loaded successfully")
            except Exception as nav_error:
                print(f"[ROBUST TV ERROR] Navigation failed: {nav_error}")
                # Try brute-force BINANCE fallback before giving up
                return await self._try_brute_force_fallback(symbol, tjde_score, decision)
            
            # Wait for chart with progressive timeouts
            chart_ready = await self._wait_for_chart_progressive()
            if not chart_ready:
                print("[ROBUST TV ERROR] Chart failed to load")
                return None
            
            # Check for "Invalid symbol" error before taking screenshot
            try:
                # Use more specific selectors to avoid strict mode violations
                invalid_symbol = await self.page.locator(".errorCard__message-S9sXvhAu").is_visible()
                if not invalid_symbol:
                    # Fallback to other invalid symbol indicators
                    invalid_symbol = await self.page.locator("span.invalid-lu2ARROZ").is_visible()
                
                if invalid_symbol:
                    print(f"[ROBUST TV ERROR] Invalid symbol detected for {tv_symbol}")
                    # Try brute-force BINANCE fallback before returning INVALID_SYMBOL
                    return await self._try_brute_force_fallback(symbol, tjde_score, decision)
                    
                # Also check for other error indicators
                symbol_not_found = await self.page.locator("text=Symbol not found").first.is_visible()
                if symbol_not_found:
                    print(f"[ROBUST TV ERROR] Symbol not found: {tv_symbol}")
                    # Try brute-force BINANCE fallback before giving up
                    return await self._try_brute_force_fallback(symbol, tjde_score, decision)
                    
            except Exception as validation_error:
                print(f"[ROBUST TV WARNING] Symbol validation failed: {validation_error}")
                # Continue with screenshot anyway if validation fails
            
            # Take screenshot
            try:
                screenshot_data = await self.page.screenshot(
                    path=file_path,
                    full_page=False
                )
                
                # Verify screenshot quality
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > 10000:  # Minimum 10KB
                        print(f"[ROBUST TV SUCCESS] Screenshot saved: {filename} ({file_size} bytes)")
                        
                        # Save metadata
                        await self._save_metadata(file_path, symbol, tv_symbol, exchange, tjde_score, decision)
                        
                        return file_path
                    else:
                        print(f"[ROBUST TV ERROR] Screenshot too small: {file_size} bytes")
                        os.remove(file_path)
                        return None
                else:
                    print("[ROBUST TV ERROR] Screenshot file not created")
                    return None
                    
            except Exception as screenshot_error:
                print(f"[ROBUST TV ERROR] Screenshot failed: {screenshot_error}")
                return None
                
        except Exception as e:
            print(f"[ROBUST TV ERROR] Generation failed for {symbol}: {e}")
            return None
    
    async def _wait_for_chart_progressive(self) -> bool:
        """Progressive chart loading with multiple fallback strategies"""
        
        try:
            # Strategy 1: Wait for canvas with short timeout
            print("[ROBUST TV] Strategy 1: Waiting for canvas...")
            try:
                await self.page.wait_for_selector("canvas", timeout=8000)
                print("[ROBUST TV] Canvas detected")
                
                # Short wait for data
                await asyncio.sleep(3)
                return True
                
            except Exception:
                print("[ROBUST TV] Canvas timeout - trying strategy 2")
            
            # Strategy 2: Wait for chart container
            print("[ROBUST TV] Strategy 2: Waiting for chart container...")
            try:
                await self.page.wait_for_selector(".chart-container", timeout=5000)
                print("[ROBUST TV] Chart container detected")
                
                await asyncio.sleep(2)
                return True
                
            except Exception:
                print("[ROBUST TV] Container timeout - trying strategy 3")
            
            # Strategy 3: Minimal wait and proceed
            print("[ROBUST TV] Strategy 3: Minimal wait and proceed...")
            await asyncio.sleep(3)
            return True
            
        except Exception as e:
            print(f"[ROBUST TV ERROR] All chart loading strategies failed: {e}")
            return False
    
    async def _save_metadata(
        self, 
        image_path: str, 
        symbol: str, 
        tv_symbol: str, 
        exchange: str,
        tjde_score: float, 
        decision: str
    ):
        """Save enhanced metadata"""
        try:
            metadata = {
                "symbol": symbol,
                "tradingview_symbol": tv_symbol,
                "exchange": exchange,
                "tjde_score": tjde_score,
                "decision": decision,
                "timestamp": datetime.now().isoformat(),
                "generator": "robust_tradingview",
                "authentic_data": True,
                "multi_exchange_resolver": True
            }
            
            metadata_path = image_path.replace('.png', '_metadata.json')
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"[ROBUST TV] Metadata saved: {os.path.basename(metadata_path)}")
            
        except Exception as e:
            print(f"[ROBUST TV ERROR] Metadata save failed: {e}")
    
    async def _try_brute_force_fallback(self, symbol: str, tjde_score: float, decision: str) -> Optional[str]:
        """
        Last resort: Try brute-force BINANCE fallback without validation
        """
        # 🔒 CRITICAL: TOP5 HARD BLOCK even in fallback
        try:
            from .top5_selector import should_generate_training_data
            if not should_generate_training_data(symbol, tjde_score):
                print(f"🚫 [TOP5 FALLBACK BLOCK] {symbol}: BINANCE fallback BLOCKED - not in TOP 5 (TJDE: {tjde_score:.3f})")
                return None
        except Exception:
            print(f"🚫 [TOP5 FALLBACK BLOCK] {symbol}: TOP5 check failed - blocking fallback screenshot")
            return None
            
        try:
            print(f"[ROBUST TV] 🚨 BRUTE-FORCE FALLBACK: Trying BINANCE:{symbol}")
            
            # Generate filename for fallback
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            score_formatted = str(int(tjde_score * 1000)) if tjde_score > 0 else "000"
            filename = f"{symbol}_BINANCE_score-{score_formatted}_{timestamp}.png"
            
            # Ensure output directory
            os.makedirs("training_data/charts", exist_ok=True)
            file_path = f"training_data/charts/{filename}"
            
            # Try brute-force BINANCE URL
            fallback_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}&interval=15"
            print(f"[ROBUST TV] Loading fallback URL: {fallback_url}")
            
            try:
                await self.page.goto(fallback_url, wait_until='domcontentloaded', timeout=15000)
                print("[ROBUST TV] ✅ Brute-force navigation successful!")
                
                # Quick chart loading check
                chart_ready = await self._wait_for_chart_progressive()
                if not chart_ready:
                    print("[ROBUST TV] Brute-force chart loading failed")
                    return None
                
                # Take screenshot immediately (no validation for fallback)
                screenshot_data = await self.page.screenshot(
                    path=file_path,
                    full_page=False
                )
                
                # Verify screenshot quality
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > 10000:  # Minimum 10KB
                        print(f"[ROBUST TV SUCCESS] ✅ Brute-force success: {filename} ({file_size} bytes)")
                        
                        # Save metadata
                        await self._save_metadata(file_path, symbol, f"BINANCE:{symbol}", "BINANCE", tjde_score, decision)
                        
                        return file_path
                    else:
                        print(f"[ROBUST TV] Brute-force screenshot too small: {file_size} bytes")
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        return None
                        
            except Exception as fallback_nav_error:
                print(f"[ROBUST TV] ❌ Brute-force navigation failed: {fallback_nav_error}")
                return None
                
        except Exception as e:
            print(f"[ROBUST TV ERROR] Brute-force fallback failed: {e}")
            return None

async def test_robust_generator():
    """Quick test of robust generator"""
    async with RobustTradingViewGenerator() as generator:
        result = await generator.generate_screenshot("BTCUSDT", 0.726, "test")
        if result:
            print(f"✅ Test successful: {result}")
        else:
            print("❌ Test failed")

if __name__ == "__main__":
    asyncio.run(test_robust_generator())