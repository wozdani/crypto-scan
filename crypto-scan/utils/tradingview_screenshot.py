"""
TradingView Screenshot Generator - Real Chart Capture for Vision-AI Training
Captures authentic TradingView charts for TOP 5 TJDE tokens eliminating matplotlib artifacts
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import global warning system
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crypto_scan_service import log_warning
except ImportError:
    # Fallback logging if crypto_scan_service not available
    def log_warning(label, exception=None, additional_info=None):
        msg = f"[{label}]"
        if exception:
            msg += f" {str(exception)}"
        if additional_info:
            msg += f" - {additional_info}"
        print(f"⚠️ {msg}")

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    log_warning("PLAYWRIGHT NOT AVAILABLE", None, "Falling back to professional matplotlib charts")

class TradingViewScreenshotGenerator:
    """Generate authentic TradingView screenshots for TOP 5 TJDE tokens"""
    
    def __init__(self):
        self.output_dir = "training_charts"
        self.browser = None
        self.context = None
        self.viewport = {"width": 1920, "height": 1080}
        self.timeout = 15000  # 15 seconds timeout
        
    async def __aenter__(self):
        """Async context manager entry"""
        if not PLAYWRIGHT_AVAILABLE:
            raise Exception("Playwright not available")
            
        try:
            self.playwright = await async_playwright().start()
            
            # Launch browser with optimized settings for chart capture
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-dev-shm-usage',
                    '--no-first-run',
                    '--disable-default-apps',
                    '--disable-extensions',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding'
                ]
            )
            
            # Create context with optimal settings
            self.context = await self.browser.new_context(
                viewport=self.viewport,
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            print("[TRADINGVIEW] Browser context initialized successfully")
            return self
            
        except Exception as e:
            print(f"[TRADINGVIEW ERROR] Failed to initialize browser: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
        except Exception as e:
            print(f"[TRADINGVIEW] Browser cleanup error: {e}")
    
    async def generate_tradingview_screenshot(
        self, 
        symbol: str, 
        tjde_score: float = 0.0,
        market_phase: str = "unknown",
        decision: str = "unknown"
    ) -> Optional[str]:
        """
        Generate TradingView screenshot for single symbol
        
        Args:
            symbol: Trading symbol (e.g., PROMUSDT)
            tjde_score: TJDE score for filename
            market_phase: Market phase for metadata
            decision: TJDE decision for metadata
            
        Returns:
            Path to saved screenshot or None if failed
        """
        try:
            # Prepare TradingView symbol format
            tv_symbol = f"BINANCE:{symbol.upper()}"
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"{symbol}_{timestamp}_tradingview_tjde.png"
            output_path = os.path.join(self.output_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            print(f"[TRADINGVIEW] Capturing {tv_symbol} (TJDE: {tjde_score:.3f})")
            
            # Create new page for this symbol
            page = await self.context.new_page()
            
            try:
                # Navigate to TradingView chart with clean URL
                chart_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}&interval=15"
                
                print(f"[TRADINGVIEW] Loading: {chart_url}")
                await page.goto(chart_url, timeout=self.timeout, wait_until='domcontentloaded')
                
                # Wait for chart to load
                print(f"[TRADINGVIEW] Waiting for chart elements to load...")
                await page.wait_for_timeout(8000)  # 8 seconds for chart loading
                
                # Try to wait for chart canvas
                try:
                    await page.wait_for_selector('canvas', timeout=5000)
                    print(f"[TRADINGVIEW] Chart canvas detected")
                except:
                    print(f"[TRADINGVIEW] Canvas not detected, continuing...")
                
                # Clean up TradingView interface elements
                print(f"[TRADINGVIEW] Cleaning interface...")
                await page.evaluate("""
                    () => {
                        // Hide common TradingView interface elements
                        const elementsToHide = [
                            '[data-name="legend"]',
                            '.chart-gui-wrapper .header-wrapper',
                            '.bottom-widgetbar',
                            '.left-toolbar',
                            '.right-toolbar', 
                            '.top-toolbar',
                            '.header-wrapper',
                            '.widgetbar-widgetbody',
                            '.layout__area--top',
                            '.layout__area--left',
                            '.layout__area--right',
                            '.layout__area--bottom',
                            '.floating-toolbar-react',
                            '.chart-markup-table',
                            '.pane-legend',
                            '.chart-widget',
                            '.alerts-wrapper'
                        ];
                        
                        elementsToHide.forEach(selector => {
                            document.querySelectorAll(selector).forEach(el => {
                                if (el) el.style.display = 'none';
                            });
                        });
                        
                        // Hide any popup/modal elements
                        document.querySelectorAll('[class*="modal"], [class*="popup"], [class*="dialog"]').forEach(el => {
                            el.style.display = 'none';
                        });
                        
                        // Maximize chart area
                        const chartContainer = document.querySelector('.chart-container, .tradingview-chart, [class*="chart"]');
                        if (chartContainer) {
                            chartContainer.style.position = 'fixed';
                            chartContainer.style.top = '0';
                            chartContainer.style.left = '0';
                            chartContainer.style.width = '100vw';
                            chartContainer.style.height = '100vh';
                            chartContainer.style.zIndex = '9999';
                        }
                    }
                """)
                
                # Wait after cleanup
                await page.wait_for_timeout(2000)
                
                # Set 15-minute interval if not already set
                try:
                    await page.evaluate("""
                        () => {
                            // Try to set 15m interval
                            const intervalButtons = document.querySelectorAll('[data-value="15"], [data-interval="15"]');
                            if (intervalButtons.length > 0) {
                                intervalButtons[0].click();
                            }
                        }
                    """)
                    await page.wait_for_timeout(1000)
                except:
                    pass
                
                # Scroll to latest candles (right edge)
                try:
                    await page.evaluate("""
                        () => {
                            // Scroll chart to the right edge (latest candles)
                            const chart = document.querySelector('canvas, .chart-container');
                            if (chart) {
                                chart.scrollLeft = chart.scrollWidth;
                            }
                            
                            // Try keyboard shortcut to go to latest
                            document.dispatchEvent(new KeyboardEvent('keydown', {key: 'End'}));
                        }
                    """)
                    await page.wait_for_timeout(1000)
                except:
                    pass
                
                # Take screenshot
                await page.screenshot(
                    path=output_path,
                    full_page=False,
                    quality=95,
                    type='png'
                )
                
                # Verify file was created and has reasonable size
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    if file_size > 10000:  # At least 10KB
                        print(f"[TRADINGVIEW] ✅ Screenshot saved: {output_path} ({file_size} bytes)")
                        
                        # Generate metadata JSON
                        await self._save_screenshot_metadata(
                            output_path, symbol, tjde_score, market_phase, decision
                        )
                        
                        return output_path
                    else:
                        print(f"[TRADINGVIEW] ❌ Screenshot too small: {file_size} bytes")
                        return None
                else:
                    print(f"[TRADINGVIEW] ❌ Screenshot file not created")
                    return None
                    
            except Exception as e:
                print(f"[TRADINGVIEW ERROR] {symbol}: {e}")
                return None
            finally:
                await page.close()
                
        except Exception as e:
            print(f"[TRADINGVIEW ERROR] Failed to capture {symbol}: {e}")
            return None
    
    async def _save_screenshot_metadata(
        self, 
        image_path: str, 
        symbol: str, 
        tjde_score: float,
        market_phase: str,
        decision: str
    ):
        """Save metadata for TradingView screenshot"""
        try:
            metadata = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "chart_type": "tradingview_screenshot",
                "source": "authentic_tradingview",
                "phase": market_phase,
                "decision": decision,
                "tjde_score": tjde_score,
                "interval": "15m",
                "viewport": self.viewport,
                "authentic_data": True,
                "vision_ai_ready": True,
                "created_at": datetime.now().isoformat()
            }
            
            json_path = image_path.replace('.png', '.json')
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Metadata saved successfully
            
        except Exception as e:
            log_warning("TRADINGVIEW METADATA SAVE ERROR", e, f"Failed to save metadata to {json_path}")

async def generate_tradingview_screenshots_for_top_tjde(
    tjde_results: List[Dict], 
    min_score: float = 0.5,
    max_symbols: int = 5
) -> List[str]:
    """
    Generate TradingView screenshots for TOP 5 TJDE tokens
    
    Args:
        tjde_results: List of TJDE analysis results
        min_score: Minimum TJDE score threshold
        max_symbols: Maximum number of symbols to process
        
    Returns:
        List of generated screenshot paths
    """
    if not PLAYWRIGHT_AVAILABLE:
        log_warning("PLAYWRIGHT NOT AVAILABLE", None, "Skipping TradingView screenshots")
        return []
    
    try:
        # Filter and sort by TJDE score
        filtered_results = [
            result for result in tjde_results 
            if result.get('tjde_score', 0) >= min_score
        ]
        
        # Sort by TJDE score (highest first)
        top_results = sorted(
            filtered_results, 
            key=lambda x: x.get('tjde_score', 0), 
            reverse=True
        )[:max_symbols]
        
        if not top_results:
            print(f"[TRADINGVIEW] No tokens meet minimum TJDE score {min_score}")
            return []
        
        print(f"[TRADINGVIEW] Generating screenshots for TOP {len(top_results)} TJDE tokens:")
        for i, result in enumerate(top_results, 1):
            symbol = result.get('symbol', 'UNKNOWN')
            score = result.get('tjde_score', 0)
            print(f"  {i}. {symbol}: TJDE {score:.3f}")
        
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
                
                if screenshot_path:
                    screenshot_paths.append(screenshot_path)
                
                # Small delay between captures
                await asyncio.sleep(1)
        
        # Success - return screenshot paths
        return screenshot_paths
        
    except Exception as e:
        log_warning("TRADINGVIEW BULK GENERATION ERROR", e, "Bulk generation failed")
        return []

def sync_generate_tradingview_screenshots(
    tjde_results: List[Dict], 
    min_score: float = 0.5,
    max_symbols: int = 5
) -> List[str]:
    """
    Synchronous wrapper for TradingView screenshot generation
    
    Args:
        tjde_results: List of TJDE analysis results
        min_score: Minimum TJDE score threshold  
        max_symbols: Maximum number of symbols to process
        
    Returns:
        List of generated screenshot paths
    """
    try:
        # Run async function in event loop
        return asyncio.run(
            generate_tradingview_screenshots_for_top_tjde(
                tjde_results, min_score, max_symbols
            )
        )
    except Exception as e:
        log_warning("TRADINGVIEW SYNC GENERATION ERROR", e, "Generation failed")
        return []

# Test function
async def test_tradingview_screenshot():
    """Test TradingView screenshot generation"""
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
    
    screenshots = await generate_tradingview_screenshots_for_top_tjde(test_results, 0.5, 2)
    
    # Return test success status
    return len(screenshots) > 0

if __name__ == "__main__":
    # Test the TradingView screenshot system
    
    if PLAYWRIGHT_AVAILABLE:
        success = asyncio.run(test_tradingview_screenshot())
        # Test result handled internally
    else:
        log_warning("PLAYWRIGHT NOT AVAILABLE", None, "Cannot test TradingView screenshots")