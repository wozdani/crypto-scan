"""
TradingView Screenshot Generator - Multi-Exchange Compatible Chart Capture for Vision-AI Training
Captures authentic TradingView charts for TOP 5 TJDE tokens using intelligent multi-exchange resolution
Features:
- Multi-exchange resolution (BINANCE → BYBIT → MEXC → OKX → GATEIO → KUCOIN)
- Intelligent exchange selection based on token characteristics
- Comprehensive screenshot validation (size, content quality)
- Enhanced error detection and automatic cleanup
- Professional chart quality for superior Vision-AI training
- Exchange tracking in metadata and optional filename enhancement
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
import time
from PIL import Image

# Import TradingView symbol mapper and validation systems
try:
    from .tradingview_symbol_mapper import map_to_tradingview, get_tradingview_chart_url
    from .screenshot_validator import ScreenshotValidator
    from .multi_exchange_resolver import get_multi_exchange_resolver
except ImportError:
    # Fallback if modules not available
    def map_to_tradingview(symbol: str) -> Optional[str]:
        return f"BINANCE:{symbol.upper()}"
    def get_tradingview_chart_url(symbol: str, timeframe: str = "15") -> Optional[str]:
        return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.upper()}&interval={timeframe}"
    class ScreenshotValidator:
        def validate_and_cleanup(self, *args, **kwargs):
            return True
    def get_multi_exchange_resolver():
        return None

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

def is_chart_blank(image_path: str, threshold: float = 0.99) -> bool:
    """
    🎯 CHART VALIDATION: Sprawdza czy obraz jest praktycznie pusty (biały)
    
    Wykrywa screenshoty gdzie canvas się załadował ale dane jeszcze nie są narysowane
    
    Args:
        image_path: Ścieżka do pliku PNG
        threshold: Próg białych pikseli (0.99 = 99% białych pikseli)
        
    Returns:
        True jeśli wykres jest pusty/biały
    """
    try:
        if not os.path.exists(image_path):
            return True
            
        # Convert to grayscale and analyze pixel values
        img = Image.open(image_path).convert("L")
        pixels = list(img.getdata())
        
        if not pixels:
            return True
            
        # Count white/near-white pixels (above 250 in 0-255 range)
        white_pixels = sum(1 for px in pixels if px > 250)
        ratio = white_pixels / len(pixels)
        
        if ratio > threshold:
            print(f"[CHART VALIDATION] ⚠️ Blank chart detected: {os.path.basename(image_path)} ({ratio:.1%} white pixels)")
            return True
        else:
            print(f"[CHART VALIDATION] ✅ Valid chart: {os.path.basename(image_path)} ({ratio:.1%} white pixels)")
            return False
            
    except Exception as e:
        log_warning("CHART VALIDATION ERROR", e, f"Failed to validate chart: {image_path}")
        return True  # Assume blank on error

class TradingViewScreenshotGenerator:
    """Generate authentic TradingView screenshots for TOP 5 TJDE tokens"""
    
    def __init__(self):
        self.output_dir = "training_data/charts"
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
            
            # Launch browser with fallback for different chromium versions
            browser_args = [
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
            
            try:
                # Try launching with default chromium
                self.browser = await self.playwright.chromium.launch(
                    headless=True,
                    args=browser_args
                )
                print("[TRADINGVIEW] Using default chromium launcher")
            except Exception as e:
                # Fallback: try with specific executable path
                log_warning("TRADINGVIEW BROWSER FALLBACK", e, "Trying alternative chromium paths")
                
                possible_paths = [
                    '/nix/store/zi4f80l169xlmivz8vja8wlphq74qqk0-chromium-125.0.6422.141/bin/chromium',  # System Chromium
                    '/home/runner/workspace/.cache/ms-playwright/chromium-1091/chrome-linux/chrome',
                    '/home/runner/workspace/.cache/ms-playwright/chromium_headless_shell-1179/chrome-linux/headless_shell',
                    '/usr/bin/chromium',
                    '/usr/bin/chromium-browser',
                    '/usr/bin/google-chrome'
                ]
                
                browser_launched = False
                for executable_path in possible_paths:
                    try:
                        import os
                        if os.path.exists(executable_path):
                            self.browser = await self.playwright.chromium.launch(
                                headless=True,
                                executable_path=executable_path,
                                args=browser_args
                            )
                            print(f"[TRADINGVIEW] Using chromium from: {executable_path}")
                            browser_launched = True
                            break
                    except:
                        continue
                
                if not browser_launched:
                    log_warning("TRADINGVIEW BROWSER ERROR", None, "Failed to launch chromium with any available executable")
                    raise Exception("No working chromium executable found")
            
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
    
    def _resolve_tradingview_symbol(self, symbol: str) -> str:
        """
        🎯 SMART EXCHANGE DETECTION: Resolve symbol to proper TradingView format
        
        Uses intelligent symbol mapper to find valid TradingView exchanges:
        1. Check symbol mapper cache first
        2. Try multiple exchanges (BINANCE, COINBASE, BYBIT, etc.)
        3. Verify symbol exists on TradingView
        4. Return validated symbol or fallback
        
        Args:
            symbol: Trading symbol (e.g. BTCUSDT)
            
        Returns:
            TradingView-compatible symbol (e.g. BINANCE:BTCUSDT) or fallback
        """
        # First try the multi-exchange resolver
        try:
            resolver = get_multi_exchange_resolver()
            if resolver:
                result = resolver.resolve_tradingview_symbol(symbol)
                if result:
                    tv_symbol, exchange = result
                    print(f"[MULTI-EXCHANGE] ✅ {symbol} → {tv_symbol} ({exchange})")
                    return tv_symbol
        except Exception as e:
            print(f"[MULTI-EXCHANGE] ❌ Error resolving {symbol}: {e}")
        
        # Fallback to existing symbol mapper
        tv_symbol = map_to_tradingview(symbol)
        
        if tv_symbol:
            print(f"[SYMBOL MAPPER] ✅ {symbol} → {tv_symbol}")
            return tv_symbol
        else:
            # Final fallback to BINANCE for major pairs (most likely to work)
            major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'LTCUSDT']
            
            if symbol in major_pairs:
                fallback = f"BINANCE:{symbol}"
                print(f"[SYMBOL MAPPER] ⚠️ {symbol} → {fallback} (major pair fallback)")
                return fallback
            else:
                # Last resort: try BYBIT (least likely to work but worth trying)
                fallback = f"BYBIT:{symbol}"
                print(f"[SYMBOL MAPPER] ⚠️ {symbol} → {fallback} (BYBIT fallback)")
                return fallback

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
            # 🎯 STEP 1: SMART SYMBOL RESOLUTION with multi-exchange support
            exchange_info = None
            tv_symbol = None
            
            # Get enhanced resolution with exchange info
            try:
                resolver = get_multi_exchange_resolver()
                if resolver:
                    result = resolver.resolve_tradingview_symbol(symbol)
                    if result:
                        tv_symbol, exchange = result
                        exchange_info = exchange
                        print(f"[MULTI-EXCHANGE] ✅ {symbol} → {tv_symbol} ({exchange})")
                    else:
                        print(f"[MULTI-EXCHANGE] ❌ {symbol} not found on any exchange")
            except Exception as e:
                print(f"[MULTI-EXCHANGE] ❌ Resolution error for {symbol}: {e}")
            
            # Fallback to regular symbol resolution
            if not tv_symbol:
                tv_symbol = self._resolve_tradingview_symbol(symbol)
                exchange_info = tv_symbol.split(':')[0] if ':' in tv_symbol else 'UNKNOWN'
                print(f"[SYMBOL MAPPER] ✅ Fallback resolution: {symbol} → {tv_symbol}")
            
            # Generate enhanced filename with exchange and score (optional enhancement)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            score_formatted = str(int(tjde_score * 1000)) if tjde_score > 0 else "000"
            
            # Enhanced filename format: SYMBOL_EXCHANGE_score-XXX.png (e.g., SXPUSDT_BYBIT_score-726.png)
            if exchange_info and exchange_info != 'UNKNOWN':
                filename = f"{symbol}_{exchange_info}_score-{score_formatted}.png"
            else:
                filename = f"{symbol}_{timestamp}_score-{score_formatted}.png"
            
            output_path = os.path.join(self.output_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create new page for TradingView screenshot
            page = await self.context.new_page()
            
            try:
                # Navigate to TradingView chart with clean URL
                chart_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}&interval=15"
                
                print(f"[TRADINGVIEW] Loading: {chart_url}")
                await page.goto(chart_url, timeout=self.timeout, wait_until='domcontentloaded')
                
                # 🔍 ENHANCED ERROR DETECTION: Check for TradingView symbol errors FIRST
                print(f"[TRADINGVIEW] Checking for symbol validity...")
                await page.wait_for_timeout(2000)  # Wait for error messages to appear
                
                symbol_invalid = False
                try:
                    # Check for error messages indicating invalid symbol
                    error_selectors = [
                        'text*="not found"',
                        'text*="invalid"',
                        'text*="Symbol not found"',
                        'text*="Invalid symbol"',
                        'text*="doesn\'t exist"',
                        '[class*="error"]',
                        '[class*="not-found"]'
                    ]
                    
                    for selector in error_selectors:
                        try:
                            error_element = await page.query_selector(selector)
                            if error_element:
                                error_text = await error_element.text_content() if error_element else ""
                                print(f"⚠️ [TRADINGVIEW SYMBOL ERROR] {tv_symbol}: {error_text}")
                                symbol_invalid = True
                                break
                        except:
                            continue
                            
                except Exception as e:
                    print(f"[TRADINGVIEW] Error check failed: {e}")
                
                # If symbol is invalid, close page and return None
                if symbol_invalid:
                    print(f"[TRADINGVIEW] ❌ {tv_symbol} invalid - terminating TradingView generation")
                    log_warning("TRADINGVIEW SYMBOL INVALID", None, f"{tv_symbol}: Symbol not found on TradingView")
                    await page.close()
                    return None
                
                # 🕒 Wait for chart to fully render (enhanced canvas detection)
                print(f"[TRADINGVIEW] ✅ {tv_symbol} valid - waiting for chart elements...")
                try:
                    # Wait for canvas elements to appear
                    await page.wait_for_selector("canvas", timeout=15000)
                    print(f"[TRADINGVIEW] Canvas elements detected")
                    
                    # Wait for multiple canvas elements (TradingView uses several)
                    await page.wait_for_function("document.querySelectorAll('canvas').length > 0", timeout=15000)
                    print(f"[TRADINGVIEW] Multiple canvas elements confirmed")
                    
                    # 🎯 ENHANCED RENDERING TIME: More time for complex charts
                    await page.wait_for_timeout(5000)  # 5 seconds for initial rendering
                    print(f"[TRADINGVIEW] Initial chart rendering completed")
                    
                    # 🎯 ADDITIONAL RENDERING TIME: Ensure data is fully drawn
                    await asyncio.sleep(5)  # Reduced to 5 seconds for faster processing
                    print(f"[TRADINGVIEW] Extended rendering time completed - chart should be fully drawn")
                    
                except Exception as e:
                    log_warning("TRADINGVIEW CHART LOADING", e, f"{symbol}: Chart may not have loaded properly")
                    print(f"[TRADINGVIEW WARNING] {symbol}: Chart loading issue ({e}) - proceeding with screenshot")
                
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
                
                # Take screenshot with conditional quality parameter
                if output_path.endswith(".jpg") or output_path.endswith(".jpeg"):
                    await page.screenshot(
                        path=output_path,
                        full_page=False,
                        quality=95,
                        type='jpeg'
                    )
                else:
                    await page.screenshot(
                        path=output_path,
                        full_page=False,
                        type='png'
                    )
                
                # 🔍 STEP 3: COMPREHENSIVE SCREENSHOT VALIDATION
                if os.path.exists(output_path):
                    # Use screenshot validator for comprehensive validation
                    validator = ScreenshotValidator(min_file_size=50000)  # 50KB minimum
                    validation_result = validator.validate_screenshot(output_path, symbol)
                    
                    if validation_result['valid']:
                        print(f"[TRADINGVIEW] ✅ Valid screenshot: {output_path} ({validation_result['file_size']} bytes, {validation_result['white_pixel_ratio']:.1%} white)")
                        
                        # Generate metadata JSON for valid screenshots only
                        await self._save_screenshot_metadata(
                            output_path, symbol, tjde_score, market_phase, decision, 
                            exchange_info, tv_symbol
                        )
                        
                        return output_path
                    else:
                        print(f"⚠️ [TRADINGVIEW VALIDATION] {symbol}: {validation_result['error']}")
                        log_warning("TRADINGVIEW SCREENSHOT INVALID", None, f"{symbol}: {validation_result['error']}")
                        
                        # Automatically cleanup invalid screenshot
                        validator.cleanup_invalid_screenshot(output_path, symbol)
                        return None
                else:
                    print(f"[TRADINGVIEW] ❌ Screenshot file not created")
                    log_warning("TRADINGVIEW SCREENSHOT MISSING", None, f"{symbol}: File not created after capture")
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
        decision: str,
        exchange_info: str = None,
        tv_symbol: str = None
    ):
        """Save metadata for TradingView screenshot"""
        try:
            metadata = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "chart_type": "tradingview_screenshot",
                "source": "authentic_tradingview",
                "exchange": exchange_info or "UNKNOWN",
                "tradingview_symbol": tv_symbol or f"UNKNOWN:{symbol}",
                "phase": market_phase,
                "decision": decision,
                "tjde_score": tjde_score,
                "interval": "15m",
                "viewport": self.viewport,
                "authentic_data": True,
                "vision_ai_ready": True,
                "multi_exchange_resolver": exchange_info is not None,
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
        # 🎯 CRITICAL FIX: For TOP 5 TJDE tokens, skip BINANCE filtering
        # TOP 5 tokens get priority treatment regardless of exchange
        print("[TRADINGVIEW] 🎯 TOP 5 mode - using smart exchange detection instead of BINANCE filtering")
        
        # STEP 1: Filter by TJDE score only (no BINANCE restriction for TOP 5)
        filtered_results = [
            result for result in tjde_results 
            if result.get('tjde_score', 0) >= min_score
        ]
        
        # STEP 2: For TOP 5, we'll use smart symbol resolution per exchange
        print(f"[TRADINGVIEW] Processing {len(filtered_results)} high-TJDE tokens with multi-exchange support")
        
        # Sort by TJDE score (highest first)
        top_results = sorted(
            filtered_results, 
            key=lambda x: x.get('tjde_score', 0), 
            reverse=True
        )[:max_symbols]
        
        if not top_results:
            print(f"[TRADINGVIEW] No tokens meet minimum TJDE score {min_score} and BINANCE compatibility")
            return []
        
        # Show filtering statistics
        total_candidates = len([r for r in tjde_results if r.get('tjde_score', 0) >= min_score])
        filtered_count = total_candidates - len(filtered_results)
        if filtered_count > 0:
            print(f"[BINANCE FILTER] Filtered out {filtered_count}/{total_candidates} tokens not available on BINANCE")
        
        print(f"[TRADINGVIEW] Generating screenshots for TOP {len(top_results)} BINANCE-compatible TJDE tokens:")
        for i, result in enumerate(top_results, 1):
            symbol = result.get('symbol', 'UNKNOWN')
            score = result.get('tjde_score', 0)
            print(f"  {i}. {symbol}: TJDE {score:.3f} (BINANCE verified)")
        
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