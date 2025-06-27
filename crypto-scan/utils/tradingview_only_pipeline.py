"""
TradingView-Only Vision-AI Pipeline
Complete replacement of matplotlib chart generation with authentic TradingView screenshots
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional

try:
    from .tradingview_screenshot import TradingViewScreenshotGenerator, PLAYWRIGHT_AVAILABLE
except ImportError:
    try:
        from tradingview_screenshot import TradingViewScreenshotGenerator, PLAYWRIGHT_AVAILABLE
    except ImportError:
        PLAYWRIGHT_AVAILABLE = False

class TradingViewOnlyPipeline:
    """TradingView-only chart generation pipeline"""
    
    def __init__(self):
        self.output_dir = "charts/tradingview"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("training_charts", exist_ok=True)
    
    async def generate_tradingview_charts_async(
        self, 
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
            print("[TRADINGVIEW-ONLY] Playwright not available - cannot generate TradingView screenshots")
            return []
        
        try:
            # Filter results by TJDE score (exclude 'avoid' decisions for low scores)
            filtered_results = []
            for result in tjde_results:
                tjde_score = result.get('tjde_score', 0)
                tjde_decision = result.get('tjde_decision', 'avoid')
                
                # Only generate charts for meaningful signals
                if tjde_score >= min_score and tjde_decision != 'avoid':
                    filtered_results.append(result)
            
            # If no good signals, get TOP scoring regardless of decision (for training)
            if not filtered_results:
                print(f"[TRADINGVIEW-ONLY] No signals ≥{min_score}, using TOP scoring tokens for training")
                all_results = [r for r in tjde_results if r.get('tjde_score', 0) > 0]
                filtered_results = sorted(
                    all_results, 
                    key=lambda x: x.get('tjde_score', 0), 
                    reverse=True
                )[:max_symbols]
            
            # Sort by TJDE score (highest first) and limit
            top_results = sorted(
                filtered_results, 
                key=lambda x: x.get('tjde_score', 0), 
                reverse=True
            )[:max_symbols]
            
            if not top_results:
                print("[TRADINGVIEW-ONLY] No tokens available for chart generation")
                return []
            
            print(f"[TRADINGVIEW-ONLY] Generating charts for TOP {len(top_results)} tokens:")
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
                    
                    # Generate TradingView screenshot with new naming format
                    screenshot_path = await self._generate_formatted_screenshot(
                        generator, symbol, tjde_score, market_phase, decision
                    )
                    
                    if screenshot_path:
                        screenshot_paths.append(screenshot_path)
                    
                    # Small delay between captures
                    await asyncio.sleep(1)
            
            print(f"[TRADINGVIEW-ONLY] ✅ Generated {len(screenshot_paths)}/{len(top_results)} authentic TradingView charts")
            return screenshot_paths
            
        except Exception as e:
            print(f"[TRADINGVIEW-ONLY ERROR] Chart generation failed: {e}")
            return []
    
    async def _generate_formatted_screenshot(
        self,
        generator: TradingViewScreenshotGenerator,
        symbol: str,
        tjde_score: float,
        market_phase: str,
        decision: str
    ) -> Optional[str]:
        """Generate screenshot with formatted filename"""
        try:
            # Format: TOKENNAME_TJDE_0.537_PHASE_breakout.png
            formatted_filename = f"{symbol}_TJDE_{tjde_score:.3f}_PHASE_{market_phase}.png"
            formatted_path = os.path.join(self.output_dir, formatted_filename)
            
            # Use TradingView generator but override the output path
            original_output_dir = generator.output_dir
            generator.output_dir = self.output_dir
            
            try:
                # Generate screenshot
                screenshot_path = await generator.generate_tradingview_screenshot(
                    symbol=symbol,
                    tjde_score=tjde_score,
                    market_phase=market_phase,
                    decision=decision
                )
                
                if screenshot_path:
                    # Rename to formatted filename if needed
                    if screenshot_path != formatted_path:
                        if os.path.exists(screenshot_path):
                            os.rename(screenshot_path, formatted_path)
                            # Also rename JSON metadata
                            old_json = screenshot_path.replace('.png', '.json')
                            new_json = formatted_path.replace('.png', '.json')
                            if os.path.exists(old_json):
                                os.rename(old_json, new_json)
                            screenshot_path = formatted_path
                    
                    print(f"[TRADINGVIEW-ONLY] ✅ Generated: {formatted_filename}")
                    return screenshot_path
                
            finally:
                generator.output_dir = original_output_dir
                
        except Exception as e:
            print(f"[TRADINGVIEW-ONLY ERROR] {symbol}: {e}")
            return None
    
    def sync_generate_tradingview_charts(
        self, 
        tjde_results: List[Dict],
        min_score: float = 0.5,
        max_symbols: int = 5
    ) -> List[str]:
        """
        Synchronous wrapper for TradingView chart generation
        Fixes the asyncio 'coroutine was never awaited' warning
        """
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - create task instead of run
                print("[TRADINGVIEW-ONLY] Running in async context - creating task")
                task = asyncio.create_task(
                    self.generate_tradingview_charts_async(
                        tjde_results, min_score, max_symbols
                    )
                )
                # Get current event loop and run until complete
                return loop.run_until_complete(task)
                
            except RuntimeError:
                # No running loop - safe to use asyncio.run
                print("[TRADINGVIEW-ONLY] No async context - using asyncio.run")
                return asyncio.run(
                    self.generate_tradingview_charts_async(
                        tjde_results, min_score, max_symbols
                    )
                )
                
        except Exception as e:
            print(f"[TRADINGVIEW-ONLY SYNC ERROR] {e}")
            return []
    
    def replace_matplotlib_charts(self, tjde_results: List[Dict]) -> Dict[str, str]:
        """
        Replace all matplotlib chart generation with TradingView screenshots
        
        Returns:
            Dictionary mapping symbols to TradingView chart paths
        """
        print("[TRADINGVIEW-ONLY] 🔄 Replacing matplotlib charts with authentic TradingView screenshots")
        
        # Generate TradingView charts
        screenshot_paths = self.sync_generate_tradingview_charts(
            tjde_results=tjde_results,
            min_score=0.4,  # Lower threshold for training data
            max_symbols=5
        )
        
        # Create symbol-to-path mapping
        chart_mapping = {}
        for path in screenshot_paths:
            if os.path.exists(path):
                filename = os.path.basename(path)
                # Extract symbol from filename: SYMBOL_TJDE_0.537_PHASE_breakout.png
                if '_TJDE_' in filename:
                    symbol = filename.split('_TJDE_')[0]
                    chart_mapping[symbol] = path
        
        if chart_mapping:
            print(f"[TRADINGVIEW-ONLY] ✅ Generated {len(chart_mapping)} TradingView charts:")
            for symbol, path in chart_mapping.items():
                print(f"  • {symbol}: {os.path.basename(path)}")
        else:
            print("[TRADINGVIEW-ONLY] ❌ No TradingView charts generated")
        
        return chart_mapping
    
    def cleanup_old_matplotlib_charts(self):
        """Remove old matplotlib chart files to prevent confusion"""
        try:
            # Remove old chart directories that might contain matplotlib charts
            old_dirs = [
                "training_data/charts",
                "charts/fallback",
                "charts/matplotlib"
            ]
            
            for old_dir in old_dirs:
                if os.path.exists(old_dir):
                    import shutil
                    shutil.rmtree(old_dir)
                    print(f"[TRADINGVIEW-ONLY] 🗑️ Removed old matplotlib directory: {old_dir}")
            
            # Move any existing training_charts/*.png to archive if they're matplotlib
            training_dir = "training_charts"
            if os.path.exists(training_dir):
                archive_dir = "archive/old_matplotlib_charts"
                os.makedirs(archive_dir, exist_ok=True)
                
                moved_count = 0
                for filename in os.listdir(training_dir):
                    if filename.endswith('.png') and 'matplotlib' in filename.lower():
                        old_path = os.path.join(training_dir, filename)
                        new_path = os.path.join(archive_dir, filename)
                        os.rename(old_path, new_path)
                        moved_count += 1
                
                if moved_count > 0:
                    print(f"[TRADINGVIEW-ONLY] 📦 Archived {moved_count} old matplotlib charts")
            
        except Exception as e:
            print(f"[TRADINGVIEW-ONLY] Cleanup error: {e}")

# Global instance
tradingview_pipeline = TradingViewOnlyPipeline()

def generate_tradingview_only_charts(tjde_results: List[Dict]) -> Dict[str, str]:
    """
    Main function to replace all matplotlib chart generation with TradingView screenshots
    
    Args:
        tjde_results: List of TJDE analysis results
        
    Returns:
        Dictionary mapping symbols to TradingView chart paths
    """
    return tradingview_pipeline.replace_matplotlib_charts(tjde_results)

def cleanup_matplotlib_legacy():
    """Remove old matplotlib chart files"""
    tradingview_pipeline.cleanup_old_matplotlib_charts()