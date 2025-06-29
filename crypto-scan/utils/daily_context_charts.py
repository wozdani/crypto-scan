#!/usr/bin/env python3
"""
Daily Context Charts Generator
Generates TradingView charts for ALL tokens once per 24h for historical context
Separate from TOP 5 training charts - purely for market overview and analysis
"""

import os
import sys
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# Add crypto-scan to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tradingview_robust import RobustTradingViewGenerator

class DailyContextChartsGenerator:
    """Daily context charts generator for market overview"""
    
    def __init__(self):
        self.output_dir = Path("context_charts_daily")
        self.output_dir.mkdir(exist_ok=True)
        self.last_run_file = Path("data/daily_context_last_run.json")
        self.generator = RobustTradingViewGenerator()
        
    def should_generate_daily_charts(self) -> bool:
        """Check if 24 hours have passed since last generation"""
        if not self.last_run_file.exists():
            return True
            
        try:
            with open(self.last_run_file, 'r') as f:
                data = json.load(f)
            
            last_run = datetime.fromisoformat(data['last_run'])
            hours_since = (datetime.now() - last_run).total_seconds() / 3600
            
            print(f"[DAILY CHARTS] Last run: {hours_since:.1f}h ago")
            return hours_since >= 24.0
            
        except Exception as e:
            print(f"[DAILY CHARTS] Error reading last run: {e}")
            return True
    
    def mark_daily_run_complete(self):
        """Mark current timestamp as last successful run"""
        try:
            data = {
                'last_run': datetime.now().isoformat(),
                'charts_generated': True,
                'run_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Ensure data directory exists
            self.last_run_file.parent.mkdir(exist_ok=True)
            
            with open(self.last_run_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"[DAILY CHARTS] Marked run complete: {data['last_run']}")
            
        except Exception as e:
            print(f"[DAILY CHARTS] Error marking run complete: {e}")
    
    def get_all_symbols(self) -> List[str]:
        """Get all available symbols for daily chart generation"""
        try:
            # Load from bybit cache or fallback list
            cache_file = Path("data/bybit_symbols_cache.json")
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    symbols = cache_data.get('symbols', [])
                    print(f"[DAILY CHARTS] Loaded {len(symbols)} symbols from cache")
                    return symbols
            
            # Fallback to recent scan results
            scan_results = Path("data/scan_results.json")
            if scan_results.exists():
                with open(scan_results, 'r') as f:
                    data = json.load(f)
                    symbols = [result['symbol'] for result in data.get('results', [])]
                    print(f"[DAILY CHARTS] Using {len(symbols)} symbols from scan results")
                    return symbols
            
            print(f"[DAILY CHARTS] No symbols found - using minimal set")
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Minimal fallback
            
        except Exception as e:
            print(f"[DAILY CHARTS] Error loading symbols: {e}")
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    def generate_chart_filename(self, symbol: str, exchange: str) -> str:
        """Generate filename for daily context chart"""
        date_str = datetime.now().strftime('%Y%m%d')
        return f"DAILY_{symbol}_{exchange}_15M_{date_str}.png"
    
    async def generate_daily_chart(self, symbol: str) -> Optional[str]:
        """Generate single daily context chart"""
        try:
            print(f"[DAILY CHART] Generating context chart for {symbol}...")
            
            # Import required modules for custom screenshot generation
            from playwright.async_api import async_playwright
            from .multi_exchange_resolver import get_multi_exchange_resolver
            
            # Generate daily chart using custom implementation
            resolver = get_multi_exchange_resolver()
            tv_symbol, exchange = resolver.resolve_symbol(symbol)
            
            date_str = datetime.now().strftime('%Y%m%d')
            daily_filename = f"DAILY_{symbol}_{exchange}_15M_{date_str}.png"
            chart_path = self.output_dir / daily_filename
            
            # Take screenshot directly
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                try:
                    # Navigate to TradingView
                    url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}&interval=15"
                    await page.goto(url, wait_until='domcontentloaded', timeout=15000)
                    
                    # Wait for chart to load
                    await page.wait_for_selector("canvas", timeout=8000)
                    await asyncio.sleep(3)  # Additional wait for data
                    
                    # Take screenshot
                    await page.screenshot(path=str(chart_path), full_page=False)
                    
                    # Verify file was created
                    if chart_path.exists() and chart_path.stat().st_size > 10000:
                        # Save metadata
                        metadata = {
                            'symbol': symbol,
                            'exchange': exchange,
                            'tradingview_symbol': tv_symbol,
                            'type': 'daily_context_chart',
                            'generated_at': datetime.now().isoformat(),
                            'interval': '15M',
                            'purpose': 'historical_context_only',
                            'not_for_training': True,
                            'filename': daily_filename
                        }
                        
                        metadata_file = chart_path.with_suffix('.json')
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        print(f"   ✅ Context chart: {daily_filename}")
                        return str(chart_path)
                    else:
                        print(f"   ❌ Chart file too small or not created for {symbol}")
                        return None
                        
                except Exception as screenshot_error:
                    print(f"   ❌ Screenshot error for {symbol}: {screenshot_error}")
                    return None
                    
                finally:
                    await browser.close()
                
        except Exception as e:
            print(f"   ❌ Error generating chart for {symbol}: {e}")
            return None
    
    async def generate_all_daily_charts(self, max_concurrent: int = 5) -> Dict:
        """Generate daily context charts for all symbols"""
        symbols = self.get_all_symbols()
        
        print(f"🗓️ [DAILY CHARTS] Starting daily context generation for {len(symbols)} symbols")
        print(f"📁 Output directory: {self.output_dir}")
        
        start_time = datetime.now()
        successful_charts = []
        failed_symbols = []
        
        # Process symbols in batches to avoid overwhelming TradingView
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(symbol):
            async with semaphore:
                result = await self.generate_daily_chart(symbol)
                if result:
                    successful_charts.append(result)
                else:
                    failed_symbols.append(symbol)
                return result
        
        # Execute all chart generations
        tasks = [generate_with_semaphore(symbol) for symbol in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        summary = {
            'total_symbols': len(symbols),
            'successful_charts': len(successful_charts),
            'failed_symbols': len(failed_symbols),
            'success_rate': len(successful_charts) / len(symbols) * 100,
            'duration_seconds': duration,
            'charts_per_minute': len(successful_charts) / (duration / 60) if duration > 0 else 0,
            'generated_at': datetime.now().isoformat(),
            'output_directory': str(self.output_dir)
        }
        
        print(f"\n📊 [DAILY CHARTS] Generation Complete:")
        print(f"   ✅ Successful: {summary['successful_charts']}/{summary['total_symbols']} ({summary['success_rate']:.1f}%)")
        print(f"   ⏱️  Duration: {summary['duration_seconds']:.1f}s")
        print(f"   📈 Rate: {summary['charts_per_minute']:.1f} charts/min")
        
        if failed_symbols:
            print(f"   ❌ Failed symbols: {', '.join(failed_symbols[:10])}" + ("..." if len(failed_symbols) > 10 else ""))
        
        # Save summary
        summary_file = self.output_dir / f"daily_summary_{datetime.now().strftime('%Y%m%d')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def cleanup_old_daily_charts(self, keep_days: int = 7):
        """Remove daily charts older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            removed_count = 0
            
            for chart_file in self.output_dir.glob("DAILY_*.png"):
                try:
                    file_time = datetime.fromtimestamp(chart_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        # Remove chart and metadata
                        chart_file.unlink()
                        metadata_file = chart_file.with_suffix('.json')
                        if metadata_file.exists():
                            metadata_file.unlink()
                        removed_count += 1
                except Exception as e:
                    print(f"[DAILY CHARTS] Error removing {chart_file}: {e}")
            
            if removed_count > 0:
                print(f"🧹 [DAILY CHARTS] Cleaned up {removed_count} old charts (>{keep_days} days)")
                
        except Exception as e:
            print(f"[DAILY CHARTS] Error during cleanup: {e}")

async def run_daily_context_generation():
    """Main function to run daily context chart generation"""
    generator = DailyContextChartsGenerator()
    
    # Check if generation needed
    if not generator.should_generate_daily_charts():
        print("⏭️ [DAILY CHARTS] Skipping - already generated within 24h")
        return False
    
    print("🗓️ [DAILY CHARTS] Starting daily context chart generation...")
    
    # Cleanup old charts first
    generator.cleanup_old_daily_charts()
    
    # Generate new charts
    summary = await generator.generate_all_daily_charts()
    
    # Mark run as complete
    generator.mark_daily_run_complete()
    
    print(f"✅ [DAILY CHARTS] Daily generation complete - {summary['successful_charts']} charts generated")
    return True

def main():
    """CLI entry point for daily chart generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate daily context charts')
    parser.add_argument('--force', action='store_true', help='Force generation even if already run today')
    parser.add_argument('--cleanup-only', action='store_true', help='Only cleanup old charts')
    parser.add_argument('--max-concurrent', type=int, default=5, help='Maximum concurrent chart generations')
    
    args = parser.parse_args()
    
    if args.cleanup_only:
        generator = DailyContextChartsGenerator()
        generator.cleanup_old_daily_charts()
        print("✅ Cleanup complete")
        return
    
    if args.force:
        # Temporarily remove last run marker
        generator = DailyContextChartsGenerator()
        if generator.last_run_file.exists():
            generator.last_run_file.unlink()
        print("🔄 Forced generation mode - ignoring 24h limit")
    
    # Run async generation
    result = asyncio.run(run_daily_context_generation())
    
    if result:
        print("🎉 Daily context charts generated successfully!")
    else:
        print("ℹ️ Daily context charts not needed - check back in 24h")

if __name__ == "__main__":
    main()