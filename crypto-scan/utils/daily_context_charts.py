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
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# Add crypto-scan to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tradingview_robust import RobustTradingViewGenerator

class DailyContextChartsGenerator:
    """Daily context charts generator for market overview"""
    
    def __init__(self, target_hour: int = 0):
        """
        Initialize daily context charts generator
        
        Args:
            target_hour: Hour of day (0-23) when to generate daily charts (default: 0 = 00:00 UTC - end of 1D candle)
        """
        self.target_hour = target_hour  # Hour when daily charts should be generated (0 = midnight UTC)
        self.output_dir = Path("context_charts_daily")
        self.output_dir.mkdir(exist_ok=True)
        self.last_run_file = Path("data/daily_context_last_run.json")
        self.generator = RobustTradingViewGenerator()
        
    def should_generate_daily_charts(self) -> bool:
        """
        Check if charts should be generated based on target hour and last run date
        
        Charts are generated once per day at 00:00 UTC (midnight) - end of 1D candle
        This ensures daily context charts capture complete 24h market data
        """
        current_time = datetime.now()
        
        # If no last run file, generate if we're at or past target hour
        if not self.last_run_file.exists():
            should_run = current_time.hour >= self.target_hour
            print(f"[DAILY CHARTS] First run - current hour: {current_time.hour}, target: {self.target_hour}, should run: {should_run}")
            return should_run
            
        try:
            with open(self.last_run_file, 'r') as f:
                data = json.load(f)
            
            last_run = datetime.fromisoformat(data['last_run'])
            last_run_date = last_run.date()
            current_date = current_time.date()
            
            # If it's a new day and we're at or past target hour
            if current_date > last_run_date and current_time.hour >= self.target_hour:
                print(f"[DAILY CHARTS] New day ({current_date}) and past target hour ({self.target_hour}:00), should generate")
                return True
            
            # If same day but we haven't run yet today and we're at target hour
            if current_date == last_run_date:
                hours_since = (current_time - last_run).total_seconds() / 3600
                print(f"[DAILY CHARTS] Same day, last run: {hours_since:.1f}h ago")
                return False
            
            print(f"[DAILY CHARTS] Current: {current_time.hour}:00, Target: {self.target_hour}:00, Last: {last_run_date}")
            return False
            
        except Exception as e:
            print(f"[DAILY CHARTS] Error reading last run: {e}")
            return current_time.hour >= self.target_hour
    
    def get_target_hour_info(self) -> dict:
        """Get information about target hour configuration"""
        current_time = datetime.now()
        return {
            'target_hour_utc': self.target_hour,
            'target_time_formatted': f"{self.target_hour:02d}:00 UTC",
            'current_hour_utc': current_time.hour,
            'current_time_formatted': f"{current_time.hour:02d}:{current_time.minute:02d} UTC",
            'next_run_today': current_time.hour < self.target_hour,
            'description': f"Daily charts generate once per day at {self.target_hour:02d}:00 UTC"
        }
    
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
        """Generate single daily context chart using existing TradingView generator"""
        try:
            print(f"[DAILY CHART] Generating context chart for {symbol}...")
            
            # Use existing robust TradingView generator system
            from .tradingview_robust import RobustTradingViewGenerator
            from .multi_exchange_resolver import get_multi_exchange_resolver
            
            # Resolve symbol
            resolver = get_multi_exchange_resolver()
            result = resolver.resolve_tradingview_symbol(symbol)
            if result:
                tv_symbol, exchange = result
            else:
                # Fallback to BINANCE
                tv_symbol = f"BINANCE:{symbol}"
                exchange = "BINANCE"
            
            # Generate chart using existing system
            async with RobustTradingViewGenerator() as generator:
                # Temporarily modify screenshot directory
                original_generate = generator.generate_screenshot
                
                async def custom_generate_screenshot(symbol_param, tjde_score=0.0, decision="context"):
                    # Generate chart with regular system
                    result = await original_generate(symbol_param, tjde_score, decision)
                    
                    if result and result != "INVALID_SYMBOL":
                        # Move to daily context directory and rename
                        original_path = Path(result)
                        date_str = datetime.now().strftime('%Y%m%d')
                        daily_filename = f"DAILY_{symbol}_{exchange}_15M_{date_str}.png"
                        daily_path = self.output_dir / daily_filename
                        
                        # Move file
                        import shutil
                        shutil.move(str(original_path), str(daily_path))
                        
                        # Move metadata if exists
                        original_metadata = original_path.with_suffix('.json')
                        if original_metadata.exists():
                            # Update metadata for daily context
                            with open(original_metadata, 'r') as f:
                                metadata = json.load(f)
                            
                            # Update for daily context
                            metadata.update({
                                'type': 'daily_context_chart',
                                'purpose': 'historical_context_only',
                                'not_for_training': True,
                                'original_filename': original_path.name,
                                'daily_filename': daily_filename
                            })
                            
                            daily_metadata = daily_path.with_suffix('.json')
                            with open(daily_metadata, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            
                            # Remove original metadata
                            original_metadata.unlink()
                        
                        print(f"   ✅ Context chart: {daily_filename}")
                        return str(daily_path)
                    
                    return result
                
                # Generate screenshot
                chart_result = await custom_generate_screenshot(symbol)
                
                if chart_result and chart_result != "INVALID_SYMBOL":
                    return chart_result
                else:
                    print(f"   ❌ Failed to generate chart for {symbol}")
                    return None
                
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