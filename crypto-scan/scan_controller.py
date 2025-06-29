#!/usr/bin/env python3
"""
Scan Controller - Central management for all scanning operations
Manages both regular async scans and daily context chart generation
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

# Add crypto-scan to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.daily_context_charts import run_daily_context_generation
from scan_all_tokens_async import main as async_scan_main

class ScanController:
    """Central controller for all scanning operations"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
    async def run_regular_scan(self):
        """Run regular async token scanning"""
        print("🔄 [SCAN CONTROLLER] Starting regular async token scan...")
        start_time = datetime.now()
        
        try:
            # Run the main async scanning function
            await async_scan_main()
            
            duration = (datetime.now() - start_time).total_seconds()
            print(f"✅ [SCAN CONTROLLER] Regular scan completed in {duration:.1f}s")
            return True
            
        except Exception as e:
            print(f"❌ [SCAN CONTROLLER] Regular scan failed: {e}")
            return False
    
    async def run_daily_context_charts(self, force: bool = False):
        """Run daily context chart generation"""
        print("📊 [SCAN CONTROLLER] Checking daily context chart generation...")
        
        try:
            # Force generation if requested
            if force:
                from utils.daily_context_charts import DailyContextChartsGenerator
                generator = DailyContextChartsGenerator()
                if generator.last_run_file.exists():
                    generator.last_run_file.unlink()
                print("🔄 [SCAN CONTROLLER] Forced daily generation mode")
            
            # Run daily context generation
            result = await run_daily_context_generation()
            
            if result:
                print("✅ [SCAN CONTROLLER] Daily context charts generated successfully")
            else:
                print("ℹ️ [SCAN CONTROLLER] Daily context charts not needed - already generated within 24h")
            
            return result
            
        except Exception as e:
            print(f"❌ [SCAN CONTROLLER] Daily context chart generation failed: {e}")
            return False
    
    async def run_full_cycle(self, include_daily: bool = True, force_daily: bool = False):
        """Run complete scan cycle with optional daily charts"""
        print("🚀 [SCAN CONTROLLER] Starting full scan cycle...")
        cycle_start = datetime.now()
        
        results = {
            'regular_scan': False,
            'daily_charts': False,
            'total_duration': 0,
            'timestamp': cycle_start.isoformat()
        }
        
        # 1. Run regular async scan
        results['regular_scan'] = await self.run_regular_scan()
        
        # 2. Run daily context charts if requested
        if include_daily:
            results['daily_charts'] = await self.run_daily_context_charts(force=force_daily)
        
        # Calculate total duration
        results['total_duration'] = (datetime.now() - cycle_start).total_seconds()
        
        # Generate summary
        print(f"\n📋 [SCAN CONTROLLER] Full Cycle Summary:")
        print(f"   ✅ Regular Scan: {'SUCCESS' if results['regular_scan'] else 'FAILED'}")
        
        if include_daily:
            daily_status = 'GENERATED' if results['daily_charts'] else 'SKIPPED'
            print(f"   📊 Daily Charts: {daily_status}")
        
        print(f"   ⏱️  Total Duration: {results['total_duration']:.1f}s")
        
        # Save cycle results
        self.save_cycle_results(results)
        
        return results
    
    def save_cycle_results(self, results: dict):
        """Save scan cycle results for monitoring"""
        try:
            import json
            
            results_file = self.data_dir / "scan_controller_history.json"
            
            # Load existing history
            history = []
            if results_file.exists():
                with open(results_file, 'r') as f:
                    history = json.load(f)
            
            # Add current results
            history.append(results)
            
            # Keep only last 100 entries
            history = history[-100:]
            
            # Save updated history
            with open(results_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            print(f"📝 [SCAN CONTROLLER] Cycle results saved to {results_file}")
            
        except Exception as e:
            print(f"⚠️ [SCAN CONTROLLER] Error saving cycle results: {e}")

async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Crypto Scan Controller')
    parser.add_argument('--mode', choices=['regular', 'daily', 'full'], default='full',
                        help='Scan mode: regular (async only), daily (context charts only), full (both)')
    parser.add_argument('--force-daily', action='store_true',
                        help='Force daily chart generation even if already run today')
    parser.add_argument('--no-daily', action='store_true',
                        help='Skip daily chart generation in full mode')
    
    args = parser.parse_args()
    
    controller = ScanController()
    
    print(f"🎯 [SCAN CONTROLLER] Starting in {args.mode.upper()} mode")
    
    if args.mode == 'regular':
        # Run only regular async scan
        await controller.run_regular_scan()
        
    elif args.mode == 'daily':
        # Run only daily context charts
        await controller.run_daily_context_charts(force=args.force_daily)
        
    elif args.mode == 'full':
        # Run full cycle
        include_daily = not args.no_daily
        await controller.run_full_cycle(
            include_daily=include_daily, 
            force_daily=args.force_daily
        )
    
    print("🏁 [SCAN CONTROLLER] Controller execution complete")

if __name__ == "__main__":
    asyncio.run(main())