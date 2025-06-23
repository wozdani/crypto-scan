#!/usr/bin/env python3
"""
Demo Chart Export for Computer Vision Training Data
Demonstrates the chart export functionality with real market data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.chart_exporter import export_chart_image, export_multiple_charts, get_export_stats


def demo_single_exports():
    """Demo individual chart exports in different styles"""
    print("📊 Demo: Single Chart Exports")
    print("-" * 40)
    
    # Test symbols
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    styles = ["professional", "clean", "detailed"]
    
    for symbol in symbols[:2]:  # Test first 2 symbols
        print(f"\n🔍 Exporting {symbol} charts...")
        
        for style in styles:
            try:
                result = export_chart_image(
                    symbol=symbol,
                    timeframe="15m",
                    limit=72,  # 18 hours of 15m candles
                    chart_style=style,
                    include_volume=True,
                    include_ema=True
                )
                
                if result:
                    print(f"  ✅ {style}: {os.path.basename(result)}")
                else:
                    print(f"  ❌ {style}: Export failed")
                    
            except Exception as e:
                print(f"  ❌ {style}: Error - {e}")


def demo_training_batch():
    """Demo batch export for training data generation"""
    print("\n🤖 Demo: Training Data Batch Export")
    print("-" * 40)
    
    # Smaller batch for demo
    training_symbols = ["ADAUSDT", "SOLUSDT"]
    timeframes = ["15m"]
    styles = ["professional", "clean"]
    
    try:
        results = export_multiple_charts(
            symbols=training_symbols,
            timeframes=timeframes,
            styles=styles,
            limit=48  # 12 hours of data
        )
        
        print("\n📈 Batch Export Results:")
        total_files = 0
        for symbol, files in results.items():
            print(f"  {symbol}: {len(files)} charts")
            total_files += len(files)
            
        print(f"\nTotal training images generated: {total_files}")
        
    except Exception as e:
        print(f"❌ Batch export failed: {e}")


def demo_chart_variations():
    """Demo different chart configurations"""
    print("\n⚙️ Demo: Chart Configuration Variations")
    print("-" * 40)
    
    symbol = "ETHUSDT"
    
    # Different configurations for diverse training data
    configs = [
        {
            "name": "Standard",
            "config": {"include_volume": True, "include_ema": True, "limit": 96}
        },
        {
            "name": "Price Only", 
            "config": {"include_volume": False, "include_ema": False, "limit": 72}
        },
        {
            "name": "With EMA",
            "config": {"include_volume": False, "include_ema": True, "limit": 48}
        }
    ]
    
    for variant in configs:
        try:
            result = export_chart_image(
                symbol=symbol,
                timeframe="15m",
                chart_style="professional",
                save_as=f"demo_{variant['name'].lower().replace(' ', '_')}.png",
                **variant["config"]
            )
            
            if result:
                print(f"  ✅ {variant['name']}: {os.path.basename(result)}")
            else:
                print(f"  ❌ {variant['name']}: Failed")
                
        except Exception as e:
            print(f"  ❌ {variant['name']}: {e}")


def demo_export_analytics():
    """Show export statistics and analytics"""
    print("\n📊 Demo: Export Analytics")
    print("-" * 40)
    
    try:
        stats = get_export_stats()
        
        print(f"📁 Export Folder: {stats.get('export_folder', 'Unknown')}")
        print(f"📄 Total Files: {stats.get('total_files', 0)}")
        print(f"💾 Total Size: {stats.get('total_size_mb', 0)} MB")
        
        latest = stats.get('latest_exports', [])
        if latest:
            print(f"\n🕐 Latest Exports:")
            for filename in latest:
                print(f"  - {filename}")
        else:
            print("\n📭 No exports found")
            
    except Exception as e:
        print(f"❌ Analytics failed: {e}")


def main():
    """Run chart export demonstrations"""
    print("🚀 Chart Export Module Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    demo_single_exports()
    demo_training_batch()
    demo_chart_variations()
    demo_export_analytics()
    
    print("\n" + "=" * 50)
    print("✅ Chart Export Demo Completed!")
    
    # Final summary
    final_stats = get_export_stats()
    print(f"\n📊 Final Summary:")
    print(f"  Charts Generated: {final_stats.get('total_files', 0)}")
    print(f"  Storage Used: {final_stats.get('total_size_mb', 0)} MB")
    print(f"  Ready for CV Training: YES" if final_stats.get('total_files', 0) > 0 else "  Ready for CV Training: NO")


if __name__ == "__main__":
    main()