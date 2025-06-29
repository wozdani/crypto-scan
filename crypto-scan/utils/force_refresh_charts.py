"""
Force Refresh Chart Generation
Ensures fresh screenshot generation for Vision-AI training data
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

# Import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from crypto_scan_service import log_warning
except ImportError:
    def log_warning(label, exception=None, additional_info=None):
        print(f"⚠️ [{label}] {exception} - {additional_info}")

def should_regenerate_chart(chart_path: str, max_age_minutes: int = 60) -> bool:
    """
    Check if chart should be regenerated based on age and freshness
    
    Args:
        chart_path: Path to existing chart file
        max_age_minutes: Maximum age in minutes before regeneration
        
    Returns:
        True if chart should be regenerated
    """
    try:
        if not os.path.exists(chart_path):
            return True  # File doesn't exist, need to generate
        
        # Check file age
        file_time = datetime.fromtimestamp(os.path.getmtime(chart_path))
        age_minutes = (datetime.now() - file_time).total_seconds() / 60
        
        if age_minutes > max_age_minutes:
            print(f"[FORCE REFRESH] {os.path.basename(chart_path)} is {age_minutes:.1f} minutes old (>{max_age_minutes}m), regenerating")
            return True
        
        # Check if timestamp in filename is stale
        filename = os.path.basename(chart_path)
        if '_20' in filename:  # Contains timestamp
            try:
                # Extract timestamp from filename like BTCUSDT_2025-06-28_21:18:05_tjde.png
                parts = filename.split('_')
                for i, part in enumerate(parts):
                    if '20' in part and len(part) >= 10:  # Date part
                        date_str = part
                        if i + 1 < len(parts) and ':' in parts[i + 1]:
                            time_str = parts[i + 1]
                            timestamp_str = f"{date_str} {time_str}"
                            
                            # Parse timestamp
                            try:
                                chart_timestamp = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S")
                                timestamp_age_minutes = (datetime.now() - chart_timestamp).total_seconds() / 60
                                
                                if timestamp_age_minutes > max_age_minutes:
                                    print(f"[FORCE REFRESH] {filename} timestamp is {timestamp_age_minutes:.1f} minutes old, regenerating")
                                    return True
                            except ValueError:
                                pass  # Skip if timestamp parsing fails
                        break
            except Exception:
                pass  # Skip timestamp analysis if parsing fails
        
        return False  # Chart is fresh enough
        
    except Exception as e:
        log_warning("CHART AGE CHECK ERROR", e, f"Error checking {chart_path}")
        return True  # Regenerate on error to be safe

def clean_old_charts_for_symbol(symbol: str, chart_dirs: List[str] = None) -> int:
    """
    Clean old charts for a specific symbol before generating new ones
    
    Args:
        symbol: Trading symbol
        chart_dirs: List of directories to check
        
    Returns:
        Number of old charts removed
    """
    if chart_dirs is None:
        chart_dirs = ["training_data/charts", "data/charts", "screenshots"]
    
    removed_count = 0
    
    for chart_dir in chart_dirs:
        if not os.path.exists(chart_dir):
            continue
            
        try:
            # Find old charts for this symbol
            for filename in os.listdir(chart_dir):
                if filename.startswith(symbol) and filename.endswith(('.png', '.jpg', '.webp')):
                    chart_path = os.path.join(chart_dir, filename)
                    
                    # Check if chart should be regenerated
                    if should_regenerate_chart(chart_path, max_age_minutes=30):
                        try:
                            os.remove(chart_path)
                            removed_count += 1
                            print(f"[FORCE REFRESH] Removed old chart: {filename}")
                            
                            # Also remove associated JSON metadata if exists
                            json_path = chart_path.replace('.png', '.json').replace('.jpg', '.json').replace('.webp', '.json')
                            if os.path.exists(json_path):
                                os.remove(json_path)
                                print(f"[FORCE REFRESH] Removed old metadata: {os.path.basename(json_path)}")
                                
                        except Exception as e:
                            log_warning("CHART REMOVAL ERROR", e, f"Failed to remove {chart_path}")
                            
        except Exception as e:
            log_warning("CHART CLEANUP ERROR", e, f"Error cleaning {chart_dir}")
    
    return removed_count

def force_refresh_vision_ai_charts(
    tjde_results: List[Dict], 
    min_score: float = 0.5, 
    max_symbols: int = 5,
    force_regenerate: bool = True
) -> Dict[str, str]:
    """
    Force refresh Vision-AI charts with fresh TradingView screenshots
    
    Args:
        tjde_results: TJDE analysis results
        min_score: Minimum TJDE score
        max_symbols: Maximum symbols to process
        force_regenerate: If True, remove old charts before generating new ones
        
    Returns:
        Dictionary mapping symbols to new chart paths
    """
    try:
        # Filter and sort eligible tokens
        eligible = [r for r in tjde_results if r.get('tjde_score', 0) >= min_score]
        eligible.sort(key=lambda x: x.get('tjde_score', 0), reverse=True)
        top_results = eligible[:max_symbols]
        
        if not top_results:
            print("[FORCE REFRESH] No eligible tokens for chart generation")
            return {}
        
        print(f"[FORCE REFRESH] Generating fresh charts for {len(top_results)} tokens")
        
        generated_charts = {}
        
        # 🎯 CRITICAL FIX: Generate all charts in single batch instead of 5 separate calls
        # This prevents multiple TradingView browser sessions and eliminates mass generation
        
        if force_regenerate:
            # Clean old charts for all symbols first
            for result in top_results:
                symbol = result.get('symbol', 'UNKNOWN')
                removed_count = clean_old_charts_for_symbol(symbol)
                if removed_count > 0:
                    print(f"[FORCE REFRESH] {symbol}: Removed {removed_count} old charts")
        
        # Generate ALL charts in single batch call
        try:
            from .tradingview_async_fix import generate_tradingview_charts_safe
            
            # ✅ FIXED: Generate charts for ALL symbols at once instead of individual calls
            chart_paths = generate_tradingview_charts_safe(top_results, 0.0, len(top_results))
            
            if chart_paths:
                # Map generated paths to symbols
                for i, result in enumerate(top_results):
                    symbol = result.get('symbol', 'UNKNOWN')
                    if i < len(chart_paths) and chart_paths[i]:
                        generated_charts[symbol] = chart_paths[i]
                        print(f"[FORCE REFRESH] ✅ {symbol}: Generated fresh chart: {os.path.basename(chart_paths[i])}")
                    else:
                        print(f"[FORCE REFRESH] ❌ {symbol}: Failed to generate fresh chart")
            else:
                print("[FORCE REFRESH] ❌ No charts generated from batch call")
                
        except Exception as e:
            log_warning("FORCE REFRESH BATCH ERROR", e, f"Failed to generate batch charts for {len(top_results)} tokens")
        
        if generated_charts:
            print(f"[FORCE REFRESH] ✅ Generated {len(generated_charts)} fresh charts")
        else:
            print("[FORCE REFRESH] ❌ No fresh charts generated")
        
        return generated_charts
        
    except Exception as e:
        log_warning("FORCE REFRESH ERROR", e, "Force refresh failed")
        return {}

def check_chart_freshness_status(chart_dirs: List[str] = None) -> Dict[str, int]:
    """
    Check freshness status of all charts
    
    Args:
        chart_dirs: List of directories to check
        
    Returns:
        Dictionary with freshness statistics
    """
    if chart_dirs is None:
        chart_dirs = ["training_data/charts", "data/charts", "screenshots"]
    
    stats = {
        'total_charts': 0,
        'fresh_charts': 0,
        'stale_charts': 0,
        'very_old_charts': 0
    }
    
    for chart_dir in chart_dirs:
        if not os.path.exists(chart_dir):
            continue
            
        try:
            for filename in os.listdir(chart_dir):
                if filename.endswith(('.png', '.jpg', '.webp')):
                    chart_path = os.path.join(chart_dir, filename)
                    stats['total_charts'] += 1
                    
                    file_time = datetime.fromtimestamp(os.path.getmtime(chart_path))
                    age_minutes = (datetime.now() - file_time).total_seconds() / 60
                    
                    if age_minutes <= 30:
                        stats['fresh_charts'] += 1
                    elif age_minutes <= 120:
                        stats['stale_charts'] += 1
                    else:
                        stats['very_old_charts'] += 1
                        
        except Exception as e:
            log_warning("CHART FRESHNESS CHECK ERROR", e, f"Error checking {chart_dir}")
    
    return stats

def main():
    """Test force refresh functionality"""
    print("🔄 FORCE REFRESH CHART SYSTEM TEST")
    print("=" * 50)
    
    # Check current chart freshness
    stats = check_chart_freshness_status()
    print(f"📊 Chart Freshness Status:")
    print(f"• Total charts: {stats['total_charts']}")
    print(f"• Fresh (<30m): {stats['fresh_charts']}")
    print(f"• Stale (30m-2h): {stats['stale_charts']}")
    print(f"• Very old (>2h): {stats['very_old_charts']}")
    
    # Test with sample data
    test_results = [
        {
            'symbol': 'BTCUSDT',
            'tjde_score': 0.75,
            'market_phase': 'trend-following',
            'tjde_decision': 'consider_entry'
        }
    ]
    
    generated = force_refresh_vision_ai_charts(test_results, 0.5, 1, True)
    print(f"\n🎯 Test Results: {len(generated)} charts generated")

if __name__ == "__main__":
    main()