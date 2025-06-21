"""
Trend Mode Debugger - Complete diagnostic tool for S/R trend detection
Analyzes each condition step by step to identify why alerts aren't generated
"""

import os
import json
from datetime import datetime
from detectors.trend_mode_sr import (
    fetch_15m_prices_extended,
    fetch_5m_prices_recent,
    get_current_price,
    get_orderbook_volumes_sr,
    is_strong_recent_trend,
    find_support_resistance_levels,
    is_price_near_support,
    is_entry_after_correction,
    detect_sr_trend_mode
)

def create_debug_log(symbol, analysis_data, result):
    """Create detailed debug log entry"""
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"\n[{timestamp}] TREND MODE DEBUG - {symbol}\n"
    log_entry += "=" * 60 + "\n"
    
    # Data availability
    log_entry += f"Data Status:\n"
    log_entry += f"  15M candles: {analysis_data.get('prices_15m_count', 0)}\n"
    log_entry += f"  5M candles: {analysis_data.get('prices_5m_count', 0)}\n"
    log_entry += f"  Current price: {analysis_data.get('current_price', 0)}\n"
    log_entry += f"  Orderbook available: {analysis_data.get('orderbook_available', False)}\n"
    
    # Condition analysis
    log_entry += f"\nCondition Analysis:\n"
    log_entry += f"  Strong 3h trend: {analysis_data.get('strong_trend_3h', False)}\n"
    if 'up_ratio' in analysis_data:
        log_entry += f"    - Up ratio: {analysis_data['up_ratio']:.3f} (required: ≥0.5)\n"
        log_entry += f"    - Price progress: {analysis_data.get('price_progress', False)}\n"
    
    log_entry += f"  Near support: {analysis_data.get('near_support', False)}\n"
    if 'support_levels_count' in analysis_data:
        log_entry += f"    - S/R levels found: {analysis_data['support_levels_count']}\n"
        log_entry += f"    - Closest support: {analysis_data.get('closest_support', 'N/A')}\n"
        log_entry += f"    - Distance to support: {analysis_data.get('support_distance', 'N/A')}\n"
    
    log_entry += f"  Entry after correction: {analysis_data.get('entry_after_correction', False)}\n"
    if 'recent_reds' in analysis_data:
        log_entry += f"    - Recent red candles: {analysis_data['recent_reds']}\n"
        log_entry += f"    - Ask volume declining: {analysis_data.get('ask_down', False)}\n"
        log_entry += f"    - Bid volume rising: {analysis_data.get('bid_up', False)}\n"
    
    # Final result
    log_entry += f"\nResult:\n"
    log_entry += f"  Score: {result.get('trend_score', 0)}/100\n"
    log_entry += f"  Trend Mode: {result.get('trend_mode', False)}\n"
    log_entry += f"  Entry Triggered: {result.get('entry_triggered', False)}\n"
    log_entry += f"  Description: {result.get('description', 'N/A')}\n"
    
    # Failure reason
    if not result.get('trend_mode', False):
        failure_reasons = []
        if not analysis_data.get('strong_trend_3h', False):
            if 'up_ratio' in analysis_data:
                failure_reasons.append(f"trend too weak (up_ratio={analysis_data['up_ratio']:.3f})")
            else:
                failure_reasons.append("trend analysis failed")
        
        if not analysis_data.get('near_support', False):
            failure_reasons.append("no support near current price")
        
        if not analysis_data.get('entry_after_correction', False):
            failure_reasons.append("bid pressure too low")
        
        log_entry += f"  Failure reason: {', '.join(failure_reasons) if failure_reasons else 'unknown'}\n"
    
    log_entry += "\n"
    
    # Write to log file
    with open("trend_debug_log.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)
    
    return log_entry

def debug_symbol_trend_mode(symbol):
    """Debug trend mode detection for a specific symbol"""
    
    print(f"🔍 Debugging Trend Mode for {symbol}")
    print("=" * 50)
    
    analysis_data = {}
    
    try:
        # Step 1: Fetch data
        print("Step 1: Fetching market data...")
        prices_15m_full = fetch_15m_prices_extended(symbol, 24)
        prices_5m = fetch_5m_prices_recent(symbol, 12)
        current_price = get_current_price(symbol)
        ask_vols, bid_vols = get_orderbook_volumes_sr(symbol)
        
        analysis_data.update({
            'prices_15m_count': len(prices_15m_full),
            'prices_5m_count': len(prices_5m),
            'current_price': current_price,
            'orderbook_available': bool(ask_vols and bid_vols)
        })
        
        print(f"  15M candles: {len(prices_15m_full)}")
        print(f"  5M candles: {len(prices_5m)}")
        print(f"  Current price: {current_price}")
        print(f"  Orderbook data: {'Available' if ask_vols and bid_vols else 'Missing'}")
        
        if len(prices_15m_full) < 24:
            print(f"❌ Insufficient 15M data (need minimum 24 candles)")
            result = {'trend_mode': False, 'trend_score': 0, 'description': 'Insufficient data'}
            create_debug_log(symbol, analysis_data, result)
            return result
        
        # Step 2: Recent trend analysis (last 3h = 12 candles)
        print("\nStep 2: Analyzing recent 3h trend...")
        prices_15m_recent = prices_15m_full[:12]
        strong_trend = is_strong_recent_trend(prices_15m_recent)
        
        # Calculate up ratio for debugging
        if len(prices_15m_recent) > 1:
            up_moves = sum(1 for i in range(1, len(prices_15m_recent)) 
                          if prices_15m_recent[i] > prices_15m_recent[i-1])
            up_ratio = up_moves / (len(prices_15m_recent) - 1)
            price_progress = prices_15m_recent[-1] > prices_15m_recent[0]
            
            analysis_data.update({
                'strong_trend_3h': strong_trend,
                'up_ratio': up_ratio,
                'price_progress': price_progress
            })
            
            print(f"  Up ratio: {up_ratio:.3f} (required: ≥0.5)")
            print(f"  Price progress: {price_progress}")
            print(f"  Strong trend: {strong_trend}")
        
        # Step 3: Support/Resistance analysis
        print("\nStep 3: Analyzing S/R levels...")
        prices_15m_history = prices_15m_full[12:96] if len(prices_15m_full) >= 96 else prices_15m_full[12:]
        support_levels = find_support_resistance_levels(prices_15m_history)
        near_support = is_price_near_support(current_price, support_levels, margin=0.004)  # Lowered margin
        
        closest_support = None
        support_distance = None
        if support_levels and current_price > 0:
            supports = [l for l in support_levels if l < current_price]
            if supports:
                closest_support = max(supports)
                support_distance = abs(current_price - closest_support) / closest_support
        
        analysis_data.update({
            'near_support': near_support,
            'support_levels_count': len(support_levels),
            'closest_support': closest_support,
            'support_distance': support_distance
        })
        
        print(f"  S/R levels found: {len(support_levels)}")
        print(f"  Closest support: {closest_support}")
        print(f"  Distance to support: {support_distance:.4f if support_distance else 'N/A'}")
        print(f"  Near support: {near_support}")
        
        # Step 4: Entry signal analysis
        print("\nStep 4: Analyzing entry signal...")
        entry_signal = is_entry_after_correction(prices_5m, ask_vols, bid_vols)
        
        # Detailed entry analysis
        if len(prices_5m) >= 5 and len(ask_vols) >= 3 and len(bid_vols) >= 3:
            recent_reds = sum(1 for i in range(-4, -1) if prices_5m[i] < prices_5m[i-1])
            ask_down = ask_vols[-3] > ask_vols[-2] > ask_vols[-1]
            bid_up = bid_vols[-3] < bid_vols[-2] < bid_vols[-1]
            
            analysis_data.update({
                'entry_after_correction': entry_signal,
                'recent_reds': recent_reds,
                'ask_down': ask_down,
                'bid_up': bid_up
            })
            
            print(f"  Recent red candles: {recent_reds}/3 (required: ≥2)")
            print(f"  Ask volume declining: {ask_down}")
            print(f"  Bid volume rising: {bid_up}")
            print(f"  Entry signal: {entry_signal}")
        
        # Step 5: Final scoring
        print("\nStep 5: Final scoring...")
        result = detect_sr_trend_mode(symbol)
        
        print(f"  Final score: {result.get('trend_score', 0)}/100")
        print(f"  Trend Mode: {result.get('trend_mode', False)}")
        print(f"  Entry Triggered: {result.get('entry_triggered', False)}")
        
        # Create debug log
        log_entry = create_debug_log(symbol, analysis_data, result)
        print(f"\n📝 Debug log saved to trend_debug_log.txt")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        result = {'trend_mode': False, 'trend_score': 0, 'description': f'Debug error: {str(e)}'}
        create_debug_log(symbol, analysis_data, result)
        return result

def batch_debug_symbols(symbols=None):
    """Debug multiple symbols to find patterns"""
    
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']
    
    print("🔍 Batch Debugging Trend Mode System")
    print("=" * 60)
    
    results = {}
    
    for symbol in symbols:
        print(f"\n🔍 Analyzing {symbol}...")
        result = debug_symbol_trend_mode(symbol)
        results[symbol] = result
        print(f"Result: Score {result.get('trend_score', 0)}, Mode: {result.get('trend_mode', False)}")
    
    # Summary
    print(f"\n📊 Batch Debug Summary:")
    print("=" * 30)
    
    total_symbols = len(results)
    trend_active = sum(1 for r in results.values() if r.get('trend_mode', False))
    
    print(f"Symbols analyzed: {total_symbols}")
    print(f"Trend mode active: {trend_active}")
    print(f"Success rate: {trend_active/total_symbols*100:.1f}%")
    
    # Show top scores
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('trend_score', 0), reverse=True)
    print(f"\nTop 3 scores:")
    for symbol, result in sorted_results[:3]:
        print(f"  {symbol}: {result.get('trend_score', 0)} points")
    
    return results

if __name__ == "__main__":
    print("🚀 Trend Mode Debugger")
    print("Choose option:")
    print("1. Debug single symbol")
    print("2. Batch debug multiple symbols")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        symbol = input("Enter symbol (e.g., BTCUSDT): ").strip().upper()
        debug_symbol_trend_mode(symbol)
    else:
        batch_debug_symbols()