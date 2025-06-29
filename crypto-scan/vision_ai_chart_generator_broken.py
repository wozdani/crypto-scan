"""
Vision-AI Chart Generator with Professional TradingView Styling
Complete implementation with matplotlib.dates for CLIP training optimization
"""

import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from typing import Optional, Dict, List

# Import candlestick_ohlc with proper error handling
try:
    from mplfinance.original_flavor import candlestick_ohlc
except ImportError:
    try:
        from matplotlib.finance import candlestick_ohlc
    except ImportError:
        # Manual candlestick implementation as fallback
        def candlestick_ohlc(ax, quotes, width=0.2, colorup='g', colordown='r'):
            """Simple candlestick implementation"""
            for quote in quotes:
                date, open_price, high, low, close = quote
                color = colorup if close >= open_price else colordown
                ax.plot([date, date], [low, high], color='black', linewidth=1)
                height = abs(close - open_price)
                bottom = min(open_price, close)
                ax.add_patch(plt.Rectangle((date - width/2, bottom), width, height, 
                                         facecolor=color, edgecolor='black', alpha=0.8))

def plot_chart_vision_ai(symbol: str, candles: List = None, alert_index: int = None, alert_indices: List = None, 
                        score: float = None, decision: str = None, phase: str = None, setup: str = None,
                        save_path: str = None, context_days: int = 2, force_fresh: bool = True) -> Optional[str]:
    """
    Generate Vision-AI chart with FRESH data - Fixed stale data issue
    
    Args:
        symbol: Trading symbol
        candles: List of candle data (if None, fetches fresh data)
        alert_index: Single alert index (backward compatibility)
        alert_indices: Multiple alert indices for memory training
        score: TJDE score
        decision: Trading decision
        phase: Market phase
        setup: Setup description
        save_path: Custom save path
        context_days: Days of context to include
        force_fresh: Force fresh data fetch regardless of cache
        
    Returns:
        Path to saved chart file
    """
    try:
        # CRITICAL FIX: Always fetch fresh data for chart generation
        if candles is None or force_fresh:
            print(f"[FRESH CHART DATA] 🔄 {symbol}: Fetching fresh candles for chart generation")
            from utils.fresh_candles import get_fresh_candles_for_charts, debug_candle_timestamps
            
            fresh_candles = get_fresh_candles_for_charts(symbol, interval="15m", limit=96)
            
            if fresh_candles:
                candles = fresh_candles
                print(f"[FRESH CHART DATA] ✅ {symbol}: Using {len(candles)} fresh candles")
                debug_candle_timestamps(candles, symbol)
            else:
                print(f"[FRESH CHART DATA] ❌ {symbol}: No fresh candles available - chart generation disabled")
                return None
        else:
            print(f"[CHART DATA] {symbol}: Using provided candles ({len(candles) if candles else 0} candles)")
            
            # Validate provided candles freshness
            if candles:
                from utils.fresh_candles import validate_candle_freshness
                if not validate_candle_freshness(candles, symbol, max_age_minutes=60):
                    print(f"[STALE DATA WARNING] {symbol}: Provided candles are stale - fetching fresh data")
                    from utils.fresh_candles import get_fresh_candles_for_charts
                    fresh_candles = get_fresh_candles_for_charts(symbol, interval="15m", limit=96)
                    if fresh_candles:
                        candles = fresh_candles
                        print(f"[FRESH OVERRIDE] ✅ {symbol}: Replaced stale data with {len(candles)} fresh candles")
        
        if not candles or len(candles) < 10:
            print(f"[CHART ERROR] {symbol}: Insufficient candle data for chart generation")
            return None
        
        print(f"[CHART VALIDATION] {symbol}: Processing {len(candles)} candles")
        
        # CRITICAL: Price range validation to prevent BANANAS31USDT-style flat charts
        price_samples = []
        valid_candles = []
        
        for candle in candles:
            try:
                if isinstance(candle, dict):
                    open_price = float(candle.get('open', 0))
                    high_price = float(candle.get('high', 0))
                    low_price = float(candle.get('low', 0))
                    close_price = float(candle.get('close', 0))
                elif isinstance(candle, list) and len(candle) >= 5:
                    open_price = float(candle[1])
                    high_price = float(candle[2])
                    low_price = float(candle[3])
                    close_price = float(candle[4])
                else:
                    continue
                
                # FIX 4: Enhanced OHLC data integrity validation
                if all(p > 0 for p in [open_price, high_price, low_price, close_price]):
                    # Comprehensive OHLC relationship validation
                    ohlc_valid = (
                        low_price <= high_price and  # Basic high >= low
                        min(open_price, close_price) >= low_price and  # Open/close >= low
                        max(open_price, close_price) <= high_price and  # Open/close <= high
                        abs(high_price - low_price) > 0  # Prevent zero-range candles
                    )
                    
                    # Additional sanity checks for extreme values
                    price_range = high_price - low_price
                    mid_price = (high_price + low_price) / 2
                    
                    # Reject candles with extreme price ratios or impossible ranges
                    if (ohlc_valid and 
                        price_range / mid_price < 1.0 and  # Max 100% range
                        high_price / low_price < 50):  # Max 50x ratio
                        price_samples.extend([open_price, high_price, low_price, close_price])
                        valid_candles.append(candle)
            except (ValueError, TypeError, IndexError):
                continue
        
        # FIX 4: Reduce strict candle requirements for better chart generation
        if len(valid_candles) < 10:
            print(f"[CHART ERROR] {symbol}: Insufficient valid candles - got {len(valid_candles)} valid from {len(candles)} total")
            return None
        
        # CRITICAL: Detect abnormal price ranges that create distorted charts
        if price_samples:
            min_price = min(price_samples)
            max_price = max(price_samples)
            price_range = max_price - min_price
            avg_price = sum(price_samples) / len(price_samples)
            
            relative_range = price_range / avg_price if avg_price > 0 else 0
            
            # Reject flat-line charts like BANANAS31USDT (0.008-0.008 range)
            if relative_range < 0.005:  # Less than 0.5% price movement
                print(f"[SKIP CHART] {symbol}: Abnormal price range - {price_range:.6f} ({relative_range*100:.3f}% of avg price) - likely distorted data")
                return None
            
            if price_range < 0.0001:  # Absolute minimum threshold
                print(f"[SKIP CHART] {symbol}: Price range too small: {price_range:.8f}")
                return None
                
            print(f"[CHART VALIDATION] {symbol}: ✅ Valid price range {min_price:.6f} - {max_price:.6f} (range: {price_range:.6f}, {relative_range*100:.2f}%)")
        
        candles = valid_candles
        
        # Prepare save path
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            phase_str = phase or "unknown"
            decision_str = decision or "unknown"
            save_path = f"training_data/charts/{symbol}_{timestamp}_{phase_str}_{decision_str}_vision.png"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # CRITICAL: Clear any existing plots to prevent overlay issues
        plt.clf()
        plt.cla()
        plt.close('all')
        
        # Configure matplotlib for professional output
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]}, 
                                      facecolor='#1e1e1e')
        
        # Prepare OHLC data
        ohlc_data = []
        volumes = []
        
        for i, candle in enumerate(candles):
            try:
                if isinstance(candle, dict):
                    # Dictionary format
                    timestamp = candle.get('timestamp', i)
                    open_price = float(candle.get('open', candle.get('Open', 1.0)))
                    high_price = float(candle.get('high', candle.get('High', 1.0)))
                    low_price = float(candle.get('low', candle.get('Low', 1.0)))
                    close_price = float(candle.get('close', candle.get('Close', 1.0)))
                    volume = float(candle.get('volume', candle.get('Volume', 1000.0)))
                elif isinstance(candle, list) and len(candle) >= 6:
                    # List format: [timestamp, open, high, low, close, volume]
                    timestamp = int(candle[0])
                    open_price = float(candle[1])
                    high_price = float(candle[2])
                    low_price = float(candle[3])
                    close_price = float(candle[4])
                    volume = float(candle[5])
                else:
                    continue
                    
                # Convert timestamp to matplotlib date format
                if timestamp > 1e12:  # Milliseconds
                    date_num = mdates.date2num(datetime.fromtimestamp(timestamp / 1000))
                else:  # Seconds
                    date_num = mdates.date2num(datetime.fromtimestamp(timestamp))
                
                ohlc_data.append([date_num, open_price, high_price, low_price, close_price])
                volumes.append(volume)
                
            except (ValueError, TypeError, KeyError) as e:
                print(f"[VISION-AI CHART] {symbol}: Skipping invalid candle {i}: {e}")
                continue
        
        if len(ohlc_data) < 10:
            print(f"[VISION-AI CHART] {symbol}: Insufficient valid OHLC data ({len(ohlc_data)})")
            return None
        
        # Plot candlesticks
        candlestick_ohlc(ax1, ohlc_data, width=0.0006, colorup='#00ff00', colordown='#ff3333')
        
        # Highlight alert points
        alert_positions = []
        if alert_indices:
            alert_positions = alert_indices
        elif alert_index is not None:
            alert_positions = [alert_index]
        
        for alert_pos in alert_positions:
            if 0 <= alert_pos < len(ohlc_data):
                # Current alert in lime green
                ax1.axvspan(ohlc_data[alert_pos][0], ohlc_data[alert_pos][0], 
                           alpha=0.15, color='lime', label='Alert')
                ax1.scatter(ohlc_data[alert_pos][0], ohlc_data[alert_pos][3], 
                           color='lime', s=100, marker='o', zorder=5)
        
        # Configure price chart
        ax1.set_facecolor('#0a0a0a')
        ax1.grid(True, alpha=0.3, color='#333333')
        ax1.set_title(f"{symbol} - {phase or 'Unknown Phase'} | Score: {score:.3f} | {decision or 'No Decision'}", 
                     fontsize=14, color='white', pad=20)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot volume
        date_nums = [ohlc[0] for ohlc in ohlc_data]
        colors = ['#00ff00' if ohlc_data[i][4] >= ohlc_data[i][1] else '#ff3333' 
                 for i in range(len(ohlc_data))]
        
        ax2.bar(date_nums, volumes, width=0.0006, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Highlight alert volume
        for alert_pos in alert_positions:
            if 0 <= alert_pos < len(volumes):
                ax2.axvspan(date_nums[alert_pos], date_nums[alert_pos], 
                           alpha=0.15, color='lime')
        
        # Configure volume chart
        ax2.set_facecolor('#0a0a0a')
        ax2.grid(True, alpha=0.3, color='#333333')
        ax2.set_ylabel('Volume', color='white')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Style improvements
        for ax in [ax1, ax2]:
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#333333')
            ax.spines['top'].set_color('#333333')
            ax.spines['right'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
        
        # Enhanced chart title with validation info
        title_info = f"Score: {score:.2f} | {decision.upper()}" if score and decision else "Training Data"
        fig.suptitle(f"{symbol} - Vision AI | {title_info} | Candles: {len(ohlc_data)}", 
                    fontsize=14, color='white')
        
        # Save chart with enhanced error handling
        try:
            plt.tight_layout()
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#1e1e1e')
            plt.close('all')  # Complete cleanup
            
            # Validate saved file
            if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:  # At least 1KB
                print(f"[VISION CHART] ✅ Saved quality chart: {save_path} ({os.path.getsize(save_path)} bytes)")
            else:
                print(f"[CHART ERROR] {symbol}: Generated file too small or missing")
                return None
                
        except Exception as e:
            print(f"[CHART ERROR] {symbol}: Save failed: {e}")
            plt.close('all')
            return None
        
        # Create metadata file
        metadata_path = save_path.replace('.png', '.json')
        metadata = {
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
            "tjde_score": score or 0.5,
            "tjde_decision": decision or "unknown",
            "market_phase": phase or "unknown",
            "setup_type": setup or f"{phase}_{decision}",
            "chart_path": save_path,
            "alert_count": len(alert_positions) if alert_positions else 1,
            "multi_alert": len(alert_positions) > 1 if alert_positions else False,
            "candles_count": len(ohlc_data),
            "vision_ai_optimized": True,
            "created_at": datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[VISION-AI CHART] {symbol}: Generated chart and metadata: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"[VISION-AI CHART ERROR] {symbol}: {e}")
        return None

def generate_training_chart_with_context(symbol: str, market_data: Dict, tjde_result: Dict, 
                                        output_dir: str = "training_data/charts") -> Optional[str]:
    """
    Generate training chart with complete market context for Vision-AI
    
    Args:
        symbol: Trading symbol
        market_data: Complete market data with candles
        tjde_result: TJDE analysis result
        output_dir: Output directory for charts
        
    Returns:
        Path to generated chart
    """
    try:
        candles = market_data.get('candles', [])
        if not candles or len(candles) < 10:
            print(f"[TRAINING CHART] {symbol}: Insufficient candle data")
            return None
        
        # Extract TJDE information
        tjde_score = tjde_result.get('score', 0)
        decision = tjde_result.get('decision', 'unknown')
        phase = tjde_result.get('market_phase', 'unknown')
        
        # Generate chart path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        chart_path = f"{output_dir}/{symbol}_{timestamp}_{phase}_{decision}_training.png"
        
        # Generate chart
        saved_path = plot_chart_vision_ai(
            symbol=symbol,
            candles=candles,
            score=tjde_score,
            decision=decision,
            phase=phase,
            setup=f"{phase}_{decision}",
            save_path=chart_path,
            context_days=2
        )
        
        return saved_path
        
    except Exception as e:
        print(f"[TRAINING CHART ERROR] {symbol}: {e}")
        return None

def validate_chart_requirements(candles: List, min_candles: int = 20) -> bool:
    """Validate if candle data meets chart generation requirements"""
    if not candles or len(candles) < min_candles:
        return False
    
    # Check for valid OHLC data in first few candles
    valid_count = 0
    for candle in candles[:min(5, len(candles))]:
        try:
            if isinstance(candle, dict):
                required_keys = ['open', 'high', 'low', 'close']
                if all(key in candle for key in required_keys):
                    valid_count += 1
            elif isinstance(candle, list) and len(candle) >= 5:
                valid_count += 1
        except:
            continue
    
    return valid_count >= min(3, len(candles))

if __name__ == "__main__":
    # Test chart generation
    print("Testing Vision-AI chart generator...")
    
    # Create sample data
    test_candles = []
    base_price = 1.0
    
    for i in range(50):
        timestamp = int(datetime.now().timestamp()) - (50-i) * 900  # 15min intervals
        price_change = (i % 10 - 5) * 0.01  # Simple price movement
        
        open_price = base_price + price_change
        high_price = open_price * 1.005
        low_price = open_price * 0.995
        close_price = open_price + price_change * 0.5
        volume = 1000 + (i % 20) * 100
        
        test_candles.append({
            'timestamp': timestamp * 1000,  # Milliseconds
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        base_price = close_price
    
    # Generate test chart
    result = plot_chart_vision_ai(
        symbol="TESTUSDT",
        candles=test_candles,
        alert_index=30,
        score=0.65,
        decision="consider_entry",
        phase="trend-following"
    )
    
    if result:
        print(f"Test chart generated: {result}")
    else:
        print("Test chart generation failed")