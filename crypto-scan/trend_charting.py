"""
Vision-AI Optimized Chart Generation for CLIP Training
Professional TradingView-style charts with clean dark theme
"""

import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
import matplotlib.ticker as mticker

# Import candlestick_ohlc with robust fallback
try:
    from mplfinance.original_flavor import candlestick_ohlc
except ImportError:
    try:
        from matplotlib.finance import candlestick_ohlc
    except ImportError:
        def candlestick_ohlc(ax, quotes, width=0.2, colorup='g', colordown='r'):
            """Simple candlestick implementation fallback"""
            for quote in quotes:
                date, open_price, high, low, close = quote
                color = colorup if close >= open_price else colordown
                ax.plot([date, date], [low, high], color='black', linewidth=1)
                height = abs(close - open_price)
                bottom = min(open_price, close)
                ax.add_patch(plt.Rectangle((date - width/2, bottom), width, height, 
                                         facecolor=color, edgecolor='black', alpha=0.8))


def plot_chart_with_context(symbol, candles, alert_indices=None, alert_index=None, score=None, decision=None, phase=None, setup=None, save_path="chart.png", context_days=2):
    """
    Enhanced chart generation with extended context for better Vision-AI training
    
    Args:
        symbol: Trading symbol
        candles: List of candle data with timestamp, open, high, low, close, volume
        alert_indices: List of historical alert indices for memory learning
        alert_index: Single alert index for backward compatibility
        score: TJDE score
        decision: Trading decision
        phase: Market phase
        setup: Setup description
        save_path: Path to save chart
        context_days: Days of historical context to include (default 2)
        
    Returns:
        Path to saved chart
    """
    try:
        # Enhanced context - use more historical data for better pattern recognition
        context_candles = int(context_days * 96)  # 96 candles per day (15m intervals)
        display_candles = candles[-context_candles:] if len(candles) > context_candles else candles
        
        # Prepare OHLC data with enhanced timestamp validation
        ohlc = []
        timestamps_for_debug = []
        
        for i, c in enumerate(display_candles):
            try:
                if isinstance(c, dict):
                    # Validate timestamp format
                    timestamp_raw = c.get('timestamp', 0)
                    if timestamp_raw > 1e12:  # Milliseconds
                        timestamp = timestamp_raw / 1000
                    else:  # Already in seconds
                        timestamp = timestamp_raw
                    
                    ts = mdates.date2num(datetime.fromtimestamp(timestamp))
                    timestamps_for_debug.append(datetime.fromtimestamp(timestamp))
                    ohlc.append((ts, c['open'], c['high'], c['low'], c['close']))
                elif isinstance(c, list) and len(c) >= 5:
                    # Handle list format [timestamp, open, high, low, close, volume]
                    timestamp_raw = int(c[0])
                    if timestamp_raw > 1e12:  # Milliseconds
                        timestamp = timestamp_raw / 1000
                    else:  # Already in seconds
                        timestamp = timestamp_raw
                    
                    ts = mdates.date2num(datetime.fromtimestamp(timestamp))
                    timestamps_for_debug.append(datetime.fromtimestamp(timestamp))
                    ohlc.append((ts, float(c[1]), float(c[2]), float(c[3]), float(c[4])))
            except (ValueError, OSError, OverflowError) as e:
                print(f"[CHART WARNING] {symbol}: Invalid timestamp at index {i}: {e}")
                continue
        
        if len(ohlc) < 10:
            print(f"[CHART ERROR] {symbol}: Insufficient valid OHLC data after timestamp validation: {len(ohlc)}")
            return None
            
        # Debug timestamp range
        if timestamps_for_debug:
            print(f"[CHART TIMESTAMPS] {symbol}: {timestamps_for_debug[0]} → {timestamps_for_debug[-1]}")

        volume = []
        timestamps = []
        for c in display_candles:
            try:
                if isinstance(c, dict):
                    vol = c.get('volume', 0)
                    if vol > 0:  # Only include positive volumes
                        volume.append(vol)
                        timestamp_raw = c.get('timestamp', 0)
                        timestamp = timestamp_raw / 1000 if timestamp_raw > 1e12 else timestamp_raw
                        timestamps.append(datetime.fromtimestamp(timestamp))
                elif isinstance(c, list) and len(c) >= 6:
                    vol = float(c[5])
                    if vol > 0:  # Only include positive volumes
                        volume.append(vol)
                        timestamp_raw = int(c[0])
                        timestamp = timestamp_raw / 1000 if timestamp_raw > 1e12 else timestamp_raw
                        timestamps.append(datetime.fromtimestamp(timestamp))
            except (ValueError, OSError, OverflowError, IndexError):
                continue
        
        if not volume:
            print(f"[CHART WARNING] {symbol}: No valid volume data available")
            volume = [1] * len(ohlc)  # Use minimal volume as fallback

        # CRITICAL: Clear any existing plots to prevent overlay issues
        plt.clf()
        plt.cla()
        plt.close('all')  # Ensure clean slate
        
        # Professional styling
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        plt.style.use("dark_background")

        fig, (ax_main, ax_volume) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), 
                                               gridspec_kw={'height_ratios': [3, 1]})
        fig.tight_layout(pad=2)
        
        print(f"[CHART RENDER] {symbol}: Creating clean chart with {len(ohlc)} price points")

        # Professional candlesticks
        candlestick_ohlc(ax_main, ohlc, width=0.4, colorup='#00ff00', colordown='#ff3333', alpha=0.9)

        # Memory-aware multi-alert highlighting system
        alerts_to_highlight = []
        
        # Support both new alert_indices and legacy alert_index
        if alert_indices:
            alerts_to_highlight = alert_indices
        elif alert_index is not None:
            alerts_to_highlight = [alert_index]
        
        # Highlight multiple historical alerts for memory learning
        if alerts_to_highlight:
            for i, idx in enumerate(alerts_to_highlight):
                if idx < len(ohlc):
                    # Color coding: current alert = lime, historical = orange/yellow
                    is_current = (i == len(alerts_to_highlight) - 1)
                    color = 'lime' if is_current else 'orange'
                    alpha = 0.20 if is_current else 0.12
                    
                    ax_main.axvspan(
                        ohlc[idx][0] - 0.25, 
                        ohlc[idx][0] + 0.25,
                        color=color, 
                        alpha=alpha,
                        label=f"Current Alert" if is_current and i == 0 else None
                    )
                    
                    # Add alert markers on volume chart for better visibility
                    if idx < len(volume):
                        ax_volume.axvline(x=timestamps[idx], color=color, linestyle='--', 
                                        alpha=0.8, linewidth=1.5)

        # Professional volume bars
        ax_volume.bar(timestamps, volume, color='steelblue', edgecolor='black', 
                     alpha=0.7, width=0.0008)

        # Enhanced title with context information
        score_text = f"{score:.3f}" if score is not None else "0.000"
        phase_text = phase.upper() if phase else ''
        setup_text = setup.upper() if setup else ''
        context_text = f"{context_days}D" if context_days != 2 else "2D"
        title = f"{symbol} | {phase_text} | TJDE: {score_text} | {setup_text} | {context_text} Context"
        ax_main.set_title(title, fontsize=12, pad=10)

        # Clean grid and formatting
        ax_main.grid(True, linestyle='--', alpha=0.3)
        ax_volume.grid(True, linestyle='--', alpha=0.2)
        ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()

        # Labels
        ax_main.set_ylabel('Price (USDT)', fontsize=10)
        ax_volume.set_ylabel('Volume', fontsize=10)
        ax_volume.set_xlabel('Time', fontsize=10)

        # Save with high quality for Vision-AI
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create directory if path has directory component
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='black')
        plt.close()
        
        # Export training metadata to JSON for enhanced AI training
        import json
        from datetime import datetime as dt
        
        metadata = {
            "symbol": symbol,
            "alerts": alert_indices if alert_indices else ([] if alert_index is None else [alert_index]),
            "phase": phase,
            "setup": setup,
            "decision": decision,
            "score": score,
            "timestamp": dt.utcnow().isoformat(),
            "chart_type": "vision_ai_training",
            "multi_alert": len(alert_indices) > 1 if alert_indices else False,
            "alert_count": len(alert_indices) if alert_indices else (1 if alert_index is not None else 0),
            "context_days": context_days
        }
        
        json_path = save_path.replace(".png", ".json")
        try:
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[VISION-AI] Metadata saved: {json_path}")
        except Exception as e:
            print(f"[VISION-AI] Failed to save metadata: {e}")
        
        # Generate GPT commentary if score is significant
        if score and score >= 0.5:
            try:
                from gpt_commentary import generate_chart_commentary
                
                # Basic TJDE data for commentary
                commentary = generate_chart_commentary(
                    save_path, score, decision or "unknown", None, symbol
                )
                
                if commentary:
                    print(f"[GPT COMMENTARY] Generated for {symbol}")
                    
            except ImportError:
                pass
            except Exception as e:
                print(f"[GPT COMMENTARY ERROR] {e}")
        
        return save_path
        
    except Exception as e:
        print(f"[VISION-AI CHART ERROR] {e}")
        plt.close('all')
        return None


def plot_custom_candlestick_chart(df_ohlc: pd.DataFrame, df_volume: pd.DataFrame, 
                                title: str, save_path: str, 
                                clip_confidence: Optional[float] = None, 
                                tjde_score: Optional[float] = None,
                                market_phase: Optional[str] = None,
                                decision: Optional[str] = None,
                                tjde_breakdown: Optional[Dict] = None,
                                alert_sent: Optional[bool] = None) -> str:
    """
    Rysuje wykres świecowy i zapisuje jako PNG z pełną kontrolą wizualizacji.

    Args:
        df_ohlc: DataFrame z kolumnami ['timestamp', 'open', 'high', 'low', 'close']
        df_volume: DataFrame z kolumnami ['timestamp', 'volume']
        title: Tytuł wykresu (np. 'BTCUSDT - Trend Mode')
        save_path: Ścieżka zapisu do pliku PNG
        clip_confidence: (opcjonalne) confidence z CLIP
        tjde_score: (opcjonalne) score z simulate_trader_decision_advanced()
        market_phase: (opcjonalne) faza rynku
        decision: (opcjonalne) decyzja TJDE
        
    Returns:
        Path to saved chart
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                     gridspec_kw={'height_ratios': [4, 1]})
        
        # Konwersja timestampów dla candlestick_ohlc
        df_ohlc_copy = df_ohlc.copy()
        df_ohlc_copy['timestamp'] = mdates.date2num(df_ohlc_copy['timestamp'])
        
        # UPGRADE A: Świece - zmniejszenie szerokości i grubości  
        candlestick_ohlc(ax1, df_ohlc_copy[['timestamp', 'open', 'high', 'low', 'close']].values, 
                        width=0.4, colorup='#00FF88', colordown='#FF4444', alpha=0.9)

        # UPGRADE B: Volume - wyraźniejsze kolory i krawędzie
        colors = ['steelblue' if close >= open else '#AA2222' 
                 for close, open in zip(df_ohlc['close'], df_ohlc['open'])]
        ax2.bar(df_volume['timestamp'], df_volume['volume'], 
               width=0.6, align='center', edgecolor='black', alpha=0.7, color=colors)

        # 1. MARKET PHASE BACKGROUND COLORS
        phase_backgrounds = {
            'trend-following': '#001100',    # Dark green
            'accumulation': '#000011',       # Dark blue
            'distribution': '#110000',       # Dark red
            'breakout': '#111100',          # Dark yellow
            'pullback': '#001111',          # Dark cyan
            'consolidation': '#110011'       # Dark magenta
        }
        
        phase_bg = phase_backgrounds.get(market_phase.lower() if market_phase else 'unknown', '#1a1a1a')
        ax1.set_facecolor(phase_bg)
        ax2.set_facecolor(phase_bg)
        
        # Enhanced styling
        ax1.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
        ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax2.grid(True, linestyle='--', alpha=0.2, color='gray')
        
        # Format x-axis
        ax1.xaxis_date()
        ax2.xaxis_date()
        fig.autofmt_xdate()
        
        # 2. SCORING ANNOTATIONS - Left side comprehensive
        if tjde_breakdown:
            annotations_y = 0.95
            annotation_spacing = 0.05
            
            # TJDE Score with color coding
            tjde_color = '#00FF00' if tjde_score >= 0.75 else '#FFFF00' if tjde_score >= 0.5 else '#FF6600'
            ax1.text(0.02, annotations_y, f'TJDE: {tjde_score:.3f}', 
                    transform=ax1.transAxes, fontsize=10, fontweight='bold',
                    color=tjde_color, bbox=dict(boxstyle="round,pad=0.2", 
                    facecolor='black', alpha=0.8))
            annotations_y -= annotation_spacing
            
            # Individual component scores
            components = [
                ('Trend', tjde_breakdown.get('trend_strength', 0)),
                ('Pullback', tjde_breakdown.get('pullback_quality', 0)), 
                ('Support', tjde_breakdown.get('support_reaction_strength', 0)),
                ('Volume', tjde_breakdown.get('volume_behavior_score', 0)),
                ('Psych', tjde_breakdown.get('psych_score', 0))
            ]
            
            for comp_name, comp_score in components:
                comp_color = '#00AA00' if comp_score >= 0.7 else '#AAAA00' if comp_score >= 0.4 else '#AA4400'
                ax1.text(0.02, annotations_y, f'{comp_name}: {comp_score:.2f}', 
                        transform=ax1.transAxes, fontsize=9,
                        color=comp_color, bbox=dict(boxstyle="round,pad=0.15", 
                        facecolor='black', alpha=0.7))
                annotations_y -= (annotation_spacing - 0.01)
        
        # 3. CLIP PHASE ANNOTATION - Right side top
        if clip_confidence is not None and clip_confidence > 0:
            clip_color = '#00FFFF' if clip_confidence >= 0.7 else '#FFAA00'
            ax1.text(0.98, 0.95, f'CLIP: {market_phase}', 
                    transform=ax1.transAxes, fontsize=10, fontweight='bold',
                    color=clip_color, bbox=dict(boxstyle="round,pad=0.2", 
                    facecolor='black', alpha=0.8), ha='right')
            
            ax1.text(0.98, 0.90, f'Conf: {clip_confidence:.3f}', 
                    transform=ax1.transAxes, fontsize=9,
                    color=clip_color, bbox=dict(boxstyle="round,pad=0.15", 
                    facecolor='black', alpha=0.7), ha='right')
        
        # 4. GRADIENT SCORING BAR - Right edge
        if tjde_score is not None:
            # Create gradient effect
            x_positions = df_ohlc['timestamp'].values
            x_max = x_positions[-1]
            y_min, y_max = ax1.get_ylim()
            
            # Calculate offset for gradient bar
            x_range = x_positions[-1] - x_positions[0]
            x_offset = x_range * 0.01
            
            # Gradient from red to green based on score
            gradient_color = (1-tjde_score, tjde_score, 0.1)  # RGB with slight blue
            
            # Vertical gradient bar
            gradient_height = (y_max - y_min) * tjde_score
            bar_x1 = x_max + x_offset
            bar_x2 = x_max + (x_offset * 1.5)
            
            ax1.fill_between([bar_x1, bar_x2], 
                           y_min, y_min + gradient_height,
                           color=gradient_color, alpha=0.8, 
                           label=f'Score: {tjde_score:.1%}')
        
        # 5. DECISION ARROWS AND ENTRY POINTS
        if tjde_score is not None and tjde_score >= 0.7:
            # Find optimal entry point (highest volume in recent candles)
            recent_candles = min(10, len(df_ohlc))
            entry_idx = len(df_ohlc) - recent_candles + np.argmax(df_volume['volume'][-recent_candles:])
            
            entry_x = df_ohlc.iloc[entry_idx]['timestamp']
            entry_y = df_ohlc.iloc[entry_idx]['high']
            entry_y_offset = entry_y + (entry_y * 0.02)  # 2% above high
            
            arrow_color = '#00FF00' if decision == 'join_trend' else '#FFFF00'
            ax1.annotate('🎯 ENTRY', xy=(entry_x, entry_y), 
                        xytext=(entry_x, entry_y_offset),
                        arrowprops=dict(facecolor=arrow_color, shrink=0.05, width=2),
                        fontsize=10, fontweight='bold', color=arrow_color,
                        ha='center')
        
        # UPGRADE C: Podświetlenie świecy alertowej
        if alert_sent is not None and alert_sent:
            # Znajdź moment alertu (najwyższy wolumen w ostatnich 20% danych)
            volume_data = df_volume['volume'].values
            alert_start_idx = int(len(volume_data) * 0.8)
            if alert_start_idx < len(volume_data):
                local_max_idx = np.argmax(volume_data[alert_start_idx:]) + alert_start_idx
                if local_max_idx < len(df_volume):
                    alert_timestamp = df_volume.iloc[local_max_idx]['timestamp']
                    # Highlight alert candle with green background
                    ax1.axvspan(alert_timestamp - 0.5, alert_timestamp + 0.5, color='green', alpha=0.15)
                    # Alert lines on both charts
                    ax1.axvline(x=alert_timestamp, linestyle='--', color='green', alpha=0.7, linewidth=2)
                    ax2.axvline(x=alert_timestamp, linestyle='--', color='green', alpha=0.7, linewidth=2)
            
            # 6. Alert status indicator
            alert_text = "ALERT SENT" if alert_sent else "NO ALERT"
            alert_color = '#00FF00' if alert_sent else '#FF4444'
            
            ax1.text(0.98, 0.05, alert_text, 
                    transform=ax1.transAxes, fontsize=11, fontweight='bold',
                    color=alert_color, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8), ha='right')

        # UPGRADE D: Profesjonalny styl boxa z fazą/setupem/decision
        info_text = f"TJDE Score: {tjde_score:.3f}\nPhase: {market_phase}\nDecision: {decision}"
        if clip_confidence is not None:
            info_text += f"\nCLIP Confidence: {clip_confidence:.3f}"
        
        ax1.text(0.02, 0.15, info_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4', alpha=0.8),
                family='monospace')

        # Professional dark theme
        fig.patch.set_facecolor('#1a1a1a')
        ax1.set_facecolor('#1a1a1a')
        ax2.set_facecolor('#1a1a1a')

        # White text for readability
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')

        # UPGRADE E: Dodaj interwał świec do tytułu 
        interval = "15M"
        if hasattr(title, 'split') and ' - ' in title:
            symbol_part = title.split(' - ')[0]
        else:
            symbol_part = title
        enhanced_title = f"{symbol_part} | {interval} | {market_phase.upper() if market_phase else 'UNKNOWN'} | TJDE: {tjde_score:.2f} | {decision.upper() if decision else 'UNKNOWN'}"
        ax1.set_title(enhanced_title, fontsize=12, fontweight='bold', color='white', pad=20)

        # UPGRADE F: Włącz siatkę i lepszy font
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # Format axes  
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.grid(True, which='major', linestyle='--', alpha=0.3)
        ax2.grid(True, which='major', linestyle='--', alpha=0.3)

        # Volume label
        ax2.set_ylabel('Volume', color='white', fontweight='bold')
        ax1.set_ylabel('Price (USDT)', color='white', fontweight='bold')
        ax2.set_xlabel('Time', color='white', fontweight='bold')

        # 6. ✍️ Dodaj podpis auto-labelingu jako tekst pod wykresem
        if tjde_breakdown and isinstance(tjde_breakdown, dict):
            breakdown_text = f"Trend: {tjde_breakdown.get('trend_strength', 0):.2f} | "
            breakdown_text += f"Pullback: {tjde_breakdown.get('pullback_quality', 0):.2f} | "
            breakdown_text += f"Support: {tjde_breakdown.get('support_reaction_strength', 0):.2f} | "
            breakdown_text += f"Volume: {tjde_breakdown.get('volume_behavior_score', 0):.2f} | "
            breakdown_text += f"Psychology: {tjde_breakdown.get('psych_score', 0):.2f}"
            
            ax2.text(0.5, -0.25, breakdown_text, transform=ax2.transAxes, 
                    fontsize=10, ha='center', style='italic', color='lightgray',
                    bbox=dict(facecolor='lightgray', alpha=0.5, pad=5))

        # Save with high quality
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), 
                   bbox_inches='tight', edgecolor='none')
        plt.close(fig)
        
        print(f"[CUSTOM CHART] Saved: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"[CUSTOM CHART ERROR] Failed to generate chart: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def prepare_ohlcv_dataframes(candles: List) -> tuple:
    """
    Prepare OHLCV data from candle list for custom chart generation
    
    Args:
        candles: List of candle data [timestamp, open, high, low, close, volume]
        
    Returns:
        Tuple of (df_ohlc, df_volume)
    """
    try:
        if not candles or len(candles) < 5:
            raise ValueError("Insufficient candle data")
            
        # Limit to reasonable number of candles for chart clarity
        chart_candles = candles[-100:] if len(candles) > 100 else candles
        
        # Create OHLC DataFrame
        ohlc_data = []
        volume_data = []
        
        for candle in chart_candles:
            try:
                # Support multiple candle formats: dict, list, tuple
                if isinstance(candle, dict):
                    timestamp_raw = candle.get('timestamp', candle.get('time', 0))
                    open_price = candle.get('open', candle.get('o', 0))
                    high_price = candle.get('high', candle.get('h', 0))
                    low_price = candle.get('low', candle.get('l', 0))
                    close_price = candle.get('close', candle.get('c', 0))
                    volume = candle.get('volume', candle.get('v', 0))
                elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                    timestamp_raw = candle[0]
                    open_price = candle[1]
                    high_price = candle[2]
                    low_price = candle[3]
                    close_price = candle[4]
                    volume = candle[5]
                else:
                    print(f"[CHART WARNING] Skipping candle with unknown format: {type(candle)}")
                    continue
                
                # Handle timestamp conversion
                if timestamp_raw > 1e12:  # Milliseconds
                    timestamp = datetime.fromtimestamp(timestamp_raw / 1000)
                else:  # Seconds
                    timestamp = datetime.fromtimestamp(timestamp_raw)
                
                ohlc_data.append({
                    'timestamp': timestamp,
                    'open': float(open_price),
                    'high': float(high_price),
                    'low': float(low_price),
                    'close': float(close_price)
                })
                
                volume_data.append({
                    'timestamp': mdates.date2num(timestamp),
                    'volume': float(volume)
                })
                
            except (ValueError, OSError, OverflowError, TypeError, KeyError) as e:
                print(f"[CHART WARNING] Skipping invalid candle: {e}")
                continue
        
        df_ohlc = pd.DataFrame(ohlc_data)
        df_volume = pd.DataFrame(volume_data)
        
        return df_ohlc, df_volume
        
    except Exception as e:
        print(f"[CHART DATA ERROR] Failed to prepare dataframes: {e}")
        return None, None


def generate_trend_mode_chart(symbol: str, candles_15m: List, tjde_result: Dict,
                            output_dir: str = "training_data/charts", 
                            alert_sent: bool = False) -> Optional[str]:
    """
    Generate custom trend-mode chart for Vision-AI training
    
    Args:
        symbol: Trading symbol
        candles_15m: 15-minute candle data
        tjde_result: TJDE analysis result
        output_dir: Output directory for charts
        
    Returns:
        Path to generated chart or None
    """
    try:
        # Prepare data
        df_ohlc, df_volume = prepare_ohlcv_dataframes(candles_15m)
        
        if df_ohlc is None or df_volume is None:
            print(f"[TREND CHART] {symbol}: Failed to prepare data")
            return None
        
        # Extract TJDE information
        tjde_score = tjde_result.get('final_score', 0.0)
        market_phase = tjde_result.get('market_phase', 'unknown')
        decision = tjde_result.get('decision', 'unknown')
        clip_confidence = tjde_result.get('clip_confidence', None)
        tjde_breakdown = tjde_result.get('breakdown', {})
        
        # Generate enhanced filename with TJDE score
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        score_str = f"score{tjde_score:.2f}".replace('.', '')
        
        filename = f"{symbol}_{timestamp}_{score_str}.png"
        save_path = os.path.join(output_dir, filename)
        
        # Create enhanced title with interval
        interval = "15M"
        title = f"{symbol} | {interval} | {market_phase.upper()} | TJDE: {tjde_score:.2f} | {decision.upper()}"
        
        # Generate chart
        chart_path = plot_custom_candlestick_chart(
            df_ohlc=df_ohlc,
            df_volume=df_volume,
            title=title,
            save_path=save_path,
            clip_confidence=clip_confidence,
            tjde_score=tjde_score,
            market_phase=market_phase,
            decision=decision
        )
        
        return chart_path
        
    except Exception as e:
        print(f"[TREND CHART ERROR] {symbol}: {e}")
        return None


def test_custom_charting():
    """Test custom charting system"""
    print("[CUSTOM CHART TEST] Testing custom candlestick generation...")
    
    # Generate sample data
    timestamps = pd.date_range(start='2025-01-01', periods=50, freq='15T')
    base_price = 100
    
    ohlc_data = []
    volume_data = []
    
    for i, ts in enumerate(timestamps):
        # Simulate price movement
        open_price = base_price + np.random.normal(0, 1)
        close_price = open_price + np.random.normal(0, 2)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 1))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 1))
        volume = np.random.uniform(1000, 5000)
        
        ohlc_data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        })
        
        volume_data.append({
            'timestamp': mdates.date2num(ts),
            'volume': volume
        })
        
        base_price = close_price
    
    df_ohlc = pd.DataFrame(ohlc_data)
    df_volume = pd.DataFrame(volume_data)
    
    # Test chart generation
    test_path = plot_custom_candlestick_chart(
        df_ohlc=df_ohlc,
        df_volume=df_volume,
        title="TESTUSDT - Custom Chart Test",
        save_path="test_charts/TESTUSDT_custom_test.png",
        clip_confidence=0.85,
        tjde_score=0.74,
        market_phase="trend-following",
        decision="consider_entry"
    )
    
    if test_path:
        print(f"✅ Custom chart test successful: {test_path}")
    else:
        print("❌ Custom chart test failed")
    
    return test_path


if __name__ == "__main__":
    test_custom_charting()