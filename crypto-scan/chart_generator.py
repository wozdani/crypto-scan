#!/usr/bin/env python3
"""
Alert-Focused Chart Generator for TJDE Training
Generuje wykresy treningowe skupione na momencie alertu z kontekstem decyzyjnym
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def detect_alert_moment(candles_15m, tjde_score=None, tjde_decision=None):
    """
    Wykrywa dokładny moment alertu TJDE w danych świecowych
    
    Args:
        candles_15m: Dane świec 15-minutowych
        tjde_score: Score TJDE dla kontekstu
        tjde_decision: Decyzja TJDE
        
    Returns:
        Index świecy, która wygenerowała alert
    """
    if not candles_15m or len(candles_15m) < 10:
        return max(0, len(candles_15m) - 1)
    
    try:
        # Szukamy ostatniego znaczącego momentu decision-making
        recent_length = min(15, len(candles_15m))
        recent_candles = candles_15m[-recent_length:]
        
        closes = [float(c[4]) for c in recent_candles]
        volumes = [float(c[5]) for c in recent_candles]
        highs = [float(c[2]) for c in recent_candles]
        lows = [float(c[3]) for c in recent_candles]
        
        # 1. Szukamy volume spike + price action
        avg_volume = sum(volumes[:-3]) / len(volumes[:-3]) if len(volumes) > 3 else sum(volumes) / len(volumes)
        
        # 2. Priorytet dla ostatnich 5 świec z volume spike
        for i in range(len(recent_candles) - 5, len(recent_candles)):
            if i >= 0 and volumes[i] > avg_volume * 1.3:
                return len(candles_15m) - (len(recent_candles) - i)
        
        # 3. Szukamy breakout lub bounce pattern
        for i in range(len(recent_candles) - 3, len(recent_candles)):
            if i > 0:
                price_move = abs(closes[i] - closes[i-1]) / closes[i-1]
                if price_move > 0.01:  # 1%+ move
                    return len(candles_15m) - (len(recent_candles) - i)
        
        # 4. Default - ostatnie 2-3 świece
        return len(candles_15m) - 2
        
    except Exception as e:
        print(f"[ALERT DETECTION ERROR] {e}")
        return max(0, len(candles_15m) - 2)


def generate_alert_focused_training_chart(
    symbol: str, 
    candles_15m: List, 
    tjde_score: float, 
    tjde_phase: str, 
    tjde_decision: str, 
    tjde_clip_confidence: float = None, 
    setup_label: str = None
) -> Optional[str]:
    """
    Generuje wykres treningowy skupiony na momencie alertu TJDE
    
    KONTEKST DECYZYJNY:
    - 100 świec przed alertem, 20 po (maks 120 total)
    - Oznaczenie momentu alertu pionową linią i kropką
    - Tytuł z pełnym kontekstem: PHASE | SETUP | SCORE | DECISION
    - Kolor tła/ramki zależny od fazy rynku
    
    Args:
        symbol: Symbol tradingowy
        candles_15m: Dane świec 15-minutowych
        tjde_score: Final score TJDE 
        tjde_phase: Faza rynku z TJDE
        tjde_decision: Decyzja TJDE
        tjde_clip_confidence: CLIP confidence (opcjonalne)
        setup_label: Opis setupu (opcjonalne)
        
    Returns:
        Ścieżka do wygenerowanego wykresu lub None
    """
    try:
        if not candles_15m or len(candles_15m) < 20:
            print(f"[CHART WARNING] {symbol}: Za mało danych świecowych ({len(candles_15m) if candles_15m else 0})")
            return None
        
        # 1. WYKRYJ MOMENT ALERTU
        alert_idx = detect_alert_moment(candles_15m, tjde_score, tjde_decision)
        print(f"[ALERT MOMENT] {symbol}: Alert wykryty na świecy {alert_idx}/{len(candles_15m)}")
        
        # 2. KONTEKST OKNO: 100 przed, 20 po alertem
        context_before = 100
        context_after = 20
        
        start_idx = max(0, alert_idx - context_before)
        end_idx = min(len(candles_15m), alert_idx + context_after)
        
        context_candles = candles_15m[start_idx:end_idx]
        alert_relative_idx = alert_idx - start_idx  # Pozycja alertu w kontekście
        
        print(f"[CONTEXT WINDOW] {symbol}: Świece {start_idx}-{end_idx} (total: {len(context_candles)}, alert na: {alert_relative_idx})")
        
        if len(context_candles) < 10:
            print(f"[CHART ERROR] {symbol}: Za mało świec w kontekście ({len(context_candles)})")
            return None
        
        # 3. PRZYGOTUJ DANE DLA WYKRESU
        df_data = []
        for candle in context_candles:
            timestamp = datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc)
            df_data.append({
                'Date': timestamp,
                'Open': float(candle[1]),
                'High': float(candle[2]),
                'Low': float(candle[3]),
                'Close': float(candle[4]),
                'Volume': float(candle[5])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        
        # 4. KOLORY FAZOWE DLA KONTEKSTU WIZUALNEGO
        phase_colors = {
            "trend-following": "#2E8B57",      # Zielony - trend
            "pullback-in-trend": "#4682B4",   # Niebieski - pullback  
            "breakout-continuation": "#FF8C00", # Pomarańczowy - breakout
            "consolidation": "#9370DB",        # Fioletowy - konsolidacja
            "trend-reversal": "#DC143C",       # Czerwony - reversal
            "fakeout": "#8B0000",             # Ciemno czerwony - fakeout
            "bullish-momentum": "#32CD32",     # Lime - momentum wzrostowy
            "bearish-momentum": "#FF6347",     # Tomato - momentum spadkowy
            "exhaustion": "#800080",          # Purple - wyczerpanie
            "volume-backed": "#FFD700",       # Gold - volume spike
            "no-trend": "#708090",            # Szary - brak trendu
            "unknown": "#696969"              # Ciemno szary - nieznane
        }
        
        phase_color = phase_colors.get(tjde_phase, phase_colors["unknown"])
        
        # 5. GENERUJ WYKRES Z KONTEKSTEM
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                       gridspec_kw={'height_ratios': [4, 1]}, 
                                       facecolor='white')
        
        # Candlestick chart z volume
        mpf.plot(df, type='candle', style='charles', 
                 ax=ax1, volume=ax2, 
                 show_nontrading=False,
                 warn_too_much_data=False)
        
        # 6. OZNACZ MOMENT ALERTU
        if 0 <= alert_relative_idx < len(df):
            alert_time = df.index[alert_relative_idx]
            alert_price = df.iloc[alert_relative_idx]['Close']
            alert_high = df.iloc[alert_relative_idx]['High']
            
            # Pionowa linia alertu
            ax1.axvline(x=alert_relative_idx, color='red', linestyle='--', linewidth=3, alpha=0.8, label='🚨 ALERT MOMENT')
            
            # Kropka na świecy alertu
            ax1.scatter(alert_relative_idx, alert_price, color='red', s=150, zorder=10, 
                       marker='o', edgecolors='darkred', linewidths=2, label='TJDE ENTRY')
            
            # Strzałka wskazująca alert
            ax1.annotate('ALERT HERE', xy=(alert_relative_idx, alert_high), 
                        xytext=(alert_relative_idx, alert_high * 1.02),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, fontweight='bold', color='red', ha='center')
        
        # 7. TYTUŁ Z PEŁNYM KONTEKSTEM DECYZYJNYM
        setup_text = f" | SETUP: {setup_label.upper()}" if setup_label else ""
        clip_text = f" | CLIP: {tjde_clip_confidence:.3f}" if tjde_clip_confidence and tjde_clip_confidence > 0 else ""
        
        title_main = f"{symbol} | PHASE: {tjde_phase.upper()}{setup_text}"
        title_metrics = f"SCORE: {tjde_score:.3f} | DECISION: {tjde_decision.upper()}{clip_text}"
        
        ax1.set_title(title_main, fontsize=14, fontweight='bold', color=phase_color, pad=10)
        ax1.text(0.5, 0.98, title_metrics, transform=ax1.transAxes, fontsize=12, 
                ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor=phase_color, alpha=0.3))
        
        # 8. RAMKA FAZOWA
        for spine in ax1.spines.values():
            spine.set_color(phase_color)
            spine.set_linewidth(3)
        
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax2.set_title('Volume', fontsize=10)
        
        # 9. ZAPISZ WYKRES
        os.makedirs("training_charts", exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Nazwa pliku z kontekstem
        phase_short = tjde_phase.replace('-', '').replace('_', '')[:8]
        decision_short = tjde_decision.replace('_', '')[:4]
        filename = f"{symbol}_{timestamp_str}_{phase_short}_{decision_short}_alert.png"
        filepath = os.path.join("training_charts", filename)
        
        plt.tight_layout()
        
        # Save with proper parameters (linewidth not supported in savefig)
        fig = plt.gcf()
        fig.patch.set_edgecolor(phase_color)
        fig.patch.set_linewidth(3)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 10. METADATA Z KONTEKSTEM ALERTU
        metadata = {
            "symbol": symbol,
            "timestamp": timestamp_str,
            "chart_type": "alert_focused_training",
            "tjde_analysis": {
                "score": tjde_score,
                "phase": tjde_phase,
                "decision": tjde_decision,
                "setup_label": setup_label
            },
            "clip_analysis": {
                "confidence": tjde_clip_confidence
            },
            "alert_context": {
                "alert_candle_index": alert_idx,
                "alert_relative_position": alert_relative_idx,
                "context_window_start": start_idx,
                "context_window_end": end_idx,
                "total_context_candles": len(context_candles),
                "candles_before_alert": context_before,
                "candles_after_alert": context_after
            },
            "visual_cues": {
                "phase_color": phase_color,
                "alert_marked": True,
                "has_volume": True
            }
        }
        
        metadata_path = filepath.replace('.png', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ [ALERT CHART] {symbol}: Wykres skupiony na momencie alertu → {filepath}")
        print(f"📊 [CONTEXT] Alert na świecy {alert_idx}, kontekst: {len(context_candles)} świec")
        
        return filepath
        
    except Exception as e:
        print(f"❌ [CHART ERROR] {symbol}: Błąd generowania wykresu alertu: {e}")
        return None


def generate_tjde_training_chart(
    symbol: str, 
    candles_15m: List, 
    candles_5m: List = None, 
    tjde_result: Dict = None,
    clip_info: Dict = None,
    output_dir: str = "training_charts"
) -> Optional[str]:
    """
    Generate TJDE-based training chart with complete analysis overlay
    
    Args:
        symbol: Trading symbol
        candles_15m: 15-minute candle data (used for chart display)
        candles_5m: 5-minute candle data (optional, for additional analysis)  
        tjde_result: Complete TJDE result dictionary with score, decision, breakdown
        clip_info: CLIP analysis results with phase and confidence
        output_dir: Output directory for charts
        
    Returns:
        Path to generated chart or None if failed
    """
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Validate inputs
        if not candles_15m or len(candles_15m) < 10:
            print(f"[CHART ERROR] {symbol}: Insufficient 15m candles ({len(candles_15m) if candles_15m else 0})")
            return None
            
        # Extract TJDE data
        if not tjde_result or not isinstance(tjde_result, dict):
            print(f"[CHART ERROR] {symbol}: Missing or invalid TJDE result")
            return None
            
        tjde_score = tjde_result.get("final_score", 0.0)
        decision = tjde_result.get("decision", "unknown")
        market_phase = tjde_result.get("market_phase", "unknown")
        setup_type = tjde_result.get("setup_type", "unknown")
        
        # Extract CLIP data if available
        clip_phase = "N/A"
        clip_confidence = 0.0
        if clip_info and isinstance(clip_info, dict):
            clip_phase = clip_info.get("predicted_phase", "N/A")
            clip_confidence = clip_info.get("confidence", 0.0)
            
        # Use last 96 candles (24 hours at 15M intervals)
        candles_to_plot = candles_15m[-96:] if len(candles_15m) >= 96 else candles_15m
        
        # Prepare data for plotting
        timestamps = [candle[0] for candle in candles_to_plot]
        opens = [float(candle[1]) for candle in candles_to_plot]
        highs = [float(candle[2]) for candle in candles_to_plot]
        lows = [float(candle[3]) for candle in candles_to_plot]
        closes = [float(candle[4]) for candle in candles_to_plot]
        volumes = [float(candle[5]) for candle in candles_to_plot]
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        for i in range(len(timestamps)):
            color = 'green' if closes[i] >= opens[i] else 'red'
            ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1)
            ax1.plot([i, i], [opens[i], closes[i]], color=color, linewidth=3)
        
        # Create TJDE-focused title
        title = f"{symbol} – TJDE Training Chart"
        ax1.set_title(title, fontsize=16, fontweight='bold')
        
        # Create analysis text overlay
        analysis_text = f"""Phase: {market_phase}
Setup: {setup_type}
TJDE Score: {tjde_score:.3f}
Decision: {decision.upper()}"""
        
        if clip_confidence > 0:
            analysis_text += f"\nCLIP: {clip_phase} ({clip_confidence:.2f})"
            
        # Add TJDE analysis overlay in top-right corner
        ax1.text(0.98, 0.98, analysis_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.9),
                family='monospace')
        
        # Volume chart
        ax2.bar(range(len(volumes)), volumes, alpha=0.7, color='blue')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time (15M candles)')
        
        # Format and save
        ax1.set_ylabel('Price (USDT)')
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{timestamp}_tjde_chart.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[CHART SUCCESS] {symbol}: TJDE chart saved to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"[CHART ERROR] {symbol}: Failed to generate chart - {e}")
        import traceback
        traceback.print_exc()
        return None


def detect_alert_point(candles_15m, tjde_score=None):
    """
    Detect the optimal alert point in candle data based on price action
    
    Args:
        candles_15m: 15-minute candle data
        tjde_score: Optional TJDE score for weighting
        
    Returns:
        Index of alert point in candles array
    """
    try:
        if not candles_15m or len(candles_15m) < 20:
            return len(candles_15m) - 10 if candles_15m else 0
            
        closes = [float(c[4]) for c in candles_15m]
        volumes = [float(c[5]) for c in candles_15m]
        
        # Calculate price momentum and volume spikes
        alert_scores = []
        for i in range(10, len(closes) - 5):
            # Price momentum (recent vs past)
            recent_price = sum(closes[i-3:i+1]) / 4
            past_price = sum(closes[i-10:i-6]) / 4
            price_momentum = abs(recent_price - past_price) / past_price if past_price > 0 else 0
            
            # Volume spike
            recent_vol = volumes[i]
            avg_vol = sum(volumes[i-10:i]) / 10 if i >= 10 else recent_vol
            volume_spike = recent_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Combined alert score
            alert_score = price_momentum * 0.7 + min(volume_spike - 1, 2.0) * 0.3
            alert_scores.append((i, alert_score))
        
        # Find best alert point (highest combined score)
        if alert_scores:
            best_point = max(alert_scores, key=lambda x: x[1])
            return best_point[0]
        
        # Fallback: use 75% through the data
        return int(len(candles_15m) * 0.75)
        
    except Exception as e:
        print(f"[ALERT POINT ERROR] {e}")
        return len(candles_15m) - 10 if candles_15m else 0


def generate_tjde_training_chart_contextual(symbol, candles_15m, tjde_score, tjde_phase, tjde_decision, tjde_clip_confidence=None, setup_label=None):
    """
    Generate context-aware TJDE training chart focused on alert moment
    
    Args:
        symbol: Trading symbol
        candles_15m: 15-minute candle data
        tjde_score: TJDE final score
        tjde_phase: Market phase from TJDE
        tjde_decision: TJDE decision
        tjde_clip_confidence: Optional CLIP confidence
        setup_label: Optional setup description
        
    Returns:
        Path to generated chart file or None
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime, timezone
        import os
        import numpy as np

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        folder = "training_charts"
        os.makedirs(folder, exist_ok=True)

        # Detect alert point in the data
        alert_index = detect_alert_point(candles_15m, tjde_score)
        
        # Extract context window: 100 candles before, 20 after alert
        context_before = 100
        context_after = 20
        
        start_idx = max(0, alert_index - context_before)
        end_idx = min(len(candles_15m), alert_index + context_after)
        
        context_candles = candles_15m[start_idx:end_idx]
        alert_position = alert_index - start_idx  # Position of alert in context window
        
        if len(context_candles) < 10:
            print(f"[CHART ERROR] {symbol}: Insufficient context data")
            return None
        
        # Extract OHLCV data
        timestamps = [datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc) for c in context_candles]
        opens = [float(c[1]) for c in context_candles]
        highs = [float(c[2]) for c in context_candles]
        lows = [float(c[3]) for c in context_candles]
        closes = [float(c[4]) for c in context_candles]
        volumes = [float(c[5]) for c in context_candles]

        # Determine phase colors
        phase_colors = {
            'trend-following': '#4CAF50',    # Green
            'pullback': '#2196F3',           # Blue  
            'breakout': '#FF9800',           # Orange
            'consolidation': '#9C27B0',      # Purple
            'reversal': '#F44336',           # Red
            'accumulation': '#607D8B'        # Blue Grey
        }
        
        primary_color = phase_colors.get(tjde_phase.lower(), '#2196F3')
        
        # Create figure with enhanced layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                      gridspec_kw={'height_ratios': [4, 1]},
                                      facecolor='white')
        
        # Price chart with candlesticks
        for i in range(len(timestamps)):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            
            # Highlight alert area
            if i == alert_position:
                color = primary_color
                alpha = 1.0
                linewidth = 3
            else:
                alpha = 0.8
                linewidth = 1.5
                
            # Candlestick wicks
            ax1.plot([timestamps[i], timestamps[i]], [lows[i], highs[i]], 
                    color=color, linewidth=linewidth, alpha=alpha)
            
            # Candlestick bodies
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            width = 0.6 if i == alert_position else 0.4
            
            ax1.bar(timestamps[i], body_height, bottom=body_bottom, 
                   color=color, width=width/24, alpha=alpha)

        # Add trend line
        ax1.plot(timestamps, closes, color=primary_color, linewidth=1, alpha=0.6)
        
        # Mark alert point
        if alert_position < len(timestamps):
            ax1.axvline(x=timestamps[alert_position], color=primary_color, 
                       linestyle='--', linewidth=2, alpha=0.8, label='Alert Point')
            ax1.scatter(timestamps[alert_position], closes[alert_position], 
                       color=primary_color, s=100, zorder=5)

        # Enhanced title with full context
        title = f"{symbol} | {tjde_phase.upper()} | TJDE: {tjde_score:.2f}"
        if setup_label:
            title += f" | {setup_label.upper()}"
        ax1.set_title(title, fontsize=14, weight='bold', color=primary_color)

        # Create comprehensive annotation
        annotation_text = f"PHASE: {tjde_phase}\nSETUP: {setup_label or 'N/A'}\nTJDE: {tjde_score:.3f}\nDECISION: {tjde_decision}"
        if tjde_clip_confidence is not None:
            annotation_text += f"\nCLIP: {tjde_clip_confidence:.3f}"
        
        # Add phase-colored background box
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor=primary_color, alpha=0.15, edgecolor=primary_color, linewidth=2)
        ax1.text(0.02, 0.98, annotation_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', fontweight='bold',
                bbox=bbox_props, family='monospace')

        # Volume chart with alert highlighting
        colors_vol = [primary_color if i == alert_position else '#64B5F6' for i in range(len(volumes))]
        ax2.bar(timestamps, volumes, color=colors_vol, alpha=0.7, width=0.5/24)
        ax2.axvline(x=timestamps[alert_position], color=primary_color, 
                   linestyle='--', linewidth=2, alpha=0.8)

        # Format axes
        ax1.set_ylabel('Price (USDT)', fontweight='bold')
        ax2.set_ylabel('Volume', fontweight='bold')
        ax2.set_xlabel('Time (Alert Context)', fontweight='bold')
        
        # Format time axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()

        # Generate filename with context info
        filename = f"{symbol}_{timestamp}_{tjde_phase}_{tjde_decision}_tjde.png"
        filepath = os.path.join(folder, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[CONTEXTUAL CHART] {symbol}: Alert context saved to {filepath}")
        print(f"[CHART CONTEXT] Alert at candle {alert_position}/{len(context_candles)}, Score: {tjde_score:.3f}")
        
        return filepath
        
    except Exception as e:
        print(f"[CONTEXTUAL CHART ERROR] {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_tjde_training_chart_simple(symbol, price_series, tjde_score, tjde_phase, tjde_decision, tjde_clip_confidence=None, setup_label=None):
    """
    Fallback: Generate simplified TJDE training chart 
    """
    try:
        import matplotlib.pyplot as plt
        from datetime import datetime
        import os

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        folder = "training_charts"
        os.makedirs(folder, exist_ok=True)

        label = f"{tjde_phase.upper()} | TJDE: {round(tjde_score, 3)}"
        if tjde_clip_confidence is not None:
            label += f" | CLIP: {round(tjde_clip_confidence, 3)}"
        if setup_label:
            label += f" | Setup: {setup_label}"

        plt.figure(figsize=(10, 5))
        plt.plot(price_series, linewidth=1.5, color='#2196F3')
        plt.title(f"{symbol} - TJDE Chart", fontsize=14, weight='bold')
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)

        plt.gca().annotate(label,
                           xy=(0.02, 0.95), xycoords='axes fraction',
                           fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1))

        filename = f"{symbol}_{timestamp}_tjde.png"
        filepath = os.path.join(folder, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[TJDE CHART] {symbol}: Saved to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"[TJDE CHART ERROR] {symbol}: {e}")
        return None


def flatten_candles(candles_15m, candles_5m=None):
    """
    Flatten candle data to price series for chart plotting
    
    Args:
        candles_15m: 15-minute candles
        candles_5m: Optional 5-minute candles
        
    Returns:
        List of close prices
    """
    try:
        price_series = []
        
        # Use 15m candles as primary data
        if candles_15m:
            price_series.extend([float(candle[4]) for candle in candles_15m])
        
        # Optionally add 5m candles for more detail
        if candles_5m and len(price_series) < 200:
            price_5m = [float(candle[4]) for candle in candles_5m]
            price_series.extend(price_5m)
        
        # Limit to reasonable size for visualization
        if len(price_series) > 300:
            price_series = price_series[-300:]
            
        return price_series
        
    except Exception as e:
        print(f"[FLATTEN ERROR] Failed to flatten candles: {e}")
        return []
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Validate candle data
        if not candles_15m or len(candles_15m) < 10:
            print(f"[CHART ERROR] {symbol}: Insufficient 15M candles ({len(candles_15m) if candles_15m else 0})")
            return None
            
        # Use last 96 candles (24 hours at 15M intervals)
        candles_to_plot = candles_15m[-96:] if len(candles_15m) >= 96 else candles_15m
        
        # Extract data for plotting
        times = [datetime.fromtimestamp(c[0] / 1000) for c in candles_to_plot]
        opens = [float(c[1]) for c in candles_to_plot]
        highs = [float(c[2]) for c in candles_to_plot]
        lows = [float(c[3]) for c in candles_to_plot]
        closes = [float(c[4]) for c in candles_to_plot]
        volumes = [float(c[5]) for c in candles_to_plot]
        
        # Create figure with professional layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                      gridspec_kw={'height_ratios': [4, 1]},
                                      facecolor='white')
        
        # Price chart with candlesticks
        for i in range(len(times)):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            
            # Candlestick wicks
            ax1.plot([times[i], times[i]], [lows[i], highs[i]], 
                    color=color, linewidth=1.5, alpha=0.8)
            
            # Candlestick bodies
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            ax1.bar(times[i], body_height, bottom=body_bottom, 
                   color=color, width=0.0008, alpha=0.9)
        
        # Add trend line for context
        ax1.plot(times, closes, color='#2196f3', linewidth=1, alpha=0.6, label='Close Price')
        
        # Format decision color
        decision_colors = {
            'join_trend': '#4caf50',
            'consider_entry': '#ff9800', 
            'avoid': '#f44336',
            'long': '#4caf50',
            'short': '#f44336'
        }
        decision_color = decision_colors.get(decision.lower(), '#757575')
        
        # Chart title with TJDE results
        title = f"{symbol} | TJDE: {tjde_score:.3f} ({decision.upper()})"
        if tjde_breakdown:
            title += f" | Trend: {tjde_breakdown.get('trend_strength', 0):.2f}"
        
        ax1.set_title(title, fontsize=14, fontweight='bold', color=decision_color)
        ax1.set_ylabel("Price (USDT)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Format time axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Volume chart
        volume_colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' for i in range(len(times))]
        ax2.bar(times, volumes, color=volume_colors, alpha=0.7, width=0.0008)
        ax2.set_ylabel("Volume", fontsize=10)
        ax2.set_xlabel("Time (UTC)", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format volume axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add TJDE score annotation
        score_text = f"TJDE Score: {tjde_score:.3f}\nDecision: {decision.upper()}"
        if tjde_breakdown:
            score_text += f"\nTrend: {tjde_breakdown.get('trend_strength', 0):.2f}"
            score_text += f"\nPullback: {tjde_breakdown.get('pullback_quality', 0):.2f}"
        
        ax1.text(0.02, 0.98, score_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor=decision_color, alpha=0.1),
                verticalalignment='top', fontsize=10)
        
        # Save chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{timestamp}_chart.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"[CHART SUCCESS] {symbol}: Professional chart saved to {filepath}")
        
        # Save metadata JSON
        _save_chart_metadata(symbol, timestamp, tjde_score, decision, tjde_breakdown, output_dir)
        
        return filepath
        
    except Exception as e:
        print(f"[CHART ERROR] {symbol}: Failed to generate chart - {e}")
        return None


def _save_chart_metadata(
    symbol: str, 
    timestamp: str, 
    tjde_score: float, 
    decision: str,
    tjde_breakdown: Dict = None,
    output_dir: str = "training_charts"
) -> bool:
    """Save chart metadata as JSON for training purposes"""
    try:
        metadata = {
            "symbol": symbol,
            "timestamp": timestamp,
            "chart_file": f"{symbol}_{timestamp}_chart.png",
            "tjde_score": tjde_score,
            "decision": decision,
            "tjde_breakdown": tjde_breakdown or {},
            "created_at": datetime.now().isoformat(),
            "chart_type": "trend_mode_training",
            "candle_timeframe": "15M",
            "quality_check": {
                "min_candles": True,
                "valid_ohlcv": True,
                "tjde_calculated": tjde_score is not None
            }
        }
        
        metadata_file = os.path.join(output_dir, f"{symbol}_{timestamp}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"[METADATA] {symbol}: Saved metadata to {metadata_file}")
        return True
        
    except Exception as e:
        print(f"[METADATA ERROR] {symbol}: {e}")
        return False


def validate_chart_quality(candles_15m: List, candles_5m: List = None) -> Dict[str, bool]:
    """
    Quality checklist for chart generation
    
    Returns:
        Dictionary with quality validation results
    """
    quality = {
        "sufficient_candles_15m": len(candles_15m) >= 20 if candles_15m else False,
        "sufficient_candles_5m": len(candles_5m) >= 60 if candles_5m else True,  # Optional
        "valid_ohlcv_data": False,
        "no_nan_values": False,
        "price_variance": False
    }
    
    if candles_15m and len(candles_15m) >= 10:
        try:
            # Check OHLCV data validity
            sample_candle = candles_15m[-1]
            if len(sample_candle) >= 6:
                quality["valid_ohlcv_data"] = True
                
            # Check for NaN values
            closes = [float(c[4]) for c in candles_15m[-10:]]
            quality["no_nan_values"] = all(not (c != c) for c in closes)  # NaN check
            
            # Check price variance (not flat line)
            if closes:
                price_std = (max(closes) - min(closes)) / max(closes) if max(closes) > 0 else 0
                quality["price_variance"] = price_std > 0.001  # At least 0.1% price movement
                
        except Exception:
            pass
    
    return quality


def generate_chart_async_safe(
    symbol: str,
    market_data: Dict,
    tjde_result: Dict,
    tjde_breakdown: Dict = None
) -> Optional[str]:
    """
    Async-safe chart generation for integration with scan_token_async
    
    Args:
        symbol: Trading symbol
        market_data: Market data dictionary with candles
        tjde_result: TJDE result with score and decision
        tjde_breakdown: Optional detailed TJDE breakdown
        
    Returns:
        Path to generated chart or None
    """
    try:
        candles_15m = market_data.get("candles", [])
        candles_5m = market_data.get("candles_5m", [])
        
        # Quality validation
        quality = validate_chart_quality(candles_15m, candles_5m)
        if not quality["sufficient_candles_15m"]:
            print(f"[CHART SKIP] {symbol}: Insufficient candles for quality chart")
            return None
            
        # Extract TJDE results
        tjde_score = tjde_result.get("final_score", 0.0)
        decision = tjde_result.get("decision", "unknown")
        
        # Generate chart
        chart_path = generate_alert_focused_training_chart(
            symbol=symbol,
            candles_15m=candles_15m,
            tjde_score=tjde_score,
            decision=decision,
            tjde_breakdown=tjde_breakdown,
            output_dir="training_charts"
        )
        
        return chart_path
        
    except Exception as e:
        print(f"[CHART ASYNC ERROR] {symbol}: {e}")
        return None


def test_chart_generation():
    """Test chart generation with sample data"""
    print("Testing chart generation...")
    
    # Generate sample candle data
    import time
    base_time = int(time.time() * 1000)
    base_price = 100.0
    
    sample_candles = []
    for i in range(96):
        timestamp = base_time - (96 - i) * 15 * 60 * 1000  # 15 minutes intervals
        price_change = (i % 10 - 5) * 0.5  # Small price movements
        
        open_price = base_price + price_change
        close_price = open_price + (i % 3 - 1) * 0.2
        high_price = max(open_price, close_price) + 0.1
        low_price = min(open_price, close_price) - 0.1
        volume = 1000 + (i % 20) * 50
        
        sample_candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
    
    # Test chart generation
    result = generate_alert_focused_training_chart(
        symbol="TESTUSDT",
        candles_15m=sample_candles,
        tjde_score=0.75,
        decision="consider_entry",
        tjde_breakdown={
            "trend_strength": 0.8,
            "pullback_quality": 0.7,
            "support_reaction": 0.6
        }
    )
    
    if result:
        print(f"✅ Test chart generated: {result}")
        # Clean up test file
        try:
            os.remove(result)
            metadata_file = result.replace("_chart.png", "_metadata.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            print("✅ Test files cleaned up")
        except:
            pass
    else:
        print("❌ Test chart generation failed")


if __name__ == "__main__":
    test_chart_generation()