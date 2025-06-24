"""
Custom Candlestick Chart Generation for Trend-Mode System
Migration from mplfinance to matplotlib + candlestick_ohlc for Vision-AI training
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
from mplfinance.original_flavor import candlestick_ohlc


def plot_custom_candlestick_chart(df_ohlc: pd.DataFrame, df_volume: pd.DataFrame, 
                                title: str, save_path: str, 
                                clip_confidence: Optional[float] = None, 
                                tjde_score: Optional[float] = None,
                                market_phase: Optional[str] = None,
                                decision: Optional[str] = None) -> str:
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
        
        # Świece z enhanced styling
        candlestick_ohlc(ax1, df_ohlc_copy[['timestamp', 'open', 'high', 'low', 'close']].values, 
                        width=0.0008, colorup='#00FF88', colordown='#FF4444', alpha=0.8)

        # Volume z gradient effect
        colors = ['#00AA44' if close >= open else '#AA2222' 
                 for close, open in zip(df_ohlc['close'], df_ohlc['open'])]
        ax2.bar(df_volume['timestamp'], df_volume['volume'], 
               width=0.0008, color=colors, alpha=0.6)

        # Enhanced styling
        ax1.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
        ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax2.grid(True, linestyle='--', alpha=0.2, color='gray')
        
        # Format x-axis
        ax1.xaxis_date()
        ax2.xaxis_date()
        fig.autofmt_xdate()

        # Professional metadata overlay
        metadata_y = 0.95
        if tjde_score is not None:
            # Color-coded TJDE score
            tjde_color = '#00FF00' if tjde_score >= 0.75 else '#FFFF00' if tjde_score >= 0.5 else '#FF6600'
            ax1.text(0.02, metadata_y, f'TJDE Score: {tjde_score:.3f}', 
                    transform=ax1.transAxes, fontsize=10, fontweight='bold',
                    color=tjde_color, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='black', alpha=0.7))
            metadata_y -= 0.06

        if clip_confidence is not None:
            # CLIP confidence with reliability indicator
            clip_color = '#00FFFF' if clip_confidence >= 0.7 else '#FFAA00'
            ax1.text(0.02, metadata_y, f'CLIP Confidence: {clip_confidence:.3f}', 
                    transform=ax1.transAxes, fontsize=10, fontweight='bold',
                    color=clip_color, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='black', alpha=0.7))
            metadata_y -= 0.06

        if market_phase is not None:
            # Market phase with phase-specific colors
            phase_colors = {
                'trend-following': '#00AA00',
                'breakout': '#FF8800', 
                'pullback': '#0088FF',
                'consolidation': '#AA00AA',
                'distribution': '#FF0000'
            }
            phase_color = phase_colors.get(market_phase.lower(), '#FFFFFF')
            ax1.text(0.02, metadata_y, f'Phase: {market_phase.upper()}', 
                    transform=ax1.transAxes, fontsize=10, fontweight='bold',
                    color=phase_color, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='black', alpha=0.7))
            metadata_y -= 0.06

        if decision is not None:
            # Decision with action-specific colors
            decision_colors = {
                'join_trend': '#00FF00',
                'consider_entry': '#FFFF00',
                'watch_closely': '#FF8800',
                'avoid': '#FF0000'
            }
            decision_color = decision_colors.get(decision.lower(), '#FFFFFF')
            ax1.text(0.02, metadata_y, f'Decision: {decision.upper()}', 
                    transform=ax1.transAxes, fontsize=10, fontweight='bold',
                    color=decision_color, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='black', alpha=0.7))

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

        # Volume label
        ax2.set_ylabel('Volume', color='white', fontweight='bold')
        ax1.set_ylabel('Price', color='white', fontweight='bold')

        # Save with high quality
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), 
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
            timestamp = datetime.fromtimestamp(candle[0] / 1000)
            
            ohlc_data.append({
                'timestamp': timestamp,
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4])
            })
            
            volume_data.append({
                'timestamp': mdates.date2num(timestamp),
                'volume': float(candle[5])
            })
        
        df_ohlc = pd.DataFrame(ohlc_data)
        df_volume = pd.DataFrame(volume_data)
        
        return df_ohlc, df_volume
        
    except Exception as e:
        print(f"[CHART DATA ERROR] Failed to prepare dataframes: {e}")
        return None, None


def generate_trend_mode_chart(symbol: str, candles_15m: List, tjde_result: Dict,
                            output_dir: str = "training_charts") -> Optional[str]:
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
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        phase_short = market_phase.replace('-', '')[:8]
        decision_short = decision.replace('_', '')[:4]
        
        filename = f"{symbol}_{timestamp}_{phase_short}_{decision_short}_trend.png"
        save_path = os.path.join(output_dir, filename)
        
        # Create title
        title = f"{symbol} - Trend Mode Analysis"
        
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