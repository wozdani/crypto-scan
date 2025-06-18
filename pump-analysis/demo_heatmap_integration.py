#!/usr/bin/env python3
"""
Demo: Heatmap integration with GPT prompts
Shows how orderbook heatmap analysis appears in GPT contexts
"""

import logging
from main import GPTAnalyzer
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_heatmap_prompt():
    """Demonstrate heatmap integration in GPT prompts"""
    
    # Create sample pre-pump data
    sample_data = {
        'symbol': 'BTCUSDT',
        'pump_start_time': '2025-06-18 18:45:00',
        'pump_increase_pct': 22.3,
        'pre_pump_period': '60 minutes',
        'price_volatility': 0.0028,
        'price_trend': 'accumulation',
        'price_compression': {
            'compression_ratio_pct': 1.5,
            'is_compressed': True
        },
        'avg_volume': 180000,
        'volume_trend': 'increasing',
        'volume_spikes': [
            {'time_minutes_before_pump': 18, 'volume_multiplier': 2.1},
            {'time_minutes_before_pump': 6, 'volume_multiplier': 3.4}
        ],
        'rsi': 48.7,
        'vwap': {
            'vwap_value': 0.025400,
            'price_vs_vwap_pct': 1.1,
            'above_vwap': True
        },
        'fake_rejects': [
            {'time_minutes_before_pump': 22, 'wick_size_pct': 3.8, 'recovery_strength': 78.0}
        ],
        'support_resistance': {
            'key_support': 0.025200,
            'key_resistance': 0.026000
        },
        'liquidity_gaps': [
            {'type': 'upward_gap', 'size_pct': 1.2, 'time_minutes_before_pump': 14}
        ],
        'onchain_insights': [
            "Wykryto transfer wieloryba o wartości $52,000 na giełdę 12 minut przed pumpem",
            "Napływ DEX +18% w ostatnich 25 minutach"
        ]
    }
    
    # Check if we have API key for full demo
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if openai_key:
        try:
            # Create GPT analyzer
            gpt_analyzer = GPTAnalyzer(openai_key)
            
            # Generate prompt with heatmap integration
            prompt = gpt_analyzer._format_analysis_prompt(sample_data)
            
            # Extract heatmap section
            lines = prompt.split('\n')
            heatmap_section = []
            in_heatmap = False
            
            for line in lines:
                if "=== ANALIZA HEATMAPY ORDERBOOKU ===" in line:
                    in_heatmap = True
                    heatmap_section.append(line)
                elif in_heatmap and line.startswith("===") and "HEATMAP" not in line:
                    break
                elif in_heatmap:
                    heatmap_section.append(line)
            
            if heatmap_section:
                print("\n" + "="*60)
                print("PRZYKŁAD INTEGRACJI HEATMAPY W PROMPTACH GPT")
                print("="*60)
                for line in heatmap_section:
                    print(line)
                print("="*60)
                print("✅ Heatmapa jest dodawana jako opisowy kontekst")
                print("✅ GPT otrzymuje sygnały strukturalne orderbooku")
                print("✅ System rozpoznaje dostępność danych heatmapy")
            else:
                print("❌ Sekcja heatmapy nie została znaleziona")
                
        except Exception as e:
            print(f"❌ Błąd demonstracji: {e}")
    else:
        print("Demo template - heatmap integration format:")
        print("""
=== ANALIZA HEATMAPY ORDERBOOKU ===
• Zniknięcie ścian podaży: TAK (ask side, -42.1%)
• Pinning ceny do płynności: NIE
• Reakcja ceny na pustkę (void): TAK (ruch: 2.3%)
• Nachylenie klastrów wolumenu: bullish (siła: 31.5%)
• Kontekst strukturalny: Strukturalne sygnały orderbooku dostępne

UWAGA: Powyższe sygnały heatmapy traktuj jako dodatkowy kontekst strukturalny, 
nie jako decydujące warunki. Użyj ich do wzbogacenia analizy pre-pump.
        """)

if __name__ == "__main__":
    demo_heatmap_prompt()