#!/usr/bin/env python3
"""
Simplified Pump Analysis System - Pre-Pump 2.0 Compatible
No numpy/pandas dependencies - pure Python implementation
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv('../.env')

class SimplePumpAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        
    def analyze_market_data(self, symbol, days=7):
        """Analyze pump patterns using simple price calculations"""
        print(f"🔍 Analyzing {symbol} for pump patterns...")
        
        try:
            # Simple price analysis without external dependencies
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': '15',
                'limit': str(days * 96)  # 96 candles per day for 15min intervals
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                return None
                
            data = response.json()
            if not data.get('result', {}).get('list'):
                return None
                
            candles = data['result']['list']
            
            # Simple pump detection logic
            pumps_detected = []
            for i in range(len(candles) - 30):  # Look for 30-candle patterns
                current_candle = candles[i]
                previous_candles = candles[i+1:i+31]
                
                current_price = float(current_candle[4])  # Close price
                avg_price = sum(float(c[4]) for c in previous_candles) / len(previous_candles)
                
                price_increase = (current_price - avg_price) / avg_price * 100
                
                if price_increase >= 15.0:  # 15%+ pump detected
                    pump_info = {
                        'symbol': symbol,
                        'timestamp': current_candle[0],
                        'price_increase': round(price_increase, 2),
                        'current_price': current_price,
                        'avg_price': round(avg_price, 4)
                    }
                    pumps_detected.append(pump_info)
                    
            return pumps_detected
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    def generate_simple_report(self, symbol, pumps):
        """Generate simple analysis report"""
        if not pumps:
            return f"No significant pumps detected for {symbol}"
            
        report = f"🚀 PUMP ANALYSIS: {symbol}\n\n"
        for i, pump in enumerate(pumps[:3], 1):  # Show top 3 pumps
            dt = datetime.fromtimestamp(int(pump['timestamp'])/1000, tz=timezone.utc)
            report += f"Pump #{i}:\n"
            report += f"  Time: {dt.strftime('%Y-%m-%d %H:%M UTC')}\n"
            report += f"  Price increase: {pump['price_increase']}%\n"
            report += f"  Price: ${pump['current_price']}\n\n"
            
        return report
    
    def run_analysis(self, symbols=None):
        """Run simplified pump analysis"""
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']
            
        print("🚀 Starting Simplified Pump Analysis...")
        total_pumps = 0
        
        for symbol in symbols:
            pumps = self.analyze_market_data(symbol)
            if pumps:
                total_pumps += len(pumps)
                report = self.generate_simple_report(symbol, pumps)
                print(report)
                
                # Save to file
                os.makedirs('reports', exist_ok=True)
                with open(f'reports/{symbol}_pump_analysis.txt', 'w') as f:
                    f.write(report)
                    
        print(f"✅ Analysis complete. Found {total_pumps} pump patterns.")
        return total_pumps

def main():
    """Main function for simplified pump analysis"""
    analyzer = SimplePumpAnalyzer()
    
    # Run analysis on major symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT', 
               'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT', 'LTCUSDT']
    
    try:
        total_pumps = analyzer.run_analysis(symbols)
        print(f"\n🎯 Summary: Analyzed {len(symbols)} symbols, found {total_pumps} pump patterns")
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis stopped by user")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()