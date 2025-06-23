#!/usr/bin/env python3
"""
Test script for OnChain Insights module
Tests the descriptive on-chain analysis functionality
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from onchain_insights import OnChainAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_onchain_analyzer():
    """Test the OnChain Analyzer functionality"""
    
    print("🧪 Testing OnChain Analyzer Module")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = OnChainAnalyzer()
    print("✅ OnChain Analyzer initialized")
    
    # Test symbols to analyze
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
    
    for symbol in test_symbols:
        print(f"\n🔍 Analyzing on-chain activity for {symbol}")
        print("-" * 30)
        
        try:
            # Analyze on-chain activity
            insights = analyzer.analyze_onchain_activity(symbol, timeframe_hours=1)
            
            if insights:
                print(f"📊 Found {len(insights)} on-chain insights:")
                for i, insight in enumerate(insights, 1):
                    print(f"  {i}. [{insight.category}] {insight.message}")
                    print(f"     Confidence: {insight.confidence:.2f} | Source: {insight.source}")
                
                # Get formatted messages for GPT
                gpt_messages = analyzer.format_insights_for_gpt(insights)
                print(f"\n💬 GPT-formatted messages ({len(gpt_messages)}):")
                for msg in gpt_messages:
                    print(f"  • {msg}")
                
                # Get summary
                summary = analyzer.get_insights_summary(insights)
                print(f"\n📈 Summary:")
                print(f"  • Total insights: {summary['total_insights']}")
                print(f"  • Average confidence: {summary['avg_confidence']:.2f}")
                print(f"  • Categories: {summary['categories']}")
                print(f"  • Sources: {summary['sources']}")
                
            else:
                print("ℹ️ No significant on-chain activity detected")
                
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ OnChain Insights test completed")

def test_contract_lookup():
    """Test contract address lookup functionality"""
    
    print("\n🔍 Testing Contract Lookup")
    print("-" * 30)
    
    analyzer = OnChainAnalyzer()
    
    test_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'MATIC', 'UNI']
    
    for symbol in test_symbols:
        try:
            contract_info = analyzer._get_contract_info(symbol)
            if contract_info:
                print(f"✅ {symbol}: {contract_info['address'][:10]}... on {contract_info['chain']}")
            else:
                print(f"❌ {symbol}: No contract found")
        except Exception as e:
            print(f"⚠️ {symbol}: Error - {e}")

def test_descriptive_insights():
    """Test the descriptive insight generation"""
    
    print("\n💬 Testing Descriptive Insight Generation")
    print("-" * 40)
    
    analyzer = OnChainAnalyzer()
    
    # Test different insight scenarios
    test_scenarios = [
        {
            'name': 'Large whale transaction',
            'symbol': 'ETHUSDT',
            'expected_categories': ['whale_activity']
        },
        {
            'name': 'DEX inflow activity', 
            'symbol': 'UNIUSDT',
            'expected_categories': ['dex_inflow']
        },
        {
            'name': 'Bridge activity',
            'symbol': 'MATICUSDT', 
            'expected_categories': ['bridge_activity']
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        
        try:
            insights = analyzer.analyze_onchain_activity(scenario['symbol'], timeframe_hours=2)
            
            if insights:
                categories_found = set(insight.category for insight in insights)
                print(f"  Categories found: {categories_found}")
                
                # Check if we found expected categories
                expected = set(scenario['expected_categories'])
                if expected.intersection(categories_found):
                    print(f"  ✅ Found expected insights")
                else:
                    print(f"  ℹ️ Different insights found than expected")
                
                # Show sample messages
                messages = analyzer.format_insights_for_gpt(insights[:3])
                print(f"  Sample messages:")
                for msg in messages:
                    print(f"    • {msg}")
            else:
                print(f"  ℹ️ No insights generated for this scenario")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    """Main test function"""
    
    print("🚀 OnChain Insights Module Test Suite")
    print("====================================")
    
    # Check if API keys are available
    api_keys = {
        'ETHERSCAN_API_KEY': os.environ.get('ETHERSCAN_API_KEY'),
        'BSCSCAN_API_KEY': os.environ.get('BSCSCAN_API_KEY'),
        'POLYGONSCAN_API_KEY': os.environ.get('POLYGONSCAN_API_KEY'),
    }
    
    available_keys = [key for key, value in api_keys.items() if value]
    print(f"📋 Available API keys: {len(available_keys)}/3")
    
    if not available_keys:
        print("⚠️ Warning: No blockchain scanner API keys detected")
        print("   Some functionality may be limited to demo mode")
    
    # Run tests
    test_contract_lookup()
    test_onchain_analyzer()
    test_descriptive_insights()
    
    print("\n🏁 All tests completed!")
    print("\nℹ️ Note: Real on-chain data analysis requires:")
    print("  • Valid blockchain scanner API keys")
    print("  • Contract address mappings from crypto-scan cache")
    print("  • Active network connections to APIs")

if __name__ == "__main__":
    main()