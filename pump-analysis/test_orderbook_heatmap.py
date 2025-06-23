#!/usr/bin/env python3
"""
Test module for orderbook heatmap simulation system
Demonstrates wall disappearance, liquidity pinning, void reactions, and cluster tilts
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from modules import (
    OrderbookHeatmapAnalyzer, 
    HeatmapFeatures, 
    OrderbookSnapshot, 
    OrderbookLevel,
    BybitOrderbookFetcher,
    get_heatmap_manager
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderbookHeatmapTester:
    """Test suite for orderbook heatmap analysis"""
    
    def __init__(self):
        self.analyzer = OrderbookHeatmapAnalyzer()
        self.test_symbol = "BTCUSDT"
        
    def create_synthetic_orderbook(self, symbol: str, base_price: float, 
                                 bid_depth: float, ask_depth: float,
                                 timestamp: datetime = None) -> OrderbookSnapshot:
        """Create synthetic orderbook data for testing"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate bid levels (below current price)
        bids = []
        for i in range(20):
            price = base_price - (i + 1) * 0.01  # $0.01 increments below
            size = bid_depth * (1 - i * 0.05)  # Decreasing depth
            if size > 0:
                bids.append(OrderbookLevel(price, size, timestamp))
        
        # Generate ask levels (above current price)
        asks = []
        for i in range(20):
            price = base_price + (i + 1) * 0.01  # $0.01 increments above
            size = ask_depth * (1 - i * 0.05)  # Decreasing depth
            if size > 0:
                asks.append(OrderbookLevel(price, size, timestamp))
        
        return OrderbookSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            current_price=base_price
        )
    
    def test_wall_disappearance(self):
        """Test detection of wall disappearance"""
        logger.info("🧪 Testing wall disappearance detection...")
        
        base_price = 50000.0
        base_time = datetime.now()
        
        # Create initial orderbook with large bid wall
        snapshot1 = self.create_synthetic_orderbook(
            self.test_symbol, base_price, bid_depth=1000.0, ask_depth=500.0,
            timestamp=base_time
        )
        self.analyzer.add_orderbook_snapshot(snapshot1)
        
        # Wait and create snapshot with disappeared bid wall
        time.sleep(1)
        snapshot2 = self.create_synthetic_orderbook(
            self.test_symbol, base_price, bid_depth=300.0, ask_depth=500.0,  # 70% bid reduction
            timestamp=base_time + timedelta(minutes=2)
        )
        self.analyzer.add_orderbook_snapshot(snapshot2)
        
        # Analyze features
        features = self.analyzer.analyze_heatmap_features(self.test_symbol)
        
        # Check results
        if features.wall_disappeared:
            logger.info(f"✅ Wall disappearance detected: {features.wall_disappeared_side} side, {features.wall_disappeared_size:.1%} reduction")
            return True
        else:
            logger.warning("❌ Wall disappearance NOT detected")
            return False
    
    def test_liquidity_pinning(self):
        """Test detection of liquidity pinning"""
        logger.info("🧪 Testing liquidity pinning detection...")
        
        base_price = 50000.0
        pinning_price = 50005.0  # Price level with high liquidity
        base_time = datetime.now()
        
        # Create multiple snapshots with price staying near high liquidity level
        for i in range(6):
            # Price oscillates around pinning level
            current_price = pinning_price + np.random.uniform(-2, 2)
            
            snapshot = self.create_synthetic_orderbook(
                self.test_symbol, current_price, bid_depth=500.0, ask_depth=500.0,
                timestamp=base_time + timedelta(minutes=i)
            )
            
            # Add extra liquidity at pinning level
            if i >= 2:  # From 3rd snapshot onwards
                # Add large ask at pinning level
                large_ask = OrderbookLevel(pinning_price, 2000.0, snapshot.timestamp)
                snapshot.asks.insert(0, large_ask)
                snapshot.asks.sort(key=lambda x: x.price)
            
            self.analyzer.add_orderbook_snapshot(snapshot)
            time.sleep(0.2)
        
        # Analyze features
        features = self.analyzer.analyze_heatmap_features(self.test_symbol)
        
        # Check results
        if features.liquidity_pinning:
            logger.info(f"✅ Liquidity pinning detected at level: {features.pinning_level:.2f}")
            return True
        else:
            logger.warning("❌ Liquidity pinning NOT detected")
            return False
    
    def test_void_reaction(self):
        """Test detection of liquidity void reactions"""
        logger.info("🧪 Testing liquidity void reaction detection...")
        
        base_price = 50000.0
        base_time = datetime.now()
        
        # Create initial orderbook
        snapshot1 = self.create_synthetic_orderbook(
            self.test_symbol, base_price, bid_depth=500.0, ask_depth=500.0,
            timestamp=base_time
        )
        self.analyzer.add_orderbook_snapshot(snapshot1)
        
        time.sleep(0.5)
        
        # Create orderbook with liquidity void above current price
        higher_price = base_price * 1.015  # 1.5% price increase
        snapshot2 = self.create_synthetic_orderbook(
            self.test_symbol, higher_price, bid_depth=500.0, ask_depth=100.0,  # Low ask depth = void
            timestamp=base_time + timedelta(minutes=3)
        )
        
        # Reduce ask liquidity to create void
        for ask in snapshot2.asks[:5]:
            ask.size *= 0.2  # Reduce top 5 ask levels to 20%
        
        self.analyzer.add_orderbook_snapshot(snapshot2)
        
        # Analyze features
        features = self.analyzer.analyze_heatmap_features(self.test_symbol)
        
        # Check results
        if features.liquidity_void_reaction:
            logger.info(f"✅ Liquidity void reaction detected: {features.void_size_percent:.1%} movement")
            return True
        else:
            logger.warning("❌ Liquidity void reaction NOT detected")
            return False
    
    def test_cluster_tilt(self):
        """Test detection of volume cluster tilt"""
        logger.info("🧪 Testing volume cluster tilt detection...")
        
        base_price = 50000.0
        base_time = datetime.now()
        
        # Create sequence showing bid/ask ratio change (bullish tilt)
        for i in range(8):
            if i < 4:
                # Initial balanced orderbook
                bid_depth = 500.0
                ask_depth = 500.0
            else:
                # Shift towards more bid depth (bullish)
                bid_depth = 800.0  # Increased bids
                ask_depth = 300.0  # Decreased asks
            
            snapshot = self.create_synthetic_orderbook(
                self.test_symbol, base_price, bid_depth=bid_depth, ask_depth=ask_depth,
                timestamp=base_time + timedelta(minutes=i)
            )
            
            self.analyzer.add_orderbook_snapshot(snapshot)
            time.sleep(0.2)
        
        # Analyze features
        features = self.analyzer.analyze_heatmap_features(self.test_symbol)
        
        # Check results
        if features.volume_cluster_tilt != "neutral":
            logger.info(f"✅ Volume cluster tilt detected: {features.volume_cluster_tilt} (strength: {features.cluster_tilt_strength:.1%})")
            return True
        else:
            logger.warning("❌ Volume cluster tilt NOT detected")
            return False
    
    def test_heatmap_integration(self):
        """Test integration with heatmap manager"""
        logger.info("🧪 Testing heatmap integration manager...")
        
        try:
            # Get heatmap manager
            heatmap_manager = get_heatmap_manager()
            
            # Add symbol for analysis
            heatmap_manager.add_symbol_for_analysis(self.test_symbol)
            
            # Create test orderbook data
            snapshot = self.create_synthetic_orderbook(
                self.test_symbol, 50000.0, bid_depth=1000.0, ask_depth=500.0
            )
            
            # Add to analyzer through manager
            heatmap_manager.heatmap_analyzer.add_orderbook_snapshot(snapshot)
            
            # Get GPT-formatted data
            gpt_data = heatmap_manager.get_heatmap_for_gpt(self.test_symbol)
            
            # Check results
            if gpt_data and 'heatmap_analysis' in gpt_data:
                logger.info("✅ Heatmap integration working")
                logger.info(f"   Summary: {gpt_data.get('heatmap_summary', 'N/A')}")
                return True
            else:
                logger.warning("❌ Heatmap integration failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Heatmap integration error: {e}")
            return False
    
    def test_data_persistence(self):
        """Test saving and loading heatmap features"""
        logger.info("🧪 Testing data persistence...")
        
        # Create test features
        test_features = HeatmapFeatures(
            wall_disappeared=True,
            liquidity_pinning=False,
            liquidity_void_reaction=True,
            volume_cluster_tilt="bullish",
            wall_disappeared_side="ask",
            wall_disappeared_size=0.45,
            void_size_percent=0.025,
            cluster_tilt_strength=0.35
        )
        
        # Save features
        timestamp = datetime.now()
        filepath = self.analyzer.save_heatmap_features(self.test_symbol, test_features, timestamp)
        
        if filepath:
            logger.info(f"✅ Features saved to: {filepath}")
            
            # Try to load them back
            loaded_features = self.analyzer.load_heatmap_features(self.test_symbol, timestamp)
            
            if loaded_features and loaded_features.wall_disappeared == test_features.wall_disappeared:
                logger.info("✅ Features loaded successfully")
                return True
            else:
                logger.warning("❌ Features loading failed")
                return False
        else:
            logger.warning("❌ Features saving failed")
            return False
    
    def run_all_tests(self):
        """Run all heatmap tests"""
        logger.info("🚀 Starting Orderbook Heatmap Test Suite")
        logger.info("=" * 60)
        
        tests = [
            ("Wall Disappearance", self.test_wall_disappearance),
            ("Liquidity Pinning", self.test_liquidity_pinning),
            ("Void Reaction", self.test_void_reaction),
            ("Cluster Tilt", self.test_cluster_tilt),
            ("Heatmap Integration", self.test_heatmap_integration),
            ("Data Persistence", self.test_data_persistence)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n🔍 Running {test_name} test...")
            try:
                result = test_func()
                results[test_name] = result
                status = "PASSED" if result else "FAILED"
                logger.info(f"📊 {test_name}: {status}")
            except Exception as e:
                logger.error(f"❌ {test_name} ERROR: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            logger.info("🎉 All heatmap tests PASSED! System is ready for production.")
        else:
            logger.warning(f"⚠️  {total-passed} test(s) failed. Review implementation.")
        
        return results

def demo_heatmap_features():
    """Demonstrate heatmap features with synthetic pump scenario"""
    logger.info("🎬 Demonstrating heatmap features during synthetic pump scenario")
    
    analyzer = OrderbookHeatmapAnalyzer()
    symbol = "ETHUSDT"
    base_price = 3000.0
    
    # Scenario: Pre-pump orderbook behavior
    scenarios = [
        {"time": 0, "price": 3000.0, "bid_depth": 800, "ask_depth": 1200, "desc": "Normal market conditions"},
        {"time": 2, "price": 2999.5, "bid_depth": 850, "ask_depth": 800, "desc": "Ask wall starts disappearing"},
        {"time": 4, "price": 3001.0, "bid_depth": 900, "ask_depth": 400, "desc": "Major ask wall disappeared"},
        {"time": 6, "price": 3003.0, "bid_depth": 1100, "ask_depth": 300, "desc": "Bullish cluster tilt"},
        {"time": 8, "price": 3008.0, "bid_depth": 1200, "ask_depth": 200, "desc": "Void reaction upward"},
    ]
    
    logger.info("📊 Simulating orderbook evolution:")
    
    for scenario in scenarios:
        timestamp = datetime.now() + timedelta(minutes=scenario["time"])
        
        # Create orderbook snapshot
        tester = OrderbookHeatmapTester()
        snapshot = tester.create_synthetic_orderbook(
            symbol, scenario["price"], scenario["bid_depth"], scenario["ask_depth"], timestamp
        )
        
        # Modify orderbook for specific scenarios
        if "wall disappeared" in scenario["desc"]:
            # Remove most ask levels
            snapshot.asks = snapshot.asks[:5]
            for ask in snapshot.asks:
                ask.size *= 0.3
        
        analyzer.add_orderbook_snapshot(snapshot)
        
        # Analyze current state
        features = analyzer.analyze_heatmap_features(symbol)
        
        logger.info(f"⏰ T+{scenario['time']}min: {scenario['desc']}")
        logger.info(f"   Price: ${scenario['price']:.1f}")
        logger.info(f"   Wall disappeared: {features.wall_disappeared}")
        logger.info(f"   Liquidity pinning: {features.liquidity_pinning}")
        logger.info(f"   Void reaction: {features.liquidity_void_reaction}")
        logger.info(f"   Cluster tilt: {features.volume_cluster_tilt}")
        
        time.sleep(0.5)
    
    # Final analysis
    final_features = analyzer.analyze_heatmap_features(symbol)
    
    logger.info("\n🔍 FINAL HEATMAP ANALYSIS:")
    logger.info(f"Wall Disappeared: {final_features.wall_disappeared}")
    if final_features.wall_disappeared:
        logger.info(f"  Side: {final_features.wall_disappeared_side}")
        logger.info(f"  Size: {final_features.wall_disappeared_size:.1%}")
    
    logger.info(f"Liquidity Pinning: {final_features.liquidity_pinning}")
    logger.info(f"Void Reaction: {final_features.liquidity_void_reaction}")
    logger.info(f"Cluster Tilt: {final_features.volume_cluster_tilt}")
    if final_features.cluster_tilt_strength > 0:
        logger.info(f"  Strength: {final_features.cluster_tilt_strength:.1%}")
    
    # Save results
    filepath = analyzer.save_heatmap_features(symbol, final_features)
    if filepath:
        logger.info(f"💾 Results saved to: {filepath}")
    
    # Show GPT integration format
    summary = analyzer.get_heatmap_summary(symbol)
    logger.info("\n📋 GPT Integration Format:")
    logger.info(json.dumps(summary, indent=2))

if __name__ == "__main__":
    print("🔬 Orderbook Heatmap Analysis Test Suite")
    print("========================================")
    
    # Run comprehensive tests
    tester = OrderbookHeatmapTester()
    test_results = tester.run_all_tests()
    
    print("\n" + "=" * 60)
    
    # Run demonstration
    demo_heatmap_features()
    
    print("\n🏁 Testing complete!")