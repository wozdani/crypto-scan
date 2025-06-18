"""
Orderbook Heatmap Simulation Module
Analyzes local orderbook data to simulate heatmap behavior and detect key patterns
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class OrderbookLevel:
    """Represents a single orderbook level"""
    price: float
    size: float
    timestamp: datetime

@dataclass
class OrderbookSnapshot:
    """Complete orderbook snapshot"""
    symbol: str
    timestamp: datetime
    bids: List[OrderbookLevel]
    asks: List[OrderbookLevel]
    current_price: float

@dataclass
class HeatmapFeatures:
    """Detected heatmap features"""
    wall_disappeared: bool = False
    liquidity_pinning: bool = False
    liquidity_void_reaction: bool = False
    volume_cluster_tilt: str = "neutral"  # bullish, bearish, neutral
    wall_disappeared_side: Optional[str] = None  # bid/ask
    wall_disappeared_size: float = 0.0
    pinning_level: Optional[float] = None
    void_size_percent: float = 0.0
    cluster_tilt_strength: float = 0.0

class OrderbookHeatmapAnalyzer:
    """Analyzes orderbook data to simulate heatmap patterns"""
    
    def __init__(self, 
                 top_levels: int = 20,
                 wall_threshold_percent: float = 0.3,  # 30% depth disappears
                 wall_timeframe_minutes: int = 5,
                 pinning_tolerance_percent: float = 0.1,  # 0.1% price tolerance
                 void_threshold_percent: float = 0.5):  # 50% liquidity gap
        
        self.top_levels = top_levels
        self.wall_threshold_percent = wall_threshold_percent
        self.wall_timeframe_minutes = wall_timeframe_minutes
        self.pinning_tolerance_percent = pinning_tolerance_percent
        self.void_threshold_percent = void_threshold_percent
        
        # Historical data storage
        self.orderbook_history: Dict[str, List[OrderbookSnapshot]] = defaultdict(list)
        self.heatmap_data_dir = "heatmap_data"
        
        # Ensure data directory exists
        os.makedirs(self.heatmap_data_dir, exist_ok=True)
    
    def add_orderbook_snapshot(self, snapshot: OrderbookSnapshot):
        """Add orderbook snapshot to historical data"""
        symbol = snapshot.symbol
        self.orderbook_history[symbol].append(snapshot)
        
        # Keep only recent data (last 2 hours)
        cutoff_time = datetime.now() - timedelta(hours=2)
        self.orderbook_history[symbol] = [
            s for s in self.orderbook_history[symbol] 
            if s.timestamp >= cutoff_time
        ]
    
    def analyze_heatmap_features(self, symbol: str, analysis_window_minutes: int = 60) -> HeatmapFeatures:
        """Analyze orderbook data to detect heatmap features"""
        
        if symbol not in self.orderbook_history:
            logger.warning(f"No orderbook history for {symbol}")
            return HeatmapFeatures()
        
        snapshots = self.orderbook_history[symbol]
        if len(snapshots) < 2:
            logger.warning(f"Insufficient orderbook data for {symbol}")
            return HeatmapFeatures()
        
        # Filter to analysis window
        cutoff_time = datetime.now() - timedelta(minutes=analysis_window_minutes)
        recent_snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return HeatmapFeatures()
        
        features = HeatmapFeatures()
        
        # Analyze wall disappearance
        self._detect_wall_disappearance(recent_snapshots, features)
        
        # Analyze liquidity pinning
        self._detect_liquidity_pinning(recent_snapshots, features)
        
        # Analyze liquidity void reactions
        self._detect_void_reactions(recent_snapshots, features)
        
        # Analyze volume cluster tilt
        self._analyze_cluster_tilt(recent_snapshots, features)
        
        return features
    
    def _detect_wall_disappearance(self, snapshots: List[OrderbookSnapshot], features: HeatmapFeatures):
        """Detect sudden disappearance of large liquidity walls"""
        
        if len(snapshots) < 2:
            return
        
        # Compare recent snapshots for significant depth changes
        cutoff_time = datetime.now() - timedelta(minutes=self.wall_timeframe_minutes)
        recent_snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return
        
        # Analyze bid and ask sides separately
        for side in ['bids', 'asks']:
            depth_changes = []
            
            for i in range(1, len(recent_snapshots)):
                prev_snapshot = recent_snapshots[i-1]
                curr_snapshot = recent_snapshots[i]
                
                prev_levels = getattr(prev_snapshot, side)
                curr_levels = getattr(curr_snapshot, side)
                
                # Calculate total depth for top levels
                prev_depth = sum(level.size for level in prev_levels[:self.top_levels])
                curr_depth = sum(level.size for level in curr_levels[:self.top_levels])
                
                if prev_depth > 0:
                    depth_change = (prev_depth - curr_depth) / prev_depth
                    depth_changes.append(depth_change)
            
            # Check if significant wall disappeared
            if depth_changes:
                max_depth_reduction = max(depth_changes)
                
                if max_depth_reduction >= self.wall_threshold_percent:
                    features.wall_disappeared = True
                    features.wall_disappeared_side = side.rstrip('s')  # 'bids' -> 'bid'
                    features.wall_disappeared_size = max_depth_reduction
                    
                    logger.info(f"Wall disappeared detected: {side} side, {max_depth_reduction:.1%} reduction")
    
    def _detect_liquidity_pinning(self, snapshots: List[OrderbookSnapshot], features: HeatmapFeatures):
        """Detect price pinning to high liquidity levels"""
        
        if len(snapshots) < 5:
            return
        
        # Find levels with consistently high liquidity
        liquidity_levels = defaultdict(list)
        
        for snapshot in snapshots[-10:]:  # Last 10 snapshots
            # Analyze both sides
            all_levels = snapshot.bids + snapshot.asks
            
            # Group by price ranges (0.1% tolerance)
            for level in all_levels:
                price_bucket = round(level.price / (level.price * self.pinning_tolerance_percent / 100))
                liquidity_levels[price_bucket].append(level.size)
        
        # Find high liquidity levels
        high_liquidity_levels = []
        for price_bucket, sizes in liquidity_levels.items():
            if len(sizes) >= 3:  # Appeared in at least 3 snapshots
                avg_size = np.mean(sizes)
                if avg_size > 0:
                    actual_price = price_bucket * (snapshots[-1].current_price * self.pinning_tolerance_percent / 100)
                    high_liquidity_levels.append((actual_price, avg_size))
        
        # Check if price is staying near high liquidity levels
        current_price = snapshots[-1].current_price
        recent_prices = [s.current_price for s in snapshots[-5:]]
        
        for liquidity_price, liquidity_size in high_liquidity_levels:
            price_distance = abs(current_price - liquidity_price) / current_price
            
            # Check if price has been staying near this level
            if price_distance <= self.pinning_tolerance_percent / 100:
                # Verify price stability around this level
                prices_near_level = [
                    p for p in recent_prices 
                    if abs(p - liquidity_price) / p <= self.pinning_tolerance_percent / 100
                ]
                
                if len(prices_near_level) >= 3:  # At least 3 of last 5 prices
                    features.liquidity_pinning = True
                    features.pinning_level = liquidity_price
                    
                    logger.info(f"Liquidity pinning detected at {liquidity_price:.4f}")
                    break
    
    def _detect_void_reactions(self, snapshots: List[OrderbookSnapshot], features: HeatmapFeatures):
        """Detect price reactions to liquidity voids"""
        
        if len(snapshots) < 3:
            return
        
        # Analyze recent price movements and liquidity gaps
        for i in range(2, len(snapshots)):
            prev_snapshot = snapshots[i-2]
            curr_snapshot = snapshots[i]
            
            price_change = (curr_snapshot.current_price - prev_snapshot.current_price) / prev_snapshot.current_price
            
            # Significant price movement (>1%)
            if abs(price_change) >= 0.01:
                # Check for liquidity void in the direction of movement
                void_detected = self._check_liquidity_void(
                    curr_snapshot, 
                    price_change > 0  # True for upward movement
                )
                
                if void_detected:
                    features.liquidity_void_reaction = True
                    features.void_size_percent = abs(price_change)
                    
                    logger.info(f"Liquidity void reaction detected: {price_change:.2%} movement")
                    break
    
    def _check_liquidity_void(self, snapshot: OrderbookSnapshot, upward_movement: bool) -> bool:
        """Check if there's a liquidity void in the movement direction"""
        
        current_price = snapshot.current_price
        levels = snapshot.asks if upward_movement else snapshot.bids
        
        # Calculate average liquidity density
        if len(levels) < 5:
            return False
        
        total_size = sum(level.size for level in levels[:10])
        avg_size_per_level = total_size / min(10, len(levels))
        
        # Look for gaps with significantly lower liquidity
        void_count = 0
        for level in levels[:5]:  # Check first 5 levels
            if level.size < avg_size_per_level * (1 - self.void_threshold_percent):
                void_count += 1
        
        # Void detected if majority of near levels have low liquidity
        return void_count >= 3
    
    def _analyze_cluster_tilt(self, snapshots: List[OrderbookSnapshot], features: HeatmapFeatures):
        """Analyze volume cluster tilt (shift from ask to bid or vice versa)"""
        
        if len(snapshots) < 5:
            return
        
        # Calculate bid/ask depth ratios over time
        depth_ratios = []
        
        for snapshot in snapshots[-10:]:  # Last 10 snapshots
            bid_depth = sum(level.size for level in snapshot.bids[:self.top_levels])
            ask_depth = sum(level.size for level in snapshot.asks[:self.top_levels])
            
            if ask_depth > 0:
                ratio = bid_depth / ask_depth
                depth_ratios.append(ratio)
        
        if len(depth_ratios) < 3:
            return
        
        # Analyze trend in depth ratios
        recent_ratios = depth_ratios[-5:]
        early_ratios = depth_ratios[-10:-5] if len(depth_ratios) >= 10 else depth_ratios[:-5]
        
        if not early_ratios:
            return
        
        recent_avg = np.mean(recent_ratios)
        early_avg = np.mean(early_ratios)
        
        ratio_change = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0
        
        # Significant tilt threshold (20% change)
        tilt_threshold = 0.2
        
        if ratio_change >= tilt_threshold:
            features.volume_cluster_tilt = "bullish"
            features.cluster_tilt_strength = ratio_change
            logger.info(f"Bullish cluster tilt detected: {ratio_change:.2%}")
        elif ratio_change <= -tilt_threshold:
            features.volume_cluster_tilt = "bearish"
            features.cluster_tilt_strength = abs(ratio_change)
            logger.info(f"Bearish cluster tilt detected: {ratio_change:.2%}")
        else:
            features.volume_cluster_tilt = "neutral"
    
    def save_heatmap_features(self, symbol: str, features: HeatmapFeatures, timestamp: datetime = None) -> str:
        """Save heatmap features to JSON file"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Convert features to dict
        features_dict = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "wall_disappeared": features.wall_disappeared,
            "liquidity_pinning": features.liquidity_pinning,
            "liquidity_void_reaction": features.liquidity_void_reaction,
            "volume_cluster_tilt": features.volume_cluster_tilt,
            "wall_disappeared_side": features.wall_disappeared_side,
            "wall_disappeared_size": features.wall_disappeared_size,
            "pinning_level": features.pinning_level,
            "void_size_percent": features.void_size_percent,
            "cluster_tilt_strength": features.cluster_tilt_strength
        }
        
        # Save to file
        filename = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}_heatmap.json"
        filepath = os.path.join(self.heatmap_data_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(features_dict, f, indent=2)
            
            logger.info(f"Heatmap features saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving heatmap features: {e}")
            return ""
    
    def load_heatmap_features(self, symbol: str, timestamp: datetime = None) -> Optional[HeatmapFeatures]:
        """Load heatmap features from JSON file"""
        
        if timestamp is None:
            # Find most recent file for symbol
            pattern = f"{symbol}_*_heatmap.json"
            import glob
            files = glob.glob(os.path.join(self.heatmap_data_dir, pattern))
            
            if not files:
                return None
            
            # Get most recent file
            files.sort(reverse=True)
            filepath = files[0]
        else:
            filename = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}_heatmap.json"
            filepath = os.path.join(self.heatmap_data_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert back to HeatmapFeatures
            features = HeatmapFeatures(
                wall_disappeared=data.get("wall_disappeared", False),
                liquidity_pinning=data.get("liquidity_pinning", False),
                liquidity_void_reaction=data.get("liquidity_void_reaction", False),
                volume_cluster_tilt=data.get("volume_cluster_tilt", "neutral"),
                wall_disappeared_side=data.get("wall_disappeared_side"),
                wall_disappeared_size=data.get("wall_disappeared_size", 0.0),
                pinning_level=data.get("pinning_level"),
                void_size_percent=data.get("void_size_percent", 0.0),
                cluster_tilt_strength=data.get("cluster_tilt_strength", 0.0)
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error loading heatmap features: {e}")
            return None
    
    def get_heatmap_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of heatmap analysis for GPT integration"""
        
        features = self.analyze_heatmap_features(symbol)
        
        summary = {
            "heatmap_analysis": {
                "wall_disappeared": features.wall_disappeared,
                "liquidity_pinning": features.liquidity_pinning,
                "liquidity_void_reaction": features.liquidity_void_reaction,
                "volume_cluster_tilt": features.volume_cluster_tilt
            },
            "heatmap_details": {
                "wall_side": features.wall_disappeared_side,
                "wall_size_reduction": f"{features.wall_disappeared_size:.1%}" if features.wall_disappeared else None,
                "pinning_level": features.pinning_level,
                "void_reaction_size": f"{features.void_size_percent:.1%}" if features.liquidity_void_reaction else None,
                "cluster_tilt_strength": f"{features.cluster_tilt_strength:.1%}" if features.cluster_tilt_strength > 0 else None
            }
        }
        
        return summary

def create_orderbook_snapshot_from_api(symbol: str, orderbook_data: Dict) -> OrderbookSnapshot:
    """Create OrderbookSnapshot from Bybit API response"""
    
    timestamp = datetime.now()
    
    # Parse bids and asks
    bids = []
    asks = []
    
    if 'b' in orderbook_data:  # Bybit format
        for bid_data in orderbook_data['b'][:20]:  # Top 20 levels
            price = float(bid_data[0])
            size = float(bid_data[1])
            bids.append(OrderbookLevel(price, size, timestamp))
    
    if 'a' in orderbook_data:  # Bybit format
        for ask_data in orderbook_data['a'][:20]:  # Top 20 levels
            price = float(ask_data[0])
            size = float(ask_data[1])
            asks.append(OrderbookLevel(price, size, timestamp))
    
    # Estimate current price (mid-price)
    current_price = 0.0
    if bids and asks:
        current_price = (bids[0].price + asks[0].price) / 2
    elif bids:
        current_price = bids[0].price
    elif asks:
        current_price = asks[0].price
    
    return OrderbookSnapshot(
        symbol=symbol,
        timestamp=timestamp,
        bids=bids,
        asks=asks,
        current_price=current_price
    )