"""
Heatmap Integration Module
Integrates orderbook heatmap analysis with pump analysis system
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import threading
import time

from .orderbook_heatmap import OrderbookHeatmapAnalyzer, HeatmapFeatures, create_orderbook_snapshot_from_api
from .bybit_orderbook import BybitOrderbookFetcher, OrderbookDataCollector

logger = logging.getLogger(__name__)

class HeatmapIntegrationManager:
    """Manages integration between orderbook heatmap analysis and pump analysis"""
    
    def __init__(self):
        # Initialize components
        self.bybit_fetcher = BybitOrderbookFetcher()
        self.orderbook_collector = OrderbookDataCollector(self.bybit_fetcher)
        self.heatmap_analyzer = OrderbookHeatmapAnalyzer()
        
        # Configuration
        self.collection_enabled = True
        self.collection_thread = None
        self.analyzed_symbols = set()
        
        # Data storage
        self.heatmap_cache: Dict[str, HeatmapFeatures] = {}
        self.last_analysis_time: Dict[str, datetime] = {}
        
        logger.info("Heatmap integration manager initialized")
    
    def start_collection(self):
        """Start background orderbook data collection"""
        if self.collection_thread and self.collection_thread.is_alive():
            logger.warning("Collection already running")
            return
        
        self.collection_enabled = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started orderbook collection thread")
    
    def stop_collection(self):
        """Stop background data collection"""
        self.collection_enabled = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped orderbook collection")
    
    def _collection_loop(self):
        """Background loop for collecting orderbook data"""
        while self.collection_enabled:
            try:
                # Collect orderbooks for active symbols
                orderbooks = self.orderbook_collector.collect_orderbooks()
                
                # Process each orderbook
                for symbol, orderbook_data in orderbooks.items():
                    # Create snapshot and add to analyzer
                    snapshot = create_orderbook_snapshot_from_api(symbol, orderbook_data)
                    self.heatmap_analyzer.add_orderbook_snapshot(snapshot)
                
                # Sleep for collection interval
                time.sleep(30)  # 30 seconds
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def add_symbol_for_analysis(self, symbol: str):
        """Add symbol for orderbook analysis"""
        self.orderbook_collector.add_symbol(symbol)
        self.analyzed_symbols.add(symbol)
        logger.info(f"Added {symbol} for heatmap analysis")
    
    def remove_symbol_from_analysis(self, symbol: str):
        """Remove symbol from analysis"""
        self.orderbook_collector.remove_symbol(symbol)
        self.analyzed_symbols.discard(symbol)
        if symbol in self.heatmap_cache:
            del self.heatmap_cache[symbol]
        if symbol in self.last_analysis_time:
            del self.last_analysis_time[symbol]
        logger.info(f"Removed {symbol} from heatmap analysis")
    
    def analyze_symbol_heatmap(self, symbol: str, force_refresh: bool = False) -> Optional[HeatmapFeatures]:
        """Analyze heatmap features for a symbol"""
        
        # Check cache first
        if not force_refresh and symbol in self.heatmap_cache:
            last_analysis = self.last_analysis_time.get(symbol)
            if last_analysis and datetime.now() - last_analysis < timedelta(minutes=10):
                return self.heatmap_cache[symbol]
        
        # Ensure symbol is being collected
        if symbol not in self.analyzed_symbols:
            self.add_symbol_for_analysis(symbol)
            # Wait a bit for some data to be collected
            time.sleep(5)
        
        # Analyze features
        features = self.heatmap_analyzer.analyze_heatmap_features(symbol)
        
        # Cache results
        self.heatmap_cache[symbol] = features
        self.last_analysis_time[symbol] = datetime.now()
        
        # Save to file
        self.heatmap_analyzer.save_heatmap_features(symbol, features)
        
        return features
    
    def get_heatmap_for_gpt(self, symbol: str) -> Dict[str, Any]:
        """Get heatmap analysis formatted for GPT integration"""
        
        features = self.analyze_symbol_heatmap(symbol)
        if not features:
            return {
                "heatmap_analysis": {
                    "wall_disappeared": False,
                    "liquidity_pinning": False,
                    "liquidity_void_reaction": False,
                    "volume_cluster_tilt": "neutral"
                },
                "heatmap_summary": "No orderbook data available for heatmap analysis"
            }
        
        # Create descriptive summary
        summary_parts = []
        
        if features.wall_disappeared:
            side = features.wall_disappeared_side or "unknown"
            size = features.wall_disappeared_size
            summary_parts.append(f"Large {side} wall disappeared ({size:.1%} depth reduction)")
        
        if features.liquidity_pinning:
            level = features.pinning_level
            if level:
                summary_parts.append(f"Price pinning detected at level {level:.4f}")
            else:
                summary_parts.append("Price showing liquidity pinning behavior")
        
        if features.liquidity_void_reaction:
            void_size = features.void_size_percent
            summary_parts.append(f"Price reacted to liquidity void ({void_size:.1%} movement)")
        
        if features.volume_cluster_tilt != "neutral":
            strength = features.cluster_tilt_strength
            tilt = features.volume_cluster_tilt
            summary_parts.append(f"Volume cluster shows {tilt} tilt (strength: {strength:.1%})")
        
        if not summary_parts:
            summary_parts.append("No significant heatmap patterns detected")
        
        return {
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
            },
            "heatmap_summary": "; ".join(summary_parts)
        }
    
    def enhance_pre_pump_data_with_heatmap(self, symbol: str, pre_pump_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance pre-pump analysis data with heatmap features"""
        
        # Get heatmap data
        heatmap_data = self.get_heatmap_for_gpt(symbol)
        
        # Add to pre-pump data
        enhanced_data = pre_pump_data.copy()
        enhanced_data.update(heatmap_data)
        
        # Add orderbook insights to existing insights
        if 'insights' in enhanced_data:
            if isinstance(enhanced_data['insights'], list):
                enhanced_data['insights'].append(heatmap_data['heatmap_summary'])
            else:
                enhanced_data['orderbook_insights'] = heatmap_data['heatmap_summary']
        else:
            enhanced_data['orderbook_insights'] = heatmap_data['heatmap_summary']
        
        return enhanced_data
    
    def get_heatmap_conditions_for_detector(self, symbol: str) -> List[str]:
        """Generate heatmap conditions for detector function generation"""
        
        features = self.analyze_symbol_heatmap(symbol)
        if not features:
            return []
        
        conditions = []
        
        if features.wall_disappeared:
            side = features.wall_disappeared_side or "bid"
            conditions.append(f"# Heatmap: {side.title()} wall disappeared")
            conditions.append(f"wall_disappeared_{side} = detect_wall_disappearance(orderbook_data, '{side}')")
        
        if features.liquidity_pinning:
            conditions.append("# Heatmap: Liquidity pinning detected")
            conditions.append("liquidity_pinning = detect_liquidity_pinning(price_data, orderbook_data)")
        
        if features.liquidity_void_reaction:
            conditions.append("# Heatmap: Liquidity void reaction")
            conditions.append("void_reaction = detect_void_reaction(price_data, orderbook_data)")
        
        if features.volume_cluster_tilt != "neutral":
            tilt = features.volume_cluster_tilt
            conditions.append(f"# Heatmap: Volume cluster {tilt} tilt")
            conditions.append(f"cluster_tilt_{tilt} = detect_cluster_tilt(orderbook_data, '{tilt}')")
        
        return conditions
    
    def save_analysis_results(self, symbol: str, pump_timestamp: datetime) -> str:
        """Save complete heatmap analysis results for a pump event"""
        
        features = self.analyze_symbol_heatmap(symbol, force_refresh=True)
        if not features:
            return ""
        
        # Create comprehensive analysis record
        analysis_record = {
            "symbol": symbol,
            "pump_timestamp": pump_timestamp.isoformat(),
            "analysis_timestamp": datetime.now().isoformat(),
            "heatmap_features": {
                "wall_disappeared": features.wall_disappeared,
                "liquidity_pinning": features.liquidity_pinning,
                "liquidity_void_reaction": features.liquidity_void_reaction,
                "volume_cluster_tilt": features.volume_cluster_tilt,
                "wall_disappeared_side": features.wall_disappeared_side,
                "wall_disappeared_size": features.wall_disappeared_size,
                "pinning_level": features.pinning_level,
                "void_size_percent": features.void_size_percent,
                "cluster_tilt_strength": features.cluster_tilt_strength
            },
            "gpt_integration_data": self.get_heatmap_for_gpt(symbol),
            "detector_conditions": self.get_heatmap_conditions_for_detector(symbol)
        }
        
        # Save to heatmap_data directory
        filename = f"{symbol}_{pump_timestamp.strftime('%Y%m%d_%H%M%S')}_pump_heatmap.json"
        filepath = os.path.join(self.heatmap_analyzer.heatmap_data_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(analysis_record, f, indent=2)
            
            logger.info(f"Pump heatmap analysis saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving pump heatmap analysis: {e}")
            return ""
    
    def load_historical_heatmap_data(self, symbol: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Load historical heatmap data for a specific pump event"""
        
        filename = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}_pump_heatmap.json"
        filepath = os.path.join(self.heatmap_analyzer.heatmap_data_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Could not load historical heatmap data: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of heatmap integration system"""
        
        return {
            "collection_enabled": self.collection_enabled,
            "collection_thread_active": self.collection_thread and self.collection_thread.is_alive(),
            "analyzed_symbols_count": len(self.analyzed_symbols),
            "analyzed_symbols": list(self.analyzed_symbols),
            "cached_analyses": len(self.heatmap_cache),
            "bybit_connection": self.bybit_fetcher.test_connection()
        }

# Global instance for easy access
_heatmap_manager = None

def get_heatmap_manager() -> HeatmapIntegrationManager:
    """Get global heatmap integration manager instance"""
    global _heatmap_manager
    if _heatmap_manager is None:
        _heatmap_manager = HeatmapIntegrationManager()
    return _heatmap_manager

def initialize_heatmap_system():
    """Initialize and start the heatmap system"""
    manager = get_heatmap_manager()
    manager.start_collection()
    return manager

def shutdown_heatmap_system():
    """Shutdown the heatmap system"""
    global _heatmap_manager
    if _heatmap_manager:
        _heatmap_manager.stop_collection()
        _heatmap_manager = None