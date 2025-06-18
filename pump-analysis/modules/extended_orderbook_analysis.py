"""
Extended Orderbook Heatmap Analysis for Pump-Analysis Project

This module provides comprehensive orderbook and price action analysis
that generates contextual data for GPT analysis of pump events.

Features:
- Orderbook depth analysis (top 25 bid/ask levels)
- Multi-timeframe candlestick analysis (15min and 1min)
- Wall disappearance detection
- Price pinning analysis
- Void reaction detection
- Volume cluster slope analysis
"""

import os
import time
import json
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ExtendedOrderbookAnalyzer:
    """Enhanced orderbook and price action analyzer for pump prediction"""
    
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY')
        self.base_url = "https://api.bybit.com"
        
        # Analysis parameters
        self.wall_threshold = 0.30  # 30% depth disappearance
        self.pinning_threshold = 0.15  # 15% price proximity to large liquidity
        self.void_threshold = 0.20  # 20% volume drop indicating void
        self.cluster_min_change = 0.20  # 20% change for slope detection
        
        logger.info("🔬 Extended Orderbook Analyzer initialized")
    
    def _generate_signature(self, params: Dict, timestamp: str) -> str:
        """Generate HMAC SHA256 signature for Bybit API"""
        if not self.api_secret:
            return ""
        
        # Sort parameters and create query string
        sorted_params = sorted(params.items())
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
        
        # Create signature payload
        payload = f"{timestamp}{self.api_key}{query_string}"
        
        # Generate signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _make_authenticated_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make authenticated request to Bybit API"""
        try:
            timestamp = str(int(time.time() * 1000))
            signature = self._generate_signature(params, timestamp)
            
            headers = {
                'X-BAPI-API-KEY': self.api_key or '',
                'X-BAPI-SIGN': signature,
                'X-BAPI-SIGN-TYPE': '2',
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-RECV-WINDOW': '5000'
            }
            
            url = f"{self.base_url}{endpoint}"
            
            logger.debug(f"📡 Extended API request: {url} with params: {params}")
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('retCode') == 0:
                return data.get('result', {})
            else:
                logger.error(f"API error: {data.get('retMsg', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error making authenticated request: {e}")
            return None
    
    def get_orderbook_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch top 25 bid/ask levels from orderbook
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            Dictionary with bids and asks data
        """
        params = {
            'category': 'linear',
            'symbol': symbol,
            'limit': 25
        }
        
        result = self._make_authenticated_request('/v5/market/orderbook', params)
        
        if result and 'b' in result and 'a' in result:
            return {
                'bids': [[float(price), float(size)] for price, size in result['b']],
                'asks': [[float(price), float(size)] for price, size in result['a']],
                'timestamp': result.get('ts', int(time.time() * 1000))
            }
        
        return None
    
    def get_kline_data(self, symbol: str, interval: str, limit: int) -> Optional[List[List]]:
        """
        Fetch kline data for specified interval
        
        Args:
            symbol: Trading symbol
            interval: Kline interval ('1' or '15')
            limit: Number of candles (60 for 1min, 672 for 15min)
            
        Returns:
            List of kline data [timestamp, open, high, low, close, volume]
        """
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        result = self._make_authenticated_request('/v5/market/kline', params)
        
        if result and 'list' in result:
            # Convert to float values
            klines = []
            for kline in result['list']:
                klines.append([
                    int(kline[0]),      # timestamp
                    float(kline[1]),    # open
                    float(kline[2]),    # high
                    float(kline[3]),    # low
                    float(kline[4]),    # close
                    float(kline[5])     # volume
                ])
            return sorted(klines, key=lambda x: x[0])  # Sort by timestamp
        
        return None
    
    def detect_walls_disappear(self, current_orderbook: Dict, klines_15m: List[List], 
                             klines_1m: List[List]) -> Dict:
        """
        Detect if significant orderbook walls have disappeared
        
        Args:
            current_orderbook: Current orderbook data
            klines_15m: 15-minute kline data
            klines_1m: 1-minute kline data
            
        Returns:
            Dictionary with wall disappearance analysis
        """
        try:
            if not current_orderbook or not klines_1m:
                return {'detected': False, 'reason': 'insufficient_data'}
            
            bids = current_orderbook.get('bids', [])
            asks = current_orderbook.get('asks', [])
            
            if len(bids) < 10 or len(asks) < 10:
                return {'detected': False, 'reason': 'insufficient_orderbook_depth'}
            
            # Calculate total bid/ask depth
            total_bid_depth = sum([size for price, size in bids])
            total_ask_depth = sum([size for price, size in asks])
            
            # Analyze recent volume spikes in 1-minute data
            recent_volumes = [kline[5] for kline in klines_1m[-10:]]  # Last 10 minutes
            avg_volume = np.mean(recent_volumes) if recent_volumes else 0
            max_recent_volume = max(recent_volumes) if recent_volumes else 0
            
            # Check for volume spikes coinciding with thin orderbook
            volume_spike = max_recent_volume > (avg_volume * 2) if avg_volume > 0 else False
            thin_book = total_bid_depth < 1000 or total_ask_depth < 1000  # Threshold for "thin"
            
            # Analyze price movement in recent candles
            recent_prices = [kline[4] for kline in klines_1m[-5:]]  # Last 5 minutes closes
            price_volatility = (max(recent_prices) - min(recent_prices)) / min(recent_prices) if recent_prices else 0
            
            high_volatility = price_volatility > 0.02  # 2% price movement
            
            walls_disappeared = volume_spike and thin_book and high_volatility
            
            return {
                'detected': walls_disappeared,
                'confidence': 0.8 if walls_disappeared else 0.2,
                'details': {
                    'volume_spike': volume_spike,
                    'thin_orderbook': thin_book,
                    'high_volatility': high_volatility,
                    'bid_depth': total_bid_depth,
                    'ask_depth': total_ask_depth,
                    'max_recent_volume': max_recent_volume,
                    'avg_volume': avg_volume,
                    'price_volatility_pct': price_volatility * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error in walls_disappear detection: {e}")
            return {'detected': False, 'reason': 'analysis_error', 'error': str(e)}
    
    def detect_pinning(self, current_orderbook: Dict, klines_1m: List[List]) -> Dict:
        """
        Detect if price is "pinned" to large liquidity levels
        
        Args:
            current_orderbook: Current orderbook data
            klines_1m: 1-minute kline data
            
        Returns:
            Dictionary with pinning analysis
        """
        try:
            if not current_orderbook or not klines_1m:
                return {'detected': False, 'reason': 'insufficient_data'}
            
            bids = current_orderbook.get('bids', [])
            asks = current_orderbook.get('asks', [])
            
            if len(bids) < 5 or len(asks) < 5 or len(klines_1m) < 10:
                return {'detected': False, 'reason': 'insufficient_data'}
            
            # Get current price (last close)
            current_price = klines_1m[-1][4]
            
            # Find largest liquidity levels
            largest_bid = max(bids, key=lambda x: x[1]) if bids else [0, 0]
            largest_ask = max(asks, key=lambda x: x[1]) if asks else [0, 0]
            
            # Calculate distance to large liquidity
            bid_distance = abs(current_price - largest_bid[0]) / current_price
            ask_distance = abs(current_price - largest_ask[0]) / current_price
            
            # Check if price is close to large liquidity (within threshold)
            close_to_large_bid = bid_distance < self.pinning_threshold
            close_to_large_ask = ask_distance < self.pinning_threshold
            
            # Analyze price behavior - is it "stuck" near this level?
            recent_prices = [kline[4] for kline in klines_1m[-10:]]
            price_range = max(recent_prices) - min(recent_prices)
            price_stability = price_range / current_price < 0.01  # Less than 1% movement
            
            # Volume analysis - is there sustained interest at this level?
            recent_volumes = [kline[5] for kline in klines_1m[-10:]]
            avg_volume = np.mean(recent_volumes)
            high_volume = max(recent_volumes) > avg_volume * 1.5
            
            pinning_detected = (close_to_large_bid or close_to_large_ask) and price_stability and high_volume
            
            return {
                'detected': pinning_detected,
                'confidence': 0.7 if pinning_detected else 0.3,
                'side': 'bid' if close_to_large_bid and pinning_detected else 'ask' if close_to_large_ask and pinning_detected else 'none',
                'details': {
                    'current_price': current_price,
                    'largest_bid': largest_bid,
                    'largest_ask': largest_ask,
                    'bid_distance_pct': bid_distance * 100,
                    'ask_distance_pct': ask_distance * 100,
                    'price_stability': price_stability,
                    'high_volume': high_volume,
                    'price_range_pct': (price_range / current_price) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error in pinning detection: {e}")
            return {'detected': False, 'reason': 'analysis_error', 'error': str(e)}
    
    def detect_void_reaction(self, klines_15m: List[List], klines_1m: List[List]) -> Dict:
        """
        Detect void reaction - dynamic response after volume void
        
        Args:
            klines_15m: 15-minute kline data
            klines_1m: 1-minute kline data
            
        Returns:
            Dictionary with void reaction analysis
        """
        try:
            if not klines_15m or not klines_1m or len(klines_15m) < 20 or len(klines_1m) < 30:
                return {'detected': False, 'reason': 'insufficient_data'}
            
            # Analyze 15-minute data for void periods
            volumes_15m = [kline[5] for kline in klines_15m[-20:]]  # Last 20 periods
            avg_volume_15m = np.mean(volumes_15m)
            
            # Find recent low-volume periods (voids)
            void_periods = []
            for i, vol in enumerate(volumes_15m[-10:]):  # Last 10 periods
                if vol < avg_volume_15m * (1 - self.void_threshold):  # 20% below average
                    void_periods.append(i)
            
            if not void_periods:
                return {'detected': False, 'reason': 'no_void_found'}
            
            # Check for reaction in 1-minute data after void
            recent_1m_volumes = [kline[5] for kline in klines_1m[-15:]]  # Last 15 minutes
            avg_1m_volume = np.mean(recent_1m_volumes)
            max_recent_1m = max(recent_1m_volumes)
            
            # Check for price movement after void
            recent_1m_prices = [kline[4] for kline in klines_1m[-15:]]
            price_change = (recent_1m_prices[-1] - recent_1m_prices[0]) / recent_1m_prices[0]
            
            # Void reaction criteria
            volume_surge = max_recent_1m > avg_1m_volume * 2  # 2x volume surge
            significant_move = abs(price_change) > 0.015  # 1.5% price move
            
            void_reaction = len(void_periods) > 0 and volume_surge and significant_move
            
            return {
                'detected': void_reaction,
                'confidence': 0.75 if void_reaction else 0.25,
                'direction': 'bullish' if price_change > 0 else 'bearish' if price_change < 0 else 'neutral',
                'details': {
                    'void_periods_count': len(void_periods),
                    'volume_surge': volume_surge,
                    'price_change_pct': price_change * 100,
                    'max_recent_volume': max_recent_1m,
                    'avg_volume': avg_1m_volume,
                    'void_threshold_used': self.void_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error in void_reaction detection: {e}")
            return {'detected': False, 'reason': 'analysis_error', 'error': str(e)}
    
    def analyze_cluster_slope(self, klines_15m: List[List], current_orderbook: Dict) -> Dict:
        """
        Analyze volume cluster slope direction
        
        Args:
            klines_15m: 15-minute kline data
            current_orderbook: Current orderbook data
            
        Returns:
            Dictionary with cluster slope analysis
        """
        try:
            if not klines_15m or not current_orderbook or len(klines_15m) < 10:
                return {'slope': 'neutral', 'reason': 'insufficient_data'}
            
            bids = current_orderbook.get('bids', [])
            asks = current_orderbook.get('asks', [])
            
            # Calculate bid/ask depth ratio over time
            current_bid_depth = sum([size for price, size in bids[:10]])  # Top 10 levels
            current_ask_depth = sum([size for price, size in asks[:10]])
            
            current_ratio = current_bid_depth / current_ask_depth if current_ask_depth > 0 else 1
            
            # Analyze volume trend in recent 15-minute candles
            recent_volumes = [kline[5] for kline in klines_15m[-10:]]
            recent_closes = [kline[4] for kline in klines_15m[-10:]]
            
            # Calculate volume-weighted price trend
            total_volume = sum(recent_volumes)
            if total_volume == 0:
                return {'slope': 'neutral', 'reason': 'no_volume'}
            
            vwap = sum([close * vol for close, vol in zip(recent_closes, recent_volumes)]) / total_volume
            current_price = recent_closes[-1]
            
            # Analyze slope based on multiple factors
            price_vs_vwap = (current_price - vwap) / vwap
            volume_trend = (recent_volumes[-1] - np.mean(recent_volumes[:-1])) / np.mean(recent_volumes[:-1])
            
            # Determine slope direction
            bullish_signals = 0
            bearish_signals = 0
            
            # Price above VWAP
            if price_vs_vwap > 0.005:  # 0.5% above VWAP
                bullish_signals += 1
            elif price_vs_vwap < -0.005:
                bearish_signals += 1
            
            # Volume increasing
            if volume_trend > 0.2:  # 20% volume increase
                bullish_signals += 1
            elif volume_trend < -0.2:
                bearish_signals += 1
            
            # Bid depth dominance
            if current_ratio > 1.2:  # 20% more bid depth
                bullish_signals += 1
            elif current_ratio < 0.8:  # 20% more ask depth
                bearish_signals += 1
            
            # Determine overall slope
            if bullish_signals >= 2:
                slope = 'bullish'
            elif bearish_signals >= 2:
                slope = 'bearish'
            else:
                slope = 'neutral'
            
            confidence = max(bullish_signals, bearish_signals) / 3.0  # Max possible signals
            
            return {
                'slope': slope,
                'confidence': confidence,
                'details': {
                    'price_vs_vwap_pct': price_vs_vwap * 100,
                    'volume_trend_pct': volume_trend * 100,
                    'bid_ask_ratio': current_ratio,
                    'bullish_signals': bullish_signals,
                    'bearish_signals': bearish_signals,
                    'current_price': current_price,
                    'vwap': vwap
                }
            }
            
        except Exception as e:
            logger.error(f"Error in cluster_slope analysis: {e}")
            return {'slope': 'neutral', 'reason': 'analysis_error', 'error': str(e)}
    
    def analyze_symbol_extended(self, symbol: str) -> Dict:
        """
        Perform complete extended orderbook analysis for a symbol
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info(f"🔬 Starting extended orderbook analysis for {symbol}")
        
        try:
            # Fetch all required data
            orderbook = self.get_orderbook_data(symbol)
            klines_15m = self.get_kline_data(symbol, '15', 672)  # 7 days of 15min data
            klines_1m = self.get_kline_data(symbol, '1', 60)     # 1 hour of 1min data
            
            if not orderbook:
                logger.warning(f"⚠️ No orderbook data for {symbol}")
                return self._empty_analysis("no_orderbook_data")
            
            if not klines_15m:
                logger.warning(f"⚠️ No 15min kline data for {symbol}")
                return self._empty_analysis("no_15m_data")
            
            if not klines_1m:
                logger.warning(f"⚠️ No 1min kline data for {symbol}")
                return self._empty_analysis("no_1m_data")
            
            # Perform all detections
            walls_analysis = self.detect_walls_disappear(orderbook, klines_15m, klines_1m)
            pinning_analysis = self.detect_pinning(orderbook, klines_1m)
            void_analysis = self.detect_void_reaction(klines_15m, klines_1m)
            cluster_analysis = self.analyze_cluster_slope(klines_15m, orderbook)
            
            # Compile results
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data_quality': {
                    'orderbook_depth': len(orderbook.get('bids', [])),
                    'klines_15m_count': len(klines_15m),
                    'klines_1m_count': len(klines_1m)
                },
                'detectors': {
                    'walls_disappear': walls_analysis,
                    'pinning': pinning_analysis,
                    'void_reaction': void_analysis,
                    'cluster_slope': cluster_analysis
                },
                'summary': self._generate_summary(walls_analysis, pinning_analysis, 
                                                void_analysis, cluster_analysis)
            }
            
            logger.info(f"✅ Extended analysis completed for {symbol}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Error in extended analysis for {symbol}: {e}")
            return self._empty_analysis("analysis_error", str(e))
    
    def _empty_analysis(self, reason: str, error: str = None) -> Dict:
        """Return empty analysis structure with reason"""
        return {
            'symbol': 'unknown',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_quality': {'status': 'failed', 'reason': reason, 'error': error},
            'detectors': {
                'walls_disappear': {'detected': False, 'reason': reason},
                'pinning': {'detected': False, 'reason': reason},
                'void_reaction': {'detected': False, 'reason': reason},
                'cluster_slope': {'slope': 'neutral', 'reason': reason}
            },
            'summary': f"Analysis failed: {reason}"
        }
    
    def _generate_summary(self, walls: Dict, pinning: Dict, void: Dict, cluster: Dict) -> str:
        """Generate human-readable summary of analysis"""
        
        summary_parts = []
        
        # Walls analysis
        if walls.get('detected'):
            summary_parts.append("Wykryto zniknięcie ścian orderbooku z spike'em wolumenu")
        
        # Pinning analysis
        if pinning.get('detected'):
            side = pinning.get('side', 'unknown')
            summary_parts.append(f"Cena przyklejona do dużej płynności po stronie {side}")
        
        # Void reaction analysis
        if void.get('detected'):
            direction = void.get('direction', 'neutral')
            summary_parts.append(f"Reakcja {direction} po okresie niskiego wolumenu")
        
        # Cluster slope analysis
        slope = cluster.get('slope', 'neutral')
        if slope != 'neutral':
            confidence = cluster.get('confidence', 0)
            summary_parts.append(f"Nachylenie klastrów wolumenowych: {slope} (pewność: {confidence:.1%})")
        
        if not summary_parts:
            return "Brak znaczących wzorców w orderbooku i wolumenie"
        
        return "; ".join(summary_parts)
    
    def format_for_gpt_context(self, analysis: Dict) -> str:
        """
        Format analysis results for GPT prompt context
        
        Args:
            analysis: Complete analysis results
            
        Returns:
            Formatted string for GPT context
        """
        if not analysis or analysis.get('data_quality', {}).get('status') == 'failed':
            return "Rozszerzona analiza orderbooku: Brak danych lub błąd analizy"
        
        context_parts = []
        context_parts.append("=== ROZSZERZONA ANALIZA ORDERBOOKU ===")
        
        detectors = analysis.get('detectors', {})
        
        # Walls disappear
        walls = detectors.get('walls_disappear', {})
        if walls.get('detected'):
            details = walls.get('details', {})
            context_parts.append(f"🔥 WALLS DISAPPEAR: Wykryto zniknięcie ścian orderbooku")
            context_parts.append(f"   - Głębokość bid: {details.get('bid_depth', 0):.0f}")
            context_parts.append(f"   - Głębokość ask: {details.get('ask_depth', 0):.0f}")
            context_parts.append(f"   - Max wolumen: {details.get('max_recent_volume', 0):.0f}")
            context_parts.append(f"   - Zmienność ceny: {details.get('price_volatility_pct', 0):.2f}%")
        
        # Pinning
        pinning = detectors.get('pinning', {})
        if pinning.get('detected'):
            details = pinning.get('details', {})
            side = pinning.get('side', 'unknown')
            context_parts.append(f"📌 PINNING: Cena przyklejona do płynności ({side})")
            context_parts.append(f"   - Aktualna cena: ${details.get('current_price', 0):.4f}")
            context_parts.append(f"   - Dystans do bid: {details.get('bid_distance_pct', 0):.2f}%")
            context_parts.append(f"   - Dystans do ask: {details.get('ask_distance_pct', 0):.2f}%")
            context_parts.append(f"   - Zakres ceny: {details.get('price_range_pct', 0):.2f}%")
        
        # Void reaction
        void = detectors.get('void_reaction', {})
        if void.get('detected'):
            details = void.get('details', {})
            direction = void.get('direction', 'neutral')
            context_parts.append(f"⚡ VOID REACTION: Reakcja {direction} po pustce")
            context_parts.append(f"   - Zmiana ceny: {details.get('price_change_pct', 0):.2f}%")
            context_parts.append(f"   - Spike wolumenu: {details.get('max_recent_volume', 0):.0f}")
            context_parts.append(f"   - Okresy pustki: {details.get('void_periods_count', 0)}")
        
        # Cluster slope
        cluster = detectors.get('cluster_slope', {})
        slope = cluster.get('slope', 'neutral')
        if slope != 'neutral':
            details = cluster.get('details', {})
            confidence = cluster.get('confidence', 0)
            context_parts.append(f"📊 CLUSTER SLOPE: {slope.upper()} (pewność: {confidence:.1%})")
            context_parts.append(f"   - Cena vs VWAP: {details.get('price_vs_vwap_pct', 0):.2f}%")
            context_parts.append(f"   - Trend wolumenu: {details.get('volume_trend_pct', 0):.2f}%")
            context_parts.append(f"   - Stosunek bid/ask: {details.get('bid_ask_ratio', 1):.2f}")
        
        # Summary
        summary = analysis.get('summary', '')
        if summary and 'Brak znaczących' not in summary:
            context_parts.append(f"\n💡 PODSUMOWANIE: {summary}")
        
        if len(context_parts) == 1:  # Only header
            context_parts.append("Brak wykrytych wzorców w rozszerzonej analizie orderbooku")
        
        return "\n".join(context_parts)


def initialize_extended_orderbook_analyzer() -> ExtendedOrderbookAnalyzer:
    """Initialize extended orderbook analyzer"""
    return ExtendedOrderbookAnalyzer()


# Global instance for pump-analysis integration
extended_analyzer = None

def get_extended_orderbook_analyzer() -> ExtendedOrderbookAnalyzer:
    """Get global extended orderbook analyzer instance"""
    global extended_analyzer
    if extended_analyzer is None:
        extended_analyzer = initialize_extended_orderbook_analyzer()
    return extended_analyzer