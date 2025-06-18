"""
Simplified Heatmap Detectors Module
Qualitative context for price behavior relative to orderbook liquidity
No numerical thresholds - structure and interpretation based detection
"""

import numpy as np
import requests
import time
import os
import hmac
import hashlib
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SimplifiedHeatmapDetectors:
    """Simplified orderbook analysis with qualitative context"""
    
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY', '')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY', '')
        self.base_url = "https://api.bybit.com"
    
    def _generate_signature(self, params: Dict, timestamp: str) -> str:
        """Generate HMAC SHA256 signature for Bybit API"""
        if not self.api_secret:
            return ""
        
        sorted_params = sorted(params.items())
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
        payload = f"{timestamp}{self.api_key}{query_string}"
        
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
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-SIGN-TYPE': '2',
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-RECV-WINDOW': '5000'
            }
            
            url = f"{self.base_url}{endpoint}"
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
    
    def get_klines(self, symbol: str, interval: str, limit: int) -> Optional[List]:
        """Get kline data from Bybit"""
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        result = self._make_authenticated_request('/v5/market/kline', params)
        
        if result and 'list' in result:
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
            return sorted(klines, key=lambda x: x[0])
        
        return None
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get orderbook data from Bybit"""
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
    
    def walls_disappear(self, symbol: str) -> str:
        """
        Detect if large orderbook walls have disappeared
        Returns: ✅ (walls gone) or ❌ (walls present)
        """
        try:
            ob = self.get_orderbook(symbol)
            if not ob or not ob.get('bids') or not ob.get('asks'):
                return "❌"
            
            bids = [b[1] for b in ob['bids']]  # bid sizes
            asks = [a[1] for a in ob['asks']]  # ask sizes
            
            if not bids or not asks:
                return "❌"
            
            bid_median = np.median(bids)
            ask_median = np.median(asks)
            
            # Check for unusually large walls (10x median size)
            large_bid = any(b > 10 * bid_median for b in bids)
            large_ask = any(a > 10 * ask_median for a in asks)
            
            # Walls disappeared = no large walls present
            return "✅" if not (large_bid or large_ask) else "❌"
            
        except Exception as e:
            logger.error(f"Error in walls_disappear for {symbol}: {e}")
            return "❌"
    
    def pinning(self, symbol: str) -> str:
        """
        Detect if price is pinned to liquidity levels
        Returns: ✅ (pinned) or ❌ (not pinned)
        """
        try:
            klines = self.get_klines(symbol, "1", 30)  # Last 30 minutes
            ob = self.get_orderbook(symbol)
            
            if not klines or not ob or not ob.get('bids') or not ob.get('asks'):
                return "❌"
            
            prices = [float(c[4]) for c in klines]  # close prices
            
            # Calculate mid price from best bid/ask
            best_bid = float(ob['bids'][0][0])
            best_ask = float(ob['asks'][0][0])
            mid_price = (best_bid + best_ask) / 2
            
            # Check if all recent prices are within tight range of mid price (0.3%)
            pinned = all(abs(p - mid_price) / mid_price < 0.003 for p in prices)
            
            return "✅" if pinned else "❌"
            
        except Exception as e:
            logger.error(f"Error in pinning for {symbol}: {e}")
            return "❌"
    
    def void_reaction(self, symbol: str) -> str:
        """
        Detect void reaction - dynamic response after low volatility
        Returns: ✅ (reaction detected) or ❌ (no reaction)
        """
        try:
            klines = self.get_klines(symbol, "1", 5)  # Last 5 minutes
            if not klines:
                return "❌"
            
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            
            # Calculate price ranges (high - low) for each candle
            ranges = [abs(h - l) for h, l in zip(highs, lows)]
            
            if len(ranges) < 2:
                return "❌"
            
            mean_range = np.mean(ranges[:-1])  # Average of previous ranges
            recent_range = ranges[-1]  # Most recent range
            
            # Void reaction = recent range significantly larger than average
            if mean_range > 0 and recent_range > 1.5 * mean_range:
                return "✅"
            
            return "❌"
            
        except Exception as e:
            logger.error(f"Error in void_reaction for {symbol}: {e}")
            return "❌"
    
    def cluster_slope(self, symbol: str) -> str:
        """
        Analyze volume cluster slope direction
        Returns: "bullish", "bearish", or "neutral"
        """
        try:
            ob = self.get_orderbook(symbol)
            if not ob or not ob.get('bids') or not ob.get('asks'):
                return "neutral"
            
            bids = [(float(p), float(q)) for p, q in ob['bids']]
            asks = [(float(p), float(q)) for p, q in ob['asks']]
            
            if len(bids) < 3 or len(asks) < 3:
                return "neutral"
            
            # Extract quantities for trend analysis
            bid_qs = np.array([q for _, q in bids])
            ask_qs = np.array([q for _, q in asks])
            
            # Calculate linear trends in bid/ask quantities
            bid_trend = np.polyfit(range(len(bid_qs)), bid_qs, 1)[0]
            ask_trend = np.polyfit(range(len(ask_qs)), ask_qs, 1)[0]
            
            # Bullish: increasing bid depth, decreasing ask depth
            if bid_trend > 0 and ask_trend < 0:
                return "bullish"
            # Bearish: decreasing bid depth, increasing ask depth
            elif bid_trend < 0 and ask_trend > 0:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error in cluster_slope for {symbol}: {e}")
            return "neutral"
    
    def analyze_symbol_simplified(self, symbol: str) -> Dict[str, str]:
        """
        Perform complete simplified heatmap analysis for a symbol
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Dictionary with simplified analysis results
        """
        logger.debug(f"🔍 Running simplified heatmap analysis for {symbol}")
        
        try:
            results = {
                'walls_disappear': self.walls_disappear(symbol),
                'pinning': self.pinning(symbol), 
                'void_reaction': self.void_reaction(symbol),
                'cluster_slope': self.cluster_slope(symbol)
            }
            
            logger.debug(f"✅ Simplified analysis completed for {symbol}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in simplified analysis for {symbol}: {e}")
            return {
                'walls_disappear': "❌",
                'pinning': "❌", 
                'void_reaction': "❌",
                'cluster_slope': "neutral"
            }
    
    def format_for_gpt_prompt(self, symbol: str) -> str:
        """
        Format simplified heatmap analysis for GPT prompt
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Formatted string for GPT context
        """
        analysis = self.analyze_symbol_simplified(symbol)
        
        return f"""=== ANALIZA HEATMAPY ORDERBOOKU ===
• Zniknięcie ścian: {analysis['walls_disappear']}
• Pinning płynności: {analysis['pinning']}
• Reakcja na void: {analysis['void_reaction']}
• Nachylenie klastrów: {analysis['cluster_slope']}"""


# Global instance
simplified_detector = None

def get_simplified_heatmap_detector() -> SimplifiedHeatmapDetectors:
    """Get global simplified heatmap detector instance"""
    global simplified_detector
    if simplified_detector is None:
        simplified_detector = SimplifiedHeatmapDetectors()
    return simplified_detector

# Convenience functions for direct usage
def walls_disappear(symbol: str) -> str:
    """Convenience function for walls_disappear detection"""
    return get_simplified_heatmap_detector().walls_disappear(symbol)

def pinning(symbol: str) -> str:
    """Convenience function for pinning detection"""
    return get_simplified_heatmap_detector().pinning(symbol)

def void_reaction(symbol: str) -> str:
    """Convenience function for void_reaction detection"""
    return get_simplified_heatmap_detector().void_reaction(symbol)

def cluster_slope(symbol: str) -> str:
    """Convenience function for cluster_slope analysis"""
    return get_simplified_heatmap_detector().cluster_slope(symbol)