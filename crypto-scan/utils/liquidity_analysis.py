#!/usr/bin/env python3
"""
Liquidity Behavior Analysis
Analizuje zachowanie płynności na podstawie orderbook i volume patterns
"""

import numpy as np
from typing import List, Dict, Optional
import json


def analyze_liquidity_behavior(symbol: str, market_data: Dict = None, candles: List[List] = None) -> Dict:
    """
    Liquidity Behavior Analysis
    
    Analizuje:
    - Stacking bidów (layered support)
    - Pinning pod EMA (price manipulation)
    - Buy pressure / absorpcja (institutional activity)
    
    Args:
        symbol: Trading symbol
        market_data: Optional market data containing orderbook
        candles: Optional candle data for volume analysis
        
    Returns:
        dict: {
            "liquidity_pattern_score": float,  # 0.0-1.0
            "bid_stacking": bool,
            "absorption_detected": bool,
            "pinning_behavior": bool,
            "volume_profile": dict
        }
    """
    try:
        # Initialize result structure
        result = {
            "liquidity_pattern_score": 0.5,
            "bid_stacking": False,
            "absorption_detected": False,
            "pinning_behavior": False,
            "volume_profile": {},
            "orderbook_analysis": {},
            "data_quality": "insufficient"
        }
        
        # 1. Orderbook Analysis (if available)
        orderbook_score = 0.5
        if market_data and market_data.get('orderbook'):
            orderbook_analysis = _analyze_orderbook_liquidity(market_data['orderbook'])
            result.update(orderbook_analysis)
            orderbook_score = orderbook_analysis.get("orderbook_score", 0.5)
        
        # 2. Volume Profile Analysis (if candles available)
        volume_score = 0.5
        if candles and len(candles) >= 10:
            volume_analysis = _analyze_volume_liquidity(candles)
            result["volume_profile"] = volume_analysis
            volume_score = volume_analysis.get("volume_score", 0.5)
        
        # 3. Calculate overall liquidity pattern score
        liquidity_score = (orderbook_score * 0.6 + volume_score * 0.4)
        result["liquidity_pattern_score"] = round(liquidity_score, 3)
        
        # 4. Determine data quality
        if market_data and candles:
            result["data_quality"] = "complete"
        elif market_data or candles:
            result["data_quality"] = "partial"
        else:
            result["data_quality"] = "insufficient"
        
        print(f"[LIQUIDITY] {symbol}: Score {liquidity_score:.3f} | Stacking: {result['bid_stacking']} | Absorption: {result['absorption_detected']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Liquidity analysis error for {symbol}: {e}")
        return {
            "liquidity_pattern_score": 0.5,
            "bid_stacking": False,
            "absorption_detected": False,
            "pinning_behavior": False,
            "volume_profile": {},
            "orderbook_analysis": {},
            "data_quality": "error"
        }


def _analyze_orderbook_liquidity(orderbook_data: Dict) -> Dict:
    """Analyze orderbook for liquidity patterns"""
    try:
        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])
        
        if not bids or not asks:
            return {"orderbook_score": 0.5, "bid_stacking": False, "orderbook_analysis": {}}
        
        # Extract bid and ask data
        bid_prices = [float(bid[0]) for bid in bids[:10]]
        bid_sizes = [float(bid[1]) for bid in bids[:10]]
        ask_prices = [float(ask[0]) for ask in asks[:10]]
        ask_sizes = [float(ask[1]) for ask in asks[:10]]
        
        # 1. Bid Stacking Analysis
        bid_stacking_result = _detect_bid_stacking(bid_prices, bid_sizes)
        
        # 2. Liquidity Imbalance Analysis
        imbalance_result = _analyze_liquidity_imbalance(bid_sizes, ask_sizes)
        
        # 3. Spread Analysis
        spread_result = _analyze_bid_ask_spread(bid_prices[0], ask_prices[0])
        
        # 4. Large Order Detection
        large_order_result = _detect_large_orders(bid_sizes, ask_sizes)
        
        # Calculate overall orderbook score
        orderbook_score = (
            bid_stacking_result["stacking_score"] * 0.3 +
            imbalance_result["imbalance_score"] * 0.3 +
            spread_result["spread_score"] * 0.2 +
            large_order_result["large_order_score"] * 0.2
        )
        
        return {
            "orderbook_score": round(orderbook_score, 3),
            "bid_stacking": bid_stacking_result["bid_stacking"],
            "orderbook_analysis": {
                "bid_stacking_details": bid_stacking_result,
                "imbalance_details": imbalance_result,
                "spread_details": spread_result,
                "large_order_details": large_order_result,
                "total_bid_volume": sum(bid_sizes),
                "total_ask_volume": sum(ask_sizes)
            }
        }
        
    except Exception as e:
        print(f"❌ Orderbook liquidity analysis error: {e}")
        return {"orderbook_score": 0.5, "bid_stacking": False, "orderbook_analysis": {}}


def _detect_bid_stacking(bid_prices: List[float], bid_sizes: List[float]) -> Dict:
    """Detect bid stacking patterns"""
    try:
        if len(bid_sizes) < 5:
            return {"bid_stacking": False, "stacking_score": 0.5}
        
        # Check for large bids in top 5 levels
        avg_bid_size = np.mean(bid_sizes)
        large_bids = [size for size in bid_sizes[:5] if size > avg_bid_size * 1.5]
        
        # Stacking criteria
        bid_stacking = len(large_bids) >= 3
        
        # Calculate stacking score
        if bid_stacking:
            stacking_score = min(0.9, 0.6 + (len(large_bids) * 0.1))
        else:
            stacking_score = 0.3 + (len(large_bids) * 0.1)
        
        return {
            "bid_stacking": bid_stacking,
            "stacking_score": stacking_score,
            "large_bids_count": len(large_bids),
            "avg_bid_size": avg_bid_size,
            "top_bid_size": max(bid_sizes[:5]) if bid_sizes else 0
        }
        
    except Exception:
        return {"bid_stacking": False, "stacking_score": 0.5}


def _analyze_liquidity_imbalance(bid_sizes: List[float], ask_sizes: List[float]) -> Dict:
    """Analyze bid/ask liquidity imbalance"""
    try:
        total_bid_volume = sum(bid_sizes[:5])  # Top 5 levels
        total_ask_volume = sum(ask_sizes[:5])
        total_volume = total_bid_volume + total_ask_volume
        
        if total_volume == 0:
            return {"imbalance_score": 0.5, "imbalance_ratio": 0}
        
        imbalance_ratio = (total_bid_volume - total_ask_volume) / total_volume
        
        # Score based on imbalance (positive = more bids = bullish)
        if imbalance_ratio > 0.3:  # Strong bid imbalance
            imbalance_score = 0.8
        elif imbalance_ratio > 0.1:  # Moderate bid imbalance
            imbalance_score = 0.7
        elif imbalance_ratio > -0.1:  # Balanced
            imbalance_score = 0.5
        elif imbalance_ratio > -0.3:  # Moderate ask imbalance
            imbalance_score = 0.3
        else:  # Strong ask imbalance
            imbalance_score = 0.2
        
        return {
            "imbalance_score": imbalance_score,
            "imbalance_ratio": round(imbalance_ratio, 3),
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume
        }
        
    except Exception:
        return {"imbalance_score": 0.5, "imbalance_ratio": 0}


def _analyze_bid_ask_spread(best_bid: float, best_ask: float) -> Dict:
    """Analyze bid-ask spread"""
    try:
        if best_bid <= 0 or best_ask <= 0:
            return {"spread_score": 0.5, "spread_pct": 0}
        
        spread_pct = (best_ask - best_bid) / best_bid * 100
        
        # Score based on spread (tighter = better liquidity)
        if spread_pct < 0.1:  # Very tight spread
            spread_score = 0.9
        elif spread_pct < 0.2:  # Tight spread
            spread_score = 0.7
        elif spread_pct < 0.5:  # Normal spread
            spread_score = 0.5
        elif spread_pct < 1.0:  # Wide spread
            spread_score = 0.3
        else:  # Very wide spread
            spread_score = 0.1
        
        return {
            "spread_score": spread_score,
            "spread_pct": round(spread_pct, 4),
            "best_bid": best_bid,
            "best_ask": best_ask
        }
        
    except Exception:
        return {"spread_score": 0.5, "spread_pct": 0}


def _detect_large_orders(bid_sizes: List[float], ask_sizes: List[float]) -> Dict:
    """Detect unusually large orders"""
    try:
        all_sizes = bid_sizes + ask_sizes
        if not all_sizes:
            return {"large_order_score": 0.5, "large_orders_detected": False}
        
        avg_size = np.mean(all_sizes)
        max_size = max(all_sizes)
        
        # Large order detection
        large_order_threshold = avg_size * 3
        large_orders = [size for size in all_sizes if size > large_order_threshold]
        
        large_orders_detected = len(large_orders) > 0
        
        # Score based on large orders presence
        if large_orders_detected:
            # Large orders can indicate institutional interest (positive)
            # but also potential manipulation (negative)
            large_order_score = 0.6 + min(0.2, len(large_orders) * 0.05)
        else:
            large_order_score = 0.5
        
        return {
            "large_order_score": large_order_score,
            "large_orders_detected": large_orders_detected,
            "large_orders_count": len(large_orders),
            "max_order_size": max_size,
            "avg_order_size": avg_size
        }
        
    except Exception:
        return {"large_order_score": 0.5, "large_orders_detected": False}


def _analyze_volume_liquidity(candles: List[List]) -> Dict:
    """Analyze volume patterns for liquidity insights"""
    try:
        if len(candles) < 10:
            return {"volume_score": 0.5, "absorption_detected": False}
        
        # Extract volume and price data
        volumes = [float(candle[5]) for candle in candles[-20:]]  # Last 20 candles
        closes = [float(candle[4]) for candle in candles[-20:]]
        highs = [float(candle[2]) for candle in candles[-20:]]
        lows = [float(candle[3]) for candle in candles[-20:]]
        
        # 1. Volume Profile Analysis
        volume_profile = _analyze_volume_profile(volumes, closes)
        
        # 2. Absorption Detection
        absorption_analysis = _detect_absorption_patterns(volumes, closes, highs, lows)
        
        # 3. Volume Trend Analysis
        volume_trend = _analyze_volume_trend(volumes)
        
        # Calculate overall volume score
        volume_score = (
            volume_profile["profile_score"] * 0.4 +
            absorption_analysis["absorption_score"] * 0.4 +
            volume_trend["trend_score"] * 0.2
        )
        
        return {
            "volume_score": round(volume_score, 3),
            "absorption_detected": absorption_analysis["absorption_detected"],
            "volume_profile_details": volume_profile,
            "absorption_details": absorption_analysis,
            "volume_trend_details": volume_trend
        }
        
    except Exception as e:
        print(f"❌ Volume liquidity analysis error: {e}")
        return {"volume_score": 0.5, "absorption_detected": False}


def _analyze_volume_profile(volumes: List[float], closes: List[float]) -> Dict:
    """Analyze volume profile patterns"""
    try:
        if not volumes:
            return {"profile_score": 0.5}
        
        recent_volumes = volumes[-5:]  # Last 5 candles
        earlier_volumes = volumes[-15:-5] if len(volumes) >= 15 else volumes[:-5]
        
        recent_avg = np.mean(recent_volumes)
        earlier_avg = np.mean(earlier_volumes) if earlier_volumes else recent_avg
        
        # Volume change analysis
        if earlier_avg > 0:
            volume_change_ratio = recent_avg / earlier_avg
        else:
            volume_change_ratio = 1.0
        
        # Score based on volume pattern
        if volume_change_ratio > 1.5:  # Increasing volume
            profile_score = 0.8
            volume_pattern = "increasing"
        elif volume_change_ratio > 1.2:  # Moderate increase
            profile_score = 0.6
            volume_pattern = "moderate_increase"
        elif volume_change_ratio < 0.7:  # Declining volume
            profile_score = 0.4
            volume_pattern = "declining"
        else:  # Stable volume
            profile_score = 0.5
            volume_pattern = "stable"
        
        return {
            "profile_score": profile_score,
            "volume_pattern": volume_pattern,
            "volume_change_ratio": round(volume_change_ratio, 3),
            "recent_avg_volume": recent_avg,
            "earlier_avg_volume": earlier_avg
        }
        
    except Exception:
        return {"profile_score": 0.5}


def _detect_absorption_patterns(volumes: List[float], closes: List[float], 
                               highs: List[float], lows: List[float]) -> Dict:
    """Detect volume absorption patterns"""
    try:
        if len(volumes) < 5:
            return {"absorption_detected": False, "absorption_score": 0.5}
        
        # Look for high volume with small price movement (absorption)
        absorption_signals = 0
        avg_volume = np.mean(volumes)
        
        for i in range(-5, 0):  # Last 5 candles
            if i < len(volumes):
                volume = volumes[i]
                price_range = (highs[i] - lows[i]) / closes[i]  # Normalized range
                
                # High volume + small range = potential absorption
                if volume > avg_volume * 1.5 and price_range < 0.02:  # <2% range
                    absorption_signals += 1
        
        absorption_detected = absorption_signals >= 2
        
        # Score based on absorption signals
        if absorption_detected:
            absorption_score = 0.7 + min(0.2, absorption_signals * 0.05)
        else:
            absorption_score = 0.4 + (absorption_signals * 0.1)
        
        return {
            "absorption_detected": absorption_detected,
            "absorption_score": absorption_score,
            "absorption_signals": absorption_signals,
            "avg_volume": avg_volume
        }
        
    except Exception:
        return {"absorption_detected": False, "absorption_score": 0.5}


def _analyze_volume_trend(volumes: List[float]) -> Dict:
    """Analyze volume trend"""
    try:
        if len(volumes) < 10:
            return {"trend_score": 0.5, "volume_trend": "unknown"}
        
        recent_volumes = volumes[-5:]
        earlier_volumes = volumes[-10:-5]
        
        recent_avg = np.mean(recent_volumes)
        earlier_avg = np.mean(earlier_volumes)
        
        if earlier_avg > 0:
            trend_ratio = recent_avg / earlier_avg
        else:
            trend_ratio = 1.0
        
        # Determine trend
        if trend_ratio > 1.3:
            volume_trend = "strongly_increasing"
            trend_score = 0.8
        elif trend_ratio > 1.1:
            volume_trend = "increasing"
            trend_score = 0.6
        elif trend_ratio < 0.8:
            volume_trend = "decreasing"
            trend_score = 0.4
        else:
            volume_trend = "stable"
            trend_score = 0.5
        
        return {
            "trend_score": trend_score,
            "volume_trend": volume_trend,
            "trend_ratio": round(trend_ratio, 3)
        }
        
    except Exception:
        return {"trend_score": 0.5, "volume_trend": "unknown"}