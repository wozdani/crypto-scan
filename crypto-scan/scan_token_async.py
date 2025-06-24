#!/usr/bin/env python3
"""
Async Token Scanner - Single token analysis with full async pipeline
Replaces blocking scan_token() with high-performance async implementation
"""

import aiohttp
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import scoring and analysis modules
try:
    from utils.scoring import compute_ppwcs_score
    from trader_ai_engine import simulate_trader_decision_advanced
    from utils.alerts import send_alert
    from utils.whale_priority import check_whale_priority
except ImportError as e:
    print(f"Warning: Could not import module {e} - some features may be limited")

async def get_candles_async(symbol: str, interval: str, session: aiohttp.ClientSession, limit: int = 96) -> list:
    """Async candle fetcher replacing safe_candles.get_candles()"""
    try:
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": str(limit)
        }
        
        async with session.get(url, params=params, timeout=5) as response:
            if response.status != 200:
                return []
            
            data = await response.json()
            candles_raw = data.get("result", {}).get("list", [])
            
            # Convert to standard format
            candles = []
            for candle_data in reversed(candles_raw):  # Bybit returns newest first
                try:
                    candles.append([
                        int(candle_data[0]),      # timestamp
                        float(candle_data[1]),    # open
                        float(candle_data[2]),    # high
                        float(candle_data[3]),    # low
                        float(candle_data[4]),    # close
                        float(candle_data[5])     # volume
                    ])
                except (ValueError, IndexError):
                    continue
            
            return candles
            
    except Exception as e:
        print(f"Error fetching {interval}m candles for {symbol}: {e}")
        return []

async def get_ticker_async(symbol: str, session: aiohttp.ClientSession) -> Optional[Dict]:
    """Async ticker data fetch"""
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        
        async with session.get(url, params=params, timeout=5) as response:
            if response.status != 200:
                return None
            
            data = await response.json()
            if not data.get("result", {}).get("list"):
                return None
            
            ticker = data["result"]["list"][0]
            return {
                "symbol": symbol,
                "price": float(ticker.get("lastPrice", 0)),
                "volume_24h": float(ticker.get("volume24h", 0)),
                "price_change_24h": float(ticker.get("price24hPcnt", 0)),
                "high_24h": float(ticker.get("highPrice24h", 0)),
                "low_24h": float(ticker.get("lowPrice24h", 0))
            }
            
    except Exception:
        return None

async def get_orderbook_async(symbol: str, session: aiohttp.ClientSession, depth: int = 25) -> Optional[Dict]:
    """Async orderbook fetch with production diagnostics"""
    try:
        url = "https://api.bybit.com/v5/market/orderbook"
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": str(depth)
        }
        
        async with session.get(url, params=params, timeout=5) as response:
            if response.status != 200:
                print(f"[ORDERBOOK PROD] {symbol} → HTTP {response.status}")
                return None
            
            data = await response.json()
            result = data.get("result", {})
            
            orderbook_data = {
                "symbol": symbol,
                "bids": [[float(bid[0]), float(bid[1])] for bid in result.get("b", [])],
                "asks": [[float(ask[0]), float(ask[1])] for ask in result.get("a", [])]
            }
            
            print(f"[ORDERBOOK PROD] {symbol} → {len(orderbook_data['bids'])} bids, {len(orderbook_data['asks'])} asks")
            return orderbook_data
            
    except Exception as e:
        print(f"[ORDERBOOK PROD ERROR] {symbol} → {e}")
        return None

async def scan_token_async(symbol: str, session: aiohttp.ClientSession, priority_info: Dict = None) -> Optional[Dict]:
    """
    Complete async token scan with scoring, TJDE analysis, and alerts
    Replaces blocking scan_token() function
    """
    print(f"[SCAN START] {symbol} → Beginning async scan")
    
    try:
        # Parallel data fetching
        ticker_task = get_ticker_async(symbol, session)
        candles_15m_task = get_candles_async(symbol, "15", session, 96)
        candles_5m_task = get_candles_async(symbol, "5", session, 50)
        orderbook_task = get_orderbook_async(symbol, session, 25)
        
        ticker, candles_15m, candles_5m, orderbook = await asyncio.gather(
            ticker_task, candles_15m_task, candles_5m_task, orderbook_task,
            return_exceptions=True
        )
        
        # Validate core data with debugging
        print(f"[DATA VALIDATION] {symbol} → ticker: {type(ticker)}, 15m: {type(candles_15m)}, 5m: {type(candles_5m)}, orderbook: {type(orderbook)}")
        
        if isinstance(ticker, Exception) or not ticker:
            print(f"[DATA VALIDATION FAILED] {symbol} → No ticker data: {ticker}")
            return None
            
        if isinstance(candles_15m, Exception):
            print(f"[DATA VALIDATION] {symbol} → 15m candles failed: {candles_15m}")
            candles_15m = []
        if isinstance(candles_5m, Exception):
            print(f"[DATA VALIDATION] {symbol} → 5m candles failed: {candles_5m}")
            candles_5m = []
        if isinstance(orderbook, Exception):
            print(f"[DATA VALIDATION] {symbol} → orderbook failed: {orderbook}")
            orderbook = {"bids": [], "asks": []}
        
        # Basic filtering
        price = ticker["price"]
        volume_24h = ticker["volume_24h"]
        
        if price <= 0 or volume_24h < 500_000:
            return None
        
        # Build market data structure
        market_data = {
            "symbol": symbol,
            "price_usd": price,
            "volume_24h": volume_24h,
            "price_change_24h": ticker["price_change_24h"],
            "candles": candles_15m,
            "candles_5m": candles_5m,
            "orderbook": orderbook,
            "ticker": ticker
        }
        
        # PPWCS Scoring with detailed debugging
        try:
            print(f"[DEBUG] {symbol} → Running PPWCS scoring")
            print(f"[DEBUG] {symbol} → market_data keys: {list(market_data.keys()) if isinstance(market_data, dict) else 'NOT DICT'}")
            print(f"[DEBUG] {symbol} → market_data type: {type(market_data)}")
            
            signals = compute_ppwcs_score(market_data)
            ppwcs_score = signals.get("final_score", 0) if signals else 0
            
            print(f"[PPWCS SUCCESS] {symbol}: score={ppwcs_score}, signals_count={len(signals.get('signals', {}) if signals else {})}")
            
        except Exception as e:
            print(f"[PPWCS FALLBACK] {symbol}: {type(e).__name__}: {e}")
            print(f"[PPWCS FALLBACK] {symbol} → market_data structure:")
            if isinstance(market_data, dict):
                for key, value in market_data.items():
                    print(f"  {key}: {type(value)} = {str(value)[:100]}")
            ppwcs_score = calculate_basic_score(market_data)
            signals = {"final_score": ppwcs_score, "signals": {}}
        
        # TJDE Analysis with detailed debugging
        try:
            print(f"[DEBUG] {symbol} → Running TJDE analysis")
            print(f"[DEBUG] {symbol} → candles_15m: {len(candles_15m) if candles_15m and isinstance(candles_15m, list) else 'INVALID'}")
            print(f"[DEBUG] {symbol} → candles_5m: {len(candles_5m) if candles_5m and isinstance(candles_5m, list) else 'INVALID'}")
            print(f"[DEBUG] {symbol} → orderbook: {type(orderbook)} = {str(orderbook)[:100] if orderbook else 'None'}")
            print(f"[DEBUG] {symbol} → price: {price} ({type(price)})")
            print(f"[DEBUG] {symbol} → volume_24h: {volume_24h} ({type(volume_24h)})")
            
            tjde_result = simulate_trader_decision_advanced(
                symbol=symbol,
                market_data=market_data,
                enable_debug=False
            )
            
            tjde_score = tjde_result.get("score", 0) if tjde_result else 0
            tjde_decision = tjde_result.get("decision", "avoid") if tjde_result else "avoid"
            
            print(f"[TJDE SUCCESS] {symbol}: score={tjde_score}, decision={tjde_decision}")
            if tjde_result and isinstance(tjde_result, dict):
                print(f"[TJDE SUCCESS] {symbol} → result keys: {list(tjde_result.keys())}")
            
        except Exception as e:
            print(f"[TJDE FALLBACK] {symbol}: {type(e).__name__}: {e}")
            print(f"[TJDE FALLBACK] {symbol} → Input validation:")
            print(f"  candles_15m: {type(candles_15m)} len={len(candles_15m) if candles_15m and hasattr(candles_15m, '__len__') else 'N/A'}")
            print(f"  candles_5m: {type(candles_5m)} len={len(candles_5m) if candles_5m and hasattr(candles_5m, '__len__') else 'N/A'}")
            print(f"  orderbook: {type(orderbook)}")
            print(f"  price: {price} ({type(price)})")
            print(f"  volume_24h: {volume_24h} ({type(volume_24h)})")
            
            tjde_score = ppwcs_score / 100 if ppwcs_score else 0.4  # Convert to 0-1 range
            tjde_decision = "monitor" if tjde_score > 0.5 else "avoid"
        
        # Alert processing with diagnostics
        alert_sent = False
        print(f"[ALERT CHECK] {symbol} → PPWCS: {ppwcs_score}, TJDE: {tjde_score}")
        
        if ppwcs_score >= 40 or tjde_score >= 0.7:
            print(f"[ALERT TRIGGER] {symbol} → Sending alert (PPWCS≥40 or TJDE≥0.7)")
            try:
                alert_sent = await send_async_alert(symbol, ppwcs_score, tjde_score, tjde_decision, market_data)
                print(f"[ALERT RESULT] {symbol} → Alert sent: {alert_sent}")
            except Exception as e:
                print(f"[ALERT ERROR] {symbol} → {e}")
        else:
            print(f"[ALERT SKIP] {symbol} → Below thresholds (PPWCS: {ppwcs_score:.1f}, TJDE: {tjde_score:.3f})")
        
        # Save results with diagnostics
        try:
            save_async_result(symbol, ppwcs_score, tjde_score, tjde_decision, market_data)
            print(f"[SAVE RESULT] {symbol} → Results saved successfully")
        except Exception as e:
            print(f"[SAVE ERROR] {symbol} → {e}")
        
        result = {
            "symbol": symbol,
            "ppwcs_score": ppwcs_score,
            "tjde_score": tjde_score,
            "tjde_decision": tjde_decision,
            "price": price,
            "volume_24h": volume_24h,
            "alert_sent": alert_sent,
            "candles_count": len(candles_15m),
            "timestamp": datetime.now().isoformat()
        }
        
        # Final result summary
        print(f"[FINAL RESULT] {symbol} → PPWCS: {ppwcs_score:.1f}, TJDE: {tjde_score:.3f} ({tjde_decision})")
        
        # Check if scores are suspicious (identical fallback values)
        if ppwcs_score == 40 and tjde_score == 0.4:
            print(f"[SUSPICIOUS] {symbol} → Both scores are fallback values - likely errors in scoring functions")
        
        print(f"✅ {symbol}: PPWCS {ppwcs_score:.1f}, TJDE {tjde_score:.3f} ({tjde_decision}), {len(candles_15m)}x15M, {len(candles_5m)}x5M")
        return result
        
    except Exception as e:
        print(f"❌ {symbol} → error: {e}")
        return None

def calculate_basic_score(market_data: Dict) -> float:
    """Basic scoring when PPWCS unavailable"""
    print(f"[BASIC SCORE] Calculating fallback score")
    
    if not market_data:
        print(f"[BASIC SCORE] No market_data provided")
        return 40.0
    
    try:
        print(f"[BASIC SCORE] market_data available: {list(market_data.keys()) if isinstance(market_data, dict) else 'NOT DICT'}")
        
        score = 0.0
        volume_24h = market_data.get("volume_24h", 0)
        price_change = abs(market_data.get("price_change_24h", 0))
        
        print(f"[BASIC SCORE] volume_24h: {volume_24h}, price_change_24h: {price_change}%")
        
        # Volume scoring
        if volume_24h > 10_000_000:
            score += 40
        elif volume_24h > 5_000_000:
            score += 25
        elif volume_24h > 1_000_000:
            score += 15
        
        # Price movement scoring
        if price_change > 15:
            score += 35
        elif price_change > 10:
            score += 25
        elif price_change > 5:
            score += 15
        
        final_score = min(100, score) if score > 0 else 45.0
        print(f"[BASIC SCORE] Final calculated score: {final_score}")
        return final_score
        
    except Exception as e:
        print(f"[BASIC SCORE] Error calculating: {e}")
        return 40.0

async def send_async_alert(symbol: str, ppwcs_score: float, tjde_score: float, tjde_decision: str, market_data: Dict) -> bool:
    """Send alert for high-scoring tokens"""
    try:
        alert_message = f"🚀 ASYNC ALERT: {symbol}\n"
        alert_message += f"PPWCS: {ppwcs_score:.1f} | TJDE: {tjde_score:.3f} ({tjde_decision})\n"
        alert_message += f"Price: ${market_data['price_usd']:.6f} | Volume: ${market_data['volume_24h']:,.0f}"
        
        # Save alert to file
        os.makedirs("data/alerts", exist_ok=True)
        alert_data = {
            "symbol": symbol,
            "ppwcs_score": ppwcs_score,
            "tjde_score": tjde_score,
            "tjde_decision": tjde_decision,
            "message": alert_message,
            "timestamp": datetime.now().isoformat(),
            "type": "async_scan"
        }
        
        with open(f"data/alerts/{symbol}_async_alert.json", "w") as f:
            json.dump(alert_data, f, indent=2)
        
        print(f"ALERT: {alert_message}")
        return True
        
    except Exception as e:
        print(f"Alert error for {symbol}: {e}")
        return False

def save_async_result(symbol: str, ppwcs_score: float, tjde_score: float, tjde_decision: str, market_data: Dict):
    """Save scan result"""
    try:
        result = {
            "symbol": symbol,
            "ppwcs_score": ppwcs_score,
            "tjde_score": tjde_score,
            "tjde_decision": tjde_decision,
            "price_usd": market_data["price_usd"],
            "volume_24h": market_data["volume_24h"],
            "timestamp": datetime.now().isoformat(),
            "scan_method": "async_token_scan"
        }
        
        os.makedirs("data/async_results", exist_ok=True)
        with open(f"data/async_results/{symbol}_async.json", "w") as f:
            json.dump(result, f, indent=2)
            
    except Exception as e:
        print(f"Save error for {symbol}: {e}")

# Test function
async def test_single_token():
    """Test async scan on single token"""
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    async with aiohttp.ClientSession() as session:
        for symbol in test_symbols:
            result = await scan_token_async(symbol, session)
            if result:
                print(f"Test result for {symbol}: {result}")

if __name__ == "__main__":
    asyncio.run(test_single_token())