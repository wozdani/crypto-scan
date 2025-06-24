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
    """Async orderbook fetch"""
    try:
        url = "https://api.bybit.com/v5/market/orderbook"
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": str(depth)
        }
        
        async with session.get(url, params=params, timeout=5) as response:
            if response.status != 200:
                return None
            
            data = await response.json()
            result = data.get("result", {})
            
            return {
                "symbol": symbol,
                "bids": [[float(bid[0]), float(bid[1])] for bid in result.get("b", [])],
                "asks": [[float(ask[0]), float(ask[1])] for ask in result.get("a", [])]
            }
            
    except Exception:
        return None

async def scan_token_async(symbol: str, session: aiohttp.ClientSession, priority_info: Dict = None) -> Optional[Dict]:
    """
    Complete async token scan with scoring, TJDE analysis, and alerts
    Replaces blocking scan_token() function
    """
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
        
        # Validate core data
        if isinstance(ticker, Exception) or not ticker:
            return None
            
        if isinstance(candles_15m, Exception):
            candles_15m = []
        if isinstance(candles_5m, Exception):
            candles_5m = []
        if isinstance(orderbook, Exception):
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
        
        # PPWCS Scoring (if available)
        try:
            signals = compute_ppwcs_score(market_data)
            ppwcs_score = signals.get("final_score", 0)
        except:
            ppwcs_score = calculate_basic_score(market_data)
            signals = {"final_score": ppwcs_score, "signals": {}}
        
        # TJDE Analysis (if available)
        try:
            tjde_result = simulate_trader_decision_advanced(
                symbol=symbol,
                market_data=market_data,
                enable_debug=False
            )
            tjde_score = tjde_result.get("score", 0)
            tjde_decision = tjde_result.get("decision", "avoid")
        except:
            tjde_score = ppwcs_score / 100  # Convert to 0-1 range
            tjde_decision = "monitor" if tjde_score > 0.5 else "avoid"
        
        # Alert processing
        alert_sent = False
        if ppwcs_score >= 40 or tjde_score >= 0.7:
            try:
                alert_sent = await send_async_alert(symbol, ppwcs_score, tjde_score, tjde_decision, market_data)
            except:
                pass
        
        # Save results
        save_async_result(symbol, ppwcs_score, tjde_score, tjde_decision, market_data)
        
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
        
        print(f"✅ {symbol}: PPWCS {ppwcs_score:.1f}, TJDE {tjde_score:.3f} ({tjde_decision}), {len(candles_15m)}x15M, {len(candles_5m)}x5M")
        return result
        
    except Exception as e:
        print(f"❌ {symbol} → error: {e}")
        return None

def calculate_basic_score(market_data: Dict) -> float:
    """Basic scoring when PPWCS unavailable"""
    score = 0.0
    
    volume_24h = market_data.get("volume_24h", 0)
    price_change = abs(market_data.get("price_change_24h", 0))
    
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
    
    return min(100, score)

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