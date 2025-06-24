#!/usr/bin/env python3
"""
Async Trend Mode Scanner - Full asyncio + aiohttp refactor
Replaces blocking safe_candles.get_candles() with async implementation
Target: <15 seconds for 500+ tokens
"""

import asyncio
import aiohttp
import time
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# Import essential modules
from utils.bybit_cache_manager import get_bybit_symbols_cached
from utils.coingecko import build_coingecko_cache
from utils.whale_priority import prioritize_whale_tokens

class AsyncTrendModeScanner:
    """High-performance async trend mode scanner"""
    
    def __init__(self, max_concurrent: int = 20):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.api_calls_count = 0
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=10, connect=3)
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=25,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": "TrendMode/2.0"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        print(f"Total API calls made: {self.api_calls_count}")
    
    async def get_candles_async(self, symbol: str, interval: str = "15", limit: int = 100) -> Optional[List[Dict]]:
        """Async replacement for safe_candles.get_candles() - main bottleneck fix"""
        try:
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": str(limit)
            }
            
            self.api_calls_count += 1
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                if not data.get("result", {}).get("list"):
                    return None
                
                candles = []
                for candle_raw in data["result"]["list"]:
                    candles.append({
                        "timestamp": int(candle_raw[0]),
                        "open": float(candle_raw[1]),
                        "high": float(candle_raw[2]),
                        "low": float(candle_raw[3]),
                        "close": float(candle_raw[4]),
                        "volume": float(candle_raw[5]),
                        "turnover": float(candle_raw[6]) if len(candle_raw) > 6 else 0
                    })
                
                return candles
                
        except Exception as e:
            print(f"Error fetching candles for {symbol}: {e}")
            return None
    
    async def get_ticker_async(self, symbol: str) -> Optional[Dict]:
        """Async ticker data fetch"""
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {"category": "linear", "symbol": symbol}
            
            self.api_calls_count += 1
            
            async with self.session.get(url, params=params) as response:
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
                    "low_24h": float(ticker.get("lowPrice24h", 0)),
                    "bid": float(ticker.get("bid1Price", 0)),
                    "ask": float(ticker.get("ask1Price", 0))
                }
                
        except Exception:
            return None
    
    async def get_orderbook_async(self, symbol: str, depth: int = 25) -> Optional[Dict]:
        """Async orderbook data fetch"""
        try:
            url = "https://api.bybit.com/v5/market/orderbook"
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": str(depth)
            }
            
            self.api_calls_count += 1
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                result = data.get("result", {})
                
                bids = []
                asks = []
                
                if result.get("b"):
                    for bid in result["b"]:
                        bids.append({
                            "price": float(bid[0]),
                            "size": float(bid[1])
                        })
                
                if result.get("a"):
                    for ask in result["a"]:
                        asks.append({
                            "price": float(ask[0]),
                            "size": float(ask[1])
                        })
                
                return {
                    "symbol": symbol,
                    "bids": bids,
                    "asks": asks,
                    "best_bid": bids[0]["price"] if bids else 0,
                    "best_ask": asks[0]["price"] if asks else 0
                }
                
        except Exception:
            return None
    
    async def fetch_all_market_data_async(self, symbol: str) -> Optional[Dict]:
        """Fetch all required market data in parallel - eliminates sequential bottleneck"""
        try:
            # Parallel fetch all data types
            ticker_task = self.get_ticker_async(symbol)
            candles_15m_task = self.get_candles_async(symbol, "15", 50)
            candles_5m_task = self.get_candles_async(symbol, "5", 50)
            orderbook_task = self.get_orderbook_async(symbol, 25)
            
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
                orderbook = {"bids": [], "asks": [], "best_bid": 0, "best_ask": 0}
            
            return {
                "symbol": symbol,
                "ticker": ticker,
                "candles_15m": candles_15m,
                "candles_5m": candles_5m,
                "orderbook": orderbook,
                "price": ticker["price"],
                "volume_24h": ticker["volume_24h"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return None
    
    def analyze_liquidity_fast(self, market_data: Dict) -> Dict:
        """Fast liquidity analysis from fetched data"""
        signals = {}
        orderbook = market_data.get("orderbook", {})
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        # Liquidity depth analysis
        if bids and asks:
            bid_depth = sum(bid["size"] for bid in bids[:5])
            ask_depth = sum(ask["size"] for ask in asks[:5])
            
            signals["liquidity_depth"] = bid_depth + ask_depth
            signals["bid_ask_ratio"] = bid_depth / ask_depth if ask_depth > 0 else 1
            signals["spread"] = (asks[0]["price"] - bids[0]["price"]) / bids[0]["price"] if bids and asks else 0
        
        return signals
    
    def analyze_candles_fast(self, market_data: Dict) -> Dict:
        """Fast candle analysis from fetched data"""
        signals = {}
        candles_15m = market_data.get("candles_15m", [])
        candles_5m = market_data.get("candles_5m", [])
        
        if candles_15m:
            # Recent price action
            recent_candles = candles_15m[-10:]
            closes = [c["close"] for c in recent_candles]
            volumes = [c["volume"] for c in recent_candles]
            
            if len(closes) >= 3:
                signals["price_trend"] = "up" if closes[-1] > closes[-3] else "down"
                signals["volume_trend"] = "increasing" if volumes[-1] > volumes[-3] else "decreasing"
                signals["volatility"] = (max(closes) - min(closes)) / closes[-1] if closes[-1] > 0 else 0
        
        if candles_5m:
            # Short-term momentum
            recent_5m = candles_5m[-5:]
            if recent_5m:
                signals["momentum_5m"] = "bullish" if recent_5m[-1]["close"] > recent_5m[0]["open"] else "bearish"
        
        return signals
    
    def calculate_trend_score(self, market_data: Dict, liquidity_signals: Dict, candle_signals: Dict) -> float:
        """Calculate trend mode score from all signals"""
        score = 0.0
        
        # Volume scoring
        volume_24h = market_data.get("volume_24h", 0)
        if volume_24h > 10_000_000:
            score += 35
        elif volume_24h > 5_000_000:
            score += 25
        elif volume_24h > 1_000_000:
            score += 15
        
        # Price trend scoring
        if candle_signals.get("price_trend") == "up":
            score += 20
        
        # Volume trend scoring
        if candle_signals.get("volume_trend") == "increasing":
            score += 15
        
        # Momentum scoring
        if candle_signals.get("momentum_5m") == "bullish":
            score += 10
        
        # Liquidity scoring
        liquidity_depth = liquidity_signals.get("liquidity_depth", 0)
        if liquidity_depth > 1000:
            score += 10
        
        # Spread penalty
        spread = liquidity_signals.get("spread", 0)
        if spread > 0.01:  # 1% spread penalty
            score -= 10
        
        return min(100, max(0, score))
    
    def simulate_trader_decision(self, score: float, market_data: Dict) -> str:
        """Simulate trader decision based on score and market context"""
        if score >= 75:
            return "join_trend"
        elif score >= 50:
            return "consider_entry"
        elif score >= 25:
            return "monitor"
        else:
            return "avoid"
    
    async def scan_token_async(self, symbol: str) -> Optional[Dict]:
        """Async scan single token - replaces blocking scan_token()"""
        async with self.semaphore:
            try:
                start_time = time.time()
                
                # Fetch all market data in parallel
                market_data = await self.fetch_all_market_data_async(symbol)
                if not market_data:
                    return None
                
                # Basic validation
                price = market_data["price"]
                volume_24h = market_data["volume_24h"]
                
                if price <= 0 or volume_24h < 500_000:  # Minimum liquidity threshold
                    return None
                
                # Fast analysis pipeline
                liquidity_signals = self.analyze_liquidity_fast(market_data)
                candle_signals = self.analyze_candles_fast(market_data)
                
                # Scoring and decision
                score = self.calculate_trend_score(market_data, liquidity_signals, candle_signals)
                decision = self.simulate_trader_decision(score, market_data)
                
                # Save result
                self.save_trend_result(symbol, score, decision, market_data)
                
                scan_time = time.time() - start_time
                
                return {
                    "symbol": symbol,
                    "score": score,
                    "decision": decision,
                    "price": price,
                    "volume_24h": volume_24h,
                    "scan_time": scan_time,
                    "liquidity_signals": liquidity_signals,
                    "candle_signals": candle_signals
                }
                
            except Exception as e:
                print(f"Error in async scan for {symbol}: {e}")
                return None
    
    def save_trend_result(self, symbol: str, score: float, decision: str, market_data: Dict):
        """Save trend mode result"""
        try:
            result = {
                "symbol": symbol,
                "score": score,
                "decision": decision,
                "price": market_data["price"],
                "volume_24h": market_data["volume_24h"],
                "timestamp": datetime.now().isoformat(),
                "scan_method": "async_trend_mode"
            }
            
            os.makedirs("data/trend_results", exist_ok=True)
            with open(f"data/trend_results/{symbol}_trend.json", "w") as f:
                json.dump(result, f, indent=2)
                
            # Send alert for high-score decisions
            if score >= 70 and decision in ["join_trend", "consider_entry"]:
                self.send_trend_alert(symbol, score, decision)
                
        except Exception as e:
            print(f"Save error for {symbol}: {e}")
    
    def send_trend_alert(self, symbol: str, score: float, decision: str):
        """Send trend mode alert for high-score setups"""
        try:
            alert_message = f"🚀 TREND-MODE: {symbol} {decision.upper()} - Score: {score:.1f}"
            
            # Save alert to file for processing
            os.makedirs("data/alerts", exist_ok=True)
            alert_data = {
                "symbol": symbol,
                "score": score,
                "decision": decision,
                "message": alert_message,
                "timestamp": datetime.now().isoformat(),
                "type": "trend_mode"
            }
            
            with open(f"data/alerts/{symbol}_trend_alert.json", "w") as f:
                json.dump(alert_data, f, indent=2)
                
            print(f"ALERT: {alert_message}")
            
        except Exception as e:
            print(f"Alert error for {symbol}: {e}")

async def async_trend_scan_cycle():
    """Main async trend mode scan cycle"""
    print(f"Starting async trend mode scan at {datetime.now().strftime('%H:%M:%S')}")
    
    # Get symbols
    symbols = get_bybit_symbols_cached()
    print(f"Scanning {len(symbols)} symbols with async trend mode")
    
    # Build cache
    build_coingecko_cache()
    
    # Whale priority
    symbols, priority_symbols, priority_info = prioritize_whale_tokens(symbols)
    
    # High-performance async scanning
    async with AsyncTrendModeScanner(max_concurrent=20) as scanner:
        start_time = time.time()
        
        # Create tasks for all symbols
        tasks = [scanner.scan_token_async(symbol) for symbol in symbols]
        
        # Execute with progress tracking
        results = []
        completed = 0
        
        # Process in chunks for memory efficiency
        chunk_size = 50
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i + chunk_size]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            for j, result in enumerate(chunk_results):
                completed += 1
                symbol = symbols[i + j]
                
                if isinstance(result, Exception):
                    print(f"[{completed}/{len(symbols)}] {symbol}: Error - {result}")
                elif result:
                    results.append(result)
                    print(f"[{completed}/{len(symbols)}] {symbol}: Score {result['score']:.1f} ({result['scan_time']:.3f}s) - {result['decision'].upper()}")
                else:
                    print(f"[{completed}/{len(symbols)}] {symbol}: Skipped")
        
        total_time = time.time() - start_time
        tokens_per_second = len(symbols) / total_time if total_time > 0 else 0
        
        print(f"\n🎯 ASYNC TREND MODE RESULTS:")
        print(f"- Processed: {len(results)}/{len(symbols)} tokens")
        print(f"- Total time: {total_time:.1f}s (TARGET: <15s)")
        print(f"- Performance: {tokens_per_second:.1f} tokens/second")
        print(f"- API calls made: {scanner.api_calls_count}")
        print(f"- Average scan time: {total_time/len(symbols):.3f}s per token")
        
        # Show top performers
        if results:
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            print(f"\n🔥 TOP 5 TREND SETUPS:")
            for i, result in enumerate(sorted_results[:5], 1):
                print(f"{i}. {result['symbol']}: {result['score']:.1f} - {result['decision'].upper()}")
        
        return results

def wait_for_next_candle():
    """Wait for next 15-minute candle"""
    now = datetime.now(timezone.utc)
    next_candle = now.replace(second=5, microsecond=0)
    
    if now.second >= 5:
        next_candle += timedelta(minutes=15 - (now.minute % 15))
    else:
        next_candle += timedelta(minutes=15 - (now.minute % 15))
    
    wait_seconds = (next_candle - now).total_seconds()
    
    if wait_seconds > 0:
        print(f"Waiting {wait_seconds:.1f}s for next candle at {next_candle.strftime('%H:%M:%S')} UTC")
        time.sleep(wait_seconds)

async def main():
    """Main async trend mode loop"""
    print("🚀 Starting Async Trend Mode Scanner")
    print("Target: <15 seconds for 500+ tokens")
    
    try:
        while True:
            try:
                await async_trend_scan_cycle()
                wait_for_next_candle()
            except KeyboardInterrupt:
                print("\nShutting down async trend mode...")
                break
            except Exception as e:
                print(f"Scan error: {e}")
                await asyncio.sleep(60)
                
    except KeyboardInterrupt:
        print("Async trend mode stopped.")

if __name__ == "__main__":
    asyncio.run(main())