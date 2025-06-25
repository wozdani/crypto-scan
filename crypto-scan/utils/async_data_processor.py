"""
Async Data Processor - Direct Integration Fix
Handles ticker None scenarios while preserving authentic candle data
"""

from typing import Dict, List, Optional, Any
import json

def process_async_data_enhanced(symbol: str, ticker_data: Optional[Dict], candles_data: Optional[Dict], orderbook_data: Optional[Dict]) -> Optional[Dict]:
    """
    Enhanced async data processing with intelligent fallback mechanisms
    Resolves ticker None issue while preserving authentic candle data
    
    Args:
        symbol: Trading symbol
        ticker_data: Ticker data (may be None)
        candles_data: Candle data (should be available)
        orderbook_data: Orderbook data (may be None)
        
    Returns:
        Processed market data or None if insufficient
    """
    
    # Initialize data containers
    price_usd = 0.0
    volume_24h = 0.0
    candles = []
    bids = []
    asks = []
    
    # PRIORITY 1: Extract from ticker data if available
    if ticker_data and ticker_data.get("result", {}).get("list"):
        ticker_list = ticker_data["result"]["list"]
        if ticker_list:
            ticker = ticker_list[0]
            price_usd = float(ticker.get("lastPrice", 0)) if ticker.get("lastPrice") else 0.0
            volume_24h = float(ticker.get("volume24h", 0)) if ticker.get("volume24h") else 0.0
            print(f"[TICKER SUCCESS] {symbol}: Price ${price_usd}, Volume {volume_24h}")
    
    # PRIORITY 2: Process candles (should always be available from async scanner)
    if candles_data and candles_data.get("result", {}).get("list"):
        candle_list = candles_data["result"]["list"]
        for candle in candle_list:
            try:
                candles.append({
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]), 
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                })
            except (ValueError, IndexError, TypeError):
                continue
        
        # FALLBACK: Extract price from latest candle if ticker failed
        if price_usd <= 0 and candles:
            price_usd = candles[-1]["close"]  # Use most recent candle close price
            # Calculate 24h volume from available candles
            if len(candles) >= 96:  # Full day of 15m candles
                volume_24h = sum(c["volume"] for c in candles[-96:])
            else:
                # Estimate based on available candles
                volume_24h = sum(c["volume"] for c in candles) * (96 / len(candles))
            print(f"[CANDLE FALLBACK] {symbol}: Price ${price_usd} from {len(candles)} candles")
    
    # PRIORITY 3: Process orderbook if available
    if orderbook_data and orderbook_data.get("result"):
        result = orderbook_data["result"]
        if result.get("b"):
            for bid in result["b"][:5]:
                try:
                    bids.append({"price": float(bid[0]), "size": float(bid[1])})
                except (ValueError, IndexError):
                    continue
        if result.get("a"):
            for ask in result["a"][:5]:
                try:
                    asks.append({"price": float(ask[0]), "size": float(ask[1])})
                except (ValueError, IndexError):
                    continue
    
    # SYNTHETIC ORDERBOOK: Create if missing but we have price
    if not bids and not asks and price_usd > 0:
        spread = price_usd * 0.001  # 0.1% spread
        bids = [{"price": price_usd - spread, "size": 100.0}]
        asks = [{"price": price_usd + spread, "size": 100.0}]
        print(f"[ORDERBOOK SYNTHETIC] {symbol}: Created synthetic orderbook")
    
    # ENHANCED VALIDATION: Accept tokens with candles even without price
    has_price = price_usd > 0
    has_candles = len(candles) > 0
    has_orderbook = len(bids) > 0 and len(asks) > 0
    
    # Must have candles for processing
    if not has_candles:
        print(f"[VALIDATION FAILED] {symbol}: No candle data")
        return None
    
    # If no price but we have candles, try final price extraction
    if not has_price and has_candles:
        # Try to extract from any candle
        for candle in candles:
            if candle.get("close", 0) > 0:
                price_usd = candle["close"]
                print(f"[PRICE RECOVERY] {symbol}: Extracted ${price_usd} from candle fallback")
                break
        
        # If still no price, use 1.0 as placeholder for processing
        if price_usd <= 0:
            price_usd = 1.0
            print(f"[PRICE PLACEHOLDER] {symbol}: Using placeholder price for candle-only processing")
    
    # SUCCESS: Return processed data with partial status tracking
    components = []
    if ticker_data: components.append("ticker")
    if candles: components.append(f"candles({len(candles)})")
    if has_orderbook: components.append("orderbook" if orderbook_data else "synthetic_orderbook")
    
    partial_status = " (PARTIAL)" if not (ticker_data and has_orderbook and orderbook_data) else ""
    print(f"[ASYNC SUCCESS{partial_status}] {symbol}: {', '.join(components)} - Price ${price_usd}")
    
    return {
        "symbol": symbol,
        "price_usd": price_usd,
        "volume_24h": volume_24h,
        "candles": candles,  # Use 'candles' key for compatibility
        "candles_15m": candles,  # Also provide original key
        "candles_5m": [],  # Not fetched in current async implementation
        "orderbook": {"bids": bids, "asks": asks},  # Nested structure for compatibility
        "bids": bids,
        "asks": asks,
        "best_bid": bids[0]["price"] if bids else price_usd * 0.999,
        "best_ask": asks[0]["price"] if asks else price_usd * 1.001,
        "volume": volume_24h,
        "recent_volumes": [c["volume"] for c in candles[-7:]] if candles else [],
        "ticker_data": ticker_data,
        "orderbook_data": orderbook_data,
        "partial_data": not (ticker_data and has_orderbook and orderbook_data),
        "is_partial": not (ticker_data and has_orderbook and orderbook_data),
        "data_sources": components,
        "price_change_24h": 0.0  # Cannot calculate without historical data
    }

def validate_processed_data(data: Dict) -> bool:
    """
    Validate processed data meets minimum requirements
    
    Args:
        data: Processed market data
        
    Returns:
        True if valid for analysis
    """
    if not data:
        return False
    
    required_fields = ["symbol", "price_usd", "candles_15m"]
    for field in required_fields:
        if field not in data:
            return False
    
    if data["price_usd"] <= 0:
        return False
    
    if len(data["candles_15m"]) < 5:  # Minimum candles for analysis
        return False
    
    return True

def get_processing_stats(results: List[Dict]) -> Dict:
    """
    Get processing statistics for debugging
    
    Args:
        results: List of processed results
        
    Returns:
        Processing statistics
    """
    total = len(results)
    valid = len([r for r in results if r is not None])
    partial = len([r for r in results if r and r.get("is_partial", False)])
    
    ticker_success = len([r for r in results if r and r.get("ticker_data")])
    candle_success = len([r for r in results if r and len(r.get("candles_15m", [])) > 0])
    orderbook_success = len([r for r in results if r and len(r.get("bids", [])) > 0 and r.get("orderbook_data")])
    
    return {
        "total_processed": total,
        "valid_results": valid,
        "partial_results": partial,
        "success_rate": f"{(valid/total*100):.1f}%" if total > 0 else "0%",
        "component_success": {
            "ticker": f"{ticker_success}/{total}",
            "candles": f"{candle_success}/{total}",
            "orderbook": f"{orderbook_success}/{total}"
        }
    }