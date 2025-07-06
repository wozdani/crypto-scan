"""
Async Data Processor - Direct Integration Fix
Handles ticker None scenarios while preserving authentic candle data
"""

from typing import Dict, List, Optional, Any
import json
import os

def process_async_data_enhanced_with_5m(symbol: str, ticker_data: Optional[Dict], candles_data: Optional[Dict], candles_5m_data: Optional[Dict], orderbook_data: Optional[Dict]) -> Optional[Dict]:
    """
    Enhanced async data processing with 5M candle support
    
    Args:
        symbol: Trading symbol
        ticker_data: Ticker data (may be None)
        candles_data: 15M candle data (should be available)
        candles_5m_data: 5M candle data (optional)
        orderbook_data: Orderbook data (may be None)
        
    Returns:
        Processed market data with both 15M and 5M candles or None if insufficient
    """
    
    # Enhanced debug logging for input validation
    print(f"[PROCESSOR DEBUG] {symbol} → Processing inputs:")
    print(f"  ticker_data: {type(ticker_data)} - {bool(ticker_data)}")
    print(f"  candles_data: {type(candles_data)} - {bool(candles_data)}")
    print(f"  candles_5m_data: {type(candles_5m_data)} - {bool(candles_5m_data)}")
    print(f"  orderbook_data: {type(orderbook_data)} - {bool(orderbook_data)}")
    
    if candles_data:
        candles_structure = candles_data.get("result", {}) if isinstance(candles_data, dict) else "Invalid"
        candles_list = candles_structure.get("list", []) if isinstance(candles_structure, dict) else "Invalid"
        print(f"  candles_15m: {type(candles_structure)} - len={len(candles_list) if hasattr(candles_list, '__len__') else 'N/A'}")
    
    if candles_5m_data:
        candles_5m_structure = candles_5m_data.get("result", {}) if isinstance(candles_5m_data, dict) else "Invalid"
        candles_5m_list = candles_5m_structure.get("list", []) if isinstance(candles_5m_structure, dict) else "Invalid"
        print(f"  candles_5m: {type(candles_5m_structure)} - len={len(candles_5m_list) if hasattr(candles_5m_list, '__len__') else 'N/A'}")
    
    # Initialize data containers
    price_usd = 0.0
    volume_24h = 0.0
    candles_15m = []
    candles_5m = []
    bids = []
    asks = []
    
    # PRIORITY 1: Extract from ticker data if available
    if ticker_data and ticker_data.get("result", {}).get("list"):
        ticker_list = ticker_data["result"]["list"]
        if ticker_list:
            ticker = ticker_list[0]
            price_usd = float(ticker.get("lastPrice", 0)) if ticker.get("lastPrice") else 0.0
            volume_24h = float(ticker.get("volume24h", 0)) if ticker.get("volume24h") else 0.0
            
            # FIX: Validate ticker data quality - mark as partial if price or volume is 0
            if price_usd <= 0.0 or volume_24h <= 0.0:
                print(f"[TICKER INVALID] {symbol}: Price ${price_usd}, Volume {volume_24h} - rejecting invalid ticker")
                has_price = False
                has_ticker = False
                ticker_invalid = True  # Flag for partial marking
            else:
                print(f"[TICKER SUCCESS] {symbol}: Price ${price_usd}, Volume {volume_24h}")
                has_ticker = True
                has_price = True
                ticker_invalid = False
    
    # PRIORITY 2: Process 15M candles (should always be available from async scanner)
    if candles_data and candles_data.get("result", {}).get("list"):
        candle_list = candles_data["result"]["list"]
        for candle in candle_list:
            try:
                candles_15m.append({
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
        if price_usd <= 0 and candles_15m:
            price_usd = candles_15m[-1]["close"]  # Use most recent candle close price
            # Calculate 24h volume from available candles
            if len(candles_15m) >= 96:  # Full day of 15m candles
                volume_24h = sum(c["volume"] for c in candles_15m[-96:])
            else:
                # Estimate based on available candles
                volume_24h = sum(c["volume"] for c in candles_15m) * (96 / len(candles_15m))
            print(f"[CANDLE FALLBACK] {symbol}: Price ${price_usd} from {len(candles_15m)} 15M candles")
    
    # PRIORITY 3: Process 5M candles if available
    if candles_5m_data and candles_5m_data.get("result", {}).get("list"):
        candle_5m_list = candles_5m_data["result"]["list"]
        for candle in candle_5m_list:
            try:
                candles_5m.append({
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]), 
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                })
            except (ValueError, IndexError, TypeError):
                continue
        print(f"[5M CANDLES SUCCESS] {symbol}: Processed {len(candles_5m)} 5M candles")
    else:
        print(f"[5M FALLBACK] {symbol} → No 5M candles, using 15M-only mode")
    
    # PRIORITY 4: Process orderbook if available
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
    has_15m_candles = len(candles_15m) > 0
    has_5m_candles = len(candles_5m) > 0
    has_orderbook = len(bids) > 0 and len(asks) > 0
    
    # Must have 15M candles for processing
    if not has_15m_candles:
        print(f"[VALIDATION FAILED] {symbol}: No 15M candle data")
        print(f"[DATA VALIDATION FAILED] {symbol} → Enhanced processor rejected data")
        return None
    
    # If no price but we have candles, try final price extraction
    if not has_price and has_15m_candles:
        # Try to extract from any candle
        for candle in candles_15m:
            if candle.get("close", 0) > 0:
                price_usd = candle["close"]
                print(f"[PRICE RECOVERY] {symbol}: Extracted ${price_usd} from candle fallback")
                break
        
    # Final price validation 
    if price_usd <= 0:
        print(f"[VALIDATION FAILED] {symbol}: No valid price found")
        return None
    
    # Determine data completeness - CRITICAL for TOP5 filtering
    # Token is considered partial if it lacks 5M candles OR has invalid ticker
    ticker_invalid = locals().get('ticker_invalid', False)
    is_partial_data = not has_5m_candles or ticker_invalid  # Missing 5M or invalid ticker indicates partial data
    
    if ticker_invalid and has_5m_candles:
        data_quality = "PARTIAL_TICKER_INVALID"  
    elif not has_5m_candles and not ticker_invalid:
        data_quality = "PARTIAL_15M_ONLY"
    elif not has_5m_candles and ticker_invalid:
        data_quality = "PARTIAL_TICKER_AND_5M"
    else:
        data_quality = "COMPLETE"
    
    if is_partial_data:
        print(f"[DATA QUALITY] {symbol}: ⚠️ PARTIAL - 15M only (missing 5M candles)")
    else:
        print(f"[DATA QUALITY] {symbol}: ✅ COMPLETE - 15M + 5M candles available")
    
    # FIX 1: Load AI-EYE label from Vision-AI labeling system
    ai_label_data = load_ai_label_for_symbol(symbol)
    
    # FIX 2: Generate HTF candles from 15M candles if not available separately
    htf_candles = generate_htf_candles_from_15m(candles_15m)
    print(f"[AI LABEL] {symbol}: {ai_label_data.get('label', 'No existing AI label found') if ai_label_data else 'No existing AI label found'}")
    print(f"[HTF GEN] Generated {len(htf_candles)} HTF candles from {len(candles_15m)} 15M candles")
    
    # Return enhanced market data with both timeframes
    market_data = {
        "symbol": symbol,
        "price_usd": price_usd,
        "volume_24h": volume_24h,
        "candles": candles_15m,  # Legacy compatibility
        "candles_15m": candles_15m,
        "candles_5m": candles_5m,
        "htf_candles": htf_candles,  # FIX 2: HTF candles for HTF Overlay module
        "ai_label": ai_label_data.get("label", "unknown") if ai_label_data else "unknown",  # FIX 1: AI label
        "ai_confidence": ai_label_data.get("confidence", 0.0) if ai_label_data else 0.0,
        "orderbook": {"bids": bids, "asks": asks},
        "bids": bids,
        "asks": asks,
        "best_bid": bids[0]["price"] if bids else price_usd * 0.999,
        "best_ask": asks[0]["price"] if asks else price_usd * 1.001,
        "volume": volume_24h,
        "recent_volumes": [c["volume"] for c in candles_15m[-10:]] if len(candles_15m) >= 10 else [],
        "ticker_data": ticker_data,
        "orderbook_data": orderbook_data,
        "partial_data": is_partial_data,  # CRITICAL: Mark tokens without 5M candles
        "is_partial": is_partial_data,    # Alias for compatibility
        "data_quality": data_quality,     # Human-readable status
        "data_sources": ["async_enhanced"],
        "price_change_24h": ((candles_15m[-1]["close"] - candles_15m[-96]["close"]) / candles_15m[-96]["close"] * 100) if len(candles_15m) >= 96 else 0.0
    }
    
    print(f"[ENHANCED SUCCESS] {symbol}: 15M={len(candles_15m)}, 5M={len(candles_5m)}, Price=${price_usd:.6f}")
    return market_data


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
    
    # Enhanced debug logging for input validation
    print(f"[PROCESSOR DEBUG] {symbol} → Processing inputs:")
    print(f"  ticker_data: {type(ticker_data)} - {bool(ticker_data)}")
    print(f"  candles_data: {type(candles_data)} - {bool(candles_data)}")
    print(f"  orderbook_data: {type(orderbook_data)} - {bool(orderbook_data)}")
    
    if candles_data:
        candles_structure = candles_data.get("result", {}) if isinstance(candles_data, dict) else "Invalid"
        candles_list = candles_structure.get("list", []) if isinstance(candles_structure, dict) else "Invalid"
        print(f"  candles_structure: {type(candles_structure)} - {bool(candles_structure)}")
        print(f"  candles_list: {type(candles_list)} len={len(candles_list) if hasattr(candles_list, '__len__') else 'N/A'}")
    
    # Initialize data containers
    price_usd = 0.0
    volume_24h = 0.0
    candles = []
    bids = []
    asks = []
    
    # PRIORITY 1: Extract from ticker data if available
    ticker_success = False
    if ticker_data and ticker_data.get("result", {}).get("list"):
        ticker_list = ticker_data["result"]["list"]
        if ticker_list:
            ticker = ticker_list[0]
            ticker_price = float(ticker.get("lastPrice", 0)) if ticker.get("lastPrice") else 0.0
            ticker_volume = float(ticker.get("volume24h", 0)) if ticker.get("volume24h") else 0.0
            
            # Validate ticker data quality
            if ticker_price > 0.0 and ticker_volume > 0.0:
                price_usd = ticker_price
                volume_24h = ticker_volume
                ticker_success = True
                print(f"[TICKER SUCCESS] {symbol}: Price ${price_usd}, Volume {volume_24h}")
            else:
                print(f"[TICKER PARTIAL] {symbol}: Price ${ticker_price}, Volume {ticker_volume} - will try candle fallback")
    
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
        if not ticker_success and candles:
            price_usd = candles[-1]["close"]  # Use most recent candle close price
            # Calculate 24h volume from available candles
            if len(candles) >= 96:  # Full day of 15m candles
                volume_24h = sum(c["volume"] for c in candles[-96:])
            else:
                # Estimate based on available candles
                volume_24h = sum(c["volume"] for c in candles) * (96 / len(candles))
            print(f"[CANDLE FALLBACK SUCCESS] {symbol}: Price ${price_usd} from {len(candles)} candles")
            ticker_success = True  # Mark as success since fallback worked
    
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
    
    # Determine if this is a partial success or full success
    partial_status = " (PARTIAL)" if not ticker_success or not has_orderbook else ""
    status_message = f"[ASYNC SUCCESS{partial_status}] {symbol}: {', '.join(components)} - Price ${price_usd}"
    
    if not ticker_success and price_usd > 0:
        status_message += " (candle fallback)"
    
    print(status_message)
    
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
        "partial_data": not (ticker_success and has_orderbook and orderbook_data),
        "is_partial": not (ticker_success and has_orderbook and orderbook_data),
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

def load_ai_label_for_symbol(symbol: str) -> Optional[Dict]:
    """
    FIX 1: Load AI-EYE label from Vision-AI labeling system
    Checks for existing AI labels in training data metadata
    """
    try:
        # Check training_data/charts for recent labeled charts
        charts_dir = "training_data/charts"
        if os.path.exists(charts_dir):
            # Look for recent metadata files for this symbol
            for filename in os.listdir(charts_dir):
                if filename.startswith(f"{symbol}_") and filename.endswith("_metadata.json"):
                    metadata_path = os.path.join(charts_dir, filename)
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        setup_label = metadata.get("setup_label")
                        if setup_label and setup_label != "unknown":
                            return {
                                "label": setup_label,
                                "confidence": metadata.get("ai_confidence", 0.85),
                                "source": "vision_ai_metadata"
                            }
        
        # Check for cached AI labels
        ai_cache_path = "data/ai_labels_cache.json"
        if os.path.exists(ai_cache_path):
            with open(ai_cache_path, 'r') as f:
                cache = json.load(f)
                if symbol in cache:
                    return cache[symbol]
        
        print(f"[AI LABEL] {symbol}: No existing AI label found")
        return None
        
    except Exception as e:
        print(f"[AI LABEL ERROR] {symbol}: {e}")
        return None

def generate_htf_candles_from_15m(candles_15m: List[Dict]) -> List[Dict]:
    """
    FIX 2: Generate HTF (1H) candles from 15M candles if not available separately
    Creates higher timeframe data for HTF Overlay module
    """
    try:
        if not candles_15m or len(candles_15m) < 4:
            print(f"[HTF GEN] Insufficient 15M candles for HTF generation: {len(candles_15m) if candles_15m else 0}")
            return []
        
        htf_candles = []
        # Group 15M candles into 1H periods (4 candles per hour)
        for i in range(0, len(candles_15m) - 3, 4):
            candle_group = candles_15m[i:i+4]
            
            # Create 1H candle from 4x15M candles
            htf_candle = {
                "timestamp": candle_group[0]["timestamp"],
                "open": candle_group[0]["open"],
                "high": max(c["high"] for c in candle_group),
                "low": min(c["low"] for c in candle_group),
                "close": candle_group[-1]["close"],
                "volume": sum(c["volume"] for c in candle_group)
            }
            htf_candles.append(htf_candle)
        
        print(f"[HTF GEN] Generated {len(htf_candles)} HTF candles from {len(candles_15m)} 15M candles")
        return htf_candles
        
    except Exception as e:
        print(f"[HTF GEN ERROR] Failed to generate HTF candles: {e}")
        return []