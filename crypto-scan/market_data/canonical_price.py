import math
from typing import Optional, Dict, Any, Tuple

def _safe(x): 
    try: 
        return float(x)
    except: 
        return None

def compute_canonical_price(ticker: Dict[str, Any], orderbook: Dict[str, Any], candles_15m, candles_5m) -> Tuple[Optional[float], str]:
    """
    Compute canonical price with strict priority:
    1. Last trade from ticker (if >0 and finite)
    2. Mid price from orderbook
    3. Close from latest 15m candle
    4. Close from latest 5m candle
    
    Returns: (price, source) or (None, "none")
    """
    # 1) last trade (jeśli >0 i finite)
    last = _safe(ticker.get("last")) or _safe(ticker.get("price"))
    if last and math.isfinite(last) and last > 0:
        return last, "ticker_last"

    # 2) mid z orderbooku
    bids = orderbook.get("bids") or [] if orderbook else []
    asks = orderbook.get("asks") or [] if orderbook else []
    if bids and asks:
        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            if best_bid > 0 and best_ask > 0:
                return (best_bid + best_ask) / 2.0, "orderbook_mid"
        except Exception:
            pass

    # 3) close z 15m
    if candles_15m:
        try:
            c15 = float(candles_15m[-1]["close"])
            if c15 > 0: 
                return c15, "candle_15m"
        except (Exception, IndexError, KeyError):
            pass

    # 4) close z 5m
    if candles_5m:
        try:
            c5 = float(candles_5m[-1]["close"])
            if c5 > 0: 
                return c5, "candle_5m"
        except (Exception, IndexError, KeyError):
            pass

    return None, "none"

def apply_canonical_price_to_market_data(market_data: Dict[str, Any], ticker: Dict[str, Any], orderbook: Dict[str, Any], candles_15m, candles_5m, symbol: str) -> bool:
    """
    Apply canonical price as single source of truth to market_data.
    Returns True if valid price found, False if should skip token.
    """
    canon_price, canon_src = compute_canonical_price(ticker, orderbook, candles_15m, candles_5m)
    
    if not canon_price:
        print(f"[{symbol}] No canonical price available - hard skip")
        return False
        
    # Set canonical price as single source of truth
    market_data["price"] = canon_price
    market_data["price_source"] = canon_src
    
    print(f"[CANONICAL PRICE] {symbol}: price=${canon_price:.6f} source={canon_src}")
    
    # Check for ticker desync and log appropriately
    ticker_price = _safe(ticker.get("price")) or _safe(ticker.get("last")) if ticker else None
    ticker_invalid = not ticker_price or ticker_price <= 0
    
    if ticker_invalid and canon_src in ("orderbook_mid", "candle_15m", "candle_5m"):
        print(f"[{symbol}] TICKER INVALID, using canonical={canon_src} → OK")
    
    return True