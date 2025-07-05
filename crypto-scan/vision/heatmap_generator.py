"""
Heatmap Generator - Orderbook Visualization
Generates visual heatmaps from orderbook data for enhanced pattern recognition
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Optional
import os

logger = logging.getLogger(__name__)

def generate_orderbook_heatmap(symbol: str, bids: List[List], asks: List[List], 
                              save_path: str) -> bool:
    """
    Generate orderbook heatmap visualization
    
    Args:
        symbol: Trading symbol
        bids: List of [price, size] bid orders
        asks: List of [price, size] ask orders  
        save_path: Path to save heatmap image
        
    Returns:
        True if heatmap generated successfully
    """
    try:
        # Validate input data
        if not bids or not asks:
            logger.warning(f"[HEATMAP] ❌ {symbol}: Empty orderbook data")
            return False
            
        # Convert to float and combine orderbook data
        all_orders = []
        
        # Process bids (negative for visual distinction)
        for bid in bids[:20]:  # Top 20 levels
            try:
                price = float(bid[0])
                size = -float(bid[1])  # Negative for bids
                all_orders.append((price, size))
            except (ValueError, IndexError):
                continue
                
        # Process asks (positive)
        for ask in asks[:20]:  # Top 20 levels
            try:
                price = float(ask[0])
                size = float(ask[1])  # Positive for asks
                all_orders.append((price, size))
            except (ValueError, IndexError):
                continue
                
        if len(all_orders) < 5:
            logger.warning(f"[HEATMAP] ❌ {symbol}: Insufficient orderbook data")
            return False
            
        # Sort by price
        all_orders.sort(key=lambda x: x[0])
        
        # Extract prices and sizes
        prices = [order[0] for order in all_orders]
        sizes = [order[1] for order in all_orders]
        
        # Create price levels grid
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:
            logger.warning(f"[HEATMAP] ❌ {symbol}: No price spread in orderbook")
            return False
            
        # Create liquidity levels
        liquidity_levels = {}
        
        # Map orders to price levels
        for price, size in all_orders:
            # Round to nearest level
            level = round((price - min_price) / price_range * 100)
            level = max(0, min(100, level))  # Clamp to 0-100
            
            if level not in liquidity_levels:
                liquidity_levels[level] = 0
            liquidity_levels[level] += size
            
        # Create heatmap array
        heatmap_data = np.zeros((1, 101))  # Single row, 101 columns (0-100)
        
        for level, size in liquidity_levels.items():
            heatmap_data[0, level] = size
            
        # Normalize for better visualization
        max_abs_size = max(abs(x) for x in heatmap_data[0] if x != 0) if any(heatmap_data[0]) else 1
        if max_abs_size > 0:
            heatmap_data = heatmap_data / max_abs_size
            
        # Generate heatmap
        plt.figure(figsize=(12, 2))
        
        # Use custom colormap: blue for bids, red for asks
        plt.imshow(heatmap_data, cmap="RdBu_r", aspect="auto", 
                  extent=[min_price, max_price, 0, 1])
        
        # Styling
        plt.title(f"{symbol} Orderbook Heatmap", fontsize=12, fontweight='bold')
        plt.xlabel("Price Level", fontsize=10)
        plt.ylabel("Liquidity", fontsize=10)
        plt.colorbar(label="Order Size (Normalized)", shrink=0.8)
        
        # Remove y-axis ticks
        plt.yticks([])
        
        # Format x-axis
        plt.ticklabel_format(style='plain', axis='x')
        
        # Tight layout
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save heatmap
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"[HEATMAP] ✅ {symbol}: Generated heatmap → {os.path.basename(save_path)}")
        return True
        
    except Exception as e:
        logger.error(f"[HEATMAP] ❌ {symbol}: Generation failed - {e}")
        return False

def generate_simple_heatmap(symbol: str, orderbook: dict, save_dir: str = "vision_data/heatmaps") -> Optional[str]:
    """
    Generate simple heatmap with standardized path format
    
    Args:
        symbol: Trading symbol
        orderbook: Orderbook dictionary with 'bids' and 'asks'
        save_dir: Directory to save heatmap
        
    Returns:
        Path to generated heatmap or None if failed
    """
    try:
        # Extract bids and asks
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            logger.warning(f"[HEATMAP] ❌ {symbol}: No orderbook data")
            return None
            
        # Create save path
        filename = f"{symbol}_heatmap.png"
        save_path = os.path.join(save_dir, filename)
        
        # Generate heatmap
        success = generate_orderbook_heatmap(symbol, bids, asks, save_path)
        
        return save_path if success else None
        
    except Exception as e:
        logger.error(f"[HEATMAP] ❌ {symbol}: Simple heatmap failed - {e}")
        return None

def create_liquidity_profile(bids: List[List], asks: List[List]) -> dict:
    """
    Create liquidity profile analysis from orderbook
    
    Args:
        bids: Bid orders as [price, size] pairs
        asks: Ask orders as [price, size] pairs
        
    Returns:
        Dictionary with liquidity metrics
    """
    try:
        # Calculate bid/ask sizes
        total_bid_size = sum(float(bid[1]) for bid in bids[:10] if len(bid) >= 2)
        total_ask_size = sum(float(ask[1]) for ask in asks[:10] if len(ask) >= 2)
        
        # Calculate spread
        if bids and asks:
            best_bid = float(bids[0][0]) if bids[0] else 0
            best_ask = float(asks[0][0]) if asks[0] else 0
            spread = best_ask - best_bid if best_ask > best_bid else 0
            spread_pct = (spread / best_ask * 100) if best_ask > 0 else 0
        else:
            spread = 0
            spread_pct = 0
            
        # Liquidity imbalance
        total_liquidity = total_bid_size + total_ask_size
        bid_dominance = (total_bid_size / total_liquidity) if total_liquidity > 0 else 0.5
        
        return {
            "total_bid_size": total_bid_size,
            "total_ask_size": total_ask_size,
            "spread": spread,
            "spread_pct": spread_pct,
            "bid_dominance": bid_dominance,
            "liquidity_imbalance": abs(bid_dominance - 0.5) * 2  # 0-1 scale
        }
        
    except Exception as e:
        logger.error(f"[LIQUIDITY] ❌ Profile creation failed: {e}")
        return {
            "total_bid_size": 0,
            "total_ask_size": 0,
            "spread": 0,
            "spread_pct": 0,
            "bid_dominance": 0.5,
            "liquidity_imbalance": 0
        }

def test_heatmap_generation():
    """Test heatmap generation with sample data"""
    
    # Sample orderbook data
    sample_bids = [
        ["50000.5", "1.5"],
        ["50000.0", "2.0"],
        ["49999.5", "1.2"],
        ["49999.0", "0.8"],
        ["49998.5", "3.0"]
    ]
    
    sample_asks = [
        ["50001.0", "1.0"],
        ["50001.5", "1.8"],
        ["50002.0", "0.9"],
        ["50002.5", "2.2"],
        ["50003.0", "1.1"]
    ]
    
    # Test generation
    test_path = "test_heatmap.png"
    success = generate_orderbook_heatmap("TESTUSDT", sample_bids, sample_asks, test_path)
    
    print(f"Heatmap generation test: {'✅ Success' if success else '❌ Failed'}")
    
    # Test liquidity profile
    profile = create_liquidity_profile(sample_bids, sample_asks)
    print(f"Liquidity profile: {profile}")

if __name__ == "__main__":
    test_heatmap_generation()