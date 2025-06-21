"""
Trend Mode Mock Data Generator for Testing (Temporary)
Generates realistic market data when API is unavailable
"""

import random
from datetime import datetime, timedelta

def generate_mock_15m_prices(symbol: str, count: int = 96) -> list[float]:
    """Generate realistic 15M price data for testing"""
    
    # Base prices for different symbols
    base_prices = {
        "BTCUSDT": 65000.0,
        "ETHUSDT": 3500.0,
        "SOLUSDT": 150.0,
        "ADAUSDT": 0.45,
        "DOTUSDT": 6.5,
        "AVAXUSDT": 35.0,
        "MATICUSDT": 0.55,
        "LINKUSDT": 15.0,
        "UNIUSDT": 8.5,
        "LTCUSDT": 85.0
    }
    
    base_price = base_prices.get(symbol, 100.0)
    prices = []
    current_price = base_price
    
    # Generate trend with some randomness
    trend_direction = random.choice([1, -1, 0])  # up, down, sideways
    
    for i in range(count):
        # Add some volatility
        volatility = random.uniform(-0.015, 0.015)  # ±1.5%
        
        # Add trend component
        if trend_direction == 1:  # Uptrend
            trend_component = random.uniform(0.001, 0.008)  # 0.1-0.8% up
        elif trend_direction == -1:  # Downtrend
            trend_component = random.uniform(-0.008, -0.001)  # 0.1-0.8% down
        else:  # Sideways
            trend_component = random.uniform(-0.003, 0.003)  # ±0.3%
        
        price_change = volatility + trend_component
        current_price *= (1 + price_change)
        prices.append(current_price)
    
    return prices

def generate_mock_5m_prices(symbol: str, count: int = 12) -> list[float]:
    """Generate realistic 5M price data for testing"""
    base_prices = {
        "BTCUSDT": 65000.0,
        "ETHUSDT": 3500.0,
        "SOLUSDT": 150.0,
        "ADAUSDT": 0.45,
        "DOTUSDT": 6.5
    }
    
    base_price = base_prices.get(symbol, 100.0)
    prices = []
    current_price = base_price
    
    # Generate correction pattern (few red candles then recovery)
    for i in range(count):
        if i < 4:  # First 4 candles - small correction
            change = random.uniform(-0.008, -0.002)  # -0.2% to -0.8%
        else:  # Recovery
            change = random.uniform(0.001, 0.006)   # +0.1% to +0.6%
            
        current_price *= (1 + change)
        prices.append(current_price)
    
    return prices

def generate_mock_current_price(symbol: str) -> float:
    """Generate current price for symbol"""
    base_prices = {
        "BTCUSDT": 65000.0,
        "ETHUSDT": 3500.0,
        "SOLUSDT": 150.0,
        "ADAUSDT": 0.45,
        "DOTUSDT": 6.5,
        "AVAXUSDT": 35.0,
        "MATICUSDT": 0.55,
        "LINKUSDT": 15.0,
        "UNIUSDT": 8.5,
        "LTCUSDT": 85.0
    }
    
    base_price = base_prices.get(symbol, 100.0)
    # Add small random variation
    variation = random.uniform(-0.02, 0.02)  # ±2%
    return base_price * (1 + variation)

def generate_mock_orderbook_volumes(symbol: str) -> tuple[list[float], list[float]]:
    """Generate orderbook volume data"""
    
    # Generate volumes that might trigger entry signal
    scenario = random.choice(["entry_signal", "no_signal", "balanced"])
    
    if scenario == "entry_signal":
        # Declining ask volume, rising bid volume
        ask_volumes = [1200.0, 1000.0, 800.0]
        bid_volumes = [600.0, 800.0, 1000.0]
    elif scenario == "no_signal":
        # Random pattern
        ask_volumes = [random.uniform(800, 1500) for _ in range(3)]
        bid_volumes = [random.uniform(500, 1200) for _ in range(3)]
    else:  # balanced
        ask_volumes = [1000.0, 1050.0, 980.0]
        bid_volumes = [950.0, 1020.0, 990.0]
    
    return ask_volumes, bid_volumes

def should_use_mock_data() -> bool:
    """Check if we should use mock data (when API is unavailable)"""
    # For now, always use mock data when API fails
    return True

# Test the mock data generation
if __name__ == "__main__":
    print("Testing mock data generation...")
    
    symbol = "BTCUSDT"
    prices_15m = generate_mock_15m_prices(symbol, 24)
    prices_5m = generate_mock_5m_prices(symbol, 12)
    current_price = generate_mock_current_price(symbol)
    ask_vols, bid_vols = generate_mock_orderbook_volumes(symbol)
    
    print(f"15M prices (24): {prices_15m[:5]}...{prices_15m[-5:]}")
    print(f"5M prices (12): {prices_5m}")
    print(f"Current price: {current_price}")
    print(f"Ask volumes: {ask_vols}")
    print(f"Bid volumes: {bid_vols}")