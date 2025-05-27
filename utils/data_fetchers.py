import os
import json
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# API Configuration
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
BSCSCAN_API_KEY = os.getenv("BSCSCAN_API_KEY")
POLYGONSCAN_API_KEY = os.getenv("POLYGONSCAN_API_KEY")
ARBISCAN_API_KEY = os.getenv("ARBISCAN_API_KEY")
OPTIMISMSCAN_API_KEY = os.getenv("OPTIMISMSCAN_API_KEY")
TRONGRID_API_KEY = os.getenv("TRONGRID_API_KEY")

# Cache configuration
SYMBOLS_CACHE_FILE = "data/cache/symbols.json"
CACHE_DURATION = 3600  # 1 hour in seconds

def get_symbols_cached():
    """Get cryptocurrency symbols with caching"""
    try:
        # Check if cache exists and is fresh
        if os.path.exists(SYMBOLS_CACHE_FILE):
            with open(SYMBOLS_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
                
                if (datetime.utcnow() - cache_time).seconds < CACHE_DURATION:
                    print(f"📦 Using cached symbols ({len(cache_data['symbols'])} symbols)")
                    return cache_data['symbols']
        
        # Fetch fresh symbols
        print("🔄 Fetching fresh symbol list...")
        symbols = fetch_top_symbols()
        
        # Cache the results
        cache_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbols': symbols
        }
        
        os.makedirs(os.path.dirname(SYMBOLS_CACHE_FILE), exist_ok=True)
        with open(SYMBOLS_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
            
        print(f"✅ Cached {len(symbols)} symbols")
        return symbols
        
    except Exception as e:
        print(f"❌ Error getting symbols: {e}")
        # Return fallback list
        return get_fallback_symbols()

def fetch_top_symbols():
    """Fetch top cryptocurrency symbols from CryptoCompare"""
    try:
        url = "https://min-api.cryptocompare.com/data/top/mktcapfull"
        params = {
            'limit': 200,
            'tsym': 'USD',
            'api_key': CRYPTOCOMPARE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'Success':
                symbols = []
                for coin in data.get('Data', []):
                    symbol = coin.get('CoinInfo', {}).get('Name')
                    if symbol and len(symbol) <= 10:  # Filter out long symbols
                        symbols.append(symbol)
                return symbols[:150]  # Return top 150
            else:
                print(f"❌ CryptoCompare API error: {data.get('Message')}")
                
    except Exception as e:
        print(f"❌ Error fetching symbols from CryptoCompare: {e}")
        
    return get_fallback_symbols()

def get_fallback_symbols():
    """Return fallback symbol list"""
    return [
        'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'TRX', 'DOT', 'MATIC',
        'LTC', 'SHIB', 'AVAX', 'UNI', 'LINK', 'XLM', 'BCH', 'ATOM', 'VET', 'FIL',
        'ETC', 'XMR', 'ALGO', 'MANA', 'SAND', 'CRO', 'FTM', 'NEAR', 'GRT', 'LRC'
    ]

def get_market_data(symbol):
    """Fetch comprehensive market data for a symbol"""
    try:
        # Get price and volume data
        price_data = get_price_data(symbol)
        if not price_data:
            return None
            
        # Get blockchain-specific data
        blockchain_data = get_blockchain_data(symbol)
        
        # Combine all data
        market_data = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'price': price_data,
            'blockchain': blockchain_data,
            'volume_24h': price_data.get('volume_24h', 0),
            'market_cap': price_data.get('market_cap', 0),
            'price_change_24h': price_data.get('price_change_24h', 0)
        }
        
        return market_data
        
    except Exception as e:
        print(f"❌ Error getting market data for {symbol}: {e}")
        return None

def get_price_data(symbol):
    """Get price and volume data from CryptoCompare"""
    try:
        url = "https://min-api.cryptocompare.com/data/pricemultifull"
        params = {
            'fsyms': symbol,
            'tsyms': 'USD',
            'api_key': CRYPTOCOMPARE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'Success' and symbol in data.get('RAW', {}):
                raw_data = data['RAW'][symbol]['USD']
                return {
                    'price': raw_data.get('PRICE', 0),
                    'volume_24h': raw_data.get('VOLUME24HOUR', 0),
                    'market_cap': raw_data.get('MKTCAP', 0),
                    'price_change_24h': raw_data.get('CHANGEPCT24HOUR', 0),
                    'high_24h': raw_data.get('HIGH24HOUR', 0),
                    'low_24h': raw_data.get('LOW24HOUR', 0),
                    'supply': raw_data.get('SUPPLY', 0)
                }
                
    except Exception as e:
        print(f"❌ Error getting price data for {symbol}: {e}")
        
    return None

def get_blockchain_data(symbol):
    """Get blockchain-specific data for supported networks"""
    blockchain_data = {}
    
    # Ethereum-based tokens
    if symbol in ['ETH'] or True:  # For now, try to get data for all symbols
        eth_data = get_ethereum_data(symbol)
        if eth_data:
            blockchain_data['ethereum'] = eth_data
    
    # BSC data
    bsc_data = get_bsc_data(symbol)
    if bsc_data:
        blockchain_data['bsc'] = bsc_data
        
    # Polygon data
    polygon_data = get_polygon_data(symbol)
    if polygon_data:
        blockchain_data['polygon'] = polygon_data
        
    return blockchain_data

def get_ethereum_data(symbol):
    """Get Ethereum blockchain data"""
    try:
        # This is a simplified implementation
        # In production, you'd need contract addresses and more sophisticated queries
        url = f"https://api.etherscan.io/api"
        params = {
            'module': 'stats',
            'action': 'ethprice',
            'apikey': ETHERSCAN_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == '1':
                return {
                    'network': 'ethereum',
                    'gas_price': data.get('result', {}).get('ethbtc', 0),
                    'status': 'active'
                }
                
    except Exception as e:
        print(f"⚠️ Error getting Ethereum data: {e}")
        
    return None

def get_bsc_data(symbol):
    """Get Binance Smart Chain data"""
    try:
        url = f"https://api.bscscan.com/api"
        params = {
            'module': 'stats',
            'action': 'bnbprice',
            'apikey': BSCSCAN_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == '1':
                return {
                    'network': 'bsc',
                    'bnb_price': data.get('result', {}).get('ethusd', 0),
                    'status': 'active'
                }
                
    except Exception as e:
        print(f"⚠️ Error getting BSC data: {e}")
        
    return None

def get_polygon_data(symbol):
    """Get Polygon network data"""
    try:
        url = f"https://api.polygonscan.com/api"
        params = {
            'module': 'stats',
            'action': 'maticprice',
            'apikey': POLYGONSCAN_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == '1':
                return {
                    'network': 'polygon',
                    'matic_price': data.get('result', {}).get('maticusd', 0),
                    'status': 'active'
                }
                
    except Exception as e:
        print(f"⚠️ Error getting Polygon data: {e}")
        
    return None

def get_historical_data(symbol, days=30):
    """Get historical price data"""
    try:
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            'fsym': symbol,
            'tsym': 'USD',
            'limit': days,
            'api_key': CRYPTOCOMPARE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'Success':
                return data.get('Data', {}).get('Data', [])
                
    except Exception as e:
        print(f"❌ Error getting historical data for {symbol}: {e}")
        
    return []
