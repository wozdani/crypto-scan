#!/usr/bin/env python3
"""
Data Fetchers Module
Handles data fetching from various sources including Bybit API
Transferred from crypto-scan for pump-analysis independence
"""

import os
import requests
import hmac
import hashlib
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)

class BybitClient:
    """Bybit API client for fetching market data"""
    
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.secret_key = os.getenv("BYBIT_SECRET_KEY")
    
    def _generate_signature(self, params: str, timestamp: str) -> str:
        """Generate signature for authenticated requests"""
        if not self.secret_key:
            return ""
        
        param_str = f"{timestamp}{self.api_key}{params}"
        return hmac.new(
            self.secret_key.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, params: Dict = None, authenticated: bool = False) -> Optional[Dict]:
        """Make API request to Bybit"""
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {"Content-Type": "application/json"}
            
            if authenticated and self.api_key and self.secret_key:
                timestamp = str(int(time.time() * 1000))
                param_str = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
                signature = self._generate_signature(param_str, timestamp)
                
                headers.update({
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-SIGN-TYPE": "2",
                    "X-BAPI-TIMESTAMP": timestamp
                })
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Bybit API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error making Bybit request to {endpoint}: {e}")
            return None
    
    def get_symbols(self, category: str = "linear") -> List[str]:
        """Get available trading symbols"""
        try:
            symbols = []
            cursor = ""
            
            while True:
                params = {"category": category, "limit": 1000}
                if cursor:
                    params["cursor"] = cursor
                
                data = self._make_request("/v5/market/instruments-info", params)
                
                if not data or data.get("retCode") != 0:
                    break
                
                instruments = data.get("result", {}).get("list", [])
                for instrument in instruments:
                    symbol = instrument.get("symbol", "")
                    if symbol.endswith("USDT"):
                        symbols.append(symbol)
                
                # Check for next page
                cursor = data.get("result", {}).get("nextPageCursor", "")
                if not cursor:
                    break
            
            logger.info(f"Fetched {len(symbols)} symbols from Bybit")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
    
    def get_kline_data(self, symbol: str, interval: str = "15", limit: int = 1000, 
                       start_time: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Get candlestick data for a symbol"""
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            if start_time:
                params["start"] = start_time
            
            data = self._make_request("/v5/market/kline", params)
            
            if not data or data.get("retCode") != 0:
                logger.warning(f"No data for {symbol}")
                return None
            
            klines = data.get("result", {}).get("list", [])
            if not klines:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert data types
            df['start_time'] = pd.to_numeric(df['start_time'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['turnover'] = pd.to_numeric(df['turnover'])
            
            # Sort by timestamp (oldest first)
            df = df.sort_values('start_time').reset_index(drop=True)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['start_time'], unit='ms')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching kline data for {symbol}: {e}")
            return None
    
    def get_ticker_24hr(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 24hr ticker statistics"""
        try:
            params = {
                "category": "linear",
                "symbol": symbol
            }
            
            data = self._make_request("/v5/market/tickers", params)
            
            if data and data.get("retCode") == 0:
                tickers = data.get("result", {}).get("list", [])
                if tickers:
                    return tickers[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

def get_fallback_symbols() -> List[str]:
    """Get fallback list of popular trading symbols"""
    return [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT', 'DOTUSDT',
        'DOGEUSDT', 'AVAXUSDT', 'SHIBUSDT', 'MATICUSDT', 'LTCUSDT', 'LINKUSDT', 'UNIUSDT',
        'ATOMUSDT', 'ETCUSDT', 'XLMUSDT', 'BCHUSDT', 'FILUSDT', 'THETAUSDT', 'TRXUSDT',
        'EOSUSDT', 'AAVEUSDT', 'MKRUSDT', 'COMPUSDT', 'YFIUSDT', 'SNXUSDT', 'CRVUSDT',
        'SUSHIUSDT', '1INCHUSDT', 'BALUSDT', 'ZRXUSDT', 'LRCUSDT', 'ENJUSDT', 'MANAUSDT',
        'SANDUSDT', 'AXSUSDT', 'CHZUSDT', 'BATUSDT', 'ZILUSDT', 'HOTUSDT', 'ICXUSDT',
        'IOSTUSDT', 'QTUMUSDT', 'OMGUSDT', 'REPUSDT', 'ZENUSDT', 'STORJUSDT', 'GNTUSDT',
        'RENUSDT', 'KNCUSDT', 'ANTUSDT', 'MLNUSDT', 'NMRUSDT', 'DNTUSDT', 'WBTCUSDT',
        'CAKEUSDT', 'BUSDUSDT', 'XVSUSDT', 'BAKEUSDT', 'BURGERUSDT', 'AUTOUSDT', 'ALPHAUSDT',
        'SXPUSDT', 'TWTUSDT', 'VAIUSDT', 'WINUSDT', 'BTTUSDT', 'QUICKUSDT', 'GHSTUSDT',
        'DFYNUSDT', 'ARBUSDT', 'GMXUSDT', 'MAGICUSDT', 'DPXUSDT', 'RDNTUSDT', 'OPUSDT',
        'VELOUSDT', 'NEOUSDT', 'GASUSDT', 'VETUSDT', 'ALGOUSDT', 'XTZUSDT', 'EGLDUSDT',
        'FLOWUSDT', 'ICPUSDT', 'NEARUSDT', 'FTMUSDT', 'DASHUSDT', 'ZECUSDT', 'WAVESUSDT',
        'KSMUSDT', 'DCRUSDT', 'ONTUSDT', 'RVNUSDT', 'CELOUSDT', 'RSRUSDT', 'OCEANUSDT',
        'BANDUSDT', 'BNTUSDT', 'LENDUSDT', 'RUNEUSDT', 'KAVAUSDT', 'YFIIUSDT', 'SRMLUSDT',
        'RAYUSDT', 'FIDAUSDT', 'APEUSDT', 'GALUSDT', 'JASMYUSDT', 'ROSEUSDT', 'IMXUSDT',
        'ENSUSDT', 'STGUSDT', 'API3USDT', 'ANKRUSDT', 'CHRUSDT', 'WOOUSDT', 'FTTUSDT',
        'BSWUSDT', 'BICOLUSDT', 'FLOKIUSDT', 'LEVERUSDT', 'STXUSDT', 'AGIXUSDT', 'GMTUSDT',
        'KDAUSDT', 'APEUSDT', 'JSTUSDT', 'BNXUSDT', 'RNDRUSDT', 'DENTUSDT', 'POLYXUSDT',
        'MASKUSDT', 'DYDXUSDT', 'MINAUSDT', 'CTSIUSDT', 'GALAUSDT', 'LPTUSDT', 'FXSUSDT',
        'DUSKUSDT', 'DEFIUSDT', 'AUDIOUSDT', 'CTXCUSDT', 'DARUSDT', 'ARPAUSDT', 'NKNUSDT',
        'STRAXUSDT', 'UNFIUSDT', 'ROSEUSDT', 'AVAUSDT', 'XEMUSDT', 'SKLUSDT', 'GRTUSDT',
        'CELRUSDT', 'ATMUSDT', 'ASRUSDT', 'COTIUSDT', 'STMXUSDT', 'BELUSDT', 'RIFUSDT',
        'RLCUSDT', 'MCOUSDT', 'XVGUSDT', 'RVNUSDT', 'DGBUSDT', 'SCUSDT', 'ZENUSDT',
        'FETUSDT', 'TFUELUSDT', 'DREPUSDT', 'WINUSDT', 'WAXPUSDT', 'HNTUSDT', 'CRVUSDT',
        'FIDAUSDT', 'ORIONUSDT', 'PONDUSDT', 'DEGOUSDT', 'ALICEUSDT', 'CHZUSDT', 'SANDUSDT',
        'AXSUSDT', 'BURGERUSDT', 'SLPUSDT', 'ERNUSDT', 'KLAYUSDT', 'PHAUSDT', 'BONDUSDT'
    ]

def fetch_symbols_with_fallback() -> List[str]:
    """Fetch symbols from Bybit API with fallback to hardcoded list"""
    try:
        client = BybitClient()
        symbols = client.get_symbols()
        
        if symbols:
            logger.info(f"Successfully fetched {len(symbols)} symbols from Bybit API")
            return symbols
        else:
            logger.warning("Failed to fetch symbols from API, using fallback list")
            return get_fallback_symbols()
            
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return get_fallback_symbols()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for price data"""
    try:
        if df is None or len(df) < 20:
            return df
        
        df = df.copy()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        
        # VWAP calculation
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['volume_price'] = df['typical_price'] * df['volume']
        df['cum_volume_price'] = df['volume_price'].cumsum()
        df['cum_volume'] = df['volume'].cumsum()
        df['vwap'] = df['cum_volume_price'] / df['cum_volume']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR (Average True Range)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift()).abs()
        df['tr3'] = (df['low'] - df['close'].shift()).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change indicators
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Clean up temporary columns
        df.drop(['typical_price', 'volume_price', 'cum_volume_price', 'cum_volume',
                'tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True, errors='ignore')
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return df

def detect_pump_patterns(df: pd.DataFrame, threshold: float = 15.0, window_minutes: int = 30) -> List[Dict]:
    """Detect pump patterns in price data"""
    try:
        if df is None or len(df) < 10:
            return []
        
        pumps = []
        window_candles = window_minutes // 15  # Assuming 15-minute candles
        
        for i in range(window_candles, len(df)):
            current_price = df.iloc[i]['close']
            start_idx = max(0, i - window_candles)
            window_low = df.iloc[start_idx:i]['low'].min()
            
            if window_low > 0:
                price_increase = ((current_price - window_low) / window_low) * 100
                
                if price_increase >= threshold:
                    # Calculate additional metrics
                    volume_spike = df.iloc[i]['volume'] / df.iloc[start_idx:i]['volume'].mean()
                    duration = window_minutes
                    
                    pump = {
                        'start_time': df.iloc[start_idx]['timestamp'],
                        'end_time': df.iloc[i]['timestamp'],
                        'start_price': window_low,
                        'end_price': current_price,
                        'price_increase_pct': price_increase,
                        'volume_spike': volume_spike,
                        'duration_minutes': duration,
                        'candle_index': i
                    }
                    
                    pumps.append(pump)
        
        # Remove overlapping pumps (keep the largest)
        filtered_pumps = []
        for pump in sorted(pumps, key=lambda x: x['price_increase_pct'], reverse=True):
            is_overlapping = False
            for existing in filtered_pumps:
                if (pump['start_time'] <= existing['end_time'] and 
                    pump['end_time'] >= existing['start_time']):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_pumps.append(pump)
        
        return filtered_pumps
        
    except Exception as e:
        logger.error(f"Error detecting pump patterns: {e}")
        return []