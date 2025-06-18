#!/usr/bin/env python3
"""
Pump Analysis System - Automatic detection and analysis of crypto pumps
Analyzes historical pump data and generates GPT insights for learning purposes
"""

import os
import time
import json
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import logging
from learning_system import LearningSystem
from onchain_insights import OnChainAnalyzer
from functions_history import FunctionHistoryManager, PerformanceTracker, GPTLearningEngine
from functions_history.function_manager import FunctionMetadata
from modules import get_heatmap_manager, initialize_heatmap_system
from modules.extended_orderbook_analysis import get_extended_orderbook_analyzer
from modules.heatmap_detectors import get_simplified_heatmap_detector

# Configure logging with DEBUG level for detailed pump analysis debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PumpEvent:
    """Data class for storing pump event information"""
    symbol: str
    start_time: datetime
    price_before: float
    price_peak: float
    price_increase_pct: float
    duration_minutes: int
    volume_spike: float

class BybitDataFetcher:
    """Handles data fetching from Bybit API"""
    
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        self.api_key = os.getenv('BYBIT_API_KEY', '')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY', '')
        
        # Headers for all requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            self.headers['X-BAPI-API-KEY'] = self.api_key
    
    def _get_authenticated_headers(self, params=None):
        """Generate authenticated headers for Bybit API using same logic as crypto-scan"""
        if not self.api_key or not self.api_secret:
            return {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
        
        import hmac
        import hashlib
        
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        if params:
            param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            raw_str = timestamp + self.api_key + recv_window + param_str
        else:
            raw_str = timestamp + self.api_key + recv_window
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            raw_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }
        
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol exists on Bybit futures perpetual
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        endpoint = f"{self.base_url}/v5/market/tickers"
        params = {
            'category': 'linear',
            'symbol': symbol
        }
        
        try:
            headers = self._get_authenticated_headers(params)
            response = requests.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if data['retCode'] == 0 and data['result']['list']:
                return True
            else:
                logger.warning(f"⚠️ Symbol {symbol} is invalid or not available")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Symbol validation failed for {symbol}: {e}")
            return False

    def get_kline_data(self, symbol: str, interval: str = "15", start_time: int = None, limit: int = 1000) -> List[Dict]:
        """
        Fetch kline data from Bybit with symbol validation
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            start_time: Start timestamp in milliseconds
            limit: Number of data points (max 1000)
        """
        endpoint = f"{self.base_url}/v5/market/kline"
        
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['start'] = start_time
            
        try:
            logger.debug(f"📡 Bybit API request: {endpoint} with params: {params}")
            headers = self._get_authenticated_headers(params)
            response = requests.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if data['retCode'] == 0:
                result_data = data['result']['list']
                logger.debug(f"✅ Bybit API success for {symbol}: {len(result_data)} candles retrieved")
                return result_data
            else:
                # Check for invalid symbol error specifically
                if "Symbol Is Invalid" in data.get('retMsg', ''):
                    logger.warning(f"⚠️ Invalid symbol {symbol} - skipping analysis")
                else:
                    logger.error(f"Bybit API error for {symbol}: {data['retMsg']}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return []
    
    def get_active_symbols(self, limit: int = None) -> List[str]:
        """
        Get active trading symbols from Bybit futures perpetual (linear category)
        Uses proven crypto-scan logic with enhanced fallback support
        """
        logger.info(f"📊 Fetching USDT perpetual futures symbols from Bybit...")
        
        # First try public API without authentication (like crypto-scan)
        symbols = self._fetch_symbols_public()
        if symbols and len(symbols) > 100:
            logger.info(f"✅ Retrieved {len(symbols)} symbols via public API")
            return symbols
        
        # Fallback to authenticated API
        symbols = self._fetch_symbols_authenticated()
        if symbols and len(symbols) > 100:
            logger.info(f"✅ Retrieved {len(symbols)} symbols via authenticated API")
            return symbols
        
        # Try to use crypto-scan's symbol data if available
        try:
            crypto_scan_symbols_path = "../crypto-scan/utils/data/cache/bybit_symbols.json"
            if os.path.exists(crypto_scan_symbols_path):
                with open(crypto_scan_symbols_path, 'r') as f:
                    cached_symbols = json.load(f)
                    if cached_symbols and len(cached_symbols) > 100:
                        logger.info(f"✅ Using crypto-scan cached symbols: {len(cached_symbols)} symbols")
                        return [s for s in cached_symbols if s.endswith('USDT')]
        except Exception as e:
            logger.debug(f"Failed to load crypto-scan symbols: {e}")
        
        # Enhanced fallback with comprehensive symbol list for production use
        enhanced_fallback = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT',
            'XLMUSDT', 'UNIUSDT', 'FILUSDT', 'TRXUSDT', 'ETCUSDT', 'NEARUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT',
            'THETAUSDT', 'HBARUSDT', 'EGLDUSDT', 'AAVEUSDT', 'EOSUSDT', 'AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAUSDT', 'CHZUSDT',
            'ALICEUSDT', 'DEGOUSDT', 'GMTUSDT', 'APTUSDT', 'OPUSDT', 'AVAXUSDT', 'SHIBUSDT', 'MATICUSDT', 'DOGEUSDT', 'LUNAUSDT',
            'FTMUSDT', 'CRVUSDT', 'COMPUSDT', 'YFIUSDT', 'MKRUSDT', 'SNXUSDT', 'BANDUSDT', 'KSMUSDT', 'RUNEUSDT', 'OCEANUSDT',
            'RSRUSDT', 'KAVAUSDT', 'INJUSDT', 'ONEUSDT', 'ZILUSDT', 'ZENUSDT', 'RVNUSDT', 'SCUSDT', 'DGBUSDT', 'BTTUSDT',
            'HOTUSDT', 'WINUSDT', 'TLMUSDT', 'DUSKUSDT', 'ANKRUSDT', 'CELRUSDT', 'MTLUSDT', 'OGNUSDT', 'NKNUSDT', 'REQUSDT',
            'AUDIOUSDT', 'CTKUSDT', 'AKROUSDT', 'AXSUSDT', 'HARDUSDT', 'DNTUSDT', 'STRAXUSDT', 'UNFIUSDT', 'ROSEUSDT', 'AVAUSDT',
            'XEMUSDT', 'SKLUSDT', 'GRTUSDT', 'JUVUSDT', 'PSGUSDT', 'CITYUSDT', 'ASRUSDT', 'BARUSDT', 'ATMUSDT', 'ALPACAUSDT',
            'DEXEUSDT', 'FORUSDT', 'EASYUSDT', 'AUTOUSDT', 'TKOUSDT', 'PUNDIXUSDT', 'BAKEUSDT', 'BURGERUSDT', 'SLPUSDT', 'BONDUSDT',
            'MLNUSDT', 'QUICKUSDT', 'C98USDT', 'CLOPUSDT', 'ADXUSDT', 'CFXUSDT', 'QIUSDT', 'IDUSDT', 'BIGTIMEUSDT', 'TRADEJOEUSDT',
            'IMXUSDT', 'GALAUSDT', 'FLMUSDT', 'STGUSDT', 'GMXUSDT', 'NEBULAUSDT', 'BICOUSDT', 'FLOKIUSDT', 'PEPEUSDT', 'SUIUSDT',
            'SEIUSDT', 'CYBERUSDT', 'ARKMUSDT', 'ARKUSDT', 'TIAUSDT', 'BEAMXUSDT', 'PIVXUSDT', 'VICUSDT', 'STXUSDT', 'AGLDUSDT',
            'RAYDIUMUSDT', 'MOVRUSDT', 'NFTUSDT', 'HIGHUSDT', 'CVXUSDT', 'PEOPLEUSDT', 'OOKIUSDT', 'SPELLUSDT', 'USTUSDT', 'JASMYUSDT',
            'DARUSDT', 'OPUSDT', 'REIUSDT', 'STEEMUSDT', 'FISUSDT', 'COWUSDT', 'SPEEDUSDT', 'METAMASKUSDT', 'LSKUSDT', 'ELFUSDT',
            'DYDXUSDT', 'AMBUSDT', 'PHBUSDT', 'BALUSDT', 'NCTUSDT', 'OXTUSDT', 'OBSUSDT', 'SNTUSDT', 'DENTUSDT', 'CKBUSDT',
            'REQUSDT', 'VIBUSDT', 'BTGUSDT', 'REPUSDT', 'ARDRUSDT', 'MDTUSDT', 'STORMXUSDT', 'RIFUSDT', 'PONDUSDT', 'DEGOUSDT',
            'ALICEUSDT', 'LINAUSDT', 'PERPUSDT', 'RAMPUSDT', 'SUPERUSDT', 'CFXUSDT', 'EPSUSDT', 'AUTOUSDT', 'GHSTUSDT', 'ALPINEUSDT',
            'TUSDT', 'ASTRUSDT', 'NBTUSDT', 'GMTUSDT', 'KDAUSDT', 'APEUSDT', 'BSWUSDT', 'MULTIUSDT', 'STEEMUSDT', 'MOBUSDT',
            'NEXOUSDT', 'REIUSDT', 'GALUSDT', 'LDOUSDT', 'EPXUSDT', 'USTCUSDT', 'LOOMUSDT', 'OMGUSDT', 'MASKUSDT', 'LRCUSDT',
            'MAVIAUSDT', 'EDUUSDT', 'SUIUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'COMBOUSDT', 'MAVUSDT', 'PENDLEUSDT', 'ARKMUSDT', 'WLDUSDT',
            'YGGUSDT', 'DODOXUSDT', 'OPENSUSDT', 'POWRUSDT', 'SLERFUSDT', 'AGIUSDT', 'XAIUSDT', 'MANTAUSDT', 'ONDOUSDT', 'LSKUSDT',
            'ALTUSDT', 'JUPUSDT', 'ZETAUSDT', 'RONINUSDT', 'DYMUSDT', 'OMUSDT', 'PIXELUSDT', 'STRKUSDT', 'PORTALUSDT', 'PDAUSDT',
            'AEVOUSDT', 'VANRYUSDT', 'BOMEUSDT', 'ETHFIUSDT', 'ENAUSDT', 'WUSDT', 'TNSRUSDT', 'SAGAUSDT', 'TAOUSDT', 'OMNIUSDT',
            'REZUSDT', 'BBUSDT', 'NOTUSDT', 'TURBOUSDT', 'IOUSDT', 'ZKUSDT', 'MEWUSDT', 'LISTAUSDT', 'ZROUSDT', 'GUSDT',
            'BANUSDT', 'RENDERUSDT', 'TONUSDT', 'DOGUSDT', 'KASUSDT', 'CATIUSDT', 'POPCATUSDT', 'MOODENGUUSDT', 'SUNUSDT', 'RAYUSDT',
            'EIGENUSDT', 'ALPACAUSDT', 'LUMAUSDT', 'NEIROUSDT', 'ACTUSDT', 'PNUTUSDT', 'CHEEMSUSDT', 'SCRUSDT', 'HIPPOUSDT', 'CETUS',
            'MOVEUSDT', 'MEUSDT', 'TRBUSDT', 'ORDIUSDT', 'ETHUSDT', 'RATSUSDT', '1000SATSUSDT', 'BONKUSDT', 'PYTHUSDT', 'WIFUSDT'
        ]
        
        logger.warning(f"⚠️ Using enhanced fallback symbol list: {len(enhanced_fallback)} symbols")
        return enhanced_fallback
    
    def _fetch_symbols_public(self) -> List[str]:
        """Fetch symbols using public API without authentication (crypto-scan approach)"""
        symbols = set()
        cursor = ""
        
        try:
            while True:
                url = "https://api.bybit.com/v5/market/tickers"
                params = {
                    "category": "linear",
                    "limit": 1000
                }
                if cursor:
                    params["cursor"] = cursor
                
                # Public API call without authentication
                response = requests.get(url, params=params, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('retCode') == 0:
                        page_symbols = data.get('result', {}).get('list', [])
                        
                        for item in page_symbols:
                            symbol = item.get('symbol', '')
                            status = item.get('status', '')
                            if symbol and symbol.endswith('USDT') and status == 'Trading':
                                symbols.add(symbol)
                        
                        cursor = data.get('result', {}).get('nextPageCursor')
                        logger.debug(f"📄 Public API: {len(page_symbols)} symbols processed, cursor: {bool(cursor)}")
                        
                        if not cursor:
                            break
                        time.sleep(0.1)
                    else:
                        logger.warning(f"Public API error: {data.get('retMsg', 'Unknown error')}")
                        break
                else:
                    logger.warning(f"Public API HTTP error: {response.status_code}")
                    break
                    
        except Exception as e:
            logger.debug(f"Public API fetch failed: {e}")
            return []
        
        if symbols:
            symbol_list = sorted(list(symbols))
            logger.info(f"📊 Public API retrieved {len(symbol_list)} symbols")
            return symbol_list
        
        return []
    
    def _fetch_symbols_authenticated(self) -> List[str]:
        """Fetch symbols using authenticated API (original approach)"""
        symbols = set()
        cursor = ""
        
        try:
            while True:
                endpoint = f"{self.base_url}/v5/market/tickers"
                params = {
                    "category": "linear",
                    "limit": 1000
                }
                if cursor:
                    params["cursor"] = cursor
                
                headers = self._get_authenticated_headers(params)
                
                response = requests.get(endpoint, params=params, headers=headers, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('retCode') == 0:
                        page_symbols = data.get('result', {}).get('list', [])
                        
                        for item in page_symbols:
                            symbol = item.get('symbol', '')
                            status = item.get('status', '')
                            if symbol and symbol.endswith('USDT') and status == 'Trading':
                                symbols.add(symbol)
                        
                        cursor = data.get('result', {}).get('nextPageCursor')
                        logger.debug(f"📄 Auth API: {len(page_symbols)} symbols processed, cursor: {bool(cursor)}")
                        
                        if not cursor:
                            break
                        time.sleep(0.1)
                    else:
                        logger.warning(f"Auth API error: {data.get('retMsg', 'Unknown error')}")
                        break
                else:
                    logger.warning(f"Auth API HTTP error: {response.status_code}")
                    break
                    
        except Exception as e:
            logger.debug(f"Auth API fetch failed: {e}")
            return []
        
        if symbols:
            symbol_list = sorted(list(symbols))
            logger.info(f"📊 Auth API retrieved {len(symbol_list)} symbols")
            return symbol_list
        
        return []

class PumpDetector:
    """Detects pump events in price data"""
    
    def __init__(self, min_increase_pct: float = 15.0, detection_window_minutes: int = 30):
        self.min_increase_pct = min_increase_pct
        self.detection_window_minutes = detection_window_minutes
        
    def detect_pumps_in_data(self, kline_data: List[List], symbol: str) -> List[PumpEvent]:
        """
        Detect pump events in kline data
        
        Args:
            kline_data: List of kline data from Bybit API
            symbol: Trading symbol
            
        Returns:
            List of detected pump events
        """
        if len(kline_data) < 4:  # Need at least 1 hour of 15-min data
            return []
            
        pumps = []
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(kline_data, columns=[
            'start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Convert to numeric
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['start_time'] = pd.to_numeric(df['start_time'])
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['start_time'], unit='ms')
        df = df.sort_values('datetime')
        
        # Rolling window analysis for pump detection
        window_size = self.detection_window_minutes // 15  # 15-min candles
        processed_periods = set()  # Track processed time periods to avoid duplicates
        
        i = window_size
        while i < len(df):
            window_start = i - window_size
            window_data = df.iloc[window_start:i+1]
            
            # Create unique period identifier
            period_key = (window_data.iloc[0]['start_time'], window_data.iloc[-1]['start_time'])
            
            # Skip if this period was already processed
            if period_key in processed_periods:
                i += 1
                continue
            
            price_start = window_data.iloc[0]['close']
            price_peak = window_data['high'].max()
            
            increase_pct = ((price_peak - price_start) / price_start) * 100
            
            if increase_pct >= self.min_increase_pct:
                # Calculate additional metrics
                volume_before = df.iloc[max(0, window_start-6):window_start]['volume'].mean()
                volume_during = window_data['volume'].mean()
                volume_spike = volume_during / volume_before if volume_before > 0 else 1
                
                pump_event = PumpEvent(
                    symbol=symbol,
                    start_time=window_data.iloc[0]['datetime'],
                    price_before=price_start,
                    price_peak=price_peak,
                    price_increase_pct=increase_pct,
                    duration_minutes=self.detection_window_minutes,
                    volume_spike=volume_spike
                )
                
                pumps.append(pump_event)
                processed_periods.add(period_key)
                
                # Skip ahead to avoid overlapping pumps
                i += window_size
            else:
                i += 1
                
        # Additional deduplication - merge pumps that are too close in time
        if len(pumps) > 1:
            pumps = self._deduplicate_pumps(pumps)
                
        return pumps
    
    def _deduplicate_pumps(self, pumps: List[PumpEvent]) -> List[PumpEvent]:
        """
        Remove duplicate pumps that are too close in time (within 15 minutes)
        Keep the pump with highest price increase
        """
        if len(pumps) <= 1:
            return pumps
            
        # Sort pumps by start time
        pumps.sort(key=lambda p: p.start_time)
        
        deduplicated = []
        
        for current_pump in pumps:
            # Check if this pump is too close to any existing pump
            is_duplicate = False
            
            for existing_pump in deduplicated:
                time_diff = abs((current_pump.start_time - existing_pump.start_time).total_seconds() / 60)
                
                # If pumps are within 15 minutes, they're duplicates
                if time_diff <= 15:
                    is_duplicate = True
                    # Keep the one with higher price increase
                    if current_pump.price_increase_pct > existing_pump.price_increase_pct:
                        # Replace existing with current
                        deduplicated.remove(existing_pump)
                        deduplicated.append(current_pump)
                    break
            
            if not is_duplicate:
                deduplicated.append(current_pump)
        
        logger.info(f"🔄 Deduplication: {len(pumps)} → {len(deduplicated)} pumps")
        return deduplicated

class PrePumpAnalyzer:
    """Analyzes pre-pump conditions"""
    
    def __init__(self, bybit_fetcher: BybitDataFetcher):
        self.bybit = bybit_fetcher
        
    def analyze_pre_pump_conditions(self, pump_event: PumpEvent) -> Dict:
        """
        Analyze conditions 60 minutes before pump
        
        Args:
            pump_event: Detected pump event
            
        Returns:
            Dictionary with pre-pump analysis
        """
        # Calculate time 60 minutes before pump
        pre_pump_start = pump_event.start_time - timedelta(minutes=60)
        start_timestamp = int(pre_pump_start.timestamp() * 1000)
        
        # Get 1-hour of data before pump (4 x 15-min candles)
        logger.info(f"📊 Fetching pre-pump data for {pump_event.symbol}: 60min before {pump_event.start_time}")
        pre_pump_data = self.bybit.get_kline_data(
            symbol=pump_event.symbol,
            interval="15",
            start_time=start_timestamp,
            limit=4
        )
        
        if not pre_pump_data:
            logger.warning(f"⚠️ No pre-pump data available for {pump_event.symbol} at timestamp {start_timestamp}")
            return None
        
        logger.info(f"✅ Retrieved {len(pre_pump_data)} candles for {pump_event.symbol} pre-pump analysis")
            
        # Convert to DataFrame
        df = pd.DataFrame(pre_pump_data, columns=[
            'start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Calculate indicators
        analysis = {
            'symbol': pump_event.symbol,
            'pump_start_time': pump_event.start_time.isoformat(),
            'pump_increase_pct': pump_event.price_increase_pct,
            'pre_pump_period': '60_minutes_before',
            'kline_data': df.to_dict('records'),  # Add raw kline data for testing
            
            # Price analysis
            'price_volatility': df['close'].std(),
            'price_trend': self._calculate_trend(df['close']),
            'price_compression': self._detect_compression(df),
            
            # Volume analysis
            'avg_volume': df['volume'].mean(),
            'volume_trend': self._calculate_trend(df['volume']),
            'volume_spikes': self._detect_volume_spikes(df),
            
            # Technical indicators
            'rsi': self._calculate_rsi(df['close']),
            'vwap': self._calculate_vwap(df),
            'fake_rejects': self._detect_fake_rejects(df),
            
            # Market structure
            'support_resistance': self._identify_support_resistance(df),
            'liquidity_gaps': self._detect_liquidity_gaps(df),
        }
        
        return analysis
    
    def _calculate_trend(self, prices: pd.Series) -> str:
        """Calculate price trend direction"""
        if len(prices) < 2:
            return "neutral"
            
        first_half = prices[:len(prices)//2].mean()
        second_half = prices[len(prices)//2:].mean()
        
        # Prevent division by zero
        if first_half == 0 or pd.isna(first_half) or pd.isna(second_half):
            return "neutral"
        
        change_pct = ((second_half - first_half) / first_half) * 100
        
        if change_pct > 1:
            return "bullish"
        elif change_pct < -1:
            return "bearish"
        else:
            return "neutral"
    
    def _detect_compression(self, df: pd.DataFrame) -> Dict:
        """Detect price compression patterns"""
        price_range = df['high'].max() - df['low'].min()
        avg_price = df['close'].mean()
        compression_ratio = (price_range / avg_price) * 100
        
        return {
            'compression_ratio_pct': compression_ratio,
            'is_compressed': compression_ratio < 2.0  # Less than 2% range
        }
    
    def _detect_volume_spikes(self, df: pd.DataFrame) -> List[Dict]:
        """Detect volume spikes in pre-pump period"""
        avg_volume = df['volume'].mean()
        spikes = []
        
        for i, row in df.iterrows():
            if row['volume'] > avg_volume * 2:  # 2x average volume
                spikes.append({
                    'time_minutes_before_pump': (len(df) - i) * 5,
                    'volume_multiplier': row['volume'] / avg_volume
                })
                
        return spikes
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period:
            return 50.0  # Neutral RSI
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_vwap(self, df: pd.DataFrame) -> Dict:
        """Calculate VWAP and analyze position relative to it"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        
        final_price = df['close'].iloc[-1]
        
        return {
            'vwap_value': float(vwap),
            'price_vs_vwap_pct': ((final_price - vwap) / vwap) * 100,
            'above_vwap': final_price > vwap
        }
    
    def _detect_fake_rejects(self, df: pd.DataFrame) -> List[Dict]:
        """Detect fake rejection patterns (long wicks followed by recovery)"""
        fake_rejects = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Calculate wick sizes
            lower_wick = abs(current['close'] - current['low']) / current['close']
            upper_wick = abs(current['high'] - current['close']) / current['close']
            
            # Detect long lower wick followed by recovery
            if (lower_wick > 0.02 and  # Wick > 2%
                current['close'] > previous['close'] and  # Recovery
                current['close'] > (current['low'] + (current['high'] - current['low']) * 0.6)):  # Close in upper 40%
                
                fake_rejects.append({
                    'time_minutes_before_pump': (len(df) - i) * 5,
                    'wick_size_pct': lower_wick * 100,
                    'recovery_strength': ((current['close'] - current['low']) / (current['high'] - current['low'])) * 100
                })
                
        return fake_rejects
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Identify key support and resistance levels"""
        highs = df['high'].values
        lows = df['low'].values
        
        # Simple pivot identification
        resistance_levels = []
        support_levels = []
        
        for i in range(1, len(highs)-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                resistance_levels.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                support_levels.append(lows[i])
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'key_resistance': max(resistance_levels) if resistance_levels else None,
            'key_support': min(support_levels) if support_levels else None
        }
    
    def _detect_liquidity_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Detect potential liquidity gaps"""
        gaps = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Gap up
            if current['low'] > previous['high']:
                gap_size = ((current['low'] - previous['high']) / previous['high']) * 100
                gaps.append({
                    'type': 'gap_up',
                    'size_pct': gap_size,
                    'time_minutes_before_pump': (len(df) - i) * 5
                })
            
            # Gap down
            elif current['high'] < previous['low']:
                gap_size = ((previous['low'] - current['high']) / current['high']) * 100
                gaps.append({
                    'type': 'gap_down',
                    'size_pct': gap_size,
                    'time_minutes_before_pump': (len(df) - i) * 5
                })
        
        return gaps

class GPTAnalyzer:
    """Handles GPT analysis of pre-pump conditions with function history context"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.function_history_file = "function_history.json"
        self.max_history_size = 5  # Przechowujemy 5 ostatnich funkcji dla kontekstu
        self.function_history = self._load_function_history()
    
    def _load_function_history(self) -> List[Dict]:
        """Load function history from JSON file"""
        try:
            if os.path.exists(self.function_history_file):
                with open(self.function_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.warning(f"Error loading function history: {e}")
            return []
    
    def _save_function_history(self):
        """Save function history to JSON file"""
        try:
            with open(self.function_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.function_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving function history: {e}")
    
    def _add_to_function_history(self, symbol: str, date: str, function_code: str, pump_increase: float):
        """Add new function to history and maintain max size"""
        function_entry = {
            'symbol': symbol,
            'date': date,
            'function_code': function_code,
            'pump_increase': pump_increase,
            'created_at': datetime.now().isoformat()
        }
        
        # Dodaj na początek listy
        self.function_history.insert(0, function_entry)
        
        # Utrzymuj maksymalny rozmiar historii
        if len(self.function_history) > self.max_history_size:
            self.function_history = self.function_history[:self.max_history_size]
        
        # Zapisz do pliku
        self._save_function_history()
        logger.info(f"📚 Added function to history: detect_{symbol}_{date}_preconditions (pump: +{pump_increase:.1f}%)")
    
    def _format_function_history_context(self) -> str:
        """Format function history for GPT context"""
        if not self.function_history:
            return ""
        
        context = "\n=== WCZEŚNIEJSZE FUNKCJE DETEKCJI ===\n"
        context += "Poniżej znajdują się ostatnie wygenerowane funkcje detekcyjne z podobnych przypadków pump'ów:\n\n"
        
        for i, entry in enumerate(self.function_history, 1):
            context += f"# Funkcja {i}: {entry['symbol']} (+{entry['pump_increase']:.1f}%) - {entry['date']}\n"
            context += entry['function_code'] + "\n\n"
        
        context += "=== KONIEC HISTORII FUNKCJI ===\n\n"
        context += "UWAGI DO NOWEJ FUNKCJI:\n"
        context += "- Porównaj nowy przypadek z powyższymi wzorcami\n"
        context += "- Unikaj powielania identycznej logiki\n"
        context += "- Wykorzystaj najlepsze elementy z poprzednich funkcji\n"
        context += "- Dostosuj thresholdy do specyfiki nowego przypadku\n"
        context += "- Jeśli wzorzec jest podobny, ulepsz istniejącą logikę\n\n"
        
        return context
        
    def generate_pump_analysis(self, pre_pump_data: Dict) -> str:
        """
        Generate GPT analysis of pre-pump conditions
        
        Args:
            pre_pump_data: Dictionary with pre-pump analysis data
            
        Returns:
            GPT analysis text
        """
        
        # Format data for GPT prompt
        prompt = self._format_analysis_prompt(pre_pump_data)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "system",
                        "content": """Jesteś ekspertem analizy technicznej rynku kryptowalut. 
                        Analizujesz warunki pre-pump (60 minut przed nagłym wzrostem ceny) aby 
                        zidentyfikować wzorce i sygnały, które mogły przewidzieć nadchodzący pump.
                        
                        Odpowiadaj w języku polskim. Skup się na:
                        1. Identyfikacji kluczowych sygnałów pre-pump
                        2. Analizie struktury rynku przed ruchem
                        3. Ocenie siły sygnałów akumulacyjnych
                        4. Wskazaniu najważniejszych wskaźników ostrzegawczych
                        5. Podsumowaniu lekcji do zastosowania w przyszłości"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.7,
                timeout=45  # 45 second timeout to prevent hanging
            )
            
            if response and response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.warning("GPT response was empty or malformed")
                return "Analiza GPT: Otrzymano pustą odpowiedź z OpenAI API"
            
        except Exception as e:
            logger.error(f"GPT analysis error: {e}")
            return f"Błąd podczas generowania analizy GPT: {e}"

    def generate_strategic_analysis(self, pre_pump_data: Dict, pump_event: 'PumpEvent', candle_data: List = None) -> str:
        """
        Generate strategic pre-pump analysis based on real pump case
        
        Args:
            pre_pump_data: Dictionary with pre-pump analysis data
            pump_event: PumpEvent with pump details
            candle_data: Raw candle data from 60 minutes before pump
            
        Returns:
            Strategic analysis with pattern identification
        """
        
        # Format 60-minute pre-pump window data for GPT
        formatted_data = self._format_pre_pump_window_data(pre_pump_data, pump_event, candle_data)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "system",
                        "content": """Jesteś analitykiem strategii crypto, specjalizującym się w identyfikacji unikalnych wzorców pre-pump.
                        
                        ZADANIE:
                        Analizuj 60-minutowe okno przed pumpem i zidentyfikuj unikalne cechy sygnału.
                        
                        PODEJŚCIE:
                        - Działaj jak analityk strategii, nie programista
                        - Identyfikuj unikalne cechy tego konkretnego przypadku
                        - Unikaj sztywnych warunków typu "RSI == 50.0"
                        - Szukaj wzorców: kompresja, akumulacja, VWAP pinning, volume breakout
                        - Opisuj logikę i kontekst każdego sygnału
                        
                        FORMAT ODPOWIEDZI:
                        1. **Identyfikacja wzorca** - jaki unikalny wzorzec pre-pump wykryłeś
                        2. **Kluczowe sygnały** - które wskaźniki były najważniejsze
                        3. **Timing i sekwencja** - jak sygnały się rozwijały w czasie
                        4. **Przykładowa detekcja** - jak można wykryć podobne przypadki
                        5. **Komentarze strategiczne** - dlaczego ten wzorzec działał
                        
                        Odpowiadaj w języku polskim, ton ekspercki i praktyczny."""
                    },
                    {
                        "role": "user",
                        "content": formatted_data
                    }
                ],
                max_tokens=2000,
                temperature=0.4,  # Slightly higher for more creative analysis
                timeout=45  # 45 second timeout to prevent hanging
            )
            
            if response and response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.warning("GPT strategic analysis response was empty or malformed")
                return "Analiza strategiczna: Otrzymano pustą odpowiedź z OpenAI API"
            
        except Exception as e:
            logger.error(f"GPT strategic analysis error: {e}")
            return f"Błąd podczas generowania analizy strategicznej GPT: {e}"

    def _format_pre_pump_window_data(self, pre_pump_data: Dict, pump_event: 'PumpEvent', candle_data: List = None) -> str:
        """
        Format 60-minute pre-pump window data for GPT analysis
        
        Args:
            pre_pump_data: Pre-pump analysis dictionary
            pump_event: PumpEvent with pump details
            candle_data: Raw candle data (if available)
            
        Returns:
            Formatted data string for GPT
        """
        
        # Format pump event details
        pump_info = f"""
PUMP EVENT DETAILS:
• Symbol: {pump_event.symbol}
• Start Time: {pump_event.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}
• Price Increase: {pump_event.price_increase_pct:.1f}%
• Duration: {pump_event.duration_minutes} minutes
• Volume Spike: {pump_event.volume_spike:.1f}x
• Price Before: ${pump_event.price_before:.4f}
• Price Peak: ${pump_event.price_peak:.4f}
"""

        # Format 60-minute pre-pump analysis
        analysis_summary = f"""
PRE-PUMP ANALYSIS (60 minutes before):
• RSI (14): {pre_pump_data.get('rsi_14', 'N/A')}
• Price Trend: {pre_pump_data.get('price_trend', 'N/A')}
• Volume Spike Detected: {'Yes' if pre_pump_data.get('volume_spike', False) else 'No'}
• Compression Detected: {'Yes' if pre_pump_data.get('compression_detected', False) else 'No'}
• VWAP Analysis: {pre_pump_data.get('vwap_analysis', {}).get('price_vs_vwap', 'N/A')} vs VWAP
• Support Level: ${pre_pump_data.get('support_resistance', {}).get('support', 'N/A')}
• Resistance Level: ${pre_pump_data.get('support_resistance', {}).get('resistance', 'N/A')}
"""

        # Format fake rejects if present
        fake_rejects = pre_pump_data.get('fake_rejects', [])
        if fake_rejects:
            rejects_info = f"\nFAKE REJECTS DETECTED:\n"
            for i, reject in enumerate(fake_rejects[:3], 1):  # Limit to 3 most recent
                rejects_info += f"• Reject {i}: {reject.get('severity', 'unknown')} severity\n"
        else:
            rejects_info = "\nFAKE REJECTS: None detected\n"

        # Format volume spikes if present
        volume_spikes = pre_pump_data.get('volume_spikes', [])
        if volume_spikes:
            volume_info = f"\nVOLUME SPIKES:\n"
            for i, spike in enumerate(volume_spikes[:3], 1):  # Limit to 3 most significant
                volume_info += f"• Spike {i}: {spike.get('multiplier', 'N/A')}x volume increase\n"
        else:
            volume_info = "\nVOLUME SPIKES: Standard volume patterns\n"

        # Format liquidity gaps if present
        liquidity_gaps = pre_pump_data.get('liquidity_gaps', [])
        if liquidity_gaps:
            liquidity_info = f"\nLIQUIDITY GAPS:\n"
            for i, gap in enumerate(liquidity_gaps[:2], 1):  # Limit to 2 most significant
                liquidity_info += f"• Gap {i}: {gap.get('size_pct', 'N/A')}% gap detected\n"
        else:
            liquidity_info = "\nLIQUIDITY GAPS: None detected\n"

        # Add candle data summary if available
        candle_info = ""
        if candle_data and len(candle_data) > 0:
            # Analyze last 4 candles (60 minutes = 4x 15-minute candles)
            recent_candles = candle_data[-4:] if len(candle_data) >= 4 else candle_data
            
            candle_info = f"""
CANDLE DATA SUMMARY (Last 60 minutes):
• Number of Candles: {len(recent_candles)}
• Timeframe: 15-minute intervals
• Data Format: [timestamp, open, high, low, close, volume, vwap, rsi]

CANDLE PATTERNS:
"""
            for i, candle in enumerate(recent_candles, 1):
                if isinstance(candle, dict):
                    # Handle dict format
                    open_price = candle.get('open', 0)
                    close_price = candle.get('close', 0)
                    volume = candle.get('volume', 0)
                    rsi = candle.get('rsi', 0)
                else:
                    # Handle list format [timestamp, open, high, low, close, volume]
                    open_price = candle[1] if len(candle) > 1 else 0
                    close_price = candle[4] if len(candle) > 4 else 0
                    volume = candle[5] if len(candle) > 5 else 0
                    rsi = candle[7] if len(candle) > 7 else "N/A"
                
                body_pct = ((close_price - open_price) / open_price * 100) if open_price > 0 else 0
                candle_info += f"• Candle {i}: {body_pct:+.2f}% body, Volume: {volume:.0f}, RSI: {rsi}\n"

        # Add heatmap analysis for strategic context
        heatmap_info = ""
        try:
            from modules.heatmap_integration import get_heatmap_manager
            heatmap_manager = get_heatmap_manager()
            heatmap_data = heatmap_manager.get_heatmap_for_gpt(pump_event.symbol)
            
            if heatmap_data and heatmap_data.get('heatmap_analysis'):
                analysis = heatmap_data['heatmap_analysis']
                
                wall_status = "TAK" if analysis.get('wall_disappeared', False) else "NIE"
                if analysis.get('wall_disappeared', False):
                    wall_side = analysis.get('wall_disappeared_side', 'nieznana')
                    wall_size = analysis.get('wall_disappeared_size', 0) * 100
                    wall_status = f"TAK ({wall_side} side, -{wall_size:.1f}%)"
                
                pinning_status = "TAK" if analysis.get('liquidity_pinning', False) else "NIE"
                void_status = "TAK" if analysis.get('liquidity_void_reaction', False) else "NIE"
                cluster_tilt = analysis.get('volume_cluster_tilt', 'neutral')
                
                heatmap_info = f"""
HEATMAP ORDERBOOKU (60 min przed pumpem):
• Zniknięcie ścian podaży: {wall_status}
• Pinning ceny do płynności: {pinning_status}  
• Reakcja ceny na pustkę (void): {void_status}
• Nachylenie klastrów wolumenu: {cluster_tilt}
• Kontekst: {heatmap_data.get('heatmap_summary', 'Strukturalne sygnały orderbooku dostępne')}
"""
            else:
                heatmap_info = f"""
HEATMAP ORDERBOOKU:
• Status: Brak danych orderbooku dla {pump_event.symbol}
• Analiza: Oparta na sygnałach cenowych i wolumenowych
"""
        except Exception as e:
            heatmap_info = f"""
HEATMAP ORDERBOOKU:
• Status: System heatmapy niedostępny
• Uwaga: Analiza skupiona na tradycyjnych wskaźnikach
"""

        # Combine all information
        formatted_data = f"""
{pump_info}
{analysis_summary}
{rejects_info}
{volume_info}
{liquidity_info}
{heatmap_info}
{candle_info}

ANALYSIS REQUEST:
Przeanalizuj powyższe dane z 60-minutowego okna przed pumpem {pump_event.symbol} (+{pump_event.price_increase_pct:.1f}%).
Zidentyfikuj unikalne wzorce i sygnały, które mogły przewidzieć ten pump.
Skup się na sekwencji zdarzeń, timing'u sygnałów i unikalnych cechach tego przypadku.
Uwzględnij sygnały heatmapy jako dodatkowy kontekst strukturalny, nie jako główne warunki.
"""
        
        return formatted_data
    
    def _format_analysis_prompt(self, data: Dict) -> str:
        """Format pre-pump data into GPT prompt"""
        
        prompt = f"""
ANALIZA PRE-PUMP - {data['symbol']}

=== INFORMACJE O PUMPIE ===
• Symbol: {data['symbol']}
• Czas rozpoczęcia pumpu: {data['pump_start_time']}
• Wzrost ceny: +{data['pump_increase_pct']:.1f}%
• Okres analizy: {data['pre_pump_period']}

=== ANALIZA CENY (60 min przed pumpem) ===
• Zmienność ceny: {data['price_volatility']:.6f}
• Trend cenowy: {data['price_trend']}
• Kompresja cenowa: {data['price_compression']['compression_ratio_pct']:.2f}% (skompresowana: {data['price_compression']['is_compressed']})

=== ANALIZA WOLUMENU ===
• Średni wolumen: {data['avg_volume']:.0f}
• Trend wolumenu: {data['volume_trend']}
• Wykryte spike'i wolumenu: {len(data['volume_spikes'])}
"""

        if data['volume_spikes']:
            prompt += "\n• Szczegóły spike'ów wolumenu:\n"
            for spike in data['volume_spikes']:
                prompt += f"  - {spike['time_minutes_before_pump']} min przed: {spike['volume_multiplier']:.1f}x średniego\n"

        prompt += f"""
=== WSKAŹNIKI TECHNICZNE ===
• RSI: {data['rsi']:.1f}
• VWAP: {data['vwap']['vwap_value']:.6f}
• Pozycja vs VWAP: {data['vwap']['price_vs_vwap_pct']:.2f}% ({'powyżej' if data['vwap']['above_vwap'] else 'poniżej'} VWAP)

=== STRUKTURY RYNKOWE ===
• Fake reject'y wykryte: {len(data['fake_rejects'])}
"""

        if data['fake_rejects']:
            prompt += "• Szczegóły fake reject'ów:\n"
            for fr in data['fake_rejects']:
                prompt += f"  - {fr['time_minutes_before_pump']} min przed: wick {fr['wick_size_pct']:.1f}%, recovery {fr['recovery_strength']:.1f}%\n"

        if data['support_resistance']['key_support'] or data['support_resistance']['key_resistance']:
            support_text = f"{data['support_resistance']['key_support']:.6f}" if data['support_resistance']['key_support'] is not None else 'brak'
            resistance_text = f"{data['support_resistance']['key_resistance']:.6f}" if data['support_resistance']['key_resistance'] is not None else 'brak'
            prompt += f"""
• Kluczowe poziomy:
  - Wsparcie: {support_text}
  - Opór: {resistance_text}
"""

        if data['liquidity_gaps']:
            prompt += f"\n• Luki płynnościowe: {len(data['liquidity_gaps'])}\n"
            for gap in data['liquidity_gaps']:
                prompt += f"  - {gap['type']}: {gap['size_pct']:.2f}% ({gap['time_minutes_before_pump']} min przed)\n"

        # Add extended orderbook heatmap analysis with simplified fallback
        try:
            extended_analyzer = get_extended_orderbook_analyzer()
            orderbook_analysis = extended_analyzer.analyze_symbol_extended(data['symbol'])
            
            if orderbook_analysis and orderbook_analysis.get('detectors'):
                # Format extended analysis for GPT context
                formatted_context = extended_analyzer.format_for_gpt_context(orderbook_analysis)
                prompt += f"\n{formatted_context}\n"
            else:
                # Fallback to simplified heatmap detectors
                try:
                    simplified_detector = get_simplified_heatmap_detector()
                    simplified_analysis = simplified_detector.format_for_gpt_prompt(data['symbol'])
                    prompt += f"\n{simplified_analysis}\n"
                except Exception as fallback_error:
                    prompt += f"""
=== ANALIZA HEATMAPY ORDERBOOKU ===
• Status: Brak dostępu do danych orderbooku dla {data['symbol']}
• Uwaga: Analiza skupiona na tradycyjnych wskaźnikach technicznych
• Kontekst: GPT powinien skupić się na sygnałach cenowych, wolumenowych i on-chain
"""
        except Exception as e:
            # Try simplified detectors as final fallback
            try:
                simplified_detector = get_simplified_heatmap_detector()
                simplified_analysis = simplified_detector.format_for_gpt_prompt(data['symbol'])
                prompt += f"\n{simplified_analysis}\n"
            except Exception as final_error:
                prompt += f"""
=== ANALIZA HEATMAPY ORDERBOOKU ===
• Status: System analizy orderbooku niedostępny
• Uwaga: GPT generuje analizę bez kontekstu orderbooku
"""

        # Add on-chain insights section
        if 'onchain_insights' in data and data['onchain_insights']:
            prompt += f"""
=== AKTYWNOŚĆ ON-CHAIN ===
Wykryte sygnały z blockchain'a w okresie przed pumpem:
"""
            for insight in data['onchain_insights']:
                prompt += f"• {insight}\n"
        
        prompt += """

=== ZADANIE ANALIZY ===
Na podstawie powyższych danych z 60 minut przed pumpem oraz sygnałów on-chain, przeanalizuj:

1. Jakie były najważniejsze sygnały ostrzegawcze?
2. Czy struktura rynku wskazywała na przygotowania do ruchu?
3. Które wskaźniki były najbardziej predykcyjne?
4. Jakie wzorce akumulacji można zidentyfikować?
5. Co można było zauważyć wcześniej, żeby przewidzieć pump?

Podaj konkretną, praktyczną analizę z naciskiem na aplikowalność w przyszłych sytuacjach.
"""

        return prompt

    def _format_detector_prompt(self, data: Dict, pump_event: 'PumpEvent') -> str:
        """Format data into prompt for detector function generation"""
        
        # Extract date from pump event
        pump_date = pump_event.start_time.strftime("%Y%m%d")
        symbol = pump_event.symbol
        
        prompt = f"""
GENEROWANIE FUNKCJI DETEKTORA PRE-PUMP

=== DANE PRZYPADKU ===
• Symbol: {symbol}
• Data pumpu: {pump_date}
• Wzrost ceny: +{pump_event.price_increase_pct:.1f}%
• Czas trwania: {pump_event.duration_minutes} minut

=== KLUCZOWE CHARAKTERYSTYKI PRE-PUMP ===
• Kompresja cenowa: {data['price_compression']['compression_ratio_pct']:.2f}% (skompresowana: {data['price_compression']['is_compressed']})
• Trend cenowy: {data['price_trend']}
• RSI: {data['rsi']:.1f}
• Pozycja vs VWAP: {data['vwap']['price_vs_vwap_pct']:.2f}% ({'powyżej' if data['vwap']['above_vwap'] else 'poniżej'})
• Średni wolumen: {data['avg_volume']:.0f}
• Trend wolumenu: {data['volume_trend']}
• Liczba volume spike'ów: {len(data['volume_spikes'])}
• Liczba fake reject'ów: {len(data['fake_rejects'])}
• Kluczowe wsparcie: {data['support_resistance']['key_support'] or 'brak'}
• Kluczowy opór: {data['support_resistance']['key_resistance'] or 'brak'}
"""

        if data['volume_spikes']:
            prompt += "\n=== SZCZEGÓŁY VOLUME SPIKE'ÓW ===\n"
            for spike in data['volume_spikes'][:3]:  # Top 3 spikes
                prompt += f"• {spike['time_minutes_before_pump']} min przed: {spike['volume_multiplier']:.1f}x średniego\n"

        if data['fake_rejects']:
            prompt += "\n=== SZCZEGÓŁY FAKE REJECT'ÓW ===\n"
            for fr in data['fake_rejects'][:2]:  # Top 2 fake rejects
                prompt += f"• {fr['time_minutes_before_pump']} min przed: wick {fr['wick_size_pct']:.1f}%, recovery {fr['recovery_strength']:.1f}%\n"

        prompt += f"""

=== ZADANIE ===
Wygeneruj funkcję w języku Python, która potrafiłaby wykryć taki przypadek pre-pump.

WYMAGANIA FUNKCJI:
1. Nazwa: detect_{symbol}_{pump_date}_preconditions(df)
2. Parameter: pandas DataFrame z kolumnami ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'rsi']
3. Zwraca: True jeśli setup został spełniony, False w przeciwnym razie
4. Używaj tylko pandas i numpy (bez zewnętrznych bibliotek)
5. Implementuj konkretną logikę bazującą na dostarczonych parametrach
6. Dodaj docstring opisujący wzorzec

PODEJŚCIE DO IMPLEMENTACJI:
• Sprawdź kompresję cenową (zakres high-low vs średnia)
• Zweryfikuj pozycję vs VWAP
• Wykryj volume spike'y (porównaj z moving average)
• Sprawdź fake reject'y (długie wicki + recovery)
• Uwzględnij trend RSI i ceny
• Użyj realistic thresholds bazujących na podanych wartościach

Wygeneruj kompletną, działającą funkcję Python gotową do zapisania w pliku.
"""

        return prompt
    
    def generate_pump_analysis_with_context(self, data: Dict, memory_context: str) -> str:
        """
        Generate GPT analysis of pre-pump conditions with memory context
        
        Args:
            data: Dictionary with pre-pump analysis data
            memory_context: Enhanced context from GPT Memory Engine
            
        Returns:
            GPT analysis text
        """
        
        prompt = memory_context + "\n\n" + self._format_analysis_prompt(data)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """Jesteś ekspertem analizy rynku kryptowalut specjalizującym się w wykrywaniu wzorców pre-pump.
                        Wykorzystaj kontekst z poprzednich analiz aby wygenerować precyzyjną analizę. 
                        Porównaj obecny przypadek z podobnymi wzorcami z przeszłości.
                        Skup się na praktycznych wskazówkach bazujących na historycznych danych."""
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                timeout=45
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating GPT analysis with context: {e}")
            return f"Błąd analizy GPT: {str(e)}"
    
    def generate_detector_function_with_context(self, data: Dict, pump_event: 'PumpEvent', memory_context: str) -> str:
        """
        Generate Python detector function with memory context and pattern recognition
        
        Args:
            data: Dictionary with pre-pump analysis data
            pump_event: PumpEvent with pump details
            memory_context: Enhanced context from GPT Memory Engine
            
        Returns:
            Python function code as string
        """
        
        prompt = memory_context + "\n\n" + self._format_detector_prompt(data, pump_event)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """Jesteś ekspertem programowania funkcji detekcyjnych pre-pump.
                        Wykorzystaj kontekst poprzednich funkcji aby stworzyć ulepszony detektor.
                        - Unikaj duplikowania logiki z poprzednich funkcji
                        - Kombinuj najlepsze elementy z udanych detektorów
                        - Dostosuj progi na podstawie historycznych danych
                        - Twórz unikalne wzorce dla tego konkretnego przypadku
                        - Użyj tylko pandas i numpy, bez zewnętrznych bibliotek"""
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                timeout=45
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating detector function with context: {e}")
            # Return a basic fallback function
            return f"""
def detect_{pump_event.symbol.lower()}_{pump_event.start_time.strftime('%Y%m%d')}_preconditions(df):
    \"\"\"Fallback detector function due to GPT error: {str(e)}\"\"\"
    return False, 0.0, ['error']
"""

class TelegramNotifier:
    """Handles Telegram notifications"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_message(self, message: str) -> bool:
        """
        Send message to Telegram chat
        
        Args:
            message: Message text to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        
        url = f"{self.base_url}/sendMessage"
        
        # Split long messages
        max_length = 4000
        if len(message) > max_length:
            parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]
            for i, part in enumerate(parts):
                success = self._send_single_message(f"📊 CZĘŚĆ {i+1}/{len(parts)}\n\n{part}")
                if not success:
                    return False
                time.sleep(1)  # Avoid rate limiting
            return True
        else:
            return self._send_single_message(message)
    
    def _send_single_message(self, message: str) -> bool:
        """Send single message to Telegram"""
        url = f"{self.base_url}/sendMessage"
        
        # Escape HTML characters that might cause 400 Bad Request
        message = self._escape_html(message)
        
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 400:
                # If HTML parsing fails, try without parse_mode
                logger.warning("HTML parse error, retrying without formatting")
                payload['parse_mode'] = None
                payload['text'] = message.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')
                response = requests.post(url, json=payload, timeout=30)
            
            response.raise_for_status()
            
            result = response.json()
            if result.get('ok'):
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {result.get('description')}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            if hasattr(e, 'response') and e.response:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Telegram error details: {error_detail}")
                except:
                    logger.error(f"Raw response: {e.response.text}")
            return False
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters for Telegram"""
        # Replace problematic characters that cause 400 Bad Request
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Re-add allowed HTML tags
        text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
        text = text.replace('&lt;code&gt;', '<code>').replace('&lt;/code&gt;', '</code>')
        
        return text

class PumpAnalysisSystem:
    """Main pump analysis system orchestrator"""
    
    def __init__(self):
        # Load environment variables
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Validate required environment variables
        if not all([self.openai_api_key, self.telegram_bot_token, self.telegram_chat_id]):
            raise ValueError("Missing required environment variables. Please check OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, and TELEGRAM_CHAT_ID")
        
        # Initialize components
        self.bybit = BybitDataFetcher()
        self.pump_detector = PumpDetector(min_increase_pct=15.0, detection_window_minutes=30)
        self.pre_pump_analyzer = PrePumpAnalyzer(self.bybit)
        self.gpt_analyzer = GPTAnalyzer(self.openai_api_key)
        self.telegram = TelegramNotifier(self.telegram_bot_token, self.telegram_chat_id)
        
        # Initialize learning system
        self.learning_system = LearningSystem()
        logger.info("🧠 Learning system initialized")
        
        # Initialize GPT Memory Engine
        from gpt_memory_engine import GPTMemoryEngine
        self.gpt_memory = GPTMemoryEngine()
        logger.info("🧠 GPT Memory Engine initialized")
        
        # Initialize crypto-scan integration
        from crypto_scan_integration import CryptoScanIntegration
        self.crypto_scan_integration = CryptoScanIntegration()
        logger.info("🔗 Crypto-scan integration initialized")
        
        # Initialize heatmap system for orderbook analysis
        try:
            self.heatmap_manager = initialize_heatmap_system()
            logger.info("📊 Heatmap integration system initialized")
        except Exception as e:
            logger.warning(f"Heatmap system initialization failed: {e}")
            self.heatmap_manager = None
        
        # Initialize function history system
        self.function_manager = FunctionHistoryManager()
        self.performance_tracker = PerformanceTracker()
        self.gpt_learning_engine = GPTLearningEngine(self.openai_api_key)
        logger.info("📚 Function history system initialized")
        
        # Initialize OnChain Analyzer
        self.onchain_analyzer = OnChainAnalyzer()
        logger.info("⛓️ OnChain Analyzer initialized")
        
        # Create data directory
        os.makedirs('pump_data', exist_ok=True)
    
    def _get_symbols_from_crypto_scan(self) -> List[str]:
        """Get symbols using crypto-scan logic - works on production server"""
        try:
            # Import crypto-scan's proven data fetchers module
            from utils.data_fetchers import get_symbols_cached, build_bybit_symbol_cache_all_categories
            
            # Try to get symbols from cache first
            symbols = get_symbols_cached(require_chain=False)
            if symbols and len(symbols) > 50:
                logger.info(f"✅ Retrieved {len(symbols)} symbols from crypto-scan cache")
                return symbols
                
        except Exception as e:
            logger.debug(f"Crypto-scan cache method failed: {e}")
        
        # Fallback: try to build cache and retry
        try:
            from utils.data_fetchers import build_bybit_symbol_cache_all_categories
            logger.info("Building fresh symbol cache...")
            build_bybit_symbol_cache_all_categories()
            
            symbols = get_symbols_cached(require_chain=False)
            if symbols and len(symbols) > 50:
                logger.info(f"✅ Built and retrieved {len(symbols)} symbols from fresh cache")
                return symbols
                
        except Exception as e:
            logger.debug(f"Cache building failed: {e}")
        
        # Production fallback for development environment
        comprehensive_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT',
            'XLMUSDT', 'UNIUSDT', 'FILUSDT', 'TRXUSDT', 'ETCUSDT', 'NEARUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT',
            'THETAUSDT', 'HBARUSDT', 'EGLDUSDT', 'AAVEUSDT', 'EOSUSDT', 'AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAUSDT', 'PAWSUSDT'
        ]
        
        logger.warning(f"⚠️ Using development fallback: {len(comprehensive_symbols)} symbols")
        logger.info("On production server, crypto-scan cache will provide 500+ symbols")
        return comprehensive_symbols
        
    def run_analysis(self, days_back: float = 7, max_symbols: int = 999999):
        """
        Run complete pump analysis
        
        Args:
            days_back: Number of days to analyze
            max_symbols: Maximum number of symbols to analyze
        """
        
        logger.info("🚀 Starting Pump Analysis System")
        logger.info(f"📊 Analyzing {days_back} days of data for up to {max_symbols} symbols")
        
        # Get active symbols using proven crypto-scan method (unlimited)
        symbols = self._get_symbols_from_crypto_scan()
        
        if not symbols:
            logger.error("❌ No symbols retrieved from Bybit")
            return
            
        # Limit symbols if max_symbols is specified
        if max_symbols and max_symbols < len(symbols):
            symbols = symbols[:max_symbols]
            logger.info(f"📈 Limited to {len(symbols)} symbols for analysis: {symbols[:10]}...")
        else:
            logger.info(f"📈 Processing all {len(symbols)} symbols: {symbols[:10]}...")
        
        total_pumps_found = 0
        total_analyses_sent = 0
        
        for i, symbol in enumerate(symbols):
            logger.info(f"🔍 Analyzing {symbol} ({i+1}/{len(symbols)})")
            
            try:
                # Get historical data
                end_time = int(datetime.now().timestamp() * 1000)
                start_time = end_time - (days_back * 24 * 60 * 60 * 1000)  # days_back days ago
                
                kline_data = self.bybit.get_kline_data(
                    symbol=symbol,
                    interval="15",
                    start_time=start_time,
                    limit=1000
                )
                
                if not kline_data:
                    logger.warning(f"⚠️ No data for {symbol}")
                    continue
                
                # Detect pumps
                pumps = self.pump_detector.detect_pumps_in_data(kline_data, symbol)
                
                if pumps:
                    logger.info(f"🎯 Found {len(pumps)} pump(s) in {symbol}")
                    total_pumps_found += len(pumps)
                    
                    # 🔥 DODAJ SYMBOL DO HEATMAPY TYLKO DLA WYKRYTYCH PUMPÓW
                    logger.info(f"📊 Adding {symbol} to heatmap analysis (pump detected)")
                    try:
                        from modules.heatmap_integration import get_heatmap_manager
                        heatmap_manager = get_heatmap_manager()
                        if heatmap_manager:
                            heatmap_manager.add_symbol_for_analysis(symbol)
                            logger.info(f"✅ {symbol} added to heatmap collection for pump analysis")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to add {symbol} to heatmap: {e}")
                    
                    # Analyze each pump
                    for pump_idx, pump in enumerate(pumps):
                        logger.info(f"🔍 Processing pump {pump_idx+1}/{len(pumps)} for {symbol}: +{pump.price_increase_pct:.1f}% at {pump.start_time}")
                        
                        try:
                            # Get pre-pump analysis
                            logger.info(f"📊 Analyzing pre-pump conditions for {symbol} pump...")
                            pre_pump_analysis = self.pre_pump_analyzer.analyze_pre_pump_conditions(pump)
                            
                            if pre_pump_analysis:
                                logger.info(f"✅ Pre-pump analysis successful for {symbol}")
                                
                                # 🧠 LEARNING SYSTEM INTEGRATION - Test existing functions on new pump
                                logger.info(f"🧪 Testing existing functions on new pump {symbol}...")
                                learning_test_results = {}
                                try:
                                    # Get pre-pump candle data for testing
                                    pre_pump_candles = self._get_pre_pump_candles_for_testing(pump)
                                    if pre_pump_candles is not None:
                                        learning_test_results = self.learning_system.test_functions_on_new_pump(
                                            {
                                                'symbol': pump.symbol,
                                                'start_time': pump.start_time.isoformat(),
                                                'price_increase_pct': pump.price_increase_pct,
                                                'duration_minutes': pump.duration_minutes
                                            },
                                            pre_pump_candles
                                        )
                                        logger.info(f"📊 Learning test complete: {learning_test_results['functions_tested']} functions tested, {len(learning_test_results['successful_detections'])} detected")
                                except Exception as e:
                                    logger.warning(f"⚠️ Learning system test failed: {e}")
                                
                                # Generate enhanced GPT analysis with memory context
                                logger.info(f"🤖 Generating GPT analysis for {symbol} with memory context...")
                                
                                # Get context from GPT Memory Engine
                                memory_context = self.gpt_memory.generate_context_for_gpt(pre_pump_analysis, {
                                    'symbol': pump.symbol,
                                    'price_increase_pct': pump.price_increase_pct,
                                    'duration_minutes': pump.duration_minutes,
                                    'volume_spike': pump.volume_spike
                                })
                                
                                # Generate strategic analysis with enhanced context and 60-minute window data
                                logger.info(f"🔍 Generating strategic analysis for {symbol} with 60-minute pre-pump window...")
                                
                                # Get 60-minute pre-pump candle data for enhanced GPT analysis
                                pre_pump_candles = self._get_pre_pump_candles_for_testing(pump)
                                
                                # Analyze on-chain activity for descriptive insights
                                logger.info(f"⛓️ Analyzing on-chain activity for {symbol}...")
                                onchain_insights = self.onchain_analyzer.analyze_onchain_activity(symbol, timeframe_hours=1)
                                onchain_messages = self.onchain_analyzer.format_insights_for_gpt(onchain_insights)
                                logger.info(f"⛓️ Generated {len(onchain_messages)} on-chain insights for {symbol}")
                                
                                # Add on-chain insights to pre-pump analysis
                                pre_pump_analysis['onchain_insights'] = onchain_messages
                                
                                # Generate strategic analysis using new dynamic approach with on-chain data
                                gpt_analysis = self.gpt_analyzer.generate_strategic_analysis(pre_pump_analysis, pump, pre_pump_candles)
                                
                                # Generate and store detector function for learning system
                                logger.info(f"🔧 Generating detector function for {symbol}...")
                                detector_function = self.generate_and_store_detector_function(pump, pre_pump_analysis)
                                
                                # 🧠 GPT MEMORY ENGINE INTEGRATION - Register detector with full context
                                logger.info(f"🧠 Registering detector in GPT Memory Engine...")
                                try:
                                    # Check if crypto-scan detected this pump beforehand
                                    crypto_scan_pre_pump_signal = self.crypto_scan_integration.check_symbol_pre_pump_detected(
                                        pump.symbol, 
                                        pump.start_time, 
                                        window_hours=2
                                    )
                                    
                                    # Get additional crypto-scan context
                                    crypto_scan_signals = {
                                        'pre_pump_detected': crypto_scan_pre_pump_signal is not None,
                                        'pre_pump_signal': crypto_scan_pre_pump_signal,
                                        'recent_performance': self.crypto_scan_integration.get_ppwcs_performance_stats(),
                                        'symbol_history': self.crypto_scan_integration.get_symbol_signal_history(pump.symbol, days=7)
                                    }
                                    
                                    # Register strategic analysis instead of rigid function
                                    analysis_id = self.gpt_memory.register_strategic_analysis(
                                        pump.symbol,
                                        pump.start_time.strftime('%Y%m%d'),
                                        gpt_analysis,
                                        {
                                            'symbol': pump.symbol,
                                            'price_increase_pct': pump.price_increase_pct,
                                            'duration_minutes': pump.duration_minutes,
                                            'volume_spike': pump.volume_spike
                                        },
                                        pre_pump_analysis,
                                        crypto_scan_signals,
                                        pre_pump_candles
                                    )
                                    
                                    if crypto_scan_pre_pump_signal:
                                        logger.info(f"✅ Crypto-scan successfully detected this pump beforehand! PPWCS: {crypto_scan_pre_pump_signal.get('ppwcs_score', 0)}")
                                    else:
                                        logger.info(f"⚠️ Crypto-scan missed this pump - opportunity for improvement")
                                    
                                    logger.info(f"✅ Strategic analysis registered in GPT Memory Engine: {analysis_id}")
                                except Exception as e:
                                    logger.error(f"❌ Failed to register detector in GPT Memory Engine: {e}")
                                
                                # 🧠 LEARNING SYSTEM INTEGRATION - Save strategic analysis with metadata
                                logger.info(f"💾 Saving strategic analysis to learning system...")
                                active_signals = self._extract_active_signals(pre_pump_analysis)
                                try:
                                    analysis_path = self.learning_system.save_strategic_analysis(
                                        gpt_analysis,
                                        pump.symbol,
                                        pump.start_time.strftime('%Y%m%d'),
                                        active_signals,
                                        {
                                            'pump_event': {
                                                'price_increase_pct': pump.price_increase_pct,
                                                'duration_minutes': pump.duration_minutes
                                            },
                                            'analysis': pre_pump_analysis,
                                            'candle_data': pre_pump_candles
                                        }
                                    )
                                    logger.info(f"✅ Strategic analysis saved to learning system: {analysis_path}")
                                except Exception as e:
                                    logger.error(f"❌ Failed to save strategic analysis to learning system: {e}")
                                
                                # Strategic analysis mode - no rigid function testing needed
                                test_result = {
                                    "success": True, 
                                    "strategic_analysis": True, 
                                    "insights_generated": True,
                                    "mode": "strategic_analysis"
                                }
                                
                                # Format message for Telegram (including test result and learning results)
                                telegram_message = self._format_telegram_message(pump, gpt_analysis, test_result if test_result else {}, learning_test_results)
                                
                                # Send to Telegram
                                logger.info(f"📤 Sending Telegram message for {symbol}...")
                                if self.telegram.send_message(telegram_message):
                                    total_analyses_sent += 1
                                    logger.info(f"✅ Analysis sent for {symbol} pump at {pump.start_time}")
                                    
                                    # Save analysis to file
                                    self._save_analysis_to_file(pump, pre_pump_analysis, gpt_analysis)
                                else:
                                    logger.error(f"❌ Failed to send analysis for {symbol} - Telegram error")
                                
                                # Small delay to avoid overwhelming Telegram
                                time.sleep(2)
                            else:
                                logger.warning(f"⚠️ PUMP SKIPPED: Pre-pump analysis returned None for {symbol} pump +{pump.price_increase_pct:.1f}%")
                                logger.warning(f"   Pump details: start={pump.start_time}, price_before={pump.price_before:.6f}, price_peak={pump.price_peak:.6f}")
                                
                        except Exception as e:
                            logger.error(f"❌ Error analyzing pump for {symbol}: {e}")
                            logger.error(f"   Pump details: +{pump.price_increase_pct:.1f}% at {pump.start_time}")
                            continue
                
                # Small delay between symbols
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        logger.info(f"🏁 Analysis complete!")
        logger.info(f"📊 Total pumps found: {total_pumps_found}")
        logger.info(f"📤 Total analyses sent: {total_analyses_sent}")
        
        # Send summary to Telegram
        summary_message = f"""
🏁 <b>PUMP ANALYSIS - PODSUMOWANIE</b>

📊 Przeanalizowane symbole: {len(symbols)}
🎯 Znalezione pumpy: {total_pumps_found}
📤 Wysłane analizy: {total_analyses_sent}
⏰ Czas zakończenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ Analiza zakończona pomyślnie!
        """
        
        self.telegram.send_message(summary_message)
    
    def _format_telegram_message(self, pump: PumpEvent, gpt_analysis: str, test_result: dict = None, learning_results: dict = None) -> str:
        """Format message for Telegram"""
        
        # Clean GPT analysis text to prevent HTML parsing errors
        clean_gpt_analysis = self._clean_text_for_telegram(gpt_analysis)
        
        message = f"""
🎯 <b>WYKRYTY PUMP - ANALIZA PRE-PUMP</b>

💰 <b>Symbol:</b> {pump.symbol}
📈 <b>Wzrost:</b> +{pump.price_increase_pct:.1f}%
⏰ <b>Czas pumpu:</b> {pump.start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
💵 <b>Cena przed:</b> ${pump.price_before:.6f}
🚀 <b>Cena szczyt:</b> ${pump.price_peak:.6f}
📊 <b>Volume spike:</b> {pump.volume_spike:.1f}x

<b>📝 ANALIZA GPT (60 min przed pumpem):</b>

{clean_gpt_analysis}

<b>🧪 TEST WYGENEROWANEJ FUNKCJI:</b>
{self._format_test_result(test_result) if test_result else "❌ Test nie został wykonany"}

---
🤖 <i>Automatyczna analiza pump_analysis system</i>
        """
        
        return message
    
    def _clean_text_for_telegram(self, text: str) -> str:
        """Clean text to prevent Telegram HTML parsing errors"""
        if not text:
            return ""
        
        # Remove or escape problematic characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Remove excessive whitespace and normalize
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # Limit length to prevent message too long error
        result = '\n'.join(cleaned_lines)
        if len(result) > 3000:
            result = result[:3000] + "...\n[TEKST SKRÓCONY]"
        
        return result
    
    def _clean_detector_function_code(self, code: str) -> str:
        """Clean detector function code to prevent syntax errors"""
        import re
        
        # Remove markdown code blocks if present
        code = re.sub(r'```python\s*\n?', '', code)
        code = re.sub(r'```\s*$', '', code)
        
        # Fix common decimal literal issues
        # Replace numbers like .5 with 0.5
        code = re.sub(r'(?<!\d)\.(\d+)', r'0.\1', code)
        
        # Fix scientific notation issues (e.g., 1.5e-3)
        code = re.sub(r'(\d+\.?\d*)e([+-]?\d+)', r'\1e\2', code)
        
        # Remove any non-ASCII characters that might cause issues
        code = ''.join(char for char in code if ord(char) < 128)
        
        # Fix common formatting issues
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove any trailing whitespace
            line = line.rstrip()
            
            # Skip empty lines at the beginning
            if not cleaned_lines and not line.strip():
                continue
                
            # Fix indentation issues
            if line.strip() and not line.startswith(' ') and not line.startswith('\t') and line.strip().startswith('def '):
                # Function definition should start at column 0
                pass
            elif line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.strip().startswith('#'):
                # Other lines should be indented if they're inside a function
                if cleaned_lines and any('def ' in prev_line for prev_line in cleaned_lines):
                    line = '    ' + line.strip()
            
            cleaned_lines.append(line)
        
        # Ensure proper function structure
        result = '\n'.join(cleaned_lines)
        
        # Validate basic Python syntax
        try:
            compile(result, '<string>', 'exec')
            return result
        except SyntaxError as e:
            # If there's still a syntax error, try to fix common issues
            logger.warning(f"Syntax error in generated function, attempting to fix: {e}")
            
            # Try to fix the specific error mentioned by user (line 62 decimal literal)
            lines = result.split('\n')
            for i, line in enumerate(lines):
                # Look for problematic decimal literals
                if re.search(r'\d+\.\d*[a-zA-Z]', line):
                    # Remove any letters after decimal numbers
                    lines[i] = re.sub(r'(\d+\.\d*)[a-zA-Z]+', r'\1', line)
                    logger.info(f"Fixed decimal literal in line {i+1}: {line} -> {lines[i]}")
            
            result = '\n'.join(lines)
            
            # Try to compile again
            try:
                compile(result, '<string>', 'exec')
                return result
            except SyntaxError:
                # If still failing, return a simple fallback function
                logger.error("Failed to fix generated function, using fallback")
                return """
def fallback_detector(df):
    \"\"\"Fallback detector function\"\"\"
    return False
"""
    
    def _save_detector_function(self, pump: PumpEvent, detector_function: str):
        """Save generated detector function to Python file"""
        
        # Extract date from pump event  
        pump_date = pump.start_time.strftime("%Y%m%d")
        symbol = pump.symbol
        
        # Create filename: SYMBOL_YYYYMMDD.py
        filename = f"generated_detectors/{symbol}_{pump_date}.py"
        
        # Clean up function code (remove markdown if present)
        function_code = detector_function
        if "```python" in function_code:
            # Extract code between ```python and ```
            start = function_code.find("```python") + 9
            end = function_code.find("```", start)
            if end != -1:
                function_code = function_code[start:end].strip()
        elif "```" in function_code:
            # Extract code between ``` and ```
            start = function_code.find("```") + 3
            end = function_code.find("```", start)
            if end != -1:
                function_code = function_code[start:end].strip()
        
        # Add header comment
        header = f'''"""
Generated Detector Function - {symbol} {pump_date}
Auto-generated by Pump Analysis System based on real pump data

Pump details:
- Symbol: {symbol}
- Date: {pump.start_time.strftime("%Y-%m-%d %H:%M:%S")}
- Price increase: +{pump.price_increase_pct:.1f}%
- Duration: {pump.duration_minutes} minutes

This function detects pre-pump conditions that preceded the actual pump event.
"""

import pandas as pd
import numpy as np

'''
        
        full_content = header + function_code
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(full_content)
            logger.info(f"🐍 Detector function saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Error saving detector function: {e}")

    def _save_analysis_to_file(self, pump: PumpEvent, pre_pump_data: Dict, gpt_analysis: str):
        """Save analysis to JSON file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pump_data/{pump.symbol}_{timestamp}_pump_analysis.json"
        
        analysis_data = {
            'pump_event': {
                'symbol': pump.symbol,
                'start_time': pump.start_time.isoformat(),
                'price_before': pump.price_before,
                'price_peak': pump.price_peak,
                'price_increase_pct': pump.price_increase_pct,
                'duration_minutes': pump.duration_minutes,
                'volume_spike': pump.volume_spike
            },
            'pre_pump_analysis': pre_pump_data,
            'gpt_analysis': gpt_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"💾 Analysis saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Error saving analysis to file: {e}")
    
    def generate_and_store_detector_function(self, pump_event: PumpEvent, pre_pump_data: Dict) -> str:
        """Generate and store detector function using function history system"""
        try:
            # Generate detector function using GPT Learning Engine
            detector_function = self.gpt_learning_engine.generate_detector_function(
                pre_pump_data, pump_event
            )
            
            # Create function metadata
            metadata = FunctionMetadata(
                symbol=pump_event.symbol,
                date=pump_event.start_time.strftime('%Y%m%d'),
                pump_increase=pump_event.price_increase_pct,
                generation_time=datetime.now(),
                active_signals=self._extract_active_signals(pre_pump_data),
                pre_pump_analysis=pre_pump_data
            )
            
            # Store function in function manager
            function_id = self.function_manager.store_function(
                detector_function, metadata
            )
            
            # Log generation in performance tracker
            self.performance_tracker.log_generation(
                function_id, pump_event.symbol, pump_event.start_time.strftime('%Y%m%d'), 
                detector_function, pump_event.price_increase_pct
            )
            
            logger.info(f"✅ Generated and stored detector function: {function_id}")
            return detector_function
            
        except Exception as e:
            logger.error(f"❌ Error generating detector function: {e}")
            return f"def detect_{pump_event.symbol.lower()}_{pump_event.start_time.strftime('%Y%m%d')}_preconditions(df): return False, 0.0, ['error']"

    def _test_detector_function(self, pump: PumpEvent, pre_pump_data: Dict, detector_function: str) -> dict:
        """Test the generated detector function on pre-pump data"""
        
        try:
            # Extract function name from the detector code
            import re
            function_match = re.search(r'def\s+(\w+)\s*\(', detector_function)
            if not function_match:
                return {
                    'success': False,
                    'error': 'Nie znaleziono nazwy funkcji w kodzie',
                    'detected': False
                }
            
            function_name = function_match.group(1)
            
            # Create a safe execution environment
            namespace = {
                'pd': pd,
                'np': np,
                '__builtins__': {},
                'len': len,
                'min': min,
                'max': max,
                'abs': abs,
                'sum': sum,
                'any': any,
                'all': all
            }
            
            # Clean and validate the detector function code before execution
            cleaned_function = self._clean_detector_function_code(detector_function)
            
            # Execute the detector function
            exec(cleaned_function, namespace)
            
            # Create test DataFrame from pre-pump data
            if 'kline_data' in pre_pump_data:
                test_df = pd.DataFrame(pre_pump_data['kline_data'])
            else:
                # Create minimal test data based on available metrics
                test_data = []
                for i in range(4):  # 4 x 15min = 60 minutes
                    test_data.append({
                        'open': 1.0,
                        'high': 1.02,
                        'low': 0.98,
                        'close': 1.01,
                        'volume': 1000,
                        'timestamp': int(time.time()) - (4-i)*900
                    })
                test_df = pd.DataFrame(test_data)
            
            # Test the function
            detector_func = namespace[function_name]
            result = detector_func(test_df)
            
            # Determine if detection was successful - handle multiple return formats
            if isinstance(result, bool):
                detected = result
            elif isinstance(result, (int, float)):
                detected = bool(result)
            elif isinstance(result, dict):
                # Check multiple possible keys for detection status
                detected = (result.get('signal_detected', False) or 
                           result.get('detected', False) or 
                           result.get('pump_detected', False) or
                           bool(result.get('confidence', 0) > 0.5))
            elif isinstance(result, (list, tuple)) and len(result) >= 1:
                # Handle tuple returns like (detected, confidence, signals)
                detected = bool(result[0])
            else:
                detected = bool(result)
            
            return {
                'success': True,
                'detected': detected,
                'function_name': function_name,
                'result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'detected': False
            }
    
    def _format_test_result(self, test_result: dict) -> str:
        """Format test result for display"""
        
        # Handle strategic analysis mode
        if test_result and test_result.get('mode') == 'strategic_analysis':
            if test_result.get('strategic_analysis') and test_result.get('insights_generated'):
                return "✅ <b>ANALIZA STRATEGICZNA:</b> Wygenerowano szczegółowe wzorce i rekomendacje pre-pump"
            else:
                return "⚠️ <b>ANALIZA STRATEGICZNA:</b> Częściowo ukończona"
        
        # Handle traditional function testing
        if not test_result or not test_result.get('success', False):
            error = test_result.get('error', 'Nieznany błąd') if test_result else 'Brak wyniku testu'
            return f"❌ <b>BŁĄD TESTU:</b> {error}"
        
        detected = test_result.get('detected', False)
        function_name = test_result.get('function_name', 'unknown')
        
        if detected:
            return f"✅ <b>SUKCES:</b> Funkcja <code>{function_name}()</code> wykryła pre-pump warunki!"
        else:
            return f"⚠️ <b>UWAGA:</b> Funkcja <code>{function_name}()</code> NIE wykryła pre-pump warunków"
    
    def _get_pre_pump_candles_for_testing(self, pump: PumpEvent) -> Optional[pd.DataFrame]:
        """Get pre-pump candle data for testing existing functions"""
        try:
            # Get data 2 hours before pump for comprehensive testing
            start_time = int((pump.start_time - timedelta(hours=2)).timestamp() * 1000)
            
            kline_data = self.bybit.get_kline_data(
                symbol=pump.symbol,
                interval="15",
                start_time=start_time,
                limit=1000
            )
            
            if not kline_data:
                return None
                
            # Convert to DataFrame for function testing
            df_data = []
            for kline in kline_data:
                if isinstance(kline, dict):
                    timestamp = int(kline['0'])  # timestamp
                    open_price = float(kline['1'])  # open
                    high_price = float(kline['2'])  # high
                    low_price = float(kline['3'])   # low
                    close_price = float(kline['4']) # close
                    volume = float(kline['5'])      # volume
                    
                    df_data.append({
                        'timestamp': pd.to_datetime(timestamp, unit='ms'),
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume
                    })
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                return df
                
        except Exception as e:
            logger.warning(f"Failed to get pre-pump candles for {pump.symbol}: {e}")
            
        return None
    
    def _extract_active_signals(self, pre_pump_analysis: Dict) -> List[str]:
        """Extract active signals from pre-pump analysis for metadata"""
        signals = []
        
        try:
            # Extract signals from different analysis sections
            if 'volume_spikes' in pre_pump_analysis and pre_pump_analysis['volume_spikes']:
                signals.append('volume_spike')
                
            if 'compression' in pre_pump_analysis:
                compression = pre_pump_analysis['compression']
                if compression.get('detected', False):
                    signals.append('compression')
                    
            if 'fake_rejects' in pre_pump_analysis and pre_pump_analysis['fake_rejects']:
                signals.append('fake_reject')
                
            if 'support_resistance' in pre_pump_analysis:
                sr = pre_pump_analysis['support_resistance']
                if sr.get('key_support') or sr.get('key_resistance'):
                    signals.append('support_resistance')
                    
            if 'trend' in pre_pump_analysis:
                signals.append(f"trend_{pre_pump_analysis['trend']}")
                
            if 'vwap_analysis' in pre_pump_analysis:
                vwap = pre_pump_analysis['vwap_analysis']
                if vwap.get('position') == 'above':
                    signals.append('above_vwap')
                elif vwap.get('position') == 'below':
                    signals.append('below_vwap')
                    
            if 'rsi' in pre_pump_analysis:
                rsi = pre_pump_analysis['rsi']
                if rsi < 30:
                    signals.append('rsi_oversold')
                elif rsi > 70:
                    signals.append('rsi_overbought')
                elif 45 <= rsi <= 55:
                    signals.append('rsi_neutral')
                    
            if 'liquidity_gaps' in pre_pump_analysis and pre_pump_analysis['liquidity_gaps']:
                signals.append('liquidity_gaps')
                
        except Exception as e:
            logger.warning(f"Error extracting signals: {e}")
            
        return signals if signals else ['unknown']

def main():
    """Main function to run pump analysis"""
    
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        # Initialize and run analysis system
        system = PumpAnalysisSystem()
        
        # Run analysis for last 7 days, unlimited symbols
        system.run_analysis(days_back=7)
        
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        raise

if __name__ == "__main__":
    main()