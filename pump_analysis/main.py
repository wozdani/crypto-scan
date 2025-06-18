#!/usr/bin/env python3
"""
Pump Analysis System - Automatic detection and analysis of crypto pumps
Analyzes historical pump data and generates GPT insights for learning purposes
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        
    def get_kline_data(self, symbol: str, interval: str = "5", start_time: int = None, limit: int = 200) -> List[Dict]:
        """
        Fetch kline data from Bybit
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            start_time: Start timestamp in milliseconds
            limit: Number of data points (max 1000)
        """
        endpoint = f"{self.base_url}/v5/market/kline"
        
        params = {
            'category': 'spot',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['start'] = start_time
            
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['retCode'] == 0:
                return data['result']['list']
            else:
                logger.error(f"Bybit API error for {symbol}: {data['retMsg']}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return []
    
    def get_active_symbols(self, limit: int = 50) -> List[str]:
        """Get list of active trading symbols sorted by volume"""
        # Try multiple endpoints for better reliability
        endpoints = [
            f"{self.base_url}/v5/market/tickers",
            f"{self.base_url}/v2/public/tickers"
        ]
        
        # Fallback list of popular symbols if API fails
        fallback_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT',
            'XLMUSDT', 'UNIUSDT', 'FILUSDT', 'TRXUSDT', 'ETCUSDT',
            'NEARUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT',
            'THETAUSDT', 'XLMUSDT', 'HBARUSDT', 'EGLDUSDT', 'AAVEUSDT',
            'EOSUSDT', 'AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAUSDT'
        ]
        
        for endpoint in endpoints:
            try:
                if 'v5' in endpoint:
                    params = {'category': 'spot'}
                else:
                    params = {}
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json'
                }
                
                response = requests.get(endpoint, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Handle v5 API response
                    if 'result' in data and data.get('retCode') == 0:
                        tickers = data['result']['list']
                        usdt_pairs = [
                            ticker for ticker in tickers 
                            if ticker['symbol'].endswith('USDT') and float(ticker.get('volume24h', 0)) > 100000
                        ]
                        usdt_pairs.sort(key=lambda x: float(x.get('volume24h', 0)), reverse=True)
                        symbols = [ticker['symbol'] for ticker in usdt_pairs[:limit]]
                        if symbols:
                            logger.info(f"Retrieved {len(symbols)} symbols from Bybit v5 API")
                            return symbols
                    
                    # Handle v2 API response
                    elif 'result' in data and isinstance(data['result'], list):
                        tickers = data['result']
                        usdt_pairs = [
                            ticker for ticker in tickers 
                            if ticker['symbol'].endswith('USDT') and float(ticker.get('volume_24h', 0)) > 100000
                        ]
                        usdt_pairs.sort(key=lambda x: float(x.get('volume_24h', 0)), reverse=True)
                        symbols = [ticker['symbol'] for ticker in usdt_pairs[:limit]]
                        if symbols:
                            logger.info(f"Retrieved {len(symbols)} symbols from Bybit v2 API")
                            return symbols
                            
            except Exception as e:
                logger.warning(f"Endpoint {endpoint} failed: {e}")
                continue
        
        # If all API calls fail, return fallback symbols
        logger.warning("All Bybit API endpoints failed, using fallback symbol list")
        return fallback_symbols[:limit]

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
        if len(kline_data) < 12:  # Need at least 1 hour of 5-min data
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
        window_size = self.detection_window_minutes // 5  # 5-min candles
        
        for i in range(window_size, len(df)):
            window_start = i - window_size
            window_data = df.iloc[window_start:i+1]
            
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
                
                # Skip overlapping windows
                i += window_size
                
        return pumps

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
        
        # Get 1-hour of data before pump (12 x 5-min candles)
        pre_pump_data = self.bybit.get_kline_data(
            symbol=pump_event.symbol,
            interval="5",
            start_time=start_timestamp,
            limit=12
        )
        
        if not pre_pump_data:
            return {}
            
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
        if first_half == 0:
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
    """Handles GPT analysis of pre-pump conditions"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
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
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"GPT analysis error: {e}")
            return f"Błąd podczas generowania analizy GPT: {e}"
    
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
            support_text = f"{data['support_resistance']['key_support']:.6f}" if data['support_resistance']['key_support'] else 'brak'
            resistance_text = f"{data['support_resistance']['key_resistance']:.6f}" if data['support_resistance']['key_resistance'] else 'brak'
            prompt += f"""
• Kluczowe poziomy:
  - Wsparcie: {support_text}
  - Opór: {resistance_text}
"""

        if data['liquidity_gaps']:
            prompt += f"\n• Luki płynnościowe: {len(data['liquidity_gaps'])}\n"
            for gap in data['liquidity_gaps']:
                prompt += f"  - {gap['type']}: {gap['size_pct']:.2f}% ({gap['time_minutes_before_pump']} min przed)\n"

        prompt += """

=== ZADANIE ANALIZY ===
Na podstawie powyższych danych z 60 minut przed pumpem, przeanalizuj:

1. Jakie były najważniejsze sygnały ostrzegawcze?
2. Czy struktura rynku wskazywała na przygotowania do ruchu?
3. Które wskaźniki były najbardziej predykcyjne?
4. Jakie wzorce akumulacji można zidentyfikować?
5. Co można było zauważyć wcześniej, żeby przewidzieć pump?

Podaj konkretną, praktyczną analizę z naciskiem na aplikowalność w przyszłych sytuacjach.
"""

        return prompt

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
        
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        try:
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
            return False

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
        
        # Create data directory
        os.makedirs('pump_data', exist_ok=True)
        
    def run_analysis(self, days_back: int = 7, max_symbols: int = 30):
        """
        Run complete pump analysis
        
        Args:
            days_back: Number of days to analyze
            max_symbols: Maximum number of symbols to analyze
        """
        
        logger.info("🚀 Starting Pump Analysis System")
        logger.info(f"📊 Analyzing {days_back} days of data for up to {max_symbols} symbols")
        
        # Get active symbols
        symbols = self.bybit.get_active_symbols(limit=max_symbols)
        
        if not symbols:
            logger.error("❌ No symbols retrieved from Bybit")
            return
            
        logger.info(f"📈 Retrieved {len(symbols)} symbols: {symbols[:10]}...")
        
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
                    interval="5",
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
                    
                    # Analyze each pump
                    for pump in pumps:
                        try:
                            # Get pre-pump analysis
                            pre_pump_analysis = self.pre_pump_analyzer.analyze_pre_pump_conditions(pump)
                            
                            if pre_pump_analysis:
                                # Generate GPT analysis
                                gpt_analysis = self.gpt_analyzer.generate_pump_analysis(pre_pump_analysis)
                                
                                # Format message for Telegram
                                telegram_message = self._format_telegram_message(pump, gpt_analysis)
                                
                                # Send to Telegram
                                if self.telegram.send_message(telegram_message):
                                    total_analyses_sent += 1
                                    logger.info(f"✅ Analysis sent for {symbol} pump at {pump.start_time}")
                                    
                                    # Save analysis to file
                                    self._save_analysis_to_file(pump, pre_pump_analysis, gpt_analysis)
                                else:
                                    logger.error(f"❌ Failed to send analysis for {symbol}")
                                
                                # Small delay to avoid overwhelming Telegram
                                time.sleep(2)
                                
                        except Exception as e:
                            logger.error(f"❌ Error analyzing pump for {symbol}: {e}")
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
    
    def _format_telegram_message(self, pump: PumpEvent, gpt_analysis: str) -> str:
        """Format message for Telegram"""
        
        message = f"""
🎯 <b>WYKRYTY PUMP - ANALIZA PRE-PUMP</b>

💰 <b>Symbol:</b> {pump.symbol}
📈 <b>Wzrost:</b> +{pump.price_increase_pct:.1f}%
⏰ <b>Czas pumpu:</b> {pump.start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
💵 <b>Cena przed:</b> ${pump.price_before:.6f}
🚀 <b>Cena szczyt:</b> ${pump.price_peak:.6f}
📊 <b>Volume spike:</b> {pump.volume_spike:.1f}x

<b>📝 ANALIZA GPT (60 min przed pumpem):</b>

{gpt_analysis}

---
🤖 <i>Automatyczna analiza pump_analysis system</i>
        """
        
        return message
    
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

def main():
    """Main function to run pump analysis"""
    
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        # Initialize and run analysis system
        system = PumpAnalysisSystem()
        
        # Run analysis for last 7 days, maximum 30 symbols
        system.run_analysis(days_back=7, max_symbols=30)
        
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        raise

if __name__ == "__main__":
    main()