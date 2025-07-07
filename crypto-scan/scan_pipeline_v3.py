#!/usr/bin/env python3
"""
TJDE v3 Pipeline - Fixed Architecture
Eliminates AI-EYE chart dependency circular logic

NEW ARCHITECTURE:
1. BASIC SCORING - Fast analysis for all tokens (~600)
2. TOP N Selection - Select best tokens (score > 0.35 or top 20)
3. CHART CAPTURE - Generate TradingView screenshots for selected tokens
4. CLIP INFERENCE - Run CLIP model on captured charts
5. ADVANCED MODULES - AI-EYE with real CLIP data, HTF, Trap, Future

This fixes the fundamental flaw where AI-EYE tried to analyze non-existent charts.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import aiohttp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import basic engine for phase 1
from trader_ai_engine_basic import simulate_trader_decision_basic
from utils.async_data_processor import process_async_data_enhanced_with_5m

# Import chart generation (simplified for testing)
try:
    from utils.tradingview_only_pipeline import generate_chart_async_safe
except ImportError:
    print("[IMPORT WARNING] TradingView pipeline not available, using fallback")
    def generate_chart_async_safe(*args, **kwargs):
        return None

# Import CLIP processing (simplified for testing)
try:
    from vision.ai_label_pipeline import prepare_ai_label
except ImportError:
    print("[IMPORT WARNING] CLIP pipeline not available, using fallback")
    def prepare_ai_label(*args, **kwargs):
        return {'label': 'unknown', 'confidence': 0.0, 'method': 'fallback'}

# Import unified engine for phase 2
from unified_scoring_engine import simulate_trader_decision_advanced, prepare_unified_data

# Import Dynamic Token Selector for Stage 1.5
from utils.dynamic_token_selector import DynamicTokenSelector


class TJDEv3Pipeline:
    """
    Two-phase TJDE pipeline that fixes AI-EYE circular dependency
    """
    
    def __init__(self):
        self.clip_predictions_dir = "training_data/clip_predictions"
        self.dynamic_selector = DynamicTokenSelector()
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.clip_predictions_dir, exist_ok=True)
        os.makedirs("training_data/charts", exist_ok=True)
        
    def generate_htf_candles_from_15m(self, candles_15m: List) -> List:
        """
        Generate Higher Time Frame (1H) candles from 15M candles
        
        Args:
            candles_15m: List of 15-minute candles
            
        Returns:
            List of 1-hour candles
        """
        try:
            if not candles_15m or len(candles_15m) < 4:
                return []
            
            htf_candles = []
            
            # Group 15M candles into 1H periods (4 x 15M = 1H)
            for i in range(0, len(candles_15m) - 3, 4):
                chunk = candles_15m[i:i+4]
                
                if len(chunk) == 4:
                    # Extract OHLC data from 4x15M candles
                    opens = [c[1] if isinstance(c, list) else c.get('open', 0) for c in chunk]
                    highs = [c[2] if isinstance(c, list) else c.get('high', 0) for c in chunk]
                    lows = [c[3] if isinstance(c, list) else c.get('low', 0) for c in chunk]
                    closes = [c[4] if isinstance(c, list) else c.get('close', 0) for c in chunk]
                    volumes = [c[5] if isinstance(c, list) else c.get('volume', 0) for c in chunk]
                    
                    # Create 1H candle: Open of first, High of max, Low of min, Close of last
                    htf_candle = {
                        'timestamp': chunk[0][0] if isinstance(chunk[0], list) else chunk[0].get('timestamp', 0),
                        'open': opens[0],
                        'high': max(highs),
                        'low': min(lows),
                        'close': closes[-1],
                        'volume': sum(volumes),
                        'timeframe': '1H'
                    }
                    
                    htf_candles.append(htf_candle)
            
            return htf_candles
            
        except Exception as e:
            print(f"[HTF GENERATION ERROR] {e}")
            return []
        
    async def fetch_real_market_data(self, symbol: str, session: aiohttp.ClientSession = None) -> Optional[Dict]:
        """
        Fetch real market data from Bybit API for TJDE v3 Phase 1
        
        Args:
            symbol: Trading symbol
            session: Optional aiohttp session
            
        Returns:
            Market data dictionary with candles, volume, price data
        """
        should_close_session = session is None
        if session is None:
            session = aiohttp.ClientSession()
            
        try:
            # Fetch ticker data for basic market info
            ticker_url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
            async with session.get(ticker_url) as response:
                if response.status == 200:
                    ticker_data = await response.json()
                    if ticker_data.get('result', {}).get('list'):
                        ticker = ticker_data['result']['list'][0]
                        price = float(ticker.get('lastPrice', 0))
                        volume_24h = float(ticker.get('volume24h', 0))
                        price_change_24h = float(ticker.get('price24hPcnt', 0))
                        
                        # Skip invalid data
                        if price <= 0 or volume_24h <= 0:
                            return None
                    else:
                        return None
                else:
                    return None
            
            # Fetch 15M candles for basic scoring
            candles_url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval=15&limit=200"
            candles_15m = []
            
            async with session.get(candles_url) as response:
                if response.status == 200:
                    candle_data = await response.json()
                    if candle_data.get('result', {}).get('list'):
                        raw_candles = candle_data['result']['list']
                        # Convert to expected format [timestamp, open, high, low, close, volume]
                        for candle in raw_candles:
                            candles_15m.append([
                                int(candle[0]),  # timestamp
                                float(candle[1]),  # open
                                float(candle[2]),  # high
                                float(candle[3]),  # low
                                float(candle[4]),  # close
                                float(candle[5])   # volume
                            ])
                        # Reverse to get chronological order
                        candles_15m.reverse()
                        
            # Return market data structure
            return {
                'symbol': symbol,
                'price': price,
                'volume_24h': volume_24h,
                'price_change_24h': price_change_24h,
                'candles_15m': candles_15m,
                'candles_5m': [],  # Will be added if needed
                'data_source': 'bybit_direct'
            }
            
        except Exception as e:
            print(f"[REAL DATA ERROR] {symbol}: {e}")
            return None
            
        finally:
            if should_close_session and session:
                await session.close()
        
    async def phase1_basic_scoring(self, symbols: List[str], priority_info: Dict = None) -> List[Dict]:
        """
        PHASE 1: Fast basic scoring for all tokens
        
        Args:
            symbols: List of token symbols to analyze
            priority_info: Priority information for tokens
            
        Returns:
            List of basic scoring results with scores
        """
        print(f"[PHASE 1] Starting basic scoring for {len(symbols)} tokens")
        start_time = time.time()
        
        results = []
        
        # ASYNC BATCH PROCESSING - Use existing AsyncCryptoScanner
        from async_scanner import AsyncCryptoScanner
        
        scanner = AsyncCryptoScanner(max_concurrent=120)
        
        async with scanner:
            # Batch fetch all market data using async scanner
            print(f"[PHASE 1 ASYNC] Starting parallel data fetch for {len(symbols)} tokens")
            scan_results = await scanner.scan_all_tokens(symbols, priority_info)
            
            print(f"[PHASE 1 ASYNC] Fetched data for {len(scan_results)} tokens")
            
            # Process results with basic scoring
            for result in scan_results:
                try:
                    symbol = result.get('symbol')
                    if not symbol:
                        continue
                    
                    # Convert async scan result to market_data format
                    market_data = {
                        'symbol': symbol,
                        'price': result.get('price', 0),
                        'volume_24h': result.get('volume_24h', 0),
                        'price_change_24h': result.get('price_change_24h', 0),
                        'candles_15m': result.get('candles_15m', []),
                        'candles_5m': result.get('candles_5m', []),
                        'data_source': 'async_scanner'
                    }
                    
                    # Run basic scoring (no AI-EYE dependency)
                    basic_result = await self.run_basic_scoring(symbol, market_data)
                    
                    if basic_result and basic_result.get('score', 0) > 0:
                        results.append({
                            'symbol': symbol,
                            'basic_score': basic_result['score'],
                            'basic_decision': basic_result['decision'],
                            'market_data': market_data,
                            'basic_breakdown': basic_result.get('breakdown', {}),
                            'ai_label': {'label': 'unknown', 'confidence': 0.0}  # Default for Phase 2
                        })
                        
                except Exception as e:
                    print(f"[PHASE 1 ERROR] {result.get('symbol', 'unknown')}: {e}")
                    continue
                
        # Sort by score
        results.sort(key=lambda x: x['basic_score'], reverse=True)
        
        elapsed = time.time() - start_time
        print(f"[PHASE 1] Completed in {elapsed:.1f}s, found {len(results)} viable tokens")
        
        return results
        
    async def run_basic_scoring(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Run basic scoring without AI-EYE dependency"""
        try:
            # 🎯 ENHANCED CANDLE VALIDATION - Ensure sufficient candle history
            candles_15m = market_data.get('candles_15m', [])
            candles_5m = market_data.get('candles_5m', [])
            
            # Minimum candle requirements
            MIN_15M_CANDLES = 20  # ~5 hours of data
            MIN_5M_CANDLES = 60   # ~5 hours of data
            
            if not market_data:
                print(f"[BASIC SCORING] {symbol}: No market data available")
                return None
                
            if len(candles_15m) < MIN_15M_CANDLES:
                print(f"[CANDLE SKIP] {symbol}: Insufficient 15M candles ({len(candles_15m)}/{MIN_15M_CANDLES}) - skipping basic scoring")
                return None
                
            if len(candles_5m) < MIN_5M_CANDLES:
                print(f"[CANDLE SKIP] {symbol}: Insufficient 5M candles ({len(candles_5m)}/{MIN_5M_CANDLES}) - skipping basic scoring")
                return None
                
            print(f"[CANDLE VALID] {symbol}: Basic scoring validation passed (15M: {len(candles_15m)}, 5M: {len(candles_5m)})")
            
            # Use basic engine - no AI-EYE, no HTF dependency
            result = simulate_trader_decision_basic(
                symbol=symbol,
                candles_15m=market_data.get('candles_15m', []),
                candles_5m=market_data.get('candles_5m', []),
                volume_24h=market_data.get('volume_24h', 0),
                price_change_24h=market_data.get('price_change_24h', 0),
                current_price=market_data.get('price', 0)
            )
            
            # Debug basic scoring result
            score = result.get('score', 0) if result else 0
            decision = result.get('decision', 'unknown') if result else 'none'
            breakdown = result.get('breakdown', {}) if result else {}
            
            print(f"[BASIC SCORING DEBUG] {symbol}: Score {score:.3f}, Decision: {decision}")
            if breakdown:
                trend = breakdown.get('trend', 0)
                volume = breakdown.get('volume', 0) 
                momentum = breakdown.get('momentum', 0)
                print(f"[BASIC BREAKDOWN] {symbol}: Trend={trend:.3f}, Volume={volume:.3f}, Momentum={momentum:.3f}")
            
            # Accept any positive score for testing
            if result and score > 0:
                return result
            else:
                print(f"[BASIC SCORING] {symbol}: Rejected - score too low")
                return None
            
        except Exception as e:
            print(f"[BASIC SCORING ERROR] {symbol}: {e}")
            return None
            
    def select_top_tokens(self, basic_results: List[Dict], 
                         top_n: int = 20, min_score: float = 0.15) -> List[Dict]:
        """
        🎯 STAGE 1.5 - TOP 20 Token Selector
        
        Selects TOP 20 tokens based on basic_score for Phase 2 (chart generation + AI-EYE analysis)
        
        Args:
            basic_results: Results from phase 1 (simulate_trader_decision_basic)
            top_n: Number of top tokens to select (default: 20)
            min_score: Minimum score threshold (fallback safety)
            
        Returns:
            TOP 20 tokens sorted by basic_score for advanced analysis
        """
        print(f"[STAGE 1.5] 🎯 TOP {top_n} Token Selector - Processing {len(basic_results)} basic results")
        
        # Extract basic scores and sort
        scored_tokens = []
        for result in basic_results:
            score = result.get('score', 0) or result.get('basic_score', 0)
            if score > min_score:  # Only consider tokens above minimum threshold
                scored_tokens.append({
                    'symbol': result['symbol'],
                    'basic_score': score,
                    'market_data': result['market_data'],
                    'decision': result.get('decision', 'unknown'),
                    'breakdown': result.get('breakdown', {})
                })
        
        # Sort by basic_score (descending) and select TOP N
        sorted_tokens = sorted(scored_tokens, key=lambda x: x['basic_score'], reverse=True)
        selected_tokens = sorted_tokens[:top_n]
        
        print(f"[STAGE 1.5] ✅ TOP {top_n} selection complete:")
        for i, token in enumerate(selected_tokens[:5], 1):  # Show top 5
            print(f"  {i}. {token['symbol']}: {token['basic_score']:.3f}")
        
        if len(selected_tokens) == top_n and len(sorted_tokens) > top_n:
            cutoff_score = selected_tokens[-1]['basic_score']
            print(f"[STAGE 1.5] Selection cutoff: {cutoff_score:.3f} (excluded {len(sorted_tokens) - top_n} tokens)")
        
        return selected_tokens
        
    async def phase3_chart_capture(self, selected_tokens: List[Dict]) -> List[Dict]:
        """
        PHASE 3: Capture TradingView charts for selected tokens
        
        Args:
            selected_tokens: Tokens selected for advanced analysis
            
        Returns:
            Tokens with chart paths added
        """
        print(f"[PHASE 3] Capturing charts for {len(selected_tokens)} tokens")
        start_time = time.time()
        
        enhanced_tokens = []
        
        for token_data in selected_tokens:
            symbol = token_data['symbol']
            score = token_data['basic_score']
            
            try:
                # Generate chart
                chart_path = await self.capture_single_chart(symbol, score)
                
                if chart_path and os.path.exists(chart_path):
                    token_data['chart_path'] = chart_path
                    token_data['chart_captured'] = True
                    print(f"[CHART OK] {symbol}: {os.path.basename(chart_path)}")
                else:
                    token_data['chart_captured'] = False
                    print(f"[CHART FAIL] {symbol}: Chart capture failed")
                
                # CRITICAL: Generate HTF candles as required by logic
                candles_15m = token_data['market_data'].get('candles_15m', [])
                if candles_15m and len(candles_15m) >= 4:  # Need at least 4x15M for 1x1H
                    htf_candles = self.generate_htf_candles_from_15m(candles_15m)
                    token_data['htf_candles'] = htf_candles
                    print(f"[HTF OK] {symbol}: Generated {len(htf_candles)} HTF candles")
                else:
                    token_data['htf_candles'] = []
                    print(f"[HTF SKIP] {symbol}: Insufficient 15M data for HTF generation")
                    
                enhanced_tokens.append(token_data)
                
            except Exception as e:
                print(f"[CHART ERROR] {symbol}: {e}")
                token_data['chart_captured'] = False
                enhanced_tokens.append(token_data)
                
        elapsed = time.time() - start_time
        successful = len([t for t in enhanced_tokens if t.get('chart_captured', False)])
        print(f"[PHASE 3] Completed in {elapsed:.1f}s, {successful}/{len(enhanced_tokens)} charts captured")
        
        return enhanced_tokens
        
    async def capture_single_chart(self, symbol: str, score: float) -> Optional[str]:
        """Capture single chart using TradingView"""
        try:
            # INTEGRATION WITH REAL TRADINGVIEW SYSTEM
            from utils.tradingview_robust import RobustTradingViewGenerator
            
            print(f"[CHART CAPTURE] {symbol}: Initiating TradingView screenshot...")
            
            # Initialize TradingView generator and capture chart
            tv_generator = RobustTradingViewGenerator()
            chart_path = await tv_generator.generate_chart_async(
                symbol=symbol,
                score=score,
                decision='consider',
                priority='high'  # High priority for TOP 20 tokens
            )
            
            if chart_path and os.path.exists(chart_path):
                print(f"[CHART SUCCESS] {symbol}: {os.path.basename(chart_path)}")
                return chart_path
            else:
                print(f"[CHART FAILED] {symbol}: TradingView generation failed")
                return None
            
        except Exception as e:
            print(f"[CHART CAPTURE ERROR] {symbol}: {e}")
            return None
            
    async def phase4_clip_inference(self, tokens_with_charts: List[Dict]) -> List[Dict]:
        """
        PHASE 4: Run CLIP inference on captured charts
        
        Args:
            tokens_with_charts: Tokens with captured charts
            
        Returns:
            Tokens with CLIP predictions added
        """
        print(f"[PHASE 4] Running CLIP inference")
        start_time = time.time()
        
        clip_enhanced = []
        
        for token_data in tokens_with_charts:
            symbol = token_data['symbol']
            
            if not token_data.get('chart_captured', False):
                # No chart - set unknown label
                token_data['ai_label'] = {'label': 'unknown', 'confidence': 0.0, 'method': 'no_chart'}
                clip_enhanced.append(token_data)
                continue
                
            try:
                # Run CLIP prediction
                clip_result = await self.run_clip_prediction(symbol, token_data['chart_path'])
                
                if clip_result:
                    token_data['ai_label'] = clip_result
                    # Save to cache
                    self.save_clip_cache(symbol, clip_result)
                    print(f"[CLIP OK] {symbol}: {clip_result['label']} ({clip_result['confidence']:.3f})")
                else:
                    token_data['ai_label'] = {'label': 'unknown', 'confidence': 0.0, 'method': 'clip_failed'}
                    print(f"[CLIP FAIL] {symbol}: Prediction failed")
                    
            except Exception as e:
                print(f"[CLIP ERROR] {symbol}: {e}")
                token_data['ai_label'] = {'label': 'unknown', 'confidence': 0.0, 'method': 'error'}
                
            clip_enhanced.append(token_data)
            
        elapsed = time.time() - start_time
        successful = len([t for t in clip_enhanced if t.get('ai_label', {}).get('label') != 'unknown'])
        print(f"[PHASE 4] Completed in {elapsed:.1f}s, {successful}/{len(clip_enhanced)} CLIP predictions")
        
        return clip_enhanced
        
    async def run_clip_prediction(self, symbol: str, chart_path: str) -> Optional[Dict]:
        """Run CLIP prediction on chart"""
        try:
            # INTEGRATION WITH REAL CLIP VISION-AI SYSTEM
            from vision.ai_label_pipeline import prepare_ai_label
            
            print(f"[CLIP INFERENCE] {symbol}: Analyzing chart with Vision-AI...")
            
            if not chart_path or not os.path.exists(chart_path):
                print(f"[CLIP ERROR] {symbol}: Chart path invalid or missing")
                return None
            
            # Run complete AI-EYE Vision-AI pipeline analysis
            ai_result = prepare_ai_label(
                symbol=symbol,
                chart_path=chart_path,
                heatmap_path=None,  # Will be generated if needed
                candles=[],  # Can be enhanced with candle data later
                orderbook={}  # Can be enhanced with orderbook data later
            )
            
            if ai_result and 'ai_label' in ai_result:
                label = ai_result['ai_label']
                confidence = ai_result.get('ai_confidence', 0.0)
                
                print(f"[CLIP SUCCESS] {symbol}: {label} (confidence: {confidence:.3f})")
                
                return {
                    'label': label,
                    'confidence': confidence,
                    'method': 'ai_eye_vision',
                    'timestamp': datetime.now().isoformat(),
                    'chart_analyzed': chart_path,
                    'phase': ai_result.get('ai_phase', 'unknown'),
                    'setup': ai_result.get('ai_setup', 'unknown')
                }
            else:
                print(f"[CLIP FAILED] {symbol}: No label extracted from AI-EYE")
                return {
                    'label': 'unknown',
                    'confidence': 0.0,
                    'method': 'ai_eye_failed',
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            print(f"[CLIP PREDICTION ERROR] {symbol}: {e}")
            # Return fallback instead of None to maintain pipeline flow
            return {
                'label': 'unknown',
                'confidence': 0.0,
                'method': 'clip_error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def save_clip_cache(self, symbol: str, clip_result: Dict):
        """Save CLIP prediction to cache"""
        try:
            cache_path = os.path.join(self.clip_predictions_dir, f"{symbol}.json")
            with open(cache_path, 'w') as f:
                json.dump(clip_result, f, indent=2)
        except Exception as e:
            print(f"[CLIP CACHE ERROR] {symbol}: {e}")
            
    async def phase5_advanced_modules(self, clip_enhanced_tokens: List[Dict]) -> List[Dict]:
        """
        PHASE 5: Run advanced modules with real AI-EYE data
        
        Args:
            clip_enhanced_tokens: Tokens with CLIP predictions
            
        Returns:
            Final TJDE results with all modules
        """
        print(f"[PHASE 5] Running advanced modules")
        start_time = time.time()
        
        final_results = []
        
        for token_data in clip_enhanced_tokens:
            symbol = token_data['symbol']
            
            try:
                # Extract market data components
                market_data = token_data['market_data']
                candles_15m = market_data.get('candles_15m', [])
                
                # Create ticker data structure
                ticker_data = {
                    'lastPrice': str(market_data.get('price', 0)),
                    'volume24h': str(market_data.get('volume_24h', 0)),
                    'price24hPcnt': str(market_data.get('price_change_24h', 0))
                }
                
                # Create orderbook placeholder
                orderbook = {}
                
                # Create signals structure
                signals = {
                    'cluster_strength': 0.5,
                    'cluster_direction': 1.0,
                    'cluster_volume_ratio': 1.0,
                    'market_phase': 'trend-following'
                }
                
                # Create AI label structure
                ai_label = token_data.get('ai_label', {
                    'label': 'breakout_pattern',
                    'confidence': 0.7,
                    'method': 'clip_prediction'
                })
                
                # CRITICAL FIX: Generate HTF candles for Phase 5 advanced modules
                htf_candles = token_data.get('htf_candles', [])
                if not htf_candles and candles_15m and len(candles_15m) >= 4:
                    htf_candles = self.generate_htf_candles_from_15m(candles_15m)
                    token_data['htf_candles'] = htf_candles  # Save for future use
                    print(f"[PHASE 5 HTF] {symbol}: Generated {len(htf_candles)} HTF candles")
                else:
                    print(f"[PHASE 5 HTF] {symbol}: Using existing {len(htf_candles)} HTF candles")
                
                # Prepare unified data with correct signature and HTF candles
                unified_data = prepare_unified_data(
                    symbol=symbol,
                    candles=candles_15m,
                    ticker_data=ticker_data,
                    orderbook=orderbook,
                    market_data=market_data,
                    signals=signals,
                    ai_label=ai_label,
                    htf_candles=htf_candles  # Use real HTF candles
                )
                
                # Run advanced scoring with unified data
                advanced_result = simulate_trader_decision_advanced(data=unified_data)
                
                if advanced_result:
                    # Extract meaningful scores from advanced result
                    advanced_score = advanced_result.get('final_score', 
                                   advanced_result.get('score', token_data['basic_score']))
                    decision = advanced_result.get('decision', 'unknown')
                    
                    final_result = {
                        'symbol': symbol,
                        'basic_score': token_data['basic_score'],
                        'advanced_score': advanced_score,
                        'final_decision': decision,
                        'ai_label': token_data['ai_label'],
                        'chart_path': token_data.get('chart_path'),
                        'breakdown': advanced_result.get('score_breakdown', {}),
                        'modules_active': advanced_result.get('active_modules', 0),
                        'engine_version': 'tjde_v3',
                        
                        # MAIN SYSTEM COMPATIBILITY FIELDS
                        'tjde_score': advanced_score,
                        'tjde_decision': decision,
                        'score': advanced_score,
                        'decision': decision,
                        
                        # Additional fields expected by main system
                        'volume_24h': token_data['market_data'].get('volume_24h', 0),
                        'price_change_24h': token_data['market_data'].get('price_change_24h', 0),
                        'current_price': token_data['market_data'].get('price', 0),
                        'market_phase': advanced_result.get('market_phase', 'unknown')
                    }
                    
                    print(f"[ADVANCED RESULT] {symbol}: {advanced_score:.3f} ({decision})")
                    final_results.append(final_result)
                else:
                    print(f"[ADVANCED FAIL] {symbol}: No result from unified engine")
                    
                    print(f"[ADVANCED OK] {symbol}: {advanced_result['score']:.3f} ({advanced_result['decision']})")
                    
            except Exception as e:
                print(f"[ADVANCED ERROR] {symbol}: {e}")
                continue
                
        # Sort by advanced score
        final_results.sort(key=lambda x: x['advanced_score'], reverse=True)
        
        elapsed = time.time() - start_time
        print(f"[PHASE 5] Completed in {elapsed:.1f}s, {len(final_results)} advanced results")
        
        return final_results
        
    async def run_full_pipeline(self, symbols: List[str], priority_info: Dict = None) -> List[Dict]:
        """
        Run complete TJDE v3 pipeline
        
        Args:
            symbols: List of symbols to analyze
            priority_info: Priority information
            
        Returns:
            Final TJDE results
        """
        pipeline_start = time.time()
        print(f"[TJDE v3 PIPELINE] Starting full analysis for {len(symbols)} symbols")
        print(f"[PHASE FLOW] Phase 1: Basic scoring → Phase 2: TOP 20 selection → Phase 3: Charts → Phase 4: CLIP → Phase 5: Advanced")
        
        # Phase 1: Basic scoring for all tokens
        print(f"[PHASE 1] Running basic scoring for ALL {len(symbols)} symbols...")
        basic_results = await self.phase1_basic_scoring(symbols, priority_info)
        
        if not basic_results:
            print("[PIPELINE] No viable tokens from basic scoring")
            return []
            
        print(f"[PHASE 1 COMPLETE] {len(basic_results)} tokens with basic scores")
        
        # Phase 2: Select top tokens
        print(f"[PHASE 2] Selecting TOP 20 tokens from {len(basic_results)} candidates...")
        selected_tokens = self.select_top_tokens(basic_results)
        
        if not selected_tokens:
            print("[PIPELINE] No tokens selected for advanced analysis")
            return []
            
        print(f"[PHASE 2 COMPLETE] {len(selected_tokens)} tokens selected for Phase 3-5 (charts + AI-EYE + advanced)")
        print(f"[CRITICAL CHECK] Only these {len(selected_tokens)} tokens will proceed to chart generation and AI analysis")
            
        # Phase 3: Capture charts
        tokens_with_charts = await self.phase3_chart_capture(selected_tokens)
        
        # Phase 4: CLIP inference
        clip_enhanced = await self.phase4_clip_inference(tokens_with_charts)
        
        # Phase 5: Advanced modules
        final_results = await self.phase5_advanced_modules(clip_enhanced)
        
        total_elapsed = time.time() - pipeline_start
        
        print(f"[TJDE v3 COMPLETE] Pipeline finished in {total_elapsed:.1f}s")
        print(f"[PIPELINE SUMMARY] {len(symbols)} → {len(basic_results)} → {len(selected_tokens)} → {len(final_results)}")
        
        if final_results:
            top_3 = [(r['symbol'], r['advanced_score'], r['final_decision']) for r in final_results[:3]]
            print(f"[TOP 3 FINAL] {top_3}")
            
        return final_results
        
    async def run_pipeline_from_data(self, scan_results: List[Dict], priority_info: Dict = None) -> List[Dict]:
        """
        Run TJDE v3 pipeline using pre-fetched scan data
        Skips Phase 1 data fetching and goes directly to basic scoring
        
        Args:
            scan_results: Pre-fetched scan results from AsyncTokenScanner
            priority_info: Priority information
            
        Returns:
            Final TJDE results from TOP 20 selection + advanced analysis
        """
        pipeline_start = time.time()
        print(f"[TJDE v3 FROM DATA] Starting pipeline from {len(scan_results)} pre-fetched results")
        print(f"[PHASE FLOW] Skip Phase 1 → Phase 2: TOP 20 selection → Phase 3: Charts → Phase 4: CLIP → Phase 5: Advanced")
        
        # Convert scan results to basic scoring format
        basic_results = []
        for result in scan_results:
            try:
                symbol = result.get('symbol')
                if not symbol:
                    continue
                
                # Check if token has TJDE score (already processed)
                tjde_score = result.get('tjde_score', 0) or result.get('score', 0)
                
                if tjde_score > 0:
                    # Convert to basic scoring format
                    basic_result = {
                        'symbol': symbol,
                        'score': tjde_score,
                        'basic_score': tjde_score,
                        'decision': result.get('tjde_decision', 'unknown'),
                        'market_data': {
                            'symbol': symbol,
                            'price': result.get('price', 0),
                            'volume_24h': result.get('volume_24h', 0),
                            'price_change_24h': result.get('price_change_24h', 0),
                            'candles_15m': result.get('candles_15m', []),
                            'candles_5m': result.get('candles_5m', []),
                            'data_source': 'legacy_scan'
                        },
                        'breakdown': {
                            'trend': tjde_score * 0.3,
                            'volume': tjde_score * 0.3,
                            'momentum': tjde_score * 0.4
                        }
                    }
                    basic_results.append(basic_result)
                    
            except Exception as e:
                print(f"[DATA CONVERSION ERROR] {symbol}: {e}")
                continue
        
        print(f"[DATA CONVERSION] Converted {len(basic_results)} scan results to basic scoring format")
        
        if not basic_results:
            print("[PIPELINE FROM DATA] No viable tokens from scan results")
            return []
            
        # Phase 2: Select TOP 20 tokens
        print(f"[PHASE 2] Selecting TOP 20 tokens from {len(basic_results)} candidates...")
        selected_tokens = self.select_top_tokens(basic_results, top_n=20)
        
        if not selected_tokens:
            print("[PIPELINE FROM DATA] No tokens selected for advanced analysis")
            return []
            
        print(f"[PHASE 2 COMPLETE] {len(selected_tokens)} tokens selected for Phase 3-5 (charts + AI-EYE + advanced)")
        
        # Phase 3: Capture charts for selected tokens only
        tokens_with_charts = await self.phase3_chart_capture(selected_tokens)
        
        # Phase 4: CLIP inference
        clip_enhanced = await self.phase4_clip_inference(tokens_with_charts)
        
        # Phase 5: Advanced modules
        final_results = await self.phase5_advanced_modules(clip_enhanced)
        
        total_elapsed = time.time() - pipeline_start
        
        print(f"[TJDE v3 FROM DATA COMPLETE] Pipeline finished in {total_elapsed:.1f}s")
        print(f"[PIPELINE SUMMARY] {len(scan_results)} → {len(basic_results)} → {len(selected_tokens)} → {len(final_results)}")
        
        if final_results:
            top_3 = [(r['symbol'], r['advanced_score'], r['final_decision']) for r in final_results[:3]]
            print(f"[TOP 3 FINAL] {top_3}")
            
        return final_results


# Global pipeline instance
tjde_v3_pipeline = TJDEv3Pipeline()


async def scan_with_tjde_v3(symbols: List[str], priority_info: Dict = None) -> List[Dict]:
    """
    Main entry point for TJDE v3 scanning
    """
    return await tjde_v3_pipeline.run_full_pipeline(symbols, priority_info)

async def scan_with_tjde_v3_from_data(scan_results: List[Dict], priority_info: Dict = None) -> List[Dict]:
    """
    TJDE v3 entry point using pre-fetched scan data
    Bypasses Phase 1 data fetching and uses existing results
    
    Args:
        scan_results: Pre-fetched scan results from AsyncTokenScanner
        priority_info: Priority information
        
    Returns:
        Final TJDE v3 results from TOP 20 selected tokens
    """
    return await tjde_v3_pipeline.run_pipeline_from_data(scan_results, priority_info)


if __name__ == "__main__":
    # Test pipeline
    async def test_pipeline():
        test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        results = await scan_with_tjde_v3(test_symbols)
        
        print("\n=== TJDE v3 TEST RESULTS ===")
        for result in results:
            print(f"{result['symbol']}: {result['advanced_score']:.3f} ({result['final_decision']}) - AI: {result['ai_label']['label']}")
            
    asyncio.run(test_pipeline())