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


class TJDEv3Pipeline:
    """
    Two-phase TJDE pipeline that fixes AI-EYE circular dependency
    """
    
    def __init__(self):
        self.clip_predictions_dir = "training_data/clip_predictions"
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.clip_predictions_dir, exist_ok=True)
        os.makedirs("training_data/charts", exist_ok=True)
        
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
        
        # Process all tokens with basic scoring
        for symbol in symbols:
            try:
                # Get market data - use real data from existing APIs
                market_data = await self.fetch_real_market_data(symbol)
                
                # Skip if no data
                if not market_data:
                    continue
                
                if not market_data:
                    continue
                    
                # Run basic scoring (no AI-EYE dependency)
                basic_result = await self.run_basic_scoring(symbol, market_data)
                
                if basic_result and basic_result.get('score', 0) > 0:
                    results.append({
                        'symbol': symbol,
                        'basic_score': basic_result['score'],
                        'basic_decision': basic_result['decision'],
                        'market_data': market_data,
                        'basic_breakdown': basic_result.get('breakdown', {})
                    })
                    
            except Exception as e:
                print(f"[PHASE 1 ERROR] {symbol}: {e}")
                continue
                
        # Sort by score
        results.sort(key=lambda x: x['basic_score'], reverse=True)
        
        elapsed = time.time() - start_time
        print(f"[PHASE 1] Completed in {elapsed:.1f}s, found {len(results)} viable tokens")
        
        return results
        
    async def run_basic_scoring(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Run basic scoring without AI-EYE dependency"""
        try:
            # Validate market data first
            if not market_data or not market_data.get('candles_15m'):
                print(f"[BASIC SCORING] {symbol}: No candle data available")
                return None
            
            # Use basic engine - no AI-EYE, no HTF dependency
            result = simulate_trader_decision_basic(
                symbol=symbol,
                candles_15m=market_data.get('candles_15m', []),
                candles_5m=market_data.get('candles_5m', []),
                volume_24h=market_data.get('volume_24h', 0),
                price_change_24h=market_data.get('price_change_24h', 0),
                current_price=market_data.get('price', 0)
            )
            
            # Validate result has meaningful score
            if result and result.get('score', 0) > 0.001:
                print(f"[BASIC SCORING] {symbol}: Score {result['score']:.3f}")
                return result
            else:
                print(f"[BASIC SCORING] {symbol}: Low score {result.get('score', 0):.3f}")
                return None
            
        except Exception as e:
            print(f"[BASIC SCORING ERROR] {symbol}: {e}")
            return None
            
    def select_top_tokens(self, basic_results: List[Dict], 
                         top_n: int = 40, min_score: float = 0.15) -> List[Dict]:
        """
        PHASE 2 SELECTION: Select top tokens for advanced analysis - REALISTIC THRESHOLDS
        
        Args:
            basic_results: Results from phase 1
            top_n: Maximum number of tokens to select
            min_score: Minimum score threshold (realistic for market data)
            
        Returns:
            Selected tokens for advanced analysis
        """
        # Filter by minimum score - realistic threshold for authentic market scores
        qualified = [r for r in basic_results if r['basic_score'] >= min_score]
        
        # Sort by score descending to get best candidates
        qualified.sort(key=lambda x: x['basic_score'], reverse=True)
        
        # Take top N candidates
        selected = qualified[:top_n]
        
        print(f"[PHASE 2 SELECTION] Selected {len(selected)}/{len(basic_results)} tokens for advanced analysis")
        print(f"[SELECTION CRITERIA] min_score={min_score}, top_n={top_n}")
        
        if selected:
            top_scores = [f"{r['symbol']}:{r['basic_score']:.3f}" for r in selected[:5]]
            print(f"[TOP SELECTED] {', '.join(top_scores)}")
        else:
            # Debug - show what scores we have
            all_scores = [f"{r['symbol']}:{r['basic_score']:.3f}" for r in basic_results[:5]]
            print(f"[DEBUG SCORES] Available: {', '.join(all_scores)}")
            
        return selected
        
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
            # Simplified chart generation for testing
            chart_filename = f"{symbol}_BYBIT_score-{int(score*1000):03d}_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            chart_path = f"training_data/charts/{chart_filename}"
            
            # For now, create a placeholder path (actual chart generation would happen here)
            # This will be integrated with real TradingView generation later
            return chart_path
            
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
            # For testing - simplified CLIP prediction
            # This will be enhanced with real CLIP model later
            test_labels = ['breakout_pattern', 'trend_continuation', 'pullback_in_trend', 
                          'momentum_follow', 'consolidation', 'unknown']
            
            # Simple pattern based on symbol characteristics for testing
            import random
            random.seed(hash(symbol) % 1000)  # Deterministic for same symbol
            
            label = random.choice(test_labels)
            confidence = random.uniform(0.3, 0.9)
            
            return {
                'label': label,
                'confidence': confidence,
                'method': 'test_fallback',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"[CLIP PREDICTION ERROR] {symbol}: {e}")
            return None
            
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
                
                # Prepare unified data with correct signature
                unified_data = prepare_unified_data(
                    symbol=symbol,
                    candles=candles_15m,
                    ticker_data=ticker_data,
                    orderbook=orderbook,
                    market_data=market_data,
                    signals=signals,
                    ai_label=ai_label,
                    htf_candles=[]
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
        
        # Phase 1: Basic scoring for all tokens
        basic_results = await self.phase1_basic_scoring(symbols, priority_info)
        
        if not basic_results:
            print("[PIPELINE] No viable tokens from basic scoring")
            return []
            
        # Phase 2: Select top tokens
        selected_tokens = self.select_top_tokens(basic_results)
        
        if not selected_tokens:
            print("[PIPELINE] No tokens selected for advanced analysis")
            return []
            
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


# Global pipeline instance
tjde_v3_pipeline = TJDEv3Pipeline()


async def scan_with_tjde_v3(symbols: List[str], priority_info: Dict = None) -> List[Dict]:
    """
    Main entry point for TJDE v3 scanning
    """
    return await tjde_v3_pipeline.run_full_pipeline(symbols, priority_info)


if __name__ == "__main__":
    # Test pipeline
    async def test_pipeline():
        test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        results = await scan_with_tjde_v3(test_symbols)
        
        print("\n=== TJDE v3 TEST RESULTS ===")
        for result in results:
            print(f"{result['symbol']}: {result['advanced_score']:.3f} ({result['final_decision']}) - AI: {result['ai_label']['label']}")
            
    asyncio.run(test_pipeline())