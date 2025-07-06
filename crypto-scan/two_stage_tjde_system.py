#!/usr/bin/env python3
"""
Two-Stage TJDE System - Solves the critical contextual data timing issue

PHASE 1: Basic screening of all tokens using lightweight analysis
PHASE 2: Advanced analysis with full contextual data only for TOP 5 candidates

This fixes the problem where advanced modules receive 0.0 scores due to missing
ai_label, htf_candles, and pattern detection data during initial scanning.
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

# Import both engines
from trader_ai_engine_basic import simulate_trader_decision_basic
from unified_scoring_engine import simulate_trader_decision_advanced, prepare_unified_data

class TwoStageTJDESystem:
    """
    Implements delayed full scoring after initial filtering
    """
    
    def __init__(self, basic_threshold: float = 0.30, top_n: int = 5):
        self.basic_threshold = basic_threshold
        self.top_n = top_n
        self.stage1_results = []
        self.stage2_results = []
        
    async def stage1_basic_screening(self, tokens_data: List[Dict]) -> List[Dict]:
        """
        Stage 1: Basic screening of all tokens
        Uses only 15M/5M candles + orderbook for quick filtering
        """
        print(f"[STAGE 1] Starting basic screening of {len(tokens_data)} tokens")
        stage1_candidates = []
        
        for token_data in tokens_data:
            symbol = token_data.get('symbol', 'UNKNOWN')
            
            try:
                # Extract basic data
                current_price = token_data.get('price', 0)
                candles_15m = token_data.get('candles_15m', [])
                candles_5m = token_data.get('candles_5m', [])
                orderbook_data = token_data.get('orderbook', {})
                volume_24h = token_data.get('volume_24h', 0)
                price_change_24h = token_data.get('price_change_24h', 0)
                
                # Basic TJDE analysis - NO ai_label or htf_candles required
                basic_result = simulate_trader_decision_basic(
                    symbol=symbol,
                    current_price=current_price,
                    candles_15m=candles_15m,
                    candles_5m=candles_5m,
                    orderbook_data=orderbook_data,
                    volume_24h=volume_24h,
                    price_change_24h=price_change_24h
                )
                
                basic_score = basic_result.get('final_score', 0.0)
                basic_decision = basic_result.get('decision', 'avoid')
                
                print(f"[STAGE 1] {symbol}: Score={basic_score:.4f}, Decision={basic_decision}")
                
                # Check if token qualifies for Stage 2
                if basic_score >= self.basic_threshold and basic_decision in ['consider', 'wait']:
                    candidate = {
                        'symbol': symbol,
                        'basic_score': basic_score,
                        'basic_decision': basic_decision,
                        'basic_confidence': basic_result.get('confidence', 0.0),
                        'token_data': token_data,  # Preserve all original data
                        'stage1_timestamp': datetime.now().isoformat()
                    }
                    stage1_candidates.append(candidate)
                    print(f"[STAGE 1 QUALIFIED] {symbol}: Score={basic_score:.4f} → Candidate for Stage 2")
                else:
                    print(f"[STAGE 1 FILTERED] {symbol}: Score={basic_score:.4f} → Below threshold ({self.basic_threshold})")
                    
            except Exception as e:
                print(f"[STAGE 1 ERROR] {symbol}: {e}")
                continue
        
        # Sort by basic score and select TOP N
        stage1_candidates.sort(key=lambda x: x['basic_score'], reverse=True)
        top_candidates = stage1_candidates[:self.top_n]
        
        print(f"[STAGE 1 COMPLETE] Selected {len(top_candidates)}/{len(stage1_candidates)} candidates for Stage 2")
        for i, candidate in enumerate(top_candidates, 1):
            print(f"[TOP {i}] {candidate['symbol']}: Basic Score={candidate['basic_score']:.4f}")
        
        self.stage1_results = top_candidates
        return top_candidates
    
    async def stage2_advanced_analysis(self, top_candidates: List[Dict]) -> List[Dict]:
        """
        Stage 2: Advanced analysis with full contextual data
        Only for TOP N candidates from Stage 1
        """
        print(f"[STAGE 2] Starting advanced analysis for {len(top_candidates)} TOP candidates")
        stage2_results = []
        
        for candidate in top_candidates:
            symbol = candidate['symbol']
            token_data = candidate['token_data']
            
            try:
                print(f"[STAGE 2] Processing {symbol} with full contextual data...")
                
                # Generate contextual data for advanced analysis
                contextual_data = await self._generate_contextual_data(symbol, token_data)
                
                # Initialize variables
                final_score = 0.0
                final_decision = 'wait'
                advanced_result = None
                basic_result = None
                
                # FIX: Enhanced validation - check for meaningful contextual data
                has_ai_data = contextual_data and contextual_data.get('ai_label') and contextual_data.get('ai_label') != {}
                has_htf_data = contextual_data and contextual_data.get('htf_candles') and len(contextual_data.get('htf_candles', [])) > 20
                
                if not contextual_data or (not has_ai_data and not has_htf_data):
                    print(f"[STAGE 2 SKIP] {symbol}: Insufficient contextual data (AI: {has_ai_data}, HTF: {has_htf_data}) - using basic engine")
                    
                    # FIX: Fallback to basic engine when advanced modules unavailable
                    basic_result = simulate_trader_decision_basic(
                        symbol=symbol,
                        candles_15m=token_data.get('candles_15m', []),
                        candles_5m=token_data.get('candles_5m', []),
                        orderbook=token_data.get('orderbook_data', {}),
                        volume_24h=token_data.get('volume_24h', 0.0),
                        price_change_24h=token_data.get('price_change_24h', 0.0),
                        current_price=token_data.get('price_usd', 0.0)
                    )
                    
                    final_score = basic_result.get('score', 0.0)
                    final_decision = basic_result.get('decision', 'wait')
                    
                    print(f"[STAGE 2 BASIC] {symbol}: Score={final_score:.4f}, Decision={final_decision}")
                    
                else:
                    print(f"[STAGE 2 ADVANCED] {symbol}: Sufficient contextual data - proceeding with advanced analysis")
                    
                    # Prepare unified data with all contextual information
                    unified_data = prepare_unified_data(
                        symbol=symbol,
                        candles=token_data.get('candles_15m', []),
                        ticker_data=token_data.get('ticker_data', {}),
                        orderbook=token_data.get('orderbook', {}),
                        market_data=token_data,
                        signals=contextual_data['signals'],
                        ai_label=contextual_data.get('ai_label', {}),
                        htf_candles=contextual_data.get('htf_candles', [])
                    )
                    
                    # Advanced TJDE analysis with complete contextual data
                    advanced_result = simulate_trader_decision_advanced(
                        symbol=symbol,
                        market_data=unified_data.get('market_data', {}),
                        signals=unified_data.get('signals', {}),
                        debug_info=unified_data
                    )
                    
                    final_score = advanced_result.get('final_score', 0.0)
                    final_decision = advanced_result.get('decision', 'wait')
                
                print(f"[STAGE 2] {symbol}: Score={final_score:.4f}, Decision={final_decision}")
                
                # Compile complete result with proper fallback handling
                if 'advanced_result' in locals():
                    # Advanced scoring was used
                    complete_result = {
                        'symbol': symbol,
                        'basic_score': candidate['basic_score'],
                        'advanced_score': final_score,
                        'final_decision': final_decision,
                        'confidence': advanced_result.get('confidence', 0.5) if advanced_result else 0.5,
                        'score_breakdown': advanced_result.get('score_breakdown', {}) if advanced_result else {},
                        'active_modules': advanced_result.get('active_modules', 0) if advanced_result else 0,
                        'strongest_component': advanced_result.get('strongest_component', 'unknown') if advanced_result else 'unknown',
                        'market_phase': advanced_result.get('market_phase', 'unknown') if advanced_result else 'unknown',
                        'ai_label': contextual_data.get('ai_label', {}) if contextual_data else {},
                        'chart_path': contextual_data.get('chart_path') if contextual_data else None
                    }
                else:
                    # Basic scoring was used
                    complete_result = {
                        'symbol': symbol,
                        'basic_score': candidate['basic_score'],
                        'advanced_score': final_score,
                        'final_decision': final_decision,
                        'confidence': basic_result.get('confidence', 0.5),
                        'score_breakdown': {'basic_engine': basic_result.get('components', {})},
                        'active_modules': 1,  # Only basic engine
                        'strongest_component': 'basic_engine',
                        'market_phase': 'basic_engine',
                        'ai_label': {},
                        'htf_analysis': {},
                        'stage2_timestamp': datetime.now().isoformat(),
                        'full_result': basic_result
                    }
                
                stage2_results.append(complete_result)
                print(f"[STAGE 2 SUCCESS] {symbol}: Complete analysis finished")
                
            except Exception as e:
                print(f"[STAGE 2 ERROR] {symbol}: {e}")
                continue
        
        # Sort by advanced score
        stage2_results.sort(key=lambda x: x['advanced_score'], reverse=True)
        
        print(f"[STAGE 2 COMPLETE] Advanced analysis completed for {len(stage2_results)} tokens")
        for i, result in enumerate(stage2_results, 1):
            print(f"[FINAL TOP {i}] {result['symbol']}: Advanced Score={result['advanced_score']:.4f}, Decision={result['final_decision']}")
        
        self.stage2_results = stage2_results
        return stage2_results
    
    async def _generate_contextual_data(self, symbol: str, token_data: Dict) -> Optional[Dict]:
        """
        Generate contextual data required for advanced analysis:
        - ai_label (GPT + CLIP analysis)
        - htf_candles (higher timeframe data)
        - pattern detection data
        """
        try:
            print(f"[CONTEXTUAL] {symbol}: Generating TradingView chart...")
            
            # Generate TradingView chart for AI analysis
            chart_path = await self._generate_tradingview_chart(symbol)
            if not chart_path:
                print(f"[CONTEXTUAL] {symbol}: Chart generation failed")
                return None
            
            print(f"[CONTEXTUAL] {symbol}: Running GPT + CLIP analysis...")
            
            # Generate AI label through GPT + CLIP pipeline
            ai_label = await self._generate_ai_label(symbol, chart_path)
            
            print(f"[CONTEXTUAL] {symbol}: Fetching HTF candles...")
            
            # Fetch HTF candles for macro analysis
            htf_candles = await self._fetch_htf_candles(symbol)
            
            print(f"[CONTEXTUAL] {symbol}: Extracting pattern signals...")
            
            # Extract enhanced signals for advanced modules
            signals = await self._extract_enhanced_signals(symbol, token_data, ai_label, htf_candles)
            
            contextual_data = {
                'ai_label': ai_label,
                'htf_candles': htf_candles,
                'signals': signals,
                'chart_path': chart_path,
                'htf_analysis': self._analyze_htf_structure(htf_candles)
            }
            
            print(f"[CONTEXTUAL SUCCESS] {symbol}: All contextual data generated")
            return contextual_data
            
        except Exception as e:
            print(f"[CONTEXTUAL ERROR] {symbol}: {e}")
            return None
    
    async def _generate_tradingview_chart(self, symbol: str) -> Optional[str]:
        """Generate TradingView chart for AI analysis"""
        try:
            from utils.tradingview_robust import RobustTradingViewGenerator
            
            generator = RobustTradingViewGenerator()
            chart_path = await generator.generate_chart_async(symbol)
            
            if chart_path and "TRADINGVIEW_FAILED" not in chart_path:
                return chart_path
            
            return None
            
        except Exception as e:
            print(f"[CHART ERROR] {symbol}: {e}")
            return None
    
    async def _generate_ai_label(self, symbol: str, chart_path: str) -> Dict:
        """Generate AI label through CLIP + GPT pipeline"""
        try:
            from vision.ai_label_pipeline import prepare_ai_label
            
            ai_label = prepare_ai_label(symbol, chart_path)
            return ai_label if ai_label else {}
            
        except Exception as e:
            print(f"[AI LABEL ERROR] {symbol}: {e}")
            return {}
    
    async def _fetch_htf_candles(self, symbol: str) -> List:
        """Fetch higher timeframe candles for macro analysis"""
        try:
            from utils.bybit_api import get_candles_safe
            
            # Fetch 1H candles for HTF analysis
            htf_candles = get_candles_safe(symbol, "1h", limit=100)
            return htf_candles if htf_candles else []
            
        except Exception as e:
            print(f"[HTF CANDLES ERROR] {symbol}: {e}")
            return []
    
    async def _extract_enhanced_signals(self, symbol: str, token_data: Dict, ai_label: Dict, htf_candles: List) -> Dict:
        """Extract enhanced signals for advanced modules"""
        try:
            from utils.feature_extractor import extract_all_features_for_token
            
            # Enhanced feature extraction with contextual data
            market_data = {
                'candles': token_data.get('candles_15m', []),
                'candles_5m': token_data.get('candles_5m', []),
                'price': token_data.get('price', 0),
                'volume_24h': token_data.get('volume_24h', 0)
            }
            
            features = extract_all_features_for_token(symbol, market_data['candles'], market_data)
            
            # Add contextual data to signals
            signals = features.copy()
            signals.update({
                'ai_label': ai_label,
                'htf_candles': htf_candles,
                'market_phase': ai_label.get('market_phase', 'unknown'),
                'pattern_type': ai_label.get('pattern_type', 'unknown'),
                'chart_path': token_data.get('chart_path', 'none')
            })
            
            return signals
            
        except Exception as e:
            print(f"[SIGNALS ERROR] {symbol}: {e}")
            return {}
    
    def _analyze_htf_structure(self, htf_candles: List) -> Dict:
        """Analyze HTF market structure"""
        if not htf_candles or len(htf_candles) < 20:
            return {'phase': 'unknown', 'strength': 0.0}
        
        try:
            # Simple HTF trend analysis
            closes = [float(candle[4]) if isinstance(candle, list) else float(candle.get('close', 0)) for candle in htf_candles[-20:]]
            
            if len(closes) < 10:
                return {'phase': 'insufficient_data', 'strength': 0.0}
            
            # Calculate trend direction
            recent_avg = sum(closes[-5:]) / 5
            older_avg = sum(closes[:5]) / 5
            
            if recent_avg > older_avg * 1.02:
                phase = 'uptrend'
                strength = min(1.0, (recent_avg - older_avg) / older_avg * 10)
            elif recent_avg < older_avg * 0.98:
                phase = 'downtrend'
                strength = min(1.0, (older_avg - recent_avg) / older_avg * 10)
            else:
                phase = 'sideways'
                strength = 0.5
            
            return {'phase': phase, 'strength': strength}
            
        except Exception as e:
            print(f"[HTF ANALYSIS ERROR]: {e}")
            return {'phase': 'error', 'strength': 0.0}
    
    async def run_complete_two_stage_analysis(self, tokens_data: List[Dict]) -> Dict:
        """
        Run complete two-stage TJDE analysis
        Returns both Stage 1 and Stage 2 results
        """
        start_time = datetime.now()
        
        print(f"[TWO-STAGE START] Beginning complete analysis of {len(tokens_data)} tokens")
        
        # Stage 1: Basic screening
        stage1_start = datetime.now()
        top_candidates = await self.stage1_basic_screening(tokens_data)
        stage1_time = (datetime.now() - stage1_start).total_seconds()
        
        # Stage 2: Advanced analysis
        stage2_start = datetime.now()
        final_results = await self.stage2_advanced_analysis(top_candidates)
        stage2_time = (datetime.now() - stage2_start).total_seconds()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        print(f"[TWO-STAGE COMPLETE] Total time: {total_time:.1f}s (Stage 1: {stage1_time:.1f}s, Stage 2: {stage2_time:.1f}s)")
        
        return {
            'stage1_results': self.stage1_results,
            'stage2_results': self.stage2_results,
            'final_results': final_results,
            'timing': {
                'total_time': total_time,
                'stage1_time': stage1_time,
                'stage2_time': stage2_time,
                'tokens_processed': len(tokens_data),
                'top_candidates': len(top_candidates),
                'final_results': len(final_results)
            }
        }

# Test function
async def test_two_stage_system():
    """Test the two-stage system with sample data"""
    
    # Sample token data for testing
    sample_tokens = [
        {
            'symbol': 'BTCUSDT',
            'price': 45000.0,
            'volume_24h': 1000000,
            'price_change_24h': 2.5,
            'candles_15m': [[1640995200, 44000, 45500, 43500, 45000, 1000]] * 30,
            'candles_5m': [[1640995200, 44000, 45500, 43500, 45000, 1000]] * 90,
            'orderbook': {'bids': [[44950, 1.0]], 'asks': [[45050, 1.0]]}
        },
        {
            'symbol': 'ETHUSDT',
            'price': 3200.0,
            'volume_24h': 800000,
            'price_change_24h': 1.8,
            'candles_15m': [[1640995200, 3100, 3250, 3050, 3200, 800]] * 30,
            'candles_5m': [[1640995200, 3100, 3250, 3050, 3200, 800]] * 90,
            'orderbook': {'bids': [[3195, 2.0]], 'asks': [[3205, 2.0]]}
        }
    ]
    
    system = TwoStageTJDESystem(basic_threshold=0.25, top_n=2)
    results = await system.run_complete_two_stage_analysis(sample_tokens)
    
    print("\n[TEST RESULTS]")
    print(f"Stage 1 candidates: {len(results['stage1_results'])}")
    print(f"Stage 2 results: {len(results['stage2_results'])}")
    print(f"Total time: {results['timing']['total_time']:.1f}s")

if __name__ == "__main__":
    asyncio.run(test_two_stage_system())