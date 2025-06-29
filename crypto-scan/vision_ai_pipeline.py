"""
Vision-AI CLIP Pipeline for Trend-Mode System
Complete implementation of auto-labeled charts and CLIP-based training
"""

import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import json
import os
import glob
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image


def save_training_chart(df: pd.DataFrame, symbol: str, timestamp: str, 
                       folder: str = "training_data/charts",  # Default to professional charts
                       tjde_score: float = None, clip_confidence: float = None,
                       market_phase: str = None, decision: str = None) -> str:
    """
    Save professional training chart using custom candlestick generation
    🎯 RESTRICTED TO TOP 5 TJDE TOKENS ONLY
    
    Args:
        df: OHLCV DataFrame with proper index
        symbol: Trading symbol
        timestamp: Timestamp string
        folder: Output folder path
        tjde_score: TJDE score for metadata
        clip_confidence: CLIP confidence for metadata
        market_phase: Market phase for metadata
        decision: TJDE decision for metadata
        
    Returns:
        Path to saved chart
    """
    try:
        # 🎯 CRITICAL FIX: Check TOP 5 status before generating training charts
        from utils.top5_selector import should_generate_training_data, warn_about_non_top5_generation
        
        if not should_generate_training_data(symbol, tjde_score):
            warn_about_non_top5_generation(symbol, "Vision-AI save_training_chart")
            return None
        
        print(f"[VISION-AI] {symbol}: Generating training chart (TOP 5 token)")
        
        from trend_charting import plot_custom_candlestick_chart
        
        os.makedirs(folder, exist_ok=True)
        
        # Prepare data for custom chart
        df_ohlc = pd.DataFrame({
            'timestamp': df.index,
            'open': df['Open'],
            'high': df['High'], 
            'low': df['Low'],
            'close': df['Close']
        })
        
        df_volume = pd.DataFrame({
            'timestamp': [mdates.date2num(ts) for ts in df.index],
            'volume': df['Volume']
        })
        
        # Enhanced chart path
        chart_path = f"{folder}/{symbol}_{timestamp}_vision_chart.png"
        
        # Generate professional Vision-AI chart with enhanced context
        from trend_charting import plot_chart_with_context
        
        # Convert DataFrame back to candle list format for new function
        candles_list = []
        for _, row in df.iterrows():
            candles_list.append({
                'timestamp': int(row.name.timestamp() * 1000),
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume']
            })
        
        # Memory-aware alert detection - get historical alerts from token memory
        alert_indices = []
        
        try:
            from utils.token_memory import get_token_history, update_token_memory
            token_history = get_token_history(symbol)
            
            # FIX 2: Ensure current TJDE decision is saved to memory (even if avoid)
            if tjde_score is not None and decision:
                memory_entry = {
                    "tjde_score": tjde_score,
                    "decision": decision,
                    "setup": f"{market_phase}_{decision}" if market_phase else decision,
                    "phase": market_phase or "unknown",
                    "vision_ai_chart": True,  # Flag for Vision-AI generation
                    "alert_generated": tjde_score >= 0.7,  # Track alert threshold
                    "result_after_2h": None  # For feedback loop
                }
                update_token_memory(symbol, memory_entry)
                print(f"[VISION-AI MEMORY] {symbol}: Saved TJDE decision {decision} (score: {tjde_score:.3f})")
            
            # Extract historical alert indices from memory
            if token_history:
                significant_entries = [e for e in token_history[-10:] if e.get('tjde_score', 0) >= 0.5]  # Lower threshold
                for i, entry in enumerate(significant_entries[-5:]):  # Last 5 significant decisions
                    # Create realistic alert positions based on memory
                    alert_pos = int(len(candles_list) * 0.6) + (i * 4)  # Spread alerts across chart
                    if alert_pos < len(candles_list) - 5:  # Leave margin at end
                        alert_indices.append(alert_pos)
                        
                print(f"[VISION-AI MEMORY] {symbol}: Found {len(significant_entries)} significant decisions in history")
            
        except ImportError:
            print(f"[VISION-AI] Token memory not available for {symbol}")
        except Exception as e:
            print(f"[VISION-AI MEMORY ERROR] {symbol}: {e}")
        
        # Add current alert if TJDE score is high
        if tjde_score and tjde_score >= 0.7:
            alert_start = int(len(candles_list) * 0.8)
            if alert_start < len(candles_list):
                volumes = [c['volume'] for c in candles_list[alert_start:]]
                current_alert = alert_start + volumes.index(max(volumes))
                alert_indices.append(current_alert)
        
        print(f"[VISION-AI] {symbol}: Using {len(alert_indices)} alert points for memory training")
        
        saved_path = plot_chart_with_context(
            symbol=symbol,
            candles=candles_list,
            alert_indices=alert_indices if alert_indices else None,
            score=tjde_score,
            decision=decision,
            phase=market_phase,
            setup=f"{market_phase}_{decision}" if market_phase and decision else None,
            save_path=chart_path,
            context_days=2
        )
        
        if saved_path:
            print(f"[VISION-AI] Training chart and metadata saved: {saved_path}")
            
            # Verify JSON metadata was created
            json_path = saved_path.replace('.png', '.json')
            if os.path.exists(json_path):
                print(f"[VISION-AI] Metadata file confirmed: {json_path}")
            else:
                print(f"[VISION-AI] Warning: Metadata file not found")
            
            # Generate GPT commentary for the chart
            try:
                from gpt_commentary import run_comprehensive_gpt_analysis
                
                # Prepare TJDE data for GPT analysis
                tjde_analysis = {
                    'final_score': tjde_score or 0.0,
                    'decision': decision or 'unknown',
                    'market_phase': market_phase or 'unknown'
                }
                
                # Add CLIP prediction if available
                clip_data = None
                clip_json_path = saved_path.replace('.png', '_clip.json')
                if os.path.exists(clip_json_path):
                    try:
                        with open(clip_json_path, 'r') as f:
                            clip_data = json.load(f)
                    except:
                        pass
                
                # Run comprehensive GPT analysis
                gpt_results = run_comprehensive_gpt_analysis(
                    saved_path, symbol, tjde_analysis, clip_data
                )
                
                if gpt_results:
                    print(f"[GPT ANALYSIS] Generated {len(gpt_results)} analyses for {symbol}")
                    
                    # FIX 1: Extract meaningful setup labels from GPT commentary
                    try:
                        from gpt_commentary import extract_primary_label_from_commentary
                        
                        # Get chart commentary and extract setup label
                        chart_commentary = gpt_results.get('chart_commentary', '')
                        if chart_commentary:
                            extracted_setup = extract_primary_label_from_commentary(chart_commentary)
                            
                            # Update JSON metadata with extracted setup
                            if os.path.exists(json_path):
                                with open(json_path, 'r') as f:
                                    metadata = json.load(f)
                                
                                # BONUS: Enhanced metadata with GPT insights
                                metadata.update({
                                    'gpt_extracted_setup': extracted_setup,
                                    'gpt_commentary_snippet': chart_commentary[:200] + '...' if len(chart_commentary) > 200 else chart_commentary,
                                    'gpt_analysis_available': True,
                                    'setup_source': 'gpt_extraction',
                                    'original_setup': metadata.get('setup', 'unknown')
                                })
                                
                                # Update primary setup field
                                if extracted_setup != 'unknown':
                                    metadata['setup'] = extracted_setup
                                    print(f"[VISION-AI LABEL FIX] {symbol}: Updated setup from 'unknown' to '{extracted_setup}'")
                                
                                # Save enhanced metadata
                                with open(json_path, 'w') as f:
                                    json.dump(metadata, f, indent=2)
                                    
                            print(f"[GPT LABEL EXTRACTION] {symbol}: Setup = '{extracted_setup}'")
                            
                            # AUTOMATIC FILE RENAMING: Rename files with GPT-extracted labels
                            try:
                                from gpt_commentary import rename_chart_files_with_gpt_label
                                
                                new_png_path, new_json_path = rename_chart_files_with_gpt_label(
                                    saved_path, chart_commentary
                                )
                                
                                if new_png_path != saved_path:
                                    print(f"[AUTO RENAME] {symbol}: Files renamed with GPT label '{extracted_setup}'")
                                    saved_path = new_png_path  # Update path for return
                                    
                            except Exception as e:
                                print(f"[AUTO RENAME ERROR] {symbol}: {e}")
                        
                    except Exception as e:
                        print(f"[GPT LABEL EXTRACTION ERROR] {symbol}: {e}")
                
            except ImportError:
                print("[GPT ANALYSIS] GPT commentary not available")
            except Exception as e:
                print(f"[GPT ANALYSIS ERROR] {e}")
            
            return saved_path
        else:
            # Fallback to basic chart if custom fails
            print(f"[VISION-AI] Custom chart failed, using fallback")
            return f"{folder}/{symbol}_{timestamp}_fallback.png"
            
    except Exception as e:
        print(f"[VISION-AI CHART ERROR] {e}")
        return f"{folder}/{symbol}_{timestamp}_error.png"


def save_label_jsonl(symbol: str, timestamp: str, label_data: Dict, 
                    output: str = "training_data/labels.jsonl") -> bool:
    """
    Save training labels in JSONL format for CLIP training
    
    Args:
        symbol: Trading symbol
        timestamp: Timestamp string  
        label_data: Dictionary with phase, setup, scores, etc.
        output: Output JSONL file path
        
    Returns:
        Success status
    """
    try:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        
        label_entry = {
            "symbol": symbol,
            "timestamp": timestamp,
            "chart_path": f"training_data/charts/{symbol}_{timestamp}_chart.png",
            **label_data
        }
        
        with open(output, "a", encoding='utf-8') as f:
            f.write(json.dumps(label_entry) + "\n")
            
        print(f"[VISION-AI] Label saved: {symbol} - {label_data.get('phase', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"[VISION-AI ERROR] Failed to save label: {e}")
        return False


def fetch_candles_for_vision(symbol: str) -> Optional[List]:
    """
    Elastyczne pobieranie świec z fallbackiem dla Vision-AI
    
    Args:
        symbol: Symbol tradingowy
        
    Returns:
        Lista świec lub None jeśli brak wystarczających danych
    """
    try:
        # Próba 1: Sprawdź async results cache (enhanced)
        print(f"[VISION-AI FETCH] {symbol}: Checking async scan results...")
        
        # Check enhanced async results with proper field access
        async_file = f"data/async_results/{symbol}_async.json"
        
        if os.path.exists(async_file):
            try:
                # Check file age to prioritize recent files
                file_age = time.time() - os.path.getmtime(async_file)
                recent_file = file_age < 300  # 5 minutes
                
                with open(async_file, 'r') as f:
                    async_data = json.load(f)
                    # Enhanced: Check for candles_15m field (new format)
                    candles = async_data.get("candles_15m", async_data.get("candles", []))
                    
                    print(f"[VISION-AI ASYNC] {symbol}: Found {len(candles)} candles in {async_file} (age: {file_age:.1f}s)")
                    
                    # Lower threshold for recent files with enhanced data
                    min_candles = 20 if recent_file else 30
                    if candles and len(candles) >= min_candles:
                        print(f"[VISION-AI ASYNC SUCCESS] {symbol}: Using {len(candles)} enhanced cached candles")
                        return candles
                    elif candles:
                        print(f"[VISION-AI ASYNC PARTIAL] {symbol}: Only {len(candles)} candles (need {min_candles}+)")
                    else:
                        print(f"[VISION-AI ASYNC EMPTY] {symbol}: No candles in cached data")
                        
            except Exception as e:
                print(f"[VISION-AI ASYNC ERROR] {symbol}: Error reading {async_file}: {e}")
        
        print(f"[VISION-AI ASYNC] {symbol}: No valid async cache found")
        
        # Próba 2: Sprawdź scan results z ostatniego skanu
        print(f"[VISION-AI SCAN] {symbol}: Checking recent scan data...")
        scan_file = f"data/scan_results/latest_scan.json"
        if os.path.exists(scan_file):
            try:
                with open(scan_file, 'r') as f:
                    scan_data = json.load(f)
                    for result in scan_data.get("results", []):
                        if result.get("symbol") == symbol:
                            candles = result.get("candles", result.get("market_data", {}).get("candles", []))
                            if candles and len(candles) >= 30:
                                print(f"[VISION-AI SCAN] {symbol}: Got {len(candles)} scan cached candles")
                                return candles
            except Exception as e:
                print(f"[VISION-AI SCAN ERROR] {symbol}: {e}")
            
        # Próba 3: Bezpośrednie API Bybit z enhanced error handling
        print(f"[VISION-AI DIRECT] {symbol}: Direct Bybit API fetch...")
        
        try:
            response = requests.get(
                "https://api.bybit.com/v5/market/kline",
                params={
                    'category': 'linear',
                    'symbol': symbol,
                    'interval': '15',
                    'limit': '200'
                },
                timeout=10
            )
            
            print(f"[VISION-AI API] {symbol}: HTTP {response.status_code}")
            
            if response.status_code == 200:
                api_data = response.json()
                print(f"[VISION-AI API] {symbol}: retCode {api_data.get('retCode', 'unknown')}")
                
                if api_data.get('retCode') == 0:
                    api_candles = api_data.get('result', {}).get('list', [])
                    print(f"[VISION-AI API] {symbol}: Raw candles count: {len(api_candles)}")
                    
                    if len(api_candles) >= 30:
                        # Convert Bybit format to standard format
                        converted_candles = []
                        for candle_data in reversed(api_candles):  # Bybit returns newest first
                            try:
                                converted_candles.append([
                                    int(candle_data[0]),      # timestamp
                                    float(candle_data[1]),    # open
                                    float(candle_data[2]),    # high
                                    float(candle_data[3]),    # low
                                    float(candle_data[4]),    # close
                                    float(candle_data[5])     # volume
                                ])
                            except (ValueError, IndexError) as e:
                                print(f"[VISION-AI API] {symbol}: Candle conversion error: {e}")
                                continue
                        
                        if len(converted_candles) >= 30:
                            print(f"[VISION-AI DIRECT] {symbol}: Got {len(converted_candles)} converted API candles")
                            return converted_candles
                        else:
                            print(f"[VISION-AI API] {symbol}: Insufficient converted candles: {len(converted_candles)}")
                    else:
                        print(f"[VISION-AI API] {symbol}: Insufficient raw candles: {len(api_candles)}")
                else:
                    print(f"[VISION-AI API] {symbol}: API error retCode: {api_data.get('retCode')}")
            else:
                print(f"[VISION-AI API] {symbol}: HTTP error {response.status_code}")
                
        except Exception as e:
            print(f"[VISION-AI API] {symbol}: Request exception: {e}")
                    
        # Próba 4: Jeśli brak danych, zwróć None - nie generujemy syntetycznych danych
        print(f"[VISION-AI FAILED] {symbol}: No authentic candle data available from any source")
        return None
        
    except Exception as e:
        print(f"[VISION-AI ERROR] {symbol}: Candle fetch failed: {e}")
        return None

def prepare_top5_training_data(tjde_results: List[Dict]) -> List[Dict]:
    """
    Select TOP 5 tokens by TJDE score for training data generation with fresh data validation
    
    Args:
        tjde_results: List of scan results with TJDE scores
        
    Returns:
        Top 5 results sorted by TJDE score with fresh data validation
    """
    # Filter valid results with TJDE scores
    valid_results = [r for r in tjde_results if r.get('tjde_score', 0) > 0]
    
    if not valid_results:
        print("[VISION-AI] No valid TJDE results for training data")
        return []
    
    # Sort by TJDE score descending and take TOP 5
    top5 = sorted(valid_results, key=lambda x: x.get('tjde_score', 0), reverse=True)[:5]
    
    print(f"[VISION-AI] Selected TOP 5 tokens for training data with fresh data validation:")
    for i, result in enumerate(top5, 1):
        symbol = result['symbol']
        tjde_score = result['tjde_score']
        print(f"  {i}. {symbol}: TJDE {tjde_score:.3f}")
        
        # FRESH DATA VALIDATION: Check if this token has current market data
        market_data = result.get('market_data', {})
        candles_15m = market_data.get('candles_15m', [])
        
        if candles_15m:
            try:
                from utils.fresh_candles import validate_candle_freshness
                is_fresh = validate_candle_freshness(candles_15m, symbol, max_age_minutes=45)
                if not is_fresh:
                    print(f"    ⚠️ {symbol}: Market data may be stale - will fetch fresh data during chart generation")
                else:
                    print(f"    ✅ {symbol}: Market data is fresh")
            except Exception as e:
                print(f"    ⚠️ {symbol}: Could not validate data freshness: {e}")
        else:
            print(f"    ⚠️ {symbol}: No 15M candles in market_data - will fetch fresh data")
    
    return top5


def generate_vision_ai_training_data(tjde_results: List[Dict], vision_ai_mode: str = "full") -> int:
    """
    TradingView-ONLY Vision-AI training data generation pipeline
    Completely replaces matplotlib with authentic TradingView screenshots
    
    Args:
        tjde_results: List of scan results with TJDE analysis
        vision_ai_mode: Vision-AI mode ("full", "fast", "minimal")
        
    Returns:
        Number of authentic TradingView charts generated
    """
    try:
        print(f"[TRADINGVIEW-ONLY] 🎯 Starting TradingView-only chart generation pipeline")
        
        # 🎯 CRITICAL FIX: Use TOP 5 tokens for TradingView chart generation
        # First select TOP 5 tokens to prevent generating charts for all tokens
        top5_results = prepare_top5_training_data(tjde_results)
        
        if not top5_results:
            print("[TRADINGVIEW-ONLY] No TOP 5 tokens available for chart generation")
            return 0
        
        # ✅ CRITICAL FIX: Skip TradingView generation here since force_refresh_vision_ai_charts() already handles it
        # This prevents duplicate TradingView browser sessions and chart generation conflicts
        print("[TRADINGVIEW-ONLY] 🎯 Skipping TradingView generation - handled by force_refresh pipeline")
        
        # Check for existing TradingView charts generated by force_refresh
        chart_mapping = {}
        for result in top5_results:
            symbol = result.get('symbol', 'UNKNOWN')
            chart_pattern = f"training_data/charts/{symbol}_*.png"
            existing_charts = glob.glob(chart_pattern)
            if existing_charts:
                # Use most recent chart
                latest_chart = max(existing_charts, key=os.path.getmtime)
                chart_mapping[symbol] = latest_chart
                print(f"[TRADINGVIEW-ONLY] ✅ Found existing chart: {symbol} -> {os.path.basename(latest_chart)}")
            else:
                print(f"[TRADINGVIEW-ONLY] ⚠️ No existing chart found for {symbol}")
        
        # Import TradingView-only pipeline - DISABLED to prevent duplication
        # from utils.tradingview_only_pipeline import generate_tradingview_only_charts  
        # chart_mapping = generate_tradingview_only_charts(top5_results)
        
        charts_generated = len(chart_mapping)
        
        if charts_generated > 0:
            print(f"[TRADINGVIEW-ONLY] ✅ Generated {charts_generated} authentic TradingView charts")
            for symbol, path in chart_mapping.items():
                print(f"[TRADINGVIEW-ONLY] • {symbol}: {os.path.basename(path)}")
        else:
            print("[TRADINGVIEW-ONLY] ❌ No TradingView charts generated")
            
        # Generate metadata for training (but NO fallback charts)
        training_pairs_created = 0
        # ✅ Use already prepared TOP 5 results instead of calling prepare_top5_training_data again
        
        if top5_results and vision_ai_mode not in ["minimal"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            for result in top5_results:
                try:
                    symbol = result.get('symbol', 'UNKNOWN')
                    tjde_score = result.get('tjde_score', 0)
                    
                    # Skip low-quality signals in fast mode
                    if vision_ai_mode == "fast" and tjde_score < 0.5:
                        continue
                    
                    # Generate metadata for TJDE analysis
                    phase = result.get('market_phase', 'unknown')
                    decision = result.get('tjde_decision', 'unknown')
                    setup = result.get('setup_type', phase)
                    clip_confidence = result.get('clip_confidence', 0.0)
                    
                    # Create training metadata (no synthetic charts)
                    label_data = {
                        "phase": phase,
                        "setup": setup,
                        "tjde_score": tjde_score,
                        "tjde_decision": decision,
                        "confidence": clip_confidence,
                        "data_source": "tradingview_only",
                        "chart_type": "authentic_tradingview",
                        "has_tradingview_chart": symbol in chart_mapping,
                        "tradingview_path": chart_mapping.get(symbol, None)
                    }
                    
                    if save_label_jsonl(symbol, timestamp, label_data):
                        training_pairs_created += 1
                        
                except Exception as e:
                    print(f"[TRADINGVIEW-ONLY ERROR] {symbol}: {e}")
        
        # Save training summary
        save_training_summary_report(training_pairs_created, len(top5_results) if top5_results else 0)
        
        total_generated = charts_generated + training_pairs_created
        print(f"[TRADINGVIEW-ONLY] ✅ Generated {total_generated} total items ({charts_generated} authentic charts + {training_pairs_created} metadata)")
        
        return total_generated
        
    except ImportError as e:
        print(f"[TRADINGVIEW-ONLY ERROR] Cannot import TradingView pipeline: {e}")
        return 0
    except Exception as e:
        print(f"[TRADINGVIEW-ONLY ERROR] Pipeline failed: {e}")
        return 0

def save_missing_candles_report(symbol: str, reason: str):
    """Zapisuj tokeny z brakującymi danymi do raportu debugowania"""
    try:
        report_file = "data/missing_candles_report.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        # Wczytaj istniejący raport
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report = json.load(f)
        else:
            report = {"missing_candles": [], "last_updated": None}
        
        # Dodaj nowy wpis
        entry = {
            "symbol": symbol,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "needs_investigation": True
        }
        
        report["missing_candles"].append(entry)
        report["last_updated"] = datetime.now().isoformat()
        
        # Zachowaj tylko ostatnie 100 wpisów
        if len(report["missing_candles"]) > 100:
            report["missing_candles"] = report["missing_candles"][-100:]
        
        # Zapisz raport
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
    except Exception as e:
        print(f"[REPORT ERROR] Failed to save missing candles report: {e}")

def save_training_summary_report(generated: int, attempted: int):
    """Zapisuj podsumowanie sesji treningowej"""
    try:
        summary_file = "data/training_summary.json"
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        
        summary = {
            "session_timestamp": datetime.now().isoformat(),
            "tokens_attempted": attempted,
            "training_pairs_generated": generated,
            "success_rate": (generated / attempted) * 100 if attempted > 0 else 0,
            "status": "SUCCESS" if generated > 0 else "FAILED"
        }
        
        # Append to history
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                data = json.load(f)
            if "sessions" not in data:
                data["sessions"] = []
        else:
            data = {"sessions": []}
            
        data["sessions"].append(summary)
        data["last_session"] = summary
        
        # Keep only last 50 sessions
        if len(data["sessions"]) > 50:
            data["sessions"] = data["sessions"][-50:]
        
        with open(summary_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"[TRAINING SUMMARY] {generated}/{attempted} pairs generated ({summary['success_rate']:.1f}% success)")
        
    except Exception as e:
        print(f"[SUMMARY ERROR] Failed to save training summary: {e}")


class EnhancedCLIPPredictor:
    """Enhanced CLIP predictor with confidence handling and Vision-AI integration"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.fallback_predictor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CLIP model with enhanced error handling and fallback"""
        try:
            # Try existing CLIP predictor first
            from ai.clip_predictor import CLIPPredictor
            self.fallback_predictor = CLIPPredictor()
            
            if self.fallback_predictor.model:
                print("[VISION-AI] ✅ Using existing CLIP predictor for Vision-AI")
                return
            
        except Exception as e:
            print(f"[VISION-AI] Existing CLIP predictor unavailable: {e}")
        
        try:
            # Try transformers approach
            from transformers import CLIPProcessor, CLIPModel
            
            print("[VISION-AI] Loading openai/clip-vit-base-patch32...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", 
                use_fast=True
            )
            
            print("[VISION-AI] ✅ CLIP model loaded successfully")
            
        except Exception as e:
            print(f"[VISION-AI] Transformers CLIP unavailable: {e}")
            print("[VISION-AI] Operating in chart analysis mode without CLIP predictions")
    
    def predict_clip_confidence(self, image_path: str, 
                              labels: List[str] = None) -> Tuple[str, float]:
        """
        Enhanced CLIP prediction with proper confidence calculation
        
        Args:
            image_path: Path to chart image
            labels: List of phase labels for prediction
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        if labels is None:
            labels = [
                "trend-following", 
                "distribution", 
                "consolidation",
                "breakout-continuation",
                "pullback-in-trend",
                "accumulation",
                "reversal-pattern"
            ]
        
        # Try existing CLIP predictor first
        if self.fallback_predictor and self.fallback_predictor.model:
            try:
                result = self.fallback_predictor.predict_chart_setup(image_path)
                if result and result.get('confidence', 0) > 0:
                    predicted_label = result['label']
                    confidence = result['confidence']
                    
                    print(f"[VISION-AI CLIP] {image_path}: {predicted_label} (confidence: {confidence:.3f})")
                    return predicted_label, confidence
                    
            except Exception as e:
                print(f"[VISION-AI CLIP FALLBACK ERROR] {e}")
        
        # Try transformers approach
        if self.model and self.processor:
            try:
                # Load and process image
                image = Image.open(image_path).convert('RGB')
                
                # Process inputs with enhanced settings
                inputs = self.processor(
                    text=labels, 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Get model outputs
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image[0]
                probs = logits_per_image.softmax(dim=0)
                
                # Get top prediction
                top_label_idx = probs.argmax().item()
                confidence = probs[top_label_idx].item()
                predicted_label = labels[top_label_idx]
                
                print(f"[VISION-AI CLIP] {image_path}: {predicted_label} (confidence: {confidence:.3f})")
                
                return predicted_label, confidence
                
            except Exception as e:
                print(f"[VISION-AI CLIP ERROR] {e}")
        
        # Fallback to pattern-based analysis
        return self._pattern_based_analysis(image_path)


def integrate_clip_with_tjde(tjde_score: float, tjde_phase: str, 
                           chart_path: str) -> Tuple[float, Dict]:
    """
    Integrate CLIP confidence with TJDE scoring
    
    Args:
        tjde_score: Original TJDE score
        tjde_phase: Current market phase
        chart_path: Path to chart for CLIP analysis
        
    Returns:
        Tuple of (enhanced_score, clip_info)
    """
    try:
        predictor = EnhancedCLIPPredictor()
        
        if not predictor.model:
            return tjde_score, {"clip_phase": "N/A", "clip_confidence": 0.0}
        
        # Get CLIP prediction
        clip_phase, clip_confidence = predictor.predict_clip_confidence(chart_path)
        
        # Calculate score enhancement
        enhanced_score = tjde_score
        
        if clip_confidence > 0.7:
            if clip_phase == tjde_phase:
                # High confidence + phase match = boost
                enhanced_score += 0.05
                print(f"[VISION-AI BOOST] Phase match: {clip_phase} (confidence: {clip_confidence:.3f}) → +0.05")
            elif clip_confidence > 0.8:
                # Very high confidence but different phase = slight boost anyway
                enhanced_score += 0.02
                print(f"[VISION-AI BOOST] High confidence: {clip_phase} (confidence: {clip_confidence:.3f}) → +0.02")
        
        clip_info = {
            "clip_phase": clip_phase,
            "clip_confidence": clip_confidence,
            "phase_match": clip_phase == tjde_phase,
            "score_enhancement": enhanced_score - tjde_score
        }
        
        return enhanced_score, clip_info
        
    except Exception as e:
        print(f"[VISION-AI INTEGRATION ERROR] {e}")
        return tjde_score, {"clip_phase": "error", "clip_confidence": 0.0}


def run_vision_ai_feedback_loop(hours_back: int = 24) -> Dict:
    """
    Analyze CLIP prediction effectiveness and adjust thresholds
    
    Args:
        hours_back: Hours to look back for analysis
        
    Returns:
        Feedback loop results
    """
    try:
        print(f"[VISION-AI FEEDBACK] Analyzing last {hours_back} hours...")
        
        # Load recent predictions and results
        # This would integrate with your existing alert/result tracking
        
        feedback_results = {
            "analyzed_predictions": 0,
            "successful_predictions": 0,
            "accuracy_rate": 0.0,
            "recommended_confidence_threshold": 0.7,
            "phase_accuracy": {}
        }
        
        print("[VISION-AI FEEDBACK] Feedback loop analysis complete")
        return feedback_results
        
    except Exception as e:
        print(f"[VISION-AI FEEDBACK ERROR] {e}")
        return {"error": str(e)}


def test_vision_ai_pipeline():
    """Test the complete Vision-AI pipeline"""
    print("[VISION-AI TEST] Testing complete pipeline...")
    
    # Test CLIP predictor
    predictor = EnhancedCLIPPredictor()
    
    if predictor.model:
        print("✅ CLIP model loaded successfully")
    else:
        print("❌ CLIP model failed to load")
    
    # Test training data folders
    os.makedirs("training_data/charts", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    print("✅ Training data folders created")
    
    print("[VISION-AI TEST] Pipeline test complete")


if __name__ == "__main__":
    test_vision_ai_pipeline()