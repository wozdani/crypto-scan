"""
Phase 1: Perception Synchronization - Clean Implementation
CLIP + TJDE + GPT Integration for Master-Level Market Perception
"""

import os
import json
import glob
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


def load_clip_prediction(symbol: str) -> Optional[Dict]:
    """Load CLIP prediction for symbol from multiple potential sources"""
    alert_time = datetime.now().strftime('%Y%m%d_%H%M')
    
    potential_paths = [
        f"training_charts/{symbol}_{alert_time}_clip.json",
        f"training_charts/{symbol}_clip.json", 
        f"data/clip_predictions/{symbol}_latest.json",
        f"data/clip_predictions/{symbol}.json"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    print(f"[CLIP LOADER] {symbol}: Loaded from {path}")
                    return data
            except Exception as e:
                print(f"[CLIP LOADER ERROR] {symbol}: Failed to load {path}: {e}")
    
    print(f"[CLIP LOADER] {symbol}: No prediction found")
    return None


def calculate_enhanced_tjde_score(features: Dict) -> float:
    """Enhanced TJDE scoring with CLIP feature integration"""
    # Base TJDE components
    trend_score = features.get("trend_strength", 0.0) * 0.25
    pullback_score = features.get("pullback_quality", 0.0) * 0.20
    support_score = features.get("support_reaction_strength", 0.0) * 0.15
    volume_score = features.get("volume_behavior_score", 0.0) * 0.15
    psych_score = features.get("psych_score", 0.0) * 0.10
    orderbook_score = features.get("orderbook_alignment", 0.0) * 0.15
    
    base_score = trend_score + pullback_score + support_score + volume_score + psych_score + orderbook_score
    
    # CLIP enhancement
    clip_confidence = features.get("clip_confidence", 0.0)
    if clip_confidence > 0.4:
        clip_boost = 0.05 * clip_confidence  # Up to 5% boost for high confidence
        base_score += clip_boost
        print(f"[ENHANCED SCORING] CLIP boost: +{clip_boost:.3f} (confidence: {clip_confidence:.3f})")
    
    return min(1.0, max(0.0, base_score))


def find_recent_chart(symbol: str) -> Optional[str]:
    """Find most recent chart for symbol in training_charts directory"""
    pattern = f"training_charts/{symbol}_*.png"
    charts = glob.glob(pattern)
    
    if charts:
        # Return most recent by filename (timestamp-based)
        return sorted(charts, reverse=True)[0]
    
    return None


def generate_gpt_chart_comment(symbol: str, features: Dict, clip_prediction: Optional[Dict], chart_path: str) -> str:
    """Generate GPT commentary on chart + features + CLIP prediction"""
    try:
        from gpt_commentary import generate_chart_commentary
        
        # Prepare context for GPT
        tjde_score = features.get("combined_score", 0.0)
        decision_hint = "consider_entry" if tjde_score > 0.6 else "avoid"
        
        # Generate comprehensive commentary
        commentary = generate_chart_commentary(
            chart_path, 
            tjde_score, 
            decision_hint,
            clip_prediction,
            symbol
        )
        
        if commentary:
            print(f"[GPT COMMENTARY] {symbol}: Generated chart analysis")
            return commentary
        else:
            print(f"[GPT COMMENTARY] {symbol}: Generation failed")
            return f"Technical analysis for {symbol}: Score {tjde_score:.3f}, monitoring market structure"
            
    except Exception as e:
        print(f"[GPT COMMENTARY ERROR] {symbol}: {e}")
        return f"Chart analysis for {symbol}: TJDE score {features.get('combined_score', 0):.3f}"


def rename_chart_with_gpt_insights(chart_path: str, gpt_comment: str, features: Dict) -> Optional[str]:
    """Rename chart file based on GPT analysis insights"""
    try:
        # Extract key insights from GPT comment for filename
        setup_hint = "unknown"
        decision_hint = "neutral"
        
        comment_lower = gpt_comment.lower()
        
        # Identify setup type from GPT comment
        if "pullback" in comment_lower or "cofnięcie" in comment_lower:
            setup_hint = "pullback"
        elif "breakout" in comment_lower or "wybicie" in comment_lower:
            setup_hint = "breakout"
        elif "support" in comment_lower or "wsparcie" in comment_lower:
            setup_hint = "support"
        elif "trend" in comment_lower:
            setup_hint = "trend-following"
        elif "consolidation" in comment_lower or "konsolidacja" in comment_lower:
            setup_hint = "consolidation"
        
        # Identify decision from features and GPT analysis
        tjde_score = features.get("combined_score", 0.0)
        if tjde_score > 0.7:
            decision_hint = "consider_entry"
        elif tjde_score > 0.5:
            decision_hint = "monitor"
        else:
            decision_hint = "avoid"
        
        # Build new filename
        base_name = os.path.basename(chart_path)
        parts = base_name.split('_')
        
        if len(parts) >= 4:
            symbol = parts[0]
            timestamp = parts[1] + '_' + parts[2]
            new_filename = f"{symbol}_{timestamp}_{setup_hint}_{decision_hint}_tjde.png"
            
            new_path = os.path.join(os.path.dirname(chart_path), new_filename)
            
            # Rename file
            shutil.move(chart_path, new_path)
            print(f"[CHART RENAME] {chart_path} → {new_filename}")
            
            return new_path
        
    except Exception as e:
        print(f"[CHART RENAME ERROR] {e}")
    
    return chart_path


def save_perception_metadata(symbol: str, metadata: Dict):
    """Save integrated CLIP + GPT + TJDE metadata"""
    try:
        os.makedirs("metadata", exist_ok=True)
        
        timestamp = metadata.get("timestamp", datetime.now().strftime('%Y%m%d_%H%M'))
        filename = f"metadata/{symbol}_{timestamp}.json"
        
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[PERCEPTION METADATA] Saved: {filename}")
        
    except Exception as e:
        print(f"[PERCEPTION METADATA ERROR] {symbol}: {e}")


def simulate_trader_decision_perception_sync(symbol: str, market_data: dict, signals: dict, debug_info: dict = None) -> dict:
    """
    Phase 1: Perception Synchronization Implementation
    
    Unified pipeline connecting CLIP predictions, TJDE scoring, and GPT interpretation
    for master-level market perception and decision making.
    """
    try:
        print(f"[PERCEPTION SYNC] {symbol}: Starting Phase 1 analysis")
        
        # === STEP 1: CLIP PREDICTION INTEGRATION ===
        clip_prediction = load_clip_prediction(symbol)
        clip_features = {}
        
        if clip_prediction and clip_prediction.get("clip_confidence", 0) > 0.4:
            clip_features["clip_trend_match"] = clip_prediction.get("trend_label", "unknown")
            clip_features["clip_confidence"] = clip_prediction.get("clip_confidence", 0.0)
            clip_features["clip_setup_type"] = clip_prediction.get("setup_type", "unknown")
            
            print(f"[CLIP INTEGRATION] {symbol}: {clip_features['clip_trend_match']} (confidence: {clip_features['clip_confidence']:.3f})")
        else:
            print(f"[CLIP INTEGRATION] {symbol}: No high-confidence prediction available")
        
        # === STEP 2: UNIFIED SCORING WITH CLIP FEATURES ===
        # Combine traditional TJDE features with CLIP insights
        all_features = {
            **signals,
            **clip_features,
            "symbol": symbol,
            "combined_score": 0.0  # Will be calculated
        }
        
        # Enhanced scoring with CLIP integration
        base_score = calculate_enhanced_tjde_score(all_features)
        all_features["combined_score"] = base_score
        
        # === STEP 3: GPT CHART COMMENTARY GENERATION ===
        gpt_comment = ""
        chart_path = find_recent_chart(symbol)
        
        if chart_path:
            gpt_comment = generate_gpt_chart_comment(symbol, all_features, clip_prediction, chart_path)
            
            # === STEP 4: RENAME CHART BASED ON GPT ANALYSIS ===
            if gpt_comment:
                new_chart_path = rename_chart_with_gpt_insights(chart_path, gpt_comment, all_features)
                if new_chart_path:
                    chart_path = new_chart_path
        
        # === STEP 5: FINAL DECISION LOGIC ===
        final_score = base_score
        decision = "avoid"
        quality_grade = "neutral-watch"
        
        if final_score >= 0.70:
            decision = "join_trend"
            quality_grade = "strong"
        elif final_score >= 0.45:
            decision = "consider_entry"
            quality_grade = "moderate"
        else:
            decision = "avoid"
            quality_grade = "weak"
        
        print(f"[PERCEPTION RESULT] {symbol}: {decision} (score: {final_score:.3f})")
        
        # === STEP 6: SAVE INTEGRATED METADATA ===
        metadata = {
            "symbol": symbol,
            "trend_label": clip_features.get("clip_trend_match", "unknown"),
            "setup_type": clip_features.get("clip_setup_type", "unknown"), 
            "clip_confidence": clip_features.get("clip_confidence", 0.0),
            "tjde_score": round(final_score, 3),
            "decision": decision,
            "gpt_comment": gpt_comment,
            "chart_path": chart_path,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M'),
            "perception_sync": True
        }
        
        save_perception_metadata(symbol, metadata)
        
        return {
            "combined_score": round(base_score, 3),
            "clip_features": clip_features,
            "final_score": round(final_score, 3),
            "decision": decision,
            "quality_grade": quality_grade,
            "gpt_comment": gpt_comment,
            "metadata": metadata,
            "perception_sync": True
        }
        
    except Exception as e:
        print(f"❌ [PERCEPTION SYNC ERROR] {symbol}: {e}")
        return {
            "decision": "avoid",
            "combined_score": 0.0,
            "final_score": 0.0,
            "quality_grade": "error",
            "perception_sync": False,
            "error": str(e)
        }


def test_perception_synchronization():
    """Test the complete Phase 1 Perception Synchronization system"""
    print('Testing Phase 1: Perception Synchronization...')
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create test environment
        test_symbol = 'PHASE1USDT'
        
        # 1. Create CLIP prediction file
        os.makedirs('data/clip_predictions', exist_ok=True)
        clip_data = {
            'trend_label': 'pullback',
            'setup_type': 'support-bounce',
            'clip_confidence': 0.72,
            'decision': 'consider_entry',
            'phase': 'trend-following'
        }
        
        with open(f'data/clip_predictions/{test_symbol}_latest.json', 'w') as f:
            json.dump(clip_data, f)
        
        # 2. Create test chart
        os.makedirs('training_charts', exist_ok=True)
        
        # Generate realistic pullback pattern
        x = np.arange(60)
        trend_base = 100 + x * 0.3  # Uptrend
        pullback = np.where(x > 40, trend_base - (x - 40) * 0.5, trend_base)  # Pullback after x=40
        noise = np.random.normal(0, 0.5, 60)
        price = pullback + noise
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(x, price, 'b-', linewidth=1.5, label='Price')
        plt.axvline(x=45, color='lime', linestyle='--', alpha=0.7, label='Support Bounce')
        plt.title(f'{test_symbol} - Pullback to Support Setup')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        volume = 1000 + np.random.randint(-200, 400, 60)
        volume[45:50] = (volume[45:50] * 1.5).astype(int)  # Volume spike at support
        plt.bar(x, volume, width=0.8, alpha=0.7, color='steelblue')
        plt.axvline(x=45, color='lime', linestyle='--', alpha=0.7)
        plt.ylabel('Volume')
        plt.xlabel('Time')
        plt.grid(True, alpha=0.3)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        chart_path = f'training_charts/{test_symbol}_{timestamp}_pullback_setup.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f'Test chart created: {chart_path}')
        
        # 3. Test enhanced TJDE with perception sync
        test_market_data = {
            'ticker': {'price': 110.5, 'volume': 15000},
            'candles_15m': [
                [1640995200000, 112.0, 113.0, 109.5, 110.5, 15000],
                [1640995800000, 110.5, 111.5, 109.8, 111.0, 16200]
            ]
        }
        
        test_signals = {
            'trend_strength': 0.75,
            'pullback_quality': 0.82,
            'support_reaction_strength': 0.88,
            'bounce_confirmation_strength': 0.75,
            'orderbook_alignment': 0.65,
            'volume_behavior_score': 0.78,
            'psych_score': 0.62
        }
        
        # Run enhanced TJDE with perception synchronization
        print(f'Running perception sync for {test_symbol}...')
        result = simulate_trader_decision_perception_sync(test_symbol, test_market_data, test_signals)
        
        print()
        print('PHASE 1 PERCEPTION SYNCHRONIZATION RESULTS:')
        print(f'  Symbol: {test_symbol}')
        print(f'  Perception Sync: {result.get("perception_sync", False)}')
        print(f'  Base Score: {result.get("combined_score", 0):.3f}')
        print(f'  Final Score: {result.get("final_score", 0):.3f}')
        print(f'  Decision: {result.get("decision", "unknown")}')
        print(f'  Quality: {result.get("quality_grade", "unknown")}')
        
        clip_features = result.get('clip_features', {})
        if clip_features:
            print(f'  CLIP Trend: {clip_features.get("clip_trend_match", "unknown")}')
            print(f'  CLIP Confidence: {clip_features.get("clip_confidence", 0):.3f}')
            print(f'  CLIP Setup: {clip_features.get("clip_setup_type", "unknown")}')
        
        gpt_comment = result.get('gpt_comment', '')
        if gpt_comment and len(gpt_comment) > 10:
            print(f'  GPT Comment: {gpt_comment[:80]}...')
        
        metadata = result.get('metadata', {})
        if metadata:
            print(f'  Metadata: {metadata.get("perception_sync", False)} sync enabled')
        
        # 4. Verify files were created
        metadata_files = glob.glob(f'metadata/{test_symbol}_*.json')
        if metadata_files:
            print(f'Metadata saved: {os.path.basename(metadata_files[0])}')
        
        # Clean up test files
        test_files = [
            f'data/clip_predictions/{test_symbol}_latest.json',
            chart_path
        ] + glob.glob(f'training_charts/{test_symbol}_{timestamp}_*.png') + metadata_files
        
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        print('Test files cleaned up')
        
        if result.get('perception_sync'):
            print('Phase 1 Perception Synchronization working correctly')
            return True
        else:
            print('Perception sync flag not set')
            return False
        
    except Exception as e:
        print(f'Phase 1 test error: {e}')
        return False


if __name__ == "__main__":
    test_perception_synchronization()