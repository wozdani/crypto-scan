#!/usr/bin/env python3
"""
Cluster Analysis Enhancement Module
Enhanced volume cluster analysis with pattern detection and quality scoring
"""

def cluster_analysis_enhancement(symbol, market_data, debug=False):
    """
    Enhanced cluster analysis with volume pattern detection and quality scoring
    
    Args:
        symbol: Trading symbol
        market_data: Dictionary containing candles, orderbook, price data
        debug: Enable detailed debug logging (default: False)
        
    Returns:
        tuple: (modifier, quality_score)
    """
    print(f"[CLUSTER ENTRY] Starting cluster_analysis_enhancement for {symbol}")
    
    try:
        # Extract data from market_data dictionary
        candles_15m = market_data.get('candles_15m', market_data.get('candles', []))
        orderbook_data = market_data.get('orderbook', {})
        price_usd = market_data.get('price_usd', market_data.get('price', 0))
        
        # Enhanced debug logging
        print(f"[CLUSTER DEBUG] {symbol}")
        print(f"- Input validation: {len(candles_15m) if candles_15m else 0} candles")
        print(f"- Orderbook available: {bool(orderbook_data)}")
        print(f"- Price: {price_usd}")
        
        if candles_15m and len(candles_15m) > 0:
            first_candle = candles_15m[0]
            if isinstance(first_candle, dict):
                print(f"[CLUSTER DEBUG] Candle format: dict, Keys: {list(first_candle.keys())}")
            elif isinstance(first_candle, list):
                print(f"[CLUSTER DEBUG] Candle format: list, Length: {len(first_candle)}")
            else:
                print(f"[CLUSTER DEBUG] Candle format: {type(first_candle)}")
        else:
            print(f"[CLUSTER DEBUG] No candles available for format check")
        
        if not candles_15m or len(candles_15m) < 20:
            print(f"[DATA WARNING] {symbol} has insufficient candles: {len(candles_15m) if candles_15m else 0} (need ≥20)")
            if debug:
                print(f"- Insufficient data: Need ≥20 candles, got {len(candles_15m) if candles_15m else 0}")
                print(f"- Final Modifier: 0.000, Quality: 0.500")
            return 0.0, 0.5

        # Calculate volume clusters and patterns
        volume_clusters = []
        recent_candles = candles_15m[-20:]  # Last 20 candles
        
        if debug:
            print(f"- Processing recent candles: {len(recent_candles)}")
        
        for i, candle in enumerate(recent_candles):
            volume = float(candle[5])
            price = float(candle[4])  # Close price
            
            if volume > 0:
                volume_clusters.append({
                    'volume': volume,
                    'price': price,
                    'index': i,
                    'timestamp': candle[0]
                })
        
        if debug:
            print(f"- Detected volume clusters: {len(volume_clusters)}")
            if volume_clusters:
                avg_volume = sum(c['volume'] for c in volume_clusters) / len(volume_clusters)
                print(f"- Average volume in clusters: {avg_volume:.0f}")

        # Analyze volume cluster density and distribution
        if len(volume_clusters) < 5:
            if debug:
                print(f"- Insufficient clusters: Need ≥5, got {len(volume_clusters)}")
                print(f"- Fallback triggered: No valid volume pattern found")
                print(f"- Final Modifier: 0.000, Quality: 0.500")
            return 0.0, 0.5
        
        # Calculate average cluster density
        total_volume = sum(c['volume'] for c in volume_clusters)
        avg_volume = total_volume / len(volume_clusters)
        
        # Find high-volume clusters (above 150% of average)
        high_volume_clusters = [c for c in volume_clusters if c['volume'] > avg_volume * 1.5]
        cluster_density = len(high_volume_clusters) / len(volume_clusters)
        
        if debug:
            print(f"- High-volume clusters: {len(high_volume_clusters)}/{len(volume_clusters)}")
            print(f"- Avg cluster density: {cluster_density:.3f}")
            print(f"- Volume threshold (150% avg): {avg_volume * 1.5:.0f}")

        # Calculate cluster slope (trend direction)
        if len(high_volume_clusters) >= 2:
            price_points = [c['price'] for c in high_volume_clusters]
            time_points = list(range(len(price_points)))
            
            # Simple linear regression for slope
            n = len(price_points)
            sum_x = sum(time_points)
            sum_y = sum(price_points)
            sum_xy = sum(x * y for x, y in zip(time_points, price_points))
            sum_x2 = sum(x * x for x in time_points)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                cluster_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            else:
                cluster_slope = 0.0
                
            if debug:
                print(f"- Price points for slope: {len(price_points)} points")
                print(f"- Cluster slope: {cluster_slope:.6f}")
        else:
            cluster_slope = 0.0
            if debug:
                print(f"- Insufficient points for slope calculation: {len(high_volume_clusters)} clusters")
                print(f"- Cluster slope: 0.000 (default)")

        # Check for volume peaks near support/resistance levels
        peak_near_support = False
        support_levels = []
        
        if orderbook_data and 'bids' in orderbook_data:
            support_levels = [bid['price'] for bid in orderbook_data['bids'][:3]]
            
            if debug:
                print(f"- Support levels from orderbook: {len(support_levels)}")
                if support_levels:
                    print(f"- Top 3 bid levels: {[f'{level:.6f}' for level in support_levels[:3]]}")
            
            for cluster in high_volume_clusters:
                for support in support_levels:
                    if abs(cluster['price'] - support) / support < 0.02:  # Within 2%
                        peak_near_support = True
                        if debug:
                            print(f"- Found volume peak near support: cluster@{cluster['price']:.6f} vs support@{support:.6f}")
                        break
                if peak_near_support:
                    break
        else:
            if debug:
                print(f"- No orderbook data available for support level analysis")
        
        if debug:
            print(f"- Peak near support: {bool(peak_near_support)}")

        # Calculate pattern score based on multiple factors
        pattern_score = 0.0
        score_breakdown = {}
        
        # Density factor (higher density = better)
        density_score = 0.0
        if cluster_density > 0.3:
            density_score = 0.3
            pattern_score += density_score
        score_breakdown['density'] = density_score
        
        # Slope factor (positive slope = accumulation)
        slope_score = 0.0
        if cluster_slope > 0:
            slope_score = min(0.4, cluster_slope * 1000)  # Scale slope
            pattern_score += slope_score
        score_breakdown['slope'] = slope_score
        
        # Support factor
        support_score = 0.3 if peak_near_support else 0.0
        pattern_score += support_score
        score_breakdown['support'] = support_score
        
        # Volume consistency factor
        volume_std = (sum((c['volume'] - avg_volume) ** 2 for c in volume_clusters) / len(volume_clusters)) ** 0.5
        volume_cv = volume_std / avg_volume if avg_volume > 0 else 1.0
        
        consistency_score = 0.0
        if volume_cv < 0.5:  # Low coefficient of variation = consistent volume
            consistency_score = 0.2
            pattern_score += consistency_score
        score_breakdown['consistency'] = consistency_score
        
        if debug:
            print(f"- Score breakdown:")
            print(f"  * Density ({cluster_density:.3f} > 0.3): +{density_score:.3f}")
            print(f"  * Slope ({cluster_slope:.6f} > 0): +{slope_score:.3f}")
            print(f"  * Support (near levels): +{support_score:.3f}")
            print(f"  * Consistency (CV {volume_cv:.3f} < 0.5): +{consistency_score:.3f}")
            print(f"- Pattern score: {pattern_score:.3f}")

        # Determine final modifier and quality
        pattern_found = pattern_score >= 0.3
        
        if not pattern_found:
            if debug:
                print(f"- Pattern threshold check: {pattern_score:.3f} < 0.3 (minimum)")
                print(f"- Fallback triggered: No valid volume pattern found")
                print(f"- Final Modifier: 0.000, Quality: 0.500")
            return 0.0, 0.5  # Weak pattern
        
        # Scale modifier based on pattern strength
        modifier = min(0.15, pattern_score * 0.5)  # Max 0.15 boost
        quality = min(1.0, 0.5 + pattern_score)  # Quality score 0.5-1.0
        
        if debug:
            print(f"- Pattern validation: PASSED ({pattern_score:.3f} ≥ 0.3)")
            print(f"- Modifier calculation: min(0.15, {pattern_score:.3f} * 0.5) = {modifier:.3f}")
            print(f"- Quality calculation: min(1.0, 0.5 + {pattern_score:.3f}) = {quality:.3f}")
            print(f"- Final Modifier: {modifier:.3f}, Quality: {quality:.3f}")
        
        # Check for fallback values before returning
        if modifier == 0.0 and quality == 0.5:
            print(f"[CLUSTER FALLBACK] {symbol}: Default values returned - insufficient data or calculation failed")
            print(f"[CLUSTER FALLBACK] Candles: {len(candles_15m) if candles_15m else 0}, Orderbook: {bool(orderbook_data)}")
        
        return modifier, quality
        
    except Exception as e:
        print(f"[CLUSTER ERROR] {symbol}: Unexpected exception: {e}")
        print(f"[CLUSTER FALLBACK] {symbol}: Exception triggered fallback (Modifier: 0.000, Quality: 0.500)")
        import traceback
        print(f"[CLUSTER TRACEBACK] {traceback.format_exc()}")
        return 0.0, 0.5

def test_cluster_analysis_debug():
    """Test cluster analysis with different scenarios"""
    print("🔍 TESTING CLUSTER ANALYSIS DEBUG MODE")
    print("=" * 60)
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    for scenario_name, (candles, orderbook, price) in scenarios.items():
        print(f"\n📊 SCENARIO: {scenario_name.upper()}")
        print("-" * 40)
        
        modifier, quality = cluster_analysis_enhancement(
            symbol=f"TEST_{scenario_name.upper()}", 
            candles_15m=candles,
            orderbook_data=orderbook,
            price_usd=price,
            debug=True
        )
        
        print(f"\n🎯 SCENARIO RESULT: Modifier {modifier:.3f}, Quality {quality:.3f}")
        print("-" * 40)

def create_test_scenarios():
    """Create different test scenarios for cluster analysis"""
    
    # Scenario 1: Strong volume pattern
    strong_pattern_candles = []
    base_price = 50000
    for i in range(25):
        # Ascending price with increasing volume clusters
        price = base_price + (i * 20)
        volume = 800000 + (i * 80000) + (i % 5 * 200000)  # Strong clusters
        candle = [1640995200000 + (i * 900000), price-10, price+15, price-15, price, volume]
        strong_pattern_candles.append(candle)
    
    # Scenario 2: Weak volume pattern  
    weak_pattern_candles = []
    for i in range(25):
        price = base_price + (i % 3 * 5)  # Sideways movement
        volume = 500000 + (i % 2 * 50000)  # Low, inconsistent volume
        candle = [1640995200000 + (i * 900000), price-3, price+3, price-3, price, volume]
        weak_pattern_candles.append(candle)
    
    # Scenario 3: Insufficient data
    insufficient_candles = strong_pattern_candles[:10]  # Only 10 candles
    
    orderbook = {
        'bids': [
            {'price': base_price - 5, 'size': 1000},
            {'price': base_price - 25, 'size': 1500}, 
            {'price': base_price - 50, 'size': 2000}
        ]
    }
    
    return {
        'strong_pattern': (strong_pattern_candles, orderbook, base_price),
        'weak_pattern': (weak_pattern_candles, orderbook, base_price),
        'insufficient_data': (insufficient_candles, orderbook, base_price)
    }

if __name__ == "__main__":
    test_cluster_analysis_debug()