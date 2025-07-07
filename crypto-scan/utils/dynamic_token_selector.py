"""
Stage 1.5 - Dynamic Token Selector v2.0
Inteligentny selektor tokenów po fast scan z adaptacyjną logiką + samouczenie się progu

Zastępuje sztywny próg basic_score > 0.35 z dynamicznym threshold
bazującym na max score, kontekście rynkowym i samouczącym się progu z feedback loop.
"""

import json
import os
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np

try:
    from feedback_loop.adaptive_threshold_integration import get_dynamic_selection_threshold
    ADAPTIVE_THRESHOLD_AVAILABLE = True
except ImportError:
    print("[DYNAMIC SELECTOR] Adaptive threshold not available, using fallback logic")
    ADAPTIVE_THRESHOLD_AVAILABLE = False

class DynamicTokenSelector:
    """
    Inteligentny selektor tokenów z adaptacyjną logiką
    """
    
    def __init__(self):
        self.sentry_cutoff = 0.25  # Minimalny sensowny próg bezpieczeństwa
        self.max_tokens_limit = 40  # Maksymalnie tokenów do zaawansowanej analizy
        self.stats_file = "data/selection_statistics.json"
        
    def select_top_basic_candidates(self, all_basic_results: List[Dict]) -> List[Dict]:
        """
        🎯 Stage 1.5 - Dynamiczny selektor najlepszych tokenów po fast scan
        
        Args:
            all_basic_results: Lista wszystkich wyników z simulate_trader_decision_basic()
            
        Returns:
            Lista wybranych tokenów do zaawansowanej analizy (AI-EYE, HTF, CLIP)
        """
        if not all_basic_results:
            print("[DYNAMIC SELECTOR] No basic results to process")
            return []
            
        # Filter out invalid results - support both 'score' and 'basic_score' keys
        valid_results = []
        for r in all_basic_results:
            if not r:
                continue
            score = r.get('score', r.get('basic_score', 0))
            if score > 0:
                # Normalize to use 'score' key consistently
                r_normalized = r.copy()
                r_normalized['score'] = score
                valid_results.append(r_normalized)
        
        if not valid_results:
            print("[DYNAMIC SELECTOR] No valid basic results found")
            return []
            
        print(f"[DYNAMIC SELECTOR] Processing {len(valid_results)} valid results from {len(all_basic_results)} total")
        
        # Phase 1: Safety cutoff - remove garbage tokens
        candidates = [r for r in valid_results if r.get('score', 0) >= self.sentry_cutoff]
        
        if not candidates:
            print(f"[DYNAMIC SELECTOR] No tokens passed sentry cutoff ({self.sentry_cutoff})")
            return []
            
        print(f"[DYNAMIC SELECTOR] {len(candidates)} tokens passed sentry cutoff")
        
        # Phase 2: Market context analysis
        scores = [r.get('score', 0) for r in candidates]
        max_score = max(scores)
        min_score = min(scores)
        avg_score = np.mean(scores)
        
        print(f"[MARKET CONTEXT] Max: {max_score:.3f}, Min: {min_score:.3f}, Avg: {avg_score:.3f}")
        
        # Phase 3: Dynamic threshold strategy
        strategy, threshold, selected_tokens = self._apply_selection_strategy(candidates, max_score, avg_score)
        
        # Phase 4: Token limit enforcement
        if len(selected_tokens) > self.max_tokens_limit:
            print(f"[LIMIT ENFORCEMENT] Reducing from {len(selected_tokens)} to {self.max_tokens_limit} tokens")
            selected_tokens = sorted(selected_tokens, key=lambda x: x.get('score', 0), reverse=True)[:self.max_tokens_limit]
            
        # Phase 5: Statistics logging for future ML optimization
        self._log_selection_statistics(
            total_candidates=len(candidates),
            selected_count=len(selected_tokens),
            max_score=max_score,
            strategy=strategy,
            threshold=threshold
        )
        
        selected_symbols = [t.get('symbol', 'UNKNOWN') for t in selected_tokens]
        print(f"[DYNAMIC SELECTOR] Final selection: {len(selected_tokens)} tokens")
        print(f"[SELECTED TOKENS] {', '.join(selected_symbols[:10])}{'...' if len(selected_symbols) > 10 else ''}")
        
        return selected_tokens
        
    def _apply_selection_strategy(self, candidates: List[Dict], max_score: float, avg_score: float) -> Tuple[str, float, List[Dict]]:
        """
        Stosuje odpowiednią strategię selekcji na podstawie kontekstu rynkowego
        
        Returns:
            (strategy_name, threshold_used, selected_tokens)
        """
        
        # Strategy 1: HIGH-QUALITY MARKET (max_score >= 0.5) - with adaptive learning
        if max_score >= 0.5:
            # Try to use adaptive threshold with learned intelligence
            if ADAPTIVE_THRESHOLD_AVAILABLE:
                try:
                    adaptive_threshold = get_dynamic_selection_threshold(max_score, self.sentry_cutoff)
                    threshold = adaptive_threshold
                    print(f"[ADAPTIVE THRESHOLD] Using learned threshold: {threshold:.3f}")
                except Exception as e:
                    print(f"[ADAPTIVE THRESHOLD ERROR] Fallback to 70% threshold: {e}")
                    threshold = max(0.7 * max_score, self.sentry_cutoff)
            else:
                # Fallback to traditional 70% threshold
                threshold = max(0.7 * max_score, self.sentry_cutoff)
            
            selected = [r for r in candidates if r.get('score', 0) >= threshold]
            
            # Limit to best 15 tokens in high-quality markets (more selective)
            if len(selected) > 15:
                selected = sorted(selected, key=lambda x: x.get('score', 0), reverse=True)[:15]
                
            strategy_name = "HIGH-QUALITY-ADAPTIVE" if ADAPTIVE_THRESHOLD_AVAILABLE else "HIGH-QUALITY"
            print(f"[SELECTION STRATEGY] {strategy_name}: threshold={threshold:.3f}, selected={len(selected)}")
            return strategy_name, threshold, selected
            
        # Strategy 2: MODERATE MARKET (0.25 <= max_score < 0.5)
        elif max_score >= 0.25:
            # Use top 10% or best 20 tokens, whichever is smaller
            top_10_percent = max(1, int(0.1 * len(candidates)))
            top_n = min(top_10_percent, 20)
            
            selected = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)[:top_n]
            threshold = selected[-1].get('score', 0) if selected else 0
            
            print(f"[SELECTION STRATEGY] MODERATE MARKET: top {top_n} tokens, qualified={len(selected)}")
            return "MODERATE", threshold, selected
            
        # Strategy 3: WEAK MARKET (max_score < 0.25)
        else:
            # Use adaptive threshold based on statistical distribution
            scores = [r.get('score', 0) for r in candidates]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Adaptive threshold: mean + 0.5 * standard deviation
            adaptive_threshold = max(mean_score + 0.5 * std_score, self.sentry_cutoff)
            selected = [r for r in candidates if r.get('score', 0) >= adaptive_threshold]
            
            # In weak markets, limit to max 30 tokens to avoid noise
            if len(selected) > 30:
                selected = sorted(selected, key=lambda x: x.get('score', 0), reverse=True)[:30]
                
            print(f"[SELECTION STRATEGY] WEAK MARKET: adaptive_threshold={adaptive_threshold:.3f}")
            print(f"  📈 Mean: {mean_score:.3f}, StdDev: {std_score:.3f}, Selected: {len(selected)}")
            return "WEAK", adaptive_threshold, selected
            
    def _log_selection_statistics(self, total_candidates: int, selected_count: int, 
                                 max_score: float, strategy: str, threshold: float):
        """
        Zapisuje statystyki selekcji dla przyszłego machine learning i optymalizacji
        """
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "total_candidates": total_candidates,
                "selected_count": selected_count,
                "selection_ratio": selected_count / total_candidates if total_candidates > 0 else 0,
                "max_score": max_score,
                "threshold_used": threshold,
                "strategy": strategy
            }
            
            # Load existing statistics
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    all_stats = json.load(f)
            else:
                all_stats = []
                
            # Add new stats
            all_stats.append(stats)
            
            # Keep last 1000 entries
            if len(all_stats) > 1000:
                all_stats = all_stats[-1000:]
                
            # Save updated statistics
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump(all_stats, f, indent=2)
                
            print(f"[SELECTION STATS] Saved: {strategy} strategy, {selected_count}/{total_candidates} selected")
            
        except Exception as e:
            print(f"[SELECTION STATS] Error saving statistics: {e}")
            
    def get_selection_analytics(self) -> Dict:
        """
        Pobiera analitykę historycznych selekcji dla optymalizacji
        
        Returns:
            Dictionary z analizą performance różnych strategii
        """
        try:
            if not os.path.exists(self.stats_file):
                return {"error": "No selection statistics available"}
                
            with open(self.stats_file, 'r') as f:
                all_stats = json.load(f)
                
            if not all_stats:
                return {"error": "Empty statistics file"}
                
            # Analyze last 100 selections
            recent_stats = all_stats[-100:]
            
            strategies = {}
            for stat in recent_stats:
                strategy = stat.get('strategy', 'UNKNOWN')
                if strategy not in strategies:
                    strategies[strategy] = {
                        'count': 0,
                        'avg_selection_ratio': 0,
                        'avg_max_score': 0,
                        'avg_threshold': 0
                    }
                    
                strategies[strategy]['count'] += 1
                strategies[strategy]['avg_selection_ratio'] += stat.get('selection_ratio', 0)
                strategies[strategy]['avg_max_score'] += stat.get('max_score', 0)
                strategies[strategy]['avg_threshold'] += stat.get('threshold_used', 0)
                
            # Calculate averages
            for strategy_data in strategies.values():
                count = strategy_data['count']
                if count > 0:
                    strategy_data['avg_selection_ratio'] /= count
                    strategy_data['avg_max_score'] /= count
                    strategy_data['avg_threshold'] /= count
                    
            return {
                "total_selections": len(recent_stats),
                "strategies": strategies,
                "latest_selection": recent_stats[-1] if recent_stats else None
            }
            
        except Exception as e:
            return {"error": f"Analytics error: {e}"}

# Global instance for easy import
dynamic_selector = DynamicTokenSelector()

def select_top_basic_candidates(all_basic_results: List[Dict]) -> List[Dict]:
    """
    Convenience function for easy import and use
    """
    return dynamic_selector.select_top_basic_candidates(all_basic_results)