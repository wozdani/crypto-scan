#!/usr/bin/env python3
"""
Dynamic Selection Adapter - Future Feedback Loop Integration
Adapts selection thresholds based on historical performance and market conditions
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np


class DynamicSelectionAdapter:
    """
    Adapter for optimizing token selection thresholds based on feedback data
    """
    
    def __init__(self, feedback_dir: str = 'feedback_loop', stats_dir: str = 'data'):
        self.feedback_dir = feedback_dir
        self.stats_dir = stats_dir
        self.thresholds_file = f'{feedback_dir}/base_thresholds.json'
        self.stats_file = f'{stats_dir}/selection_statistics.json'
        
    def analyze_selection_performance(self, days_back: int = 7) -> Dict:
        """
        Analyze historical selection performance to optimize thresholds
        """
        try:
            if not os.path.exists(self.stats_file):
                return self._get_default_analysis()
            
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            
            # Filter recent data
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_stats = [
                s for s in stats 
                if datetime.fromisoformat(s['timestamp']) > cutoff_date
            ]
            
            if not recent_stats:
                return self._get_default_analysis()
            
            # Analysis metrics
            analysis = {
                'total_cycles': len(recent_stats),
                'strategies': self._analyze_strategies(recent_stats),
                'score_distribution': self._analyze_score_distribution(recent_stats),
                'selection_efficiency': self._analyze_selection_efficiency(recent_stats),
                'recommendations': self._generate_recommendations(recent_stats)
            }
            
            print(f"[SELECTION ANALYSIS] Analyzed {len(recent_stats)} cycles over {days_back} days")
            return analysis
            
        except Exception as e:
            print(f"[SELECTION ANALYSIS] Error: {e}")
            return self._get_default_analysis()
    
    def _analyze_strategies(self, stats: List[Dict]) -> Dict:
        """Analyze strategy usage and effectiveness"""
        strategy_counts = {}
        strategy_selections = {}
        
        for stat in stats:
            strategy = stat['strategy_used']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            if strategy not in strategy_selections:
                strategy_selections[strategy] = []
            strategy_selections[strategy].append(stat['selection_ratio'])
        
        strategy_analysis = {}
        for strategy, count in strategy_counts.items():
            ratios = strategy_selections[strategy]
            strategy_analysis[strategy] = {
                'usage_count': count,
                'usage_percentage': count / len(stats) * 100,
                'avg_selection_ratio': np.mean(ratios),
                'selection_consistency': 1.0 - np.std(ratios) if len(ratios) > 1 else 1.0
            }
        
        return strategy_analysis
    
    def _analyze_score_distribution(self, stats: List[Dict]) -> Dict:
        """Analyze score distribution patterns"""
        top_scores = [stat['top_score'] for stat in stats]
        
        return {
            'mean_top_score': np.mean(top_scores),
            'std_top_score': np.std(top_scores),
            'median_top_score': np.median(top_scores),
            'score_trend': self._calculate_trend(top_scores),
            'high_quality_days': len([s for s in top_scores if s >= 0.5]),
            'moderate_days': len([s for s in top_scores if 0.25 <= s < 0.5]),
            'weak_days': len([s for s in top_scores if s < 0.25])
        }
    
    def _analyze_selection_efficiency(self, stats: List[Dict]) -> Dict:
        """Analyze selection efficiency metrics"""
        ratios = [stat['selection_ratio'] for stat in stats]
        selected_counts = [stat['selected_count'] for stat in stats]
        
        return {
            'avg_selection_ratio': np.mean(ratios),
            'selection_stability': 1.0 - np.std(ratios),
            'avg_selected_count': np.mean(selected_counts),
            'optimal_range': self._find_optimal_selection_range(stats)
        }
    
    def _generate_recommendations(self, stats: List[Dict]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze strategy distribution
        strategies = self._analyze_strategies(stats)
        
        if strategies.get('WEAK', {}).get('usage_percentage', 0) > 60:
            recommendations.append("Consider lowering base thresholds - market shows consistent weakness")
        
        if strategies.get('HIGH-QUALITY', {}).get('usage_percentage', 0) > 50:
            recommendations.append("Consider raising thresholds - market shows strong performance")
        
        # Analyze selection efficiency
        efficiency = self._analyze_selection_efficiency(stats)
        
        if efficiency['avg_selection_ratio'] > 0.15:
            recommendations.append("Selection ratio high - consider more selective criteria")
        elif efficiency['avg_selection_ratio'] < 0.05:
            recommendations.append("Selection ratio low - consider relaxing criteria")
        
        if not recommendations:
            recommendations.append("Current selection parameters appear optimal")
        
        return recommendations
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple trend analysis
        recent_avg = np.mean(values[-3:])
        older_avg = np.mean(values[:-3])
        
        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"
    
    def _find_optimal_selection_range(self, stats: List[Dict]) -> Dict:
        """Find optimal selection range based on historical data"""
        ratios = [stat['selection_ratio'] for stat in stats]
        
        if not ratios:
            return {'min': 0.05, 'max': 0.15, 'optimal': 0.10}
        
        q25 = np.percentile(ratios, 25)
        q75 = np.percentile(ratios, 75)
        median = np.median(ratios)
        
        return {
            'min': max(0.02, q25),
            'max': min(0.25, q75),
            'optimal': median
        }
    
    def _get_default_analysis(self) -> Dict:
        """Default analysis when no data available"""
        return {
            'total_cycles': 0,
            'strategies': {},
            'score_distribution': {
                'mean_top_score': 0.3,
                'std_top_score': 0.1,
                'trend': 'stable'
            },
            'selection_efficiency': {
                'avg_selection_ratio': 0.1,
                'optimal_range': {'min': 0.05, 'max': 0.15, 'optimal': 0.10}
            },
            'recommendations': ['No historical data - using default parameters']
        }
    
    def update_adaptive_thresholds(self) -> Dict:
        """
        Update adaptive thresholds based on performance analysis
        """
        analysis = self.analyze_selection_performance()
        
        # Load current thresholds
        current_thresholds = self._load_current_thresholds()
        
        # Calculate new thresholds based on analysis
        new_thresholds = self._calculate_adaptive_thresholds(analysis, current_thresholds)
        
        # Save updated thresholds
        self._save_thresholds(new_thresholds)
        
        print(f"[ADAPTIVE THRESHOLDS] Updated based on {analysis['total_cycles']} cycles")
        return new_thresholds
    
    def _load_current_thresholds(self) -> Dict:
        """Load current threshold configuration"""
        try:
            if os.path.exists(self.thresholds_file):
                with open(self.thresholds_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        
        # Default thresholds
        return {
            'high_quality_threshold': 0.5,
            'moderate_threshold': 0.25,
            'weak_market_multiplier': 0.7,
            'last_updated': None
        }
    
    def _calculate_adaptive_thresholds(self, analysis: Dict, current: Dict) -> Dict:
        """Calculate new adaptive thresholds"""
        score_dist = analysis['score_distribution']
        
        # Adaptive high quality threshold
        if score_dist['score_trend'] == 'improving':
            high_threshold = min(0.7, current['high_quality_threshold'] * 1.1)
        elif score_dist['score_trend'] == 'declining':
            high_threshold = max(0.3, current['high_quality_threshold'] * 0.9)
        else:
            high_threshold = current['high_quality_threshold']
        
        # Adaptive moderate threshold
        moderate_threshold = max(0.15, score_dist['mean_top_score'] * 0.5)
        
        # Adaptive weak market multiplier
        efficiency = analysis['selection_efficiency']
        if efficiency['avg_selection_ratio'] > 0.15:
            weak_multiplier = max(0.5, current['weak_market_multiplier'] * 0.9)
        else:
            weak_multiplier = min(0.8, current['weak_market_multiplier'] * 1.1)
        
        return {
            'high_quality_threshold': round(high_threshold, 3),
            'moderate_threshold': round(moderate_threshold, 3),
            'weak_market_multiplier': round(weak_multiplier, 3),
            'last_updated': datetime.now().isoformat(),
            'analysis_summary': {
                'cycles_analyzed': analysis['total_cycles'],
                'score_trend': score_dist['score_trend'],
                'avg_selection_ratio': efficiency['avg_selection_ratio']
            }
        }
    
    def _save_thresholds(self, thresholds: Dict):
        """Save updated thresholds to file"""
        os.makedirs(self.feedback_dir, exist_ok=True)
        with open(self.thresholds_file, 'w') as f:
            json.dump(thresholds, f, indent=2)


def analyze_selection_performance(days_back: int = 7) -> Dict:
    """Convenience function for analyzing selection performance"""
    adapter = DynamicSelectionAdapter()
    return adapter.analyze_selection_performance(days_back)


def update_adaptive_thresholds() -> Dict:
    """Convenience function for updating adaptive thresholds"""
    adapter = DynamicSelectionAdapter()
    return adapter.update_adaptive_thresholds()


if __name__ == "__main__":
    # Test the adapter
    adapter = DynamicSelectionAdapter()
    analysis = adapter.analyze_selection_performance()
    
    print("📊 Selection Performance Analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Update thresholds
    thresholds = adapter.update_adaptive_thresholds()
    print("\n🎯 Updated Adaptive Thresholds:")
    print(json.dumps(thresholds, indent=2))