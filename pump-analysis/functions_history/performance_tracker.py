#!/usr/bin/env python3
"""
Performance Tracker
Tracks function performance across multiple pump events and provides feedback scoring
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class PerformanceResult:
    """Result of testing a function on pump data"""
    function_name: str
    test_symbol: str
    test_date: str
    detected: bool
    confidence_score: float
    execution_time_ms: float
    error_message: Optional[str] = None
    test_timestamp: str = None
    
    def __post_init__(self):
        if self.test_timestamp is None:
            self.test_timestamp = datetime.now().isoformat()

class PerformanceTracker:
    """Tracks and analyzes function performance across multiple test cases"""
    
    def __init__(self, base_dir: str = "functions_history"):
        self.base_dir = base_dir
        self.performance_db = os.path.join(base_dir, "performance_database.json")
        self.test_results = os.path.join(base_dir, "test_results.json")
        
        # Load existing data
        self.performance_data = self._load_performance_data()
        self.results_history = self._load_results_history()
    
    def _load_performance_data(self) -> Dict[str, Dict]:
        """Load performance database"""
        if os.path.exists(self.performance_db):
            try:
                with open(self.performance_db, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading performance data: {e}")
        return {}
    
    def _save_performance_data(self):
        """Save performance database"""
        try:
            with open(self.performance_db, 'w', encoding='utf-8') as f:
                json.dump(self.performance_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _load_results_history(self) -> List[Dict]:
        """Load test results history"""
        if os.path.exists(self.test_results):
            try:
                with open(self.test_results, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading results history: {e}")
        return []
    
    def _save_results_history(self):
        """Save test results history"""
        try:
            with open(self.test_results, 'w', encoding='utf-8') as f:
                json.dump(self.results_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving results history: {e}")
    
    def record_test_result(self, result: PerformanceResult):
        """Record a test result for a function"""
        
        # Add to results history
        result_dict = {
            'function_name': result.function_name,
            'test_symbol': result.test_symbol,
            'test_date': result.test_date,
            'detected': result.detected,
            'confidence_score': result.confidence_score,
            'execution_time_ms': result.execution_time_ms,
            'error_message': result.error_message,
            'test_timestamp': result.test_timestamp
        }
        
        self.results_history.append(result_dict)
        self._save_results_history()
        
        # Update performance database
        if result.function_name not in self.performance_data:
            self.performance_data[result.function_name] = {
                'total_tests': 0,
                'successful_detections': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'average_confidence': 0.0,
                'average_execution_time': 0.0,
                'last_tested': result.test_timestamp,
                'test_history': []
            }
        
        func_data = self.performance_data[result.function_name]
        func_data['total_tests'] += 1
        func_data['last_tested'] = result.test_timestamp
        func_data['test_history'].append(result_dict)
        
        # Update statistics
        if result.detected:
            func_data['successful_detections'] += 1
        
        # Calculate running averages
        all_confidences = [t['confidence_score'] for t in func_data['test_history']]
        all_times = [t['execution_time_ms'] for t in func_data['test_history'] if t['execution_time_ms'] > 0]
        
        func_data['average_confidence'] = sum(all_confidences) / len(all_confidences)
        if all_times:
            func_data['average_execution_time'] = sum(all_times) / len(all_times)
        
        self._save_performance_data()
        logger.info(f"Recorded test result for {result.function_name}: {result.detected}")
    
    def calculate_feedback_score(self, function_name: str) -> float:
        """
        Calculate feedback score (0-10) based on performance metrics
        
        Scoring criteria:
        - Detection success rate (40%)
        - Average confidence (30%)
        - Consistency (20%)
        - Execution efficiency (10%)
        """
        
        if function_name not in self.performance_data:
            return 0.0
        
        data = self.performance_data[function_name]
        
        if data['total_tests'] == 0:
            return 0.0
        
        # Detection success rate (0-4 points)
        success_rate = data['successful_detections'] / data['total_tests']
        success_score = success_rate * 4.0
        
        # Average confidence (0-3 points)
        confidence_score = (data['average_confidence'] / 100.0) * 3.0
        
        # Consistency score (0-2 points) - based on confidence variance
        confidences = [t['confidence_score'] for t in data['test_history']]
        if len(confidences) > 1:
            confidence_std = pd.Series(confidences).std()
            consistency_score = max(0, 2.0 - (confidence_std / 50.0))
        else:
            consistency_score = 1.0
        
        # Execution efficiency (0-1 points)
        if data['average_execution_time'] > 0:
            # Penalize slow functions (>1000ms gets 0 points)
            efficiency_score = max(0, 1.0 - (data['average_execution_time'] / 1000.0))
        else:
            efficiency_score = 1.0
        
        total_score = success_score + confidence_score + consistency_score + efficiency_score
        return min(10.0, total_score)
    
    def get_function_performance(self, function_name: str) -> Optional[Dict]:
        """Get detailed performance data for a function"""
        if function_name not in self.performance_data:
            return None
        
        data = self.performance_data[function_name].copy()
        data['feedback_score'] = self.calculate_feedback_score(function_name)
        
        return data
    
    def get_top_performers(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top performing functions by feedback score"""
        performers = []
        for func_name in self.performance_data.keys():
            score = self.calculate_feedback_score(func_name)
            performers.append((func_name, score))
        
        performers.sort(key=lambda x: x[1], reverse=True)
        return performers[:limit]
    
    def get_worst_performers(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get worst performing functions that need improvement"""
        performers = []
        for func_name in self.performance_data.keys():
            score = self.calculate_feedback_score(func_name)
            if self.performance_data[func_name]['total_tests'] >= 3:  # Only consider well-tested functions
                performers.append((func_name, score))
        
        performers.sort(key=lambda x: x[1])
        return performers[:limit]
    
    def analyze_performance_trends(self, function_name: str, days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if function_name not in self.performance_data:
            return {}
        
        data = self.performance_data[function_name]
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent tests
        recent_tests = []
        for test in data['test_history']:
            test_date = datetime.fromisoformat(test['test_timestamp'])
            if test_date >= cutoff_date:
                recent_tests.append(test)
        
        if not recent_tests:
            return {'message': f'No tests in the last {days} days'}
        
        # Calculate trends
        chronological_tests = sorted(recent_tests, key=lambda x: x['test_timestamp'])
        
        # Split into first and second half to see improvement
        mid_point = len(chronological_tests) // 2
        first_half = chronological_tests[:mid_point] if mid_point > 0 else []
        second_half = chronological_tests[mid_point:]
        
        first_success_rate = sum(t['detected'] for t in first_half) / len(first_half) if first_half else 0
        second_success_rate = sum(t['detected'] for t in second_half) / len(second_half) if second_half else 0
        
        first_avg_confidence = sum(t['confidence_score'] for t in first_half) / len(first_half) if first_half else 0
        second_avg_confidence = sum(t['confidence_score'] for t in second_half) / len(second_half) if second_half else 0
        
        return {
            'total_recent_tests': len(recent_tests),
            'success_rate_trend': second_success_rate - first_success_rate,
            'confidence_trend': second_avg_confidence - first_avg_confidence,
            'recent_success_rate': second_success_rate,
            'recent_avg_confidence': second_avg_confidence,
            'is_improving': second_success_rate > first_success_rate
        }
    
    def generate_improvement_suggestions(self, function_name: str) -> List[str]:
        """Generate suggestions for function improvement based on performance"""
        if function_name not in self.performance_data:
            return ["Function not found in performance database"]
        
        data = self.performance_data[function_name]
        suggestions = []
        
        if data['total_tests'] < 5:
            suggestions.append("Need more test data - function requires at least 5 test cases for reliable assessment")
        
        success_rate = data['successful_detections'] / data['total_tests'] if data['total_tests'] > 0 else 0
        
        if success_rate < 0.3:
            suggestions.append("Low detection rate - consider relaxing detection criteria or adding alternative signal patterns")
        elif success_rate < 0.6:
            suggestions.append("Moderate detection rate - fine-tune threshold values and signal combinations")
        
        if data['average_confidence'] < 50:
            suggestions.append("Low confidence scores - strengthen signal validation and add confirmation logic")
        
        if data['average_execution_time'] > 500:
            suggestions.append("Slow execution - optimize calculations and reduce computational complexity")
        
        # Analyze recent trend
        trends = self.analyze_performance_trends(function_name)
        if trends.get('success_rate_trend', 0) < -0.1:
            suggestions.append("Declining performance trend - function may be overfitting to old patterns")
        
        if not suggestions:
            suggestions.append("Function performing well - consider using as template for similar cases")
        
        return suggestions
    
    def export_performance_report(self, output_file: str = None) -> str:
        """Export comprehensive performance report"""
        if output_file is None:
            output_file = os.path.join(self.base_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_functions': len(self.performance_data),
            'total_tests': sum(data['total_tests'] for data in self.performance_data.values()),
            'functions': {}
        }
        
        for func_name, data in self.performance_data.items():
            func_report = data.copy()
            func_report['feedback_score'] = self.calculate_feedback_score(func_name)
            func_report['improvement_suggestions'] = self.generate_improvement_suggestions(func_name)
            func_report['performance_trends'] = self.analyze_performance_trends(func_name)
            
            report['functions'][func_name] = func_report
        
        # Add summary statistics
        scores = [self.calculate_feedback_score(name) for name in self.performance_data.keys()]
        if scores:
            report['summary'] = {
                'average_score': sum(scores) / len(scores),
                'top_score': max(scores),
                'functions_above_7': sum(1 for s in scores if s >= 7.0),
                'functions_below_3': sum(1 for s in scores if s < 3.0)
            }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Performance report exported to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
        
        return output_file
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up test results older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean results history
        initial_count = len(self.results_history)
        self.results_history = [
            result for result in self.results_history
            if datetime.fromisoformat(result['test_timestamp']) >= cutoff_date
        ]
        
        # Clean performance data test history
        for func_name, data in self.performance_data.items():
            initial_tests = len(data['test_history'])
            data['test_history'] = [
                test for test in data['test_history']
                if datetime.fromisoformat(test['test_timestamp']) >= cutoff_date
            ]
            
            # Recalculate statistics
            if data['test_history']:
                data['total_tests'] = len(data['test_history'])
                data['successful_detections'] = sum(1 for t in data['test_history'] if t['detected'])
                
                confidences = [t['confidence_score'] for t in data['test_history']]
                times = [t['execution_time_ms'] for t in data['test_history'] if t['execution_time_ms'] > 0]
                
                data['average_confidence'] = sum(confidences) / len(confidences)
                data['average_execution_time'] = sum(times) / len(times) if times else 0
            else:
                # Remove function if no recent data
                data['total_tests'] = 0
                data['successful_detections'] = 0
                data['average_confidence'] = 0
                data['average_execution_time'] = 0
        
        self._save_performance_data()
        self._save_results_history()
        
        cleaned_count = initial_count - len(self.results_history)
        logger.info(f"Cleaned {cleaned_count} old test results (older than {days} days)")
        
        return cleaned_count