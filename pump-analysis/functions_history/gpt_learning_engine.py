#!/usr/bin/env python3
"""
GPT Learning Engine
Advanced AI system that learns from function performance and generates improved detectors
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI

from .function_manager import FunctionHistoryManager, FunctionMetadata
from .performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

class GPTLearningEngine:
    """Advanced GPT-4o learning system for detector function improvement"""
    
    def __init__(self, api_key: str, base_dir: str = "functions_history"):
        self.openai = OpenAI(api_key=api_key)
        self.function_manager = FunctionHistoryManager(base_dir)
        self.performance_tracker = PerformanceTracker(base_dir)
        self.base_dir = base_dir
        
        # Learning configuration
        self.model = "gpt-4o"  # Latest OpenAI model
        self.learning_context_file = os.path.join(base_dir, "learning_context.json")
        self.improvement_log = os.path.join(base_dir, "improvement_log.json")
        
        # Load learning context
        self.learning_context = self._load_learning_context()
        self.improvements_made = self._load_improvement_log()
    
    def _load_learning_context(self) -> Dict:
        """Load accumulated learning context"""
        if os.path.exists(self.learning_context_file):
            try:
                with open(self.learning_context_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading learning context: {e}")
        return {
            'successful_patterns': [],
            'failed_patterns': [],
            'improvement_strategies': [],
            'best_practices': []
        }
    
    def _save_learning_context(self):
        """Save learning context"""
        try:
            with open(self.learning_context_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_context, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving learning context: {e}")
    
    def _load_improvement_log(self) -> List[Dict]:
        """Load improvement history"""
        if os.path.exists(self.improvement_log):
            try:
                with open(self.improvement_log, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading improvement log: {e}")
        return []
    
    def _save_improvement_log(self):
        """Save improvement log"""
        try:
            with open(self.improvement_log, 'w', encoding='utf-8') as f:
                json.dump(self.improvements_made, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving improvement log: {e}")
    
    def generate_detector_function_with_history(self, pre_pump_data: Dict, pump_event: Any, 
                                              similar_cases: List = None) -> str:
        """
        Generate detector function with full historical context and learning
        
        Args:
            pre_pump_data: Pre-pump analysis data
            pump_event: Pump event details
            similar_cases: Similar historical cases
            
        Returns:
            Generated function code
        """
        
        # Get historical context
        symbol = pump_event.symbol
        pump_increase = pump_event.price_increase_pct
        
        # Find similar cases from history
        if similar_cases is None:
            similar_cases = self.function_manager.get_similar_cases(
                symbol, pump_increase, tolerance=10.0
            )
        
        # Get top performing functions for reference
        top_performers = self.function_manager.get_top_performing_functions(5)
        
        # Get performance insights
        performance_insights = self._generate_performance_insights()
        
        # Create comprehensive prompt with historical learning
        prompt = self._create_learning_prompt(
            pre_pump_data, pump_event, similar_cases, 
            top_performers, performance_insights
        )
        
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI system that learns from historical pump detection data. "
                        "Generate Python detector functions that improve upon past performance. "
                        "Focus on patterns that have proven successful and avoid approaches that have failed."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            function_code = response.choices[0].message.content
            
            # Log the generation for learning
            self._log_function_generation(pump_event, pre_pump_data, function_code)
            
            return function_code
            
        except Exception as e:
            logger.error(f"Error generating function with GPT: {e}")
            return self._generate_fallback_function(pre_pump_data, pump_event)
    
    def _create_learning_prompt(self, pre_pump_data: Dict, pump_event: Any,
                               similar_cases: List, top_performers: List,
                               performance_insights: Dict) -> str:
        """Create comprehensive learning prompt for GPT"""
        
        pump_date = pump_event.start_time.strftime("%Y%m%d")
        
        prompt = f"""
ADVANCED DETECTOR FUNCTION GENERATION WITH HISTORICAL LEARNING

=== TARGET PUMP CASE ===
Symbol: {pump_event.symbol}
Date: {pump_date}
Price Increase: +{pump_event.price_increase_pct:.1f}%
Duration: {pump_event.duration_minutes} minutes
Volume Spike: {pump_event.volume_spike:.1f}x

=== PRE-PUMP ANALYSIS DATA ===
"""
        
        # Add pre-pump data
        for key, value in pre_pump_data.items():
            if key == 'onchain_insights' and isinstance(value, list):
                prompt += f"On-chain Insights: {len(value)} signals detected\n"
                for insight in value[:3]:  # Show first 3
                    prompt += f"  • {insight}\n"
            else:
                prompt += f"{key}: {value}\n"
        
        # Add similar historical cases
        if similar_cases:
            prompt += f"\n=== SIMILAR HISTORICAL CASES ({len(similar_cases)}) ===\n"
            for i, (func_code, metadata) in enumerate(similar_cases[:3]):
                perf = self.performance_tracker.get_function_performance(metadata.function_name)
                score = perf.get('feedback_score', 0) if perf else 0
                prompt += f"Case {i+1}: {metadata.symbol} (+{metadata.pump_increase_pct:.1f}%) - Score: {score:.1f}/10\n"
        
        # Add top performing functions insights
        if top_performers:
            prompt += f"\n=== TOP PERFORMING PATTERNS ===\n"
            for func_code, metadata in top_performers[:3]:
                perf = self.performance_tracker.get_function_performance(metadata.function_name)
                if perf:
                    prompt += f"• {metadata.function_name}: {perf['feedback_score']:.1f}/10 score "
                    prompt += f"({perf['successful_detections']}/{perf['total_tests']} success rate)\n"
        
        # Add performance insights
        if performance_insights:
            prompt += f"\n=== PERFORMANCE INSIGHTS ===\n"
            for insight in performance_insights.get('key_lessons', []):
                prompt += f"• {insight}\n"
        
        # Add learning context
        if self.learning_context['successful_patterns']:
            prompt += f"\n=== SUCCESSFUL PATTERNS FROM HISTORY ===\n"
            for pattern in self.learning_context['successful_patterns'][-5:]:
                prompt += f"• {pattern}\n"
        
        if self.learning_context['failed_patterns']:
            prompt += f"\n=== PATTERNS TO AVOID ===\n"
            for pattern in self.learning_context['failed_patterns'][-3:]:
                prompt += f"• {pattern}\n"
        
        prompt += f"""

=== FUNCTION GENERATION TASK ===
Generate a Python function named 'detect_{pump_event.symbol}_{pump_date}_preconditions' that:

1. LEARNS from historical performance data above
2. INCORPORATES successful patterns from top performers
3. AVOIDS failed approaches from low-scoring functions
4. ADAPTS to the specific characteristics of this pump case
5. USES the pre-pump data and on-chain insights effectively

Requirements:
- Function must return (detected: bool, confidence: float, signals: list)
- Use the exact pre-pump data structure provided
- Include confidence scoring (0-100)
- Add detailed comments explaining the logic
- Consider on-chain insights in decision making
- Make the function robust and not overfitted

Focus on creating a detector that balances precision and recall based on the learning insights.
"""
        
        return prompt
    
    def _generate_performance_insights(self) -> Dict:
        """Generate insights from performance data"""
        insights = {
            'key_lessons': [],
            'common_failures': [],
            'success_factors': []
        }
        
        # Analyze top and worst performers
        top_performers = self.performance_tracker.get_top_performers(10)
        worst_performers = self.performance_tracker.get_worst_performers(5)
        
        # Extract lessons from top performers
        for func_name, score in top_performers:
            if score >= 7.0:
                perf = self.performance_tracker.get_function_performance(func_name)
                if perf:
                    insights['success_factors'].append(f"High confidence averaging {perf['average_confidence']:.1f}")
                    if perf['average_execution_time'] < 200:
                        insights['success_factors'].append("Fast execution under 200ms")
        
        # Extract lessons from poor performers
        for func_name, score in worst_performers:
            if score < 3.0:
                suggestions = self.performance_tracker.generate_improvement_suggestions(func_name)
                insights['common_failures'].extend(suggestions[:2])
        
        # Generate key lessons
        if insights['success_factors']:
            insights['key_lessons'].append("Top performers focus on high confidence thresholds and efficient execution")
        if insights['common_failures']:
            insights['key_lessons'].append("Avoid overly strict criteria that reduce detection rates")
        
        return insights
    
    def improve_underperforming_function(self, function_name: str) -> Optional[str]:
        """
        Generate improved version of an underperforming function
        
        Args:
            function_name: Name of function to improve
            
        Returns:
            New improved function name or None if failed
        """
        
        # Get function and performance data
        result = self.function_manager.get_function_by_name(function_name)
        if not result:
            logger.error(f"Function {function_name} not found")
            return None
        
        func_code, metadata = result
        performance = self.performance_tracker.get_function_performance(function_name)
        
        if not performance:
            logger.error(f"No performance data for {function_name}")
            return None
        
        # Check if improvement is needed
        if performance['feedback_score'] >= 7.0:
            logger.info(f"Function {function_name} already performing well (score: {performance['feedback_score']:.1f})")
            return None
        
        # Generate improvement suggestions
        suggestions = self.performance_tracker.generate_improvement_suggestions(function_name)
        
        # Get similar better-performing functions
        similar_cases = self.function_manager.get_similar_cases(
            metadata.symbol, metadata.pump_increase_pct, tolerance=15.0
        )
        
        better_functions = []
        for similar_code, similar_meta in similar_cases:
            similar_perf = self.performance_tracker.get_function_performance(similar_meta.function_name)
            if similar_perf and similar_perf['feedback_score'] > performance['feedback_score']:
                better_functions.append((similar_code, similar_meta, similar_perf))
        
        # Create improvement prompt
        prompt = self._create_improvement_prompt(
            func_code, metadata, performance, suggestions, better_functions
        )
        
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI that improves underperforming detector functions. "
                        "Analyze the issues and create an enhanced version that addresses the specific problems."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.6,
                max_tokens=3000
            )
            
            improved_code = response.choices[0].message.content
            
            # Create improved version
            improvement_notes = f"Improved from v{metadata.version} - Score: {performance['feedback_score']:.1f}/10"
            new_function_name = self.function_manager.create_improved_version(
                function_name, improved_code, improvement_notes
            )
            
            # Log improvement
            self._log_improvement(function_name, new_function_name, suggestions)
            
            logger.info(f"Generated improved function: {new_function_name}")
            return new_function_name
            
        except Exception as e:
            logger.error(f"Error improving function: {e}")
            return None
    
    def _create_improvement_prompt(self, func_code: str, metadata: FunctionMetadata,
                                 performance: Dict, suggestions: List[str],
                                 better_functions: List) -> str:
        """Create prompt for function improvement"""
        
        prompt = f"""
FUNCTION IMPROVEMENT TASK

=== CURRENT FUNCTION ANALYSIS ===
Function: {metadata.function_name}
Symbol: {metadata.symbol} (+{metadata.pump_increase_pct:.1f}%)
Current Score: {performance['feedback_score']:.1f}/10
Success Rate: {performance['successful_detections']}/{performance['total_tests']} ({performance['successful_detections']/performance['total_tests']*100:.1f}%)
Average Confidence: {performance['average_confidence']:.1f}

=== PERFORMANCE ISSUES ===
"""
        for suggestion in suggestions:
            prompt += f"• {suggestion}\n"
        
        prompt += f"\n=== CURRENT FUNCTION CODE ===\n{func_code}\n"
        
        if better_functions:
            prompt += f"\n=== BETTER PERFORMING SIMILAR FUNCTIONS ===\n"
            for i, (better_code, better_meta, better_perf) in enumerate(better_functions[:2]):
                prompt += f"Function {i+1}: {better_meta.function_name} (Score: {better_perf['feedback_score']:.1f}/10)\n"
                prompt += f"Key differences to analyze:\n{better_code[:500]}...\n\n"
        
        prompt += """
=== IMPROVEMENT TASK ===
Create an improved version that:
1. Addresses the specific performance issues identified
2. Learns from better-performing similar functions
3. Maintains the core detection logic but improves precision/recall
4. Increases confidence scoring accuracy
5. Optimizes execution efficiency

Return the complete improved function with:
- Same function signature
- Enhanced logic addressing the issues
- Improved confidence calculation
- Detailed comments explaining improvements
"""
        
        return prompt
    
    def _log_function_generation(self, pump_event: Any, pre_pump_data: Dict, function_code: str):
        """Log function generation for learning"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'generation',
            'symbol': pump_event.symbol,
            'pump_increase': pump_event.price_increase_pct,
            'pre_pump_signals': list(pre_pump_data.keys()),
            'function_length': len(function_code),
            'has_onchain_data': 'onchain_insights' in pre_pump_data
        }
        
        self.improvements_made.append(log_entry)
        self._save_improvement_log()
    
    def _log_improvement(self, original_func: str, improved_func: str, suggestions: List[str]):
        """Log function improvement for learning"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'improvement',
            'original_function': original_func,
            'improved_function': improved_func,
            'issues_addressed': suggestions
        }
        
        self.improvements_made.append(log_entry)
        self._save_improvement_log()
    
    def update_learning_context(self, function_name: str, performance_score: float):
        """Update learning context based on function performance"""
        
        result = self.function_manager.get_function_by_name(function_name)
        if not result:
            return
        
        func_code, metadata = result
        
        # Extract patterns from function
        patterns = self._extract_patterns_from_function(func_code)
        
        # Categorize based on performance
        if performance_score >= 7.0:
            self.learning_context['successful_patterns'].extend(patterns)
            # Keep only recent successful patterns
            self.learning_context['successful_patterns'] = self.learning_context['successful_patterns'][-20:]
        elif performance_score < 3.0:
            self.learning_context['failed_patterns'].extend(patterns)
            # Keep only recent failed patterns
            self.learning_context['failed_patterns'] = self.learning_context['failed_patterns'][-10:]
        
        self._save_learning_context()
    
    def _extract_patterns_from_function(self, func_code: str) -> List[str]:
        """Extract key patterns from function code"""
        patterns = []
        
        # Simple pattern extraction
        if 'rsi' in func_code.lower():
            patterns.append("Uses RSI indicator")
        if 'volume' in func_code.lower():
            patterns.append("Analyzes volume patterns")
        if 'vwap' in func_code.lower():
            patterns.append("Incorporates VWAP analysis")
        if 'onchain' in func_code.lower():
            patterns.append("Considers on-chain insights")
        if 'confidence =' in func_code:
            patterns.append("Has confidence calculation")
        if 'and' in func_code and 'or' in func_code:
            patterns.append("Uses complex boolean logic")
        
        return patterns

    def generate_detector_function(self, pre_pump_data: Dict, pump_event: 'PumpEvent') -> str:
        """Generate detector function based on pre-pump analysis (interface compatibility)"""
        return self.generate_detector_function_with_history(pre_pump_data, pump_event, None)

    def create_improved_version(self, function_id: str, original_function: str, performance_score: float, feedback: List[str]) -> str:
        """Create improved version of existing function"""
        try:
            # Analyze performance issues
            improvement_context = self._analyze_performance_issues(performance_score, feedback)
            
            # Generate improved version with GPT
            prompt = self._create_improvement_prompt(original_function, improvement_context)
            
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at improving detector functions based on performance feedback. Focus on accuracy and reliability."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.5
            )
            
            improved_code = response.choices[0].message.content.strip()
            improved_code = self._clean_function_code(improved_code)
            
            # Log improvement
            self._log_function_improvement(function_id, improved_code, performance_score)
            
            return improved_code
            
        except Exception as e:
            logger.error(f"Error creating improved function: {e}")
            return original_function

    def _analyze_performance_issues(self, score: float, feedback: List[str]) -> str:
        """Analyze performance issues for improvement"""
        issues = []
        if score < 0.5:
            issues.append("Low accuracy - detector may be too sensitive or missing key patterns")
        if 'false_positive' in str(feedback):
            issues.append("High false positive rate - tighten detection criteria")
        if 'false_negative' in str(feedback):
            issues.append("Missing valid signals - broaden detection scope")
        
        return "; ".join(issues) if issues else "General optimization needed"

    def _create_improvement_prompt(self, original_function: str, improvement_context: str) -> str:
        """Create prompt for function improvement"""
        return f"""
Improve this detector function based on performance feedback:

Original Function:
{original_function}

Performance Issues:
{improvement_context}

Create an improved version that:
1. Addresses the identified issues
2. Maintains the same function signature
3. Improves accuracy and reduces false positives
4. Uses more robust detection logic

Return only the improved Python function code.
"""

    def _clean_function_code(self, code: str) -> str:
        """Clean and validate function code"""
        # Remove markdown formatting
        if "```python" in code:
            start = code.find("```python") + 9
            end = code.find("```", start)
            if end != -1:
                code = code[start:end].strip()
        elif "```" in code:
            start = code.find("```") + 3
            end = code.find("```", start)
            if end != -1:
                code = code[start:end].strip()
        
        # Validate syntax
        try:
            compile(code, '<string>', 'exec')
            return code
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {e}")
            # Return basic fallback
            return """
def detector_function(df):
    if len(df) < 10:
        return False, 0.0, ['insufficient_data']
    return False, 0.0, ['syntax_error']
"""

    def _log_function_improvement(self, function_id: str, improved_code: str, score: float):
        """Log function improvement"""
        try:
            improvement_data = {
                'original_id': function_id,
                'improved_code': improved_code,
                'score': score,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to improvements log
            if not hasattr(self, 'improvements'):
                self.improvements = []
            self.improvements.append(improvement_data)
            
            logger.info(f"Function improvement logged for {function_id}")
            
        except Exception as e:
            logger.error(f"Error logging improvement: {e}")
    
    def _generate_fallback_function(self, pre_pump_data: Dict, pump_event: Any) -> str:
        """Generate basic fallback function if GPT fails"""
        
        pump_date = pump_event.start_time.strftime("%Y%m%d")
        
        return f'''def detect_{pump_event.symbol}_{pump_date}_preconditions(data):
    """
    Fallback detector function for {pump_event.symbol}
    Generated when GPT was unavailable
    """
    detected = False
    confidence = 0.0
    signals = []
    
    try:
        # Basic volume and RSI check
        if data.get('volume_trend') == 'increasing':
            signals.append('volume_increasing')
            confidence += 30
        
        rsi = data.get('rsi', 0)
        if 45 <= rsi <= 65:
            signals.append('rsi_neutral')
            confidence += 25
        
        # On-chain insights check
        onchain = data.get('onchain_insights', [])
        if onchain:
            signals.append('onchain_activity')
            confidence += 45
        
        detected = confidence >= 50
        
    except Exception as e:
        return False, 0.0, [f"error: {{e}}"]
    
    return detected, min(confidence, 100.0), signals
'''
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        
        total_functions = len(self.function_manager.metadata)
        improvements = len([log for log in self.improvements_made if log['type'] == 'improvement'])
        generations = len([log for log in self.improvements_made if log['type'] == 'generation'])
        
        # Performance distribution
        scores = []
        for func_name in self.function_manager.metadata.keys():
            score = self.performance_tracker.calculate_feedback_score(func_name)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        high_performers = sum(1 for s in scores if s >= 7.0)
        
        return {
            'total_functions': total_functions,
            'total_improvements': improvements,
            'total_generations': generations,
            'average_score': round(avg_score, 2),
            'high_performers': high_performers,
            'successful_patterns': len(self.learning_context['successful_patterns']),
            'failed_patterns': len(self.learning_context['failed_patterns']),
            'learning_entries': len(self.improvements_made)
        }