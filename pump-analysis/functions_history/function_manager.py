#!/usr/bin/env python3
"""
Function History Manager
Manages detector function storage, versioning, and metadata tracking
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class FunctionMetadata:
    """Metadata for a detector function"""
    symbol: str
    date: str
    pump_increase: float
    generation_time: datetime
    active_signals: List[str]
    pre_pump_analysis: Dict[str, Any]
    function_name: Optional[str] = None
    pump_date: Optional[str] = None
    generation_date: Optional[str] = None
    pump_increase_pct: Optional[float] = None
    pump_duration_minutes: Optional[int] = None
    version: int = 1
    feedback_score: Optional[float] = None
    performance_tests: List[Dict] = None
    improvement_notes: str = ""
    parent_function: Optional[str] = None  # For improved versions
    
    def __post_init__(self):
        if self.performance_tests is None:
            self.performance_tests = []

class FunctionHistoryManager:
    """Manages detector function history and versioning"""
    
    def __init__(self, base_dir: str = "functions_history"):
        self.base_dir = base_dir
        self.functions_dir = os.path.join(base_dir, "functions")
        self.metadata_file = os.path.join(base_dir, "functions_metadata.json")
        self.performance_file = os.path.join(base_dir, "performance_history.json")
        
        # Create directories
        os.makedirs(self.functions_dir, exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        self.performance_history = self._load_performance_history()
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load function metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save function metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _load_performance_history(self) -> Dict[str, List]:
        """Load performance history from JSON file"""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading performance history: {e}")
        return {}
    
    def _save_performance_history(self):
        """Save performance history to JSON file"""
        try:
            with open(self.performance_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def store_function(self, function_code: str, metadata: FunctionMetadata) -> str:
        """Store a function with its metadata"""
        return self.save_function(function_code, metadata)
    
    def get_function(self, function_id: str) -> tuple:
        """Get function code and metadata by ID"""
        try:
            function_file = os.path.join(self.functions_dir, f"{function_id}.py")
            
            if not os.path.exists(function_file):
                return None, None
            
            # Load function code
            with open(function_file, 'r', encoding='utf-8') as f:
                function_code = f.read()
            
            # Get metadata
            metadata = self.metadata.get(function_id)
            
            return function_code, metadata
            
        except Exception as e:
            logger.error(f"Error getting function {function_id}: {e}")
            return None, None

    def list_functions(self) -> List[str]:
        """List all stored function IDs"""
        return list(self.metadata.keys())

    def save_function(self, function_code: str, metadata: FunctionMetadata) -> str:
        """
        Save a detector function with its metadata
        
        Args:
            function_code: The Python function code
            metadata: Function metadata
            
        Returns:
            Function file path
        """
        
        # Generate filename
        filename = f"{metadata.function_name}_v{metadata.version}.py"
        filepath = os.path.join(self.functions_dir, filename)
        
        # Save function code
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f'"""\n')
                f.write(f'Detector Function: {metadata.function_name}\n')
                f.write(f'Generated: {metadata.generation_date}\n')
                f.write(f'Symbol: {metadata.symbol}\n')
                f.write(f'Pump Date: {metadata.pump_date}\n')
                f.write(f'Pump Increase: +{metadata.pump_increase_pct:.1f}%\n')
                f.write(f'Version: {metadata.version}\n')
                if metadata.parent_function:
                    f.write(f'Improved from: {metadata.parent_function}\n')
                f.write(f'"""\n\n')
                f.write(function_code)
            
            logger.info(f"Function saved: {filepath}")
        except Exception as e:
            logger.error(f"Error saving function {filename}: {e}")
            return ""
        
        # Save metadata
        self.metadata[metadata.function_name] = asdict(metadata)
        self._save_metadata()
        
        return filepath
    
    def get_function_by_name(self, function_name: str) -> Optional[tuple]:
        """
        Get function code and metadata by name
        
        Returns:
            Tuple of (function_code, metadata) or None
        """
        if function_name not in self.metadata:
            return None
        
        metadata = self.metadata[function_name]
        filename = f"{function_name}_v{metadata['version']}.py"
        filepath = os.path.join(self.functions_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Function file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                function_code = f.read()
            return function_code, FunctionMetadata(**metadata)
        except Exception as e:
            logger.error(f"Error reading function {filepath}: {e}")
            return None
    
    def get_functions_for_symbol(self, symbol: str) -> List[tuple]:
        """Get all functions for a specific symbol"""
        functions = []
        for func_name, metadata in self.metadata.items():
            if metadata['symbol'] == symbol:
                result = self.get_function_by_name(func_name)
                if result:
                    functions.append(result)
        return functions
    
    def get_top_performing_functions(self, limit: int = 10) -> List[tuple]:
        """Get top performing functions by feedback score"""
        scored_functions = []
        for func_name, metadata in self.metadata.items():
            if metadata.get('feedback_score') is not None:
                result = self.get_function_by_name(func_name)
                if result:
                    scored_functions.append((result[0], result[1], metadata['feedback_score']))
        
        # Sort by feedback score (descending)
        scored_functions.sort(key=lambda x: x[2], reverse=True)
        return [(code, meta) for code, meta, score in scored_functions[:limit]]
    
    def get_similar_cases(self, symbol: str, pump_increase_pct: float, tolerance: float = 5.0) -> List[tuple]:
        """
        Find similar pump cases based on symbol and pump percentage
        
        Args:
            symbol: Target symbol
            pump_increase_pct: Target pump percentage
            tolerance: Percentage tolerance for similarity
            
        Returns:
            List of (function_code, metadata) tuples
        """
        similar_functions = []
        for func_name, metadata in self.metadata.items():
            # Check symbol match
            if metadata['symbol'] == symbol:
                # Check pump percentage similarity
                pump_diff = abs(metadata['pump_increase_pct'] - pump_increase_pct)
                if pump_diff <= tolerance:
                    result = self.get_function_by_name(func_name)
                    if result:
                        similar_functions.append(result)
        
        return similar_functions
    
    def update_feedback_score(self, function_name: str, score: float, notes: str = ""):
        """Update feedback score for a function"""
        if function_name in self.metadata:
            self.metadata[function_name]['feedback_score'] = score
            if notes:
                self.metadata[function_name]['improvement_notes'] = notes
            self._save_metadata()
            logger.info(f"Updated feedback score for {function_name}: {score}/10")
    
    def add_performance_test(self, function_name: str, test_result: Dict):
        """Add performance test result to function history"""
        if function_name in self.metadata:
            self.metadata[function_name]['performance_tests'].append({
                'timestamp': datetime.now().isoformat(),
                'result': test_result
            })
            self._save_metadata()
        
        # Also add to performance history
        if function_name not in self.performance_history:
            self.performance_history[function_name] = []
        
        self.performance_history[function_name].append({
            'timestamp': datetime.now().isoformat(),
            'test_result': test_result
        })
        self._save_performance_history()
    
    def create_improved_version(self, parent_function_name: str, new_function_code: str, 
                              improvement_notes: str) -> str:
        """
        Create an improved version of an existing function
        
        Args:
            parent_function_name: Name of the function to improve
            new_function_code: Improved function code
            improvement_notes: Description of improvements
            
        Returns:
            New function name
        """
        if parent_function_name not in self.metadata:
            raise ValueError(f"Parent function {parent_function_name} not found")
        
        parent_metadata = FunctionMetadata(**self.metadata[parent_function_name])
        
        # Create new metadata for improved version
        new_version = parent_metadata.version + 1
        new_function_name = f"detect_{parent_metadata.symbol}_{parent_metadata.pump_date}_preconditions_v{new_version}"
        
        improved_metadata = FunctionMetadata(
            function_name=new_function_name,
            symbol=parent_metadata.symbol,
            pump_date=parent_metadata.pump_date,
            generation_date=datetime.now().isoformat(),
            pump_increase_pct=parent_metadata.pump_increase_pct,
            pump_duration_minutes=parent_metadata.pump_duration_minutes,
            version=new_version,
            improvement_notes=improvement_notes,
            parent_function=parent_function_name
        )
        
        # Save improved function
        filepath = self.save_function(new_function_code, improved_metadata)
        logger.info(f"Created improved version: {new_function_name} (v{new_version})")
        
        return new_function_name
    
    def get_function_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored functions"""
        total_functions = len(self.metadata)
        scored_functions = sum(1 for meta in self.metadata.values() 
                             if meta.get('feedback_score') is not None)
        
        # Average score
        scores = [meta['feedback_score'] for meta in self.metadata.values() 
                 if meta.get('feedback_score') is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Version distribution
        versions = {}
        for meta in self.metadata.values():
            v = meta['version']
            versions[v] = versions.get(v, 0) + 1
        
        # Symbol distribution
        symbols = {}
        for meta in self.metadata.values():
            s = meta['symbol']
            symbols[s] = symbols.get(s, 0) + 1
        
        return {
            'total_functions': total_functions,
            'scored_functions': scored_functions,
            'average_score': round(avg_score, 2),
            'version_distribution': versions,
            'symbol_distribution': symbols,
            'top_symbols': sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def clean_old_versions(self, keep_versions: int = 3):
        """Clean old versions, keeping only the latest N versions per symbol"""
        symbol_functions = {}
        
        # Group functions by symbol
        for func_name, metadata in self.metadata.items():
            symbol = metadata['symbol']
            if symbol not in symbol_functions:
                symbol_functions[symbol] = []
            symbol_functions[symbol].append((func_name, metadata))
        
        # For each symbol, keep only the latest versions
        deleted_count = 0
        for symbol, functions in symbol_functions.items():
            # Sort by version (descending)
            functions.sort(key=lambda x: x[1]['version'], reverse=True)
            
            # Delete old versions
            for func_name, metadata in functions[keep_versions:]:
                filename = f"{func_name}_v{metadata['version']}.py"
                filepath = os.path.join(self.functions_dir, filename)
                
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    del self.metadata[func_name]
                    deleted_count += 1
                    logger.info(f"Deleted old function: {func_name}")
                except Exception as e:
                    logger.error(f"Error deleting {func_name}: {e}")
        
        if deleted_count > 0:
            self._save_metadata()
            logger.info(f"Cleaned {deleted_count} old function versions")
        
        return deleted_count