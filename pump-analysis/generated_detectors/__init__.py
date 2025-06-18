"""
Generated Detectors Module - Pump Analysis System

This module contains automatically generated Python detection functions
based on real pump analysis cases. Each function represents pattern
recognition logic derived from actual pre-pump conditions.

Usage:
    from generated_detectors import detect_BTCUSDT_20250613_preconditions
    result = detect_BTCUSDT_20250613_preconditions(df)

Functions are named: detect_<SYMBOL>_<YYYYMMDD>_preconditions()
"""

import os
import importlib.util
from typing import List, Callable
import pandas as pd

def get_available_detectors() -> List[str]:
    """
    Get list of all available detector functions
    
    Returns:
        List of detector function names
    """
    detectors = []
    current_dir = os.path.dirname(__file__)
    
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and filename != '__init__.py':
            # Extract function name from filename
            symbol_date = filename[:-3]  # Remove .py extension
            function_name = f"detect_{symbol_date}_preconditions"
            detectors.append(function_name)
    
    return detectors

def load_detector(symbol: str, date: str) -> Callable:
    """
    Dynamically load a specific detector function
    
    Args:
        symbol: Trading symbol (e.g. 'BTCUSDT')
        date: Date in YYYYMMDD format (e.g. '20250613')
    
    Returns:
        Detector function
    """
    filename = f"{symbol}_{date}.py"
    function_name = f"detect_{symbol}_{date}_preconditions"
    
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Detector file not found: {filename}")
    
    spec = importlib.util.spec_from_file_location(function_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return getattr(module, function_name)

def test_all_detectors(df: pd.DataFrame) -> dict:
    """
    Test all available detectors against provided data
    
    Args:
        df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'rsi']
    
    Returns:
        Dictionary with detector results
    """
    results = {}
    detectors = get_available_detectors()
    
    for detector_name in detectors:
        try:
            # Extract symbol and date from function name
            parts = detector_name.replace('detect_', '').replace('_preconditions', '').split('_')
            if len(parts) >= 2:
                symbol = '_'.join(parts[:-1])
                date = parts[-1]
                
                detector_func = load_detector(symbol, date)
                result = detector_func(df)
                results[detector_name] = result
        except Exception as e:
            results[detector_name] = f"Error: {str(e)}"
    
    return results