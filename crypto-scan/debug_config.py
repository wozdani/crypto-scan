#!/usr/bin/env python3
"""
Debug Configuration for Crypto Scanner
Centralized debug logging setup for all modules
"""

import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure debug logging
logging.basicConfig(
    filename='logs/debug.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def debug_print(message: str, module: str = "GENERAL"):
    """
    Print to console and log to file
    
    Args:
        message: Debug message
        module: Module name for categorization
    """
    formatted_msg = f"[{module}] {message}"
    print(formatted_msg)
    logging.debug(formatted_msg)

def log_debug(message: str):
    """Simple debug logger"""
    logging.debug(message)
    print(f"[DEBUG] {message}")