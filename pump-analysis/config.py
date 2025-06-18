#!/usr/bin/env python3
"""
Configuration module for pump analysis system
Handles environment variables and system settings
"""

import os
from typing import Optional

class Config:
    """Configuration class for pump analysis system"""
    
    def __init__(self):
        # Load environment variables with defaults
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Bybit API credentials (required for data fetching)
        self.bybit_api_key = os.getenv('BYBIT_API_KEY')
        self.bybit_secret_key = os.getenv('BYBIT_SECRET_KEY')
        
        # Analysis parameters with defaults
        self.min_pump_increase_pct = float(os.getenv('MIN_PUMP_INCREASE_PCT', '15.0'))
        self.detection_window_minutes = int(os.getenv('DETECTION_WINDOW_MINUTES', '30'))
        self.analysis_days_back = int(os.getenv('ANALYSIS_DAYS_BACK', '7'))
        self.max_symbols_to_analyze = int(os.getenv('MAX_SYMBOLS_TO_ANALYZE', '999999'))
        
        # System settings
        self.data_directory = 'pump_data'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration
        
        Returns:
            tuple: (is_valid, list_of_missing_keys)
        """
        missing_keys = []
        
        if not self.openai_api_key:
            missing_keys.append('OPENAI_API_KEY')
        if not self.bybit_api_key:
            missing_keys.append('BYBIT_API_KEY')
        if not self.bybit_secret_key:
            missing_keys.append('BYBIT_SECRET_KEY')
        if not self.telegram_bot_token:
            missing_keys.append('TELEGRAM_BOT_TOKEN')
        if not self.telegram_chat_id:
            missing_keys.append('TELEGRAM_CHAT_ID')
            
        return len(missing_keys) == 0, missing_keys
    
    def get_summary(self) -> dict:
        """Get configuration summary for logging"""
        return {
            'min_pump_increase_pct': self.min_pump_increase_pct,
            'detection_window_minutes': self.detection_window_minutes,
            'analysis_days_back': self.analysis_days_back,
            'max_symbols_to_analyze': self.max_symbols_to_analyze,
            'data_directory': self.data_directory,
            'openai_configured': bool(self.openai_api_key),
            'telegram_configured': bool(self.telegram_bot_token and self.telegram_chat_id)
        }