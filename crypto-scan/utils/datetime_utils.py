"""
DateTime utilities for crypto-scan system
"""

from datetime import datetime, timezone


def get_utc_hour() -> int:
    """
    Get current UTC hour (0-23)
    
    Returns:
        int: Current hour in UTC timezone
    """
    return datetime.now(timezone.utc).hour


def get_utc_timestamp() -> str:
    """
    Get current UTC timestamp as ISO string
    
    Returns:
        str: Current UTC timestamp
    """
    return datetime.now(timezone.utc).isoformat()


def get_utc_datetime() -> datetime:
    """
    Get current UTC datetime object
    
    Returns:
        datetime: Current UTC datetime
    """
    return datetime.now(timezone.utc)