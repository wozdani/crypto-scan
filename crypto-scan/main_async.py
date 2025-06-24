#!/usr/bin/env python3
"""
Main Async Entry Point for Crypto Scanner
Replaces crypto_scan_service.py with full async architecture
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scan_all_tokens_async import main as async_main

if __name__ == "__main__":
    print("🚀 Starting Crypto Scanner with Full Async Architecture")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Replacing sequential scanning with parallel async execution")
    print("Target: <15 seconds for 500+ tokens")
    
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nCrypto Scanner stopped by user")
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)