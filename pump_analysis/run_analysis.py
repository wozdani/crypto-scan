#!/usr/bin/env python3
"""
Simple runner script for pump analysis system
Can be used for cron jobs or manual execution
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    print("🚀 Starting Pump Analysis System...")
    try:
        main()
        print("✅ Analysis completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Analysis stopped by user")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)