#!/usr/bin/env python3
"""
Browser Installation Fix
Handles Playwright browser installation issues and provides fallback mechanisms
"""

import os
import subprocess
import sys
from typing import Optional

def check_chromium_installation() -> bool:
    """Check if Chromium is properly installed for Playwright"""
    
    # Common Chromium paths in Replit/Nix environment
    chromium_paths = [
        "/home/runner/workspace/.cache/ms-playwright/chromium-1091/chrome-linux/chrome",
        "/home/runner/workspace/.cache/ms-playwright/chromium_headless_shell-1179/chrome-linux/headless_shell",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser", 
        "/usr/bin/google-chrome"
    ]
    
    for path in chromium_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            print(f"[BROWSER CHECK] ✅ Found working Chromium: {path}")
            return True
    
    print(f"[BROWSER CHECK] ❌ No working Chromium executable found")
    return False

def install_playwright_browsers() -> bool:
    """Install Playwright browsers with proper error handling"""
    
    print("[BROWSER INSTALL] 🔄 Installing Playwright browsers...")
    
    try:
        # Try to install Chromium
        result = subprocess.run([
            sys.executable, "-m", "playwright", "install", "chromium"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("[BROWSER INSTALL] ✅ Chromium installation successful")
            return True
        else:
            print(f"[BROWSER INSTALL] ❌ Installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[BROWSER INSTALL] ⏰ Installation timeout - trying alternative approach")
        return False
    except Exception as e:
        print(f"[BROWSER INSTALL] ❌ Installation error: {e}")
        return False

def get_working_chromium_path() -> Optional[str]:
    """Get path to working Chromium executable"""
    
    chromium_paths = [
        "/home/runner/workspace/.cache/ms-playwright/chromium-1091/chrome-linux/chrome",
        "/home/runner/workspace/.cache/ms-playwright/chromium_headless_shell-1179/chrome-linux/headless_shell",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome"
    ]
    
    for path in chromium_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None

def fix_browser_installation() -> bool:
    """Main function to fix browser installation issues"""
    
    print("[BROWSER FIX] 🔧 Checking browser installation...")
    
    # Check if browsers are already working
    if check_chromium_installation():
        print("[BROWSER FIX] ✅ Browser installation is working")
        return True
    
    # Try to install browsers
    print("[BROWSER FIX] 🔄 Attempting browser installation...")
    if install_playwright_browsers():
        return check_chromium_installation()
    
    # Alternative: try to find system Chromium
    print("[BROWSER FIX] 🔍 Looking for system Chromium...")
    system_chromium = get_working_chromium_path()
    if system_chromium:
        print(f"[BROWSER FIX] ✅ Found system Chromium: {system_chromium}")
        return True
    
    print("[BROWSER FIX] ❌ Could not fix browser installation")
    return False

def create_fallback_tradingview_generator():
    """Create fallback TradingView generator that handles browser issues"""
    
    print("[BROWSER FIX] 🔄 Creating fallback TradingView generator...")
    
    fallback_code = '''
import os
from datetime import datetime
from typing import Optional

class FallbackTradingViewGenerator:
    """Fallback generator for when Playwright browsers are not available"""
    
    def __init__(self):
        self.browser_available = False
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def generate_tradingview_screenshot(
        self,
        symbol: str,
        tjde_score: float = 0.0,
        market_phase: str = "unknown",
        decision: str = "unknown"
    ) -> Optional[str]:
        """Generate placeholder when browser not available"""
        
        print(f"[FALLBACK] Browser not available - creating placeholder for {symbol}")
        
        # Create placeholder file in failed_charts directory
        os.makedirs("training_data/failed_charts", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        score_formatted = str(int(tjde_score * 1000)) if tjde_score > 0 else "000"
        
        placeholder_filename = f"{symbol}_{timestamp}_score-{score_formatted}_BROWSER_UNAVAILABLE.txt"
        placeholder_path = f"training_data/failed_charts/{placeholder_filename}"
        
        with open(placeholder_path, 'w') as f:
            f.write(f"TradingView screenshot placeholder for {symbol}\\n")
            f.write(f"TJDE Score: {tjde_score}\\n")
            f.write(f"Market Phase: {market_phase}\\n")
            f.write(f"Decision: {decision}\\n")
            f.write(f"Created: {datetime.now().isoformat()}\\n")
            f.write(f"Reason: Browser executable not available\\n")
        
        return None  # Return None to indicate failure
'''
    
    # Save fallback generator
    with open("crypto-scan/utils/fallback_tradingview.py", "w") as f:
        f.write(fallback_code)
    
    print("[BROWSER FIX] ✅ Fallback generator created")

if __name__ == "__main__":
    success = fix_browser_installation()
    if not success:
        create_fallback_tradingview_generator()
        print("[BROWSER FIX] 📋 Fallback system ready")
    else:
        print("[BROWSER FIX] 🎯 Browser installation fixed successfully")