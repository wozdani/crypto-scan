"""
Screenshot Validator for TradingView Charts
Validates file size, content quality, and prevents empty chart generation
"""

import os
from PIL import Image
import numpy as np
from typing import Optional, Dict, Tuple
from debug_config import log_debug

class ScreenshotValidator:
    """Validates TradingView screenshots for quality and authenticity"""
    
    def __init__(self, min_file_size: int = 50000):  # 50KB minimum
        self.min_file_size = min_file_size
        self.max_white_pixels = 0.95  # Max 95% white pixels
        
    def validate_screenshot(self, file_path: str, symbol: str = "") -> Dict[str, any]:
        """
        Comprehensive screenshot validation
        
        Args:
            file_path: Path to screenshot file
            symbol: Trading symbol for logging
            
        Returns:
            Validation result dictionary with status and details
        """
        result = {
            'valid': False,
            'file_exists': False,
            'file_size': 0,
            'size_valid': False,
            'content_valid': False,
            'white_pixel_ratio': 0.0,
            'error': None,
            'symbol': symbol
        }
        
        try:
            # Check file existence
            if not os.path.exists(file_path):
                result['error'] = "File does not exist"
                log_debug(f"[SCREENSHOT VALIDATOR] {symbol}: File not found - {file_path}")
                return result
                
            result['file_exists'] = True
            
            # Check file size
            file_size = os.path.getsize(file_path)
            result['file_size'] = file_size
            
            if file_size < self.min_file_size:
                result['error'] = f"File too small: {file_size} bytes (min: {self.min_file_size})"
                log_debug(f"[SCREENSHOT VALIDATOR] {symbol}: File too small - {file_size} bytes")
                return result
                
            result['size_valid'] = True
            
            # Analyze image content
            white_ratio = self._analyze_image_content(file_path)
            result['white_pixel_ratio'] = white_ratio
            
            if white_ratio > self.max_white_pixels:
                result['error'] = f"Too many white pixels: {white_ratio:.1%} (max: {self.max_white_pixels:.1%})"
                log_debug(f"[SCREENSHOT VALIDATOR] {symbol}: Mostly white image - {white_ratio:.1%}")
                return result
                
            result['content_valid'] = True
            result['valid'] = True
            
            log_debug(f"[SCREENSHOT VALIDATOR] {symbol}: ✅ Valid screenshot - {file_size} bytes, {white_ratio:.1%} white")
            return result
            
        except Exception as e:
            result['error'] = f"Validation error: {str(e)}"
            log_debug(f"[SCREENSHOT VALIDATOR] {symbol}: Error - {e}")
            return result
    
    def _analyze_image_content(self, file_path: str) -> float:
        """Analyze image content for white pixel ratio"""
        try:
            with Image.open(file_path) as img:
                # Convert to grayscale for analysis
                gray = img.convert('L')
                
                # Convert to numpy array
                img_array = np.array(gray)
                
                # Count white-ish pixels (>240 in grayscale)
                white_pixels = np.sum(img_array > 240)
                total_pixels = img_array.size
                
                return white_pixels / total_pixels if total_pixels > 0 else 1.0
                
        except Exception as e:
            log_debug(f"[SCREENSHOT VALIDATOR] Image analysis error: {e}")
            return 1.0  # Assume invalid if can't analyze
    
    def cleanup_invalid_screenshot(self, file_path: str, symbol: str = "") -> bool:
        """Remove invalid screenshot and its metadata"""
        try:
            # Remove PNG file
            if os.path.exists(file_path):
                os.remove(file_path)
                log_debug(f"[SCREENSHOT CLEANUP] {symbol}: Removed invalid PNG - {file_path}")
            
            # Remove JSON metadata
            json_path = file_path.replace('.png', '.json')
            if os.path.exists(json_path):
                os.remove(json_path)
                log_debug(f"[SCREENSHOT CLEANUP] {symbol}: Removed invalid JSON - {json_path}")
            
            return True
            
        except Exception as e:
            log_debug(f"[SCREENSHOT CLEANUP] {symbol}: Error removing files - {e}")
            return False
    
    def validate_and_cleanup(self, file_path: str, symbol: str = "") -> bool:
        """Validate screenshot and cleanup if invalid"""
        validation = self.validate_screenshot(file_path, symbol)
        
        if not validation['valid']:
            log_debug(f"[SCREENSHOT VALIDATOR] {symbol}: Invalid - {validation['error']}")
            self.cleanup_invalid_screenshot(file_path, symbol)
            return False
        
        return True
    
    def get_validation_stats(self) -> Dict[str, any]:
        """Get validator configuration stats"""
        return {
            'min_file_size': self.min_file_size,
            'max_white_pixels': self.max_white_pixels,
            'min_size_kb': self.min_file_size // 1024
        }

def validate_tradingview_screenshot(file_path: str, symbol: str = "", min_size_kb: int = 50) -> bool:
    """
    Convenience function to validate TradingView screenshot
    
    Args:
        file_path: Path to screenshot file
        symbol: Trading symbol for logging
        min_size_kb: Minimum file size in KB
        
    Returns:
        True if screenshot is valid, False otherwise
    """
    validator = ScreenshotValidator(min_file_size=min_size_kb * 1024)
    return validator.validate_and_cleanup(file_path, symbol)

def is_screenshot_valid(file_path: str) -> Tuple[bool, str]:
    """
    Quick validation check
    
    Returns:
        (is_valid, error_message)
    """
    validator = ScreenshotValidator()
    result = validator.validate_screenshot(file_path)
    return result['valid'], result.get('error', '')

def main():
    """Test screenshot validation"""
    validator = ScreenshotValidator()
    
    print("🔍 Screenshot Validator Configuration:")
    stats = validator.get_validation_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test with dummy file paths
    test_files = [
        "/path/to/valid_chart.png",
        "/path/to/empty_chart.png",
        "/path/to/nonexistent.png"
    ]
    
    print(f"\n🧪 Testing validation logic:")
    for file_path in test_files:
        result = validator.validate_screenshot(file_path, "TEST")
        status = "✅ VALID" if result['valid'] else f"❌ INVALID: {result['error']}"
        print(f"  {os.path.basename(file_path)}: {status}")

if __name__ == "__main__":
    main()