"""
Enhanced Chart Validator with OCR Support
Waliduje wykresy TradingView wykrywając błędne komunikaty i placeholdery
"""
import os
import pytesseract
from PIL import Image
import json
from typing import Dict, Optional, Any

class ChartValidator:
    """
    Comprehensive chart validation using OCR and file analysis
    """
    
    def __init__(self):
        self.validation_cache = {}
        self.error_keywords = [
            "invalid symbol",
            "symbol not found", 
            "no data",
            "not available",
            "failed to load",
            "tradingview error",
            "connection error",
            "page not found",
            "404",
            "symbol is invalid",
            "chart not available"
        ]
    
    def is_chart_valid(self, image_path: str) -> Dict[str, Any]:
        """
        Sprawdza czy wykres z TradingView jest prawidłowy
        
        Args:
            image_path: Ścieżka do pliku wykresu
            
        Returns:
            Dict z wynikami walidacji:
            {
                'valid': bool,
                'reason': str,
                'file_size': int,
                'validation_method': str
            }
        """
        if not os.path.exists(image_path):
            return {
                'valid': False,
                'reason': 'File does not exist',
                'file_size': 0,
                'validation_method': 'file_check'
            }
        
        # Check if it's a placeholder text file
        if image_path.endswith('.txt') or 'TRADINGVIEW_FAILED' in image_path:
            return {
                'valid': False,
                'reason': 'Placeholder file detected',
                'file_size': os.path.getsize(image_path),
                'validation_method': 'filename_check'
            }
        
        file_size = os.path.getsize(image_path)
        
        # File size validation - suspicious if too small
        if file_size < 5000:  # Less than 5KB
            return {
                'valid': False,
                'reason': f'File too small: {file_size} bytes (likely error page)',
                'file_size': file_size,
                'validation_method': 'file_size'
            }
        
        # Cache check
        cache_key = f"{image_path}_{file_size}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # OCR validation
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img).lower()
            
            # Check for error keywords
            for keyword in self.error_keywords:
                if keyword in text:
                    result = {
                        'valid': False,
                        'reason': f'OCR detected error: "{keyword}"',
                        'file_size': file_size,
                        'validation_method': 'ocr_content'
                    }
                    self.validation_cache[cache_key] = result
                    return result
            
            # Valid chart
            result = {
                'valid': True,
                'reason': 'Chart validated successfully',
                'file_size': file_size,
                'validation_method': 'ocr_content'
            }
            self.validation_cache[cache_key] = result
            return result
            
        except Exception as e:
            # OCR failed - fallback to file size validation
            print(f"[CHART VALIDATOR] OCR failed for {image_path}: {e}")
            
            if file_size > 10000:  # > 10KB probably valid
                result = {
                    'valid': True,
                    'reason': 'File size indicates valid chart (OCR unavailable)',
                    'file_size': file_size,
                    'validation_method': 'file_size_fallback'
                }
            else:
                result = {
                    'valid': False,
                    'reason': f'OCR failed and file size suspicious: {file_size} bytes',
                    'file_size': file_size,
                    'validation_method': 'file_size_fallback'
                }
            
            self.validation_cache[cache_key] = result
            return result
    
    def cleanup_invalid_chart(self, image_path: str) -> bool:
        """
        Usuwa nieprawidłowy wykres i powiązane pliki
        
        Args:
            image_path: Ścieżka do wykresu
            
        Returns:
            True jeśli usunięto, False jeśli błąd
        """
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"[CHART VALIDATOR] 🗑️ Removed invalid chart: {image_path}")
                
                # Remove metadata file if exists
                metadata_path = image_path.replace('.png', '_metadata.json')
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    print(f"[CHART VALIDATOR] 🗑️ Removed metadata: {metadata_path}")
                
                return True
            return False
        except Exception as e:
            print(f"[CHART VALIDATOR] ❌ Failed to cleanup {image_path}: {e}")
            return False
    
    def get_validation_stats(self) -> Dict:
        """Zwraca statystyki walidacji z cache"""
        valid_count = sum(1 for r in self.validation_cache.values() if r['valid'])
        total_count = len(self.validation_cache)
        
        return {
            'total_validated': total_count,
            'valid_charts': valid_count,
            'invalid_charts': total_count - valid_count,
            'validation_rate': valid_count / total_count if total_count > 0 else 0
        }

# Global validator instance
chart_validator = ChartValidator()

def is_chart_valid(image_path: str) -> bool:
    """
    Convenience function for quick validation
    
    Args:
        image_path: Path to chart image
        
    Returns:
        True if chart is valid, False otherwise
    """
    result = chart_validator.is_chart_valid(image_path)
    return result['valid']

def validate_and_cleanup_chart(image_path: str) -> Optional[str]:
    """
    Validates chart and returns path if valid, None if invalid (with cleanup)
    
    Args:
        image_path: Path to chart image
        
    Returns:
        image_path if valid, None if invalid
    """
    validation = chart_validator.is_chart_valid(image_path)
    
    if validation['valid']:
        print(f"[CHART FILTER] ✅ Chart validation passed: {image_path}")
        return image_path
    else:
        print(f"[CHART FILTER] ❌ Chart validation failed: {validation['reason']}")
        chart_validator.cleanup_invalid_chart(image_path)
        return None

def test_chart_validation():
    """Test chart validation system"""
    print("[CHART VALIDATOR] Testing validation system...")
    
    # Test with non-existent file
    result = chart_validator.is_chart_valid("nonexistent.png")
    print(f"Non-existent file: {result}")
    
    # Test with placeholder
    result = chart_validator.is_chart_valid("TRADINGVIEW_FAILED_placeholder.txt")
    print(f"Placeholder file: {result}")
    
    print("[CHART VALIDATOR] Test completed")

if __name__ == "__main__":
    test_chart_validation()