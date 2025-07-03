#!/usr/bin/env python3
"""
Debug OCR functionality
"""
import sys
sys.path.insert(0, '/home/runner/workspace/crypto-scan')

from utils.tradingview_robust import RobustTradingViewGenerator
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import os

def create_simple_test():
    """Create simple invalid symbol test"""
    # Create white image with black text
    img = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    # Use default font and black text
    draw.text((50, 100), "Invalid symbol", fill='black')
    draw.text((50, 150), "Error loading chart", fill='red')
    
    img.save('simple_test.png')
    print("Created simple_test.png")

def test_ocr_directly():
    """Test OCR directly"""
    create_simple_test()
    
    # Load and test with OCR
    img = Image.open('simple_test.png')
    text = pytesseract.image_to_string(img)
    
    print(f"OCR raw text: {repr(text)}")
    print(f"OCR lower: {repr(text.lower())}")
    
    # Test detection patterns
    patterns = ['invalid', 'error', 'symbol', 'loading']
    for pattern in patterns:
        found = pattern in text.lower()
        print(f"Pattern '{pattern}': {found}")

def test_chart_validator():
    """Test the chart validator function"""
    create_simple_test()
    
    generator = RobustTradingViewGenerator()
    result = generator.is_chart_valid('simple_test.png')
    
    print(f"Chart validator result: {result}")
    print("Expected: False (should detect invalid chart)")

if __name__ == "__main__":
    print("🔍 OCR Debug Test")
    print("=" * 30)
    
    try:
        test_ocr_directly()
        print("\n" + "=" * 30)
        test_chart_validator()
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Cleanup
        try:
            os.remove('simple_test.png')
        except:
            pass