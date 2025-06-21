"""
Fix symbol fetching by implementing fallback symbol list
"""

def get_fallback_symbols():
    """Return hardcoded symbol list as fallback"""
    return [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "AVAXUSDT",
        "MATICUSDT", "LINKUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "FILUSDT",
        "NEARUSDT", "AAVEUSDT", "CRVUSDT", "SUSHIUSDT", "1INCHUSDT", "CKBUSDT",
        "MANAUSDT", "SANDUSDT", "AXSUSDT", "CHZUSDT", "ENJUSDT", "GALAUSDT"
    ]

def patch_crypto_scan_service():
    """Add fallback to crypto_scan_service.py"""
    
    fallback_code = '''
    # Fallback symbol list if Bybit API fails
    if not symbols_to_scan:
        print("🔄 Using fallback symbol list...")
        symbols_to_scan = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "AVAXUSDT",
            "MATICUSDT", "LINKUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "FILUSDT",
            "NEARUSDT", "AAVEUSDT", "CRVUSDT", "SUSHIUSDT", "1INCHUSDT", "CKBUSDT",
            "MANAUSDT", "SANDUSDT", "AXSUSDT", "CHZUSDT", "ENJUSDT", "GALAUSDT"
        ]
        print(f"📋 Fallback symbols loaded: {len(symbols_to_scan)}")
    '''
    
    print("Fallback code to add after symbol fetching:")
    print(fallback_code)

if __name__ == "__main__":
    patch_crypto_scan_service()