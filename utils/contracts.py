from utils.coingecko import get_contract

def get_or_fetch_token_contract(symbol):
    try:
        token_info = get_contract(symbol)
        return token_info
    except Exception as e:
        print(f"❌ Błąd pobierania kontraktu dla {symbol}: {e}")
        return None

def normalize_token_name(symbol):
    """
    Usuwa cyfry z nazwy tokena i sufiks 'USDT', 'PERP'.
    Przykład: '1000000CHEEMSUSDT' -> 'CHEEMS'
    """
    # Usuń cyfry z początku
    cleaned = ''.join([c for c in symbol if not c.isdigit()])
    # Usuń sufiksy
    cleaned = cleaned.replace('USDT', '').replace('PERP', '')
    return cleaned.strip()