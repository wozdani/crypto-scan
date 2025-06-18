def normalize_token_name(symbol, cache):
    
    # Usuń typowe sufiksy stablecoinów
    for suffix in ["USDT", "USD", "PERP"]:
        if symbol.upper().endswith(suffix):
            symbol = symbol[:-len(suffix)]
            break

    symbol_upper = symbol.upper()
    symbol_lower = symbol.lower()

    # Najpierw sprawdź czy symbol w cache jako klucz pasuje dokładnie
    if symbol in cache:
        return symbol
    if symbol_upper in cache:
        return symbol_upper
    if symbol_lower in cache:
        return symbol_lower

    # Jeśli nie – szukaj po skrócie w nazwach cache
    for token_name in cache:
        if token_name.lower() == symbol_lower:
            return token_name
        if token_name.replace(" ", "").lower() == symbol_lower:
            return token_name
        if token_name.replace("-", "").lower() == symbol_lower:
            return token_name

    # Domyślnie zwróć wersję lower
    return symbol_lower
