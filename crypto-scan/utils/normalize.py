def normalize_token_name(symbol, cache):
    """
    CRITICAL FIX: Add timeout protection to prevent hanging on loop operations
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("normalize_token_name operation timeout")
    
    try:
        # EMERGENCY TIMEOUT: 1-second timeout for normalization
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)
    
        # Usuń typowe sufiksy stablecoinów
        for suffix in ["USDT", "USD", "PERP"]:
            if symbol.upper().endswith(suffix):
                symbol = symbol[:-len(suffix)]
                break

        symbol_upper = symbol.upper()
        symbol_lower = symbol.lower()

        # Najpierw sprawdź czy symbol w cache jako klucz pasuje dokładnie
        if symbol in cache:
            signal.alarm(0)
            return symbol
        if symbol_upper in cache:
            signal.alarm(0)
            return symbol_upper
        if symbol_lower in cache:
            signal.alarm(0)
            return symbol_lower

        # Jeśli nie – szukaj po skrócie w nazwach cache
        for token_name in cache:
            if token_name.lower() == symbol_lower:
                signal.alarm(0)
                return token_name
            if token_name.replace(" ", "").lower() == symbol_lower:
                signal.alarm(0)
                return token_name
            if token_name.replace("-", "").lower() == symbol_lower:
                signal.alarm(0)
                return token_name

        # Domyślnie zwróć wersję lower
        signal.alarm(0)  # Cancel timeout
        return symbol_lower
        
    except TimeoutError:
        signal.alarm(0)
        print(f"[NORMALIZE TIMEOUT] {symbol} - normalization timed out, returning original")
        return symbol.lower()
    except Exception as e:
        signal.alarm(0)
        print(f"[NORMALIZE ERROR] {symbol} - normalization error: {e}")
        return symbol.lower()