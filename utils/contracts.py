import requests
import json
import os

# Cache globalny dla symbol -> coingecko_id
coingecko_symbol_map = {}

def load_coingecko_symbol_map():
    """Jednorazowe pobranie pełnej listy tokenów z CoinGecko"""
    global coingecko_symbol_map
    
    if coingecko_symbol_map:
        return  # Już załadowane
    
    try:
        headers = {}
        api_key = os.getenv("COINGECKO_API_KEY")
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
            
        response = requests.get("https://api.coingecko.com/api/v3/coins/list", headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            # Tworzymy mapę symbol -> id, np. {"PEPE": "pepe", ...}
            for item in data:
                symbol = item['symbol'].upper()
                coingecko_symbol_map[symbol] = item["id"]
                # Dodaj też wersję z USDT
                coingecko_symbol_map[f"{symbol}USDT"] = item["id"]
            
            print(f"✅ Załadowano {len(data)} tokenów z CoinGecko do cache")
        else:
            print(f"❌ Błąd ładowania listy CoinGecko: {response.status_code}")
    except Exception as e:
        print(f"❌ Wyjątek podczas ładowania listy CoinGecko: {e}")

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

def get_token_contract_from_coingecko(symbol):
    """
    Pobiera kontrakt tokena z CoinGecko API
    Zwraca: {"address": "...", "chain": "bsc"} lub None
    """
    try:
        # Użyj klucza API jeśli dostępny
        api_key = os.getenv('COINGECKO_API_KEY')
        headers = {}
        if api_key:
            headers['x-cg-demo-api-key'] = api_key
            
        # Załaduj mapę jeśli nie jest załadowana
        if not coingecko_symbol_map:
            load_coingecko_symbol_map()

        # Spróbuj znaleźć token w cache'u
        normalized_symbol = normalize_token_name(symbol)
        coingecko_id = coingecko_symbol_map.get(symbol) or coingecko_symbol_map.get(normalized_symbol)
        
        if not coingecko_id:
            print(f"⚠️ Brak CoinGecko ID dla {symbol}")
            return None

        url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            platforms = data.get("platforms", {})
            
            # Preferuj BSC, potem Ethereum, Polygon
            preferred_chains = ["binance-smart-chain", "ethereum", "polygon-pos", "arbitrum-one", "optimistic-ethereum"]
            
            for chain in preferred_chains:
                if chain in platforms and platforms[chain]:
                    return {
                        "address": platforms[chain], 
                        "chain": chain.replace("binance-smart-chain", "bsc")
                                    .replace("polygon-pos", "polygon")
                                    .replace("arbitrum-one", "arbitrum")
                                    .replace("optimistic-ethereum", "optimism")
                    }
            
            # Jeśli nie ma preferowanych, weź pierwszy dostępny
            for chain, address in platforms.items():
                if address:
                    return {
                        "address": address,
                        "chain": chain.replace("binance-smart-chain", "bsc")
                                    .replace("polygon-pos", "polygon")
                                    .replace("arbitrum-one", "arbitrum")
                                    .replace("optimistic-ethereum", "optimism")
                    }
        else:
            print(f"⚠️ CoinGecko API error {response.status_code} for {symbol}")
        
        return None
        
    except Exception as e:
        print(f"❌ Błąd pobierania z CoinGecko dla {symbol}: {e}")
        return None

def load_token_map():
    """Ładuje mapę tokenów z pliku"""
    try:
        if os.path.exists("token_contract_map.json"):
            with open("token_contract_map.json", "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"❌ Błąd ładowania token map: {e}")
    return {}

def save_token_map(token_map):
    """Zapisuje mapę tokenów do pliku"""
    try:
        with open("token_contract_map.json", "w", encoding="utf-8") as f:
            json.dump(token_map, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Błąd zapisywania token map: {e}")
        return False

def log_missing_contract(symbol):
    """Loguje token bez kontraktu"""
    try:
        with open("missing_contracts_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{symbol}\n")
    except Exception as e:
        print(f"❌ Błąd logowania missing contract: {e}")

def get_or_fetch_token_contract(symbol):
    """
    Główna funkcja - pobiera kontrakt z mapy lub z CoinGecko
    """
    # Załaduj mapę tokenów
    token_map = load_token_map()
    
    # Sprawdź czy token już istnieje w mapie
    if symbol in token_map:
        return token_map[symbol]
    
    # Normalizuj nazwę tokena dla CoinGecko
    clean_symbol = normalize_token_name(symbol)
    print(f"🔍 Szukam kontraktu dla {symbol} -> {clean_symbol}")
    
    # Pobierz z CoinGecko
    token_info = get_token_contract_from_coingecko(clean_symbol)
    
    if token_info:
        # Dodaj do mapy i zapisz
        token_map[symbol] = token_info
        if save_token_map(token_map):
            print(f"✅ Dodano kontrakt dla {symbol}: {token_info}")
        return token_info
    else:
        # Loguj brakujący kontrakt
        log_missing_contract(symbol)
        print(f"⚠️ Brak kontraktu dla {symbol}")
        return None
