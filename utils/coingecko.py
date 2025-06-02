import requests
import time
import os

# Cache ID list for performance
COINGECKO_TOKEN_LIST = []

def fetch_coingecko_token_list():
    """Pobiera listę wszystkich tokenów z CoinGecko - jedno zapytanie na start"""
    global COINGECKO_TOKEN_LIST
    if COINGECKO_TOKEN_LIST:
        return COINGECKO_TOKEN_LIST
    
    try:
        headers = {}
        api_key = os.getenv("COINGECKO_API_KEY")
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
            
        response = requests.get("https://api.coingecko.com/api/v3/coins/list", headers=headers, timeout=15)
        response.raise_for_status()
        COINGECKO_TOKEN_LIST = response.json()
        print(f"✅ Załadowano {len(COINGECKO_TOKEN_LIST)} tokenów z CoinGecko")
        return COINGECKO_TOKEN_LIST
    except Exception as e:
        print(f"❌ Błąd pobierania listy tokenów z CoinGecko: {e}")
        return []

def normalize_symbol_for_search(symbol):
    """Normalizuje symbol do wyszukiwania w CoinGecko"""
    # Usuń USDT, cyfry z początku
    cleaned = symbol.upper().replace("USDT", "").replace("PERP", "")
    cleaned = ''.join([c for c in cleaned if not c.isdigit()])
    return cleaned.strip()

def get_multiple_token_contracts_from_coingecko(symbols):
    """Pobiera kontrakty dla wielu tokenów naraz - minimalizuje zapytania API"""
    token_list = fetch_coingecko_token_list()
    if not token_list:
        return {}
    
    result = {}
    found_tokens = {}
    
    # Najpierw znajdź wszystkie tokeny w liście
    for symbol in symbols:
        normalized = normalize_symbol_for_search(symbol)
        
        # Szukaj dokładnego dopasowania
        match = next((t for t in token_list if t["symbol"].upper() == normalized), None)
        
        if match:
            found_tokens[symbol] = match["id"]
        else:
            result[symbol] = None
    
    print(f"🔍 Znaleziono {len(found_tokens)}/{len(symbols)} tokenów w CoinGecko")
    
    # Teraz pobierz szczegóły tylko dla znalezionych tokenów
    headers = {}
    api_key = os.getenv("COINGECKO_API_KEY")
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    
    for symbol, token_id in found_tokens.items():
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                platforms = data.get("platforms", {})
                
                # Preferuj BSC, potem Ethereum
                preferred_chains = ["binance-smart-chain", "ethereum", "polygon-pos", "arbitrum-one", "optimistic-ethereum"]
                
                contract_found = False
                for chain in preferred_chains:
                    if chain in platforms and platforms[chain]:
                        result[symbol] = {
                            "address": platforms[chain],
                            "chain": chain.replace("binance-smart-chain", "bsc")
                                        .replace("polygon-pos", "polygon")
                                        .replace("arbitrum-one", "arbitrum")
                                        .replace("optimistic-ethereum", "optimism")
                        }
                        contract_found = True
                        break
                
                # Jeśli nie ma preferowanych, weź pierwszy dostępny
                if not contract_found:
                    for chain, address in platforms.items():
                        if address:
                            result[symbol] = {
                                "address": address,
                                "chain": chain.replace("binance-smart-chain", "bsc")
                                            .replace("polygon-pos", "polygon")
                                            .replace("arbitrum-one", "arbitrum")
                                            .replace("optimistic-ethereum", "optimism")
                            }
                            break
                
                if symbol not in result:
                    result[symbol] = None
                    
            else:
                print(f"⚠️ CoinGecko API error {response.status_code} for {symbol}")
                result[symbol] = None
            
            # Rate limiting między zapytaniami
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Błąd pobierania danych dla {symbol}: {e}")
            result[symbol] = None
    
    return result