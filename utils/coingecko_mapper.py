# utils/coingecko_mapper.py

import json
import time
import requests
import os

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/list"
CONTRACTS_URL = "https://api.coingecko.com/api/v3/coins/{id}"

def load_contract_map():
    try:
        with open("token_contract_map.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_contract_map(token_map):
    with open("token_contract_map.json", "w", encoding="utf-8") as f:
        json.dump(token_map, f, indent=2)

def normalize_symbol(symbol):
    return symbol.replace("USDT", "").replace("1000000", "").lower()

def get_coingecko_id_map():
    try:
        # Sprawdź czy mamy klucz API CoinGecko
        api_key = os.getenv("COINGECKO_API_KEY")
        headers = {}
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
            
        response = requests.get(COINGECKO_URL, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {item["symbol"]: item["id"] for item in data}
    except Exception as e:
        print(f"❌ Błąd przy pobieraniu listy CoinGecko: {e}")
        return {}

def fetch_contract_info(gecko_id):
    try:
        # Sprawdź czy mamy klucz API CoinGecko
        api_key = os.getenv("COINGECKO_API_KEY")
        headers = {}
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
            
        url = CONTRACTS_URL.format(id=gecko_id)
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        platforms = data.get("platforms", {})
        for chain, address in platforms.items():
            if address:
                # Mapowanie nazw chainów na standardowe nazwy
                chain_mapping = {
                    "ethereum": "ethereum",
                    "binance-smart-chain": "bsc",
                    "polygon-pos": "polygon",
                    "arbitrum-one": "arbitrum",
                    "optimistic-ethereum": "optimism",
                    "tron": "tron"
                }
                standard_chain = chain_mapping.get(chain.lower(), chain.lower())
                return {"chain": standard_chain, "address": address}
    except Exception as e:
        print(f"❌ Błąd przy pobieraniu kontraktu dla {gecko_id}: {e}")
    return None

def bulk_update_contracts(symbols):
    """Aktualizuje kontrakty dla symboli, których brakuje w mapie"""
    if not symbols:
        return
        
    token_map = load_contract_map()
    
    # Filtruj tylko symbole, których brakuje w mapie
    missing_symbols = [symbol for symbol in symbols if symbol not in token_map]
    
    if not missing_symbols:
        print(f"✅ Wszystkie {len(symbols)} symboli już ma kontrakty w mapie")
        return
    
    print(f"🔍 Pobieranie kontraktów dla {len(missing_symbols)} brakujących symboli...")
    
    gecko_ids = get_coingecko_id_map()
    if not gecko_ids:
        print("❌ Nie udało się pobrać listy CoinGecko")
        return

    updated_count = 0
    for symbol in missing_symbols:
        norm = normalize_symbol(symbol)
        gecko_id = gecko_ids.get(norm)
        if not gecko_id:
            print(f"⚠️ Brak CoinGecko ID dla {symbol}")
            continue

        contract_info = fetch_contract_info(gecko_id)
        if contract_info:
            token_map[symbol] = contract_info
            updated_count += 1
            print(f"✅ Dodano {symbol} → {contract_info['chain']} ({contract_info['address'][:10]}...)")
        else:
            print(f"❌ Nie udało się pobrać kontraktu dla {symbol}")

        # Rate limiting - 1.2s między zapytaniami
        time.sleep(1.2)

    if updated_count > 0:
        save_contract_map(token_map)
        print(f"💾 Zapisano {updated_count} nowych kontraktów do token_contract_map.json")
    else:
        print("❌ Nie dodano żadnych nowych kontraktów")