import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

def build_token_contract_map(bybit_symbols: list[str]) -> dict:
    url = f"https://min-api.cryptocompare.com/data/all/coinlist?summary=true&api_key={CRYPTOCOMPARE_API_KEY}"
    response = requests.get(url, timeout=15)
    data = response.json()["Data"]

    token_map = {}

    for symbol in bybit_symbols:
        base = symbol.replace("USDT", "")
        token_info = data.get(base)

        if token_info:
            contract = token_info.get("SmartContractAddress")
            platform = token_info.get("PlatformType")

            if contract and platform and platform.lower() != "native":
                token_map[symbol] = {
                    "chain": platform.upper(),
                    "address": contract
                }

    return token_map

def save_token_map_to_json(symbols: list[str], filename="token_contract_map.json"):
    token_map = build_token_contract_map(symbols)
    with open(filename, "w") as f:
        json.dump(token_map, f, indent=2)
    print(f"✅ Zapisano mapę tokenów do {filename}")

# Przykład użycia:
if __name__ == "__main__":
    from utils.data_fetchers import get_symbols_cached
    symbols = get_symbols_cached()
    save_token_map_to_json(symbols)