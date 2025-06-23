import requests

def normalize_token_name(symbol):
    """
    Usuwa cyfry z nazwy tokena i sufiks 'USDT'.
    Przykład: '1000000PEPEUSDT' -> 'PEPE'
    """
    return ''.join([c for c in symbol if not c.isdigit()]).replace('USDT', '')


def get_token_contract_from_coingecko(symbol):
    try:
        url = f"https://api.coingecko.com/api/v3/search?query={symbol}"
        response = requests.get(url, timeout=10)
        data = response.json()

        for item in data.get("coins", []):
            if item["symbol"].lower() == symbol.lower().replace("usdt", ""):
                token_id = item["id"]
                detail_url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
                detail = requests.get(detail_url, timeout=10).json()

                platforms = detail.get("platforms", {})
                for chain, address in platforms.items():
                    if address:
                        return chain.lower(), address

        return None, None
    except Exception as e:
        print(f"? Blad CoinGecko dla {symbol}: {e}")
        return None, None
