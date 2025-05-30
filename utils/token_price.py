import requests
import os

def get_token_price_usd(symbol):
    try:
        coingecko_id_map = {
            "PEPEUSDT": "pepe",
            "FLOKIUSDT": "floki",
            # Dodaj więcej symboli → id z coingecko
        }

        token_id = coingecko_id_map.get(symbol)
        if not token_id:
            print(f"⚠️ Brak CoinGecko ID dla {symbol}")
            return None

        api_key = os.getenv("COINGECKO_API_KEY")
        
        if api_key:
            # Try with API key in URL parameter
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_id}&vs_currencies=usd&x_cg_demo_api_key={api_key}"
            res = requests.get(url, timeout=10)
        else:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_id}&vs_currencies=usd"
            res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data[token_id]["usd"]
    except Exception as e:
        print(f"❌ Błąd CoinGecko dla {symbol}: {e}")
        return None