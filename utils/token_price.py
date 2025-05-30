import requests

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

        url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_id}&vs_currencies=usd"
        res = requests.get(url, timeout=10)
        data = res.json()
        return data[token_id]["usd"]
    except Exception as e:
        print(f"❌ Błąd CoinGecko dla {symbol}: {e}")
        return None