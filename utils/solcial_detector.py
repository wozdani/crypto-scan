import requests
import time
import os

SOCIAL_SPIKE_THRESHOLD = 2.5  # czyli 250% wzrostu
_social_cache = {}

def get_social_mentions(symbol):
    """
    Pobiera dane spolecznosciowe dla danego symbolu z LunarCrush API
    """
    try:
        api_key = os.getenv('LUNARCRUSH_API_KEY')
        if not api_key:
            print(f"❌ Brak LUNARCRUSH_API_KEY dla {symbol}")
            return None
            
        url = f"https://api.lunarcrush.com/v2?data=assets&key={api_key}&symbol={symbol}"
        response = requests.get(url, timeout=10, proxies=PROXY)

        
        if response.status_code != 200:
            print(f"❌ Blad HTTP {response.status_code} dla social {symbol}")
            return None
            
        data = response.json()
        
        # Sprawdzamy czy dane są dostępne
        if not data.get("data") or len(data["data"]) == 0:
            return None
            
        # Mozemy uzywac realnych metryk: social_score, tweet_mentions, reddit_score
        mentions = data["data"][0].get("social_score", 0)
        return mentions
        
    except Exception as e:
        print(f"❌ Błąd pobierania social dla {symbol}: {e}")
        return None

def detect_social_spike(symbol):
    """
    Wykrywa nagly wzrost aktywnosci spolecznosciowej dla danego symbolu
    """
    current_mentions = get_social_mentions(symbol)
    if current_mentions is None:
        return False

    prev_mentions = _social_cache.get(symbol)
    _social_cache[symbol] = current_mentions

    if prev_mentions is None:
        return False  # Brak danych do porownania

    if prev_mentions == 0:
        return False  # Unikamy dzielenia przez zero

    spike_ratio = current_mentions / prev_mentions

    if spike_ratio >= SOCIAL_SPIKE_THRESHOLD:
        print(f"📣 Spike social dla {symbol} (×{spike_ratio:.2f})")
        return True

    return False

def get_social_stats(symbol):
    """
    Zwraca szczegolowe statystyki spolecznosciowe dla danego symbolu
    """
    try:
        api_key = os.getenv('LUNARCRUSH_API_KEY')
        if not api_key:
            return None
            
        url = f"https://api.lunarcrush.com/v2?data=assets&key={api_key}&symbol={symbol}"
        response = requests.get(url, timeout=10, proxies=PROXY)

        if response.status_code != 200:
            return None
            
        data = response.json()
        
        if not data.get("data") or len(data["data"]) == 0:
            return None
            
        asset_data = data["data"][0]
        
        return {
            "social_score": asset_data.get("social_score", 0),
            "tweet_mentions": asset_data.get("tweet_mentions", 0),
            "reddit_score": asset_data.get("reddit_score", 0),
            "social_volume": asset_data.get("social_volume", 0),
            "social_dominance": asset_data.get("social_dominance", 0)
        }
        
    except Exception as e:
        print(f"❌ Blad pobierania social stats dla {symbol}: {e}")
        return None
